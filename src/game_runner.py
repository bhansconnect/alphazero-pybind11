import glob
import importlib.util
import os
from collections import namedtuple
from multiprocessing.pool import ThreadPool as Pool
import math
import time
import torch
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import threading
import tqdm
import queue
import numpy as np
import gc

src_path = os.path.dirname(os.path.realpath(__file__))
build_path = os.path.join(os.path.dirname(src_path), 'build/src')
lib_path = glob.glob(os.path.join(build_path, 'alphazero*.so'))[0]

spec = importlib.util.spec_from_file_location(
    'alphazero', lib_path)
alphazero = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alphazero)

GRArgs = namedtuple(
    'GRArgs', ['title', 'game', 'max_batch_size', 'iteration',  'data_save_size', 'data_folder', 'concurrent_batches', 'batch_workers', 'nn_workers', 'result_workers', 'mcts_workers', 'cuda'], defaults=(0, 30000, 'data/history', 0, 0, 1, 1, os.cpu_count() - 1, torch.cuda.is_available()))


def calculate_win_draw_rates(n, p1_score, draws):
    p1_wins = (n+p1_score-draws)/2
    p1_adj_wins = p1_wins + draws/Game.NUM_PLAYERS()
    p1_rate = p1_adj_wins/n
    draw_rate = draws/n
    return p1_rate, draw_rate


class GameRunner:
    def __init__(self, players, pm, args):
        self.players = players
        self.pm = pm
        self.args = args
        self.num_players = self.args.game.NUM_PLAYERS()
        self.batch_workers = self.args.batch_workers
        if self.batch_workers == 0:
            self.batch_workers = self.num_players
        self.concurrent_batches = self.args.concurrent_batches
        if self.concurrent_batches == 0:
            self.concurrent_batches = self.num_players
        if self.batch_workers % self.num_players != 0:
            raise Exception(
                'batch workers should be a multiple of the number of players')
        if self.concurrent_batches % self.batch_workers != 0:
            raise Exception(
                'concurrent batches should be a multiple of the number of batch workers')
        if len(self.players) != self.num_players:
            raise Exception('There must be a player for each player')
        self.ready_queues = []
        for i in range(self.batch_workers):
            self.ready_queues.append(queue.SimpleQueue())
        self.batch_queue = queue.SimpleQueue()
        self.result_queue = queue.SimpleQueue()
        self.monitor_queue = queue.SimpleQueue()
        self.saved_samples = 0
        cs = self.args.game.CANONICAL_SHAPE()
        self.hist_canonical = torch.zeros(
            self.args.data_save_size, cs[0], cs[1], cs[2])
        self.hist_v = torch.zeros(
            self.args.data_save_size, self.num_players)
        self.hist_pi = torch.zeros(
            self.args.data_save_size, self.args.game.NUM_MOVES())
        shape = (self.args.max_batch_size, cs[0], cs[1], cs[2])
        self.batches = []
        for i in range(self.concurrent_batches):
            self.batches.append(torch.zeros(shape))
            self.batches[i].pin_memory()
            self.ready_queues[i % self.num_players].put(i)

    def run(self):
        batch_workers = []
        for i in range(self.batch_workers):
            batch_workers.append(threading.Thread(
                target=self.batch_builder, args=(i % self.num_players,)))
            batch_workers[i].start()
        result_workers = []
        for i in range(self.args.result_workers):
            result_workers.append(threading.Thread(
                target=self.result_processor))
            result_workers[i].start()
        player_workers = []
        for i in range(self.num_players):
            player_workers.append(threading.Thread(
                target=self.player_executor))
            player_workers[i].start()

        monitor = threading.Thread(target=self.monitor)
        monitor.start()
        hist_saver = threading.Thread(target=self.hist_saver)
        hist_saver.start()

        with Pool(self.args.mcts_workers) as pool:
            pool.map(lambda _: self.pm.play(),
                     range(self.args.mcts_workers), 1)

        for bw in batch_workers:
            bw.join()
        for rw in result_workers:
            rw.join()
        for pw in player_workers:
            pw.join()
        monitor.join()
        hist_saver.join()
        # clear queues before returning.
        while not self.batch_queue.empty():
            self.batch_queue.get()
        while not self.result_queue.empty():
            self.result_queue.get()
        while not self.monitor_queue.empty():
            self.monitor_queue.get()
        for ready_queue in self.ready_queues:
            while not ready_queue.empty():
                ready_queue.get()

    def monitor(self):
        last_completed = 0
        last_update = time.time()
        n = self.pm.params().games_to_play
        pbar = tqdm.tqdm(total=n,
                         unit='games', desc=self.args.title, leave=False)
        while(self.pm.remaining_games() > 0):
            try:
                self.monitor_queue.get(timeout=1)
            except queue.Empty:
                continue
            if time.time() - last_update > 1:
                hr = 0
                hits = self.pm.cache_hits()
                total = hits + self.pm.cache_misses()
                if total > 0:
                    hr = hits/total
                draw_rate = 0
                p1_rate = 0
                completed = self.pm.games_completed()
                if completed > 0:
                    p1_rate, draw_rate = calculate_win_draw_rates(
                        completed, self.pm.scores()[0], self.pm.draws())
                pbar.set_postfix({
                    'p1 rate': p1_rate,
                    'draw rate': draw_rate,
                    'cache rate': hr})
                pbar.update(completed-last_completed)
                last_completed = completed
                last_update = time.time()
        hr = 0
        hits = self.pm.cache_hits()
        total = hits + self.pm.cache_misses()
        if total > 0:
            hr = hits/total
        p1_rate, draw_rate = calculate_win_draw_rates(
            n, self.pm.scores()[0], self.pm.draws())
        pbar.set_postfix({
            'p1 rate': p1_rate,
            'draw rate': draw_rate,
            'cache hit': hr})
        pbar.update(n - last_completed)
        pbar.close()

    def batch_builder(self, player):
        while(self.pm.remaining_games() > 0):
            try:
                batch_index = self.ready_queues[player].get(timeout=1)
            except queue.Empty:
                continue
            batch = self.batches[batch_index]
            game_indices = self.pm.build_batch(
                batch_index % self.num_players, batch, self.batch_workers)
            out = batch[:len(game_indices)]
            if self.args.cuda:
                out = out.contiguous().cuda(non_blocking=True)
            self.batch_queue.put((out, batch_index, game_indices))

    def player_executor(self):
        while(self.pm.remaining_games() > 0):
            try:
                batch, batch_index, game_indices = self.batch_queue.get(
                    timeout=1)
            except queue.Empty:
                continue
            v, pi = self.players[batch_index % self.num_players].process(batch)
            self.result_queue.put((v, pi, batch_index, game_indices))

    def result_processor(self):
        while(self.pm.remaining_games() > 0):
            try:
                v, pi, batch_index, game_indices = self.result_queue.get(
                    timeout=1)
            except queue.Empty:
                continue
            v = v.cpu().numpy()
            pi = pi.cpu().numpy()
            self.pm.update_inferences(batch_index % self.num_players,
                                      game_indices, v, pi)
            self.ready_queues[batch_index % self.num_players].put(batch_index)
            self.monitor_queue.put(v.shape[0])

    def hist_saver(self):
        batch = 0
        os.makedirs(self.args.data_folder, exist_ok=True)
        while(self.pm.remaining_games() > 0 or self.pm.hist_count() > 0):
            size = self.pm.build_history_batch(
                self.hist_canonical, self.hist_v, self.hist_pi)
            if size == 0:
                continue
            torch.save(
                self.hist_canonical[:size], f'{self.args.data_folder}/{self.args.iteration:04d}-{batch:04d}-canonical.pt')
            torch.save(
                self.hist_v[:size], f'{self.args.data_folder}/{self.args.iteration:04d}-{batch:04d}-v.pt')
            torch.save(
                self.hist_pi[:size], f'{self.args.data_folder}/{self.args.iteration:04d}-{batch:04d}-pi.pt')
            self.saved_samples += size
            batch += 1


class RandPlayer:
    def __init__(self, game, max_batch_size):
        self.v = torch.zeros((max_batch_size, game.NUM_PLAYERS()))
        # I'm not really sure why, but e^-2 seems to be the average value here
        # for a new network and leads to good results. Zero or 1 performs quite poorly.
        self.pi = torch.exp(-2*torch.ones((max_batch_size, game.NUM_MOVES())))

    def predict(self, canonical):
        v, pi = self.process(canonical.unsqueeze(0))
        return v[0], pi[0]

    def process(self, batch):
        return self.v[:batch.shape[0]], self.pi[:batch.shape[0]]


if __name__ == '__main__':
    import shutil
    import neural_net

    EXPECTED_OPENING_LENGTH = 6
    CPUCT = 2
    SELF_PLAY_TEMP = 1
    EVAL_TEMP = 0.5
    TEMP_DECAY_HALF_LIFE = EXPECTED_OPENING_LENGTH
    FINAL_TEMP = 0.2
    MAX_CACHE_SIZE = 1000000

    # To decide on the following numbers, I would advise graphing the equation: scalar*(1+beta*(((iter+1)/scalar)**alpha-1)/alpha)
    WINDOW_SIZE_ALPHA = 0.5  # This decides how fast the curve flattens to a max
    WINDOW_SIZE_BETA = 0.7  # This decides the rough overall slope.
    WINDOW_SIZE_SCALAR = 6  # This ends up being approximately first time history doesn't grow

    CONCURRENT_BATCHES = 4
    RESULT_WORKERS = 2

    def base_params(Game, start_temp, bs):
        params = alphazero.PlayParams()
        params.max_cache_size = MAX_CACHE_SIZE
        params.cpuct = CPUCT
        params.start_temp = start_temp
        params.temp_decay_half_life = TEMP_DECAY_HALF_LIFE
        params.final_temp = FINAL_TEMP
        params.max_batch_size = bs
        params.concurrent_games = bs * CONCURRENT_BATCHES
        return params

    def create_init_net(Game, nnargs):
        nn = neural_net.NNWrapper(Game, nnargs)
        nn.save_checkpoint('data/checkpoint', '0000.pt')

    def calc_hist_size(i):
        return int(WINDOW_SIZE_SCALAR*(1 + WINDOW_SIZE_BETA*(((i+1)/WINDOW_SIZE_SCALAR)**WINDOW_SIZE_ALPHA-1)/WINDOW_SIZE_ALPHA))

    def train(Game, nnargs, iteration, hist_size):
        nn = neural_net.NNWrapper(Game, nnargs)
        nn.load_checkpoint('data/checkpoint', f'{iteration:04d}.pt')
        datasets = []
        for i in range(max(0, iteration - hist_size), iteration + 1):
            c = sorted(glob.glob(f'data/history/{i:04d}-*-canonical.pt'))
            v = sorted(glob.glob(f'data/history/{i:04d}-*-v.pt'))
            p = sorted(glob.glob(f'data/history/{i:04d}-*-pi.pt'))
            for j in range(len(c)):
                datasets.append(TensorDataset(
                    torch.load(c[j]), torch.load(v[j]), torch.load(p[j])))

        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True,
                                num_workers=11, pin_memory=True)

        v_loss, pi_loss = nn.train(dataloader, min(250*5, len(dataloader)))
        nn.save_checkpoint('data/checkpoint', f'{iteration+1:04d}.pt')
        del nn
        del dataset
        del dataloader
        return v_loss, pi_loss

    def self_play(Game, nnargs, best, iteration, depth):
        bs = 512
        n = bs*12*4
        params = base_params(Game, SELF_PLAY_TEMP, bs)
        params.games_to_play = n
        params.mcts_depth = [depth] * Game.NUM_PLAYERS()
        params.self_play = True
        params.history_enabled = True
        params.add_noise = True
        params.playout_cap_randomization = True
        params.playout_cap_depth = 50
        params.playout_cap_percent = 0.75
        pm = alphazero.PlayManager(Game(), params)

        grargs = GRArgs(title='Self Play', game=Game, iteration=iteration,
                        max_batch_size=bs, concurrent_batches=CONCURRENT_BATCHES, result_workers=RESULT_WORKERS)
        nn = neural_net.NNWrapper(Game, nnargs)
        nn.load_checkpoint('data/checkpoint', f'{best:04d}.pt')
        players = []
        for _ in range(Game.NUM_PLAYERS()):
            players.append(nn)
        gr = GameRunner(players, pm, grargs)
        gr.run()
        p1_rate, draw_rate = calculate_win_draw_rates(
            n, pm.scores()[0], pm.draws())
        hits = pm.cache_hits()
        total = hits + pm.cache_misses()
        hr = hits/total
        agl = pm.avg_game_length()
        del pm
        del nn
        return p1_rate, draw_rate, hr, agl

    def play_past(Game, nnargs, depth, iteration, past_iter):
        nn_rate = 0
        draw_rate = 0
        hr = 0
        agl = 0
        nn = neural_net.NNWrapper(Game, nnargs)
        nn.load_checkpoint('data/checkpoint', f'{iteration:04d}.pt')
        nn_past = neural_net.NNWrapper(Game, nnargs)
        nn_past.load_checkpoint('data/checkpoint', f'{past_iter:04d}.pt')
        for i in range(Game.NUM_PLAYERS()):
            params = alphazero.PlayParams()
            bs = 64
            n = bs*4
            params = base_params(Game, EVAL_TEMP, bs)
            params.games_to_play = n
            params.mcts_depth = [depth] * Game.NUM_PLAYERS()
            pm = alphazero.PlayManager(Game(), params)

            grargs = GRArgs(title=f'Bench {iteration} v {past_iter}({i+1}/{Game.NUM_PLAYERS()})', game=Game, iteration=iteration,
                            max_batch_size=bs, concurrent_batches=CONCURRENT_BATCHES, result_workers=RESULT_WORKERS)
            players = []
            for _ in range(Game.NUM_PLAYERS()):
                players.append(nn_past)
            players[i] = nn
            gr = GameRunner(players, pm, grargs)
            gr.run()
            wr, dr = calculate_win_draw_rates(n, pm.scores()[i], pm.draws())
            nn_rate += wr
            draw_rate += dr
            hits = pm.cache_hits()
            total = hits + pm.cache_misses()
            hr += hits/total
            agl += pm.avg_game_length()
        del pm
        del nn
        del nn_past
        return nn_rate / Game.NUM_PLAYERS(), draw_rate / Game.NUM_PLAYERS(), hr / Game.NUM_PLAYERS(), agl / Game.NUM_PLAYERS()

    def play_rand(Game, nnargs, iteration, nn_depth, rand_depth):
        nn_rate = 0
        draw_rate = 0
        hr = 0
        agl = 0
        nn = neural_net.NNWrapper(Game, nnargs)
        nn.load_checkpoint('data/checkpoint', f'{iteration:04d}.pt')
        for i in range(Game.NUM_PLAYERS()):
            params = alphazero.PlayParams()
            bs = 128
            n = bs*4
            params = base_params(Game, EVAL_TEMP, bs)
            params.games_to_play = n
            mcts_depth = [rand_depth] * Game.NUM_PLAYERS()
            mcts_depth[i] = nn_depth
            params.mcts_depth = mcts_depth
            pm = alphazero.PlayManager(Game(), params)

            grargs = GRArgs(title=f'Bench Rand MCTS-{rand_depth}({i+1}/{Game.NUM_PLAYERS()})', game=Game, iteration=iteration,
                            max_batch_size=bs, concurrent_batches=CONCURRENT_BATCHES, result_workers=RESULT_WORKERS)
            rand = RandPlayer(Game, bs)
            players = []
            for _ in range(Game.NUM_PLAYERS()):
                players.append(rand)
            players[i] = nn
            gr = GameRunner(players, pm, grargs)
            gr.run()
            wr, dr = calculate_win_draw_rates(n, pm.scores()[i], pm.draws())
            nn_rate += wr
            draw_rate += dr
            hits = pm.cache_hits()
            total = hits + pm.cache_misses()
            hr += hits/total
            agl += pm.avg_game_length()
        del pm
        del nn
        return nn_rate / Game.NUM_PLAYERS(), draw_rate / Game.NUM_PLAYERS(), hr / Game.NUM_PLAYERS(), agl / Game.NUM_PLAYERS()

    def elo_prob(r1, r2):
        return 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (r1-r2) / 400))

    def get_elo(past_elo, win_rates, new_agent):
        # Everything but the newest agent is anchored for consitency.
        # Start with the assumption it is equal to the agent before it.
        if new_agent != 0:
            past_elo[new_agent] = past_elo[new_agent-1]
        iters = 5000
        for _ in tqdm.trange(iters, leave=False):
            mean_update = 0
            for j in range(past_elo.shape[0]):
                if not math.isnan(wr[new_agent, j]):
                    mean_update += wr[new_agent, j] - \
                        elo_prob(past_elo[j], past_elo[new_agent])
            past_elo[new_agent] += mean_update*32
        return past_elo

    start = 0
    iters = 200
    depth = 5
    channels = 32
    nn_selfplay_mcts_depth = 250
    nn_compare_mcts_depth = nn_selfplay_mcts_depth//2
    rand_mcts_depth = 400
    past_compares = [20]
    gating_percent = 0.52
    total_agents = iters+2  # + base and mcts
    mcts_agent = total_agents - 1

    run_name = f'c_{channels}_d_{depth}_kata_5x'
    nnargs = neural_net.NNArgs(
        num_channels=channels, depth=depth, lr_milestones=[150])
    Game = alphazero.Connect4GS

    if start == 0:
        create_init_net(Game, nnargs,)
        wr = np.empty((total_agents, total_agents))
        wr[:] = np.NAN
        elo = np.zeros(total_agents)
        current_best = 0
    else:
        wr = np.genfromtxt('data/win_rate.csv', delimiter=',')
        elo = np.genfromtxt('data/melo.csv', delimiter=',')
        current_best = np.argmax(elo[:mcts_agent])

    postfix = {'current best': 0, 'nn vs best': 0, 'best vs mcts': 0, 'elo': 0, 'p1 rate': 0, 'draw rate': 0,
               'v loss': 0, 'pi loss': 0}
    writer = SummaryWriter(f'runs/{run_name}')
    with tqdm.trange(start, iters, desc='Build Amazing Network') as pbar:
        for i in pbar:
            writer.add_scalar(
                f'Elo/Current Best', current_best, i)
            if i > 0:
                for past_compare in past_compares:
                    past_iter = max(0, i - past_compare)
                    nn_rate, draw_rate, hit_rate, game_length = play_past(
                        Game, nnargs, nn_compare_mcts_depth,  i, past_iter)
                    writer.add_scalar(
                        f'Win Rate/NN vs -{past_compare}', nn_rate, i)
                    writer.add_scalar(
                        f'Draw Rate/NN vs -{past_compare}', draw_rate, i)
                    writer.add_scalar(
                        f'Cache Hit Rate/NN vs -{past_compare}', hit_rate, i)
                    writer.add_scalar(
                        f'Average Game Length/NN vs -{past_compare}', game_length, i)
                    wr[i, past_iter] = nn_rate
                    wr[past_iter, i] = 1-nn_rate
                    if i == current_best:
                        writer.add_scalar(
                            f'Win Rate/Best vs -{past_compare}', nn_rate, i)
                        writer.add_scalar(
                            f'Draw Rate/Best vs -{past_compare}', draw_rate, i)
                        writer.add_scalar(
                            f'Cache Hit Rate/Best vs -{past_compare}', hit_rate, i)
                        writer.add_scalar(
                            f'Average Game Length/Best vs -{past_compare}', game_length, i)
                    gc.collect()

            nn_rate, draw_rate, hit_rate, game_length = play_rand(
                Game, nnargs, i, nn_compare_mcts_depth, rand_mcts_depth)
            wr[i, mcts_agent] = nn_rate
            wr[mcts_agent, i] = 1-nn_rate
            elo = get_elo(elo, wr, i)
            if i == 0:
                # Anchor first net to zero.
                elo[mcts_agent] -= elo[0]
                elo[0] -= elo[0]
            postfix['elo'] = elo[i]
            pbar.set_postfix(postfix)
            writer.add_scalar(
                f'Win Rate/NN vs MCTS-{rand_mcts_depth}', nn_rate, i)
            writer.add_scalar(
                f'Draw Rate/NN vs MCTS-{rand_mcts_depth}', draw_rate, i)
            writer.add_scalar(
                f'Cache Hit Rate/NN vs MCTS-{rand_mcts_depth}', hit_rate, i)
            writer.add_scalar(
                f'Average Game Length/NN vs MCTS-{rand_mcts_depth}', game_length, i)
            writer.add_scalar(
                f'Elo/NN', elo[i], i)
            if i == current_best:
                writer.add_scalar(
                    f'Elo/Best', elo[i], i)
                writer.add_scalar(
                    f'Win Rate/Best vs MCTS-{rand_mcts_depth}', nn_rate, i)
                writer.add_scalar(
                    f'Draw Rate/Best vs MCTS-{rand_mcts_depth}', draw_rate, i)
                writer.add_scalar(
                    f'Cache Hit Rate/Best vs MCTS-{rand_mcts_depth}', hit_rate, i)
                writer.add_scalar(
                    f'Average Game Length/Best vs MCTS-{rand_mcts_depth}', game_length, i)
                postfix['best vs mcts'] = nn_rate
                pbar.set_postfix(postfix)
            gc.collect()

            p1_rate, draw_rate, hit_rate, game_length = self_play(
                Game, nnargs, current_best, i, nn_selfplay_mcts_depth)
            writer.add_scalar('Win Rate/Self Play P1', p1_rate, i)
            writer.add_scalar('Draw Rate/Self Play', draw_rate, i)
            writer.add_scalar('Cache Hit Rate/Self Play', hit_rate, i)
            writer.add_scalar('Average Game Length/Self Play', game_length, i)
            postfix['p1 rate'] = p1_rate
            postfix['draw rate'] = draw_rate
            pbar.set_postfix(postfix)
            gc.collect()

            hist_size = calc_hist_size(i)
            writer.add_scalar('History Size', hist_size, i)
            v_loss, pi_loss = train(Game, nnargs, i, hist_size)
            writer.add_scalar('Loss/V', v_loss, i)
            writer.add_scalar('Loss/Pi', pi_loss, i)
            writer.add_scalar('Loss/Total', v_loss+pi_loss, i)
            postfix['v loss'] = v_loss
            postfix['pi loss'] = pi_loss
            pbar.set_postfix(postfix)
            gc.collect()

            # Eval for gating
            next_net = i + 1
            nn_rate, draw_rate, hit_rate, game_length = play_past(
                Game, nnargs, nn_compare_mcts_depth, next_net, current_best)
            wr[next_net, current_best] = nn_rate
            wr[current_best, next_net] = 1-nn_rate
            writer.add_scalar(
                f'Win Rate/NN vs Best', nn_rate, next_net)
            writer.add_scalar(
                f'Draw Rate/NN vs Best', draw_rate, next_net)
            writer.add_scalar(
                f'Cache Hit Rate/NN vs Best', hit_rate, next_net)
            writer.add_scalar(
                f'Average Game Length/NN vs Best', game_length, next_net)
            postfix['nn vs best'] = nn_rate
            pbar.set_postfix(postfix)
            if nn_rate > gating_percent:
                current_best = next_net
                postfix['current best'] = current_best
                pbar.set_postfix(postfix)
            gc.collect()
            np.savetxt("data/win_rate.csv", wr, delimiter=",")
            np.savetxt("data/elo.csv", elo, delimiter=",")

    writer.close()
