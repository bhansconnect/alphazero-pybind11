import glob
import importlib.util
import os
from collections import namedtuple
import math
import random
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

HIST_SIZE = 30000
HIST_LOCATION = 'data/history'
TMP_HIST_LOCATION = 'data/tmp_history'
GRArgs = namedtuple(
    'GRArgs', ['title', 'game', 'max_batch_size', 'iteration',  'data_save_size', 'data_folder', 'concurrent_batches', 'batch_workers', 'nn_workers', 'result_workers', 'mcts_workers', 'cuda'], defaults=(0, HIST_SIZE, TMP_HIST_LOCATION, 0, 0, 1, 1, os.cpu_count() - 1, torch.cuda.is_available()))


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
            self.args.data_save_size, self.num_players+1)
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
        mcts_workers = []
        for i in range(self.args.mcts_workers):
            mcts_workers.append(threading.Thread(
                target=self.pm.play))
            mcts_workers[i].start()

        monitor = threading.Thread(target=self.monitor)
        monitor.start()
        if self.pm.params().history_enabled:
            hist_saver = threading.Thread(target=self.hist_saver)
            hist_saver.start()

        for bw in batch_workers:
            bw.join()
        for rw in result_workers:
            rw.join()
        for pw in player_workers:
            pw.join()
        for mw in mcts_workers:
            mw.join()
        monitor.join()
        if self.pm.params().history_enabled:
            hist_saver.join()

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
                completed = self.pm.games_completed()
                scores = self.pm.scores()
                win_rates = [0] * len(scores)
                if completed > 0:
                    for i in range(len(scores)-1):
                        win_rates[i] = (scores[i] + scores[-1] /
                                        self.num_players)/completed
                    win_rates[-1] = scores[-1]/completed
                win_rates = list(map(lambda x: f'{x:0.3f}', win_rates))
                pbar.set_postfix({
                    'win rates': win_rates,
                    'cache rate': hr})
                pbar.update(completed-last_completed)
                last_completed = completed
                last_update = time.time()
        hr = 0
        hits = self.pm.cache_hits()
        total = hits + self.pm.cache_misses()
        if total > 0:
            hr = hits/total
        scores = self.pm.scores()
        win_rates = [0] * len(scores)
        for i in range(len(scores)-1):
            win_rates[i] = (scores[i] + scores[-1] /
                            self.num_players)/completed
        win_rates[-1] = scores[-1]/completed
        win_rates = list(map(lambda x: f'{x:0.3f}', win_rates))
        pbar.set_postfix({
            'win rates': win_rates,
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

            cs = self.args.game.CANONICAL_SHAPE()
            c_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                f'{self.args.data_folder}/{self.args.iteration:04d}-{batch:04d}-canonical-{size}.pt', shared=True, size=size*cs[0]*cs[1]*cs[2])).reshape(size, cs[0], cs[1], cs[2])
            v_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                f'{self.args.data_folder}/{self.args.iteration:04d}-{batch:04d}-v-{size}.pt', shared=True, size=size*(self.num_players+1))).reshape(size, self.num_players+1)
            p_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                f'{self.args.data_folder}/{self.args.iteration:04d}-{batch:04d}-pi-{size}.pt', shared=True, size=size*(self.args.game.NUM_MOVES()))).reshape(size, self.args.game.NUM_MOVES())
            c_tensor[:] = self.hist_canonical[:size]
            v_tensor[:] = self.hist_v[:size]
            p_tensor[:] = self.hist_pi[:size]
            self.saved_samples += size
            batch += 1


class RandPlayer:
    def __init__(self, game, max_batch_size):
        self.v = torch.ones((max_batch_size, game.NUM_PLAYERS()+1))
        self.v /= torch.sum(self.v)
        self.pi = torch.ones((max_batch_size, game.NUM_MOVES()))
        self.pi /= torch.sum(self.pi)

    def predict(self, canonical):
        v, pi = self.process(canonical.unsqueeze(0))
        return v[0], pi[0]

    def process(self, batch):
        return self.v[:batch.shape[0]], self.pi[:batch.shape[0]]


EXPECTED_OPENING_LENGTH = 6
CPUCT = 2
SELF_PLAY_TEMP = 1
EVAL_TEMP = 0.5
TEMP_DECAY_HALF_LIFE = EXPECTED_OPENING_LENGTH
FINAL_TEMP = 0.2
MAX_CACHE_SIZE = 200000

# To decide on the following numbers, I would advise graphing the equation: scalar*(1+beta*(((iter+1)/scalar)**alpha-1)/alpha)
WINDOW_SIZE_ALPHA = 0.5  # This decides how fast the curve flattens to a max
WINDOW_SIZE_BETA = 0.7  # This decides the rough overall slope.
WINDOW_SIZE_SCALAR = 6  # This ends up being approximately first time history doesn't grow

RESULT_WORKERS = 2


def base_params(Game, start_temp, bs, cb):
    params = alphazero.PlayParams()
    params.max_cache_size = MAX_CACHE_SIZE
    params.cpuct = CPUCT
    params.start_temp = start_temp
    params.temp_decay_half_life = TEMP_DECAY_HALF_LIFE
    params.final_temp = FINAL_TEMP
    params.max_batch_size = bs
    params.concurrent_games = bs * cb
    return params


if __name__ == '__main__':
    import shutil
    import neural_net

    def create_init_net(Game, nnargs):
        nn = neural_net.NNWrapper(Game, nnargs)
        nn.save_checkpoint('data/checkpoint', '0000.pt')

    def calc_hist_size(i):
        return int(WINDOW_SIZE_SCALAR*(1 + WINDOW_SIZE_BETA*(((i+1)/WINDOW_SIZE_SCALAR)**WINDOW_SIZE_ALPHA-1)/WINDOW_SIZE_ALPHA))

    def resample_by_surprise(Game, nnargs, iteration):
        # Used to resample the latest iteration by how surprising each sample is.
        # Each sample is given 0.5 weight as a base.
        # The other half of the weight is distributed based on the sample loss.
        # The sample is then added to the dataset floor(weight) times.
        # It is also added an extra time with the probability of weight - floor(weight)
        nn = neural_net.NNWrapper(Game, nnargs)
        nn.load_checkpoint('data/checkpoint', f'{iteration:04d}.pt')

        c = sorted(
            glob.glob(f'{TMP_HIST_LOCATION}/{iteration:04d}-*-canonical-*.pt'))
        v = sorted(glob.glob(f'{TMP_HIST_LOCATION}/{iteration:04d}-*-v-*.pt'))
        p = sorted(glob.glob(f'{TMP_HIST_LOCATION}/{iteration:04d}-*-pi-*.pt'))

        datasets = []
        for j in range(len(c)):
            size = int(c[j].split('-')[-1].split('.')[0])
            cs = Game.CANONICAL_SHAPE()
            c_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                c[j], shared=False, size=size*cs[0]*cs[1]*cs[2])).reshape(size, cs[0], cs[1], cs[2])
            v_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                v[j], shared=False, size=size*(Game.NUM_PLAYERS()+1))).reshape(size, Game.NUM_PLAYERS()+1)
            p_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                p[j], shared=False, size=size*(Game.NUM_MOVES()))).reshape(size, Game.NUM_MOVES())
            datasets.append(TensorDataset(c_tensor, v_tensor, p_tensor))
            del c_tensor
            del v_tensor
            del p_tensor

        dataset = ConcatDataset(datasets)
        sample_count = len(dataset)
        dataloader = DataLoader(dataset, batch_size=512,
                                shuffle=False, num_workers=11)
        loss = nn.sample_loss(dataloader, sample_count)
        total_loss = np.sum(loss)

        i_out = 0
        batch_out = 0
        cs = Game.CANONICAL_SHAPE()
        c_out = torch.zeros(
            HIST_SIZE, cs[0], cs[1], cs[2])
        v_out = torch.zeros(
            HIST_SIZE, Game.NUM_PLAYERS()+1)
        p_out = torch.zeros(
            HIST_SIZE, Game.NUM_MOVES())
        os.makedirs(HIST_LOCATION, exist_ok=True)

        def maybe_save(c, v, p, size, batch, force=False):
            if size == HIST_SIZE or (force and size > 0):
                c_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                    f'{HIST_LOCATION}/{iteration:04d}-{batch:04d}-canonical-{size}.pt', shared=True, size=size*cs[0]*cs[1]*cs[2])).reshape(size, cs[0], cs[1], cs[2])
                v_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                    f'{HIST_LOCATION}/{iteration:04d}-{batch:04d}-v-{size}.pt', shared=True, size=size*(Game.NUM_PLAYERS()+1))).reshape(size, Game.NUM_PLAYERS()+1)
                p_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                    f'{HIST_LOCATION}/{iteration:04d}-{batch:04d}-pi-{size}.pt', shared=True, size=size*(Game.NUM_MOVES()))).reshape(size, Game.NUM_MOVES())
                c_tensor[:] = c[:size]
                v_tensor[:] = v[:size]
                p_tensor[:] = p[:size]
                return True
            return False

        for i in range(sample_count):
            sample_weight = 0.5 + (loss[i]/total_loss) * 0.5 * sample_count
            for _ in range(math.floor(sample_weight)):
                c, v, pi = dataset[i]
                c_out[i_out] = c
                v_out[i_out] = v
                p_out[i_out] = pi
                i_out += 1
                if maybe_save(c_out, v_out, p_out, i_out, batch_out):
                    i_out = 0
                    batch_out += 1
            if random.random() < sample_weight - math.floor(sample_weight):
                c, v, pi = dataset[i]
                c_out[i_out] = c
                v_out[i_out] = v
                p_out[i_out] = pi
                i_out += 1
                if maybe_save(c_out, v_out, p_out, i_out, batch_out):
                    i_out = 0
                    batch_out += 1

        maybe_save(c_out, v_out, p_out, i_out, batch_out, force=True)

        del datasets[:]
        del dataset
        del dataloader
        del nn
        del c_out
        del v_out
        del p_out

        for fn in glob.glob(f'{TMP_HIST_LOCATION}/*'):
            os.remove(fn)

    def train(Game, nnargs, iteration, hist_size):
        nn = neural_net.NNWrapper(Game, nnargs)
        nn.load_checkpoint('data/checkpoint', f'{iteration:04d}.pt')

        datasets = []
        for i in range(max(0, iteration - hist_size), iteration + 1):
            c = sorted(glob.glob(f'{HIST_LOCATION}/{i:04d}-*-canonical-*.pt'))
            v = sorted(glob.glob(f'{HIST_LOCATION}/{i:04d}-*-v-*.pt'))
            p = sorted(glob.glob(f'{HIST_LOCATION}/{i:04d}-*-pi-*.pt'))
            for j in range(len(c)):
                size = int(c[j].split('-')[-1].split('.')[0])
                cs = Game.CANONICAL_SHAPE()
                c_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                    c[j], shared=False, size=size*cs[0]*cs[1]*cs[2])).reshape(size, cs[0], cs[1], cs[2])
                v_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                    v[j], shared=False, size=size*(Game.NUM_PLAYERS()+1))).reshape(size, Game.NUM_PLAYERS()+1)
                p_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                    p[j], shared=False, size=size*(Game.NUM_MOVES()))).reshape(size, Game.NUM_MOVES())
                datasets.append(TensorDataset(c_tensor, v_tensor, p_tensor))
                del c_tensor
                del v_tensor
                del p_tensor

        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=512,
                                shuffle=True, num_workers=11)

        v_loss, pi_loss = nn.train(dataloader, 250*4)
        nn.save_checkpoint('data/checkpoint', f'{iteration+1:04d}.pt')
        del datasets[:]
        del dataset
        del dataloader
        del nn
        return v_loss, pi_loss

    def self_play(Game, nnargs, best, iteration, depth, fast_depth):
        bs = 512
        cb = Game.NUM_PLAYERS()*2
        n = bs*cb*12
        params = base_params(Game, SELF_PLAY_TEMP, bs, cb)
        params.games_to_play = n
        params.mcts_depth = [depth] * Game.NUM_PLAYERS()
        params.self_play = True
        params.history_enabled = True
        params.add_noise = True
        params.playout_cap_randomization = True
        params.playout_cap_depth = fast_depth
        params.playout_cap_percent = 0.75
        pm = alphazero.PlayManager(Game(), params)

        grargs = GRArgs(title='Self Play', game=Game, iteration=iteration,
                        max_batch_size=bs, concurrent_batches=cb, result_workers=RESULT_WORKERS)
        nn = neural_net.NNWrapper(Game, nnargs)
        nn.load_checkpoint('data/checkpoint', f'{best:04d}.pt')
        players = []
        for _ in range(Game.NUM_PLAYERS()):
            players.append(nn)
        gr = GameRunner(players, pm, grargs)
        gr.run()
        scores = pm.scores()
        win_rates = [0] * len(scores)
        for i in range(len(scores)-1):
            win_rates[i] = (scores[i] + scores[-1] /
                            Game.NUM_PLAYERS())/n
        win_rates[-1] = scores[-1]/n
        hits = pm.cache_hits()
        total = hits + pm.cache_misses()
        hr = 0
        if total > 0:
            hr = hits/total
        agl = pm.avg_game_length()
        del pm
        del nn
        return win_rates, hr, agl

    def play_past(Game, nnargs, depth, iteration, past_iter):
        nn_rate = 0
        draw_rate = 0
        hr = 0
        agl = 0
        nn = neural_net.NNWrapper(Game, nnargs)
        nn.load_checkpoint('data/checkpoint', f'{iteration:04d}.pt')
        nn_past = neural_net.NNWrapper(Game, nnargs)
        nn_past.load_checkpoint('data/checkpoint', f'{past_iter:04d}.pt')
        cb = Game.NUM_PLAYERS()
        if Game.NUM_PLAYERS() > 2:
            bs = 16
            n = bs*cb
            for i in tqdm.trange(Game.NUM_PLAYERS(), leave=False, desc=f"Bench 1 new vs {Game.NUM_PLAYERS() - 1} old"):
                params = base_params(Game, EVAL_TEMP, bs, cb)
                params.games_to_play = n
                params.mcts_depth = [depth] * Game.NUM_PLAYERS()
                pm = alphazero.PlayManager(Game(), params)

                grargs = GRArgs(title=f'Bench {iteration} v {past_iter} as p{i+1}', game=Game, iteration=iteration,
                                max_batch_size=bs, concurrent_batches=cb, result_workers=RESULT_WORKERS)
                players = []
                for _ in range(Game.NUM_PLAYERS()):
                    players.append(nn_past)
                players[i] = nn
                gr = GameRunner(players, pm, grargs)
                gr.run()
                scores = pm.scores()
                nn_rate += (scores[i] + scores[-1]/Game.NUM_PLAYERS())/n
                draw_rate += scores[-1]/n
                hits = pm.cache_hits()
                total = hits + pm.cache_misses()
                if total > 0:
                    hr += hits/total
                agl += pm.avg_game_length()
                del pm
                gc.collect()
            for i in tqdm.trange(Game.NUM_PLAYERS(), leave=False, desc=f"Bench {Game.NUM_PLAYERS() - 1} new vs 1 old"):
                params = base_params(Game, EVAL_TEMP, bs, cb)
                params.games_to_play = n
                params.mcts_depth = [depth] * Game.NUM_PLAYERS()
                pm = alphazero.PlayManager(Game(), params)

                grargs = GRArgs(title=f'Bench {iteration} v {past_iter} as p{i+1}', game=Game, iteration=iteration,
                                max_batch_size=bs, concurrent_batches=cb, result_workers=RESULT_WORKERS)
                players = []
                for _ in range(Game.NUM_PLAYERS()):
                    players.append(nn)
                players[i] = nn_past
                gr = GameRunner(players, pm, grargs)
                gr.run()
                scores = pm.scores()
                for j in range(1, Game.NUM_PLAYERS()):
                    nn_rate += (scores[(i+j) % Game.NUM_PLAYERS()] +
                                scores[-1]/Game.NUM_PLAYERS())/n
                draw_rate += scores[-1]/n
                hits = pm.cache_hits()
                total = hits + pm.cache_misses()
                if total > 0:
                    hr += hits/total
                agl += pm.avg_game_length()
                del pm
                gc.collect()
            nn_rate /= 2*Game.NUM_PLAYERS()
            draw_rate /= 2*Game.NUM_PLAYERS()
            hr /= 2*Game.NUM_PLAYERS()
            agl /= 2*Game.NUM_PLAYERS()
        else:
            bs = 64
            n = bs*cb
            for i in tqdm.trange(Game.NUM_PLAYERS(), leave=False, desc=f"Bench new vs old"):
                params = base_params(Game, EVAL_TEMP, bs, cb)
                params.games_to_play = n
                params.mcts_depth = [depth] * Game.NUM_PLAYERS()
                pm = alphazero.PlayManager(Game(), params)

                grargs = GRArgs(title=f'Bench {iteration} v {past_iter} as p{i+1}', game=Game, iteration=iteration,
                                max_batch_size=bs, concurrent_batches=cb, result_workers=RESULT_WORKERS)
                players = []
                for _ in range(Game.NUM_PLAYERS()):
                    players.append(nn_past)
                players[i] = nn
                gr = GameRunner(players, pm, grargs)
                gr.run()
                scores = pm.scores()
                nn_rate += (scores[i] + scores[-1]/Game.NUM_PLAYERS())/n
                draw_rate += scores[-1]/n
                hits = pm.cache_hits()
                total = hits + pm.cache_misses()
                if total > 0:
                    hr += hits/total
                agl += pm.avg_game_length()
                del pm
                gc.collect()
            nn_rate /= Game.NUM_PLAYERS()
            draw_rate /= Game.NUM_PLAYERS()
            hr /= Game.NUM_PLAYERS()
            agl /= Game.NUM_PLAYERS()

        del nn
        del nn_past
        return nn_rate, draw_rate, hr, agl

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

    bootstrap_iters = 0
    start = 0
    iters = 200
    depth = 5
    channels = 32
    nn_selfplay_mcts_depth = 250
    nn_selfplay_fast_mcts_depth = 50
    nn_compare_mcts_depth = nn_selfplay_mcts_depth//2
    compare_past = 20
    gating_percent = 0.52
    total_agents = iters+1  # + base

    run_name = f'c_{channels}_d_{depth}'
    writer = SummaryWriter(f'runs/{run_name}')
    nnargs = neural_net.NNArgs(
        num_channels=channels, depth=depth, lr_milestone=150)
    Game = alphazero.Connect4GS

    if start == 0:
        create_init_net(Game, nnargs,)
        wr = np.empty((total_agents, total_agents))
        wr[:] = np.NAN
        elo = np.zeros(total_agents)
        current_best = 0
    else:
        tmp_wr = np.genfromtxt('data/win_rate.csv', delimiter=',')
        wr = np.zeros_like(tmp_wr)
        wr[:start+1][:start+1] = tmp_wr[:start+1][:start+1]
        tmp_elo = np.genfromtxt('data/elo.csv', delimiter=',')
        elo = np.zeros_like(tmp_elo)
        elo[:start+1] = tmp_elo[:start+1]
        current_best = np.argmax(elo[:start+1])

    postfix = {'best': current_best}
    if bootstrap_iters > 0 and bootstrap_iters > start:
        # We are just going to assume the new nets have similar elo to the past instead of running many comparisons matches.
        prev_elo = np.genfromtxt('data/elo.csv', delimiter=',')
        prev_wr = np.genfromtxt('data/win_rate.csv', delimiter=',')
        with tqdm.trange(start, bootstrap_iters, desc='Bootstraping Network') as pbar:
            for i in pbar:
                elo[i] = prev_elo[i]
                wr[i][:bootstrap_iters] = prev_wr[i][:bootstrap_iters]
                hist_size = calc_hist_size(i)
                v_loss, pi_loss = train(Game, nnargs, i, hist_size)
                writer.add_scalar('Loss/V', v_loss, i)
                writer.add_scalar('Loss/Pi', pi_loss, i)
                writer.add_scalar('Loss/Total', v_loss+pi_loss, i)
                postfix['vloss'] = v_loss
                postfix['ploss'] = pi_loss
                pbar.set_postfix(postfix)
                gc.collect()
        current_best = bootstrap_iters
        start = bootstrap_iters
        postfix['best'] = current_best

    with tqdm.trange(start, iters, desc='Build Amazing Network') as pbar:
        for i in pbar:
            writer.add_scalar(
                f'Misc/Current Best', current_best, i)
            past_iter = max(0, i - compare_past)
            if past_iter != current_best:
                nn_rate, draw_rate, hit_rate, game_length = play_past(
                    Game, nnargs, nn_compare_mcts_depth,  i, past_iter)
                wr[i, past_iter] = nn_rate
                wr[past_iter, i] = 1-nn_rate
                writer.add_scalar(
                    f'Win Rate/NN vs -{compare_past}', nn_rate, i)
                writer.add_scalar(
                    f'Draw Rate/NN vs -{compare_past}', draw_rate, i)
                writer.add_scalar(
                    f'Cache Hit Rate/NN vs -{compare_past}', hit_rate, i)
                writer.add_scalar(
                    f'Average Game Length/NN vs -{compare_past}', game_length, i)
                if i == current_best:
                    writer.add_scalar(
                        f'Win Rate/Best vs -{compare_past}', nn_rate, i)
                    writer.add_scalar(
                        f'Draw Rate/Best vs -{compare_past}', draw_rate, i)
                    writer.add_scalar(
                        f'Cache Hit Rate/Best vs -{compare_past}', hit_rate, i)
                    writer.add_scalar(
                        f'Average Game Length/Best vs -{compare_past}', game_length, i)
                postfix[f'vs -{compare_past}'] = nn_rate
                gc.collect()

            elo = get_elo(elo, wr, i)
            writer.add_scalar(
                f'Elo/NN', elo[i], i)
            if i == current_best:
                writer.add_scalar(
                    f'Elo/Best', elo[i], i)
            postfix['elo'] = int(elo[i])
            pbar.set_postfix(postfix)
            np.savetxt("data/elo.csv", elo, delimiter=",")

            win_rates, hit_rate, game_length = self_play(
                Game, nnargs, current_best, i, nn_selfplay_mcts_depth, nn_selfplay_fast_mcts_depth)
            for j in range(len(win_rates)-1):
                writer.add_scalar(
                    f'Win Rate/Self Play P{j+1}', win_rates[j], i)
            writer.add_scalar('Draw Rate/Self Play', win_rates[-1], i)
            writer.add_scalar('Cache Hit Rate/Self Play', hit_rate, i)
            writer.add_scalar(
                'Average Game Length/Self Play', game_length, i)
            postfix['win_rates'] = list(map(lambda x: f'{x:0.3f}', win_rates))
            pbar.set_postfix(postfix)
            gc.collect()

            resample_by_surprise(Game, nnargs, i)
            gc.collect()

            hist_size = calc_hist_size(i)
            writer.add_scalar('Misc/History Size', hist_size, i)
            v_loss, pi_loss = train(Game, nnargs, i, hist_size)
            writer.add_scalar('Loss/V', v_loss, i)
            writer.add_scalar('Loss/Pi', pi_loss, i)
            writer.add_scalar('Loss/Total', v_loss+pi_loss, i)
            postfix['vloss'] = v_loss
            postfix['ploss'] = pi_loss
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
            postfix['vs best'] = nn_rate
            pbar.set_postfix(postfix)
            if nn_rate > gating_percent:
                current_best = next_net
                postfix['best'] = current_best
                pbar.set_postfix(postfix)
            gc.collect()
            np.savetxt("data/win_rate.csv", wr, delimiter=",")

    writer.close()
