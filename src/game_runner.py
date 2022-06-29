import glob
import os
from collections import namedtuple
import math
import random
import time
import torch
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import aim
import threading
import tqdm
import queue
import numpy as np
import gc
from load_lib import load_alphazero

alphazero = load_alphazero()

HIST_SIZE = 30_000
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
                    for i in range(len(scores)):
                        win_rates[i] = scores[i]/completed
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
        completed = self.pm.games_completed()
        scores = self.pm.scores()
        win_rates = [0] * len(scores)
        for i in range(len(scores)):
            win_rates[i] = scores[i]/completed
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
        self.v = torch.full(
            (max_batch_size, game.NUM_PLAYERS()+1), 1.0/(game.NUM_PLAYERS()+1))
        self.pi = torch.full(
            (max_batch_size, game.NUM_MOVES()), 1.0/game.NUM_MOVES())

    def predict(self, canonical):
        v, pi = self.process(canonical.unsqueeze(0))
        return v[0], pi[0]

    def process(self, batch):
        return self.v[:batch.shape[0]], self.pi[:batch.shape[0]]


EXPECTED_OPENING_LENGTH = 10
CPUCT = 1.25
SELF_PLAY_TEMP = 1.0
EVAL_TEMP = 0.5
TEMP_DECAY_HALF_LIFE = EXPECTED_OPENING_LENGTH
FINAL_TEMP = 0.2
FPU_REDUCTION = 0.25
MAX_CACHE_SIZE = 200_000

# Concurrent games played is batch size * num player * concurrent batch mult
# Total games per iteration is batch size * num players * concurrent batch mult * chunks
SELF_PLAY_BATCH_SIZE = 1024
SELF_PLAY_CONCURRENT_BATCH_MULT = 2
SELF_PLAY_CHUNKS = 4

TRAIN_BATCH_SIZE = 1024
# Note: If the game has a high number of symetries generated, this number should likely get lowered.
TRAIN_SAMPLE_RATE = 1

# To decide on the following numbers, I would advise graphing the equation: scalar*(1+beta*(((iter+1)/scalar)**alpha-1)/alpha)
WINDOW_SIZE_ALPHA = 0.5  # This decides how fast the curve flattens to a max
WINDOW_SIZE_BETA = 0.7  # This decides the rough overall slope.
WINDOW_SIZE_SCALAR = 6  # This ends up being approximately first time history doesn't grow

RESULT_WORKERS = 2

# Panel based gating has the network play against multiple previous best agents before being promoted.
# This is muhch more imortant with games where the draw rate is high betwen new networks and the best.
# It is also important in grame that lead to rock-paper-scissor type network oscillations.
GATING_PANEL_SIZE = 1
# Ensure it is at least this good against the entire panel of networks.
GATING_PANEL_WIN_RATE = 0.52
# Ensure it is at least this good against the best network.
# Generally it is ok to be slightly worse than the best if you crush the panel. Especially in high draw games.
GATING_BEST_WIN_RATE = 0.52

# A win/loss/draw will happen if it has a lower percent than this to not happen.
# EX: 0.02 means that if the chance to draw is greater than 98% it will automatically happen.
# This must be zero if there are more than 2 players.
RESIGN_PERCENT = 0.02
# The percent of resignations that will be played to the end anyway.
RESIGN_PLAYTHROUGH_PERCENT = 0.20

bootstrap_iters = 0
start = 0
iters = 200
depth = 4
channels = 12
kernel_size = 5
dense_net = True
network_name = 'densenet' if dense_net else 'resnet'
nn_selfplay_mcts_depth = 450
nn_selfplay_fast_mcts_depth = 75
nn_compare_mcts_depth = nn_selfplay_mcts_depth//2
compare_past = 20
lr_milestone = 150

Game = alphazero.Connect4GS
game_name = 'connect4'

run_name = f'{game_name}-{network_name}-{depth}d-{channels}c-{kernel_size}k-{nn_selfplay_mcts_depth}sims'

# When you change game, define initialization here.
# For example some games could change version or exact ruleset here.


def new_game():
    return Game()


def base_params(Game, start_temp, bs, cb):
    params = alphazero.PlayParams()
    params.max_cache_size = MAX_CACHE_SIZE
    params.cpuct = CPUCT
    params.start_temp = start_temp
    params.temp_decay_half_life = TEMP_DECAY_HALF_LIFE
    params.final_temp = FINAL_TEMP
    params.max_batch_size = bs
    params.concurrent_games = bs * cb
    params.fpu_reduction = FPU_REDUCTION
    return params


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
            if not math.isnan(win_rates[new_agent, j]):
                rate = win_rates[new_agent, j]
                rate = max(0.001, rate)
                rate = min(0.999, rate)
                mean_update += rate - \
                    elo_prob(past_elo[j], past_elo[new_agent])
        past_elo[new_agent] += mean_update*32
    return past_elo


if __name__ == '__main__':
    import shutil
    import neural_net

    def create_init_net(Game, nnargs):
        nn = neural_net.NNWrapper(Game, nnargs)
        nn.save_checkpoint('data/checkpoint', f'0000-{run_name}.pt')

    def calc_hist_size(i):
        return int(WINDOW_SIZE_SCALAR*(1 + WINDOW_SIZE_BETA*(((i+1)/WINDOW_SIZE_SCALAR)**WINDOW_SIZE_ALPHA-1)/WINDOW_SIZE_ALPHA))

    def maybe_save(Game, c, v, p, size, batch, iteration, location=HIST_LOCATION, name='', force=False):
        cs = Game.CANONICAL_SHAPE()
        if size == HIST_SIZE or (force and size > 0):
            c_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                f'{location}/{iteration:04d}-{batch:04d}{name}-canonical-{size}.pt', shared=True, size=size*cs[0]*cs[1]*cs[2])).reshape(size, cs[0], cs[1], cs[2])
            v_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                f'{location}/{iteration:04d}-{batch:04d}{name}-v-{size}.pt', shared=True, size=size*(Game.NUM_PLAYERS()+1))).reshape(size, Game.NUM_PLAYERS()+1)
            p_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                f'{location}/{iteration:04d}-{batch:04d}{name}-pi-{size}.pt', shared=True, size=size*(Game.NUM_MOVES()))).reshape(size, Game.NUM_MOVES())
            c_tensor[:] = c[:size]
            v_tensor[:] = v[:size]
            p_tensor[:] = p[:size]
            return True
        return False

    def exploit_symmetries(Game, iteration):
        # In games with symmetries, create symmetric samples of the data.
        if Game.NUM_SYMMETRIES() <= 1:
            return

        c_names = sorted(
            glob.glob(f'{TMP_HIST_LOCATION}/{iteration:04d}-*-canonical-*.pt'))
        v_names = sorted(
            glob.glob(f'{TMP_HIST_LOCATION}/{iteration:04d}-*-v-*.pt'))
        p_names = sorted(
            glob.glob(f'{TMP_HIST_LOCATION}/{iteration:04d}-*-pi-*.pt'))

        datasets = []
        for j in range(len(c_names)):
            size = int(c_names[j].split('-')[-1].split('.')[0])
            cs = Game.CANONICAL_SHAPE()
            c_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                c_names[j], shared=False, size=size*cs[0]*cs[1]*cs[2])).reshape(size, cs[0], cs[1], cs[2])
            v_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                v_names[j], shared=False, size=size*(Game.NUM_PLAYERS()+1))).reshape(size, Game.NUM_PLAYERS()+1)
            p_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                p_names[j], shared=False, size=size*(Game.NUM_MOVES()))).reshape(size, Game.NUM_MOVES())
            datasets.append(TensorDataset(c_tensor, v_tensor, p_tensor))
            del c_tensor
            del v_tensor
            del p_tensor

        dataset = ConcatDataset(datasets)
        sample_count = len(dataset)
        dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE,
                                shuffle=False, num_workers=11)

        i_out = 0
        batch_out = 0
        cs = Game.CANONICAL_SHAPE()
        c_out = torch.zeros(
            HIST_SIZE, cs[0], cs[1], cs[2])
        v_out = torch.zeros(
            HIST_SIZE, Game.NUM_PLAYERS()+1)
        p_out = torch.zeros(
            HIST_SIZE, Game.NUM_MOVES())

        for i in tqdm.trange(sample_count, desc='Creating Symmetric Samples', leave=False):
            c, v, pi = dataset[i]
            ph = alphazero.PlayHistory(c, v, pi)
            syms = new_game().symmetries(ph)
            for sym in syms:
                c_out[i_out] = torch.from_numpy(np.array(sym.canonical()))
                v_out[i_out] = torch.from_numpy(np.array(sym.v()))
                p_out[i_out] = torch.from_numpy(np.array(sym.pi()))
                i_out += 1
                if maybe_save(Game, c_out, v_out, p_out, i_out, batch_out, iteration, location=TMP_HIST_LOCATION, name='syms'):
                    i_out = 0
                    batch_out += 1
        maybe_save(Game, c_out, v_out, p_out, i_out,
                   batch_out, iteration, location=TMP_HIST_LOCATION, name='syms', force=True)

        del datasets[:]
        del dataset
        del dataloader
        del c_out
        del v_out
        del p_out

        for fn in c_names + v_names + p_names:
            os.remove(fn)

    def resample_by_surprise(Game, iteration):
        # Used to resample the latest iteration by how surprising each sample is.
        # Each sample is given 0.5 weight as a base.
        # The other half of the weight is distributed based on the sample loss.
        # The sample is then added to the dataset floor(weight) times.
        # It is also added an extra time with the probability of weight - floor(weight)
        c_names = sorted(
            glob.glob(f'{TMP_HIST_LOCATION}/{iteration:04d}-*-canonical-*.pt'))
        v_names = sorted(
            glob.glob(f'{TMP_HIST_LOCATION}/{iteration:04d}-*-v-*.pt'))
        p_names = sorted(
            glob.glob(f'{TMP_HIST_LOCATION}/{iteration:04d}-*-pi-*.pt'))

        datasets = []
        for j in range(len(c_names)):
            size = int(c_names[j].split('-')[-1].split('.')[0])
            cs = Game.CANONICAL_SHAPE()
            c_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                c_names[j], shared=False, size=size*cs[0]*cs[1]*cs[2])).reshape(size, cs[0], cs[1], cs[2])
            v_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                v_names[j], shared=False, size=size*(Game.NUM_PLAYERS()+1))).reshape(size, Game.NUM_PLAYERS()+1)
            p_tensor = torch.FloatTensor(torch.FloatStorage().from_file(
                p_names[j], shared=False, size=size*(Game.NUM_MOVES()))).reshape(size, Game.NUM_MOVES())
            datasets.append(TensorDataset(c_tensor, v_tensor, p_tensor))
            del c_tensor
            del v_tensor
            del p_tensor

        dataset = ConcatDataset(datasets)
        sample_count = len(dataset)
        dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE,
                                shuffle=False, num_workers=11)

        nn = neural_net.NNWrapper.load_checkpoint(
            Game, 'data/checkpoint', f'{iteration:04d}-{run_name}.pt')
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

        # Clear old history for iteration before saving new history.
        for fn in glob.glob(f'{HIST_LOCATION}/{iteration:04d}-*.pt'):
            os.remove(fn)

        for i in tqdm.trange(sample_count, desc='Resampling Data', leave=False):
            sample_weight = 0.5 + (loss[i]/total_loss) * 0.5 * sample_count
            for _ in range(math.floor(sample_weight)):
                c, v, pi = dataset[i]
                c_out[i_out] = c
                v_out[i_out] = v
                p_out[i_out] = pi
                i_out += 1
                if maybe_save(Game, c_out, v_out, p_out, i_out, batch_out, iteration):
                    i_out = 0
                    batch_out += 1
            if random.random() < sample_weight - math.floor(sample_weight):
                c, v, pi = dataset[i]
                c_out[i_out] = c
                v_out[i_out] = v
                p_out[i_out] = pi
                i_out += 1
                if maybe_save(Game, c_out, v_out, p_out, i_out, batch_out, iteration):
                    i_out = 0
                    batch_out += 1

        maybe_save(Game, c_out, v_out, p_out, i_out,
                   batch_out, iteration, force=True)

        del datasets[:]
        del dataset
        del dataloader
        del nn
        del c_out
        del v_out
        del p_out

        for fn in glob.glob(f'{TMP_HIST_LOCATION}/*'):
            os.remove(fn)

    def iteration_loss(Game, iteration):
        datasets = []
        c = sorted(
            glob.glob(f'{HIST_LOCATION}/{iteration:04d}-*-canonical-*.pt'))
        v = sorted(glob.glob(f'{HIST_LOCATION}/{iteration:04d}-*-v-*.pt'))
        p = sorted(glob.glob(f'{HIST_LOCATION}/{iteration:04d}-*-pi-*.pt'))
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
        dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE,
                                shuffle=True, num_workers=11)

        nn = neural_net.NNWrapper.load_checkpoint(
            Game, 'data/checkpoint', f'{iteration:04d}-{run_name}.pt')
        v_loss, pi_loss = nn.losses(dataloader)

        del datasets[:]
        del dataset
        del dataloader
        del nn

        return v_loss, pi_loss

    def train(Game, iteration, hist_size, run, total_train_steps):
        total_size = 0
        datasets = []
        for i in range(max(0, iteration - hist_size), iteration + 1):
            c = sorted(glob.glob(f'{HIST_LOCATION}/{i:04d}-*-canonical-*.pt'))
            v = sorted(glob.glob(f'{HIST_LOCATION}/{i:04d}-*-v-*.pt'))
            p = sorted(glob.glob(f'{HIST_LOCATION}/{i:04d}-*-pi-*.pt'))
            for j in range(len(c)):
                size = int(c[j].split('-')[-1].split('.')[0])
                total_size += size
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

        bs = TRAIN_BATCH_SIZE
        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=bs,
                                shuffle=True, num_workers=11)

        average_generation = total_size/min(hist_size, iteration+1)
        nn = neural_net.NNWrapper.load_checkpoint(
            Game, 'data/checkpoint', f'{iteration:04d}-{run_name}.pt')
        steps_to_train = int(
            math.ceil(average_generation/bs*TRAIN_SAMPLE_RATE))
        v_loss, pi_loss = nn.train(
            dataloader, steps_to_train, run, iteration, total_train_steps)
        total_train_steps += steps_to_train
        nn.save_checkpoint('data/checkpoint',
                           f'{iteration+1:04d}-{run_name}.pt')
        del datasets[:]
        del dataset
        del dataloader
        del nn
        return v_loss, pi_loss, total_train_steps

    def self_play(Game, best, iteration, depth, fast_depth):
        bs = SELF_PLAY_BATCH_SIZE
        cb = Game.NUM_PLAYERS()*SELF_PLAY_CONCURRENT_BATCH_MULT
        n = bs*cb*SELF_PLAY_CHUNKS
        params = base_params(Game, SELF_PLAY_TEMP, bs, cb)
        params.games_to_play = n
        params.mcts_depth = [depth] * Game.NUM_PLAYERS()
        params.tree_reuse = False
        params.self_play = True
        params.history_enabled = True
        params.add_noise = True
        params.playout_cap_randomization = True
        params.playout_cap_depth = fast_depth
        params.playout_cap_percent = 0.75
        params.resign_percent = RESIGN_PERCENT
        params.resign_playthrough_percent = RESIGN_PLAYTHROUGH_PERCENT

        # Just use a random agent when generating data with network zero.
        # They are equivalent.
        use_rand = best == 0
        if use_rand:
            nn = RandPlayer(Game, bs)
            params.max_cache_size = 0
        else:
            nn = neural_net.NNWrapper.load_checkpoint(
                Game, 'data/checkpoint', f'{best:04d}-{run_name}.pt')

        pm = alphazero.PlayManager(new_game(), params)
        grargs = GRArgs(title='Self Play', game=Game, iteration=iteration,
                        max_batch_size=bs, concurrent_batches=cb, result_workers=RESULT_WORKERS, cuda=not use_rand)

        players = []
        for _ in range(Game.NUM_PLAYERS()):
            players.append(nn)

        gr = GameRunner(players, pm, grargs)
        gr.run()
        scores = pm.scores()
        win_rates = [0] * len(scores)
        for i in range(len(scores)):
            win_rates[i] = scores[i] / n
        resign_scores = pm.resign_scores()
        resign_win_rates = [0] * len(resign_scores)
        for i in range(len(resign_scores)):
            resign_win_rates[i] = resign_scores[i] / n
        resign_rate = sum(resign_scores)/sum(scores)
        hits = pm.cache_hits()
        total = hits + pm.cache_misses()
        hr = 0
        if total > 0:
            hr = hits/total
        agl = pm.avg_game_length()
        del pm
        del nn
        return win_rates, hr, agl, resign_win_rates, resign_rate

    def play_past(Game, depth, iteration, past_iter):
        nn_rate = 0
        draw_rate = 0
        hr = 0
        agl = 0
        nn = neural_net.NNWrapper.load_checkpoint(
            Game, 'data/checkpoint', f'{iteration:04d}-{run_name}.pt')
        if past_iter == 0:
            nn_past = RandPlayer(Game, 64)
        else:
            nn_past = neural_net.NNWrapper.load_checkpoint(
                Game, 'data/checkpoint', f'{past_iter:04d}-{run_name}.pt')
        cb = Game.NUM_PLAYERS()
        if Game.NUM_PLAYERS() > 2:
            bs = 16
            n = bs*cb
            for i in tqdm.trange(Game.NUM_PLAYERS(), leave=False, desc=f'Bench 1 new vs {Game.NUM_PLAYERS() - 1} old'):
                params = base_params(Game, EVAL_TEMP, bs, cb)
                params.games_to_play = n
                params.mcts_depth = [depth] * Game.NUM_PLAYERS()
                pm = alphazero.PlayManager(new_game(), params)

                grargs = GRArgs(title=f'Bench {iteration} v {past_iter} as p{i+1}', game=Game, iteration=iteration,
                                max_batch_size=bs, concurrent_batches=cb, result_workers=RESULT_WORKERS)
                players = []
                for _ in range(Game.NUM_PLAYERS()):
                    players.append(nn_past)
                players[i] = nn
                gr = GameRunner(players, pm, grargs)
                gr.run()
                scores = pm.scores()
                nn_rate += scores[i]/n
                draw_rate += scores[-1]/n
                hits = pm.cache_hits()
                total = hits + pm.cache_misses()
                if total > 0:
                    hr += hits/total
                agl += pm.avg_game_length()
                del pm
                gc.collect()
            for i in tqdm.trange(Game.NUM_PLAYERS(), leave=False, desc=f'Bench {Game.NUM_PLAYERS() - 1} new vs 1 old'):
                params = base_params(Game, EVAL_TEMP, bs, cb)
                params.games_to_play = n
                params.mcts_depth = [depth] * Game.NUM_PLAYERS()
                pm = alphazero.PlayManager(new_game(), params)

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
                    nn_rate += scores[(i+j) % Game.NUM_PLAYERS()]/n
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
            for i in tqdm.trange(Game.NUM_PLAYERS(), leave=False, desc=f'Bench new vs old'):
                params = base_params(Game, EVAL_TEMP, bs, cb)
                params.games_to_play = n
                params.mcts_depth = [depth] * Game.NUM_PLAYERS()
                pm = alphazero.PlayManager(new_game(), params)

                grargs = GRArgs(title=f'Bench {iteration} v {past_iter} as p{i+1}', game=Game, iteration=iteration,
                                max_batch_size=bs, concurrent_batches=cb, result_workers=RESULT_WORKERS)
                players = []
                for _ in range(Game.NUM_PLAYERS()):
                    players.append(nn_past)
                players[i] = nn
                gr = GameRunner(players, pm, grargs)
                gr.run()
                scores = pm.scores()
                nn_rate += scores[i]/n
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

    run = aim.Run(
        experiment=game_name,
        run_hash=run_name,
    )

    run['hparams'] = {
        'network': network_name,
        'panel_size': GATING_PANEL_SIZE,
        'panel_win_rate': GATING_PANEL_WIN_RATE,
        'best_win_rate': GATING_BEST_WIN_RATE,
        'expected_opening_length': EXPECTED_OPENING_LENGTH,
        'cpuct': CPUCT,
        'fpu_reduction': FPU_REDUCTION,
        'self_play_temp': SELF_PLAY_TEMP,
        'eval_temp': EVAL_TEMP,
        'final_temp': FINAL_TEMP,
        'training_sample_rate': TRAIN_SAMPLE_RATE,
        'bootstrap_iters': bootstrap_iters,
        'depth': depth,
        'channels': channels,
        'kernel_size': kernel_size,
        'lr_milestone': lr_milestone,
        'full_mcts_depth': nn_selfplay_mcts_depth,
        'fast_mcts_depth': nn_selfplay_fast_mcts_depth,
    }

    total_agents = iters+1  # + base

    nnargs = neural_net.NNArgs(
        num_channels=channels, depth=depth, lr_milestone=lr_milestone, dense_net=dense_net, kernel_size=kernel_size)

    if start == 0:
        create_init_net(Game, nnargs)
        wr = np.empty((total_agents, total_agents))
        wr[:] = np.NAN
        elo = np.zeros(total_agents)
        current_best = 0
        total_train_steps = 0
        if bootstrap_iters == 0:
            np.savetxt('data/elo.csv', elo, delimiter=',')
            np.savetxt('data/win_rate.csv', wr, delimiter=',')
            np.savetxt('data/total_train_steps.txt',
                       [total_train_steps], delimiter=',')
    else:
        tmp_wr = np.genfromtxt('data/win_rate.csv', delimiter=',')
        wr = np.full_like(tmp_wr, np.NAN)
        wr[:start+1][:start+1] = tmp_wr[:start+1][:start+1]
        tmp_elo = np.genfromtxt('data/elo.csv', delimiter=',')
        elo = np.zeros_like(tmp_elo)
        elo[:start+1] = tmp_elo[:start+1]
        current_best = np.argmax(elo[:start+1])
        total_train_steps = int(np.genfromtxt('data/total_train_steps.txt'))

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
                v_loss, pi_loss, total_train_steps = train(
                    Game, i, hist_size, run, total_train_steps)
                np.savetxt('data/total_train_steps.txt',
                           [total_train_steps], delimiter=',')
                postfix['vloss'] = v_loss
                postfix['ploss'] = pi_loss
                pbar.set_postfix(postfix)
                gc.collect()
        current_best = bootstrap_iters
        start = bootstrap_iters
        postfix['best'] = current_best

    panel = [current_best]

    with tqdm.trange(start, iters, desc='Build Amazing Network') as pbar:
        for i in pbar:
            run.track(current_best, name='best_network',
                      epoch=i, step=total_train_steps)
            past_iter = max(0, i - compare_past)
            if past_iter != i and math.isnan(wr[i, past_iter]):
                nn_rate, draw_rate, _, game_length = play_past(
                    Game, nn_compare_mcts_depth,  i, past_iter)
                wr[i, past_iter] = (nn_rate + draw_rate/Game.NUM_PLAYERS())
                wr[past_iter, i] = 1-(nn_rate + draw_rate/Game.NUM_PLAYERS())
                run.track(nn_rate, name='win_rate', epoch=i, step=total_train_steps,
                          context={'vs': f'-{compare_past}', 'from': 'all_games'})
                run.track(draw_rate, name='draw_rate', epoch=i, step=total_train_steps,
                          context={'vs': f'-{compare_past}', 'from': 'all_games'})
                run.track(game_length, name='average_game_length',
                          epoch=i, step=total_train_steps, context={'vs': f'-{compare_past}'})
                postfix[f'vs -{compare_past}'] = (nn_rate +
                                                  draw_rate/Game.NUM_PLAYERS())
                gc.collect()

            elo = get_elo(elo, wr, i)
            run.track(elo[i], name='elo', epoch=i,
                      step=total_train_steps, context={'type': 'current'})
            run.track(elo[current_best], name='elo',
                      epoch=i, step=total_train_steps, context={'type': 'best'})
            postfix['elo'] = int(elo[i])
            pbar.set_postfix(postfix)
            np.savetxt('data/elo.csv', elo, delimiter=',')

            win_rates, hit_rate, game_length, resign_win_rates, resignation_rate = self_play(
                Game, current_best, i, nn_selfplay_mcts_depth, nn_selfplay_fast_mcts_depth)
            for j in range(len(win_rates)-1):
                run.track(win_rates[j], name='win_rate',
                          epoch=i, step=total_train_steps, context={'vs': f'self', 'player': j+1, 'from': 'all_games'})
            for j in range(len(resign_win_rates)-1):
                run.track(resign_win_rates[j], name='win_rate',
                          epoch=i, step=total_train_steps, context={'vs': f'self', 'player': j+1, 'from': 'resignation'})
            run.track(resignation_rate, name='resignation_rate',
                      epoch=i, step=total_train_steps, context={'vs': f'self'})
            run.track(win_rates[-1], name='draw_rate',
                      epoch=i, step=total_train_steps, context={'vs': f'self', 'from': 'all_games'})
            run.track(resign_win_rates[-1], name='draw_rate',
                      epoch=i, step=total_train_steps, context={'vs': f'self', 'from': 'resignation'})
            run.track(float(hit_rate), name='cache_hit_rate',
                      epoch=i, step=total_train_steps, context={'vs': f'self'})
            run.track(game_length, name='average_game_length',
                      epoch=i, step=total_train_steps, context={'vs': f'self'})
            postfix['win_rates'] = list(map(lambda x: f'{x:0.3f}', win_rates))
            pbar.set_postfix(postfix)
            gc.collect()

            exploit_symmetries(Game, i)
            gc.collect()

            resample_by_surprise(Game, i)
            gc.collect()

            hist_size = calc_hist_size(i)
            run.track(hist_size, name='history_size',
                      epoch=i, step=total_train_steps)
            v_loss, pi_loss, total_train_steps = train(
                Game, i, hist_size, run, total_train_steps)
            np.savetxt('data/total_train_steps.txt',
                       [total_train_steps], delimiter=',')
            postfix['vloss'] = v_loss
            postfix['ploss'] = pi_loss
            pbar.set_postfix(postfix)
            gc.collect()

            # Eval for gating
            next_net = i + 1
            panel_nn_rate = 0
            panel_draw_rate = 0
            panel_game_length = 0
            best_win_rate = 0
            for gate_net in tqdm.tqdm(panel, desc=f'Pitting against Panel {panel}', leave=False):
                nn_rate, draw_rate, _, game_length = play_past(
                    Game, nn_compare_mcts_depth, next_net, gate_net)
                panel_nn_rate += nn_rate
                panel_draw_rate += draw_rate
                panel_game_length += game_length
                wr[next_net, gate_net] = (
                    nn_rate + draw_rate/Game.NUM_PLAYERS())
                wr[gate_net, next_net] = 1 - \
                    (nn_rate + draw_rate/Game.NUM_PLAYERS())
                gc.collect()
                if gate_net == current_best:
                    run.track(nn_rate, name='win_rate', epoch=next_net, step=total_train_steps,
                              context={'vs': 'best', 'from': 'all_games'})
                    run.track(draw_rate, name='draw_rate', epoch=next_net, step=total_train_steps,
                              context={'vs': 'best', 'from': 'all_games'})
                    run.track(game_length, name='average_game_length',
                              epoch=next_net, context={'vs': 'best'})
                    best_win_rate = nn_rate + draw_rate/Game.NUM_PLAYERS()
                    postfix['vs best'] = best_win_rate
                    pbar.set_postfix(postfix)
            panel_nn_rate /= len(panel)
            panel_draw_rate /= len(panel)
            panel_game_length /= len(panel)
            run.track(panel_nn_rate, name='win_rate', epoch=next_net, step=total_train_steps,
                      context={'vs': 'panel', 'from': 'all_games'})
            run.track(panel_draw_rate, name='draw_rate', epoch=next_net, step=total_train_steps,
                      context={'vs': 'panel', 'from': 'all_games'})
            run.track(panel_game_length, name='average_game_length',
                      epoch=next_net, context={'vs': 'panel'})
            panel_win_rate = panel_nn_rate + panel_draw_rate/Game.NUM_PLAYERS()
            postfix['vs panel'] = panel_win_rate
            # Scale panel win rate based on size of the panel.
            panel_ratio = len(panel) / GATING_PANEL_SIZE
            wanted_panel_win_rate = (
                GATING_PANEL_WIN_RATE * panel_ratio) + (GATING_BEST_WIN_RATE * (1.0 - panel_ratio))
            if panel_win_rate > wanted_panel_win_rate and best_win_rate > GATING_BEST_WIN_RATE:
                current_best = next_net
                postfix['best'] = current_best
                pbar.set_postfix(postfix)
                panel.append(current_best)
                while len(panel) > GATING_PANEL_SIZE:
                    panel = panel[1:]
            np.savetxt('data/win_rate.csv', wr, delimiter=',')
