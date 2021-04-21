import glob
import importlib.util
import os
from collections import namedtuple
from multiprocessing.pool import ThreadPool as Pool
import math
import time
import torch
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import threading
import tqdm
import queue
import numpy as np

src_path = os.path.dirname(os.path.realpath(__file__))
build_path = os.path.join(os.path.dirname(src_path), 'build/src')
lib_path = glob.glob(os.path.join(build_path, 'alphazero*.so'))[0]

spec = importlib.util.spec_from_file_location(
    'alphazero', lib_path)
alphazero = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alphazero)

GRArgs = namedtuple(
    'GRArgs', ['game', 'nnargs', 'max_batch_size', 'iteration',  'data_save_size', 'data_folder', 'concurrent_batches', 'nn_workers', 'batch_workers', 'result_workers', 'mcts_workers'], defaults=(0, 30000, 'data/history', 2, 1, 1, 1, os.cpu_count() - 1))


class GameRunner:
    def __init__(self, pm, args):
        self.pm = pm
        self.args = args
        self.ready_queue = queue.SimpleQueue()
        self.batch_queue = queue.SimpleQueue()
        self.result_queue = queue.SimpleQueue()
        self.monitor_queue = queue.SimpleQueue()
        self.saved_samples = 0
        self.batches = []
        cs = self.args.game.CANONICAL_SHAPE()
        self.hist_canonical = torch.zeros(
            self.args.data_save_size, cs[0], cs[1], cs[2])
        self.hist_v = torch.zeros(
            self.args.data_save_size, self.args.game.NUM_PLAYERS())
        self.hist_pi = torch.zeros(
            self.args.data_save_size, self.args.game.NUM_MOVES())
        shape = (self.args.max_batch_size, cs[0], cs[1], cs[2])
        for i in range(self.args.concurrent_batches):
            self.batches.append(torch.zeros(shape))
            self.batches[i].pin_memory()
            self.ready_queue.put(i)

    def run(self):
        batch_workers = []
        for i in range(self.args.batch_workers):
            batch_workers.append(threading.Thread(target=self.batch_builder))
            batch_workers[i].start()
        result_workers = []
        for i in range(self.args.result_workers):
            result_workers.append(threading.Thread(
                target=self.result_processor))
            result_workers[i].start()
        nn_workers = []
        for i in range(self.args.nn_workers):
            nn = neural_net.NNWrapper(Game, self.args.nnargs)
            nn.load_checkpoint('data/checkpoint',
                               f'{self.args.iteration:04d}.pt')
            nn_workers.append(threading.Thread(
                target=self.network_executor, args=(nn,)))
            nn_workers[i].start()

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
        for nw in nn_workers:
            nw.join()
        monitor.join()
        hist_saver.join()
        print(f'Saved {self.saved_samples} samples')

    def monitor(self):
        last_completed = 0
        last_update = time.time()
        n = self.pm.params().games_to_play
        pbar = tqdm.tqdm(total=n,
                         unit='games', desc='Playing Games')
        while(self.pm.remaining_games() > 0):
            try:
                self.monitor_queue.get(timeout=1)
            except queue.Empty:
                continue
            if time.time() - last_update > 1:
                hits = self.pm.cache_hits()
                total = hits + self.pm.cache_misses() + 1
                completed = self.pm.games_completed()
                p1_score = 0
                if completed > 0:
                    p1_score = self.pm.scores()[0]/completed
                pbar.set_postfix({
                    'p1 score': p1_score,
                    'cache hit': hits/total})
                completed = self.pm.games_completed()
                pbar.update(completed-last_completed)
                last_completed = completed
                last_update = time.time()
        hits = self.pm.cache_hits()
        total = hits + self.pm.cache_misses() + 1
        pbar.set_postfix({
            'p1 score': self.pm.scores()[0]/n,
            'cache hit': hits/total})
        pbar.update(self.pm.params().games_to_play - last_completed)
        pbar.close()

    def batch_builder(self):
        while(self.pm.remaining_games() > 0):
            try:
                batch_index = self.ready_queue.get(timeout=1)
            except queue.Empty:
                continue
            batch = self.batches[batch_index]
            game_indices = self.pm.build_batch(
                batch, self.args.concurrent_batches)
            out = batch[:len(game_indices)]
            if self.args.nnargs.cuda:
                out = out.contiguous().cuda(non_blocking=True)
            self.batch_queue.put((out, batch_index, game_indices))

    def network_executor(self, nn):
        while(self.pm.remaining_games() > 0):
            try:
                batch, batch_index, game_indices = self.batch_queue.get(
                    timeout=1)
            except queue.Empty:
                continue
            v, pi = nn.process(batch)
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
            self.pm.update_inferences(
                game_indices, v, pi)
            self.ready_queue.put(batch_index)
            self.monitor_queue.put(v.shape[0])

    def hist_saver(self):
        batch = 0
        os.makedirs(self.args.data_folder, exist_ok=True)
        while(self.pm.remaining_games() > 0 or self.pm.hist_count() > 0):
            size = self.pm.build_history_batch(
                self.hist_canonical, self.hist_v, self.hist_pi)
            torch.save(
                self.hist_canonical[:size], f'{self.args.data_folder}/{self.args.iteration:04d}-{batch:04d}-canonical.pt')
            torch.save(
                self.hist_v[:size], f'{self.args.data_folder}/{self.args.iteration:04d}-{batch:04d}-v.pt')
            torch.save(
                self.hist_pi[:size], f'{self.args.data_folder}/{self.args.iteration:04d}-{batch:04d}-pi.pt')
            self.saved_samples += size
            batch += 1


if __name__ == '__main__':
    import neural_net

    nnargs = neural_net.NNArgs(num_channels=32, depth=5)
    Game = alphazero.Connect4GS

    def create_init_net():
        nn = neural_net.NNWrapper(Game, nnargs)
        nn.save_checkpoint('data/checkpoint', '0000.pt')

    def train(iteration):
        nn = neural_net.NNWrapper(Game, nnargs)
        nn.load_checkpoint('data/checkpoint', f'{iteration:04d}.pt')
        datasets = []
        currentHistorySize = min(max(4, (iteration + 4)//2), 20)
        for i in range(max(0, iteration - currentHistorySize), iteration + 1):
            c = sorted(glob.glob(f'data/history/{i:04d}-*-canonical.pt'))
            v = sorted(glob.glob(f'data/history/{i:04d}-*-v.pt'))
            p = sorted(glob.glob(f'data/history/{i:04d}-*-pi.pt'))
            for j in range(len(c)):
                datasets.append(TensorDataset(
                    torch.load(c[j]), torch.load(v[j]), torch.load(p[j])))

        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True,
                                num_workers=11, pin_memory=True)

        nn.train(dataloader, 100)
        nn.save_checkpoint('data/checkpoint', f'{iteration+1:04d}.pt')

    def play(iteration):
        params = alphazero.PlayParams()
        bs = 512
        n = bs*12
        cb = 3
        params.games_to_play = n
        params.concurrent_games = bs * (cb+1)
        params.max_batch_size = bs
        params.mcts_depth = 100
        params.history_enabled = True
        params.max_cache_size = 100000
        params.temp_minimization_turn = 10
        params.temp = 2
        pm = alphazero.PlayManager(Game(), params)

        grargs = GRArgs(game=Game, nnargs=nnargs, iteration=iteration,
                        max_batch_size=bs, concurrent_batches=cb, result_workers=2, batch_workers=2)
        gr = GameRunner(pm, grargs)
        gr.run()

    create_init_net()

    for i in range(10):
        play(i)
        train(i)
