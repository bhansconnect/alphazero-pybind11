import glob
import importlib.util
import os
from collections import namedtuple
from multiprocessing.pool import ThreadPool as Pool
import math
import time
import torch
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
    'GRArgs', ['game', 'nnargs', 'max_batch_size', 'concurrent_batches', 'nn_workers', 'batch_workers', 'result_workers', 'mcts_workers'], defaults=(2, 1, 1, 1, os.cpu_count() - 1))


class GameRunner:
    def __init__(self, pm, args):
        self.pm = pm
        self.args = args
        self.ready_queue = queue.SimpleQueue()
        self.batch_queue = queue.SimpleQueue()
        self.result_queue = queue.SimpleQueue()
        self.monitor_queue = queue.SimpleQueue()
        self.awaiting_gpu = 0
        self.batches = []
        cs = self.args.game.CANONICAL_SHAPE()
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
            nn_workers.append(threading.Thread(
                target=self.network_executor, args=(neural_net.NNWrapper(Game, self.args.nnargs),)))
            nn_workers[i].start()

        monitor = threading.Thread(target=self.monitor)
        monitor.start()

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

    def monitor(self):
        last_completed = 0
        sample_count = 0
        start = time.time()
        last_update = time.time()
        pbar = tqdm.tqdm(total=n,
                         unit='games', desc='Playing Games')
        while(self.pm.remaining_games() > 0):
            try:
                batch_size = self.monitor_queue.get(timeout=1)
            except queue.Empty:
                continue
            sample_count += batch_size
            if time.time() - last_update > 1:
                hits = self.pm.cache_hits()
                total = hits + self.pm.cache_misses() + 1
                completed = self.pm.games_completed()
                p1_score = 0
                if completed > 0:
                    p1_score = self.pm.scores()[0]/completed
                pbar.set_postfix({
                    'samples/s': sample_count/(time.time()-start),
                    'p1 score': p1_score,
                    'cache hit': hits/total})
                completed = self.pm.games_completed()
                pbar.update(completed-last_completed)
                last_completed = completed
                last_update = time.time()
        hits = self.pm.cache_hits()
        total = hits + self.pm.cache_misses() + 1
        pbar.set_postfix({
            'samples/s': sample_count/(time.time()-start),
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
            self.awaiting_gpu += 1
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
            self.awaiting_gpu -= 1
            self.pm.update_inferences(
                game_indices, v, pi)
            self.ready_queue.put(batch_index)
            self.monitor_queue.put(v.shape[0])


if __name__ == '__main__':
    import neural_net
    Game = alphazero.Connect4GS
    params = alphazero.PlayParams()
    bs = 512
    n = bs*12
    cb = 3
    params.games_to_play = n
    params.concurrent_games = bs * (cb+1)
    params.max_batch_size = bs
    params.mcts_depth = 100
    params.history_enabled = False
    params.max_cache_size = 100000

    pm = alphazero.PlayManager(Game(), params)

    grargs = GRArgs(game=Game, nnargs=neural_net.NNArgs(num_channels=32, depth=5), max_batch_size=bs,
                    concurrent_batches=cb)
    gr = GameRunner(pm, grargs)
    gr.run()
