import glob
import importlib.util
import os
from multiprocessing.pool import ThreadPool
import threading
import time
import numpy as np

src_path = os.path.dirname(os.path.realpath(__file__))
build_path = os.path.join(os.path.dirname(src_path), 'build/src')
lib_path = glob.glob(os.path.join(build_path, 'alphazero*.so'))[0]

spec = importlib.util.spec_from_file_location(
    'alphazero', lib_path)
alphazero = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alphazero)

Game = alphazero.Connect4GS


def dumb_eval(pm):
    while(pm.remaining_games() > 0):
        i = pm.pop_game()
        if i is None:
            continue
        gd = pm.game_data(i)
        gd.v().fill(0)
        pi = gd.pi()
        pi[:] = gd.valid_moves()
        total = np.sum(pi)
        if total != 0:
            pi /= total
        pm.push_inference(i)


if __name__ == '__main__':
    procs = os.cpu_count()
    workers = procs - 1
    params = alphazero.PlayParams()
    n = 128*workers
    params.games_to_play = n
    params.concurrent_games = 16 * workers
    params.mcts_depth = 100

    pm = alphazero.PlayManager(alphazero.Connect4GS(), params)

    inference = threading.Thread(target=dumb_eval, args=(pm,))
    inference.start()
    start = time.time()
    with ThreadPool(workers) as pool:
        pool.map(lambda _: pm.play(), range(workers), 1)
    inference.join()
    elapsed = time.time() - start
    print("Score: ", 100*(pm.scores()[0]+n)/(2*n))
    print("Execution time: ", elapsed)
