import os
from multiprocessing.pool import ThreadPool
import threading
import time
import numpy as np
from load_lib import load_alphazero

alphazero = load_alphazero()

Game = alphazero.Connect4GS


def dumb_eval(pm, bs):
    while pm.remaining_games() > 0:
        batch = []
        start = time.time()
        while len(batch) < min(bs, pm.remaining_games()):
            i = pm.pop_game()
            if i is None:
                continue
            batch.append(i)
            gd = pm.game_data(i)
            v = gd.v()
            v.fill(1.0 / len(v))
            pi = gd.pi()
            pi[:] = gd.valid_moves()
            total = np.sum(pi)
            if total != 0:
                pi /= total
        elapsed = time.time() - start
        remaining = 0.059 - elapsed
        if remaining > 0:
            time.sleep(remaining)
        for i in batch:
            pm.push_inference(i)


if __name__ == "__main__":
    procs = os.cpu_count()
    workers = procs - 1
    params = alphazero.PlayParams()
    n = 512 * 12
    bs = 512
    params.games_to_play = n
    params.concurrent_games = bs * 2
    params.mcts_depth = 50

    pm = alphazero.PlayManager(alphazero.Connect4GS(), params)

    inference = threading.Thread(
        target=dumb_eval,
        args=(
            pm,
            bs,
        ),
    )
    inference.start()
    start = time.time()
    with ThreadPool(workers) as pool:
        pool.map(lambda _: pm.play(), range(workers), 1)
    inference.join()
    elapsed = time.time() - start
    print("Score: ", 100 * (pm.scores()[0] + n) / (2 * n))
    print("Execution time: ", elapsed)
