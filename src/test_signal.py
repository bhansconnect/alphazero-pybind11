"""Ctrl+C / SIGINT must reach the main thread during GameRunner.run().

Bug: with many workers parked in long C++ pm.play loops (GIL released),
the main thread sat in Thread.join() → sem_wait(-1). A process-directed
SIGINT delivered to a worker thread set the pending-signal flag without
ever returning EINTR on main's sem_wait, so the Python handler never
ran — Ctrl+C silently disappeared.

Fix: block SIGINT in workers so it can only land on main.
"""

import os
import signal
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
from game_runner import (
    RandPlayer, base_params, set_eval_types, set_model_groups,
    GameRunner, GRArgs,
)
from config import TrainConfig


class _PmWrap:
    """Lets us intercept pm.play (used as a thread target) while delegating
    every other attribute to the pybind11 PlayManager."""
    def __init__(self, pm, play):
        self.__dict__["_pm"] = pm
        self.__dict__["play"] = play
    def __getattr__(self, name):
        return getattr(self._pm, name)


def _make_pm(num_games=1_000_000, mcts_visits=8):
    Game = alphazero.Connect4GS
    params = base_params(TrainConfig(game="connect4"), 0.5, 4, 2)
    params.max_cache_size = 0
    params.games_to_play = num_games
    params.concurrent_games = 16
    params.max_batch_size = 4
    params.mcts_visits = [mcts_visits] * Game.NUM_PLAYERS()
    players = [RandPlayer()] * Game.NUM_PLAYERS()
    set_model_groups(params, players)
    set_eval_types(params, players)
    return Game, alphazero.PlayManager(Game(), params), players


def test_mcts_workers_have_sigint_blocked():
    """Each mcts worker thread spawned by _run_inner must inherit a
    SIGINT-blocked mask. Otherwise the kernel can deliver SIGINT to a
    worker and the Python handler (which only runs on the main thread)
    never fires."""
    Game, pm, players = _make_pm()
    N = 4
    masks = []
    lock = threading.Lock()
    enough = threading.Event()
    real_play = pm.play

    def sampling_play():
        m = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        with lock:
            masks.append(m)
            if len(masks) >= N:
                enough.set()
        real_play()

    gr = GameRunner(
        players, _PmWrap(pm, sampling_play),
        GRArgs(title="mask", game=Game, max_batch_size=4,
               iteration=0, mcts_workers=N),
    )

    def stopper():
        enough.wait(timeout=10)
        pm.stop()
    t = threading.Thread(target=stopper)
    t.start()
    try:
        gr.run()
    finally:
        t.join(timeout=10)

    assert len(masks) == N
    for m in masks:
        assert signal.SIGINT in m, f"worker had SIGINT unblocked: {m}"


def test_sigint_during_gr_run_shuts_down():
    """End-to-end: a single process-directed SIGINT during gr.run()
    raises KeyboardInterrupt promptly."""
    Game, pm, players = _make_pm(mcts_visits=200)
    gr = GameRunner(
        players, pm,
        GRArgs(title="e2e", game=Game, max_batch_size=4,
               iteration=0, mcts_workers=16),
    )

    def fire():
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGINT)
    threading.Thread(target=fire, daemon=True).start()

    start = time.time()
    with pytest.raises(KeyboardInterrupt):
        gr.run()
    assert time.time() - start < 15
    assert pm.games_completed() < 1_000_000
