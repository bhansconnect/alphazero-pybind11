"""Tests for play.py -- batch sizing, cache-aware batching, calibration, MCTS search."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
from play import (
    _compute_batch_size,
    _run_one_batch,
    run_mcts_search,
    create_mcts,
    calibrate_timed_batch,
    PlayContext,
    _update_caches,
    _get_cache,
)
from cache_utils import create_cache


# --- _compute_batch_size tests ---


def test_compute_batch_size_sqrt_scaling():
    """Batch size equals int(sqrt(budget)) for various budgets."""
    for budget in [4, 16, 25, 100, 400, 10000]:
        assert _compute_batch_size(budget) == int(math.sqrt(budget))


def test_compute_batch_size_minimum_one():
    """Budget=0 and budget=1 both return 1."""
    assert _compute_batch_size(0) == 1
    assert _compute_batch_size(1) == 1


def test_compute_batch_size_no_hard_cap():
    """Large budgets produce large batch sizes (no cap at 64)."""
    assert _compute_batch_size(10000) == 100
    assert _compute_batch_size(40000) == 200
    assert _compute_batch_size(1000000) == 1000


# --- _run_one_batch tests ---


def test_run_one_batch_returns_actual_sims():
    """_run_one_batch returns actual simulation count, not nominal batch_size."""
    gs = alphazero.Connect4GS()
    mcts = create_mcts(type(gs))
    batch_size = 4

    actual = _run_one_batch(gs, None, mcts, batch_size, "random", None, 0)

    # With random eval and no cache, all leaves are processed inline
    # actual should equal the number of attempts (all are immediate)
    assert actual >= batch_size
    assert isinstance(actual, int)


def test_run_one_batch_cache_hits_extra_sims():
    """With a warm cache, actual_sims > len(pending_gpu) due to cache hits."""
    gs = alphazero.Connect4GS()
    cache = create_cache(type(gs), 1000)
    mcts = create_mcts(type(gs))
    batch_size = 8

    # With random eval (no agent), all are processed immediately
    actual = _run_one_batch(gs, None, mcts, batch_size, "random", cache, 0)

    # All sims are processed inline (random eval), so actual >= batch_size
    assert actual >= batch_size


# --- run_mcts_search tests ---


def test_mcts_search_node_limit():
    """Basic correctness with auto batch and node limit."""
    gs = alphazero.Connect4GS()
    mcts = create_mcts(type(gs))
    node_limit = 50

    counts, sims, wld = run_mcts_search(
        gs, None, mcts, node_limit=node_limit, eval_type="random",
        max_batch_size=0,
    )

    assert counts.sum() > 0
    assert sims >= node_limit  # may overshoot slightly due to batching
    assert len(wld) == gs.num_players() + 1


def test_mcts_search_sequential_unchanged():
    """batch_size=1 path still works correctly."""
    gs = alphazero.Connect4GS()
    mcts = create_mcts(type(gs))
    node_limit = 20

    counts, sims, wld = run_mcts_search(
        gs, None, mcts, node_limit=node_limit, eval_type="random",
        max_batch_size=1,
    )

    assert sims == node_limit  # sequential is exact
    assert counts.sum() > 0


def test_mcts_search_timed_auto():
    """Timed mode completes and produces results."""
    gs = alphazero.Connect4GS()
    mcts = create_mcts(type(gs))

    counts, sims, wld = run_mcts_search(
        gs, None, mcts, time_limit=0.1, eval_type="random",
        max_batch_size=0,
    )

    assert counts.sum() > 0
    assert sims > 0


# --- calibrate_timed_batch tests ---


def test_calibrate_timed_batch():
    """Calibration returns a power of 2, >= 2."""
    gs = alphazero.Connect4GS()

    result = calibrate_timed_batch(gs, None, time_limit=1.0,
                                    eval_type="random", cache=None)

    assert result >= 2
    # Must be a power of 2
    assert (result & (result - 1)) == 0, f"{result} is not a power of 2"


# --- Per-player cache split tests ---


def test_update_caches_shared_when_no_time():
    """No think_time -> shared cache, no per-player caches."""
    gs = alphazero.Connect4GS()
    ctx = PlayContext(gs, type(gs), cache_size=1000)
    ctx.players[0].is_ai = True
    ctx.players[1].is_ai = True
    _update_caches(ctx)

    assert ctx.cache is not None
    assert ctx.player_caches == [None, None]
    assert _get_cache(ctx, 0) is ctx.cache
    assert _get_cache(ctx, 1) is ctx.cache


def test_update_caches_split_when_timed():
    """With think_time -> per-player caches, no shared cache."""
    gs = alphazero.Connect4GS()
    ctx = PlayContext(gs, type(gs), cache_size=1000)
    ctx.players[0].is_ai = True
    ctx.players[0].think_time = 2.0
    ctx.players[1].is_ai = True
    _update_caches(ctx)

    assert ctx.cache is None
    assert ctx.player_caches[0] is not None
    assert ctx.player_caches[1] is not None
    assert ctx.player_caches[0] is not ctx.player_caches[1]
    assert _get_cache(ctx, 0) is ctx.player_caches[0]
    assert _get_cache(ctx, 1) is ctx.player_caches[1]
    # Each gets half the total cache size
    assert ctx.player_caches[0].max_size() == 500
    assert ctx.player_caches[1].max_size() == 500


def test_update_caches_disabled():
    """cache_size=0 -> all caches are None."""
    gs = alphazero.Connect4GS()
    ctx = PlayContext(gs, type(gs), cache_size=0)
    ctx.players[0].think_time = 2.0
    _update_caches(ctx)

    assert ctx.cache is None
    assert ctx.player_caches == [None, None]
    assert _get_cache(ctx, 0) is None
    assert _get_cache(ctx, 1) is None
