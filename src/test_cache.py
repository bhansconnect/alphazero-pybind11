"""Tests for S3FIFOCache pybind11 bindings and cache_utils helpers."""

import os
import sys
import threading

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
from cache_utils import create_cache, create_sharded_cache, cached_inference
from game_runner import RandPlayer, PlayoutPlayer, base_params, set_eval_types, set_model_groups, GameRunner, GRArgs
from config import TrainConfig


# --- Binding tests ---


def test_hash_game_state_deterministic():
    """Same state hashes to same value."""
    gs = alphazero.Connect4GS()
    h1 = alphazero.hash_game_state(gs)
    h2 = alphazero.hash_game_state(gs)
    assert h1 == h2


def test_hash_game_state_different():
    """Different states hash differently."""
    gs1 = alphazero.Connect4GS()
    gs2 = alphazero.Connect4GS()
    gs2.play_move(0)
    h1 = alphazero.hash_game_state(gs1)
    h2 = alphazero.hash_game_state(gs2)
    assert h1 != h2


def test_cache_miss_returns_none():
    """find on empty cache returns None."""
    cache = alphazero.S3FIFOCache(100, 90, 7, 3)
    result = cache.find(12345, 7, 3)
    assert result is None


def test_cache_insert_and_find():
    """insert then find returns correct values."""
    num_policy = 7
    num_value = 3
    cache = alphazero.S3FIFOCache(100, 90, num_policy, num_value)

    policy = np.array([0.1, 0.2, 0.3, 0.15, 0.1, 0.05, 0.1], dtype=np.float32)
    value = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    h = 42

    cache.insert(h, policy, value)
    result = cache.find(h, num_policy, num_value)

    assert result is not None
    pi_out, v_out = result
    np.testing.assert_allclose(pi_out, policy, atol=1e-6)
    np.testing.assert_allclose(v_out, value, atol=1e-6)


def test_cache_hit_miss_counts():
    """hits()/misses() track correctly."""
    cache = alphazero.S3FIFOCache(100, 90, 7, 3)
    policy = np.zeros(7, dtype=np.float32)
    value = np.zeros(3, dtype=np.float32)

    assert cache.hits() == 0
    assert cache.misses() == 0

    # Miss
    cache.find(1, 7, 3)
    assert cache.misses() == 1
    assert cache.hits() == 0

    # Insert and hit
    cache.insert(1, policy, value)
    cache.find(1, 7, 3)
    assert cache.hits() == 1
    assert cache.misses() == 1


def test_cache_eviction():
    """inserting beyond max_size evicts entries."""
    max_size = 10
    cache = alphazero.S3FIFOCache(max_size, 9, 7, 3)
    policy = np.zeros(7, dtype=np.float32)
    value = np.zeros(3, dtype=np.float32)

    # Fill cache
    for i in range(max_size):
        cache.insert(i, policy, value)
    assert cache.size() == max_size

    # Insert more - should evict
    for i in range(max_size, max_size + 5):
        cache.insert(i, policy, value)
    assert cache.size() == max_size


def test_cache_size_and_max_size():
    """size() and max_size() report correctly."""
    cache = alphazero.S3FIFOCache(50, 45, 7, 3)
    assert cache.max_size() == 50
    assert cache.size() == 0

    policy = np.zeros(7, dtype=np.float32)
    value = np.zeros(3, dtype=np.float32)
    cache.insert(1, policy, value)
    assert cache.size() == 1


# --- cache_utils tests ---


def test_create_cache_zero_size():
    """create_cache returns None for size <= 0."""
    assert create_cache(alphazero.Connect4GS, 0) is None
    assert create_cache(alphazero.Connect4GS, -1) is None


def test_create_cache_valid():
    """create_cache returns working cache."""
    cache = create_cache(alphazero.Connect4GS, 100)
    assert cache is not None
    assert cache.max_size() == 100
    assert cache.size() == 0


def test_cached_inference_miss_and_hit():
    """First call misses, second call hits with same result."""

    class FakeAgent:
        """Minimal agent that returns deterministic predictions."""
        def __init__(self, game_class):
            self.num_moves = game_class.NUM_MOVES()
            self.num_values = game_class.NUM_PLAYERS() + 1
            self.call_count = 0

        def predict(self, canonical):
            import torch
            self.call_count += 1
            v = torch.full((1, self.num_values), 0.33)
            pi = torch.ones(1, self.num_moves) / self.num_moves
            return v, pi

    Game = alphazero.Connect4GS
    cache = create_cache(Game, 100)
    agent = FakeAgent(Game)
    gs = Game()

    # First call - miss
    v1, pi1, hit1 = cached_inference(cache, gs, agent)
    assert not hit1
    assert agent.call_count == 1

    # Second call - hit
    v2, pi2, hit2 = cached_inference(cache, gs, agent)
    assert hit2
    assert agent.call_count == 1  # no additional inference

    np.testing.assert_allclose(v1, v2, atol=1e-6)
    np.testing.assert_allclose(pi1, pi2, atol=1e-6)


# --- MCTS equivalence test ---


class DeterministicAgent:
    """Agent with deterministic predictions based on state hash for reproducibility."""
    def __init__(self, game_class):
        self.num_moves = game_class.NUM_MOVES()
        self.num_values = game_class.NUM_PLAYERS() + 1
        self.call_count = 0

    def predict(self, canonical):
        self.call_count += 1
        # Use canonical tensor to generate deterministic but non-uniform output
        flat = canonical.flatten().float()
        seed = int(abs(flat.sum().item()) * 1000) % (2**31)
        rng = np.random.RandomState(seed)
        pi = rng.dirichlet(np.ones(self.num_moves)).astype(np.float32)
        v = rng.dirichlet(np.ones(self.num_values)).astype(np.float32)
        return torch.from_numpy(v.reshape(1, -1)), torch.from_numpy(pi.reshape(1, -1))


def _run_mcts_game(Game, agent, cache, num_sims=50, max_moves=20):
    """Run a short game using MCTS, return list of (probs, root_value) per move."""
    gs = Game()
    results = []
    for _ in range(max_moves):
        if gs.scores() is not None:
            break
        mcts = alphazero.MCTS(1.25, gs.num_players(), gs.num_moves(), 0.0, 1.4, 0.25,
                              gs.relative_values())
        for sim in range(num_sims):
            leaf = mcts.find_leaf(gs)
            if cache is not None:
                h = alphazero.hash_game_state(leaf)
                result = cache.find(h, leaf.num_moves(), leaf.num_players() + 1)
                if result is not None:
                    pi_c, v_c = result
                    mcts.process_result(gs, v_c, pi_c, False)
                    continue
            v, pi = agent.predict(torch.from_numpy(np.array(leaf.canonicalized())))
            v = v.cpu().numpy().flatten()
            pi = pi.cpu().numpy().flatten()
            if cache is not None:
                cache.insert(alphazero.hash_game_state(leaf), pi, v)
            mcts.process_result(gs, v, pi, False)

        probs = np.array(mcts.probs(1.0))
        root_val = np.array(mcts.root_value())
        results.append((probs, root_val))

        # Pick best move deterministically
        move = int(np.argmax(probs))
        gs.play_move(move)

    return results


def test_mcts_with_and_without_cache_identical():
    """MCTS produces identical policies and values with and without cache."""
    Game = alphazero.Connect4GS

    # Run without cache
    agent_no_cache = DeterministicAgent(Game)
    results_no_cache = _run_mcts_game(Game, agent_no_cache, cache=None, num_sims=50)

    # Run with cache
    cache = create_cache(Game, 10000)
    agent_with_cache = DeterministicAgent(Game)
    results_with_cache = _run_mcts_game(Game, agent_with_cache, cache, num_sims=50)

    assert len(results_no_cache) == len(results_with_cache), \
        f"Different game lengths: {len(results_no_cache)} vs {len(results_with_cache)}"

    for i, ((p1, v1), (p2, v2)) in enumerate(zip(results_no_cache, results_with_cache)):
        np.testing.assert_allclose(p1, p2, atol=1e-6,
                                   err_msg=f"Policy mismatch at move {i}")
        np.testing.assert_allclose(v1, v2, atol=1e-6,
                                   err_msg=f"Value mismatch at move {i}")

    # Cache should have hits (positions revisited across MCTS sims)
    assert cache.hits() > 0, "Expected some cache hits"
    # With cache, fewer NN calls should have been made
    assert agent_with_cache.call_count <= agent_no_cache.call_count, \
        f"Cache should reduce inference calls: {agent_with_cache.call_count} vs {agent_no_cache.call_count}"


# --- ShardedS3FIFOCache binding tests ---


def test_sharded_cache_construction():
    """ShardedS3FIFOCache can be constructed and exposes stats methods."""
    cache = alphazero.ShardedS3FIFOCache(1000, 2, 900, 7, 3)
    assert cache.size() == 0
    assert cache.max_size() == 1000
    assert cache.hits() == 0
    assert cache.misses() == 0
    assert cache.evictions() == 0
    assert cache.reinserts() == 0


def test_sharded_cache_shared_ptr_holder():
    """ShardedS3FIFOCache uses shared_ptr holder -- multiple references work."""
    cache = alphazero.ShardedS3FIFOCache(100, 1, 90, 7, 3)
    ref1 = cache
    ref2 = cache
    # Both refs point to the same C++ object
    assert ref1.size() == ref2.size()
    assert ref1.max_size() == ref2.max_size()


def test_create_sharded_cache_zero():
    """create_sharded_cache returns None for size <= 0."""
    assert create_sharded_cache(alphazero.Connect4GS, 0) is None
    assert create_sharded_cache(alphazero.Connect4GS, -1) is None


def test_create_sharded_cache_valid():
    """create_sharded_cache returns working cache with correct dimensions."""
    cache = create_sharded_cache(alphazero.Connect4GS, 1000, shards=2)
    assert cache is not None
    assert cache.max_size() == 1000
    assert cache.size() == 0


def test_create_sharded_cache_default_shards():
    """create_sharded_cache defaults to 1 shard."""
    cache = create_sharded_cache(alphazero.Connect4GS, 100)
    assert cache is not None
    assert cache.max_size() == 100


# --- PlayManager with external caches ---


def _run_play_manager_with_nn(Game, params, caches=None, num_games=8):
    """Helper: run games through PlayManager with NN eval using dummy inference.

    This simulates what GameRunner does: MCTS workers in threads produce
    leaf states, an inference thread pops them, generates random v/pi,
    and pushes results back. This exercises the cache code path.
    """
    nplayers = Game.NUM_PLAYERS()
    params.games_to_play = num_games
    params.concurrent_games = min(4, num_games)
    params.max_batch_size = 4
    params.mcts_visits = [8] * nplayers
    # NN eval type -- this exercises the cache
    params.eval_type = [alphazero.EvalType.NN] * nplayers
    params.self_play = True
    params.model_groups = [0] * nplayers  # All players share group 0

    if caches is not None:
        pm = alphazero.PlayManager(Game(), params, caches=caches)
    else:
        pm = alphazero.PlayManager(Game(), params)

    num_values = nplayers + 1
    num_moves = Game.NUM_MOVES()

    # MCTS worker threads
    workers = []
    for _ in range(2):
        t = threading.Thread(target=pm.play)
        t.start()
        workers.append(t)

    # Inference thread: pop batches, return random v/pi
    def inference_loop():
        rng = np.random.RandomState(42)
        batch = np.zeros((params.max_batch_size, *Game.CANONICAL_SHAPE()), dtype=np.float32)
        while pm.remaining_games() > 0:
            indices = pm.build_batch(0, batch)
            if not indices:
                continue
            n = len(indices)
            v = rng.dirichlet(np.ones(num_values), size=n).astype(np.float32)
            pi = rng.dirichlet(np.ones(num_moves), size=n).astype(np.float32)
            pm.update_inferences(0, indices, v, pi)

    inf_thread = threading.Thread(target=inference_loop)
    inf_thread.start()
    inf_thread.join()
    for t in workers:
        t.join()

    assert pm.games_completed() == num_games
    return pm


def _run_play_manager_random(Game, params, caches=None, num_games=8):
    """Helper: run games with RANDOM eval (no inference needed)."""
    nplayers = Game.NUM_PLAYERS()
    params.games_to_play = num_games
    params.concurrent_games = min(4, num_games)
    params.max_batch_size = 4
    params.mcts_visits = [8] * nplayers
    params.eval_type = [alphazero.EvalType.RANDOM] * nplayers

    if caches is not None:
        pm = alphazero.PlayManager(Game(), params, caches=caches)
    else:
        pm = alphazero.PlayManager(Game(), params)

    workers = []
    for _ in range(2):
        t = threading.Thread(target=pm.play)
        t.start()
        workers.append(t)
    for t in workers:
        t.join()

    assert pm.games_completed() == num_games
    return pm


def test_playmanager_internal_cache_still_works():
    """PlayManager with max_cache_size > 0 (no external caches) still works."""
    Game = alphazero.Connect4GS
    config = TrainConfig(game="connect4")
    params = base_params(config, 0.5, 4, 2)
    params.max_cache_size = 1000

    pm = _run_play_manager_with_nn(Game, params, caches=None, num_games=8)
    # Internal caches should have been created (may be slightly less due to shard rounding)
    assert pm.cache_max_size() > 0
    # NN eval should have populated the cache
    assert pm.cache_size() > 0


def test_playmanager_no_cache():
    """PlayManager with no cache (max_cache_size=0, no external caches)."""
    Game = alphazero.Connect4GS
    config = TrainConfig(game="connect4")
    params = base_params(config, 0.5, 4, 2)
    params.max_cache_size = 0

    pm = _run_play_manager_random(Game, params, caches=None, num_games=8)
    assert pm.cache_max_size() == 0
    assert pm.cache_hits() == 0
    assert pm.cache_misses() == 0


def test_playmanager_external_cache():
    """PlayManager with externally-provided ShardedS3FIFOCache works and populates it."""
    Game = alphazero.Connect4GS
    config = TrainConfig(game="connect4")
    params = base_params(config, 0.5, 4, 2)
    params.max_cache_size = 0

    external_cache = create_sharded_cache(Game, 5000)
    caches = [external_cache]  # Single group (self_play)

    pm = _run_play_manager_with_nn(Game, params, caches=caches, num_games=8)
    # External cache should have been used
    assert pm.cache_max_size() > 0
    # The external cache object itself should reflect usage
    assert external_cache.size() > 0
    assert external_cache.misses() > 0


def test_playmanager_external_cache_shared_across_instances():
    """Two PlayManager instances sharing the same external cache accumulate entries."""
    Game = alphazero.Connect4GS
    config = TrainConfig(game="connect4")

    shared_cache = create_sharded_cache(Game, 10000)

    # Run first batch of games
    params1 = base_params(config, 0.5, 4, 2)
    params1.max_cache_size = 0
    caches = [shared_cache]
    _run_play_manager_with_nn(Game, params1, caches=caches, num_games=8)
    size_after_first = shared_cache.size()
    hits_after_first = shared_cache.hits()
    assert size_after_first > 0, "Cache should have entries after first batch"

    # Run second batch of games with the same shared cache
    params2 = base_params(config, 0.5, 4, 2)
    params2.max_cache_size = 0
    _run_play_manager_with_nn(Game, params2, caches=caches, num_games=8)
    hits_after_second = shared_cache.hits()

    # Second batch should benefit from cached positions from first batch
    assert hits_after_second > hits_after_first, \
        f"Second batch should have more cache hits: {hits_after_second} vs {hits_after_first}"


def test_playmanager_external_cache_with_none_entries():
    """PlayManager handles None entries in the caches list (for non-NN groups)."""
    Game = alphazero.Connect4GS
    config = TrainConfig(game="connect4")

    # One NN cache, one None (simulating NN vs random where random gets no cache)
    nn_cache = create_sharded_cache(Game, 5000)
    caches = [nn_cache, None]

    params = base_params(config, 0.5, 4, 2)
    params.max_cache_size = 0
    # Both players use RANDOM for simplicity -- the point is that None cache entries don't crash
    pm = _run_play_manager_random(Game, params, caches=caches, num_games=8)
    assert pm.games_completed() == 8


def test_playmanager_cache_max_size_sums_external():
    """cache_max_size() returns sum of external caches' max sizes."""
    Game = alphazero.Connect4GS
    config = TrainConfig(game="connect4")

    cache1 = create_sharded_cache(Game, 3000)
    cache2 = create_sharded_cache(Game, 5000)
    caches = [cache1, cache2]

    params = base_params(config, 0.5, 4, 2)
    params.max_cache_size = 0
    pm = _run_play_manager_random(Game, params, caches=caches, num_games=4)
    assert pm.cache_max_size() == 3000 + 5000


# --- seat_visits tests ---


def test_seat_visits_binding_roundtrip():
    """seat_visits can be set and read back on PlayParams."""
    params = alphazero.PlayParams()
    assert params.seat_visits == []

    sv = [[1, 480], [480, 1]]
    params.seat_visits = sv
    assert params.seat_visits == sv


def test_seat_visits_overrides_visit_count():
    """seat_visits causes asymmetric play when same model shares a group.

    With seat_visits=[1, 200] (perm 0) and [200, 1] (perm 1), one player
    uses 1 visit (essentially random) while the other uses 200.
    The player with more visits should win significantly more.
    Without seat_visits, both get the same visits and results are even.
    """
    Game = alphazero.Connect4GS
    config = TrainConfig(game="connect4")
    nplayers = Game.NUM_PLAYERS()
    num_games = 64

    # --- Run WITH seat_visits: player 0 gets 200 visits, player 1 gets 1 ---
    params = base_params(config, 0.5, 4, 2)
    params.games_to_play = num_games
    params.concurrent_games = min(8, num_games)
    params.max_batch_size = 8
    params.mcts_visits = [200, 1]
    params.eval_type = [alphazero.EvalType.PLAYOUT] * nplayers
    params.model_groups = [0, 0]  # same group
    params.seat_perms = [[0, 0], [0, 0]]
    params.seat_visits = [[200, 1], [1, 200]]

    pm = alphazero.PlayManager(Game(), params)
    workers = []
    for _ in range(2):
        t = threading.Thread(target=pm.play)
        t.start()
        workers.append(t)
    for t in workers:
        t.join()

    assert pm.games_completed() == num_games

    # Perm 0: seat 0 gets 200 visits, seat 1 gets 1
    # Perm 1: seat 0 gets 1 visit, seat 1 gets 200
    # The player with 200 visits should dominate in each perm.
    perm0_scores = np.array(pm.perm_scores(0))
    perm1_scores = np.array(pm.perm_scores(1))
    perm0_games = pm.perm_games_completed(0)
    perm1_games = pm.perm_games_completed(1)

    if perm0_games > 0:
        # In perm 0, seat 0 has 200 visits — should win most
        assert perm0_scores[0] > perm0_scores[1], \
            f"Perm 0: seat 0 (200 visits) should beat seat 1 (1 visit): {perm0_scores}"
    if perm1_games > 0:
        # In perm 1, seat 1 has 200 visits — should win most
        assert perm1_scores[1] > perm1_scores[0], \
            f"Perm 1: seat 1 (200 visits) should beat seat 0 (1 visit): {perm1_scores}"
