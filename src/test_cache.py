"""Tests for S3FIFOCache pybind11 bindings and cache_utils helpers."""

import numpy as np
import pytest
import torch

import alphazero
from cache_utils import create_cache, cached_inference


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
