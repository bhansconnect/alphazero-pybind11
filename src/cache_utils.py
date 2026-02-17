"""Shared helpers for S3-FIFO cache integration with MCTS scripts."""

import numpy as np
import torch

import alphazero


def create_cache(game_class, cache_size):
    """Create an S3FIFOCache for the given game, or return None if cache_size <= 0."""
    if cache_size <= 0:
        return None
    num_moves = game_class.NUM_MOVES()
    num_values = game_class.NUM_PLAYERS() + 1
    ghost_size = cache_size * 9 // 10
    return alphazero.S3FIFOCache(cache_size, ghost_size, num_moves, num_values)


def cached_inference(cache, leaf, agent):
    """Check cache for leaf; on miss, run agent.predict() and insert.

    Returns (v, pi, was_hit) where v and pi are numpy arrays.
    """
    h = alphazero.hash_game_state(leaf)
    if cache is not None:
        result = cache.find(h, leaf.num_moves(), leaf.num_players() + 1)
        if result is not None:
            pi, v = result
            return np.array(v), np.array(pi), True
    # Cache miss - run inference
    v, pi = agent.predict(torch.from_numpy(np.array(leaf.canonicalized())))
    v = v.cpu().numpy().flatten()
    pi = pi.cpu().numpy().flatten()
    if cache is not None:
        cache.insert(h, pi, v)
    return v, pi, False


def create_sharded_cache(game_class, cache_size, shards=None):
    """Create a ShardedS3FIFOCache for PlayManager, or None if cache_size <= 0."""
    if cache_size <= 0:
        return None
    if shards is None:
        shards = 1
    num_moves = game_class.NUM_MOVES()
    num_values = game_class.NUM_PLAYERS() + 1
    ghost_size = cache_size * 9 // 10
    return alphazero.ShardedS3FIFOCache(cache_size, shards, ghost_size, num_moves, num_values)


def print_cache_stats(cache):
    """Print cache hit rate and saturation."""
    if cache is None:
        return
    hits, misses = cache.hits(), cache.misses()
    total = hits + misses
    if total == 0:
        return
    print(f"Cache: {hits/total:.1%} hit rate, {cache.size()}/{cache.max_size()} entries")


def print_sharded_cache_stats(cache, label="Shared cache"):
    """Print stats for a ShardedS3FIFOCache."""
    if cache is None:
        return
    hits, misses = cache.hits(), cache.misses()
    total = hits + misses
    if total == 0:
        return
    print(f"{label}: {hits/total:.1%} hit rate, {cache.size()}/{cache.max_size()} entries"
          f" ({hits} hits, {misses} misses, {cache.evictions()} evictions, {cache.reinserts()} reinserts)")
