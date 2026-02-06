#!/usr/bin/env python3
"""Minimal E2E training test for Star Gambit, including multi-size support."""

import os
import sys
import gc
import shutil
import threading

import alphazero
from neural_net import NNWrapper, NNArgs, get_device
import torch
import numpy as np

# Use Star Gambit (Skirmish variant - 3F, 1C, 0D, 39 actions)
Game = alphazero.StarGambitSkirmishGS

# Unified game for multi-size tests
UnifiedGame = alphazero.StarGambitUnifiedGS

# Minimal network for fast testing
DEPTH = 2
CHANNELS = 8
KERNEL_SIZE = 3
DENSE_NET = True

# Minimal self-play parameters
BATCH_SIZE = 16
CONCURRENT_GAMES = 32
GAMES_TO_PLAY = 64
MCTS_DEPTH = 10

# Data directories
TEST_DATA_DIR = "data/test_star_gambit"


def new_game():
    return Game()


def create_test_network():
    """Create a fresh network for testing."""
    nnargs = NNArgs(
        num_channels=CHANNELS,
        depth=DEPTH,
        kernel_size=KERNEL_SIZE,
        dense_net=DENSE_NET,
    )
    return NNWrapper(Game, nnargs)


class RandPlayer:
    """Random player for initial self-play."""
    def __init__(self, game, max_batch_size):
        self.v = torch.full(
            (max_batch_size, game.NUM_PLAYERS() + 1), 1.0 / (game.NUM_PLAYERS() + 1)
        )
        self.pi = torch.full((max_batch_size, game.NUM_MOVES()), 1.0 / game.NUM_MOVES())

    def predict(self, canonical):
        v, pi = self.process(canonical.unsqueeze(0))
        return v[0], pi[0]

    def process(self, batch):
        return self.v[: batch.shape[0]], self.pi[: batch.shape[0]]


def test_network_creation():
    """Test that we can create a network for Star Gambit."""
    print("=" * 60)
    print("Testing network creation...")

    nn = create_test_network()
    print(f"  Device: {nn.device}")
    print(f"  Network created successfully!")

    # Test forward pass
    cs = Game.CANONICAL_SHAPE()
    print(f"  Canonical shape: {cs}")
    dummy_input = torch.randn(1, cs[0], cs[1], cs[2])
    v, pi = nn.predict(dummy_input[0])
    print(f"  Value output shape: {v.shape}")
    print(f"  Policy output shape: {pi.shape}")
    print(f"  Value sum: {v.sum().item():.4f} (should be ~1.0)")
    print(f"  Policy sum: {pi.sum().item():.4f} (should be ~1.0)")
    print("  Network forward pass OK!")

    del nn
    return True


def test_game_canonical():
    """Test that game canonicalization works."""
    print("=" * 60)
    print("Testing game canonicalization...")

    game = new_game()
    canonical = game.canonicalized()
    canonical_np = np.array(canonical)
    print(f"  Canonical shape: {canonical_np.shape}")
    print(f"  Expected shape: {Game.CANONICAL_SHAPE()}")
    assert canonical_np.shape == tuple(Game.CANONICAL_SHAPE()), "Shape mismatch!"
    print("  Canonicalization OK!")

    return True


def test_play_manager_basic():
    """Test basic PlayManager functionality."""
    print("=" * 60)
    print("Testing PlayManager basic functionality...")

    params = alphazero.PlayParams()
    params.games_to_play = 2
    params.max_batch_size = 2
    params.concurrent_games = 2
    params.mcts_depth = [5, 5]
    params.cpuct = 1.25
    params.start_temp = 1.0
    params.final_temp = 0.2
    params.temp_decay_half_life = 10
    params.fpu_reduction = 0.25
    params.self_play = True
    params.history_enabled = False
    params.max_cache_size = 0

    pm = alphazero.PlayManager(new_game(), params)
    print(f"  PlayManager created for {params.games_to_play} games")

    player = RandPlayer(Game, 2)
    device = get_device()
    cs = Game.CANONICAL_SHAPE()

    # MCTS worker thread
    def mcts_worker():
        while pm.remaining_games() > 0:
            pm.play()

    # Inference worker for each player (separate threads like C++ test)
    def inference_worker(p):
        batch = torch.zeros((params.max_batch_size, cs[0], cs[1], cs[2]))
        while pm.remaining_games() > 0 or pm.games_completed() < params.games_to_play:
            game_indices = pm.build_batch(p, batch, 2)
            if len(game_indices) > 0:
                out = batch[:len(game_indices)]
                out = out.contiguous().to(device, non_blocking=True)
                v, pi = player.process(out)
                v_np = v.cpu().numpy()
                pi_np = pi.cpu().numpy()
                pm.update_inferences(p, game_indices, v_np, pi_np)

    play_thread = threading.Thread(target=mcts_worker)
    infer_p0 = threading.Thread(target=inference_worker, args=(0,))
    infer_p1 = threading.Thread(target=inference_worker, args=(1,))

    play_thread.start()
    infer_p0.start()
    infer_p1.start()

    play_thread.join()
    infer_p0.join()
    infer_p1.join()

    scores = pm.scores()
    print(f"  Games completed: {pm.games_completed()}")
    print(f"  Scores: {scores}")
    print("  PlayManager basic test OK!")

    del pm
    return True


def test_self_play_with_history():
    """Test self-play with history collection."""
    print("=" * 60)
    print("Testing self-play with history...")

    params = alphazero.PlayParams()
    params.games_to_play = GAMES_TO_PLAY
    params.max_batch_size = BATCH_SIZE
    params.concurrent_games = CONCURRENT_GAMES
    params.mcts_depth = [MCTS_DEPTH, MCTS_DEPTH]
    params.cpuct = 1.25
    params.start_temp = 1.0
    params.final_temp = 0.2
    params.temp_decay_half_life = 10
    params.fpu_reduction = 0.25
    params.self_play = True
    params.history_enabled = True
    params.max_cache_size = 100000
    params.cache_shards = 4
    params.tree_reuse = False
    params.add_noise = True
    params.playout_cap_randomization = False
    params.resign_percent = 0.0

    pm = alphazero.PlayManager(new_game(), params)
    print(f"  Starting self-play for {params.games_to_play} games...")

    player = RandPlayer(Game, BATCH_SIZE)
    device = get_device()
    cs = Game.CANONICAL_SHAPE()

    # History collection tensors
    hist_canonical = torch.zeros(5000, cs[0], cs[1], cs[2])
    hist_v = torch.zeros(5000, Game.NUM_PLAYERS() + 1)
    hist_pi = torch.zeros(5000, Game.NUM_MOVES())

    total_samples = 0

    # MCTS worker thread
    def mcts_worker():
        while pm.remaining_games() > 0:
            pm.play()

    # Inference worker for each player
    def inference_worker(p):
        batch = torch.zeros((BATCH_SIZE, cs[0], cs[1], cs[2]))
        while pm.remaining_games() > 0 or pm.games_completed() < params.games_to_play:
            game_indices = pm.build_batch(p, batch, 2)
            if len(game_indices) > 0:
                out = batch[:len(game_indices)]
                out = out.contiguous().to(device, non_blocking=True)
                v, pi = player.process(out)
                v_np = v.cpu().numpy()
                pi_np = pi.cpu().numpy()
                pm.update_inferences(p, game_indices, v_np, pi_np)

    play_thread = threading.Thread(target=mcts_worker)
    infer_p0 = threading.Thread(target=inference_worker, args=(0,))
    infer_p1 = threading.Thread(target=inference_worker, args=(1,))

    play_thread.start()
    infer_p0.start()
    infer_p1.start()

    play_thread.join()
    infer_p0.join()
    infer_p1.join()

    # Collect all history after gameplay completes
    while pm.hist_count() > 0:
        size = pm.build_history_batch(hist_canonical, hist_v, hist_pi)
        if size > 0:
            total_samples += size

    scores = pm.scores()
    agl = pm.avg_game_length()

    print(f"  Games completed: {pm.games_completed()}")
    print(f"  Scores: P0={scores[0]:.2f}, P1={scores[1]:.2f}, Draw={scores[2]:.2f}")
    print(f"  Average game length: {agl:.1f} moves")
    print(f"  Total training samples collected: {total_samples}")
    print("  Self-play with history OK!")

    del pm
    return total_samples


def test_training_step(num_samples):
    """Test a training step with generated data."""
    print("=" * 60)
    print("Testing training step...")

    nn = create_test_network()
    cs = Game.CANONICAL_SHAPE()

    # Create some random training data
    batch_size = min(32, num_samples) if num_samples > 0 else 32
    canonical = torch.randn(batch_size, cs[0], cs[1], cs[2])
    target_v = torch.randn(batch_size, Game.NUM_PLAYERS() + 1)
    target_v = torch.softmax(target_v, dim=1)
    target_pi = torch.randn(batch_size, Game.NUM_MOVES())
    target_pi = torch.softmax(target_pi, dim=1)

    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(canonical, target_v, target_pi)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Dummy logger
    class DummyRun:
        def track(*args, **kwargs):
            pass

    run = DummyRun()

    print(f"  Training for 5 steps with batch size {batch_size}...")
    v_loss, pi_loss = nn.train(dataloader, 5, run, 0, 0)
    print(f"  Value loss: {v_loss:.4f}")
    print(f"  Policy loss: {pi_loss:.4f}")
    print("  Training step OK!")

    # Test save/load
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    nn.save_checkpoint(TEST_DATA_DIR, "test_checkpoint.pt")
    print(f"  Checkpoint saved to {TEST_DATA_DIR}/test_checkpoint.pt")

    nn2 = NNWrapper.load_checkpoint(Game, TEST_DATA_DIR, "test_checkpoint.pt")
    print("  Checkpoint loaded successfully!")

    del nn, nn2
    return True


def test_full_e2e():
    """Full E2E test: self-play -> training."""
    print("=" * 60)
    print("FULL E2E TEST: Self-play -> Training")
    print("=" * 60)

    # Clean up test data
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    # Create network
    print("\n1. Creating network...")
    nn = create_test_network()
    nn.save_checkpoint(TEST_DATA_DIR, "0000-star_gambit.pt")
    print("   Initial network saved.")

    # Self-play
    print("\n2. Running self-play...")
    params = alphazero.PlayParams()
    params.games_to_play = 32  # Small number for test
    params.max_batch_size = 8
    params.concurrent_games = 16
    params.mcts_depth = [8, 8]
    params.cpuct = 1.25
    params.start_temp = 1.0
    params.final_temp = 0.2
    params.temp_decay_half_life = 10
    params.fpu_reduction = 0.25
    params.self_play = True
    params.history_enabled = True
    params.max_cache_size = 50000
    params.cache_shards = 4
    params.tree_reuse = False
    params.add_noise = True
    params.playout_cap_randomization = False
    params.resign_percent = 0.0

    pm = alphazero.PlayManager(new_game(), params)
    device = get_device()
    cs = Game.CANONICAL_SHAPE()

    # History collection
    hist_canonical = torch.zeros(3000, cs[0], cs[1], cs[2])
    hist_v = torch.zeros(3000, Game.NUM_PLAYERS() + 1)
    hist_pi = torch.zeros(3000, Game.NUM_MOVES())

    all_canonical = []
    all_v = []
    all_pi = []

    # MCTS worker thread
    def mcts_worker():
        while pm.remaining_games() > 0:
            pm.play()

    # Inference worker for each player
    def inference_worker(p):
        batch = torch.zeros((params.max_batch_size, cs[0], cs[1], cs[2]))
        while pm.remaining_games() > 0 or pm.games_completed() < params.games_to_play:
            game_indices = pm.build_batch(p, batch, 2)
            if len(game_indices) > 0:
                out = batch[:len(game_indices)]
                out = out.contiguous().to(device, non_blocking=True)
                v, pi = nn.process(out)
                v_np = v.cpu().numpy()
                pi_np = pi.cpu().numpy()
                pm.update_inferences(p, game_indices, v_np, pi_np)

    play_thread = threading.Thread(target=mcts_worker)
    infer_p0 = threading.Thread(target=inference_worker, args=(0,))
    infer_p1 = threading.Thread(target=inference_worker, args=(1,))

    play_thread.start()
    infer_p0.start()
    infer_p1.start()

    play_thread.join()
    infer_p0.join()
    infer_p1.join()

    # Collect all history after gameplay
    while pm.hist_count() > 0:
        size = pm.build_history_batch(hist_canonical, hist_v, hist_pi)
        if size > 0:
            all_canonical.append(hist_canonical[:size].clone())
            all_v.append(hist_v[:size].clone())
            all_pi.append(hist_pi[:size].clone())

    scores = pm.scores()
    print(f"   Games: {pm.games_completed()}, Scores: P0={scores[0]:.1f}, P1={scores[1]:.1f}, Draw={scores[2]:.1f}")
    print(f"   Avg game length: {pm.avg_game_length():.1f}")

    if not all_canonical:
        print("   WARNING: No training samples collected!")
        return False

    train_c = torch.cat(all_canonical, dim=0)
    train_v = torch.cat(all_v, dim=0)
    train_pi = torch.cat(all_pi, dim=0)
    print(f"   Collected {train_c.shape[0]} training samples")

    del pm
    gc.collect()

    # Training
    print("\n3. Training network...")
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(train_c, train_v, train_pi)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    class DummyRun:
        def track(*args, **kwargs):
            pass

    run = DummyRun()
    steps = min(20, len(dataloader))
    v_loss, pi_loss = nn.train(dataloader, steps, run, 0, 0)
    print(f"   Training complete: v_loss={v_loss:.4f}, pi_loss={pi_loss:.4f}")

    nn.save_checkpoint(TEST_DATA_DIR, "0001-star_gambit.pt")
    print("   Trained network saved.")

    # Verify trained network
    print("\n4. Verifying trained network...")
    nn2 = NNWrapper.load_checkpoint(Game, TEST_DATA_DIR, "0001-star_gambit.pt")
    game = new_game()
    canonical = torch.tensor(np.array(game.canonicalized()), dtype=torch.float32)
    v, pi = nn2.predict(canonical)
    print(f"   Prediction check: v_sum={v.sum().item():.4f}, pi_sum={pi.sum().item():.4f}")

    # Clean up
    shutil.rmtree(TEST_DATA_DIR)

    print("\n" + "=" * 60)
    print("FULL E2E TEST PASSED!")
    print("=" * 60)
    return True


# ============================================================================
# Multi-Size Support Tests
# ============================================================================

def test_unified_game_creation():
    """Test creating unified games with all variants."""
    print("=" * 60)
    print("Testing unified game creation...")

    variants = [
        ('SKIRMISH', alphazero.StarGambitVariant.SKIRMISH),
        ('CLASH', alphazero.StarGambitVariant.CLASH),
        ('BATTLE', alphazero.StarGambitVariant.BATTLE),
    ]

    expected_shape = UnifiedGame.CANONICAL_SHAPE()
    expected_moves = UnifiedGame.NUM_MOVES()

    print(f"  Expected canonical shape: {expected_shape}")
    print(f"  Expected num moves: {expected_moves}")

    for name, variant in variants:
        game = alphazero.StarGambitUnifiedGS(variant)
        canonical = np.array(game.canonicalized())

        print(f"  {name}:")
        print(f"    Variant name: {game.get_variant_name()}")
        print(f"    Canonical shape: {canonical.shape}")
        print(f"    Valid moves: {np.sum(game.valid_moves())}")

        assert canonical.shape == tuple(expected_shape), f"Shape mismatch for {name}"
        assert game.get_variant_name() == name.capitalize(), f"Variant name mismatch for {name}"

    print("  Unified game creation OK!")
    return True


def test_fixup_network_creation():
    """Test creating a network with Fixup initialization."""
    print("=" * 60)
    print("Testing Fixup network creation...")

    nnargs = NNArgs(
        num_channels=CHANNELS,
        depth=DEPTH,
        kernel_size=KERNEL_SIZE,
        dense_net=DENSE_NET,
        use_fixup=True,
        multi_size=True,
    )

    # Test with unified game
    nn = NNWrapper(UnifiedGame, nnargs)
    print(f"  Device: {nn.device}")
    print(f"  Fixup enabled: True")
    print(f"  Multi-size enabled: True")

    # Test forward pass with unified shape
    cs = UnifiedGame.CANONICAL_SHAPE()
    print(f"  Canonical shape: {cs}")
    dummy_input = torch.randn(1, cs[0], cs[1], cs[2])
    v, pi = nn.predict(dummy_input[0])
    print(f"  Value output shape: {v.shape}")
    print(f"  Policy output shape: {pi.shape}")
    print(f"  Value sum: {v.sum().item():.4f} (should be ~1.0)")
    print(f"  Policy sum: {pi.sum().item():.4f} (should be ~1.0)")

    # Test that value and policy sums are valid probabilities
    assert abs(v.sum().item() - 1.0) < 0.01, "Value should sum to ~1.0"
    assert abs(pi.sum().item() - 1.0) < 0.01, "Policy should sum to ~1.0"

    print("  Fixup network creation OK!")
    del nn
    return True


def test_curriculum_sampling():
    """Test SizeDistribution curriculum learning."""
    print("=" * 60)
    print("Testing curriculum sampling...")

    from size_distribution import SizeDistribution, DEFAULT_CURRICULUM

    dist = SizeDistribution(DEFAULT_CURRICULUM)

    # Test at iteration 0 (should be 80/15/5)
    dist.step(0)
    probs = dist.get_probs()
    print(f"  Iteration 0: {dist.summary()}")
    assert abs(probs['skirmish'] - 0.80) < 0.01, "Expected 80% Skirmish at iter 0"
    assert abs(probs['clash'] - 0.15) < 0.01, "Expected 15% Clash at iter 0"
    assert abs(probs['battle'] - 0.05) < 0.01, "Expected 5% Battle at iter 0"

    # Test at iteration 100 (should be 56/26/18)
    dist.step(100)
    probs = dist.get_probs()
    print(f"  Iteration 100: {dist.summary()}")
    # Interpolated values: 0.80 + 100/200 * (0.33 - 0.80) = 0.80 - 0.235 = 0.565
    assert abs(probs['skirmish'] - 0.565) < 0.02, f"Expected ~56.5% Skirmish at iter 100, got {probs['skirmish']}"

    # Test at iteration 200 (should be 33/33/34)
    dist.step(200)
    probs = dist.get_probs()
    print(f"  Iteration 200: {dist.summary()}")
    assert abs(probs['skirmish'] - 0.33) < 0.02, "Expected 33% Skirmish at iter 200"
    assert abs(probs['clash'] - 0.33) < 0.02, "Expected 33% Clash at iter 200"
    assert abs(probs['battle'] - 0.34) < 0.02, "Expected 34% Battle at iter 200"

    # Test sampling (should produce valid variants)
    dist.step(100)
    samples = dist.sample_n(100)
    variant_counts = {}
    for s in samples:
        name = str(s).split('.')[-1].lower()
        variant_counts[name] = variant_counts.get(name, 0) + 1
    print(f"  Sample 100 at iter 100: {variant_counts}")

    # Skirmish should be most common at iter 100
    assert variant_counts.get('skirmish', 0) > variant_counts.get('battle', 0), \
        "Skirmish should be more common than Battle at iter 100"

    print("  Curriculum sampling OK!")
    return True


def test_unified_playmanager():
    """Test PlayManager with unified games."""
    print("=" * 60)
    print("Testing PlayManager with unified games...")

    # Test with each variant
    for name, variant in [('Skirmish', alphazero.StarGambitVariant.SKIRMISH),
                          ('Clash', alphazero.StarGambitVariant.CLASH),
                          ('Battle', alphazero.StarGambitVariant.BATTLE)]:

        params = alphazero.PlayParams()
        params.games_to_play = 2
        params.max_batch_size = 2
        params.concurrent_games = 2
        params.mcts_depth = [5, 5]
        params.cpuct = 1.25
        params.start_temp = 1.0
        params.final_temp = 0.2
        params.temp_decay_half_life = 10
        params.fpu_reduction = 0.25
        params.self_play = True
        params.history_enabled = False
        params.max_cache_size = 0

        game = alphazero.StarGambitUnifiedGS(variant)
        pm = alphazero.PlayManager(game, params)
        print(f"  Created PlayManager for {name}")

        player = RandPlayer(UnifiedGame, 2)
        device = get_device()
        cs = UnifiedGame.CANONICAL_SHAPE()

        # MCTS worker thread
        def mcts_worker():
            while pm.remaining_games() > 0:
                pm.play()

        # Inference worker for each player
        def inference_worker(p):
            batch = torch.zeros((params.max_batch_size, cs[0], cs[1], cs[2]))
            while pm.remaining_games() > 0 or pm.games_completed() < params.games_to_play:
                game_indices = pm.build_batch(p, batch, 2)
                if len(game_indices) > 0:
                    out = batch[:len(game_indices)]
                    out = out.contiguous().to(device, non_blocking=True)
                    v, pi = player.process(out)
                    v_np = v.cpu().numpy()
                    pi_np = pi.cpu().numpy()
                    pm.update_inferences(p, game_indices, v_np, pi_np)

        play_thread = threading.Thread(target=mcts_worker)
        infer_p0 = threading.Thread(target=inference_worker, args=(0,))
        infer_p1 = threading.Thread(target=inference_worker, args=(1,))

        play_thread.start()
        infer_p0.start()
        infer_p1.start()

        play_thread.join()
        infer_p0.join()
        infer_p1.join()

        print(f"    Games completed: {pm.games_completed()}")
        del pm

    print("  Unified PlayManager OK!")
    return True


def test_e2e_multisize():
    """Mini E2E test with multi-size unified training."""
    print("=" * 60)
    print("Testing E2E multi-size training...")

    # Create Fixup network for unified game
    nnargs = NNArgs(
        num_channels=CHANNELS,
        depth=DEPTH,
        kernel_size=KERNEL_SIZE,
        dense_net=DENSE_NET,
        use_fixup=True,
        multi_size=True,
    )

    nn = NNWrapper(UnifiedGame, nnargs)
    print(f"  Created Fixup network for unified game")

    # Run self-play on each variant
    from size_distribution import SizeDistribution, DEFAULT_CURRICULUM
    dist = SizeDistribution(DEFAULT_CURRICULUM)
    dist.step(0)  # Start of training

    all_canonical = []
    all_v = []
    all_pi = []

    cs = UnifiedGame.CANONICAL_SHAPE()
    device = get_device()

    for name, variant in [('Skirmish', alphazero.StarGambitVariant.SKIRMISH),
                          ('Clash', alphazero.StarGambitVariant.CLASH)]:

        print(f"  Running self-play on {name}...")

        params = alphazero.PlayParams()
        params.games_to_play = 8
        params.max_batch_size = 4
        params.concurrent_games = 8
        params.mcts_depth = [5, 5]
        params.cpuct = 1.25
        params.start_temp = 1.0
        params.final_temp = 0.2
        params.temp_decay_half_life = 10
        params.fpu_reduction = 0.25
        params.self_play = True
        params.history_enabled = True
        params.max_cache_size = 10000
        params.cache_shards = 2
        params.tree_reuse = False
        params.add_noise = True
        params.playout_cap_randomization = False
        params.resign_percent = 0.0

        game = alphazero.StarGambitUnifiedGS(variant)
        pm = alphazero.PlayManager(game, params)

        hist_canonical = torch.zeros(1000, cs[0], cs[1], cs[2])
        hist_v = torch.zeros(1000, UnifiedGame.NUM_PLAYERS() + 1)
        hist_pi = torch.zeros(1000, UnifiedGame.NUM_MOVES())

        # MCTS worker thread
        def mcts_worker():
            while pm.remaining_games() > 0:
                pm.play()

        # Inference worker for each player
        def inference_worker(p):
            batch = torch.zeros((params.max_batch_size, cs[0], cs[1], cs[2]))
            while pm.remaining_games() > 0 or pm.games_completed() < params.games_to_play:
                game_indices = pm.build_batch(p, batch, 2)
                if len(game_indices) > 0:
                    out = batch[:len(game_indices)]
                    out = out.contiguous().to(device, non_blocking=True)
                    v, pi = nn.process(out)
                    v_np = v.cpu().numpy()
                    pi_np = pi.cpu().numpy()
                    pm.update_inferences(p, game_indices, v_np, pi_np)

        play_thread = threading.Thread(target=mcts_worker)
        infer_p0 = threading.Thread(target=inference_worker, args=(0,))
        infer_p1 = threading.Thread(target=inference_worker, args=(1,))

        play_thread.start()
        infer_p0.start()
        infer_p1.start()

        play_thread.join()
        infer_p0.join()
        infer_p1.join()

        # Collect all history after gameplay
        while pm.hist_count() > 0:
            size = pm.build_history_batch(hist_canonical, hist_v, hist_pi)
            if size > 0:
                all_canonical.append(hist_canonical[:size].clone())
                all_v.append(hist_v[:size].clone())
                all_pi.append(hist_pi[:size].clone())

        print(f"    {name} games completed: {pm.games_completed()}")
        del pm
        gc.collect()

    if not all_canonical:
        print("  WARNING: No training samples collected!")
        return False

    # Combine and train
    train_c = torch.cat(all_canonical, dim=0)
    train_v = torch.cat(all_v, dim=0)
    train_pi = torch.cat(all_pi, dim=0)
    print(f"  Total training samples: {train_c.shape[0]}")

    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(train_c, train_v, train_pi)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    class DummyRun:
        def track(*args, **kwargs):
            pass

    run = DummyRun()
    steps = min(10, len(dataloader))
    v_loss, pi_loss = nn.train(dataloader, steps, run, 0, 0)
    print(f"  Training: v_loss={v_loss:.4f}, pi_loss={pi_loss:.4f}")

    # Verify trained network works on all variants
    print("  Verifying on all variants...")
    for name, variant in [('Skirmish', alphazero.StarGambitVariant.SKIRMISH),
                          ('Clash', alphazero.StarGambitVariant.CLASH),
                          ('Battle', alphazero.StarGambitVariant.BATTLE)]:
        game = alphazero.StarGambitUnifiedGS(variant)
        canonical = torch.tensor(np.array(game.canonicalized()), dtype=torch.float32)
        v, pi = nn.predict(canonical)
        print(f"    {name}: v_sum={v.sum().item():.4f}, pi_sum={pi.sum().item():.4f}")
        assert abs(v.sum().item() - 1.0) < 0.01, f"Value should sum to 1 for {name}"
        assert abs(pi.sum().item() - 1.0) < 0.01, f"Policy should sum to 1 for {name}"

    del nn
    print("  E2E multi-size training OK!")
    return True


def main():
    print("\n" + "=" * 60)
    print("Star Gambit E2E Training Test Suite")
    print("=" * 60)
    print(f"Game: {Game}")
    print(f"NUM_PLAYERS: {Game.NUM_PLAYERS()}")
    print(f"NUM_MOVES: {Game.NUM_MOVES()}")
    print(f"NUM_SYMMETRIES: {Game.NUM_SYMMETRIES()}")
    print(f"CANONICAL_SHAPE: {Game.CANONICAL_SHAPE()}")
    print(f"Device: {get_device()}")
    print()
    print(f"UnifiedGame: {UnifiedGame}")
    print(f"Unified NUM_MOVES: {UnifiedGame.NUM_MOVES()}")
    print(f"Unified CANONICAL_SHAPE: {UnifiedGame.CANONICAL_SHAPE()}")
    print()

    tests = [
        ("Network Creation", test_network_creation),
        ("Game Canonicalization", test_game_canonical),
        ("PlayManager Basic", test_play_manager_basic),
        # Multi-size tests
        ("Unified Game Creation", test_unified_game_creation),
        ("Fixup Network Creation", test_fixup_network_creation),
        ("Curriculum Sampling", test_curriculum_sampling),
        ("Unified PlayManager", test_unified_playmanager),
    ]

    all_passed = True
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                print(f"PASSED: {name}\n")
            else:
                print(f"FAILED: {name}\n")
                all_passed = False
        except Exception as e:
            print(f"FAILED: {name} - {e}\n")
            import traceback
            traceback.print_exc()
            all_passed = False
        gc.collect()

    # Self-play with history
    try:
        num_samples = test_self_play_with_history()
        print(f"PASSED: Self-play with history\n")
    except Exception as e:
        print(f"FAILED: Self-play with history - {e}\n")
        import traceback
        traceback.print_exc()
        all_passed = False
        num_samples = 0
    gc.collect()

    # Training step
    try:
        test_training_step(num_samples)
        print(f"PASSED: Training step\n")
    except Exception as e:
        print(f"FAILED: Training step - {e}\n")
        import traceback
        traceback.print_exc()
        all_passed = False
    gc.collect()

    # Full E2E test
    try:
        test_full_e2e()
        print(f"PASSED: Full E2E test\n")
    except Exception as e:
        print(f"FAILED: Full E2E test - {e}\n")
        import traceback
        traceback.print_exc()
        all_passed = False

    # E2E Multi-size test
    try:
        test_e2e_multisize()
        print(f"PASSED: E2E Multi-size test\n")
    except Exception as e:
        print(f"FAILED: E2E Multi-size test - {e}\n")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
