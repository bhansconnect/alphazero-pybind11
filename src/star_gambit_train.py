#!/usr/bin/env python3
"""Training script for Star Gambit AI.

This script configures game_runner.py for Star Gambit Skirmish training.
Run with: python src/star_gambit_train.py
"""

import os
import sys

# Ensure we can import from the src directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
import game_runner

# === Game Configuration ===
Game = alphazero.StarGambitSkirmishGS
game_name = "star_gambit_skirmish"


def new_game():
    return Game()


# === Network Configuration ===
# Star Gambit Skirmish: 39 actions, 61 hexes on 5-side board
# Moderate complexity - 3 fighters + 1 cruiser per side
depth = 4              # 4 residual/dense blocks
channels = 16          # 16 channels per layer
kernel_size = 3        # 3x3 kernels
dense_net = True       # DenseNet tends to train faster

# === MCTS Configuration ===
nn_selfplay_mcts_depth = 100       # Full search for quality games
nn_selfplay_fast_mcts_depth = 25   # Fast depth for playout cap
nn_compare_mcts_depth = 50         # Comparison depth for gating

# === Training Parameters ===
CPUCT = 1.25                       # Exploration constant
FPU_REDUCTION = 0.25               # First Play Urgency reduction
EXPECTED_OPENING_LENGTH = 15       # ~15 move opening phase
SELF_PLAY_TEMP = 1.0               # Temperature during self-play
EVAL_TEMP = 0.5                    # Temperature during evaluation
FINAL_TEMP = 0.2                   # Final temperature
TEMP_DECAY_HALF_LIFE = EXPECTED_OPENING_LENGTH

# === Batch Configuration ===
SELF_PLAY_BATCH_SIZE = 128         # Games per batch
SELF_PLAY_CONCURRENT_BATCH_MULT = 2
SELF_PLAY_CHUNKS = 4

# === Gating Thresholds ===
GATING_PANEL_SIZE = 1
GATING_PANEL_WIN_RATE = 0.52
GATING_BEST_WIN_RATE = 0.52

# === Training Iterations ===
bootstrap_iters = 0
start = 0
iters = 200
lr_milestone = 150

# === Derived ===
network_name = "densenet" if dense_net else "resnet"
run_name = f"{game_name}-{network_name}-{depth}d-{channels}c-{kernel_size}k-{nn_selfplay_mcts_depth}sims"


def configure_game_runner():
    """Override game_runner module globals with Star Gambit settings."""
    game_runner.Game = Game
    game_runner.game_name = game_name
    game_runner.new_game = new_game

    # Network
    game_runner.depth = depth
    game_runner.channels = channels
    game_runner.kernel_size = kernel_size
    game_runner.dense_net = dense_net
    game_runner.network_name = network_name
    game_runner.run_name = run_name
    game_runner.lr_milestone = lr_milestone

    # MCTS
    game_runner.nn_selfplay_mcts_depth = nn_selfplay_mcts_depth
    game_runner.nn_selfplay_fast_mcts_depth = nn_selfplay_fast_mcts_depth
    game_runner.nn_compare_mcts_depth = nn_compare_mcts_depth

    # Training parameters
    game_runner.CPUCT = CPUCT
    game_runner.FPU_REDUCTION = FPU_REDUCTION
    game_runner.EXPECTED_OPENING_LENGTH = EXPECTED_OPENING_LENGTH
    game_runner.SELF_PLAY_TEMP = SELF_PLAY_TEMP
    game_runner.EVAL_TEMP = EVAL_TEMP
    game_runner.FINAL_TEMP = FINAL_TEMP
    game_runner.TEMP_DECAY_HALF_LIFE = TEMP_DECAY_HALF_LIFE

    # Batch
    game_runner.SELF_PLAY_BATCH_SIZE = SELF_PLAY_BATCH_SIZE
    game_runner.SELF_PLAY_CONCURRENT_BATCH_MULT = SELF_PLAY_CONCURRENT_BATCH_MULT
    game_runner.SELF_PLAY_CHUNKS = SELF_PLAY_CHUNKS

    # Gating
    game_runner.GATING_PANEL_SIZE = GATING_PANEL_SIZE
    game_runner.GATING_PANEL_WIN_RATE = GATING_PANEL_WIN_RATE
    game_runner.GATING_BEST_WIN_RATE = GATING_BEST_WIN_RATE

    # Iterations
    game_runner.bootstrap_iters = bootstrap_iters
    game_runner.start = start
    game_runner.iters = iters


if __name__ == "__main__":
    print(f"=== Star Gambit Training ===")
    print(f"Game: {game_name}")
    print(f"Network: {network_name} {depth}d {channels}c {kernel_size}k")
    print(f"MCTS depth: {nn_selfplay_mcts_depth}")
    print(f"Iterations: {iters}")
    print()

    # Configure game_runner with our settings
    configure_game_runner()

    # Read game_runner.py and extract the main block
    game_runner_path = game_runner.__file__
    with open(game_runner_path, 'r') as f:
        source = f.read()

    # Find and extract the main block content
    main_marker = 'if __name__ == "__main__":'
    main_start = source.find(main_marker)
    if main_start == -1:
        raise RuntimeError("Could not find main block in game_runner.py")

    # Get the main block (everything after the if statement, properly indented)
    main_block = source[main_start + len(main_marker):]

    # Dedent the main block (remove one level of indentation)
    import textwrap
    main_block = textwrap.dedent(main_block)

    # Execute the main block in game_runner's namespace
    exec(main_block, game_runner.__dict__)
