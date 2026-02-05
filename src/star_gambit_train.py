#!/usr/bin/env python3
"""Training script for Star Gambit AI.

This script configures game_runner.py for Star Gambit training.
Supports all three variants: Skirmish, Clash, and Battle.
Run with: python src/star_gambit_train.py
"""

import os
import sys

# Ensure we can import from the src directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
import game_runner

# === Game Variants ===
VARIANTS = {
    '1': ('skirmish', alphazero.StarGambitSkirmishGS, '3F, 1C, 0D - 39 actions, 5-side board'),
    '2': ('clash', alphazero.StarGambitClashGS, '3F, 2C, 1D - 55 actions, 5-side board'),
    '3': ('battle', alphazero.StarGambitBattleGS, '4F, 3C, 2D - 75 actions, 6-side board'),
}

# Variant-specific opening lengths (half-life of temperature decay)
OPENING_LENGTHS = {
    'skirmish': 8,
    'clash': 10,
    'battle': 12,
}

# === Game Configuration (set by select_variant) ===
Game = None
game_name = None
variant_name = None


def select_variant():
    """Prompt user to select a game variant."""
    global Game, game_name, variant_name

    print("Select Star Gambit variant:")
    for key, (name, _, desc) in VARIANTS.items():
        print(f"  {key}. {name.capitalize()} ({desc})")

    variant_input = input("Variant (1/2/3) [1]: ").strip() or '1'
    if variant_input not in VARIANTS:
        print(f"Invalid selection '{variant_input}', defaulting to Skirmish")
        variant_input = '1'

    variant_name, Game, _ = VARIANTS[variant_input]
    game_name = f"star_gambit_{variant_name}"
    print(f"Selected: {variant_name.capitalize()}\n")


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
SELF_PLAY_TEMP = 1.0               # Temperature during self-play
EVAL_TEMP = 0.5                    # Temperature during evaluation
FINAL_TEMP = 0.2                   # Final temperature

# === Batch Configuration ===
SELF_PLAY_BATCH_SIZE = 128         # Games per batch
SELF_PLAY_CONCURRENT_BATCH_MULT = 2
SELF_PLAY_CHUNKS = 4
TRAIN_BATCH_SIZE = 1024            # Training batch size
TRAIN_SAMPLE_RATE = 1              # Training sample multiplier

# === Gating Thresholds ===
GATING_PANEL_SIZE = 1
GATING_PANEL_WIN_RATE = 0.52
GATING_BEST_WIN_RATE = 0.52

# === Resignation ===
RESIGN_PERCENT = 0.02              # Resign if win prob < this
RESIGN_PLAYTHROUGH_PERCENT = 0.20  # % of resignations played out anyway

# === Cache Configuration ===
MAX_CACHE_SIZE = 200_000           # MCTS position cache size
CACHE_SHARDS = None                # None = use cpu_count()

# === History Window ===
HIST_SIZE = 30_000                 # History buffer size per batch
WINDOW_SIZE_ALPHA = 0.5            # History window growth rate
WINDOW_SIZE_BETA = 0.7             # History window slope
WINDOW_SIZE_SCALAR = 6             # History window base scale

# === Workers ===
RESULT_WORKERS = 2                 # Result processing threads
DATA_WORKERS = None                # None = use cpu_count() - 1

# === Comparison ===
compare_past = 20                  # Iterations back to compare ELO

# === Training Iterations ===
bootstrap_iters = 0
start = 0
iters = 200
lr_milestone = 150

# === Derived (set in main after variant selection) ===
network_name = None
run_name = None


def configure_game_runner():
    """Override game_runner module globals with Star Gambit settings."""
    game_runner.Game = Game
    game_runner.game_name = game_name
    game_runner.new_game = new_game

    # Variant-specific data paths
    game_runner.CHECKPOINT_LOCATION = os.path.join("data", "checkpoint", variant_name)
    game_runner.HIST_LOCATION = os.path.join("data", "history", variant_name)
    game_runner.TMP_HIST_LOCATION = os.path.join("data", "tmp_history", variant_name)

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
    opening_length = OPENING_LENGTHS[variant_name]
    game_runner.EXPECTED_OPENING_LENGTH = opening_length
    game_runner.SELF_PLAY_TEMP = SELF_PLAY_TEMP
    game_runner.EVAL_TEMP = EVAL_TEMP
    game_runner.FINAL_TEMP = FINAL_TEMP
    game_runner.TEMP_DECAY_HALF_LIFE = opening_length

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

    # Training batch
    game_runner.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
    game_runner.TRAIN_SAMPLE_RATE = TRAIN_SAMPLE_RATE

    # Resignation
    game_runner.RESIGN_PERCENT = RESIGN_PERCENT
    game_runner.RESIGN_PLAYTHROUGH_PERCENT = RESIGN_PLAYTHROUGH_PERCENT

    # Cache
    game_runner.MAX_CACHE_SIZE = MAX_CACHE_SIZE
    if CACHE_SHARDS is not None:
        game_runner.CACHE_SHARDS = CACHE_SHARDS

    # History window
    game_runner.HIST_SIZE = HIST_SIZE
    game_runner.WINDOW_SIZE_ALPHA = WINDOW_SIZE_ALPHA
    game_runner.WINDOW_SIZE_BETA = WINDOW_SIZE_BETA
    game_runner.WINDOW_SIZE_SCALAR = WINDOW_SIZE_SCALAR

    # Workers
    game_runner.RESULT_WORKERS = RESULT_WORKERS
    if DATA_WORKERS is not None:
        game_runner.DATA_WORKERS = DATA_WORKERS

    # Comparison
    game_runner.compare_past = compare_past


if __name__ == "__main__":
    print(f"=== Star Gambit Training ===\n")

    # Select game variant first
    select_variant()

    # Update derived values now that game_name is set
    network_name = "densenet" if dense_net else "resnet"
    run_name = f"{game_name}-{network_name}-{depth}d-{channels}c-{kernel_size}k-{nn_selfplay_mcts_depth}sims"

    print(f"Game: {game_name}")
    print(f"Network: {network_name} {depth}d {channels}c {kernel_size}k")
    print(f"MCTS depth: {nn_selfplay_mcts_depth}")
    print(f"Iterations: {iters}")
    print(f"Checkpoint path: data/checkpoint/{variant_name}/")
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
