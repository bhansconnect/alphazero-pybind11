#!/usr/bin/env python3
"""Training CLI.

Usage:
    python src/train.py configs/connect4.yaml
    python src/train.py configs/star_gambit_skirmish.yaml --iterations 400 --depth 6
    python src/train.py configs/connect4.yaml --experiment my-experiment-v2
    python src/train.py configs/connect4.yaml --experiment densenet-4d-12c-5k-100sims --start 50
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
import game_runner


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Training")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--experiment", default=None,
        help="Explicit experiment name (overrides auto-generated name)"
    )
    parser.add_argument(
        "--base-dir", default="data",
        help="Base data directory (default: data)"
    )

    # Parse known args, treat rest as --key val config overrides
    args, remaining = parser.parse_known_args()

    # Parse remaining as --key val pairs
    overrides = {}
    i = 0
    while i < len(remaining):
        if remaining[i].startswith("--"):
            key = remaining[i][2:]
            if i + 1 < len(remaining) and not remaining[i + 1].startswith("--"):
                overrides[key] = remaining[i + 1]
                i += 2
            else:
                overrides[key] = "true"
                i += 1
        else:
            print(f"Warning: ignoring unexpected argument: {remaining[i]}")
            i += 1

    return args, overrides


def main():
    args, overrides = parse_args()

    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config, overrides)

    experiment_dir = config.resolve_experiment_dir(
        base=args.base_dir,
        explicit_name=args.experiment,
    )

    print(f"Game: {config.game}")
    print(f"Experiment: {os.path.basename(experiment_dir)}")
    print(f"Directory: {experiment_dir}")
    print(f"Network: {config.network_name} {config.depth}d {config.channels}c {config.kernel_size}k")
    print(f"MCTS depth: {config.selfplay_mcts_depth}")
    print(f"Iterations: {config.start} -> {config.iterations}")
    print()

    game_runner.main(config, experiment_dir)


if __name__ == "__main__":
    main()
