#!/usr/bin/env python3
"""Training CLI.

Usage:
    python src/train.py configs/connect4.yaml
    python src/train.py configs/star_gambit_skirmish.yaml --iterations 400 --depth 6
    python src/train.py configs/connect4.yaml --experiment my-experiment-v2
    python src/train.py --resume data/star_gambit_skirmish/densenet-4d-16c-3k-120sims
    python src/train.py --resume data/star_gambit_skirmish/densenet-4d-16c-3k-120sims --iterations 600
    python src/train.py --bootstrap data/star_gambit_skirmish/densenet-4d-16c-3k-120sims
    python src/train.py --bootstrap data/star_gambit_skirmish/exp1 --channels 32
"""

import argparse
import dataclasses
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, TrainConfig, find_latest_checkpoint
import game_runner


def _build_config_epilog():
    """Build epilog listing all overridable TrainConfig fields."""
    lines = ["Config overrides (--key value):"]
    for f in dataclasses.fields(TrainConfig):
        if f.type is list or f.type == "list":
            continue
        type_name = f.type if isinstance(f.type, str) else f.type.__name__
        default = f.default if f.default is not dataclasses.MISSING else ""
        flag_name = f.name.replace("_", "-")
        lines.append(f"  --{flag_name:30s} {type_name:5s}  (default: {default})")
    return "\n".join(lines)


def parse_args():
    epilog = _build_config_epilog()
    parser = argparse.ArgumentParser(
        description="AlphaZero Training",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [-h] [config] [--resume PATH] [--bootstrap PATH] "
              "[--experiment NAME] [--base-dir DIR] [overrides...]",
    )
    # config is extracted manually from remaining args to avoid
    # parse_known_args consuming override values as the positional
    # (e.g. "--bootstrap dir --depth 30" would set config="30")
    parser.add_argument(
        "--resume", default=None, metavar="PATH",
        help="Path to experiment directory to resume from"
    )
    parser.add_argument(
        "--bootstrap", default=None, metavar="PATH",
        help="Path to source experiment directory to bootstrap from"
    )
    parser.add_argument(
        "--experiment", default=None, metavar="NAME",
        help="Explicit experiment name (overrides auto-generated name)"
    )
    parser.add_argument(
        "--base-dir", default="data", metavar="DIR",
        help="Base data directory (default: data)"
    )

    args, remaining = parser.parse_known_args()

    # Parse --key val overrides first, collect leftover non-flag args
    overrides = {}
    non_flag = []
    i = 0
    while i < len(remaining):
        if remaining[i].startswith("--"):
            key = remaining[i][2:].replace("-", "_")
            if i + 1 < len(remaining) and not remaining[i + 1].startswith("--"):
                overrides[key] = remaining[i + 1]
                i += 2
            else:
                overrides[key] = "true"
                i += 1
        else:
            non_flag.append(remaining[i])
            i += 1

    # First non-flag leftover is the config path
    args.config = non_flag[0] if non_flag else None
    for extra in non_flag[1:]:
        print(f"Warning: ignoring unexpected argument: {extra}")

    return args, overrides


def _check_existing_experiment(experiment_dir, explicit_name):
    """If an explicit experiment name collides with an existing directory, ask the user."""
    if not explicit_name or not os.path.exists(experiment_dir):
        return
    print(f"\nWarning: experiment '{explicit_name}' already exists at {experiment_dir}")
    print("  [o] Overwrite - delete all existing data and start fresh")
    print("  [r] Resume    - continue from latest checkpoint")
    print("  [a] Abort")
    while True:
        choice = input("\nChoice [o/r/a]: ").strip().lower()
        if choice in ("o", "r", "a"):
            break
        print("Please enter 'o', 'r', or 'a'.")
    if choice == "a":
        print("Aborted.")
        sys.exit(0)
    return choice


def main():
    args, overrides = parse_args()

    modes = sum([args.config is not None, args.resume is not None, args.bootstrap is not None])
    if modes > 1:
        print("Error: specify only one of: config file, --resume, or --bootstrap")
        sys.exit(1)
    if modes == 0:
        print("Error: must specify a config file, --resume <dir>, or --bootstrap <dir>")
        sys.exit(1)

    if args.resume:
        # Resume mode
        experiment_dir = args.resume.rstrip("/")
        config_path = os.path.join(experiment_dir, "config.yaml")
        if not os.path.exists(config_path):
            print(f"Error: no config.yaml found in {experiment_dir}")
            sys.exit(1)
        config = load_config(config_path, overrides)

        checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
        start = find_latest_checkpoint(checkpoint_dir)
        if start == 0:
            print(f"Error: no checkpoints found in {checkpoint_dir}")
            sys.exit(1)
        if start >= config.iterations:
            print(f"Error: checkpoint is at iteration {start} but --iterations is {config.iterations}")
            sys.exit(1)

        bootstrap_from = ""

    elif args.bootstrap:
        # Bootstrap mode
        source_dir = args.bootstrap.rstrip("/")
        config_path = os.path.join(source_dir, "config.yaml")
        if not os.path.exists(config_path):
            print(f"Error: no config.yaml found in {source_dir}")
            sys.exit(1)
        config = load_config(config_path, overrides)

        experiment_dir = config.resolve_experiment_dir(
            base=args.base_dir,
            explicit_name=args.experiment,
        )
        choice = _check_existing_experiment(experiment_dir, args.experiment)
        if choice == "o":
            shutil.rmtree(experiment_dir)
            start = 0
            bootstrap_from = source_dir
        elif choice == "r":
            checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
            start = find_latest_checkpoint(checkpoint_dir)
            if start == 0:
                print(f"Error: no checkpoints found in {checkpoint_dir}, cannot resume")
                sys.exit(1)
            bootstrap_from = ""
        else:
            start = 0
            bootstrap_from = source_dir

    else:
        # Fresh start
        if not os.path.exists(args.config):
            print(f"Error: config file not found: {args.config}")
            sys.exit(1)

        config = load_config(args.config, overrides)
        experiment_dir = config.resolve_experiment_dir(
            base=args.base_dir,
            explicit_name=args.experiment,
        )
        choice = _check_existing_experiment(experiment_dir, args.experiment)
        if choice == "o":
            shutil.rmtree(experiment_dir)
            start = 0
        elif choice == "r":
            checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
            start = find_latest_checkpoint(checkpoint_dir)
            if start == 0:
                print(f"Error: no checkpoints found in {checkpoint_dir}, cannot resume")
                sys.exit(1)
        else:
            start = 0
        bootstrap_from = ""

    print(f"Game: {config.game}")
    print(f"Experiment: {os.path.basename(experiment_dir)}")
    print(f"Directory: {experiment_dir}")
    print(f"Network: {config.network_name} {config.depth}d {config.channels}c {config.kernel_size}k")
    print(f"MCTS visits: {config.selfplay_mcts_visits}")
    if config.max_cache_size > 0:
        game = config.Game
        entry_bytes = (game.NUM_MOVES() + game.NUM_PLAYERS() + 1) * 4 + 208
        cache_mb = config.max_cache_size * entry_bytes / (1024 * 1024)
        print(f"Cache: {config.max_cache_size:,} entries, ~{cache_mb:.0f} MB")
    print(f"Iterations: {start} -> {config.iterations}")
    print(f"LR schedule: {config.lr_schedule}")
    if bootstrap_from:
        print(f"Bootstrap from: {bootstrap_from}")
    print()

    game_runner.main(config, experiment_dir, start=start, bootstrap_from=bootstrap_from)


if __name__ == "__main__":
    main()
