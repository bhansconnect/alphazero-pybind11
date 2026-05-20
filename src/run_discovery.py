"""Shared interactive helpers for discovering games, experiments, and aim runs.

Used by the diagnostic CLI scripts (frozen_eval_backfill, visit_sweep_elo).
Initial implementation lifts existing logic from play.py and mcts_analysis.py
so the new scripts get the same UX without copy-paste. Existing scripts can
migrate to these helpers in a future cleanup.
"""

import glob
import os
import re

# Enable line-editing in input() prompts (arrow keys, history, backspace).
# Importing readline is enough — Python wires it into the input() builtin
# globally for the rest of this process.
try:
    import readline  # noqa: F401  (side-effect import)
except ImportError:
    pass

from config import GAME_REGISTRY


def discover_experiments(game_name: str, base: str = "data") -> dict:
    """Return {experiment_name: [(iter_num, full_path), ...]} for one game.

    Each experiment's checkpoints sorted by iteration descending.
    Empty dict if no checkpoints found.
    """
    game_dir = os.path.join(base, game_name)
    if not os.path.isdir(game_dir):
        return {}

    experiments = {}
    for exp_name in sorted(os.listdir(game_dir)):
        checkpoint_dir = os.path.join(game_dir, exp_name, "checkpoint")
        if not os.path.isdir(checkpoint_dir):
            continue

        checkpoints = []
        for pt_file in glob.glob(os.path.join(checkpoint_dir, "*.pt")):
            filename = os.path.basename(pt_file)
            match = re.match(r"^(\d+).*\.pt$", filename)
            if match:
                checkpoints.append((int(match.group(1)), pt_file))

        if checkpoints:
            checkpoints.sort(key=lambda x: x[0], reverse=True)
            experiments[exp_name] = checkpoints

    return experiments


def discover_games(base: str = "data") -> dict:
    """Return {game_name: {experiment_name: [(iter_num, path), ...]}}.

    Only includes games that exist in GAME_REGISTRY and have checkpoints.
    """
    if not os.path.isdir(base):
        return {}

    games = {}
    for entry in sorted(os.listdir(base)):
        if entry not in GAME_REGISTRY:
            continue
        experiments = discover_experiments(entry, base)
        if experiments:
            games[entry] = experiments
    return games


def select_game_interactive(games: dict) -> str:
    """Prompt for a game from the discovered set. Returns the selected name."""
    names = sorted(games.keys())
    if len(names) == 1:
        print(f"Game: {names[0]}")
        return names[0]

    print("Available games:")
    for i, name in enumerate(names):
        n_exps = len(games[name])
        print(f"  {i+1}. {name} ({n_exps} experiment{'s' if n_exps != 1 else ''})")

    while True:
        choice = input("Select game [1]: ").strip()
        if choice == "":
            return names[0]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(names):
                return names[idx]
        except ValueError:
            if choice in names:
                return choice
        print(f"Enter 1-{len(names)} or a game name")


def select_experiment_interactive(experiments: dict) -> tuple:
    """Prompt for an experiment. Returns (experiment_name, checkpoints_list)."""
    names = list(experiments.keys())
    if len(names) == 1:
        print(f"Experiment: {names[0]} ({len(experiments[names[0]])} checkpoints)")
        return names[0], experiments[names[0]]

    print("Available experiments:")
    for i, name in enumerate(names):
        cpts = experiments[name]
        print(f"  {i+1}. {name} ({len(cpts)} ckpts, latest: iter {cpts[0][0]:04d})")

    while True:
        choice = input("Select experiment [1]: ").strip()
        if choice == "":
            return names[0], experiments[names[0]]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(names):
                return names[idx], experiments[names[idx]]
        except ValueError:
            pass
        print(f"Enter 1-{len(names)}")


def select_checkpoint_interactive(checkpoints: list, prompt: str = "Select checkpoint") -> tuple:
    """Prompt for one checkpoint from a sorted-descending list.

    Returns (iter_num, path). Supports shortcuts:
      Enter / l  = latest
      -N         = back N entries (so -0 = latest, -1 = previous, ...)
      iN         = specific iteration number
      <index>    = position in the displayed list
    """
    latest_iter, latest_path = checkpoints[0]
    print(f"\nCheckpoints (newest first, {len(checkpoints)} total):")
    print(f"  l. Latest -> iter {latest_iter:04d}")
    show = min(10, len(checkpoints))
    for i in range(show):
        print(f"  {i}. iter {checkpoints[i][0]:04d}")
    if len(checkpoints) > show:
        print(f"  ... ({len(checkpoints) - show} more; use iN for specific iter)")
    print("  Shortcuts: Enter=latest, -N=back N, iN=specific iter")

    while True:
        choice = input(f"{prompt} [latest]: ").strip().lower()
        if choice in ("", "l", "latest"):
            return checkpoints[0]
        if choice.startswith("-"):
            try:
                idx = -int(choice)
                if 0 <= idx < len(checkpoints):
                    return checkpoints[idx]
            except ValueError:
                pass
            print(f"  Need -0..-{len(checkpoints)-1}")
            continue
        m = re.match(r"^i(\d+)$", choice)
        if m:
            target = int(m.group(1))
            for it, path in checkpoints:
                if it == target:
                    return it, path
            print(f"  Iter {target} not in checkpoint list")
            continue
        try:
            idx = int(choice)
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx]
            print(f"  Enter 0-{len(checkpoints)-1}")
        except ValueError:
            print("  Invalid input")


def experiment_dir_from_checkpoint(checkpoint_path: str) -> str:
    """data/{game}/{exp}/checkpoint/{iter}-{name}.pt -> data/{game}/{exp}/"""
    return os.path.dirname(os.path.dirname(checkpoint_path))


def aim_hash_for_experiment(experiment_dir: str) -> str:
    """Return aim run hash from .aim_run_hash in experiment dir, or empty string.

    The training loop writes this file when it opens / creates the aim Run.
    Scripts that want to resume the same run read it.
    """
    path = os.path.join(experiment_dir, ".aim_run_hash")
    if not os.path.isfile(path):
        return ""
    with open(path) as f:
        return f.read().strip()


def prompt_int(label: str, default: int, lo: int = None, hi: int = None) -> int:
    """Prompt for an integer with optional bounds. Returns default on empty input."""
    bounds = ""
    if lo is not None and hi is not None:
        bounds = f" [{lo}-{hi}]"
    elif lo is not None:
        bounds = f" [>= {lo}]"
    elif hi is not None:
        bounds = f" [<= {hi}]"
    while True:
        raw = input(f"{label}{bounds} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            v = int(raw)
        except ValueError:
            print("  Need an integer")
            continue
        if lo is not None and v < lo:
            print(f"  Must be >= {lo}")
            continue
        if hi is not None and v > hi:
            print(f"  Must be <= {hi}")
            continue
        return v


def prompt_yes_no(label: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{label} {suffix}: ").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes")
