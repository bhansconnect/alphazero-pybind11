"""Interactive backfill for frozen eval metrics.

Use when you've set `frozen_eval_anchor_iter` mid-training and want the
metric trace populated from iter 0 (or some earlier point) to current.
Also works for re-running evals against a different anchor.

UX mirrors src/mcts_analysis.py — fully interactive, no CLI flags required.
"""

import os
import re
import sys

import numpy as np
import yaml

# Make sure src/ is on the path when run as a script
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import tqdm

import frozen_eval
from config import TrainConfig, load_config
from run_discovery import (
    aim_hash_for_experiment,
    discover_experiments,
    discover_games,
    experiment_dir_from_checkpoint,
    prompt_int,
    prompt_yes_no,
    select_experiment_interactive,
    select_game_interactive,
)


CONFIGS_DIR = os.path.join(os.path.dirname(_SRC_DIR), "configs")


def _load_experiment_config(game_name: str, experiment_dir: str) -> TrainConfig:
    """Load the experiment's saved config.yaml if present, falling back to
    the game's default config in configs/."""
    saved = os.path.join(experiment_dir, "config.yaml")
    if os.path.isfile(saved):
        return load_config(saved, {}, warn=False)
    default_path = os.path.join(CONFIGS_DIR, f"{game_name}.yaml")
    if os.path.isfile(default_path):
        return load_config(default_path, {}, warn=False)
    return TrainConfig(game=game_name)


def _extract_experiment_name(checkpoint_filename: str) -> str:
    """0050-densenet-4d-16c-3k-300sims.pt -> densenet-4d-16c-3k-300sims"""
    m = re.match(r"^\d+-(.+)\.pt$", checkpoint_filename)
    if not m:
        raise ValueError(f"Unexpected checkpoint filename: {checkpoint_filename}")
    return m.group(1)


def _resume_aim_run(experiment_dir: str):
    """Resume the aim Run for this experiment, or return a Dummy that no-ops."""
    try:
        import aim
    except ImportError:
        print("aim not installed; metrics will print only, not log.")
        class _Dummy:
            def track(self, *a, **kw):
                pass
        return _Dummy()

    hash_ = aim_hash_for_experiment(experiment_dir)
    if not hash_:
        print("  Warning: no .aim_run_hash in experiment dir; creating a new run.")
        return aim.Run(experiment=os.path.basename(experiment_dir))
    return aim.Run(run_hash=hash_)


def main():
    print("=" * 60)
    print("Frozen eval backfill")
    print("=" * 60)

    games = discover_games()
    if not games:
        print("No experiments with checkpoints found under data/. Nothing to do.")
        return

    game_name = select_game_interactive(games)
    print()

    experiments = games[game_name]
    exp_name, checkpoints = select_experiment_interactive(experiments)
    print()

    sample_ckpt_path = checkpoints[0][1]
    experiment_dir = experiment_dir_from_checkpoint(sample_ckpt_path)
    experiment_name = _extract_experiment_name(os.path.basename(sample_ckpt_path))

    config = _load_experiment_config(game_name, experiment_dir)
    Game = config.Game
    paths = config.resolve_paths(experiment_dir)

    iters = sorted(i for i, _ in checkpoints)
    earliest, latest = iters[0], iters[-1]
    print(f"Available iter range: {earliest} .. {latest}  ({len(iters)} checkpoints)")
    print()

    anchor = prompt_int("Anchor iteration", default=latest, lo=earliest, hi=latest)
    start = prompt_int("Start iter (inclusive)", default=earliest, lo=earliest, hi=latest)
    end = prompt_int("End iter (inclusive)", default=latest, lo=start, hi=latest)
    interval = prompt_int("Interval (every N iters)", default=1, lo=1)
    print()

    # Resume the aim run; first time around, this also confirms write access.
    run = _resume_aim_run(experiment_dir)

    # Ensure the snapshot. This may take 30-60s if not already created.
    if not frozen_eval.ensure_snapshot(config, paths, experiment_name, anchor):
        print(f"\nError: anchor checkpoint {anchor:04d}-{experiment_name}.pt not "
              f"found. Pick a different anchor.")
        return

    iters_to_run = [it for it in iters if start <= it <= end and (it - start) % interval == 0]
    print(f"\nWill run frozen eval for {len(iters_to_run)} checkpoints against "
          f"anchor {anchor:04d}.")
    if not prompt_yes_no("Proceed?", default=True):
        return

    import neural_net  # deferred to avoid import cost when user bails out

    pbar = tqdm.tqdm(iters_to_run, desc="Backfilling", unit="iter")
    for it in pbar:
        pbar.set_postfix({"iter": it})
        try:
            nn = neural_net.NNWrapper.load_checkpoint(
                Game, paths["checkpoint"], f"{it:04d}-{experiment_name}.pt",
            )
        except Exception as exc:
            print(f"  Skipping iter {it}: {exc}")
            continue
        metrics = frozen_eval.evaluate_checkpoint(nn, config, paths, anchor)
        frozen_eval.log_metrics_to_aim(
            run, metrics, anchor_iter=anchor, epoch=it, step=int(it),
        )
        del nn
        import gc
        gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()
