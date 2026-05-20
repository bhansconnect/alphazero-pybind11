"""Interactive visit-sweep Elo curve.

For a single test checkpoint (or a range, sweeping iter-by-iter), play it at
several MCTS visit budgets against a fixed anchor checkpoint at its standard
visits. Per-budget win rate -> Elo delta. Curve slope = marginal value of
search at the test network's current state.

k=0 means "raw policy, no MCTS" — internally implemented as mcts_visits=2,
which (with default PUCT + FPU) reduces to argmax of the network prior.

Asymmetric: anchor stays at config.compare_mcts_visits while test sweeps k.
This is the cleanest Jones-style saturation curve.

Outputs:
- CSV at data/{game}/{exp}/visit_sweep/anchor_{K:04d}.csv (appends rows).
- Optional aim logging: per-(k, test_iter) `visit_sweep/elo_delta` traces
  plus a `visit_sweep/delta_search` scalar per test_iter.
"""

import csv
import math
import os
import re
import sys
import time

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import tqdm

import neural_net
from cache_utils import create_sharded_cache
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
from tournament import pit_agents


CONFIGS_DIR = os.path.join(os.path.dirname(_SRC_DIR), "configs")
RAW_POLICY_VISITS = 2  # mcts_visits=2 with default PUCT == argmax of net prior

# Variant id -> human-readable name (matches UNIFIED_VARIANT_NAMES in game_runner).
UNIFIED_VARIANT_NAMES = ["skirmish", "showdown", "clash", "battle"]


def _is_unified(config) -> bool:
    return getattr(config, "game", None) == "star_gambit_unified"


def _load_experiment_config(game_name: str, experiment_dir: str) -> TrainConfig:
    saved = os.path.join(experiment_dir, "config.yaml")
    if os.path.isfile(saved):
        return load_config(saved, {}, warn=False)
    default_path = os.path.join(CONFIGS_DIR, f"{game_name}.yaml")
    if os.path.isfile(default_path):
        return load_config(default_path, {}, warn=False)
    return TrainConfig(game=game_name)


def _extract_experiment_name(checkpoint_filename: str) -> str:
    m = re.match(r"^\d+-(.+)\.pt$", checkpoint_filename)
    if not m:
        raise ValueError(f"Unexpected checkpoint filename: {checkpoint_filename}")
    return m.group(1)


def _parse_visits_input(raw: str, default: list) -> list:
    if raw == "":
        return default
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            print(f"  Skipping non-integer visit budget: {tok!r}")
    return out or default


def _elo_delta(win_rate: float) -> float:
    """Two-player Elo gap from win rate (with draws weighted 0.5)."""
    wr = max(0.001, min(0.999, win_rate))
    return 400.0 * math.log10(wr / (1.0 - wr))


def _resume_aim_run(experiment_dir: str):
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


def _run_sweep_one_iter(config, Game, paths, experiment_name,
                        test_iter: int, anchor_iter: int,
                        visits_list: list, games_per_point: int,
                        anchor_visits: int, cache_size: int):
    """Run the sweep for one test iter against the anchor.

    Single PM run per visit budget. For star_gambit_unified, games are split
    evenly across the 4 variants in one PM (default equal probs); per-variant
    win rates are extracted from PM's variant_perm_scores tracking. Aggregate
    win rate is the total across all variants for that budget.

    Returns list of (variant_name_or_None, k, win_rate, elo_delta) rows.
    The variant_name is None for the aggregate trace; a string per variant
    for the unified breakdown.
    """
    test_nn = neural_net.NNWrapper.load_checkpoint(
        Game, paths["checkpoint"], f"{test_iter:04d}-{experiment_name}.pt",
    )
    anchor_nn = neural_net.NNWrapper.load_checkpoint(
        Game, paths["checkpoint"], f"{anchor_iter:04d}-{experiment_name}.pt",
    )

    num_players = Game.NUM_PLAYERS()
    bs = max(1, games_per_point // max(1, num_players * num_players))

    players = [anchor_nn] * num_players
    players[0] = test_nn

    # One ShardedS3FIFOCache per unique model group. Shared across visit
    # budgets within this iter.
    test_cache = create_sharded_cache(Game, cache_size) if cache_size > 0 else None
    anchor_cache = create_sharded_cache(Game, cache_size) if cache_size > 0 else None
    caches = [test_cache, anchor_cache] if cache_size > 0 else None

    is_unified = _is_unified(config)
    rows = []
    for k in visits_list:
        effective_k = RAW_POLICY_VISITS if k == 0 else k
        visits = [anchor_visits] * num_players
        visits[0] = effective_k
        result = pit_agents(
            config, Game, players, visits, bs,
            name=f"sweep iter={test_iter} k={k}",
            caches=caches,
            return_per_variant=is_unified,
        )
        if is_unified:
            win_rates, per_variant = result
        else:
            win_rates, per_variant = result, {}

        # Aggregate trace (no variant key). For non-unified games this is the
        # only line; for unified, it averages all 4 variants' games.
        test_wr = win_rates[0]
        elo = _elo_delta(test_wr)
        rows.append((None, k, test_wr, elo))
        print(f"    k={k:>5d}: test_win_rate={test_wr:.3f}  elo_delta={elo:+.1f}")

        # Per-variant traces.
        for vid in sorted(per_variant.keys()):
            vname = (UNIFIED_VARIANT_NAMES[vid]
                     if 0 <= vid < len(UNIFIED_VARIANT_NAMES) else f"v{vid}")
            v_wr = per_variant[vid][0]
            v_elo = _elo_delta(v_wr)
            rows.append((vname, k, v_wr, v_elo))
            print(f"      {vname:>10s}: test_win_rate={v_wr:.3f}  elo_delta={v_elo:+.1f}")

    del test_nn, anchor_nn, test_cache, anchor_cache, caches
    import gc
    gc.collect()
    return rows


def _write_csv(csv_path: str, anchor_iter: int, anchor_visits: int,
               test_iter: int, rows: list):
    """Append rows to the per-anchor CSV, creating header if needed."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header_needed = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["anchor_iter", "anchor_visits", "test_iter",
                        "variant", "visits", "win_rate", "elo_delta"])
        for variant_name, k, wr, elo in rows:
            w.writerow([anchor_iter, anchor_visits, test_iter,
                        variant_name or "", k,
                        f"{wr:.6f}", f"{elo:.3f}"])


def main():
    print("=" * 60)
    print("Visit-sweep Elo curve")
    print("=" * 60)

    games = discover_games()
    if not games:
        print("No experiments with checkpoints found under data/. Nothing to do.")
        return

    game_name = select_game_interactive(games)
    print()
    exp_name, checkpoints = select_experiment_interactive(games[game_name])
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

    anchor_iter = prompt_int("Anchor iteration", default=latest,
                             lo=earliest, hi=latest)
    test_start = prompt_int("Test iter start", default=earliest, lo=earliest, hi=latest)
    test_end = prompt_int("Test iter end (inclusive)", default=latest,
                          lo=test_start, hi=latest)
    interval = prompt_int("Iter interval", default=max(1, (test_end - test_start) // 5 or 1), lo=1)
    anchor_visits = prompt_int(
        "Anchor MCTS visits (Enter for compare_mcts_visits)",
        default=int(config.compare_mcts_visits), lo=1,
    )

    default_visits = [0, 8, 32, 128, 512, 1024]
    print(f"Default visit budgets: {default_visits} (0 = raw policy, no MCTS)")
    visits_raw = input("Visit budgets (comma-separated, Enter for default): ").strip()
    visits_list = _parse_visits_input(visits_raw, default_visits)
    games_per_point = prompt_int("Games per visit point", default=100, lo=2)
    cache_size = prompt_int(
        "Per-net cache size (0 = disabled)",
        default=int(getattr(config, "max_cache_size", 200_000)), lo=0,
    )

    log_aim = prompt_yes_no("Log to aim run?", default=False)
    run = _resume_aim_run(experiment_dir) if log_aim else None

    test_iters = [it for it in iters
                  if test_start <= it <= test_end and (it - test_start) % interval == 0]
    total_matches = len(test_iters) * len(visits_list)
    print()
    print(f"Will sweep {len(test_iters)} test iters x {len(visits_list)} budgets "
          f"= {total_matches} PM runs (each ~{games_per_point} games).")
    if _is_unified(config):
        print(f"Games split across 4 variants per budget "
              f"(~{games_per_point // 4} games/variant).")
    print(f"Anchor plays at {anchor_visits} visits.")
    if not prompt_yes_no("Proceed?", default=True):
        return

    csv_path = os.path.join(experiment_dir, "visit_sweep", f"anchor_{anchor_iter:04d}.csv")
    t0 = time.time()
    for test_iter in tqdm.tqdm(test_iters, desc="Test iters"):
        print(f"\nTest iter {test_iter} vs anchor {anchor_iter}:")
        try:
            rows = _run_sweep_one_iter(
                config, Game, paths, experiment_name,
                test_iter, anchor_iter, visits_list, games_per_point,
                anchor_visits, cache_size,
            )
        except Exception as exc:
            print(f"  Skipping iter {test_iter}: {exc}")
            continue

        _write_csv(csv_path, anchor_iter, anchor_visits, test_iter, rows)

        if run is not None:
            base_ctx = {
                "anchor": f"{anchor_iter:04d}",
                "anchor_visits": str(anchor_visits),
                "test_iter": f"{test_iter:04d}",
            }
            # Per-k traces: aggregate (no variant key) and per-variant.
            for variant_name, k, _wr, elo in rows:
                ctx = dict(base_ctx)
                if variant_name is not None:
                    ctx["variant"] = variant_name
                run.track(
                    elo, name="visit_sweep/elo_delta",
                    epoch=test_iter, step=int(k), context=ctx,
                )
            # delta_search scalar per (variant or aggregate) per test_iter.
            ks = sorted({k for _v, k, _wr, _e in rows})
            elos_by_kv = {(v, k): e for v, k, _wr, e in rows}
            variant_names = sorted({v for v, _, _, _ in rows},
                                    key=lambda x: ("" if x is None else x))
            if len(ks) >= 2:
                k_low, k_high = min(ks), max(ks)
                ds_ctx_base = {
                    "anchor": f"{anchor_iter:04d}",
                    "anchor_visits": str(anchor_visits),
                    "k_low": str(k_low),
                    "k_high": str(k_high),
                }
                for vname in variant_names:
                    if (vname, k_low) in elos_by_kv and (vname, k_high) in elos_by_kv:
                        d = elos_by_kv[(vname, k_high)] - elos_by_kv[(vname, k_low)]
                        ctx = dict(ds_ctx_base)
                        if vname is not None:
                            ctx["variant"] = vname
                        run.track(d, name="visit_sweep/delta_search",
                                  epoch=test_iter, step=int(test_iter), context=ctx)
    print(f"\nDone in {time.time() - t0:.1f}s. CSV at {csv_path}")


if __name__ == "__main__":
    main()
