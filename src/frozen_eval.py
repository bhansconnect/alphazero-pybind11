"""Frozen eval set: live-MCTS diagnostic on a fixed set of game states
captured at a chosen anchor iteration.

Diagnostic question (Grill et al. 2020): how big is the policy-improvement
gap `KL(pi_MCTS_current || pi_net_raw_current)` on a fixed position set?
Falling = network is absorbing what search finds. Plateaued high = network
is capacity-bound (search keeps finding things the net can't represent).
Near zero = search isn't earning visits; either grow net or at a ceiling.

This metric is anchor-quality-independent: only the *positions* come from
the anchor; both sides of the comparison (MCTS, raw) use the *current*
network. The anchor only matters insofar as the positions it explores are
diverse enough to be informative.

Snapshot generation: when an anchor is hit and no snapshot exists, load the
anchor checkpoint and run a brief burst of self-play with it to capture
~`frozen_eval_positions` game states total, sampled across variants in
proportion to `variant_fractions`. Game states are pickled directly to a
permanent location outside the reservoir — all 17 game classes implement
`py::pickle` via `to_bytes/from_bytes` on the GameState interface.
"""

import glob
import hashlib
import os
import pickle
import tempfile
import time
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

import alphazero


UNIFIED_VARIANT_NAMES = ["skirmish", "showdown", "clash", "battle"]


def frozen_set_dir(paths: dict, anchor_iter: int) -> str:
    """Permanent location for snapshots — outside the reservoir-managed history/ tree."""
    return os.path.join(paths["experiment"], "frozen_eval", f"anchor_{anchor_iter:04d}")


def _states_path(paths: dict, anchor_iter: int) -> str:
    return os.path.join(frozen_set_dir(paths, anchor_iter), "states.pkl")


def snapshot_exists(paths: dict, anchor_iter: int) -> bool:
    """True iff the snapshot file exists AND is non-empty. Rejects truncated
    files left over from a previous failed write."""
    p = _states_path(paths, anchor_iter)
    return os.path.isfile(p) and os.path.getsize(p) > 0


# ---------------------------------------------------------------------------
# MCTS helpers (mirrors mcts_analysis._make_base_mcts but local to keep
# frozen_eval self-contained)
# ---------------------------------------------------------------------------

def _make_base_mcts(config, Game):
    """Eval-mode MCTS: no Dirichlet noise, root temperature 1.0."""
    num_players = Game.NUM_PLAYERS()
    num_moves = Game.NUM_MOVES()
    relative_values = Game().relative_values()
    return alphazero.MCTS(
        config.cpuct, num_players, num_moves,
        0.0,  # dirichlet_alpha
        1.0,  # mcts_root_temp
        config.fpu_reduction, relative_values,
        config.root_fpu_zero, config.shaped_dirichlet,
    )


def _make_game_for_variant(config, variant_id: Optional[int]):
    """Create a game state, optionally forcing a variant for unified games.

    variant_id is 0..3 for unified games, None otherwise.
    """
    Game = config.Game
    if config.game == "star_gambit_unified" and variant_id is not None:
        probs = [0.0] * 4
        probs[variant_id] = 1.0
        return Game(probs=probs)
    return Game()


def _is_unified(config) -> bool:
    return config.game == "star_gambit_unified"


def linfit_slope(pts) -> float:
    """Least-squares slope of y vs x for (x, y) pairs. NaN for <2 points.

    Used for trajectory analysis of metrics like KL gap and effective rank
    across training iterations. Gracefully handles non-consecutive iter
    points (e.g., when only some iters have frozen_eval data due to
    interval > 1 or partial backfill) — the linear regression doesn't
    require evenly-spaced x values.
    """
    pts = list(pts)
    if len(pts) < 2:
        return float("nan")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return float("nan")
    num = sum((x - mx) * (y - my) for x, y in pts)
    return num / den


def find_checkpoint(primary_dir: str, experiment_name: str, iteration: int,
                    fallback_dir: Optional[str] = None) -> Optional[str]:
    """Locate the checkpoint .pt for `iteration`, with bootstrap fallback.

    Search order:
      1. {primary_dir}/{iteration:04d}-{experiment_name}.pt
      2. {fallback_dir}/{iteration:04d}-{experiment_name}.pt
      3. {fallback_dir}/{iteration:04d}-*.pt   (bootstrap source may use
         a different experiment name)

    Returns the full path of the first match, or None if not found.
    """
    primary = os.path.join(primary_dir, f"{iteration:04d}-{experiment_name}.pt")
    if os.path.isfile(primary):
        return primary
    if not fallback_dir:
        return None
    secondary = os.path.join(fallback_dir, f"{iteration:04d}-{experiment_name}.pt")
    if os.path.isfile(secondary):
        return secondary
    cps = sorted(glob.glob(os.path.join(fallback_dir, f"{iteration:04d}-*.pt")))
    return cps[0] if cps else None


def print_kl_health(run, history, *, epoch, step, anchor_iter) -> None:
    """Compute and log the rolling-window slope of frozen-eval KL, print
    a one-line health summary.

    `history` is a list of (iter, mean_kl_across_variants) tuples. Slope is
    fitted in KL-per-iter units across all points in the window. Linear
    regression handles non-consecutive iters (gaps from `frozen_eval_interval`
    > 1 or partial backfill) cleanly.

    Status classifications:
      near-converged: KL < 0.05 (no slope check needed, already low)
      absorbing:     slope < -0.0015 (KL falling at meaningful rate)
      regressing ⚠:  slope >  0.0015 (network actively losing ground)
      stuck ⚠:       |slope| ≤ 0.0015 and KL > 0.10
      stable:        |slope| ≤ 0.0015 and KL ≤ 0.10

    Only emits a slope when there are 3+ points; otherwise prints a "need
    more points" line so the trajectory is visible from the first eval.

    `run` may be a Dummy (no-op track) for the no-aim case; the print
    happens either way.
    """
    if len(history) < 3:
        if history:
            it_now, kl_now = history[-1]
            tqdm.write(f"  frozen_eval health: KL={kl_now:.3f}  "
                       f"(slope needs 3+ points; have {len(history)})")
        return

    current_kl = history[-1][1]
    span = history[-1][0] - history[0][0]
    slope = linfit_slope(history)
    n = len(history)

    if current_kl < 0.05:
        status = "near-converged"
    elif slope < -0.0015:
        status = "absorbing"
    elif slope > 0.0015:
        status = "regressing ⚠"
    else:
        status = "stuck ⚠" if current_kl > 0.10 else "stable"

    if run is not None:
        try:
            run.track(
                slope, name="frozen_eval/kl_slope_recent",
                epoch=epoch, step=step,
                context={
                    "anchor_iter": f"{anchor_iter:04d}",
                    "window_points": str(n),
                },
            )
        except Exception:
            pass

    sign = "+" if slope >= 0 else ""
    tqdm.write(
        f"  frozen_eval health: KL={current_kl:.3f}  "
        f"slope={sign}{slope:.4f}/iter "
        f"over {n} pts (span {span} iters)  [{status}]"
    )


def read_bootstrap_fallback(experiment_dir: str) -> Optional[str]:
    """Return the source checkpoint dir for a bootstrapped experiment, or None.

    Bootstrap mode writes {experiment_dir}/bootstrap_meta.json with the
    source's checkpoint dir. Used so anchor checkpoints that pre-date the
    new run can still be resolved.
    """
    import json
    p = os.path.join(experiment_dir, "bootstrap_meta.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f).get("source_checkpoint_dir")
    except Exception:
        return None


def _variant_targets(config, total: int) -> dict:
    """Per-variant target counts {variant_id: count} summing to total.

    For non-unified games, returns {None: total}.
    """
    if not _is_unified(config) or not config.variant_fractions:
        return {None: total}
    # Normalize fractions in case they don't quite sum to 1.0
    raw = {
        UNIFIED_VARIANT_NAMES.index(name): float(frac)
        for name, frac in config.variant_fractions.items()
        if name in UNIFIED_VARIANT_NAMES
    }
    s = sum(raw.values()) or 1.0
    raw = {vid: f / s for vid, f in raw.items()}
    targets = {vid: int(round(total * frac)) for vid, frac in raw.items()}
    # Fix rounding drift
    diff = total - sum(targets.values())
    if diff != 0 and targets:
        # Give extra/short to the largest bucket
        largest = max(targets, key=lambda k: targets[k])
        targets[largest] += diff
    return {vid: c for vid, c in targets.items() if c > 0}


# ---------------------------------------------------------------------------
# Snapshot generation: burst self-play with anchor checkpoint
# ---------------------------------------------------------------------------

def _resolve_anchor_checkpoint(paths: dict, experiment_name: str, anchor_iter: int):
    """Load the anchor checkpoint, or return None if not yet on disk."""
    import neural_net
    path = os.path.join(paths["checkpoint"], f"{anchor_iter:04d}-{experiment_name}.pt")
    if not os.path.isfile(path):
        return None
    Game_cls = None  # will be set by caller; just verify file exists here
    return path


_POOL_MULTIPLIER = 4   # pool size target = target_count * this
_MAX_GAMES_FACTOR = 10  # hard ceiling: stop after this * min_games even if
                        # pool target isn't met (small state space safeguard)


def _canonical_key(gs):
    """Hash of canonical-form board for deduplication. Same canonical means
    same network input means same MCTS/raw output, so it's the right dedup
    granularity. blake2b is fast and has no fixed-seed nondeterminism."""
    canonical = np.asarray(gs.canonicalized())
    return hashlib.blake2b(canonical.tobytes(), digest_size=16).digest()




def _burst_capture_one_variant(config, anchor_nn, variant_id: Optional[int],
                                target_count: int, visits: int,
                                min_games: int):
    """Play complete games with anchor_nn, deduplicate positions by canonical
    form, then uniformly sample target_count states from the pool.

    Stops after a complete game once BOTH:
      - pool size >= target_count * _POOL_MULTIPLIER (sampling-rate guarantee)
      - games_played >= min_games (source-diversity guarantee)

    Hard ceiling at min_games * _MAX_GAMES_FACTOR games to bound the worst
    case (small state space where dedup keeps rejecting; warn and proceed).

    Games are never truncated mid-play. Each game contributes its full set
    of unique-canonical positions to the pool.
    """
    from play import run_mcts_search
    Game = config.Game

    batch_size = max(1, min(16, int(visits ** 0.5)))
    pool_target = target_count * _POOL_MULTIPLIER
    max_games = max(min_games * _MAX_GAMES_FACTOR, 50)

    pool = []
    seen_keys = set()
    games_played = 0
    duplicates_skipped = 0

    while True:
        if len(pool) >= pool_target and games_played >= min_games:
            break
        if games_played >= max_games:
            tqdm.write(f"      Warning: hit max_games={max_games} before reaching "
                       f"pool target ({len(pool)}/{pool_target}); state space "
                       f"may be small. Proceeding with what we have.")
            break

        gs = _make_game_for_variant(config, variant_id)
        mcts = _make_base_mcts(config, Game)
        move_number = 0

        while gs.scores() is None:
            key = _canonical_key(gs)
            if key in seen_keys:
                duplicates_skipped += 1
            else:
                seen_keys.add(key)
                # Game states are picklable directly (via py::pickle on each
                # game class). gs.copy() gives us an independent snapshot.
                pool.append({
                    "gs": gs.copy(),
                    "variant": variant_id,
                    "move_number": move_number,
                    "player": int(gs.current_player()),
                })

            counts, _sims, _wld = run_mcts_search(
                gs, anchor_nn, mcts,
                node_limit=visits,
                eval_type="network",
                max_batch_size=batch_size,
            )
            # Stochastic action selection (temp 1.0 over counts) keeps games
            # diverse across the burst — argmax would funnel every game down
            # the same opening line.
            counts = counts.astype(np.float64)
            total = counts.sum()
            if total <= 0:
                break
            probs = counts / total
            action = int(np.random.choice(len(probs), p=probs))
            gs.play_move(action)
            mcts = _make_base_mcts(config, Game)
            move_number += 1
        games_played += 1

    tqdm.write(f"      ({games_played} games -> pool of {len(pool)} unique, "
               f"{duplicates_skipped} dups skipped, sampling "
               f"{min(target_count, len(pool))})")

    if len(pool) <= target_count:
        return pool
    idx = np.random.choice(len(pool), size=target_count, replace=False)
    return [pool[int(i)] for i in idx]


def ensure_snapshot(config, paths: dict, experiment_name: str,
                    anchor_iter: int,
                    fallback_checkpoint_dir: Optional[str] = None) -> bool:
    """Snapshot game states for this anchor if not already done.

    Args:
        fallback_checkpoint_dir: bootstrap source's checkpoint dir, for
            anchor iters that pre-date this experiment. If None, the
            caller can pass `read_bootstrap_fallback(experiment_dir)`.

    Returns:
      True  if snapshot exists (created now or previously).
      False if the anchor checkpoint isn't on disk yet anywhere we can
            reach (silent skip; try again next iter).
    """
    if anchor_iter <= 0:
        return False
    if snapshot_exists(paths, anchor_iter):
        return True

    import neural_net
    Game = config.Game
    ckpt_path = find_checkpoint(
        paths["checkpoint"], experiment_name, anchor_iter,
        fallback_dir=fallback_checkpoint_dir,
    )
    if ckpt_path is None:
        return False  # anchor iter not reached / fully rotated / not in bootstrap source

    src_note = "" if os.path.dirname(ckpt_path) == paths["checkpoint"] else \
               f" (from bootstrap source: {os.path.dirname(ckpt_path)})"
    tqdm.write(f"  Frozen eval: generating snapshot for anchor iter {anchor_iter}{src_note} ...")
    t0 = time.time()
    nn = neural_net.NNWrapper.load_checkpoint(
        Game, os.path.dirname(ckpt_path), os.path.basename(ckpt_path)
    )
    # Don't enable inference graphs — the snapshot run is short and graph
    # warmup overhead dominates. Plain forward passes are fine.

    total = max(1, int(config.frozen_eval_positions))
    visits = max(1, int(config.selfplay_mcts_visits))
    min_games = max(1, int(getattr(config, "frozen_eval_min_games", 20)))
    targets = _variant_targets(config, total)

    all_states = []
    for variant_id, count in targets.items():
        if count <= 0:
            continue
        label = (
            f"variant {UNIFIED_VARIANT_NAMES[variant_id]}"
            if variant_id is not None else "default"
        )
        tqdm.write(f"    Generating {count} positions ({label}) at {visits} visits "
                   f"(min {min_games} games) ...")
        states = _burst_capture_one_variant(
            config, nn, variant_id, count, visits, min_games,
        )
        all_states.extend(states)

    snap_dir = frozen_set_dir(paths, anchor_iter)
    os.makedirs(snap_dir, exist_ok=True)
    final_path = _states_path(paths, anchor_iter)
    # Atomic write: tempfile in same dir, then os.replace. Prevents an empty
    # or partial file from being left behind on a crash mid-pickle.
    fd, tmp_path = tempfile.mkstemp(dir=snap_dir, suffix=".pkl.tmp")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(all_states, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, final_path)  # atomic on POSIX
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    tqdm.write(f"  Frozen eval: saved {len(all_states)} positions in "
               f"{time.time() - t0:.1f}s to {final_path}")

    del nn
    import gc
    gc.collect()
    return True


def _load_states(paths: dict, anchor_iter: int) -> list:
    with open(_states_path(paths, anchor_iter), "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Evaluation: run MCTS on each frozen state with the current network,
# compare against the current network's raw policy.
# ---------------------------------------------------------------------------

def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """KL(p || q). Skip terms where p==0; epsilon-smooth q."""
    mask = p > 0
    if not mask.any():
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / (q[mask] + eps))))


def _batched_raw_forward(nn, gs_list, chunk_size: int = 512):
    """Compute raw network (v, pi) probabilities for many game states with one
    batched GPU pass. Replaces 1024 single-sample forwards with ceil(N/chunk)
    well-utilized forwards — biggest standalone speedup vs the old loop.

    Returns (v_array [N, num_players+1], pi_array [N, num_moves]) as float64.
    """
    canonicals = np.stack([np.array(gs.canonicalized()) for gs in gs_list])
    n = canonicals.shape[0]
    nn.nnet.eval()
    v_chunks, pi_chunks = [], []
    with torch.no_grad():
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            batch = torch.from_numpy(canonicals[s:e]).to(
                nn.device, non_blocking=True).float()
            out_v, out_pi = nn.nnet(batch)
            # Network heads emit log-softmax; convert to probabilities.
            v_chunks.append(torch.exp(out_v).cpu().numpy())
            pi_chunks.append(torch.exp(out_pi).cpu().numpy())
    return (np.concatenate(v_chunks, axis=0).astype(np.float64),
            np.concatenate(pi_chunks, axis=0).astype(np.float64))


def _eval_states(nn, config, states: list, visits: int,
                 cache_size: int = 200_000,
                 gpu_chunk_size: int = 512) -> list:
    """Parallel batched MCTS across all frozen positions.

    Architecture:
      * One MCTS instance per position, all run concurrently.
      * Each round, every active position calls find_leaf, producing a leaf
        canonical that needs a network evaluation. Cache hits short-circuit;
        cache misses are stacked into one big GPU batch (split into chunks
        of gpu_chunk_size for memory safety).
      * After the batch returns, results are distributed via process_result
        on each MCTS, sims_done advances by 1 per active position, and any
        positions that hit the visit budget drop out of the active set.
      * Raw network policy/value for all positions is precomputed once with
        a single batched forward at the start — no per-position single-shot
        forward in the metric extraction.

    Compared to the prior sequential implementation:
      * ~5-8× faster on 1024 positions × 120 visits.
      * GPU utilization rises from ~5% (batch≈11) to near-saturation (~512).
      * Cross-position transposition hits via the shared S3FIFOCache (every
        repeated canonical across any pair of MCTS trees is a free skip).

    Memory: holds N MCTS instances + N (small) game states. For typical
    N=1024 with 120 visits each, peak memory is a few hundred MB — fine.
    """
    Game = config.Game
    n = len(states)
    if n == 0:
        return []

    # One up-front batched forward for the raw policy/value used in the
    # diagnostic comparisons. Replaces 1024 single-sample forwards.
    raw_v_all, raw_pi_all = _batched_raw_forward(nn, [s["gs"] for s in states])

    # Per-position MCTS state.
    mctss = [_make_base_mcts(config, Game) for _ in range(n)]
    gss = [s["gs"] for s in states]
    sims_done = [0] * n
    is_first = [True] * n  # 4th arg to process_result, True only on first sim

    # Shared S3FIFOCache: cross-position transposition hits are free skips.
    # Cache is per-call (fresh each evaluate_checkpoint); same NN, so insert
    # results stay valid within this call.
    from cache_utils import create_cache
    cache = create_cache(Game, cache_size) if cache_size > 0 else None

    num_players = Game.NUM_PLAYERS()
    num_moves = Game.NUM_MOVES()

    nn.nnet.eval()
    active = list(range(n))
    while active:
        # Stage 1: find_leaf on every active position, separate hits and misses.
        miss_idxs = []
        miss_keys = []
        miss_canonicals = []
        hits = []  # list of (idx, v_np, pi_np)
        for idx in active:
            leaf = mctss[idx].find_leaf(gss[idx])
            h = alphazero.hash_game_state(leaf)
            cached = None
            if cache is not None:
                cached = cache.find(h, num_moves, num_players + 1)
            if cached is not None:
                pi_cached, v_cached = cached  # cache stores (pi, v)
                hits.append((idx,
                             np.asarray(v_cached, dtype=np.float32),
                             np.asarray(pi_cached, dtype=np.float32)))
            else:
                miss_idxs.append(idx)
                miss_keys.append(h)
                miss_canonicals.append(np.array(leaf.canonicalized()))

        # Stage 2: batched GPU forward on the misses, in memory-safe chunks.
        miss_results = []
        if miss_canonicals:
            canonical_batch = np.stack(miss_canonicals)
            m = canonical_batch.shape[0]
            with torch.no_grad():
                for s in range(0, m, gpu_chunk_size):
                    e = min(s + gpu_chunk_size, m)
                    batch = torch.from_numpy(canonical_batch[s:e]).to(
                        nn.device, non_blocking=True).float()
                    out_v, out_pi = nn.nnet(batch)
                    v_chunk = torch.exp(out_v).cpu().numpy().astype(np.float32)
                    pi_chunk = torch.exp(out_pi).cpu().numpy().astype(np.float32)
                    for j in range(e - s):
                        miss_results.append((v_chunk[j], pi_chunk[j]))
            # Insert into cache and process results.
            for j, idx in enumerate(miss_idxs):
                v, pi = miss_results[j]
                if cache is not None:
                    cache.insert(miss_keys[j], pi, v)
                mctss[idx].process_result(gss[idx], v, pi, is_first[idx])
                is_first[idx] = False
                sims_done[idx] += 1

        # Stage 3: process cache hits.
        for idx, v, pi in hits:
            mctss[idx].process_result(gss[idx], v, pi, is_first[idx])
            is_first[idx] = False
            sims_done[idx] += 1

        # Drop positions that hit the visit budget.
        active = [i for i in active if sims_done[i] < visits]

    # Extract per-position metrics. Same shape as the prior sequential version,
    # so evaluate_checkpoint downstream logic is unchanged.
    out = []
    for idx, entry in enumerate(states):
        counts = np.array(mctss[idx].counts(), dtype=np.float64)
        total = counts.sum()
        pi_mcts = (counts / total) if total > 0 else np.full_like(
            counts, 1.0 / len(counts))
        v_mcts = np.asarray(mctss[idx].root_value(), dtype=np.float64)

        raw_v = raw_v_all[idx]
        raw_pi = raw_pi_all[idx]

        argmax_mcts = int(np.argmax(pi_mcts))
        argmax_raw = int(np.argmax(raw_pi))

        out.append({
            "variant": entry["variant"],
            "move_number": entry["move_number"],
            "kl_mcts_net": _kl(pi_mcts, raw_pi),
            "value_mae": float(np.mean(np.abs(raw_v[:len(v_mcts)] - v_mcts))),
            "top1_agreement": 1.0 if argmax_mcts == argmax_raw else 0.0,
        })
    return out


def log_metrics_to_aim(run, metrics: dict, anchor_iter: int,
                       epoch, step) -> None:
    """Track per-variant metrics and a variant-averaged 'mean' trace to aim.

    metrics: {variant_name: {metric_name: float}} as returned by
        evaluate_checkpoint. The "mean" trace averages across variants.

    Single source of truth used by both the live training-loop hook and the
    backfill CLI; keeps name/context shape in sync.
    """
    if not metrics:
        return
    anchor_ctx = f"{anchor_iter:04d}"
    # Per-variant traces.
    for variant_name, vm in metrics.items():
        for metric_name, value in vm.items():
            run.track(
                value,
                name=f"frozen_eval/{metric_name}",
                epoch=epoch, step=step,
                context={"variant": variant_name, "anchor_iter": anchor_ctx},
            )
    # Variant-averaged trace — single line per metric across all variants.
    # Convention (matches per-variant aggregate metrics elsewhere in the
    # repo, e.g. gating's panel win_rate): omit the "variant" context key
    # on the aggregate trace; per-variant lines include it.
    metric_names = set()
    for vm in metrics.values():
        metric_names.update(vm.keys())
    for metric_name in metric_names:
        vals = [vm[metric_name] for vm in metrics.values() if metric_name in vm]
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        run.track(
            avg,
            name=f"frozen_eval/{metric_name}",
            epoch=epoch, step=step,
            context={"anchor_iter": anchor_ctx},
        )


def evaluate_checkpoint(nn, config, paths: dict, anchor_iter: int) -> dict:
    """Evaluate the given current-iter NN on the frozen state set.

    Returns:
      {variant_name: {metric_name: float}} aggregated across all states in
      that variant. Non-unified games report under key "default".
      Empty dict if no snapshot exists.
    """
    if not snapshot_exists(paths, anchor_iter):
        return {}

    states = _load_states(paths, anchor_iter)
    if not states:
        return {}

    visits = max(1, int(config.selfplay_mcts_visits))
    per_state = _eval_states(nn, config, states, visits)

    # Group by variant_id (None for non-unified)
    by_variant = {}
    for entry in per_state:
        vid = entry["variant"]
        by_variant.setdefault(vid, []).append(entry)

    results = {}
    for vid, rows in by_variant.items():
        name = UNIFIED_VARIANT_NAMES[vid] if vid is not None else "default"
        results[name] = {
            "kl_mcts_net": float(np.mean([r["kl_mcts_net"] for r in rows])),
            "value_mae": float(np.mean([r["value_mae"] for r in rows])),
            "top1_agreement": float(np.mean([r["top1_agreement"] for r in rows])),
        }
    return results
