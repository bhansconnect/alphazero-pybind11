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
            print(f"      Warning: hit max_games={max_games} before reaching "
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

    print(f"      ({games_played} games -> pool of {len(pool)} unique, "
          f"{duplicates_skipped} dups skipped, sampling "
          f"{min(target_count, len(pool))})")

    if len(pool) <= target_count:
        return pool
    idx = np.random.choice(len(pool), size=target_count, replace=False)
    return [pool[int(i)] for i in idx]


def ensure_snapshot(config, paths: dict, experiment_name: str,
                    anchor_iter: int) -> bool:
    """Snapshot game states for this anchor if not already done.

    Returns:
      True  if snapshot exists (created now or previously).
      False if the anchor checkpoint isn't on disk yet (silent skip;
            try again next iter).
    """
    if anchor_iter <= 0:
        return False
    if snapshot_exists(paths, anchor_iter):
        return True

    import neural_net
    Game = config.Game
    ckpt_path = os.path.join(paths["checkpoint"], f"{anchor_iter:04d}-{experiment_name}.pt")
    if not os.path.isfile(ckpt_path):
        return False  # anchor iter not reached / fully rotated out

    print(f"  Frozen eval: generating snapshot for anchor iter {anchor_iter} ...")
    t0 = time.time()
    nn = neural_net.NNWrapper.load_checkpoint(
        Game, paths["checkpoint"], f"{anchor_iter:04d}-{experiment_name}.pt"
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
        print(f"    Generating {count} positions ({label}) at {visits} visits "
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
    print(f"  Frozen eval: saved {len(all_states)} positions in "
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


def _eval_states(nn, config, states: list, visits: int) -> list:
    """Run MCTS + raw forward on each state. Returns per-state metric dicts."""
    from play import run_mcts_search
    Game = config.Game
    batch_size = max(1, min(16, int(visits ** 0.5)))

    out = []
    for entry in states:
        gs = entry["gs"]
        mcts = _make_base_mcts(config, Game)
        counts, _sims, mcts_value = run_mcts_search(
            gs, nn, mcts,
            node_limit=visits,
            eval_type="network",
            max_batch_size=batch_size,
        )
        counts = counts.astype(np.float64)
        total = counts.sum()
        pi_mcts = counts / total if total > 0 else np.full_like(counts, 1.0 / len(counts))

        # Raw network policy/value: one forward pass on the canonical
        canonical = np.array(gs.canonicalized())
        batch = torch.from_numpy(canonical[None, ...]).to(nn.device).float()
        with torch.no_grad():
            out_v, out_pi = nn.nnet(batch)
        # Network heads return log-softmax probabilities
        pi_raw = torch.exp(out_pi).squeeze(0).cpu().numpy().astype(np.float64)
        v_raw = torch.exp(out_v).squeeze(0).cpu().numpy().astype(np.float64)
        # Mask invalid moves in raw policy for fair comparison
        valid_mask = (counts > 0) | (pi_mcts > 0)
        if valid_mask.any():
            pi_raw_masked = pi_raw * (counts >= 0)  # keep all; raw includes invalids
        # Use top-1 over valid moves only via pi_mcts argmax (MCTS only visits valids)
        argmax_mcts = int(np.argmax(pi_mcts))
        argmax_raw = int(np.argmax(pi_raw))

        # Value comparison: MCTS root value is W/L/D probabilities; first entry is "win for current player"
        # network's v_raw is also W/L/D probabilities. Compare win prob for current player.
        v_mcts = np.asarray(mcts_value, dtype=np.float64)

        out.append({
            "variant": entry["variant"],
            "move_number": entry["move_number"],
            "kl_mcts_net": _kl(pi_mcts, pi_raw),
            "value_mae": float(np.mean(np.abs(v_raw[:len(v_mcts)] - v_mcts))),
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
