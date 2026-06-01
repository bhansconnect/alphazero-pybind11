#!/usr/bin/env python3
"""Unified interactive play agent with pluggable game UI.

Usage:
    python src/play.py                                   # auto-discover games
    python src/play.py star_gambit_skirmish               # by game name
    python src/play.py configs/star_gambit_skirmish.yaml   # by config file
    python src/play.py star_gambit_skirmish --think-time 5
    python src/play.py star_gambit_skirmish --nodes 200
"""

import argparse
import glob
import math
import os
import re
import sys
import time

import numpy as np
import readline  # noqa: F401 - Enable line editing in input()
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
from cache_utils import create_cache, cached_inference, print_cache_stats
from config import (
    GAME_REGISTRY,
    EXPERIMENT_DEFAULT_CPUCT,
    EXPERIMENT_DEFAULT_FPU_REDUCTION,
    EXPERIMENT_DEFAULT_GUMBEL_M,
    EXPERIMENT_DEFAULT_GUMBEL_C_VISIT,
    EXPERIMENT_DEFAULT_GUMBEL_C_SCALE,
    load_experiment_config,
    load_experiment_gumbel,
)
from action_selector import ActionSelector
from game_ui import get_game_ui

try:
    import torch
    import neural_net

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch/neural_net not available. AI features disabled.")


# Default AI parameters
DEFAULT_CPUCT = EXPERIMENT_DEFAULT_CPUCT
DEFAULT_FPU_REDUCTION = EXPERIMENT_DEFAULT_FPU_REDUCTION
DEFAULT_TEMPERATURE = 0.5
DEFAULT_NODE_LIMIT = 100

# Library defaults for Gumbel (mirror mcts.h). Used when a network has no
# config.yaml — and as a starting point that load_experiment_config tweaks.
DEFAULT_GUMBEL_M = EXPERIMENT_DEFAULT_GUMBEL_M
DEFAULT_GUMBEL_C_VISIT = EXPERIMENT_DEFAULT_GUMBEL_C_VISIT
DEFAULT_GUMBEL_C_SCALE = EXPERIMENT_DEFAULT_GUMBEL_C_SCALE


class PlayerConfig:
    """Configuration for a single player (human or AI)."""

    def __init__(self):
        self.is_ai = False
        self.network_path = None
        self.network = None
        self.eval_type = "random"  # "random", "playout", or "network"
        self.think_time = None
        self.node_limit = DEFAULT_NODE_LIMIT
        self.temperature = DEFAULT_TEMPERATURE
        self.mcts = None
        self.show_hints = False
        self.greedy = False
        self.batch_size = 0  # 0 = auto, 1 = off (sequential), >1 = fixed
        # Dirichlet root noise. 0 = off (matches tournament/PlayManager). Set
        # to e.g. 0.25 to add exploration noise like AlphaZero self-play.
        self.epsilon = 0.0
        # Visit-count pruning before temperature sampling. Zero out moves whose
        # visits are below max(1, prune_frac * total_visits). Only active when
        # temperature > 0 (greedy paths already ignore low-visit tails).
        self.prune_frac = 0.02
        # Lower-confidence-bound move selection (KataGo / LeelaZero style).
        # When enabled, the greedy/argmax path picks argmax(Q - z/sqrt(N))
        # instead of argmax(visits). Enabling LCB forces greedy=True because
        # LCB is a deterministic best-move selector and has no effect under
        # stochastic sampling.
        self.lcb_enabled = False
        self.lcb_z = 2.0
        # Gumbel opening-moves override: when the move number is below this
        # threshold AND Gumbel is enabled, force G3 (sample improved policy
        # with temperature) even if greedy is set, to preserve opening
        # diversity at the start of a game. 0 disables G2.
        self.gumbel_opening_moves = 0
        # Resign: when enabled, the player resigns if their expected outcome
        # (W − L from root_value) drops below ``resign_threshold`` for
        # ``resign_consecutive`` consecutive own moves. ``_resign_streak``
        # tracks the current run.
        self.resign_enabled = False
        self.resign_threshold = -0.95
        self.resign_consecutive = 3
        self._resign_streak = 0
        # Most recent principal variation from MCTS (list of action IDs).
        # Populated by get_ai_probs after each search; consumed by display.
        self._last_pv = []
        self._calibrated_timed_bs = None
        # Per-player search algorithm. ``algo_override`` is 'puct', 'gumbel',
        # or None (= use whatever the network's yaml specifies; default).
        # Defaulting to None means each network plays with the algorithm it
        # was trained with — Gumbel-trained nets use Gumbel, PUCT-trained nets
        # use PUCT — which the head-to-head comparison on the unified-gumbel
        # run showed clearly outperforms forcing PUCT (60–75% win rate across
        # checkpoints, especially at temp=0 where PUCT is deterministic and
        # Gumbel uniquely provides opening variance via fresh perturbations).
        # The gumbel_* fields are the currently-applied parameters, refreshed
        # each time the network is loaded or the algo override changes.
        self.algo_override = None
        self.gumbel_enabled = False
        self.gumbel_m = DEFAULT_GUMBEL_M
        self.gumbel_c_visit = DEFAULT_GUMBEL_C_VISIT
        self.gumbel_c_scale = DEFAULT_GUMBEL_C_SCALE
        self.gumbel_full = False
        # Gumbel params auto-detected from the network's config.yaml on load.
        # Stored separately so we can restore them when the user flips back to
        # ':auto' after an explicit override.
        self._yaml_gumbel = None

    def __str__(self):
        if not self.is_ai:
            hints_str = ", hints=on" if self.show_hints else ""
            return f"Human{hints_str}"
        if self.eval_type == "playout":
            net_str = "playout"
        elif self.network_path:
            net_str = os.path.basename(self.network_path)
        else:
            net_str = "random"
        time_str = f"{self.think_time}s" if self.think_time else "none"
        node_str = str(self.node_limit) if self.node_limit else "none"
        greedy_str = ", greedy" if self.greedy else ""
        batch_str = ""
        if self.batch_size != 1:
            batch_str = f", batch={'auto' if self.batch_size == 0 else self.batch_size}"
        algo_str = ""
        if self.eval_type == "network" or self.eval_type == "playout":
            actual = "gumbel" if self.gumbel_enabled else "puct"
            tag = f"{actual}(auto)" if self.algo_override is None else actual
            algo_str = f", algo={tag}"
        eps_str = f", eps={self.epsilon}" if self.epsilon > 0 else ""
        lcb_str = f", lcb=z{self.lcb_z}" if self.lcb_enabled else ""
        gopen_str = (
            f", gumbel-opening={self.gumbel_opening_moves}"
            if self.gumbel_enabled and self.gumbel_opening_moves > 0
            else ""
        )
        cscale_str = (
            f", cscale={self.gumbel_c_scale}"
            if self.gumbel_enabled and self.gumbel_c_scale != DEFAULT_GUMBEL_C_SCALE
            else ""
        )
        resign_str = (
            f", resign={self.resign_threshold}/{self.resign_consecutive}"
            if self.resign_enabled else ""
        )
        return f"AI(net={net_str}, nodes={node_str}, time={time_str}, temp={self.temperature}{greedy_str}{batch_str}{algo_str}{eps_str}{lcb_str}{gopen_str}{cscale_str}{resign_str})"


class PlayContext:
    """Game context for play sessions."""

    def __init__(self, game, game_class, cache_size=0):
        self.game = game
        self.game_class = game_class
        self.players = [PlayerConfig(), PlayerConfig()]
        self.auto_play = False
        self.auto_full = False
        self.cpuct = DEFAULT_CPUCT
        self.fpu_reduction = DEFAULT_FPU_REDUCTION
        self.cache_size = cache_size
        self.cache = create_cache(game_class, cache_size)
        self.player_caches = [None, None]
        self.tree_reuse = True
        self.auto_delay = 0.0


def _update_caches(ctx):
    """Rebuild shared or per-player caches based on current config."""
    if ctx.cache_size <= 0:
        ctx.cache = None
        ctx.player_caches = [None, None]
        return
    if any(p.think_time is not None for p in ctx.players):
        per_size = ctx.cache_size // 2
        ctx.cache = None
        ctx.player_caches = [
            create_cache(ctx.game_class, per_size),
            create_cache(ctx.game_class, per_size),
        ]
    else:
        ctx.cache = create_cache(ctx.game_class, ctx.cache_size)
        ctx.player_caches = [None, None]


def _get_cache(ctx, player_idx):
    """Return per-player cache if split, otherwise shared cache."""
    if ctx.player_caches[player_idx] is not None:
        return ctx.player_caches[player_idx]
    return ctx.cache


def create_mcts(game_class, cpuct=DEFAULT_CPUCT, fpu_reduction=DEFAULT_FPU_REDUCTION,
                epsilon=0.0, gumbel_enabled=False, gumbel_m=DEFAULT_GUMBEL_M,
                gumbel_c_visit=DEFAULT_GUMBEL_C_VISIT,
                gumbel_c_scale=DEFAULT_GUMBEL_C_SCALE, gumbel_full=False):
    """Create a new MCTS instance.

    Defaults to ``epsilon=0`` (no Dirichlet root noise) so play.py matches
    tournament.py / PlayManager (``PlayParams::epsilon=0.0``) by default;
    users can opt back in via the ``epsilon`` CLI flag or interactive command.
    """
    return alphazero.MCTS(
        cpuct,
        game_class.NUM_PLAYERS(),
        game_class.NUM_MOVES(),
        float(epsilon),  # epsilon (Dirichlet)
        1.0,   # root_policy_temp
        fpu_reduction,
        game_class().relative_values(),
        False,  # root_fpu_zero
        False,  # shaped_dirichlet
        bool(gumbel_enabled),
        int(gumbel_m),
        float(gumbel_c_visit),
        float(gumbel_c_scale),
        bool(gumbel_full),
    )


def advance_mcts_trees(ctx, action):
    """Advance all MCTS trees to reflect the played move (tree reuse)."""
    if not ctx.tree_reuse:
        for p in ctx.players:
            p.mcts = None
        return
    for p in ctx.players:
        if p.mcts is not None:
            try:
                p.mcts.update_root(ctx.game, action)
            except Exception:
                p.mcts = None


def apply_temperature(probs, temperature):
    """Apply temperature to probabilities."""
    if temperature == 0:
        result = np.zeros_like(probs)
        result[np.argmax(probs)] = 1.0
        return result
    elif temperature == 1.0:
        return probs
    else:
        scaled = np.power(probs + 1e-10, 1.0 / temperature)
        return scaled / scaled.sum()


def _compute_batch_size(budget):
    """Batch size as sqrt(budget), minimum 1."""
    return max(1, int(math.sqrt(budget)))


def _run_one_batch(gs, agent, mcts, batch_size, eval_type, cache, sims_so_far):
    """Collect leaves via WU-UCT until batch_size GPU misses or 2*batch_size attempts.

    Cache hits and terminals are backpropagated immediately so subsequent
    find_leaf_batched calls see updated tree statistics.

    Returns actual number of simulations processed (may exceed batch_size due to
    cache hits/terminals).
    """
    pending_gpu = []      # [(leaf_index, hash, root_noise)] needing GPU
    canonical_list = []   # numpy canonical states for GPU batch
    actual_sims = 0
    max_attempts = 2 * batch_size

    for attempt in range(max_attempts):
        if len(pending_gpu) >= batch_size:
            break

        leaf_gs = mcts.find_leaf_batched(gs)
        leaf_idx = mcts.in_flight_count() - 1
        root_noise = (sims_so_far + actual_sims == 0 and attempt == 0)

        # Terminal: process immediately
        if leaf_gs.scores() is not None:
            v = np.array(leaf_gs.scores())
            pi = np.zeros(gs.num_moves())
            mcts.process_result_batched(gs, leaf_idx, v, pi, root_noise)
            actual_sims += 1
            continue

        # Playout eval: process immediately
        if eval_type == "playout":
            v, pi = alphazero.playout_eval(leaf_gs)
            mcts.process_result_batched(gs, leaf_idx, np.array(v), np.array(pi), root_noise)
            actual_sims += 1
            continue

        # Random eval: process immediately
        if agent is None:
            v = np.full(gs.num_players() + 1, 1.0 / (gs.num_players() + 1))
            pi = np.ones(gs.num_moves()) / gs.num_moves()
            mcts.process_result_batched(gs, leaf_idx, v, pi, root_noise)
            actual_sims += 1
            continue

        # Cache check
        h = alphazero.hash_game_state(leaf_gs)
        if cache is not None:
            result = cache.find(h, leaf_gs.num_moves(), leaf_gs.num_players() + 1)
            if result is not None:
                pi_c, v_c = result
                mcts.process_result_batched(gs, leaf_idx, np.array(v_c),
                                            np.array(pi_c), root_noise)
                actual_sims += 1
                continue

        # Cache miss: queue for GPU
        canonical_list.append(np.array(leaf_gs.canonicalized()))
        pending_gpu.append((leaf_idx, h, root_noise))

    # Batched GPU inference for cache misses
    if pending_gpu:
        batch_tensor = torch.from_numpy(np.stack(canonical_list))
        v_batch, pi_batch = agent.process(batch_tensor)
        v_np, pi_np = v_batch.cpu().numpy(), pi_batch.cpu().numpy()
        for j, (leaf_idx, h, rn) in enumerate(pending_gpu):
            v, pi = v_np[j].flatten(), pi_np[j].flatten()
            if cache is not None:
                cache.insert(h, pi, v)
            mcts.process_result_batched(gs, leaf_idx, v, pi, rn)
        actual_sims += len(pending_gpu)

    mcts.reset_batch()
    return actual_sims


def calibrate_timed_batch(gs, agent, time_limit, eval_type, cache):
    """One-time calibration to find self-consistent batch size for timed search.

    Tests powers of 2, measuring throughput at each. Returns the largest B where
    B <= sqrt(throughput * time_limit) (self-consistent: batch size doesn't exceed
    sqrt of estimated budget at that throughput).

    Returns the chosen batch size (minimum 2).
    """
    game_class = type(gs)
    best_bs = 2

    bs = 2
    while True:
        mcts_obj = create_mcts(game_class)
        t0 = time.time()
        _run_one_batch(gs, agent, mcts_obj, bs, eval_type, cache, 0)
        elapsed = time.time() - t0

        if elapsed > 0:
            throughput = bs / elapsed
            total_budget = throughput * time_limit
            ideal_bs = math.sqrt(total_budget)

            if bs <= ideal_bs:
                best_bs = bs
            else:
                # Past the self-consistent point
                break

        bs *= 2

    return best_bs


def run_mcts_search(gs, agent, mcts, time_limit=None, node_limit=None, eval_type="random",
                    cache=None, max_batch_size=1):
    """Run MCTS search. Returns (visit_counts, num_simulations, wld).

    max_batch_size: 0=auto, 1=sequential (no batching), >1=fixed batch size.

    When ``mcts.gumbel_enabled()`` is true, ``set_gumbel_num_sims`` is called
    once before the search loop with the resolved sim budget. Gumbel requires
    a fixed budget; for the time-only case we estimate from the batch size.
    """
    if time_limit is None and node_limit is None:
        node_limit = DEFAULT_NODE_LIMIT

    # Determine effective batch size
    if max_batch_size == 0:
        # Auto mode
        if node_limit is not None:
            effective_bs = _compute_batch_size(node_limit)
        else:
            effective_bs = max(4, int(math.sqrt(100)))  # safe default for timed auto
    elif max_batch_size == 1:
        effective_bs = 1
    else:
        if node_limit is not None:
            effective_bs = min(_compute_batch_size(node_limit), max_batch_size)
        else:
            effective_bs = max_batch_size

    # Gumbel needs the sim target before the first find_leaf. When only a
    # time limit is set we don't know the true budget, so fall back to a
    # heuristic large enough to allow Sequential Halving to make progress.
    if mcts.gumbel_enabled():
        if node_limit is not None:
            gumbel_target = max(1, int(node_limit))
        else:
            gumbel_target = max(16, effective_bs * 16)
        mcts.set_gumbel_num_sims(gumbel_target)

    start = time.time()
    sims = 0

    # Sequential path (batch_size=1): original loop, zero overhead
    if effective_bs <= 1 and max_batch_size == 1:
        while True:
            if time_limit is not None and time.time() - start >= time_limit:
                break
            if node_limit is not None and sims >= node_limit:
                break

            leaf = mcts.find_leaf(gs)
            if eval_type == "playout":
                v, pi = alphazero.playout_eval(leaf)
                v = np.array(v)
                pi = np.array(pi)
            elif agent is None:
                v = np.full(gs.num_players() + 1, 1.0 / (gs.num_players() + 1))
                pi = np.ones(gs.num_moves()) / gs.num_moves()
            else:
                v, pi, _ = cached_inference(cache, leaf, agent)

            mcts.process_result(gs, v, pi, sims == 0)
            sims += 1
    else:
        # Batched path
        while True:
            if time_limit is not None and time.time() - start >= time_limit:
                break
            if node_limit is not None and sims >= node_limit:
                break
            actual = _run_one_batch(gs, agent, mcts, effective_bs, eval_type, cache, sims)
            sims += actual

    counts = np.array(mcts.counts())
    wld = np.array(mcts.root_value())
    return counts, sims, wld


def _effective_greedy(pcfg, move_number):
    """Apply Gumbel G2 opening-moves override.

    During the first ``gumbel_opening_moves`` plies of a game with a
    Gumbel-enabled player, force non-greedy sampling so the G3 path
    (gumbel_improved_policy with temperature) drives opening diversity.
    """
    if (pcfg.gumbel_enabled
            and pcfg.gumbel_opening_moves > 0
            and move_number < pcfg.gumbel_opening_moves):
        return False
    return pcfg.greedy


def get_ai_probs(ctx, player_idx, valids, move_number=0):
    """Get AI move probabilities. Returns (probs, source_str, sims, wld).

    When the player's MCTS is in Gumbel mode and greedy is enabled, the
    returned ``probs`` is a one-hot vector on ``mcts.gumbel_final_action()``
    so that the move selected matches the Sequential Halving result rather
    than naive visit-count argmax.

    For Gumbel players running in non-greedy mode (G3, including the G2
    opening-moves override), the returned distribution is
    ``gumbel_improved_policy()`` with temperature applied, so that what gets
    sampled matches what's displayed in hints. Visit-count pruning is
    skipped on this path since the improved policy is already Q-aware.
    """
    pcfg = ctx.players[player_idx]
    wld = None

    if pcfg.mcts is None:
        pcfg.mcts = create_mcts(
            ctx.game_class, ctx.cpuct, ctx.fpu_reduction,
            epsilon=pcfg.epsilon,
            gumbel_enabled=pcfg.gumbel_enabled,
            gumbel_m=pcfg.gumbel_m,
            gumbel_c_visit=pcfg.gumbel_c_visit,
            gumbel_c_scale=pcfg.gumbel_c_scale,
            gumbel_full=pcfg.gumbel_full,
        )

    should_search = (pcfg.think_time is not None and pcfg.think_time > 0) or (
        pcfg.node_limit is not None and pcfg.node_limit > 0
    )

    if should_search and (
        pcfg.network is not None or pcfg.eval_type == "playout" or TORCH_AVAILABLE
    ):
        # Resolve batch size (pre-game calibration handles reporting)
        player_cache = _get_cache(ctx, player_idx)
        batch_size = pcfg.batch_size
        if batch_size == 0:
            if pcfg.think_time is not None:
                if pcfg._calibrated_timed_bs is None:
                    pcfg._calibrated_timed_bs = calibrate_timed_batch(
                        ctx.game, pcfg.network, pcfg.think_time,
                        pcfg.eval_type, player_cache,
                    )
                batch_size = pcfg._calibrated_timed_bs
            elif pcfg.node_limit is not None:
                batch_size = _compute_batch_size(pcfg.node_limit)

        counts, sims, wld = run_mcts_search(
            ctx.game,
            pcfg.network,
            pcfg.mcts,
            time_limit=pcfg.think_time,
            node_limit=pcfg.node_limit,
            eval_type=pcfg.eval_type,
            cache=player_cache,
            max_batch_size=batch_size,
        )
        eff_greedy = _effective_greedy(pcfg, move_number)
        used_improved = False
        # G3: when Gumbel is enabled and we'll be sampling, prefer the
        # gumbel_improved_policy() distribution over raw visit counts. It
        # matches the paper's pi' target and naturally biases toward higher-Q
        # moves; we still apply temperature on top to honor the player's
        # temperature setting.
        if (pcfg.gumbel_enabled
                and not eff_greedy
                and pcfg.mcts is not None
                and sims > 0):
            try:
                ip = np.array(pcfg.mcts.gumbel_improved_policy())
                if ip.sum() > 0:
                    probs = ip.astype(float)
                    used_improved = True
            except Exception:
                pass

        if not used_improved:
            if counts.sum() > 0:
                probs = counts.astype(float) / counts.sum()
                # Visit-count pruning before temperature sampling: zero out
                # moves below the floor, but only when we'll actually sample
                # (temp > 0) and the player has not opted out. Greedy/argmax
                # already picks the head; pruning the tail only matters
                # under stochastic play. Skipped for Gumbel-improved-policy
                # since that distribution is already Q-aware.
                if (not eff_greedy
                        and pcfg.temperature > 0
                        and pcfg.prune_frac > 0):
                    floor = max(1.0, pcfg.prune_frac * counts.sum())
                    pruned = probs.copy()
                    pruned[counts < floor] = 0.0
                    pruned_sum = pruned.sum()
                    if pruned_sum > 0:
                        probs = pruned / pruned_sum
                    # else: pruning would zero everything — fall back
            else:
                probs = np.ones(ctx.game.NUM_MOVES()) / ctx.game.NUM_MOVES()
        # Report batch size in source string
        algo_tag = "g3" if used_improved else ("g1" if pcfg.gumbel_enabled else "")
        algo_tag = f", {algo_tag}" if algo_tag else ""
        if batch_size > 1:
            source = f"MCTS ({sims} sims, bs={batch_size}{algo_tag})"
        else:
            source = f"MCTS ({sims} sims{algo_tag})"
        # Capture principal variation for display. Failures (e.g. very small
        # sim budgets) are harmless — _last_pv stays an empty list.
        try:
            pv = pcfg.mcts.principal_variation(5)
            pcfg._last_pv = [int(a) for a in pv]
        except Exception:
            pcfg._last_pv = []
    else:
        sims = 0
        pcfg._last_pv = []
        if pcfg.eval_type == "playout":
            v, pi = alphazero.playout_eval(ctx.game)
            probs = np.array(pi)
            source = "playout (single rollout)"
        elif pcfg.network is not None:
            canonical = torch.from_numpy(np.array(ctx.game.canonicalized()))
            _, pi = pcfg.network.predict(canonical)
            probs = pi.cpu().numpy().flatten()
            source = "policy network"
        else:
            probs = np.ones(ctx.game.NUM_MOVES()) / ctx.game.NUM_MOVES()
            source = "uniform random"

    probs = apply_temperature(probs, pcfg.temperature)
    probs[valids == 0] = 0
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        valid_count = valids.sum()
        if valid_count > 0:
            probs[valids == 1] = 1.0 / valid_count

    return probs, source, sims, wld


def _format_pv(game, pv, ui):
    """Format a principal-variation move list for display. Walks a copy of
    ``game`` so each move can be rendered in its proper context via the
    game UI."""
    if not pv:
        return ""
    parts = []
    gs = game.copy()
    for action in pv:
        try:
            parts.append(ui.format_move(gs, int(action)))
            gs.play_move(int(action))
        except Exception:
            break
    return " ".join(parts)


def _print_pv(game, pcfg, ui):
    """Print the principal variation line if we have one to show."""
    if not pcfg._last_pv:
        return
    rendered = _format_pv(game, pcfg._last_pv, ui)
    if rendered:
        print(f"  PV: {rendered}")


def _check_resign(pcfg, wld):
    """Update the player's resign streak from a fresh root WLD and return
    True if the player should resign now. Caller-owned state lives on
    ``pcfg``; PUCT/Gumbel WLD semantics are identical (current-player
    perspective)."""
    if not pcfg.resign_enabled or wld is None:
        pcfg._resign_streak = 0
        return False
    try:
        v = float(wld[0]) - float(wld[1])  # expected score for current player
    except (TypeError, IndexError):
        pcfg._resign_streak = 0
        return False
    if v <= pcfg.resign_threshold:
        pcfg._resign_streak += 1
    else:
        pcfg._resign_streak = 0
    return pcfg._resign_streak >= max(1, pcfg.resign_consecutive)


def _greedy_ai_action(pcfg, probs, valids, sims):
    """Greedy AI move pick. Honors Gumbel final-action and LCB when enabled,
    falling back to argmax over ``probs``. Used by step/manual/auto-step
    paths that need the AI's chosen move without re-running get_ai_move."""
    if pcfg.is_ai and pcfg.gumbel_enabled and pcfg.mcts is not None and sims > 0:
        try:
            action = int(pcfg.mcts.gumbel_final_action())
            if valids[action] != 0:
                return action
        except Exception:
            pass
    if pcfg.is_ai and pcfg.lcb_enabled and sims > 0:
        lcb_action = _lcb_action(pcfg, valids)
        if lcb_action is not None and valids[lcb_action] != 0:
            return lcb_action
    return int(np.argmax(probs))


def _lcb_action(pcfg, valids):
    """Argmax over Q - z/sqrt(N) for visited valid moves.

    Returns None if no visited valid move exists or the MCTS lacks the
    needed accessors. Caller falls back to argmax(probs) in that case.
    """
    if pcfg.mcts is None:
        return None
    try:
        q = np.array(pcfg.mcts.root_q_values(), dtype=float)
        n = np.array(pcfg.mcts.counts(), dtype=float)
    except Exception:
        return None
    if q.size == 0 or n.size == 0:
        return None
    visited = (n > 0) & (valids != 0)
    if not visited.any():
        return None
    score = np.full_like(q, -np.inf)
    score[visited] = q[visited] - pcfg.lcb_z / np.sqrt(n[visited])
    return int(np.argmax(score))


def get_ai_move(ctx, player_idx, valids, greedy=False, move_number=0):
    """Get AI move. Returns (action, probs, source, sims, wld).

    Selection precedence in the greedy/argmax branch:
      1. Gumbel: ``mcts.gumbel_final_action()`` (Sequential Halving result)
      2. LCB (if ``pcfg.lcb_enabled``): argmax(Q - z/sqrt(N))
      3. Plain argmax over the post-temperature/pruned probability vector

    ``greedy`` is honored unless the Gumbel G2 opening-moves override forces
    sampling for the first plies of the game.
    """
    pcfg = ctx.players[player_idx]
    eff_greedy = greedy and _effective_greedy(pcfg, move_number)
    probs, source, sims, wld = get_ai_probs(ctx, player_idx, valids,
                                            move_number=move_number)
    if eff_greedy:
        action = _greedy_ai_action(pcfg, probs, valids, sims)
    else:
        action = np.random.choice(len(probs), p=probs)
    return action, probs, source, sims, wld


# ---------------------------------------------------------------------------
# Game and network discovery
# ---------------------------------------------------------------------------


def discover_checkpoints(game_name, base="data"):
    """Discover checkpoints from experiment directories.

    Returns: {experiment_name: [(iter_num, full_path), ...]}
    Each experiment's checkpoints sorted by iteration descending.
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
                iter_num = int(match.group(1))
                checkpoints.append((iter_num, pt_file))

        if checkpoints:
            checkpoints.sort(key=lambda x: x[0], reverse=True)
            experiments[exp_name] = checkpoints

    return experiments


def discover_games(base="data"):
    """Scan data/ for games with trained checkpoints.

    Returns: {game_name: {experiment_name: [(iter_num, path), ...]}}
    Only includes games that exist in GAME_REGISTRY.
    """
    if not os.path.isdir(base):
        return {}

    games = {}
    for entry in sorted(os.listdir(base)):
        if entry not in GAME_REGISTRY:
            continue
        experiments = discover_checkpoints(entry, base)
        if experiments:
            games[entry] = experiments

    return games


def _apply_player_gumbel(pcfg):
    """Refresh ``pcfg``'s gumbel_* fields from its current ``algo_override``
    and the cached yaml config (``_yaml_gumbel``).

    The override only flips ``gumbel_enabled``; m/c_visit/c_scale/full are
    inherited from the yaml so that a manual ':gumbel' uses the same Gumbel
    hyperparameters the network was trained with. Resets the player's MCTS
    so the next search rebuilds it with the new configuration.
    """
    y = pcfg._yaml_gumbel or {
        "gumbel_enabled": False,
        "gumbel_m": DEFAULT_GUMBEL_M,
        "gumbel_c_visit": DEFAULT_GUMBEL_C_VISIT,
        "gumbel_c_scale": DEFAULT_GUMBEL_C_SCALE,
        "gumbel_full": False,
    }
    was_gumbel = pcfg.gumbel_enabled
    if pcfg.algo_override == "puct":
        pcfg.gumbel_enabled = False
    elif pcfg.algo_override == "gumbel":
        pcfg.gumbel_enabled = True
    else:
        pcfg.gumbel_enabled = bool(y["gumbel_enabled"])
    pcfg.gumbel_m = int(y["gumbel_m"])
    pcfg.gumbel_c_visit = float(y["gumbel_c_visit"])
    pcfg.gumbel_c_scale = float(y["gumbel_c_scale"])
    pcfg.gumbel_full = bool(y["gumbel_full"])
    pcfg.mcts = None
    # Gumbel's paper-faithful behavior is deterministic argmax-over-Sequential-
    # Halving (G1). When a player transitions into Gumbel mode, default greedy
    # to True so that variance comes from fresh Gumbel perturbations each
    # search rather than visit-count sampling. Users can still flip greedy off
    # via the greedy command (which engages G3: sample improved policy).
    if pcfg.gumbel_enabled and not was_gumbel:
        pcfg.greedy = True


def load_network(game_class, path, players, ctx):
    """Load a network and assign to specified players.

    The gumbel_* fields on each player are refreshed from the network's
    experiment config.yaml (auto-detect). Existing per-player algo_override
    settings are preserved across reloads.
    """
    if path == "playout":
        for p in players:
            ctx.players[p].network = None
            ctx.players[p].network_path = None
            ctx.players[p].eval_type = "playout"
            ctx.players[p].mcts = None
            ctx.players[p]._yaml_gumbel = None
            _apply_player_gumbel(ctx.players[p])
        return True
    if not TORCH_AVAILABLE:
        print("Error: torch not available")
        return False
    try:
        net = neural_net.NNWrapper.load_checkpoint(
            game_class, os.path.dirname(path), os.path.basename(path)
        )
        net.enable_inference_optimizations()
        yaml_gumbel = load_experiment_gumbel(path)
        for p in players:
            ctx.players[p].network = net
            ctx.players[p].network_path = path
            ctx.players[p].eval_type = "network"
            ctx.players[p]._yaml_gumbel = dict(yaml_gumbel)
            _apply_player_gumbel(ctx.players[p])
        return True
    except Exception as e:
        print(f"Error loading network: {e}")
        return False


def select_checkpoint(checkpoints):
    """Select a checkpoint interactively with rich shortcuts.

    Returns path, None (random), or 'playout'.

    Shortcuts:
      Enter / l  = latest
      -N         = back N iterations from latest
      iN         = specific iteration number
      r          = random policy
      p          = playout policy
      <number>   = index from list
    """
    latest_iter, latest_path = checkpoints[0]

    print(f"\n  Checkpoints (newest first):")
    print(f"    l. Latest -> iter {latest_iter:04d}")
    print(f"    r. Random policy")
    print(f"    p. Playout policy")
    print("    " + "-" * 40)

    show = min(10, len(checkpoints))
    for i in range(show):
        iter_num, _ = checkpoints[i]
        print(f"    {i}. iter {iter_num:04d}")
    if len(checkpoints) > show:
        print(f"    ... ({len(checkpoints) - show} more, use -N to go back N from latest)")

    print("  Shortcuts: Enter=latest, -N=back N iters, iN=specific iter")

    while True:
        choice = input("  Select checkpoint: ").strip().lower()

        if choice in ["", "l", "latest"]:
            return latest_path
        if choice == "r":
            return None
        if choice == "p":
            return "playout"

        # Negative number = go back N from latest
        if choice.startswith("-"):
            try:
                offset = int(choice)  # negative
                idx = -offset
                if 0 <= idx < len(checkpoints):
                    return checkpoints[idx][1]
                print(f"  Only {len(checkpoints)} checkpoints available")
            except ValueError:
                print("  Invalid offset")
            continue

        # iN or iter:N = specific iteration
        iter_match = re.match(r"^i(?:ter:?)?(\d+)$", choice)
        if iter_match:
            target_iter = int(iter_match.group(1))
            for iter_num, path in checkpoints:
                if iter_num == target_iter:
                    return path
            print(f"  Iteration {target_iter} not found")
            continue

        # Numeric index
        try:
            idx = int(choice)
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx][1]
            print(f"  Enter 0-{len(checkpoints)-1}")
        except ValueError:
            print("  Invalid input")


def _select_experiment(experiments, prompt_prefix=""):
    """Select an experiment from a dict. Returns (name, checkpoints) or None."""
    exp_names = list(experiments.keys())
    if len(exp_names) == 1:
        name = exp_names[0]
        print(f"{prompt_prefix}Experiment: {name} ({len(experiments[name])} checkpoints)")
        return name, experiments[name]

    print(f"{prompt_prefix}Experiments:")
    for i, name in enumerate(exp_names):
        cpts = experiments[name]
        print(
            f"  {i+1}. {name} ({len(cpts)} ckpts, latest: iter {cpts[0][0]:04d})"
        )

    while True:
        choice = input(f"{prompt_prefix}Select experiment (Enter=first, r=random, p=playout): ").strip().lower()
        if choice == "r":
            return None, "random"
        if choice == "p":
            return None, "playout"
        if choice in ["", "1"] and len(exp_names) >= 1:
            return exp_names[0], experiments[exp_names[0]]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(exp_names):
                return exp_names[idx], experiments[exp_names[idx]]
        except ValueError:
            pass


def select_network_interactive(ctx, game_name, base="data"):
    """Interactive network selection with per-player experiment+checkpoint choice.

    Returns True if any network was loaded.
    """
    if not TORCH_AVAILABLE:
        print("torch not available - using random policy")
        return False

    experiments = discover_checkpoints(game_name, base)

    if not experiments:
        print(f"No checkpoints found in {base}/{game_name}/*/checkpoint/")
        print("Using random policy")
        return False

    print("\n=== Network Selection ===")

    # Same or different networks?
    choice = (
        input(
            "Same network for both players? [y]es / [n]o / [r]andom / [p]layout (default=yes): "
        )
        .strip()
        .lower()
    )

    if choice == "r":
        print("Using random policy")
        return False
    if choice == "p":
        load_network(ctx.game_class, "playout", [0, 1], ctx)
        print("Using playout policy")
        return True

    if choice == "n":
        # Different networks per player - each independently selects experiment AND checkpoint
        any_loaded = False
        for player in [0, 1]:
            print(f"\n--- Player {player} ---")
            result = _select_experiment(experiments, f"  P{player} ")
            if result[1] == "random":
                print(f"  Player {player}: random")
                continue
            if result[1] == "playout":
                load_network(ctx.game_class, "playout", [player], ctx)
                print(f"  Player {player}: playout")
                any_loaded = True
                continue

            exp_name, checkpoints = result
            path = select_checkpoint(checkpoints)
            if path == "playout":
                load_network(ctx.game_class, "playout", [player], ctx)
                print(f"  Player {player}: playout")
                any_loaded = True
            elif path is None:
                print(f"  Player {player}: random")
            elif load_network(ctx.game_class, path, [player], ctx):
                # Load cpuct/fpu_reduction from experiment config
                cpuct, fpu = load_experiment_config(path)
                ctx.cpuct = cpuct
                ctx.fpu_reduction = fpu
                print(f"  Player {player}: {os.path.basename(path)}")
                any_loaded = True
        return any_loaded
    else:
        # Same network for both
        result = _select_experiment(experiments)
        if result[1] == "random":
            print("Using random policy")
            return False
        if result[1] == "playout":
            load_network(ctx.game_class, "playout", [0, 1], ctx)
            print("Using playout policy")
            return True

        exp_name, checkpoints = result
        path = select_checkpoint(checkpoints)
        if path == "playout":
            load_network(ctx.game_class, "playout", [0, 1], ctx)
            print("Using playout policy")
            return True
        if path is None:
            print("Using random policy")
            return False
        if load_network(ctx.game_class, path, [0, 1], ctx):
            # Load cpuct/fpu_reduction from experiment config
            cpuct, fpu = load_experiment_config(path)
            ctx.cpuct = cpuct
            ctx.fpu_reduction = fpu
            print(f"Loaded: {os.path.basename(path)}")
            return True
        return False


# ---------------------------------------------------------------------------
# Display and command handling
# ---------------------------------------------------------------------------


def print_status(ctx):
    """Print current player configuration."""
    for i in range(2):
        print(f"  Player {i}: {ctx.players[i]}")
    reuse_str = "on" if ctx.tree_reuse else "off"
    if any(c is not None for c in ctx.player_caches):
        sizes = [str(c.max_size()) if c is not None else "off" for c in ctx.player_caches]
        cache_str = f"split ({sizes[0]}/{sizes[1]})"
    elif ctx.cache is not None:
        cache_str = str(ctx.cache.max_size())
    else:
        cache_str = "off"
    delay_str = f", Delay: {ctx.auto_delay}s" if ctx.auto_delay > 0 else ""
    print(f"  Tree reuse: {reuse_str}, Cache: {cache_str}{delay_str}")


def handle_config_command(parts, ctx, game_name, base_dir):
    """Handle AI configuration commands. Returns command string."""
    cmd = parts[0]

    if cmd == "net":
        select_network_interactive(ctx, game_name, base_dir)
        print_status(ctx)
        return "config"

    # Get target player
    player = None
    if len(parts) >= 2:
        try:
            p = int(parts[1])
            if p in [0, 1]:
                player = p
        except ValueError:
            pass

    targets = [player] if player is not None else [0, 1]

    if cmd == "nodes":
        if len(parts) >= (3 if player is not None else 2):
            val = parts[-1]
            for p in targets:
                if val in ["off", "0", "none"]:
                    ctx.players[p].node_limit = None
                else:
                    try:
                        ctx.players[p].node_limit = int(val)
                    except ValueError:
                        print(f"Invalid node count: {val}")
                        return "config"
            print_status(ctx)
    elif cmd == "time":
        if len(parts) >= (3 if player is not None else 2):
            val = parts[-1]
            for p in targets:
                if val in ["off", "0", "none"]:
                    ctx.players[p].think_time = None
                else:
                    try:
                        ctx.players[p].think_time = float(val)
                    except ValueError:
                        print(f"Invalid time: {val}")
                        return "config"
            _update_caches(ctx)
            print_status(ctx)
    elif cmd == "temp":
        if len(parts) >= (3 if player is not None else 2):
            val = parts[-1]
            for p in targets:
                try:
                    ctx.players[p].temperature = float(val)
                except ValueError:
                    print(f"Invalid temperature: {val}")
                    return "config"
            print_status(ctx)
    elif cmd == "hints":
        if len(parts) >= (3 if player is not None else 2):
            val = parts[-1]
            for p in targets:
                ctx.players[p].show_hints = val in ["on", "true", "1", "yes"]
            print_status(ctx)
    elif cmd == "greedy":
        if len(parts) >= (3 if player is not None else 2):
            val = parts[-1]
            for p in targets:
                ctx.players[p].greedy = val in ["on", "true", "1", "yes"]
            print_status(ctx)
    elif cmd == "batch":
        if len(parts) >= (3 if player is not None else 2):
            val = parts[-1]
            for p in targets:
                if val in ["auto", "0"]:
                    ctx.players[p].batch_size = 0
                elif val in ["off", "1"]:
                    ctx.players[p].batch_size = 1
                else:
                    try:
                        ctx.players[p].batch_size = int(val)
                    except ValueError:
                        print(f"Invalid batch size: {val}")
                        return "config"
            print_status(ctx)
    elif cmd == "cscale":
        # cscale <0|1> <value>: override gumbel_c_scale for player(s).
        # Lower c_scale = more variance, higher = sharper Q-driven play. Paper
        # default is 1.0; empirical strength peak around 2.0 on this run.
        if len(parts) >= (3 if player is not None else 2):
            val = parts[-1]
            try:
                cs = float(val)
            except ValueError:
                print(f"Invalid c_scale: {val}")
                return "config"
            if cs <= 0:
                print(f"c_scale must be > 0, got {cs}")
                return "config"
            for p in targets:
                ctx.players[p].gumbel_c_scale = cs
                ctx.players[p].mcts = None  # rebuild on next search
            print_status(ctx)
    elif cmd == "resign":
        # resign <0|1> <on|off|threshold>: enable/disable resign; passing a
        # numeric value also sets the threshold (and enables).
        if len(parts) >= (3 if player is not None else 2):
            val = parts[-1]
            enable = None
            thresh_override = None
            if val in ("on", "true", "yes", "1"):
                enable = True
            elif val in ("off", "false", "no", "0"):
                enable = False
            else:
                try:
                    thresh_override = float(val)
                    enable = True
                except ValueError:
                    print(f"Invalid resign value: {val} (use on/off or numeric threshold)")
                    return "config"
                if thresh_override < -1.0 or thresh_override > 1.0:
                    print(f"resign threshold must be in [-1, 1], got {thresh_override}")
                    return "config"
            for p in targets:
                ctx.players[p].resign_enabled = bool(enable)
                if thresh_override is not None:
                    ctx.players[p].resign_threshold = thresh_override
                ctx.players[p]._resign_streak = 0
            print_status(ctx)
    elif cmd == "gumbel-opening":
        if len(parts) >= (3 if player is not None else 2):
            val = parts[-1]
            if val in ("off", "none"):
                n = 0
            else:
                try:
                    n = int(val)
                except ValueError:
                    print(f"Invalid gumbel-opening: {val}")
                    return "config"
                if n < 0:
                    print(f"gumbel-opening must be non-negative, got {n}")
                    return "config"
            for p in targets:
                ctx.players[p].gumbel_opening_moves = n
            print_status(ctx)
    elif cmd == "lcb":
        # lcb <0|1> <on|off|Z>: enable/disable LCB (sets greedy=True with warning
        # when enabling), or pass a numeric Z to also tune the strictness.
        if len(parts) >= (3 if player is not None else 2):
            val = parts[-1]
            enable = None
            z_override = None
            if val in ("on", "true", "yes", "1"):
                enable = True
            elif val in ("off", "false", "no", "0"):
                enable = False
            else:
                try:
                    z_override = float(val)
                    enable = True
                except ValueError:
                    print(f"Invalid lcb value: {val} (use on/off or a numeric Z)")
                    return "config"
            for p in targets:
                ctx.players[p].lcb_enabled = bool(enable)
                if z_override is not None:
                    ctx.players[p].lcb_z = z_override
                if enable and not ctx.players[p].greedy:
                    print(
                        f"[lcb] Enabling LCB also forces greedy=True for player "
                        f"{p} (LCB is a deterministic best-move selector; it "
                        f"has no effect under stochastic sampling)."
                    )
                    ctx.players[p].greedy = True
            print_status(ctx)
    elif cmd == "prune":
        if len(parts) >= (3 if player is not None else 2):
            val = parts[-1]
            if val in ("off", "0", "none"):
                frac = 0.0
            else:
                try:
                    frac = float(val)
                except ValueError:
                    print(f"Invalid prune fraction: {val}")
                    return "config"
                if frac < 0.0 or frac >= 1.0:
                    print(f"prune_frac must be in [0, 1), got {frac}")
                    return "config"
            for p in targets:
                ctx.players[p].prune_frac = frac
            print_status(ctx)
    elif cmd == "epsilon":
        if len(parts) >= (3 if player is not None else 2):
            val = parts[-1]
            try:
                eps = float(val)
            except ValueError:
                print(f"Invalid epsilon: {val}")
                return "config"
            if eps < 0.0 or eps > 1.0:
                print(f"Epsilon must be in [0, 1], got {eps}")
                return "config"
            for p in targets:
                ctx.players[p].epsilon = eps
                ctx.players[p].mcts = None  # rebuild on next search
            print_status(ctx)
    elif cmd == "algo":
        if len(parts) >= (3 if player is not None else 2):
            val = parts[-1]
            aliases = {
                "auto": None, "default": None, "yaml": None,
                "puct": "puct", "p": "puct",
                "gumbel": "gumbel", "g": "gumbel",
            }
            if val not in aliases:
                print(f"Invalid algo: {val} (use auto/puct/gumbel)")
                return "config"
            for p in targets:
                ctx.players[p].algo_override = aliases[val]
                _apply_player_gumbel(ctx.players[p])
            print_status(ctx)
    elif cmd == "delay":
        if len(parts) >= 2:
            try:
                ctx.auto_delay = float(parts[1])
                print_status(ctx)
            except ValueError:
                print(f"Invalid delay: {parts[1]}")

    return "config"


def parse_meta_command(cmd, ctx, game_name="", base_dir="data"):
    """Parse meta-commands (help, undo, quit, status, config, etc.).

    Returns command string or None if not a meta-command.
    """
    lower = cmd.lower().strip()
    if lower in ["q", "quit", "exit"]:
        return "quit"
    if lower in ["h", "help", "?"]:
        return "help"
    if lower in ["u", "undo"]:
        return "undo"
    if lower in ["s", "status"]:
        return "status"
    if lower in ["v", "valid", "moves"]:
        return "valid"
    if lower.startswith("auto"):
        sub = lower.split()
        if len(sub) >= 2 and sub[1] in ("full", "f"):
            return "auto-full"
        if len(sub) >= 2 and sub[1] in ("step", "s"):
            return "auto-step"
        choice = input("  Auto mode: [s]tep (press Enter each move) or [f]ull (no pauses)? ").strip().lower()
        if choice in ("f", "full"):
            return "auto-full"
        return "auto-step"
    if lower == "manual":
        return "manual"
    parts = lower.split()
    if parts and parts[0] in ["net", "nodes", "time", "temp", "hints", "greedy", "batch", "algo", "epsilon", "prune", "lcb", "gumbel-opening", "cscale", "resign", "delay"]:
        return handle_config_command(parts, ctx, game_name, base_dir)
    return None


def print_generic_help():
    """Print generic play help."""
    print("\nCommands:")
    print("  <move>     - Play a move (game-specific syntax or action ID)")
    print("  undo / u   - Undo last move")
    print("  valid / v  - List valid moves")
    print("  help / h   - Show this help")
    print("  status / s - Show player configuration")
    print("  auto       - Enable AI auto-play (step or full)")
    print("  manual     - Disable AI auto-play")
    print("  quit / q   - Quit game")
    print("\nAI configuration:")
    print("  net                      - Re-select network")
    print("  nodes <0|1> <count|off>  - Node limit")
    print("  time <0|1> <secs|off>    - Time limit")
    print("  temp <0|1> <value>       - Temperature")
    print("  hints <0|1> <on|off>     - AI hints for human player")
    print("  greedy <0|1> <on|off>    - Always play best move")
    print("  batch <0|1> <0|1|N>      - Batch size (0=auto, 1=off, N=fixed)")
    print("  algo <0|1> <auto|puct|gumbel> - Search algorithm (auto=use network's config)")
    print("  epsilon <0|1> <value>    - Dirichlet root noise (0=off, 0.25=AZ self-play)")
    print("  prune <0|1> <frac|off>   - Visit-count floor for temperature sampling (e.g. 0.02)")
    print("  lcb <0|1> <on|off|Z>     - LCB best-move selection (forces greedy when on; Z=2.0)")
    print("  gumbel-opening <0|1> <N> - Force G3 sampling for first N plies (Gumbel only)")
    print("  cscale <0|1> <value>     - Gumbel c_scale (1.0 paper default; 2.0 sharper, 0.5 looser)")
    print("  resign <0|1> <on|off|thresh> - Auto-resign in auto-full mode (default thresh=-0.95)")
    print("  delay <seconds>          - Turn delay for auto-full mode")


# ---------------------------------------------------------------------------
# Game resolution
# ---------------------------------------------------------------------------


def resolve_game(arg, base_dir):
    """Resolve game_or_config argument to (game_name, Game class).

    arg can be:
      - None: auto-discover from data/
      - A .yaml/.yml path: load config file
      - A GAME_REGISTRY key: use directly
    """
    # YAML config file
    if arg and (arg.endswith(".yaml") or arg.endswith(".yml")):
        from config import load_config
        config = load_config(arg, {})
        return config.game, config.Game

    # Direct game name
    if arg and arg in GAME_REGISTRY:
        cls_name = GAME_REGISTRY[arg]
        return arg, getattr(alphazero, cls_name)

    # Auto-discover
    games = discover_games(base_dir)
    all_games = list(GAME_REGISTRY.keys())

    if not games:
        # No trained checkpoints - show all available games
        print("No trained checkpoints found. Available games:")
        for i, name in enumerate(all_games):
            print(f"  {i+1}. {name}")
        while True:
            choice = input("Select game: ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(all_games):
                    name = all_games[idx]
                    cls_name = GAME_REGISTRY[name]
                    return name, getattr(alphazero, cls_name)
            except ValueError:
                if choice in GAME_REGISTRY:
                    cls_name = GAME_REGISTRY[choice]
                    return choice, getattr(alphazero, cls_name)
            print("Invalid selection")
    else:
        # Show games with checkpoints first
        game_names = list(games.keys())
        print("Games with trained networks:")
        for i, name in enumerate(game_names):
            exps = games[name]
            total_ckpts = sum(len(c) for c in exps.values())
            print(f"  {i+1}. {name} ({len(exps)} experiment(s), {total_ckpts} checkpoints)")

        # Also mention other available games
        other_games = [g for g in all_games if g not in games]
        if other_games:
            print(f"\nOther games (no checkpoints): {', '.join(other_games)}")

        while True:
            choice = input("Select game (Enter=first): ").strip()
            if choice == "":
                name = game_names[0]
                cls_name = GAME_REGISTRY[name]
                return name, getattr(alphazero, cls_name)
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(game_names):
                    name = game_names[idx]
                    cls_name = GAME_REGISTRY[name]
                    return name, getattr(alphazero, cls_name)
            except ValueError:
                if choice in GAME_REGISTRY:
                    cls_name = GAME_REGISTRY[choice]
                    return choice, getattr(alphazero, cls_name)
            print("Invalid selection")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _prompt_value(prompt, default, parse_fn, allow_none=False):
    """Prompt for a value with a default. Returns parsed value or default."""
    default_str = str(default) if default is not None else "none"
    raw = input(f"  {prompt} [{default_str}]: ").strip()
    if not raw:
        return default
    if allow_none and raw.lower() in ("none", "off", "0"):
        return None
    try:
        return parse_fn(raw)
    except (ValueError, TypeError):
        print(f"  Invalid value, using default: {default_str}")
        return default


def _prompt_bool(prompt, default):
    """Prompt for a boolean with a default."""
    default_str = "on" if default else "off"
    raw = input(f"  {prompt} [{default_str}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("on", "true", "1", "yes", "y")


def prompt_ai_config(ctx, args):
    """Prompt for key AI settings at startup."""
    print("\n=== AI Configuration ===")

    # Node limit
    node_default = args.nodes if args.nodes is not None else DEFAULT_NODE_LIMIT
    node_limit = _prompt_value("Node limit per move", node_default, int, allow_none=True)
    for p in ctx.players:
        if p.is_ai:
            p.node_limit = node_limit

    # Think time
    time_default = args.think_time
    think_time = _prompt_value("Think time in seconds", time_default, float,
                               allow_none=True)
    for p in ctx.players:
        if p.is_ai:
            p.think_time = think_time
            if think_time is not None:
                p.node_limit = None

    # Temperature
    temp = _prompt_value("Temperature", DEFAULT_TEMPERATURE, float)
    for p in ctx.players:
        if p.is_ai:
            p.temperature = temp

    # Greedy
    greedy = _prompt_bool("Greedy (always play best move)", False)
    for p in ctx.players:
        if p.is_ai:
            p.greedy = greedy

    # Batch size (only when a network is loaded)
    has_network = any(p.network is not None for p in ctx.players if p.is_ai)
    if has_network:
        batch_val = _prompt_value("Batch size (0=auto, 1=off)", 0, int)
        for p in ctx.players:
            if p.is_ai:
                p.batch_size = batch_val

    # Dirichlet root noise (epsilon). Default 0 (matches tournament). CLI
    # --epsilon skips the prompt; otherwise prompt once.
    if args.epsilon is not None:
        eps = args.epsilon
    else:
        eps = _prompt_value(
            "Dirichlet root noise epsilon (0=off, 0.25=AZ self-play)", 0.0, float)
    for p in ctx.players:
        if p.is_ai:
            p.epsilon = eps

    # LCB best-move selection (CLI --lcb). Auto-forces greedy=True for AI
    # players because LCB has no effect under stochastic sampling.
    if args.lcb:
        for p in ctx.players:
            if p.is_ai:
                p.lcb_enabled = True
                if args.lcb_z is not None:
                    p.lcb_z = args.lcb_z
                if not p.greedy:
                    p.greedy = True
        print("[lcb] LCB enabled for AI players (greedy forced to True).")

    # Gumbel c_scale override (CLI --gumbel-c-scale).
    if args.gumbel_c_scale is not None:
        for p in ctx.players:
            if p.is_ai:
                p.gumbel_c_scale = args.gumbel_c_scale
                p.mcts = None
        print(f"[gumbel] c_scale overridden to {args.gumbel_c_scale} for AI players.")

    # Resign threshold (CLI --resign). Only meaningful in auto-full mode but
    # we set it for all AI players unconditionally.
    if args.resign is not None:
        for p in ctx.players:
            if p.is_ai:
                p.resign_enabled = True
                p.resign_threshold = args.resign
                if args.resign_moves is not None:
                    p.resign_consecutive = args.resign_moves
        print(f"[resign] Auto-resign enabled for AI players at threshold {args.resign}.")

    # Tree reuse
    ctx.tree_reuse = _prompt_bool("Tree reuse", ctx.tree_reuse)

    # Cache size
    cache_default = args.cache_size
    cache_size = _prompt_value("Cache size", cache_default, int)
    ctx.cache_size = cache_size if cache_size and cache_size > 0 else 0
    _update_caches(ctx)


def main():
    parser = argparse.ArgumentParser(description="Play against AI")
    parser.add_argument(
        "game_or_config", nargs="?", default=None,
        help="Game name or YAML config path (optional, auto-discovers if omitted)"
    )
    parser.add_argument(
        "--think-time", type=float, default=None, help="AI think time in seconds"
    )
    parser.add_argument("--nodes", type=int, default=None, help="AI node limit")
    parser.add_argument(
        "--base-dir", default="data", help="Base data directory (default: data)"
    )
    parser.add_argument(
        "--cache_size", type=int, default=10000, help="S3-FIFO cache size (default: 10000)"
    )
    parser.add_argument(
        "--epsilon", type=float, default=None,
        help="Dirichlet root noise weight (default: 0 = off; AlphaZero self-play uses 0.25)"
    )
    parser.add_argument(
        "--lcb", action="store_true",
        help="Enable LCB best-move selection for AI players (forces greedy=True)"
    )
    parser.add_argument(
        "--lcb-z", type=float, default=None,
        help="LCB strictness Z-score (default: 2.0; larger = more conservative)"
    )
    parser.add_argument(
        "--resign", type=float, nargs="?", const=-0.95, default=None,
        metavar="THRESHOLD",
        help="Enable auto-resign for AI players at the given W-L threshold "
             "(default if flag bare: -0.95). Only fires in auto-full mode."
    )
    parser.add_argument(
        "--resign-moves", type=int, default=None,
        help="Consecutive bad-WLD moves before resigning (default: 3)"
    )
    parser.add_argument(
        "--gumbel-c-scale", type=float, default=None,
        help="Override Gumbel c_scale for AI players (default: yaml value, "
             "library default 1.0). Lower = more variance; higher = sharper. "
             "Empirical sweet spot for our nets: 2.0."
    )
    args = parser.parse_args()
    if args.epsilon is not None and not (0.0 <= args.epsilon <= 1.0):
        parser.error(f"--epsilon must be in [0, 1], got {args.epsilon}")
    if args.gumbel_c_scale is not None and args.gumbel_c_scale <= 0:
        parser.error(f"--gumbel-c-scale must be > 0, got {args.gumbel_c_scale}")

    game_name, Game = resolve_game(args.game_or_config, args.base_dir)
    ui = get_game_ui(game_name)
    # checkpoint_game_name: used for data/checkpoint discovery; stays as the base
    # game even after variant selection (pinned variants share the same data dir).
    checkpoint_game_name = game_name

    # Offer variant selection if the UI supports it (e.g. unified games).
    variant = ui.select_variant()
    if variant and variant != game_name and variant in GAME_REGISTRY:
        game_name = variant
        Game = getattr(alphazero, GAME_REGISTRY[variant])
        ui = get_game_ui(game_name)

    print(f"\n=== {game_name} Interactive Player ===\n")

    gs = Game()
    ctx = PlayContext(gs, Game, cache_size=args.cache_size)

    # Mode selection
    print("Select mode:")
    print("  1. Human vs Human")
    print("  2. Human (P0) vs AI (P1)")
    print("  3. AI (P0) vs Human (P1)")
    print("  4. AI vs AI (watch)")

    mode = input("Mode (1/2/3/4) [2]: ").strip()
    if mode == "1":
        ctx.players[0].is_ai = False
        ctx.players[1].is_ai = False
    elif mode == "3":
        ctx.players[0].is_ai = True
        ctx.players[1].is_ai = False
    elif mode == "4":
        ctx.players[0].is_ai = True
        ctx.players[1].is_ai = True
        ctx.auto_play = True
        full = input("Fully automatic (no pauses)? [y]/n: ").strip().lower()
        ctx.auto_full = full not in ("n", "no")
        if ctx.auto_full:
            ctx.auto_delay = _prompt_value(
                "Turn delay in seconds (0=no pause)", 0.0, float)
    else:  # Default: mode 2
        ctx.players[0].is_ai = False
        ctx.players[1].is_ai = True

    # Network selection and configuration
    if ctx.players[0].is_ai or ctx.players[1].is_ai:
        select_network_interactive(ctx, checkpoint_game_name, args.base_dir)
        prompt_ai_config(ctx, args)

        print()
        print_status(ctx)
        print("\nFine-tune settings or Enter to start (type 'help' for commands):")

        while True:
            cmd = input("\nConfig> ").strip()
            if cmd.lower() in ["", "start", "go"]:
                break
            result = parse_meta_command(cmd, ctx, checkpoint_game_name, args.base_dir)
            if result == "quit":
                return
            if result == "help":
                print_generic_help()
            elif result == "status":
                print_status(ctx)
            elif result is None:
                print(f"Unknown command: {cmd}. Type 'help' for available commands.")

    # Pre-game batch size calibration for timed auto players
    for pi, pcfg in enumerate(ctx.players):
        if not pcfg.is_ai or pcfg.batch_size != 0:
            continue
        if pcfg.think_time is not None and pcfg._calibrated_timed_bs is None:
            print(f"  P{pi} calibrating timed batch size...")
            pcfg._calibrated_timed_bs = calibrate_timed_batch(
                ctx.game, pcfg.network, pcfg.think_time,
                pcfg.eval_type, _get_cache(ctx, pi),
            )
            print(f"  P{pi} batch size: {pcfg._calibrated_timed_bs} (calibrated for {pcfg.think_time}s)")
        elif pcfg.node_limit is not None:
            bs = _compute_batch_size(pcfg.node_limit)
            print(f"  P{pi} batch size: {bs} (sqrt of {pcfg.node_limit} nodes)")

    print("\nStarting game...")

    # Game loop
    history = []

    while True:
        print("\n" + "=" * 50)
        print(ui.display_board(ctx.game))

        scores = ctx.game.scores()
        if scores is not None:
            scores_arr = np.array(scores)
            winners = np.where(scores_arr == 1)[0]
            if len(winners) > 0:
                print(f"\nPlayer {winners[0]} wins!")
            else:
                print("\nDraw!")
            if any(c is not None for c in ctx.player_caches):
                for i, c in enumerate(ctx.player_caches):
                    print_cache_stats(c, label=f"P{i} cache")
            elif ctx.cache is not None:
                print_cache_stats(ctx.cache)
            break

        valids = np.array(ctx.game.valid_moves())
        current = ctx.game.current_player()
        pcfg = ctx.players[current]

        if pcfg.is_ai:
            if ctx.auto_play:
                if ctx.auto_full:
                    # Fully automatic: play with optional delay
                    start = time.time()
                    action, probs, source, sims, wld = get_ai_move(
                        ctx, current, valids, greedy=pcfg.greedy,
                        move_number=len(history))
                    if action is None:
                        print("No valid moves!")
                        break

                    print(f"\nAI (P{current}) [{source}]")
                    _print_pv(ctx.game, pcfg, ui)
                    ui.display_actions_menu(ctx.game, probs, valids, wld=wld)

                    if _check_resign(pcfg, wld):
                        opponent = 1 - current
                        print(f"\n>>> Player {current} resigns (V={float(wld[0])-float(wld[1]):.3f} ≤ {pcfg.resign_threshold} for {pcfg.resign_consecutive} moves)")
                        print(f"\nPlayer {opponent} wins by resignation!")
                        break

                    elapsed = time.time() - start
                    if ctx.auto_delay > 0 and elapsed < ctx.auto_delay:
                        time.sleep(ctx.auto_delay - elapsed)

                    print(
                        f"\n>>> Plays: {ui.format_move(ctx.game, action)}  [{probs[action]*100:.1f}%]"
                    )
                    advance_mcts_trees(ctx, action)
                    history.append(ctx.game.copy())
                    ctx.game.play_move(action)
                else:
                    # Step mode: run MCTS, show probs, interactive prompt
                    probs, source, sims, wld = get_ai_probs(
                        ctx, current, valids, move_number=len(history))
                    print(f"\nAI (P{current}) [{source}]")
                    _print_pv(ctx.game, pcfg, ui)

                    # Pre-select AI's chosen move (respects greedy/temperature
                    # and Gumbel G2 opening-moves override)
                    if _effective_greedy(pcfg, len(history)):
                        ai_best = _greedy_ai_action(pcfg, probs, valids, sims)
                    else:
                        ai_best = int(np.random.choice(len(probs), p=probs))
                    entries = ui.build_action_menu(ctx.game, probs, valids, wld=wld)
                    use_text = False
                    try:
                        selector = ActionSelector(entries, preselect=ai_best)
                        result = selector.run()
                    except (ValueError, OSError):
                        use_text = True
                        result = None
                        ui.display_actions_menu(ctx.game, probs, valids, wld=wld)

                    if result is None and not use_text:
                        use_text = True
                        ui.display_actions_menu(ctx.game, probs, valids, wld=wld)

                    if not use_text:
                        if isinstance(result, int):
                            print(
                                f"\n>>> Plays: {ui.format_move(ctx.game, result)}  [{probs[result]*100:.1f}%]"
                            )
                            advance_mcts_trees(ctx, result)
                            history.append(ctx.game.copy())
                            ctx.game.play_move(result)
                            continue
                        if result == 'quit':
                            return
                        if result == 'undo':
                            if history:
                                ctx.game = history.pop()
                                for p in ctx.players:
                                    p.mcts = None
                                print("Move undone")
                            else:
                                print("No moves to undo")
                            continue
                        if result == 'help':
                            print_generic_help()
                            ui.show_help(ctx.game)
                            continue
                        if result == 'status':
                            print_status(ctx)
                            continue
                        if result == 'valid':
                            descs = ui.get_valid_move_descriptions(ctx.game, valids)
                            for aid, desc in descs:
                                prob_str = f"  [{probs[aid]*100:5.1f}%]"
                                print(f"  {aid:4d}: {desc}{prob_str}")
                            continue

                    # Text input fallback
                    while True:
                        cmd = input(f"\nAuto-step P{current} (Enter=play AI move): ").strip()
                        if not cmd:
                            # Sample and play the AI move
                            if pcfg.greedy:
                                action = _greedy_ai_action(pcfg, probs, valids, sims)
                            else:
                                action = np.random.choice(len(probs), p=probs)
                            print(
                                f"\n>>> Plays: {ui.format_move(ctx.game, action)}  [{probs[action]*100:.1f}%]"
                            )
                            advance_mcts_trees(ctx, action)
                            history.append(ctx.game.copy())
                            ctx.game.play_move(action)
                            break

                        meta = parse_meta_command(cmd, ctx, checkpoint_game_name, args.base_dir)
                        if meta == "quit":
                            return
                        if meta == "help":
                            print_generic_help()
                            ui.show_help(ctx.game)
                            continue
                        if meta == "undo":
                            if history:
                                ctx.game = history.pop()
                                for p in ctx.players:
                                    p.mcts = None
                                print("Move undone")
                                break
                            print("No moves to undo")
                            continue
                        if meta == "status":
                            print_status(ctx)
                            continue
                        if meta == "valid":
                            descs = ui.get_valid_move_descriptions(ctx.game, valids)
                            for aid, desc in descs:
                                prob_str = f"  [{probs[aid]*100:5.1f}%]"
                                print(f"  {aid:4d}: {desc}{prob_str}")
                            continue
                        if meta == "auto-full":
                            ctx.auto_full = True
                            print("Switched to full auto")
                            # Play this move and continue
                            if pcfg.greedy:
                                action = _greedy_ai_action(pcfg, probs, valids, sims)
                            else:
                                action = np.random.choice(len(probs), p=probs)
                            print(
                                f"\n>>> Plays: {ui.format_move(ctx.game, action)}  [{probs[action]*100:.1f}%]"
                            )
                            advance_mcts_trees(ctx, action)
                            history.append(ctx.game.copy())
                            ctx.game.play_move(action)
                            break
                        if meta == "auto-step":
                            continue  # Already in step mode
                        if meta == "manual":
                            ctx.auto_play = False
                            print("Auto-play disabled")
                            break
                        if meta == "config":
                            continue

                        # Try game-specific move parse
                        action = ui.parse_move(ctx.game, cmd, valids)
                        if action is not None:
                            print(f"Playing: {ui.format_move(ctx.game, action)}")
                            advance_mcts_trees(ctx, action)
                            history.append(ctx.game.copy())
                            ctx.game.play_move(action)
                            break
                        print(
                            "Invalid command. Enter=play AI move, 'help' for commands."
                        )
            else:
                # Manual mode: show AI suggestions, human picks
                probs, source, sims, wld = get_ai_probs(
                    ctx, current, valids, move_number=len(history))
                print(f"\nAI (P{current}) suggests [{source}]:")
                _print_pv(ctx.game, pcfg, ui)

                # Pre-select AI's chosen move (respects greedy/temperature
                # and Gumbel G2 opening-moves override)
                if _effective_greedy(pcfg, len(history)):
                    ai_best = _greedy_ai_action(pcfg, probs, valids, sims)
                else:
                    ai_best = int(np.random.choice(len(probs), p=probs))
                entries = ui.build_action_menu(ctx.game, probs, valids, wld=wld)
                use_text = False
                try:
                    selector = ActionSelector(entries, preselect=ai_best)
                    result = selector.run()
                except (ValueError, OSError):
                    use_text = True
                    result = None
                    ui.display_actions_menu(ctx.game, probs, valids, wld=wld)

                if result is None and not use_text:
                    use_text = True
                    ui.display_actions_menu(ctx.game, probs, valids, wld=wld)

                if not use_text:
                    if isinstance(result, int):
                        print(f"Playing: {ui.format_move(ctx.game, result)}")
                        advance_mcts_trees(ctx, result)
                        history.append(ctx.game.copy())
                        ctx.game.play_move(result)
                        continue
                    if result == 'quit':
                        return
                    if result == 'undo':
                        if history:
                            ctx.game = history.pop()
                            for p in ctx.players:
                                p.mcts = None
                            print("Move undone")
                        else:
                            print("No moves to undo")
                        continue
                    if result == 'help':
                        print_generic_help()
                        ui.show_help(ctx.game)
                        continue
                    if result == 'status':
                        print_status(ctx)
                        continue
                    if result in ("auto-step", "auto-full"):
                        ctx.auto_play = True
                        ctx.auto_full = result == "auto-full"
                        print(f"Auto-play enabled")
                        continue

                # Text input fallback
                while True:
                    cmd = input(f"\nPlayer {current} (AI-assisted) move: ").strip()
                    if not cmd:
                        # Play the AI-suggested move
                        if pcfg.greedy:
                            action = _greedy_ai_action(pcfg, probs, valids, sims)
                        else:
                            action = np.random.choice(len(probs), p=probs)
                        print(
                            f"\n>>> Plays: {ui.format_move(ctx.game, action)}  [{probs[action]*100:.1f}%]"
                        )
                        advance_mcts_trees(ctx, action)
                        history.append(ctx.game.copy())
                        ctx.game.play_move(action)
                        break

                    meta = parse_meta_command(cmd, ctx, checkpoint_game_name, args.base_dir)
                    if meta == "quit":
                        return
                    if meta == "help":
                        print_generic_help()
                        ui.show_help(ctx.game)
                        continue
                    if meta == "undo":
                        if history:
                            ctx.game = history.pop()
                            for p in ctx.players:
                                p.mcts = None
                            print("Move undone")
                            break
                        print("No moves to undo")
                        continue
                    if meta == "status":
                        print_status(ctx)
                        continue
                    if meta == "valid":
                        descs = ui.get_valid_move_descriptions(ctx.game, valids)
                        for aid, desc in descs:
                            prob_str = f"  [{probs[aid]*100:5.1f}%]"
                            print(f"  {aid:4d}: {desc}{prob_str}")
                        continue
                    if meta in ("auto-step", "auto-full"):
                        ctx.auto_play = True
                        ctx.auto_full = meta == "auto-full"
                        print(f"Auto-play enabled ({meta.split('-')[1]})")
                        break
                    if meta == "config":
                        continue

                    # Try game-specific parse
                    action = ui.parse_move(ctx.game, cmd, valids)
                    if action is not None:
                        print(f"Playing: {ui.format_move(ctx.game, action)}")
                        advance_mcts_trees(ctx, action)
                        history.append(ctx.game.copy())
                        ctx.game.play_move(action)
                        break
                    print(
                        "Invalid move. Enter=play AI move, 'help' for commands."
                    )
        else:
            # Human turn
            probs = None
            wld = None
            if pcfg.show_hints and (
                pcfg.network is not None or pcfg.eval_type == "playout"
            ):
                probs, source, _, wld = get_ai_probs(
                    ctx, current, valids, move_number=len(history))
                print(f"\nHints [{source}]:")
                _print_pv(ctx.game, pcfg, ui)

            # Try arrow-key selector first
            entries = ui.build_action_menu(ctx.game, probs, valids, wld=wld)
            use_text = False
            try:
                selector = ActionSelector(entries)
                result = selector.run()
            except (ValueError, OSError):
                use_text = True
                result = None
                ui.display_actions_menu(ctx.game, probs, valids, wld=wld)

            if result is None and not use_text:
                # Tab pressed - switch to text input
                use_text = True
                ui.display_actions_menu(ctx.game, probs, valids, wld=wld)

            if not use_text:
                if isinstance(result, int):
                    print(f"Playing: {ui.format_move(ctx.game, result)}")
                    advance_mcts_trees(ctx, result)
                    history.append(ctx.game.copy())
                    ctx.game.play_move(result)
                    continue
                if result == 'quit':
                    return
                if result == 'undo':
                    if history:
                        ctx.game = history.pop()
                        for p in ctx.players:
                            p.mcts = None
                        print("Move undone")
                    else:
                        print("No moves to undo")
                    continue
                if result == 'help':
                    print_generic_help()
                    ui.show_help(ctx.game)
                    continue
                if result == 'status':
                    print_status(ctx)
                    continue
                if result == 'valid':
                    descs = ui.get_valid_move_descriptions(ctx.game, valids)
                    for aid, desc in descs:
                        prob_str = (
                            f"  [{probs[aid]*100:5.1f}%]"
                            if probs is not None
                            else ""
                        )
                        print(f"  {aid:4d}: {desc}{prob_str}")
                    continue

            # Text input fallback
            while True:
                cmd = input(f"\nPlayer {current} move: ").strip()
                if not cmd:
                    continue

                meta = parse_meta_command(cmd, ctx, checkpoint_game_name, args.base_dir)
                if meta == "quit":
                    return
                if meta == "help":
                    print_generic_help()
                    ui.show_help(ctx.game)
                    continue
                if meta == "undo":
                    if history:
                        ctx.game = history.pop()
                        for p in ctx.players:
                            p.mcts = None
                        print("Move undone")
                        break
                    print("No moves to undo")
                    continue
                if meta == "status":
                    print_status(ctx)
                    continue
                if meta == "valid":
                    descs = ui.get_valid_move_descriptions(ctx.game, valids)
                    for aid, desc in descs:
                        prob_str = (
                            f"  [{probs[aid]*100:5.1f}%]"
                            if probs is not None
                            else ""
                        )
                        print(f"  {aid:4d}: {desc}{prob_str}")
                    continue
                if meta in ("auto-step", "auto-full"):
                    ctx.auto_play = True
                    ctx.auto_full = meta == "auto-full"
                    print(f"Auto-play enabled ({meta.split('-')[1]})")
                    break
                if meta == "config":
                    continue

                # Try game-specific parse
                action = ui.parse_move(ctx.game, cmd, valids)
                if action is not None:
                    print(f"Playing: {ui.format_move(ctx.game, action)}")
                    advance_mcts_trees(ctx, action)
                    history.append(ctx.game.copy())
                    ctx.game.play_move(action)
                    break
                print(
                    "Invalid move. Type 'help' for commands, 'valid' for valid moves."
                )


if __name__ == "__main__":
    main()
