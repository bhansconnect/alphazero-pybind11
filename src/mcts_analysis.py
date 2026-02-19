#!/usr/bin/env python3
"""MCTS Threshold Analysis -- game-generic.

Determines the optimal MCTS visit count for self-play and evaluation by:
1. Running an Elo tournament between different visit count configurations
2. Measuring policy convergence (JSD, TV, Hellinger, top-k agreement vs anchor policy)
3. Measuring value estimate convergence (correlation + MAE vs anchor, and vs game outcomes)
"""

import os
import sys

# macOS nano zone allocator corrupts heap when C++ threads run inside Python.
# Must be set before process starts (before first malloc), so re-exec if needed.
if __name__ == "__main__" and sys.platform == "darwin" and os.environ.get("MallocNanoZone") != "0":
    os.environ["MallocNanoZone"] = "0"
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)
import glob
import math
import gc
import re
import numpy as np
import torch
import tqdm

import alphazero
import neural_net
from config import TrainConfig, GAME_REGISTRY, load_config
from game_runner import (
    GameRunner, GRArgs, RandPlayer, PlayoutPlayer, base_params, elo_prob,
    set_eval_types,
)
from cache_utils import create_cache, create_sharded_cache, print_cache_stats, print_sharded_cache_stats
from tournament import pit_agents, calc_elo

np.set_printoptions(precision=3, suppress=True)

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")

# --- Constants ---
DEFAULT_VISIT_COUNTS = [1, 25, 50, 100, 200, 400, 800]
TOURNAMENT_BATCH_SIZE = 64
TOURNAMENT_CONCURRENT_BATCHES = 2
ANALYSIS_GAMES = 64


# --- Entry helpers ---

def entry_label(entry):
    """Label for an (vc, mode) entry: '200' or '200sp'."""
    vc, mode = entry
    return f"{vc}sp" if mode == "selfplay" else str(vc)


def entry_sort_key(entry):
    """Sort key: by VC, then base before selfplay."""
    vc, mode = entry
    return (vc, 0 if mode == "base" else 1)


def calc_temp(config, turn):
    """Compute temperature with exponential decay (matches game_runner)."""
    ln2 = 0.693
    ld = ln2 / config.temp_decay_half_life
    temp = config.eval_temp - config.final_temp
    temp *= math.exp(-ld * turn)
    temp += config.final_temp
    return temp


def calc_temp_selfplay(config, turn):
    """Compute temperature using self_play_temp (for selfplay mode analysis)."""
    ln2 = 0.693
    ld = ln2 / config.temp_decay_half_life
    temp = config.self_play_temp - config.final_temp
    temp *= math.exp(-ld * turn)
    temp += config.final_temp
    return temp


def jensen_shannon_divergence(p, q):
    """JSD(p, q). Bounded [0, ln(2)]."""
    m = 0.5 * (p + q)
    jsd = 0.0
    mask_p = p > 0
    jsd += 0.5 * np.sum(p[mask_p] * np.log(p[mask_p] / m[mask_p]))
    mask_q = q > 0
    jsd += 0.5 * np.sum(q[mask_q] * np.log(q[mask_q] / m[mask_q]))
    return float(jsd)


def total_variation(p, q):
    """TV(p, q). Bounded [0, 1]."""
    return 0.5 * float(np.sum(np.abs(p - q)))


def hellinger_distance(p, q):
    """Hellinger distance. Bounded [0, 1]."""
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2)))


def top_k_agreement(p, q, k):
    """Fraction of top-k moves by p that appear in top-k of q."""
    top_p = set(np.argsort(p)[-k:])
    top_q = set(np.argsort(q)[-k:])
    return len(top_p & top_q) / k


# --- Game discovery ---

def get_available_games():
    """Return list of all available game names from the registry."""
    return sorted(GAME_REGISTRY.keys())


def discover_experiment_checkpoints(game_name, base="data"):
    """Discover checkpoints from experiment directories for a game.

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


def select_checkpoint(checkpoints):
    """Select a checkpoint interactively. Returns path, None (random), or 'playout'."""
    latest_iter, latest_path = checkpoints[0]

    print(f"\nCheckpoints (newest first):")
    print(f"  l. Latest -> iter {latest_iter:04d}")
    print(f"  r. Random policy")
    print(f"  p. Playout policy")
    print("  " + "-" * 40)

    show = min(10, len(checkpoints))
    for i in range(show):
        iter_num, _ = checkpoints[i]
        print(f"  {i}. iter {iter_num:04d}")
    if len(checkpoints) > show:
        print(f"  ... ({len(checkpoints) - show} more)")

    while True:
        choice = input("\nSelect checkpoint (Enter=latest): ").strip().lower()
        if choice in ["", "l"]:
            return latest_path
        if choice == "r":
            return None
        if choice == "p":
            return "playout"
        try:
            idx = int(choice)
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx][1]
            print(f"Enter 0-{len(checkpoints)-1}")
        except ValueError:
            print("Invalid input")


# --- Phase 1: Interactive Configuration ---

def interactive_config():
    """Interactive configuration. Returns (config, Game, network_path, entries, anchor, phases, use_playout, cache_size, tree_reuse)."""
    print("=== MCTS Threshold Analysis ===\n")

    # Game selection from registry
    games = get_available_games()
    print("Select game:")
    for i, name in enumerate(games):
        print(f"  {i + 1}. {name}")
    choice = input(f"Game [1]: ").strip() or "1"
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(games):
            game_name = games[idx]
        else:
            print(f"Invalid choice, using {games[0]}")
            game_name = games[0]
    except ValueError:
        # Try matching by name
        if choice in games:
            game_name = choice
        else:
            print(f"Invalid choice, using {games[0]}")
            game_name = games[0]
    print(f"  -> {game_name}\n")

    # Load config from YAML if available, else defaults
    yaml_path = os.path.join(CONFIGS_DIR, f"{game_name}.yaml")
    if os.path.exists(yaml_path):
        config = load_config(yaml_path, {})
    else:
        config = TrainConfig(game=game_name)
    Game = config.Game

    # Network selection from experiment directories
    experiments = discover_experiment_checkpoints(game_name)
    if not experiments:
        print(f"No checkpoints found in data/{game_name}/*/checkpoint/")
        print("Will use random or playout policy.\n")
        network_path = None
        use_playout = False
        playout_choice = input("Use playout (rollout) evaluation instead of uniform random? (y/n) [n]: ").strip().lower()
        if playout_choice == 'y':
            use_playout = True
            print("  -> Playout policy (random rollout)\n")
        else:
            print("  -> Random policy (uniform)\n")
    else:
        # Experiment selection
        exp_names = list(experiments.keys())
        if len(exp_names) == 1:
            selected_exp = exp_names[0]
            print(f"Experiment: {selected_exp}")
        else:
            print("Available experiments:")
            for i, name in enumerate(exp_names):
                cpts = experiments[name]
                print(f"  {i + 1}. {name} ({len(cpts)} checkpoints, latest: iter {cpts[0][0]:04d})")
            while True:
                rc = input("\nSelect experiment (number) [1]: ").strip()
                if rc == "":
                    selected_exp = exp_names[0]
                    break
                try:
                    idx = int(rc) - 1
                    if 0 <= idx < len(exp_names):
                        selected_exp = exp_names[idx]
                        break
                    print(f"Enter 1-{len(exp_names)}")
                except ValueError:
                    print("Invalid input")

        checkpoints = experiments[selected_exp]
        network_path = select_checkpoint(checkpoints)
        use_playout = False
        if network_path == "playout":
            network_path = None
            use_playout = True
            print("  -> Playout policy (random rollout)\n")
        elif network_path is None:
            playout_choice = input("Use playout (rollout) evaluation instead of uniform random? (y/n) [n]: ").strip().lower()
            if playout_choice == 'y':
                use_playout = True
                print("  -> Playout policy (random rollout)\n")
            else:
                print("  -> Random policy (uniform)\n")
        else:
            print(f"  -> {os.path.basename(network_path)}\n")

    # Visit counts as (vc, mode) entries (append 'sp' for selfplay mode, e.g. 200sp)
    print(f"Default visit counts: {DEFAULT_VISIT_COUNTS}")
    print("  (1 = raw network policy, no MCTS search)")
    print("  (append 'sp' for selfplay mode, e.g. 200sp)")
    print("  (same VC can appear in both modes, e.g. 200, 200sp)")
    vc_input = input("Enter comma-separated visit counts or press Enter for defaults: ").strip()
    if vc_input:
        entries = []
        for token in vc_input.split(","):
            token = token.strip().lower()
            if token.endswith("sp"):
                vc = int(token[:-2])
                mode = "selfplay"
            else:
                vc = int(token)
                mode = "base"
            entry = (vc, mode)
            if entry not in entries:
                entries.append(entry)
        anchor = entries[-1]  # last entry typed
        entries.sort(key=entry_sort_key)
    else:
        entries = [(vc, "base") for vc in DEFAULT_VISIT_COUNTS]
        anchor = entries[-1]
    # Display with mode indicators
    print(f"  -> [{', '.join(entry_label(e) for e in entries)}]")
    print(f"  -> anchor: {entry_label(anchor)}\n")

    # Phase selection
    print("Select phases to run:")
    print("  1. Elo Tournament only")
    print("  2. Policy & Value Analysis only")
    print("  3. Both (default)")
    phase_input = input("Phase [3]: ").strip() or "3"
    if phase_input == "1":
        phases = {"tournament"}
    elif phase_input == "2":
        phases = {"analysis"}
    else:
        phases = {"tournament", "analysis"}
    print(f"  -> {phases}\n")

    # Cache size
    cache_input = input("Cache size (0 to disable) [200000]: ").strip()
    if cache_input:
        cache_size = int(cache_input)
    else:
        cache_size = 200000
    print(f"  -> cache_size={cache_size}\n")

    # Tree reuse
    tree_reuse_input = input("Tree reuse (persistent MCTS across positions)? [Y/n]: ").strip().lower()
    tree_reuse = tree_reuse_input != 'n'
    print(f"  -> tree_reuse={tree_reuse}\n")

    return config, Game, network_path, entries, anchor, phases, use_playout, cache_size, tree_reuse


# --- Phase 2: Elo Tournament (Round-Robin) ---

def run_tournament(config, Game, network_path, entries, use_playout=False, cache_size=0):
    """Run round-robin Elo tournament between visit count configs.

    Returns (elo_ratings, win_matrix) indexed by unique numeric visit counts.
    """
    print("=" * 60)
    print("Phase: Elo Tournament (Round-Robin)")
    print("=" * 60)

    # Tournament uses unique numeric VCs (mode doesn't affect tournament play)
    visit_counts = sorted(set(vc for vc, _ in entries))
    count = len(visit_counts)

    # Load network (shared across players)
    num_players = Game.NUM_PLAYERS()
    if network_path is None:
        if use_playout:
            agent = PlayoutPlayer()
        else:
            agent = RandPlayer()
        def make_players():
            return [agent] * num_players
    else:
        net_dir = os.path.dirname(network_path)
        net_file = os.path.basename(network_path)
        agent = neural_net.NNWrapper.load_checkpoint(Game, net_dir, net_file)
        agent.enable_inference_optimizations()
        def make_players():
            return [agent] * num_players

    win_matrix = np.full((count, count), np.nan)

    # Create a single shared cache for all matchups (same network throughout)
    shared_cache = create_sharded_cache(Game, cache_size) if cache_size > 0 else None

    total_matchups = count * (count - 1) // 2
    with tqdm.tqdm(total=total_matchups, desc="Tournament") as pbar:
        for i in range(count):
            for j in range(i + 1, count):
                d1 = visit_counts[i]
                d2 = visit_counts[j]

                players = make_players()
                depths = [d2] * num_players
                depths[0] = d1

                # All model groups share the same cache (same network)
                if shared_cache is not None:
                    caches = [shared_cache] * num_players
                else:
                    caches = None

                name = f"v{d1}-v{d2}"
                win_rates = pit_agents(
                    config, Game, players, depths,
                    TOURNAMENT_BATCH_SIZE, name,
                    cache_size=0,
                    caches=caches,
                )
                win_matrix[i, j] = win_rates[0]
                win_matrix[j, i] = win_rates[1]
                pbar.update()
                wr_str = "-".join(f"{r*100:.1f}%" for r in win_rates)
                pbar.set_postfix_str(f"{name}: {wr_str}")

    # Print shared cache stats
    print_sharded_cache_stats(shared_cache)

    # Compute Elo
    elo = np.zeros(count)
    elo = calc_elo(elo, win_matrix)

    print("\n--- Tournament Results ---")
    print(f"{'Visits':>8s} {'Elo':>8s}")
    for i, vc in enumerate(visit_counts):
        print(f"{vc:>8d} {elo[i]:>8.0f}")

    gc.collect()
    return elo, win_matrix


# --- Phase 3: Policy & Value Analysis ---

def _get_policy(mcts_obj, is_selfplay, config):
    """Get policy from MCTS tree, using pruned policy for selfplay if configured."""
    if is_selfplay and config.policy_target_pruning:
        return np.array(mcts_obj.probs_pruned(1.0))
    return np.array(mcts_obj.probs(1.0))


def _evaluate_leaves(agent, items, game_states, num_players, num_moves, cache):
    """Evaluate leaf nodes for MCTS. Handles random, playout, and NN agents.

    items: list of (gid, key, mcts_obj, leaf, hash, is_sp, slot, gs) tuples.
    Returns list of (v, pi) numpy arrays aligned with items.
    """
    if agent == "playout":
        leaves = [item[3] for item in items]
        v_np, pi_np = alphazero.playout_eval_batch(leaves)
        return list(zip(v_np, pi_np))
    elif agent == "random":
        np_ = num_players
        results = []
        for _ in items:
            v = np.full(np_ + 1, 1.0 / (np_ + 1), dtype=np.float32)
            pi = np.full(num_moves, 1.0 / num_moves, dtype=np.float32)
            results.append((v, pi))
        return results
    else:
        # NN agent — batch GPU inference
        canonicals = [np.array(item[3].canonicalized()) for item in items]
        batch = torch.from_numpy(np.stack(canonicals))
        v_batch, pi_batch = agent.process(batch)
        v_np = v_batch.cpu().numpy()
        pi_np = pi_batch.cpu().numpy()
        # Insert into cache
        if cache is not None:
            for j, item in enumerate(items):
                cache.insert(item[4], pi_np[j], v_np[j])
        return list(zip(v_np, pi_np))


def run_analysis(config, Game, network_path, entries, anchor, use_playout=False,
                 cache_size=0, tree_reuse=False):
    """Run policy & value convergence analysis with batched inference.

    Plays ANALYSIS_GAMES concurrently, batching all NN inference calls across
    games for maximum cache sharing.

    Per-entry modes: each (vc, mode) entry can be "base" (no noise, eval temp)
    or "selfplay" (Dirichlet noise, root policy temp, selfplay temp).

    Tree reuse OFF: entries within the same mode share one fresh tree per position.
    Tree reuse ON: each entry gets its own persistent tree across positions.

    Returns dict with all collected metrics.
    """
    base_entries = [e for e in entries if e[1] == "base"]
    selfplay_entries = [e for e in entries if e[1] == "selfplay"]
    has_selfplay = len(selfplay_entries) > 0
    has_base = len(base_entries) > 0

    mode_str = []
    if has_base and has_selfplay:
        mode_str.append("mixed base/selfplay")
    elif has_selfplay:
        mode_str.append("selfplay")
    else:
        mode_str.append("base")
    if tree_reuse:
        mode_str.append("tree reuse")
    header = f"Phase: Policy & Value Analysis ({', '.join(mode_str)})"

    print("=" * 60)
    print(header)
    print("=" * 60)

    if network_path is None and use_playout:
        agent = "playout"
    elif network_path is None:
        agent = "random"
    else:
        net_dir = os.path.dirname(network_path)
        net_file = os.path.basename(network_path)
        agent = neural_net.NNWrapper.load_checkpoint(Game, net_dir, net_file)
        agent.enable_inference_optimizations()

    num_players = Game.NUM_PLAYERS()
    num_moves = Game.NUM_MOVES()
    relative_values = Game().relative_values()

    anchor_vc = anchor[0]
    has_vc1_base = (1, "base") in entries
    has_vc1_selfplay = (1, "selfplay") in entries

    base_vcs = sorted(set(e[0] for e in base_entries))
    selfplay_vcs = sorted(set(e[0] for e in selfplay_entries))

    # The base reference tree always runs to anchor_vc for move selection + Q-values
    base_ref_target = anchor_vc

    # Per-game state
    game_states = [Game() for _ in range(ANALYSIS_GAMES)]
    game_positions = [[] for _ in range(ANALYSIS_GAMES)]
    active = list(range(ANALYSIS_GAMES))
    game_scores = [None] * ANALYSIS_GAMES

    cache = create_cache(Game, cache_size) if not isinstance(agent, str) else None

    def _make_base_mcts():
        return alphazero.MCTS(config.cpuct, num_players, num_moves, 0.0, 1.0,
                              config.fpu_reduction, relative_values,
                              config.root_fpu_zero, config.shaped_dirichlet)

    def _make_selfplay_mcts():
        return alphazero.MCTS(config.cpuct, num_players, num_moves, 0.25,
                              config.mcts_root_temp, config.fpu_reduction,
                              relative_values, config.root_fpu_zero,
                              config.shaped_dirichlet)

    # Tree storage: trees[gid] = {key: (mcts_obj, target_vc, is_selfplay)}
    # key is either an entry tuple (vc, mode) for tree_reuse, or "base"/"selfplay"
    # for shared trees, plus "base_ref" if needed
    trees = {}

    if tree_reuse:
        # One persistent tree per entry per game
        for gid in range(ANALYSIS_GAMES):
            trees[gid] = {}
            for entry in entries:
                vc, mode = entry
                is_sp = mode == "selfplay"
                mcts_obj = _make_selfplay_mcts() if is_sp else _make_base_mcts()
                trees[gid][entry] = (mcts_obj, vc, is_sp)
            # If anchor is selfplay, we need a separate base ref tree
            if anchor[1] == "selfplay":
                trees[gid]["base_ref"] = (_make_base_mcts(), anchor_vc, False)
    # Shared trees created per-position below

    print(f"Playing {ANALYSIS_GAMES} games with anchor {entry_label(anchor)} per position...")

    total_positions = 0
    pbar = tqdm.tqdm(desc="Positions", unit="pos")

    while active:
        n = len(active)

        # Per-position snapshot storage (keyed by entry tuples)
        pos_policies = [{} for _ in range(n)]
        pos_values = [{} for _ in range(n)]
        pos_q_values = [None] * n

        if not tree_reuse:
            # Create fresh shared trees per position
            for slot, gid in enumerate(active):
                trees[gid] = {}
                # Base shared tree — runs to max of base VCs and anchor_vc
                base_target = max(base_vcs + [anchor_vc]) if base_vcs else anchor_vc
                trees[gid]["base"] = (_make_base_mcts(), base_target, False)
                # Selfplay shared tree (if any selfplay entries)
                if selfplay_vcs:
                    trees[gid]["selfplay"] = (_make_selfplay_mcts(), max(selfplay_vcs), True)

        # Build the work list: (gid, key, tree, target_vc, is_selfplay)
        # For tree_reuse, keys are entry tuples + optional "base_ref"
        # For shared trees, keys are "base" and "selfplay"
        work_items = []
        for slot, gid in enumerate(active):
            for key, (mcts_obj, target, is_sp) in trees[gid].items():
                work_items.append((gid, key, mcts_obj, target, is_sp, slot))

        # Snapshot tracking: which entries have been captured for each slot
        snapshotted = [set() for _ in range(n)]

        def _snapshot(slot, entry, mcts_obj, is_sp):
            """Capture policy and value snapshot for an entry."""
            if entry in snapshotted[slot]:
                return
            snapshotted[slot].add(entry)
            pos_policies[slot][entry] = _get_policy(mcts_obj, is_sp, config)
            wld = np.array(mcts_obj.root_value())
            pos_values[slot][entry] = float(wld[0])

        def _check_snapshots(gid, key, mcts_obj, is_sp, slot):
            """Check if root_n has reached any target entries for this tree."""
            rn = mcts_obj.root_n()
            if tree_reuse:
                # key is an entry tuple or "base_ref"
                if isinstance(key, tuple) and rn >= key[0]:
                    _snapshot(slot, key, mcts_obj, is_sp)
            else:
                # Shared tree — check all entries of matching mode
                entries_to_check = selfplay_entries if is_sp else base_entries
                for entry in entries_to_check:
                    if entry[0] <= rn:
                        _snapshot(slot, entry, mcts_obj, is_sp)

        # Check if any trees are already at their target (tree reuse: root_n > 0)
        for gid, key, mcts_obj, target, is_sp, slot in work_items:
            _check_snapshots(gid, key, mcts_obj, is_sp, slot)

        # vc=1 is special: captured after 1 simulation from the appropriate tree
        vc1_captured_base = [False] * n
        vc1_captured_selfplay = [False] * n

        # Main simulation loop — interleaved across all trees
        while True:
            miss_items = []  # (gid, key, mcts_obj, leaf, hash, is_sp, slot, gs)

            for gid, key, mcts_obj, target, is_sp, slot in work_items:
                if mcts_obj.root_n() >= target:
                    continue  # this tree is done

                gs = game_states[gid]

                # Process cache hits inline until miss or target reached
                while mcts_obj.root_n() < target:
                    leaf = mcts_obj.find_leaf(gs)

                    if isinstance(agent, str):
                        # Random/playout: no caching, always a "miss"
                        miss_items.append((gid, key, mcts_obj, leaf, None, is_sp, slot, gs))
                        break

                    h = alphazero.hash_game_state(leaf)
                    if cache is not None:
                        result = cache.find(h, num_moves, num_players + 1)
                        if result is not None:
                            pi_cached, v_cached = result
                            mcts_obj.process_result(gs, v_cached, pi_cached, is_sp)
                            _check_snapshots(gid, key, mcts_obj, is_sp, slot)
                            # Check vc=1
                            if has_vc1_base and not is_sp and mcts_obj.root_n() >= 1 and not vc1_captured_base[slot]:
                                _snapshot(slot, (1, "base"), mcts_obj, is_sp)
                            if has_vc1_selfplay and is_sp and mcts_obj.root_n() >= 1 and not vc1_captured_selfplay[slot]:
                                _snapshot(slot, (1, "selfplay"), mcts_obj, is_sp)
                            continue  # try next leaf
                    miss_items.append((gid, key, mcts_obj, leaf, h, is_sp, slot, gs))
                    break  # need eval, stop this tree

            if not miss_items:
                break  # all trees done

            # Batch evaluate all misses
            results = _evaluate_leaves(agent, miss_items, game_states,
                                       num_players, num_moves, cache)

            for j, (gid, key, mcts_obj, leaf, h, is_sp, slot, gs) in enumerate(miss_items):
                v, pi = results[j]
                mcts_obj.process_result(gs, v, pi, is_sp)
                _check_snapshots(gid, key, mcts_obj, is_sp, slot)
                # Check vc=1 after first sim
                if has_vc1_base and not is_sp and mcts_obj.root_n() >= 1 and not vc1_captured_base[slot]:
                    _snapshot(slot, (1, "base"), mcts_obj, is_sp)
                    vc1_captured_base[slot] = True
                if has_vc1_selfplay and is_sp and mcts_obj.root_n() >= 1 and not vc1_captured_selfplay[slot]:
                    _snapshot(slot, (1, "selfplay"), mcts_obj, is_sp)
                    vc1_captured_selfplay[slot] = True

        # Capture Q-values from the base reference tree at anchor_vc
        for slot, gid in enumerate(active):
            if tree_reuse:
                if "base_ref" in trees[gid]:
                    ref_tree = trees[gid]["base_ref"][0]
                elif anchor in trees[gid] and not trees[gid][anchor][2]:
                    # anchor is a base entry
                    ref_tree = trees[gid][anchor][0]
                else:
                    ref_tree = trees[gid][anchor][0]
            else:
                ref_tree = trees[gid]["base"][0]
            pos_q_values[slot] = np.array(ref_tree.root_q_values())

            # Always store base reference policy/value under "base_ref" key
            pos_policies[slot]["base_ref"] = _get_policy(ref_tree, False, config)
            pos_values[slot]["base_ref"] = float(np.array(ref_tree.root_value())[0])
            # Fill anchor entry from base_ref if anchor is base mode and not yet captured
            if anchor[1] == "base" and anchor not in pos_policies[slot]:
                pos_policies[slot][anchor] = pos_policies[slot]["base_ref"]
                pos_values[slot][anchor] = pos_values[slot]["base_ref"]

        # Record position data and advance games
        next_active = []
        for slot, gid in enumerate(active):
            gs = game_states[gid]
            game_positions[gid].append(
                (gs.current_player(), pos_values[slot], pos_policies[slot], pos_q_values[slot])
            )
            total_positions += 1
            pbar.update(1)

            # Select move from base ref tree's policy with temperature
            if tree_reuse:
                if "base_ref" in trees[gid]:
                    ref_tree = trees[gid]["base_ref"][0]
                else:
                    ref_tree = trees[gid][anchor][0]
            else:
                ref_tree = trees[gid]["base"][0]

            probs = _get_policy(ref_tree, False, config)
            temp = calc_temp(config, gs.current_turn())
            if temp > 0 and temp != 1.0:
                probs = np.power(probs + 1e-10, 1.0 / temp)
                probs /= probs.sum()
            move = np.random.choice(len(probs), p=probs)

            if tree_reuse:
                # Update root for all persistent trees
                for key, (mcts_obj, target, is_sp) in trees[gid].items():
                    mcts_obj.update_root(gs, move)

            gs.play_move(move)

            if gs.scores() is not None:
                game_scores[gid] = np.array(gs.scores())
            else:
                next_active.append(gid)
                # Re-apply noise/temp on reused subtrees for selfplay trees
                if tree_reuse:
                    for key, (mcts_obj, target, is_sp) in trees[gid].items():
                        if is_sp and mcts_obj.root_n() > 0:
                            mcts_obj.apply_root_policy_temp()
                            mcts_obj.add_root_noise()

        active = next_active

    pbar.close()
    print(f"Collected {total_positions} positions across {ANALYSIS_GAMES} games")

    # Aggregate metrics from all games (keyed by entry tuples)
    all_jsd = {e: [] for e in entries}
    all_tv = {e: [] for e in entries}
    all_hellinger = {e: [] for e in entries}
    all_top1 = {e: [] for e in entries}
    all_top3 = {e: [] for e in entries}
    all_top9 = {e: [] for e in entries}
    position_values = {e: [] for e in entries}
    position_max_values = []
    position_outcomes = []
    # Expected reward: V(pi_entry) = sum(pi_entry * Q_base_ref)
    expected_reward = {e: [] for e in entries}

    for gid in range(ANALYSIS_GAMES):
        scores = game_scores[gid]
        for current_player, values_at_pos, policies_at_pos, q_values in game_positions[gid]:
            # Outcome for current player
            if scores[current_player] == 1:
                outcome = 1.0
            elif scores[-1] == 1:
                outcome = 0.5
            else:
                outcome = 0.0
            position_outcomes.append(outcome)

            # Policy divergence metrics vs anchor entry
            if anchor in policies_at_pos:
                ref_policy = policies_at_pos[anchor]
                for entry in entries:
                    if entry in policies_at_pos:
                        q = policies_at_pos[entry]
                        all_jsd[entry].append(jensen_shannon_divergence(ref_policy, q))
                        all_tv[entry].append(total_variation(ref_policy, q))
                        all_hellinger[entry].append(hellinger_distance(ref_policy, q))
                        all_top1[entry].append(top_k_agreement(ref_policy, q, 1))
                        all_top3[entry].append(top_k_agreement(ref_policy, q, 3))
                        all_top9[entry].append(top_k_agreement(ref_policy, q, 9))

            # Store values
            for entry in entries:
                if entry in values_at_pos:
                    position_values[entry].append(values_at_pos[entry])
            if anchor in values_at_pos:
                position_max_values.append(values_at_pos[anchor])

            # Expected reward using Q-values from base ref tree (objective evaluation)
            for entry in entries:
                if entry in policies_at_pos:
                    v_pi = float(np.sum(policies_at_pos[entry] * q_values))
                    expected_reward[entry].append(v_pi)

    position_outcomes = np.array(position_outcomes)
    position_max_values = np.array(position_max_values)

    metrics = {
        "entries": entries,
        "anchor": anchor,
        "anchor_vc": anchor_vc,
        "total_positions": total_positions,
        "num_games": ANALYSIS_GAMES,
    }

    # Policy divergence metrics
    policy_metric_names = ["jsd", "tv", "hellinger", "top1", "top3", "top9"]
    policy_metric_data = {
        "jsd": all_jsd, "tv": all_tv, "hellinger": all_hellinger,
        "top1": all_top1, "top3": all_top3, "top9": all_top9,
    }
    for name in policy_metric_names:
        means = {}
        all_vals = {}
        for entry in entries:
            data = policy_metric_data[name]
            if entry in data and len(data[entry]) > 0:
                arr = np.array(data[entry])
                means[entry] = float(np.mean(arr))
                all_vals[entry] = arr
        metrics[f"{name}_means"] = means
        metrics[f"{name}_all"] = all_vals

    # Value metrics
    value_corr_vs_max = {}
    value_mae_vs_max = {}
    value_corr_vs_outcome = {}
    value_mae_vs_outcome = {}

    for entry in entries:
        vals = np.array(position_values[entry])
        n = min(len(vals), len(position_max_values), len(position_outcomes))
        if n < 3:
            continue
        vals = vals[:n]
        max_vals = position_max_values[:n]
        outcomes = position_outcomes[:n]

        # vs anchor entry
        corr = np.corrcoef(vals, max_vals)[0, 1] if np.std(vals) > 1e-9 and np.std(max_vals) > 1e-9 else 0.0
        mae = float(np.mean(np.abs(vals - max_vals)))
        value_corr_vs_max[entry] = float(corr)
        value_mae_vs_max[entry] = mae

        # vs game outcome
        corr = np.corrcoef(vals, outcomes)[0, 1] if np.std(vals) > 1e-9 and np.std(outcomes) > 1e-9 else 0.0
        mae = float(np.mean(np.abs(vals - outcomes)))
        value_corr_vs_outcome[entry] = float(corr)
        value_mae_vs_outcome[entry] = mae

    metrics["value_corr_vs_max"] = value_corr_vs_max
    metrics["value_mae_vs_max"] = value_mae_vs_max
    metrics["value_corr_vs_outcome"] = value_corr_vs_outcome
    metrics["value_mae_vs_outcome"] = value_mae_vs_outcome

    # Expected reward and regret metrics
    reward_means = {}
    reward_all = {}
    regret_means = {}
    regret_all = {}
    # Anchor entry expected reward (for regret computation)
    anchor_reward_list = []
    for gid in range(ANALYSIS_GAMES):
        for current_player, values_at_pos, policies_at_pos, q_values in game_positions[gid]:
            if anchor in policies_at_pos:
                anchor_reward_list.append(float(np.sum(policies_at_pos[anchor] * q_values)))
    anchor_reward = np.array(anchor_reward_list) if anchor_reward_list else None

    for entry in entries:
        if entry in expected_reward and len(expected_reward[entry]) > 0:
            arr = np.array(expected_reward[entry])
            reward_means[entry] = float(np.mean(arr))
            reward_all[entry] = arr
    # Regret = V(pi_anchor) - V(pi_entry)
    if anchor_reward is not None:
        for entry in entries:
            if entry in reward_all:
                n_common = min(len(anchor_reward), len(reward_all[entry]))
                reg = anchor_reward[:n_common] - reward_all[entry][:n_common]
                regret_means[entry] = float(np.mean(reg))
                regret_all[entry] = reg
    metrics["reward_means"] = reward_means
    metrics["reward_all"] = reward_all
    metrics["regret_means"] = regret_means
    metrics["regret_all"] = regret_all

    # Print summary
    print(f"\n--- Policy Analysis (divergence from {entry_label(anchor)}) ---")
    print(f"{'Visits':>8s} {'JSD':>10s} {'TV':>10s} {'Hellinger':>10s} {'Top-1':>10s} {'Top-3':>10s} {'Top-9':>10s}")
    all_policy_entries = sorted(
        set().union(*(metrics[f"{n}_means"].keys() for n in policy_metric_names)),
        key=entry_sort_key
    )
    for entry in all_policy_entries:
        vals = []
        for name in policy_metric_names:
            v = metrics[f"{name}_means"].get(entry)
            vals.append(f"{v:>10.4f}" if v is not None else f"{'N/A':>10s}")
        print(f"{entry_label(entry):>8s} {''.join(vals)}")

    print(f"\n--- Value Analysis vs {entry_label(anchor)} ---")
    print(f"{'Visits':>8s} {'Corr':>8s} {'MAE':>8s}")
    for entry in sorted(value_corr_vs_max.keys(), key=entry_sort_key):
        print(f"{entry_label(entry):>8s} {value_corr_vs_max[entry]:>8.4f} {value_mae_vs_max[entry]:>8.4f}")

    print("\n--- Value Analysis vs Game Outcome ---")
    print(f"{'Visits':>8s} {'Corr':>8s} {'MAE':>8s}")
    for entry in sorted(value_corr_vs_outcome.keys(), key=entry_sort_key):
        print(f"{entry_label(entry):>8s} {value_corr_vs_outcome[entry]:>8.4f} {value_mae_vs_outcome[entry]:>8.4f}")

    if reward_means:
        print("\n--- Expected Reward (V(pi) = sum(pi * Q_anchor)) ---")
        print(f"{'Visits':>8s} {'E[reward]':>10s}")
        for entry in sorted(reward_means.keys(), key=entry_sort_key):
            print(f"{entry_label(entry):>8s} {reward_means[entry]:>10.4f}")

    if regret_means:
        print("\n--- Policy Regret (V(pi_anchor) - V(pi_entry)) ---")
        print(f"{'Visits':>8s} {'Regret':>10s}")
        for entry in sorted(regret_means.keys(), key=entry_sort_key):
            print(f"{entry_label(entry):>8s} {regret_means[entry]:>10.4f}")

    print_cache_stats(cache)
    del agent
    gc.collect()
    return metrics


# --- Phase 4: Visualization & Data Save ---

def _has_display():
    """Check if a graphical display is available."""
    if os.environ.get('SSH_TTY') and not os.environ.get('DISPLAY'):
        return False
    import matplotlib
    return matplotlib.get_backend().lower() != 'agg'


def visualize_and_save(entries, anchor, elo=None, win_matrix=None, metrics=None):
    """Generate plots and save raw data."""
    if not _has_display():
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    has_mixed = len(set(mode for _, mode in entries)) > 1

    def _plot_by_mode(ax, sorted_entries, values, base_color="tab:blue", sp_color="tab:orange", **kwargs):
        """Plot values split by mode with appropriate markers/lines."""
        base = [(e[0], v) for e, v in zip(sorted_entries, values) if e[1] == "base"]
        sp = [(e[0], v) for e, v in zip(sorted_entries, values) if e[1] == "selfplay"]
        if base:
            bx, by = zip(*base)
            ax.plot(bx, by, "o-", markersize=8, linewidth=2, color=base_color,
                    label="base" if has_mixed else None, **kwargs)
        if sp:
            sx, sy = zip(*sp)
            ax.plot(sx, sy, "^--", markersize=8, linewidth=2, color=sp_color,
                    label="selfplay" if has_mixed else None, **kwargs)
        if has_mixed:
            ax.legend(fontsize=8)

    save_dir = os.path.join("data", "threshold_analysis")
    os.makedirs(save_dir, exist_ok=True)

    # Tournament uses unique numeric VCs
    tournament_vcs = sorted(set(vc for vc, _ in entries))

    fig_num = 1

    # Figure 1: Elo vs visit count
    if elo is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(tournament_vcs, elo, "o-", markersize=8, linewidth=2)
        for i, vc in enumerate(tournament_vcs):
            ax.annotate(f"{elo[i]:.0f}", (vc, elo[i]),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("MCTS Visit Count")
        ax.set_ylabel("Elo Rating")
        ax.set_title("Elo vs MCTS Visit Count")
        ax.set_xticks(tournament_vcs)
        ax.set_xticklabels([str(v) for v in tournament_vcs])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(save_dir, "elo_vs_visits.png")
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
        fig_num += 1

    # Figure 2: Policy divergence metrics (2x3 grid)
    policy_metric_info = [
        ("jsd", "Jensen-Shannon Divergence", "lightblue"),
        ("tv", "Total Variation", "lightgreen"),
        ("hellinger", "Hellinger Distance", "lightyellow"),
        ("top1", "Top-1 Agreement", "lightsalmon"),
        ("top3", "Top-3 Agreement", "plum"),
        ("top9", "Top-9 Agreement", "lightskyblue"),
    ]
    has_policy_metrics = metrics and any(
        metrics.get(f"{name}_means") for name, _, _ in policy_metric_info
    )
    if has_policy_metrics:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (name, title, base_color) in enumerate(policy_metric_info):
            ax = axes[idx]
            means = metrics.get(f"{name}_means", {})
            all_data = metrics.get(f"{name}_all", {})
            if not means:
                ax.set_visible(False)
                continue

            sorted_entries = sorted(means.keys(), key=entry_sort_key)
            labels = [entry_label(e) for e in sorted_entries]

            # Box plot with mean overlay
            if all_data:
                bp_data = [all_data[e] for e in sorted_entries]
                bp = ax.boxplot(bp_data, tick_labels=labels,
                                showfliers=False, patch_artist=True)
                for i, entry in enumerate(sorted_entries):
                    if entry[1] == "selfplay":
                        bp["boxes"][i].set_facecolor("lightyellow" if base_color != "lightyellow" else "lightblue")
                    else:
                        bp["boxes"][i].set_facecolor(base_color)
                # Overlay mean line
                ax.plot(range(1, len(sorted_entries) + 1), [means[e] for e in sorted_entries],
                        "D-", markersize=6, linewidth=2, color="tab:blue",
                        zorder=3, label="mean")
                ax.legend(fontsize=8)
            else:
                ax.plot(range(1, len(sorted_entries) + 1), [means[e] for e in sorted_entries],
                        "o-", markersize=8, linewidth=2)
                ax.set_xticks(range(1, len(sorted_entries) + 1))
                ax.set_xticklabels(labels)

            ax.set_xlabel("MCTS Visit Count")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(save_dir, "policy_metrics.png")
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
        fig_num += 1

    # Figure 3: Value analysis (2x2)
    if metrics and "value_corr_vs_max" in metrics:
        corr_max = metrics["value_corr_vs_max"]
        mae_max = metrics["value_mae_vs_max"]
        corr_out = metrics["value_corr_vs_outcome"]
        mae_out = metrics["value_mae_vs_outcome"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        def _set_entry_xticks(ax, sorted_entries):
            unique_vcs = sorted(set(e[0] for e in sorted_entries))
            ax.set_xscale("log")
            ax.set_xticks(unique_vcs)
            ax.set_xticklabels([str(v) for v in unique_vcs])

        # Top-left: correlation vs anchor
        if corr_max:
            es = sorted(corr_max.keys(), key=entry_sort_key)
            _plot_by_mode(axes[0, 0], es, [corr_max[e] for e in es])
            _set_entry_xticks(axes[0, 0], es)
        axes[0, 0].set_xlabel("MCTS Visit Count")
        axes[0, 0].set_ylabel("Pearson Correlation")
        axes[0, 0].set_title(f"Value Correlation vs {entry_label(anchor)}")
        axes[0, 0].grid(True, alpha=0.3)

        # Top-right: MAE vs anchor
        if mae_max:
            es = sorted(mae_max.keys(), key=entry_sort_key)
            _plot_by_mode(axes[0, 1], es, [mae_max[e] for e in es],
                          base_color="tab:orange", sp_color="tab:red")
            _set_entry_xticks(axes[0, 1], es)
        axes[0, 1].set_xlabel("MCTS Visit Count")
        axes[0, 1].set_ylabel("MAE")
        axes[0, 1].set_title(f"Value MAE vs {entry_label(anchor)}")
        axes[0, 1].grid(True, alpha=0.3)

        # Bottom-left: correlation vs game outcome
        if corr_out:
            es = sorted(corr_out.keys(), key=entry_sort_key)
            _plot_by_mode(axes[1, 0], es, [corr_out[e] for e in es],
                          base_color="tab:green", sp_color="tab:olive")
            _set_entry_xticks(axes[1, 0], es)
        axes[1, 0].set_xlabel("MCTS Visit Count")
        axes[1, 0].set_ylabel("Pearson Correlation")
        axes[1, 0].set_title("Value Correlation vs Game Outcome")
        axes[1, 0].grid(True, alpha=0.3)

        # Bottom-right: MAE vs game outcome
        if mae_out:
            es = sorted(mae_out.keys(), key=entry_sort_key)
            _plot_by_mode(axes[1, 1], es, [mae_out[e] for e in es],
                          base_color="tab:red", sp_color="tab:purple")
            _set_entry_xticks(axes[1, 1], es)
        axes[1, 1].set_xlabel("MCTS Visit Count")
        axes[1, 1].set_ylabel("MAE")
        axes[1, 1].set_title("Value MAE vs Game Outcome")
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(save_dir, "value_analysis.png")
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")

    # Figure 4: Expected reward & policy regret
    if metrics and "reward_means" in metrics and metrics["reward_means"]:
        reward_means = metrics["reward_means"]
        reward_all = metrics["reward_all"]
        regret_means = metrics.get("regret_means", {})
        regret_all = metrics.get("regret_all", {})

        n_panels = 1 + (1 if regret_means else 0) + (1 if regret_all else 0)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

        panel = 0

        # Left: expected reward curve
        reward_es = sorted(reward_means.keys(), key=entry_sort_key)
        _plot_by_mode(axes[panel], reward_es, [reward_means[e] for e in reward_es])
        reward_unique_vcs = sorted(set(e[0] for e in reward_es))
        axes[panel].set_xscale("log")
        axes[panel].set_xlabel("MCTS Visit Count")
        axes[panel].set_ylabel("Expected Reward V(pi)")
        axes[panel].set_title("Expected Reward vs Visit Count")
        axes[panel].set_xticks(reward_unique_vcs)
        axes[panel].set_xticklabels([str(v) for v in reward_unique_vcs])
        axes[panel].grid(True, alpha=0.3)
        panel += 1

        # Center: mean regret curve
        if regret_means:
            regret_es = sorted(regret_means.keys(), key=entry_sort_key)
            _plot_by_mode(axes[panel], regret_es, [regret_means[e] for e in regret_es],
                          base_color="tab:red", sp_color="tab:purple")
            regret_unique_vcs = sorted(set(e[0] for e in regret_es))
            axes[panel].set_xscale("log")
            axes[panel].set_xlabel("MCTS Visit Count")
            axes[panel].set_ylabel("Regret")
            axes[panel].set_title("Mean Policy Regret vs Visit Count")
            axes[panel].set_xticks(regret_unique_vcs)
            axes[panel].set_xticklabels([str(v) for v in regret_unique_vcs])
            axes[panel].grid(True, alpha=0.3)
            panel += 1

        # Right: regret distribution box plot
        if regret_all:
            regret_box_es = sorted(regret_all.keys(), key=entry_sort_key)
            bp_data = [regret_all[e] for e in regret_box_es]
            labels = [entry_label(e) for e in regret_box_es]
            bp = axes[panel].boxplot(bp_data, tick_labels=labels,
                                     showfliers=False, patch_artist=True)
            for i, entry in enumerate(regret_box_es):
                bp["boxes"][i].set_facecolor("lightyellow" if entry[1] == "selfplay" else "lightsalmon")
            axes[panel].set_xlabel("MCTS Visit Count")
            axes[panel].set_ylabel("Regret")
            axes[panel].set_title("Policy Regret Distribution")
            axes[panel].grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(save_dir, "policy_regret.png")
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")

    # Save raw data
    save_data = {
        "visit_counts": np.array(tournament_vcs),
        "entry_labels": np.array([entry_label(e) for e in entries]),
        "anchor": np.array(entry_label(anchor)),
    }
    if elo is not None:
        save_data["elo"] = elo
    if win_matrix is not None:
        save_data["win_matrix"] = win_matrix
    if metrics:
        for key in ["total_positions", "num_games", "anchor_vc"]:
            if key in metrics:
                save_data[key] = np.array(metrics[key])
        # Policy divergence metrics
        for prefix in ["jsd", "tv", "hellinger", "top1", "top3", "top9"]:
            if f"{prefix}_means" in metrics:
                for entry, val in metrics[f"{prefix}_means"].items():
                    save_data[f"{prefix}_mean_{entry_label(entry)}"] = np.array(val)
            if f"{prefix}_all" in metrics:
                for entry, arr in metrics[f"{prefix}_all"].items():
                    save_data[f"{prefix}_all_{entry_label(entry)}"] = arr
        # Value data
        for prefix in ["value_corr_vs_max", "value_mae_vs_max",
                        "value_corr_vs_outcome", "value_mae_vs_outcome"]:
            if prefix in metrics:
                for entry, val in metrics[prefix].items():
                    save_data[f"{prefix}_{entry_label(entry)}"] = np.array(val)
        # Expected reward and regret data
        for prefix in ["reward_means", "regret_means"]:
            if prefix in metrics:
                for entry, val in metrics[prefix].items():
                    save_data[f"{prefix}_{entry_label(entry)}"] = np.array(val)
        for prefix in ["reward_all", "regret_all"]:
            if prefix in metrics:
                for entry, arr in metrics[prefix].items():
                    save_data[f"{prefix}_{entry_label(entry)}"] = arr

    npz_path = os.path.join(save_dir, "threshold_data.npz")
    np.savez(npz_path, **save_data)
    print(f"Saved: {npz_path}")

    if _has_display():
        plt.show()
    else:
        print("No display detected, skipping interactive display")


def main():
    config, Game, network_path, entries, anchor, phases, use_playout, cache_size, tree_reuse = interactive_config()

    elo = None
    win_matrix = None
    metrics = None

    if "tournament" in phases:
        elo, win_matrix = run_tournament(config, Game, network_path, entries, use_playout,
                                         cache_size=cache_size)

    if "analysis" in phases:
        metrics = run_analysis(config, Game, network_path, entries, anchor, use_playout,
                               cache_size=cache_size, tree_reuse=tree_reuse)

    visualize_and_save(entries, anchor, elo=elo, win_matrix=win_matrix, metrics=metrics)

    print("\nDone!")


if __name__ == "__main__":
    main()
