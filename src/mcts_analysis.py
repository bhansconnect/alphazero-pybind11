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
import time as _time
from typing import NamedTuple
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
from policy_metrics import (
    jensen_shannon_divergence, total_variation, hellinger_distance,
    top_k_agreement, kl_divergence, policy_entropy,
)

np.set_printoptions(precision=3, suppress=True)

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")

# --- Constants ---
DEFAULT_VISIT_COUNTS = [1, 25, 50, 100, 200, 400, 800]
TOURNAMENT_BATCH_SIZE = 64
TOURNAMENT_CONCURRENT_BATCHES = 2
ANALYSIS_GAMES = 64


class Entry(NamedTuple):
    """An analysis entry: (visit_count, mode, batch_size).

    mode: "base" (no noise, eval temp) or "selfplay" (Dirichlet, root temp).
    batch_size: WU-UCT batch size. 1 = sequential (default).
    """
    vc: int
    mode: str
    batch_size: int = 1


# --- Entry helpers ---

def _to_entry(e):
    """Normalize a tuple to an Entry (handles 2-tuples for backward compat)."""
    if isinstance(e, Entry):
        return e
    if len(e) == 2:
        return Entry(e[0], e[1], 1)
    return Entry(*e)


def entry_label(entry):
    """Label for an entry: '200', '200sp', '200b8', '200spb8'."""
    e = _to_entry(entry)
    label = f"{e.vc}sp" if e.mode == "selfplay" else str(e.vc)
    if e.batch_size > 1:
        label += f"b{e.batch_size}"
    elif e.batch_size == 0:
        label += "b0"
    return label


def _resolve_batch_size(entry):
    """Resolve batch_size: 0 (auto) -> sqrt(vc), else as-is."""
    if entry.batch_size == 0:
        return max(1, int(math.sqrt(entry.vc)))
    return entry.batch_size


def entry_sort_key(entry):
    """Sort key: by VC, then base before selfplay, then batch_size."""
    e = _to_entry(entry)
    return (e.vc, 0 if e.mode == "base" else 1, e.batch_size)


def _load_agent(Game, network_path, use_playout):
    """Load network agent, or return 'playout'/'random' string for non-NN agents."""
    if network_path is None and use_playout:
        return "playout"
    elif network_path is None:
        return "random"
    net_dir = os.path.dirname(network_path)
    net_file = os.path.basename(network_path)
    agent = neural_net.NNWrapper.load_checkpoint(Game, net_dir, net_file)
    agent.enable_inference_optimizations()
    return agent


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


# --- Benchmark Phase ---

BENCHMARK_POSITIONS = 30


def _make_mcts(is_selfplay, config, Game):
    """Create an MCTS instance with config params for the given mode."""
    num_players = Game.NUM_PLAYERS()
    num_moves = Game.NUM_MOVES()
    relative_values = Game().relative_values()
    if is_selfplay:
        return alphazero.MCTS(config.cpuct, num_players, num_moves, 0.25,
                              config.mcts_root_temp, config.fpu_reduction,
                              relative_values, config.root_fpu_zero,
                              config.shaped_dirichlet)
    return alphazero.MCTS(config.cpuct, num_players, num_moves, 0.0, 1.0,
                          config.fpu_reduction, relative_values,
                          config.root_fpu_zero, config.shaped_dirichlet)


def run_benchmark(config, Game, agent, entries, cache_size=0, tree_reuse=False):
    """Benchmark: measure sims/s for each entry with independent MCTS + cache."""
    from play import run_mcts_search

    results = {}  # entry -> {'sims_per_sec': float, 'total_time': float, 'positions': int}
    bench_entries = [e for e in entries if e.vc > 1]
    pbar_entry = tqdm.tqdm(bench_entries, desc="Benchmark", unit="entry")

    for entry in pbar_entry:
        pbar_entry.set_description(f"Bench {entry_label(entry)}")
        gs = Game()
        cache = create_cache(Game, cache_size) if not isinstance(agent, str) else None
        is_sp = entry.mode == "selfplay"
        mcts = _make_mcts(is_sp, config, Game)

        nn_agent = None if isinstance(agent, str) else agent
        eval_type = agent if isinstance(agent, str) else "network"

        total_time = 0.0
        positions = 0
        pbar_pos = tqdm.tqdm(total=BENCHMARK_POSITIONS, desc="Positions",
                             unit="pos", leave=False)

        while gs.scores() is None and positions < BENCHMARK_POSITIONS:
            t0 = _time.time()
            counts, sims, _ = run_mcts_search(
                gs, nn_agent, mcts, node_limit=entry.vc,
                eval_type=eval_type, cache=cache,
                max_batch_size=entry.batch_size,
            )
            total_time += _time.time() - t0
            positions += 1
            pbar_pos.update(1)

            action = int(np.argmax(counts))
            gs.play_move(action)

            if tree_reuse:
                try:
                    mcts.update_root(gs, action)
                except Exception:
                    mcts = _make_mcts(is_sp, config, Game)
            else:
                mcts = _make_mcts(is_sp, config, Game)

        sps = (entry.vc * positions) / total_time if total_time > 0 else 0
        results[entry] = {
            'sims_per_sec': sps,
            'total_time': total_time,
            'positions': positions,
        }
        pbar_pos.close()
        pbar_entry.set_postfix(sps=f"{sps:,.0f}")

    pbar_entry.close()

    # Print benchmark summary
    if results:
        print(f"\n--- Benchmark Timing ---")
        print(f"{'Entry':>8s} {'Total(s)':>10s} {'Per-pos(s)':>12s} {'Sims/s':>10s}")
        for entry in sorted(results.keys(), key=entry_sort_key):
            data = results[entry]
            per_pos = data['total_time'] / data['positions'] if data['positions'] > 0 else 0
            print(f"{entry_label(entry):>8s} {data['total_time']:>10.2f} {per_pos:>12.4f} {data['sims_per_sec']:>10.0f}")

    return results


# --- Phase 1: Interactive Configuration ---

def interactive_config():
    """Interactive configuration. Returns (config, Game, network_path, entries, anchor, phases, use_playout, cache_size, tree_reuse, experiment_dir)."""
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

    # Visit counts as (vc, mode, batch_size) entries
    print(f"Default visit counts: {DEFAULT_VISIT_COUNTS}")
    print("  (1 = raw network policy, no MCTS search)")
    print("  (append 'sp' for selfplay mode, e.g. 200sp)")
    print("  (append 'b<N>' for batched WU-UCT, e.g. 200b8, 200spb16, 120b0 for auto)")
    print("  (same VC can appear in both modes, e.g. 200, 200sp)")
    vc_input = input("Enter comma-separated visit counts or press Enter for defaults: ").strip()
    if vc_input:
        entries = []
        for token in vc_input.split(","):
            token = token.strip().lower()
            # Parse optional bN suffix
            batch_size = 1
            batch_match = re.search(r'b(\d+)$', token)
            if batch_match:
                batch_size = int(batch_match.group(1))
                token = token[:batch_match.start()]
            if token.endswith("sp"):
                vc = int(token[:-2])
                mode = "selfplay"
            else:
                vc = int(token)
                mode = "base"
            entry = Entry(vc, mode, batch_size)
            if entry not in entries:
                entries.append(entry)
        anchor = entries[-1]  # last entry typed
        entries.sort(key=entry_sort_key)
    else:
        entries = [Entry(vc, "base") for vc in DEFAULT_VISIT_COUNTS]
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

    # Derive experiment directory from network_path for saving analysis output
    if network_path is not None:
        # network_path is data/{game}/{experiment}/checkpoint/{iter}-{name}.pt
        experiment_dir = os.path.dirname(os.path.dirname(network_path))
    else:
        experiment_dir = os.path.join("data", game_name)

    return config, Game, network_path, entries, anchor, phases, use_playout, cache_size, tree_reuse, experiment_dir


# --- Phase 2: Elo Tournament (Round-Robin) ---

class MctsSettings(NamedTuple):
    epsilon: float
    mcts_root_temp: float
    root_fpu_zero: bool


def _entry_mcts_settings(entry, config) -> MctsSettings:
    """Return MctsSettings for a player's entry.

    Selfplay entries get Dirichlet noise, root policy temp, and root FPU zero.
    Base entries get clean tournament settings (no noise, root_temp=1.0, no root FPU zero).
    """
    entry = _to_entry(entry)
    mode = entry.mode
    if mode == "selfplay":
        return MctsSettings(0.25, config.mcts_root_temp, True)
    return MctsSettings(0.0, 1.0, False)


def run_tournament(config, Game, agent, entries, cache_size=0):
    """Run Monrad (Swiss-style) Elo tournament between entry configs.

    Each entry is a (visit_count, mode) tuple where mode is "base" or "selfplay".
    agent: NNWrapper, 'playout', or 'random' (from _load_agent).
    Returns (elo_ratings, win_matrix) indexed by entries.
    """
    entries = [_to_entry(e) for e in entries]
    print("=" * 60)
    print("Phase: Elo Tournament (Monrad)")
    print("=" * 60)

    count = len(entries)

    # Wrap agent for tournament (needs list of players)
    num_players = Game.NUM_PLAYERS()
    if agent == "playout":
        tournament_agent = PlayoutPlayer()
    elif agent == "random":
        tournament_agent = RandPlayer()
    else:
        tournament_agent = agent
    def make_players():
        return [tournament_agent] * num_players

    win_matrix = np.full((count, count), np.nan)
    elo = np.zeros(count)
    rankings = list(range(count))
    rounds = int(np.ceil(np.log2(count)))
    dist = count

    # Create a single shared cache for all matchups (same network throughout)
    shared_cache = create_sharded_cache(Game, cache_size) if cache_size > 0 else None

    # Check if any entry uses selfplay mode (needs per-seat overrides)
    has_selfplay = any(e.mode == "selfplay" for e in entries)

    for r in tqdm.trange(rounds, desc="Rounds"):
        dist = math.ceil(dist / 2)

        # --- Monrad pairing ---
        pairings = []
        current = len(rankings) - 1
        played = [False] * count

        while current >= 0:
            if played[rankings[current]]:
                current -= 1
                continue
            played[rankings[current]] = True

            # Find opponent at target distance
            offset = dist
            while current - offset >= 0 and (
                played[rankings[current - offset]]
                or not math.isnan(
                    win_matrix[rankings[current], rankings[current - offset]]
                )
            ):
                offset += 1

            if current - offset < 0:
                offset = 1
                while current - offset >= 0 and (
                    played[rankings[current - offset]]
                    or not math.isnan(
                        win_matrix[rankings[current], rankings[current - offset]]
                    )
                ):
                    offset += 1
                if current - offset < 0:
                    offset = 1
                    while not math.isnan(
                        win_matrix[rankings[current], rankings[current - offset]]
                    ):
                        offset += 1

            played[rankings[current - offset]] = True
            pairings.append((rankings[current], rankings[current - offset]))
            current -= 1

        # --- Play matchups with progress bar ---
        for i, j in tqdm.tqdm(pairings, desc="Matchups", leave=False):
            entry_i = entries[i]
            entry_j = entries[j]

            players = make_players()
            depths = [entry_j[0]] * num_players
            depths[0] = entry_i[0]

            if shared_cache is not None:
                caches = [shared_cache] * num_players
            else:
                caches = None

            kwargs = {}
            if has_selfplay:
                eps_i, rpt_i, rfz_i = _entry_mcts_settings(entry_i, config)
                eps_j, rpt_j, rfz_j = _entry_mcts_settings(entry_j, config)
                kwargs["player_epsilon"] = [eps_i] + [eps_j] * (num_players - 1)
                kwargs["player_mcts_root_temp"] = [rpt_i] + [rpt_j] * (num_players - 1)
                kwargs["player_root_fpu_zero"] = [rfz_i] + [rfz_j] * (num_players - 1)

            label_i = entry_label(entry_i)
            label_j = entry_label(entry_j)
            name = f"{label_i}-{label_j}"
            win_rates = pit_agents(
                config, Game, players, depths,
                TOURNAMENT_BATCH_SIZE, name,
                cache_size=0,
                caches=caches,
                **kwargs,
            )
            win_matrix[i, j] = win_rates[0]
            win_matrix[j, i] = win_rates[1]

        # Update ELO and rankings after each round
        elo = calc_elo(elo, win_matrix)
        rankings = list(np.argsort(elo))

    # Print shared cache stats
    print_sharded_cache_stats(shared_cache)

    print("\n--- Tournament Results ---")
    print(f"{'Entry':>10s} {'Elo':>8s}")
    for i, entry in enumerate(entries):
        print(f"{entry_label(entry):>10s} {elo[i]:>8.0f}")

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


def run_analysis(config, Game, agent, entries, anchor,
                 cache_size=0, tree_reuse=False):
    """Run policy & value convergence analysis with batched inference.

    Plays ANALYSIS_GAMES concurrently, batching all NN inference calls across
    games for maximum cache sharing.

    Per-entry modes: each (vc, mode) entry can be "base" (no noise, eval temp)
    or "selfplay" (Dirichlet noise, root policy temp, selfplay temp).

    Tree reuse OFF: entries within the same mode share one fresh tree per position.
    Tree reuse ON: each entry gets its own persistent tree across positions.

    agent: NNWrapper, 'playout', or 'random' (from _load_agent).

    Returns (metrics, position_snapshots).
    position_snapshots: list of dicts with 'gs', 'player', 'values' keys.
    """
    entries = [_to_entry(e) for e in entries]
    anchor = _to_entry(anchor)
    base_entries = [e for e in entries if e.mode == "base"]
    selfplay_entries = [e for e in entries if e.mode == "selfplay"]
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

    num_players = Game.NUM_PLAYERS()
    num_moves = Game.NUM_MOVES()
    relative_values = Game().relative_values()

    anchor_vc = anchor.vc
    has_vc1_base = any(e.vc == 1 and e.mode == "base" for e in entries)
    has_vc1_selfplay = any(e.vc == 1 and e.mode == "selfplay" for e in entries)

    # PIO: always capture vc=1 for gap analysis
    _pio_inject_base = not has_vc1_base  # Always needed as PIO baseline
    _pio_inject_selfplay = not has_vc1_selfplay and has_selfplay
    has_vc1_base = has_vc1_base or _pio_inject_base
    has_vc1_selfplay = has_vc1_selfplay or _pio_inject_selfplay

    base_vcs = sorted(set(e.vc for e in base_entries))
    selfplay_vcs = sorted(set(e.vc for e in selfplay_entries))

    # The base reference tree always runs to anchor_vc for move selection + Q-values
    base_ref_target = anchor_vc

    # Per-game state
    game_states = [Game() for _ in range(ANALYSIS_GAMES)]
    game_positions = [[] for _ in range(ANALYSIS_GAMES)]
    active = list(range(ANALYSIS_GAMES))
    game_scores = [None] * ANALYSIS_GAMES

    # Position snapshots (one per turn, not per sub-move)
    position_snapshots = []
    snapshot_prev_player = [game_states[i].current_player() for i in range(ANALYSIS_GAMES)]

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
        # One persistent tree per entry per game (unbatched only)
        # Batched entries grouped by (mode, batch_size) share one tree per group
        for gid in range(ANALYSIS_GAMES):
            trees[gid] = {}
            for entry in entries:
                if entry.batch_size != 1:
                    continue
                is_sp = entry.mode == "selfplay"
                mcts_obj = _make_selfplay_mcts() if is_sp else _make_base_mcts()
                trees[gid][entry] = (mcts_obj, entry.vc, is_sp)
            # Group batched entries
            _tr_groups = {}
            for entry in entries:
                if entry.batch_size != 1:
                    _tr_groups.setdefault((entry.mode, entry.batch_size), []).append(entry)
            for (mode, bs), group in _tr_groups.items():
                is_sp = mode == "selfplay"
                mcts_obj = _make_selfplay_mcts() if is_sp else _make_base_mcts()
                target = max(e.vc for e in group)
                trees[gid][("batched", mode, bs)] = (mcts_obj, target, is_sp)
            # Need a separate base ref tree if anchor is selfplay or batched
            if anchor.mode == "selfplay" or anchor.batch_size != 1:
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

        # Identify batched entries (batch_size != 1) — these need their own trees
        batched_entries = [e for e in entries if _to_entry(e).batch_size != 1]
        # Group batched entries by (mode, batch_size) to share trees
        batched_groups = {}
        for entry in batched_entries:
            gk = (entry.mode, entry.batch_size)
            batched_groups.setdefault(gk, []).append(entry)

        if not tree_reuse:
            # Create fresh shared trees per position
            for slot, gid in enumerate(active):
                trees[gid] = {}
                # Base shared tree — runs to max of base VCs (unbatched only) and anchor_vc
                unbatched_base_vcs = sorted(set(e.vc for e in base_entries if _to_entry(e).batch_size == 1))
                base_target = max(unbatched_base_vcs + [anchor_vc]) if unbatched_base_vcs else anchor_vc
                trees[gid]["base"] = (_make_base_mcts(), base_target, False)
                # Selfplay shared tree (if any unbatched selfplay entries)
                unbatched_sp_vcs = sorted(set(e.vc for e in selfplay_entries if _to_entry(e).batch_size == 1))
                if unbatched_sp_vcs:
                    trees[gid]["selfplay"] = (_make_selfplay_mcts(), max(unbatched_sp_vcs), True)
                # Batched entries: one shared tree per (mode, batch_size) group
                for (mode, bs), group in batched_groups.items():
                    is_sp = mode == "selfplay"
                    mcts_obj = _make_selfplay_mcts() if is_sp else _make_base_mcts()
                    target = max(e.vc for e in group)
                    group_key = ("batched", mode, bs)
                    trees[gid][group_key] = (mcts_obj, target, is_sp)

        # Build the work list: (gid, key, tree, target_vc, is_selfplay)
        # For tree_reuse, keys are entry tuples + optional "base_ref"
        # For shared trees, keys are "base" and "selfplay"
        # Batched entries (batch_size != 1) are processed separately.
        work_items = []
        batched_work_items = []
        for slot, gid in enumerate(active):
            for key, (mcts_obj, target, is_sp) in trees[gid].items():
                item = (gid, key, mcts_obj, target, is_sp, slot)
                if isinstance(key, tuple) and len(key) == 3 and key[0] == "batched":
                    batched_work_items.append(item)
                else:
                    work_items.append(item)

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
                if isinstance(key, tuple) and rn >= key.vc:
                    _snapshot(slot, key, mcts_obj, is_sp)
            else:
                # Shared tree — check all unbatched entries of matching mode
                # (batched entries snapshot from their own trees)
                entries_to_check = selfplay_entries if is_sp else base_entries
                for entry in entries_to_check:
                    if _to_entry(entry).batch_size != 1:
                        continue
                    if entry.vc <= rn:
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
                                _snapshot(slot, Entry(1, "base"), mcts_obj, is_sp)
                            if has_vc1_selfplay and is_sp and mcts_obj.root_n() >= 1 and not vc1_captured_selfplay[slot]:
                                _snapshot(slot, Entry(1, "selfplay"), mcts_obj, is_sp)
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
                    _snapshot(slot, Entry(1, "base"), mcts_obj, is_sp)
                    vc1_captured_base[slot] = True
                if has_vc1_selfplay and is_sp and mcts_obj.root_n() >= 1 and not vc1_captured_selfplay[slot]:
                    _snapshot(slot, Entry(1, "selfplay"), mcts_obj, is_sp)
                    vc1_captured_selfplay[slot] = True

        # Process batched work items — grouped by (mode, batch_size), shared tree
        for gid, key, mcts_obj, target, is_sp, slot in batched_work_items:
            _, mode, batch_size = key  # ("batched", mode, bs)
            group_entries = batched_groups[(mode, batch_size)]
            sorted_group = sorted(group_entries, key=lambda e: e.vc)
            gs = game_states[gid]
            bs = _resolve_batch_size(group_entries[0])  # all same batch_size

            while mcts_obj.root_n() < target:
                # Cache-aware batch: accumulate up to bs GPU misses,
                # with max 2*bs total attempts. Terminals, cache hits,
                # and random evals are backpropagated immediately.
                pending_gpu = []    # (leaf_idx, hash)
                canonical_list = []
                playout_leaves = []  # (leaf_idx, leaf) for playout batch
                max_attempts = 2 * bs

                for attempt in range(max_attempts):
                    if len(pending_gpu) + len(playout_leaves) >= bs:
                        break

                    leaf = mcts_obj.find_leaf_batched(gs)
                    leaf_idx = mcts_obj.in_flight_count() - 1

                    # Terminal
                    if leaf.scores() is not None:
                        v = np.array(leaf.scores(), dtype=np.float32)
                        pi = np.zeros(num_moves, dtype=np.float32)
                        mcts_obj.process_result_batched(gs, leaf_idx, v, pi, is_sp)
                        continue

                    if agent == "playout":
                        playout_leaves.append((leaf_idx, leaf))
                        continue
                    elif agent == "random":
                        v = np.full(num_players + 1, 1.0 / (num_players + 1), dtype=np.float32)
                        pi = np.full(num_moves, 1.0 / num_moves, dtype=np.float32)
                        mcts_obj.process_result_batched(gs, leaf_idx, v, pi, is_sp)
                        continue

                    # NN agent — check cache
                    h = alphazero.hash_game_state(leaf)
                    if cache is not None:
                        result = cache.find(h, num_moves, num_players + 1)
                        if result is not None:
                            pi_c, v_c = result
                            mcts_obj.process_result_batched(gs, leaf_idx,
                                                            np.array(v_c), np.array(pi_c), is_sp)
                            continue

                    canonical_list.append(np.array(leaf.canonicalized()))
                    pending_gpu.append((leaf_idx, h))

                # Batch playout evaluation
                if playout_leaves:
                    leaves = [lf for _, lf in playout_leaves]
                    v_np, pi_np = alphazero.playout_eval_batch(leaves)
                    for j, (leaf_idx, _) in enumerate(playout_leaves):
                        mcts_obj.process_result_batched(gs, leaf_idx,
                                                        v_np[j], pi_np[j], is_sp)

                # Batched GPU inference for cache misses
                if pending_gpu:
                    batch_tensor = torch.from_numpy(np.stack(canonical_list))
                    v_batch, pi_batch = agent.process(batch_tensor)
                    v_np, pi_np = v_batch.cpu().numpy(), pi_batch.cpu().numpy()
                    for j, (leaf_idx, h) in enumerate(pending_gpu):
                        v, pi = v_np[j].flatten(), pi_np[j].flatten()
                        if cache is not None:
                            cache.insert(h, pi, v)
                        mcts_obj.process_result_batched(gs, leaf_idx, v, pi, is_sp)

                mcts_obj.reset_batch()

                # Snapshot entries whose vc threshold has been reached
                rn = mcts_obj.root_n()
                for entry in sorted_group:
                    if entry.vc <= rn:
                        _snapshot(slot, entry, mcts_obj, is_sp)

            # Final snapshot for any remaining entries
            for entry in sorted_group:
                _snapshot(slot, entry, mcts_obj, is_sp)

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
            if anchor.mode == "base" and anchor not in pos_policies[slot]:
                pos_policies[slot][anchor] = pos_policies[slot]["base_ref"]
                pos_values[slot][anchor] = pos_values[slot]["base_ref"]

        # Record position data and advance games
        next_active = []
        for slot, gid in enumerate(active):
            gs = game_states[gid]
            game_positions[gid].append(
                (gs.current_player(), pos_values[slot], pos_policies[slot], pos_q_values[slot])
            )
            # Save snapshot (once per turn, not every sub-move in
            # multi-move-per-turn games)
            cur_player = gs.current_player()
            if cur_player != snapshot_prev_player[gid] or gs.current_turn() == 0:
                position_snapshots.append({
                    "gs": gs.copy(),
                    "player": cur_player,
                    "values": {entry: pos_values[slot].get(entry)
                               for entry in entries if entry in pos_values[slot]},
                })
                snapshot_prev_player[gid] = cur_player
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

    # PIO gap accumulators (only for entries with vc > 1)
    pio_entries = [e for e in entries if e.vc > 1]
    pio_kl = {e: [] for e in pio_entries}
    pio_top1_flip = {e: [] for e in pio_entries}
    pio_entropy_raw = {e: [] for e in pio_entries}
    pio_entropy_mcts = {e: [] for e in pio_entries}
    pio_entropy_reduction = {e: [] for e in pio_entries}
    pio_value_correction = {e: [] for e in pio_entries}
    pio_value_sign_flip = {e: [] for e in pio_entries}
    pio_value_accuracy_gain = {e: [] for e in pio_entries}
    pio_correction_quality = {e: [] for e in pio_entries}

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

            # PIO gap metrics: compare each entry (vc > 1) against vc=1 baseline
            for entry in pio_entries:
                baseline_entry = Entry(1, "base")
                if entry not in policies_at_pos or baseline_entry not in policies_at_pos:
                    continue
                pi_mcts = policies_at_pos[entry]
                pi_raw = policies_at_pos[baseline_entry]

                # KL(pi_mcts || pi_raw)
                pio_kl[entry].append(kl_divergence(pi_mcts, pi_raw))

                # Top-1 flip
                move_mcts = int(np.argmax(pi_mcts))
                move_raw = int(np.argmax(pi_raw))
                flipped = move_mcts != move_raw
                pio_top1_flip[entry].append(float(flipped))

                # Entropy
                h_raw = policy_entropy(pi_raw)
                h_mcts = policy_entropy(pi_mcts)
                pio_entropy_raw[entry].append(h_raw)
                pio_entropy_mcts[entry].append(h_mcts)
                pio_entropy_reduction[entry].append(h_raw - h_mcts)

                # Value correction
                if entry in values_at_pos and baseline_entry in values_at_pos:
                    v_mcts = values_at_pos[entry]
                    v_raw = values_at_pos[baseline_entry]
                    pio_value_correction[entry].append(abs(v_mcts - v_raw))
                    pio_value_sign_flip[entry].append(float((v_raw - 0.5) * (v_mcts - 0.5) < 0))
                    # Value accuracy gain: does search make the value more accurate?
                    pio_value_accuracy_gain[entry].append(
                        float(abs(v_mcts - outcome) < abs(v_raw - outcome))
                    )

                # Correction quality: when top-1 flips, is Q[move_mcts] > Q[move_raw]?
                if flipped:
                    pio_correction_quality[entry].append(float(q_values[move_mcts] > q_values[move_raw]))

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

    # Value calibration (ECE) per entry — adaptive equal-mass bins
    NUM_CAL_BINS = 10
    value_ece = {}
    value_calibration_data = {}
    for entry in entries:
        vals = np.array(position_values[entry])
        n = min(len(vals), len(position_outcomes))
        if n < NUM_CAL_BINS:
            continue
        vals = vals[:n]
        outcomes = position_outcomes[:n]

        # Sort by predicted value, split into equal-mass bins
        order = np.argsort(vals)
        bin_size = n // NUM_CAL_BINS
        bin_pred = np.zeros(NUM_CAL_BINS)
        bin_actual = np.zeros(NUM_CAL_BINS)
        bin_count = np.zeros(NUM_CAL_BINS)
        for b in range(NUM_CAL_BINS):
            start = b * bin_size
            end = n if b == NUM_CAL_BINS - 1 else (b + 1) * bin_size
            idx = order[start:end]
            bin_pred[b] = vals[idx].mean()
            bin_actual[b] = outcomes[idx].mean()
            bin_count[b] = len(idx)

        total = bin_count.sum()
        ece = float(np.sum(bin_count * np.abs(bin_pred - bin_actual)) / total)
        value_ece[entry] = ece
        value_calibration_data[entry] = (bin_pred, bin_actual, bin_count)

    metrics["value_ece"] = value_ece
    metrics["value_calibration_data"] = value_calibration_data

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

    # PIO gap metrics aggregation
    pio_metric_names = ["pio_kl", "pio_top1_flip", "pio_entropy_raw", "pio_entropy_mcts",
                        "pio_entropy_reduction", "pio_value_correction",
                        "pio_value_sign_flip", "pio_value_accuracy_gain",
                        "pio_correction_quality"]
    pio_metric_data = {
        "pio_kl": pio_kl, "pio_top1_flip": pio_top1_flip,
        "pio_entropy_raw": pio_entropy_raw, "pio_entropy_mcts": pio_entropy_mcts,
        "pio_entropy_reduction": pio_entropy_reduction,
        "pio_value_correction": pio_value_correction,
        "pio_value_sign_flip": pio_value_sign_flip,
        "pio_value_accuracy_gain": pio_value_accuracy_gain,
        "pio_correction_quality": pio_correction_quality,
    }
    for name in pio_metric_names:
        means = {}
        all_vals = {}
        for entry in pio_entries:
            data = pio_metric_data[name]
            if entry in data and len(data[entry]) > 0:
                arr = np.array(data[entry])
                means[entry] = float(np.mean(arr))
                all_vals[entry] = arr
        metrics[f"{name}_means"] = means
        metrics[f"{name}_all"] = all_vals

    # Marginal KL: KL(pi_N2 || pi_N1) for consecutive visit-count pairs within each mode
    pio_marginal_kl = {}
    for mode in ("base", "selfplay"):
        mode_entries = sorted([e for e in entries if e.mode == mode], key=entry_sort_key)
        if not mode_entries:
            continue
        # Always use base vc=1 as chain start (universal PIO baseline)
        if mode_entries[0][0] > 1 and Entry(1, "base") not in set(mode_entries):
            mode_entries = [Entry(1, "base")] + mode_entries
        for i in range(1, len(mode_entries)):
            e_prev, e_curr = mode_entries[i-1], mode_entries[i]
            vals = []
            for gid in range(ANALYSIS_GAMES):
                for _, _, policies_at_pos, _ in game_positions[gid]:
                    if e_prev in policies_at_pos and e_curr in policies_at_pos:
                        vals.append(kl_divergence(policies_at_pos[e_curr], policies_at_pos[e_prev]))
            if vals:
                pio_marginal_kl[e_curr] = np.array(vals)
    metrics["pio_marginal_kl_means"] = {e: float(np.mean(v)) for e, v in pio_marginal_kl.items()}
    metrics["pio_marginal_kl_all"] = pio_marginal_kl

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

    if value_ece:
        print(f"\n--- Value Calibration (ECE, lower is better) ---")
        print(f"{'Visits':>8s} {'ECE':>10s}")
        for entry in sorted(value_ece.keys(), key=entry_sort_key):
            print(f"{entry_label(entry):>8s} {value_ece[entry]:>10.4f}")

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

    # PIO gap summary
    pio_kl_means = metrics.get("pio_kl_means", {})
    if pio_kl_means:
        print(f"\n--- PIO Gap Analysis (vs raw network, vc=1 base) ---")
        print(f"{'Visits':>8s} {'KL':>10s} {'Top1Flip':>10s} {'EntReduc':>10s} {'ValCorr':>10s} {'ValFlip':>10s} {'ValGain':>10s} {'CorrQual':>10s}")
        for entry in sorted(pio_kl_means.keys(), key=entry_sort_key):
            def _fmt(name):
                v = metrics.get(f"{name}_means", {}).get(entry)
                return f"{v:>10.4f}" if v is not None else f"{'N/A':>10s}"
            print(f"{entry_label(entry):>8s} {_fmt('pio_kl')}{_fmt('pio_top1_flip')}{_fmt('pio_entropy_reduction')}{_fmt('pio_value_correction')}{_fmt('pio_value_sign_flip')}{_fmt('pio_value_accuracy_gain')}{_fmt('pio_correction_quality')}")

    marginal_kl_means = metrics.get("pio_marginal_kl_means", {})
    if marginal_kl_means:
        print(f"\n--- Marginal KL (KL between consecutive visit counts) ---")
        print(f"{'Visits':>8s} {'MargKL':>10s}")
        for entry in sorted(marginal_kl_means.keys(), key=entry_sort_key):
            print(f"{entry_label(entry):>8s} {marginal_kl_means[entry]:>10.4f}")

    print_cache_stats(cache)
    gc.collect()
    return metrics, position_snapshots



# --- Scaling Readiness Report ---

def compute_scaling_report(entries, anchor, elo=None, metrics=None):
    """Compute derived scaling readiness metrics from raw analysis data.

    Returns dict with:
      elo_per_doubling: list of (vc1, vc2, elo_per_2x, mode) quads
      elo_regressions: {mode: (slope, r2)}
      policy_improvement: {entry: float}  (pio_top1_flip, ascending with VC)
      capacity_score: {entry: float}  (vc > 1 only, pio_top1_flip * correction_quality)
      mcts_utilization: {entry: float}
      value_ece: {entry: float}  (Expected Calibration Error, lower is better)
      value_accuracy_gain: {entry: float}  (rate where search improves value, vc > 1 only)
    """
    entries = [_to_entry(e) for e in entries]
    anchor = _to_entry(anchor)

    if elo is None and metrics is None:
        return {}

    result = {}

    # 1. Elo per Doubling — computed separately per mode to avoid index mismatch
    if elo is not None:
        modes_present = sorted(set(e.mode for e in entries))
        elo_per_doubling = []
        regressions = {}

        for mode in modes_present:
            mode_entries = [e for e in entries if e.mode == mode]
            mode_vcs = sorted(e.vc for e in mode_entries)
            mode_elos = [elo[entries.index(e)] for e in sorted(mode_entries, key=lambda e: e.vc)]

            for i in range(len(mode_vcs) - 1):
                vc1, vc2 = mode_vcs[i], mode_vcs[i + 1]
                log2_ratio = math.log2(vc2 / vc1)
                if log2_ratio > 0:
                    elo_per_doubling.append((vc1, vc2, (mode_elos[i + 1] - mode_elos[i]) / log2_ratio, mode))

            if len(mode_vcs) >= 2:
                log2_vcs = np.array([math.log2(vc) for vc in mode_vcs])
                elo_arr = np.array(mode_elos)
                if np.std(log2_vcs) > 1e-9:
                    slope, intercept = np.polyfit(log2_vcs, elo_arr, 1)
                    ss_res = np.sum((elo_arr - (slope * log2_vcs + intercept)) ** 2)
                    ss_tot = np.sum((elo_arr - np.mean(elo_arr)) ** 2)
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
                    regressions[mode] = (float(slope), float(r2))

        if elo_per_doubling:
            result["elo_per_doubling"] = elo_per_doubling
        if regressions:
            result["elo_regressions"] = regressions

    if metrics is None:
        return result

    # 2. Policy Improvement: pio_top1_flip (ascending with VC — measures how much
    #    MCTS improves over raw network policy)
    pio_flip = metrics.get("pio_top1_flip_means", {})
    if pio_flip:
        result["policy_improvement"] = dict(pio_flip)

    # 3. Capacity Score: pio_top1_flip * correction_quality (vc > 1 only)
    #    Measures both how much search changes the policy AND whether those changes
    #    are correct (ascending with VC when search is beneficial)
    correction_quality = metrics.get("pio_correction_quality_means", {})
    if pio_flip and correction_quality:
        capacity = {}
        for entry in correction_quality:
            e = _to_entry(entry)
            if e.vc > 1 and entry in pio_flip:
                capacity[entry] = pio_flip[entry] * correction_quality[entry]
        if capacity:
            result["capacity_score"] = capacity

    # 4. Value calibration (ECE) — lower is better
    value_ece = metrics.get("value_ece", {})
    if value_ece:
        result["value_ece"] = dict(value_ece)

    # 5. Value accuracy gain — rate where search improves value (vc > 1 only)
    vag = metrics.get("pio_value_accuracy_gain_means", {})
    if vag:
        result["value_accuracy_gain"] = dict(vag)

    # 6. MCTS Utilization: (reward_entry - reward_vc1) / (reward_anchor - reward_vc1)
    reward_means = metrics.get("reward_means", {})
    if reward_means:
        # Find vc=1 baseline reward (prefer base mode)
        vc1_reward = None
        for mode in ("base", "selfplay"):
            key = Entry(1, mode)
            if key in reward_means:
                vc1_reward = reward_means[key]
                break
        anchor_reward = reward_means.get(anchor)
        if vc1_reward is not None and anchor_reward is not None:
            denom = anchor_reward - vc1_reward
            if abs(denom) > 1e-9:
                utilization = {}
                for entry, reward in reward_means.items():
                    utilization[entry] = (reward - vc1_reward) / denom
                result["mcts_utilization"] = utilization

    # 7. Training Signal: target entropy + KL gap vs visit count
    pio_entropy = metrics.get("pio_entropy_mcts_means", {})
    pio_kl_data = metrics.get("pio_kl_means", {})
    if pio_entropy:
        result["training_signal_entropy"] = dict(pio_entropy)
    if pio_kl_data:
        result["training_signal_kl"] = dict(pio_kl_data)

    return result


def print_scaling_report(scaling, entries, anchor):
    """Print formatted terminal summary of scaling readiness metrics."""
    print(f"\n--- Scaling Readiness Report ---")

    # Elo per doubling
    if "elo_per_doubling" in scaling:
        print(f"\n  Elo per Doubling:")
        for vc1, vc2, epd, mode in scaling["elo_per_doubling"]:
            suffix = "sp" if mode == "selfplay" else ""
            print(f"    {vc1}{suffix:>3s} -> {vc2}{suffix:<3s}  {epd:>8.1f} Elo/2x")
        if "elo_regressions" in scaling:
            for mode, (slope, r2) in scaling["elo_regressions"].items():
                label = f" ({mode})" if len(scaling["elo_regressions"]) > 1 else ""
                print(f"    Log-linear fit{label}: slope={slope:.1f} Elo/2x, R2={r2:.3f}")

    # Per-entry table
    has_pi = "policy_improvement" in scaling
    has_ece = "value_ece" in scaling
    has_vag = "value_accuracy_gain" in scaling
    has_cs = "capacity_score" in scaling
    has_mu = "mcts_utilization" in scaling
    has_tse = "training_signal_entropy" in scaling
    has_tsk = "training_signal_kl" in scaling
    if has_pi or has_ece or has_vag or has_cs or has_mu or has_tse or has_tsk:
        header = f"\n  {'Visits':>8s}"
        if has_pi:
            header += f" {'PolImpr':>8s}"
        if has_ece:
            header += f" {'ECE':>8s}"
        if has_vag:
            header += f" {'ValGain':>8s}"
        if has_cs:
            header += f" {'CapScore':>8s}"
        if has_mu:
            header += f" {'MCTSUtil':>8s}"
        if has_tse:
            header += f" {'TgtEntr':>8s}"
        if has_tsk:
            header += f" {'KLGap':>8s}"
        print(header)

        sorted_entries = sorted(
            set().union(
                scaling.get("policy_improvement", {}).keys(),
                scaling.get("value_ece", {}).keys(),
                scaling.get("value_accuracy_gain", {}).keys(),
                scaling.get("capacity_score", {}).keys(),
                scaling.get("mcts_utilization", {}).keys(),
                scaling.get("training_signal_entropy", {}).keys(),
                scaling.get("training_signal_kl", {}).keys(),
            ),
            key=entry_sort_key,
        )
        for entry in sorted_entries:
            row = f"  {entry_label(entry):>8s}"
            if has_pi:
                v = scaling["policy_improvement"].get(entry)
                row += f" {v:>8.4f}" if v is not None else f" {'N/A':>8s}"
            if has_ece:
                v = scaling["value_ece"].get(entry)
                row += f" {v:>8.4f}" if v is not None else f" {'N/A':>8s}"
            if has_vag:
                v = scaling["value_accuracy_gain"].get(entry)
                row += f" {v:>8.4f}" if v is not None else f" {'N/A':>8s}"
            if has_cs:
                v = scaling["capacity_score"].get(entry)
                row += f" {v:>8.4f}" if v is not None else f" {'N/A':>8s}"
            if has_mu:
                v = scaling["mcts_utilization"].get(entry)
                row += f" {v:>8.4f}" if v is not None else f" {'N/A':>8s}"
            if has_tse:
                v = scaling["training_signal_entropy"].get(entry)
                row += f" {v:>8.4f}" if v is not None else f" {'N/A':>8s}"
            if has_tsk:
                v = scaling["training_signal_kl"].get(entry)
                row += f" {v:>8.4f}" if v is not None else f" {'N/A':>8s}"
            print(row)


# --- Phase 4: Visualization & Data Save ---

def _has_display():
    """Check if a graphical display is available."""
    if os.environ.get('SSH_TTY') and not os.environ.get('DISPLAY'):
        return False
    import matplotlib
    return matplotlib.get_backend().lower() != 'agg'


# Metrics that are binary or discrete rates — bar charts with Wilson CI instead of boxplots
_rate_metrics = {"top1", "top3", "top9",
                 "pio_top1_flip", "pio_value_sign_flip", "pio_value_accuracy_gain",
                 "pio_correction_quality"}


def _wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion. Returns (lower, upper)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _plot_metric_panel(ax, metric_name, title, base_color, sorted_entries, means, all_data):
    """Plot a single metric panel as either a bar+CI chart (rate metrics) or boxplot.

    Rate/binary metrics get bar charts with Wilson 95% CI error bars.
    Continuous metrics keep the existing boxplot + mean overlay.
    """
    labels = [entry_label(e) for e in sorted_entries]

    if metric_name in _rate_metrics and all_data:
        # Bar chart with Wilson CI
        positions = range(len(sorted_entries))
        bar_means = []
        ci_lo = []
        ci_hi = []
        bar_colors = []
        for entry in sorted_entries:
            arr = all_data[entry]
            n = len(arr)
            k = float(np.sum(arr))
            p = k / n if n > 0 else 0.0
            lo, hi = _wilson_ci(k, n)
            bar_means.append(p)
            ci_lo.append(max(0.0, p - lo))
            ci_hi.append(max(0.0, hi - p))
            bar_colors.append("lightyellow" if entry.mode == "selfplay" else base_color)

        bars = ax.bar(positions, bar_means, color=bar_colors, edgecolor="gray",
                       linewidth=0.8, zorder=2)
        ax.errorbar(positions, bar_means, yerr=[ci_lo, ci_hi],
                     fmt="none", ecolor="black", capsize=4, linewidth=1.5, zorder=3)
        # Annotate mean values
        for i, m in enumerate(bar_means):
            ax.annotate(f"{m:.2f}", (i, m), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, min(1.05, max(bar_means) + 0.15) if bar_means else 1.05)
    elif all_data:
        # Boxplot with mean overlay (existing logic)
        bp_data = [all_data[e] for e in sorted_entries]
        bp = ax.boxplot(bp_data, tick_labels=labels,
                        showfliers=False, patch_artist=True)
        for i, entry in enumerate(sorted_entries):
            if entry.mode == "selfplay":
                bp["boxes"][i].set_facecolor("lightyellow" if base_color != "lightyellow" else "lightblue")
            else:
                bp["boxes"][i].set_facecolor(base_color)
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


def _set_entry_xticks(ax, sorted_entries):
    """Set log-scale x-axis with ticks at unique visit counts."""
    unique_vcs = sorted(set(e.vc for e in sorted_entries))
    ax.set_xscale("log")
    ax.set_xticks(unique_vcs)
    ax.set_xticklabels([str(v) for v in unique_vcs])


def visualize_and_save(entries, anchor, elo=None, win_matrix=None, metrics=None, save_dir=None):
    """Generate plots and save raw data."""
    if not _has_display():
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    entries = [_to_entry(e) for e in entries]
    anchor = _to_entry(anchor)

    has_mixed = len(set(e.mode for e in entries)) > 1

    def _plot_by_mode(ax, sorted_entries, values, base_color="tab:blue", sp_color="tab:orange", **kwargs):
        """Plot values split by mode with appropriate markers/lines."""
        base = [(e.vc, v) for e, v in zip(sorted_entries, values) if e.mode == "base"]
        sp = [(e.vc, v) for e, v in zip(sorted_entries, values) if e.mode == "selfplay"]
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

    if save_dir is None:
        save_dir = os.path.join("data", "mcts_analysis")
    os.makedirs(save_dir, exist_ok=True)

    fig_num = 1

    # Figure 1: Elo vs visit count (indexed by entries)
    if elo is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        sorted_entries = sorted(entries, key=entry_sort_key)
        sorted_elo = [elo[entries.index(e)] for e in sorted_entries]
        _plot_by_mode(ax, sorted_entries, sorted_elo)
        for e, el in zip(sorted_entries, sorted_elo):
            ax.annotate(f"{el:.0f}", (e.vc, el),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("MCTS Visit Count")
        ax.set_ylabel("Elo Rating")
        ax.set_title("Elo vs MCTS Visit Count")
        unique_vcs = sorted(set(e.vc for e in entries))
        ax.set_xticks(unique_vcs)
        ax.set_xticklabels([str(v) for v in unique_vcs])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(save_dir, "mcts_elo_vs_visits.png")
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
            _plot_metric_panel(ax, name, title, base_color, sorted_entries, means, all_data)

        fig.tight_layout()
        path = os.path.join(save_dir, "mcts_policy_metrics.png")
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
        fig_num += 1

    # Figure 3: Value analysis (2x3)
    if metrics and "value_corr_vs_max" in metrics:
        corr_max = metrics["value_corr_vs_max"]
        mae_max = metrics["value_mae_vs_max"]
        corr_out = metrics["value_corr_vs_outcome"]
        mae_out = metrics["value_mae_vs_outcome"]
        value_ece = metrics.get("value_ece", {})
        calibration_data = metrics.get("value_calibration_data", {})

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Top-left: correlation vs anchor
        if corr_max:
            es = sorted(corr_max.keys(), key=entry_sort_key)
            _plot_by_mode(axes[0, 0], es, [corr_max[e] for e in es])
            _set_entry_xticks(axes[0, 0], es)
        axes[0, 0].set_xlabel("MCTS Visit Count")
        axes[0, 0].set_ylabel("Pearson Correlation")
        axes[0, 0].set_title(f"Value Correlation vs {entry_label(anchor)}")
        axes[0, 0].grid(True, alpha=0.3)

        # Top-center: MAE vs anchor
        if mae_max:
            es = sorted(mae_max.keys(), key=entry_sort_key)
            _plot_by_mode(axes[0, 1], es, [mae_max[e] for e in es],
                          base_color="tab:orange", sp_color="tab:red")
            _set_entry_xticks(axes[0, 1], es)
        axes[0, 1].set_xlabel("MCTS Visit Count")
        axes[0, 1].set_ylabel("MAE")
        axes[0, 1].set_title(f"Value MAE vs {entry_label(anchor)}")
        axes[0, 1].grid(True, alpha=0.3)

        # Top-right: ECE bar chart
        if value_ece:
            ece_entries = sorted(value_ece.keys(), key=entry_sort_key)
            _plot_by_mode(axes[0, 2], ece_entries, [value_ece[e] for e in ece_entries],
                          base_color="tab:red", sp_color="tab:pink")
            _set_entry_xticks(axes[0, 2], ece_entries)
            axes[0, 2].set_xlabel("MCTS Visit Count")
            axes[0, 2].set_ylabel("ECE (lower=better)")
            axes[0, 2].set_title("Expected Calibration Error")
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].set_visible(False)

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

        # Bottom-center: MAE vs game outcome
        if mae_out:
            es = sorted(mae_out.keys(), key=entry_sort_key)
            _plot_by_mode(axes[1, 1], es, [mae_out[e] for e in es],
                          base_color="tab:red", sp_color="tab:purple")
            _set_entry_xticks(axes[1, 1], es)
        axes[1, 1].set_xlabel("MCTS Visit Count")
        axes[1, 1].set_ylabel("MAE")
        axes[1, 1].set_title("Value MAE vs Game Outcome")
        axes[1, 1].grid(True, alpha=0.3)

        # Bottom-right: Calibration curves for all entries
        if calibration_data:
            cal_entries = sorted(calibration_data.keys(), key=entry_sort_key)
            cmap = plt.cm.viridis
            n_cal = len(cal_entries)
            for idx, entry in enumerate(cal_entries):
                bin_pred, bin_actual, bin_count = calibration_data[entry]
                color = cmap(idx / max(n_cal - 1, 1))
                axes[1, 2].plot(bin_pred, bin_actual, "o-", color=color,
                                markersize=3, linewidth=1.2, alpha=0.8,
                                label=entry_label(entry))
            axes[1, 2].plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
            axes[1, 2].set_xlabel("Predicted Value")
            axes[1, 2].set_ylabel("Actual Win Rate")
            axes[1, 2].set_title("Calibration Curves")
            axes[1, 2].set_xlim(-0.05, 1.05)
            axes[1, 2].set_ylim(-0.05, 1.05)
            axes[1, 2].set_aspect("equal")
            axes[1, 2].legend(fontsize=6, loc="upper left")
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].set_visible(False)

        fig.tight_layout()
        path = os.path.join(save_dir, "mcts_value_analysis.png")
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
        reward_unique_vcs = sorted(set(e.vc for e in reward_es))
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
            regret_unique_vcs = sorted(set(e.vc for e in regret_es))
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
                bp["boxes"][i].set_facecolor("lightyellow" if entry.mode == "selfplay" else "lightsalmon")
            axes[panel].set_xlabel("MCTS Visit Count")
            axes[panel].set_ylabel("Regret")
            axes[panel].set_title("Policy Regret Distribution")
            axes[panel].grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(save_dir, "mcts_policy_regret.png")
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")

    # Figure 5: PIO Gap Summary (2x4 grid)
    has_pio = metrics and metrics.get("pio_kl_means")
    if has_pio:
        pio_panel_info = [
            ("pio_kl", "KL(pi_mcts || pi_raw)", "lightblue"),
            ("pio_top1_flip", "Top-1 Flip Rate", "lightgreen"),
            ("pio_entropy_reduction", "Entropy Reduction", "lightyellow"),
            ("pio_value_correction", "|V_mcts - V_raw|", "lightsalmon"),
            ("pio_value_sign_flip", "Value Sign Flip Rate", "plum"),
            ("pio_value_accuracy_gain", "Value Accuracy Gain", "lightcyan"),
            ("pio_correction_quality", "Correction Quality", "lightskyblue"),
        ]
        n_panels = len(pio_panel_info)
        n_cols = 4
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 5 * n_rows))
        axes = axes.flatten()

        for idx, (name, title, base_color) in enumerate(pio_panel_info):
            ax = axes[idx]
            means = metrics.get(f"{name}_means", {})
            all_data = metrics.get(f"{name}_all", {})
            if not means:
                ax.set_visible(False)
                continue

            sorted_entries = sorted(means.keys(), key=entry_sort_key)
            _plot_metric_panel(ax, name, title, base_color, sorted_entries, means, all_data)

        # Hide unused axes
        for idx in range(len(pio_panel_info), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("PIO Gap Analysis (MCTS vs Raw Network)", fontsize=14, y=1.02)
        fig.tight_layout()
        path = os.path.join(save_dir, "mcts_pio_gap_summary.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    # Figure 6: Marginal Returns
    has_marginal = metrics and metrics.get("pio_marginal_kl_means")
    has_reward = metrics and metrics.get("reward_means")
    if has_marginal or has_reward:
        n_panels = (1 if has_marginal else 0) + (1 if has_reward else 0)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]
        panel = 0

        if has_marginal:
            marginal_means = metrics["pio_marginal_kl_means"]
            marginal_es = sorted(marginal_means.keys(), key=entry_sort_key)
            _plot_by_mode(axes[panel], marginal_es, [marginal_means[e] for e in marginal_es])
            marginal_vcs = sorted(set(e.vc for e in marginal_es))
            axes[panel].set_xscale("log")
            axes[panel].set_xlabel("MCTS Visit Count")
            axes[panel].set_ylabel("Marginal KL")
            axes[panel].set_title("Marginal KL Between Consecutive Visit Counts")
            axes[panel].set_xticks(marginal_vcs)
            axes[panel].set_xticklabels([str(v) for v in marginal_vcs])
            axes[panel].grid(True, alpha=0.3)
            panel += 1

        if has_reward:
            reward_means = metrics["reward_means"]
            reward_es = sorted(reward_means.keys(), key=entry_sort_key)
            _plot_by_mode(axes[panel], reward_es, [reward_means[e] for e in reward_es])

            # Horizontal dashed lines for vc=1 baselines
            for mode, color in [("base", "tab:blue"), ("selfplay", "tab:orange")]:
                baseline = (1, mode)
                if baseline in reward_means:
                    axes[panel].axhline(y=reward_means[baseline], color=color,
                                        linestyle="--", alpha=0.5,
                                        label=f"vc=1 {mode}")
            axes[panel].legend(fontsize=8)

            reward_vcs = sorted(set(e.vc for e in reward_es))
            axes[panel].set_xscale("log")
            axes[panel].set_xlabel("MCTS Visit Count")
            axes[panel].set_ylabel("Expected Reward V(pi)")
            axes[panel].set_title("Expected Reward vs Visit Count")
            axes[panel].set_xticks(reward_vcs)
            axes[panel].set_xticklabels([str(v) for v in reward_vcs])
            axes[panel].grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(save_dir, "mcts_pio_marginal_returns.png")
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")

    # Figure 7: Scaling Readiness Report
    scaling = compute_scaling_report(entries, anchor, elo=elo, metrics=metrics)
    if scaling:
        print_scaling_report(scaling, entries, anchor)

        fig, axes = plt.subplots(2, 3, figsize=(21, 10))

        # Top-left: Elo per Doubling (bar chart)
        if "elo_per_doubling" in scaling:
            epd = scaling["elo_per_doubling"]
            mode_colors = {"base": "tab:blue", "selfplay": "tab:orange"}
            bar_labels = []
            bar_vals = []
            bar_colors = []
            for vc1, vc2, val, mode in epd:
                suffix = "sp" if mode == "selfplay" else ""
                bar_labels.append(f"{vc1}{suffix}->{vc2}{suffix}")
                bar_vals.append(val)
                bar_colors.append(mode_colors.get(mode, "tab:blue"))
            positions = range(len(bar_labels))
            axes[0, 0].bar(positions, bar_vals, color=bar_colors, edgecolor="gray",
                           linewidth=0.8, zorder=2)
            if "elo_regressions" in scaling:
                for mode, (slope, r2) in scaling["elo_regressions"].items():
                    label = f"{mode} R2={r2:.3f}" if len(scaling["elo_regressions"]) > 1 else f"R2={r2:.3f}"
                    axes[0, 0].axhline(y=slope, color=mode_colors.get(mode, "red"),
                                       linestyle="--", linewidth=1.5, zorder=3, label=label)
                axes[0, 0].legend(fontsize=8)
            axes[0, 0].set_xticks(list(positions))
            axes[0, 0].set_xticklabels(bar_labels, rotation=45, ha="right")
            axes[0, 0].set_ylabel("Elo per Doubling")
            axes[0, 0].set_title("Elo per Doubling")
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].set_visible(False)

        # Top-center: Capacity Score
        if "capacity_score" in scaling:
            cs = scaling["capacity_score"]
            cs_entries = sorted(cs.keys(), key=entry_sort_key)
            _plot_by_mode(axes[0, 1], cs_entries, [cs[e] for e in cs_entries])
            _set_entry_xticks(axes[0, 1], cs_entries)
            axes[0, 1].set_xlabel("MCTS Visit Count")
            axes[0, 1].set_ylabel("Capacity Score")
            axes[0, 1].set_title("Capacity Score (Top-1 Flip x Correction Quality)")
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].set_visible(False)

        # Top-right: Training Signal — Target Entropy H(pi_mcts)
        if "training_signal_entropy" in scaling:
            tse = scaling["training_signal_entropy"]
            tse_entries = sorted(tse.keys(), key=entry_sort_key)
            _plot_by_mode(axes[0, 2], tse_entries, [tse[e] for e in tse_entries],
                          base_color="tab:purple", sp_color="tab:pink")
            _set_entry_xticks(axes[0, 2], tse_entries)
            axes[0, 2].set_xlabel("MCTS Visit Count")
            axes[0, 2].set_ylabel("Target Entropy H(pi_mcts)")
            axes[0, 2].set_title("Training Signal: Target Entropy")
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].set_visible(False)

        # Bottom-left: MCTS Utilization
        if "mcts_utilization" in scaling:
            mu = scaling["mcts_utilization"]
            mu_entries = sorted(mu.keys(), key=entry_sort_key)
            _plot_by_mode(axes[1, 0], mu_entries, [mu[e] for e in mu_entries],
                          base_color="tab:green", sp_color="tab:olive")
            _set_entry_xticks(axes[1, 0], mu_entries)
            axes[1, 0].axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
            axes[1, 0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
            axes[1, 0].set_xlabel("MCTS Visit Count")
            axes[1, 0].set_ylabel("MCTS Utilization")
            axes[1, 0].set_title("MCTS Utilization (fraction of search improvement)")
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].set_visible(False)

        # Bottom-center: Value Calibration & Accuracy Gain (dual y-axis)
        has_ece = "value_ece" in scaling
        has_vag = "value_accuracy_gain" in scaling
        if has_ece or has_vag:
            ax_left = axes[1, 1]
            all_entries = []
            if has_ece:
                ece = scaling["value_ece"]
                ece_entries = sorted(ece.keys(), key=entry_sort_key)
                _plot_by_mode(ax_left, ece_entries, [ece[e] for e in ece_entries],
                              base_color="tab:red", sp_color="tab:pink")
                ax_left.set_ylabel("ECE (lower=better)", color="tab:red")
                ax_left.tick_params(axis="y", labelcolor="tab:red")
                ax_left.invert_yaxis()
                all_entries = ece_entries
            if has_vag:
                ax_right = ax_left.twinx()
                vag = scaling["value_accuracy_gain"]
                vag_entries = sorted(vag.keys(), key=entry_sort_key)
                _plot_by_mode(ax_right, vag_entries, [vag[e] for e in vag_entries],
                              base_color="tab:blue", sp_color="tab:cyan")
                ax_right.set_ylabel("Value Accuracy Gain", color="tab:blue")
                ax_right.tick_params(axis="y", labelcolor="tab:blue")
                ax_right.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
                if not all_entries:
                    all_entries = vag_entries
            _set_entry_xticks(ax_left, all_entries)
            ax_left.set_xlabel("MCTS Visit Count")
            ax_left.set_title("Value Calibration & Accuracy Gain")
            ax_left.grid(True, alpha=0.3)
        else:
            axes[1, 1].set_visible(False)

        # Bottom-right: Training Signal — KL Gap KL(pi_mcts || pi_raw)
        if "training_signal_kl" in scaling:
            tsk = scaling["training_signal_kl"]
            tsk_entries = sorted(tsk.keys(), key=entry_sort_key)
            _plot_by_mode(axes[1, 2], tsk_entries, [tsk[e] for e in tsk_entries],
                          base_color="tab:red", sp_color="tab:orange")
            _set_entry_xticks(axes[1, 2], tsk_entries)
            axes[1, 2].set_xlabel("MCTS Visit Count")
            axes[1, 2].set_ylabel("KL Gap (pi_mcts || pi_raw)")
            axes[1, 2].set_title("Training Signal: KL Gap")
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].set_visible(False)

        fig.suptitle("Scaling Readiness Report", fontsize=14)
        fig.tight_layout()
        path = os.path.join(save_dir, "mcts_scaling_report.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    # Figure: Search Timing from benchmark (Sims/s and per-position time)
    benchmark_timing = metrics.get("benchmark_timing", {}) if metrics else {}
    if benchmark_timing:
        from matplotlib.cm import get_cmap
        from matplotlib.patches import Patch

        sorted_timing_entries = sorted(benchmark_timing.keys(), key=entry_sort_key)
        labels = [entry_label(e) for e in sorted_timing_entries]
        sims_per_sec = [benchmark_timing[e]['sims_per_sec'] for e in sorted_timing_entries]
        per_pos = [benchmark_timing[e]['total_time'] / benchmark_timing[e]['positions']
                   if benchmark_timing[e]['positions'] > 0 else 0
                   for e in sorted_timing_entries]

        # Color by batch_size
        unique_bs = sorted(set(e.batch_size for e in sorted_timing_entries))
        cmap = get_cmap("viridis", max(len(unique_bs), 2))
        bs_to_color = {bs: cmap(i / max(len(unique_bs) - 1, 1)) for i, bs in enumerate(unique_bs)}
        colors = [bs_to_color[e.batch_size] for e in sorted_timing_entries]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(labels))

        ax1.bar(x, sims_per_sec, color=colors)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax1.set_ylabel("Simulations / second")
        ax1.set_title("Search Throughput (Sims/s)")
        ax1.grid(True, alpha=0.3, axis="y")

        ax2.bar(x, per_pos, color=colors)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right")
        ax2.set_ylabel("Seconds / position")
        ax2.set_title("Per-Position Search Time")
        ax2.grid(True, alpha=0.3, axis="y")

        # Legend mapping colors to batch sizes
        legend_elements = [
            Patch(facecolor=bs_to_color[bs],
                  label="seq" if bs == 1 else f"b={bs}")
            for bs in unique_bs
        ]
        fig.legend(handles=legend_elements, loc="upper right", fontsize=9,
                   title="Batch Size")
        fig.tight_layout(rect=[0, 0, 0.92, 1])
        path = os.path.join(save_dir, "mcts_search_timing.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    # Save raw data
    save_data = {
        "visit_counts": np.array(sorted(set(e.vc for e in entries))),
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
        # PIO gap metrics
        pio_prefixes = ["pio_kl", "pio_top1_flip", "pio_entropy_raw", "pio_entropy_mcts",
                        "pio_entropy_reduction", "pio_value_correction",
                        "pio_value_sign_flip", "pio_value_accuracy_gain",
                        "pio_correction_quality"]
        for prefix in pio_prefixes:
            if f"{prefix}_means" in metrics:
                for entry, val in metrics[f"{prefix}_means"].items():
                    save_data[f"{prefix}_mean_{entry_label(entry)}"] = np.array(val)
            if f"{prefix}_all" in metrics:
                for entry, arr in metrics[f"{prefix}_all"].items():
                    save_data[f"{prefix}_all_{entry_label(entry)}"] = arr
        # Marginal KL
        if "pio_marginal_kl_means" in metrics:
            for entry, val in metrics["pio_marginal_kl_means"].items():
                save_data[f"pio_marginal_kl_mean_{entry_label(entry)}"] = np.array(val)
        if "pio_marginal_kl_all" in metrics:
            for entry, arr in metrics["pio_marginal_kl_all"].items():
                save_data[f"pio_marginal_kl_all_{entry_label(entry)}"] = arr
        # Benchmark timing data
        if "benchmark_timing" in metrics:
            for entry, data in metrics["benchmark_timing"].items():
                save_data[f"timing_sps_{entry_label(entry)}"] = np.array(data['sims_per_sec'])

    # Scaling report data
    if scaling:
        if "elo_per_doubling" in scaling:
            epd = scaling["elo_per_doubling"]
            save_data["scaling_elo_per_doubling_vc1"] = np.array([vc1 for vc1, _, _, _ in epd])
            save_data["scaling_elo_per_doubling_vc2"] = np.array([vc2 for _, vc2, _, _ in epd])
            save_data["scaling_elo_per_doubling_values"] = np.array([v for _, _, v, _ in epd])
            save_data["scaling_elo_per_doubling_modes"] = np.array([m for _, _, _, m in epd])
        if "elo_regressions" in scaling:
            for mode, (slope, r2) in scaling["elo_regressions"].items():
                suffix = f"_{mode}" if len(scaling["elo_regressions"]) > 1 else ""
                save_data[f"scaling_elo_slope{suffix}"] = np.array(slope)
                save_data[f"scaling_elo_r2{suffix}"] = np.array(r2)
        for key in ["policy_improvement", "capacity_score", "mcts_utilization",
                     "value_ece", "value_accuracy_gain"]:
            if key in scaling:
                for entry, val in scaling[key].items():
                    save_data[f"scaling_{key}_{entry_label(entry)}"] = np.array(val)

    # Value calibration data
    if metrics and "value_ece" in metrics:
        for entry, val in metrics["value_ece"].items():
            save_data[f"value_ece_{entry_label(entry)}"] = np.array(val)

    npz_path = os.path.join(save_dir, "mcts_results.npz")
    np.savez(npz_path, **save_data)
    print(f"Saved: {npz_path}")

    if _has_display():
        plt.show()
    else:
        print("No display detected, skipping interactive display")


def main():
    (config, Game, network_path, entries, anchor, phases, use_playout,
     cache_size, tree_reuse, experiment_dir) = interactive_config()

    agent = _load_agent(Game, network_path, use_playout)

    elo = None
    win_matrix = None
    metrics = None

    if "tournament" in phases:
        elo, win_matrix = run_tournament(config, Game, agent, entries,
                                         cache_size=cache_size)

    if "analysis" in phases:
        benchmark = run_benchmark(config, Game, agent, entries,
                                  cache_size=cache_size, tree_reuse=tree_reuse)
        metrics, snapshots = run_analysis(
            config, Game, agent, entries, anchor,
            cache_size=cache_size, tree_reuse=tree_reuse,
        )
        # Merge benchmark timing into metrics for visualization
        if benchmark:
            metrics["benchmark_timing"] = benchmark

    del agent
    gc.collect()

    save_dir = os.path.join(experiment_dir, "analysis")
    visualize_and_save(entries, anchor, elo=elo, win_matrix=win_matrix,
                       metrics=metrics, save_dir=save_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
