#!/usr/bin/env python3
"""MCTS Threshold Analysis -- game-generic.

Determines the optimal MCTS visit count for self-play and evaluation by:
1. Running an Elo tournament between different visit count configurations
2. Measuring policy convergence (JSD, TV, Hellinger, top-k agreement vs max-visits policy)
3. Measuring value estimate convergence (correlation + MAE vs max-visits, and vs game outcomes)
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
from tournament import pit_agents, calc_elo

np.set_printoptions(precision=3, suppress=True)

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")

# --- Constants ---
DEFAULT_VISIT_COUNTS = [1, 25, 50, 100, 200, 400, 800]
TOURNAMENT_BATCH_SIZE = 64
TOURNAMENT_CONCURRENT_BATCHES = 2
ANALYSIS_GAMES = 64


def calc_temp(config, turn):
    """Compute temperature with exponential decay (matches game_runner)."""
    ln2 = 0.693
    ld = ln2 / config.temp_decay_half_life
    temp = config.eval_temp - config.final_temp
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
    """Interactive configuration. Returns (config, Game, network_path, visit_counts, phases, use_playout)."""
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

    # Visit counts
    print(f"Default visit counts: {DEFAULT_VISIT_COUNTS}")
    print("  (1 = raw network policy, no MCTS search)")
    vc_input = input("Enter comma-separated visit counts or press Enter for defaults: ").strip()
    if vc_input:
        visit_counts = sorted(set(int(x.strip()) for x in vc_input.split(",")))
    else:
        visit_counts = list(DEFAULT_VISIT_COUNTS)
    print(f"  -> {visit_counts}\n")

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

    return config, Game, network_path, visit_counts, phases, use_playout


# --- Phase 2: Elo Tournament (Round-Robin) ---

def run_tournament(config, Game, network_path, visit_counts, use_playout=False):
    """Run round-robin Elo tournament between visit count configs.

    Returns (elo_ratings, win_matrix) indexed by visit_counts order.
    """
    print("=" * 60)
    print("Phase: Elo Tournament (Round-Robin)")
    print("=" * 60)

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

    total_matchups = count * (count - 1) // 2
    with tqdm.tqdm(total=total_matchups, desc="Tournament") as pbar:
        for i in range(count):
            for j in range(i + 1, count):
                d1 = visit_counts[i]
                d2 = visit_counts[j]

                players = make_players()
                depths = [d2] * num_players
                depths[0] = d1

                name = f"v{d1}-v{d2}"
                win_rates = pit_agents(
                    config, Game, players, depths,
                    TOURNAMENT_BATCH_SIZE, name,
                )
                win_matrix[i, j] = win_rates[0]
                win_matrix[j, i] = win_rates[1]
                pbar.update()
                pbar.set_postfix_str(f"{name}: {win_rates[0]:.3f}-{win_rates[1]:.3f}")

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

def run_analysis(config, Game, network_path, visit_counts, use_playout=False):
    """Run policy & value convergence analysis with batched inference.

    Plays ANALYSIS_GAMES concurrently, batching all NN inference calls across
    games for ~25x speedup over sequential single-sample inference.

    Returns dict with all collected metrics.
    """
    print("=" * 60)
    print("Phase: Policy & Value Analysis")
    print("=" * 60)

    if network_path is None and use_playout:
        agent = "playout"  # Will use playout_eval per-leaf below
    elif network_path is None:
        agent = "random"  # Will use dumb_eval per-leaf below
    else:
        net_dir = os.path.dirname(network_path)
        net_file = os.path.basename(network_path)
        agent = neural_net.NNWrapper.load_checkpoint(Game, net_dir, net_file)
        agent.enable_inference_optimizations()

    num_players = Game.NUM_PLAYERS()
    num_moves = Game.NUM_MOVES()
    relative_values = Game().relative_values()
    max_visits = max(visit_counts)
    target_vcs = sorted(vc for vc in visit_counts if vc > 1)
    has_vc1 = 1 in visit_counts

    # Per-game state
    game_states = [Game() for _ in range(ANALYSIS_GAMES)]
    # Per-game accumulated data: list of (current_player, {vc: value}, {vc: policy})
    game_positions = [[] for _ in range(ANALYSIS_GAMES)]
    active = list(range(ANALYSIS_GAMES))
    game_scores = [None] * ANALYSIS_GAMES  # final scores per game

    print(f"Playing {ANALYSIS_GAMES} games with max {max_visits} visits per position...")

    total_positions = 0
    pbar = tqdm.tqdm(desc="Positions", unit="pos")

    while active:
        n = len(active)

        # Create fresh MCTS for each active game's current position
        mcts_list = [
            alphazero.MCTS(config.cpuct, num_players, num_moves, 0.0, 1.4, config.fpu_reduction,
                           relative_values)
            for _ in range(n)
        ]

        # Per-position snapshot storage (indexed by slot in active list)
        pos_policies = [{} for _ in range(n)]
        pos_values = [{} for _ in range(n)]
        pos_q_values = [None] * n

        # Run MCTS simulations in lockstep
        for sim in range(max_visits):
            # find_leaf for each active game
            leaves = [
                mcts_list[i].find_leaf(game_states[active[i]])
                for i in range(n)
            ]

            # Evaluate leaves
            if agent == "playout":
                v_np, pi_np = alphazero.playout_eval_batch(leaves)
            elif agent == "random":
                # Random evaluation: uniform policy, equal value
                np_ = num_players
                v_np = np.full((n, np_ + 1), 1.0 / (np_ + 1), dtype=np.float32)
                pi_np = np.full((n, num_moves), 1.0 / num_moves, dtype=np.float32)
            else:
                # Stack canonicals into batch tensor
                canonicals = [np.array(leaf.canonicalized()) for leaf in leaves]
                batch = torch.from_numpy(np.stack(canonicals))

                # Single batched GPU inference
                v_batch, pi_batch = agent.process(batch)
                v_np = v_batch.cpu().numpy()
                pi_np = pi_batch.cpu().numpy()

            # process_result for each active game
            for i in range(n):
                mcts_list[i].process_result(
                    game_states[active[i]], v_np[i], pi_np[i], False
                )

            sims_done = sim + 1

            # Capture vc=1 snapshot from the first simulation
            if sims_done == 1 and has_vc1:
                for i in range(n):
                    # probs(1.0) after 1 sim returns the prior policy
                    probs_arr = np.array(mcts_list[i].probs(1.0))
                    pos_policies[i][1] = probs_arr
                    # Value from NN output of sim 1 (root was the leaf)
                    pos_values[i][1] = float(v_np[i][0])

            # Check if we hit any target visit count
            for vc in target_vcs:
                if sims_done == vc:
                    for i in range(n):
                        pos_policies[i][vc] = np.array(mcts_list[i].probs(1.0))
                        wld = np.array(mcts_list[i].root_value())
                        pos_values[i][vc] = float(wld[0])

        # Capture Q-values from the max-visits MCTS tree
        for i in range(n):
            pos_q_values[i] = np.array(mcts_list[i].root_q_values())

        # Record position data and advance games
        next_active = []
        for slot, gid in enumerate(active):
            gs = game_states[gid]
            game_positions[gid].append(
                (gs.current_player(), pos_values[slot], pos_policies[slot], pos_q_values[slot])
            )
            total_positions += 1
            pbar.update(1)

            # Select move from max-visits policy with temperature
            if max_visits in pos_policies[slot]:
                probs = pos_policies[slot][max_visits].copy()
                temp = calc_temp(config, gs.current_turn())
                if temp > 0 and temp != 1.0:
                    probs = np.power(probs + 1e-10, 1.0 / temp)
                    probs /= probs.sum()
                move = np.random.choice(len(probs), p=probs)
            else:
                valids = np.array(gs.valid_moves())
                valid_indices = np.where(valids == 1)[0]
                move = np.random.choice(valid_indices)

            gs.play_move(move)

            if gs.scores() is not None:
                game_scores[gid] = np.array(gs.scores())
            else:
                next_active.append(gid)

        active = next_active

    pbar.close()
    print(f"Collected {total_positions} positions across {ANALYSIS_GAMES} games")

    # Aggregate metrics from all games
    all_jsd = {vc: [] for vc in visit_counts}
    all_tv = {vc: [] for vc in visit_counts}
    all_hellinger = {vc: [] for vc in visit_counts}
    all_top1 = {vc: [] for vc in visit_counts}
    all_top3 = {vc: [] for vc in visit_counts}
    all_top9 = {vc: [] for vc in visit_counts}
    position_values = {vc: [] for vc in visit_counts}
    position_max_values = []
    position_outcomes = []
    # Expected reward: V(pi_vc) = sum(pi_vc * Q_max)
    expected_reward = {vc: [] for vc in visit_counts}

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

            # Policy divergence metrics vs max visits
            if max_visits in policies_at_pos:
                max_policy = policies_at_pos[max_visits]
                for vc in visit_counts:
                    if vc in policies_at_pos:
                        q = policies_at_pos[vc]
                        all_jsd[vc].append(jensen_shannon_divergence(max_policy, q))
                        all_tv[vc].append(total_variation(max_policy, q))
                        all_hellinger[vc].append(hellinger_distance(max_policy, q))
                        all_top1[vc].append(top_k_agreement(max_policy, q, 1))
                        all_top3[vc].append(top_k_agreement(max_policy, q, 3))
                        all_top9[vc].append(top_k_agreement(max_policy, q, 9))

            # Store values
            for vc in visit_counts:
                if vc in values_at_pos:
                    position_values[vc].append(values_at_pos[vc])
            if max_visits in values_at_pos:
                position_max_values.append(values_at_pos[max_visits])

            # Expected reward using Q-values from max-visits tree
            for vc in visit_counts:
                if vc in policies_at_pos:
                    v_pi = float(np.sum(policies_at_pos[vc] * q_values))
                    expected_reward[vc].append(v_pi)

    position_outcomes = np.array(position_outcomes)
    position_max_values = np.array(position_max_values)

    metrics = {
        "visit_counts": visit_counts,
        "max_visits": max_visits,
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
        for vc in visit_counts:
            data = policy_metric_data[name]
            if vc != max_visits and vc in data and len(data[vc]) > 0:
                arr = np.array(data[vc])
                means[vc] = float(np.mean(arr))
                all_vals[vc] = arr
        metrics[f"{name}_means"] = means
        metrics[f"{name}_all"] = all_vals

    # Value metrics
    value_corr_vs_max = {}
    value_mae_vs_max = {}
    value_corr_vs_outcome = {}
    value_mae_vs_outcome = {}

    for vc in visit_counts:
        vals = np.array(position_values[vc])
        n = min(len(vals), len(position_max_values), len(position_outcomes))
        if n < 3:
            continue
        vals = vals[:n]
        max_vals = position_max_values[:n]
        outcomes = position_outcomes[:n]

        # vs max-visits
        corr = np.corrcoef(vals, max_vals)[0, 1] if np.std(vals) > 1e-9 and np.std(max_vals) > 1e-9 else 0.0
        mae = float(np.mean(np.abs(vals - max_vals)))
        value_corr_vs_max[vc] = float(corr)
        value_mae_vs_max[vc] = mae

        # vs game outcome
        corr = np.corrcoef(vals, outcomes)[0, 1] if np.std(vals) > 1e-9 and np.std(outcomes) > 1e-9 else 0.0
        mae = float(np.mean(np.abs(vals - outcomes)))
        value_corr_vs_outcome[vc] = float(corr)
        value_mae_vs_outcome[vc] = mae

    metrics["value_corr_vs_max"] = value_corr_vs_max
    metrics["value_mae_vs_max"] = value_mae_vs_max
    metrics["value_corr_vs_outcome"] = value_corr_vs_outcome
    metrics["value_mae_vs_outcome"] = value_mae_vs_outcome

    # Expected reward and regret metrics
    reward_means = {}
    reward_all = {}
    regret_means = {}
    regret_all = {}
    for vc in visit_counts:
        if vc in expected_reward and len(expected_reward[vc]) > 0:
            arr = np.array(expected_reward[vc])
            reward_means[vc] = float(np.mean(arr))
            reward_all[vc] = arr
    # Regret = V(pi_max) - V(pi_vc)
    if max_visits in reward_all:
        max_reward = reward_all[max_visits]
        for vc in visit_counts:
            if vc in reward_all:
                reg = max_reward[:len(reward_all[vc])] - reward_all[vc]
                regret_means[vc] = float(np.mean(reg))
                regret_all[vc] = reg
    metrics["reward_means"] = reward_means
    metrics["reward_all"] = reward_all
    metrics["regret_means"] = regret_means
    metrics["regret_all"] = regret_all

    # Print summary
    print("\n--- Policy Analysis (divergence from max-visits policy) ---")
    print(f"{'Visits':>8s} {'JSD':>10s} {'TV':>10s} {'Hellinger':>10s} {'Top-1':>10s} {'Top-3':>10s} {'Top-9':>10s}")
    all_policy_vcs = sorted(set().union(*(metrics[f"{n}_means"].keys() for n in policy_metric_names)))
    for vc in all_policy_vcs:
        vals = []
        for name in policy_metric_names:
            v = metrics[f"{name}_means"].get(vc)
            vals.append(f"{v:>10.4f}" if v is not None else f"{'N/A':>10s}")
        print(f"{vc:>8d} {''.join(vals)}")

    print("\n--- Value Analysis vs Max-Visits ---")
    print(f"{'Visits':>8s} {'Corr':>8s} {'MAE':>8s}")
    for vc in sorted(value_corr_vs_max.keys()):
        print(f"{vc:>8d} {value_corr_vs_max[vc]:>8.4f} {value_mae_vs_max[vc]:>8.4f}")

    print("\n--- Value Analysis vs Game Outcome ---")
    print(f"{'Visits':>8s} {'Corr':>8s} {'MAE':>8s}")
    for vc in sorted(value_corr_vs_outcome.keys()):
        print(f"{vc:>8d} {value_corr_vs_outcome[vc]:>8.4f} {value_mae_vs_outcome[vc]:>8.4f}")

    if reward_means:
        print("\n--- Expected Reward (V(pi) = sum(pi * Q_max)) ---")
        print(f"{'Visits':>8s} {'E[reward]':>10s}")
        for vc in sorted(reward_means.keys()):
            print(f"{vc:>8d} {reward_means[vc]:>10.4f}")

    if regret_means:
        print("\n--- Policy Regret (V(pi_max) - V(pi_vc)) ---")
        print(f"{'Visits':>8s} {'Regret':>10s}")
        for vc in sorted(regret_means.keys()):
            print(f"{vc:>8d} {regret_means[vc]:>10.4f}")

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


def visualize_and_save(visit_counts, elo=None, win_matrix=None, metrics=None):
    """Generate plots and save raw data."""
    if not _has_display():
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    save_dir = os.path.join("data", "threshold_analysis")
    os.makedirs(save_dir, exist_ok=True)

    fig_num = 1

    # Figure 1: Elo vs visit count
    if elo is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(visit_counts, elo, "o-", markersize=8, linewidth=2)
        for i, vc in enumerate(visit_counts):
            ax.annotate(f"{elo[i]:.0f}", (vc, elo[i]),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("MCTS Visit Count")
        ax.set_ylabel("Elo Rating")
        ax.set_title("Elo vs MCTS Visit Count")
        ax.set_xticks(visit_counts)
        ax.set_xticklabels([str(v) for v in visit_counts])
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

        for idx, (name, title, color) in enumerate(policy_metric_info):
            ax = axes[idx]
            means = metrics.get(f"{name}_means", {})
            all_data = metrics.get(f"{name}_all", {})
            if not means:
                ax.set_visible(False)
                continue

            vcs = sorted(means.keys())
            labels = [str(v) for v in vcs]

            # Box plot with mean overlay
            if all_data:
                bp_data = [all_data[vc] for vc in vcs]
                bp = ax.boxplot(bp_data, tick_labels=labels,
                                showfliers=False, patch_artist=True)
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                # Overlay mean line
                ax.plot(range(1, len(vcs) + 1), [means[vc] for vc in vcs],
                        "D-", markersize=6, linewidth=2, color="tab:blue",
                        zorder=3, label="mean")
                ax.legend(fontsize=8)
            else:
                ax.plot(range(1, len(vcs) + 1), [means[vc] for vc in vcs],
                        "o-", markersize=8, linewidth=2)
                ax.set_xticks(range(1, len(vcs) + 1))
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

        # Top-left: correlation vs max-visits
        if corr_max:
            vcs = sorted(corr_max.keys())
            axes[0, 0].plot(vcs, [corr_max[vc] for vc in vcs], "o-", markersize=8, linewidth=2)
            axes[0, 0].set_xscale("log")
            axes[0, 0].set_xticks(vcs)
            axes[0, 0].set_xticklabels([str(v) for v in vcs])
        axes[0, 0].set_xlabel("MCTS Visit Count")
        axes[0, 0].set_ylabel("Pearson Correlation")
        axes[0, 0].set_title("Value Correlation vs Max-Visits")
        axes[0, 0].grid(True, alpha=0.3)

        # Top-right: MAE vs max-visits
        if mae_max:
            vcs = sorted(mae_max.keys())
            axes[0, 1].plot(vcs, [mae_max[vc] for vc in vcs], "o-", markersize=8, linewidth=2, color="tab:orange")
            axes[0, 1].set_xscale("log")
            axes[0, 1].set_xticks(vcs)
            axes[0, 1].set_xticklabels([str(v) for v in vcs])
        axes[0, 1].set_xlabel("MCTS Visit Count")
        axes[0, 1].set_ylabel("MAE")
        axes[0, 1].set_title("Value MAE vs Max-Visits")
        axes[0, 1].grid(True, alpha=0.3)

        # Bottom-left: correlation vs game outcome
        if corr_out:
            vcs = sorted(corr_out.keys())
            axes[1, 0].plot(vcs, [corr_out[vc] for vc in vcs], "o-", markersize=8, linewidth=2, color="tab:green")
            axes[1, 0].set_xscale("log")
            axes[1, 0].set_xticks(vcs)
            axes[1, 0].set_xticklabels([str(v) for v in vcs])
        axes[1, 0].set_xlabel("MCTS Visit Count")
        axes[1, 0].set_ylabel("Pearson Correlation")
        axes[1, 0].set_title("Value Correlation vs Game Outcome")
        axes[1, 0].grid(True, alpha=0.3)

        # Bottom-right: MAE vs game outcome
        if mae_out:
            vcs = sorted(mae_out.keys())
            axes[1, 1].plot(vcs, [mae_out[vc] for vc in vcs], "o-", markersize=8, linewidth=2, color="tab:red")
            axes[1, 1].set_xscale("log")
            axes[1, 1].set_xticks(vcs)
            axes[1, 1].set_xticklabels([str(v) for v in vcs])
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
        reward_vcs = sorted(reward_means.keys())
        axes[panel].plot(reward_vcs, [reward_means[vc] for vc in reward_vcs],
                         "o-", markersize=8, linewidth=2)
        axes[panel].set_xscale("log")
        axes[panel].set_xlabel("MCTS Visit Count")
        axes[panel].set_ylabel("Expected Reward V(pi)")
        axes[panel].set_title("Expected Reward vs Visit Count")
        axes[panel].set_xticks(reward_vcs)
        axes[panel].set_xticklabels([str(v) for v in reward_vcs])
        axes[panel].grid(True, alpha=0.3)
        panel += 1

        # Center: mean regret curve
        if regret_means:
            regret_vcs = sorted(regret_means.keys())
            axes[panel].plot(regret_vcs, [regret_means[vc] for vc in regret_vcs],
                             "o-", markersize=8, linewidth=2, color="tab:red")
            axes[panel].set_xscale("log")
            axes[panel].set_xlabel("MCTS Visit Count")
            axes[panel].set_ylabel("Regret")
            axes[panel].set_title("Mean Policy Regret vs Visit Count")
            axes[panel].set_xticks(regret_vcs)
            axes[panel].set_xticklabels([str(v) for v in regret_vcs])
            axes[panel].grid(True, alpha=0.3)
            panel += 1

        # Right: regret distribution box plot
        if regret_all:
            regret_vcs = sorted(regret_all.keys())
            bp_data = [regret_all[vc] for vc in regret_vcs]
            bp = axes[panel].boxplot(bp_data, tick_labels=[str(v) for v in regret_vcs],
                                     showfliers=False, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightsalmon")
            axes[panel].set_xlabel("MCTS Visit Count")
            axes[panel].set_ylabel("Regret")
            axes[panel].set_title("Policy Regret Distribution")
            axes[panel].grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(save_dir, "policy_regret.png")
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")

    # Save raw data
    save_data = {"visit_counts": np.array(visit_counts)}
    if elo is not None:
        save_data["elo"] = elo
    if win_matrix is not None:
        save_data["win_matrix"] = win_matrix
    if metrics:
        for key in ["total_positions", "num_games", "max_visits"]:
            if key in metrics:
                save_data[key] = np.array(metrics[key])
        # Policy divergence metrics
        for prefix in ["jsd", "tv", "hellinger", "top1", "top3", "top9"]:
            if f"{prefix}_means" in metrics:
                for vc, val in metrics[f"{prefix}_means"].items():
                    save_data[f"{prefix}_mean_{vc}"] = np.array(val)
            if f"{prefix}_all" in metrics:
                for vc, arr in metrics[f"{prefix}_all"].items():
                    save_data[f"{prefix}_all_{vc}"] = arr
        # Value data
        for prefix in ["value_corr_vs_max", "value_mae_vs_max",
                        "value_corr_vs_outcome", "value_mae_vs_outcome"]:
            if prefix in metrics:
                for vc, val in metrics[prefix].items():
                    save_data[f"{prefix}_{vc}"] = np.array(val)
        # Expected reward and regret data
        for prefix in ["reward_means", "regret_means"]:
            if prefix in metrics:
                for vc, val in metrics[prefix].items():
                    save_data[f"{prefix}_{vc}"] = np.array(val)
        for prefix in ["reward_all", "regret_all"]:
            if prefix in metrics:
                for vc, arr in metrics[prefix].items():
                    save_data[f"{prefix}_{vc}"] = arr

    npz_path = os.path.join(save_dir, "threshold_data.npz")
    np.savez(npz_path, **save_data)
    print(f"Saved: {npz_path}")

    if _has_display():
        plt.show()
    else:
        print("No display detected, skipping interactive display")


def main():
    config, Game, network_path, visit_counts, phases, use_playout = interactive_config()

    elo = None
    win_matrix = None
    metrics = None

    if "tournament" in phases:
        elo, win_matrix = run_tournament(config, Game, network_path, visit_counts, use_playout)

    if "analysis" in phases:
        metrics = run_analysis(config, Game, network_path, visit_counts, use_playout)

    visualize_and_save(visit_counts, elo=elo, win_matrix=win_matrix, metrics=metrics)

    print("\nDone!")


if __name__ == "__main__":
    main()
