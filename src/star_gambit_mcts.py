#!/usr/bin/env python3
"""MCTS Threshold Analysis for Star Gambit.

Determines the optimal MCTS visit count for self-play and evaluation by:
1. Running an Elo tournament between different visit count configurations
2. Measuring policy convergence (KL divergence from max-visits policy)
3. Measuring value estimate convergence (correlation + MAE vs max-visits, and vs game outcomes)
"""

import os
import sys
import math
import gc
import numpy as np
import torch
import tqdm

import alphazero
import neural_net
from game_runner import (
    GameRunner, GRArgs, RandPlayer, base_params, elo_prob,
    CPUCT, FPU_REDUCTION, EVAL_TEMP, FINAL_TEMP, TEMP_DECAY_HALF_LIFE,
)
from monrad import pit_agents, calc_elo
from star_gambit_play import discover_networks, select_checkpoint_from_run

np.set_printoptions(precision=3, suppress=True)

# --- Constants ---
DEFAULT_VISIT_COUNTS = [1, 25, 50, 100, 200, 400, 800]
TOURNAMENT_BATCH_SIZE = 64
TOURNAMENT_CONCURRENT_BATCHES = 2
ANALYSIS_GAMES = 64


def calc_temp(turn):
    """Compute temperature with exponential decay (matches game_runner)."""
    ln2 = 0.693
    ld = ln2 / TEMP_DECAY_HALF_LIFE
    temp = EVAL_TEMP - FINAL_TEMP
    temp *= math.exp(-ld * turn)
    temp += FINAL_TEMP
    return temp


def kl_divergence(p, q):
    """KL(p || q) with epsilon smoothing."""
    eps = 1e-9
    result = 0.0
    for i in range(len(p)):
        if p[i] > eps:
            q_val = max(q[i], eps)
            result += p[i] * math.log(p[i] / q_val)
    return result


# --- Phase 1: Interactive Configuration ---

GAME_VARIANTS = {
    "1": ("Skirmish", alphazero.StarGambitSkirmishGS),
    "2": ("Clash", alphazero.StarGambitClashGS),
    "3": ("Battle", alphazero.StarGambitBattleGS),
}


def interactive_config():
    """Interactive configuration. Returns (Game, network_path, visit_counts, phases)."""
    print("=== MCTS Threshold Analysis for Star Gambit ===\n")

    # Game variant
    print("Select game variant:")
    for k, (name, _) in GAME_VARIANTS.items():
        print(f"  {k}. {name}")
    choice = input("Variant [1]: ").strip() or "1"
    if choice not in GAME_VARIANTS:
        print(f"Invalid choice '{choice}', using Skirmish")
        choice = "1"
    variant_name, Game = GAME_VARIANTS[choice]
    print(f"  -> {variant_name}\n")

    # Network selection
    variants = discover_networks()
    if not variants:
        print("No checkpoints found in data/checkpoint/")
        sys.exit(1)

    # Variant selection
    variant_names = list(variants.keys())
    if len(variant_names) == 1:
        selected_variant = variant_names[0]
        print(f"Variant: {selected_variant}")
    else:
        print("Available variants:")
        for i, name in enumerate(variant_names):
            runs = variants[name]
            total_cpts = sum(len(cpts) for cpts in runs.values())
            print(f"  {i + 1}. {name} ({len(runs)} run(s), {total_cpts} checkpoints)")
        while True:
            rc = input("\nSelect variant (number) [1]: ").strip()
            if rc == "":
                selected_variant = variant_names[0]
                break
            try:
                idx = int(rc) - 1
                if 0 <= idx < len(variant_names):
                    selected_variant = variant_names[idx]
                    break
                print(f"Enter 1-{len(variant_names)}")
            except ValueError:
                print("Invalid input")

    runs = variants[selected_variant]

    # Run selection
    run_names = sorted(runs.keys())
    if len(run_names) == 1:
        selected_run = run_names[0]
        print(f"Run: {selected_run}")
    else:
        print("Available runs:")
        for i, name in enumerate(run_names):
            cpts = runs[name]
            iter_range = f"{cpts[-1][0]:04d}-{cpts[0][0]:04d}"
            print(f"  {i + 1}. {name} ({len(cpts)} checkpoints: {iter_range})")
        while True:
            rc = input("\nSelect run (number) [1]: ").strip()
            if rc == "":
                selected_run = run_names[0]
                break
            try:
                idx = int(rc) - 1
                if 0 <= idx < len(run_names):
                    selected_run = run_names[idx]
                    break
                print(f"Enter 1-{len(run_names)}")
            except ValueError:
                print("Invalid input")

    checkpoints = runs[selected_run]
    network_path = select_checkpoint_from_run(checkpoints)
    if network_path is None:
        print("Random policy selected - analysis requires a network. Exiting.")
        sys.exit(1)
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

    return Game, network_path, visit_counts, phases


# --- Phase 2: Elo Tournament (Round-Robin) ---

def run_tournament(Game, network_path, visit_counts):
    """Run round-robin Elo tournament between visit count configs.

    Returns (elo_ratings, win_matrix) indexed by visit_counts order.
    """
    print("=" * 60)
    print("Phase: Elo Tournament (Round-Robin)")
    print("=" * 60)

    net_dir = os.path.dirname(network_path)
    net_file = os.path.basename(network_path)
    count = len(visit_counts)

    # Load a single network, reuse for all configs
    agent = neural_net.NNWrapper.load_checkpoint(Game, net_dir, net_file)

    win_matrix = np.full((count, count), np.nan)

    total_matchups = count * (count - 1) // 2
    with tqdm.tqdm(total=total_matchups, desc="Tournament") as pbar:
        for i in range(count):
            for j in range(i + 1, count):
                d1 = visit_counts[i]
                d2 = visit_counts[j]

                players = [agent] * Game.NUM_PLAYERS()
                depths = [d2] * Game.NUM_PLAYERS()
                players[0] = agent
                depths[0] = d1

                name = f"v{d1}-v{d2}"
                win_rates = pit_agents(
                    Game, players, depths,
                    TOURNAMENT_BATCH_SIZE, name,
                    tree_reuse=False,
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

    del agent
    gc.collect()
    return elo, win_matrix


# --- Phase 3: Policy & Value Analysis ---

def run_analysis(Game, network_path, visit_counts):
    """Run policy & value convergence analysis with batched inference.

    Plays ANALYSIS_GAMES concurrently, batching all NN inference calls across
    games for ~25x speedup over sequential single-sample inference.

    Returns dict with all collected metrics.
    """
    print("=" * 60)
    print("Phase: Policy & Value Analysis")
    print("=" * 60)

    net_dir = os.path.dirname(network_path)
    net_file = os.path.basename(network_path)
    agent = neural_net.NNWrapper.load_checkpoint(Game, net_dir, net_file)

    num_players = Game.NUM_PLAYERS()
    num_moves = Game.NUM_MOVES()
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
            alphazero.MCTS(CPUCT, num_players, num_moves, 0.0, 1.4, FPU_REDUCTION)
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
                temp = calc_temp(gs.current_turn())
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
    all_kl = {vc: [] for vc in visit_counts}
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

            # Policy KL divergence vs max visits
            if max_visits in policies_at_pos:
                max_policy = policies_at_pos[max_visits]
                for vc in visit_counts:
                    if vc in policies_at_pos:
                        kl = kl_divergence(max_policy, policies_at_pos[vc])
                        all_kl[vc].append(kl)

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

    # Policy KL divergence
    kl_means = {}
    kl_all = {}
    for vc in visit_counts:
        if vc != max_visits and vc in all_kl and len(all_kl[vc]) > 0:
            arr = np.array(all_kl[vc])
            kl_means[vc] = float(np.mean(arr))
            kl_all[vc] = arr
    metrics["kl_means"] = kl_means
    metrics["kl_all"] = kl_all

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
    print("\n--- Policy Analysis (KL divergence from max-visits policy) ---")
    print(f"{'Visits':>8s} {'Mean KL':>10s}")
    for vc in sorted(kl_means.keys()):
        print(f"{vc:>8d} {kl_means[vc]:>10.4f}")

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

    # Figure 2: Policy KL divergence
    if metrics and "kl_means" in metrics and metrics["kl_means"]:
        kl_means = metrics["kl_means"]
        kl_all = metrics["kl_all"]
        kl_vcs = sorted(kl_means.keys())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Mean line
        ax1.plot(kl_vcs, [kl_means[vc] for vc in kl_vcs], "o-", markersize=8, linewidth=2)
        ax1.set_xscale("log")
        ax1.set_xlabel("MCTS Visit Count")
        ax1.set_ylabel("KL Divergence")
        ax1.set_title("Mean Policy KL Divergence vs Max-Visits")
        ax1.set_xticks(kl_vcs)
        ax1.set_xticklabels([str(v) for v in kl_vcs])
        ax1.grid(True, alpha=0.3)

        # Box plot
        bp_data = [kl_all[vc] for vc in kl_vcs]
        bp = ax2.boxplot(bp_data, labels=[str(v) for v in kl_vcs],
                         showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        ax2.set_xlabel("MCTS Visit Count")
        ax2.set_ylabel("KL Divergence")
        ax2.set_title("Policy KL Divergence Distribution")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(save_dir, "policy_kl.png")
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
            bp = axes[panel].boxplot(bp_data, labels=[str(v) for v in regret_vcs],
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
        # KL data
        if "kl_means" in metrics:
            for vc, val in metrics["kl_means"].items():
                save_data[f"kl_mean_{vc}"] = np.array(val)
            for vc, arr in metrics.get("kl_all", {}).items():
                save_data[f"kl_all_{vc}"] = arr
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
    Game, network_path, visit_counts, phases = interactive_config()

    elo = None
    win_matrix = None
    metrics = None

    if "tournament" in phases:
        elo, win_matrix = run_tournament(Game, network_path, visit_counts)

    if "analysis" in phases:
        metrics = run_analysis(Game, network_path, visit_counts)

    visualize_and_save(visit_counts, elo=elo, win_matrix=win_matrix, metrics=metrics)

    print("\nDone!")


if __name__ == "__main__":
    main()
