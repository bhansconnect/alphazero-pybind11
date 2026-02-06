#!/usr/bin/env python3
"""Monrad tournament system for Star Gambit networks.

This script runs a Swiss-style (Monrad) tournament between trained networks
to establish ELO rankings. Supports all three variants: Skirmish, Clash, and Battle.
Also supports unified (multi-size) networks with flexible variant selection.

Usage:
    python src/star_gambit_monrad.py

Networks are loaded from data/bench/{variant}/ (if exists) or data/checkpoint/{variant}/.
Results are saved to data/star_gambit_{variant}_monrad_wr.csv.
"""

import os
import sys

# Ensure we can import from the src directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero

# Import shared functions from monrad.py
from monrad import calc_elo, pit_agents

import neural_net
from game_runner import RandPlayer
import glob
import math
import numpy as np
import tqdm

np.set_printoptions(precision=3, suppress=True)

# === Game Variants ===
VARIANTS = {
    '1': ('skirmish', alphazero.StarGambitSkirmishGS, '3F, 1C, 0D'),
    '2': ('clash', alphazero.StarGambitClashGS, '3F, 2C, 1D'),
    '3': ('battle', alphazero.StarGambitBattleGS, '4F, 3C, 2D'),
    '4': ('unified', alphazero.StarGambitUnifiedGS, 'Multi-size network'),
}

# Variant enum mappings for unified mode
VARIANT_ENUMS = {
    'skirmish': alphazero.StarGambitVariant.SKIRMISH,
    'clash': alphazero.StarGambitVariant.CLASH,
    'battle': alphazero.StarGambitVariant.BATTLE,
}

# === Star Gambit Configuration ===
BATCH_SIZE = 64
NN_MCTS_DEPTH = 200


def select_variant():
    """Prompt user to select a game variant.

    Returns:
        (variant_name, Game class, selected_variants)
        where selected_variants is a list of variant names for unified mode,
        or None for single-size mode.
    """
    print("Select Star Gambit variant:")
    for key, (name, _, desc) in VARIANTS.items():
        print(f"  {key}. {name.capitalize()} ({desc})")

    variant_input = input("Variant (1/2/3/4) [1]: ").strip() or '1'
    if variant_input not in VARIANTS:
        print(f"Invalid selection '{variant_input}', defaulting to Skirmish")
        variant_input = '1'

    variant_name, Game, _ = VARIANTS[variant_input]

    # For unified mode, allow selecting which variants to include
    if variant_name == 'unified':
        print("\nSelect game sizes for tournament:")
        print("  1. Skirmish only")
        print("  2. Clash only")
        print("  3. Battle only")
        print("  4. Skirmish + Clash")
        print("  5. Skirmish + Battle")
        print("  6. Clash + Battle")
        print("  7. All sizes (Recommended)")

        size_input = input("Selection (1-7) [7]: ").strip() or '7'

        size_selections = {
            '1': ['skirmish'],
            '2': ['clash'],
            '3': ['battle'],
            '4': ['skirmish', 'clash'],
            '5': ['skirmish', 'battle'],
            '6': ['clash', 'battle'],
            '7': ['skirmish', 'clash', 'battle'],
        }

        selected_variants = size_selections.get(size_input, ['skirmish', 'clash', 'battle'])
        print(f"Selected: Unified ({', '.join(v.capitalize() for v in selected_variants)})\n")
        return variant_name, Game, selected_variants

    print(f"Selected: {variant_name.capitalize()}\n")
    return variant_name, Game, None


def create_unified_game(variant_name):
    """Create a unified game for the specified variant."""
    variant_enum = VARIANT_ENUMS[variant_name]
    return alphazero.StarGambitUnifiedGS(variant_enum)


def pit_agents_unified(players, depths, batch_size, selected_variants, desc=""):
    """Pit agents across multiple variants for unified networks.

    Plays equal number of games for each variant and returns combined win rates.
    """
    Game = alphazero.StarGambitUnifiedGS
    total_wins = np.zeros(2)
    total_draws = 0
    total_games = 0

    for variant_name in selected_variants:
        # Create game factory for this variant
        variant_enum = VARIANT_ENUMS[variant_name]

        def new_game():
            return alphazero.StarGambitUnifiedGS(variant_enum)

        # Run games for this variant
        variant_desc = f"{desc} ({variant_name})" if desc else variant_name
        win_rates = pit_agents(
            Game, players, depths, batch_size, variant_desc,
            new_game_fn=new_game
        )

        total_wins += win_rates[:2] * batch_size
        total_draws += win_rates[2] * batch_size if len(win_rates) > 2 else 0
        total_games += batch_size

    # Return averaged win rates
    return total_wins / total_games


def main():
    """Run Monrad tournament for Star Gambit networks."""
    # Select game variant
    variant_name, Game, selected_variants = select_variant()

    # Determine model path (variant-specific subdirectories)
    model_path = os.path.join("data", "checkpoint", variant_name)
    bench_path = os.path.join("data", "bench", variant_name)
    if os.path.isdir(bench_path):
        model_path = bench_path

    # Load agents
    nn_agents = [
        os.path.basename(x)
        for x in sorted(glob.glob(os.path.join(model_path, "*.pt")), reverse=False)
    ]
    rand_agents = []  # Add random agents if desired, e.g., [5000]
    agents = rand_agents + nn_agents

    if len(agents) < 2:
        print("Need at least 2 agents for a tournament")
        print(f"Found {len(agents)} agents in {model_path}")
        return

    if len(agents) % 2 == 1:
        agents.insert(0, "dummy")

    count = len(agents)
    print(f"=== Star Gambit Monrad Tournament ===")
    print(f"Game: {variant_name.capitalize()} ({Game.__name__})")
    if selected_variants:
        print(f"Variants: {', '.join(v.capitalize() for v in selected_variants)}")
    print(f"MCTS depth: {NN_MCTS_DEPTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Model path: {model_path}")
    print(f"Agents ({count}): {agents}")
    print()

    win_matrix = np.full((count, count), np.nan)
    elo = np.zeros(count)
    rankings = list(range(count))
    rounds = int(np.ceil(np.log2(count)))
    dist = count

    # Track per-variant stats for unified mode
    if selected_variants:
        per_variant_wins = {v: np.full((count, count), np.nan) for v in selected_variants}

    for r in range(rounds):
        print(f"Round {r + 1}/{rounds}")
        dist = math.ceil(dist / 2)

        with tqdm.trange(count // 2, desc="Games") as pbar:
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
                    print("Falling back to shorter distance games")
                    offset = 1
                    while current - offset >= 0 and (
                        played[rankings[current - offset]]
                        or not math.isnan(
                            win_matrix[rankings[current], rankings[current - offset]]
                        )
                    ):
                        offset += 1
                    if current - offset < 0:
                        print("No one to play? Relaxing constraints")
                        offset = 1
                        while not math.isnan(
                            win_matrix[rankings[current], rankings[current - offset]]
                        ):
                            offset += 1

                played[rankings[current - offset]] = True

                i = rankings[current]
                j = rankings[current - offset]

                # Handle dummy player
                if agents[i] == "dummy":
                    win_matrix[i, j] = 0.0
                    win_matrix[j, i] = 1.0
                    current -= 1
                    continue
                elif agents[j] == "dummy":
                    win_matrix[i, j] = 1.0
                    win_matrix[j, i] = 0.0
                    current -= 1
                    continue

                # Load players
                if agents[i] in rand_agents:
                    p1 = RandPlayer(Game, BATCH_SIZE)
                    d1 = agents[i]
                else:
                    p1 = neural_net.NNWrapper.load_checkpoint(
                        Game, model_path, agents[i]
                    )
                    d1 = NN_MCTS_DEPTH

                if agents[j] in rand_agents:
                    p2 = RandPlayer(Game, BATCH_SIZE)
                    d2 = agents[j]
                else:
                    p2 = neural_net.NNWrapper.load_checkpoint(
                        Game, model_path, agents[j]
                    )
                    d2 = NN_MCTS_DEPTH

                # Pit agents
                players = [p2] * Game.NUM_PLAYERS()
                depths = [d2] * Game.NUM_PLAYERS()
                players[0] = p1
                depths[0] = d1

                if selected_variants:
                    # Unified mode: play across all selected variants
                    win_rates = pit_agents_unified(
                        players, depths, BATCH_SIZE, selected_variants,
                        f"{agents[i]}-{agents[j]}"
                    )
                else:
                    # Single-size mode
                    win_rates = pit_agents(
                        Game, players, depths, BATCH_SIZE, f"{agents[i]}-{agents[j]}"
                    )

                win_matrix[i, j] = win_rates[0]
                win_matrix[j, i] = win_rates[1]
                pbar.update()
                current -= 1

        # Update ELO and rankings
        elo = calc_elo(elo, win_matrix)
        rankings = list(np.argsort(elo))
        print(f"ELO: {np.array2string(elo, precision=0)}")
        print(f"Rankings: {rankings}")

    # Final results
    print()
    print("=== Final Results ===")
    print("Win matrix:")
    print(win_matrix)
    print()
    print("Agents:", agents)
    print("ELO:", elo)
    print("Rankings (worst to best):", rankings)
    print()

    # Sort agents by ELO for display
    sorted_agents = [(agents[i], elo[i]) for i in rankings]
    print("Leaderboard:")
    for rank, (agent, rating) in enumerate(reversed(sorted_agents)):
        print(f"  {rank+1}. {agent}: {rating:.0f}")

    # Save results
    output_file = os.path.join("data", f"star_gambit_{variant_name}_monrad_wr.csv")
    np.savetxt(
        output_file,
        win_matrix,
        delimiter=",",
        header=",".join([str(a) for a in agents]),
    )
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
