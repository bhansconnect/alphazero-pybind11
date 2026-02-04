#!/usr/bin/env python3
"""Monrad tournament system for Star Gambit networks.

This script runs a Swiss-style (Monrad) tournament between trained networks
to establish ELO rankings.

Usage:
    python src/star_gambit_monrad.py

Networks are loaded from data/bench/ (if exists) or data/checkpoint/.
Results are saved to data/monrad_wr.csv.
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

# === Star Gambit Configuration ===
Game = alphazero.StarGambitSkirmishGS
BATCH_SIZE = 64
NN_MCTS_DEPTH = 200


def main():
    """Run Monrad tournament for Star Gambit networks."""
    # Determine model path
    model_path = os.path.join("data", "checkpoint")
    if os.path.isdir(os.path.join("data", "bench")):
        model_path = os.path.join("data", "bench")

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
    print(f"Game: StarGambitSkirmishGS")
    print(f"MCTS depth: {NN_MCTS_DEPTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Agents ({count}): {agents}")
    print()

    win_matrix = np.full((count, count), np.nan)
    elo = np.zeros(count)
    rankings = list(range(count))
    rounds = int(np.ceil(np.log2(count)))
    dist = count

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
    np.savetxt(
        os.path.join("data", "star_gambit_monrad_wr.csv"),
        win_matrix,
        delimiter=",",
        header=",".join([str(a) for a in agents]),
    )
    print(f"\nResults saved to data/star_gambit_monrad_wr.csv")


if __name__ == "__main__":
    main()
