#!/usr/bin/env python3
"""Unified tournament runner supporting monrad and round-robin formats.

Usage:
    # Interactive mode (fallback to network_selector)
    python src/tournament.py --game connect4 --format monrad

    # Directory-based selection
    python src/tournament.py --game star_gambit_skirmish --format monrad \
      --dir data/star_gambit_skirmish/densenet-4d-16c-3k-300sims --count 10

    # Multiple experiments + special agents
    python src/tournament.py --game connect4 --format roundrobin \
      --dir data/connect4/exp1 --count 5 \
      --dir data/connect4/exp2 --count 3 \
      --random --playout --mcts_visits 200 --batch_size 64
"""

import argparse
import gc
import glob
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tqdm

import alphazero
import neural_net
from config import TrainConfig, GAME_REGISTRY, load_config
from game_runner import (
    GameRunner,
    GRArgs,
    RandPlayer,
    PlayoutPlayer,
    base_params,
    elo_prob,
    set_eval_types,
    set_model_groups,
)
from network_selector import (
    discover_runs,
    interactive_select,
    create_tournament_dir,
    save_tournament_results,
)
from tracy_utils import tracy_zone

np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------------------
# ELO calculation
# ---------------------------------------------------------------------------


@tracy_zone
def calc_elo(past_elo, win_rates):
    """Calculate ELO ratings from win rate matrix."""
    iters = 5000
    anchor = -1200
    anchor_wr = elo_prob(anchor, 0)
    for _ in tqdm.trange(iters, leave=False):
        mean_update = 0
        for i in range(past_elo.shape[0]):
            for j in range(past_elo.shape[0]):
                if not math.isnan(win_rates[i, j]):
                    rate = win_rates[i, j]
                    rate = max(0.001, rate)
                    rate = min(0.999, rate)
                    mean_update += rate - elo_prob(past_elo[j], past_elo[i])
            mean_update += anchor_wr - elo_prob(anchor, past_elo[i])
            past_elo[i] += mean_update * 32
        anchor_update = 0
        for i in range(past_elo.shape[0]):
            anchor_update += (1 - anchor_wr) - elo_prob(past_elo[i], anchor)
        anchor += anchor_update * 32
    past_elo -= min(past_elo)
    return past_elo


# ---------------------------------------------------------------------------
# Agent pitting
# ---------------------------------------------------------------------------


@tracy_zone
def pit_agents(config, Game, players, mcts_visits, bs, name, tree_reuse=True, cache_size=0):
    """Play agents against each other across all seat permutations in one run.

    Returns list of win rates per player index.
    """
    num_players = Game.NUM_PLAYERS()
    cb = num_players

    # Build model groups from player identity
    params = base_params(config, 0.5, bs, cb)
    params.max_cache_size = cache_size
    params.tree_reuse = tree_reuse
    params.mcts_visits = list(mcts_visits) if not isinstance(mcts_visits, list) else mcts_visits
    # Expand scalar mcts_visits to per-player
    if len(params.mcts_visits) == 1:
        params.mcts_visits = params.mcts_visits * num_players
    elif not isinstance(params.mcts_visits, list):
        params.mcts_visits = [params.mcts_visits] * num_players

    set_model_groups(params, players)
    model_groups = list(params.model_groups)

    # Generate seat permutations (all rotations)
    seat_perms = []
    for i in range(num_players):
        perm = [model_groups[(j + i) % num_players] for j in range(num_players)]
        seat_perms.append(perm)
    params.seat_perms = seat_perms

    n = bs * cb * num_players  # all perms at once
    params.games_to_play = n
    set_eval_types(params, players)

    # Build unique players list for GameRunner
    # (deduplicate by model group)
    seen_groups = {}
    unique_players = [None] * num_players
    for i, p in enumerate(players):
        g = model_groups[i]
        if g not in seen_groups:
            seen_groups[g] = i
        unique_players[i] = players[seen_groups[g]]

    pm = alphazero.PlayManager(Game(), params)
    grargs = GRArgs(
        title=name,
        game=Game,
        max_batch_size=bs,
    )
    gr = GameRunner(unique_players, pm, grargs)
    gr.run()

    # Extract per-perm results
    win_rates = [0.0] * num_players
    for perm_idx in range(pm.num_seat_perms()):
        perm_scores = pm.perm_scores(perm_idx)
        perm_games = pm.perm_games_completed(perm_idx)
        if perm_games == 0:
            continue
        rotation = perm_idx  # perm_idx corresponds to rotation offset
        for seat in range(num_players):
            original_player = (seat + rotation) % num_players
            wins = perm_scores[seat] + perm_scores[num_players] / num_players
            win_rates[original_player] += wins / perm_games

    for i in range(num_players):
        win_rates[i] /= num_players

    gc.collect()
    return win_rates


# ---------------------------------------------------------------------------
# Network selection from experiment directories
# ---------------------------------------------------------------------------


def select_from_experiment_dir(exp_dir, count, elo_path=None):
    """Select networks from an experiment directory.

    Selection priority:
    1. Best network (highest ELO)
    2. Latest network (highest iteration)
    3. Remaining slots evenly distributed

    Returns list of (iter_num, full_path) tuples.
    """
    checkpoint_dir = os.path.join(exp_dir, "checkpoint")
    if not os.path.isdir(checkpoint_dir):
        print(f"Warning: no checkpoint dir in {exp_dir}")
        return []

    # Discover checkpoints
    checkpoints = []
    for pt_file in sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt"))):
        filename = os.path.basename(pt_file)
        match = re.match(r"^(\d+).*\.pt$", filename)
        if match:
            iter_num = int(match.group(1))
            checkpoints.append((iter_num, pt_file))

    if not checkpoints:
        print(f"Warning: no checkpoints in {checkpoint_dir}")
        return []

    checkpoints.sort(key=lambda x: x[0])

    if count >= len(checkpoints):
        return checkpoints

    # Find best (highest ELO) iteration
    best_iter = None
    if elo_path is None:
        elo_path = os.path.join(exp_dir, "elo.csv")
    if os.path.exists(elo_path):
        try:
            elo_data = np.loadtxt(elo_path, delimiter=",")
            if elo_data.ndim == 1:
                best_iter = int(np.argmax(elo_data))
            else:
                # Last column might be the ELO values
                best_iter = int(np.argmax(elo_data[-1]))
        except Exception:
            pass

    selected = set()

    # 1. Best by ELO
    if best_iter is not None:
        for iter_num, _ in checkpoints:
            if iter_num == best_iter:
                selected.add(iter_num)
                break

    # 2. Latest
    selected.add(checkpoints[-1][0])

    # 3. Evenly distributed for remaining slots
    remaining = count - len(selected)
    if remaining > 0:
        total = len(checkpoints)
        for i in range(1, remaining + 1):
            idx = round(i * (total - 1) / (remaining + 1))
            selected.add(checkpoints[idx][0])

    # Build result
    iter_to_path = {it: path for it, path in checkpoints}
    result = sorted(
        [(it, iter_to_path[it]) for it in selected], key=lambda x: x[0]
    )
    return result


def load_agent(Game, agent_name, model_path, mcts_visits, rand_agents):
    """Load an agent from its descriptor. Returns (player, depth)."""
    if agent_name == "playout":
        return PlayoutPlayer(), mcts_visits
    if agent_name in rand_agents:
        return RandPlayer(), agent_name
    # Network agent - agent_name is either a filename or full path
    if os.path.isabs(agent_name) or os.path.exists(agent_name):
        nn = neural_net.NNWrapper.load_checkpoint(
            Game, os.path.dirname(agent_name), os.path.basename(agent_name)
        )
        nn.enable_inference_optimizations()
        return nn, mcts_visits
    nn = neural_net.NNWrapper.load_checkpoint(Game, model_path, agent_name)
    nn.enable_inference_optimizations()
    return nn, mcts_visits


# ---------------------------------------------------------------------------
# Tournament formats
# ---------------------------------------------------------------------------


def run_monrad(config, Game, agents, mcts_visits, bs, model_path, rand_agents, cache_size=0):
    """Run a Monrad (Swiss-style) tournament."""
    if len(agents) % 2 == 1:
        agents.insert(0, "dummy")

    count = len(agents)
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
                            win_matrix[
                                rankings[current], rankings[current - offset]
                            ]
                        )
                    ):
                        offset += 1
                    if current - offset < 0:
                        print("No one to play? Relaxing constraints")
                        offset = 1
                        while not math.isnan(
                            win_matrix[
                                rankings[current], rankings[current - offset]
                            ]
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

                p1, d1 = load_agent(Game, agents[i], model_path, mcts_visits, rand_agents)
                p2, d2 = load_agent(Game, agents[j], model_path, mcts_visits, rand_agents)

                players = [p2] * Game.NUM_PLAYERS()
                depths = [d2] * Game.NUM_PLAYERS()
                players[0] = p1
                depths[0] = d1

                win_rates = pit_agents(
                    config,
                    Game,
                    players,
                    depths,
                    bs,
                    f"{_agent_label(agents[i])}-{_agent_label(agents[j])}",
                    cache_size=cache_size,
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

    return agents, elo, win_matrix


def run_roundrobin(config, Game, agents, mcts_visits, bs, model_path, rand_agents, cache_size=0):
    """Run a round-robin tournament."""
    count = len(agents)
    win_matrix = np.zeros((count, count))

    with tqdm.trange(count * (count - 1) // 2, desc="Pit Agents") as pbar:
        for i in range(count):
            p1, d1 = load_agent(Game, agents[i], model_path, mcts_visits, rand_agents)
            for j in range(i + 1, count):
                p2, d2 = load_agent(
                    Game, agents[j], model_path, mcts_visits, rand_agents
                )
                players = [p2] * Game.NUM_PLAYERS()
                depths = [d2] * Game.NUM_PLAYERS()
                players[0] = p1
                depths[0] = d1

                win_rates = pit_agents(
                    config,
                    Game,
                    players,
                    depths,
                    bs,
                    f"{_agent_label(agents[i])}-{_agent_label(agents[j])}",
                    cache_size=cache_size,
                )

                if Game.NUM_PLAYERS() == 2:
                    win_matrix[i, j] = win_rates[0]
                    win_matrix[j, i] = win_rates[1]
                    pbar.update()
                    continue

                # For >2 players, swap positions and average
                players = [p1] * Game.NUM_PLAYERS()
                depths = [d1] * Game.NUM_PLAYERS()
                players[0] = p2
                depths[0] = d2
                win_rates2 = pit_agents(
                    config,
                    Game,
                    players,
                    depths,
                    bs,
                    f"{_agent_label(agents[j])}-{_agent_label(agents[i])}",
                    cache_size=cache_size,
                )

                wr1 = win_rates[0]
                wr2 = win_rates2[0]
                for k in range(1, len(win_rates)):
                    wr1 += win_rates2[k]
                    wr2 += win_rates[k]
                win_matrix[i, j] = wr1 / 2
                win_matrix[j, i] = wr2 / 2
                pbar.update()

    elo = calc_elo(np.zeros(count), win_matrix)
    return agents, elo, win_matrix


def _agent_label(agent_name):
    """Short label for agent name (truncate long paths)."""
    if isinstance(agent_name, (int, float)):
        return f"rand-{agent_name}"
    name = os.path.basename(str(agent_name))
    if name.endswith(".pt"):
        name = name[:-3]
    return name[:30]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Tournament runner")
    parser.add_argument(
        "--game",
        required=True,
        choices=list(GAME_REGISTRY.keys()),
        help="Game to play",
    )
    parser.add_argument(
        "--format",
        choices=["monrad", "roundrobin"],
        default="monrad",
        help="Tournament format (default: monrad)",
    )
    parser.add_argument(
        "--dir",
        action="append",
        default=[],
        help="Experiment directory to load checkpoints from (can repeat)",
    )
    parser.add_argument(
        "--count",
        action="append",
        type=int,
        default=[],
        help="Number of networks per --dir (must match number of --dir args)",
    )
    parser.add_argument("--random", action="store_true", help="Include random agent")
    parser.add_argument("--playout", action="store_true", help="Include playout agent")
    parser.add_argument(
        "--mcts_visits", type=int, default=200, help="MCTS visits (default: 200)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Games per position (default: 64)"
    )
    parser.add_argument(
        "--base-dir", default="data", help="Base data directory (default: data)"
    )
    parser.add_argument(
        "--cache_size", type=int, default=200000, help="S3-FIFO cache size (default: 200000)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create a minimal config for base_params
    config = TrainConfig(game=args.game)
    Game = config.Game

    # Build agent list
    agents = []
    rand_agents = []
    model_path = None

    if args.dir:
        # Directory-based selection
        if args.count and len(args.count) != len(args.dir):
            print(
                f"Error: {len(args.dir)} --dir args but {len(args.count)} --count args"
            )
            sys.exit(1)

        for idx, exp_dir in enumerate(args.dir):
            count = args.count[idx] if idx < len(args.count) else 10
            selected = select_from_experiment_dir(exp_dir, count)
            for iter_num, path in selected:
                agents.append(path)
            print(
                f"Selected {len(selected)} networks from {exp_dir}"
            )

        # Use first dir's checkpoint for model_path fallback
        model_path = os.path.join(args.dir[0], "checkpoint")
    else:
        # Interactive selection via network_selector
        # Try new-style experiment dirs first, then old-style
        game_dir = os.path.join(args.base_dir, args.game)
        old_model_path = os.path.join(args.base_dir, "checkpoint")
        old_bench_path = os.path.join(args.base_dir, "bench")

        # Check for experiment-style dirs
        found_experiments = False
        if os.path.isdir(game_dir):
            for entry in os.listdir(game_dir):
                if os.path.isdir(os.path.join(game_dir, entry, "checkpoint")):
                    found_experiments = True
                    break

        if found_experiments:
            # Flatten all checkpoints into old-style runs for network_selector
            exp_model_path = os.path.join(game_dir, "_tournament_tmp")
            # Point to the first experiment's checkpoint dir
            for entry in sorted(os.listdir(game_dir)):
                cdir = os.path.join(game_dir, entry, "checkpoint")
                if os.path.isdir(cdir):
                    model_path = cdir
                    break
            # Discover across all experiment checkpoint dirs
            all_runs = {}
            for entry in sorted(os.listdir(game_dir)):
                cdir = os.path.join(game_dir, entry, "checkpoint")
                if os.path.isdir(cdir):
                    runs = discover_runs(cdir)
                    all_runs.update(runs)
            if all_runs:
                nn_agents, mcts_visits, num_random, num_playout = interactive_select(
                    all_runs, default_mcts=args.mcts_visits
                )
                args.mcts_visits = mcts_visits
                if num_random:
                    args.random = True
                if num_playout:
                    args.playout = True
                agents.extend(nn_agents)
        elif os.path.isdir(old_bench_path):
            model_path = old_bench_path
            runs = discover_runs(model_path)
            if runs:
                nn_agents, mcts_visits, num_random, num_playout = interactive_select(
                    runs, default_mcts=args.mcts_visits
                )
                args.mcts_visits = mcts_visits
                if num_random:
                    args.random = True
                if num_playout:
                    args.playout = True
                agents.extend(nn_agents)
        elif os.path.isdir(old_model_path):
            model_path = old_model_path
            runs = discover_runs(model_path)
            if runs:
                nn_agents, mcts_visits, num_random, num_playout = interactive_select(
                    runs, default_mcts=args.mcts_visits
                )
                args.mcts_visits = mcts_visits
                if num_random:
                    args.random = True
                if num_playout:
                    args.playout = True
                agents.extend(nn_agents)
        else:
            print(f"No checkpoints found for {args.game}")
            sys.exit(1)

    # Add special agents
    if args.random:
        rand_agents = [args.mcts_visits]
        agents = rand_agents + agents
    if args.playout:
        agents = ["playout"] + agents

    if len(agents) < 2:
        print("Need at least 2 agents for a tournament")
        sys.exit(1)

    print(f"\n=== {args.game} {args.format.capitalize()} Tournament ===")
    print(f"MCTS visits: {args.mcts_visits}")
    print(f"Batch size: {args.batch_size}")
    print(f"Agents ({len(agents)}):")
    for a in agents:
        print(f"  {_agent_label(a)}")
    print()

    # Run tournament
    if args.format == "monrad":
        agents, elo, win_matrix = run_monrad(
            config, Game, agents, args.mcts_visits, args.batch_size, model_path, rand_agents,
            cache_size=args.cache_size,
        )
    else:
        agents, elo, win_matrix = run_roundrobin(
            config, Game, agents, args.mcts_visits, args.batch_size, model_path, rand_agents,
            cache_size=args.cache_size,
        )

    # Print results
    print()
    print("=== Final Results ===")
    print("Win matrix:")
    print(win_matrix)
    print()

    rankings = list(np.argsort(elo))
    print("Leaderboard:")
    for rank, i in enumerate(reversed(rankings)):
        print(f"  {rank+1}. {_agent_label(agents[i])}: {elo[i]:.0f}")

    # Save results
    num_random = 1 if args.random else 0
    num_playout = 1 if args.playout else 0
    output_dir = create_tournament_dir(".", args.game, args.format)
    save_tournament_results(
        output_dir,
        [_agent_label(a) for a in agents],
        elo,
        win_matrix,
        mcts_visits=args.mcts_visits,
        num_random=num_random,
        num_playout=num_playout,
        variant=args.game,
        fmt=args.format,
    )


if __name__ == "__main__":
    main()
