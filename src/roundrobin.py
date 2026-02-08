import neural_net
from game_runner import GameRunner, GRArgs, RandPlayer, PlayoutPlayer, base_params, set_eval_types
import glob
import os
import numpy as np
import gc
import tqdm
import alphazero
from tracy_utils import tracy_zone

np.set_printoptions(precision=3, suppress=True)


@tracy_zone
def pit_agents(Game, players, mcts_depths, bs, name):
    np = Game.NUM_PLAYERS()
    win_rates = [0] * np
    for i in tqdm.trange(np, leave=False, desc=name):
        cb = Game.NUM_PLAYERS()
        n = bs * cb
        ordered_players = [None] * np
        ordered_depths = [None] * np
        for j in range(np):
            ordered_players[j] = players[(j + i) % np]
            ordered_depths[j] = mcts_depths[(j + i) % np]

        params = base_params(Game, 0.5, bs, cb)
        # Disable the cache because it does not work well in arena.
        params.max_cache_size = 0
        params.games_to_play = n
        params.mcts_depth = ordered_depths
        set_eval_types(params, ordered_players)
        pm = alphazero.PlayManager(Game(), params)

        grargs = GRArgs(
            title=f"{name}({i + 1}/{np})",
            game=Game,
            max_batch_size=bs,
            concurrent_batches=cb,
            result_workers=2,
        )
        gr = GameRunner(ordered_players, pm, grargs)
        gr.run()
        scores = pm.scores()
        for j in range(np):
            wins = scores[j] + scores[-1] / np
            win_rates[(j + i) % np] += wins / n
        gc.collect()
    for i in range(np):
        win_rates[i] /= np
    return win_rates


if __name__ == "__main__":
    from monrad import calc_elo
    from network_selector import (
        discover_runs, interactive_select, create_tournament_dir,
        save_tournament_results,
    )

    model_path = os.path.join("data", "checkpoint")
    if os.path.isdir(os.path.join("data", "bench")):
        model_path = os.path.join("data", "bench")

    runs = discover_runs(model_path)
    if not runs:
        print(f"No checkpoints found in {model_path}")
        exit(1)

    nn_agents, mcts_visits, num_random, num_playout = interactive_select(runs, default_mcts=200)

    rand_agents = [mcts_visits] * num_random
    playout_agents = ["playout"] * num_playout
    agents = rand_agents + playout_agents + nn_agents

    Game = alphazero.Connect4GS
    bs = 64

    count = len(agents)
    print(agents)
    win_matrix = np.zeros((count, count))
    with tqdm.trange(count * (count - 1) // 2, desc="Pit Agents") as pbar:
        for i in range(count):
            p1 = None
            d1 = 0
            if agents[i] == "playout":
                p1 = PlayoutPlayer()
                d1 = mcts_visits
            elif agents[i] in rand_agents:
                p1 = RandPlayer()
                d1 = agents[i]
            else:
                p1 = neural_net.NNWrapper.load_checkpoint(Game, model_path, agents[i])
                d1 = mcts_visits
            for j in range(i + 1, count):
                p2 = None
                d2 = 0
                if agents[j] == "playout":
                    p2 = PlayoutPlayer()
                    d2 = mcts_visits
                elif agents[j] in rand_agents:
                    p2 = RandPlayer()
                    d2 = agents[j]
                else:
                    p2 = neural_net.NNWrapper.load_checkpoint(
                        Game, model_path, agents[j]
                    )
                    d2 = mcts_visits
                players = [p2] * Game.NUM_PLAYERS()
                depths = [d2] * Game.NUM_PLAYERS()
                players[0] = p1
                depths[0] = d1
                win_rates = pit_agents(
                    Game, players, depths, bs, f"{agents[i]}-{agents[j]}"
                )
                if Game.NUM_PLAYERS() == 2:
                    win_matrix[i, j] = win_rates[0]
                    win_matrix[j, i] = win_rates[1]
                    print(win_matrix[i, j])
                    pbar.update()
                    continue
                players = [p1] * Game.NUM_PLAYERS()
                depths = [d1] * Game.NUM_PLAYERS()
                players[0] = p2
                depths[0] = d2
                win_rates2 = pit_agents(
                    Game, players, depths, bs, f"{agents[j]}-{agents[i]}"
                )
                wr1 = win_rates[0]
                wr2 = win_rates2[0]
                for i in range(1, len(win_rates)):
                    wr1 += win_rates2[i]
                    wr2 += win_rates[i]
                win_matrix[i, j] = wr1 / 2
                win_matrix[j, i] = wr2 / 2
                print(win_matrix[i, j])
                pbar.update()
            print(win_matrix[i])
    print()
    print(win_matrix)
    print()
    print(agents)

    elo = calc_elo(np.zeros(count), win_matrix)
    output_dir = create_tournament_dir(".", "connect4", "roundrobin")
    save_tournament_results(
        output_dir, agents, elo, win_matrix,
        mcts_visits=mcts_visits,
        num_random=num_random,
        num_playout=num_playout,
        variant="connect4",
        fmt="roundrobin",
    )
