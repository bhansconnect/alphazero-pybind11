import neural_net
from game_runner import GameRunner, GRArgs, RandPlayer
import glob
import importlib.util
import os
import numpy as np
import gc
import tqdm
np.set_printoptions(precision=3, suppress=True)

src_path = os.path.dirname(os.path.realpath(__file__))
build_path = os.path.join(os.path.dirname(src_path), 'build/src')
lib_path = glob.glob(os.path.join(build_path, 'alphazero*.so'))[0]

spec = importlib.util.spec_from_file_location(
    'alphazero', lib_path)
alphazero = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alphazero)


def pit_agents(Game, players, mcts_depths, bs, name):
    np = Game.NUM_PLAYERS()
    win_rates = [0]*np
    for i in range(np):
        params = alphazero.PlayParams()
        n = bs*4
        cb = 4
        ordered_players = [None]*np
        for j in range(np):
            ordered_players[j] = players[(j+i) % np]

        params.games_to_play = n
        params.concurrent_games = bs * cb
        params.max_batch_size = bs
        params.mcts_depth = mcts_depths
        params.history_enabled = False
        params.self_play = False
        params.max_cache_size = 1000000
        params.temp_minimization_turn = 10
        params.temp = 0.1
        params.cpuct = 2
        pm = alphazero.PlayManager(Game(), params)

        grargs = GRArgs(title=f'{name}({i+1}/{np})', game=Game,
                        max_batch_size=bs, concurrent_batches=cb, result_workers=2)
        gr = GameRunner(ordered_players, pm, grargs)
        gr.run()
        scores = pm.scores()
        draws = pm.draws()
        for j in range(np):
            wins = (n+scores[j]-draws)/2
            adj_wins = wins + draws/np
            win_rates[(j+i) % np] += adj_wins/n
        gc.collect()
    for i in range(np):
        win_rates[i] /= np
    return win_rates


if __name__ == '__main__':
    nn_agents = [os.path.basename(x) for x in sorted(
        glob.glob('data/roundrobin/*.pt'))]
    rand_agents = [100, 400, 1600, 6400]
    agents = nn_agents + rand_agents

    Game = alphazero.Connect4GS
    bs = 32
    depth = 10
    channels = 32
    nn_mtcs_depth = 100

    nnargs = neural_net.NNArgs(
        num_channels=channels, depth=depth)

    count = len(agents)
    print(agents)
    win_matrix = np.zeros((count, count))
    with tqdm.trange(count*(count-1)//2, desc='Pit Agents') as pbar:
        for i in range(count):
            p1 = None
            d1 = 0
            if agents[i] in rand_agents:
                p1 = RandPlayer(Game, bs)
                d1 = agents[i]
            else:
                p1 = neural_net.NNWrapper(Game, nnargs)
                p1.load_checkpoint('data/roundrobin', agents[i])
                d1 = nn_mtcs_depth
            for j in range(i+1, count):
                p2 = None
                d2 = 0
                if agents[j] in rand_agents:
                    p2 = RandPlayer(Game, bs)
                    d2 = agents[j]
                else:
                    p2 = neural_net.NNWrapper(Game, nnargs)
                    p2.load_checkpoint('data/roundrobin', agents[j])
                    d2 = nn_mtcs_depth
                win_rates = pit_agents(
                    Game, [p1, p2], [d1, d2], bs, f'{agents[i]}-{agents[j]}')
                win_matrix[i, j] = win_rates[0]
                win_matrix[j, i] = win_rates[1]
                pbar.update()
            print(win_matrix[i])
    print()
    print(win_matrix)
