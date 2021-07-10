import neural_net
from game_runner import GameRunner, GRArgs, RandPlayer, base_params
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
    for i in tqdm.trange(np, leave=False, desc=name):
        params = alphazero.PlayParams()
        cb = Game.NUM_PLAYERS()
        n = bs*cb
        ordered_players = [None]*np
        ordered_depths = [None]*np
        for j in range(np):
            ordered_players[j] = players[(j+i) % np]
            ordered_depths[j] = mcts_depths[(j+i) % np]

        params = base_params(Game, 0.5, bs, cb)
        params.games_to_play = n
        params.mcts_depth = ordered_depths
        pm = alphazero.PlayManager(Game(), params)

        grargs = GRArgs(title=f'{name}({i+1}/{np})', game=Game,
                        max_batch_size=bs, concurrent_batches=cb, result_workers=2)
        gr = GameRunner(ordered_players, pm, grargs)
        gr.run()
        scores = pm.scores()
        for j in range(np):
            wins = scores[j] + scores[-1]/np
            win_rates[(j+i) % np] += wins/n
        gc.collect()
    for i in range(np):
        win_rates[i] /= np
    return win_rates


if __name__ == '__main__':
    nn_agents = [os.path.basename(x) for x in sorted(
        glob.glob('data/roundrobin/*.pt'), reverse=True)]
    # rand_agents = [1000]
    rand_agents = []
    agents = nn_agents + rand_agents

    Game = alphazero.Connect4GS
    bs = 16
    depth = 5
    channels = 32
    nn_mtcs_depth = 500

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
                _, depth, channels = os.path.splitext(agents[i])[0].split('-')
                nnargs = neural_net.NNArgs(
                    num_channels=int(channels), depth=int(depth))
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
                    _, depth, channels = os.path.splitext(agents[j])[
                        0].split('-')
                    nnargs = neural_net.NNArgs(
                        num_channels=int(channels), depth=int(depth))
                    p2 = neural_net.NNWrapper(Game, nnargs)
                    p2.load_checkpoint('data/roundrobin', agents[j])
                    d2 = nn_mtcs_depth
                players = [p2] * Game.NUM_PLAYERS()
                depths = [d2] * Game.NUM_PLAYERS()
                players[0] = p1
                depths[0] = d1
                win_rates = pit_agents(
                    Game, players, depths, bs, f'{agents[i]}-{agents[j]}')
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
                    Game, players, depths, bs, f'{agents[j]}-{agents[i]}')
                wr1 = win_rates[0]
                wr2 = win_rates2[0]
                for i in range(1, len(win_rates)):
                    wr1 += win_rates2[i]
                    wr2 += win_rates[i]
                win_matrix[i, j] = wr1/2
                win_matrix[j, i] = wr2/2
                print(win_matrix[i, j])
                pbar.update()
            print(win_matrix[i])
    print()
    print(win_matrix)
    print()
    print(agents)
    np.savetxt('data/roundrobin_wr.csv', win_matrix,
               delimiter=',', header=','.join([str(a) for a in agents]))
