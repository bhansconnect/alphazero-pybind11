from load_lib import load_alphazero
import neural_net
from game_runner import GameRunner, GRArgs, RandPlayer, base_params, elo_prob
import glob
import os
import math
import numpy as np
import gc
import tqdm

np.set_printoptions(precision=3, suppress=True)
alphazero = load_alphazero()


def calc_elo(past_elo, win_rates):
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
                    mean_update += rate - \
                        elo_prob(past_elo[j], past_elo[i])
            mean_update += anchor_wr - \
                elo_prob(anchor, past_elo[i])
            past_elo[i] += mean_update*32
        anchor_update = 0
        for i in range(past_elo.shape[0]):
            anchor_update += (1-anchor_wr) - \
                elo_prob(past_elo[i], anchor)
        anchor += anchor_update*32
    past_elo -= min(past_elo)
    return past_elo


def pit_agents(Game, players, mcts_depths, bs, name):
    np = Game.NUM_PLAYERS()
    win_rates = [0]*np
    for i in tqdm.trange(np, leave=False, desc=name):
        cb = Game.NUM_PLAYERS()
        n = bs*cb
        ordered_players = [None]*np
        ordered_depths = [None]*np
        for j in range(np):
            ordered_players[j] = players[(j+i) % np]
            ordered_depths[j] = mcts_depths[(j+i) % np]

        params = base_params(Game, 0.5, bs, cb)
        # Disable the cache because it does not work well in arena.
        params.max_cache_size = 0
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
        glob.glob('data/bench/*.pt'), reverse=False)]
    # rand_agents = [1000]
    rand_agents = []
    agents = rand_agents + nn_agents

    Game = alphazero.TawlbwrddGS
    bs = 128
    nn_mtcs_depth = 125

    if len(agents) % 2 == 1:
        agents.insert(0, "dummy")

    count = len(agents)
    print(agents)
    win_matrix = np.full((count, count), np.NAN)
    elo = np.zeros(count)
    rankings = list(range(count))
    rounds = int(np.ceil(np.log2(count)))
    dist = count
    for r in range(rounds):
        print(f'Round {r+1}')
        dist = math.ceil(dist/2)
        with tqdm.trange(count//2, desc='Games') as pbar:
            current = len(rankings) - 1
            played = [False]*count
            while current >= 0:
                if played[rankings[current]]:
                    current -= 1
                    continue
                played[rankings[current]] = True
                offset = dist
                while (current-offset >= 0 and (played[rankings[current-offset]] or not math.isnan(win_matrix[rankings[current], rankings[current-offset]]))):
                    offset += 1
                if current-offset < 0:
                    print('No one to play? Relaxing constraints')
                    offset = dist
                    while not math.isnan(win_matrix[rankings[current], rankings[current-offset]]):
                        offset += 1
                played[rankings[current-offset]] = True

                i = rankings[current]
                j = rankings[current-offset]
                # print(f'Pairing {current} vs {current-offset} -> {i} vs {j}')
                if agents[i] == 'dummy':
                    win_matrix[i, j] = 0.0
                    win_matrix[j, i] = 1.0
                    continue
                elif agents[j] == 'dummy':
                    win_matrix[i, j] = 1.0
                    win_matrix[j, i] = 0.0
                    continue

                if agents[i] in rand_agents:
                    p1 = RandPlayer(Game, bs)
                    d1 = agents[i]
                else:
                    _, depth, channels = os.path.splitext(
                        agents[i])[0].split('-')[:3]
                    nnargs = neural_net.NNArgs(
                        num_channels=int(channels), depth=int(depth))
                    p1 = neural_net.NNWrapper(Game, nnargs)
                    p1.load_checkpoint('data/bench', agents[i])
                    d1 = nn_mtcs_depth
                if agents[j] in rand_agents:
                    p2 = RandPlayer(Game, bs)
                    d2 = agents[j]
                else:
                    _, depth, channels = os.path.splitext(
                        agents[j])[0].split('-')[:3]
                    nnargs = neural_net.NNArgs(
                        num_channels=int(channels), depth=int(depth))
                    p2 = neural_net.NNWrapper(Game, nnargs)
                    p2.load_checkpoint('data/bench', agents[j])
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
                    # print(win_matrix[i, j])
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
                # print(win_matrix[i, j])
                pbar.update()
        # Update elo and rankings.
        elo = calc_elo(elo, win_matrix)
        rankings = np.argsort(elo)
        print(np.array2string(elo, precision=0))
        print(rankings)

    print()
    print(win_matrix)
    print()
    print(agents)
    print(elo)
    print(rankings)
    np.savetxt('data/monrad_wr.csv', win_matrix,
               delimiter=',', header=','.join([str(a) for a in agents]))
