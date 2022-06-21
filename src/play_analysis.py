import glob
import os
import torch
import numpy as np
import time
import neural_net
from load_lib import load_alphazero

alphazero = load_alphazero()

Game = alphazero.Connect4GS

CPUCT = 1.25
START_TEMP = 1
END_TEMP = 0.2
TEMP_DECAY_HALF_LIFE = 10

MODEL_COUNT = 4
ANALYSIS_GAMES = 3
MAX_ANALYSIS_PLAYOUTS = 5000
ANALYSIS_GROUPING = 25


def calc_temp(turn):
    ln2 = 0.693
    ld = ln2 / TEMP_DECAY_HALF_LIFE
    temp = START_TEMP - END_TEMP
    temp *= np.exp(-ld * turn)
    temp += END_TEMP
    return temp


assert MAX_ANALYSIS_PLAYOUTS % ANALYSIS_GROUPING == 0


def kl_divergence(target, actual, epsilon=1e9):
    out = 0
    for i in range(len(target)):
        if target[i] > 0 and actual[i] > 0:
            out += -target[i]*np.log(actual[i]/target[i])
        elif target[i] > 0:
            out += -target[i]*np.log(epsilon/target[i])
    return out


def mcts_predictions(gs, agent):
    mcts = alphazero.MCTS(CPUCT, gs.num_players(),
                          gs.num_moves(), 0, 1.4, 0.25)
    _, raw_pi = agent.predict(torch.from_numpy(gs.canonicalized()))
    predictions = [raw_pi.cpu().numpy()]
    for i in range(MAX_ANALYSIS_PLAYOUTS):
        leaf = mcts.find_leaf(gs)
        v, pi = agent.predict(torch.from_numpy(leaf.canonicalized()))
        v = v.cpu().numpy()
        pi = pi.cpu().numpy()
        mcts.process_result(gs, v, pi, False)
        if (i+1) % ANALYSIS_GROUPING == 0:
            predictions.append(mcts.probs(1.0))

    probs = mcts.probs(calc_temp(gs.current_turn()))
    rand = np.random.choice(probs.shape[0], p=probs)
    return (predictions, rand)


def update_analysis(predictions, oracle_predictions, agent_id):
    global policy_loss
    global best_move_change
    global move_changed
    best_move = np.argmax(oracle_predictions[-1])
    for i, pi in enumerate(predictions):
        e = 1e-9 if i == 0 else 1.0/MAX_ANALYSIS_PLAYOUTS
        policy_loss[agent_id,
                    i] += kl_divergence(oracle_predictions[-1], pi, e)
        best_move_change[agent_id, i] += 100 * \
            (pi[best_move] - predictions[0][best_move])
        if best_move != np.argmax(pi):
            move_changed[agent_id, i] += 100


def playout_analysis(gs, nn_folder, nn_files):
    global move_count
    oracle = neural_net.NNWrapper.load_checkpoint(
        Game, nn_folder, nn_files[-1])
    (oracle_predictions, chosen_move) = mcts_predictions(gs, oracle)
    del oracle
    update_analysis(oracle_predictions, oracle_predictions, -1)

    for agent_id, nn_file in enumerate(nn_files[0:-1]):
        agent = neural_net.NNWrapper.load_checkpoint(
            Game, nn_folder, nn_file)
        (predictions, _) = mcts_predictions(gs, agent)
        del agent

        update_analysis(predictions, oracle_predictions, agent_id)

    move_count += 1
    return chosen_move


def finalize_analysis(nn_files):
    import matplotlib.pyplot as plt
    global policy_loss
    global best_move_change
    global move_changed
    global move_count
    for agent_id in range(len(nn_files)):
        policy_loss[agent_id] /= move_count
        best_move_change[agent_id] /= move_count
        move_changed[agent_id] /= move_count
    visits = list(range(0, MAX_ANALYSIS_PLAYOUTS, ANALYSIS_GROUPING))
    visits.append(MAX_ANALYSIS_PLAYOUTS)

    loss_per_visit = np.copy(policy_loss)
    loss_per_visit = policy_loss[:, 0, np.newaxis] - loss_per_visit
    best_move_change_per_visit = np.copy(best_move_change)
    move_changed_per_visit = np.copy(move_changed)
    move_changed_per_visit = move_changed_per_visit[:,
                                                    0, np.newaxis] - move_changed_per_visit
    for agent_id in range(len(nn_files)):
        for i in range(1, len(policy_loss[agent_id])):
            loss_per_visit[agent_id, i] /= i*ANALYSIS_GROUPING
            if loss_per_visit[agent_id, i] < 0:
                loss_per_visit[agent_id, i] = 0
            move_changed_per_visit[agent_id, i] /= i*ANALYSIS_GROUPING
            if move_changed_per_visit[agent_id, i] < 0:
                move_changed_per_visit[agent_id, i] = 0
            best_move_change_per_visit[agent_id, i] /= i*ANALYSIS_GROUPING
    plt.figure(1)
    plt.subplot(221)
    plt.title('Loss vs visits')
    for i, name in enumerate(nn_files):
        plt.plot(visits, policy_loss[i], label=name)
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('visits')
    plt.subplot(222)
    plt.title('Loss change per one visits')
    for i, name in enumerate(nn_files):
        plt.plot(visits[1:], loss_per_visit[i, 1:], label=name)
    plt.legend()
    plt.ylabel('loss change')
    plt.xlabel('visits')
    plt.subplot(223)
    plt.title('Best move change vs visits')
    for i, name in enumerate(nn_files):
        plt.plot(visits, best_move_change[i], label=name)
    plt.legend()
    plt.ylabel('percent')
    plt.xlabel('visits')
    plt.subplot(224)
    plt.title('Best move change per one visits')
    for i, name in enumerate(nn_files):
        plt.plot(visits[1:], best_move_change_per_visit[i, 1:], label=name)
    plt.legend()
    plt.ylabel('percent change')
    plt.xlabel('visits')

    plt.figure(2)
    plt.subplot(211)
    for i, name in enumerate(nn_files):
        plt.plot(visits, move_changed[i], label=name)
    plt.legend()
    plt.title('Selected move change percent')
    plt.ylabel('percent')
    plt.xlabel('visits')
    plt.subplot(212)
    for i, name in enumerate(nn_files):
        plt.plot(visits[1:], move_changed_per_visit[i, 1:], label=name)
    plt.legend()
    plt.title('Selected move change percent')
    plt.ylabel('percent change')
    plt.xlabel('visits')
    plt.show()


def main():
    global policy_loss
    global best_move_change
    global move_changed
    global move_count

    np.set_printoptions(precision=3, suppress=True)
    nn_folder = 'data/checkpoint'
    base_params = '4d-12c-5k'
    all_nn_paths = sorted(
        glob.glob(os.path.join(nn_folder, f'*-{base_params}-*.pt')))
    chunksize = max(len(all_nn_paths)//MODEL_COUNT, 1)
    selected_nn_paths = list(
        reversed(list(reversed(all_nn_paths))[::chunksize]))
    nn_files = [os.path.basename(x) for x in selected_nn_paths]
    print(nn_files)
    agent_count = len(nn_files)

    # Note, the last network is considered the oracle.
    # Its moves are used for testing how good the other networks are.

    groups = (MAX_ANALYSIS_PLAYOUTS // ANALYSIS_GROUPING+1)
    policy_loss = np.full((agent_count, groups), 0, dtype=float)
    best_move_change = np.full((agent_count, groups), 0, dtype=float)
    move_changed = np.full((agent_count, groups), 0, dtype=float)
    move_count = 0

    for i in range(ANALYSIS_GAMES):
        gs = Game()
        while gs.scores() is None:
            print()
            print(f'Game:    {i+1:3d}/{ANALYSIS_GAMES:3d}')
            print(gs)
            rand = playout_analysis(gs, nn_folder, nn_files)
            print()

            gs.play_move(rand)
        print(gs)
        print(gs.scores())
    finalize_analysis(nn_files)


if __name__ == '__main__':
    main()
