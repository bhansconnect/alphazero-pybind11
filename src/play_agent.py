import glob
import os
import torch
import numpy as np
import time
from load_lib import load_alphazero

alphazero = load_alphazero()

THINK_TIME = 9.5
CPUCT = 1.25
START_TEMP = 1
END_TEMP = 0.2
TEMP_DECAY_HALF_LIFE = 10

Game = alphazero.OnitamaGS

bf = 0
bfc = 0


def calc_temp(turn):
    ln2 = 0.693
    ld = ln2 / TEMP_DECAY_HALF_LIFE
    temp = START_TEMP - END_TEMP
    temp *= np.exp(-ld * turn)
    temp += END_TEMP
    return temp


ANALYZE = False
PLAY_GAMES = 1
ANALYSIS_GAMES = 5
MAX_ANALYSIS_PLAYOUTS = 5000
ANALYSIS_GROUPING = 25
assert MAX_ANALYSIS_PLAYOUTS % ANALYSIS_GROUPING == 0
policy_loss = np.full((MAX_ANALYSIS_PLAYOUTS //
                       ANALYSIS_GROUPING)+1, 0, dtype=float)
best_move_change = np.full((MAX_ANALYSIS_PLAYOUTS //
                            ANALYSIS_GROUPING)+1, 0, dtype=float)
move_changed = np.full((MAX_ANALYSIS_PLAYOUTS //
                        ANALYSIS_GROUPING)+1, 0, dtype=float)
move_count = 0


def kl_divergence(target, actual, epsilon=1e9):
    out = 0
    for i in range(len(target)):
        if target[i] > 0 and actual[i] > 0:
            out += -target[i]*np.log(actual[i]/target[i])
        elif target[i] > 0:
            out += -target[i]*np.log(epsilon/target[i])
    return out


def playout_analysis(gs, agent):
    global policy_loss
    global best_move_change
    global move_changed
    global move_count
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

    best_move = np.argmax(predictions[-1])
    for i, pi in enumerate(predictions):
        e = 1e-9 if i == 0 else 1.0/MAX_ANALYSIS_PLAYOUTS
        policy_loss[i] += kl_divergence(predictions[-1], pi, e)
        best_move_change[i] += 100*(pi[best_move] - predictions[0][best_move])
        if best_move != np.argmax(pi):
            move_changed[i] += 1

    move_count += 1

    probs = mcts.probs(calc_temp(gs.current_turn()))
    rand = np.random.choice(probs.shape[0], p=probs)
    return rand


def finalize_analysis(name):
    import matplotlib.pyplot as plt
    global policy_loss
    global best_move_change
    global move_changed
    global move_count
    policy_loss /= move_count
    best_move_change /= move_count
    move_changed /= move_count
    visits = list(range(0, MAX_ANALYSIS_PLAYOUTS, ANALYSIS_GROUPING))
    visits.append(MAX_ANALYSIS_PLAYOUTS)

    loss_per_visit = np.copy(policy_loss)
    loss_per_visit = policy_loss[0] - loss_per_visit
    best_move_change_per_visit = np.copy(best_move_change)
    move_changed_per_visit = np.copy(move_changed)
    move_changed_per_visit = move_changed_per_visit[0] - move_changed_per_visit
    for i in range(1, len(policy_loss)):
        loss_per_visit[i] /= i*ANALYSIS_GROUPING
        if loss_per_visit[i] < 0:
            loss_per_visit[i] = 0
        move_changed_per_visit[i] /= i*ANALYSIS_GROUPING
        if move_changed_per_visit[i] < 0:
            move_changed_per_visit[i] = 0
        best_move_change_per_visit[i] /= i*ANALYSIS_GROUPING
    plt.figure(1)
    plt.suptitle(name)
    plt.subplot(221)
    plt.title('Loss vs visits')
    plt.plot(visits, policy_loss)
    plt.ylabel('loss')
    plt.xlabel('visits')
    plt.subplot(222)
    plt.title('Loss change per one visits')
    plt.plot(visits[1:], loss_per_visit[1:])
    plt.ylabel('loss change')
    plt.xlabel('visits')
    plt.subplot(223)
    plt.title('Best move change vs visits')
    plt.plot(visits, best_move_change)
    plt.ylabel('percent')
    plt.xlabel('visits')
    plt.subplot(224)
    plt.title('Best move change per one visits')
    plt.plot(visits[1:], best_move_change_per_visit[1:])
    plt.ylabel('percent change')
    plt.xlabel('visits')

    plt.figure(2)
    plt.suptitle(name)
    plt.subplot(211)
    plt.plot(visits, move_changed)
    plt.title('Selected move change percent')
    plt.ylabel('percent')
    plt.xlabel('visits')
    plt.subplot(212)
    plt.plot(visits[1:], move_changed_per_visit[1:])
    plt.title('Selected move change percent')
    plt.ylabel('percent change')
    plt.xlabel('visits')
    plt.show()


def eval_position(gs, agent):
    mcts = alphazero.MCTS(CPUCT, gs.num_players(),
                          gs.num_moves(), 0, 1.4, 0.25)
    global bf
    global bfc
    bf += np.sum(gs.valid_moves())
    bfc += 1
    start = time.time()
    sims = 0
    while time.time() - start < THINK_TIME:
        # for _ in range(500):
        leaf = mcts.find_leaf(gs)
        v, pi = agent.predict(torch.from_numpy(leaf.canonicalized()))
        v = v.cpu().numpy()
        pi = pi.cpu().numpy()
        mcts.process_result(gs, v, pi, False)
        sims += 1
    print(f'\tRan {sims} simulations in {round(time.time()-start, 3)} seconds')
    # print('Press enter for ai analysis')
    # input()
    v, pi = agent.predict(torch.from_numpy(gs.canonicalized()))
    v = v.cpu().numpy()
    pi = pi.cpu().numpy()
    print(f'\tRaw Score: {v}')
    thing = {x: round(100*pi[x], 1) for x in reversed(np.argsort(pi)[-10:])}
    print(f'\tRaw Top Probs: {thing}')
    print(f'\tRaw Best: {np.argmax(pi)}')
    print(f'\tRaw Rand: {np.random.choice(pi.shape[0], p=pi)}')

    print(f'\tMCTS Value Current Player: {mcts.root_value()}')
    counts = mcts.counts()
    thing = {x: counts[x] for x in reversed(np.argsort(counts)[-10:])}
    print(f'\tMCTS Top Counts: {thing}')
    probs = mcts.probs(calc_temp(gs.current_turn()))
    thing = {x: round(100*probs[x], 1)
             for x in reversed(np.argsort(probs)[-10:])}
    print(f'\tMCTS Top Probs: {thing}')
    print(f'\tMCTS Best: {np.argmax(probs)}')
    rand = np.random.choice(probs.shape[0], p=probs)
    print(f'\tMCTS Rand: {rand}')
    return rand


if __name__ == '__main__':
    import neural_net
    import game_runner

    np.set_printoptions(precision=3, suppress=True)
    nn_folder = 'data/checkpoint'
    nn_file = os.path.basename(
        sorted(glob.glob(os.path.join(nn_folder, '*.pt')))[-1])

    print(f'Using network: {nn_file}')
    nn = neural_net.NNWrapper.load_checkpoint(Game, nn_folder, nn_file)
    for _ in range(ANALYSIS_GAMES if ANALYZE else PLAY_GAMES):
        gs = Game()
        pc = 0
        hist = []
        while gs.scores() is None:
            hist.append(gs.copy())
            print()
            print()
            print(gs)
            if ANALYZE:
                rand = playout_analysis(gs, nn)
            else:
                rand = eval_position(gs, nn)
            print()
            valids = gs.valid_moves()
            valid = False

            def get_input_num(stuff, max_stuff):
                valid = False
                while not valid:
                    try:
                        print(f'Enter {stuff}: ', end='')
                        selection = int(input())
                        if selection == 0:
                            return 0
                        if 0 < selection <= max_stuff:
                            return selection
                        else:
                            raise Exception('Sad')
                    except KeyboardInterrupt:
                        exit()
                    except:
                        print('You suck at typing numbers. Get Gut!')

            # def print_ref_board():
            #     for h in range(7):
            #         for w in range(7):
            #             if not ((h == 0 and w == 0) or (h == 1 and w == 0) or (h == 0 and w == 1) or
            #                     (h == 1 and w == 1) or (h == 2 and w == 0) or (h == 0 and w == 2) or
            #                     (h == 6 and w == 6) or (h == 5 and w == 6) or (h == 6 and w == 5) or
            #                     (h == 5 and w == 5) or (h == 4 and w == 6) or (h == 6 and w == 4)):
            #                 print(f'{w+h*7:2d}', end=' ')
            #             else:
            #                 print('  ', end=' ')
            #         print()
            # valid = False
            # move = 0

            # # grow, seed, buy, pass, by number
            # # grow locaction
            # # seed from locaction
            # # seed to locaction

            # while not valid:
            #     choice = get_input_num(
            #         '0-undo, 1-buy, 2-grow, 3-seed, 4-pass, 5-wild card', 5)
            #     if choice == 0:
            #         if len(hist) == 1:
            #             gs = hist[-1].copy()
            #             hist = []
            #         else:
            #             gs = hist[-2].copy()
            #             hist = hist[:-2]
            #         move = -1
            #         break
            #     if choice == 1:
            #         buy_choice = get_input_num(
            #             '1-seed, 2-small, 3-med, 4-large', 4)
            #         move = 2449+buy_choice
            #     elif choice == 2:
            #         print_ref_board()
            #         grow_choice = get_input_num('grow tile location (0-49ish)', 49)
            #         move = grow_choice
            #     elif choice == 3:
            #         print_ref_board()
            #         from_choice = get_input_num('from tile location (0-49ish)', 49)
            #         to_choice = get_input_num('to tile location (0-49ish)', 49)
            #         move = 49 + from_choice*49 + to_choice
            #     elif choice == 4:
            #         move = 2454
            #     elif choice == 5:
            #         move = get_input_num('anything(0-2454)', 2454)
            #     valid = valids[move]
            #     if not valid:
            #         print(f'Move {move} is sad :(')
            #     print()
            # if rand == 2454:
            #     pc += 1

            # move = -1
            # while not valid:
            #     try:
            #         print('Enter Move(-1 is undo): ', end='')
            #         move = int(input())
            #         if 0 <= move < len(valids):
            #             valid = valids[move]
            #         elif move == -1:
            #             if len(hist) == 1:
            #                 gs = hist[-1].copy()
            #                 hist = []
            #             else:
            #                 gs = hist[-2].copy()
            #                 hist = hist[:-2]
            #             move = -1
            #             break
            #         else:
            #             raise Exception('Sad')
            #     except KeyboardInterrupt:
            #         exit()
            #     except:
            #         print('You suck at typing numbers. Get Gut!')
            # if move != -1:
            #     gs.play_move(move)

            HEIGHT = Game.CANONICAL_SHAPE()[1]
            WIDTH = Game.CANONICAL_SHAPE()[2]

            def gen_move(from_h, from_w, h_move, to_loc):
                if h_move:
                    return (from_h * WIDTH + from_w) * (WIDTH + HEIGHT) + WIDTH + to_loc
                return (from_h * WIDTH + from_w) * (WIDTH + HEIGHT) + to_loc

            # move = -1
            # while not valid:
            #     try:
            #         print('Enter Move(-1 is undo, -2 human input): ', end='')
            #         move = int(input())
            #         if 0 <= move <= len(valids):
            #             valid = valids[move]
            #         elif move == -1:
            #             if len(hist) == 1:
            #                 gs = hist[-1].copy()
            #                 hist = []
            #             else:
            #                 gs = hist[-2].copy()
            #                 hist = hist[:-2]
            #             move = -1
            #             break
            #         elif move == -2:
            #             print('locations are 1 indexed')
            #             from_h = get_input_num('from height', HEIGHT) - 1
            #             from_w = get_input_num('from width', WIDTH) - 1
            #             to_h = get_input_num('to height', HEIGHT) - 1
            #             to_w = get_input_num('to width', WIDTH) - 1
            #             if (from_h != to_h and from_w != to_w):
            #                 raise Exception(
            #                     'move not valid. Needs to be rook like move')
            #             if from_h == to_h:
            #                 move = gen_move(from_h, from_w, False, to_w)
            #                 valid = valids[move]
            #             else:
            #                 move = gen_move(from_h, from_w, True, to_h)
            #                 valid = valids[move]
            #         else:
            #             raise Exception('Sad')
            #     except KeyboardInterrupt:
            #         exit()
            #     except Exception as e:
            #         print('You suck at typing numbers. Get Gut!')
            #         print(e)
            # if move != -1:
            #     gs.play_move(move)

            gs.play_move(rand)
        print(gs)
        print(gs.scores())
    if ANALYZE:
        finalize_analysis(nn_file)
    else:
        print('Passes:', pc)
        print('BranchFactor:', bf/bfc)
        print(bf)
        print(bfc)
