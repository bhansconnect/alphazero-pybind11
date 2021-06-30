import glob
import importlib.util
import os
import torch
import numpy as np
import time

src_path = os.path.dirname(os.path.realpath(__file__))
build_path = os.path.join(os.path.dirname(src_path), 'build/src')
lib_path = glob.glob(os.path.join(build_path, 'alphazero*.so'))[0]

spec = importlib.util.spec_from_file_location(
    'alphazero', lib_path)
alphazero = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alphazero)

THINK_TIME = 0.5
CPUCT = 1.1
TEMP = 0.5

bf = 0
bfc = 0


def eval_posistion(gs, agent):
    mcts = alphazero.MCTS(CPUCT, gs.num_players(), gs.num_moves())
    v, pi = agent.predict(torch.from_numpy(gs.canonicalized()))
    v = v.cpu().numpy()
    pi = pi.cpu().numpy()
    global bf
    global bfc
    bf += np.sum(gs.valid_moves())
    bfc += 1
    start = time.time()
    # while time.time() - start < THINK_TIME:
    for _ in range(250):
        leaf = mcts.find_leaf(gs)
        v, pi = agent.predict(torch.from_numpy(leaf.canonicalized()))
        v = v.cpu().numpy()
        pi = pi.cpu().numpy()
        mcts.process_result(gs, v, pi, False)
    # print('Press enter for ai analysis')
    # input()
    print(f'\tRaw Score: {v}')
    thing = {x[0]: pi[x[0]] for x in np.argwhere(pi > 0.05)}
    print(f'\tRaw Probs: {thing}')
    print(f'\tRaw Best: {np.argmax(pi)}')
    print(f'\tRaw Rand: {np.random.choice(pi.shape[0], p=pi)}')

    print(f'\tMCTS Value Current Player: {mcts.root_value()}')
    print(f'\tMCTS Counts: {mcts.counts()}')
    probs = mcts.probs(TEMP)
    thing = {x[0]: probs[x[0]] for x in np.argwhere(probs > 0.05)}
    print(f'\tMCTS Probs: {thing}')
    print(f'\tMCTS Best: {np.argmax(probs)}')
    rand = np.random.choice(probs.shape[0], p=probs)
    print(f'\tMCTS Rand: {rand}')
    return rand


if __name__ == '__main__':
    import neural_net
    import game_runner

    np.set_printoptions(precision=3, suppress=True)
    Game = alphazero.PhotosynthesisGS4
    nn_folder = 'data/checkpoint'
    nn_file = os.path.basename(
        sorted(glob.glob(os.path.join(nn_folder, '*.pt')))[-1])

    print(f'Using network: {nn_file}')
    depth = 5
    channels = 32
    nnargs = neural_net.NNArgs(num_channels=channels, depth=depth)
    nn = neural_net.NNWrapper(Game, nnargs)
    nn.load_checkpoint(nn_folder, nn_file)
    gs = Game()
    pc = 0
    hist = []
    while gs.scores() is None:
        hist.append(gs.copy())
        print()
        print()
        print(gs)
        rand = eval_posistion(gs, nn)
        print()
        valids = gs.valid_moves()
        valid = False

        # grow, seed, buy, pass, by number
        # grow locaction
        # seed from locaction
        # seed to locaction

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

        def print_ref_board():
            for h in range(7):
                for w in range(7):
                    if not ((h == 0 and w == 0) or (h == 1 and w == 0) or (h == 0 and w == 1) or
                            (h == 1 and w == 1) or (h == 2 and w == 0) or (h == 0 and w == 2) or
                            (h == 6 and w == 6) or (h == 5 and w == 6) or (h == 6 and w == 5) or
                            (h == 5 and w == 5) or (h == 4 and w == 6) or (h == 6 and w == 4)):
                        print(f'{w+h*7:2d}', end=' ')
                    else:
                        print('  ', end=' ')
                print()
        valid = False
        move = 0

        while not valid:
            choice = get_input_num(
                '0-undo, 1-buy, 2-grow, 3-seed, 4-pass, 5-wild card', 5)
            if choice == 0:
                if len(hist) == 1:
                    gs = hist[-1].copy()
                    hist = []
                else:
                    gs = hist[-2].copy()
                    hist = hist[:-2]
                move = -1
                break
            if choice == 1:
                buy_choice = get_input_num(
                    '1-seed, 2-small, 3-med, 4-large', 4)
                move = 2449+buy_choice
            elif choice == 2:
                print_ref_board()
                grow_choice = get_input_num('grow tile location (0-49ish)', 49)
                move = grow_choice
            elif choice == 3:
                print_ref_board()
                from_choice = get_input_num('from tile location (0-49ish)', 49)
                to_choice = get_input_num('to tile location (0-49ish)', 49)
                move = 49 + from_choice*49 + to_choice
            elif choice == 4:
                move = 2454
            elif choice == 5:
                move = get_input_num('anything(0-2454)', 2454)
            valid = valids[move]
            if not valid:
                print(f'Move {move} is sad :(')
            print()

        if move != -1:
            gs.play_move(move)

        # selection = -1
        # while not valid:
        #     try:
        #         print('Enter Move: ', end='')
        #         selection = int(input())
        #         if 0 <= selection < len(valids):
        #             valid = valids[selection]
        #         else:
        #             raise Exception('Sad')
        #     except KeyboardInterrupt:
        #         exit()
        #     except:
        #         print('You suck at typing numbers. Get Gut!')
        # gs.play_move(selection)

        # if rand == 2454:
        #     pc += 1
        # gs.play_move(rand)
    print(gs)
    print(gs.scores())
    print('Passes:', pc)
    print('BranchFactor:', bf/bfc)
    print(bf)
    print(bfc)
