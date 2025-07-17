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

Game = alphazero.Connect4GS

bf = 0
bfc = 0


def calc_temp(turn):
    ln2 = 0.693
    ld = ln2 / TEMP_DECAY_HALF_LIFE
    temp = START_TEMP - END_TEMP
    temp *= np.exp(-ld * turn)
    temp += END_TEMP
    return temp


def eval_position(gs, agent):
    mcts = alphazero.MCTS(CPUCT, gs.num_players(), gs.num_moves(), 0, 1.4, 0.25)
    global bf
    global bfc
    bf += np.sum(gs.valid_moves())
    bfc += 1
    start = time.time()
    sims = 0
    while time.time() - start < THINK_TIME:
        # for _ in range(200):
        leaf = mcts.find_leaf(gs)
        v, pi = agent.predict(torch.from_numpy(leaf.canonicalized()))
        v = v.cpu().numpy()
        pi = pi.cpu().numpy()
        mcts.process_result(gs, v, pi, False)
        sims += 1
    print(f"\tRan {sims} simulations in {round(time.time() - start, 3)} seconds")
    # print('Press enter for ai analysis')
    # input()
    v, pi = agent.predict(torch.from_numpy(gs.canonicalized()))
    v = v.cpu().numpy()
    pi = pi.cpu().numpy()
    print(f"\tRaw Score: {v}")
    thing = {x: round(100 * pi[x], 1) for x in reversed(np.argsort(pi)[-10:])}
    print(f"\tRaw Top Probs: {thing}")
    print(f"\tRaw Best: {np.argmax(pi)}")
    print(f"\tRaw Rand: {np.random.choice(pi.shape[0], p=pi)}")

    print(f"\tMCTS Current Player WLD: {mcts.root_value()}")
    counts = mcts.counts()
    thing = {x: counts[x] for x in reversed(np.argsort(counts)[-10:])}
    print(f"\tMCTS Top Counts: {thing}")
    probs = mcts.probs(calc_temp(gs.current_turn()))
    thing = {x: round(100 * probs[x], 1) for x in reversed(np.argsort(probs)[-10:])}
    print(f"\tMCTS Top Probs: {thing}")
    print(f"\tMCTS Best: {np.argmax(probs)}")
    rand = np.random.choice(probs.shape[0], p=probs)
    print(f"\tMCTS Rand: {rand}")
    return rand


if __name__ == "__main__":
    import neural_net

    np.set_printoptions(precision=3, suppress=True)
    nn_folder = os.path.join("data", "checkpoint")
    nn_file = os.path.basename(sorted(glob.glob(os.path.join(nn_folder, "*.pt")))[-1])

    print(f"Using network: {nn_file}")
    nn = neural_net.NNWrapper.load_checkpoint(Game, nn_folder, nn_file)
    gs = Game()
    pc = 0
    hist = []
    while gs.scores() is None:
        hist.append(gs.copy())
        print()
        print()
        print(gs)
        rand = eval_position(gs, nn)
        print()
        valids = gs.valid_moves()
        valid = False

        def get_input_num(stuff, max_stuff):
            valid = False
            while not valid:
                try:
                    print(f"Enter {stuff}: ", end="")
                    selection = int(input())
                    if -1 <= selection <= max_stuff:
                        return selection
                    else:
                        raise Exception("Sad")
                except KeyboardInterrupt:
                    exit()
                except ValueError:
                    print("You suck at typing numbers. Get Gud!")

        HEIGHT = Game.CANONICAL_SHAPE()[1]
        WIDTH = Game.CANONICAL_SHAPE()[2]

        def print_ref_board():
            for h in range(7):
                for w in range(7):
                    if not (
                        (h == 0 and w == 0)
                        or (h == 1 and w == 0)
                        or (h == 0 and w == 1)
                        or (h == 1 and w == 1)
                        or (h == 2 and w == 0)
                        or (h == 0 and w == 2)
                        or (h == 6 and w == 6)
                        or (h == 5 and w == 6)
                        or (h == 6 and w == 5)
                        or (h == 5 and w == 5)
                        or (h == 4 and w == 6)
                        or (h == 6 and w == 4)
                    ):
                        print(f"{w + h * 7:2d}", end=" ")
                    else:
                        print("  ", end=" ")
                print()

        valid = False
        move = 0

        # random
        gs.play_move(rand)

        # onitama
        # while not valid:
        #     P0_MASTER_LAYER = 0
        #     P0_PAWN_LAYER = 1
        #     P1_MASTER_LAYER = 2
        #     P1_PAWN_LAYER = 3
        #     p = gs.current_player()
        #     if p == 0:
        #         card0, card1 = gs.p0_card0(), gs.p0_card1()
        #         master_layer, pawn_layer = P0_MASTER_LAYER, P0_PAWN_LAYER
        #     else:
        #         card0, card1 = gs.p1_card0(), gs.p1_card1()
        #         master_layer, pawn_layer = P1_MASTER_LAYER, P1_PAWN_LAYER
        #     choice = get_input_num(
        #         f'0-undo, 1-{card0.name}, 2-{card1.name}, 3-wild card', 3)
        #     if choice == 0:
        #         if len(hist) == 1:
        #             gs = hist[-1].copy()
        #             hist = []
        #         else:
        #             gs = hist[-2].copy()
        #             hist = hist[:-2]
        #         move = -1
        #         break
        #     elif choice == 3:
        #         move = get_input_num('anything(0-1252)', 1252)
        #     else:
        #         # Playing a card. Need to pick a piece.
        #         if choice == 1:
        #             card = card0
        #         else:
        #             card = card1
        #         board = gs.canonicalized()
        #         pieces = []
        #         print()
        #         for h in range(HEIGHT):
        #             for w in range(WIDTH):
        #                 if board[master_layer, h, w] or board[pawn_layer, h, w]:
        #                     print(f'{len(pieces)}', end='')
        #                     pieces.append((h, w))
        #                 else:
        #                     print('.', end='')
        #             print()
        #         piece = get_input_num(
        #             f'select piece(0-{len(pieces)-1})', len(pieces)-1)
        #         if piece == -1:
        #             print()
        #             print(gs)
        #             continue
        #         base = pieces[piece]
        #         base_h, base_w = base
        #         squares = []
        #         for (h, w) in card.movements:
        #             if p == 1:
        #                 h *= -1
        #                 w *= -1
        #             h += base_h
        #             w += base_w
        #             if 0 <= h < HEIGHT and 0 <= w < WIDTH and board[master_layer, h, w] == 0 and board[pawn_layer, h, w] == 0:
        #                 squares.append((h, w))
        #         if len(squares) == 0:
        #             print()
        #             print("=== No valid moves with chosen card and piece ===")
        #             print()
        #             print(gs)
        #             continue
        #         print()
        #         for h in range(HEIGHT):
        #             for w in range(WIDTH):
        #                 found = False
        #                 for i, (to_h, to_w) in enumerate(squares):
        #                     if h == to_h and w == to_w:
        #                         print(f'{i}', end='')
        #                         found = True
        #                 if not found:
        #                     print('.', end='')
        #             print()
        #         target = get_input_num(
        #             f'select target(0-{len(squares)-1})', len(squares)-1)
        #         if target == -1:
        #             print()
        #             print(gs)
        #             continue
        #         target_h, target_w = squares[target]
        #         print(base)
        #         print((target_h, target_w))
        #         move = (choice-1) * (HEIGHT * WIDTH * HEIGHT * WIDTH) + base_h * (WIDTH *
        #                                                                           HEIGHT * WIDTH) + base_w * (HEIGHT * WIDTH) + target_h * WIDTH + target_w
        #     valid = valids[move]
        #     if not valid:
        #         print(f'Move {move} is not valid for some reason :(')
        #         print()
        #         print(gs)
        # if move == -1:
        #     continue
        # gs.play_move(move)
        # continue

        # photosynthesis
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

        # def gen_move(from_h, from_w, h_move, to_loc):
        #     if h_move:
        #         return (from_h * WIDTH + from_w) * (WIDTH + HEIGHT) + WIDTH + to_loc
        #     return (from_h * WIDTH + from_w) * (WIDTH + HEIGHT) + to_loc

        # generic
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

    print(gs)
    print(gs.scores())
    print("Passes:", pc)
    print("BranchFactor:", bf / bfc)
    print(bf)
    print(bfc)
