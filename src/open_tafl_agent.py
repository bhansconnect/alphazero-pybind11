# Used to enable the network to play against other AI in OpenTafl.
# Long term it may want to be moved to ONNX(probably in C++)

from load_lib import load_alphazero
import argparse
import neural_net
import time
import glob
import os
import torch
import numpy as np

np.set_printoptions(precision=3, suppress=True)
alphazero = load_alphazero()


def calc_temp(start_temp, end_temp, decay, turn):
    ln2 = 0.693
    ld = ln2 / decay
    temp = start_temp - end_temp
    temp *= np.exp(-ld * turn)
    temp += end_temp
    return temp


def move_to_string(move, height, width):
    new_loc = move % (width + height)
    height_move = new_loc >= width
    if height_move:
        new_loc -= width
    piece_loc = (move // (width + height))
    piece_w = piece_loc % width
    piece_h = piece_loc // width

    new_h = piece_h
    new_w = piece_w
    if height_move:
        new_h = new_loc
    else:
        new_w = new_loc
    return f"{chr(ord('a')+piece_w)}{piece_h+1}-{chr(ord('a')+new_w)}{new_h+1}"


def eval_position(gs, mcts, agent, args, time_limit):
    height = gs.CANONICAL_SHAPE()[1]
    width = gs.CANONICAL_SHAPE()[2]
    start = time.time()
    while time.time() - start < time_limit:
        leaf = mcts.find_leaf(gs)
        v, pi = agent.predict(torch.from_numpy(leaf.canonicalized()))
        v = v.cpu().numpy()
        pi = pi.cpu().numpy()
        mcts.process_result(gs, v, pi, False)

    v, pi = agent.predict(torch.from_numpy(gs.canonicalized()))
    v = v.cpu().numpy()
    pi = pi.cpu().numpy()
    high_prob = {move_to_string(x[0], height, width): pi[x[0]]
                 for x in np.argwhere(pi > 0.05)}
    print(f'status Raw win probs: {v}')
    print(f'status Raw top moves: {high_prob}')

    probs = mcts.probs(calc_temp(args.start_temp, args.end_temp,
                                 args.temp_half_life, gs.current_turn()))
    high_prob = {move_to_string(x[0], height, width): probs[x[0]]
                 for x in np.argwhere(probs > 0.05)}
    rand = np.random.choice(probs.shape[0], p=probs)
    print(
        f'status MCTS ran with {sum(mcts.counts())} simulations in {time.time()-start} seconds')
    print(f'status MCTS win-loss-draw prob: {mcts.root_value()}')
    print(f'status MCTS top moves: {high_prob}')
    print(
        f'status MCTS best: {move_to_string(np.argmax(probs), height, width)}')
    print(f'status MCTS chosen: {move_to_string(rand, height, width)}')
    return rand


def gen_move(height, width, from_h, from_w, to_h, to_w):
    if (from_h != to_h and from_w != to_w) or (from_h == to_h and from_w == to_w):
        raise Exception('Invalid move recieved')
    if from_h != to_h:
        return (from_h * width + from_w) * (width + height) + width + to_h
    return (from_h * width + from_w) * (width + height) + to_w


if __name__ == '__main__':
    print('hello', flush=True)
    parser = argparse.ArgumentParser(description='AlphaZero OpenTafl Agent')
    parser.add_argument('--folder', type=str, required=True,
                        help='The folder where networks are stored.')
    parser.add_argument('--network', type=str, default='latest',
                        help='The neural network weights file name.')
    parser.add_argument('--time', type=float, default=9.5,
                        help='Base time to think per move in seconds.')
    parser.add_argument('--start-temp', type=float, default=0.5,
                        help='How much randomness to use when picking a move at start of game. 0 is pick best move.')
    parser.add_argument('--end-temp', type=float, default=0.1,
                        help='How much randomness to use when picking a move at end of game. 0 is pick best move.')
    parser.add_argument('--temp-half-life', type=float, default=10,
                        help='How many turns for temp to become half of what it used to be.')
    parser.add_argument('--cpuct', type=float, default=2,
                        help='How much to explore when searching. Should be smaller for shorter time limits.')
    parser.add_argument('--fpu-redux', type=float, default=0.25,
                        help='First play urgency reduction')
    parser.add_argument('--game', type=str, default='computer-brandubh',
                        help='The game/ruleset being played.')
    args = parser.parse_args()

    if args.game.lower() == 'computer-brandubh':
        Game = alphazero.BrandubhGS
        name = 'brandubh'
    elif args.game.lower() == 'computer-tawlbwrdd':
        Game = alphazero.TawlbwrddGS
        name = 'tawlbwrdd'
    elif args.game.lower() == 'computer-fetlar':
        Game = alphazero.OpenTaflGS
        name = 'opentafl'
    else:
        print(f'status Unsupported ruleset: {args.game.lower()}')
        print('error -1')
        exit()

    if args.network == 'latest':
        nn_files = [os.path.basename(x) for x in sorted(
            glob.glob(os.path.join(args.folder, f'*-{name}-*.pt')))]
        args.network = nn_files[-1]
    start = time.time()
    print(
        f'status Loading network {args.network} in {os.path.abspath(args.folder)}')
    try:
        nn = neural_net.NNWrapper.load_checkpoint(
            Game, args.folder, args.network)
    except Exception as e:
        print(f'status Failed to load network: {e}')
        print('error -1')
        exit()

    print(f'status Loaded network in {time.time()-start} seconds')

    max_turns = 2**16-1
    gs = Game(max_turns)
    height = gs.CANONICAL_SHAPE()[1]
    width = gs.CANONICAL_SHAPE()[2]
    mcts = alphazero.MCTS(args.cpuct, gs.num_players(),
                          gs.num_moves(), 0, 1.4, args.fpu_redux)

    time_limit = args.time
    try:
        while True:
            command = input().strip()
            if command.startswith('play'):
                move = eval_position(gs, mcts, nn, args, time_limit)
                print(
                    f'move {move_to_string(move, width, height)}', flush=True)
                mcts.update_root(gs, move)
                gs.play_move(move)
            elif command.startswith('opponent-move'):
                move = command.split()[1]
                from_loc, to_loc = move.split('-')
                from_w = ord(from_loc[0]) - ord('a')
                from_h = int(from_loc[1:]) - 1
                to_w = ord(to_loc[0]) - ord('a')
                to_h = int(to_loc[1:]) - 1
                move = gen_move(height, width, from_h, from_w, to_h, to_w)
                mcts.update_root(gs, move)
                gs.play_move(move)
            elif command.startswith('finish'):
                gs = Game()
                mcts = alphazero.MCTS(
                    args.cpuct, gs.num_players(), gs.num_moves(), 0, 1.4, args.fpu_redux)
            elif command.startswith('clock'):
                # Grab the over time length.
                # Subtract 200ms to play it safe.
                time_limit = float(command.split()[3]) - 0.2
            elif command.startswith('move'):
                continue
            elif command.startswith('error'):
                print('error -1')
                break
            elif command.startswith('goodbye'):
                break
            else:
                print(f'status Unused/Unknown Command: "{command}"')
    except Exception as e:
        print(f'status Failure: {e}')
        print('error -1')
        exit()
