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
CPUCT = 2
TEMP = 0.5


def eval_posistion(gs, agent):
    mcts = alphazero.MCTS(CPUCT, gs.num_players(), gs.num_moves())
    v, pi = agent.predict(torch.from_numpy(gs.canonicalized()))
    v = v.cpu().numpy()
    pi = pi.cpu().numpy()
    print(f'\tRaw Score: {v}')
    print(f'\tRaw Probs: {pi}')
    print(f'\tRaw Best: {np.argmax(pi)}')
    print(f'\tRaw Rand: {np.random.choice(pi.shape[0], p=pi)}')
    start = time.time()
    while time.time() - start < THINK_TIME:
        leaf = mcts.find_leaf(gs)
        v, pi = agent.predict(torch.from_numpy(leaf.canonicalized()))
        v = v.cpu().numpy()
        pi = pi.cpu().numpy()
        mcts.process_result(gs, v, pi, False)
    print(f'\tMCTS Value Current Player: {mcts.root_value()}')
    print(f'\tMCTS Counts: {mcts.counts()}')
    probs = mcts.probs(TEMP)
    print(f'\tMCTS Probs: {probs}')
    print(f'\tMCTS Best: {np.argmax(probs)}')
    print(f'\tMCTS Rand: {np.random.choice(probs.shape[0], p=probs)}')


if __name__ == '__main__':
    import neural_net
    import game_runner

    np.set_printoptions(precision=3, suppress=True)
    Game = alphazero.Connect4GS
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
    while gs.scores() is None:
        print(gs)
        print('NN: ')
        eval_posistion(gs, nn)
        print()
        print('Which Move? ')
        gs.play_move(int(input()))
