import glob
import importlib.util
import os
import torch
import numpy as np

src_path = os.path.dirname(os.path.realpath(__file__))
build_path = os.path.join(os.path.dirname(src_path), 'build/src')
lib_path = glob.glob(os.path.join(build_path, 'alphazero*.so'))[0]

spec = importlib.util.spec_from_file_location(
    'alphazero', lib_path)
alphazero = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alphazero)

CPUCT = 2
TEMP = 0.5


def eval_posistion(gs, agent, depth):
    mcts = alphazero.MCTS(CPUCT, gs.num_moves())
    v, pi = agent.predict(torch.from_numpy(gs.canonicalized()))
    pi = np.multiply(pi.cpu().numpy(), gs.valid_moves())
    pi /= np.sum(pi)
    print(f'\tRaw Score: {v.cpu().numpy()}')
    print(f'\tRaw Probs: {pi}')
    for _ in range(depth):
        leaf = mcts.find_leaf(gs)
        v, pi = agent.predict(torch.from_numpy(leaf.canonicalized()))
        mcts.process_result(gs, v.cpu().numpy(), pi.cpu().numpy(), False)
    print(f'\tMCTS Counts: {mcts.counts()}')
    print(f'\tMCTS Probs: {mcts.probs(TEMP)}')


if __name__ == '__main__':
    import neural_net
    import game_runner

    np.set_printoptions(precision=3, suppress=True)
    Game = alphazero.Connect4GS
    nn_folder = 'data/checkpoint'
    nn_file = os.path.basename(sorted(glob.glob(os.path.join(nn_folder,'*.pt')))[-1])
    print(f'Using network: {nn_file}')
    depth = 5
    channels = 32
    nnargs = neural_net.NNArgs(num_channels=channels, depth=depth)
    nn = neural_net.NNWrapper(Game, nnargs)
    nn.load_checkpoint(nn_folder, nn_file)
    rand = game_runner.RandPlayer(Game, 1)
    gs = Game()
    while gs.scores() is None:
        print(gs)
        print('MCTS-400: ')
        eval_posistion(gs, rand, 400)
        print('NN-125: ')
        eval_posistion(gs, nn, 125)
        print()
        print('Which Move? ')
        gs.play_move(int(input()))
