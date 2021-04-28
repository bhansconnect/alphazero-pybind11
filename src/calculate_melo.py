# This is used to post hoc calculate a more accurate elo for the agents.
# It understands that there can be cycles that effect performance.
# It tends to rate overfit agents much lower.
import numpy as np
import tqdm
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    wr = np.genfromtxt('data/win_rate.csv', delimiter=',')

    size = wr.shape[0]
    agents = set()
    matchups = set()
    for i in range(wr.shape[0]):
        for j in range(i+1,wr.shape[0]):
            if not math.isnan(wr[i, j]):
                agents.add(i)
                matchups.add((i, j))

    alpha = math.log(10)/400

    def sig(x):
        return 1.0 / (1.0 + math.exp(-alpha*x))

    def melo2_update(K, i, j, p_ij, r, c):
        p_hat_ij = sig(r[i] - r[j] + c[i, 0] * c[j, 1] - c[j, 0] * c[i, 1])
        delta = (p_ij - p_hat_ij) * K
        r_update = [16 * delta, -16 * delta]
        # r has higher learning rate than c
        c_update = [
            [+delta * c[j, 1], -delta * c[j, 0]],
            [-delta * c[i, 1], +delta * c[i, 0]]
        ]
        return r_update, c_update

    melo = np.zeros(wr.shape[0])
    c = np.random.uniform(-1, 1, (wr.shape[0], 2))
    rounds = 10000
    with tqdm.tqdm(range(rounds)) as pbar:
        for it in pbar:
            if it == 1:
                print(flush=True)
            change_elo = np.zeros_like(melo)
            change_c = np.zeros_like(c)
            for (i, j) in matchups:
                melo_update, c_update = melo2_update(
                    1, i, j, wr[i, j], melo, c)
                melo[i] += melo_update[0]
                melo[j] += melo_update[1]
                c[i] += c_update[0]
                c[j] += c_update[1]
                change_elo[i] += np.abs(melo_update[0])
                change_elo[j] += np.abs(melo_update[1])
                change_c[i] += np.abs(c_update[0])
                change_c[j] += np.abs(c_update[1])
            # Weight decay of c. This makes the algorithm converge to elo if their are no cycles
            c *= 0.995
            if it % 100 == 0:
                pbar.set_postfix(elo_cost=np.average(change_elo),
                                 c_cost=np.average(change_c))

    melo -= melo[0]
    np.savetxt("data/melo.csv", melo, delimiter=",")
    tmp_melo = np.zeros(len(agents))
    tmp_c = np.zeros((len(agents),2))
    tmp_melo[:len(agents)-1] = melo[:len(agents)-1]
    tmp_melo[len(agents)-1] = melo[wr.shape[0]-1]
    tmp_c[:len(agents)-1] = c[:len(agents)-1]
    tmp_c[len(agents)-1] = c[wr.shape[0]-1]
    print(tmp_c)
    print(tmp_melo)

    fig, ax1 = plt.subplots()
    ax1.plot(sorted(agents)[:len(agents)-1],
             melo[:len(agents)-1], '-o', color='tab:orange')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('elo rating')
    plt.show()
