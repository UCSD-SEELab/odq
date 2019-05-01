import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    N_w = 10

    w1 = np.random.rand(N_w)

    w2 = np.exp(w1)

    w3 = np.exp(2*w1)

    w4 = np.exp(5*w1)

    w5 = np.exp(10*w1)

    w6 = np.exp(w1 ** 2)

    w_all = np.stack((w1/sum(w1), w2/sum(w2), w3/sum(w3), w4/sum(w4), w5/sum(w5), w6/sum(w6)))

    ind_w = list(range(w_all.shape[0]))
    plt.xticks(ind_w, ['x', 'exp(x)', 'exp(2x)', 'exp(5x)', 'exp(10x)', 'exp(x^2)'])

    for ind_col in range(w_all.shape[1]):
        w_offset = np.sum(w_all[:, :ind_col], axis=1)
        plt.bar(ind_w, w_all[:, ind_col], bottom=w_offset)

    plt.show()