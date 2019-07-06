import os

import numpy as np
from matplotlib import pyplot as plt
import time

FLAG_VERBOSE = False
PLOT_DELAY = 0.0001

def dist_L2(v1, m1, w=None):
    """
    Computes the L2 distance between vector v1 and the rows of matrix m1 with columns normalized by w
    """
    if w is None:
        w = np.ones(v1.shape)

    return np.sum(((m1 - v1) * w)**2, axis=1)


def calc_weights_unit_var(X, Y):
    """
    Calculate the distance weights as 1/std(val) to normalize X and Y to unit variances.
    """
    std_x = np.std(X, axis=0)
    std_y = np.std(Y, axis=0)

    w_unit_var = np.append(1/std_x, 1/std_y)

    w_unit_var[np.isinf(w_unit_var)] = 0

    w_imp = w_unit_var
    w_imp = w_imp / np.sum(w_imp)

    return w_unit_var, w_imp


def calc_weights_x_y_tradeoff(X, Y, pct=0.99):
    """
    Create weights for the distance function that
    """
    std_x = np.std(X, axis=0)
    std_y = np.std(Y, axis=0)

    w_cols = np.append((pct) / std_x, (1 - pct) / std_y)

    w_imp = w_cols / np.sum(w_cols)

    return w_cols, w_imp


def calc_weights_singlex(X, Y, ind_x=0):
    """
    Create weights for the distance function that
    """
    std_x = np.std(X, axis=0)

    w_cols = np.zeros(X.shape[-1] + Y.shape[-1])
    w_cols[ind_x] = 1 / std_x[ind_x]

    w_imp = w_cols / np.sum(w_cols)

    return w_cols, w_imp


def calc_weights_singley(X, Y, ind_y=0):
    """
    Create weights for the distance function that
    """
    std_x = np.std(X, axis=0)
    std_y = np.std(Y, axis=0)

    w_cols = np.zeros(X.shape[-1] + Y.shape[-1])
    w_cols[X.shape[-1] + ind_y] = 1 / std_y[ind_y]

    w_imp = w_cols / np.sum(w_cols)

    return w_cols, w_imp


class KennardStone(object):
    def __init__(self, num_datapoints_max, num_input_features, num_target_features):
        self.ind_max = num_datapoints_max - 1
        self.num_x = num_input_features
        self.num_y = num_target_features
        self.num_cols = self.num_x + self.num_y

        self.dataset = np.zeros((num_datapoints_max, self.num_cols))

        self.ind_curr = 0

    def select_subset(self, dataset_in):
        """
        Select num_target_features samples from dataset_in that maximize the Y distance
        """
        w_unit_var, _ = calc_weights_unit_var(dataset_in[:,:-1], dataset_in[:,-1:])

        # Initial search to find 2 points furthest apart
        dist_temp = dist_L2(dataset_in[0, :], dataset_in, w=w_unit_var)
        ind_max = np.argmax(dist_temp)
        ind_subset = [0, ind_max]
        dist_max = dist_temp.flatten()[ind_max]

        for ind_data in range(1, dataset_in.shape[0] - 1):
            dist_temp = dist_L2(dataset_in[ind_data, :], dataset_in[ind_data+1:, :], w=w_unit_var)
            ind_max = np.argmax(dist_temp)
            if dist_temp[ind_max] > dist_max:
                ind_subset = [ind_data, ind_data+ind_max]
                dist_max = dist_temp[ind_max]

        self.ind_curr = 2
        print('  Initial points selected')

        # Continue distance search until all slots are filled
        dist_min = dist_L2(dataset_in[ind_subset[0], :], dataset_in, w=w_unit_var).flatten()
        self.dataset[0, :] = dataset_in[ind_subset[0], :]
        self.dataset[1, :] = dataset_in[ind_subset[1], :]
        # self.plot(ind_dims=[0, -1]) # For debug

        print('  Processing dataset')
        for ind_curr in range(2, self.ind_max):
            if (ind_curr % (self.ind_max // 10)) == 0:
                print('  {0} of {1}'.format(ind_curr, self.ind_max))
            dist_temp = dist_L2(dataset_in[ind_subset[-1], :], dataset_in, w=w_unit_var)
            dist_min = np.minimum(dist_min, dist_temp, out=dist_min)

            ind_max = np.argmax(dist_min)
            ind_subset.append(ind_max)
            self.dataset[ind_curr, :] = dataset_in[ind_subset[-1], :]
            self.ind_curr = ind_curr
            # self.plot(ind_dims=[0, -1]) # For debug

    def get_sample_weights(self):
        """
        Returns sample weights
        """
        return np.ones(self.ind_curr)

    def get_dataset(self):
        """
        Returns X and Y values from current dataset
        """
        return self.dataset[:self.ind_curr, :self.num_x], \
               self.dataset[:self.ind_curr, self.num_x:(self.ind_features + 1)]

    def plot(self, title_in='KS', ind_dims=[0, 1], fig_num=1, b_save_fig=False, title_save='KS_plot'):
        """
        Plots distribution in up to 3 dimensions
        """
        plt.figure(fig_num)
        plt.cla()
        plt.title(title_in)
        # plt.ylim((-3, 11))
        # plt.xlim((-7, 7))
        plt.scatter(self.dataset[:self.ind_curr, ind_dims[0]], self.dataset[:self.ind_curr, ind_dims[1]])
        plt.xlabel('Data Column {0}'.format(ind_dims[0]))
        plt.ylabel('Data Column {0}'.format(ind_dims[1]))
        plt.grid(True)
        plt.tight_layout()

        if b_save_fig:
            plt.savefig(os.path.join(os.path.dirname(__file__), 'results', 'img',
                                     '{0}_{1}.png'.format(title_save, self.total_processed)))
        else:
            plt.draw()
            plt.pause(PLOT_DELAY / 2)

    def plot_hist(self, title_in='ODQ Features', fig_num=10, b_save_fig=False, title_save='ODQ_hist'):
        """
        Plot histrograms of each parameter distribution
        """
        if not(hasattr(self, 'total_processed')):
            self.total_processed = 0

        n_fig = ((self.ind_features + 1) // 12) + 1

        ind_feature = 0

        list_figs = []
        list_axs = []

        for ind_fig in range(n_fig):
            plt.figure(fig_num+ind_fig)
            plt.clf()
            temp_fig, temp_ax = plt.subplots(num=(fig_num+ind_fig), nrows=3, ncols=4)
            list_figs.append(temp_fig)
            list_axs.append(temp_ax)

            # list_figs[ind_fig].suptitle(title_in)

            for subplot_row in list_axs[ind_fig]:
                for subplot_el in subplot_row:
                    subplot_el.hist(self.dataset[:, ind_feature], bins=20, density=True)
                    subplot_el.set_xlabel('Feature {0}'.format(ind_feature))
                    ind_feature += 1
                    if ind_feature >= self.ind_features:
                        break
                if ind_feature >= self.ind_features:
                    break

            plt.tight_layout()

            if b_save_fig:
                plt.savefig(os.path.join(os.path.dirname(__file__), 'results', 'img',
                                         '{0}_hist_{1}_{2}.png'.format(title_save, ind_fig, self.total_processed)))
            else:
                plt.draw()
                plt.pause(PLOT_DELAY / 2)

            if ind_feature >= self.ind_features:
                break


if __name__ == '__main__':
    plt.ion()
    N_dataset = 10000
    N_saved   = 1000
    FLAG_PLOT = False
    N_dim = 30

    np.random.seed(1234)

    # Generate arbitrary dataset distribution
    dataset_raw = np.random.randn(N_dataset, N_dim)
    dataset_full = np.zeros(dataset_raw.shape)
    dataset_alpha = np.random.rand(N_dataset)
    for datapoint_full, datapoint_raw, alpha in zip(dataset_full, dataset_raw, dataset_alpha):
        if alpha < 0.4:
            datapoint_full[0] = 1 + 2*datapoint_raw[0]
            datapoint_full[-1] = 4 + 2*datapoint_raw[1]
        elif alpha < 0.5:
            datapoint_full[0] = 4 + 0.25*datapoint_raw[0]
            datapoint_full[-1] = 5 + 1.25*datapoint_raw[1]
        elif alpha < 0.75:
            datapoint_full[0] = 3 + 0.6*datapoint_raw[0] + 1*datapoint_raw[1]
            datapoint_full[-1] = 3 + datapoint_raw[1]
        elif alpha <= 1:
            datapoint_full[0] = 0 + 0.7*datapoint_raw[0] + 0.5*datapoint_raw[1]
            datapoint_full[-1] = 1 + 0.7*datapoint_raw[1]

    # Run N points of the distribution through ODQ
    quantizer = KennardStone(N_saved, N_dim - 1, 1)

    time_start = time.time()
    quantizer.select_subset(dataset_full)
    time_end = time.time()

    quantizer.plot(ind_dims=[0, -1])

    if FLAG_PLOT:
        # Plot ODQ, full dataset, and uniform random sampling of dataset
        quantizer.plot()

        # Full dataset
        plt.figure(2)
        plt.title('Full Dataset')
        plt.ylim((-3, 11))
        plt.xlim((-7, 7))
        plt.scatter(dataset_full[:, 0], dataset_full[:, 1], s=1)
        plt.grid(True)
        plt.draw()

        # Uniform random selection
        ind_random = np.random.choice(N_dataset, np.round(1.2*N_saved).astype(int))
        plt.figure(3)
        plt.title('Uniform Random Sampling')
        plt.ylim((-3, 11))
        plt.xlim((-7, 7))
        plt.scatter(dataset_full[ind_random, 0], dataset_full[ind_random, 1], s=1)
        plt.grid(True)
        plt.draw()
        plt.pause(1000)

