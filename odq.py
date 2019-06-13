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


def calc_weights_incorrect(X, Y):
    """
    Calculate the distance weights between the input features and targets. The weights are related to the maximum
    covariance between a column of X and any column of y, normalized by the variances of X and Y, and the variance of
    X to normalize for input scale.
    """
    std_x = np.std(X, axis=0)
    std_y = np.std(Y, axis=0)
    cov_max = np.zeros(X.shape[1])

    Y_centered = Y - np.mean(Y, axis=0)
    X_centered = X - np.mean(X, axis=0)

    for ind_x in range(X.shape[1]):
        cov_temp = np.matmul(X_centered[:, ind_x:ind_x+1].transpose(), Y_centered) / X.shape[0] / std_x[ind_x] / std_y
        cov_max[ind_x] = np.max(np.abs(cov_temp))

    range_x = np.max(X) - np.min(X)
    range_y = np.max(Y) - np.min(Y)

    w_max_cov = np.append((1 - cov_max)/std_x/range_x, (1 - np.max(cov_max))/std_y/range_y)

    w_imp = np.append((1 - cov_max)/range_x, (1 - np.max(cov_max))/range_y)
    w_imp = w_imp / np.sum(w_imp)

    return w_max_cov, w_imp


def calc_weights_max_cov2(X, Y):
    """
    Calculate the distance weights between the input features and targets. The weights are related to the maximum
    covariance between a column of X and any column of y, normalized by the variances of X and Y, and the variance of
    X to normalize for input scale.
    """
    std_x = np.std(X, axis=0)
    std_y = np.std(Y, axis=0)
    cov_max = np.zeros(X.shape[1])

    Y_centered = Y - np.mean(Y, axis=0)
    X_centered = X - np.mean(X, axis=0)

    for ind_x in range(X.shape[1]):
        cov_temp = np.matmul(X_centered[:, ind_x:ind_x+1].transpose(), Y_centered) / X.shape[0] / std_x[ind_x] / std_y
        cov_max[ind_x] = np.max(np.abs(cov_temp))

    w_cols = cov_max
    w_max_cov = np.append(w_cols/std_x, np.max(w_cols)/std_y)

    w_imp = np.append(w_cols, np.max(w_cols))
    w_imp = w_imp / np.sum(w_imp)

    return w_max_cov, w_imp


def calc_weights_max_cov(X, Y):
    """
    Calculate the distance weights between the input features and targets. The weights are related to the maximum
    covariance between a column of X and any column of y, normalized by the variances of X and Y, and the variance of
    X to normalize for input scale.
    """
    std_x = np.std(X, axis=0)
    std_y = np.std(Y, axis=0)
    cov_max = np.zeros(X.shape[1])

    Y_centered = Y - np.mean(Y, axis=0)
    X_centered = X - np.mean(X, axis=0)

    for ind_x in range(X.shape[1]):
        cov_temp = np.matmul(X_centered[:, ind_x:ind_x+1].transpose(), Y_centered) / X.shape[0] / std_x[ind_x] / std_y
        cov_max[ind_x] = np.max(np.abs(cov_temp))

    w_cols = np.sqrt(cov_max)
    w_max_cov = np.append(w_cols/std_x, np.max(w_cols)/std_y)

    w_imp = np.append(w_cols, np.max(w_cols))
    w_imp = w_imp / np.sum(w_imp)

    return w_max_cov, w_imp


def calc_weights_max_norm(X, Y):
    """
    Calculate the distance weights between the input features and targets. The weights are related to the maximum
    covariance between a column of X and any column of y, normalized by the variances of X and Y, and the variance of
    X to normalize for input scale.
    """
    std_x = np.std(X, axis=0)
    std_y = np.std(Y, axis=0)
    cov_max = np.zeros(X.shape[1])

    Y_centered = Y - np.mean(Y, axis=0)
    X_centered = X - np.mean(X, axis=0)

    for ind_x in range(X.shape[1]):
        cov_temp = np.matmul(X_centered[:, ind_x:ind_x+1].transpose(), Y_centered) / X.shape[0] / std_x[ind_x] / std_y
        cov_max[ind_x] = np.max(np.abs(cov_temp))

    w_cols = cov_max**2 / np.sum(cov_max**2)

    w_max_cov = np.append(w_cols/std_x, np.max(w_cols)/std_y)

    w_imp = np.append(w_cols, np.max(w_cols))
    w_imp = w_imp / np.sum(w_imp)

    return w_max_cov, w_imp


def calc_weights_max_cov_logit(X, Y):
    """
    Calculate the distance weights between the input features and targets. The weights are related to the maximum
    covariance between a column of X and any column of y, normalized by the variances of X and Y, and the variance of
    X to normalize for input scale.
    """
    std_x = np.std(X, axis=0)
    std_y = np.std(Y, axis=0)
    cov_max = np.zeros(X.shape[1])

    Y_centered = Y - np.mean(Y, axis=0)
    X_centered = X - np.mean(X, axis=0)

    for ind_x in range(X.shape[1]):
        cov_temp = np.matmul(X_centered[:, ind_x:ind_x+1].transpose(), Y_centered) / X.shape[0] / std_x[ind_x] / std_y
        cov_max[ind_x] = np.max(np.abs(cov_temp))

    w_cols = (np.exp(cov_max) - 1)

    w_max_cov = np.append(w_cols/std_x, np.max(w_cols)/std_y)

    w_imp = np.append(w_cols, np.max(w_cols))
    w_imp = w_imp / np.sum(w_imp)

    return w_max_cov, w_imp


def calc_weights_max_cov_gauss(X, Y):
    """
    Calculate the distance weights between the input features and targets. The weights are related to the maximum
    covariance between a column of X and any column of y, normalized by the variances of X and Y, and the variance of
    X to normalize for input scale.
    """
    std_x = np.std(X, axis=0)
    std_y = np.std(Y, axis=0)
    cov_max = np.zeros(X.shape[1])

    Y_centered = Y - np.mean(Y, axis=0)
    X_centered = X - np.mean(X, axis=0)

    for ind_x in range(X.shape[1]):
        cov_temp = np.matmul(X_centered[:, ind_x:ind_x+1].transpose(), Y_centered) / X.shape[0] / std_x[ind_x] / std_y
        cov_max[ind_x] = np.max(np.abs(cov_temp))

    w_cols = (np.exp(cov_max**2) - 1)

    w_max_cov = np.append(w_cols/std_x, np.max(w_cols)/std_y)
    w_imp = np.append(w_cols, np.max(w_cols))
    w_imp = w_imp / np.sum(w_imp)

    return w_max_cov, w_imp


def calc_weights_unit_var(X, Y):
    """
    Calculate the distance weights as 1/std(val) to normalize X and Y to unit variances.
    """
    std_x = np.std(X, axis=0)
    std_y = np.std(Y, axis=0)
    w_unit_var = np.append(1/std_x, 1/std_y)

    w_imp = np.ones((X.shape[-1], Y.shape[-1]))
    w_imp = w_imp / np.sum(w_imp)

    return w_unit_var, w_imp


def calc_weights_pr_squeeze(X, Y, depth=4):
    """
    Calculate the distance weights using a squeezed version based on projection pursuit
    """
    std_x = np.std(X, axis=0)
    std_y = np.std(Y, axis=0)

    Y_centered = Y - np.mean(Y, axis=0)
    X_centered = X - np.mean(X, axis=0)

    list_ind_sel = []

    # Perform projection pursuit to determine most important features to use for weight determination
    Y_res = Y_centered
    for ind_d in range(depth):
        # print('  Unexplained: {0:0.2f}'.format(np.sqrt(np.sum(Y_res[:, ]**2))))
        cov_max = np.zeros(X.shape[1])
        for ind_x in range(X.shape[1]):
            # If already selected for use, skip and continue
            if ind_x in list_ind_sel:
                continue
            cov_temp = np.matmul(X_centered[:, ind_x:ind_x+1].transpose(), Y_centered) / X.shape[0] / std_x[ind_x] / std_y
            cov_max[ind_x] = np.max(np.abs(cov_temp))

        ind_max = np.argmax(cov_max)
        coef_prog = np.dot(Y_res.transpose(), X_centered[:, ind_max]) / np.dot(X_centered[:, ind_max].transpose(), X_centered[:, ind_max])
        Y_exp = np.stack([coef * X_centered[:, ind_max] for coef in coef_prog], axis=-1)
        Y_res = Y_res - Y_exp
        list_ind_sel.append(ind_max)

    cov_max = np.zeros(X.shape[1])
    for ind_sel in list_ind_sel:
        cov_temp = np.matmul(X_centered[:, ind_sel:ind_sel + 1].transpose(), Y_centered) / X.shape[0] / std_x[ind_sel] / std_y
        cov_max[ind_sel] = np.max(np.abs(cov_temp))

    w_cols = cov_max / np.sum(cov_max)

    w_pr_squeeze = np.append(w_cols/std_x, np.max(w_cols)/std_y)

    w_imp = np.append(w_cols, np.max(w_cols))
    w_imp = w_imp / np.sum(w_imp)

    return w_pr_squeeze, w_imp


def calc_weights_imbalanced(X, Y):
    """
    Create weights for the distance function that are imbalanced
    """
    std_x = np.std(X, axis=0)
    std_y = np.std(Y, axis=0)

    w_cols = np.exp(10*np.random.rand(X.shape[-1] + Y.shape[-1]))

    w_imp = w_cols / np.sum(w_cols)

    return w_cols, w_imp


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


class OnlineDatasetQuantizer(object):
    def __init__(self, num_datapoints_max, num_input_features, num_target_features, b_save_dist=True, b_save_density=True, w_x_columns=None, w_y_columns=None):
        self.ind_max = num_datapoints_max - 1
        self.num_x = num_input_features
        self.num_y = num_target_features
        self.b_dist = b_save_dist
        self.b_density = b_save_density
        if w_x_columns is None:
            w_x_columns = np.ones(self.num_x)
        if w_y_columns is None:
            w_y_columns = np.ones(self.num_y)
        self.w = np.append(w_x_columns, w_y_columns)

        self.ind_features = self.num_x + self.num_y - 1

        if self.b_dist:
            self.ind_dist = self.ind_features + 1
        else:
            self.ind_dist = self.ind_features

        if self.b_density:
            self.ind_density = self.ind_dist + 1
        else:
            self.ind_density = self.ind_dist

        self.num_cols = self.ind_density + 1

        self.dataset = np.zeros((num_datapoints_max, self.num_cols))
        self.ind_curr = 0

        self.dist_min = 100 # TODO initialize minimum distance properly

        self.total_processed = 0

    def add_point(self, x, y):
        """
        Add current (x, y) pairing to dataset
        """
        self.total_processed += 1

        # TODO Check that dimensions are correct

        datapoint = np.append(x, y)

        # plt.figure(1)
        # plt.scatter(datapoint[0], datapoint[1], c='r', marker='x')
        # plt.draw()
        # plt.pause(PLOT_DELAY/2)

        # If first point,
        if self.ind_curr == 0:
            self.dataset[self.ind_curr, :(self.ind_features + 1)] = datapoint
            if self.b_dist:
                self.dataset[self.ind_curr, self.ind_dist] = np.inf
            if self.b_density:
                self.dataset[self.ind_curr, self.ind_density] = 1
            self.ind_curr = self.ind_curr + 1
            return

        # If saving distance calculations in metainformation,
        if self.b_dist:
            # Calculate distance between new point and current dataset based on selected distance metric and replace all
            # distances with minimum distance
            temp_dist = dist_L2(datapoint, self.dataset[:self.ind_curr, :(self.ind_features + 1)], w=self.w)
        else:
            # Calculate all possible distance calculations and determine minimum
            # TODO Perform brute force distance calculations
            return

        temp_ind_min = np.argmin(temp_dist)
        # If memory is not yet full,
        if self.ind_curr <= self.ind_max:
            # Update distance measurements
            if self.b_dist:
                ind_update = np.where(self.dataset[:self.ind_curr, self.ind_dist] > temp_dist)
                self.dataset[ind_update, self.ind_dist] = temp_dist[ind_update]

            # Add datapoint directly to dataset
            self.dataset[self.ind_curr, :(self.ind_features + 1)] = datapoint
            if self.b_dist:
                self.dataset[self.ind_curr, self.ind_dist] = temp_dist[temp_ind_min]
            if self.b_density:
                self.dataset[self.ind_curr, self.ind_density] = 1
            self.ind_curr = self.ind_curr + 1
        else:
            # Otherwise, determine points that share a minimum distance, evict one and combine information
            if self.b_dist:
                # If dataset has two points closer than incoming point,
                if FLAG_VERBOSE:
                    print('\n\nnew point: ({0:0.2f}, {1:0.2f})'.format(datapoint[0], datapoint[1]))
                    print('min dataset: {0:0.4f}  min new datapoint: {1:0.4f}'.format(self.dataset[:, self.ind_dist].min(), temp_dist[temp_ind_min]))
                if self.dataset[:, self.ind_dist].min() < temp_dist[temp_ind_min]:
                    # Evict one and combine information of closest 2 points
                    ind_min = np.where(self.dataset[:, self.ind_dist] == self.dataset[:, self.ind_dist].min())
                    ind_min = ind_min[0]
                    while (ind_min.shape[0] < 2):
                        self.update_min_dist(ind_min[0])
                        ind_min = np.where(self.dataset[:, self.ind_dist] == self.dataset[:, self.ind_dist].min())
                        ind_min = ind_min[0]
                    if ind_min.shape[0] > 2:
                        ind_min = ind_min[0:2] # TODO Update to random integer selection

                    if FLAG_VERBOSE:
                        print('Evicting and replacing')
                        print('closest points: ({0:0.2f}, {1:0.2f}) and ({2:0.2f}, {3:0.2f})'.format(self.dataset[ind_min[0], 0],
                                                                                                     self.dataset[ind_min[0], 1],
                                                                                                     self.dataset[ind_min[1], 0],
                                                                                                     self.dataset[ind_min[1], 1]))

                    if self.b_density:
                        # Calculate a weighted average
                        self.dataset[ind_min[0], :(self.ind_features + 1)] = np.sum(self.dataset[ind_min, :(self.ind_features + 1)] * self.dataset[ind_min, self.ind_density:(self.ind_density + 1)], axis=0) / np.sum(self.dataset[ind_min, self.ind_density])
                        self.dataset[ind_min[0], self.ind_density] = np.sum(self.dataset[ind_min, self.ind_density])
                    else:
                        # Calculate a standard average
                        self.dataset[ind_min[0], :(self.ind_features + 1)] = np.sum(self.dataset[ind_min, :]) / ind_min.shape[0]

                    # Replace other point with new datapoint
                    self.dataset[ind_min[1], :(self.ind_features + 1)] = datapoint
                    if self.b_dist:
                        self.dataset[ind_min[1], self.ind_dist] = temp_dist[temp_ind_min]
                    if self.b_density:
                        self.dataset[ind_min[1], self.ind_density] = 1

                else:
                    if FLAG_VERBOSE:
                        print('Combining point with existing')
                        print('new point is close to: ({0:0.2f}, {1:0.2f})'.format(
                            self.dataset[temp_ind_min, 0],
                            self.dataset[temp_ind_min, 1]))
                    # Add information of new datapoint to existing dataset
                    if self.b_density:
                        # Calculate a weighted average
                        self.dataset[temp_ind_min, :(self.ind_features + 1)] = (self.dataset[temp_ind_min, :(self.ind_features + 1)]*self.dataset[temp_ind_min, self.ind_density] + datapoint) / (self.dataset[temp_ind_min, self.ind_density] + 1)
                        self.dataset[temp_ind_min, self.ind_density] = self.dataset[temp_ind_min, self.ind_density] + 1
                    else:
                        # Calculate a standard average
                        self.dataset[temp_ind_min, :(self.ind_features + 1)] = (self.dataset[temp_ind_min, :(self.ind_features + 1)] + datapoint) / 2

                    ind_min = [temp_ind_min]

                # Recalculate distance measurements with new datapoint in dataset
                if self.b_dist:
                    self.update_min_dist(ind_min[0])

            else:
                # Use temporary distance vector
                # TODO
                return

    def get_sample_weights(self):
        """
        Returns sample weights
        """
        if self.b_density:
            return self.dataset[:self.ind_curr, self.ind_density]
        else:
            return np.ones(self.ind_curr)

    def get_dataset(self):
        """
        Returns X and Y values from current dataset
        """
        return self.dataset[:self.ind_curr, :self.num_x], \
               self.dataset[:self.ind_curr, self.num_x:(self.ind_features + 1)]

    def update_min_dist(self, ind_min):
        """
        Update the minimum distance calculation for point at ind_update
        """
        temp_dist = dist_L2(self.dataset[ind_min, :(self.ind_features + 1)],
                            self.dataset[:(self.ind_curr + 1), :(self.ind_features + 1)], w=self.w)
        temp_dist[ind_min] = np.inf
        ind_update = np.where(self.dataset[:self.ind_curr, self.ind_dist] > temp_dist)
        self.dataset[ind_update, self.ind_dist] = temp_dist[ind_update]

        # Update distance for point that was updated
        self.dataset[ind_min, self.ind_dist] = temp_dist.min()

    def plot(self, title_in='ODQ', ind_dims=[0, 1], fig_num=1, b_save_fig=False, title_save='ODQ_hist'):
        """
        Plots distribution in up to 3 dimensions
        """
        if not(hasattr(self, 'total_processed')):
            self.total_processed = 0

        # TODO Add ind_dims to input variables and update plots to adapt to ind_dims

        plt.figure(fig_num)
        plt.cla()
        plt.title(title_in)
        # plt.ylim((-3, 11))
        # plt.xlim((-7, 7))
        if self.b_density:
            plt.scatter(self.dataset[:self.ind_curr, ind_dims[0]], self.dataset[:self.ind_curr, ind_dims[1]], s=1 + (0.2 * (self.dataset[:self.ind_curr, self.ind_density] - 1)))
        else:
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
    quantizer = OnlineDatasetQuantizer(N_saved, N_dim - 1, 1)


    for ind, datapoint in enumerate(dataset_full):
        time_start = time.time()
        quantizer.add_point(datapoint[:-1], datapoint[-1])
        time_end = time.time()

        # print('ind: {0}  time:{1:0.2f}'.format(ind, 1000*(time_end - time_start)))

        if FLAG_PLOT:
            if ind % 100 == 0:
                quantizer.plot('ODQ Sample {0:5d}'.format(ind))


    quantizer.plot_hist()

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

