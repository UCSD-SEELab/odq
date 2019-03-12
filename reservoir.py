import numpy as np
from matplotlib import pyplot as plt

FLAG_VERBOSE = False
PLOT_DELAY = 0.0001


class ReservoirSampler(object):
    def __init__(self, num_datapoints_max, num_input_features, num_target_features):
        self.ind_max = num_datapoints_max - 1
        self.num_x = num_input_features
        self.num_y = num_target_features
        self.num_cols = self.num_x + self.num_y

        self.dataset = np.zeros((num_datapoints_max, self.num_cols))
        self.ind_curr = 0
        self.num_points_processed = 0

    def add_point(self, x, y):
        """
        Process current (x, y) data point to determine if added to dataset
        """

        datapoint = np.append(x, y)

        # If memory is not yet full,
        if self.ind_curr <= self.ind_max:
            # Add datapoint directly to dataset
            self.dataset[self.ind_curr, :] = datapoint
            self.ind_curr += 1
            self.num_points_processed += 1
        else:
            # Otherwise, determine if point should be added or tossed
            ind_rand = np.random.randint(self.ind_curr)
            if ind_rand <= self.ind_max:
                self.dataset[ind_rand, :] = datapoint
            self.num_points_processed += 1

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
               self.dataset[:self.ind_curr, self.num_x:]

    def plot(self, title_in='Reservoir', ind_dims=[0, 1], fig_num=1):
        """
        Plots distribution in up to 3 dimensions
        """

        # TODO Add ind_dims to input variables and update plots to adapt to ind_dims

        plt.figure(fig_num)
        plt.cla()
        plt.title(title_in)
        plt.scatter(self.dataset[:self.ind_curr, ind_dims[0]], self.dataset[:self.ind_curr, ind_dims[1]])
        plt.grid(True)
        plt.draw()
        plt.pause(PLOT_DELAY/2)