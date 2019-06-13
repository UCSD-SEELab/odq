import os
import sys

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

    return np.sqrt(np.sum(((m1 - v1) * w)**2))


class OnlineMaxEntropySelector(object):
    def __init__(self, num_datapoints_max, num_input_features, num_target_features=1, num_neighbors=5):
        """
        Initialize storage for all required data points, local gradients, and neighborhoods
        """
        if not(num_target_features == 1):
            print('ERROR: More than 1 target feature is not currently supported.')
            sys.exit()

        self.ind_max = num_datapoints_max - 1
        self.num_x = num_input_features
        self.num_y = num_target_features

        self.num_neighbors = num_neighbors

        self.dataset = np.zeros((num_datapoints_max, self.num_x + self.num_y))
        self.grad_local = np.zeros((num_datapoints_max, self.num_x))
        self.neighborhoods = -1*np.ones((num_datapoints_max, self.num_neighbors), dtype=np.int16)
        self.max_dist_neighbs = np.zeros(num_datapoints_max)
        self.max_dist_neighbs[:] = np.nan
        self.ind_curr = 0

        self.total_processed = 0

    def add_point(self, x, y):
        """
        Add current (x, y) pairing to dataset

        (x,y) incoming
        if first point
            add point to empty set with gradient = [1, ..., 1]
            return

        calculate distance to existing points using existing gradients

        if memory is not full
            determine neighborhood
            calculate local gradient for new point using least squares
            re-evaluate neighborhood
            while neighborhood has changed
                recalculate gradient with new neighborhood
                re-evaluate neighborhood
            add to set with gradient

        else
            if distance to nearest point is less than global min distance
                determine which point allows for a higher global min distance, remove the other

                for all points which were neighbors with previous, add to list_new_neighbors
                for points whose distance to new point is less than previous max, add to list_new_neighbors
            else
                remove point with global min distance
                add new point

                for all points which were neighbors with previous, add to list_new_neighbors
                for points whose distance to new point is less than previous max, add to list_new_neighbors

            for points in list_new_neighbors
                evaluate neighborhood using current gradient
                while neighborhood has changed
                    recalculate gradient with new neighborhood
                    re-evaluate neighborhood

        """
        self.total_processed += 1

        if (not(x.shape[0] == self.num_x) or not(y.shape[0] == self.num_y)):
            print('ERROR: OnlineMaxEntropySelector add_point: dimensions incorrect')
            return

        datapoint = np.append(x, y)

        # If first point, add to set
        if self.ind_curr == 0:
            self.dataset[self.ind_curr, :] = datapoint
            self.grad_local[self.ind_curr, :] = 1
            self.ind_curr = self.ind_curr + 1
            return

        # Determine distances based on local gradients of each neighbor
        list_dist = np.zeros((self.ind_curr, 1))
        self.calc_list_dist_L2(x, b_gradneighbor=True, out=list_dist)

        # If memory is not full
        if self.ind_curr <= self.ind_max:
            # Determine neighbors
            if self.ind_curr > self.num_neighbors:
                ind_neighbors = np.argpartition(list_dist, self.num_neighbors, axis=None)[:self.num_neighbors]
            else:
                ind_neighbors = np.arange(0, self.ind_curr)

            # Calculate local approximation of gradient using neighborhood and current point
            grad_local_new = self.calc_local_gradient(ind_neighbors, data_extra=datapoint)

            if self.ind_curr > self.num_neighbors:
                ind_neighbors, grad_local, list_dist_neighbors = self.update_neighborhood(data_in=datapoint,
                                                                                      grad_local_x=grad_local_new,
                                                                                      ind_neighbors_init=ind_neighbors,
                                                                                      b_member=False)
            else:
                grad_local = grad_local_new
                list_dist_neighbors = self.calc_list_dist_L2(x, w=grad_local_new, list_ind_valid=ind_neighbors)

            # Add to set
            self.dataset[self.ind_curr, :] = datapoint
            self.grad_local[self.ind_curr, :] = grad_local
            self.neighborhoods[self.ind_curr, :len(ind_neighbors)] = ind_neighbors
            self.max_dist_neighbs[self.ind_curr] = np.max(list_dist_neighbors)
            self.ind_curr = self.ind_curr + 1

            # Find all points that used the previous point in distance calculations and recalculate neighborhoods
            list_ind_new_neighbs = [ind for ind, dist_local, max_dist in zip(range(len(list_dist)), list_dist, self.max_dist_neighbs[:len(list_dist)])
                                    if ((np.isnan(max_dist)) or (dist_local < max_dist))]
            list_ind_new_neighbs.extend([ind for ind, neighbs_local in enumerate(self.neighborhoods[:self.ind_curr]) if
                                        np.any(neighbs_local == -1)])
            list_ind_new_neighbs = list(set(list_ind_new_neighbs))  # Removes duplicates

        else:
            # Determine neighborhood for point
            ind_neighbors = np.argpartition(list_dist, self.num_neighbors, axis=None)[:self.num_neighbors]
            grad_local_new = self.calc_local_gradient(ind_neighbors, data_extra=datapoint)
            ind_neighbors, grad_local, list_dist_neighbors = self.update_neighborhood(data_in=datapoint,
                                                                                      grad_local_x=grad_local_new,
                                                                                      ind_neighbors_init=ind_neighbors)

            # If new point, check if incoming point is improvement over existing points. Otherwise, check if replacing
            # nearest point provides benefit
            if np.min(list_dist_neighbors) > np.min(self.max_dist_neighbs):
                ind_min = np.argmin(self.max_dist_neighbs)

                # Replace previous point with incoming point
                self.dataset[ind_min] = datapoint
                self.grad_local[ind_min, :] = grad_local
                self.neighborhoods[ind_min, :len(ind_neighbors)] = ind_neighbors
                self.max_dist_neighbs[ind_min] = np.max(list_dist_neighbors)

                # Find all points that used the previous point in distance calculations and recalculate neighborhoods
                list_ind_new_neighbs = [ind for ind, neighbs_local in enumerate(self.neighborhoods) if
                                        np.any(neighbs_local == ind_min)]

                # Find all points whose distance to the new point is closer than the minimum of previous points
                list_ind_new_neighbs.extend([ind for ind, dist_local, max_dist in
                                             zip(range(len(list_dist)), list_dist,
                                                       self.max_dist_neighbs[:len(list_dist)])
                                             if dist_local < max_dist])

                list_ind_new_neighbs = list(set(list_ind_new_neighbs))  # Removes duplicates

            else:
                # Determine closest saved point to incoming point
                ind_min = ind_neighbors[np.argmin(list_dist_neighbors)]
                max_dist_saved = self.max_dist_neighbs[ind_min]
                datapoint_saved = self.dataset[ind_min]
                grad_local_saved = self.grad_local[ind_min]
                neighborhoods_saved = self.neighborhoods[ind_min]

                self.dataset[ind_min] = datapoint
                self.grad_local[ind_min, :] = grad_local
                self.neighborhoods[ind_min, :len(ind_neighbors)] = ind_neighbors
                self.max_dist_neighbs[ind_min] = np.max(list_dist_neighbors)

                # Compare their impact on global distance measurements excluding the other point, selecting the point
                # which improves the global metric the most
                ind_neighbors, grad_local, list_dist_neighbors = self.update_neighborhood(ind_in=ind_min,
                                                                                          b_member=True)
                # If previous point was better, restore
                if max_dist_saved > np.min(self.max_dist_neighbs):
                    self.dataset[ind_min] = datapoint_saved
                    self.grad_local[ind_min, :] = grad_local_saved
                    self.neighborhoods[ind_min, :] = neighborhoods_saved
                    self.max_dist_neighbs[ind_min] = max_dist_saved

                    list_ind_new_neighbs = []
                else:
                    # Find all points that used the previous point in distance calculations and recalculate neighborhoods
                    list_ind_new_neighbs = [ind for ind, neighbs_local in enumerate(self.neighborhoods) if
                                            np.any(neighbs_local == ind_min)]

                    # Find all points whose distance to the new point is closer than the minimum of previous points
                    list_ind_new_neighbs.extend([ind for ind, dist_local, max_dist in
                                                 zip(range(len(list_dist)), list_dist,
                                                           self.max_dist_neighbs[:len(list_dist)])
                                                 if dist_local < max_dist])

                    list_ind_new_neighbs = list(set(list_ind_new_neighbs))  # Removes duplicates

        for ind_new_neighb in list_ind_new_neighbs:
            ind_neighbors, grad_local, list_dist_neighbors = self.update_neighborhood(ind_in=ind_new_neighb,
                                                                                      grad_local_x=self.grad_local[ind_new_neighb],
                                                                                      ind_neighbors_init=self.neighborhoods[ind_new_neighb],
                                                                                      b_member=True)
            self.grad_local[ind_new_neighb, :] = grad_local
            self.neighborhoods[ind_new_neighb, :len(ind_neighbors)] = ind_neighbors
            self.max_dist_neighbs[ind_new_neighb] = np.max(list_dist_neighbors)

        return

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

    def update_neighborhood(self, ind_in=None, data_in=None, grad_local_x=None, ind_neighbors_init=None, b_member=False):
        """
        Updates the neighborhood of either the 'ind_in' point from self.dataset ('b_member' == True) or provided point
        'data_in' with gradient 'grad_local_x' ('b_member' == False)
        """
        if b_member == False:
            assert not(data_in is None) and not(grad_local_x is None), \
                   'omes: update_neighborhood: Incorrect values for b_member == False'
        else:
            assert not(ind_in is None), \
                   'omes: update_neighborhood: Incorrect index for b_member == True'
            data_in = self.dataset[ind_in, :]
            grad_local_x = self.grad_local[ind_in]

        # To ensure worst-case fixed time convergence, binary split the dataset until smallest possible unit, then
        # remove one potential neighbor at a time. If for any loop, the previous best set of neighbors is the same
        # as the current set of neighbors, convergence has been reached, so stop and return neighborhood.
        list_neighbors_possible = list(range(self.ind_curr))
        if b_member:
            del list_neighbors_possible[ind_in]

        list_dist = self.calc_list_dist_L2(data_in[:-1], w=grad_local_x, list_ind_valid=list_neighbors_possible)

        if list_dist.shape[0] <= self.num_neighbors:
            ind_neighbors_curr = list_neighbors_possible
            grad_local_x = self.calc_local_gradient(ind_neighbors_curr, data_extra=data_in)
            list_dist_out = np.max(list_dist)*np.ones((self.num_neighbors, 1))
            list_dist_out[:list_dist.shape[0]] = list_dist

        else:
            if not(ind_neighbors_init is None):
                assert ind_neighbors_init.shape[0] == self.num_neighbors, \
                       'omes: update_neighborhood: ind_neighbors_init incorrect dimensions'
                ind_neighbors_prev = list(ind_neighbors_init)
                ind_neighbors_prev.sort()
            else:
                ind_neighbors_prev = list(np.zeros(self.num_neighbors))

            ind_partition = np.argpartition(list_dist, self.num_neighbors, axis=None)[:self.num_neighbors]
            ind_neighbors_curr = [list_neighbors_possible[ind] for ind in ind_partition]
            ind_sorted = np.argsort(ind_neighbors_curr)
            ind_neighbors_curr = [ind_neighbors_curr[ind] for ind in ind_sorted]
            list_dist_out = list_dist[ind_partition, :][ind_sorted, :]

            # Repeated calculate the neighborhood from a shrinking set of selections until convergence
            ind = 0
            while not(ind_neighbors_prev == ind_neighbors_curr) and (len(list_neighbors_possible) > self.num_neighbors):
                # if ind > 0:
                #     print('pause')
                ind += 1

                if ind > 5*self.num_neighbors:
                    print('ERROR in update_neighborhood: ind {0}'.format(ind))
                    print('ind_neighbors_curr {0}'.format(ind_neighbors_curr))
                    print('ind_neighbors_prev {0}'.format(ind_neighbors_prev))
                    break
                # Reduce by 50% if sufficient neighbors remain. Otherwise, reduce neighbors by eliminating farthest
                # remaining neighbor according to list_dist
                if len(list_neighbors_possible) >= 4*self.num_neighbors:
                    list_ind_far_neighbors = list(np.argpartition(list_dist,
                                                         len(list_neighbors_possible) // 2, axis=None)[len(list_neighbors_possible) // 2:])
                else:
                    list_ind_far_neighbors = [np.argmax(list_dist)]

                list_ind_far_neighbors.sort(reverse=True)

                for ind_far_neighbor in list_ind_far_neighbors:
                    del list_neighbors_possible[ind_far_neighbor]

                # Recalculate gradient and distances. Then, re-evaluate neighborhood
                grad_local_x = self.calc_local_gradient(ind_neighbors_curr, data_extra=data_in)
                list_dist = self.calc_list_dist_L2(data_in[:-1], w=grad_local_x, list_ind_valid=list_neighbors_possible)
                ind_neighbors_prev = ind_neighbors_curr[:]
                if len(list_neighbors_possible) > self.num_neighbors:
                    ind_partition = np.argpartition(list_dist, self.num_neighbors, axis=None)[:self.num_neighbors]
                    ind_neighbors_curr = [list_neighbors_possible[ind] for ind in ind_partition]
                else:
                    ind_partition = list(range(self.num_neighbors))
                    ind_neighbors_curr = list_neighbors_possible

                ind_sorted = np.argsort(ind_neighbors_curr)
                ind_neighbors_curr = [ind_neighbors_curr[ind] for ind in ind_sorted]
                list_dist_out = list_dist[ind_partition, :][ind_sorted, :]
            # print('{0} neighb combos checked'.format(ind))

        return ind_neighbors_curr, grad_local_x, list_dist_out

    def calc_local_gradient(self, ind_neighborhood, data_extra=None):
        """
        Calculates a local gradient using a least squares fit of the given neighborhood
        """
        x_neighbors = self.dataset[ind_neighborhood, :-1]
        y_neighbors = self.dataset[ind_neighborhood, -1:]
        if not (data_extra is None):
            x_neighbors = np.vstack((x_neighbors, data_extra[:-1]))
            x_neighbors = np.append(x_neighbors, np.ones((x_neighbors.shape[0], 1)), axis=1)
            y_neighbors = np.vstack((y_neighbors, data_extra[-1]))
        else:
            x_neighbors = np.append(x_neighbors, np.ones((x_neighbors.shape[0], 1)), axis=1)

        grad_local = np.linalg.lstsq(x_neighbors, y_neighbors, rcond=None)

        return np.abs(grad_local[0]).flatten()[:-1]

    def calc_list_dist_L2(self, x, b_gradneighbor=False, w=None, list_ind_valid=None, out=None):
        """
        Performs a weighted L2 distance calculation between point of interest x and the entire list of saved values. If
        an array 'out' is supplied, then values will be loaded into it. Otherwise, an array will be returned. If
        'b_gradneighbor' is True, then the gradients of neighbors will be used to calculate distance. Otherwise, the
        provided weights will be used.
        """
        b_return = False
        if (w is None):
            w = np.ones(x.shape)

        if list_ind_valid is None:
            list_ind_valid = list(range(self.ind_curr))

        if out is None:
            b_return = True
            out = np.zeros((len(list_ind_valid), 1))

        if (b_gradneighbor):
            for ind_out, ind_data in enumerate(list_ind_valid):
                out[ind_out] = dist_L2(x, self.dataset[ind_data, :self.num_x], w=self.grad_local[ind_data, :])
        else:
            for ind_out, ind_data in enumerate(list_ind_valid):
                out[ind_out] = dist_L2(x, self.dataset[ind_data, :self.num_x], w=w)

        if b_return:
            return out
        else:
            return

    def update_min_dist(self, ind_min):
        """
        Update the minimum distance calculation for point at ind_update
        """
        temp_dist = dist_L2(self.dataset[ind_min, :],
                            self.dataset[:(self.ind_curr + 1), :], w=self.w)
        temp_dist[ind_min] = np.inf
        ind_update = np.where(self.dataset[:self.ind_curr, self.ind_dist] > temp_dist)
        self.dataset[ind_update, self.ind_dist] = temp_dist[ind_update]

        # Update distance for point that was updated
        self.dataset[ind_min, self.ind_dist] = temp_dist.min()

    def plot(self, title_in='OMES', ind_dims=[0, 1], fig_num=1, b_save_fig=False, title_save='omes_hist'):
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

    def plot_hist(self, title_in='OMES Features', fig_num=10, b_save_fig=False, title_save='omes_hist'):
        """
        Plot histrograms of each parameter distribution
        """
        if not(hasattr(self, 'total_processed')):
            self.total_processed = 0

        n_fig = ((self.num_x + self.num_y + 1) // 12) + 1

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
                    if ind_feature >= (self.num_x + self.num_y):
                        break
                if ind_feature >= (self.num_x + self.num_y):
                    break

            plt.tight_layout()

            if b_save_fig:
                plt.savefig(os.path.join(os.path.dirname(__file__), 'results', 'img',
                                         '{0}_hist_{1}_{2}.png'.format(title_save, ind_fig, self.total_processed)))
            else:
                plt.draw()
                plt.pause(PLOT_DELAY / 2)

            if ind_feature >= (self.num_x + self.num_y):
                break

if __name__ == '__main__':
    plt.ion()
    N_dataset = 500
    N_saved   = 100
    FLAG_PLOT = False
    N_dim = 4

    np.random.seed(1234)

    # Generate arbitrary dataset distribution
    dataset_raw = -2 + 4*np.random.rand(N_dataset, N_dim)
    # dataset_raw = np.random.randni(N_dataset, N_dim)
    dataset_full = np.zeros(dataset_raw.shape)
    dataset_alpha = np.random.rand(N_dataset)
    for datapoint_full, datapoint_raw, alpha in zip(dataset_full, dataset_raw, dataset_alpha):
        """
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
        """
        if datapoint_raw[0] < -1:
            datapoint_full[0] = datapoint_raw[0]
            datapoint_full[-1] = 2 + 2*datapoint_raw[0] + 0.01*datapoint_raw[1]
        elif datapoint_raw[0] < 0:
            datapoint_full[0] = datapoint_raw[0]
            datapoint_full[-1] = 0*datapoint_raw[0] + 0.01 * datapoint_raw[1]
        elif datapoint_raw[0] < 1:
            datapoint_full[0] = datapoint_raw[0]
            datapoint_full[-1] = 1.0*datapoint_raw[0] + 0.01 * datapoint_raw[1]
        else:
            datapoint_full[0] = datapoint_raw[0]
            datapoint_full[-1] = 4 - 3*datapoint_raw[0] + 0.01 * datapoint_raw[1]

    # Run N points of the distribution through OMES
    quantizer = OnlineMaxEntropySelector(N_saved, N_dim - 1, 1)

    for ind, datapoint in enumerate(dataset_full):
        time_start = time.time()
        quantizer.add_point(datapoint[:-1], datapoint[-1:])
        time_end = time.time()

        print('ind: {0}  time:{1:0.2f}'.format(ind, 1000*(time_end - time_start)))

        if FLAG_PLOT:
            if ind % 100 == 0:
                quantizer.plot('OMES Sample {0:5d}'.format(ind))


    data = quantizer.dataset

    plt.figure()
    plt.scatter(data[:,0], data[:,-1])

    plt.figure()
    plt.hist(data[:, -1], bins=20)
    plt.title('OMES y distribution')

    plt.figure()
    plt.hist(dataset_full[:, -1], bins=20)
    plt.title('Original y distribution')

    plt.figure()
    plt.hist(quantizer.grad_local[:,0], bins=20)
    plt.title('Approximate Gradient')
    plt.show(block=True)

