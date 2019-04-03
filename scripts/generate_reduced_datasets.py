import sys
import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from sklearn import preprocessing
import pickle as pkl
from ml_models import train_test_split, train_test_split_blocks

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from odq.odq import OnlineDatasetQuantizer, calc_weights_max_cov, calc_weights_max_cov2, calc_weights_unit_var, \
                    calc_weights_max_norm, calc_weights_max_cov_gauss
from odq.reservoir import ReservoirSampler
from odq.data import home_energy, server_power, metasense


if __name__ == '__main__':
    FLAG_VERBOSE = False
    FLAG_PLOT = False
    PLOT_DELAY = 0.0001
    DATASET = home_energy # home_energy # server_power # metasense
    directory_target = 'home_energy_test_cov_max_gauss_20190402'
    ind_assess = [-1] #2000 * np.arange(1, 35).astype(int)
    list_compression_ratio = np.append([], 2**(5 + np.arange(6)))[::-1]#2**(1 + np.arange(9))[::-1]
    N_iterations = 5
    TRAIN_TEST_RATIO = 0.8 # Server power has test/train datasets pre-split due to tasks
    TRAIN_VAL_RATIO = 0.8

    filename_base = datetime.now().strftime('{0}_data_size_%Y%m%d%H%M%S'.format(DATASET.__name__.split('.')[-1]))

    plt.ion()

    if DATASET is None:
        N_dataset = 20000
        N_dim = 2

        # Generate arbitrary dataset distribution
        dataset_raw = np.random.randn(N_dataset, N_dim)
        dataset_full = np.zeros(dataset_raw.shape)
        dataset_alpha = np.random.rand(N_dataset)
        for datapoint_full, datapoint_raw, alpha in zip(dataset_full, dataset_raw, dataset_alpha):
            if alpha < 0.4:
                datapoint_full[0] = 1 + 2 * datapoint_raw[0]
                datapoint_full[1] = 4 + 2 * datapoint_raw[1]
            elif alpha < 0.5:
                datapoint_full[0] = 4 + 0.25 * datapoint_raw[0]
                datapoint_full[1] = 5 + 1.25 * datapoint_raw[1]
            elif alpha < 0.75:
                datapoint_full[0] = 3 + 0.6 * datapoint_raw[0] + 1 * datapoint_raw[1]
                datapoint_full[1] = 3 + datapoint_raw[1]
            elif alpha <= 1:
                datapoint_full[0] = 0 + 0.7 * datapoint_raw[0] + 0.5 * datapoint_raw[1]
                datapoint_full[1] = 1 + 0.7 * datapoint_raw[1]

        X = datapoint_full[:, :-1]
        Y = datapoint_full[:, -1:]

        # Use default weights for each column
        w_x = None
        w_y = None

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, pct_train=TRAIN_TEST_RATIO)

    elif DATASET is server_power:
        X_train, Y_train, X_test, Y_test = DATASET.load()
        TRAIN_TEST_RATIO = Y_train.shape[0] / (Y_train.shape[0] + Y_test.shape[0])
        ind_random = np.random.permutation(Y_train.shape[0])
        X_train = X_train[ind_random]
        Y_train = Y_train[ind_random]

    elif DATASET is home_energy:
        X, Y = DATASET.load()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, pct_train=TRAIN_TEST_RATIO)

    elif DATASET is metasense:
        X, Y = DATASET.load()
        X_train, X_test, Y_train, Y_test = train_test_split_blocks(X, Y, pct_train=TRAIN_TEST_RATIO, n_blocks=10)

    else:
        print('Invalid dataset {0}'.format(DATASET))
        sys.exit()

    # Normalize data to between 0 and 1 for future machine learning
    min_max_scaler_x = preprocessing.MinMaxScaler()
    min_max_scaler_y = preprocessing.MinMaxScaler()

    # Construct fixed memory OnlineDatasetQuantizer
    N_datapoints = X_train.shape[0]
    N_x = X_train.shape[1]
    N_y = Y_train.shape[1]

    # Print point for index decided based on mod(ind, ind_print) == 0
    ind_print = int(N_datapoints // 10)* np.arange(1, 11).astype(int)

    for ind_loop in range(N_iterations):
        # Use same dataset for all compression levels in a given trial. Use loaded data for first loop, otherwise,
        # reshuffle the training dataset.
        if not(ind_loop == 0):
            if DATASET is home_energy:
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, pct_train=TRAIN_TEST_RATIO)

            elif DATASET is metasense:
                X_train, X_test, Y_train, Y_test = train_test_split_blocks(X, Y, pct_train=TRAIN_TEST_RATIO, n_blocks=10)

            elif DATASET is server_power:
                ind_random = np.random.permutation(N_datapoints)
                X_train = X_train[ind_random]
                Y_train = Y_train[ind_random]

        # Generate new sets of weights for selected training set
        w_max_cov, cov_max = calc_weights_max_cov2(X_train, Y_train)
        w_max_cov_norm, cov_max_norm = calc_weights_max_norm(X_train, Y_train)
        w_unit_var, _ = calc_weights_unit_var(X_train, Y_train)
        w_max_cov_gauss, cov_max_gauss = calc_weights_max_cov_gauss(X_train, Y_train)
        w_ones = np.ones((N_x + N_y))
        w_x = w_max_cov[:N_x]
        w_y = w_max_cov[N_x:]

        min_max_scaler_x.fit(X_train)
        min_max_scaler_y.fit(Y_train)

        for compression_ratio in list_compression_ratio:
            N_saved_odq = int(N_datapoints / compression_ratio * (N_x + N_y) / (N_x + N_y + 2) * TRAIN_VAL_RATIO)
            N_saved_res = int(N_datapoints / compression_ratio)
            print('\n\nCompression Ratio {0}: {1} -> odq:{2} res:{3}'.format(compression_ratio, N_datapoints, N_saved_odq, N_saved_res))
            quantizer = OnlineDatasetQuantizer(num_datapoints_max=N_saved_odq, num_input_features=N_x, num_target_features=N_y,
                                               w_x_columns=w_x, w_y_columns=w_y)
            reservoir_sampler = ReservoirSampler(num_datapoints_max=N_saved_res, num_input_features=N_x, num_target_features=N_y)

            for ind, X_new, Y_new in zip(range(N_datapoints), X_train, Y_train):
                quantizer.add_point(X_new, Y_new)
                reservoir_sampler.add_point(X_new, Y_new)

                if ind in ind_print:
                    print('  {0} / {1}'.format(ind, N_datapoints))

                if (ind in ind_assess):
                    quantizer.plot_hist(title_in='ODQ Hist ({0} samples)'.format(ind), fig_num=10, b_save_fig=True,
                                        title_save='odq_{0}_{1}_{{}}_{2}'.format(filename_base, compression_ratio, ind))
                    reservoir_sampler.plot_hist(title_in='Reservoir Hist ({0} samples)'.format(ind), fig_num=20,
                                                b_save_fig=True, title_save='res_{0}_{1}_{{}}_{2}'.format(filename_base, compression_ratio, ind))

            print('  Saving reduced datasets')
            directory_target_full = os.path.join(os.path.dirname(__file__), '..', 'results', 'quantized', directory_target)
            if not os.path.exists(directory_target_full):
                os.mkdir(directory_target_full)

            with open(os.path.join(directory_target_full, filename_base + '_{0}_{1}_quantized.pkl'.format(compression_ratio, ind_loop)), 'wb') as fid:
                pkl.dump({'quantizer': quantizer, 'reservoir_sampler': reservoir_sampler,
                          'Y_train': Y_train, 'Y_test': Y_test,
                          'X_train': X_train, 'X_test': X_test,
                          'min_max_scaler_x': min_max_scaler_x, 'min_max_scaler_y': min_max_scaler_y,
                          'TRAIN_VAL_RATIO': TRAIN_VAL_RATIO, 'TRAIN_TEST_RATIO': TRAIN_TEST_RATIO,
                          }, fid)
