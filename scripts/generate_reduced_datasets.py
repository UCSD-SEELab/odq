import sys
import os
import time
from datetime import datetime
import argparse

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from sklearn import preprocessing
import pickle as pkl

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from odq.odq import OnlineDatasetQuantizer, calc_weights_max_cov, calc_weights_max_cov2, calc_weights_unit_var, \
                    calc_weights_max_norm, calc_weights_max_cov_gauss, calc_weights_pr_squeeze, calc_weights_incorrect, \
                    calc_weights_imbalanced, calc_weights_x_y_tradeoff, calc_weights_singlex, calc_weights_singley
from odq.omes import OnlineMaxEntropySelector
from odq.reservoir import ReservoirSampler
from odq.data import home_energy, server_power, metasense


def train_test_split(X, Y, pct_train=0.8, weights=None):
    """
    Splits the datasets X and Y into training and test sets based on input percentage (pct_train)
    """
    N = X.shape[0]
    ind_split = np.round(N*pct_train).astype(int)
    ind_random = np.random.permutation(N)

    if weights is None:
        return X[ind_random[:ind_split], :], X[ind_random[ind_split:], :], \
               Y[ind_random[:ind_split], :], Y[ind_random[ind_split:], :]
    else:
        return X[ind_random[:ind_split], :], X[ind_random[ind_split:], :], \
               Y[ind_random[:ind_split], :], Y[ind_random[ind_split:], :], \
               weights[ind_random[:ind_split]]

def train_test_split_blocks(X, Y, pct_train=0.8, n_blocks=3):
    """
    Splits datasets X and Y into training and test sets based on input percentage, dividing the test set into
    n_blocks number of continuous blocks
    """
    N = X.shape[0]
    N_test = np.round(N*(1 - pct_train)).astype(int)
    list_block_sizes = []
    for _ in range(n_blocks):
        list_block_sizes.append((N_test - sum(list_block_sizes)) // (n_blocks - len(list_block_sizes)))

    list_ind_test = []

    for block_size in list_block_sizes:
        ind_block_min = np.random.randint(N - 1 - block_size)
        ind_block_max = ind_block_min + block_size

        while (ind_block_min in list_ind_test) or (ind_block_max in list_ind_test):
            ind_block_min = np.random.randint(N - 1 - block_size)
            ind_block_max = ind_block_min + block_size

        list_ind_test.extend(list(range(ind_block_min, ind_block_max)))

    list_ind_train = list(set(np.random.permutation(N)) - set(list_ind_test))

    return X[list_ind_train, :], X[list_ind_test, :], \
           Y[list_ind_train, :], Y[list_ind_test, :]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='Target directory to save generated files.')
    parser.add_argument('--N', type=int, nargs=1, help='Number of iterations to generate.', default=[5])
    parser.add_argument('--dataset', type=str, help='Dataset to use (home_energy, server_power, metasense)')
    parser.add_argument('--method', type=str, help='Quantizer to use (omes, odq)', default='omes')
    parser.add_argument('--w_type', type=int, nargs='+', help='Type of weight to use in ODQ (1: ones, 2: cov_max2, 3: unit_var, 4: pr_squeeze', default=[3])
    parser.add_argument('--brd', type=int, nargs=1, help='Board number for MetaSense tests', default=[11])
    parser.add_argument('--k', type=int, nargs='+', help='Size of neighborhood for calculating local gradient', default=[5])

    args = parser.parse_args()

    if args.dir is not None:
        directory_target = args.dir
    else:
        directory_target = 'unnamed_results'

    if args.dataset is not None:
        try:
            DATASET = eval(args.dataset)
        except:
            print('Failed to load {0}'.format(args.dataset))
            sys.exit()
    else:
        DATASET = metasense

    if not(DATASET in [home_energy, server_power, metasense]):
        print('Invalid dataset')
        sys.exit()

    FLAG_VERBOSE = False
    FLAG_PLOT = False
    PLOT_DELAY = 0.0001
    ind_assess = [-1] # 5000 * np.arange(1, 35).astype(int)
    list_compression_ratio = np.append([], 2**(2 + 2*np.arange(5)))[::-1]#2**(1 + np.arange(9))[::-1]
    N_iterations = args.N[0]
    list_weight_type = args.w_type
    metasense_brd = args.brd[0]
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
        X, Y = DATASET.load(board_num=metasense_brd)
        X_train, X_test, Y_train, Y_test = train_test_split_blocks(X, Y, pct_train=TRAIN_TEST_RATIO, n_blocks=10)

    else:
        print('Invalid dataset {0}'.format(DATASET))
        sys.exit()

    # Normalize data to between 0 and 1 for future machine learning
    min_max_scaler_x = preprocessing.MinMaxScaler()
    min_max_scaler_y = preprocessing.MinMaxScaler()

    # Construct fixed memory for quantizer
    N_datapoints = X_train.shape[0]
    N_x = X_train.shape[1]
    N_y = Y_train.shape[1]

    # Print point for index decided based on mod(ind, ind_print) == 0
    N_print_segs = 100

    ind_print = int(N_datapoints // N_print_segs)* np.arange(1, N_print_segs + 1).astype(int)

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

        min_max_scaler_x.fit(X_train)
        min_max_scaler_y.fit(Y_train)

        # Generate new sets of weights for selected training set
        dict_out = {'Y_train': Y_train, 'Y_test': Y_test, 'X_train': X_train, 'X_test': X_test,
                    'min_max_scaler_x': min_max_scaler_x, 'min_max_scaler_y': min_max_scaler_y,
                    'TRAIN_VAL_RATIO': TRAIN_VAL_RATIO, 'TRAIN_TEST_RATIO': TRAIN_TEST_RATIO}

        if args.method == 'odq':
            list_w_x = []
            list_w_y = []
            list_w_imp = []
            for ind, weight_type in enumerate(list_weight_type):
                if weight_type <= 1:
                    w_list = np.ones((N_x + N_y))
                    w_imp = w_list / sum(w_list)
                elif weight_type == 2:
                    w_list, w_imp = calc_weights_max_cov2(X_train, Y_train)
                elif weight_type == 3:
                    w_list, w_imp = calc_weights_unit_var(X_train, Y_train)
                elif weight_type == 4:
                    w_list, w_imp = calc_weights_pr_squeeze(X_train, Y_train, depth=4)
                elif weight_type == 5:
                    w_list, w_imp = calc_weights_imbalanced(X_train, Y_train)
                elif weight_type == 6:
                    w_list, w_imp = calc_weights_x_y_tradeoff(X_train, Y_train, pct=0.999)
                elif weight_type == 7:
                    w_list, w_imp = calc_weights_x_y_tradeoff(X_train, Y_train, pct=0.99)
                elif weight_type == 8:
                    w_list, w_imp = calc_weights_x_y_tradeoff(X_train, Y_train, pct=0.01)
                elif weight_type == 9:
                    w_list, w_imp = calc_weights_x_y_tradeoff(X_train, Y_train, pct=0.001)
                elif weight_type == 10:
                    w_list, w_imp = calc_weights_singlex(X_train, Y_train, ind_x=0)
                elif weight_type == 11:
                    w_list, w_imp = calc_weights_singley(X_train, Y_train, ind_y=0)

                list_w_x.append(w_list[:N_x])
                list_w_y.append(w_list[N_x:])
                list_w_imp.append(w_imp)
        else:
            list_w_x = [np.NaN]
            list_w_y = [np.NaN]
            list_w_imp = [np.NaN]
            list_weight_type = [np.NaN]

        for compression_ratio in list_compression_ratio:
            dict_out['quantizer_type'] = args.method
            dict_out['quantizers'] = []
            if args.method == 'odq':
                N_saved_quantizer = int(N_datapoints / compression_ratio * (N_x + N_y) / (N_x + N_y + 2) * TRAIN_VAL_RATIO)
            else:
                N_saved_quantizer = int(N_datapoints / compression_ratio * (N_x + N_y) / (2*N_x + N_y + args.k[0] + 1))
            N_saved_res = int(N_datapoints / compression_ratio)
            print('\n\nCompression Ratio {0}: {1} -> {4}:{2} res:{3}'.format(compression_ratio, N_datapoints, N_saved_quantizer, N_saved_res, args.method))
            list_quantizers = []
            if args.method == 'odq':
                for w_x, w_y in zip(list_w_x, list_w_y):
                    list_quantizers.append(OnlineDatasetQuantizer(num_datapoints_max=N_saved_quantizer,
                                                                  num_input_features=N_x,
                                                                  num_target_features=N_y,
                                                                  w_x_columns=w_x, w_y_columns=w_y))
            else:
                list_quantizers.append(OnlineMaxEntropySelector(num_datapoints_max=N_saved_quantizer,
                                                                num_input_features=N_x,
                                                                num_target_features=N_y,
                                                                num_neighbors=args.k[0]))

            reservoir_sampler = ReservoirSampler(num_datapoints_max=N_saved_res, num_input_features=N_x, num_target_features=N_y)

            for ind, X_new, Y_new in zip(range(N_datapoints), X_train, Y_train):
                for quantizer in list_quantizers:
                    time_start = time.time()
                    quantizer.add_point(X_new, Y_new)
                    time_end = time.time()
                    # print('ind: {0}  time:{1:0.2f}'.format(ind, 1000 * (time_end - time_start)))

                reservoir_sampler.add_point(X_new, Y_new)

                if ind in ind_print:
                    print('  {0} / {1}'.format(ind, N_datapoints))

                #if (ind in ind_assess):
                #    for quantizer, w_type in zip(list_quantizers, list_weight_type):
                #        quantizer.plot_hist(title_in='ODQ Hist (w_type {1}, {0} samples)'.format(ind, w_type), fig_num=10, b_save_fig=True,
                #                            title_save='odq_{0}_{1}_{{}}_{2}'.format(filename_base, compression_ratio, ind))
                #    reservoir_sampler.plot_hist(title_in='Reservoir Hist ({0} samples)'.format(ind), fig_num=20,
                #                                b_save_fig=True, title_save='res_{0}_{1}_{{}}_{2}'.format(filename_base, compression_ratio, ind))

            for quantizer, w_x, w_y, w_imp, w_type in zip(list_quantizers, list_w_x, list_w_y, list_w_imp, list_weight_type):
                dict_out['quantizers'].append({'quantizer':quantizer, 'w_x': w_x, 'w_y': w_y, 'w_imp': w_imp, 'w_type': w_type})
            dict_out['reservoir_sampler'] = reservoir_sampler

            print('  Saving reduced datasets')
            directory_target_full = os.path.join(os.path.dirname(__file__), '..', 'results', 'quantized', directory_target)
            if not os.path.exists(directory_target_full):
                os.mkdir(directory_target_full)

            with open(os.path.join(directory_target_full, filename_base + '_{0}_{1}_quantized.pkl'.format(compression_ratio, ind_loop)), 'wb') as fid:
                pkl.dump(dict_out, fid)
