import sys
import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

from ml_models import generate_model_server_power, generate_model_home_energy, train_test_split
import pickle as pkl

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from odq.odq import OnlineDatasetQuantizer, calc_weights_max_cov
from odq.reservoir import ReservoirSampler
from odq.data import home_energy, server_power


if __name__ == '__main__':
    FLAG_VERBOSE = False
    FLAG_PLOT = False
    FLAG_SAVE_MODEL = False
    FLAG_TRAIN_MODEL = False
    list_filename_previous_quant = ['server_power_data_size_20190315180856_261_0_quantized.pkl']
    directory_quant = os.path.join(os.path.dirname(__file__), '..', 'results', 'quantized')

    DATASET = server_power
    ind_assess = [-1] #1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000]
    list_compression_ratio = np.append([], 2**(1 + np.arange(8)))[::-1]#2**(1 + np.arange(9))[::-1]
    N_trials = 3
    TRAIN_VAL_RATIO = 0.8
    TRAIN_TEST_RATIO = 0.8 # Server power has test/train datasets pre-split due to tasks

    # np.random.seed(1237)

    plt.ion()

    for file in os.listdir(directory_quant):

        if not (file.lower().endswith('.pkl')):
            continue

        filename_match = [file.startswith(filename_base) for filename_base in list_filename_previous_quant]

        if not (any(filename_match)):
            continue

        print('Loading data from {0}'.format(file))

        try:
            with open(os.path.join(directory_quant, file), 'rb') as fid:
                data_temp = pkl.load(fid)
            min_max_scaler_x = data_temp['min_max_scaler_x']
            min_max_scaler_y = data_temp['min_max_scaler_y']
            X_train = data_temp['X_train']
            Y_train = data_temp['Y_train']
            X_test = data_temp['X_test']
            Y_test = data_temp['Y_test']

            filename_parts = file.split('_')
            DATASET = eval('{0}_{1}'.format(filename_parts[0], filename_parts[1]))
            filename_base = file.replace('quantized.pkl', '')
            filename_base = filename_base.replace('_' + filename_parts[-3], '')
            print('  SUCCESS')
        except:
            print('  ERROR. Skipping.')
            continue


        for ind_loop in range(N_trials):
            X_scaled = min_max_scaler_x.fit_transform(X_train)
            Y_scaled = min_max_scaler_y.fit_transform(Y_train)

            # Construct fixed memory OnlineDatasetQuantizer
            N_datapoints = X_train.shape[0]
            N_x = X_train.shape[1]
            N_y = Y_train.shape[1]

            if DATASET is home_energy:
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, pct_train=TRAIN_TEST_RATIO)
                X_scaled = min_max_scaler_x.fit_transform(X_train)
                Y_scaled = min_max_scaler_y.fit_transform(Y_train)

            for ind_loop in range(N_trials):
                # Create machine learning models for each evaluation step
                if DATASET is server_power:
                    model_full = generate_model_server_power(N_x, N_y)
                elif DATASET is home_energy:
                    model_full = generate_model_home_energy(N_x, N_y)

                # Save full model
                time_start = time.time()
                print('\nGenerate model from full dataset')
                cb_earlystopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                history_full = model_full.fit(X_scaled, Y_scaled, batch_size=128, epochs=400, verbose=0,
                                              validation_split=(1 - TRAIN_VAL_RATIO), callbacks=[cb_earlystopping])
                score_full = model_full.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test), verbose=0)

                Y_full_predict = min_max_scaler_y.inverse_transform(model_full.predict(min_max_scaler_x.transform(X_test)))

                print('    RMSE: {0:0.2f}'.format(np.sqrt(np.mean((Y_full_predict - Y_test) ** 2))))
                print('    Time: {0:0.2f} s'.format(time.time() - time_start))

                if FLAG_SAVE_MODEL:
                    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'raw',
                                           filename_base + 'model_{0}of{1}_full.pkl'.format(ind_loop, N_trials)), 'wb') as fid:
                        pkl.dump({'model_full': model_full, 'history_full':history_full, 'score_full': score_full,
                                  'Y_full_predict': Y_full_predict, 'Y_test': Y_test, 'X_test': X_test}, fid)
                else:
                    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'raw',
                                           filename_base + 'results_{0}of{1}_full.pkl'.format(ind_loop, N_trials)), 'wb') as fid:
                        pkl.dump({'history_full':history_full, 'score_full': score_full,
                                  'Y_full_predict': Y_full_predict, 'Y_test': Y_test, 'X_test': X_test}, fid)
