import sys
import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

from .ml_models import generate_model_server_power, generate_model_home_energy, train_test_split
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
    list_filename_previous_quant = []
    directory_quant = os.path.join(os.path.dirname(__file__), 'results', 'quantized')

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
            quantizer = data_temp['quantizer']
            reservoir_sampler = data_temp['reservoir_sampler']
            X_train = data_temp['X_train']
            Y_train = data_temp['Y_train']
            X_test = data_temp['X_test']
            Y_test = data_temp['Y_test']

            filename_parts = file.split('_')
            DATASET = eval('{0}_{1}'.format(filename_parts[0], filename_parts[1]))
            N_saved = int(filename_parts[-3])
            filename_base = file.replace('quantized.pkl', '')
            print('  SUCCESS')
        except:
            print('  ERROR. Skipping.')
            continue

        X_scaled = min_max_scaler_x.fit_transform(X_train)
        Y_scaled = min_max_scaler_y.fit_transform(Y_train)

        # Construct fixed memory OnlineDatasetQuantizer
        N_datapoints = X_train.shape[0]
        N_x = X_train.shape[1]
        N_y = Y_train.shape[1]

        compression_ratio = N_datapoints / N_saved

        if DATASET is home_energy:
            N_epochs = int(max(500, 50*compression_ratio))
        else:
            N_epochs = int(max(100, 30*compression_ratio))

        print('\n\nCompression Ratio {0}: {1} -> {2}'.format(compression_ratio, N_datapoints, N_saved))

        for ind_loop in range(N_trials):
            # Create an early stopping callback appropriate for the dataset size
            cb_earlystopping = EarlyStopping(monitor='val_loss', patience=max([40, min([10*compression_ratio, 250])]), restore_best_weights=True)

            # Create machine learning models for each evaluation step
            if DATASET is server_power:
                model_odq = generate_model_server_power(N_x, N_y)
                model_reservoir = generate_model_server_power(N_x, N_y)
                model_full = generate_model_server_power(N_x, N_y)
            elif DATASET is home_energy:
                model_odq = generate_model_home_energy(N_x, N_y)
                model_reservoir = generate_model_home_energy(N_x, N_y)
                model_full = generate_model_home_energy(N_x, N_y)

            time_start = time.time()
            print('\nGenerate model from ODQ-reduced Data')
            X_temp, Y_temp = quantizer.get_dataset()
            w_temp = quantizer.get_sample_weights()
            history_odq = model_odq.fit(X_temp, Y_temp, batch_size=128, epochs=N_epochs, sample_weight=w_temp, verbose=0,
                                        validation_split=(1 - TRAIN_VAL_RATIO), callbacks=[cb_earlystopping])
            score_odq = model_odq.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test), verbose=0)

            Y_odq_predict = min_max_scaler_y.inverse_transform(model_odq.predict(min_max_scaler_x.transform(X_test)))

            print('    RMSE: {0:0.2f}'.format(np.sqrt(np.mean((Y_odq_predict - Y_test)**2))))
            print('    Time: {0:0.2f} s'.format(time.time() - time_start))

            time_start = time.time()
            print('\nGenerate model from Reservoir-reduced Data')
            X_temp, Y_temp = reservoir_sampler.get_dataset()
            history_reservoir = model_reservoir.fit(X_temp, Y_temp, batch_size=128, epochs=N_epochs, verbose=0,
                                                    validation_split=(1 - TRAIN_VAL_RATIO), callbacks=[cb_earlystopping])
            score_reservoir = model_reservoir.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test), verbose=0)

            Y_reservoir_predict = min_max_scaler_y.inverse_transform(model_reservoir.predict(min_max_scaler_x.transform(X_test)))

            print('    RMSE: {0:0.2f}'.format(np.sqrt(np.mean((Y_reservoir_predict - Y_test)**2))))
            print('    Time: {0:0.2f} s'.format(time.time() - time_start))
            if FLAG_SAVE_MODEL:
                with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'raw',
                                       filename_base + 'models_{0}of{1}_full.pkl'.format(ind_loop, N_trials)), 'wb') as fid:
                    pkl.dump({'model_odq': model_odq, 'model_reservoir': model_reservoir,
                              'history_odq': history_odq, 'history_reservoir': history_reservoir,
                              'score_odq': score_odq, 'Y_odq_predict': Y_odq_predict,
                              'score_reservoir': score_reservoir, 'Y_reservoir_predict': Y_reservoir_predict,
                              'Y_test': Y_test, 'X_test': X_test}, fid)
            else:
                with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'raw',
                                       filename_base + 'results_{0}of{1}_full.pkl'.format(ind_loop, N_trials)), 'wb') as fid:
                    pkl.dump({'history_odq': history_odq, 'history_reservoir': history_reservoir,
                              'score_odq': score_odq, 'Y_odq_predict': Y_odq_predict,
                              'score_reservoir': score_reservoir, 'Y_reservoir_predict': Y_reservoir_predict,
                              'Y_test': Y_test, 'X_test': X_test}, fid)
