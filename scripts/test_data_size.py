import sys
import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras import backend as K

from ml_models import generate_model_server_power, generate_model_home_energy, generate_model_metasense, train_test_split, train_test_split_blocks
import pickle as pkl

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from odq.odq import OnlineDatasetQuantizer
from odq.reservoir import ReservoirSampler
from odq.data import home_energy, server_power, metasense


if __name__ == '__main__':
    FLAG_VERBOSE = False
    FLAG_PLOT = False
    FLAG_SAVE_MODEL = False
    FLAG_TRAIN_MODEL = False
    directory_quant = os.path.join(os.path.dirname(__file__), '..', 'results', 'quantized')
    directory_target = 'test_convergence'

    N_trials = 3
    TRAIN_VAL_RATIO = 0.8
    list_lr = [0.001, 0.0001, 0.00001]
    list_std_noise = [0.1, 0.01, 0.001]

    # np.random.seed(1237)

    plt.ion()

    for file in os.listdir(os.path.join(directory_quant, directory_target)):
        if not (file.lower().endswith('.pkl')):
            continue

        filename_base = file.replace('quantized.pkl', '')

        print('\n\nLoading data from {0}'.format(file))

        try:
            with open(os.path.join(directory_quant, directory_target, file), 'rb') as fid:
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
            if filename_parts[1] == 'data':
                DATASET = eval('{0}'.format(filename_parts[0]))
            else:
                DATASET = eval('{0}_{1}'.format(filename_parts[0], filename_parts[1]))
            compression_ratio = float(filename_parts[-3])
            print('  SUCCESS')
        except:
            print('  ERROR. Skipping.')
            continue

        N_datapoints = X_train.shape[0]
        N_x = X_train.shape[1]
        N_y = Y_train.shape[1]

        if DATASET is home_energy:
            N_epochs = int(max(500, 50*compression_ratio))
        else:
            N_epochs = int(max(200, 30*compression_ratio))

        print('\n\nCompression Ratio {0}'.format(compression_ratio))

        for ind_loop in range(N_trials):
            print('  Trial {0} of {1}:'.format(ind_loop + 1, N_trials))

            for lr in list_lr:
                for std_noise in list_std_noise:
                    print('  lr: {0}   std_noise: {1}:'.format(lr, std_noise))

                    if not (FLAG_PLOT) and os.path.isfile(
                            os.path.join(os.path.dirname(__file__), '..', 'results', 'raw',
                                         directory_target,
                                         filename_base + '_lr{1}_std{2}_results_trial{0}_reduced.pkl'.format(ind_loop, lr, std_noise))):
                        print('  File already processed')
                        continue

                    # Create an early stopping callback appropriate for the dataset size
                    cb_earlystopping = EarlyStopping(monitor='val_loss',
                                                     patience=max([20, min([compression_ratio*lr/0.0001, 250])]),
                                                     restore_best_weights=True)

                    # Create machine learning models for each evaluation step
                    if DATASET is server_power:
                        model_odq = generate_model_server_power(N_x, N_y)
                        model_reservoir = generate_model_server_power(N_x, N_y)

                    elif DATASET is home_energy:
                        model_odq = generate_model_home_energy(N_x, N_y)
                        model_reservoir = generate_model_home_energy(N_x, N_y)

                    elif DATASET is metasense:
                        model_odq = generate_model_metasense(N_x, N_y)
                        model_reservoir = generate_model_metasense(N_x, N_y)

                    # Perform training from reservoir data first, saving the validation set
                    time_start = time.time()
                    print('    Generating from Reservoir-reduced Data')
                    X_temp, Y_temp = reservoir_sampler.get_dataset()

                    X_temp = min_max_scaler_x.transform(X_temp)
                    Y_temp = min_max_scaler_y.transform(Y_temp)

                    X_fit, X_val, Y_fit, Y_val = train_test_split(X_temp, Y_temp, pct_train=TRAIN_VAL_RATIO)

                    # Train the model on 10 epochs before checking for early stopping conditions to prevent premature return
                    history_temp = model_reservoir.fit(X_fit, Y_fit, batch_size=32, epochs=10, verbose=0,
                                                       validation_data=(X_val, Y_val))
                    history_reservoir = history_temp.history
                    history_reservoir['epoch'] = history_temp.epoch
                    history_temp = model_reservoir.fit(X_fit, Y_fit, batch_size=32, epochs=N_epochs, verbose=0,
                                                       validation_data=(X_val, Y_val), initial_epoch=10,
                                                       callbacks=[cb_earlystopping])
                    history_reservoir['epoch'].extend(history_temp.epoch)
                    history_reservoir['val_loss'].extend(history_temp.history['val_loss'])
                    history_reservoir['val_mean_squared_error'].extend(history_temp.history['val_mean_squared_error'])
                    history_reservoir['val_mean_absolute_error'].extend(history_temp.history['val_mean_absolute_error'])
                    history_reservoir['loss'].extend(history_temp.history['loss'])
                    history_reservoir['mean_squared_error'].extend(history_temp.history['mean_squared_error'])
                    history_reservoir['mean_absolute_error'].extend(history_temp.history['mean_absolute_error'])

                    score_reservoir = model_reservoir.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test), verbose=0)

                    Y_reservoir_predict = min_max_scaler_y.inverse_transform(model_reservoir.predict(min_max_scaler_x.transform(X_test)))

                    results_reservoir_rmse = np.sqrt(np.mean((Y_reservoir_predict - Y_test)**2, axis=0))

                    print('    RMSE: {0}'.format(np.array2string(results_reservoir_rmse, precision=2, suppress_small=True)))
                    print('    Time: {0:0.2f} s'.format(time.time() - time_start))

                    time_start = time.time()
                    print('    Generating model from ODQ-reduced Data')
                    X_temp, Y_temp = quantizer.get_dataset()
                    w_temp = quantizer.get_sample_weights()

                    w_temp = w_temp * w_temp.shape[0] / np.sum(w_temp)

                    X_temp = min_max_scaler_x.transform(X_temp)
                    Y_temp = min_max_scaler_y.transform(Y_temp)

                    # Train using validation set from reservoir sampling. Size is already taken into account in
                    # generate_reduced_datasets.py
                    history_temp = model_odq.fit(X_temp, Y_temp, batch_size=32, epochs=10, sample_weight=w_temp, verbose=0,
                                                 validation_data=(X_val, Y_val))
                    history_odq = history_temp.history
                    history_odq['epoch'] = history_temp.epoch
                    history_temp = model_odq.fit(X_temp, Y_temp, batch_size=32, epochs=N_epochs, sample_weight=w_temp, verbose=0,
                                                 validation_data=(X_val, Y_val), initial_epoch=10,
                                                 callbacks=[cb_earlystopping])
                    history_odq['epoch'].extend(history_temp.epoch)
                    history_odq['val_loss'].extend(history_temp.history['val_loss'])
                    history_odq['val_mean_squared_error'].extend(history_temp.history['val_mean_squared_error'])
                    history_odq['val_mean_absolute_error'].extend(history_temp.history['val_mean_absolute_error'])
                    history_odq['loss'].extend(history_temp.history['loss'])
                    history_odq['mean_squared_error'].extend(history_temp.history['mean_squared_error'])
                    history_odq['mean_absolute_error'].extend(history_temp.history['mean_absolute_error'])

                    score_odq = model_odq.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test), verbose=0)

                    Y_odq_predict = min_max_scaler_y.inverse_transform(model_odq.predict(min_max_scaler_x.transform(X_test)))

                    results_odq_rmse = np.sqrt(np.mean((Y_odq_predict - Y_test)**2, axis=0))

                    print('    RMSE: {0}'.format(np.array2string(results_odq_rmse, precision=2, suppress_small=True)))
                    print('    Time: {0:0.2f} s'.format(time.time() - time_start))

                    # Save all results for subsequent processing
                    directory_target_full = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw', directory_target)
                    if not os.path.exists(directory_target_full):
                        os.mkdir(directory_target_full)

                    if FLAG_SAVE_MODEL:
                        with open(os.path.join(directory_target_full,
                                               filename_base + '_lr{1}_std{2}_models_trial{0}_reduced.pkl'.format(ind_loop, lr, std_noise)), 'wb') as fid:
                            pkl.dump({'model_odq': model_odq, 'model_reservoir': model_reservoir,
                                      'history_odq': history_odq, 'history_reservoir': history_reservoir,
                                      'score_odq': score_odq, 'Y_odq_predict': Y_odq_predict,
                                      'score_reservoir': score_reservoir, 'Y_reservoir_predict': Y_reservoir_predict,
                                      'Y_test': Y_test, 'X_test': X_test,
                                      'N_datapoints': N_datapoints}, fid)
                    else:
                        with open(os.path.join(directory_target_full,
                                               filename_base + '_lr{1}_std{2}_results_trial{0}_reduced.pkl'.format(ind_loop, lr, std_noise)), 'wb') as fid:
                            pkl.dump({'history_odq': history_odq, 'history_reservoir': history_reservoir,
                                      'score_odq': score_odq, 'Y_odq_predict': Y_odq_predict,
                                      'score_reservoir': score_reservoir, 'Y_reservoir_predict': Y_reservoir_predict,
                                      'Y_test': Y_test, 'X_test': X_test,
                                      'N_datapoints': N_datapoints}, fid)

                    if FLAG_PLOT:
                        plt.figure()
                        plt.rc('font', family='Liberation Serif', size=14)
                        plt.plot(history_odq['val_mean_squared_error'], 'b')
                        plt.plot(history_reservoir['val_mean_squared_error'], 'r')
                        plt.title('Validation CR = {0}'.format(compression_ratio))
                        plt.legend(('ODQ', 'Reservoir'))

                        plt.figure()
                        plt.rc('font', family='Liberation Serif', size=14)
                        plt.plot(history_odq['mean_squared_error'], 'k')
                        plt.plot(history_odq['val_mean_squared_error'], 'b')
                        plt.title('ODQ CR = {0}'.format(compression_ratio))
                        plt.legend(('Train', 'Val'))

                        plt.figure()
                        plt.rc('font', family='Liberation Serif', size=14)
                        plt.plot(history_reservoir['mean_squared_error'], 'k')
                        plt.plot(history_reservoir['val_mean_squared_error'], 'b')
                        plt.title('Reservoir CR = {0}'.format(compression_ratio))
                        plt.legend(('Train', 'Val'))
                        plt.show(block=True)

                    # Reset Tensorflow session to prevent memory growth
                    K.clear_session()
