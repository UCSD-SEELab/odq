import sys
import os
import time
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras import backend as K
import tensorflow as tf

from ml_models import generate_model_server_power, generate_model_home_energy, generate_model_metasense, train_test_split, train_test_split_blocks
import pickle as pkl

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from odq.data import home_energy, server_power, metasense


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='Target directory of files.')
    parser.add_argument('--N', type=int, nargs=1, help='Number of trials to run.', default=3)
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    if args.dir is not None:
        directory_target = args.dir
    else:
        directory_target = 'metasense_test_cov_max2_20190401'

    # Configure tensorflow
    num_cores = 4
    if args.cpu:
        num_CPU = num_cores
        num_GPU = 0
    else:
        num_GPU = 1
        num_CPU = num_cores

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores,
                            allow_soft_placement=True,
                            device_count={'CPU': num_CPU,
                                          'GPU': num_GPU}
                            )

    session = tf.Session(config=config)
    K.set_session(session)

    N_trials = args.N[0]

    FLAG_VERBOSE = False
    FLAG_PLOT = False
    FLAG_SAVE_MODEL = False
    FLAG_TRAIN_MODEL = False
    directory_quant = os.path.join(os.path.dirname(__file__), '..', 'results', 'quantized')

    TRAIN_VAL_RATIO = 0.8
    list_lr = [0.01, 0.001]
    list_decay = [0, 1e-3, 1e-6]
    list_std_noise = [0.001]

    # plt.ion()

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
            list_quantizers = data_temp['quantizers']
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
            N_epochs = 50 # int(max(500, 50*compression_ratio))
        else:
            N_epochs = 50 # int(max(200, 30*compression_ratio))

        print('\n\nCompression Ratio {0}'.format(compression_ratio))

        for ind_loop in range(N_trials):
            print('  Trial {0} of {1}:'.format(ind_loop + 1, N_trials))

            for lr in list_lr:
                for decay in list_decay:
                    print('  lr: {0}   decay: {1}:'.format(lr, decay))

                    if not (FLAG_PLOT) and os.path.isfile(
                            os.path.join(os.path.dirname(__file__), '..', 'results', 'raw',
                                         directory_target,
                                         filename_base + '_lr{1}_decay{2}_results_trial{0}_reduced.pkl'.format(ind_loop, lr, decay))):
                        print('  File already processed')
                        continue

                    # Create machine learning models for each evaluation step
                    list_model_odq = []
                    if DATASET is server_power:
                        generate_model = generate_model_server_power
                    elif DATASET is home_energy:
                        generate_model = generate_model_home_energy
                    elif DATASET is metasense:
                        generate_model = generate_model_metasense

                    for _ in list_quantizers:
                        list_model_odq.append(generate_model(N_x, N_y, std_noise=0, lr=lr, decay=decay, optimizer='sgd'))
                    model_reservoir = generate_model(N_x, N_y, std_noise=0, lr=lr, decay=decay, optimizer='sgd')

                    # Perform training from reservoir data first, saving the validation set
                    time_start = time.time()
                    print('    Generating from Reservoir-reduced Data')
                    X_temp, Y_temp = reservoir_sampler.get_dataset()

                    X_temp = min_max_scaler_x.transform(X_temp)
                    Y_temp = min_max_scaler_y.transform(Y_temp)

                    X_fit, X_val, Y_fit, Y_val = train_test_split(X_temp, Y_temp, pct_train=TRAIN_VAL_RATIO)

                    history_temp = model_reservoir.fit(X_fit, Y_fit, batch_size=32, epochs=N_epochs, verbose=0,
                                                       validation_data=(X_val, Y_val))
                    history_reservoir = history_temp.history
                    history_reservoir['epoch'] = history_temp.epoch

                    score_reservoir = model_reservoir.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test), verbose=0)

                    Y_reservoir_predict = min_max_scaler_y.inverse_transform(model_reservoir.predict(min_max_scaler_x.transform(X_test)))

                    results_reservoir_rmse = np.sqrt(np.mean((Y_reservoir_predict - Y_test)**2, axis=0))

                    print('    RMSE: {0}'.format(np.array2string(results_reservoir_rmse, precision=2, suppress_small=True)))
                    print('    Time: {0:0.2f} s'.format(time.time() - time_start))

                    plt.figure()
                    plt.title('Reservoir trial {0} lr={1} decay={2}'.format(ind_loop, lr, decay))
                    plt.plot(history_reservoir['epoch'], history_reservoir['mean_squared_error'])
                    plt.plot(history_reservoir['epoch'], history_reservoir['val_mean_squared_error'])
                    plt.legend(('Train', 'Test'))

                    dict_out = {'history_reservoir': history_reservoir, 'score_reservoir': score_reservoir,
                                'Y_reservoir_predict': Y_reservoir_predict, 'Y_test': Y_test, 'X_test': X_test,
                                'N_datapoints': N_datapoints}

                    for dict_quantizer, model_odq in zip(list_quantizers, list_model_odq):
                        quantizer = dict_quantizer['quantizer']
                        w_x = dict_quantizer['w_x']
                        w_y = dict_quantizer['w_y']
                        w_imp = dict_quantizer['w_imp']
                        w_type = dict_quantizer['w_type']

                        time_start = time.time()
                        print('    Generating model from ODQ-reduced Data, weight type {0}'.format(w_type))
                        X_temp, Y_temp = quantizer.get_dataset()
                        w_temp = quantizer.get_sample_weights()

                        w_temp = w_temp * w_temp.shape[0] / np.sum(w_temp)

                        X_temp = min_max_scaler_x.transform(X_temp)
                        Y_temp = min_max_scaler_y.transform(Y_temp)

                        # Train using validation set from reservoir sampling. Size is already taken into account in
                        # generate_reduced_datasets.py
                        history_temp = model_odq.fit(X_temp, Y_temp, batch_size=32, epochs=N_epochs, sample_weight=w_temp, verbose=0,
                                                     validation_data=(X_val, Y_val))
                        history_odq = history_temp.history
                        history_odq['epoch'] = history_temp.epoch

                        score_odq = model_odq.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test), verbose=0)

                        Y_odq_predict = min_max_scaler_y.inverse_transform(model_odq.predict(min_max_scaler_x.transform(X_test)))

                        results_odq_rmse = np.sqrt(np.mean((Y_odq_predict - Y_test)**2, axis=0))

                        print('    RMSE: {0}'.format(np.array2string(results_odq_rmse, precision=2, suppress_small=True)))
                        print('    Time: {0:0.2f} s'.format(time.time() - time_start))

                        plt.figure()
                        plt.title('Weight-type {3} trial {0} lr={1} decay={2}'.format(ind_loop, lr, decay, dict_quantizer['w_type']))
                        plt.plot(history_odq['epoch'], history_odq['mean_squared_error'])
                        plt.plot(history_odq['epoch'], history_odq['val_mean_squared_error'])
                        plt.legend(('Train', 'Test'))

                        dict_out['history_odq_w{0}'.format(w_type)] = history_odq
                        dict_out['score_odq_w{0}'.format(w_type)] = score_odq
                        dict_out['Y_odq_predict_w{0}'.format(w_type)] = Y_odq_predict

                        # Save all results for subsequent processing
                        directory_target_full = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw', directory_target)
                        if not os.path.exists(directory_target_full):
                            os.mkdir(directory_target_full)

                    if FLAG_SAVE_MODEL:
                        with open(os.path.join(directory_target_full,
                                               filename_base + 'lr{1}_decay{3}_std{2}_models_trial{0}_reduced.pkl'.format(ind_loop, lr, 0, decay)), 'wb') as fid:
                            dict_out['model_odq'] = model_odq
                            dict_out['model_reservoir'] = model_reservoir
                            pkl.dump(dict_out, fid)
                    else:
                        with open(os.path.join(directory_target_full,
                                               filename_base + 'lr{1}_decay{3}_std{2}_results_trial{0}_reduced.pkl'.format(ind_loop, lr, 0, decay)), 'wb') as fid:
                            pkl.dump(dict_out, fid)

                    # Reset Tensorflow session to prevent memory growth
                    K.clear_session()

                    # Configure tensorflow
                    num_cores = 4
                    if args.cpu:
                        num_GPU = 1
                        num_CPU = 3
                    else:
                        num_CPU = 3
                        num_GPU = 0

                    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                                            inter_op_parallelism_threads=num_cores,
                                            allow_soft_placement=True,
                                            device_count={'CPU': num_CPU,
                                                          'GPU': num_GPU}
                                            )

                    session = tf.Session(config=config)
                    K.set_session(session)

                plt.show() # Show after all learning rates and decays have been processed