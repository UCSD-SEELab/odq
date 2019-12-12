import sys
import os
import time
import argparse
from functools import partial
from multiprocessing import Pool
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot
from datetime import datetime
import configparser
import json

import numpy as np
from keras.callbacks import EarlyStopping
from keras import backend as K
import tensorflow as tf

from ml_models import generate_model_server_power, generate_model_home_energy, generate_model_metasense, train_test_split, train_test_split_blocks, generate_model_square
import pickle as pkl

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from odq.odq import OnlineDatasetQuantizer
from odq.reservoir import ReservoirSampler
from odq.data import home_energy, server_power, metasense


def config_tf_session(b_cpu):
    """
    Configure tensorflow session for CPU (b_cpu=True) or GPU (b_cpu=False) operation for up to 3 CPUs per instance
    """
    # Configure tensorflow
    num_cores = 3
    if b_cpu:
        num_GPU = 0
        num_CPU = num_cores
    else:
        num_GPU = 1
        num_CPU = num_cores

    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores,
                            allow_soft_placement=True,
                            device_count={'CPU': num_CPU,
                                          'GPU': num_GPU}
                            )

    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

def run_nn_tests(filename, dir_quant, dir_target, N_trials=3, b_cpu=True,
                 TRAIN_VAL_RATIO = 0.8, list_models=[{'desc': 'NN_default'}]):
    """
    Generate results for neural network processing of the quantized dataset

    Input dictionary for list_models:

        list_models (list)
            desc
            cfg


    Structure of input dictionary from generate_reduced_datasets.py

        dict_in
            dataset_name
            compression_ratio
            X_test
            Y_test
            X_train
            Y_train
            min_max_scaler_x
            min_max_scaler_y
            train_test_ratio
            train_val_ratio
            quantizers (list)
                desc
                quantizer

    Structure of output dictionary:

        dict_out
            dataset_name
            compression_ratio
            trial_num
            min_max_scaler_x
            min_max_scaler_y
            X_test
            Y_test
            X_val
            Y_val
            filename_datasets
            quantizer_results (list)
                desc
                model_results (list)
                    desc (from list_models)
                    cfg (from list_models)
                    history
                    score
                    Y_predict
                    rmse
                    t_train
    """

    # Check that file is valid
    if not (filename.lower().endswith('.pkl')):
        return
    print ("working on : " + filename)

    filename_base = filename.replace('quantized.pkl', '')
    str_testtime = datetime.now().strftime('%Y%m%d%H%M%S')
    try:
        with open(os.path.join(dir_quant, dir_target, filename), 'rb') as fid:
            dict_in = pkl.load(fid)
        min_max_scaler_x = dict_in['min_max_scaler_x']
        min_max_scaler_y = dict_in['min_max_scaler_y']
        list_quantizers = dict_in['quantizers']
        X_train = dict_in['X_train']
        Y_train = dict_in['Y_train']
        X_test = dict_in['X_test']
        Y_test = dict_in['Y_test']

        if 'dataset_name' in dict_in:
            dataset_name = dict_in['dataset_name']
        else:
            filename_parts = filename.split('_')
            if filename_parts[1] == 'data':
                dataset_name = '{0}'.format(filename_parts[0])
            else:
                dataset_name = '{0}_{1}'.format(filename_parts[0], filename_parts[1])
        DATASET = eval(dataset_name)

        if 'compression_ratio' in dict_in:
            compression_ratio = dict_in['compression_ratio']
        else:
            compression_ratio = float(filename_parts[-3])

    except:
        print('ERROR loading from {0}. Skipping.'.format(filename))
        print(sys.exc_info()[0])
        return

    # Process data
    N_datapoints = X_train.shape[0]
    N_x = X_train.shape[1]
    N_y = Y_train.shape[1]

    for ind_loop in range(N_trials):
        dict_out = {'dataset_name': dataset_name,
                    'compression_ratio': compression_ratio,
                    'trial_num': ind_loop,
                    'X_test': X_test,
                    'Y_test': Y_test,
                    'min_max_scaler_x': min_max_scaler_x,
                    'min_max_scaler_y': min_max_scaler_y,
                    'filename_datasets': filename}

        # Find the reservoir data, process, and save validation set
        quantizer_res = next(entry for entry in list_quantizers if entry['desc'] == 'reservoir')
       # X_temp, Y_temp = quantizer_res['quantizer'].get_dataset()

        X_temp = min_max_scaler_x.transform(X_train)
        Y_temp = min_max_scaler_y.transform(Y_train)

        X_fit_res, X_val, Y_fit_res, Y_val = train_test_split(X_temp, Y_temp, pct_train=TRAIN_VAL_RATIO)

        list_quantizer_results = []

        temp_dict = {'desc': 'NN_custom', 'model_results': [entry.copy() for entry in list_models]}

        for model_config in temp_dict['model_results']:
            config_tf_session(b_cpu)

            # Set correct number of epochs
            if 'epochs' in model_config:
                N_epochs = model_config['epochs']
            else:
                if DATASET is home_energy:
                    N_epochs = int(max(500, 50 * compression_ratio))
                elif DATASET is metasense:
                    N_epochs = int(max(500, 50 * compression_ratio))
                elif DATASET is server_power:
                    N_epochs = int(max(500, 50 * compression_ratio))
                else:
                    N_epochs = int(max(500, 50 * compression_ratio))

            if model_config['desc'] == 'NN_default':
                # Required parameters:
                #  none
                if DATASET is server_power:
                    model = generate_model_server_power(N_x, N_y)
                elif DATASET is home_energy:
                    model = generate_model_home_energy(N_x, N_y)
                elif DATASET is metasense:
                    model = generate_model_metasense(N_x, N_y)

            elif model_config['desc'] == 'NN_custom':
                # Required parameters:
                #  func_custommodel
                #  model_cfg
                if 'func_custommodel' in model_config:
                    model = model_config['func_custommodel'](N_x, N_y, model_cfg=model_config['cfg'])
                else:
                    if DATASET is server_power:
                        model = generate_model_server_power(N_x, N_y, model_cfg=model_config['cfg'])
                    elif DATASET is home_energy:
                        model = generate_model_home_energy(N_x, N_y, model_cfg=model_config['cfg'])
                    elif DATASET is metasense:
                        model = generate_model_metasense(N_x, N_y, model_cfg=model_config['cfg'])

            elif model_config['desc'] == 'lstsq':
                # Required parameters:
                #  none

                continue

            elif model_config['desc'] == 'knn':
                # Required parameters:
                #  k (number of neighboring points to use in prediction)

                continue

            else:
                print('ERROR: Unrecognized model {0}'.format(model_config['desc']))
                continue

            #if not(dict_quantizer['desc'] == 'reservoir'):
               # print ("")
               # X_temp, Y_temp = dict_quantizer['quantizer'].get_dataset()
              #  sample_weight = dict_quantizer['quantizer'].get_sample_weights()

               # X_fit = min_max_scaler_x.transform(X_temp)
                #Y_fit = min_max_scaler_y.transform(Y_temp)
          #  else:
               # X_fit = X_fit_res
                #Y_fit = Y_fit_res
                sample_weight = None

            time_start = time.time()
            history_temp = model.fit(X_fit_res, Y_fit_res, batch_size=64, epochs=N_epochs, verbose=0,
                                     validation_data=(X_val, Y_val))
            time_end = time.time()
            score_temp = model.evaluate(min_max_scaler_x.transform(X_test),
                                        min_max_scaler_y.transform(Y_test), verbose=0)
            Y_predict = min_max_scaler_y.inverse_transform(model.predict(min_max_scaler_x.transform(X_test)))
            results_rmse = np.sqrt(np.mean((Y_predict - Y_test) ** 2, axis=0))[0]
            history = history_temp.history
            history['epoch'] = history_temp.epoch


            # Save results in dictionary structure
            # (Note: history values were taking up substantial space for very large epoch numbers. Reduced the
            #  number of saved samples using ind_subsample.)
            ind_subsample = list(range(1, 500, 2)) + list(range(500, 1000, 5)) + \
                            list(range(1000, 5000, 10))
            if ind_subsample[-1] < max(history['epoch']):
                ind_subsample += list(range(5000, max(history['epoch']) + 1, 25))
            else:
                ind_subsample = [ind for ind in ind_subsample if ind <= max(history['epoch'])]
            for key in history:
                history[key] = [history[key][ind] for ind in ind_subsample]
            model_config['history'] = history
            model_config['Y_predict'] = Y_predict
            model_config['rmse'] = results_rmse
            model_config['score'] = score_temp
            model_config['t_train'] = time_end - time_start

            # Reset Tensorflow session to prevent memory growth
            K.clear_session()

            # Build list of results
        list_quantizer_results.append(temp_dict)

        dict_out['quantizer_results'] = list_quantizer_results

        print('{0} CR {1} ({2} of {3}) complete.'.format(filename_base, compression_ratio, ind_loop+1, N_trials))
        for quantizer_result in list_quantizer_results:
            print('  {0}'.format(quantizer_result['desc']))
            for model_config in quantizer_result['model_results']:
                print('    {0}: {1:0.2f} ({2:0.2f} s)'.format(model_config['desc'], model_config['rmse'], model_config['t_train']))

        # Save all results for subsequent processing
        dir_target_full = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw', dir_target)
        if not os.path.exists(dir_target_full):
            os.mkdir(dir_target_full)

        with open(os.path.join(dir_target_full,
                               filename_base + 'test{0}_trial{1}_results.pkl'.format(str_testtime, ind_loop)), 'wb') as fid:
            pkl.dump(dict_out, fid)
        with open(os.path.join(dir_target_full,
                               filename_base + 'test{0}_trial{1}_results.txt'.format(str_testtime, ind_loop)), 'wb') as fid:
            json.dump(dict_out, fid)




def run_convergence_tests(filename, dir_quant, dir_target, N_trials=3, b_cpu=True, list_lr=[0.001], list_decay=[0.001],
                 TRAIN_VAL_RATIO = 0.8, b_usefreq=True, b_costmae=False):
    """
    Generate results for neural network processing of the quantized dataset
    """
    return
    """
    FLAG_OVERWRITE = False
    FLAG_SAVEIMG = True
    dir_img_full = os.path.join(dir_quant, '..', 'img', dir_target)

    # Check that file is valid
    if not (filename.lower().endswith('.pkl')):
        return

    filename_base = filename.replace('quantized.pkl', '')

    print('{0}: Loading data'.format(filename))

    try:
        with open(os.path.join(dir_quant, dir_target, filename), 'rb') as fid:
            data_temp = pkl.load(fid)
        min_max_scaler_x = data_temp['min_max_scaler_x']
        min_max_scaler_y = data_temp['min_max_scaler_y']
        if 'quantizers' in data_temp:
            list_quantizers = data_temp['quantizers']
        else:
            list_quantizers = [{'quantizer':data_temp['quantizer']}]
        reservoir_sampler = data_temp['reservoir_sampler']
        X_train = data_temp['X_train']
        Y_train = data_temp['Y_train']
        X_test = data_temp['X_test']
        Y_test = data_temp['Y_test']
        if 'quantizer_type' in data_temp:
            quantizer_type = data_temp['quantizer_type']
        else:
            quantizer_type = 'unknown'

        filename_parts = filename.split('_')
        if filename_parts[1] == 'data':
            DATASET = eval('{0}'.format(filename_parts[0]))
        else:
            DATASET = eval('{0}_{1}'.format(filename_parts[0], filename_parts[1]))
        compression_ratio = float(filename_parts[-3])
        print('  SUCCESS')
    except:
        print('  ERROR. Skipping.')
        return

    # Process data
    N_datapoints = X_train.shape[0]
    N_x = X_train.shape[1]
    N_y = Y_train.shape[1]

    if DATASET is home_energy:
        N_epochs = int(max(200, 50*compression_ratio))
    else:
        N_epochs = int(max(200, 30*compression_ratio))

    dir_target_full = os.path.join(dir_quant, '..', 'raw', dir_target)
    if not os.path.exists(dir_target_full):
        os.mkdir(dir_target_full)

    if not os.path.exists(dir_img_full):
        os.mkdir(dir_img_full)

    for ind_loop in range(N_trials):
        print('{0}: Trial {1} of {2}:'.format(filename, ind_loop + 1, N_trials))

        for lr in list_lr:
            for decay in list_decay:

                if not(FLAG_SAVEIMG) and os.path.isfile(
                        os.path.join(dir_target_full, filename_base + '{0}_lr{1}_c{2}_results_trial{3}_reduced.pkl'.format(quantizer_type, lr, ind_loop, lr, costtype, ind_loop))):
                    print('  File already processed')
                    continue

                # Create machine learning models for each evaluation step
                list_model_quant = []
                if DATASET is server_power:
                    generate_model = generate_model_server_power
                elif DATASET is home_energy:
                    generate_model = generate_model_home_energy
                elif DATASET is metasense:
                    generate_model = generate_model_metasense

                for _ in list_quantizers:
                    list_model_quant.append(generate_model(N_x, N_y, lr=lr, decay=decay, optimizer='sgd'))
                model_reservoir = generate_model(N_x, N_y, lr=lr, decay=decay, optimizer='sgd')

                # Perform training from reservoir data first, saving the validation set
                time_start = time.time()
                X_temp, Y_temp = reservoir_sampler.get_dataset()

                X_temp = min_max_scaler_x.transform(X_temp)
                Y_temp = min_max_scaler_y.transform(Y_temp)

                X_fit, X_val, Y_fit, Y_val = train_test_split(X_temp, Y_temp, pct_train=TRAIN_VAL_RATIO)

                history_temp = model_reservoir.fit(X_fit, Y_fit, batch_size=32, epochs=N_epochs, verbose=0,
                                                   validation_data=(X_val, Y_val))
                history_reservoir = history_temp.history
                history_reservoir['epoch'] = history_temp.epoch

                score_reservoir = model_reservoir.evaluate(min_max_scaler_x.transform(X_test),
                                                           min_max_scaler_y.transform(Y_test), verbose=0)

                Y_reservoir_predict = min_max_scaler_y.inverse_transform(
                    model_reservoir.predict(min_max_scaler_x.transform(X_test)))

                results_reservoir_rmse = np.sqrt(np.mean((Y_reservoir_predict - Y_test) ** 2, axis=0))

                print('{0}: Reservoir LR: {1} Decay: {2} RMSE: {3} Time: {4:0.2f} s'.format(filename, lr, decay, np.array2string(results_reservoir_rmse, precision=2, suppress_small=True), time.time() - time_start))

                plt.figure()
                plt.title('Reservoir CR {3} trial {0} lr={1} decay={2}'.format(ind_loop, lr, decay, compression_ratio))
                plt.plot(history_reservoir['epoch'], history_reservoir['mean_squared_error'])
                plt.plot(history_reservoir['epoch'], history_reservoir['val_mean_squared_error'])
                plt.legend(('Train', 'Test'))
                plt.savefig(os.path.join(dir_img_full, 'Convergence_' + filename_base + '{0}_lr{1}_c{2}_results_trial{3}_reduced.pkl'.format(quantizer_type, lr, ind_loop, lr, costtype, ind_loop)))
                plt.close()

                dict_out = {'history_reservoir': history_reservoir, 'score_reservoir': score_reservoir,
                            'Y_reservoir_predict': Y_reservoir_predict, 'Y_test': Y_test, 'X_test': X_test,
                            'N_datapoints': N_datapoints}

                for dict_quantizer, model_quant in zip(list_quantizers, list_model_quant):
                    quantizer = dict_quantizer['quantizer']

                    time_start = time.time()
                    X_temp, Y_temp = quantizer.get_dataset()
                    w_temp = quantizer.get_sample_weights()

                    w_temp = w_temp * w_temp.shape[0] / np.sum(w_temp)

                    X_temp = min_max_scaler_x.transform(X_temp)
                    Y_temp = min_max_scaler_y.transform(Y_temp)

                    # Train using validation set from reservoir sampling. Size is already taken into account in
                    # generate_reduced_datasets.py
                    history_temp = model_quant.fit(X_temp, Y_temp, batch_size=32, epochs=N_epochs, sample_weight=w_temp,
                                                 verbose=0,
                                                 validation_data=(X_val, Y_val))
                    history_quant = history_temp.history
                    history_quant['epoch'] = history_temp.epoch

                    score_quant = model_quant.evaluate(min_max_scaler_x.transform(X_test),
                                                   min_max_scaler_y.transform(Y_test), verbose=0)

                    Y_quant_predict = min_max_scaler_y.inverse_transform(
                        model_quant.predict(min_max_scaler_x.transform(X_test)))

                    results_quant_rmse = np.sqrt(np.mean((Y_quant_predict - Y_test) ** 2, axis=0))

                    print('{0}: {5} LR: {1} Decay: {2} RMSE: {3} Time: {4:0.2f} s'.format(filename, lr, decay, np.array2string(results_quant_rmse, precision=2, suppress_small=True), time.time() - time_start, quantizer))

                    plt.figure()
                    plt.title('Quantizer {4} CR {3} trial {0} lr={1} decay={2}'.format(ind_loop, lr, decay, compression_ratio,
                                                                                  quantizer_type))
                    plt.plot(history_quant['epoch'], history_quant['mean_squared_error'])
                    plt.plot(history_quant['epoch'], history_quant['val_mean_squared_error'])
                    plt.legend(('Train', 'Test'))
                    plt.savefig(os.path.join(dir_img_full, 'Convergence_' + filename_base + '{0}_lr{1}_c{2}_results_trial{3}_reduced.pkl'.format(quantizer_type, lr, ind_loop, lr, costtype, ind_loop)))
                    plt.close()

                    dict_out['history_{0}'.format(quantizer_type)] = history_quant
                    dict_out['score_{0}'.format(quantizer_type)] = score_quant
                    dict_out['Y_quant_predict_{0}'.format(quantizer_type)] = Y_quant_predict

                # Save all results for subsequent processing
                with open(os.path.join(dir_target_full, filename_base + '{0}_lr{1}_c{2}_results_trial{3}_reduced.pkl'.format(quantizer_type, lr, ind_loop, lr, costtype, ind_loop)), 'wb') as fid:
                    pkl.dump(dict_out, fid)

                # Reset Tensorflow session to prevent memory growth
                K.clear_session()
    """

def run_nn_tests_with_lossF(filename, dir_quant, dir_target, N_trials=3, b_cpu=True, list_lr=[0.0001],
                 TRAIN_VAL_RATIO = 0.8, b_usefreq=True, b_costmae=False , loss_fnc = 'mean_absolute_error'):
    """
    Generate results for neural network processing of the quantized dataset using custom loss functions
    """
    return
    """
    FLAG_OVERWRITE = False

    # Check that file is valid
    if not (filename.lower().endswith('.pkl')):
        return

    filename_base = filename.replace('quantized.pkl', '')

    print('\n\nLoading data from {0}'.format(filename))

    try:
        with open(os.path.join(dir_quant, dir_target, filename), 'rb') as fid:
            data_temp = pkl.load(fid)
        min_max_scaler_x = data_temp['min_max_scaler_x']
        min_max_scaler_y = data_temp['min_max_scaler_y']

        X_train = data_temp['X_train']
        Y_train = data_temp['Y_train']
        X_test = data_temp['X_test']
        Y_test = data_temp['Y_test']
        if 'quantizer_type' in data_temp:
            quantizer_type = data_temp['quantizer_type']
        else:
            quantizer_type = 'unknown'

        filename_parts = filename.split('_')
        if filename_parts[1] == 'data':
            DATASET = eval('{0}'.format(filename_parts[0]))
        else:
            DATASET = eval('{0}_{1}'.format(filename_parts[0], filename_parts[1]))
        compression_ratio = float(filename_parts[-3])
        print('  SUCCESS')
    except:
        print('  ERROR. Skipping.')
        return

    # Process data
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
            config_tf_session(b_cpu)

            print('  lr: {0}'.format(lr))

            if not(FLAG_OVERWRITE) and os.path.isfile(
                    os.path.join(os.path.dirname(__file__), '..', 'results', 'raw',
                                 dir_target,
                                 filename_base + '{0}_lr{1}_c{2}_results_trial{3}_reduced.pkl'.format(quantizer_type, lr, ind_loop, lr, costtype, ind_loop))):
                print('  File already processed')
                continue

            # Create an early stopping callback appropriate for the dataset size
            cb_earlystopping = EarlyStopping(monitor='val_loss',
                                             patience=max([20, min([compression_ratio*lr/0.00005, 250])]),
                                             restore_best_weights=True)

            # Create machine learning models for each evaluation step
            list_model_quant = []
            if DATASET is server_power:
                generate_model = generate_model_server_power
            elif DATASET is home_energy:
                generate_model = generate_model_home_energy
            elif DATASET is metasense:
                generate_model = generate_model_metasense

            model_reservoir = generate_model(N_x, N_y, lr=lr, loss=loss_fnc)

            # Perform training from reservoir data first, saving the validation set
            time_start = time.time()
          #  print('    Generating from Reservoir-reduced Data')
           # X_temp, Y_temp = reservoir_sampler.get_dataset()

            X_temp = min_max_scaler_x.transform(X_train)
            Y_temp = min_max_scaler_y.transform(Y_train)

            X_fit, X_val, Y_fit, Y_val = train_test_split(X_temp, Y_temp, pct_train=TRAIN_VAL_RATIO)

            # Train the model on 10 epochs before checking for early stopping conditions to prevent premature return
            history_temp = model_reservoir.fit(X_fit, Y_fit, batch_size=64, epochs=10, verbose=0,
                                               validation_data=(X_val, Y_val))
            history_reservoir = history_temp.history
            history_reservoir['epoch'] = history_temp.epoch
            history_temp = model_reservoir.fit(X_fit, Y_fit, batch_size=64, epochs=N_epochs, verbose=0,
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

            dict_out = {'history_reservoir': history_reservoir, 'score_reservoir': score_reservoir,
                        'Y_reservoir_predict': Y_reservoir_predict, 'Y_test': Y_test, 'X_test': X_test,
                        'N_datapoints': N_datapoints}


                # Save all results for subsequent processing
            dir_target_full = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw', dir_target)
            if not os.path.exists(dir_target_full):
                os.mkdir(dir_target_full)

            with open(os.path.join(dir_target_full,
                                   filename_base + '{0}_lr{1}_c{2}_results_trial{3}_reduced.pkl'.format(quantizer_type, lr, ind_loop, lr, costtype, ind_loop)), 'wb') as fid:
                pkl.dump(dict_out, fid)

            # Reset Tensorflow session to prevent memory growth
            K.clear_session()
    """

def run_sq_nn_tests(filename, dir_quant, dir_target, N_trials=3, b_cpu=True, list_lr=[0.0001],
                 TRAIN_VAL_RATIO=0.8, N_depth=2, N_weights=4000, b_usefreq=True):
    """
    Generate results for neural network processing of the quantized dataset
    """
    return
    """
    FLAG_OVERWRITE = False

    # Check that file is valid
    if not (filename.lower().endswith('.pkl')):
        return

    filename_base = filename.replace('quantized.pkl', '')

    print('\n\nLoading data from {0}'.format(filename))

    try:
        with open(os.path.join(dir_quant, dir_target, filename), 'rb') as fid:
            data_temp = pkl.load(fid)
        min_max_scaler_x = data_temp['min_max_scaler_x']
        min_max_scaler_y = data_temp['min_max_scaler_y']
        if 'quantizers' in data_temp:
            list_quantizers = data_temp['quantizers']
        else:
            list_quantizers = [{'quantizer':data_temp['quantizer']}]
        reservoir_sampler = data_temp['reservoir_sampler']
        X_train = data_temp['X_train']
        Y_train = data_temp['Y_train']
        X_test = data_temp['X_test']
        Y_test = data_temp['Y_test']
        if 'quantizer_type' in data_temp:
            quantizer_type = data_temp['quantizer_type']
        else:
            quantizer_type = 'unknown'

        filename_parts = filename.split('_')
        if filename_parts[1] == 'data':
            DATASET = eval('{0}'.format(filename_parts[0]))
        else:
            DATASET = eval('{0}_{1}'.format(filename_parts[0], filename_parts[1]))
        compression_ratio = float(filename_parts[-3])
        print('  SUCCESS')
    except:
        print('  ERROR. Skipping.')
        return

    # Process data
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
            config_tf_session(b_cpu)

            print('  lr: {0}'.format(lr))

            if not(FLAG_OVERWRITE) and os.path.isfile(
                    os.path.join(os.path.dirname(__file__), '..', 'results', 'raw',
                                 dir_target,
                                 filename_base + '{0}_lr{1}_c{2}_depth{4}_ram{5}_results_trial{3}_reduced.pkl'.format(
                                     quantizer_type, lr, ind_loop, lr, costtype, ind_loop, N_depth, N_weights))):
                print('  File already processed')
                continue

            # Create an early stopping callback appropriate for the dataset size
            cb_earlystopping = EarlyStopping(monitor='val_loss',
                                             patience=max([20, min([compression_ratio*lr/0.00005, 250])]),
                                             restore_best_weights=True)

            # Create machine learning models for each evaluation step
            list_model_quant = []



            if DATASET is server_power:
                generate_model = generate_model_square
            elif DATASET is home_energy:
                generate_model = generate_model_square
            elif DATASET is metasense:
                generate_model = generate_model_square

            for _ in list_quantizers:
                list_model_quant.append(generate_model(N_x, N_y, N_depth=N_depth, N_weights=N_weights, lr=lr))
            model_reservoir = generate_model(N_x, N_y, N_depth=N_depth, N_weights=N_weights, lr=lr)

            # Perform training from reservoir data first, saving the validation set
            time_start = time.time()
            print('    Generating from Reservoir-reduced Data')
            X_temp, Y_temp = reservoir_sampler.get_dataset()

            X_temp = min_max_scaler_x.transform(X_temp)
            Y_temp = min_max_scaler_y.transform(Y_temp)

            X_fit, X_val, Y_fit, Y_val = train_test_split(X_temp, Y_temp, pct_train=TRAIN_VAL_RATIO)

            # Train the model on 10 epochs before checking for early stopping conditions to prevent premature return
            history_temp = model_reservoir.fit(X_fit, Y_fit, batch_size=64, epochs=10, verbose=0,
                                               validation_data=(X_val, Y_val))
            history_reservoir = history_temp.history
            history_reservoir['epoch'] = history_temp.epoch
            history_temp = model_reservoir.fit(X_fit, Y_fit, batch_size=64, epochs=N_epochs, verbose=0,
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

            dict_out = {'history_reservoir': history_reservoir, 'score_reservoir': score_reservoir,
                        'Y_reservoir_predict': Y_reservoir_predict, 'Y_test': Y_test, 'X_test': X_test,
                        'N_datapoints': N_datapoints, 'N_trials': N_trials, 'list_lr': list_lr,
                        'TRAIN_VAL_RATIO': TRAIN_VAL_RATIO, 'Number_of_Layers': N_depth, 'Device_RAM_Size': N_weights}

            for dict_quantizer, model_quant in zip(list_quantizers, list_model_quant):
                quantizer = dict_quantizer['quantizer']

                time_start = time.time()
                print('    Generating model from {0}-reduced Data'.format(quantizer_type))
                X_temp, Y_temp = quantizer.get_dataset()
                w_temp = quantizer.get_sample_weights()

                w_temp = w_temp * w_temp.shape[0] / np.sum(w_temp)

                X_temp = min_max_scaler_x.transform(X_temp)
                Y_temp = min_max_scaler_y.transform(Y_temp)

                # Train using validation set from reservoir sampling. Size is already taken into account in
                # generate_reduced_datasets.py
                if b_usefreq:
                    sample_weight = w_temp
                else:
                    sample_weight = None
                history_temp = model_quant.fit(X_temp, Y_temp, batch_size=64, epochs=10, sample_weight=sample_weight, verbose=0,
                                             validation_data=(X_val, Y_val))
                history_quant = history_temp.history
                history_quant['epoch'] = history_temp.epoch
                history_temp = model_quant.fit(X_temp, Y_temp, batch_size=64, epochs=N_epochs, sample_weight=sample_weight, verbose=0,
                                             validation_data=(X_val, Y_val), initial_epoch=10,
                                             callbacks=[cb_earlystopping])
                history_quant['epoch'].extend(history_temp.epoch)
                history_quant['val_loss'].extend(history_temp.history['val_loss'])
                history_quant['val_mean_squared_error'].extend(history_temp.history['val_mean_squared_error'])
                history_quant['val_mean_absolute_error'].extend(history_temp.history['val_mean_absolute_error'])
                history_quant['loss'].extend(history_temp.history['loss'])
                history_quant['mean_squared_error'].extend(history_temp.history['mean_squared_error'])
                history_quant['mean_absolute_error'].extend(history_temp.history['mean_absolute_error'])

                score_quant = model_quant.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test), verbose=0)

                Y_quant_predict = min_max_scaler_y.inverse_transform(model_quant.predict(min_max_scaler_x.transform(X_test)))

                results_quant_rmse = np.sqrt(np.mean((Y_quant_predict - Y_test)**2, axis=0))

                print('    RMSE: {0}'.format(np.array2string(results_quant_rmse, precision=2, suppress_small=True)))
                print('    Time: {0:0.2f} s'.format(time.time() - time_start))

                dict_out['history_{0}'.format(quantizer_type)] = history_quant
                dict_out['score_{0}'.format(quantizer_type)] = score_quant
                dict_out['Y_quant_predict_{0}'.format(quantizer_type)] = Y_quant_predict

                # Save all results for subsequent processing
                dir_target_full = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw', dir_target)
                if not os.path.exists(dir_target_full):
                    os.mkdir(dir_target_full)

            with open(os.path.join(dir_target_full,
                                   filename_base + '{0}_lr{1}_c{2}_depth{4}_ram{5}_results_trial{3}_reduced.pkl'.format(
                                       quantizer_type, lr, ind_loop, lr, costtype, ind_loop, N_depth, N_weights)), 'wb') as fid:
                pkl.dump(dict_out, fid)

            # Reset Tensorflow session to prevent memory growth
            K.clear_session()
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='Target directory of files.')
    parser.add_argument('--N', type=int, nargs=1, help='Number of trials to run.', default=[3])
    parser.add_argument('--cfg', type=str, help='Target for test configuration file.')
    parser.add_argument('--cpu', help='Flag to use CPU version of Tensorflow (GPU by default).', action='store_true')
    parser.add_argument('--Nproc', type=int, help='Number of simultaneous processes to run.', default=1)
    """
    parser.add_argument('--lr', type=float, nargs='+', help='Learning rates to use.', default=[0.0001])
    parser.add_argument('--decay', type=float, nargs='+', help='Decay rate to use for optimizer.')
    parser.add_argument('--usefreq', help='Flag to use frequency of samples in training', action='store_true')
    parser.add_argument('--test', type=str, help='Type of test to run (nn, convergence, loss, nnarch)')
    parser.add_argument('--loss', type=str, help='loss function method (sigmoid, step, mean_squared_error, mean_absolute_error)', default='mean_squared_error')
    parser.add_argument('--march', type=str, nargs=1, help='Type of ANN (sq, default).', default=['default'])
    parser.add_argument('--mdepth', type=int, nargs='+', help='Number of hidden ANN layesrs', default=[2])
    parser.add_argument('--mram', type=int, nargs='+', help='Device RAM size', default=[4000])
    """

    config = configparser.ConfigParser()

    args = parser.parse_args()

    if args.dir is not None:
        dir_target = args.dir
    else:
        print('ERROR: No directory provided. Use \'--dir DIRECTORY\' to provide a target directory.')
        sys.exit()

    if args.cfg is not None:
        config.read(os.path.join(os.path.dirname(__file__), 'configs', args.cfg))

        try:
            list_models = [json.loads(config['list_models'][entry].replace('\'', '"')) for entry in config['list_models']]
        except:
            print('ERROR: Config file parsed incorrectly.')
            print(sys.exc_info()[0])
            sys.exit()
    else:
        list_models = [{'desc': 'NN_default'}]

    N_trials = args.N[0]

    dir_quant = os.path.join(os.path.dirname(__file__), '..', 'results', 'quantized')

    p_run_tests = partial(run_nn_tests, dir_quant=dir_quant, dir_target=dir_target, N_trials=N_trials,
                          list_models=list_models, TRAIN_VAL_RATIO=0.8)

    DEBUG = False
    if DEBUG == True:
        for filename in os.listdir(os.path.join(dir_quant, dir_target)):
            p_run_tests(filename=filename)
    else:
        with Pool(args.Nproc) as p:
            p.map(p_run_tests,
                  [filename for filename in os.listdir(os.path.join(dir_quant, dir_target)) if
                   (filename.lower().endswith('.pkl'))])
