import sys
import os
import time
import argparse
from functools import partial
from multiprocessing import Pool

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

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores,
                            allow_soft_placement=True,
                            device_count={'CPU': num_CPU,
                                          'GPU': num_GPU}
                            )

    session = tf.Session(config=config)
    K.set_session(session)

def run_nn_tests(filename, dir_quant, dir_target, N_trials=3, b_cpu=True, list_lr=[0.0001], list_std_noise=[0.001],
                 TRAIN_VAL_RATIO = 0.8, b_usefreq=True, b_costmae=False):
    """
    Generate results for neural network processing of the quantized dataset
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
            list_quantizers = [{'quantizer':data_temp['quantizer'], 'w_x': [0], 'w_y': [0], 'w_imp': [0], 'w_type': 0}]
        reservoir_sampler = data_temp['reservoir_sampler']
        X_train = data_temp['X_train']
        Y_train = data_temp['Y_train']
        X_test = data_temp['X_test']
        Y_test = data_temp['Y_test']

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
            for std_noise in list_std_noise:
                config_tf_session(b_cpu)

                print('  lr: {0}   std_noise: {1}:'.format(lr, std_noise))

                if not(FLAG_OVERWRITE) and os.path.isfile(
                        os.path.join(os.path.dirname(__file__), '..', 'results', 'raw',
                                     dir_target,
                                     filename_base + 'lr{1}_std{2}_f{3}_c{4}_results_trial{0}_reduced.pkl'.format(ind_loop, lr, std_noise, int(b_usefreq), int(b_costmae)))):
                    print('  File already processed')
                    continue

                # Create an early stopping callback appropriate for the dataset size
                cb_earlystopping = EarlyStopping(monitor='val_loss',
                                                 patience=max([20, min([compression_ratio*lr/0.00005, 250])]),
                                                 restore_best_weights=True)


                '''TODO : Write a method to stop early when reaching the minimum point'''

                # Create machine learning models for each evaluation step
                list_model_odq = []
                if DATASET is server_power:
                    generate_model = generate_model_server_power
                elif DATASET is home_energy:
                    generate_model = generate_model_home_energy
                elif DATASET is metasense:
                    generate_model = generate_model_metasense

                for _ in list_quantizers:
                    list_model_odq.append(generate_model(N_x, N_y, lr=lr, std_noise=std_noise, b_costmae=b_costmae))
                model_reservoir = generate_model_home_energy(N_x, N_y, lr=lr, std_noise=std_noise, b_costmae=b_costmae)

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
                            'N_datapoints': N_datapoints, 'N_trials': N_trials, 'list_lr': list_lr, 'list_std_noise': list_std_noise,
                            'TRAIN_VAL_RATIO': TRAIN_VAL_RATIO}

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
                    if b_usefreq:
                        sample_weight = w_temp
                    else:
                        sample_weight = None
                    history_temp = model_odq.fit(X_temp, Y_temp, batch_size=64, epochs=10, sample_weight=sample_weight, verbose=0,
                                                 validation_data=(X_val, Y_val))
                    history_odq = history_temp.history
                    history_odq['epoch'] = history_temp.epoch
                    history_temp = model_odq.fit(X_temp, Y_temp, batch_size=64, epochs=N_epochs, sample_weight=sample_weight, verbose=0,
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

                    dict_out['history_odq_w{0}'.format(w_type)] = history_odq
                    dict_out['score_odq_w{0}'.format(w_type)] = score_odq
                    dict_out['Y_odq_predict_w{0}'.format(w_type)] = Y_odq_predict

                    # Save all results for subsequent processing
                    dir_target_full = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw', dir_target)
                    if not os.path.exists(dir_target_full):
                        os.mkdir(dir_target_full)

                with open(os.path.join(dir_target_full,
                                       filename_base + 'lr{1}_std{2}_f{3}_c{4}_results_trial{0}_reduced.pkl'.format(ind_loop, lr, std_noise, int(b_usefreq), int(b_costmae))), 'wb') as fid:
                    pkl.dump(dict_out, fid)

                # Reset Tensorflow session to prevent memory growth
                K.clear_session()













def run_sq_nn_tests(filename, dir_quant, dir_target, N_trials=3, b_cpu=True, list_lr=[0.0001], list_std_noise=[0.001],
                 TRAIN_VAL_RATIO = 0.8, L = 2, RAM = 4000, b_usefreq=True, b_costmae=False):
    """
    Generate results for neural network processing of the quantized dataset
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
            list_quantizers = [{'quantizer':data_temp['quantizer'], 'w_x': [0], 'w_y': [0], 'w_imp': [0], 'w_type': 0}]
        reservoir_sampler = data_temp['reservoir_sampler']
        X_train = data_temp['X_train']
        Y_train = data_temp['Y_train']
        X_test = data_temp['X_test']
        Y_test = data_temp['Y_test']

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
            for std_noise in list_std_noise:
                config_tf_session(b_cpu)

                print('  lr: {0}   std_noise: {1}:'.format(lr, std_noise))

                if not(FLAG_OVERWRITE) and os.path.isfile(
                        os.path.join(os.path.dirname(__file__), '..', 'results', 'raw',
                                     dir_target,
                                     filename_base + 'lr{1}_std{2}_f{3}_c{4}_L{5}_RAM{6}_results_trial{0}_reduced.pkl'.format(ind_loop, lr, std_noise, int(b_usefreq), int(b_costmae),L, RAM))):
                    print('  File already processed')
                    continue

                # Create an early stopping callback appropriate for the dataset size
                cb_earlystopping = EarlyStopping(monitor='val_loss',
                                                 patience=max([20, min([compression_ratio*lr/0.00005, 250])]),
                                                 restore_best_weights=True)

                # Create machine learning models for each evaluation step
                list_model_odq = []



                if DATASET is server_power:
                    generate_model = generate_model_square
                elif DATASET is home_energy:
                    generate_model = generate_model_square
                elif DATASET is metasense:
                    generate_model = generate_model_square

                for _ in list_quantizers:
                    list_model_odq.append(generate_model(N_x, N_y, L, RAM, lr=lr, std_noise=std_noise, b_costmae=b_costmae))
                model_reservoir = generate_model(N_x, N_y, lr=lr, std_noise=std_noise, b_costmae=b_costmae)

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
                            'N_datapoints': N_datapoints, 'N_trials': N_trials, 'list_lr': list_lr, 'list_std_noise': list_std_noise,
                            'TRAIN_VAL_RATIO': TRAIN_VAL_RATIO, 'Number_of_Layers': L, 'Device_RAM_Size': RAM}

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
                    if b_usefreq:
                        sample_weight = w_temp
                    else:
                        sample_weight = None
                    history_temp = model_odq.fit(X_temp, Y_temp, batch_size=64, epochs=10, sample_weight=sample_weight, verbose=0,
                                                 validation_data=(X_val, Y_val))
                    history_odq = history_temp.history
                    history_odq['epoch'] = history_temp.epoch
                    history_temp = model_odq.fit(X_temp, Y_temp, batch_size=64, epochs=N_epochs, sample_weight=sample_weight, verbose=0,
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

                    dict_out['history_odq_w{0}'.format(w_type)] = history_odq
                    dict_out['score_odq_w{0}'.format(w_type)] = score_odq
                    dict_out['Y_odq_predict_w{0}'.format(w_type)] = Y_odq_predict

                    # Save all results for subsequent processing
                    dir_target_full = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw', dir_target)
                    if not os.path.exists(dir_target_full):
                        os.mkdir(dir_target_full)

                with open(os.path.join(dir_target_full,
                                       filename_base + 'lr{1}_std{2}_f{3}_c{4}_L{5}_RAM{6}_results_trial{0}_reduced.pkl'.format(ind_loop, lr, std_noise, int(b_usefreq), int(b_costmae),L, RAM)), 'wb') as fid:
                    pkl.dump(dict_out, fid)

                # Reset Tensorflow session to prevent memory growth
                K.clear_session()







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='Target directory of files.')
    parser.add_argument('--N', type=int, nargs=1, help='Number of trials to run.', default=[3])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--lr', type=float, nargs='+', help='ADAM learning rates to use.', default=[0.0001])
    parser.add_argument('--std', type=float, nargs='+', help='Noise std dev to use for NN training.', default=[0.001])
    parser.add_argument('--usefreq', action='store_true')
    parser.add_argument('--costmae', action='store_true')
    parser.add_argument('--ANNArch', type=str, nargs=1, help='Type of ANN (sq, default).', default=['default'])
    parser.add_argument('--L', type=int, nargs=1, help='Number of hidden ANN layesrs', default=[2])
    parser.add_argument('--RAM', type=int, nargs=1, help='Device RAM size', default=[4000])



    args = parser.parse_args()

    if args.dir is not None:
        dir_target = args.dir
    else:
        dir_target = 'metasense_test_cov_max2_201905031'

    N_trials = args.N[0]

    dir_quant = os.path.join(os.path.dirname(__file__), '..', 'results', 'quantized')

    print(args.ANNArch[0])

    if str(args.ANNArch[0]) == 'sq':

        L = args.L[0]
        RAM = args.RAM[0]
        print('Yes Square ! L = ',L, 'RAM = ', RAM)
        p_run_nn_tests = partial(run_sq_nn_tests, dir_quant=dir_quant, dir_target=dir_target, N_trials=N_trials,
                             b_cpu=args.cpu, b_usefreq=args.usefreq, b_costmae=args.costmae,
                             list_lr=args.lr, L=L, RAM=RAM, list_std_noise=args.std, TRAIN_VAL_RATIO=0.8, )
    else:
        print('Deafult !')
        p_run_nn_tests = partial(run_nn_tests, dir_quant=dir_quant, dir_target=dir_target, N_trials=N_trials,
                                 b_cpu=args.cpu, b_usefreq=args.usefreq, b_costmae=args.costmae,
                                 list_lr=args.lr, list_std_noise=args.std, TRAIN_VAL_RATIO=0.8, )


    with Pool(4) as p:
        p.map(p_run_nn_tests,
              [filename for filename in os.listdir(os.path.join(dir_quant, dir_target)) if
               (filename.lower().endswith('.pkl'))])


