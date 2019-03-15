import sys
import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dropout
from keras import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import keras.backend as K
from functools import partial
from sklearn import preprocessing
import pickle as pkl

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from odq.odq import OnlineDatasetQuantizer, calc_weights_max_cov
from odq.reservoir import ReservoirSampler
from odq.data import home_energy, server_power


def generate_model_server_power():
    """
    Create neural network model
    """
    layer_input = Input(shape=(N_x, )) # Input features
    layer1 = Dense(200, activation='relu')(layer_input)
    layer1 = Dropout(0.5)(layer1)
    layer1 = Dense(200, activation='relu')(layer1)
    layer1 = Dropout(0.5)(layer1)
    layer2 = Dense(200, activation='relu')(layer1)
    layer2 = Dropout(0.5)(layer2)
    layer_out = Dense(N_y)(layer2)
    model_nn = Model(inputs=layer_input, outputs=layer_out)
    optimizer_adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model_nn.compile(optimizer=optimizer_adam, loss='mean_squared_error', metrics=['mae'])
    return model_nn

def generate_model_home_energy():
    """
    Create neural network model
    """
    layer_input = Input(shape=(N_x,))  # Input features
    layer1 = Dense(512, activation='relu')(layer_input)
    layer1 = Dropout(0)(layer1)
    layer1 = Dense(128, activation='relu')(layer1)
    layer1 = Dropout(0.5)(layer1)
    # layer1 = Dense(128, activation='relu')(layer1)
    # layer1 = Dropout(0.5)(layer1)
    layer2 = Dense(128, activation='relu')(layer1)
    layer2 = Dropout(0.5)(layer2)
    layer_out = Dense(N_y)(layer2)
    model_nn = Model(inputs=layer_input, outputs=layer_out)
    optimizer_adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model_nn.compile(optimizer=optimizer_adam, loss='mean_squared_error', metrics=['mse'])
    return model_nn

def train_test_split(X, Y, pct_train=0.8):
    """
    Splits the datasets X and Y into training and test sets based on input percentage (pct_train)
    """
    N = X.shape[0]
    ind_split = np.round(N*pct_train).astype(int)
    ind_random = np.random.permutation(N)

    return X[ind_random[:ind_split], :], X[ind_random[ind_split:], :], \
           Y[ind_random[:ind_split], :], Y[ind_random[ind_split:], :]


if __name__ == '__main__':
    FLAG_VERBOSE = False
    FLAG_PLOT = False
    FLAG_SAVE_MODEL = False
    PLOT_DELAY = 0.0001
    DATASET = home_energy
    ind_assess = [-1]
    list_compression_ratio = np.append([1.3], 2**(1 + np.arange(7)))#2**(1 + np.arange(9))[::-1]
    N_trials = 3
    TRAIN_VAL_RATIO = 0.8
    TRAIN_TEST_RATIO = 0.8 # Server power has test/train datasets pre-split due to tasks

    filename_base = datetime.now().strftime('{0}_Data_Size_%Y%m%d%H%M%S'.format(DATASET.__name__.split('.')[-1]))

    # np.random.seed(1237)

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
        # Load dataset
        X_train, Y_train, X_test, Y_test = DATASET.load()

        N_x = X_train.shape[1]
        N_y = Y_train.shape[1]

        # Calculate weights for each column
        w_max_cov = calc_weights_max_cov(X_train, Y_train)
        w_x = w_max_cov[:N_x]
        w_y = w_max_cov[N_x:]

    elif DATASET is home_energy:
        # Load dataset
        X, Y = DATASET.load()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, pct_train=TRAIN_TEST_RATIO)

        N_x = X_train.shape[1]
        N_y = Y_train.shape[1]

        # Calculate weights for each column
        w_max_cov = calc_weights_max_cov(X, Y)
        w_x = w_max_cov[:N_x]
        w_y = w_max_cov[N_x:]

    else:
        print('Invalid dataset {0}'.format(DATASET))
        sys.exit()

    min_max_scaler_x = preprocessing.MinMaxScaler()
    min_max_scaler_y = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler_x.fit_transform(X_train)
    Y_scaled = min_max_scaler_y.fit_transform(Y_train)

    # Construct fixed memory OnlineDatasetQuantizer
    N_datapoints = X_train.shape[0]
    N_x = X_train.shape[1]
    N_y = Y_train.shape[1]

    for ind_loop in range(N_trials):
        for compression_ratio in list_compression_ratio:
            N_saved = int(N_datapoints // compression_ratio)
            if DATASET is home_energy:
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, pct_train=TRAIN_TEST_RATIO)
                X_scaled = min_max_scaler_x.fit_transform(X_train)
                Y_scaled = min_max_scaler_y.fit_transform(Y_train)
                N_epochs = int(max(500, 50*compression_ratio))
            else:
                N_epochs = int(max(100, 30*compression_ratio))

            print('\n\nCompression Ratio {0}: {1} -> {2}'.format(compression_ratio, N_datapoints, N_saved))
            quantizer = OnlineDatasetQuantizer(num_datapoints_max=N_saved, num_input_features=N_x, num_target_features=N_y,
                                               w_x_columns=w_x, w_y_columns=w_y)
            reservoir_sampler = ReservoirSampler(num_datapoints_max=N_saved, num_input_features=N_x, num_target_features=N_y)

            # Create an early stopping callback appropriate for the dataset size
            cb_earlystopping = EarlyStopping(monitor='val_loss', patience=max([40, min([10*compression_ratio, 250])]), restore_best_weights=True)

            # Create machine learning models for each evaluation step
            if DATASET is server_power:
                model_odq = generate_model_server_power()
                model_reservoir = generate_model_server_power()
                model_full = generate_model_server_power()
            elif DATASET is home_energy:
                model_odq = generate_model_home_energy()
                model_reservoir = generate_model_home_energy()
                model_full = generate_model_home_energy()

            for ind, X_new, Y_new in zip(range(N_datapoints), X_scaled, Y_scaled):
                quantizer.add_point(X_new, Y_new)
                reservoir_sampler.add_point(X_new, Y_new)
                if (ind in ind_assess):
                    print('\nind {0}: Generate model from ODQ-reduced Data'.format(ind))
                    X_temp, Y_temp = quantizer.get_dataset()
                    w_temp = quantizer.get_sample_weights()
                    history_odq = model_odq.fit(X_temp, Y_temp, batch_size=32, epochs=N_epochs, sample_weight=w_temp, verbose=0,
                                                validation_split=(1 - TRAIN_VAL_RATIO), callbacks=[cb_earlystopping])
                    score_odq = model_odq.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test), verbose=0)

                    Y_odq_predict = min_max_scaler_y.inverse_transform(model_odq.predict(min_max_scaler_x.transform(X_test)))

                    print('    RMSE: {0}'.format(np.sqrt(np.mean((Y_odq_predict - Y_test)**2))))

                    print('\nind {0}: Generate model from Reservoir-reduced Data'.format(ind))
                    X_temp, Y_temp = reservoir_sampler.get_dataset()
                    history_reservoir = model_reservoir.fit(X_temp, Y_temp, batch_size=32, epochs=N_epochs, verbose=0,
                                                validation_split=(1 - TRAIN_VAL_RATIO), callbacks=[cb_earlystopping])
                    score_reservoir = model_reservoir.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test), verbose=0)

                    Y_reservoir_predict = min_max_scaler_y.inverse_transform(model_reservoir.predict(min_max_scaler_x.transform(X_test)))

                    print('    RMSE: {0}'.format(np.sqrt(np.mean((Y_reservoir_predict - Y_test)**2))))

                    # Save model progress
                    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'raw', filename_base + '_{1}_models{0:06d}_{2}_partial.pkl'.format(ind, N_saved, ind_loop)), 'wb') as fid:
                        pkl.dump({'model_odq': model_odq, 'model_reservoir': model_reservoir, 'history_odq': history_odq,
                                  'score_odq': score_odq, 'Y_odq_predict': Y_odq_predict, 'score_reservoir': score_reservoir,
                                  'Y_reservoir_predict': Y_reservoir_predict}, fid)

                    if FLAG_PLOT:
                        quantizer.plot('ODQ Sample {0:5d}'.format(ind), ind_dims=[0,1], fig_num=1)
                        quantizer.plot('ODQ Sample {0:5d}'.format(ind), ind_dims=[0,-3], fig_num=2)
                        quantizer.plot('ODQ Sample {0:5d}'.format(ind), ind_dims=[1,-3], fig_num=3)


            # Plot full dataset for comparison
            if FLAG_PLOT:
                plt.figure(4)
                plt.title('Full Dataset (X0 and X1)')
                plt.scatter(X[:, 0], X[:, 1], s=1)
                plt.grid(True)

                plt.figure(5)
                plt.title('Full Dataset (X0 and Y0)')
                plt.scatter(X[:, 0], Y[:, 0], s=1)
                plt.grid(True)

                plt.figure(6)
                plt.title('Full Dataset (X1 and Y0)')
                plt.scatter(X[:, 1], Y[:, 0], s=1)
                plt.grid(True)
                plt.draw()
                plt.pause(1)

            time_start = time.time()
            print('\nGenerate final model from ODQ-reduced Data')
            X_temp, Y_temp = quantizer.get_dataset()
            w_temp = quantizer.get_sample_weights()
            history_odq = model_odq.fit(X_temp, Y_temp, batch_size=128, epochs=N_epochs, sample_weight=w_temp, verbose=0,
                                        validation_split=(1 - TRAIN_VAL_RATIO), callbacks=[cb_earlystopping])
            score_odq = model_odq.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test), verbose=0)

            Y_odq_predict = min_max_scaler_y.inverse_transform(model_odq.predict(min_max_scaler_x.transform(X_test)))

            print('    RMSE: {0:0.2f}'.format(np.sqrt(np.mean((Y_odq_predict - Y_test)**2))))
            print('    Time: {0:0.2f} s'.format(time.time() - time_start))

            time_start = time.time()
            print('\nGenerate final model from Reservoir-reduced Data')
            X_temp, Y_temp = reservoir_sampler.get_dataset()
            history_reservoir = model_reservoir.fit(X_temp, Y_temp, batch_size=128, epochs=N_epochs, verbose=0,
                                                    validation_split=(1 - TRAIN_VAL_RATIO), callbacks=[cb_earlystopping])
            score_reservoir = model_reservoir.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test), verbose=0)

            Y_reservoir_predict = min_max_scaler_y.inverse_transform(model_reservoir.predict(min_max_scaler_x.transform(X_test)))

            print('    RMSE: {0:0.2f}'.format(np.sqrt(np.mean((Y_reservoir_predict - Y_test)**2))))
            print('    Time: {0:0.2f} s'.format(time.time() - time_start))
            if FLAG_SAVE_MODEL:
                with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'raw', filename_base + '_{0}_models_{1}_final.pkl'.format(N_saved, ind_loop)), 'wb') as fid:
                    pkl.dump({'quantizer': quantizer, 'reservoir_sampler': reservoir_sampler,
                              'model_odq': model_odq, 'model_reservoir': model_reservoir, 'history_odq': history_odq,
                              'score_odq': score_odq, 'Y_odq_predict': Y_odq_predict, 'history_reservoir': history_reservoir,
                              'score_reservoir': score_reservoir, 'Y_reservoir_predict': Y_reservoir_predict,
                              'Y_test': Y_test, 'X_test': X_test}, fid)
            else:
                with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'raw', filename_base + '_{0}_results_{1}_final.pkl'.format(N_saved, ind_loop)), 'wb') as fid:
                    pkl.dump({'quantizer': quantizer, 'reservoir_sampler': reservoir_sampler,
                              'history_odq': history_odq, 'score_odq': score_odq, 'Y_odq_predict': Y_odq_predict,
                              'history_reservoir': history_reservoir, 'score_reservoir': score_reservoir, 'Y_reservoir_predict': Y_reservoir_predict,
                              'Y_test': Y_test, 'X_test': X_test}, fid)

        # Save full model
        time_start = time.time()
        print('\nGenerate final model from full dataset')
        cb_earlystopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history_full = model_full.fit(X_scaled, Y_scaled, batch_size=128, epochs=400, verbose=0,
                                      validation_split=(1 - TRAIN_VAL_RATIO), callbacks=[cb_earlystopping])
        score_full = model_full.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test), verbose=0)

        Y_full_predict = min_max_scaler_y.inverse_transform(model_full.predict(min_max_scaler_x.transform(X_test)))

        print('    RMSE: {0:0.2f}'.format(np.sqrt(np.mean((Y_full_predict - Y_test) ** 2))))
        print('    Time: {0:0.2f} s'.format(time.time() - time_start))

        if FLAG_SAVE_MODEL:
            with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'raw',
                                   filename_base + '_{0}_model_{1}_full.pkl'.format(N_saved, ind_loop)), 'wb') as fid:
                pkl.dump({'model_full': model_full, 'history_full':history_full, 'score_full': score_full,
                          'Y_full_predict': Y_full_predict, 'Y_test': Y_test, 'X_test': X_test}, fid)
        else:
            with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'raw',
                                   filename_base + '_{0}_results_{1}_full.pkl'.format(N_saved, ind_loop)), 'wb') as fid:
                pkl.dump({'history_full':history_full, 'score_full': score_full,
                          'Y_full_predict': Y_full_predict, 'Y_test': Y_test, 'X_test': X_test}, fid)
