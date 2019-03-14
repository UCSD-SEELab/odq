import sys
import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dropout
from keras import Model
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

def generate_model_home_energy():
    """
    Create neural network model
    """
    layer_input = Input(shape=(N_x, )) # Input features
    layer1 = Dense(100, activation='relu')(layer_input)
    layer1 = Dropout(0)(layer1)
    layer1 = Dense(100, activation='relu')(layer1)
    layer1 = Dropout(0.2)(layer1)
    layer1 = Dense(100, activation='relu')(layer1)
    layer1 = Dropout(0.4)(layer1)
    layer2 = Dense(100, activation='relu')(layer1)
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
    PLOT_DELAY = 0.0001
    DATASET = home_energy
    N_saved = 2000
    ind_assess = [-1] #[2000, 4000, 8000, 12000]
    TRAIN_VAL_RATIO = 0.8
    TRAIN_TEST_RATIO = 0.9 # Server power has test/train datasets pre-split due to tasks

    filename_base = datetime.now().strftime('Inc_Add_%Y%m%d%H%M%S')

    np.random.seed(1234)

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

    quantizer = OnlineDatasetQuantizer(num_datapoints_max=N_saved, num_input_features=N_x, num_target_features=N_y,
                                       w_x_columns=w_x, w_y_columns=w_y)
    reservoir_sampler = ReservoirSampler(num_datapoints_max=N_saved, num_input_features=N_x, num_target_features=N_y)

    # Create machine learning models for each evaluation step
    if DATASET is server_power:
        model_odq = generate_model_server_power()
        model_reservoir = generate_model_server_power()
        model_full = generate_model_server_power()
    elif DATASET is home_energy:
        model_odq = generate_model_home_energy()
        model_reservoir = generate_model_server_power()
        model_full = generate_model_server_power()

    for ind, X_new, Y_new in zip(range(N_datapoints), X_scaled, Y_scaled):
        quantizer.add_point(X_new, Y_new)
        reservoir_sampler.add_point(X_new, Y_new)
        if (ind in ind_assess):
            print('\nind {0}: Generate model from ODQ-reduced Data'.format(ind))
            X_temp, Y_temp = quantizer.get_dataset()
            w_temp = quantizer.get_sample_weights()
            history_odq = model_odq.fit(X_temp, Y_temp, batch_size=32, epochs=500, sample_weight=w_temp, verbose=0,
                                        validation_split=(1 - TRAIN_VAL_RATIO))
            score_odq = model_odq.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test))

            Y_odq_predict = min_max_scaler_y.inverse_transform(model_odq.predict(min_max_scaler_x.transform(X_test)))

            print('    RMSE: {0}'.format(np.sqrt(np.sum((Y_odq_predict - Y_test)**2))))

            print('\nind {0}: Generate model from Reservoir-reduced Data'.format(ind))
            X_temp, Y_temp = reservoir_sampler.get_dataset()
            history_reservoir = model_reservoir.fit(X_temp, Y_temp, batch_size=32, epochs=500, verbose=0,
                                        validation_split=(1 - TRAIN_VAL_RATIO))
            score_reservoir = model_reservoir.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test))

            Y_reservoir_predict = min_max_scaler_y.inverse_transform(model_reservoir.predict(min_max_scaler_x.transform(X_test)))

            print('    RMSE: {0}'.format(np.sqrt(np.sum((Y_reservoir_predict - Y_test)**2))))

            # Save model progress
            with open(os.path.join(os.path.dirname(__file__), '..', 'results', filename_base + '_models{0:06d}.pkl'.format(ind)), 'wb') as fid:
                pkl.dump({'model_odq': model_odq, 'model_reservoir': model_reservoir, 'history_odq': history_odq,
                          'score_odq': score_odq, 'Y_odq_predict': Y_odq_predict, 'history_reservoir': history_reservoir,
                          'score_reservoir': score_reservoir, 'Y_reservoir_predict': Y_reservoir_predict,
                          'Y_test': Y_test, 'X_test': X_test}, fid)

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

    print('\nGenerate final model from ODQ-reduced Data')
    X_temp, Y_temp = quantizer.get_dataset()
    w_temp = quantizer.get_sample_weights()
    history_odq = model_odq.fit(X_temp, Y_temp, batch_size=32, epochs=1000, sample_weight=w_temp, verbose=1,
                                validation_split=0.1)
    score_odq = model_odq.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test))

    Y_odq_predict = min_max_scaler_y.inverse_transform(model_odq.predict(min_max_scaler_x.transform(X_test)))

    print('    RMSE: {0}'.format(np.sqrt(np.sum((Y_odq_predict - Y_test)**2))))

    print('\nind {0}: Generate final model from Reservoir-reduced Data')
    X_temp, Y_temp = reservoir_sampler.get_dataset()
    history_reservoir = model_reservoir.fit(X_temp, Y_temp, batch_size=32, epochs=1000, verbose=1,
                                            validation_split=0.1)
    score_reservoir = model_reservoir.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test))

    Y_reservoir_predict = min_max_scaler_y.inverse_transform(model_reservoir.predict(min_max_scaler_x.transform(X_test)))

    print('    RMSE: {0}'.format(np.sqrt(np.sum((Y_reservoir_predict - Y_test)**2))))

    print('\nGenerate final model from full dataset')
    history_full = model_full.fit(X_scaled, Y_scaled, batch_size=32, epochs=500, verbose=1,
                                  validation_split=0.1)
    score_full = model_full.evaluate(min_max_scaler_x.transform(X_test), min_max_scaler_y.transform(Y_test))

    Y_full_predict = min_max_scaler_y.inverse_transform(model_full.predict(min_max_scaler_x.transform(X_test)))

    print('    RMSE: {0}'.format(np.sqrt(np.sum((Y_full_predict - Y_test)**2))))

    with open(os.path.join(os.path.dirname(__file__), '..', 'results', filename_base + '_models_final.pkl'.format(ind)), 'wb') as fid:
        pkl.dump({'model_odq': model_odq, 'model_reservoir': model_reservoir, 'history_odq': history_odq,
                  'score_odq': score_odq, 'Y_odq_predict': Y_odq_predict, 'history_reservoir': history_reservoir,
                  'score_reservoir': score_reservoir, 'Y_reservoir_predict': Y_reservoir_predict,
                  'model_full': model_full, 'history_full': history_full, 'score_full': score_full,
                  'Y_full_predict': Y_full_predict, 'Y_test': Y_test, 'X_test': X_test
                  }, fid)
