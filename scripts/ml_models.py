import numpy as np

from keras import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam

def generate_model_server_power(N_x, N_y):
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
    model_nn.compile(optimizer=optimizer_adam, loss='mean_squared_error', metrics=['mse', 'mae'])
    return model_nn

def generate_model_home_energy(N_x, N_y):
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
    model_nn.compile(optimizer=optimizer_adam, loss='mean_squared_error', metrics=['mse', 'mae'])
    return model_nn

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