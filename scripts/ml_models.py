import sys
import numpy as np

from keras import Model
from keras.layers import Input, Dense, Dropout, GaussianNoise
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.constraints import maxnorm
from functools import partial
import tensorflow as tf


def custom_loss_function_sig (y_true, y_pred, min_val = -2.197, max_val = 2.197 ) :
    """
    Custom loss function for Keras using a sigmoid function of 'm' and 'b' based on y_true
    """
    m = (-2.197 - 2.197) / (min_val - max_val)
    b = (max_val * m - 2.197) / m
    alpha = K.exp(m*y_true +b) / (1 + K.exp(m*y_true + b))
    return alpha*K.mean(K.square(y_pred - y_true))

def custom_loss_function_step (y_true, y_pred , b=0 ) :
    """
    Custom loss function for Keras using a step function at 'b' based on y_true
    """
    alpha = 0.5 * (tf.sign(y_true - b) + 1)
    return alpha*K.mean(K.square(y_pred - y_true))


def generate_model_server_power(N_x, N_y, model_cfg={'lr':0.01, 'dropout':0.5, 'decay':1e-4, 'optimizer':'sgd',
                                                  'loss':'mean_squared_error', 'b_custommodel':False,
                                                  'N_layer':2, 'N_weights':10000}):
    """
    Create neural network model for the server power dataset
    """
    if not('dropout' in model_cfg):
        model_cfg['dropout'] = 0.5

    if model_cfg['b_custommodel'] == False:
        layer_input = Input(shape=(N_x,))  # Input features
        layer1 = Dense(200, activation='relu')(layer_input)
        layer1 = Dropout(model_cfg['dropout'])(layer1)
        layer1 = Dense(200, activation='relu')(layer1)
        layer1 = Dropout(model_cfg['dropout'])(layer1)
        layer2 = Dense(200, activation='relu')(layer1)
        layer2 = Dropout(model_cfg['dropout'])(layer2)
        layer_out = Dense(N_y)(layer2)
        model_nn = Model(inputs=layer_input, outputs=layer_out)

    else:
        model_nn = generate_model_architecture_square(N_x=N_x, N_y=N_y, N_layer=model_cfg['N_layer'],
                                                      N_weights=model_cfg['N_weights'], dropout=model_cfg['dropout'])

    if model_cfg['optimizer'] == 'sgd':
        optimizer = SGD(lr=model_cfg['lr'], decay=model_cfg['decay'])
    elif model_cfg['optimizer'] == 'adam':
        optimizer = Adam(lr=model_cfg['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    if model_cfg['loss'] == 'sigmoid' :
        min_val = model_cfg['min_val']
        max_val = model_cfg['max_val']
        loss = partial(custom_loss_function_sig, min_val=min_val, max_val=max_val)
    elif model_cfg['loss'] == 'step':
        loss_step_b = 1
        loss = partial(custom_loss_function_step, b=loss_step_b)
    else:
        loss = model_cfg['loss']

    model_nn.compile(optimizer=optimizer, loss=loss, metrics=['mse', 'mae'])

    return model_nn


def generate_model_home_energy(N_x, N_y, model_cfg={'lr':0.01, 'dropout':0.5, 'decay':1e-4, 'optimizer':'sgd',
                                                  'loss':'mean_squared_error', 'b_custommodel':False,
                                                  'N_layer':2, 'N_weights':10000}):
    """
    Create neural network model for the home energy dataset
    """
    if not('dropout' in model_cfg):
        model_cfg['dropout'] = 0.5

    if model_cfg['b_custommodel'] == False:
        layer_input = Input(shape=(N_x,))  # Input features
        layer1 = Dense(512, activation='relu')(layer_input)
        layer1 = Dropout(0)(layer1)
        layer1 = Dense(128, activation='relu')(layer1)
        layer1 = Dropout(model_cfg['dropout'])(layer1)
        # layer1 = Dense(128, activation='relu')(layer1)
        # layer1 = Dropout(model_cfg['dropout'])(layer1)
        layer2 = Dense(128, activation='relu')(layer1)
        layer2 = Dropout(model_cfg['dropout'])(layer2)
        layer_out = Dense(N_y)(layer2)
        model_nn = Model(inputs=layer_input, outputs=layer_out)

    else:
        model_nn = generate_model_architecture_square(N_x=N_x, N_y=N_y, N_layer=model_cfg['N_layer'],
                                                      N_weights=model_cfg['N_weights'], dropout=model_cfg['dropout'])

    if model_cfg['optimizer'] == 'sgd':
        optimizer = SGD(lr=model_cfg['lr'], decay=model_cfg['decay'])
    elif model_cfg['optimizer'] == 'adam':
        optimizer = Adam(lr=model_cfg['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    if model_cfg['loss'] == 'sigmoid' :
        min_val = model_cfg['min_val']
        max_val = model_cfg['max_val']
        loss = partial(custom_loss_function_sig, min_val=min_val, max_val=max_val)
    elif model_cfg['loss'] == 'step':
        loss_step_b = 1
        loss = partial(custom_loss_function_step, b=loss_step_b)
    else:
        loss = model_cfg['loss']

    model_nn.compile(optimizer=optimizer, loss=loss, metrics=['mse', 'mae'])

    return model_nn


def generate_model_metasense(N_x, N_y, model_cfg={'lr':0.01, 'dropout':0.5, 'decay':1e-4, 'optimizer':'sgd',
                                                  'loss':'mean_squared_error', 'b_custommodel':False,
                                                  'N_layer':2, 'N_weights':10000}):
    """
    Create neural network model.

    If b_custommodel is set to True, then a square model that adheres to size ('N_weights') and depth ('N_layer')
    requirements will be generated.
    """
    if not('dropout' in model_cfg):
        model_cfg['dropout'] = 0.5

    if model_cfg['b_custommodel'] == False:
        layer_input = Input(shape=(N_x,))  # Input features
        layer1 = Dense(100, activation='relu')(layer_input)
        layer1 = Dropout(model_cfg['dropout'])(layer1)
        layer2 = Dense(100, activation='relu')(layer1)
        layer2 = Dropout(model_cfg['dropout'])(layer2)
        layer_out = Dense(N_y)(layer2)
        model_nn = Model(inputs=layer_input, outputs=layer_out)

    else:
        model_nn = generate_model_architecture_square(N_x=N_x, N_y=N_y, N_layer=model_cfg['N_layer'],
                                                      N_weights=model_cfg['N_weights'], dropout=model_cfg['dropout'])

    if model_cfg['optimizer'] == 'sgd':
        optimizer = SGD(lr=model_cfg['lr'], decay=model_cfg['decay'])
    elif model_cfg['optimizer'] == 'adam':
        optimizer = Adam(lr=model_cfg['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    if model_cfg['loss'] == 'sigmoid' :
        min_val = model_cfg['min_val']
        max_val = model_cfg['max_val']
        loss = partial(custom_loss_function_sig, min_val=min_val, max_val=max_val)
    elif model_cfg['loss'] == 'step':
        loss_step_b = 1
        loss = partial(custom_loss_function_step, b=loss_step_b)
    else:
        loss = model_cfg['loss']

    model_nn.compile(optimizer=optimizer, loss=loss, metrics=['mse', 'mae'])

    return model_nn

def generate_model_square(N_x, N_y, N_layer, N_weights, lr=0.01, dropout=0.5, decay=1e-4, optimizer='sgd', loss='mean_squared_error'):
    """
    Wrapper to generate a fully connected neural network architecture
    """
    model_nn = generate_model_architecture_square(N_x=N_x, N_y=N_y, N_layer=N_layer, N_weights=N_weights, dropout=dropout)

    if optimizer == 'sgd':
        optimizer = SGD(lr=lr, decay=decay)
    elif optimizer == 'adam':
        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    if model_cfg['loss'] == 'sigmoid' :
        min_val = model_cfg['min_val']
        max_val = model_cfg['max_val']
        loss = partial(custom_loss_function_sig, min_val=min_val, max_val=max_val)
    elif loss == 'step':
        loss_step_b = 1
        loss = partial(custom_loss_function_step, b=loss_step_b)

    model_nn.compile(optimizer=optimizer, loss=loss, metrics=['mse', 'mae'])

    return model_nn

def generate_model_architecture_square(N_x, N_y, N_layer, N_weights, dropout):
    """
    Generate a fully connected neural network architecture that has 'N_layer' hidden layers with a maximum number of
    'N_weights' parameters

    Since the model is rectangular, we can calculate the number of neurons per layer using the following equations:

    N_weights = (N_x+1)*N_width + (N_layer)*(N_width+1)*(N_width) + (N_width+1)*N_y
                 input              hidden layers                      output

    0 = (N_layer)*N_width**2 + (N_x + N_y + N_layer + 1)*N_width + (N_y + N_weights)

    Solve using quadratic equation.
    """
    a = N_layer
    b = N_x + N_y + N_layer + 1
    c = N_y - N_weights

    N_width = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)

    N_layer = int(N_layer)
    N_width = int(N_width)

    total_weights_test = (N_x+1)*N_width + (N_layer)*(N_width+1)*(N_width) + (N_width+1)*N_y

    # print("Device RAM size = ", N_weights)
    # print("Total ANN weights using the positive value of root", total_weights_test)
    # print("Number of hidden layers = ", N_layer)

    layer_input = Input(shape=(N_x,))  # Input features

    for i in range(0, N_layer):
        if (i == 0):
            layer = Dense(N_width, activation='relu', kernel_constraint=maxnorm(3))(layer_input)
            layer = Dropout(dropout)(layer)
        else:
            layer = Dense(N_width, activation='relu', kernel_constraint=maxnorm(3))(layer)
            layer = Dropout(dropout)(layer)

    layer_out = Dense(N_y)(layer)
    model_nn = Model(inputs=layer_input, outputs=layer_out)

    # print(model_nn.summary())

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
    test_y = np.arange(100).reshape((-1, 1))
    test_x = np.concatenate((test_y, test_y), axis=1)

    x_train, x_test, y_train, y_test = train_test_split_blocks(test_x, test_y, pct_train=0.8, n_blocks=2)

    print('Done')
