import numpy as np

from keras import Model
from keras.layers import Input, Dense, Dropout, GaussianNoise
from keras.optimizers import Adam, SGD

def generate_model_server_power(N_x, N_y, std_noise=0.01, lr=0.0001, b_costmae=False):
    """
    Create neural network model for the server power dataset
    """
    layer_input = Input(shape=(N_x, )) # Input features
    layer1 = GaussianNoise(stddev=std_noise)(layer_input)
    layer1 = Dense(200, activation='relu')(layer1)
    layer1 = Dropout(0.5)(layer1)
    layer1 = Dense(200, activation='relu')(layer1)
    layer1 = Dropout(0.5)(layer1)
    layer2 = Dense(200, activation='relu')(layer1)
    layer2 = Dropout(0.5)(layer2)
    layer_out = Dense(N_y)(layer2)
    model_nn = Model(inputs=layer_input, outputs=layer_out)
    optimizer_adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    if b_costmae:
        model_nn.compile(optimizer=optimizer_adam, loss='mean_absolute_error', metrics=['mse', 'mae'])
    else:
        model_nn.compile(optimizer=optimizer_adam, loss='mean_squared_error', metrics=['mse', 'mae'])
    return model_nn

def generate_model_home_energy(N_x, N_y, std_noise=0.01, lr=0.0001, b_costmae=False):
    """
    Create neural network model for the home energy dataset
    """
    layer_input = Input(shape=(N_x,))  # Input features
    layer1 = GaussianNoise(stddev=std_noise)(layer_input)
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
    optimizer_adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    if b_costmae:
        model_nn.compile(optimizer=optimizer_adam, loss='mean_absolute_error', metrics=['mse', 'mae'])
    else:
        model_nn.compile(optimizer=optimizer_adam, loss='mean_squared_error', metrics=['mse', 'mae'])
    return model_nn

def generate_model_metasense(N_x, N_y, std_noise=0.01, lr=0.0001, b_costmae=False):
    """
    Create neural network model
    """
    layer_input = Input(shape=(N_x,))  # Input features
    layer1 = GaussianNoise(stddev=std_noise)(layer_input)
    layer1 = Dense(200, activation='relu')(layer_input)
    layer1 = Dropout(0.5)(layer1)
    layer2 = Dense(200, activation='relu')(layer1)
    layer2 = Dropout(0.5)(layer2)
    layer_out = Dense(N_y)(layer2)
    model_nn = Model(inputs=layer_input, outputs=layer_out)
    optimizer_adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    if b_costmae:
        model_nn.compile(optimizer=optimizer_adam, loss='mean_absolute_error', metrics=['mse', 'mae'])
    else:
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