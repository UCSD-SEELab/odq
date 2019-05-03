import numpy as np
import math
from keras import Model
from keras.layers import Input, Dense, Dropout, GaussianNoise
from keras.optimizers import Adam, SGD

def generate_model_server_power(N_x, N_y, std_noise=0.01, lr=0.001, decay=1e-6, b_costmae=False, optimizer='sgd'):
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
    if optimizer == 'sgd':
        optimizer = SGD(lr=lr, decay=decay)
    elif optimizer == 'adam':
        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    if b_costmae:
        model_nn.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mse', 'mae'])
    else:
        model_nn.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse', 'mae'])
    return model_nn

def generate_model_home_energy(N_x, N_y, std_noise=0.01, lr=0.001, decay=1e-6, b_costmae=False, optimizer='sgd'):
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
    if optimizer == 'sgd':
        optimizer = SGD(lr=lr, decay=decay)
    elif optimizer == 'adam':
        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    if b_costmae:
        model_nn.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mse', 'mae'])
    else:
        model_nn.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse', 'mae'])
    return model_nn

def generate_model_metasense(N_x, N_y, std_noise=0.01, lr=0.001, decay=1e-6, b_costmae=False, optimizer='sgd'):
    """
    Create neural network model
    no. of features = 6 (N_x = 6)
    ANN Maximum weights = 128000
    no. of outputs = 2 (N_y = 2)
    Defult = 41600
    """
    layer_input = Input(shape=(N_x,))  # Input features
    layer1 = GaussianNoise(stddev=std_noise)(layer_input)
    layer1 = Dense(200, activation='relu')(layer_input)
    layer1 = Dropout(0.5)(layer1)
    layer2 = Dense(200, activation='relu')(layer1)
    layer2 = Dropout(0.5)(layer2)
    layer_out = Dense(N_y)(layer2)
    model_nn = Model(inputs=layer_input, outputs=layer_out)
    if optimizer == 'sgd':
        optimizer = SGD(lr=lr, decay=decay)
    elif optimizer == 'adam':
        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    if b_costmae:
        model_nn.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mse', 'mae'])
    else:
        model_nn.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse', 'mae'])
    return model_nn


## Generate different ANN architectures
def generate_model_square(N_x, N_y, N_layer, N_weights, std_noise=0.01, lr=0.001, decay=1e-6, b_costmae=False,
                          optimizer='sgd'):
    """
    description

    N_layer: number of layers
    N_weights: is the device total weight

    """

    # Make the half of RAM size as max limit
    N_weights = N_weights / 2

    # N_weights = N_x*k + k*N_y + (N_layer-1)*K*K
    # (N_layer-1 * k**2) + ((N_x + N_y) * k) - N_weights = 0

    # with offset : N_weights = (N_x+1)*k + k*(N_y+1) + (N_layer-1)*(K+1)*(K)
    # (N_x+1)*k + k*(N_y+1) + (N_layer-1)(K**2+k) = 0

    a = N_layer - 1
    b = N_x + N_y + 2 + (N_layer - 1)
    c = - N_weights + (N_layer - 1)

    # print("\n{}x**2 + {}x + {} = 0 has ".format(a, b, c), end="")

    discriminant = b * b - 4 * a * c
    if discriminant > 0:
        rt = math.sqrt(discriminant)
        root1 = (-b + rt) / (2 * a)
        root2 = (-b - rt) / (2 * a)
        # print("two real solutions: {0:0.4f} and {1:0.4f}".format(root1, root2))
    elif discriminant == 0:
        root1 = -b / (2 * a)
        # print("one real solution: {0:0.4f}".format(root))
    else:
        root1 = -b / (2 * a)
        imag = abs(math.sqrt(-discriminant) / (2 * a))
        # print("two complex solutions: {0:0.4f} + {1:0.4f}i and {0:0.4f} - {1:0.4f}i".format(real, imag))

    k = int(root1)  # Number of neurons

    total_weights_test = N_x * k + k * N_y + (N_layer - 1) * k * k

    print("Device RAM size = ", N_weights)
    print("Total ANN weights using the positive value of root", total_weights_test)
    print("Number of hidden layers = ", N_layer)

    layers_list = []

    layer_input = Input(shape=(N_x,))  # Input features
    layers_list.append(layer_input)

    for i in range(1, N_layer + 1):
        if (i == 1):
            layer = GaussianNoise(stddev=std_noise)(layers_list[i - 1])
            layer = Dense(k, activation='relu')(layers_list[i - 1])
            layer = Dropout(0.5)(layer)
            layers_list.append(layer)

        else:
            layer = Dense(k, activation='relu')(layers_list[i - 1])
            layer = Dropout(0.5)(layer)
            layers_list.append(layer)

    layer_out = Dense(N_y)(layers_list[N_layer])
    model_nn = Model(inputs=layer_input, outputs=layer_out)

    print(model_nn.summary())

    if optimizer == 'sgd':
        optimizer = SGD(lr=lr, decay=decay)
    elif optimizer == 'adam':
        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    if b_costmae:
        model_nn.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mse', 'mae'])
    else:
        model_nn.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse', 'mae'])
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