import os
import pandas as pd
import numpy as np


def load():
    """
    Load the local server power dataset from Yeseong Kim
    """
    list_features = ['master.cpu-cycles.raw', 'master.instructions.raw', 'master.branch-instructions.raw',
                     'master.branch-misses.raw', 'master.bus-cycles.raw', 'master.cache-references.raw',
                     'master.cache-misses.raw', 'master.L1-dcache-loads.raw', 'master.L1-dcache-stores.raw',
                     'master.r1A2.raw', 'master.r2B1.raw', 'master.r10B1.raw',
                     'slave1.cpu-cycles.raw', 'slave1.instructions.raw', 'slave1.branch-instructions.raw',
                     'slave1.branch-misses.raw', 'slave1.bus-cycles.raw', 'slave1.cache-references.raw',
                     'slave1.cache-misses.raw', 'slave1.L1-dcache-loads.raw', 'slave1.L1-dcache-stores.raw',
                     'slave1.r0A0.raw', 'slave1.r1FDC.raw', 'slave1.r010.raw',
                     'slave2.cpu-cycles.raw',  'slave2.instructions.raw', 'slave2.branch-instructions.raw',
                     'slave2.branch-misses.raw', 'slave2.cache-references.raw',
                     'slave2.cache-misses.raw', 'slave2.L1-dcache-loads.raw', 'slave2.L1-dcache-stores.raw',
                     'slave2.r1C2.raw', 'slave2.r110.raw', 'slave2.r1A2.raw',
                     ] # removed 'slave2.bus-cycles.raw' due to column being all 0
    list_targets = ['total_power']

    directory_train = os.path.join(os.path.dirname(__file__), 'power_toy_train.csv')
    directory_test = os.path.join(os.path.dirname(__file__), 'power_toy_test.csv')
    data_train = pd.read_csv(directory_train, header=0, sep=',')
    data_test = pd.read_csv(directory_test, header=0, sep=',')

    # Ensure only valid datapoints
    data_train.replace([np.inf, -np.inf], np.nan)
    data_train.dropna(axis=0)

    data_test.replace([np.inf, -np.inf], np.nan)
    data_test.dropna(axis=0)

    return data_train[list_features].to_numpy(), data_train[list_targets].to_numpy(), \
           data_test[list_features].to_numpy(), data_test[list_targets].to_numpy()

if __name__ == '__main__':
    data = load()
    print(data)
