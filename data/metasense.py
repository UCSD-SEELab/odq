import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load(board_num=11):
    """
    Load the local MetaSense dataset
    """

    """
    List of possible features:
    'SensorMAC', 'Ts', 'TempCmV', 'OxWmV', 'OxAmV', 'No2WmV', 'No2AmV', 'CoWmV', 'CoAmV', 'NCmV',
    'Pres_mB', 'Pres_cT', 'Hum_pc', 'Hum_cT', 'Time', 'Co2', 'VPp', 'VIp', 'Latitude', 'Longitude',
    'Altitude', 'Accuracy', 'Bearing', 'Speed', 'no2', 'o3', 'co', 'temperature', 'humidity',
    'absolute-humidity', 'board', 'location', 'round', 'epa-o3', 'epa-no2'
    """
    list_features = ['no2', 'o3', 'co', 'temperature', 'humidity', 'absolute-humidity']
    list_targets =  ['epa-o3']#, 'epa-no2']

    cols_target = list_features + list_targets

    directory_data = os.path.join(os.path.dirname(__file__), 'metasense')

    list_data = []
    for file in os.listdir(directory_data):
        if not (file.lower().endswith('.csv')):
            continue

        try:
            list_filepieces = file.replace('.csv', '').split(sep='_')

            if not(int(list_filepieces[-1]) == board_num):
                continue
        except:
            continue

        print('Loading data from {0}'.format(file))

        try:
            with open(os.path.join(directory_data, file), 'rb') as fid:
                data = pd.read_csv(fid, header=0, sep=',', index_col='Ts')
            print('  SUCCESS')
        except:
            print('  ERROR. Skipping.')
            continue

        data.index = pd.to_datetime(data.index, unit='s')
        data_minute = data[cols_target].resample('T').median()

        # Remove 1 hour worth of data after each nan to account for sensor warmup
        ind_nan = np.any(data_minute.isna(), axis=1)
        ind_nan[0] = True
        ind_remove = np.zeros(ind_nan.shape).astype(bool)
        for ind, b_nan in enumerate(ind_nan):
            if b_nan:
                ind_end = min([ind + 60, ind_nan.shape[0]])
                ind_remove[ind:ind_end] = True

        data_minute = data_minute[np.invert(ind_remove)]

        # Remove last 1 hour of data
        t_end = data_minute.index[-1]
        data_minute = data_minute[data_minute.index < (t_end - pd.Timedelta(minutes=60))]

        # data['o3'].plot()
        # data_minute['o3'].plot()
        # plt.show()
        list_data.append(data_minute)

    data_full = pd.concat(list_data)
    # data_full['o3'].plot()
    # plt.show()
    return data_full[list_features].to_numpy(), data_full[list_targets].to_numpy()

if __name__ == '__main__':
    data = load()
    print(data)
