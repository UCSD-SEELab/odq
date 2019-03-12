import os
import pandas as pd
import numpy as np


def number_seconds_since_midnight(data_ts):
    """
    Converts a datetime64 object into number of seconds since midnight
    """
    return 3600*data_ts.dt.hour + 60*data_ts.dt.minute + data_ts.dt.second

def load():
    """
    Load the local home energy dataset from https://github.com/LuisM78/Appliances-energy-prediction-data
    and https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
    """
    list_features = ['nsm',
                     'lights',
                     'T1', 'RH_1',
                     'T2', 'RH_2',
                     'T3', 'RH_3',
                     'T4', 'RH_4',
                     'T5', 'RH_5',
                     'T6', 'RH_6',
                     'T7', 'RH_7',
                     'T8', 'RH_8',
                     'T9', 'RH_9',
                     'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint',
                     ]
    list_targets = ['Appliances']

    directory_data = os.path.join(os.path.dirname(__file__), 'energydata_complete.csv')
    data = pd.read_csv(directory_data, quotechar='"', header=0, sep=',')

    # Add nsm (number of seconds until midnight)
    data['nsm'] = number_seconds_since_midnight(pd.to_datetime(data['date']))

    # Ensure only valid datapoints
    data.replace([np.inf, -np.inf], np.nan)
    data.dropna(axis=0)

    return data[list_features].to_numpy(), data[list_targets].to_numpy()

if __name__ == '__main__':
    data = load()
    print(data)
