import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from itertools import compress


directory = os.path.join(os.path.dirname(__file__), 'raw')

list_filename_base = ['home_energy_data_size_20190327182157']

field_new = 'N_datapoints'
data_new = 15788
b_overwrite = False

if __name__ == '__main__':
    if b_overwrite:
        print('Updating pkl files. Adding \'{0}\' = {1}. Replacing if present.'.format(field_new, data_new))
    else:
        print('Updating pkl files. Adding \'{0}\' = {1} if not present'.format(field_new, data_new))

    for file in os.listdir(directory):
        if not(file.lower().endswith('.pkl')):
            continue

        filename_match = [file.startswith(filename_base) for filename_base in list_filename_base]

        if not(any(filename_match)):
            continue

        print('Loading data from {0}'.format(file))

        try:
            with open(os.path.join(directory, file), 'rb') as fid:
                data_temp = pkl.load(fid)
            print('  LOAD SUCCESS')
        except:
            print('  LOAD ERROR. Skipping.')

        b_keyinlist = any([dict_key == field_new for dict_key in data_temp.keys()])

        if b_overwrite or not(b_keyinlist):
            data_temp[field_new] = data_new

        try:
            with open(os.path.join(directory, file), 'wb') as fid:
                pkl.dump(data_temp, fid)
            print('  UPDATE SUCCESS')
        except:
            print('  UPDATE ERROR')