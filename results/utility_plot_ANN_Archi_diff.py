import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from itertools import compress
import matplotlib.font_manager
import argparse


if __name__ == '__main__':
    directory_target = 'ann_test_home_energy_201905031'
    list_filename_base = ['home_energy']
    plot_type = 'Training'  # 'Data_Size' / 'Training'
    FLAG_SAVE_COMPUTATION = False
    filename_save = 'results_metasense12_imbalanced_20190407.pkl'

    directory = os.path.join(os.path.dirname(__file__), 'raw', directory_target)

    if FLAG_SAVE_COMPUTATION and os.path.isfile(os.path.join(os.path.dirname(__file__), filename_save)):
        print('Existing results file found. Loading...')
        try:
            with open(os.path.join(os.path.dirname(__file__), filename_save), 'rb') as fid:
                data_temp = pkl.load(fid)
            data = data_temp['data']
            N_datapoints = data_temp['N_datapoints']
            print('  SUCCESS')
        except:
            print('  ERROR loading results file.')

    else:
        if not(FLAG_SAVE_COMPUTATION):
            print('Generating...')
        else:
            print('Existing results file not found. Generating...')

        # Create an initial list to hold all entries
        data = []
        N_datapoints = -1

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
                print('  SUCCESS')
            except:
                print('  ERROR. Skipping.')

            temp_filename_base = list(compress(list_filename_base, filename_match))[0]

            # list_filepieces = file.split(sep=temp_filename_base)[-1].split(sep='_')
            list_filepieces = file.split(sep='_')
            compression_ratio = float(list_filepieces[-11])
            print(compression_ratio)
            filetype = list_filepieces[-1]

            if N_datapoints < 0 and any([dict_key == 'N_datapoints' for dict_key in data_temp.keys()]):
                N_datapoints = data_temp['N_datapoints']

            if filetype == 'reduced.pkl':
                try:
                    score_odq = np.sqrt(np.mean( (data_temp['Y_test'] - data_temp['Y_odq_predict_w3'])**2 ))
                    score_res = np.sqrt(np.mean( (data_temp['Y_test'] - data_temp['Y_reservoir_predict'])**2 ))
                except:
                    print('  ERROR. Missing expected entries.')
                    continue

                # Look for entry in data that has same n_samples value
                entry = list(filter(lambda data_entry: data_entry['compression_ratio'] == compression_ratio, data))
                if entry:
                    entry[0]['score_odq'].append(score_odq)
                    entry[0]['score_res'].append(score_res)
                    entry[0]['history_odq'].append(data_temp['history_odq_w3'])
                    entry[0]['history_reservoir'].append(data_temp['history_reservoir'])
                else:
                    data.append({'compression_ratio': compression_ratio,
                                 'score_odq': [score_odq], 'score_res': [score_res],
                                 'history_odq': [data_temp['history_odq_w3']], 'history_reservoir': [data_temp['history_reservoir']]})

            elif filetype == 'full.pkl':
                try:
                    score_full = np.sqrt(np.mean( (data_temp['Y_test'] - data_temp['Y_full_predict'])**2 ))
                except:
                    print('  ERROR. Missing expected entries.')
                    continue

                # Look for entry in data that relates to full dataset (compression_ratio = 1)
                entry = list(filter(lambda data_entry: data_entry['compression_ratio'] == 1, data))
                if entry:
                    entry[0]['score_full'].append(score_full)
                else:
                    data.append({'compression_ratio': 1, 'score_full': [score_full]})

            else:
                print('  Unrecognized file type: {0}'.format(filetype))
                continue

        # Save results to remove need to regenerate
        with open(os.path.join(os.path.dirname(__file__), filename_save), 'wb') as fid:
            pkl.dump({'data':data, 'N_datapoints':N_datapoints}, fid)

    # Generate plots for Data Size
    if plot_type == 'Data_Size':
        list_n          = np.array([])
        list_odq_mean   = np.array([])
        list_odq_std    = np.array([])
        list_res_mean   = np.array([])
        list_res_std    = np.array([])
        list_full_mean = 0
        list_full_std = 1

        for data_entry in data:
            if data_entry['compression_ratio'] > 1:
                list_n = np.append(list_n, N_datapoints // data_entry['compression_ratio'])
                list_odq_mean = np.append(list_odq_mean, np.mean(data_entry['score_odq']))
                list_odq_std  = np.append(list_odq_std,  np.std(data_entry['score_odq']))
                list_res_mean = np.append(list_res_mean, np.mean(data_entry['score_res']))
                list_res_std  = np.append(list_res_std,  np.std(data_entry['score_res']))
            else:
                list_full_mean = np.mean(data_entry['score_full'])
                list_full_std  = np.std(data_entry['score_full'])

        ind_sorted = np.argsort(list_n)
        list_n = list_n[ind_sorted]
        list_odq_mean = list_odq_mean[ind_sorted]
        list_odq_std  = list_odq_std[ind_sorted]
        list_res_mean = list_res_mean[ind_sorted]
        list_res_std  = list_res_std[ind_sorted]

        plt.figure()
        plt.rc('font', family='Liberation Serif', size=14)
        plt.xscale('log')
        plt.plot([np.min(list_n), np.max(list_n)], [list_full_mean, list_full_mean], 'k-')
        plt.errorbar(list_n, list_odq_mean, yerr=list_odq_std)
        plt.errorbar(list_n, list_res_mean, yerr=list_res_std)
        plt.legend(('Full', 'ODQ', 'Reservoir'))
        #plt.plot([np.min(list_n), np.max(list_n)], [list_full_mean + list_full_std, list_full_mean + list_full_std], 'k:')
        #plt.plot([np.min(list_n), np.max(list_n)], [list_full_mean - list_full_std, list_full_mean - list_full_std], 'k:')
        plt.xlabel('Number of Samples Retained')
        plt.ylabel('Error (MSE)')
        plt.title('Home Energy Dataset')
        plt.grid('on')
        plt.tight_layout()
        plt.show()

    elif plot_type == 'Training':
        for data_entry in data:
            for history_odq, history_reservoir in zip(data_entry['history_odq'], data_entry['history_reservoir']):
                plt.figure()
                plt.rc('font', family='Liberation Serif', size=14)
                plt.plot(history_odq['val_mean_squared_error'], 'b')
                plt.plot(history_reservoir['val_mean_squared_error'], 'r')
                plt.title('Validation CR = {0}'.format(data_entry['compression_ratio']))
                plt.legend(('ODQ', 'Reservoir'))

                plt.figure()
                plt.rc('font', family='Liberation Serif', size=14)
                plt.plot(history_odq['mean_squared_error'], 'k')
                plt.plot(history_odq['val_mean_squared_error'], 'b')
                plt.title('ODQ CR = {0}'.format(data_entry['compression_ratio']))
                plt.legend(('Train', 'Val'))

                plt.figure()
                plt.rc('font', family='Liberation Serif', size=14)
                plt.plot(history_reservoir['mean_squared_error'], 'k')
                plt.plot(history_reservoir['val_mean_squared_error'], 'b')
                plt.title('Reservoir CR = {0}'.format(data_entry['compression_ratio']))
                plt.legend(('Train', 'Val'))
                plt.show()

