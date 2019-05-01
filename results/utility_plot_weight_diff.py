import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from itertools import compress
import argparse


if __name__ == '__main__':
    directory_target = 'metasense11_w3_w6_20190411_lr0.0001_std0.001_f1_c1'
    plot_type = 'Data_Size'  # 'Data_Size' / 'Training'
    FLAG_SAVE_COMPUTATION = False

    filename_save = 'results_{0}.pkl'.format(directory_target)
    directory = os.path.join(os.path.dirname(__file__), 'raw', directory_target)

    if FLAG_SAVE_COMPUTATION and os.path.isfile(os.path.join(os.path.dirname(__file__), filename_save)):
        print('Existing results file found. Loading...')
        try:
            with open(os.path.join(os.path.dirname(__file__), filename_save), 'rb') as fid:
                data_temp = pkl.load(fid)
            data = data_temp['data']
            N_datapoints = data_temp['N_datapoints']
            list_w_type = data_temp['list_w_type']
            print('  SUCCESS')
        except:
            print('  ERROR loading results file.')

    else:
        if FLAG_SAVE_COMPUTATION:
            print('Generating...')
        else:
            print('Existing results file not found. Generating...')

        # Create an initial list to hold all entries
        data = []
        N_datapoints = -1

        for file in os.listdir(directory):
            if not(file.lower().endswith('.pkl')):
                continue

            print('Loading data from {0}'.format(file))

            try:
                with open(os.path.join(directory, file), 'rb') as fid:
                    data_temp = pkl.load(fid)
                print('  SUCCESS')
            except:
                print('  ERROR. Skipping.')

            list_keys_Y_odq_predict = [key for key in data_temp if key.startswith('Y_odq_predict')]
            N_weights = len(list_keys_Y_odq_predict)
            list_filepieces = file.split(sep='_')
            ind_config = [ind for ind, filepiece in enumerate(list_filepieces) if filepiece=='data'][0]
            str_dataset = '_'.join(list_filepieces[:ind_config])
            str_date = list_filepieces[ind_config+2]
            compression_ratio = float(list_filepieces[ind_config+3])
            lr = float([filepiece[2:] for filepiece in list_filepieces if filepiece.startswith('lr')][0])
            std_noise = float([filepiece[3:] for filepiece in list_filepieces if filepiece.startswith('std')][0])
            filetype = list_filepieces[-1]

            if N_datapoints < 0 and any([dict_key == 'N_datapoints' for dict_key in data_temp.keys()]):
                N_datapoints = data_temp['N_datapoints']

            if filetype == 'reduced.pkl':
                try:
                    list_w_type = []
                    list_score_odq = []
                    for key in list_keys_Y_odq_predict:
                        list_score_odq.append(np.sqrt(np.mean( (data_temp['Y_test'] - data_temp[key])**2 )))
                        list_w_type.append(key.replace('Y_odq_predict_', ''))
                    score_res = np.sqrt(np.mean((data_temp['Y_test'] - data_temp['Y_reservoir_predict']) ** 2))
                except:
                    print('  ERROR. Missing expected entries.')
                    continue

                # Look for entry in data that has same n_samples value
                entry = list(filter(lambda data_entry: data_entry['compression_ratio'] == compression_ratio, data))
                if entry:
                    entry[0]['score_res'].append(score_res)
                    entry[0]['history_reservoir'].append(data_temp['history_reservoir'])
                    for score_odq, w_type in zip(list_score_odq, list_w_type):
                        entry[0]['score_odq_{0}'.format(w_type)].append(score_odq)
                        entry[0]['history_odq_{0}'.format(w_type)].append(data_temp['history_odq_{0}'.format(w_type)])
                else:
                    dict_temp = {'compression_ratio': compression_ratio, 'score_res': [score_res],
                                 'history_reservoir': [data_temp['history_reservoir']]}
                    for score_odq, w_type in zip(list_score_odq, list_w_type):
                        dict_temp['score_odq_{0}'.format(w_type)] = [score_odq]
                        dict_temp['history_odq_{0}'.format(w_type)] = [data_temp['history_odq_{0}'.format(w_type)]]
                    data.append(dict_temp)

            elif filetype == 'full.pkl':
                # TODO Probably broken, needs testing
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
            pkl.dump({'data':data, 'N_datapoints':N_datapoints, 'list_w_type':list_w_type}, fid)

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
                temp_list_mean = []
                temp_list_std = []
                for w_type in list_w_type:
                    temp_list_mean.append(np.mean(data_entry['score_odq_{0}'.format(w_type)]))
                    temp_list_std.append(np.std(data_entry['score_odq_{0}'.format(w_type)]))
                if list_odq_mean.shape[0] == 0:
                    list_odq_mean = np.array(temp_list_mean)
                    list_odq_std = np.array(temp_list_std)
                else:
                    list_odq_mean = np.vstack((list_odq_mean, temp_list_mean))
                    list_odq_std  = np.vstack((list_odq_std,  temp_list_std))
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
        for ind_col in range(len(list_w_type)):
            plt.errorbar(list_n, list_odq_mean[:, ind_col], yerr=list_odq_std[:, ind_col])
        plt.errorbar(list_n, list_res_mean, yerr=list_res_std)
        str_legend = ['Full'] + ['ODQ {0}'.format(w_type) for w_type in list_w_type] + ['Reservoir']
        plt.legend(str_legend)
        plt.plot([np.min(list_n), np.max(list_n)], [list_full_mean + list_full_std, list_full_mean + list_full_std], 'k:')
        plt.plot([np.min(list_n), np.max(list_n)], [list_full_mean - list_full_std, list_full_mean - list_full_std], 'k:')
        plt.xlabel('Number of Samples Retained')
        plt.ylabel('Error (MSE)')
        plt.title('')
        plt.grid('on')
        plt.tight_layout()
        plt.show()

    elif plot_type == 'Training':
        for data_entry in data:
            for history_odq, history_reservoir in zip(data_entry['history_odq_w3'], data_entry['history_reservoir']):
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

