import os
import sys
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from itertools import compress
import argparse


if __name__ == '__main__':
    directory_target = '<insert directory here>'
    plot_type = '<insert type here>'  # 'Data_Size' / 'Training' / 'Range_Acc'
    FLAG_SAVE_COMPUTATION = False

    filename_save = 'results_{0}.pkl'.format(directory_target)
    directory = os.path.join(os.path.dirname(__file__), 'raw', directory_target)

    if not(os.path.isdir(directory)):
        print('ERROR: {0} does not exist.'.format(directory))
        sys.exit()

    if FLAG_SAVE_COMPUTATION and os.path.isfile(os.path.join(os.path.dirname(__file__), filename_save)):
        print('Existing results file found. Loading...')
        try:
            with open(os.path.join(os.path.dirname(__file__), filename_save), 'rb') as fid:
                data_temp = pkl.load(fid)
            data = data_temp['data']
            N_datapoints = data_temp['N_datapoints']
            list_method = data_temp['list_method']
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

            list_keys_Y_predict = [key for key in data_temp if key.startswith('Y_predict')]
            if (len(list_keys_Y_predict) == 0):
                list_keys_Y_predict = [key for key in data_temp if key.startswith('Y_odq_predict')]
                for key_Y_predict in list_keys_Y_predict:
                    data_temp[key_Y_predict.replace('Y_odq', 'Y')] = data_temp[key_Y_predict]
            N_methods = len(list_keys_Y_predict)
            list_filepieces = file.split(sep='_')
            ind_config = [ind for ind, filepiece in enumerate(list_filepieces) if filepiece=='data'][0]
            str_dataset = '_'.join(list_filepieces[:ind_config])
            str_date = list_filepieces[ind_config+2]
            compression_ratio = float(list_filepieces[ind_config+3])
            lr = float([filepiece[2:] for filepiece in list_filepieces if filepiece.startswith('lr')][0])
            filetype = list_filepieces[-1]

            if N_datapoints < 0 and any([dict_key == 'N_datapoints' for dict_key in data_temp.keys()]):
                N_datapoints = data_temp['N_datapoints']

            if filetype == 'reduced.pkl':
                try:
                    list_method = []
                    list_score_quant = []
                    list_Y_test = []
                    for key in list_keys_Y_predict:
                        list_score_quant.append(np.sqrt(np.mean( (data_temp['Y_test'] - data_temp[key])**2 )))
                        list_method.append(key.replace('Y_odq_predict_', '').replace('Y_predict_',''))
                    score_res = np.sqrt(np.mean((data_temp['Y_test'] - data_temp['Y_reservoir_predict']) ** 2))
                except:
                    print('  ERROR. Missing expected entries.')
                    continue

                # Look for entry in data that has same n_samples value
                entry = list(filter(lambda data_entry: data_entry['compression_ratio'] == compression_ratio, data))
                if entry:
                    entry[0]['score_res'].append(score_res)
                    entry[0]['history_reservoir'].append(data_temp['history_reservoir'])
                    entry[0]['Y_res_predict'].append(data_temp['Y_reservoir_predict'][:,0])
                    entry[0]['Y_test'].append(data_temp['Y_test'][:,0])
                    for score_quant, method in zip(list_score_quant, list_method):
                        entry[0]['score_quant_{0}'.format(method)].append(score_quant)
                        entry[0]['history_{0}'.format(method)].append(data_temp['history_{0}'.format(method)])
                        entry[0]['Y_predict_{0}'.format(method)].append(data_temp['Y_predict_{0}'.format(method)][:,0])

                else:
                    dict_temp = {'compression_ratio': compression_ratio, 'score_res': [score_res],
                                 'history_reservoir': [data_temp['history_reservoir']],
                                 'Y_test': [data_temp['Y_test'][:,0]], 'Y_res_predict': [data_temp['Y_reservoir_predict'][:,0]]}
                    for score_quant, method in zip(list_score_quant, list_method):
                        dict_temp['score_quant_{0}'.format(method)]       = [score_quant]
                        dict_temp['history_{0}'.format(method)]     = [data_temp['history_{0}'.format(method)]]
                        dict_temp['Y_predict_{0}'.format(method)]   = [data_temp['Y_predict_{0}'.format(method)][:,0]]
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
            pkl.dump({'data':data, 'N_datapoints':N_datapoints, 'list_method':list_method}, fid)

    # Generate plots for each compression ratio showing the accuracy of the calculation over the measurement range
    if plot_type == 'Data_Size':
        list_n = np.array([])
        list_odq_mean = np.array([])
        list_odq_std = np.array([])
        list_res_mean = np.array([])
        list_res_std = np.array([])
        list_full_mean = 0
        list_full_std = 1

        for data_entry in data:
            if data_entry['compression_ratio'] > 1:
                list_n = np.append(list_n, N_datapoints // data_entry['compression_ratio'])
                temp_list_mean = []
                temp_list_std = []
                for method in list_method:
                    temp_list_mean.append(np.mean(data_entry['score_{0}'.format(method)]))
                    temp_list_std.append(np.std(data_entry['score_{0}'.format(method)]))
                if list_odq_mean.shape[0] == 0:
                    list_odq_mean = np.array(temp_list_mean)
                    list_odq_std = np.array(temp_list_std)
                else:
                    list_odq_mean = np.vstack((list_odq_mean, temp_list_mean))
                    list_odq_std = np.vstack((list_odq_std, temp_list_std))
                list_res_mean = np.append(list_res_mean, np.mean(data_entry['score_res']))
                list_res_std = np.append(list_res_std, np.std(data_entry['score_res']))
            else:
                list_full_mean = np.mean(data_entry['score_full'])
                list_full_std = np.std(data_entry['score_full'])

        ind_sorted = np.argsort(list_n)
        list_n = list_n[ind_sorted]
        list_odq_mean = list_odq_mean[ind_sorted]
        list_odq_std = list_odq_std[ind_sorted]
        list_res_mean = list_res_mean[ind_sorted]
        list_res_std = list_res_std[ind_sorted]

        plt.figure()
        plt.rc('font', family='Liberation Serif', size=14)
        plt.xscale('log')
        plt.plot([np.min(list_n), np.max(list_n)], [list_full_mean, list_full_mean], 'k-')
        for ind_col in range(len(list_method)):
            plt.errorbar(list_n, list_odq_mean[:, ind_col], yerr=list_odq_std[:, ind_col])
        plt.errorbar(list_n, list_res_mean, yerr=list_res_std)
        str_legend = ['Full'] + ['ODQ {0}'.format(method) for method in list_method] + ['Reservoir']
        plt.legend(str_legend)
        plt.plot([np.min(list_n), np.max(list_n)], [list_full_mean + list_full_std, list_full_mean + list_full_std],
                 'k:')
        plt.plot([np.min(list_n), np.max(list_n)], [list_full_mean - list_full_std, list_full_mean - list_full_std],
                 'k:')
        plt.xlabel('Number of Samples Retained')
        plt.ylabel('Error (MSE)')
        plt.title('')
        plt.grid('on')
        plt.tight_layout()
        plt.show()

    elif plot_type == 'Range_Acc':
        for data_entry in data:
            Y_test = np.concatenate(data_entry['Y_test']).flatten()
            Y_res_predict = np.concatenate(data_entry['Y_res_predict']).flatten()

            N_segs = 10
            err_quant_bins = np.zeros((N_segs))
            err_res_bins = np.zeros((N_segs))
            ind_bins = np.linspace(np.min(Y_test), np.max(Y_test), N_segs + 1)

            for method in list_method:
                Y_predict = np.concatenate(data_entry['Y_predict_{0}'.format(method)]).flatten()
                for ind in range(N_segs):
                    temp1 = Y_test >= ind_bins[ind]
                    temp2 = Y_test < ind_bins[ind+1]
                    b_ind_valid = np.logical_and(temp1, temp2)
                    err_quant_bins[ind] = np.sqrt(np.mean((Y_test[b_ind_valid] - Y_predict[b_ind_valid])**2))
                    err_res_bins[ind] = np.sqrt(np.mean((Y_test[b_ind_valid] - Y_res_predict[b_ind_valid])**2))

                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(5, 8))
                #plt.tight_layout()
                plt.rcParams['axes.grid'] = True
                plt.rc('font', family='Liberation Serif', size=12)
                fig.suptitle('Accuracy over FSO (CR{1}, {0})'.format(method, data_entry['compression_ratio']))
                ax1.grid('on')
                ax1.scatter(data_entry['Y_test'], data_entry['Y_res_predict'])
                ax1.scatter(data_entry['Y_test'], data_entry['Y_predict_{0}'.format(method)])
                ax1.legend(['Reservoir', '{0}'.format(method)])
                ax1.set_ylabel('Y_predict')

                bin_width = (ind_bins[1] - ind_bins[0])
                ax2.grid('on')
                ax2.bar(ind_bins[0:-1] + bin_width*0.05, err_res_bins, width=bin_width*0.45)
                ax2.bar(ind_bins[0:-1] + bin_width*0.5,  err_quant_bins, width=bin_width*0.45)
                ax2.set_ylabel('Error (rms)')

                ax3.grid('on')
                ax3.hist(Y_test, bins=ind_bins, align='left', density=True)
                ax3.set_xlabel('Y_test')

            plt.show()

    elif plot_type == 'Training':
        for data_entry in data:
            for method in list_method:
                for history_quant, history_reservoir in zip(data_entry['history_{0}'.format(method)], data_entry['history_reservoir']):
                    plt.figure()
                    plt.rc('font', family='Liberation Serif', size=14)
                    plt.plot(history_quant['val_mean_squared_error'], 'b')
                    plt.plot(history_reservoir['val_mean_squared_error'], 'r')
                    plt.title('Validation CR = {0}'.format(data_entry['compression_ratio']))
                    plt.legend(('{0}'.format(method), 'Reservoir'))

                    plt.figure()
                    plt.rc('font', family='Liberation Serif', size=14)
                    plt.plot(history_quant['mean_squared_error'], 'k')
                    plt.plot(history_quant['val_mean_squared_error'], 'b')
                    plt.title('{0} CR = {1}'.format(method, data_entry['compression_ratio']))
                    plt.legend(('Train', 'Val'))

                    plt.figure()
                    plt.rc('font', family='Liberation Serif', size=14)
                    plt.plot(history_reservoir['mean_squared_error'], 'k')
                    plt.plot(history_reservoir['val_mean_squared_error'], 'b')
                    plt.title('Reservoir CR = {0}'.format(data_entry['compression_ratio']))
                    plt.legend(('Train', 'Val'))
                    plt.show()
