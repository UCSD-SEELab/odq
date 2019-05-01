import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from itertools import compress
import argparse


if __name__ == '__main__':
    directory_target = 'metasense11_w3_w6_20190411_lr0.0001_std0.001_f1_c1'
    plot_type = 'Range_Acc'  # 'Data_Size' / 'Training'
    b_mae = False
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
            b_costmae = bool(int(([filepiece[1] for filepiece in list_filepieces if filepiece.startswith('c')][0])))
            filetype = list_filepieces[-1]

            if not(b_mae == b_costmae):
                print('  cost function incorrect')
                continue

            if N_datapoints < 0 and any([dict_key == 'N_datapoints' for dict_key in data_temp.keys()]):
                N_datapoints = data_temp['N_datapoints']

            if filetype == 'reduced.pkl':
                try:
                    list_w_type = []
                    list_score_odq = []
                    list_Y_test = []
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
                    entry[0]['Y_res_predict'].append(data_temp['Y_reservoir_predict'][:,0])
                    entry[0]['Y_test'].append(data_temp['Y_test'][:,0])
                    for score_odq, w_type in zip(list_score_odq, list_w_type):
                        entry[0]['score_odq_{0}'.format(w_type)].append(score_odq)
                        entry[0]['history_odq_{0}'.format(w_type)].append(data_temp['history_odq_{0}'.format(w_type)])
                        entry[0]['Y_odq_predict_{0}'.format(w_type)].append(data_temp['Y_odq_predict_{0}'.format(w_type)][:,0])

                else:
                    dict_temp = {'compression_ratio': compression_ratio, 'score_res': [score_res],
                                 'history_reservoir': [data_temp['history_reservoir']],
                                 'Y_test': [data_temp['Y_test'][:,0]], 'Y_res_predict': [data_temp['Y_reservoir_predict'][:,0]]}
                    for score_odq, w_type in zip(list_score_odq, list_w_type):
                        dict_temp['score_odq_{0}'.format(w_type)]       = [score_odq]
                        dict_temp['history_odq_{0}'.format(w_type)]     = [data_temp['history_odq_{0}'.format(w_type)]]
                        dict_temp['Y_odq_predict_{0}'.format(w_type)]   = [data_temp['Y_odq_predict_{0}'.format(w_type)][:,0]]
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

    # Generate plots for each compression ratio showing the accuracy of the calculation over the measurement range
    if plot_type == 'Range_Acc':
        for data_entry in data:
            Y_test = np.concatenate(data_entry['Y_test']).flatten()
            Y_res_predict = np.concatenate(data_entry['Y_res_predict']).flatten()

            N_segs = 10
            err_odq_bins = np.zeros((N_segs))
            err_res_bins = np.zeros((N_segs))
            ind_bins = np.linspace(np.min(Y_test), np.max(Y_test), N_segs + 1)

            for w_type in ['w3']: #list_w_type:
                Y_odq_predict = np.concatenate(data_entry['Y_odq_predict_{0}'.format(w_type)]).flatten()
                for ind in range(N_segs):
                    temp1 = Y_test >= ind_bins[ind]
                    temp2 = Y_test < ind_bins[ind+1]
                    b_ind_valid = np.logical_and(temp1, temp2)
                    err_odq_bins[ind] = np.sqrt(np.mean((Y_test[b_ind_valid] - Y_odq_predict[b_ind_valid])**2))
                    err_res_bins[ind] = np.sqrt(np.mean((Y_test[b_ind_valid] - Y_res_predict[b_ind_valid])**2))

                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(5, 8))
                #plt.tight_layout()
                plt.rcParams['axes.grid'] = True
                plt.rc('font', family='Liberation Serif', size=12)
                fig.suptitle('Accuracy over FSO (CR{1}, {0})'.format(w_type, data_entry['compression_ratio']))
                ax1.grid('on')
                ax1.scatter(data_entry['Y_test'], data_entry['Y_res_predict'])
                ax1.scatter(data_entry['Y_test'], data_entry['Y_odq_predict_{0}'.format(w_type)])
                ax1.legend(['Reservoir', 'ODQ {0}'.format(w_type)])
                ax1.set_ylabel('Y_predict')

                bin_width = (ind_bins[1] - ind_bins[0])
                ax2.grid('on')
                ax2.bar(ind_bins[0:-1] + bin_width*0.05, err_res_bins, width=bin_width*0.45)
                ax2.bar(ind_bins[0:-1] + bin_width*0.5,  err_odq_bins, width=bin_width*0.45)
                ax2.set_ylabel('Error (rms)')
                #ax2.legend(['Reservoir', 'ODQ'])

                ax3.grid('on')
                ax3.hist(Y_test, bins=ind_bins, align='left', density=True)
                ax3.set_xlabel('Y_test')

            plt.show()

