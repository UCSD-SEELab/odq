import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from itertools import compress


directory = os.path.join(os.path.dirname(__file__), 'raw')
#list_filename_base = ['server_power_Data_Size_20190313202112',
#                      'server_power_Data_Size_20190313235700']
#filename_target = 'results_server_power_data_size_3iter.pkl'

list_filename_base = ['home_energy_Data_Size_20190312185618',
                      'home_energy_Data_Size_20190312183943',
                      'home_energy_Data_Size_20190312182359']
filename_target = 'results_home_energy_data_size_initial.pkl'

plot_type = 'Data_Size'

if os.path.isfile(os.path.join(os.path.dirname(__file__), filename_target)):
    print('Existing results file found. Loading...')
    try:
        with open(os.path.join(os.path.dirname(__file__), filename_target), 'rb') as fid:
            data = pkl.load(fid)
        print('  SUCCESS')
    except:
        print('  ERROR loading results file.')

else:
    print('Existing results file not found. Generating...')
    data = []

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

        list_filepieces = file.split(sep=temp_filename_base)[-1].split(sep='_')
        n_datapoints = int(list_filepieces[1])
        filetype = list_filepieces[-1]

        if filetype == 'final.pkl':
            try:
                score_odq = np.sqrt(np.mean( (data_temp['Y_test'] - data_temp['Y_odq_predict'])**2 ))
                score_res = np.sqrt(np.mean( (data_temp['Y_test'] - data_temp['Y_reservoir_predict'])**2 ))
            except:
                print('  ERROR. Missing expected entries.')
                continue

            # Look for entry in data that has same n_samples value
            entry = list(filter(lambda data_entry: data_entry['n_datapoints'] == n_datapoints, data))
            if entry:
                entry[0]['score_odq'].append(score_odq)
                entry[0]['score_res'].append(score_res)
            else:
                data.append({'n_datapoints': n_datapoints, 'score_odq': [score_odq], 'score_res': [score_res] })
        elif filetype == 'full.pkl':
            try:
                score_full = np.sqrt(np.mean( (data_temp['Y_test'] - data_temp['Y_full_predict'])**2 ))
            except:
                print('  ERROR. Missing expected entries.')
                continue

            # Look for entry in data that relates to full dataset (n_datapoints = -1)
            entry = list(filter(lambda data_entry: data_entry['n_datapoints'] == -1, data))
            if entry:
                entry[0]['score_full'].append(score_full)
            else:
                data.append({'n_datapoints': -1, 'score_full': [score_full]})

        else:
            print('  Unrecognized file type: {0}'.format(filetype))
            continue

    # Save results to remove need to regenerate
    with open(os.path.join(os.path.dirname(__file__), filename_target), 'wb') as fid:
        pkl.dump(data, fid)

# Generate plots for Data Size
if plot_type == 'Data_Size':
    list_n          = np.array([])
    list_odq_mean   = np.array([])
    list_odq_std    = np.array([])
    list_res_mean   = np.array([])
    list_res_std    = np.array([])

    for data_entry in data:
        if int(data_entry['n_datapoints']) > 0:
            list_n = np.append(list_n, int(data_entry['n_datapoints']))
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
    plt.plot([np.min(list_n), np.max(list_n)], [list_full_mean, list_full_mean])
    plt.errorbar(list_n, list_odq_mean, yerr=list_odq_std)
    plt.errorbar(list_n, list_res_mean, yerr=list_res_std)
    plt.legend(('Full', 'ODQ', 'Reservoir'))
    plt.xlabel('Number of Samples Retained')
    plt.ylabel('Error (MSE)')
    plt.title('Appliance Energy Dataset')
    plt.grid('on')
    plt.show()
