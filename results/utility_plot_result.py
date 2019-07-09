import os
import sys
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from itertools import compress
import argparse

colormap = plt.get_cmap('tab10')
colormap_ind = 4
colormap_max = 10

def get_next_color():
    """
    Returns next color tuple from colormap
    """
    global colormap_ind

    color = plt.get_cmap('tab10')(colormap_ind)
    colormap_ind = (colormap_ind + 1) % colormap_max
    return color

if __name__ == '__main__':
    directory_target = 'server_power_20190614'
    plot_type = 'Range_Acc'  # 'Data_Size' / 'Training' / 'Range_Acc'
    list_quantizers_plot = ['reservoir', 'odq_3', 'omes']

    err_max = None

    colors_plot = {'reservoir': colormap(3),
                   'omes':      colormap(0),
                   'odq_3':     colormap(2),
                   'odq_11':    colormap(1),
                  }

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='Target directory of files.')
    parser.add_argument('--plot', type=str, help='plot type: Data_Size | Training | Range_Acc')

    args = parser.parse_args()

    if args.dir is not None:
        directory_target = args.dir

    if args.plot is not None:
        plot_type = args.plot


    """
    Structure of input dictionaries from files:

        dict_in
            dataset_name
            compression_ratio
            trial_num
            min_max_scaler_x
            min_max_scaler_y
            X_test
            Y_test
            X_val
            Y_val
            filename_datasets
            quantizer_results (list)
                desc
                model_results (list)
                    desc
                    cfg
                    desc
                    history
                    score
                    Y_predict
                    rmse
                    t_train
    
    Processed into the following structure:
    
        dict_collected
            results (list)
                desc_quant
                desc_model
                cfg_str_short
                dataset_name
                n_points
                error (np.array of columns CR, RMSE, t_train)
            Y_results (list)
                desc_model
                cfg_str_short
                dataset_name
                compression_ratio
                quantizer_results (list)
                    desc_quant
                    Y_predict
                    Y_test
    """
    directory = os.path.join(os.path.dirname(__file__), 'raw', directory_target)
    directory_img = os.path.join(os.path.dirname(__file__), 'img', directory_target)

    if not(os.path.isdir(directory)):
        print('ERROR: {0} does not exist.'.format(directory))
        sys.exit()
    if not(os.path.isdir(directory_img)):
        print('Creating save directory')
        os.mkdir(directory_img)

    # Create an initial list to hold all entries
    dict_collected = {'results': [], 'Y_results': []}
    N_datapoints = -1

    for file in os.listdir(directory):
        if not(file.lower().endswith('.pkl')):
            continue

        print('Loading data from {0}'.format(file))

        try:
            with open(os.path.join(directory, file), 'rb') as fid:
                dict_in = pkl.load(fid)
            print('  SUCCESS')
        except:
            print('  ERROR. Skipping.')

        if not('quantizer_results' in dict_in):
            print('ERROR: Old data file format. Skipping.')
            continue

        for quantizer_result in dict_in['quantizer_results']:
            if not(quantizer_result['desc'] in list_quantizers_plot):
                continue

            if not(quantizer_result['desc'] in colors_plot):
                colors_plot[quantizer_result['desc']] = get_next_color()

            for model_result in quantizer_result['model_results']:
                if not(model_result['desc'].endswith('default')) and ('cfg' in model_result):
                    cfg_str = '_' + '_'.join('{0}{1}'.format(key,val) for key, val in model_result['cfg'].items())
                    if ('N_layer' in model_result['cfg']) and ('N_weights' in model_result['cfg']):
                        cfg_str_short = '{0} deep, {1} total'.format(model_result['cfg']['N_layer'],
                                                                     model_result['cfg']['N_weights'])
                    else:
                        cfg_str_short = 'unknown'
                else:
                    cfg_str = ''
                    cfg_str_short = 'default'

                model_desc = '{0}{1}'.format(model_result['desc'], cfg_str)

                """
                results(list)
                    desc_quant
                    desc_model
                    cfg_str_short
                    dataset_name
                    n_points
                    error(np.array of columns of CR, RMSE, t_train)
                """
                # Add result to data_collected['results'], combining with previous entry if available
                dict_temp = next((entry for entry in dict_collected['results'] if
                    (entry['dataset_name'] == dict_in['dataset_name']) and
                    (entry['desc_quant'] == quantizer_result['desc']) and
                    (entry['desc_model'] == model_desc)), None)
                if dict_temp == None:
                    dict_temp = {'dataset_name': dict_in['dataset_name'],
                                 'desc_quant': quantizer_result['desc'],
                                 'desc_model': model_desc,
                                 'cfg_str_short': cfg_str_short,
                                 'error': np.zeros((0,3))}
                    if 'n_points' in dict_in:
                        dict_temp['n_points'] = dict_in['n_points']
                    else:
                        if dict_temp['dataset_name'] == 'metasense':
                            dict_temp['n_points'] = 83368
                        elif dict_temp['dataset_name'] == 'server_power':
                            dict_temp['n_points'] = 67017
                        elif dict_temp['dataset_name'] == 'home_energy':
                            dict_temp['n_points'] = 15788

                    dict_collected['results'].append(dict_temp)

                """
                Y_results (list)
                    desc_model
                    cfg_str_short
                    dataset_name
                    compression_ratio
                    quantizer_results (list)
                        desc_quant
                        Y_predict
                        Y_test
                """
                # Add result to data_collected['Y_results'], combining with previous entry if available
                if np.isnan(model_result['Y_predict']).any():
                    continue

                dict_temp_Y = next((entry for entry in dict_collected['Y_results'] if
                                    (entry['dataset_name'] == dict_in['dataset_name']) and
                                    (entry['compression_ratio'] == dict_in['compression_ratio']) and
                                    (entry['desc_model'] == model_desc)), None)
                if dict_temp_Y is not None:
                    dict_result_Y = next((entry for entry in dict_temp_Y['quantizer_results'] if
                                         (entry['desc_quant'] == quantizer_result['desc'])), None)
                    if dict_result_Y is None:
                        dict_result_Y = {'desc_quant': quantizer_result['desc'],
                                                                 'Y_predict': [], 'Y_test': []}
                        dict_temp_Y['quantizer_results'].append(dict_result_Y)
                    dict_result_Y['Y_predict'].extend(model_result['Y_predict'].flatten().tolist())
                    dict_result_Y['Y_test'].extend(dict_in['Y_test'].flatten().tolist())
                else:
                    dict_temp_Y = {'dataset_name': dict_in['dataset_name'],
                                   'compression_ratio': dict_in['compression_ratio'],
                                   'desc_model': model_desc, 'cfg_str_short': cfg_str_short,
                                   'quantizer_results': [{'desc_quant': quantizer_result['desc'],
                                                          'Y_predict': model_result['Y_predict'].flatten().tolist(),
                                                          'Y_test': dict_in['Y_test'].flatten().tolist()}]}
                    dict_collected['Y_results'].append(dict_temp_Y)

                if np.isnan(model_result['rmse']):
                    print('  ERROR: {0} {1} {2} RMSE value is NaN'.format(dict_in['dataset_name'], quantizer_result['desc'], model_desc))
                dict_temp['error'] = np.append(dict_temp['error'],
                                               [[dict_in['compression_ratio'], model_result['rmse'], model_result['t_train']]],
                                               axis=0)

    # Generate plots for each compression ratio showing the accuracy of the calculation over the measurement range
    if plot_type == 'Data_Size':
        """
        Generate plots showing saved data size vs test error
        
        Uses 'results' structured as follows:
        
        dict_collected
            results (list)
                desc_quant
                desc_model
                cfg_str_short
                dataset_name
                n_points
                error (np.array of columns CR, RMSE, t_train)
            Y_results (list)
                ...
        """
        list_dataset = np.unique([dataset['dataset_name'] for dataset in dict_collected['results']])

        for dataset_target in list_dataset:
            list_results_dataset  = [dataset for dataset in dict_collected['results'] if dataset['dataset_name'] == dataset_target]
            list_models = np.unique([result['desc_model'] for result in list_results_dataset])

            for model in list_models:
                list_results_model = [result for result in list_results_dataset if result['desc_model'] == model]

                data = []

                for result_model in list_results_model:
                    result = result_model['error']
                    list_cr = np.unique(result[:, 0])
                    list_n = []
                    list_err_mean = []
                    list_err_std = []
                    list_t_mean = []
                    list_t_std = []

                    for cr in list_cr:
                        temp = result[result[:,0] == cr, :]
                        temp_mean = np.nanmean(temp, axis=0)
                        temp_std = np.nanstd(temp, axis=0)
                        list_n.append(result_model['n_points'] // cr)
                        list_err_mean.append(temp_mean[1])
                        list_err_std.append(temp_std[1])
                        list_t_mean.append(temp_mean[2])
                        list_t_std.append(temp_std[2])

                    data.append({'desc_quant': result_model['desc_quant'], 'n_train': list_n,
                                 'err_mean': list_err_mean, 'err_std': list_err_std,
                                 't_mean': list_t_mean, 't_std': list_t_std})

                # Generate plot for unique model-dataset pair
                plt.figure()
                plt.rc('font', family='Arial', size=14)
                plt.xscale('log')
                # plt.plot([np.min(list_n), np.max(list_n)], [list_full_mean, list_full_mean], 'k-')
                for datum in data:
                    plt.errorbar(datum['n_train'], datum['err_mean'], yerr=datum['err_std'])
                str_legend = [datum['desc_quant'] for datum in data]
                plt.legend(str_legend)
                plt.xlabel('Number of Samples Retained')
                plt.ylabel('Error (MSE)')
                plt.title('{0} {2} ({1})'.format(dataset_target, result_model['desc_model'].split('_')[0], result_model['cfg_str_short']))
                plt.grid('on')
                plt.tight_layout()
                plt.show()

    elif plot_type == 'Range_Acc':
        """
        Generate plots for each compression ratio showing the test error over the measurement range
        
        Uses 'Y_results' structured as follows:
        
        dict_collected
            results (list)
                ...
            Y_results (list)
                desc_model
                cfg_str_short
                dataset_name
                compression_ratio
                quantizer_results (list)
                    desc_quant
                    Y_predict
                    Y_test
        """
        N_segs = 10     # number of bins used for dividing up the measurement range of the test set

        list_dataset = np.unique([Y_result['dataset_name'] for Y_result in dict_collected['Y_results']])

        for dataset_target in list_dataset:
            for Y_result in [dataset for dataset in dict_collected['Y_results'] if dataset['dataset_name'] == dataset_target]:
                # Find min and max of Y_test
                Y_min = min([min(result['Y_test']) for result in Y_result['quantizer_results']])
                Y_max = max([max(result['Y_test']) for result in Y_result['quantizer_results']])

                N_quantizers = len(Y_result['quantizer_results'])

                bin_edges = np.linspace(Y_min, Y_max, N_segs + 1)
                bin_width = bin_edges[1] - bin_edges[0]
                bar_width = (bin_width*0.9) / N_quantizers
                err_bins = np.zeros((N_quantizers, N_segs))

                # Calculate error over a given range of Y and generate bar plot comparing all available quantizers
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 6))
                plt.rcParams['axes.grid'] = True
                plt.rc('font', family='Arial', size=12)
                fig.suptitle('{0} CR{1} ({2} {3})'.format(dataset_target, Y_result['compression_ratio'],
                                                          Y_result['desc_model'].split('_')[0],
                                                          Y_result['cfg_str_short']))
                ax1.grid('on')
                ax2.grid('on')

                for ind_quant, quantizer_result in enumerate(Y_result['quantizer_results']):
                    for ind_bin in range(N_segs):
                        temp1 = quantizer_result['Y_test'] >= bin_edges[ind_bin]
                        temp2 = quantizer_result['Y_test'] < bin_edges[ind_bin + 1]
                        b_ind_valid = np.logical_and(temp1, temp2)

                        Y_test = np.array(quantizer_result['Y_test'])[b_ind_valid]
                        Y_predict = np.array(quantizer_result['Y_predict'])[b_ind_valid]

                        err_bins[ind_quant, ind_bin] = np.sqrt(np.mean((Y_test - Y_predict)**2))

                    ax1.bar(bin_edges[0:-1] + 0.05*bin_width + ind_quant*bar_width, err_bins[ind_quant, :],
                            width=bar_width, align='edge', color=colors_plot[quantizer_result['desc_quant']])

                if err_max:
                    ax1.set_ylim(bottom=0, top=err_max)
                ax1.legend([quantizer_result['desc_quant'] for quantizer_result in Y_result['quantizer_results']])
                ax1.set_ylabel('Error (rms)')

                Y_test = [element for result in Y_result['quantizer_results'] for element in result['Y_test'] ]
                ax2.hist(Y_test, bins=bin_edges, align='left', density=True)
                ax2.set_xlabel('Y_test')

                plt.savefig(os.path.join(directory_img, 'fig_range_{0}_cfg{2}_{3}_cr{1}.png'.format(dataset_target,
                                         Y_result['compression_ratio'], Y_result['desc_model'].split('_')[0],
                                         Y_result['cfg_str_short'])))

    """
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
    """
