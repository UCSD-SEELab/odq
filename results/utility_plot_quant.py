import os
import sys
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from itertools import compress
import argparse

colormap = plt.get_cmap('tab10')
colormap_ind = 5
colormap_max = 10

colormap_hist = plt.get_cmap('tab20')


def get_next_color():
    """
    Returns next color tuple from colormap
    """
    global colormap_ind

    color = plt.get_cmap('tab10')(colormap_ind)
    colormap_ind = (colormap_ind + 1) % colormap_max
    return color


def get_next_color_ind():
    """
    Returns the next color index
    """
    global colormap_ind
    colormap_ind = (colormap_ind + 1) % colormap_max
    return colormap_ind


if __name__ == '__main__':
    directory_target = 'metasense_20190828'
    plot_type = 'Quant_Spread'  # 'Data_Size' / 'Training' / 'Range_Acc' / 'Quant_Spread'
    list_quantizers_plot = ['reservoir', 'odq_3', 'omes', 'ks']

    err_max = None

    colors_ind = {'reservoir': 2,
                  'omes': 0,
                  'odq_3': 1,
                  'odq_11': 1,
                  'ks': 3,
                  }

    colors_plot = {}
    for key in colors_ind:
        colors_plot[key] = np.array(colormap(colors_ind[key])).reshape(1,-1)

    """
    Structure of input dictionaries from files:

        dict_in
            dataset_name
            compression_ratio
            X_test
            Y_test
            X_train
            Y_train
            min_max_scaler_x
            min_max_scaler_y
            train_test_ratio
            train_val_ratio
            quantizers
                desc
                quantizer
    """
    directory = os.path.join(os.path.dirname(__file__), 'quantized', directory_target)
    directory_img = os.path.join(os.path.dirname(__file__), 'img', directory_target, plot_type)

    if not (os.path.isdir(directory)):
        print('ERROR: {0} does not exist.'.format(directory))
        sys.exit()
    if not (os.path.isdir(directory_img)):
        print('Creating save directory')
        os.makedirs(directory_img)

    # Create an initial list to hold all entries
    dict_collected = {'results': [], 'Y_results': []}

    if plot_type == 'Quant_Spread':
        """
        Generate scatter plots of each input feature vs output target for all quantized files
        """
        for file in os.listdir(directory):
            if not (file.lower().endswith('.pkl')):
                continue

            print('Loading data from {0}'.format(file))

            try:
                with open(os.path.join(directory, file), 'rb') as fid:
                    dict_in = pkl.load(fid)
                print('  SUCCESS')
            except:
                print('  ERROR. Skipping.')

            filename_save = file.replace('.pkl', '')

            filename_parts = filename_save.split('_')

            Y_range = []
            X_range = []

            # Generate plot for original data
            for ind_X in range(dict_in['X_train'].shape[1]):
                X_plt = dict_in['X_train'][:, ind_X]
                Y_plt = dict_in['Y_train'].flatten()

                Y_range = (np.min(Y_plt) - 5, np.max(Y_plt) + 5)
                X_range.append((np.min(X_plt) - 5, np.max(X_plt) + 5))

                plt.figure(figsize=(4, 2.5))
                plt.rc('font', family='Liberation Serif', size=14)
                plt.scatter(X_plt, Y_plt, c='k', s=1)
                plt.title('CR{0} tr{1} Y vs X{2}'.format(filename_parts[-3], filename_parts[-2], ind_X))
                plt.xlim(X_range[ind_X])
                plt.ylim(Y_range)
                plt.xlabel('X{0}'.format(ind_X))
                plt.ylabel('Y')
                plt.savefig(os.path.join(directory_img, 'fig_{0}_Y_X{1}_full.png'.format(filename_save, ind_X)))
                plt.close()

            # Generate plot for each quantized set
            for quantizer in dict_in['quantizers']:
                print('{0}'.format(quantizer['desc'].upper()))

                if not (quantizer['desc'] in colors_plot):
                    colors_plot[quantizer['desc']] = get_next_color()

                X_quantized, Y_quantized = quantizer['quantizer'].get_dataset()

                for ind_X in range(X_quantized.shape[1]):
                    X_plt = X_quantized[:, ind_X]
                    Y_plt = Y_quantized.flatten()

                    plt.figure(figsize=(4, 2.5))
                    plt.rc('font', family='Liberation Serif', size=14)
                    plt.scatter(X_plt, Y_plt, c=colors_plot[quantizer['desc']], s=1)
                    plt.title('{3} CR{0} tr{1} Y vs X{2}'.format(filename_parts[-3], filename_parts[-2], ind_X,
                                                                 quantizer['desc'].upper()))
                    plt.xlim(X_range[ind_X])
                    plt.ylim(Y_range)
                    plt.xlabel('X{0}'.format(ind_X))
                    plt.ylabel('Y')
                    plt.savefig(os.path.join(directory_img, 'fig_{0}_Y_X{1}_{2}.png'.format(filename_save, ind_X,
                                                                                            quantizer['desc'].upper())))
                    plt.close()

        '''
        """
        Generate plots showing saved data size vs test error

        Creates 'results' structured as follows:

        dict_collected
            results (list)
                desc_quant
                desc_model
                cfg_str_short
                dataset_name
                n_points
                error (np.array of columns CR, RMSE, t_train)
        """
        # Collect all results from files
        N_datapoints = -1

        for file in os.listdir(directory):
            if not (file.lower().endswith('.pkl')):
                continue

            print('Loading data from {0}'.format(file))

            try:
                with open(os.path.join(directory, file), 'rb') as fid:
                    dict_in = pkl.load(fid)
                print('  SUCCESS')
            except:
                print('  ERROR. Skipping.')

            if not ('quantizer_results' in dict_in):
                print('ERROR: Old data file format. Skipping.')
                continue

            for quantizer_result in dict_in['quantizer_results']:
                if not (quantizer_result['desc'] in list_quantizers_plot):
                    continue

                if not (quantizer_result['desc'] in colors_plot):
                    colors_plot[quantizer_result['desc']] = get_next_color()

                for model_result in quantizer_result['model_results']:
                    if not (model_result['desc'].endswith('default')) and ('cfg' in model_result):
                        cfg_str = '_' + '_'.join('{0}{1}'.format(key, val) for key, val in model_result['cfg'].items())
                        if ('N_layer' in model_result['cfg']) and ('N_weights' in model_result['cfg']):
                            cfg_str_short = '{0} deep, {1} total'.format(model_result['cfg']['N_layer'],
                                                                         model_result['cfg']['N_weights'])
                        else:
                            cfg_str_short = 'unknown'
                    else:
                        cfg_str = 'default'
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
                                     'error': np.zeros((0, 3))}
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

                    if np.isnan(model_result['rmse']):
                        print('  ERROR: {0} {1} {2} RMSE value is NaN'.format(dict_in['dataset_name'], quantizer_result['desc'], model_desc))
                    dict_temp['error'] = np.append(dict_temp['error'],
                                                   [[dict_in['compression_ratio'], model_result['rmse'], model_result['t_train']]],
                                                   axis=0)

        # Plot processed results
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
                plt.rc('font', family='Liberation Serif', size=14)
                plt.xscale('log')
                # plt.plot([np.min(list_n), np.max(list_n)], [list_full_mean, list_full_mean], 'k-')
                for datum in data:
                    plt.errorbar(datum['n_train'], datum['err_mean'], yerr=datum['err_std'])
                str_legend = [datum['desc_quant'] for datum in data]
                plt.legend(str_legend)
                if err_max:
                    plt.ylim(bottom=0, top=err_max)
                plt.xlabel('Number of Samples Retained')
                plt.ylabel('Error (MSE)')
                plt.title('{0} {2} ({1})'.format(dataset_target, result_model['desc_model'].split('_')[0], result_model['cfg_str_short']))
                plt.grid('on')
                plt.tight_layout()
                plt.savefig(os.path.join(directory_img,
                                         'fig_data_size_{0}_{1}_{2}.png'.format(dataset_target,
                                                                            result_model['desc_model'].split('_')[0],
                                                                            result_model['cfg_str_short'])))
                plt.close()
                # plt.show()

    elif plot_type == 'Range_Acc':
        """
        Generate plots for each compression ratio showing the test error over the measurement range

        Creates 'Y_results' structured as follows:

        dict_collected
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
        # Collect all results from files
        for file in os.listdir(directory):
            if not (file.lower().endswith('.pkl')):
                continue

            print('Loading data from {0}'.format(file))

            try:
                with open(os.path.join(directory, file), 'rb') as fid:
                    dict_in = pkl.load(fid)
                print('  SUCCESS')
            except:
                print('  ERROR. Skipping.')

            if not ('quantizer_results' in dict_in):
                print('ERROR: Old data file format. Skipping.')
                continue

            for quantizer_result in dict_in['quantizer_results']:
                if not (quantizer_result['desc'] in list_quantizers_plot):
                    continue

                if not (quantizer_result['desc'] in colors_plot):
                    colors_plot[quantizer_result['desc']] = get_next_color()

                for model_result in quantizer_result['model_results']:
                    if not (model_result['desc'].endswith('default')) and ('cfg' in model_result):
                        cfg_str = '_' + '_'.join('{0}{1}'.format(key, val) for key, val in model_result['cfg'].items())
                        if ('N_layer' in model_result['cfg']) and ('N_weights' in model_result['cfg']):
                            cfg_str_short = '{0} deep, {1} total'.format(model_result['cfg']['N_layer'],
                                                                         model_result['cfg']['N_weights'])
                        else:
                            cfg_str_short = 'unknown'
                    else:
                        cfg_str = 'default'
                        cfg_str_short = 'default'

                    model_desc = '{0}{1}'.format(model_result['desc'], cfg_str)

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

        # Plot processed results
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
                plt.rc('font', family='Liberation Serif', size=12)
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
                plt.close()

    elif plot_type == 'Training':
        """
        Generate plots of training history for each file

        Uses 'history' from dict_in:

        dict_in
            quantizer_results (list)
                ...
                model_results (list)
                    history
                    ...

        'history' is structured as follows:

        history


        """
        for file in os.listdir(directory):
            if not (file.lower().endswith('.pkl')):
                continue

            print('Loading data from {0}'.format(file))

            try:
                with open(os.path.join(directory, file), 'rb') as fid:
                    dict_in = pkl.load(fid)
                print('  SUCCESS')
            except:
                print('  ERROR. Skipping.')

            filename_save = file.replace('.pkl', '')

            for quantizer_result in dict_in['quantizer_results']:
                if not (quantizer_result['desc'] in list_quantizers_plot):
                    continue

                if not (quantizer_result['desc'] in colors_ind):
                    colors_ind[quantizer_result['desc']] = get_next_color_ind()

                for model_result in quantizer_result['model_results']:
                    if not (model_result['desc'].endswith('default')) and ('cfg' in model_result):
                        if ('N_layer' in model_result['cfg']) and ('N_weights' in model_result['cfg']):
                            cfg_str_short = 'Nlayer{0}_Nweights{1}'.format(model_result['cfg']['N_layer'],
                                                                          model_result['cfg']['N_weights'])
                        else:
                            cfg_str_short = 'unknown'
                    else:
                        cfg_str_short = 'default'

                    if 'lr' in model_result['cfg']:
                        cfg_str_short += '_lr{0}'.format(model_result['cfg']['lr'])
                    else:
                        cfg_str_short += '_lrdefault'
                    if 'decay' in model_result['cfg']:
                        cfg_str_short += '_decay{0}'.format(model_result['cfg']['decay'])
                    else:
                        cfg_str_short += '_decaydefault'

                    plt.figure()
                    plt.rc('font', family='Liberation Serif', size=14)
                    plt.plot(model_result['history']['epoch'], model_result['history']['mean_squared_error'], 'k')
                    plt.plot(model_result['history']['epoch'],model_result['history']['val_mean_squared_error'], 'b')
                    if 'N_layer' in model_result['cfg']:
                        if model_result['cfg']['N_layer'] == 2:
                            plt.ylim(0, 0.04)
                        elif model_result['cfg']['N_layer'] == 5:
                            plt.ylim(0, 0.06)
                        elif model_result['cfg']['N_layer'] == 10:
                            plt.ylim(0, 0.1)
                        elif model_result['cfg']['N_layer'] == 20:
                            plt.ylim(0, 0.15)
                    plt.title('{0} {4} CR{1} \n({2} {3})'.format(dict_in['dataset_name'], dict_in['compression_ratio'],
                                                           model_result['desc'].split('_')[0], cfg_str_short,
                                                           quantizer_result['desc']))
                    plt.legend(('Train', 'Val'))
                    plt.savefig(os.path.join(directory_img,
                                             'fig_training_{0}_{1}_{2}_{3}.png'.format(filename_save,
                                                                                   model_result['desc'].split('_')[0],
                                                                                   cfg_str_short,
                                                                                   quantizer_result['desc'])))
                    plt.close()
        '''