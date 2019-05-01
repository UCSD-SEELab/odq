import os
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='Target directory for splitting.')
    args = parser.parse_args()

    directory_target = args.dir
    directory_target_full = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw', directory_target)

    if not os.path.exists(directory_target_full):
        print('Invalid directory')
        sys.exit()

    list_configs = []

    for file in os.listdir(directory_target_full):
        if not (file.lower().endswith('.pkl')):
            continue

        # Separate and compare current config
        list_filepieces = file.split(sep='_')

        filepiece_lr = [filepiece for filepiece in list_filepieces if filepiece.startswith('lr')][0]
        if filepiece_lr is None:
            filepiece_lr = 'lr0'
        filepiece_std = [filepiece for filepiece in list_filepieces if filepiece.startswith('std')][0]
        if filepiece_std is None:
            filepiece_std = 'std0'
        filepiece_f = [filepiece for filepiece in list_filepieces if filepiece.startswith('f')][0]
        if filepiece_f is None:
            filepiece_f = 'f0'
        filepiece_c = [filepiece for filepiece in list_filepieces if filepiece.startswith('c')][0]
        if filepiece_c is None:
            filepiece_c = 'c0'

        directory_new = '{0}_{1}_{2}_{3}_{4}'.format(directory_target, filepiece_lr, filepiece_std, filepiece_f, filepiece_c)
        directory_new_full = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw', directory_new)
        if not os.path.exists(directory_new_full):
            print('Making new directory: {0}'.format(directory_new))
            os.mkdir(directory_new_full)
            list_configs.append(directory_new)

        print('Moving {0}'.format(file))

        os.system('cp {0} {1}'.format(os.path.join(directory_target_full, file), os.path.join(directory_new_full, file)))



