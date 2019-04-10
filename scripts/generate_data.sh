#python test_data_size.py --N 3 --dir "home_energy_test_pr_squeeze_20190403"
#python test_data_size.py --N 3 --dir "home_energy_test_unit_var_20190403"
#python test_data_size.py --N 3 --dir "server_power_pr_squeeze_20190403"

python generate_reduced_datasets.py --N 5 --dataset "metasense" --brd 12 --dir "metasense12_ones_20190407" --w_type 1
python generate_reduced_datasets.py --N 5 --dataset "metasense" --brd 12 --dir "metasense12_cov_max2_20190407" --w_type 2
python generate_reduced_datasets.py --N 5 --dataset "metasense" --brd 12 --dir "metasense12_unit_var_20190407" --w_type 3
python generate_reduced_datasets.py --N 5 --dataset "metasense" --brd 12 --dir "metasense12_imbalanced_20190407" --w_type 5

python generate_reduced_datasets.py --N 5 --dataset "metasense" --brd 11 --dir "metasense11_ones_20190407" --w_type 1
python generate_reduced_datasets.py --N 5 --dataset "metasense" --brd 11 --dir "metasense11_cov_max2_20190407" --w_type 2
python generate_reduced_datasets.py --N 5 --dataset "metasense" --brd 11 --dir "metasense11_unit_var_20190407" --w_type 3
python generate_reduced_datasets.py --N 5 --dataset "metasense" --brd 11 --dir "metasense11_imbalanced_20190407" --w_type 5
