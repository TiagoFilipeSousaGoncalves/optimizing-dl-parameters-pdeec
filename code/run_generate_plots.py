# Imports
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cPickle
import os

# PyTorch
import torch


# Directories
# Results Directory
results = "results"
# Datasets Directories
datasets = [i for i in os.listdir(results) if not i.startswith('.')]
datasets = [i for i in datasets if not i.startswith('_')]
# print(datasets)

# Filenames
stat_data_fname = "stat_data.pickle"
test_results_fname = "test_results.pickle"


# Go through datasets folders
for dataset_idx, dataset_folder_name in enumerate(datasets):
    print(f"Dataset {dataset_idx+1}/{len(datasets)} | Name: {dataset_folder_name}")
    # Get the complete path of the stata data pickle file
    d_stat_data_path = os.path.join(results, dataset_folder_name, stat_data_fname)
    # Get the complete path of the test results pickle file
    d_test_results_path = os.path.join(results, dataset_folder_name, test_results_fname)

    # Open files
    # stat data pickle file
    with open(d_stat_data_path, "rb") as s:
        stat_data = cPickle.load(s)
    
    # Debug prints
    # print(f"Stats Data Values: {stat_data}")
    print(f"Stats Data Shape: {stat_data.shape}")

    # test results pickle file
    with open(d_test_results_path, "rb") as t:
        test_results = cPickle.load(t)
    
    # Debug prints
    print(f"Test Results Values: {test_results}")
    # print(f"Test Results Shape: {np.shape(test_results)}")