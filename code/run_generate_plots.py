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

    # TODO: Plot 1 - Search Space Size vs Nr of Layers
    
    # TODO: Review Plot 2 - Best Individual Fitnesses per Phase
    # List for phases
    phases = [i for i in range(stat_data.shape[0])]
    
    # List for best individual fitnesses per phase
    best_individual_fitnesses = list()
    # Go through all phases
    for ph in range(stat_data.shape[0]):
        # Create a list for all phase fitnesses
        phase_fitnesses = list()
        # Go through all generations
        for gen in range(stat_data.shape[1]):
            # Go through all the individuals of the population
            for p_ind in range(stat_data.shape[2]):
                # Append phase fitnesses
                phase_fitnesses.append(stat_data[ph][gen][p_ind][2])
        
        # Append the best individual fitness
        best_individual_fitnesses.append(max(phase_fitnesses))
    
    # Generate plot with best individual fitnesse per phase
    # Plot Title
    plt.title(f"Best Individual Fitnesses Per Phase | Dataset: {dataset_folder_name}")
    # Plot Axis Labels
    plt.xlabel("Phase")
    plt.ylabel("Fitness")
    # Plot Axis Limits
    plt.xticks(phases)
    # Generate plot
    plt.plot(phases, best_individual_fitnesses)
    plt.show()


    # TODO: Review Plot 3 - Best Individual Accuracy per Phase
    # List for best individual accuracies per phase
    best_individual_accuracy = list()
    # Go through all phases
    for ph in range(stat_data.shape[0]):
        # Create a list for all phase accuracies
        phase_accs = list()
        # Go through all generations
        for gen in range(stat_data.shape[1]):
            # Go through all the individuals of the population
            for p_ind in range(stat_data.shape[2]):
                # Append phase accuracies
                phase_accs.append(stat_data[ph][gen][p_ind][0])
        
        # Append the best individual accuracy
        best_individual_accuracy.append(max(phase_accs))
    
    # Generate plot with best individual accuracy per phase
    # Plot Title
    plt.title(f"Best Individual Accuracies Per Phase | Dataset: {dataset_folder_name}")
    # Plot Axis Labels
    plt.xlabel("Phase")
    plt.ylabel("Accuracy")
    # Plot Axis Limits
    plt.xticks(phases)
    # Generate plot
    plt.plot(phases, best_individual_accuracy)
    plt.show()


    # TODO: Review Plot 4 - Individual Fitness per Generation per Phase
    # Create list to append lists of individual fitnesses
    ind_ph_fitnesses = list()
    # Go through all phases
    for ph in range(stat_data.shape[0]):
        # Create a list for all individual fitnesses
        phase_ind_fit = list()
        # Go through all generations
        for gen in range(stat_data.shape[1]):
            # Go through all the individuals of the population
            for p_ind in range(stat_data.shape[2]):
                # Append phase individual fitnesses
                phase_ind_fit.append(stat_data[ph][gen][p_ind][2])
        
        # Append their phases and their respective individual fitnesses
        ind_ph_fitnesses.append([[ph for _ in range(len(phase_ind_fit))], phase_ind_fit])

    # Generate plot with individual fitness per generation per phase
    # Plot Title
    plt.title(f"Individual Fitnesses per Generation per Phase | Dataset: {dataset_folder_name}")
    # Plot Axis Labels
    plt.xlabel("Phase")
    plt.ylabel("Individuals and Fitnesses")
    # Plot Axis Limits
    plt.xticks(phases)
    # Generate scatter plot 
    for idx in range(len(phases)):
        plt.scatter(ind_ph_fitnesses[idx][0], ind_ph_fitnesses[idx][1])
    plt.show()



    # TODO: Review Plot 5 - Distribution of Individuals and Fitnesses
    # Create list for all the individual fitnesses
    ind_fitnesses = list()
    # Go through all phases
    for ph in range(stat_data.shape[0]):
        # Create a list for all individual phase fitnesses
        phase_ind_fit = list()
        # Go through all generations
        for gen in range(stat_data.shape[1]):
            # Go through all the individuals of the population
            for p_ind in range(stat_data.shape[2]):
                # Append individual phase fitnesses
                phase_ind_fit.append(stat_data[ph][gen][p_ind][2])
        
        # Concatenate the lists of individual phase fitnesses
        ind_fitnesses += phase_ind_fit
        
    # Generate plot with distribution of individuals and fitnesses
    # Plot Title
    plt.title(f"Distribution of Individuals and Fitnesses | Dataset: {dataset_folder_name}")
    # Plot Axis Labels
    plt.xlabel("Number of Individuals")
    plt.ylabel("Individual Fitnesses")
    # Generate scatter plot
    plt.scatter([i for i in range(len(ind_fitnesses))], ind_fitnesses)
    plt.show()