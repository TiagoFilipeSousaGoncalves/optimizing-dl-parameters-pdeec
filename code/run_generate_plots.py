# Imports
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cPickle
import os
from itertools import product

# PyTorch
import torch


# Directories
# Results Directory
results = "results"
# Datasets Directories
# datasets = [i for i in os.listdir(results) if not i.startswith('.')]
# datasets = [i for i in datasets if not i.startswith('_')]
# datasets = [i for i in datasets if not i.startswith('s')]
# datasets = [i for i in datasets if not i.startswith('e')]
datasets = ["mnist"]
datasets_max_phases = [2]
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


    # TODO: Review Plot 1 - Search Space Size vs Chromossome Lengths
    # Initial chromossome length
    chromossome_lengths = [2+i for i in range(4)]
    # Conv-Layer Params
    nr_conv_filters = 7
    nr_kernel_sizes = 5
    nr_conv_act_fns = 3
    nr_of_conv_dropout_probs = 1000
    nr_pool_types = 3
    nr_conv_params = nr_conv_filters * nr_kernel_sizes * nr_conv_act_fns * nr_of_conv_dropout_probs * nr_pool_types
    
    # FC-Layer Params
    nr_fc_neurons = 100
    nr_fc_act_fns = 3
    nr_of_fc_dropout_probs = 3
    nr_fc_params = nr_fc_neurons * nr_fc_act_fns * nr_of_fc_dropout_probs

    # Learning Rate
    nr_of_lrates = 3

    # Create lists to append search space sizes and chromossome legths
    sss_cl_lists = list()

    # Compute the search space sizes as function of the chromossome length 
    for c_len in chromossome_lengths:
        # Get the permutations of the types of architectures 
        c_prod = product(range(0, c_len+1), repeat=2)
        # Debug print
        # print(list(c_prod))

        # Exclude the permutations in which the sum of layers is different from c_len
        valid_perm = list()
        for p in list(c_prod):
            # print(p[0], p[1])
            if p[0] + p[1] == c_len:
                valid_perm.append([p[0], p[1]])
        
        # Debug print
        # print(valid_perm)
        

        # Compute the size of search spaces per permutation
        search_space_size = 0
        # Go through the valid permutations
        for p_val in valid_perm:
            p_sss = ((p_val[0] * nr_conv_params) + (p_val[1] * nr_fc_params)) * nr_of_lrates
            search_space_size += p_sss
            # print(f"{search_space_size}")
        
        # Append search space size per c_len
        sss_cl_lists.append(search_space_size)
    
    # Generate plot with best individual fitnesse per phase
    # Plot Title
    plt.title(f"Search Space Size per Chromosome Length")
    # Plot Axis Labels
    plt.xlabel("Chromossome Length")
    plt.ylabel("Search Space Size")
    # Plot Axis Limits
    plt.xticks(chromossome_lengths)
    # Generate plot
    plt.plot(chromossome_lengths, sss_cl_lists)
    plt.savefig(fname=os.path.join(results, "ssize_clen.png"))
    plt.show()



    
    # TODO: Review Plot 2 - Best Individual Fitnesses per Phase
    # List for phases
    phases = [i for i in range(datasets_max_phases[dataset_idx])]
    
    # List for best individual fitnesses per phase
    best_individual_fitnesses = list()
    # Go through all phases
    for ph in range(datasets_max_phases[dataset_idx]):
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
    plt.title(f"Best Individual Fitnesses per Phase | Dataset: {dataset_folder_name.upper()}")
    # Plot Axis Labels
    plt.xlabel("Phase")
    plt.ylabel("Fitness")
    # Plot Axis Limits
    plt.xticks(phases)
    # Generate plot
    plt.plot(phases, best_individual_fitnesses, 'o')
    plt.savefig(fname=os.path.join(results, dataset_folder_name, f"best_ind_fit_phase.png"))
    plt.show()


    # TODO: Review Plot 3 - Best Individual Accuracy per Phase
    # List for best individual accuracies per phase
    best_individual_accuracy = list()
    # Go through all phases
    for ph in range(datasets_max_phases[dataset_idx]):
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
    plt.title(f"Best Individual Accuracies per Phase | Dataset: {dataset_folder_name.upper()}")
    # Plot Axis Labels
    plt.xlabel("Phase")
    plt.ylabel("Accuracy")
    # Plot Axis Limits
    plt.xticks(phases)
    # Generate plot
    plt.plot(phases, best_individual_accuracy, 'o')
    plt.savefig(fname=os.path.join(results, dataset_folder_name, f"best_ind_acc_phase.png"))
    plt.show()


    # TODO: Review Plot 4 - Individual Fitness per Generation per Phase
    # Create list to append lists of individual fitnesses
    ind_ph_fitnesses = list()
    # Go through all phases
    for ph in range(datasets_max_phases[dataset_idx]):
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
    plt.title(f"Individual Fitnesses per Generation per Phase | Dataset: {dataset_folder_name.upper()}")
    # Plot Axis Labels
    plt.xlabel("Phase")
    plt.ylabel("Individuals and Fitnesses")
    # Plot Axis Limits
    plt.xticks(phases)
    # Generate scatter plot 
    for idx in range(len(phases)):
        plt.scatter(ind_ph_fitnesses[idx][0], ind_ph_fitnesses[idx][1])
    plt.savefig(fname=os.path.join(results, dataset_folder_name, f"ind_fit_per_gen_per_phase.png"))
    plt.show()



    # TODO: Review Plot 5 - Distribution of Individuals and Fitnesses
    # Create list for all the individual fitnesses
    ind_fitnesses = list()
    # Go through all phases
    for ph in range(datasets_max_phases[dataset_idx]):
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
    plt.title(f"Distribution of Individuals and Fitnesses | Dataset: {dataset_folder_name.upper()}")
    # Plot Axis Labels
    plt.xlabel("Number of Individuals")
    plt.ylabel("Individual Fitnesses")
    # Generate scatter plot
    plt.scatter([i for i in range(len(ind_fitnesses))], ind_fitnesses)
    plt.savefig(fname=os.path.join(results, dataset_folder_name, f"distribution_ind_fit.png"))
    plt.show()



    # TODO: Review Plot 6 - Accuracy per Generation per Phase
    # Go through phases
    for phase in range(datasets_max_phases[dataset_idx]):
        # Create temporary axis lists to append the results per phase
        generations = []
        fitnesses = []
        for gen in range(stat_data.shape[1]):
            for p_ind in range(stat_data.shape[2]):
                generations.append(gen)
                fitnesses.append(stat_data[phase][gen][p_ind][0])
            
        # Generate plot with distribution of individuals and fitnesses
        # Plot Title
        plt.title(f"Distribution of Fitnesses along the Generation | Phase {phase} | Dataset: {dataset_folder_name.upper()}")
        # Plot Axis Labels
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        # Generate scatter plot
        plt.scatter(generations, fitnesses)
        plt.savefig(fname=os.path.join(results, dataset_folder_name, f"ind_fit_per_gen_single_phase_{phase}.png"))
        plt.show()
        

# Finish statement
print(f"Finished.")