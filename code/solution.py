# Imports
import torch
from itertools import combinations
import numpy as np


# Define the Solution Class
# Change Class to a function that generates new solution that outputs the final array 
class Solution:
    def __init__(self, conv_filters, conv_kernel_sizes, conv_activ_functions, conv_drop_rates, conv_pool_types,
                 fc_neurons, fc_activ_functions, fc_drop_rates, learning_rate):
        # Assert conv list sizes and fc list sizes
        # Convolutional Block Sizes
        conv_block_params_comb = combinations(
            [conv_filters, conv_kernel_sizes, conv_activ_functions, conv_drop_rates, conv_pool_types], 2)
        # Go through each combination
        for comb in list(conv_block_params_comb):
            assert len(comb[0]) == len(
                comb[1]), "The length of all the lists of convolutional layer parameters must match."

        # Fully-Connected Block Sizes
        fc_block_params_comb = combinations([fc_neurons, fc_activ_functions, fc_drop_rates], 2)
        # Go through each combination
        for comb in list(fc_block_params_comb):
            assert len(comb[0]) == len(
                comb[1]), "The length of all the lists of fully-connected layers parameters must match."

        # Dictionaries to map several parameters into numbers
        # Instance some important dictionaries
        # Activation Functions
        activ_functions = dict()
        activ_functions['none'] = 0
        activ_functions['relu'] = 1
        activ_functions['tanh'] = 2

        # Convolutional Pooling Types
        pooling_types = dict()
        pooling_types['none'] = 0
        pooling_types['max'] = 1
        pooling_types['avg'] = 2

        # Convolutional Layers Parameters
        conv_filters = conv_filters
        conv_kernel_sizes = conv_kernel_sizes
        conv_activ_functions = conv_activ_functions
        conv_drop_rates = conv_drop_rates
        conv_pool_types = conv_pool_types

        # Fully-Connected Layers Parameters
        fc_neurons = fc_neurons
        fc_activ_functions = fc_activ_functions
        fc_drop_rates = fc_drop_rates


        # Create Solution structure
        # Populate Convolutional Layers Block Matrix with Parameters
        convolutional_layers = np.array(list())
        if len(conv_filters) > 0:
            convolutional_layers = np.zeros(shape=(len(conv_filters), 5), dtype='float')
            for conv_idx in range(len(conv_filters)):
                # Each row is a layer!
                # Column 0: Number of Conv-Filters
                convolutional_layers[conv_idx, 0] = conv_filters[conv_idx]

                # Column 1: Conv-Kernel Sizes
                convolutional_layers[conv_idx, 1] = conv_kernel_sizes[conv_idx]

                # Column 2: Conv-Activation Functions
                convolutional_layers[conv_idx, 2] = activ_functions[conv_activ_functions[conv_idx]]

                # Column 3: Conv-Drop Rates
                convolutional_layers[conv_idx, 3] = conv_drop_rates[conv_idx]

                # Column 4: Conv-Pool Layer Types
                convolutional_layers[conv_idx, 4] = pooling_types[conv_pool_types[conv_idx]]

        # Populate Fully-Connected Layers Block Matrix with Parameters
        fully_connected_layers = np.array(list())
        if len(fc_neurons) > 0:
            fully_connected_layers = np.zeros(shape=(len(fc_neurons), 3), dtype='float')
            for fc_idx in range(len(fc_neurons)):
                # Each row is a layer!
                # Column 0: FC Layer Number of Neurons
                fully_connected_layers[fc_idx, 0] = fc_neurons[fc_idx]

                # Column 1: FC Layer Activation Functions
                fully_connected_layers[fc_idx, 1] = activ_functions[fc_activ_functions[fc_idx]]

                # Column 2: FC Dropout Rates
                fully_connected_layers[fc_idx, 2] = fc_drop_rates[fc_idx]


        # Convert convolutional and fully-connected layers' parameters matrices to torch.Tensor()
        # Convolutional Layers
        convolutional_layers = torch.from_numpy(convolutional_layers)
        convolutional_layers = convolutional_layers.detach()
        # Fully-Connected Layers
        fully_connected_layers = torch.from_numpy(fully_connected_layers)
        fully_connected_layers = fully_connected_layers.detach()

        # Converto learning rate into a tensor
        learning_rate = torch.tensor([learning_rate]).detach()

        # Alter solution structure and save it into a class variable
        self.final_built_solution = [convolutional_layers, fully_connected_layers, learning_rate]


    # Function to return the solution matrix
    def get_solution_matrix(self):
        return [torch.clone(i).detach() for i in self.final_built_solution]