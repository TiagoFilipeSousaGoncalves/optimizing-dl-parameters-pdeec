import torch
from itertools import combinations
import numpy as np
import copy


# Define the Solution Class
# TODO: Change Class to a function that generates new solution that outputs the final array 
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
        self.activ_functions = dict()
        self.activ_functions['none'] = 0
        self.activ_functions['relu'] = 1
        self.activ_functions['tanh'] = 2

        # Convolutional Pooling Types
        self.pooling_types = dict()
        self.pooling_types['none'] = 0
        self.pooling_types['max'] = 1
        self.pooling_types['avg'] = 2

        # Convolutional Layers Parameters
        self.conv_filters = conv_filters
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_activ_functions = conv_activ_functions
        self.conv_drop_rates = conv_drop_rates
        self.conv_pool_types = conv_pool_types

        # Fully-Connected Layers Parameters
        self.fc_neurons = fc_neurons
        self.fc_activ_functions = fc_activ_functions
        self.fc_drop_rates = fc_drop_rates

        # Learning Rate
        self.learning_rate = learning_rate

    # TODO: Function to (re)build solution structure (useful to handle mutations and crossovers)
    def build_solution(self):
        # Create Solution structure
        # Populate Convolutional Layers Block Matrix with Parameters
        self.convolutional_layers = np.array(list())
        if len(self.conv_filters) > 0:
            self.convolutional_layers = np.zeros(shape=(len(self.conv_filters), 5), dtype='float')
            for conv_idx in range(len(self.conv_filters)):
                # Each row is a layer!
                # Column 0: Number of Conv-Filters
                self.convolutional_layers[conv_idx, 0] = self.conv_filters[conv_idx]

                # Column 1: Conv-Kernel Sizes
                self.convolutional_layers[conv_idx, 1] = self.conv_kernel_sizes[conv_idx]

                # Column 2: Conv-Activation Functions
                self.convolutional_layers[conv_idx, 2] = self.activ_functions[self.conv_activ_functions[conv_idx]]

                # Column 3: Conv-Drop Rates
                self.convolutional_layers[conv_idx, 3] = self.conv_drop_rates[conv_idx]

                # Column 4: Conv-Pool Layer Types
                self.convolutional_layers[conv_idx, 4] = self.pooling_types[self.conv_pool_types[conv_idx]]

        # Populate Fully-Connected Layers Block Matrix with Parameters
        self.fully_connected_layers = np.array(list())
        if len(self.fc_neurons) > 0:
            self.fully_connected_layers = np.zeros(shape=(len(self.fc_neurons), 3), dtype='float')
            for fc_idx in range(len(self.fc_neurons)):
                # Each row is a layer!
                # Column 0: FC Layer Number of Neurons
                self.fully_connected_layers[fc_idx, 0] = self.fc_neurons[fc_idx]

                # Column 1: FC Layer Activation Functions
                self.fully_connected_layers[fc_idx, 1] = self.activ_functions[self.fc_activ_functions[fc_idx]]

                # Column 2: FC Dropout Rates
                self.fully_connected_layers[fc_idx, 2] = self.fc_drop_rates[fc_idx]


        # Convert convolutional and fully-connected layers' parameters matrices to torch.Tensor()
        # Convolutional Layers
        self.convolutional_layers = torch.from_numpy(self.convolutional_layers)
        self.convolutional_layers = self.convolutional_layers.detach()
        # Fully-Connected Layers
        self.fully_connected_layers = torch.from_numpy(self.fully_connected_layers)
        self.fully_connected_layers = self.fully_connected_layers.detach()

        # Alter solution structure to:
        self.final_built_solution = [self.convolutional_layers, self.fully_connected_layers, self.learning_rate]

    # Function to return the solution matrix
    def get_solution_matrix(self):
        return copy.deepcopy(self.final_built_solution)