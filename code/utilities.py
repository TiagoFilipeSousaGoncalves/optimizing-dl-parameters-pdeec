# Imports
import os
import numpy as np 
from collections import OrderedDict
from itertools import combinations

# PyTorch Imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Torch Summary
from torchsummary import summary


# Define the Solution Class
class Solution():
    def __init__(self, conv_filters, conv_kernel_sizes, conv_activ_functions, conv_drop_rates, conv_pool_types, fc_neurons, fc_activ_functions, fc_drop_rates, learning_rate):
        # Assert conv list sizes and fc list sizes
        # Convolutional Block Sizes
        conv_block_params_comb = combinations([conv_filters, conv_kernel_sizes, conv_activ_functions, conv_drop_rates, conv_pool_types], 2)
        # Go through each combination
        for comb in list(conv_block_params_comb):
            assert len(comb[0]) == len(comb[1]), "The length of all the lists of convolutional layer parameters must match."
        
        # Fully-Connected Block Sizes
        fc_block_params_comb = combinations([fc_neurons, fc_activ_functions, fc_drop_rates], 2)
        # Go through each combination
        for comb in list(fc_block_params_comb):
            assert len(comb[0]) == len(comb[1]), "The length of all the lists of fully-connected layers parameters must match."

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
    
        # Create solution matrix
        self.number_of_layers = len(self.conv_filters) + len(self.fc_neurons)
        self.solution_matrix = np.zeros(shape=(self.number_of_layers, 9+1), dtype='object')

        # Populate Solution Matrix with Parameters
        # Convolutional Layers
        for conv_idx in range(len(self.conv_filters)):
            # Each row is a layer
            # Column 0: Binary (0, 1), indicating if it is a Conv (1) or FC (0) layers
            self.solution_matrix[conv_idx, 0] = 1
            
            # Column 1: Number of Conv-Filters
            self.solution_matrix[conv_idx, 1] = self.conv_filters[conv_idx]

            # Column 2: Conv-Kernel Sizes
            self.solution_matrix[conv_idx, 2] = self.conv_kernel_sizes[conv_idx]

            # Column 3: Conv-Activation Functions
            self.solution_matrix[conv_idx, 3] = self.conv_activ_functions[conv_idx]

            # Column 4: Conv-Drop Rates
            self.solution_matrix[conv_idx, 4] = self.conv_drop_rates[conv_idx]

            # Column 5: Conv-Pool Layer Types
            self.solution_matrix[conv_idx, 5] = self.conv_pool_types[conv_idx]
        
        # Fully-Connected Layers
        for fc_idx in range(len(self.fc_neurons)):
            # We have to go to the row after the conv-layers
            row_idx = len(self.conv_filters) + fc_idx

            # Column 6: FC Layer Number of Neurons
            self.solution_matrix[row_idx, 6] = self.fc_neurons[fc_idx]

            # Column 7: FC Layer Activation Functions
            self.solution_matrix[row_idx, 7] = self.fc_activ_functions[fc_idx]

            # Column 8: FC Dropout Rates
            self.solution_matrix[row_idx, 8] = self.fc_drop_rates[fc_idx]
        

        # TODO: Add learning rate to last column

        # TODO: Convert solution matrix to torch.Tensor()

        # TODO: Alter solution structure to:
        # self.final_built_solution = [self.convolutional_layers, self.fully_connected_layers, self.learning_rate]

    # Function to return the solution matrix
    def get_solution_matrix(self):
        return self.solution_matrix.copy()



# Define the Model Class
class Model(nn.Module):
    def __init__(self, input_shape, number_of_labels, solution):
        super(Model, self).__init__()
        # Instance some important dictionaries
        # Activation Functions
        self.activ_functions = dict()
        self.activ_functions['none'] = nn.Identity()
        self.activ_functions['relu'] = nn.ReLU()
        self.activ_functions['tanh'] = nn.Tanh()

        # Convolutional Pooling Types
        self.conv_pooling_types = dict()
        self.conv_pooling_types['none'] = nn.Identity()
        self.conv_pooling_types['max'] = nn.MaxPool2d(2, 2)
        self.conv_pooling_types['avg'] = nn.AvgPool2d(2, 2)


        # Process Input Shape
        self.rows = input_shape[0]
        self.columns = input_shape[1]
        self.channels = input_shape[2]
        

        # Create Convolutional Block
        self.convolutional_layers = OrderedDict()
        
        # Go through the solution
        input_channels = self.channels
        for conv_idx, layer in enumerate(solution):
            # Check if Column 0 == 1; if yes, it is a Conv-Layer
            if layer[0] == 1:
                self.convolutional_layers[f'conv_{conv_idx}'] = nn.Conv2d(in_channels=input_channels, out_channels=layer[1], kernel_size=layer[2])
                self.convolutional_layers[f'conv_{layer[3]}{conv_idx}'] = self.activ_functions[layer[3]]
                self.convolutional_layers[f'conv_dropout{conv_idx}'] = nn.Dropout2d(layer[4])
                self.convolutional_layers[f'conv_pool_{layer[5]}{conv_idx}'] = self.conv_pooling_types[layer[5]]
                input_channels = layer[1]
            
        
        # Convert into a conv-layer
        self.convolutional_layers = nn.Sequential(self.convolutional_layers)

        
        # Now, we have to compute the linear dimensions for the first FC Layer
        aux_tensor = torch.randn(1, self.channels, self.rows, self.columns)
        aux_tensor = self.convolutional_layers(aux_tensor)
        
        # Input features
        input_features = aux_tensor.size(0) * aux_tensor.size(1) * aux_tensor.size(2) * aux_tensor.size(3)
        # print(input_features)

        # del intermediate variables
        del aux_tensor

        # Now, we build the FC-Block
        self.fc_layers = OrderedDict()

        # Go through FC-Layers
        for fc_idx, layer in enumerate(solution):
            if layer[0] == 0:
                self.fc_layers[f'fc_{fc_idx}'] = nn.Linear(in_features=input_features, out_features=layer[6])
                self.fc_layers[f'fc_{layer[7]}{fc_idx}'] = self.activ_functions[layer[7]]
                self.fc_layers[f'fc_dropout{fc_idx}'] = nn.Dropout(layer[8])
                input_features = layer[6]
        
        # Now convert this into an fc layer
        self.fc_layers = nn.Sequential(self.fc_layers)
        
        # Last FC-layer
        self.fc_labels = nn.Linear(in_features=input_features, out_features=number_of_labels)

        # TODO: Add learning rate




    def forward(self, x):
        # Go through convolutional block
        x = self.convolutional_layers(x)
        # print(x.size())
        
        # Flatten
        x = x.view(x.size(0), -1)
        # print(x.size())
        
        # Go through fc block
        x = self.fc_layers(x)
        # print(x.size())

        # Apply last FC layer
        x = self.fc_labels(x)
        # print(x.size())

        return x



# Define the Genetic Algorithm Class
class GeneticAlgorithm():
    def __init__(self, size_of_population, nr_of_generations, mutation_rate, percentage_of_best_fit, survival_rate_of_less_fit):
        # Initialize variables
        self.size_of_population = size_of_population
        self.nr_of_generations = nr_of_generations
        self.mutation_rate = mutation_rate
        self.percentage_of_best_fit = percentage_of_best_fit
        self.survival_rate_of_less_fit = survival_rate_of_less_fit
    
    # TODO: Normalize Data (compute data mean and std manually):
    def normalize_data(self):
        pass

    # TODO: Training Method
    def train(self):
        pass
    
    
    # TODO: Thread Training Solution
    def thread_training(self):
        pass

    
    # TODO: Transfer learning
    def transfer_learning(self):
        pass

    # TODO: Mutation Method
    def apply_mutation(self):
        # TODO: Randomly change parameters inside the solution
        pass


    # TODO: Crossover Method
    def apply_crossover(self):
        # TODO: Crossover between solution (random layers to hybrid); pay attention to the number of conv layers and fc layers of mum and dad
        pass


    # TODO: Fitness Function
    def solution_fitness(self):
        # TODO: Acc + Loss + Number of Epochs Until Convergence
        pass


# Test
""" solution = Solution(
    conv_filters=[8, 16],
    conv_kernel_sizes=[1, 2],
    conv_activ_functions=['relu', 'tanh'],
    conv_drop_rates=[0.2, 0.5],
    conv_pool_types=["max", "avg"],
    fc_neurons=[100, 128],
    fc_activ_functions=['relu', 'tanh'],
    fc_drop_rates=[0.0, 1.0],
    learning_rate=0.001
) """
"""
candidate_solution = solution.get_solution_matrix()
print(candidate_solution)

model = Model(input_shape=[28, 28, 3], number_of_labels=10, solution=candidate_solution)
print(model.parameters)
tensor = torch.randn(1, 3, 28, 28)
out = model(tensor)
print(out)
# summary(model, (3, 28, 28))
print(model) """