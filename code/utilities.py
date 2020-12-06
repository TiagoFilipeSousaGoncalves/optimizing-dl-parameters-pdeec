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


        # Create Solution structure
        # Populate Convolutional Layers Block Matrix with Parameters
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
        self.fully_connected_layers = np.zeros(shape=(len(self.fc_neurons), 3), dtype='float')
        for fc_idx in range(len(self.fc_neurons)):
            # Each row is a layer!
            # Column 0: FC Layer Number of Neurons
            self.fully_connected_layers[fc_idx, 0] = self.fc_neurons[fc_idx]

            # Column 1: FC Layer Activation Functions
            self.fully_connected_layers[fc_idx, 1] = self.activ_functions[self.fc_activ_functions[fc_idx]]

            # Column 2: FC Dropout Rates
            self.fully_connected_layers[fc_idx, 2] = self.fc_drop_rates[fc_idx]
        

        # Add learning rate to the solution object
        self.learning_rate = learning_rate

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
        return self.final_built_solution.copy()



# Define the Model Class
class Model(nn.Module):
    def __init__(self, input_shape, number_of_labels, solution):
        super(Model, self).__init__()
        # Instance some important dictionaries to map numbers into parameters (PyTorch Layers)
        # Activation Functions
        self.activ_functions = dict()
        self.activ_functions[0] = nn.Identity()
        self.activ_functions[1] = nn.ReLU()
        self.activ_functions[2] = nn.Tanh()
        # Convolutional Pooling Types
        self.conv_pooling_types = dict()
        self.conv_pooling_types[0] = nn.Identity()
        self.conv_pooling_types[1] = nn.MaxPool2d(2, 2)
        self.conv_pooling_types[2] = nn.AvgPool2d(2, 2)

        # Instance some important dictionaries to inverse-map numbers into parameters names
        # Activation Functions
        self.inv_activ_functions = dict()
        self.inv_activ_functions[0] = 'none'
        self.inv_activ_functions[1] = 'relu'
        self.inv_activ_functions[2] = 'tanh'
        # Convolutional Pooling Types
        self.inv_pooling_types = dict()
        self.inv_pooling_types[0] = 'none'
        self.inv_pooling_types[1] = 'max'
        self.inv_pooling_types[2] = 'avg'


        # Process Input Shape
        self.rows = input_shape[0]
        self.columns = input_shape[1]
        self.channels = input_shape[2]
        

        # Create Convolutional Block
        self.convolutional_layers = OrderedDict()
        
        # Go through the convolutional block of the solution (index: 0)
        input_channels = self.channels
        for conv_idx, layer in enumerate(solution[0]):
            self.convolutional_layers[f'conv_{conv_idx}'] = nn.Conv2d(in_channels=input_channels, out_channels=int(layer[0].item()), kernel_size=int(layer[1].item()))
            self.convolutional_layers[f'conv_act_{self.inv_activ_functions[int(layer[2].item())]}{conv_idx}'] = self.activ_functions[int(layer[2].item())]
            self.convolutional_layers[f'conv_dropout{conv_idx}'] = nn.Dropout2d(layer[3].item())
            self.convolutional_layers[f'conv_pool_{self.inv_pooling_types[int(layer[4].item())]}{conv_idx}'] = self.conv_pooling_types[int(layer[4].item())]
            input_channels = int((layer[0].item()))
            
        
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

        # Go through the fully-connected block of the solution (index: 1)
        for fc_idx, layer in enumerate(solution[1]):
            self.fc_layers[f'fc_{fc_idx}'] = nn.Linear(in_features=input_features, out_features=int(layer[0].item()))
            self.fc_layers[f'fc_act_{self.inv_activ_functions[int(layer[1].item())]}{fc_idx}'] = self.activ_functions[int(layer[1].item())]
            self.fc_layers[f'fc_dropout{fc_idx}'] = nn.Dropout(layer[2])
            input_features = int(layer[0].item())
        
        # Now convert this into an fc layer
        self.fc_layers = nn.Sequential(self.fc_layers)
        
        # Last FC-layer
        self.fc_labels = nn.Linear(in_features=input_features, out_features=number_of_labels)

        # Add learning rate
        self.learning_rate = solution[2]




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
)

candidate_solution = solution.get_solution_matrix()
print(candidate_solution) """


""" model = Model(input_shape=[28, 28, 3], number_of_labels=10, solution=candidate_solution)
print(f"Learning Rate: {model.learning_rate}")
print(model.parameters)
tensor = torch.randn(1, 3, 28, 28)
out = model(tensor)
print(out)
summary(model, (3, 28, 28))
print(model) """