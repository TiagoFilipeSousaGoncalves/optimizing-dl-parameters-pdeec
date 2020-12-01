# Imports
import os
import numpy as np 


# PyTorch Imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Define the Solution Class
class Solution():
    def __init__(self, conv_filters, conv_kernel_sizes, conv_activ_functions, conv_drop_rates, conv_pool_types, fc_neurons, fc_activ_functions, fc_drop_rates):
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
    
        # Create solution matrix
        self.number_of_layers = len(self.conv_filters) + len(self.fc_neurons)
        self.solution_matrix = np.zeros(shape=(self.number_of_layers, 9), dtype='object')

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
    

    # Function to return the solution matrix
    def get_solution_matrix(self):
        return self.solution_matrix.copy()



# Define the Model Class
"""
class Model(nn.Module):
    def __init__(self, input_shape, number_of_labels, solution):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28*28, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
"""

# Test
solution = Solution(
    conv_filters=[8, 16],
    conv_kernel_sizes=[1, 2],
    conv_activ_functions=['ReLU', 'Tanh'],
    conv_drop_rates=[0.2, 0.5],
    conv_pool_types=["Max", "Avg"],
    fc_neurons=[100, 128],
    fc_activ_functions=['ReLU', 'Softmax'],
    fc_drop_rates=[0.0, 1.0]
)

candidate_solution = solution.get_solution_matrix()
print(candidate_solution)