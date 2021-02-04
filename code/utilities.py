# Imports
# General
import numpy as np

# PyTorch
import torch 
import torch.nn as nn
import torchvision



# Class: Utilities with some important dictionaries that we use in the building of our solutions
class Utils:
    def __init__(self):
        self.inv_conv_activ_functions = ['none',
                                         'relu',
                                         'tanh']

        self.inv_conv_pooling_types = ['none',
                                       'max',
                                       'avg']

        self.inv_fc_activ_functions = ['none',
                                       'relu',
                                       'tanh']

        # Conv Network parameters
        self.conv_nr_filters = np.array([8, 16, 32, 64, 128, 256, 512])
        self.conv_kernel_size = np.array([1, 3, 5, 7, 9])
        self.conv_activ_functions = [nn.Identity(),
                                     nn.ReLU(),
                                     nn.Tanh()]
        self.conv_drop_out_range = np.array([0.0, 1.0])
        self.conv_pooling_types = [nn.Identity(),
                                   nn.MaxPool2d(2, 2),
                                   nn.AvgPool2d(2, 2)]

        # FC Network Parameters
        self.fc_nr_neurons_range = np.array([1, 100])
        self.fc_activ_functions = [nn.Identity(),
                                   nn.ReLU(),
                                   nn.Tanh()]
        self.fc_drop_out_range = np.array([0.0, 1.0])

        # Learning rate
        self.learning_rate = np.array([0.001, 0.0001, 0.00001])


# We create this object to automatically import to other functions 
utils = Utils()



# Function: Obtain Mean and Standard Deviation of the datasets to apply a proper normalisation
# You can run this function to check our mean and std values in the training scripts
def calculate_mean_std():
    # Compute Mean and STD on MNIST
    # Get data
    mnist_data = torchvision.datasets.MNIST('data/mnist', train=True, download=True)

    # Create and empty list to append all the images of the dataset
    l = []

    # Go through all the images
    for i in range(len(mnist_data)):
        l.append(mnist_data[i][0])

    # Convert into PyTorch Tensor
    l = torch.stack(l, dim=0)

    # Get Mean and STD
    print(f"MNIST Dataset Mean and STD: {torch.std_mean(l)}")

    ####

    # Compute Mean and STD on Fashion-MNIST
    # Get data
    fashion_mnist_data = torchvision.datasets.FashionMNIST('data/fashion-mnist', train=True, download=True)

    # Create and empty list to append all the images of the dataset
    l = []

    # Go through all the images
    for i in range(len(fashion_mnist_data)):
        l.append(fashion_mnist_data[i][0])

    # Convert into PyTorch Tensor
    l = torch.stack(l, dim=0)

    # Get Mean and STD
    print(f"Fashion-MNIST Dataset Mean and STD: {torch.std_mean(l)}")

    ####

    # Compute Mean and STD on CIFAR-10
    # Get data
    cifar_10_data = torchvision.datasets.CIFAR10('data/cifar10', train=True, download=True)

    # Create and empty list to append all the images of the dataset
    l = []

    # Go through all the images
    for i in range(len(cifar_10_data)):
        l.append(cifar_10_data[i][0])

    # Convert into PyTorch Tensor
    l = torch.stack(l, dim=0)

    # Get Mean and STD
    print(f"CIFAR-10 Dataset Red Channel Mean and STD: {torch.std_mean(l[:, 0])}")
    print(f"CIFAR-10 Dataset Green Channel Mean and STD: {torch.std_mean(l[:, 1])}")
    print(f"CIFAR-10 Dataset Green Channel Mean and STD: {torch.std_mean(l[:, 2])}")

    return 

# Uncomment if you want to run this function and check mean and std values per dataset
# calculate_mean_std()