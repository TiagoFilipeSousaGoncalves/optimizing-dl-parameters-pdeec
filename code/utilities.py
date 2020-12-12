# Imports
import torch.nn as nn
import numpy as np


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


utils = Utils()

"""
# Test
solution = Solution(
    conv_filters=[8, 16],
    # conv_filters=[],
    conv_kernel_sizes=[1, 2],
    # conv_kernel_sizes=[],
    conv_activ_functions=['relu', 'tanh'],
    # conv_activ_functions=[],
    conv_drop_rates=[0.2, 0.5],
    # conv_drop_rates=[],
    conv_pool_types=["max", "avg"],
    # conv_pool_types=[],
    fc_neurons=[100, 128],
    # fc_neurons=[],
    fc_activ_functions=['relu', 'tanh'],
    # fc_activ_functions=[],
    fc_drop_rates=[0.0, 1.0],
    # fc_drop_rates=[],
    learning_rate=0.001
)

candidate_solution = solution.get_solution_matrix()
print(candidate_solution, len(candidate_solution))
"""

"""
model = Model(input_shape=[28, 28, 3], number_of_labels=10, solution=candidate_solution)
print(f"Learning Rate: {model.learning_rate}")
print(model.parameters)
tensor = torch.randn(1, 3, 28, 28)
out = model(tensor)
print(out)
summary(model, (3, 28, 28))
print(model)'''

'''
# data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=4, shuffle=True, num_workers=4)


def calculate_mean_std():
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1307],
                                                                                        std=[0.3081])])

    mnist_data = torchvision.datasets.MNIST('data/mnist', train=True, download=True, transform=train_transform)

    l = []

    for i in range(len(mnist_data)):
        l.append(mnist_data[i][0])

    l = torch.stack(l, dim=0)

    print(torch.std_mean(l))

    ####
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860],
                                                                                      std=[0.3530])])

    fashion_mnist_data = torchvision.datasets.FashionMNIST('data/fashion_mnist', train=True, download=True, transform=train_transform)

    l = []

    for i in range(len(fashion_mnist_data)):
        l.append(fashion_mnist_data[i][0])

    l = torch.stack(l, dim=0)

    print(torch.std_mean(l))

    ####
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                                      std=[0.2470, 0.2435, 0.2616])])

    cifar_10_data = torchvision.datasets.CIFAR10('data/cifar10', train=True, download=True, transform=train_transform)

    l = []

    for i in range(len(cifar_10_data)):
        l.append(cifar_10_data[i][0])

    l = torch.stack(l, dim=0)

    print(torch.std_mean(l[:, 0]))
    print(torch.std_mean(l[:, 1]))
    print(torch.std_mean(l[:, 2]))


calculate_mean_std()
'''

"""
"""
if __name__ == '__main__':
    ga = GeneticAlgorithm(10, 10, 0.5, 0.5, 0.5, 0, 1, 8)

    ga.train()
"""