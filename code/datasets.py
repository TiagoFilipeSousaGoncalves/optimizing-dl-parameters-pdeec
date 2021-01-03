# Imports
import torch
import torchvision
from torchvision import transforms

# MNIST Data Loader
def get_mnist_loader(batch_size, train=True):

    # Mean and STD
    mnist_mean = [0.1307]
    mnist_std = [0.3081]

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mnist_mean,
                                                                                      std=mnist_std)])

    mnist_data = torchvision.datasets.MNIST('data/mnist', train=train, download=True, transform=train_transform)

    data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader


# Fashion MNIST Data Loader
def get_fashion_mnist_loader(batch_size, train=True):

    # Mean and STD
    fashion_mnist_mean = [0.2860]
    fashion_mnist_std = [0.3530]

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=fashion_mnist_mean,
                                                                                      std=fashion_mnist_std)])

    fashion_mnist_data = torchvision.datasets.FashionMNIST('data/fashion_mnist', train=train, download=True, transform=train_transform)

    data_loader = torch.utils.data.DataLoader(fashion_mnist_data, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader


# CIFAR10 Data Loader
def get_cifar10_loader(batch_size, train=True):

    # Mean and STD
    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std = [0.2470, 0.2435, 0.2616]

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cifar10_mean,
                                                                                      std=cifar10_std)])

    cifar10_data = torchvision.datasets.CIFAR10('data/cifar10', train=train, download=True, transform=train_transform)

    data_loader = torch.utils.data.DataLoader(cifar10_data, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader