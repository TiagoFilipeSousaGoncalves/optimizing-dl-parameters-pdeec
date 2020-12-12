import torch
import torchvision
from torchvision import transforms

'''
mnist_mean = [0.1307]
mnist_std = [0.3081]

fashion_mnist_mean = [0.2860]
fashion_mnist_std = [0.3530]

cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2470, 0.2435, 0.2616]
'''


def get_mnist_loader(batch_size):
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1307],
                                                                                      std=[0.3081])])

    mnist_data = torchvision.datasets.MNIST('data/mnist', train=True, download=True, transform=train_transform)

    data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader

