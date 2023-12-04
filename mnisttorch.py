import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST
import train_test
import json


def read_param():
    with open("config.json", 'r') as f:
        result = json.load(f)
        batch_size = result['grad_parameters']['batch_size']
        epochs = result['grad_parameters']['epochs']
        learning_rate = result['grad_parameters']['learning_rate']
    return batch_size, epochs, learning_rate


def main():
    train_dataset = torchvision.datasets.MNIST(root='D:\PyProjects\Peceptron\Perceptron\dataset', train=True,
                                               transform=torchvision.transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='D:\PyProjects\Peceptron\Perceptron\dataset', train=False,
                                              transform=torchvision.transforms.ToTensor(),
                                              download=True)

    batch_size, _, _ = read_param()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    train_test.train(train_loader)
    train_test.test(test_loader)


if __name__ == '__main__':
    main()
