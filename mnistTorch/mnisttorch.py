import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.optim as optim

class Network(torch.nn.Modules):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = torch.nn.Linear(784, 34)
        self.fc2 = torch.nn.Sigmoid(34, 10)

    def forward(self, input):
        output = torch.sigmoid(self.fc1(input))
        output = self.fc2(output)
        return output


