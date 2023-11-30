import torch
import torch.nn

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 64)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(64, 10)
    def forward(self, input):
        input = input.view(-1, 784)
        layer1 = self.fc1(input)
        layer1 = self.relu1(layer1)
        layer2 = self.fc2(layer1)
        output = self.relu2(layer2)
        output = self.fc3(output)
        return output




