import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math_func

data = np.arange(-10, 10, 1)
target = math_func.quadratic(data)

data = torch.tensor(data, dtype=torch.float).view(-1, 1)
target = torch.tensor(target, dtype=torch.float).view(-1, 1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    net = Net()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    losses = []
    num_epochs = 2000

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(data.numpy(), target.numpy(), label='True Function')
    plt.plot(data.numpy(), net(data).detach().numpy(), label='Predicted Function')
    plt.xlabel('Data')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    data = np.linspace(-10, 10, 100)
