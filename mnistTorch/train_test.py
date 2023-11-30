import Network
import torch
import mnisttorch
import matplotlib.pyplot as plt

model = Network.Network()
def train(train_loader):

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    epochs = 10
    losses = []

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):

            outputs = model(images)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def test(test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Точность на тестовом наборе данных: {} %'.format(100 * correct / total))
