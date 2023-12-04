import Network
import torch
import matplotlib.pyplot as plt
import mnisttorch

model = Network.Network()


# def save_results(file_path, results):
#     with open(file_path, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Image Path', 'Label'])
#         writer.writerows(results)
#

def train(train_loader):
    _, epochs, _ = mnisttorch.read_param()
    _, _, learning_rate = mnisttorch.read_param()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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

    model.save_results('D:\PyProjects\Peceptron\Perceptron\dataset\Result.csv', test_loader)

    # save_results('D:\PyProjects\Peceptron\dataset\Results.csv', results)
