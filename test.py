import Network
import torch


model = Network.Network()

def test(test_loader):
    checkpoint = torch.load('D:\PyProjects\Peceptron\Perceptron\checkpoint')
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

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

