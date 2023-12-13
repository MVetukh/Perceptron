import Network
import torch
import matplotlib.pyplot as plt
import utiles
from tqdm import tqdm

model = Network.Network()


def train(train_loader):
    _, epochs, _ = utiles.read_param()
    _, _, learning_rate = utiles.read_param()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    losses = []
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(tqdm(train_loader)):
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

    # Additional information

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'D:\PyProjects\Peceptron\Perceptron\checkpoint')
