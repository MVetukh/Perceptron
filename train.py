import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.transforms import ToTensor, Compose, RandomRotation, RandomAffine, Normalize
from tqdm import tqdm
import mnist_dataloader
import network
import utiles

def validate(model, val_loader, loss_func):
    model.eval()  # Режим оценки (evaluation mode)
    val_loss = 0
    correct = 0
    with torch.no_grad():  # Отключение градиентов для ускорения и снижения использования памяти
        for images, labels in val_loader:
            outputs = model(images)
            val_loss += loss_func(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = correct / len(val_loader.dataset)
    return val_loss, val_accuracy


def train(train_loader, val_loader):
    model = network.Network()
    hyper_parameters = utiles.read_param()

    epochs = hyper_parameters.epochs
    learning_rate = hyper_parameters.learning_rate

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

            val_loss, val_accuracy = validate(model, val_loader, loss_func)
            print(f"Epoch {epoch}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    check_dir = Path('.\checkpoint.pth').resolve()
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, check_dir)


if __name__ == "__main__":
    hyper_parameters = utiles.read_param()
    batch_size = hyper_parameters.batch_size

    root_dir = Path('.\mnist_png\Training').resolve()

    # transform = ToTensor()

    transforms = Compose([
        RandomRotation(degrees=5),  # Случайный поворот изображений на угол до 5 градусов
        RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Случайные сдвиги по вертикали и горизонтали
        ToTensor(),  # Преобразование изображений в тензоры PyTorch
        Normalize((0.1307,), (0.3081,))  # Нормализация данных
    ])

    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--data_path', type=str, default=root_dir,
                        help='Path to the testing data directory')

    args = parser.parse_args()

    train_dataset = mnist_dataloader.CustomDataset(args.data_path, transform=transforms)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dataset_size = len(train_dataset)

    train_size = int(dataset_size * 0.8)  # 80% на обучающий набор
    val_size = dataset_size - train_size  # 20% на валидационный набор

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train(train_loader, val_loader)
