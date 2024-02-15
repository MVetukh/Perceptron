import network
import torch
import argparse
import mnist_dataloader
from pathlib import Path
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from mnist_dataloader import CustomDataset
from network import Network
from utiles import read_param
from sklearn.metrics import accuracy_score


# def test(test_loader):
#     model: Network = network.Network()
#     check_dir: Path = Path('.\checkpoint.pth').resolve()
#     checkpoint = torch.load(check_dir)
#     model.load_state_dict(checkpoint['model_state_dict'])
#
#     model.eval()
#
#     with torch.no_grad():
#         correct: int = 0
#         total: int = 0
#         for images, labels in tqdm(test_loader):
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     print('Точность на тестовом наборе данных: {} %'.format(100 * correct /
#                                                                 total))
#
#     model.save_results('.\dataset\Result.csv', test_loader)

def test(test_loader: DataLoader):
    model: network.Network = network.Network()
    check_dir: Path = Path('./checkpoint.pth').resolve()
    checkpoint = torch.load(check_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    all_predicted, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Тестирование"):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predicted.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # Вычисляем общую точность
    accuracy = accuracy_score(all_labels, all_predicted)
    print(f'Точность на тестовом наборе данных: {accuracy * 100:.2f} %')

    model.save_results('./Result.csv', test_loader)


if __name__ == "__main__":
    root_dir: Path = Path('D:\datasets\mnist_png\Testing').resolve()
    hyper_parameters = read_param()
    batch_size = hyper_parameters.batch_size
    transform: ToTensor = ToTensor()

    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('--data_path', type=str, default=root_dir,
                        help='Path to the testing data directory')

    args = parser.parse_args()

    test_dataset: CustomDataset = mnist_dataloader.CustomDataset(
        args.data_path, transform=transform)

    test_loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)
    test(test_loader)
