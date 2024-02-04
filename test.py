import network
import torch
import argparse
import mnist_dataloader
from pathlib import Path
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from mnist_dataloader import CustomDataset
from network import Network
from utiles import read_param


def test(test_loader):
    model: Network = network.Network()
    check_dir: Path = Path('.\checkpoint.pth').resolve()
    checkpoint = torch.load(check_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    with torch.no_grad():
        correct: int = 0
        total: int = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Точность на тестовом наборе данных: {} %'.format(100 * correct / total))

    model.save_results('.\dataset\Result.csv', test_loader)


#
if __name__ == "__main__":
    root_dir: Path = Path('.\mnist_png\Testing').resolve()
    hyper_parameters = read_param()
    batch_size = hyper_parameters.batch_size
    transform: ToTensor = ToTensor()



    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('--data_path', type=str, default=root_dir,
                        help='Path to the testing data directory')

    args = parser.parse_args()

    test_dataset: CustomDataset = mnist_dataloader.CustomDataset(args.data_path, transform=transform)

    test_loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test(test_loader)
