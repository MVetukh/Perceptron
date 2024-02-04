from typing import Any
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from pathlib import Path
from utiles import read_param


def prepare_dataset(root_dir):
    data = []
    root_path = Path(root_dir)
    # Проходимся по всем поддиректориям и собираем информацию о путях к изображениям и их классах
    for class_folder in root_path.iterdir():
        if class_folder.is_dir():
            class_id = class_folder.name
            images_paths = list(class_folder.glob('*.png'))
            data.extend([(img_path, int(class_id)) for img_path in images_paths])

    return data


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data_info = prepare_dataset(root_dir)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):

        img_path, class_id = self.data_info[idx]
        image = Image.open(img_path).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        return image, class_id


def mnist_dataloader():
    train_dataset = MNIST(root='D:\PyProjects\Peceptron\Perceptron\dataset', train=True,
                          transform=ToTensor(),
                          download=True)
    test_dataset = MNIST(root='D:\PyProjects\Peceptron\Perceptron\dataset', train=False,
                         transform=ToTensor(),
                         download=True)

    param = read_param()
    batch_size = param.batch_size

    train_loader: DataLoader[Any] = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader: DataLoader[Any] = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



# class CustomDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.classes = os.listdir(root_dir)
#
#     def __len__(self):
#         num_samples = 0
#         for class_dir in self.classes:
#             class_path = os.path.join(self.root_dir, class_dir)
#             num_samples += len(os.listdir(class_path))
#         return num_samples
#
#     def __getitem__(self, index):
#         img_path = self.get_image_path(index)
#         image = Image.open(img_path)
#
#         if self.transform:
#             image = self.transform(image)
#
#         label = int(os.path.basename(os.path.dirname(img_path)))
#         return image, label
#
#     def get_image_path(self, index):
#         class_dir = self.classes[index % len(os.listdir(self.root_dir))]
#         class_path = os.path.join(self.root_dir, class_dir)
#         image_index = index % len(os.listdir(class_path))
#         image_filename = os.listdir(class_path)[image_index]
#         image_path = os.path.join(class_path, image_filename)
#         return image_path
