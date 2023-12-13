import torch
import torchvision
import utiles
import os
from torch.utils.data import Dataset
from PIL import Image


def mnist_dataloader():
    train_dataset = torchvision.datasets.MNIST(root='D:\PyProjects\Peceptron\Perceptron\dataset', train=True,
                                               transform=torchvision.transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='D:\PyProjects\Peceptron\Perceptron\dataset', train=False,
                                              transform=torchvision.transforms.ToTensor(),
                                              download=True)

    batch_size, _, _ = utiles.read_param()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# class CustomDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.classes = sorted(os.listdir(root_dir))
#         self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
#         self.samples = self._make_dataset()
#
#     def _make_dataset(self):
#         samples = []
#         for cls_name in self.classes:
#             cls_path = os.path.join(self.root_dir, cls_name)
#             if os.path.isdir(cls_path):
#                 for img_name in sorted(os.listdir(cls_path)):
#                     img_path = os.path.join(cls_path, img_name)
#                     samples.append((img_path, self.class_to_idx[cls_name]))
#         return samples
#
#     def __getitem__(self, index):
#         img_path, label = self.samples[index]
#         img = Image.open(img_path)
#         if self.transform:
#             img = self.transform(img)
#         return img, label
#
#     def __len__(self):
#         return len(self.samples)


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)

    def __len__(self):
        num_samples = 0
        for class_dir in self.classes:
            class_path = os.path.join(self.root_dir, class_dir)
            num_samples += len(os.listdir(class_path))
        return num_samples

    def __getitem__(self, index):
        img_path = self.get_image_path(index)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label = int(os.path.basename(os.path.dirname(img_path)))
        return image, label

    def get_image_path(self, index):
        class_dir = self.classes[index % len(os.listdir(self.root_dir))]
        class_path = os.path.join(self.root_dir, class_dir)
        image_index = index % len(os.listdir(class_path))
        image_filename = os.listdir(class_path)[image_index]
        image_path = os.path.join(class_path, image_filename)
        return image_path
