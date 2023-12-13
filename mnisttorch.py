import mnist_dataloader
import test
import train
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import os
import torch
import utiles

if __name__ == '__main__':
    root_dir = 'D:\datasets\mnist_png'
    transform = ToTensor()

    #train_loader, test_loader = mnist_dataloader.mnist_dataloader()

    train_dataset = mnist_dataloader.CustomDataset(os.path.join(root_dir, 'Training'), transform=transform)
    test_dataset = mnist_dataloader.CustomDataset(os.path.join(root_dir, 'Testing'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    train.train(train_loader)
    test.test(test_loader)
