import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Transforms the dataset
transform = transforms.ToTensor()

# Loasd CIFAR-10 training dataset
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(
    trainset, batch_size=4)

# Loads CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(
    testset, batch_size=4)

# Define the classes in CIFAR-10
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')