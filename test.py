from train import NeuralNetwork
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = "cpu"

# Loads CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(
    testset, batch_size=16)

dataiter = iter(testloader)
images, labels = next(dataiter)

network = NeuralNetwork()

network.load_state_dict(torch.load("imlonetwork.pth"))