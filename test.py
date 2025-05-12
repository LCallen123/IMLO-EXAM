from train import NeuralNetwork
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = "cpu"

# Loads CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(
    testset, batch_size=32)

dataiter = iter(testloader)
images, labels = next(dataiter)

network = NeuralNetwork()

network.load_state_dict(torch.load("imlonetwork.pth"))

correct = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = network(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
print(f'Test set accuracy: {100 * correct // 10000}%')