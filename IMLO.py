import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Transforms the dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Loasd CIFAR-10 training dataset
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(
    trainset, batch_size=16)

# Loads CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(
    testset, batch_size=32)

# Define the classes in CIFAR-10
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        # Convolutional layers take 3 channel images for RGB
        # with the second layer outputting 16 channels
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

NeuralNet = NeuralNetwork()

NeuralNet.to(device)

# Create the loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimiser = optim.SGD(NeuralNet.parameters(), lr=0.0001, momentum=0.9)

def train_model(trainloader, loss_function, optimiser, epochs):
    for epoch in range(epochs):
        current_loss = 0
        for i, data in enumerate(trainloader):
            # data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimiser.zero_grad()

            outputs = NeuralNet(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimiser.step()

            # prints loss for a batch of 2000
            current_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[Epoch {epoch + 1}] loss: {current_loss/2000:.3f}')
                current_loss = 0

    print('Finished Training')

epochs = 50

train_model(trainloader, loss_function, optimiser, epochs)
