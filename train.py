import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Transforms the dataset
transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Loasd CIFAR-10 training dataset
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(
    trainset, batch_size=16)

# Define the classes in CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        # 3 convolutional layers which take 3 channels being RGB
        # and output 128 channels with a kernel size of 3.
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(3, 2)

        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the output from the convolutional layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

NeuralNet = NeuralNetwork()

NeuralNet.to(device)

# Create the loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimiser = optim.SGD(NeuralNet.parameters(), lr=0.00005, momentum=0.9)

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

            # prints loss for each epoch
            current_loss += loss.item()
            if i % 3125 == 3124:
                print(f'[Epoch {epoch + 1}] loss: {current_loss/3125:.3f}')
                current_loss = 0

    print('training finished after %d epochs' % epochs)

epochs = 200
if __name__ == '__main__':
    train_model(trainloader, loss_function, optimiser, epochs)

    # Save the model
    torch.save(NeuralNet.state_dict(), 'imlonetwork.pth')
    print('Model saved as imlonetwork.pth')