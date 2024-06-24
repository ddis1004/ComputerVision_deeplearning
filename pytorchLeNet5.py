import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.is_available())

datasets.MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
        ]

class CustomAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(CustomAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.scale = None  # Learnable scaling factor
        self.bias = None  # Learnable bias

    def forward(self, x):
        if self.scale is None or self.bias is None:
            num_channels = x.size(1)
            device = x.device  # Get the device of the input tensor
            self.scale = nn.Parameter(torch.ones(num_channels, 1, 1, device = device))  # Learnable scaling factor for each feature map
            self.bias = nn.Parameter(torch.zeros(num_channels, 1, 1, device = device))  # Learnable bias for each feature map

        x = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return self.scale * x + self.bias


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = CustomAvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.ModuleList([nn.Conv2d(len(connections), 1, kernel_size=5)
                                    for connections in [
                                        [0, 1, 2],
                                        [1, 2, 3],
                                        [2, 3, 4],
                                        [3, 4, 5],
                                        [0, 4, 5],
                                        [0, 1, 5],
                                        [0, 1, 2, 3],
                                        [1, 2, 3, 4],
                                        [2, 3, 4, 5],
                                        [0, 3, 4, 5],
                                        [0, 1, 2, 4],
                                        [1, 2, 3, 5],
                                        [0, 1, 3, 4],
                                        [0, 2, 3, 5],
                                        [0, 1, 4, 5],
                                        [1, 2, 3, 4, 5],
                                    ]])

        self.pool2 = CustomAvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Adjusted input size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.connections = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [0, 4, 5],
            [0, 1, 5],
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [0, 3, 4, 5],
            [0, 1, 2, 4],
            [1, 2, 3, 5],
            [0, 1, 3, 4],
            [0, 2, 3, 5],
            [0, 1, 4, 5],
            [1, 2, 3, 4, 5],
        ]

    def forward(self, x):
        x = F.tanh(self.conv1(x))  # First convolutional layer + Tanh activation
        x = self.pool1(x)  # First custom pooling layer

        # C3 layer
        c3 = [F.tanh(self.conv2[i](torch.cat([x[:, j:j + 1, :, :] for j in self.connections[i]], dim=1)))
              for i in range(16)]
        x = torch.cat(c3, dim=1)  # Concatenate the feature maps along the channel dimension

        x = self.pool2(x)  # Second custom pooling layer
        x = x.view(-1, 16 * 5 * 5)  # Flatten the tensor
        x = F.tanh(self.fc1(x))  # First fully connected layer + Tanh activation
        x = F.tanh(self.fc2(x))  # Second fully connected layer + Tanh activation
        x = self.fc3(x)  # Output layer (no activation, will be applied later)
        return x

class LeNet5_simplified(nn.Module):
    def __init__(self):
        super(LeNet5_simplified, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
        x = F.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)
        x = F.tanh(self.conv3(x))
        x = x.view(-1, 120)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

epoch_result = []

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Assuming LeNet5 and LeNet5_simplified are defined

def train(model, device, train_loader, optimizer, epoch, train_accuracies):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(train_loader.dataset)
    train_accuracies.append(accuracy)

def test(model, device, test_loader, test_accuracies):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'{len(test_accuracies)} : {accuracy}')
    test_accuracies.append(accuracy)

# Data loaders
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LeNet5().to(device)
simplified_model = LeNet5_simplified().to(device)

optimizer1 = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer2 = optim.SGD(simplified_model.parameters(), lr=0.01, momentum=0.9)

train_accuracies = []
test_accuracies = []

for epoch in range(1, 21):
    train(model, device, train_loader, optimizer1, epoch, train_accuracies)
    test(model, device, test_loader, test_accuracies)

simplified_train_accuracies = []
simplified_test_accuracies = []

for epoch in range(1, 21):
    train(simplified_model, device, train_loader, optimizer2, epoch, simplified_train_accuracies)
    test(simplified_model, device, test_loader, simplified_test_accuracies)

# Plotting
epochs = range(1, 21)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, test_accuracies, label='Test Accuracy')
plt.title('LeNet5 Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, simplified_train_accuracies, label='Train Accuracy')
plt.plot(epochs, simplified_test_accuracies, label='Test Accuracy')
plt.title('LeNet5 Simplified Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

