import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torch.nn.functional as F

train_data = datasets.MNIST(root='./', train=True, download=True, transform= torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5), (0.5))
]))
test_data = datasets.MNIST(root='./', train=False, download=True, transform= torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
]))

train_loader = data.DataLoader(train_data, batch_size= 10, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size= 10000, shuffle=True)

device = 'cuda'
from time import time
class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5) # 1 * 28 * 28 -> 6 * 24 * 24
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # 6 * 12 * 12 -> 16 * 8 * 8
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(2560, 100)
        self.fc2 = nn.Linear(100, 10)
    def forward(self,x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, -1)

model = CNN().to(device)

loss_fn = nn.CrossEntropyLoss()

import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs = 10

def training_loop():
    for epoch in range(epochs):
        model.train() 
        for img, label in train_loader:
                img, label = img.to(device), label.to(device)
                out = model(img)
                loss = loss_fn(out, label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print(loss)
        print(f'epoch {epoch} ended')
        print('training done!')

    torch.save(model.state_dict(), './MNIST.pth')

    #training loop

training_loop()

