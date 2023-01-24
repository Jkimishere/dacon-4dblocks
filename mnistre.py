import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
from torchvision import transforms, datasets


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])


train_data = datasets.MNIST('./data', True, transform=transforms.ToTensor(),download=True)
train_loader = utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


test_data = datasets.MNIST('./data', False, transform=transforms.ToTensor(),download=True)
test_loader = utils.data.DataLoader(test_data, batch_size=batch_size,
                                          shuffle=True, num_workers=2)




class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 10)
        self.pool = nn.MaxPool2d(2,2)


    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, -1)

if __name__ == '__main__':
    model = net().to (device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    epochs = 30

    def training_loop():
        for epoch in range(epochs):
            model.train() 
            for img, label in train_loader:
                img, label = img.to(device), label.to(device)
                out = model(img)
                loss = loss_fn(out, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(loss)
            print(f'epoch {epoch} ended')
        print('training done!')

    torch.save(model.state_dict(), './MNIST.pth')

    #training loop

    training_loop()
    
    # model.load_state_dict(torch.load('./MNIST.pth'))

    # model.eval()

    # with torch.inference_mode():
    #     # for img, label in test_loader:
    #     #     img, label = img.to(device), label.to(device)
    #     #     out = model(img)
    #     #     print(out)
    #     #     loss = loss_fn(out, label)
    #     img,label = next(iter(test_loader))

    #     img, label = img.to(device), label.to(device)
    #     outputs=model(img)
    #     _, preds = torch.max(outputs, 1)
    #     preds=preds.cpu().numpy()
    #     classes=label.cpu().numpy()
    #     print(preds)
    #     print(classes)
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #     plt.imshow(img[0].cpu().squeeze(0), cmap= 'gray')
    #     plt.show()
    # # print(f'loss : {loss}')

     




def training_loop():
    for epoch in range(epochs):
        model.train() 
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            out = model(img)
            loss = loss_fn(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'epoch {epoch} ended')

    print('training done!')

    torch.save(model.state_dict(), './MNIST.pth')

    #training loop
