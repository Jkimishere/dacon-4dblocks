import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
from PIL import Image
from time import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random


def training_loop(model, epochs, trainloader, loss_fn, optimizer, validation):
        training_start = time()
        for epoch in range(epochs):
            model.train()
            start = time()
            loss = 0.0
            print(f'epoch {epoch}')
            model.train() 
            for i, data in enumerate(trainloader):
                img, label = data
                out = model(img)
                #print(out, label)
                loss = loss_fn(((out)), label)
                if i % 100 == 0:
                    print(i)
                    print(f'loss value is {loss.item()}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            end = time()
            print(f'epoch {epoch} ended with loss {loss} ||| epoch {epoch} runtime : {end - start} seconds')
            print(f'epoch {epoch} validation')
            model.eval()
            testing_loop(model, validation)
        training_end = time()
        print(f'training done in {int(training_end - training_start)} seconds, or {int(training_end - training_start) / 60} minutes')
        torch.save(model.state_dict(), './Model.pth')

def testing_loop(model,testloader):
        correct = 0
        total = 0
        model.eval()
        allpreds = []
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = (model(images.unsqueeze(0)))
                print(outputs)
                preds = outputs > 0.5
                allpreds.append(preds.values)
                total += 10
                correct += (preds == labels).sum().item()
        print(f'total {total}, correct {correct}')
        return allpreds

class Loader(data.Dataset):
    def __init__(self, transforms, validation=False) -> None:
        super().__init__()
        self.csv = pd.read_csv('./data/train.csv')
        self.transforms = transforms
        self.imgs = os.listdir('./data/train')
        self.label = self.csv.iloc[:,2:].values
        random.shuffle(self.imgs)
        if not validation:
            self.imgs = self.imgs[0:int(len(self.imgs) * 0.9)] #split images(first 90%)
        else:
            self.imgs = self.imgs[int(len(self.imgs) * 0.9):]
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_loc = os.path.join('./data/train', self.imgs[index])
        image = Image.open(img_loc).convert("RGB") 

        tensor_image = self.transforms(image)
        tensor_label = torch.tensor(self.label[index],dtype=torch.float32)
        return tensor_image.to('cuda'), tensor_label.to('cuda')
        


# class Model(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 10, kernel_size=5), # 100x100x3 -> 96x96x10
#             nn.ReLU(),
#             nn.MaxPool2d(2), # 48 x 48 x 10
#             nn.Conv2d(10, 30, 5), # 44x44x30
#             nn.BatchNorm2d(30),
#             nn.ReLU(),
#             nn.MaxPool2d(2), # 22x22x30
#             nn.Conv2d(30, 40, 5), # 18 * 18 * 40
#             nn.BatchNorm2d(40),
#             nn.ReLU(),
#             nn.MaxPool2d(2),# 9 * 9 * 40
#         )

#         self.fc = nn.Sequential(
#             nn.Linear(40*9*9,500),
#             nn.ReLU(),
#             nn.Linear(500, 100),
#             nn.ReLU(),
#             nn.Linear(100, 10),
#             nn.Sigmoid(),
#         )
#     def forward(self, x):
#         x = self.conv(x)
#         x = torch.flatten(x,start_dim=1)
#         x = self.fc(x)
#         return x

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model.fc = nn.Sequential(nn.Linear(2048, 10), nn.Sigmoid())
print(model)
model = model.to('cuda')


img_set = Loader(transforms=T.Compose([T.Resize((100,100)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
val_set = Loader(transforms=T.Compose([T.Resize((100,100)), T.ToTensor()]), validation=True)
img_loader = data.DataLoader(img_set,batch_size = 64, shuffle=True)

#model = Model().to('cuda')
loss = nn.BCELoss()
#model.load_state_dict(torch.load('./Model.pth'))

training_loop(model, epochs=10, trainloader=img_loader,loss_fn=loss, optimizer=optim.SGD(params=model.parameters(),lr=0.001, weight_decay=5e-4, momentum=0.95), validation=val_set)



#model.eval()

#a = testing_loop(model, val_set)