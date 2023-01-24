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
import os
import torchsampler
from torch.utils.data import WeightedRandomSampler


'''
the diffs variable is the differences of 0s and 1s in each classes. 
ex)
    diffs[0] = (number of 0s in the row "A") - (number of 1s in the row "A")

the diffs are calculated using the following function: 
def count(df):
    zeros = 0
    ones = 0
    for index, row in df.iterrows():
        if row['A'] == 0:
            zeros += 1
        else:
            ones += 1

    print(zeros, ones)
'''
diffs = [2412
        ,2798
        ,3564
        ,3638
        ,3978
        ,3928
        ,3974
        ,4410
        ,4282
        ,4362]

def count(df,name):
    zeros = 0
    ones = 0
    for index, row in df.iterrows():
        if row[name] == 0:
            zeros += 1
        else:
            ones += 1

    print(zeros - ones)
def appendrow(name, df):
    rand = df.query(f'{name} == 1').sample(n = diffs[list('ABCDEFGHIJ').index(name)])
    df = pd.concat([df, rand], ignore_index=True)
    df.to_csv('testdf.csv')
    return df

def training_loop(model, epochs, trainloader, optimizer, validation, name):
        loss_fn = nn.CrossEntropyLoss()
        training_start = time()
        losses = []
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
                loss = loss_fn(out, label)
                if i % 100 == 0:
                    print(i)
                    losses.append(loss.item())
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
        plt.plot(losses)
        torch.save(model.state_dict(), f'./Model{name}.pth')

def testing_loop(model, testloader):
        correct = 0
        total = 0
        model.eval()
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = model(images.unsqueeze(0))
                # the class with the highest energy is what we choose as prediction
                print(outputs)
                _,predicted = torch.max(outputs.data, 1)
                total += 1
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on test images: {100 * correct // total} %      || correct : {correct}, total : {total}')

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5), # 100x100x3 -> 96x96x10
            nn.ReLU(),
            nn.MaxPool2d(2), # 48 x 48 x 10
            nn.Conv2d(10, 30, 5), # 44x44x30
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(2), # 22x22x30
            nn.Conv2d(30, 40, 5), # 18 * 18 * 40
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(2),# 9 * 9 * 40
        )

        self.fc = nn.Sequential(
            nn.Linear(40*9*9,500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc(x)
        return x


    
class CustomLoader(data.Dataset):
    def __init__(self, class_name:str, transforms, df, validation = False,) -> None:
        super().__init__()
        self.name = class_name
        self.csv = df
        self.transforms = transforms
        self.imgs = os.listdir('./data/train') 
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        df = self.csv
        img_loc = os.path.join('./data', df['img_path'].iloc[index])
        image = Image.open(img_loc).convert("RGB") 
        tensor_image = self.transforms(image)
        tensor_label = torch.tensor(df[self.name].iloc[index],dtype=torch.long)
        return tensor_image.to('cuda'), tensor_label.to('cuda')


#example_loader = CustomLoader('A', transforms=T.Compose([T.Resize((100,100)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.7854), (0.4284))]))



def trainer():
    classes = 'ABCDEFGHIJ'
    for i in list(classes):
        full_df = pd.read_csv('data/train.csv')
        full_df = full_df.sample(frac=1, random_state=42)
        train_df = full_df[0:int(len(full_df) * 0.9)]
        val_df = full_df[int(len(full_df) * 0.9):]
        df = appendrow(i, df= train_df)
        count(df,name=i)
        train = CustomLoader(i, transforms=T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([0.9391, 0.9257, 0.9181], [0.1404, 0.1725, 0.1931])]), df=df)
        val = CustomLoader(i, transforms=T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([0.9391, 0.9257, 0.9181], [0.1404, 0.1725, 0.1931])]), df=val_df, validation=True)
        trainloader = data.DataLoader(train, batch_size=64)
        valloader = data.DataLoader(val, batch_size=1)
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model.fc = nn.Sequential(nn.Linear(512, 2))
        model = model.to('cuda')
        training_loop(model, 5, trainloader, optimizer=optim.SGD(params=model.parameters(), lr= 0.001), validation=val, name=i)



trainer()
