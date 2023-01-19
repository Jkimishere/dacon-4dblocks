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

#TODO : make model and initialize model inside training loop 
# ? : is there a better way of making multiple models?
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


class CustomLoader(data.Dataset):
    def __init__(self, class_name:str, transforms, validation = False) -> None:
        super().__init__()
        self.name = class_name
        self.csv = pd.read_csv('./data/train.csv')
        self.transforms = transforms
        self.imgs = os.listdir('./data/train') 
        random.shuffle(self.imgs)
        if not validation:
            self.imgs = self.imgs[0:int(len(self.imgs) * 0.9)] #split images(first 90%)
        else:
            self.imgs = self.imgs[int(len(self.imgs) * 0.9):]
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        df = self.csv[self.name]
        img_loc = os.path.join('./data/train', self.imgs[index])
        image = Image.open(img_loc).convert("RGB") 
        tensor_image = self.transforms(image)
        tensor_label = torch.tensor(df[index],dtype=torch.float32)
        return tensor_image.to('cuda'), tensor_label.to('cuda')


loader = CustomLoader('A', transforms=T.Compose([T.Resize((100,100)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))


print(loader.__getitem__(10))