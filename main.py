import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.functional as F
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

def prediction_loop(model, loader):
    samplesubs = pd.read_csv('./data/sample_submission.csv')
    allpreds = []
    for data in loader:
        images = data
        outputs = (model(images))
        preds = outputs > 0.5
        preds = preds.cpu().tolist()
        print(preds)
        for i in preds[0]:
            i = int(i)
        print(preds)
        allpreds.append(preds)
    samplesubs.append(pd.DataFrame(allpreds).T)
    return samplesubs
        
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


model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model.fc = nn.Sequential(nn.Linear(2048, 10), nn.Sigmoid())
ct = 0
for child in model.children():
    ct += 1
    if ct < 10:
        for param in child.parameters():
            param.requires_grad = False

model = model.to('cuda')

img_set = Loader(transforms=T.Compose([T.Resize((100,100)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(([0.2687, 0.2129, 0.1901]), ([0.4584, 0.5564, 0.6069]))]))
val_set = Loader(transforms=T.Compose([T.Resize((100,100)), T.ToTensor()]), validation=True)
img_loader = data.DataLoader(img_set,batch_size = 64, shuffle=True)

#model = Model().to('cuda')
weights = torch.FloatTensor([0.55, 0.45])
loss = nn.BCELoss(weight=weights, reduction='none')
#model.load_state_dict(torch.load('./Model.pth'))

training_loop(model, epochs=10, trainloader=img_loader,loss_fn=loss, optimizer=optim.SGD(params=model.parameters(),lr=1e-3, weight_decay=5e-4, momentum=0.9), validation=val_set)



class pred_loader(data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.imgs = os.listdir('data/test')
        self.transforms = T.Compose([T.Resize((100, 100)), T.ToTensor()])
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_loc = os.path.join('./data/test', self.imgs[index])
        image = Image.open(img_loc).convert("RGB") 
        tensor_image = self.transforms(image)
        return tensor_image.to('cuda')


# predset = pred_loader()
# predloader = data.DataLoader(predset)
# model.load_state_dict(torch.load('./Model.pth'))

# a = prediction_loop(model,predloader)

# a.to_csv('./predictions.csv')