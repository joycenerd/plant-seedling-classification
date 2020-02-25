from torchvision import transforms
from data_preprocessing import PlantSeedlingData
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import VGG16,VGG19,GOOGLENET
import copy
import torch.nn as nn
import torch
from torch.autograd import Variable
import csv
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from callback import EarlyStopping


ROOTDIR="./ignore/plant-seedlings-classification"

def train():
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_set=PlantSeedlingData(Path(ROOTDIR).joinpath('train'), data_transform)

    valid_size=0.25
    num_train=len(train_set)
    indices=list(range(num_train))
    np.random.shuffle(indices)
    split=int(np.floor(valid_size*num_train))
    train_idx,valid_idx=indices[split:],indices[:split]

    train_sampler=SubsetRandomSampler(train_idx)
    valid_sampler=SubsetRandomSampler(valid_idx)
    train_loader=DataLoader(train_set,batch_size=32,sampler=train_sampler,num_workers=1)
    valid_loader=DataLoader(train_set,batch_size=32,sampler=valid_sampler,num_workers=1)
    

    # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Input model: ")
    which_model=input()
    if which_model=="vgg16":
        model = VGG16(num_classes=train_set.num_classes)
        num_epochs=50
    elif which_model=="vgg19":
        model=VGG19(num_classes=train_set.num_classes)
        num_epochs=100
    elif which_model=="googlenet":
        model=GOOGLENET(num_classes=train_set.num_classes)
        num_epochs=100

    if torch.cuda.is_available():
        model=model.cuda("cuda:0")
    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
    loss_acc=np.empty((0,4),dtype=float)

    early_stopping=EarlyStopping(patience=20,verbose=True)

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

        training_loss = 0.0
        training_corrects = 0
        valid_loss=0.0
        valid_corrects=0

        for i, (inputs, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda("cuda:0"))
                labels = Variable(labels.cuda("cuda:0"))
            else:
                inputs=Variable(inputs)
                labels=Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            training_loss += loss.item() * inputs.size(0)
            training_corrects += torch.sum(preds == labels.data)

        training_loss = training_loss / (len(train_set)-split)
        training_acc = float(training_corrects) / (len(train_set)-split)

        model.eval()
        for _data,target in valid_loader:
            outputs=model(_data)
            _, preds = torch.max(outputs.data, 1)
            loss=criterion(outputs,target)

            valid_loss+=loss.item()*_data.size(0)
            valid_corrects+=torch.sum(preds==target.data)
        
        valid_loss=valid_loss/split
        valid_acc=float(valid_corrects)/split
        
        loss_acc=np.append(loss_acc,np.array([[training_loss,training_acc,valid_loss,valid_acc]]),axis=0)

        print_msg=(f'train_loss: {training_loss:.4f} valid_loss: {valid_loss:.4f}\t'+
                   f'train_acc: {training_acc:.4f} valid_acc: {valid_acc:.4f}')
        print(print_msg)

        early_stopping(valid_loss,model)
        if early_stopping.early_stop:
            print("Early Stopping")
            break

    loss_acc=np.round(loss_acc,4)
    np.savetxt('googlenet2-train_loss_acc.csv',loss_acc,delimiter=',')
    
    model.load_state_dict(torch.load('checkpoint2.pt'))
    torch.save(model,'googlenet2-best-train-acc.pth')




        




    
