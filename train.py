from torchvision import transforms
from data_preprocessing import PlantSeedlingData
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import VGG16,VGG19
import copy
import torch.nn as nn
import torch
from torch.autograd import Variable
import csv
import numpy as np


ROOTDIR="./ignore/plant-seedlings-classification"

def train():
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_set = PlantSeedlingData(Path(ROOTDIR).joinpath('train'), data_transform)
    data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=1)

    # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Input model: ")
    which_model=input()
    if which_model=="vgg16":
        model = VGG16(num_classes=train_set.num_classes)
        num_epochs=50
    elif which_model=="vgg19":
        model=VGG19(num_classes=train_set.num_classes)
        num_epochs=100

    if torch.cuda.is_available():
        model=model.cuda("cuda:0")
    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
    loss_acc=np.empty((0,2),dtype=float)


    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

        training_loss = 0.0
        training_corrects = 0

        for i, (inputs, labels) in enumerate(data_loader):
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

        training_loss = training_loss / len(train_set)
        training_acc = float(training_corrects) / len(train_set)
        loss_acc=np.append(loss_acc,np.array([[training_loss,training_acc]]),axis=0)

        print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')
        if training_acc>best_acc:
            best_acc=training_acc
            best_model_params=copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_params)
    torch.save(model, f'model-{best_acc:.02f}-best_train_acc.pth')
    np.savetxt('loss_acc.csv',loss_acc,delimiter=',')





        




    
