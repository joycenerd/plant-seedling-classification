from torchvision import transforms
from data_preprocessing import PlantSeedlingData
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import VGG16
import copy
import torch.nn as nn
import torch
from torch.autograd import Variable


ROOTDIR="./ignore/plant-seedlings-classification"

def train():
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_set = PlantSeedlingDataset(ROOTDIR.joinpath('train'), data_transform)
    data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=1)

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VGG16(num_classes=train_set.num_classes)
    model = model.cuda(device)
    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_epochs = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    num_epochs=2
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

        training_loss = 0.0
        training_corrects = 0

        for i, (inputs, labels) in enumerate(data_loader):
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.data[0] * inputs.size(0)
            training_corrects += torch.sum(preds == labels.data)

        training_loss = training_loss / len(train_set)
        training_acc = training_corrects / len(train_set)

        print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n'）





        




    
