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

    # training data initialization
    data_transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    
    train_path=Path(ROOTDIR).joinpath("train")
    train_data=PlantSeedlingData(train_path,data_transform)
    data_loader=DataLoader(dataset=train_data,batch_size=32,shuffle=True,num_workers=1)

    # parameters initialization
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=VGG16(num_classes=train_data.num_classes).to(device)
    model.train()
    best_model_parameters=copy.deepcopy(model.state_dict())
    best_accuracy=0.0
    epochs=50
    loss_function=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(params=model.parameters(),lr=0.001,momentum=0.9)
    batch_size=32

    for epoch in range(2):
        show_progress="Epoch: "+str(epoch+1)+"/"+str(2)
        print(show_progress)
        print('-'*len(show_progress))

        current_loss=0.0
        current_acc=0

        for i,(_input,label) in enumerate(data_loader):
            _input=Variable(_input.cuda(device) if device=="cuda:0" else _input)
            label=Variable(label.cuda(device) if device=="cuda:0" else label)
            
            # zero the parameter gradient
            optimizer.zero_grad()

            # forward
            outputs=model(_input)
            _,predict_y=torch.max(outputs.data,1)
            loss=loss_function(outputs,label)

            # backward
            loss.backward()
            optimizer.step()

            # statistics
            current_loss+=loss.data[0]*_input.size(0)
            current_acc+=torch.sum(predict_y==label.data)

        total_loss=current_loss/len(train_data)
        total_acc=current_acc/len(train_data)
        print("Loss: "+str(total_loss))
        print("Accuracy: "+str(total_acc))





        




    
