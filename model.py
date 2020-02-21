import torch.nn as nn
import math
from torchvision import models
from collections import namedtuple


class VGG16(nn.Module):
    def __init__(self,num_classes=1000,requires_grad=False):
        super(VGG16,self).__init__()
        vgg_pretrained_features=models.vgg16(pretrained=True).features
        self.features=vgg_pretrained_features
        
        # (batch size,channels,rows,cols)
        # input=(batch size,3,224,224)
        # output=(batch size,512,7,7)
        self.classifier=nn.Sequential(
            nn.Linear(in_features=512*7*7,out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096,out_features=num_classes)
        )

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x




