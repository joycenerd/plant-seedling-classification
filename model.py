import torch.nn as nn
import math
from torchvision import models
from collections import namedtuple


class VGG16(nn.Module):
    def __init__(self,num_classes=1000,requires_grad=False):
        super(VGG16,self).__init__()
        vgg_pretrained_features=models.vgg16(pretrained=True).features
        self.slice1=nn.Sequential()
        self.slice2=nn.Sequential()
        self.slice3=nn.Sequential()
        self.slice4=nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x),vgg_pretrained_features[x])
        for x in range(4,9):
            self.slice2.add_module(str(x),vgg_pretrained_features[x])
        for x in range(9,16):
            self.slice3.add_module(str(x),vgg_pretrained_features[x])
        for x in range(16,23):
            self.slice4.add_module(str(x),vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad=False

    def forward(self,X):
        h=self.slice1(X)
        h_relu1_2=h
        h=self.slice2(h)
        h_relu2_2=h
        h=self.slice3(h)
        h_relu3_3=h
        h=self.slice4(h)
        h_relu4_3=h
        VGG_outputs=namedtuple("VGG Outputs",['relu1_2','relu2_2','relu3_3','relu4_3'])
        out=VGG_outputs(h_relu1_2,h_relu2_2,h_relu3_3,h_relu4_3)
        return out





