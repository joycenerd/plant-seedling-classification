import torch.nn as nn
import math

class VGG16(nn.Module):
    def __init__(self,num_classes=1000):
        super(VGG16,self).__init__()
        self.features=nn.Sequential(
            
        )