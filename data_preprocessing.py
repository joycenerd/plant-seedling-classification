from torch.utils.data import Dataset
from pathlib import Path
import os

class PlantSeedlingData(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir=Path(root_dir)
        self.x=[]
        self.y=[]
        self.transform=transform
        self.num_classes=0

        if(self.root_dir.name=='train'):
            for folder in os.listdir(root_dir):
                print(folder)

