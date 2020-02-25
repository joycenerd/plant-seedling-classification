from torch.utils.data import Dataset
from pathlib import Path
import os
from PIL import Image

class PlantSeedlingData(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir=Path(root_dir)
        self.x=[]
        self.y=[]
        self.transform=transform
        self.num_classes=0
        self.num_data=0

        if(self.root_dir.name=='train'):
            i=0
            for folder in os.listdir(root_dir):
                folder_path=root_dir.joinpath(folder)
                for _file in os.listdir(folder_path):
                   self.x.append(folder_path.joinpath(_file))
                   self.y.append(i)
                   self.num_data+=1
                self.num_classes+=1
                i+=1
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx):
        image=Image.open(self.x[idx]).convert('RGB')
        if self.transform:
            image=self.transform(image)
        return image,self.y[idx]

