from torchvision import transforms
from data_preprocessing import PlantSeedlingData
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


ROOTDIR="./ignore/plant-seedlings-classification"

def train():
    data_transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    
    train_path=Path(ROOTDIR).joinpath("train")
    train_data=PlantSeedlingData(train_path,data_transform)
    data_loader=DataLoader(dataset=train_data,batch_size=6,shuffle=True,num_workers=1)
    
