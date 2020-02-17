from torchvision import transforms
from data_preprocessing import PlantSeedlingData
from pathlib import Path


ROOTDIR="./ignore/plant-seedlings-classification"

def train():
    data_transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    train_path=Path(ROOTDIR).joinpath("train")
    PlantSeedlingData(train_path,data_transform)
