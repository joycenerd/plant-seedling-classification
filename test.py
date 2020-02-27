import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.autograd import Variable


ROOTDIR="./ignore/plant-seedlings-classification"

def test():
    data_transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    classes=[_dir for _dir in os.listdir(Path(ROOTDIR).joinpath('train'))]
    num_classes=len(classes)

    model=torch.load("/mnt/md0/new-home/joycenerd/plant-seedling-classification/googlenet4-best-train-acc.pth")
    if torch.cuda.is_available():
        model=model.cuda("cuda:0")
    model.eval()

    sample_submission=pd.read_csv(Path(ROOTDIR).joinpath('sample_submission.csv'))
    submission=sample_submission.copy()
    for i,filename in enumerate(sample_submission['file']):
        image=Image.open(Path(ROOTDIR).joinpath('test').joinpath(filename)).convert('RGB')
        image=data_transform(image).unsqueeze(0)
        if torch.cuda.is_available():
            inputs=Variable(image.cuda("cuda:0"))
        else:
            inputs=Variable(image)
        outputs=model(inputs)
        _,preds=torch.max(outputs.data,1)
        submission['species'][i]=classes[preds[0]]
        print(filename+" complete")
    
    submission.to_csv(Path(ROOTDIR).joinpath('submission4.csv'),index=False)






if __name__=='__main__':
    test()
