import pandas as pd
import matplotlib.pyplot as plt


DATAPATH="./googlenet4-train_loss_acc.csv"

data=pd.read_csv(DATAPATH,header=None)
print(data)

plt.figure(figsize=[8,6])
plt.plot(data[0],'r',linewidth=3.0)
plt.plot(data[2],'b',linewidth=3.0)
plt.legend(['Training Loss','Validation Loss'],fontsize=18)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.savefig("./figure/loss_curve.jpg")

plt.figure(figsize=[8,6])
plt.plot(data[1],'r',linewidth=3.0)
plt.plot(data[3],'b',linewidth=3.0)
plt.legend(['Training Accuracy','Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.savefig("./figure/acc_curve.jpg")

