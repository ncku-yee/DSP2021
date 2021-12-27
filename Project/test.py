import numpy as np
import torch, os, random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time
from sklearn.preprocessing import StandardScaler
from torch import optim
import argparse

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", dest="testdata")
args = parser.parse_args()
assert (args.testdata != None)

# Select CUDA device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Load testing data
testdata = np.load(args.testdata)
print(f"testdata.shape = {testdata.shape}")

# Preprocessing
print("===Start Preprocessing===")
scalerA = StandardScaler()
scalerA.fit(testdata[:,0,:])
tmp = scalerA.transform(testdata[:,0,:])
testdata[:,0,:] = tmp
scalerB = StandardScaler()
scalerB.fit(testdata[:,1,:])
tmp = scalerB.transform(testdata[:,1,:])
testdata[:,1,:] = tmp
print("===End Preprocessing===")

# Neural Networks
class MyDSPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 20, kernel_size=13, stride=7),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(20, 40, kernel_size=11, stride=7),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(40, 20, kernel_size=9, stride=5),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.clf = nn.Linear(1280, 3)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        output = self.clf(x)
        return output

# Load the model
def load_model(model,filename):
    model.load_state_dict(torch.load(filename))
    return model

# Load the weights and predict
model = MyDSPNet().to(device)
model = load_model(model, "./weight.pth")

with open("./result.csv","w") as f:
    f.write("id,category\n")
    for i, x in enumerate(testdata):
        x = torch.from_numpy(x.reshape(1, 2, 16000)).to(device)
        output = model(x)
        pred = output.argmax(dim=1, keepdim=True)
        f.write("%d,%d\n"%(i,pred.item()))