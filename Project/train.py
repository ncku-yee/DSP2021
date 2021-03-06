import numpy as np
import torch, os, random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import optim
import argparse

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", dest="traindata")
parser.add_argument("-l", "--label", dest="trainlabel")
args = parser.parse_args()
assert (args.traindata != None and args.trainlabel != None)

# Fix the seed
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

# Select CUDA device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Load training data and label
traindata, trainlabel = np.load(args.traindata), np.load(args.trainlabel)
print(f"traindata.shape = {traindata.shape}")
print(f"trainlabel.shape = {trainlabel.shape}")

# Preprocessing
print("===Start Preprocessing===")
# FFT(Fast Fourier Transform)
# traindata = np.abs(np.fft.fft(traindata))
scalerA = StandardScaler()
scalerA.fit(traindata[:,0,:])
tmp = scalerA.transform(traindata[:,0,:])
traindata[:,0,:] = tmp
scalerB = StandardScaler()
scalerB.fit(traindata[:,1,:])
tmp = scalerB.transform(traindata[:,1,:])
traindata[:,1,:] = tmp
print("===End Preprocessing===")

# Define Dataset Class
class RawDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = torch.from_numpy(np.array(data).astype(np.float32))
        if labels is not None:
            self.labels = torch.from_numpy(np.array(labels).astype(np.int64))
        else:
            self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.labels is not None:
            target = self.labels[idx]
            return sample, target
        else:
            return sample

# Determine the hyperparameters.
batch_size = 48
lr = 3e-3
epochs = 30

# Load the dataset.
train_dataloader = DataLoader(RawDataset(traindata, trainlabel),
                                batch_size = batch_size, shuffle = True)

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

# Create model
model = MyDSPNet().to(device)

# Show the architecture of model
for name, module in model.named_children():
    print(module)

# Get the input size of training data.
input_size = None
for X, y in train_dataloader:
    input_size = X.shape[1:]                        # input_size = (C, H, W)
    break

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

# Each training loop will call train function once.
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)          # training dataset ?????????
    num_batches = len(dataloader)           # batch size ?????????
    total_loss, correct = 0, 0              # ?????? loss ?????????
    model.train()                           # model ??? training ??????
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)   # ???????????????????????????????????? torch ???????????? device('cpu'/'cuda')

        # Compute prediction error
        pred = model(X)                     # ?????????????????????????????????????????? (Forward propagation)
        loss = loss_fn(pred, y)             # ?????? loss

        # Backpropagation
        optimizer.zero_grad()               # ????????????
        loss.backward()                     # ??? loss ????????????(backpropagate)
        optimizer.step()                    # ????????????

        # Compute the total loss and total correct
        total_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Compute the average loss among all batches and the accuracy of an epoch
    total_loss /= num_batches
    correct /= size
    return correct, total_loss

# Training
train_loss, train_acc = [], []
for epoch in range(epochs):
    print(f"Epoch: {(epoch + 1):>2d} / {epochs:>2d}\n-------------------------------")
    acc, loss = train(train_dataloader, model, loss_fn, optimizer)
    train_acc.append(100 * acc)
    train_loss.append(loss)
    print(f"Training Error:\n\tAcc: {train_acc[-1]:>3.1f}% | loss: {train_loss[-1]:>9f}\n")


# Save the model
def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)
save_model(model,"./weight.pth")