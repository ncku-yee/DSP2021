import numpy as np
import torch, os, random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time
from sklearn.preprocessing import StandardScaler
from torch import optim

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

norm = True
traindata, trainlabel = np.load("./traindata.npy"), np.load("./trainlabel.npy")
testdata = np.load("./anomalytestdata.npy")
anomalydata = np.load("./anomaly_sample.npy")
print(f"traindata.shape = {traindata.shape}")
print(f"testdata.shape = {testdata.shape}")
print(f"anomalydata.shape = {anomalydata.shape}")

norm = True
if norm:
    print("Preprocessing Data...")
    scalerA = StandardScaler()
    scalerA.fit(traindata[:,0,:])
    tmp = scalerA.transform(traindata[:,0,:])
    traindata[:,0,:] = tmp
    tmp = scalerA.transform(testdata[:,0,:])
    testdata[:,0,:] = tmp
    tmp = scalerA.transform(anomalydata[:,0,:])
    anomalydata[:,0,:] = tmp

    scalerB = StandardScaler()
    scalerB.fit(traindata[:,1,:])
    tmp = scalerB.transform(traindata[:,1,:])
    traindata[:,1,:] = tmp
    tmp = scalerB.transform(testdata[:,1,:])
    testdata[:,1,:] = tmp
    tmp = scalerB.transform(anomalydata[:,1,:])
    anomalydata[:,1,:] = tmp
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
batch_size = 128
lr = 3e-3
epochs = 30

# Load the dataset.
train_dataloader = DataLoader(RawDataset(traindata, trainlabel),
                                batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(RawDataset(testdata),
                                batch_size = 1, shuffle = False)

class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        # define your arch
        self.encoder = nn.Sequential(
            nn.Linear(16000, 8000),
            nn.Tanh(),
            nn.Linear(8000, 4000),
            nn.Tanh(),
            nn.Linear(4000, 2000),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2000, 4000),
            nn.Tanh(),
            nn.Linear(4000, 8000),
            nn.Tanh(),
            nn.Linear(8000, 16000),
            nn.Sigmoid()
        )

    def forward(self, x):
        # define your forward
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

model_autoencoder = AutoEncoder().to(device)                        # 使用內建的 model
optimizer = optim.Adam(model_autoencoder.parameters(), lr = lr)     # 選擇你想用的 optimizer
# Loss function
criterion = nn.MSELoss()                                            # 選擇想用的 loss function

# Show the architecture of model
for name, module in model_autoencoder.named_children():
    print(module)

# Each training loop will call train function once.
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)          # training dataset 的大小
    num_batches = len(dataloader)           # batch size 的大小
    total_loss = 0                          # 紀錄 loss 的總和
    model.train()                           # model 在 training 階段
    for batch, X in enumerate(dataloader):
        X = X.to(device)                    # 將訓練資料和標籤轉成當前 torch 所指定的 device('cpu'/'cuda')

        # Compute prediction error
        codes, decoded = model(X)           # 將訓練資料輸入至模型進行訓練 (Forward propagation)
        loss = loss_fn(decoded, X)          # 計算 loss

        # Backpropagation
        optimizer.zero_grad()               # 清空梯度
        loss.backward()                     # 將 loss 反向傳播(backpropagate)
        optimizer.step()                    # 更新權重

        # Compute the total loss and total correct
        total_loss += loss.item()

    # Compute the average loss among all batches and the accuracy of an epoch
    total_loss /= num_batches
    return total_loss

# Training
train_loss = []
for epoch in range(epochs):
    print(f"Epoch: {(epoch + 1):>2d} / {epochs:>2d}\n-------------------------------")
    loss = train(train_dataloader, model_autoencoder, criterion, optimizer)
    train_loss.append(loss)
    print(f"Training Error:\n\tloss: {train_loss[-1]:>9f}\n")

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
            nn.Conv1d(40, 80, kernel_size=9, stride=5),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.clf = nn.Linear(5120, 3)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        output = self.clf(x)
        return output

# Save the model
def save_model(model, filename):
    state = model.state_dict()
    torch.save(state, filename)

# Load the model
def load_model(model,filename):
    model.load_state_dict(torch.load(filename))
    return model

save_model(model_autoencoder, "autoencoder-best.pth")
model_autoencoder = AutoEncoder().to(device)    
model_autoencoder = load_model(model_autoencoder, "autoencoder-best.pth")
model_clf = MyDSPNet().to(device)
model_clf = load_model(model_clf, "./weight.pth")       # Task 1 model

with torch.no_grad():
    with open("./result.csv","w") as f:
        f.write("id,category\n")
        for idx, input in enumerate(testdata):
            input = input.astype('float32')
            x = torch.from_numpy(input.reshape(2, 16000)).to(device)
            encode, decode = model_autoencoder (x)
            decode = decode.cpu().numpy()
            mse = np.sum(np.square(input - decode))     # Sum of Square Error
            # Abnormal
            if mse > 17000:
                f.write("%d,%d\n"%(idx,3))
            # Normal
            else:
                x = torch.from_numpy(testdata[idx].astype('float32').reshape(1, 2, 16000)).to(device)
                output = model_clf(x)
                pred = output.argmax(dim=1, keepdim=True)
                f.write("%d,%d\n"%(idx,pred.item()))
            # print(mse)