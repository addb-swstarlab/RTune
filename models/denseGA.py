from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np

class RocksDBDataset(Dataset):
    def __init__(self, X, y):
        super(RocksDBDataset, self).__init__()
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#         self.fc1 = nn.Sequential(nn.Linear(22, 16), nn.ReLU(16))
        self.fc1 = nn.Sequential(nn.Linear(22, 64), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64, 16), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(16, 1))

    def forward(self, x):
        x = x.float()
        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        h3 = self.fc3(h2)
        return h3

def train(model, train_loader, lr):
    ## Construct optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    ## Set phase
    model.train()
    
    ## Train start
    total_loss = 0.
    for data, target in train_loader:
        ## data.shape = (batch_size, 22)
        ## target.shape = (batch_size, 1)
        ## initilize gradient
        optimizer.zero_grad()
        ## predict
        output = model(data) # output.shape = (batch_size, 1)
        ## loss
        loss = F.mse_loss(output, target)
        ## backpropagation
        loss.backward()
        optimizer.step()
        ## Logging
        total_loss += loss.item()
    total_loss /= len(train_loader)
    return total_loss
    

def valid(model, valid_loader):
    ## Set phase
    model.eval()
    
    ## Valid start    
    total_loss = 0.
    with torch.no_grad():
        for data, target in valid_loader:
            output = model(data)
            loss = F.mse_loss(output, target) # mean squared error
            total_loss += loss.item()
    total_loss /= len(valid_loader)
    return total_loss

def fitness_function(solution, model):
    Dataset_sol = RocksDBDataset(solution, np.zeros((len(solution), 1)))
    loader_sol = DataLoader(Dataset_sol, shuffle=False, batch_size=8)
    
    ## Set phase
    model.eval()
    
    ## Predict
    fitness = []
    with torch.no_grad():
        for data, _ in loader_sol:
            fitness_batch = model(data)
            fitness_batch = fitness_batch.detach().cpu().numpy() # scaled.shape = (batch_size, 1)
            fitness_batch = fitness_batch.ravel() # fitness_batch.shape = (batch_size,)
            fitness_batch = fitness_batch.tolist() # numpy -> list
            fitness += fitness_batch # [1,2] += [3,4,5] --> [1,2,3,4,5]
    
    return fitness