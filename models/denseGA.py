from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import copy

class RocksDBDataset(Dataset):
    def __init__(self, X, y):
        super(RocksDBDataset, self).__init__()
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

class SingleNet(nn.Module):
    def __init__(self):
        super(SingleNet, self).__init__()
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

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiNet(nn.Module):
    def __init__(self, de_time, de_rate, de_waf, de_sa, b):
        super(MultiNet, self).__init__()
        self.de_ex = np.array([de_time, de_rate, de_waf, de_sa])
        self.de_time = de_time
        self.de_rate = de_rate
        self.de_waf = de_waf
        self.de_sa = de_sa
        self.b = np.array(b) # list of balance values for calculating score [a, b, c, d]
        self.fc1 = nn.Sequential(nn.Linear(22, 64), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64, 16), nn.ReLU())
        self.fc_ex = clones(nn.Sequential(nn.Linear(16,1)), 4)
    
    def get_score(self, h_ex):
        self.score_ex = [self.de_ex[0]/h_ex[0], h_ex[1]/self.de_ex[1], self.de_ex[2]/h_ex[2], self.de_ex[3]/h_ex[3]]
        return sum(_*self.b[i] for i, _ in enumerate(self.score_ex))

    def forward(self, x):
        x = x.float()
        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        self.time = self.fc_ex[0](h2)
        self.rate = self.fc_ex[1](h2)
        self.waf = self.fc_ex[2](h2)
        self.sa = self.fc_ex[3](h2)
        self.score = self.get_score([self.time, self.rate, self.waf, self.sa])
        return self.score


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