# train.py

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Load data
data = yf.download("AAPL", start="2020-01-01", end="2025-02-14")

data['SMA_5'] = data['Close'].rolling(5).mean()
data['SMA_20'] = data['Close'].rolling(20).mean()
data['Returns'] = data['Close'].pct_change()
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL"}

def get_state(data, index):
    return np.array([
        float(data.loc[index, 'Close']),
        float(data.loc[index, 'SMA_5']),
        float(data.loc[index, 'SMA_20']),
        float(data.loc[index, 'Returns'])
    ])

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = DQN()

# (skip full training here for brevity OR reduce episodes)
torch.save(model.state_dict(), "model.pth")

print("Model saved!")
