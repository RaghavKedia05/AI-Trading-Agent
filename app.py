import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model
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

# Load trained model
model = DQN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL"}

def get_state(data, index):
    return np.array([
        float(data.loc[index, 'Close']),
        float(data.loc[index, 'SMA_5']),
        float(data.loc[index, 'SMA_20']),
        float(data.loc[index, 'Returns'])
    ])

st.title("📈 AI Trading Agent")

stock = st.text_input("Stock", "AAPL")

if st.button("Predict"):
    data = yf.download(stock, period="1y")

    data['SMA_5'] = data['Close'].rolling(5).mean()
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)

    if data.empty:
        st.error("No data found")
        st.stop()

    state = get_state(data, len(data)-1)
    state = torch.FloatTensor(state).unsqueeze(0)

    with torch.no_grad():
        action = torch.argmax(model(state)).item()

    st.success(f"AI Decision: {ACTIONS[action]}")

    # Plot
    fig, ax = plt.subplots()
    ax.plot(data['Close'], label="Price")
    ax.legend()
    st.pyplot(fig)
