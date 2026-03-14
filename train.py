import yfinance as yf
import pandas as pd

from environment import TradingEnvironment
from agent import DQNAgent

print("Downloading stock data...")

symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2025-02-14"

data = yf.download(symbol, start=start_date, end=end_date)

data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['Returns'] = data['Close'].pct_change()

data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

env = TradingEnvironment(data)

state_size = 4
action_size = 3

agent = DQNAgent(state_size, action_size)

batch_size = 32
episodes = 500

print("Starting training...")

for episode in range(episodes):

    state = env.reset()
    done = False
    total_reward = 0

    while not done:

        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        total_reward += reward

    agent.replay(batch_size)

    print(f"Episode {episode+1}/{episodes} Reward: {total_reward}")

print("Training Complete!")
