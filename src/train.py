import pandas as pd
import matplotlib.pyplot as plt

from environment import TradingEnvironment
from agent import Agent

data = pd.read_csv("data/stock_data.csv")

env = TradingEnvironment(data)

agent = Agent(6,3)

episodes = 20

rewards = []

for e in range(episodes):

    state = env.reset()

    total_reward = 0

    while True:

        action = agent.act(state)

        next_state, reward, done = env.step(action)

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        total_reward += reward

        if done:
            break

        if len(agent.memory) > 32:
            agent.replay(32)

    rewards.append(total_reward)

    print("Episode:", e, "Reward:", total_reward)

plt.plot(rewards)

plt.title("Training Reward Curve")

plt.show()
