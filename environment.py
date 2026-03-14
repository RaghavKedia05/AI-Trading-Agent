import numpy as np


def get_state(data, index):
    """
    Extract the state representation for the RL agent.
    """

    row = data.iloc[index]

    return np.array([
        float(row['Close']),
        float(row['SMA_5']),
        float(row['SMA_20']),
        float(row['Returns'])
    ])


class TradingEnvironment:

    def __init__(self, data):

        self.data = data
        self.initial_balance = 10000

        self.reset()

    def reset(self):

        self.balance = self.initial_balance
        self.holdings = 0
        self.index = 0

        return get_state(self.data, self.index)

    def step(self, action):

        price = float(self.data.iloc[self.index]['Close'])

        reward = 0

        # BUY
        if action == 1 and self.balance >= price:
            shares = self.balance // price
            self.holdings += shares
            self.balance -= shares * price

        # SELL
        elif action == 2 and self.holdings > 0:
            self.balance += self.holdings * price
            self.holdings = 0

        self.index += 1

        done = self.index >= len(self.data) - 1

        if done:
            portfolio_value = self.balance + self.holdings * price
            reward = portfolio_value - self.initial_balance

        next_state = None if done else get_state(self.data, self.index)

        return next_state, reward, done, {}
