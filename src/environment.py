import numpy as np

class TradingEnvironment:

    def __init__(self, data):

        self.data = data
        self.initial_balance = 10000

        self.reset()

    def reset(self):

        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0

        self.portfolio_values = []

        return self._get_state()

    def _get_state(self):

        row = self.data.iloc[self.current_step]

        return np.array([
            row['Close'],
            row['Return'],
            row['MA10'],
            row['MA50'],
            self.balance,
            self.shares
        ])

    def step(self, action):

        price = self.data.iloc[self.current_step]['Close']

        prev_value = self.balance + self.shares * price

        # Buy
        if action == 0 and self.balance >= price:

            self.shares += 1
            self.balance -= price

        # Sell
        elif action == 1 and self.shares > 0:

            self.shares -= 1
            self.balance += price

        self.current_step += 1

        done = self.current_step >= len(self.data) - 1

        new_price = self.data.iloc[self.current_step]['Close']

        current_value = self.balance + self.shares * new_price

        reward = current_value - prev_value

        self.portfolio_values.append(current_value)

        return self._get_state(), reward, done
