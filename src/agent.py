import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQN(nn.Module):

    def __init__(self, state_size, action_size):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):

        return self.network(x)


class Agent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=5000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.model = DQN(state_size, action_size)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.loss_fn = nn.MSELoss()

    def act(self, state):

        if np.random.rand() < self.epsilon:

            return random.randrange(self.action_size)

        state = torch.FloatTensor(state)

        q_values = self.model(state)

        return torch.argmax(q_values).item()

    def remember(self, *args):

        self.memory.append(args)

    def replay(self, batch_size):

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:

            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)

            target = reward

            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()

            target_f = self.model(state).detach().clone()

            target_f[action] = target

            output = self.model(state)

            loss = self.loss_fn(output, target_f)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
