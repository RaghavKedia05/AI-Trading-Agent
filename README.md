# 🤖 Agentic AI Trading Agent (Deep Q-Learning)

An **AI-powered autonomous trading agent** built using **Reinforcement Learning and Deep Q-Networks (DQN)**.
The agent learns how to trade stocks by interacting with a simulated trading environment and optimizing decisions to maximize profit.

This project demonstrates how **Agentic AI systems can perceive market conditions, make decisions, and continuously improve their strategy through experience**.

---

# 📌 Project Overview

Traditional algorithmic trading systems rely on predefined rules.
In contrast, this project implements an **AI agent that learns trading strategies automatically** using reinforcement learning.

The agent:

1. Observes the stock market state.
2. Chooses an action (Buy / Sell / Hold).
3. Receives feedback through rewards.
4. Improves its decision-making through training.

The system is trained on historical stock market data and learns how to maximize portfolio returns.

---

# 🧠 Core Concepts Used

This project combines several important AI and Machine Learning concepts:

### Agentic AI

Autonomous AI agents that can perceive, reason, act, and learn from their environment.

### Reinforcement Learning

A learning paradigm where agents improve through trial and error interactions with an environment.

### Deep Q-Learning (DQN)

A neural network-based approach that approximates the **Q-value function** for optimal decision making.

### Experience Replay

Stores past experiences and reuses them during training to stabilize learning.

### Exploration vs Exploitation

The agent balances exploring new strategies and exploiting learned profitable actions.

---

# 🏗️ System Architecture

The AI trading system consists of the following components:

```
Market Data → Feature Engineering → Trading Environment
        ↓
     State Representation
        ↓
      DQN Agent
        ↓
Neural Network (Q-Value Prediction)
        ↓
   Action Selection
(Buy / Sell / Hold)
        ↓
     Reward Signal
        ↓
      Learning
```

---

# 📊 State Representation

At every time step the agent observes a **state vector** containing key market indicators:

* Stock closing price
* 5-day Simple Moving Average (SMA)
* 20-day Simple Moving Average (SMA)
* Daily return percentage

These features help the agent detect **short-term and long-term trends**.

---

# 🎮 Action Space

The agent can perform three possible actions:

| Action | Description                             |
| ------ | --------------------------------------- |
| HOLD   | Do nothing                              |
| BUY    | Purchase shares using available balance |
| SELL   | Sell currently held shares              |

---

# 💰 Reward Function

The agent’s objective is to maximize trading profit.

Reward is calculated as:

```
Reward = Final Portfolio Value − Initial Balance
```

The agent learns to select actions that lead to higher long-term rewards.

---

# 🧠 Deep Q-Network

The agent uses a **neural network to estimate Q-values** for each action.

Architecture:

```
Input Layer  : 4 Features
Hidden Layer : 64 neurons
Hidden Layer : 64 neurons
Output Layer : 3 neurons (Buy, Sell, Hold)
```

Activation function: **ReLU**

The network predicts expected future rewards for each action given the current market state.

---

# 📂 Project Structure

```
ai_trading_agent
│
├── train.py          # Main training script
├── agent.py          # DQN agent implementation
├── model.py          # Neural network architecture
├── environment.py    # Trading environment simulation
├── utils.py          # State representation utilities
├── requirements.txt  # Project dependencies
└── README.md
```

---

# ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/yourusername/ai-trading-agent.git
cd ai-trading-agent
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

Dependencies include:

* yfinance
* pandas
* numpy
* torch

---

# ▶️ Running the Project

Start training the trading agent:

```
python train.py
```

The system will:

1. Download historical stock data
2. Train the reinforcement learning agent
3. Simulate trading episodes
4. Output the total reward per episode

---

# 📈 Example Output

```
Episode 1/500 Reward: -9800
Episode 2/500 Reward: -9650
...
Episode 498/500 Reward: 2100
Episode 499/500 Reward: 4200
Episode 500/500 Reward: 410
```

After training, the agent is evaluated on a trading simulation.

Example:

```
Final Balance: $10178.89
Total Profit: $178.89
```

---

# 🚀 Key Features

* Reinforcement learning trading agent
* Deep Q-Network implemented with PyTorch
* Autonomous decision making
* Experience replay for stable learning
* Real stock market data from Yahoo Finance
* Modular project architecture

---

# ⚠️ Limitations

This project is intended for **educational and research purposes**.

Limitations include:

* Uses a small set of technical indicators
* No transaction fees included
* Trained on a single stock
* Simplified reward function
* Not suitable for real financial trading

---

# 🔮 Future Improvements

Possible enhancements include:

* Adding advanced indicators (RSI, MACD, Bollinger Bands)
* Implementing Double DQN or Dueling DQN
* Training on multiple stocks
* Portfolio optimization
* Risk management strategies
* Real-time trading simulation
* Visualization of training performance

---

# 🎓 Learning Outcomes

Through this project you will understand:

* How reinforcement learning works in financial markets
* How autonomous AI agents interact with environments
* How neural networks estimate action values
* How experience replay stabilizes training


