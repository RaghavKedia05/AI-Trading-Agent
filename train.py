from agent import DQNAgent
from environment import TradingEnvironment
from utils import get_stock_data

data = get_stock_data("AAPL")

env = TradingEnvironment(data)

state_size = 4
action_size = 3

agent = DQNAgent(state_size, action_size)

episodes = 200
batch_size = 32

for episode in range(episodes):

    state = env.reset()
    done = False

    while not done:

        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)

        agent.remember(state, action, reward, next_state, done)

        state = next_state

    agent.replay(batch_size)

    print(f"Episode {episode+1}/{episodes} completed")

print("Training finished")
torch.save(agent.model.state_dict(), "trained_model.pth")
