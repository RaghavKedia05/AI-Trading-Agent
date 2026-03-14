import streamlit as st
from utils import get_stock_data
from environment import TradingEnvironment
from agent import DQNAgent

st.title("AI Trading Agent")

symbol = st.text_input("Enter Stock Symbol", "AAPL")

if st.button("Run AI Agent"):

    data = get_stock_data(symbol)

    env = TradingEnvironment(data)

    agent = DQNAgent(state_size=4, action_size=3)

    state = env.reset()

    action = agent.act(state)

    actions = {
        0: "HOLD",
        1: "BUY",
        2: "SELL"
    }

    st.success(f"AI Recommendation: {actions[action]}")
