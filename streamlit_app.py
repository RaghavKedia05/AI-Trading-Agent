import streamlit as st
import yfinance as yf
import pandas as pd

from environment import TradingEnvironment
from agent import DQNAgent

st.title("AI Trading Agent")

st.write("Enter a stock symbol to get an AI trading recommendation.")

# Stock input
symbol = st.text_input("Enter Stock Symbol (Example: AAPL, TCS, INFY)", "AAPL")

# Automatically handle Indian stocks
if symbol.isalpha() and symbol.isupper() and len(symbol) <= 10:
    indian_symbol = symbol + ".NS"
else:
    indian_symbol = symbol

if st.button("Run AI Agent"):

    # Try downloading US stock first
    data = yf.download(symbol, period="2y")

    # If empty try NSE symbol
    if data.empty:
        data = yf.download(indian_symbol, period="2y")

    # If still empty show error
    if data.empty:
        st.error("Invalid stock symbol. Try examples like AAPL, MSFT, TCS, INFY.")
        st.stop()

    # Feature engineering
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Returns'] = data['Close'].pct_change()

    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    st.subheader("Stock Data Preview")
    st.dataframe(data.tail())

    # Run trading environment
    env = TradingEnvironment(data)
    agent = DQNAgent(state_size=4, action_size=3)

    state = env.reset()

    action = agent.act(state)

    actions = {
        0: "HOLD",
        1: "BUY",
        2: "SELL"
    }

    st.subheader("AI Recommendation")

    st.success(actions[action])
