import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

from environment import TradingEnvironment
from agent import DQNAgent

# Page settings
st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Trading Agent Dashboard")

st.markdown("AI-powered trading signal generator")

# Sidebar
st.sidebar.header("Stock Settings")

symbol = st.sidebar.text_input("Stock Symbol", "AAPL")

period = st.sidebar.selectbox(
    "Select Time Period",
    ["3mo", "6mo", "1y", "2y"]
)

run_ai = st.sidebar.button("Run AI Analysis")

# Download data
data = yf.download(symbol, period=period)

if data.empty:
    st.error("Invalid stock symbol. Try AAPL, MSFT, TCS.NS, INFY.NS")
    st.stop()

# Feature engineering
data['SMA_5'] = data['Close'].rolling(5).mean()
data['SMA_20'] = data['Close'].rolling(20).mean()
data['Returns'] = data['Close'].pct_change()

data.dropna(inplace=True)
data.reset_index(inplace=True)

# Top metrics
col1, col2, col3 = st.columns(3)

current_price = round(data['Close'].iloc[-1], 2)
high_price = round(data['High'].max(), 2)
low_price = round(data['Low'].min(), 2)

col1.metric("Current Price", f"${current_price}")
col2.metric("Period High", f"${high_price}")
col3.metric("Period Low", f"${low_price}")

# Stock chart
st.subheader("📊 Stock Price Chart")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=data['Date'],
        y=data['Close'],
        name="Price",
        line=dict(width=3)
    )
)

fig.add_trace(
    go.Scatter(
        x=data['Date'],
        y=data['SMA_5'],
        name="SMA 5",
        line=dict(dash="dot")
    )
)

fig.add_trace(
    go.Scatter(
        x=data['Date'],
        y=data['SMA_20'],
        name="SMA 20",
        line=dict(dash="dash")
    )
)

fig.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# AI section
if run_ai:

    st.subheader("🤖 AI Trading Signal")

    env = TradingEnvironment(data)
    agent = DQNAgent(state_size=4, action_size=3)

    state = env.reset()
    action = agent.act(state)

    actions = {
        0: "HOLD",
        1: "BUY",
        2: "SELL"
    }

    signal = actions[action]

    if signal == "BUY":
        st.success("🟢 BUY SIGNAL")
    elif signal == "SELL":
        st.error("🔴 SELL SIGNAL")
    else:
        st.warning("🟡 HOLD POSITION")

# Data preview
with st.expander("View Raw Data"):
    st.dataframe(data.tail(20))
