import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

from environment import TradingEnvironment
from agent import DQNAgent

# Page config
st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Trading Agent")
st.write("Deep Reinforcement Learning based stock trading assistant")

# Sidebar
st.sidebar.header("Stock Settings")

symbol = st.sidebar.text_input(
    "Enter Stock Symbol",
    "AAPL"
)

period = st.sidebar.selectbox(
    "Select Time Period",
    ["3mo", "6mo", "1y", "2y"]
)

run_ai = st.sidebar.button("Run AI Analysis")


# Load stock data
@st.cache_data
def load_data(symbol, period):

    data = yf.download(symbol, period=period)

    if data.empty:
        return None

    data.reset_index(inplace=True)

    data['SMA_5'] = data['Close'].rolling(5).mean()
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['Returns'] = data['Close'].pct_change()

    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data


data = load_data(symbol, period)

if data is None:
    st.error("Invalid stock symbol. Try AAPL, TSLA, TCS.NS, INFY.NS")
    st.stop()


# Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Current Price", f"${round(data['Close'].iloc[-1],2)}")
col2.metric("High", f"${round(data['High'].max(),2)}")
col3.metric("Low", f"${round(data['Low'].min(),2)}")


# Chart
st.subheader("Stock Price Chart")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=data['Date'],
        y=data['Close'],
        name="Price"
    )
)

fig.add_trace(
    go.Scatter(
        x=data['Date'],
        y=data['SMA_5'],
        name="SMA 5"
    )
)

fig.add_trace(
    go.Scatter(
        x=data['Date'],
        y=data['SMA_20'],
        name="SMA 20"
    )
)

fig.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(fig, use_container_width=True)


# Run AI
if run_ai:

    st.subheader("AI Trading Signal")

    try:

        env = TradingEnvironment(data)

        agent = DQNAgent(
            state_size=4,
            action_size=3
        )

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
            st.warning("🟡 HOLD")

    except Exception as e:

        st.error("AI model failed to run. Try another stock.")


# Data viewer
with st.expander("View Raw Data"):
    st.dataframe(data.tail(20))
