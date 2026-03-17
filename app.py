import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Trading Agent", layout="wide")

st.title("📈 AI Trading Agent")

# User Input
stock = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", "AAPL")

if st.button("Run Analysis"):
    data = yf.download(stock, period="1y")

    if data.empty:
        st.error("Invalid stock symbol")
    else:
        st.success(f"Data loaded for {stock}")

        # Moving Averages
        data["MA50"] = data["Close"].rolling(window=50).mean()
        data["MA200"] = data["Close"].rolling(window=200).mean()

        # Plot
        fig, ax = plt.subplots()
        ax.plot(data["Close"], label="Close Price")
        ax.plot(data["MA50"], label="MA50")
        ax.plot(data["MA200"], label="MA200")
        ax.legend()

        st.pyplot(fig)

        # Simple Signal
        if data["MA50"].iloc[-1] > data["MA200"].iloc[-1]:
            st.success("📊 Signal: BUY")
        else:
            st.error("📉 Signal: SELL")
