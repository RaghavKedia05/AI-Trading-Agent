import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="AI Trading Agent", layout="wide")

st.title("📈 AI Trading Agent")

stock = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", "AAPL")

if st.button("Run Analysis"):
    try:
        data = yf.download(stock, period="1y")

        if data.empty:
            st.error("Invalid stock symbol or no data found")
            st.stop()

        # Indicators
        data["MA50"] = data["Close"].rolling(50).mean()
        data["MA200"] = data["Close"].rolling(200).mean()

        # Plot
        fig, ax = plt.subplots()
        ax.plot(data["Close"], label="Close Price")
        ax.plot(data["MA50"], label="MA50")
        ax.plot(data["MA200"], label="MA200")
        ax.legend()

        st.pyplot(fig)

        # Trading Logic
        if data["MA50"].iloc[-1] > data["MA200"].iloc[-1]:
            st.success("📊 AI Signal: BUY")
        elif data["MA50"].iloc[-1] < data["MA200"].iloc[-1]:
            st.error("📉 AI Signal: SELL")
        else:
            st.warning("⚖️ AI Signal: HOLD")

    except Exception as e:
        st.error(f"Error: {e}")
