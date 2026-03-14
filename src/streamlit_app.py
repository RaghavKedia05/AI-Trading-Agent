import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/stock_data.csv")

st.title("AI Trading Agent Dashboard")

st.subheader("Stock Price")

fig, ax = plt.subplots()

ax.plot(data['Close'])

st.pyplot(fig)

st.write("This dashboard visualizes stock prices used for training the AI trading agent.")
