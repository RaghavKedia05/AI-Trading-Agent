import yfinance as yf
import pandas as pd

def download_data():

    data = yf.download("AAPL", start="2015-01-01", end="2024-01-01")

    data = data[['Close']]

    data['Return'] = data['Close'].pct_change()
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA50'] = data['Close'].rolling(50).mean()

    data.dropna(inplace=True)

    data.to_csv("data/stock_data.csv")

    print("Stock data saved.")


if __name__ == "__main__":
    download_data()
