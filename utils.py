import yfinance as yf

def get_stock_data(symbol):

    data = yf.download(symbol, period="2y")

    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Returns'] = data['Close'].pct_change()

    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data
