"""
utils.py
--------
Data loading, preprocessing, and visualisation helpers.

Supports:
  • Downloading real data via yfinance (if installed)
  • Generating synthetic OHLCV data for offline testing
  • Train/test splitting
  • Plotting training curves and portfolio performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import os


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data(ticker: str = "AAPL",
              start:  str = "2018-01-01",
              end:    str = "2023-12-31",
              use_synthetic: bool = False) -> pd.DataFrame:
    """
    Load historical OHLCV data.

    Args:
        ticker        : Stock ticker symbol (e.g. 'AAPL').
        start / end   : Date range strings 'YYYY-MM-DD'.
        use_synthetic : If True (or yfinance unavailable), generate synthetic data.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    if not use_synthetic:
        try:
            import yfinance as yf
            df = yf.download(ticker, start=start, end=end, progress=False)
            df.dropna(inplace=True)
            df.reset_index(inplace=True)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            print(f"[Data] Downloaded {len(df)} rows for {ticker} "
                  f"({start} → {end})")
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            print(f"[Data] yfinance failed ({e}). Falling back to synthetic data.")

    return generate_synthetic_data(n_days=1500, seed=42)


def generate_synthetic_data(n_days: int = 1500,
                             start_price: float = 150.0,
                             volatility: float  = 0.015,
                             drift: float       = 0.0003,
                             seed: int          = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic OHLCV price series using geometric Brownian motion
    with occasional regime changes (bull / bear / sideways).

    Args:
        n_days      : Number of trading days to generate.
        start_price : Initial closing price.
        volatility  : Daily volatility (std of log-returns).
        drift       : Daily drift (mean of log-returns).
        seed        : Random seed for reproducibility.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    rng = np.random.default_rng(seed)

    prices = [start_price]
    regimes = []

    # Regime model: switch every ~120 days
    regime_len  = 120
    regime_types = ["bull", "bear", "sideways"]
    current_regime = "bull"
    days_in_regime = 0

    for i in range(1, n_days):
        days_in_regime += 1
        if days_in_regime >= regime_len:
            current_regime = rng.choice(regime_types)
            days_in_regime = 0

        regime_drift = {"bull": drift + 0.0005,
                        "bear": drift - 0.0008,
                        "sideways": 0.0}[current_regime]

        log_return = rng.normal(regime_drift, volatility)
        new_price  = prices[-1] * np.exp(log_return)
        new_price  = max(new_price, 1.0)      # floor at $1
        prices.append(new_price)
        regimes.append(current_regime)

    closes = np.array(prices, dtype=np.float32)

    # Derive OHLV from close
    daily_range = np.abs(rng.normal(0, volatility * closes, size=n_days)) + 0.01
    opens       = closes * (1 + rng.normal(0, 0.003, size=n_days))
    highs       = closes + daily_range * rng.uniform(0.3, 1.0, n_days)
    lows        = closes - daily_range * rng.uniform(0.3, 1.0, n_days)
    lows        = np.clip(lows, 0.5, None)
    volumes     = rng.integers(1_000_000, 50_000_000, size=n_days).astype(float)

    # Build date index
    base_date = datetime(2018, 1, 2)
    dates = [base_date + timedelta(days=i) for i in range(n_days)]

    df = pd.DataFrame({
        "Date"  : dates,
        "Open"  : opens.round(2),
        "High"  : highs.round(2),
        "Low"   : lows.round(2),
        "Close" : closes.round(2),
        "Volume": volumes,
    })
    print(f"[Data] Generated {n_days} days of synthetic OHLCV data.")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def train_test_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """Split DataFrame into train and test sets (time-ordered)."""
    split = int(len(df) * train_ratio)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df  = df.iloc[split:].reset_index(drop=True)
    print(f"[Data] Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_results(episode_rewards: list,
                          portfolio_values: list,
                          losses: list,
                          save_path: str = "training_results.png"):
    """
    Plot training curves: episode rewards, portfolio growth, and loss.

    Args:
        episode_rewards  : List of total reward per episode.
        portfolio_values : List of final portfolio value per episode.
        losses           : List of training losses.
        save_path        : File path to save the figure.
    """
    fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    _style = dict(color="#58a6ff", linewidth=1.5)
    _avg_style = dict(color="#f78166", linewidth=2.0, linestyle="--")
    _fill_alpha = 0.15

    def _smooth(arr, w=20):
        if len(arr) < w:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="valid")

    # ---- Panel 1: Episode Reward ----
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episode_rewards, alpha=0.4, **_style)
    sm = _smooth(episode_rewards)
    ax1.plot(range(len(sm)), sm, **_avg_style, label="Smoothed (20-ep avg)")
    ax1.fill_between(range(len(episode_rewards)), episode_rewards,
                     alpha=_fill_alpha, color="#58a6ff")
    ax1.axhline(0, color="#8b949e", linewidth=0.7, linestyle=":")
    ax1.set_title("Episode Reward", color="white", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Episode", color="#8b949e")
    ax1.set_ylabel("Total Reward", color="#8b949e")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=9)

    # ---- Panel 2: Portfolio Value ----
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(portfolio_values, alpha=0.4, **_style)
    sm2 = _smooth(portfolio_values)
    ax2.plot(range(len(sm2)), sm2, **_avg_style, label="Smoothed")
    ax2.fill_between(range(len(portfolio_values)), portfolio_values,
                     alpha=_fill_alpha, color="#58a6ff")
    ax2.set_title("Final Portfolio Value per Episode", color="white",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Episode", color="#8b949e")
    ax2.set_ylabel("Portfolio ($)", color="#8b949e")
    ax2.legend(facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=9)

    # ---- Panel 3: Training Loss ----
    ax3 = fig.add_subplot(gs[1, 0])
    if losses:
        ax3.plot(losses, alpha=0.4, color="#3fb950", linewidth=1.0)
        sm3 = _smooth(losses, w=50)
        ax3.plot(range(len(sm3)), sm3, color="#ffa657", linewidth=2.0,
                 linestyle="--", label="Smoothed (50-step avg)")
        ax3.set_yscale("log")
    ax3.set_title("Training Loss (Huber)", color="white",
                  fontsize=12, fontweight="bold")
    ax3.set_xlabel("Update Step", color="#8b949e")
    ax3.set_ylabel("Loss", color="#8b949e")
    ax3.legend(facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=9)

    # ---- Panel 4: ROI Distribution ----
    ax4 = fig.add_subplot(gs[1, 1])
    initial = 10_000.0
    rois = [(v - initial) / initial * 100 for v in portfolio_values]
    ax4.hist(rois, bins=30, color="#58a6ff", alpha=0.7, edgecolor="#1f6feb")
    ax4.axvline(np.mean(rois), color="#f78166", linewidth=2,
                linestyle="--", label=f"Mean ROI: {np.mean(rois):.1f}%")
    ax4.axvline(0, color="#8b949e", linewidth=1, linestyle=":")
    ax4.set_title("ROI Distribution", color="white",
                  fontsize=12, fontweight="bold")
    ax4.set_xlabel("ROI (%)", color="#8b949e")
    ax4.set_ylabel("Frequency", color="#8b949e")
    ax4.legend(facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=9)

    # Global styling
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    fig.suptitle("AI Trading Agent — DQN Training Dashboard",
                 color="white", fontsize=15, fontweight="bold", y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Training results saved → {save_path}")


def plot_backtest(df: pd.DataFrame,
                  portfolio_history: list,
                  trade_log: list,
                  initial_balance: float = 10_000.0,
                  save_path: str = "backtest_results.png"):
    """
    Plot backtest: price series with buy/sell markers + portfolio vs buy-and-hold.
    """
    closes = df["Close"].values
    n      = min(len(closes), len(portfolio_history))
    steps  = range(n)

    # Buy-and-hold benchmark
    shares_bh    = initial_balance / closes[0]
    bah_portfolio = [shares_bh * closes[min(i, len(closes)-1)] for i in steps]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), facecolor="#0d1117",
                             gridspec_kw={"height_ratios": [2, 1], "hspace": 0.3})

    # ---- Upper: Price + trades ----
    ax = axes[0]
    ax.set_facecolor("#161b22")
    ax.plot(closes[:n], color="#58a6ff", linewidth=1.2, label="Close Price")

    buys  = [t for t in trade_log if t["action"] == "BUY"]
    sells = [t for t in trade_log if t["action"] == "SELL"]

    if buys:
        bx = [t["step"] for t in buys]
        by = [t["price"] for t in buys]
        ax.scatter(bx, by, marker="^", color="#3fb950", s=60, zorder=5,
                   label=f"BUY  ({len(buys)})")

    if sells:
        sx = [t["step"] for t in sells]
        sy = [t["price"] for t in sells]
        ax.scatter(sx, sy, marker="v", color="#f78166", s=60, zorder=5,
                   label=f"SELL ({len(sells)})")

    ax.set_title("Price Series + Trading Signals", color="white",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Price ($)", color="#8b949e")
    ax.legend(facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#c9d1d9", fontsize=9)

    # ---- Lower: Portfolio vs BaH ----
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    ax2.plot(portfolio_history[:n], color="#58a6ff", linewidth=1.5,
             label="DQN Agent")
    ax2.plot(bah_portfolio[:n],     color="#ffa657", linewidth=1.5,
             linestyle="--", label="Buy & Hold")
    ax2.axhline(initial_balance, color="#8b949e", linewidth=0.8, linestyle=":")
    ax2.fill_between(steps, portfolio_history[:n], initial_balance,
                     alpha=0.1, color="#58a6ff")
    ax2.set_title("Portfolio Value vs Buy-and-Hold Benchmark",
                  color="white", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Trading Step", color="#8b949e")
    ax2.set_ylabel("Portfolio ($)", color="#8b949e")
    ax2.legend(facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=9)

    for ax in axes:
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    fig.suptitle("AI Trading Agent — Backtest Results",
                 color="white", fontsize=15, fontweight="bold")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Backtest results saved → {save_path}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(portfolio_history: list,
                    trade_log: list,
                    initial_balance: float = 10_000.0) -> dict:
    """Compute key performance metrics from a completed episode."""
    final_value = portfolio_history[-1]
    roi         = (final_value - initial_balance) / initial_balance * 100

    returns = np.diff(portfolio_history) / (np.array(portfolio_history[:-1]) + 1e-8)

    sharpe = 0.0
    if returns.std() > 1e-8:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)   # annualised

    # Max drawdown
    peak = np.maximum.accumulate(portfolio_history)
    dd   = (np.array(portfolio_history) - peak) / (peak + 1e-8)
    max_dd = float(dd.min() * 100)

    sells        = [t for t in trade_log if t["action"] == "SELL"]
    total_trades = len([t for t in trade_log if t["action"] in ("BUY", "SELL")])
    win_rate     = 0.0
    if sells:
        profitable = sum(1 for t in sells if t.get("profit", 0) > 0)
        win_rate   = profitable / len(sells) * 100

    return {
        "Final Portfolio ($)": round(final_value, 2),
        "ROI (%)":              round(roi, 2),
        "Sharpe Ratio":         round(sharpe, 4),
        "Max Drawdown (%)":     round(max_dd, 2),
        "Total Trades":         total_trades,
        "Win Rate (%)":         round(win_rate, 2),
    }
