"""
trading_env.py
--------------
Simulated stock trading environment for the AI Trading Agent.
Implements the Agent–Environment Interaction Loop from the PEAS framework.

State Space:
    [price_norm, ma5_norm, ma10_norm, ma20_norm, daily_return,
     shares_held_norm, balance_norm, unrealized_pnl_norm]

Action Space:
    0 = Hold
    1 = Buy  (uses 10% of available balance)
    2 = Sell (sells all held shares)
"""

import numpy as np
import pandas as pd


class TradingEnvironment:
    """
    A simulated stock market environment built on historical price data.
    Follows OpenAI Gym-style interface: reset(), step(), render().
    """

    HOLD = 0
    BUY  = 1
    SELL = 2
    ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10_000.0,
                 trade_pct: float = 0.1, window: int = 20):
        """
        Args:
            df              : DataFrame with at least a 'Close' column.
            initial_balance : Starting cash balance in USD.
            trade_pct       : Fraction of balance used per BUY order.
            window          : Look-back window for moving averages.
        """
        self.df              = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.trade_pct       = trade_pct
        self.window          = window
        self.n_steps         = len(df)

        # Pre-compute indicators once (fast)
        self._precompute_features()

        # State/action dimensions
        self.state_size  = 8
        self.action_size = 3

        self.reset()

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    def _precompute_features(self):
        closes = self.df["Close"].values.astype(np.float32)

        # Simple moving averages
        self.ma5  = self._rolling_mean(closes, 5)
        self.ma10 = self._rolling_mean(closes, 10)
        self.ma20 = self._rolling_mean(closes, 20)

        # Daily log-returns
        self.returns = np.zeros_like(closes)
        self.returns[1:] = np.log(closes[1:] / (closes[:-1] + 1e-8))

        # Normalisation baselines (use first 20 days as reference)
        self.price_scale   = max(closes[:20].mean(), 1.0)
        self.balance_scale = self.initial_balance

    @staticmethod
    def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        result = np.full_like(arr, arr[0])
        for i in range(1, len(arr)):
            start = max(0, i - window + 1)
            result[i] = arr[start: i + 1].mean()
        return result

    # ------------------------------------------------------------------
    # Gym Interface
    # ------------------------------------------------------------------

    def reset(self):
        """Reset environment to the start of the price series."""
        self.current_step    = self.window
        self.balance         = self.initial_balance
        self.shares_held     = 0.0
        self.total_trades    = 0
        self.profit_trades   = 0
        self.episode_reward  = 0.0
        self.portfolio_history = [self.initial_balance]
        self.trade_log       = []
        return self._get_state()

    def step(self, action: int):
        """
        Execute one trading step.

        Returns:
            next_state (np.ndarray)
            reward     (float)
            done       (bool)
            info       (dict)
        """
        prev_portfolio = self._portfolio_value()
        price          = self._current_price()

        reward_shaping = 0.0

        # ---- Execute Action ----
        if action == self.BUY:
            reward_shaping = self._execute_buy(price)
        elif action == self.SELL:
            reward_shaping = self._execute_sell(price)
        # HOLD: no transaction

        # ---- Advance time ----
        self.current_step += 1

        # ---- Reward: change in portfolio value ----
        new_portfolio = self._portfolio_value()
        reward = (new_portfolio - prev_portfolio) / (self.initial_balance + 1e-8)
        reward += reward_shaping

        self.episode_reward  += reward
        self.portfolio_history.append(new_portfolio)

        done = self.current_step >= self.n_steps - 1

        info = {
            "step"          : self.current_step,
            "price"         : price,
            "balance"       : self.balance,
            "shares_held"   : self.shares_held,
            "portfolio"     : new_portfolio,
            "episode_reward": self.episode_reward,
            "action_name"   : self.ACTION_NAMES[action],
        }

        return self._get_state(), reward, done, info

    # ------------------------------------------------------------------
    # Trading Logic
    # ------------------------------------------------------------------

    def _execute_buy(self, price: float) -> float:
        """Buy shares using trade_pct of available balance."""
        spend        = self.balance * self.trade_pct
        shares_to_buy = spend / (price + 1e-8)
        if shares_to_buy < 1e-6:
            return -0.001              # Small penalty for futile BUY
        self.balance    -= spend
        self.shares_held += shares_to_buy
        self.total_trades += 1
        self.trade_log.append({"step": self.current_step, "action": "BUY",
                                "price": price, "shares": shares_to_buy})
        return 0.0

    def _execute_sell(self, price: float) -> float:
        """Sell all held shares."""
        if self.shares_held < 1e-6:
            return -0.001              # Small penalty for selling nothing

        proceeds          = self.shares_held * price
        cost_basis        = self.shares_held * self._avg_buy_price()
        profit            = proceeds - cost_basis

        self.balance     += proceeds
        self.total_trades += 1
        if profit > 0:
            self.profit_trades += 1

        self.trade_log.append({"step": self.current_step, "action": "SELL",
                                "price": price, "shares": self.shares_held,
                                "profit": profit})
        self.shares_held  = 0.0
        return 0.0

    def _avg_buy_price(self) -> float:
        """Approximate average buy price from trade log."""
        buys = [t for t in self.trade_log if t["action"] == "BUY"]
        if not buys:
            return self._current_price()
        total_cost   = sum(t["price"] * t["shares"] for t in buys)
        total_shares = sum(t["shares"] for t in buys)
        return total_cost / (total_shares + 1e-8)

    # ------------------------------------------------------------------
    # State Construction  (SENSORS)
    # ------------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        i = self.current_step
        price = self.df["Close"].iloc[i]

        state = np.array([
            price               / self.price_scale,
            self.ma5[i]         / self.price_scale,
            self.ma10[i]        / self.price_scale,
            self.ma20[i]        / self.price_scale,
            np.clip(self.returns[i], -0.1, 0.1) / 0.1,   # normalised return
            min(self.shares_held * price / self.initial_balance, 1.0),
            self.balance        / self.balance_scale,
            self._unrealised_pnl() / self.initial_balance,
        ], dtype=np.float32)

        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_price(self) -> float:
        return float(self.df["Close"].iloc[self.current_step])

    def _portfolio_value(self) -> float:
        return self.balance + self.shares_held * self._current_price()

    def _unrealised_pnl(self) -> float:
        if self.shares_held < 1e-6:
            return 0.0
        return self.shares_held * (self._current_price() - self._avg_buy_price())

    def render(self):
        pv = self._portfolio_value()
        roi = (pv - self.initial_balance) / self.initial_balance * 100
        print(f"Step {self.current_step:4d} | "
              f"Price: ${self._current_price():8.2f} | "
              f"Balance: ${self.balance:10.2f} | "
              f"Shares: {self.shares_held:6.4f} | "
              f"Portfolio: ${pv:10.2f} | "
              f"ROI: {roi:+.2f}%")
