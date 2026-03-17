"""
train.py
--------
Training loop for the DQN Trading Agent.

Usage:
    python train.py                         # default: synthetic data, 200 episodes
    python train.py --ticker AAPL --episodes 500
    python train.py --synthetic --episodes 300 --lr 5e-4

Key behaviours:
  • Saves the best model (by portfolio value) automatically.
  • Logs metrics every `log_every` episodes.
  • Produces training dashboard plot at the end.
"""

import argparse
import time
import numpy as np

from trading_env import TradingEnvironment
from dqn_agent   import DQNAgent
from utils       import (load_data, train_test_split,
                          plot_training_results, compute_metrics)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train the DQN Trading Agent")
    p.add_argument("--ticker",      type=str,   default="AAPL",
                   help="Stock ticker (requires yfinance)")
    p.add_argument("--start",       type=str,   default="2018-01-01")
    p.add_argument("--end",         type=str,   default="2023-12-31")
    p.add_argument("--synthetic",   action="store_true",
                   help="Use synthetic data (no internet needed)")
    p.add_argument("--episodes",    type=int,   default=200)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--gamma",       type=float, default=0.99)
    p.add_argument("--epsilon",     type=float, default=1.0)
    p.add_argument("--eps_decay",   type=float, default=0.995)
    p.add_argument("--batch",       type=int,   default=64)
    p.add_argument("--balance",     type=float, default=10_000.0)
    p.add_argument("--log_every",   type=int,   default=10)
    p.add_argument("--save",        type=str,   default="dqn_trading_agent.pt")
    p.add_argument("--plot",        type=str,   default="training_results.png")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(args):
    print("\n" + "="*60)
    print("   AI TRADING AGENT  —  Deep Q-Learning Training")
    print("="*60)

    # ---- Data ----
    df = load_data(ticker=args.ticker, start=args.start, end=args.end,
                   use_synthetic=args.synthetic)
    train_df, test_df = train_test_split(df, train_ratio=0.8)

    # ---- Environment ----
    env = TradingEnvironment(train_df, initial_balance=args.balance)

    # ---- Agent ----
    agent = DQNAgent(
        state_size    = env.state_size,
        action_size   = env.action_size,
        lr            = args.lr,
        gamma         = args.gamma,
        epsilon       = args.epsilon,
        epsilon_decay = args.eps_decay,
        batch_size    = args.batch,
    )

    print(f"\n[Config] Episodes: {args.episodes} | "
          f"Train rows: {len(train_df)} | "
          f"Initial balance: ${args.balance:,.0f}")
    print(f"[Config] Device: {agent.device} | "
          f"Batch: {args.batch} | LR: {args.lr}")
    print("-"*60)

    # ---- Tracking ----
    episode_rewards  = []
    portfolio_values = []
    best_portfolio   = 0.0
    t0 = time.time()

    # ---- Main Loop ----
    for ep in range(1, args.episodes + 1):
        state = env.reset()
        done  = False

        while not done:
            action                      = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

        agent.decay_epsilon()

        ep_reward = env.episode_reward
        ep_pv     = env._portfolio_value()
        episode_rewards.append(ep_reward)
        portfolio_values.append(ep_pv)

        # Save best model
        if ep_pv > best_portfolio:
            best_portfolio = ep_pv
            agent.save(args.save)

        # Logging
        if ep % args.log_every == 0 or ep == 1:
            elapsed  = time.time() - t0
            roi      = (ep_pv - args.balance) / args.balance * 100
            avg_rew  = np.mean(episode_rewards[-args.log_every:])
            avg_loss = np.mean(agent.losses[-200:]) if agent.losses else float("nan")
            print(f"Ep {ep:4d}/{args.episodes} | "
                  f"ε={agent.epsilon:.3f} | "
                  f"Avg Reward={avg_rew:+.4f} | "
                  f"Portfolio=${ep_pv:,.0f} ({roi:+.1f}%) | "
                  f"Loss={avg_loss:.5f} | "
                  f"Elapsed={elapsed:.0f}s")

    print("-"*60)
    print(f"\n[Done] Best portfolio: ${best_portfolio:,.2f} "
          f"(ROI: {(best_portfolio-args.balance)/args.balance*100:+.2f}%)")
    print(f"[Done] Model saved → {args.save}")

    # ---- Plots ----
    plot_training_results(episode_rewards, portfolio_values,
                          agent.losses, save_path=args.plot)

    # ---- Quick test-set evaluation ----
    print("\n" + "="*60)
    print("   TEST SET EVALUATION")
    print("="*60)
    agent.load(args.save)
    test_env = TradingEnvironment(test_df, initial_balance=args.balance)
    evaluate(agent, test_env, args.balance)

    return agent, episode_rewards, portfolio_values


# ---------------------------------------------------------------------------
# Evaluation Helper
# ---------------------------------------------------------------------------

def evaluate(agent: DQNAgent,
             env:   TradingEnvironment,
             initial_balance: float = 10_000.0):
    """Run one greedy episode on `env` and print metrics."""
    state = env.reset()
    done  = False
    step  = 0

    while not done:
        action = agent.act(state, training=False)
        state, _, done, info = env.step(action)
        step += 1

    metrics = compute_metrics(env.portfolio_history, env.trade_log, initial_balance)
    print("\nPerformance Metrics")
    print("-"*35)
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")
    print("-"*35)

    return metrics


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
