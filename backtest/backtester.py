"""
NexusTrade — Backtesting Engine
=================================
Runs a simulated trading session on historical OHLC data.
Produces a full performance report with equity curve, trade log,
and per-strategy metrics.

Usage:
    bt = Backtester(candles, initial_balance=1000.0)
    results = bt.run()
    bt.print_report(results)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from core.indicators import compute_all
from core.signal_engine import SignalEngine
from risk.risk_manager import RiskManager
from config import config

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    index: int
    direction: str
    entry_price: float
    exit_price: float
    stake: float
    payout: float
    profit: float
    won: bool
    confidence: float
    duration_bars: int
    reasoning: List[str] = field(default_factory=list)


@dataclass
class BacktestResults:
    initial_balance: float
    final_balance: float
    net_pnl: float
    pnl_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    equity_curve: List[float]
    trades: List[BacktestTrade]
    signals_generated: int
    signals_skipped: int


class Backtester:
    """
    Walk-forward backtester.

    Simulates a full trading session on historical OHLC data.
    Uses the same indicator and signal engine as the live bot.
    Does NOT use future data — each bar sees only past candles.

    Assumes binary options payout: win returns stake * payout_ratio,
    loss forfeits entire stake.
    """

    DEFAULT_PAYOUT_RATIO = 0.85   # Typical Deriv Rise/Fall payout

    def __init__(self, candles: List[dict], initial_balance: float = 1000.0,
                 payout_ratio: float = DEFAULT_PAYOUT_RATIO,
                 trade_duration_bars: int = 5,
                 warm_up_bars: int = 50):
        """
        Args:
            candles:             List of OHLCV dicts sorted oldest→newest.
            initial_balance:     Starting paper balance.
            payout_ratio:        Win payout as fraction of stake (0.85 = 85% profit).
            trade_duration_bars: How many bars before trade expires.
            warm_up_bars:        Bars to skip at start (needed for indicator warm-up).
        """
        self.candles = candles
        self.initial_balance = initial_balance
        self.payout_ratio = payout_ratio
        self.trade_duration_bars = trade_duration_bars
        self.warm_up_bars = max(warm_up_bars, 50)
        self.signal_engine = SignalEngine()

    def run(self) -> BacktestResults:
        """Execute the backtest. Returns full results."""
        n = len(self.candles)
        if n < self.warm_up_bars + self.trade_duration_bars + 10:
            raise ValueError(f"Not enough candles ({n}) for backtesting")

        balance = self.initial_balance
        peak_balance = balance
        max_dd = 0.0
        equity_curve = [balance]
        trades: List[BacktestTrade] = []
        signals_generated = 0
        signals_skipped = 0

        consecutive_losses = 0
        pnl_series = []

        i = self.warm_up_bars
        while i < n - self.trade_duration_bars:
            # Compute indicators on all candles up to current bar
            visible_candles = self.candles[:i+1]
            indicators = compute_all(visible_candles)
            if indicators is None:
                i += 1
                continue

            # Score the signal
            signal = self.signal_engine.analyze(indicators)
            signals_generated += 1

            # Skip if no clear signal
            if not signal.should_trade:
                signals_skipped += 1
                i += 1
                continue

            # Calculate stake
            risk_pct = config.risk.max_risk_per_trade_pct / 100
            stake = balance * risk_pct * signal.stake_modifier
            stake = max(1.0, min(stake, config.risk.max_stake))
            stake = round(stake, 2)

            # Skip if we'd go below minimum
            if balance - stake < config.risk.min_balance:
                signals_skipped += 1
                i += 1
                continue

            # Determine outcome: look at close price N bars ahead
            entry_price = self.candles[i]["close"]
            exit_bar = min(i + self.trade_duration_bars, n - 1)
            exit_price = self.candles[exit_bar]["close"]

            # Win condition: price moved in predicted direction
            if signal.direction == "RISE":
                won = exit_price > entry_price
            else:
                won = exit_price < entry_price

            # P&L calculation (binary options)
            if won:
                profit = stake * self.payout_ratio
                consecutive_losses = 0
            else:
                profit = -stake
                consecutive_losses += 1

            balance += profit
            pnl_series.append(profit)

            # Update drawdown
            if balance > peak_balance:
                peak_balance = balance
            dd = peak_balance - balance
            if dd > max_dd:
                max_dd = dd

            equity_curve.append(round(balance, 2))

            trades.append(BacktestTrade(
                index=i,
                direction=signal.direction,
                entry_price=entry_price,
                exit_price=exit_price,
                stake=stake,
                payout=profit if won else 0,
                profit=round(profit, 2),
                won=won,
                confidence=signal.confidence,
                duration_bars=self.trade_duration_bars,
                reasoning=signal.reasoning[:3],
            ))

            # Enforce daily loss limit (simplified: check after every 10 trades)
            daily_pnl = sum(pnl_series[-20:]) if len(pnl_series) >= 20 else sum(pnl_series)
            if daily_pnl < -(self.initial_balance * config.risk.daily_loss_limit_pct / 100):
                logger.debug(f"Backtest: daily loss limit hit at bar {i}")

            # Skip ahead to after trade expires
            i += self.trade_duration_bars

        # ----------------------------------------------------------------
        # Aggregate results
        # ----------------------------------------------------------------
        total = len(trades)
        if total == 0:
            logger.warning("Backtest produced no trades")
            return self._empty_results(equity_curve)

        wins_list = [t.profit for t in trades if t.won]
        losses_list = [t.profit for t in trades if not t.won]
        win_count = len(wins_list)
        loss_count = len(losses_list)

        gross_profit = sum(wins_list) if wins_list else 0
        gross_loss = abs(sum(losses_list)) if losses_list else 0

        net_pnl = balance - self.initial_balance
        win_rate = win_count / total * 100 if total > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
        avg_win = gross_profit / win_count if win_count > 0 else 0
        avg_loss = gross_loss / loss_count if loss_count > 0 else 0
        max_dd_pct = max_dd / self.initial_balance * 100 if self.initial_balance > 0 else 0

        # Sharpe ratio (annualized, assuming daily P&L)
        if len(pnl_series) > 1:
            daily_returns = np.array(pnl_series)
            sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(252)
        else:
            sharpe = 0.0

        return BacktestResults(
            initial_balance=self.initial_balance,
            final_balance=round(balance, 2),
            net_pnl=round(net_pnl, 2),
            pnl_pct=round(net_pnl / self.initial_balance * 100, 2),
            total_trades=total,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=round(win_rate, 2),
            gross_profit=round(gross_profit, 2),
            gross_loss=round(gross_loss, 2),
            profit_factor=round(profit_factor, 3),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            best_trade=round(max(t.profit for t in trades), 2),
            worst_trade=round(min(t.profit for t in trades), 2),
            max_drawdown=round(max_dd, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            sharpe_ratio=round(float(sharpe), 3),
            equity_curve=equity_curve,
            trades=trades,
            signals_generated=signals_generated,
            signals_skipped=signals_skipped,
        )

    def _empty_results(self, equity_curve) -> BacktestResults:
        return BacktestResults(
            initial_balance=self.initial_balance, final_balance=self.initial_balance,
            net_pnl=0, pnl_pct=0, total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, gross_profit=0, gross_loss=0, profit_factor=0, avg_win=0, avg_loss=0,
            best_trade=0, worst_trade=0, max_drawdown=0, max_drawdown_pct=0, sharpe_ratio=0,
            equity_curve=equity_curve, trades=[], signals_generated=0, signals_skipped=0,
        )

    @staticmethod
    def print_report(r: BacktestResults):
        """Print a formatted backtest summary to console."""
        print("\n" + "=" * 55)
        print("  NEXUSTRADE BACKTEST REPORT")
        print("=" * 55)
        print(f"  Balance:    ${r.initial_balance:.2f} → ${r.final_balance:.2f}")
        print(f"  Net P&L:    ${r.net_pnl:+.2f}  ({r.pnl_pct:+.2f}%)")
        print("-" * 55)
        print(f"  Trades:     {r.total_trades}  "
              f"(Wins: {r.winning_trades} | Losses: {r.losing_trades})")
        print(f"  Win Rate:   {r.win_rate:.1f}%")
        print(f"  Profit Factor: {r.profit_factor:.3f}")
        print(f"  Sharpe Ratio:  {r.sharpe_ratio:.3f}")
        print("-" * 55)
        print(f"  Avg Win:    ${r.avg_win:.2f}    Avg Loss: ${r.avg_loss:.2f}")
        print(f"  Best Trade: ${r.best_trade:.2f}  Worst: ${r.worst_trade:.2f}")
        print(f"  Max Drawdown: ${r.max_drawdown:.2f} ({r.max_drawdown_pct:.2f}%)")
        print("-" * 55)
        print(f"  Signals generated: {r.signals_generated}")
        print(f"  Signals skipped:   {r.signals_skipped}")
        print("=" * 55 + "\n")
