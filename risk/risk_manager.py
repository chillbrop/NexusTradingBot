"""
NexusTrade — Risk Management Module
=====================================
Enforces all risk rules before allowing trade entry.
Tracks daily P&L, consecutive losses, and account safeguards.
This is the gatekeeper — nothing trades without passing through here.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from config import config

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    """Live risk state — resets at midnight UTC."""
    date: str = field(default_factory=lambda: str(date.today()))
    starting_balance: float = 0.0
    current_balance: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_trades_today: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    peak_balance_today: float = 0.0
    max_drawdown_today: float = 0.0
    current_stake: float = 0.0
    anti_mart_step: int = 0       # How many consecutive scale-ups
    last_loss_ts: float = 0.0     # Unix timestamp of last loss
    trading_paused: bool = False  # Manual or auto pause
    pause_reason: str = ""
    daily_profit_hit: bool = False
    daily_loss_hit: bool = False
    current_risk_level: str = "NORMAL"   # NORMAL, CONSERVATIVE, AGGRESSIVE

    @property
    def win_rate(self) -> float:
        if self.total_trades_today == 0:
            return 0.0
        return self.winning_trades / self.total_trades_today * 100

    def reset_for_new_day(self, balance: float):
        self.date = str(date.today())
        self.starting_balance = balance
        self.current_balance = balance
        self.daily_pnl = 0.0
        self.daily_pnl_pct = 0.0
        self.total_trades_today = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.peak_balance_today = balance
        self.max_drawdown_today = 0.0
        self.anti_mart_step = 0
        self.trading_paused = False
        self.pause_reason = ""
        self.daily_profit_hit = False
        self.daily_loss_hit = False
        self.current_risk_level = "NORMAL"
        logger.info(f"Risk state reset for new day. Starting balance: ${balance:.2f}")


class RiskManager:
    """
    Central risk management system.

    Responsibilities:
    - Validate each trade entry against all risk rules
    - Calculate optimal stake size (Kelly-influenced)
    - Track P&L and position exposure
    - Enforce daily profit/loss targets
    - Manage consecutive loss protection
    - Anti-martingale stake scaling
    """

    def __init__(self, initial_balance: float = 0.0):
        self.state = RiskState()
        self.state.starting_balance = initial_balance
        self.state.current_balance = initial_balance
        self.state.peak_balance_today = initial_balance
        self._cfg = config.risk

    # ----------------------------------------------------------------
    # GATE CHECK — called before every potential trade
    # ----------------------------------------------------------------

    def can_trade(self, signal_confidence: float = 50.0) -> tuple[bool, str]:
        """
        Master gate check. Returns (allowed: bool, reason: str).
        All conditions must pass for a trade to be allowed.
        """
        cfg = self._cfg
        state = self.state

        # Always check for day rollover
        self._check_day_rollover()

        # 1. Emergency balance protection
        if state.current_balance < cfg.min_balance:
            return False, f"Balance below minimum (${state.current_balance:.2f} < ${cfg.min_balance:.2f})"

        # 2. Daily loss limit
        loss_limit = state.starting_balance * (cfg.daily_loss_limit_pct / 100)
        if state.daily_pnl <= -loss_limit:
            state.daily_loss_hit = True
            return False, f"Daily loss limit hit (${state.daily_pnl:.2f})"

        # 3. Daily profit target
        profit_target = state.starting_balance * (cfg.daily_profit_target_pct / 100)
        if state.daily_pnl >= profit_target:
            state.daily_profit_hit = True
            return False, f"Daily profit target reached (+${state.daily_pnl:.2f}) 🎯"

        # 4. Consecutive loss protection
        if state.consecutive_losses >= cfg.max_consecutive_losses:
            # Cool-down period check
            elapsed = time.time() - state.last_loss_ts
            if elapsed < cfg.pause_after_loss_seconds:
                remaining = int(cfg.pause_after_loss_seconds - elapsed)
                return False, f"Loss cool-down active ({remaining}s remaining)"
            else:
                # Cool-down passed — reset and allow
                state.consecutive_losses = 0
                logger.info("Loss cool-down elapsed — trading resuming")

        # 5. Manual pause
        if state.trading_paused:
            return False, f"Trading manually paused: {state.pause_reason}"

        # 6. Signal confidence minimum
        if signal_confidence < config.strategy.min_confidence_score:
            return False, f"Signal confidence too low ({signal_confidence:.1f} < {config.strategy.min_confidence_score})"

        return True, "OK"

    # ----------------------------------------------------------------
    # STAKE SIZING
    # ----------------------------------------------------------------

    def calculate_stake(self, confidence: float) -> float:
        """
        Calculate the stake for the next trade.

        Combines:
        - Base % of balance (configurable)
        - Anti-martingale scaling (increase after wins)
        - Confidence-based adjustment
        - Recovery mode reduction
        - Hard caps
        """
        cfg = self._cfg
        state = self.state
        balance = state.current_balance

        # Base stake = % of current balance
        base_stake = balance * (cfg.max_risk_per_trade_pct / 100)

        # Confidence adjustment: high confidence = full stake, low = reduce
        # Confidence range typically 60–95 when reaching this point
        conf_factor = (confidence - 50) / 50   # 0.2 at 60, 1.0 at 100
        conf_factor = max(0.3, min(1.5, conf_factor))

        stake = base_stake * conf_factor

        # Anti-martingale: scale up after consecutive wins
        if cfg.anti_martingale and state.consecutive_wins >= 2:
            steps = min(state.consecutive_wins - 1, cfg.max_anti_mart_steps)
            am_factor = cfg.anti_martingale_multiplier ** steps
            stake *= am_factor
            logger.debug(f"Anti-martingale: x{am_factor:.2f} (streak={state.consecutive_wins})")

        # Recovery mode: reduce stake after recent losses
        if state.consecutive_losses >= cfg.recovery_mode_after_losses:
            stake *= 0.5
            state.current_risk_level = "CONSERVATIVE"
            logger.debug("Recovery mode: stake halved")
        else:
            state.current_risk_level = "NORMAL"

        # Apply hard caps
        stake = max(1.0, min(stake, cfg.max_stake))
        stake = round(stake, 2)

        state.current_stake = stake
        return stake

    # ----------------------------------------------------------------
    # TRADE RESULT RECORDING
    # ----------------------------------------------------------------

    def record_win(self, pnl: float):
        """Call this when a trade closes as a winner."""
        state = self.state
        state.current_balance += pnl
        state.daily_pnl += pnl
        state.total_trades_today += 1
        state.winning_trades += 1
        state.consecutive_wins += 1
        state.consecutive_losses = 0
        state.anti_mart_step = min(state.anti_mart_step + 1,
                                   self._cfg.max_anti_mart_steps)

        # Update peak balance
        if state.current_balance > state.peak_balance_today:
            state.peak_balance_today = state.current_balance

        state.daily_pnl_pct = state.daily_pnl / state.starting_balance * 100

        logger.info(f"✅ WIN +${pnl:.2f} | Balance: ${state.current_balance:.2f} | "
                    f"Daily P&L: {state.daily_pnl_pct:+.2f}% | Streak: {state.consecutive_wins}")

    def record_loss(self, pnl: float):
        """Call this when a trade closes as a loser (pnl should be negative)."""
        state = self.state
        pnl = -abs(pnl)  # Ensure negative
        state.current_balance += pnl
        state.daily_pnl += pnl
        state.total_trades_today += 1
        state.losing_trades += 1
        state.consecutive_losses += 1
        state.consecutive_wins = 0
        state.anti_mart_step = 0
        state.last_loss_ts = time.time()

        # Update drawdown
        drawdown = state.peak_balance_today - state.current_balance
        if drawdown > state.max_drawdown_today:
            state.max_drawdown_today = drawdown

        state.daily_pnl_pct = state.daily_pnl / state.starting_balance * 100

        logger.warning(f"❌ LOSS ${pnl:.2f} | Balance: ${state.current_balance:.2f} | "
                       f"Daily P&L: {state.daily_pnl_pct:+.2f}% | Streak: -{state.consecutive_losses}")

        # Check if we hit consecutive loss threshold
        if state.consecutive_losses >= self._cfg.max_consecutive_losses:
            logger.warning(f"⛔ {state.consecutive_losses} consecutive losses — "
                           f"entering {self._cfg.pause_after_loss_seconds}s cool-down")

    # ----------------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------------

    def update_balance(self, new_balance: float):
        """Sync balance from live API data."""
        old = self.state.current_balance
        self.state.current_balance = new_balance
        if abs(old - new_balance) > 0.01:
            logger.debug(f"Balance updated: ${old:.2f} → ${new_balance:.2f}")

    def pause_trading(self, reason: str = "Manual pause"):
        """Manually pause the bot."""
        self.state.trading_paused = True
        self.state.pause_reason = reason
        logger.warning(f"Trading paused: {reason}")

    def resume_trading(self):
        """Resume after manual pause."""
        self.state.trading_paused = False
        self.state.pause_reason = ""
        logger.info("Trading resumed")

    def _check_day_rollover(self):
        """Auto-reset state at midnight UTC."""
        today = str(date.today())
        if self.state.date != today:
            logger.info("New trading day detected — resetting risk state")
            self.state.reset_for_new_day(self.state.current_balance)

    def get_status_summary(self) -> dict:
        """Return a JSON-serializable status dict for the dashboard."""
        s = self.state
        return {
            "balance": round(s.current_balance, 2),
            "daily_pnl": round(s.daily_pnl, 2),
            "daily_pnl_pct": round(s.daily_pnl_pct, 3),
            "trades_today": s.total_trades_today,
            "win_rate": round(s.win_rate, 1),
            "consecutive_wins": s.consecutive_wins,
            "consecutive_losses": s.consecutive_losses,
            "max_drawdown_today": round(s.max_drawdown_today, 2),
            "risk_level": s.current_risk_level,
            "trading_paused": s.trading_paused,
            "pause_reason": s.pause_reason,
            "daily_profit_hit": s.daily_profit_hit,
            "daily_loss_hit": s.daily_loss_hit,
            "profit_target_usd": round(s.starting_balance * config.risk.daily_profit_target_pct / 100, 2),
            "loss_limit_usd": round(s.starting_balance * config.risk.daily_loss_limit_pct / 100, 2),
        }
