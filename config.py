"""
NexusTrade Configuration
========================
Central configuration for all bot parameters, risk limits, and API settings.
All sensitive values are loaded from environment variables.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Load .env from the project root (the directory containing config.py),
# so the bot works regardless of which directory you launch it from.
_PROJECT_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _PROJECT_ROOT / ".env"

try:
    from dotenv import load_dotenv
    if _ENV_FILE.exists():
        load_dotenv(dotenv_path=_ENV_FILE)
    else:
        # Try the current working directory as a fallback
        load_dotenv()
except ImportError:
    # python-dotenv not installed — rely on shell environment variables
    pass


@dataclass
class DerivAPIConfig:
    """Deriv API connection settings."""
    api_token: str = field(default_factory=lambda: os.getenv("DERIV_API_TOKEN", ""))
    app_id: int = field(default_factory=lambda: int(os.getenv("DERIV_APP_ID", "1089")))
    endpoint: str = "wss://ws.binaryws.com/websockets/v3"
    demo_endpoint: str = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
    is_demo: bool = field(default_factory=lambda: os.getenv("DEMO_MODE", "true").lower() == "true")


@dataclass
class RiskConfig:
    """Risk management parameters."""
    max_risk_per_trade_pct: float = 2.5          # Max % of balance per trade
    daily_profit_target_pct: float = 5.0          # Stop trading at +5% daily
    daily_loss_limit_pct: float = 2.0             # Hard stop at -2% daily
    max_consecutive_losses: int = 3               # Pause after N losses in a row
    max_open_trades: int = 1                      # Only 1 trade at a time
    min_balance: float = 50.0                     # Emergency stop below this
    default_stake: float = 10.0                   # Default trade stake
    max_stake: float = 100.0                      # Hard cap on single stake
    stake_scaling: bool = True                    # Scale stake with balance
    anti_martingale: bool = True                  # Increase stake after wins
    anti_martingale_multiplier: float = 1.5       # Scale factor on wins
    max_anti_mart_steps: int = 3                  # Max consecutive scale-ups
    pause_after_loss_seconds: int = 60            # Cool-down after loss
    recovery_mode_after_losses: int = 2           # Switch to conservative mode


@dataclass
class StrategyConfig:
    """Technical analysis and signal parameters."""
    # Indicator periods
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std_dev: float = 2.0
    ema_short: int = 10
    ema_medium: int = 20
    ema_long: int = 50
    sma_200: int = 200
    atr_period: int = 14

    # Signal thresholds
    min_confidence_score: float = 75.0           # Min AI confidence to trade
    trend_strength_threshold: float = 0.6         # Min trend strength (0–1)
    volatility_filter_max: float = 3.0            # Skip if ATR% above this
    volatility_filter_min: float = 0.1            # Skip if market too quiet

    # Trade timing
    default_duration_minutes: int = 5
    min_duration_minutes: int = 1
    max_duration_minutes: int = 15
    allow_early_exit: bool = True
    early_exit_profit_pct: float = 0.7            # Exit if P&L > 70% of max


@dataclass
class MLConfig:
    """Machine learning model settings."""
    model_path: str = "models/signal_model.pkl"
    feature_window: int = 50                      # Candles of history for features
    retrain_every_n_trades: int = 100
    min_training_samples: int = 500
    use_ml_filter: bool = True                    # Gate trades on ML score
    ml_confidence_threshold: float = 0.70


@dataclass
class NotificationConfig:
    """Telegram and Discord alerts."""
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    discord_webhook: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK", ""))
    notify_on_trade_open: bool = True
    notify_on_trade_close: bool = True
    notify_on_daily_summary: bool = True
    notify_on_errors: bool = True
    notify_on_drawdown: bool = True
    quiet_hours_start: int = 22                   # No alerts 22:00–07:00
    quiet_hours_end: int = 7


@dataclass
class DatabaseConfig:
    """Database connection settings."""
    db_type: str = "sqlite"                       # "sqlite" or "postgresql"
    sqlite_path: str = "data/nexustrade.db"
    postgres_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))


@dataclass
class AppConfig:
    """Top-level application configuration."""
    deriv: DerivAPIConfig = field(default_factory=DerivAPIConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    # Markets to trade
    markets: List[str] = field(default_factory=lambda: [
        "R_100",   # Volatility 100 Index
        "R_75",    # Volatility 75 Index
        "R_50",    # Volatility 50 Index
        "R_25",    # Volatility 25 Index
    ])

    # Market hours filter (UTC)
    avoid_low_liquidity_hours: bool = True
    low_liquidity_start: int = 22
    low_liquidity_end: int = 3

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/nexustrade.log"
    log_to_console: bool = True


# Singleton instance
config = AppConfig()
