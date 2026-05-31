"""
NexusTrade — Main Trading Bot
===============================
Orchestrates the entire trading pipeline:
  1. Connect to Deriv API
  2. Stream live candle data
  3. Compute indicators every tick
  4. Score signal via AI engine
  5. Check risk gates
  6. Execute trades
  7. Monitor open positions
  8. Record results and notify

Run from the project root:
    python -m nexustrade.bot
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

from config import config, AppConfig
from core.deriv_client import DerivClient, DerivAPIError
from core.indicators import compute_all, AllIndicators
from core.signal_engine import SignalEngine, TradeSignal
from risk.risk_manager import RiskManager
from db.trade_logger import TradeLogger
from notifications.notifier import Notifier
from utils.logger_setup import setup_logging

logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main bot controller.
    Manages the complete lifecycle from market data → trade → result → log.
    """

    def __init__(self, cfg: AppConfig = None):
        self.cfg = cfg or config
        self.client = DerivClient()
        self.signal_engine = SignalEngine()
        self.risk_manager = RiskManager()
        self.trade_logger = TradeLogger(cfg=self.cfg.database)
        self.notifier = Notifier(cfg=self.cfg.notifications)

        # Per-symbol state
        self.candle_buffers: Dict[str, List[dict]] = {}
        self.active_trades: Dict[str, dict] = {}   # symbol → trade info
        self._last_indicators: Dict[str, AllIndicators] = {}

        # Bot state
        self._running = False
        self._paused = False
        self._trade_count = 0

        # Wire lifecycle callbacks
        self.client.on_connect = self._on_api_connect
        self.client.on_disconnect = self._on_api_disconnect

    # ----------------------------------------------------------------
    # MAIN RUN LOOP
    # ----------------------------------------------------------------

    async def run(self):
        """Start the bot. Blocks until shutdown."""
        setup_logging(self.cfg.log_level, self.cfg.log_file)
        logger.info("=" * 60)
        logger.info("NexusTrade Bot Starting")
        logger.info(f"Mode: {'DEMO' if self.cfg.deriv.is_demo else '⚠ LIVE'}")
        logger.info(f"Markets: {', '.join(self.cfg.markets)}")
        logger.info("=" * 60)

        # ---- Pre-flight validation ----
        self._preflight_check()

        # Initialize database
        await self.trade_logger.initialize()

        # Register shutdown handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        self._running = True

        try:
            await self.client.connect()
        except KeyboardInterrupt:
            await self.shutdown()

    def _preflight_check(self):
        """Validate critical configuration before connecting."""
        errors = []

        if not self.cfg.deriv.api_token:
            errors.append(
                "DERIV_API_TOKEN is not set.\n"
                "  1. Copy .env.example to .env\n"
                "  2. Set DERIV_API_TOKEN=your_token_here\n"
                "  3. Get a token at: https://app.deriv.com/account/api-token"
            )

        if not self.cfg.deriv.app_id:
            errors.append("DERIV_APP_ID is missing (default: 1089)")

        if errors:
            for e in errors:
                logger.critical(f"CONFIG ERROR: {e}")
            raise SystemExit(
                "\n\nBot cannot start — fix the above configuration errors first.\n"
            )

        logger.info(f"Config OK | App ID: {self.cfg.deriv.app_id} | "
                    f"Token: ...{self.cfg.deriv.api_token[-4:]} | "
                    f"Demo: {self.cfg.deriv.is_demo}")

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down NexusTrade...")
        self._running = False
        await self.client.disconnect()
        await self.trade_logger.close()
        logger.info("Shutdown complete.")

    # ----------------------------------------------------------------
    # CONNECTION LIFECYCLE
    # ----------------------------------------------------------------

    async def _on_api_connect(self):
        """Called when WebSocket connects and authorizes."""
        logger.info("API connected — initializing market feeds")

        # Sync balance
        balance_data = await self.client.get_balance()
        balance = float(balance_data.get("balance", 0))
        self.risk_manager.update_balance(balance)
        self.risk_manager.state.starting_balance = balance
        self.risk_manager.state.peak_balance_today = balance
        logger.info(f"Account balance: ${balance:.2f} {balance_data.get('currency', 'USD')}")

        # Subscribe to live candle feeds for each market
        for symbol in self.cfg.markets:
            self.candle_buffers[symbol] = []
            # Fetch history first
            history = await self.client.get_candles(
                symbol,
                granularity=self.cfg.strategy.default_duration_minutes * 60,
                count=200
            )
            self.candle_buffers[symbol] = history
            logger.info(f"Loaded {len(history)} candles for {symbol}")

            # Subscribe to live updates
            await self.client.subscribe_candles(
                symbol,
                granularity=60,  # 1-min candles
                callback=self._make_candle_callback(symbol),
            )

        await self.notifier.send(
            "🟢 NexusTrade connected and running\n"
            f"Balance: ${balance:.2f} | Markets: {len(self.cfg.markets)}"
        )

    async def _on_api_disconnect(self):
        """Called on disconnection."""
        logger.warning("API disconnected — awaiting reconnect")
        await self.notifier.send("⚠️ NexusTrade disconnected — attempting reconnect...")

    # ----------------------------------------------------------------
    # CANDLE DATA HANDLER
    # ----------------------------------------------------------------

    def _make_candle_callback(self, symbol: str):
        """Factory to create a symbol-specific candle handler closure."""
        async def on_candle(ohlc: dict):
            await self._process_candle(symbol, ohlc)
        return on_candle

    async def _process_candle(self, symbol: str, ohlc: dict):
        """
        Main data pipeline — called on every new/updated candle.
        1. Update buffer
        2. Compute indicators
        3. Score signal
        4. Risk check
        5. Execute if approved
        """
        if not self._running:
            return

        buf = self.candle_buffers.get(symbol, [])

        # Build candle dict
        candle = {
            "time": int(ohlc.get("epoch", time.time())),
            "open":  float(ohlc.get("open",  0)),
            "high":  float(ohlc.get("high",  0)),
            "low":   float(ohlc.get("low",   0)),
            "close": float(ohlc.get("close", 0)),
            "volume": float(ohlc.get("volume", 1)),
        }

        # Append or update last candle
        if buf and buf[-1]["time"] == candle["time"]:
            buf[-1] = candle   # Update current candle in-place
        else:
            buf.append(candle)
            if len(buf) > 500:
                buf.pop(0)      # Rolling window

        # Need at least 30 candles for indicators
        if len(buf) < 30:
            return

        # Compute all indicators
        indicators = compute_all(buf)
        if indicators is None:
            return
        self._last_indicators[symbol] = indicators

        # Skip if we already have an open trade on this symbol
        if symbol in self.active_trades:
            await self._monitor_open_trade(symbol, candle["close"])
            return

        # Skip if paused
        if self._paused or not self._running:
            return

        # Market conditions filter
        if not self._market_conditions_ok(indicators, symbol):
            return

        # Score signal
        signal = self.signal_engine.analyze(indicators)

        # Log signal (DEBUG level to avoid log spam)
        logger.debug(f"{symbol} | Signal: {signal.direction} {signal.confidence:.1f}% | "
                     f"Scores: {signal.sub_scores}")

        # Risk gate check
        allowed, reason = self.risk_manager.can_trade(signal.confidence)
        if not allowed:
            logger.debug(f"Trade blocked: {reason}")
            return

        # Execute trade
        if signal.should_trade:
            await self._execute_trade(symbol, signal, candle["close"])

    # ----------------------------------------------------------------
    # MARKET CONDITIONS FILTER
    # ----------------------------------------------------------------

    def _market_conditions_ok(self, ind: AllIndicators, symbol: str) -> bool:
        """
        Extra pre-trade environment checks beyond indicator signals.
        Returns False to block trading in unfavorable conditions.
        """
        # Skip sideways/ranging markets
        if ind.ma.trend == "SIDEWAYS":
            logger.debug(f"{symbol}: Sideways market — skipping")
            return False

        # Skip if BB is in extreme squeeze (no clear direction)
        if ind.bb.squeeze and abs(ind.momentum) < 0.1:
            logger.debug(f"{symbol}: BB squeeze + low momentum — skipping")
            return False

        # Skip extreme volatility
        if ind.atr.volatility == "EXTREME":
            logger.debug(f"{symbol}: Extreme volatility — skipping")
            return False

        # Hour filter (low liquidity)
        hour = datetime.utcnow().hour
        cfg = self.cfg
        if cfg.avoid_low_liquidity_hours:
            start, end = cfg.low_liquidity_start, cfg.low_liquidity_end
            if start > end:
                if hour >= start or hour < end:
                    logger.debug(f"Low liquidity hours ({hour:02d}:xx UTC)")
                    return False
            elif start <= hour < end:
                return False

        return True

    # ----------------------------------------------------------------
    # TRADE EXECUTION
    # ----------------------------------------------------------------

    async def _execute_trade(self, symbol: str, signal: TradeSignal, price: float):
        """Place a trade via the Deriv API."""
        stake = self.risk_manager.calculate_stake(signal.confidence)

        contract_type = "CALL" if signal.direction == "RISE" else "PUT"
        duration = signal.duration_minutes
        duration_unit = "m"

        logger.info(f"📈 Opening trade: {signal.direction} {symbol} | "
                    f"Stake: ${stake:.2f} | Duration: {duration}m | "
                    f"Confidence: {signal.confidence:.1f}%")

        try:
            contract = await self.client.buy_contract(
                symbol=symbol,
                contract_type=contract_type,
                duration=duration,
                duration_unit=duration_unit,
                amount=stake,
            )
        except DerivAPIError as e:
            logger.error(f"Trade execution failed: {e}")
            await self.notifier.send(f"❌ Trade failed: {e}")
            return
        except Exception as e:
            logger.error(f"Unexpected error placing trade: {e}", exc_info=True)
            return

        contract_id = contract.get("contract_id")
        trade_info = {
            "contract_id": contract_id,
            "symbol": symbol,
            "direction": signal.direction,
            "stake": stake,
            "entry_price": price,
            "open_time": time.time(),
            "duration_minutes": duration,
            "confidence": signal.confidence,
            "reasoning": signal.reasoning,
        }

        self.active_trades[symbol] = trade_info
        self._trade_count += 1

        # Log to database
        await self.trade_logger.log_trade_open(trade_info)

        # Subscribe to live P&L updates
        if contract_id:
            await self.client.subscribe_open_contract(
                contract_id,
                callback=self._make_contract_callback(symbol, contract_id),
            )

        await self.notifier.send(
            f"📊 Trade opened\n"
            f"{'📈' if signal.direction=='RISE' else '📉'} {signal.direction} {symbol}\n"
            f"Stake: ${stake:.2f} | Duration: {duration}m\n"
            f"Confidence: {signal.confidence:.1f}%"
        )

    # ----------------------------------------------------------------
    # OPEN TRADE MONITORING
    # ----------------------------------------------------------------

    def _make_contract_callback(self, symbol: str, contract_id: int):
        """Factory for contract status update callback."""
        async def on_update(data: dict):
            await self._on_contract_update(symbol, contract_id, data)
        return on_update

    async def _on_contract_update(self, symbol: str, contract_id: int, data: dict):
        """Handle real-time contract update (P&L, status)."""
        if symbol not in self.active_trades:
            return

        trade = self.active_trades[symbol]
        status = data.get("status")
        current_spot = float(data.get("current_spot", 0))
        profit = float(data.get("profit", 0))

        # Check for early exit opportunity
        if config.strategy.allow_early_exit and profit > 0:
            buy_price = float(data.get("buy_price", trade["stake"]))
            payout = float(data.get("bid_price", 0))   # Current sell value
            max_payout = float(data.get("payout", buy_price * 1.85))

            if max_payout > 0:
                pct_of_max = payout / max_payout
                if pct_of_max >= config.strategy.early_exit_profit_pct:
                    logger.info(f"Early exit triggered: {pct_of_max*100:.0f}% of max profit secured")
                    await self._early_exit(symbol, contract_id, trade)
                    return

        # Trade has closed (won or lost)
        if status in ("won", "lost"):
            await self._close_trade(symbol, data)

    async def _monitor_open_trade(self, symbol: str, current_price: float):
        """Periodic monitoring of open trade (called from candle handler)."""
        trade = self.active_trades.get(symbol)
        if not trade:
            return

        elapsed = time.time() - trade["open_time"]
        max_duration_secs = trade["duration_minutes"] * 60 * 1.2  # 20% grace

        if elapsed > max_duration_secs:
            logger.warning(f"Trade timeout exceeded for {symbol} — may have missed close event")

    async def _early_exit(self, symbol: str, contract_id: int, trade: dict):
        """Sell contract early to lock in profit."""
        try:
            result = await self.client.sell_contract(contract_id)
            sold_for = float(result.get("sold_for", 0))
            profit = sold_for - trade["stake"]
            logger.info(f"Early exit: sold for ${sold_for:.2f} (profit: ${profit:+.2f})")
            await self._finalize_trade(symbol, profit, "EARLY_EXIT")
        except Exception as e:
            logger.error(f"Early exit failed: {e}")

    async def _close_trade(self, symbol: str, data: dict):
        """Handle natural trade close (expired)."""
        profit = float(data.get("profit", 0))
        await self._finalize_trade(symbol, profit, "EXPIRED")

    async def _finalize_trade(self, symbol: str, profit: float, close_type: str):
        """Common logic for trade close — win or loss."""
        trade = self.active_trades.pop(symbol, None)
        if not trade:
            return

        won = profit > 0

        if won:
            self.risk_manager.record_win(profit)
        else:
            self.risk_manager.record_loss(abs(profit))

        # Update database
        await self.trade_logger.log_trade_close({
            **trade,
            "profit": profit,
            "close_type": close_type,
            "won": won,
            "close_time": time.time(),
        })

        status = self.risk_manager.get_status_summary()

        logger.info(
            f"{'✅ WIN' if won else '❌ LOSS'} {trade['symbol']} | "
            f"Profit: ${profit:+.2f} | "
            f"Daily P&L: ${status['daily_pnl']:+.2f} ({status['daily_pnl_pct']:+.2f}%) | "
            f"Balance: ${status['balance']:.2f}"
        )

        await self.notifier.send(
            f"{'✅ WIN' if won else '❌ LOSS'} Trade closed\n"
            f"{trade['symbol']} | ${profit:+.2f}\n"
            f"Balance: ${status['balance']:.2f} | Win rate: {status['win_rate']:.1f}%"
        )

    # ----------------------------------------------------------------
    # MANUAL CONTROLS (for API/dashboard)
    # ----------------------------------------------------------------

    def pause(self, reason: str = "Manual"):
        self._paused = True
        self.risk_manager.pause_trading(reason)

    def resume(self):
        self._paused = False
        self.risk_manager.resume_trading()

    def stop(self):
        self._running = False

    def get_status(self) -> dict:
        """Full status snapshot for the dashboard."""
        return {
            "running": self._running,
            "paused": self._paused,
            "trade_count": self._trade_count,
            "active_trades": list(self.active_trades.keys()),
            "risk": self.risk_manager.get_status_summary(),
            "mode": "DEMO" if config.deriv.is_demo else "LIVE",
        }


# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

async def main():
    bot = TradingBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
