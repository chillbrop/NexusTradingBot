"""
NexusTrade — Notification System
===================================
Sends trade alerts, daily summaries, and error notifications via
Telegram Bot API and Discord Webhooks.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

import aiohttp

from config import NotificationConfig

logger = logging.getLogger(__name__)


class Notifier:
    """
    Unified notification dispatcher.
    Supports Telegram and Discord simultaneously.
    Respects quiet hours and rate limits.
    """

    # Rate limit: max 1 message per 3 seconds to avoid spam
    _MIN_INTERVAL = 3.0

    def __init__(self, cfg: NotificationConfig = None):
        self.cfg = cfg or NotificationConfig()
        self._last_sent = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session

    # ----------------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------------

    async def send(self, message: str, priority: bool = False):
        """
        Queue a notification for delivery.
        priority=True bypasses quiet hours (for critical errors).
        """
        if not self._is_configured():
            return

        if not priority and self._in_quiet_hours():
            logger.debug(f"Notification suppressed (quiet hours): {message[:50]}")
            return

        await self._queue.put((message, priority))

        # Start worker if not running
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())

    async def send_trade_opened(self, trade: dict):
        """Formatted alert for trade open."""
        if not self.cfg.notify_on_trade_open:
            return
        direction_emoji = "📈" if trade.get("direction") == "RISE" else "📉"
        msg = (
            f"{direction_emoji} *Trade Opened*\n"
            f"Symbol: `{trade.get('symbol')}`\n"
            f"Direction: *{trade.get('direction')}*\n"
            f"Stake: `${trade.get('stake', 0):.2f}`\n"
            f"Duration: `{trade.get('duration_minutes')}m`\n"
            f"Confidence: `{trade.get('confidence', 0):.1f}%`\n"
            f"Entry: `{trade.get('entry_price', 0):.4f}`"
        )
        await self.send(msg)

    async def send_trade_closed(self, trade: dict, risk_summary: dict):
        """Formatted alert for trade close."""
        if not self.cfg.notify_on_trade_close:
            return
        won = trade.get("won", False)
        profit = trade.get("profit", 0)
        emoji = "✅" if won else "❌"
        msg = (
            f"{emoji} *Trade {'Won' if won else 'Lost'}*\n"
            f"Symbol: `{trade.get('symbol')}`\n"
            f"P&L: `${profit:+.2f}`\n"
            f"Balance: `${risk_summary.get('balance', 0):.2f}`\n"
            f"Daily P&L: `${risk_summary.get('daily_pnl', 0):+.2f}` "
            f"({risk_summary.get('daily_pnl_pct', 0):+.2f}%)\n"
            f"Win Rate: `{risk_summary.get('win_rate', 0):.1f}%`"
        )
        await self.send(msg)

    async def send_daily_summary(self, summary: dict):
        """End-of-day performance summary."""
        if not self.cfg.notify_on_daily_summary:
            return
        pnl = summary.get("net_pnl", 0)
        emoji = "🟢" if pnl >= 0 else "🔴"
        msg = (
            f"{emoji} *Daily Summary — {datetime.utcnow().strftime('%Y-%m-%d')}*\n"
            f"Net P&L: `${pnl:+.2f}`\n"
            f"Trades: `{summary.get('total_trades', 0)}`\n"
            f"Wins: `{summary.get('winning_trades', 0)}` | "
            f"Losses: `{summary.get('losing_trades', 0)}`\n"
            f"Win Rate: `{summary.get('win_rate', 0):.1f}%`\n"
            f"Profit Factor: `{summary.get('profit_factor', 0):.2f}`\n"
            f"Max Drawdown: `${summary.get('max_drawdown', 0):.2f}`"
        )
        await self.send(msg, priority=True)

    async def send_error(self, error: str):
        """Critical error alert — always sent, bypasses quiet hours."""
        if not self.cfg.notify_on_errors:
            return
        await self.send(f"⚠️ *Bot Error*\n```{error[:300]}```", priority=True)

    async def send_drawdown_alert(self, drawdown_pct: float, balance: float):
        """Alert when drawdown exceeds threshold."""
        if not self.cfg.notify_on_drawdown:
            return
        await self.send(
            f"🚨 *Drawdown Alert*\n"
            f"Current drawdown: `{drawdown_pct:.1f}%`\n"
            f"Balance: `${balance:.2f}`",
            priority=True
        )

    # ----------------------------------------------------------------
    # WORKER LOOP
    # ----------------------------------------------------------------

    async def _worker(self):
        """Process message queue with rate limiting."""
        while True:
            try:
                message, priority = await asyncio.wait_for(
                    self._queue.get(), timeout=30.0
                )
            except asyncio.TimeoutError:
                break

            # Rate limit
            elapsed = time.time() - self._last_sent
            if elapsed < self._MIN_INTERVAL:
                await asyncio.sleep(self._MIN_INTERVAL - elapsed)

            await self._dispatch(message)
            self._last_sent = time.time()
            self._queue.task_done()

    async def _dispatch(self, message: str):
        """Send to all configured platforms."""
        tasks = []
        if self.cfg.telegram_bot_token and self.cfg.telegram_chat_id:
            tasks.append(self._send_telegram(message))
        if self.cfg.discord_webhook:
            tasks.append(self._send_discord(message))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"Notification delivery failed: {r}")

    # ----------------------------------------------------------------
    # PLATFORM SENDERS
    # ----------------------------------------------------------------

    async def _send_telegram(self, message: str):
        """Send message via Telegram Bot API."""
        url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.cfg.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        session = await self._get_session()
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Telegram error {resp.status}: {body[:100]}")

    async def _send_discord(self, message: str):
        """Send message via Discord webhook."""
        # Convert Markdown bold (*text*) to Discord format (**text**)
        discord_msg = message.replace("*", "**")
        payload = {"content": discord_msg}
        session = await self._get_session()
        async with session.post(self.cfg.discord_webhook, json=payload) as resp:
            if resp.status not in (200, 204):
                body = await resp.text()
                raise RuntimeError(f"Discord error {resp.status}: {body[:100]}")

    # ----------------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------------

    def _is_configured(self) -> bool:
        return bool(
            (self.cfg.telegram_bot_token and self.cfg.telegram_chat_id) or
            self.cfg.discord_webhook
        )

    def _in_quiet_hours(self) -> bool:
        hour = datetime.utcnow().hour
        start = self.cfg.quiet_hours_start
        end = self.cfg.quiet_hours_end
        if start > end:
            return hour >= start or hour < end
        return start <= hour < end

    async def close(self):
        """Clean up aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
