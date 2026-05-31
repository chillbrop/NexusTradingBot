"""
NexusTrade — Trade Logger (SQLite/PostgreSQL)
==============================================
Persists all trade data, indicator snapshots, and daily summaries.
Supports both SQLite (default) and PostgreSQL (production).
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import date, datetime
from typing import Dict, List, Optional

from config import DatabaseConfig

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    Async-friendly trade database using sqlite3 in a thread executor.
    Schema:
      - trades          — full trade records
      - indicator_snapshots — indicator values at entry
      - daily_summaries — end-of-day roll-ups
      - bot_sessions    — uptime tracking
    """

    def __init__(self, cfg: DatabaseConfig = None):
        self.cfg = cfg or DatabaseConfig()
        self._conn: Optional[sqlite3.Connection] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def initialize(self):
        """Create database and tables if they don't exist."""
        self._loop = asyncio.get_event_loop()
        os.makedirs(os.path.dirname(self.cfg.sqlite_path), exist_ok=True)
        await self._run(self._create_tables)
        logger.info(f"Database initialized: {self.cfg.sqlite_path}")

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.cfg.sqlite_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    async def _run(self, fn, *args):
        """Execute a blocking DB function in the thread pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)

    def _create_tables(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id      TEXT,
                symbol           TEXT NOT NULL,
                direction        TEXT NOT NULL,        -- RISE / FALL
                stake            REAL NOT NULL,
                payout           REAL,
                profit           REAL,
                entry_price      REAL,
                exit_price       REAL,
                confidence       REAL,
                open_time        REAL NOT NULL,        -- Unix timestamp
                close_time       REAL,
                duration_minutes INTEGER,
                close_type       TEXT,                 -- EXPIRED / EARLY_EXIT
                won              INTEGER,              -- 1=win, 0=loss, NULL=open
                reasoning        TEXT,                 -- JSON array
                created_at       TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS indicator_snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id    INTEGER REFERENCES trades(id) ON DELETE CASCADE,
                symbol      TEXT NOT NULL,
                timestamp   REAL NOT NULL,
                rsi         REAL,
                macd_hist   REAL,
                bb_pct_b    REAL,
                bb_bandwidth REAL,
                ema10       REAL,
                ema20       REAL,
                sma50       REAL,
                atr_pct     REAL,
                momentum    REAL,
                trend       TEXT,
                patterns    TEXT,    -- JSON array of pattern names
                raw_json    TEXT     -- Full indicator snapshot as JSON
            );

            CREATE TABLE IF NOT EXISTS daily_summaries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT UNIQUE NOT NULL,
                starting_balance REAL,
                ending_balance  REAL,
                total_trades    INTEGER DEFAULT 0,
                winning_trades  INTEGER DEFAULT 0,
                losing_trades   INTEGER DEFAULT 0,
                gross_profit    REAL DEFAULT 0,
                gross_loss      REAL DEFAULT 0,
                net_pnl         REAL DEFAULT 0,
                win_rate        REAL DEFAULT 0,
                profit_factor   REAL DEFAULT 0,
                max_drawdown    REAL DEFAULT 0,
                best_trade      REAL DEFAULT 0,
                worst_trade     REAL DEFAULT 0,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS bot_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time  TEXT NOT NULL,
                end_time    TEXT,
                mode        TEXT,    -- DEMO / LIVE
                version     TEXT DEFAULT '1.0.0'
            );

            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_open_time ON trades(open_time);
            CREATE INDEX IF NOT EXISTS idx_trades_won ON trades(won);
        """)
        conn.commit()

    # ----------------------------------------------------------------
    # TRADE CRUD
    # ----------------------------------------------------------------

    async def log_trade_open(self, trade: dict) -> int:
        """Insert a new open trade record. Returns the row ID."""
        def _insert():
            conn = self._get_conn()
            cur = conn.execute("""
                INSERT INTO trades
                    (contract_id, symbol, direction, stake, entry_price,
                     confidence, open_time, duration_minutes, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(trade.get("contract_id", "")),
                trade["symbol"],
                trade["direction"],
                trade["stake"],
                trade.get("entry_price"),
                trade.get("confidence"),
                trade.get("open_time", time.time()),
                trade.get("duration_minutes"),
                json.dumps(trade.get("reasoning", [])),
            ))
            conn.commit()
            return cur.lastrowid

        row_id = await self._run(_insert)
        logger.debug(f"Trade opened in DB: row_id={row_id}")
        return row_id

    async def log_trade_close(self, trade: dict):
        """Update a trade record with close data."""
        def _update():
            conn = self._get_conn()
            conn.execute("""
                UPDATE trades SET
                    profit     = ?,
                    exit_price = ?,
                    close_time = ?,
                    close_type = ?,
                    won        = ?
                WHERE contract_id = ?
            """, (
                trade.get("profit"),
                trade.get("exit_price"),
                trade.get("close_time", time.time()),
                trade.get("close_type", "EXPIRED"),
                1 if trade.get("won") else 0,
                str(trade.get("contract_id", "")),
            ))
            conn.commit()
        await self._run(_update)

    async def log_indicator_snapshot(self, trade_id: int, symbol: str,
                                     indicators) -> None:
        """Snapshot indicator state at trade entry for later ML training."""
        def _insert():
            conn = self._get_conn()
            patterns = [p.name for p in (indicators.patterns or [])]
            raw = {
                "rsi": indicators.rsi.value,
                "rsi_signal": indicators.rsi.signal,
                "macd_hist": indicators.macd.histogram,
                "macd_dir": indicators.macd.direction,
                "bb_pct_b": indicators.bb.percent_b,
                "bb_bandwidth": indicators.bb.bandwidth,
                "bb_squeeze": indicators.bb.squeeze,
                "ema10": indicators.ma.ema10,
                "ema20": indicators.ma.ema20,
                "sma50": indicators.ma.sma50,
                "atr_pct": indicators.atr.atr_pct,
                "volatility": indicators.atr.volatility,
                "momentum": indicators.momentum,
                "trend": indicators.ma.trend,
                "patterns": patterns,
                "volume_trend": indicators.volume_trend,
            }
            conn.execute("""
                INSERT INTO indicator_snapshots
                    (trade_id, symbol, timestamp, rsi, macd_hist, bb_pct_b,
                     bb_bandwidth, ema10, ema20, sma50, atr_pct, momentum,
                     trend, patterns, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, symbol, time.time(),
                raw["rsi"], raw["macd_hist"], raw["bb_pct_b"],
                raw["bb_bandwidth"], raw["ema10"], raw["ema20"],
                raw["sma50"], raw["atr_pct"], raw["momentum"],
                raw["trend"], json.dumps(patterns), json.dumps(raw),
            ))
            conn.commit()
        await self._run(_insert)

    # ----------------------------------------------------------------
    # QUERIES
    # ----------------------------------------------------------------

    async def get_recent_trades(self, limit: int = 50) -> List[dict]:
        """Fetch most recent closed trades."""
        def _query():
            conn = self._get_conn()
            rows = conn.execute("""
                SELECT * FROM trades
                WHERE close_time IS NOT NULL
                ORDER BY close_time DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        return await self._run(_query)

    async def get_today_summary(self) -> dict:
        """Aggregate today's trade statistics."""
        def _query():
            today = str(date.today())
            conn = self._get_conn()
            row = conn.execute("""
                SELECT
                    COUNT(*)                                     AS total,
                    SUM(CASE WHEN won=1 THEN 1 ELSE 0 END)      AS wins,
                    SUM(CASE WHEN won=0 THEN 1 ELSE 0 END)      AS losses,
                    SUM(CASE WHEN profit>0 THEN profit ELSE 0 END) AS gross_profit,
                    SUM(CASE WHEN profit<0 THEN profit ELSE 0 END) AS gross_loss,
                    SUM(profit)                                  AS net_pnl,
                    MAX(profit)                                  AS best_trade,
                    MIN(profit)                                  AS worst_trade,
                    AVG(confidence)                              AS avg_confidence
                FROM trades
                WHERE date(open_time, 'unixepoch') = ?
                  AND close_time IS NOT NULL
            """, (today,)).fetchone()
            return dict(row) if row else {}
        return await self._run(_query)

    async def get_performance_stats(self, days: int = 30) -> dict:
        """Extended performance stats for the analytics dashboard."""
        def _query():
            conn = self._get_conn()
            rows = conn.execute("""
                SELECT profit, won, stake, confidence, symbol, direction
                FROM trades
                WHERE close_time IS NOT NULL
                  AND open_time > ?
            """, (time.time() - days * 86400,)).fetchall()

            trades = [dict(r) for r in rows]
            if not trades:
                return {}

            profits = [t["profit"] for t in trades if t["profit"] is not None]
            wins = [p for p in profits if p > 0]
            losses = [p for p in profits if p < 0]

            total = len(profits)
            win_count = len(wins)
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 0

            return {
                "total_trades": total,
                "win_rate": round(win_count / total * 100, 1) if total else 0,
                "net_pnl": round(sum(profits), 2),
                "gross_profit": round(gross_profit, 2),
                "gross_loss": round(gross_loss, 2),
                "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else 999,
                "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
                "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0,
                "best_trade": round(max(profits), 2) if profits else 0,
                "worst_trade": round(min(profits), 2) if profits else 0,
                "avg_confidence": round(
                    sum(t["confidence"] for t in trades if t["confidence"]) /
                    sum(1 for t in trades if t["confidence"]), 1
                ) if any(t["confidence"] for t in trades) else 0,
            }
        return await self._run(_query)

    async def get_training_data(self, min_samples: int = 100) -> List[dict]:
        """
        Extract labeled indicator snapshots for ML model training.
        Returns list of {features: [...], label: 1/0} dicts.
        """
        def _query():
            conn = self._get_conn()
            rows = conn.execute("""
                SELECT s.raw_json, t.won
                FROM indicator_snapshots s
                JOIN trades t ON s.trade_id = t.id
                WHERE t.won IS NOT NULL
                ORDER BY s.timestamp DESC
                LIMIT 10000
            """).fetchall()
            result = []
            for row in rows:
                try:
                    features = json.loads(row["raw_json"])
                    result.append({"features": features, "label": row["won"]})
                except json.JSONDecodeError:
                    continue
            return result
        return await self._run(_query)

    async def save_daily_summary(self, summary: dict):
        """Upsert end-of-day summary."""
        def _upsert():
            conn = self._get_conn()
            conn.execute("""
                INSERT INTO daily_summaries
                    (date, starting_balance, ending_balance, total_trades,
                     winning_trades, losing_trades, gross_profit, gross_loss,
                     net_pnl, win_rate, profit_factor, max_drawdown,
                     best_trade, worst_trade)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    ending_balance = excluded.ending_balance,
                    total_trades   = excluded.total_trades,
                    winning_trades = excluded.winning_trades,
                    losing_trades  = excluded.losing_trades,
                    net_pnl        = excluded.net_pnl
            """, (
                summary.get("date", str(date.today())),
                summary.get("starting_balance", 0),
                summary.get("ending_balance", 0),
                summary.get("total_trades", 0),
                summary.get("winning_trades", 0),
                summary.get("losing_trades", 0),
                summary.get("gross_profit", 0),
                summary.get("gross_loss", 0),
                summary.get("net_pnl", 0),
                summary.get("win_rate", 0),
                summary.get("profit_factor", 0),
                summary.get("max_drawdown", 0),
                summary.get("best_trade", 0),
                summary.get("worst_trade", 0),
            ))
            conn.commit()
        await self._run(_upsert)

    async def close(self):
        """Close DB connection."""
        def _close():
            if self._conn:
                self._conn.close()
        await self._run(_close)
