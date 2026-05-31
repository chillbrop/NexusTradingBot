"""
NexusTrade — FastAPI Dashboard Backend
========================================
Provides REST endpoints and WebSocket streaming for the React dashboard.

Endpoints:
  GET  /api/status           — bot status snapshot
  GET  /api/balance          — current account balance
  GET  /api/trades           — recent trade history
  GET  /api/performance      — rolling performance stats
  GET  /api/indicators/{sym} — latest indicator values
  POST /api/bot/start        — start bot
  POST /api/bot/stop         — stop bot
  POST /api/bot/pause        — pause/resume
  POST /api/config           — update live config
  POST /api/backtest         — run backtest on stored data
  WS   /ws/live              — real-time data stream

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# App
# ----------------------------------------------------------------
app = FastAPI(
    title="NexusTrade API",
    description="AI Trading Bot for Deriv — Dashboard Backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global bot reference — injected at startup
_bot = None
_db = None
_ws_clients: List[WebSocket] = []


# ----------------------------------------------------------------
# Startup / Shutdown
# ----------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Initialize database and start bot in background."""
    global _bot, _db
    from bot import TradingBot
    from db.trade_logger import TradeLogger

    _db = TradeLogger()
    await _db.initialize()
    _bot = TradingBot()

    # Start the bot in a background task
    asyncio.create_task(_bot.run())

    # Start WebSocket broadcaster
    asyncio.create_task(_broadcast_loop())
    logger.info("NexusTrade server started")


@app.on_event("shutdown")
async def shutdown():
    if _bot:
        await _bot.shutdown()
    if _db:
        await _db.close()


# ----------------------------------------------------------------
# REST Endpoints
# ----------------------------------------------------------------

@app.get("/api/status")
async def get_status():
    if not _bot:
        raise HTTPException(503, "Bot not initialized")
    return JSONResponse(_bot.get_status())


@app.get("/api/balance")
async def get_balance():
    if not _bot:
        raise HTTPException(503, "Bot not initialized")
    risk = _bot.risk_manager.get_status_summary()
    return JSONResponse({
        "balance": risk["balance"],
        "daily_pnl": risk["daily_pnl"],
        "daily_pnl_pct": risk["daily_pnl_pct"],
    })


@app.get("/api/trades")
async def get_trades(limit: int = 50):
    if not _db:
        raise HTTPException(503, "DB not initialized")
    trades = await _db.get_recent_trades(limit=limit)
    return JSONResponse({"trades": trades})


@app.get("/api/performance")
async def get_performance(days: int = 30):
    if not _db:
        raise HTTPException(503, "DB not initialized")
    stats = await _db.get_performance_stats(days=days)
    today = await _db.get_today_summary()
    return JSONResponse({"stats": stats, "today": today})


@app.get("/api/indicators/{symbol}")
async def get_indicators(symbol: str):
    if not _bot:
        raise HTTPException(503, "Bot not initialized")
    ind = _bot._last_indicators.get(symbol)
    if ind is None:
        raise HTTPException(404, f"No indicator data for {symbol}")
    return JSONResponse({
        "symbol": symbol,
        "rsi": {"value": ind.rsi.value, "signal": ind.rsi.signal},
        "macd": {"histogram": ind.macd.histogram, "direction": ind.macd.direction},
        "bb": {"percent_b": ind.bb.percent_b, "bandwidth": ind.bb.bandwidth,
               "squeeze": ind.bb.squeeze},
        "ma": {"trend": ind.ma.trend, "ema10": ind.ma.ema10, "ema20": ind.ma.ema20,
               "golden_cross": ind.ma.golden_cross},
        "atr": {"atr_pct": ind.atr.atr_pct, "volatility": ind.atr.volatility},
        "momentum": ind.momentum,
        "patterns": [{"name": p.name, "direction": p.direction, "strength": p.strength}
                     for p in ind.patterns],
    })


@app.get("/api/candles/{symbol}")
async def get_candles(symbol: str, granularity: int = 60, count: int = 200):
    if not _bot:
        raise HTTPException(503, "Bot not initialized")
    # Return from buffer if available
    buf = _bot.candle_buffers.get(symbol, [])
    if buf:
        return JSONResponse({"candles": buf[-count:]})
    # Fetch from API
    candles = await _bot.client.get_candles(symbol, granularity, count)
    return JSONResponse({"candles": candles})


# ----------------------------------------------------------------
# Bot Control Endpoints
# ----------------------------------------------------------------

@app.post("/api/bot/start")
async def start_bot():
    if not _bot:
        raise HTTPException(503, "Bot not initialized")
    _bot.resume()
    return {"status": "started"}


@app.post("/api/bot/stop")
async def stop_bot():
    if not _bot:
        raise HTTPException(503, "Bot not initialized")
    _bot.pause("Manual stop via API")
    return {"status": "stopped"}


@app.post("/api/bot/pause")
async def pause_bot(data: dict = {}):
    if not _bot:
        raise HTTPException(503, "Bot not initialized")
    reason = data.get("reason", "Manual pause")
    if _bot._paused:
        _bot.resume()
        return {"status": "resumed"}
    else:
        _bot.pause(reason)
        return {"status": "paused"}


class ConfigUpdate(BaseModel):
    max_risk_per_trade_pct: Optional[float] = None
    daily_profit_target_pct: Optional[float] = None
    daily_loss_limit_pct: Optional[float] = None
    min_confidence_score: Optional[float] = None
    default_duration_minutes: Optional[int] = None
    default_stake: Optional[float] = None


@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    """Live config update without restarting the bot."""
    from config import config
    risk = config.risk
    strategy = config.strategy

    changes = {}
    if update.max_risk_per_trade_pct is not None:
        risk.max_risk_per_trade_pct = update.max_risk_per_trade_pct
        changes["max_risk_per_trade_pct"] = update.max_risk_per_trade_pct
    if update.daily_profit_target_pct is not None:
        risk.daily_profit_target_pct = update.daily_profit_target_pct
        changes["daily_profit_target_pct"] = update.daily_profit_target_pct
    if update.daily_loss_limit_pct is not None:
        risk.daily_loss_limit_pct = update.daily_loss_limit_pct
        changes["daily_loss_limit_pct"] = update.daily_loss_limit_pct
    if update.min_confidence_score is not None:
        strategy.min_confidence_score = update.min_confidence_score
        changes["min_confidence_score"] = update.min_confidence_score
    if update.default_duration_minutes is not None:
        strategy.default_duration_minutes = update.default_duration_minutes
        changes["default_duration_minutes"] = update.default_duration_minutes

    logger.info(f"Config updated: {changes}")
    return {"updated": changes}


class BacktestRequest(BaseModel):
    symbol: str = "R_100"
    initial_balance: float = 1000.0
    granularity: int = 60
    candle_count: int = 500
    trade_duration_bars: int = 5


@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest):
    if not _bot:
        raise HTTPException(503, "Bot not initialized")

    candles = await _bot.client.get_candles(
        req.symbol, req.granularity, req.candle_count
    )
    if len(candles) < 100:
        raise HTTPException(400, f"Insufficient candle data ({len(candles)} candles)")

    from backtest.backtester import Backtester
    bt = Backtester(
        candles=candles,
        initial_balance=req.initial_balance,
        trade_duration_bars=req.trade_duration_bars,
    )
    results = bt.run()

    return JSONResponse({
        "initial_balance": results.initial_balance,
        "final_balance": results.final_balance,
        "net_pnl": results.net_pnl,
        "pnl_pct": results.pnl_pct,
        "total_trades": results.total_trades,
        "win_rate": results.win_rate,
        "profit_factor": results.profit_factor,
        "sharpe_ratio": results.sharpe_ratio,
        "max_drawdown": results.max_drawdown,
        "max_drawdown_pct": results.max_drawdown_pct,
        "avg_win": results.avg_win,
        "avg_loss": results.avg_loss,
        "best_trade": results.best_trade,
        "worst_trade": results.worst_trade,
        "signals_generated": results.signals_generated,
        "signals_skipped": results.signals_skipped,
        "equity_curve": results.equity_curve,
        "trades": [
            {
                "index": t.index,
                "direction": t.direction,
                "profit": t.profit,
                "won": t.won,
                "confidence": t.confidence,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
            }
            for t in results.trades[:200]  # Cap payload size
        ],
    })


@app.post("/api/ml/train")
async def trigger_ml_training():
    """Manually trigger ML model retraining from trade history."""
    if not _db:
        raise HTTPException(503, "DB not initialized")
    training_data = await _db.get_training_data()
    if len(training_data) < 200:
        return {"error": "Not enough training data", "samples": len(training_data)}

    from ml.signal_model import SignalModel
    model = SignalModel()
    metrics = model.train(training_data)
    return {"metrics": metrics}


# ----------------------------------------------------------------
# WebSocket — Real-time streaming
# ----------------------------------------------------------------

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info(f"WS client connected ({len(_ws_clients)} total)")
    try:
        while True:
            # Keep connection alive — receive pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WS client error: {e}")
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


async def _broadcast_loop():
    """Broadcast bot status + indicator updates to all WS clients every second."""
    while True:
        await asyncio.sleep(1.0)
        if not _ws_clients or not _bot:
            continue

        try:
            status = _bot.get_status()
            payload = json.dumps({"type": "status", "data": status})
            dead = []
            for ws in _ws_clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                _ws_clients.remove(ws)
        except Exception as e:
            logger.debug(f"Broadcast error: {e}")
