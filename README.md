# NexusTrade — AI-Powered Deriv Trading Bot

An advanced algorithmic trading system for [Deriv](https://deriv.com) synthetic indices with multi-factor AI signal scoring, full risk management, and a real-time dashboard.

---

## Architecture

```
nexustrade/
├── bot.py                    # Main orchestrator — runs the full pipeline
├── config.py                 # All configuration (loaded from .env)
├── core/
│   ├── deriv_client.py       # Deriv WebSocket API client
│   ├── indicators.py         # RSI, MACD, BB, MA, ATR, S/R, patterns
│   └── signal_engine.py      # AI multi-factor confidence scorer
├── risk/
│   └── risk_manager.py       # Trade gating, stake sizing, daily limits
├── ml/
│   └── signal_model.py       # GBM classifier for win probability
├── db/
│   └── trade_logger.py       # SQLite persistence (trade log + analytics)
├── api/
│   └── server.py             # FastAPI REST + WebSocket server
├── backtest/
│   └── backtester.py         # Walk-forward backtesting engine
├── notifications/
│   └── notifier.py           # Telegram + Discord alerts
└── utils/
    └── logger_setup.py       # Rotating file + console logger
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — set your DERIV_API_TOKEN
```

### 3. Get a Deriv API token
- Log in to [app.deriv.com](https://app.deriv.com)
- Go to **Account Settings → API Token**
- Create a token with **Read, Trade, Payments** scopes
- Start with `DEMO_MODE=true` to test safely

### 4. Run the bot (API + bot server)
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### 5. Run the bot standalone (no web server)
```bash
python bot.py
```

---

## Signal Engine

Each potential trade is scored across six components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Trend | 25% | MA alignment (EMA10/20, SMA50/200) |
| Momentum | 20% | RSI extremes + MACD crossovers |
| Pattern | 20% | Candlestick recognition (10+ patterns) |
| Bollinger | 15% | Band position and squeeze/breakout |
| Support/Resistance | 10% | Proximity to key price levels |
| Volatility Filter | 10% | ATR-based environment quality |

A trade is only opened when the final confidence score exceeds `MIN_CONFIDENCE_SCORE` (default: 75/100).

---

## Risk Management

All risk rules enforce automatically:

- **Max risk per trade** — configurable % of balance (default 2.5%)
- **Daily profit target** — bot stops when reached (default +5%)
- **Daily loss limit** — hard stop (default -2%)
- **Consecutive loss protection** — pause + cool-down after N losses (default 3)
- **Anti-martingale** — scale stake on winning streaks, reset on losses
- **Recovery mode** — halved stakes after recent losses
- **Minimum balance guard** — emergency stop

---

## Technical Indicators

All indicators are implemented in pure NumPy with no external TA library dependency:

- **RSI (14)** — overbought/oversold detection
- **MACD (12/26/9)** — momentum and crossover signals
- **Bollinger Bands (20, 2σ)** — squeeze, breakout, mean-reversion
- **EMA 10/20 + SMA 50/200** — trend direction and golden/death cross
- **ATR (14)** — volatility measurement and trade filter
- **Support/Resistance** — pivot-point clustering over 50 bars
- **Candlestick Patterns** — 10 patterns including Hammer, Engulfing, Morning/Evening Star, Doji

---

## ML Model

The ML component trains a **Gradient Boosting Classifier** on historical trade outcomes:

- Features: 17-dimensional vector from indicator state at entry
- Target: 1 = trade won, 0 = trade lost
- Retrained automatically every 100 closed trades
- Outputs win probability — used to boost/penalize the base confidence score
- Falls back to rule-based scoring when insufficient data

---

## Backtesting

```python
from backtest.backtester import Backtester

candles = [...]  # List of OHLCV dicts
bt = Backtester(candles, initial_balance=1000.0)
results = bt.run()
Backtester.print_report(results)
```

Output includes: net P&L, win rate, profit factor, Sharpe ratio, max drawdown, equity curve.

---

## Dashboard API

The FastAPI server exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Full bot status |
| `/api/balance` | GET | Current balance + daily P&L |
| `/api/trades` | GET | Recent trade history |
| `/api/performance` | GET | Rolling statistics |
| `/api/indicators/{symbol}` | GET | Live indicator values |
| `/api/bot/start` | POST | Start trading |
| `/api/bot/stop` | POST | Stop trading |
| `/api/bot/pause` | POST | Toggle pause |
| `/api/config` | POST | Update risk/strategy params |
| `/api/backtest` | POST | Run backtest |
| `/api/ml/train` | POST | Retrain ML model |
| `/ws/live` | WebSocket | Real-time data stream |

---

## Deployment (VPS)

```bash
# Install with systemd service
sudo cp deploy/nexustrade.service /etc/systemd/system/
sudo systemctl enable nexustrade
sudo systemctl start nexustrade

# Or with Docker
docker build -t nexustrade .
docker run -d --env-file .env -p 8000:8000 nexustrade
```

---

## Important Disclaimers

- **This bot does not guarantee profits.** All trading involves risk of loss.
- Binary options / synthetic indices carry high risk — never trade money you cannot afford to lose.
- Always test thoroughly in **DEMO mode** before enabling live trading.
- Past backtest performance does not predict future results.
- The developer is not responsible for any financial losses.

---

## License

MIT — use freely, modify as needed. Credit appreciated but not required.

