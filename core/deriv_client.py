"""
NexusTrade — Deriv API Client
==============================
Manages the WebSocket connection to the Deriv API.
Compatible with websockets >= 10.x and >= 12.x (API changed significantly).

Key fixes vs v1:
  - ws.open removed in websockets 10+  → use ws.state == OPEN or try/except
  - websockets.connect() returns a context manager in v10+ (not a coroutine)
  - ClientConnection vs WebSocketClientProtocol naming changed in v12
"""

import asyncio
import json
import logging
import time
from typing import Callable, Dict, Optional, Set

import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

# Compatibility: websockets 10+ uses State enum; older versions had .open bool
try:
    from websockets.connection import State as WS_State
    def _ws_is_open(ws) -> bool:
        return ws.state is WS_State.OPEN
except ImportError:
    # Fallback for very old versions
    def _ws_is_open(ws) -> bool:
        return getattr(ws, "open", False)

from config import config

logger = logging.getLogger(__name__)


class DerivAPIError(Exception):
    """Raised when Deriv returns an API-level error response."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class DerivClient:
    """
    Async WebSocket client for the Deriv API.

    Handles:
      - Secure connection + re-authentication on reconnect
      - Request/response matching via req_id
      - Subscription routing by subscription ID or symbol
      - Exponential back-off reconnection
      - Keepalive pings

    Usage:
        client = DerivClient()
        await client.connect()          # blocks — runs the reconnect loop
    """

    def __init__(self):
        self._ws = None                           # active WebSocket connection
        self._request_id: int = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._subscriptions: Dict[str, Callable] = {}   # sub_id → callback
        self._tick_callbacks: Dict[str, Callable] = {}  # symbol → callback
        self._candle_callbacks: Dict[str, Callable] = {}
        self._reconnect_attempts: int = 0
        self._is_authorized: bool = False
        self._active_subscriptions: Set[str] = set()
        self._running: bool = False
        self._keepalive_task: Optional[asyncio.Task] = None
        self.on_connect: Optional[Callable] = None
        self.on_disconnect: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Endpoint builder
    # ------------------------------------------------------------------

    @property
    def endpoint(self) -> str:
        app_id = config.deriv.app_id
        return f"wss://ws.binaryws.com/websockets/v3?app_id={app_id}"

    # ------------------------------------------------------------------
    # Public: connect / disconnect
    # ------------------------------------------------------------------

    async def connect(self):
        """
        Start the connection loop. Blocks until self._running is False.
        Call disconnect() to stop gracefully.
        """
        self._running = True
        await self._connection_loop()

    async def disconnect(self):
        """Gracefully stop and close the WebSocket."""
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal: connection loop with exponential back-off
    # ------------------------------------------------------------------

    async def _connection_loop(self):
        while self._running:
            try:
                logger.info(
                    f"Connecting to Deriv API "
                    f"({'DEMO' if config.deriv.is_demo else 'LIVE'})... "
                    f"[attempt {self._reconnect_attempts + 1}]"
                )
                # websockets.connect() is an async context manager in v10+
                async with websockets.connect(
                    self.endpoint,
                    ping_interval=30,
                    ping_timeout=15,
                    close_timeout=5,
                    max_size=2**22,          # 4 MB — handle large candle payloads
                ) as ws:
                    self._ws = ws
                    self._reconnect_attempts = 0  # Reset on successful connect

                    # Authorize first
                    await self._authorize()

                    # Notify bot that we're live
                    if self.on_connect:
                        await self.on_connect()

                    # Start keepalive pings
                    self._keepalive_task = asyncio.create_task(self._keepalive())

                    # Block here — process messages until connection drops
                    await self._listen()

            except (ConnectionClosedError, ConnectionClosedOK) as e:
                logger.warning(f"WebSocket closed: {e}")
            except websockets.exceptions.WebSocketException as e:
                logger.error(f"WebSocket error: {e}")
            except OSError as e:
                logger.error(f"Network error: {e}")
            except Exception as e:
                logger.error(f"Unexpected connection error: {e}", exc_info=True)
            finally:
                self._ws = None
                self._is_authorized = False
                # Cancel keepalive
                if self._keepalive_task and not self._keepalive_task.done():
                    self._keepalive_task.cancel()
                    try:
                        await self._keepalive_task
                    except asyncio.CancelledError:
                        pass
                # Fail all pending requests
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(ConnectionError("WebSocket disconnected"))
                self._pending.clear()

                if self.on_disconnect:
                    try:
                        await self.on_disconnect()
                    except Exception:
                        pass

            if not self._running:
                break

            # Exponential back-off: 1s, 2s, 4s … capped at 60s
            wait = min(2 ** self._reconnect_attempts, 60)
            logger.info(f"Reconnecting in {wait}s...")
            await asyncio.sleep(wait)
            self._reconnect_attempts += 1

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------

    async def _authorize(self):
        token = config.deriv.api_token
        if not token:
            raise ValueError(
                "DERIV_API_TOKEN is not set. "
                "Add it to your .env file and restart."
            )
        resp = await self._send({"authorize": token})
        account = resp.get("authorize", {})
        self._is_authorized = True
        logger.info(
            f"Authorized ✓ | "
            f"Account: {account.get('email', 'unknown')} | "
            f"Balance: {account.get('balance')} {account.get('currency', 'USD')} | "
            f"{'DEMO' if account.get('is_virtual') else 'LIVE'}"
        )
        return account

    # ------------------------------------------------------------------
    # Message receive loop
    # ------------------------------------------------------------------

    async def _listen(self):
        """Read messages from the WebSocket and route them."""
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
                await self._dispatch(msg)
            except json.JSONDecodeError:
                logger.warning(f"Non-JSON message: {str(raw)[:80]}")
            except Exception as e:
                logger.error(f"Dispatch error: {e}", exc_info=True)

    async def _dispatch(self, msg: dict):
        """Route an inbound message to the correct waiter or callback."""
        req_id = msg.get("req_id")

        # 1. Resolve a waiting send() call
        if req_id and req_id in self._pending:
            fut = self._pending.pop(req_id)
            if not fut.done():
                fut.set_result(msg)
            return

        # 2. Route subscription stream by subscription ID
        sub_id = msg.get("subscription", {}).get("id")
        if sub_id and sub_id in self._subscriptions:
            cb = self._subscriptions[sub_id]
            if asyncio.iscoroutinefunction(cb):
                await cb(msg)
            else:
                cb(msg)
            return

        # 3. Route tick/ohlc by symbol (fallback)
        msg_type = msg.get("msg_type")
        if msg_type == "tick":
            sym = msg.get("tick", {}).get("symbol")
            if sym and sym in self._tick_callbacks:
                await self._tick_callbacks[sym](msg["tick"])
        elif msg_type == "ohlc":
            sym = msg.get("ohlc", {}).get("symbol")
            if sym and sym in self._candle_callbacks:
                await self._candle_callbacks[sym](msg["ohlc"])

    # ------------------------------------------------------------------
    # Keepalive
    # ------------------------------------------------------------------

    async def _keepalive(self):
        """Send a ping every 25 s to prevent server-side timeout."""
        while self._running:
            await asyncio.sleep(25)
            if self._ws is None:
                break
            try:
                await self._send_raw({"ping": 1})
            except Exception:
                break

    # ------------------------------------------------------------------
    # Core send / receive
    # ------------------------------------------------------------------

    async def _send(self, payload: dict, timeout: float = 20.0) -> dict:
        """
        Send a JSON request and await the matching response.
        Thread-safe: uses per-request asyncio.Future.
        """
        if self._ws is None or not _ws_is_open(self._ws):
            raise ConnectionError("WebSocket is not connected")

        self._request_id += 1
        req_id = self._request_id
        payload = {**payload, "req_id": req_id}

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut

        await self._ws.send(json.dumps(payload))

        try:
            result = await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise TimeoutError(
                f"No response within {timeout}s "
                f"(req_id={req_id}, type={list(payload.keys())[0]})"
            )

        # Surface API-level errors as exceptions
        if "error" in result:
            err = result["error"]
            raise DerivAPIError(
                err.get("code", "UNKNOWN"),
                err.get("message", "Unknown API error"),
            )

        return result

    async def _send_raw(self, payload: dict):
        """Fire-and-forget — no response expected (used for ping)."""
        if self._ws and _ws_is_open(self._ws):
            await self._ws.send(json.dumps(payload))

    # ------------------------------------------------------------------
    # Account methods
    # ------------------------------------------------------------------

    async def get_balance(self) -> dict:
        resp = await self._send({"balance": 1, "account": "current"})
        return resp.get("balance", {})

    async def get_profit_table(self, limit: int = 50) -> dict:
        resp = await self._send({
            "profit_table": 1,
            "description": 1,
            "limit": limit,
            "sort": "DESC",
        })
        return resp.get("profit_table", {})

    async def get_portfolio(self) -> dict:
        resp = await self._send({"portfolio": 1})
        return resp.get("portfolio", {})

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_candles(
        self, symbol: str, granularity: int = 60, count: int = 200
    ) -> list:
        """
        Fetch historical OHLC candles.

        Args:
            symbol:      e.g. "R_100"
            granularity: Seconds per candle — 60, 120, 300, 600, 900, 1800, 3600, 86400
            count:       Number of candles (max 5000)
        """
        resp = await self._send({
            "ticks_history": symbol,
            "style": "candles",
            "granularity": granularity,
            "count": count,
            "end": "latest",
            "adjust_start_time": 1,
        }, timeout=30.0)

        return [
            {
                "time":  int(c["epoch"]),
                "open":  float(c["open"]),
                "high":  float(c["high"]),
                "low":   float(c["low"]),
                "close": float(c["close"]),
                "volume": 1.0,   # Deriv doesn't provide volume on synthetics
            }
            for c in resp.get("candles", [])
        ]

    async def subscribe_ticks(self, symbol: str, callback: Callable):
        """Subscribe to live tick stream."""
        resp = await self._send({"ticks": symbol, "subscribe": 1})
        sub_id = resp.get("subscription", {}).get("id")
        if sub_id:
            # Wrap callback to extract the tick sub-object
            async def _cb(msg: dict):
                await callback(msg.get("tick", {}))
            self._subscriptions[sub_id] = _cb
            self._tick_callbacks[symbol] = callback
            self._active_subscriptions.add(symbol)
            logger.info(f"Subscribed to ticks: {symbol}")

    async def subscribe_candles(
        self, symbol: str, granularity: int, callback: Callable
    ):
        """Subscribe to live OHLC candle stream."""
        resp = await self._send({
            "ticks_history": symbol,
            "style": "candles",
            "granularity": granularity,
            "subscribe": 1,
            "end": "latest",
            "count": 1,
        })
        sub_id = resp.get("subscription", {}).get("id")
        if sub_id:
            async def _cb(msg: dict):
                await callback(msg.get("ohlc", {}))
            self._subscriptions[sub_id] = _cb
            self._candle_callbacks[symbol] = callback
            logger.info(f"Subscribed to candles: {symbol} ({granularity}s)")

    async def get_active_symbols(self) -> list:
        resp = await self._send({
            "active_symbols": "brief",
            "product_type": "basic",
        })
        return resp.get("active_symbols", [])

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------

    async def buy_contract(
        self,
        symbol: str,
        contract_type: str,
        duration: int,
        duration_unit: str,
        amount: float,
        basis: str = "stake",
        currency: str = "USD",
    ) -> dict:
        """
        Open a binary options trade.

        Args:
            symbol:        Market, e.g. "R_100"
            contract_type: "CALL" (rise) or "PUT" (fall)
            duration:      Numeric duration value
            duration_unit: "t" ticks | "s" seconds | "m" minutes | "h" hours | "d" days
            amount:        Stake in account currency
            basis:         "stake" or "payout"
        """
        # Step 1: get a price proposal
        proposal_resp = await self._send({
            "proposal": 1,
            "amount": str(round(amount, 2)),
            "basis": basis,
            "contract_type": contract_type,
            "currency": currency,
            "duration": duration,
            "duration_unit": duration_unit,
            "symbol": symbol,
        })

        proposal = proposal_resp.get("proposal", {})
        proposal_id = proposal.get("id")
        ask_price = proposal.get("ask_price", amount)

        if not proposal_id:
            raise DerivAPIError("NO_PROPOSAL", "Server returned no proposal ID")

        # Step 2: buy
        buy_resp = await self._send({
            "buy": proposal_id,
            "price": ask_price,
        })

        contract = buy_resp.get("buy", {})
        logger.info(
            f"Trade opened — ID: {contract.get('contract_id')} | "
            f"{contract_type} {symbol} | "
            f"Stake: ${amount:.2f} | "
            f"Duration: {duration}{duration_unit}"
        )
        return contract

    async def sell_contract(self, contract_id: int, price: float = 0) -> dict:
        """
        Early exit — sell an open contract at current market value.
        price=0 means sell at any price (immediate).
        """
        resp = await self._send({
            "sell": contract_id,
            "price": price,
        })
        return resp.get("sell", {})

    async def get_open_contracts(self) -> list:
        resp = await self._send({"portfolio": 1})
        return resp.get("portfolio", {}).get("contracts", [])

    async def subscribe_open_contract(
        self, contract_id: int, callback: Callable
    ) -> str:
        """Subscribe to P&L updates for an open contract."""
        resp = await self._send({
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 1,
        })
        sub_id = resp.get("subscription", {}).get("id")
        if sub_id:
            async def _cb(msg: dict):
                await callback(msg.get("proposal_open_contract", {}))
            self._subscriptions[sub_id] = _cb
        return sub_id or ""
