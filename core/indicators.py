"""
NexusTrade — Technical Indicators Engine
=========================================
Pure-Python/NumPy implementation of all technical indicators used by the bot.
All functions operate on numpy arrays and return named tuples or dicts.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class RSIResult:
    value: float
    signal: str          # "OVERBOUGHT", "OVERSOLD", "NEUTRAL"
    strength: float      # 0.0–1.0 signal confidence


@dataclass
class MACDResult:
    macd_line: float
    signal_line: float
    histogram: float
    direction: str       # "BULLISH", "BEARISH", "NEUTRAL"
    crossover: bool      # True if fresh crossover occurred


@dataclass
class BollingerResult:
    upper: float
    middle: float
    lower: float
    bandwidth: float     # (upper - lower) / middle  →  volatility proxy
    percent_b: float     # Where price sits within bands (0=lower, 1=upper)
    squeeze: bool        # True if BB is in squeeze (low bandwidth)
    signal: str          # "NEAR_UPPER", "NEAR_LOWER", "MID", "BREAKOUT"


@dataclass
class MAResult:
    ema10: float
    ema20: float
    sma50: float
    sma200: float
    trend: str           # "STRONG_UP", "UP", "DOWN", "STRONG_DOWN", "SIDEWAYS"
    golden_cross: bool   # EMA10 > EMA20 > SMA50
    death_cross: bool


@dataclass
class ATRResult:
    atr: float
    atr_pct: float       # ATR as % of price
    volatility: str      # "LOW", "NORMAL", "HIGH", "EXTREME"


@dataclass
class SupportResistanceResult:
    supports: List[float]
    resistances: List[float]
    nearest_support: float
    nearest_resistance: float
    distance_to_support_pct: float
    distance_to_resistance_pct: float
    at_support: bool
    at_resistance: bool


@dataclass
class CandlePattern:
    name: str
    direction: str       # "BULLISH", "BEARISH", "NEUTRAL"
    strength: float      # 0.0–1.0


@dataclass
class AllIndicators:
    """Complete set of indicator results for a single bar."""
    rsi: RSIResult
    macd: MACDResult
    bb: BollingerResult
    ma: MAResult
    atr: ATRResult
    sr: SupportResistanceResult
    patterns: List[CandlePattern]
    momentum: float      # Rate-of-change momentum score
    volume_trend: str    # "INCREASING", "DECREASING", "FLAT"


# ============================================================
# CORE INDICATOR FUNCTIONS
# ============================================================

def _ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    k = 2.0 / (period + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = prices[i] * k + ema[i-1] * (1 - k)
    return ema


def _sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    sma = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        sma[i] = np.mean(prices[i-period+1:i+1])
    return sma


def _wilder_rma(prices: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothed moving average (used in RSI)."""
    rma = np.zeros(len(prices))
    rma[period-1] = np.mean(prices[:period])
    for i in range(period, len(prices)):
        rma[i] = (rma[i-1] * (period - 1) + prices[i]) / period
    return rma


def compute_rsi(closes: np.ndarray, period: int = 14) -> RSIResult:
    """
    Relative Strength Index.
    Returns RSI value plus signal interpretation.
    """
    if len(closes) < period + 1:
        return RSIResult(value=50.0, signal="NEUTRAL", strength=0.0)

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = _wilder_rma(gains, period)
    avg_loss = _wilder_rma(losses, period)

    rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    rsi = float(rsi_series[-1])

    if rsi >= 70:
        signal = "OVERBOUGHT"
        strength = min(1.0, (rsi - 70) / 30)
    elif rsi <= 30:
        signal = "OVERSOLD"
        strength = min(1.0, (30 - rsi) / 30)
    else:
        signal = "NEUTRAL"
        strength = 1.0 - abs(rsi - 50) / 20  # Stronger near 50
        strength = max(0.0, min(1.0, strength))

    return RSIResult(value=round(rsi, 2), signal=signal, strength=round(strength, 3))


def compute_macd(closes: np.ndarray, fast: int = 12,
                 slow: int = 26, signal: int = 9) -> MACDResult:
    """MACD — Moving Average Convergence/Divergence."""
    if len(closes) < slow + signal:
        return MACDResult(0, 0, 0, "NEUTRAL", False)

    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line

    m = float(macd_line[-1])
    s = float(signal_line[-1])
    h = float(histogram[-1])
    h_prev = float(histogram[-2]) if len(histogram) > 1 else 0

    # Detect crossover (histogram changes sign)
    crossover = (h > 0 and h_prev <= 0) or (h < 0 and h_prev >= 0)

    if m > s and h > 0:
        direction = "BULLISH"
    elif m < s and h < 0:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    return MACDResult(
        macd_line=round(m, 4),
        signal_line=round(s, 4),
        histogram=round(h, 4),
        direction=direction,
        crossover=crossover,
    )


def compute_bollinger(closes: np.ndarray, period: int = 20,
                      std_dev: float = 2.0) -> BollingerResult:
    """Bollinger Bands with squeeze detection."""
    if len(closes) < period:
        p = float(closes[-1]) if len(closes) > 0 else 0
        return BollingerResult(p, p, p, 0, 0.5, False, "MID")

    price = float(closes[-1])
    window = closes[-period:]
    mid = float(np.mean(window))
    std = float(np.std(window, ddof=1))
    upper = mid + std_dev * std
    lower = mid - std_dev * std

    bandwidth = (upper - lower) / mid if mid != 0 else 0
    percent_b = (price - lower) / (upper - lower) if (upper - lower) != 0 else 0.5

    # Squeeze: bandwidth below 20-period average bandwidth
    squeeze = bandwidth < 0.025  # ~2.5% is tight for most synthetics

    if percent_b > 0.9:
        signal = "NEAR_UPPER"
    elif percent_b < 0.1:
        signal = "NEAR_LOWER"
    elif bandwidth > 0.05 and abs(percent_b - 0.5) > 0.3:
        signal = "BREAKOUT"
    else:
        signal = "MID"

    return BollingerResult(
        upper=round(upper, 4),
        middle=round(mid, 4),
        lower=round(lower, 4),
        bandwidth=round(bandwidth, 4),
        percent_b=round(percent_b, 4),
        squeeze=squeeze,
        signal=signal,
    )


def compute_moving_averages(closes: np.ndarray) -> MAResult:
    """All moving averages + trend determination."""
    n = len(closes)

    def safe_ema(p):
        return float(_ema(closes, p)[-1]) if n >= p else float(closes[-1])

    def safe_sma(p):
        return float(_sma(closes, p)[-1]) if n >= p else float(closes[-1])

    ema10 = safe_ema(10)
    ema20 = safe_ema(20)
    sma50 = safe_sma(50)
    sma200 = safe_sma(200)
    price = float(closes[-1])

    # Trend scoring: +1 for each bull condition
    bull_count = sum([
        price > ema10,
        price > ema20,
        price > sma50,
        ema10 > ema20,
        ema20 > sma50,
    ])

    if bull_count >= 5:
        trend = "STRONG_UP"
    elif bull_count >= 3:
        trend = "UP"
    elif bull_count <= 1:
        trend = "STRONG_DOWN"
    elif bull_count <= 2:
        trend = "DOWN"
    else:
        trend = "SIDEWAYS"

    golden_cross = (ema10 > ema20 > sma50)
    death_cross = (ema10 < ema20 < sma50)

    return MAResult(
        ema10=round(ema10, 4),
        ema20=round(ema20, 4),
        sma50=round(sma50, 4),
        sma200=round(sma200, 4),
        trend=trend,
        golden_cross=golden_cross,
        death_cross=death_cross,
    )


def compute_atr(highs: np.ndarray, lows: np.ndarray,
                closes: np.ndarray, period: int = 14) -> ATRResult:
    """Average True Range — measures volatility."""
    if len(closes) < period + 1:
        spread = float(np.mean(highs - lows)) if len(highs) > 0 else 0
        price = float(closes[-1]) if len(closes) > 0 else 1
        return ATRResult(spread, spread/price*100, "NORMAL")

    # True range = max of (H-L, |H-prev_C|, |L-prev_C|)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    atr = float(_wilder_rma(tr, period)[-1])
    price = float(closes[-1])
    atr_pct = (atr / price * 100) if price != 0 else 0

    if atr_pct < 0.3:
        volatility = "LOW"
    elif atr_pct < 1.0:
        volatility = "NORMAL"
    elif atr_pct < 2.5:
        volatility = "HIGH"
    else:
        volatility = "EXTREME"

    return ATRResult(atr=round(atr, 4), atr_pct=round(atr_pct, 3), volatility=volatility)


def compute_support_resistance(highs: np.ndarray, lows: np.ndarray,
                               closes: np.ndarray,
                               lookback: int = 50) -> SupportResistanceResult:
    """
    Find key support and resistance levels using pivot-point method.
    Clusters nearby levels for cleaner output.
    """
    price = float(closes[-1])
    h = highs[-lookback:]
    l = lows[-lookback:]

    # Find pivot highs and lows (local maxima/minima)
    resistances, supports = [], []

    for i in range(2, len(h) - 2):
        if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
            resistances.append(float(h[i]))
        if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
            supports.append(float(l[i]))

    def cluster_levels(levels: list, tolerance_pct: float = 0.3) -> list:
        """Merge nearby levels into clusters."""
        if not levels:
            return []
        levels = sorted(set(levels))
        clusters = [levels[0]]
        for lv in levels[1:]:
            if abs(lv - clusters[-1]) / clusters[-1] * 100 < tolerance_pct:
                clusters[-1] = (clusters[-1] + lv) / 2  # Average merge
            else:
                clusters.append(lv)
        return clusters

    supports = cluster_levels(supports)
    resistances = cluster_levels(resistances)

    # Nearest levels to current price
    sup_below = [s for s in supports if s < price]
    res_above = [r for r in resistances if r > price]

    nearest_sup = max(sup_below) if sup_below else price * 0.98
    nearest_res = min(res_above) if res_above else price * 1.02

    dist_sup = (price - nearest_sup) / price * 100
    dist_res = (nearest_res - price) / price * 100

    at_support = dist_sup < 0.5      # Within 0.5% of support
    at_resistance = dist_res < 0.5   # Within 0.5% of resistance

    return SupportResistanceResult(
        supports=supports[-5:],
        resistances=resistances[-5:],
        nearest_support=round(nearest_sup, 4),
        nearest_resistance=round(nearest_res, 4),
        distance_to_support_pct=round(dist_sup, 3),
        distance_to_resistance_pct=round(dist_res, 3),
        at_support=at_support,
        at_resistance=at_resistance,
    )


# ============================================================
# CANDLESTICK PATTERN RECOGNITION
# ============================================================

def compute_candle_patterns(opens: np.ndarray, highs: np.ndarray,
                            lows: np.ndarray, closes: np.ndarray) -> List[CandlePattern]:
    """
    Detect common candlestick reversal and continuation patterns.
    Returns a list of detected patterns on the most recent candles.
    """
    patterns = []
    if len(closes) < 3:
        return patterns

    o, h, l, c = opens[-3:], highs[-3:], lows[-3:], closes[-3:]

    # Body and shadow sizes
    body = lambda i: abs(c[i] - o[i])
    upper_shadow = lambda i: h[i] - max(o[i], c[i])
    lower_shadow = lambda i: min(o[i], c[i]) - l[i]
    candle_range = lambda i: h[i] - l[i]

    # ---- Single candle patterns ----

    # Doji (very small body)
    if candle_range(2) > 0 and body(2) / candle_range(2) < 0.1:
        patterns.append(CandlePattern("DOJI", "NEUTRAL", 0.5))

    # Hammer (small body, long lower shadow, short upper shadow)
    if (candle_range(2) > 0 and
            lower_shadow(2) > 2 * body(2) and
            upper_shadow(2) < body(2) * 0.3 and
            body(2) / candle_range(2) < 0.4):
        direction = "BULLISH" if c[1] < o[1] else "NEUTRAL"  # Bullish if prior trend down
        patterns.append(CandlePattern("HAMMER", direction, 0.75))

    # Shooting star (small body, long upper shadow)
    if (candle_range(2) > 0 and
            upper_shadow(2) > 2 * body(2) and
            lower_shadow(2) < body(2) * 0.3):
        patterns.append(CandlePattern("SHOOTING_STAR", "BEARISH", 0.72))

    # Marubozu (full body, no shadows)
    if candle_range(2) > 0 and body(2) / candle_range(2) > 0.9:
        direction = "BULLISH" if c[2] > o[2] else "BEARISH"
        patterns.append(CandlePattern("MARUBOZU", direction, 0.8))

    # ---- Two-candle patterns ----

    # Bullish Engulfing
    if (o[2] < c[1] and          # Open below prev close
            c[2] > o[1] and       # Close above prev open
            c[1] < o[1] and       # Previous candle bearish
            c[2] > o[2]):         # Current candle bullish
        patterns.append(CandlePattern("BULL_ENGULFING", "BULLISH", 0.82))

    # Bearish Engulfing
    if (o[2] > c[1] and
            c[2] < o[1] and
            c[1] > o[1] and
            c[2] < o[2]):
        patterns.append(CandlePattern("BEAR_ENGULFING", "BEARISH", 0.82))

    # ---- Three-candle patterns ----

    # Morning Star
    if (c[0] < o[0] and            # First: bearish
            body(1) < 0.3 * body(0) and  # Second: small doji-ish
            c[2] > o[2] and        # Third: bullish
            c[2] > (o[0] + c[0]) / 2):   # Third closes above midpoint of first
        patterns.append(CandlePattern("MORNING_STAR", "BULLISH", 0.88))

    # Evening Star
    if (c[0] > o[0] and
            body(1) < 0.3 * body(0) and
            c[2] < o[2] and
            c[2] < (o[0] + c[0]) / 2):
        patterns.append(CandlePattern("EVENING_STAR", "BEARISH", 0.88))

    # Three White Soldiers
    if all(c[i] > o[i] for i in range(3)) and c[2] > c[1] > c[0]:
        patterns.append(CandlePattern("THREE_WHITE_SOLDIERS", "BULLISH", 0.85))

    # Three Black Crows
    if all(c[i] < o[i] for i in range(3)) and c[2] < c[1] < c[0]:
        patterns.append(CandlePattern("THREE_BLACK_CROWS", "BEARISH", 0.85))

    return patterns


def compute_momentum(closes: np.ndarray, period: int = 14) -> float:
    """
    Rate-of-change momentum, normalized to -1 … +1.
    Positive = accelerating uptrend, Negative = accelerating downtrend.
    """
    if len(closes) < period + 1:
        return 0.0
    roc = (closes[-1] - closes[-period]) / closes[-period]
    return float(np.clip(roc * 100, -1, 1))


def compute_volume_trend(volumes: np.ndarray, period: int = 10) -> str:
    """Determine if volume is increasing, decreasing, or flat."""
    if len(volumes) < period:
        return "FLAT"
    recent = np.mean(volumes[-period//2:])
    older = np.mean(volumes[-period:-period//2])
    if older == 0:
        return "FLAT"
    ratio = recent / older
    if ratio > 1.15:
        return "INCREASING"
    elif ratio < 0.85:
        return "DECREASING"
    return "FLAT"


# ============================================================
# MASTER COMPUTE FUNCTION
# ============================================================

def compute_all(candles: list) -> Optional[AllIndicators]:
    """
    Compute every indicator from a list of OHLCV candle dicts.
    Each candle: {"time": epoch, "open": f, "high": f, "low": f, "close": f, "volume": f}
    Returns None if not enough data.
    """
    if len(candles) < 30:
        logger.warning(f"Not enough candles to compute indicators ({len(candles)} < 30)")
        return None

    o = np.array([c["open"] for c in candles], dtype=float)
    h = np.array([c["high"] for c in candles], dtype=float)
    l = np.array([c["low"] for c in candles], dtype=float)
    c = np.array([c["close"] for c in candles], dtype=float)
    v = np.array([candle.get("volume", 1) for candle in candles], dtype=float)

    rsi = compute_rsi(c)
    macd = compute_macd(c)
    bb = compute_bollinger(c)
    ma = compute_moving_averages(c)
    atr = compute_atr(h, l, c)
    sr = compute_support_resistance(h, l, c)
    patterns = compute_candle_patterns(o, h, l, c)
    momentum = compute_momentum(c)
    vol_trend = compute_volume_trend(v)

    return AllIndicators(
        rsi=rsi, macd=macd, bb=bb, ma=ma, atr=atr,
        sr=sr, patterns=patterns, momentum=momentum,
        volume_trend=vol_trend,
    )
