"""
NexusTrade — AI Signal Scoring Engine
=======================================
Converts raw indicator data into a unified confidence score (0–100),
determines trade direction, and gates entries based on configurable thresholds.

The scoring engine uses a weighted multi-factor model:
  - Technical indicator consensus
  - Candlestick pattern quality
  - Support/Resistance positioning
  - Momentum alignment
  - Volatility filter
  - (Optional) ML model probability output

Each sub-score is weighted, normalized, then blended into a single confidence.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from core.indicators import AllIndicators
from config import config

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Complete signal output from the scoring engine."""
    direction: str              # "RISE", "FALL", or "HOLD"
    confidence: float           # 0–100 — overall AI confidence
    stake_modifier: float       # Multiplier on base stake (0.5–2.0)
    duration_minutes: int       # Recommended trade duration
    reasoning: list             # Human-readable decision log
    sub_scores: dict            # Per-component scores for dashboard display

    @property
    def should_trade(self) -> bool:
        """True if confidence clears the minimum threshold."""
        return (
            self.direction in ("RISE", "FALL") and
            self.confidence >= config.strategy.min_confidence_score
        )

    @property
    def is_strong(self) -> bool:
        return self.confidence >= 85

    @property
    def is_moderate(self) -> bool:
        return 70 <= self.confidence < 85


class SignalEngine:
    """
    Multi-factor AI signal scoring.

    Weights (must sum to 1.0):
      trend       0.25   Direction of overall trend (MA alignment)
      momentum    0.20   RSI + MACD momentum
      pattern     0.20   Candlestick pattern recognition
      bb_position 0.15   Bollinger Band position
      sr_zone     0.10   Support / Resistance proximity
      volatility  0.10   Volatility filter (too high = lower score)
    """

    WEIGHTS = {
        "trend":       0.25,
        "momentum":    0.20,
        "pattern":     0.20,
        "bb_position": 0.15,
        "sr_zone":     0.10,
        "volatility":  0.10,
    }

    def analyze(self, indicators: AllIndicators,
                ml_probability: Optional[float] = None) -> TradeSignal:
        """
        Produce a trade signal from indicators + optional ML probability.

        Args:
            indicators:     Output from core.indicators.compute_all()
            ml_probability: ML model output in range 0–1 (None = not available)
        """
        reasoning = []
        sub_scores = {}
        direction_votes = {"RISE": 0.0, "FALL": 0.0}

        # ----------------------------------------------------------------
        # 1. TREND SCORE  (MA alignment)
        # ----------------------------------------------------------------
        trend_score, trend_dir = self._score_trend(indicators, reasoning)
        sub_scores["trend"] = round(trend_score * 100)
        if trend_dir:
            direction_votes[trend_dir] += trend_score * self.WEIGHTS["trend"] * 2

        # ----------------------------------------------------------------
        # 2. MOMENTUM SCORE  (RSI + MACD)
        # ----------------------------------------------------------------
        mom_score, mom_dir = self._score_momentum(indicators, reasoning)
        sub_scores["momentum"] = round(mom_score * 100)
        if mom_dir:
            direction_votes[mom_dir] += mom_score * self.WEIGHTS["momentum"] * 2

        # ----------------------------------------------------------------
        # 3. PATTERN SCORE  (Candlestick recognition)
        # ----------------------------------------------------------------
        pat_score, pat_dir = self._score_patterns(indicators, reasoning)
        sub_scores["pattern"] = round(pat_score * 100)
        if pat_dir:
            direction_votes[pat_dir] += pat_score * self.WEIGHTS["pattern"] * 2

        # ----------------------------------------------------------------
        # 4. BOLLINGER BAND SCORE
        # ----------------------------------------------------------------
        bb_score, bb_dir = self._score_bollinger(indicators, reasoning)
        sub_scores["bb_position"] = round(bb_score * 100)
        if bb_dir:
            direction_votes[bb_dir] += bb_score * self.WEIGHTS["bb_position"] * 2

        # ----------------------------------------------------------------
        # 5. SUPPORT / RESISTANCE SCORE
        # ----------------------------------------------------------------
        sr_score, sr_dir = self._score_sr(indicators, reasoning)
        sub_scores["sr_zone"] = round(sr_score * 100)
        if sr_dir:
            direction_votes[sr_dir] += sr_score * self.WEIGHTS["sr_zone"] * 2

        # ----------------------------------------------------------------
        # 6. VOLATILITY FILTER  (penalty for too much / too little noise)
        # ----------------------------------------------------------------
        vol_score, vol_ok = self._score_volatility(indicators, reasoning)
        sub_scores["volatility"] = round(vol_score * 100)

        # ----------------------------------------------------------------
        # CONSENSUS DIRECTION
        # ----------------------------------------------------------------
        total_vote = direction_votes["RISE"] + direction_votes["FALL"]
        if total_vote < 0.01:
            direction = "HOLD"
            raw_confidence = 0.0
        else:
            rise_pct = direction_votes["RISE"] / total_vote
            direction = "RISE" if rise_pct >= 0.55 else "FALL" if rise_pct <= 0.45 else "HOLD"
            agreement = abs(rise_pct - 0.5) * 2    # 0=tossup, 1=full agreement
            raw_confidence = agreement

        # ----------------------------------------------------------------
        # BLEND COMPONENT SCORES INTO SINGLE CONFIDENCE
        # ----------------------------------------------------------------
        component_confidence = sum(
            sub_scores[k] / 100 * w
            for k, w in self.WEIGHTS.items()
            if k in sub_scores
        )
        confidence = (raw_confidence * 0.5 + component_confidence * 0.5) * 100

        # Apply volatility penalty
        confidence *= vol_score

        # Boost if ML model agrees
        if ml_probability is not None:
            ml_agree = (
                (direction == "RISE" and ml_probability > 0.5) or
                (direction == "FALL" and ml_probability < 0.5)
            )
            if ml_agree:
                ml_boost = abs(ml_probability - 0.5) * 20  # max +10 pts
                confidence = min(100, confidence + ml_boost)
                reasoning.append(f"✓ ML model confirms signal (prob={ml_probability:.2f})")
            else:
                confidence *= 0.85  # Penalty when ML disagrees
                reasoning.append(f"⚠ ML model disagrees (prob={ml_probability:.2f})")
            sub_scores["ml"] = round(ml_probability * 100)

        # Hard-hold conditions
        if not vol_ok:
            direction = "HOLD"
            reasoning.append("✗ HOLD: Extreme volatility — skipping trade")

        if direction == "HOLD":
            confidence = 0.0
            reasoning.append("✗ No clear directional consensus — holding")

        # ----------------------------------------------------------------
        # STAKE MODIFIER  (size position based on confidence)
        # ----------------------------------------------------------------
        if confidence >= 90:
            stake_modifier = 1.5
        elif confidence >= 80:
            stake_modifier = 1.2
        elif confidence >= 70:
            stake_modifier = 1.0
        elif confidence >= 60:
            stake_modifier = 0.8
        else:
            stake_modifier = 0.5

        # ----------------------------------------------------------------
        # DURATION RECOMMENDATION
        # ----------------------------------------------------------------
        duration = self._recommend_duration(indicators, confidence)

        signal = TradeSignal(
            direction=direction,
            confidence=round(confidence, 1),
            stake_modifier=stake_modifier,
            duration_minutes=duration,
            reasoning=reasoning,
            sub_scores=sub_scores,
        )

        logger.debug(f"Signal: {direction} | Confidence: {confidence:.1f} | "
                     f"Scores: {sub_scores}")
        return signal

    # ----------------------------------------------------------------
    # COMPONENT SCORERS
    # ----------------------------------------------------------------

    def _score_trend(self, ind: AllIndicators,
                     log: list) -> Tuple[float, Optional[str]]:
        """Score based on MA trend alignment."""
        trend = ind.ma.trend
        score_map = {
            "STRONG_UP":   (0.9, "RISE"),
            "UP":          (0.7, "RISE"),
            "SIDEWAYS":    (0.2, None),
            "DOWN":        (0.7, "FALL"),
            "STRONG_DOWN": (0.9, "FALL"),
        }
        score, direction = score_map.get(trend, (0.3, None))

        if ind.ma.golden_cross:
            score = min(1.0, score + 0.1)
            log.append("✓ Golden cross: EMA10 > EMA20 > SMA50")
        elif ind.ma.death_cross:
            score = min(1.0, score + 0.1)
            log.append("✓ Death cross confirmed: bearish alignment")

        log.append(f"  Trend: {trend} ({direction or 'HOLD'})")
        return score, direction

    def _score_momentum(self, ind: AllIndicators,
                        log: list) -> Tuple[float, Optional[str]]:
        """Score RSI and MACD momentum."""
        rsi = ind.rsi
        macd = ind.macd
        scores = []
        directions = []

        # RSI contribution
        if rsi.signal == "OVERSOLD":
            scores.append(0.85)
            directions.append("RISE")
            log.append(f"✓ RSI oversold ({rsi.value:.1f}) — potential reversal up")
        elif rsi.signal == "OVERBOUGHT":
            scores.append(0.85)
            directions.append("FALL")
            log.append(f"✓ RSI overbought ({rsi.value:.1f}) — potential reversal down")
        else:
            # Neutral RSI — mild directional bias based on value
            if rsi.value > 55:
                scores.append(0.6)
                directions.append("RISE")
            elif rsi.value < 45:
                scores.append(0.6)
                directions.append("FALL")
            else:
                scores.append(0.3)
            log.append(f"  RSI neutral ({rsi.value:.1f})")

        # MACD contribution
        if macd.direction == "BULLISH":
            scores.append(0.8 if macd.crossover else 0.65)
            directions.append("RISE")
            log.append(f"✓ MACD bullish {'(crossover)' if macd.crossover else ''}")
        elif macd.direction == "BEARISH":
            scores.append(0.8 if macd.crossover else 0.65)
            directions.append("FALL")
            log.append(f"✓ MACD bearish {'(crossover)' if macd.crossover else ''}")
        else:
            scores.append(0.35)
            log.append("  MACD neutral")

        # Momentum rate-of-change
        mom = ind.momentum
        if abs(mom) > 0.3:
            directions.append("RISE" if mom > 0 else "FALL")
            scores.append(min(0.9, 0.5 + abs(mom)))
            log.append(f"✓ Strong momentum: {mom:+.3f}")

        if not scores:
            return 0.3, None

        avg_score = sum(scores) / len(scores)
        final_dir = max(set(directions), key=directions.count) if directions else None
        return avg_score, final_dir

    def _score_patterns(self, ind: AllIndicators,
                        log: list) -> Tuple[float, Optional[str]]:
        """Score detected candlestick patterns."""
        if not ind.patterns:
            return 0.3, None

        bull_strength = sum(p.strength for p in ind.patterns if p.direction == "BULLISH")
        bear_strength = sum(p.strength for p in ind.patterns if p.direction == "BEARISH")

        for p in ind.patterns:
            log.append(f"✓ Pattern: {p.name} ({p.direction}, strength={p.strength:.2f})")

        if bull_strength > bear_strength:
            score = min(0.95, 0.5 + bull_strength * 0.3)
            return score, "RISE"
        elif bear_strength > bull_strength:
            score = min(0.95, 0.5 + bear_strength * 0.3)
            return score, "FALL"
        else:
            return 0.4, None  # Conflicting patterns

    def _score_bollinger(self, ind: AllIndicators,
                         log: list) -> Tuple[float, Optional[str]]:
        """Score based on BB position and squeeze/breakout."""
        bb = ind.bb

        if bb.squeeze:
            log.append("⚠ BB squeeze — breakout approaching")
            return 0.6, None  # Breakout expected but direction unclear

        if bb.signal == "NEAR_LOWER":
            log.append(f"✓ Price at lower BB — potential bounce (B%={bb.percent_b:.2f})")
            return 0.78, "RISE"
        elif bb.signal == "NEAR_UPPER":
            log.append(f"✓ Price at upper BB — potential rejection (B%={bb.percent_b:.2f})")
            return 0.78, "FALL"
        elif bb.signal == "BREAKOUT":
            # Breakout: follow the direction
            direction = "RISE" if bb.percent_b > 0.5 else "FALL"
            log.append(f"✓ BB breakout {'upward' if direction=='RISE' else 'downward'}")
            return 0.72, direction
        else:
            return 0.4, None

    def _score_sr(self, ind: AllIndicators,
                  log: list) -> Tuple[float, Optional[str]]:
        """Score based on proximity to key support/resistance levels."""
        sr = ind.sr

        if sr.at_support:
            log.append(f"✓ Price at support ({sr.nearest_support:.4f}) — {sr.distance_to_support_pct:.2f}% away")
            return 0.82, "RISE"
        elif sr.at_resistance:
            log.append(f"✓ Price at resistance ({sr.nearest_resistance:.4f}) — {sr.distance_to_resistance_pct:.2f}% away")
            return 0.82, "FALL"
        else:
            # Mid-range — lower confidence on S/R component
            dist = min(sr.distance_to_support_pct, sr.distance_to_resistance_pct)
            score = max(0.2, 0.5 - dist * 0.05)
            return score, None

    def _score_volatility(self, ind: AllIndicators,
                          log: list) -> Tuple[float, bool]:
        """
        Volatility gate.
        Returns (score_multiplier, is_tradeable).
        """
        atr = ind.atr

        if atr.volatility == "EXTREME":
            log.append("✗ EXTREME volatility — unfavorable conditions")
            return 0.3, False
        elif atr.volatility == "HIGH":
            log.append("⚠ High volatility — reducing stake")
            return 0.7, True
        elif atr.volatility == "LOW":
            log.append("⚠ Low volatility — reduced opportunity")
            return 0.75, True
        else:
            log.append(f"✓ Volatility normal (ATR%={atr.atr_pct:.2f}%)")
            return 1.0, True

    def _recommend_duration(self, ind: AllIndicators,
                            confidence: float) -> int:
        """Recommend trade duration based on volatility and confidence."""
        cfg = config.strategy
        base = cfg.default_duration_minutes

        if ind.atr.volatility == "HIGH":
            # Shorter duration in high volatility to limit exposure
            return max(cfg.min_duration_minutes, base - 2)
        elif ind.atr.volatility == "LOW":
            # Longer duration when market is slow
            return min(cfg.max_duration_minutes, base + 2)
        elif confidence >= 85:
            # Strong signal — standard duration
            return base
        else:
            # Weak signal — shorter to reduce risk
            return max(cfg.min_duration_minutes, base - 1)
