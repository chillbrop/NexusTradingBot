"""
NexusTrade — Machine Learning Signal Model
============================================
Trains a gradient-boosted classifier on historical trade indicator snapshots
to predict win probability for each new signal.

Features extracted from AllIndicators at trade entry time:
  RSI value, MACD histogram, BB %B, ATR%, momentum, EMA alignment,
  trend code, pattern scores, etc.

The model outputs a probability (0–1) that the trade will be a winner.
This is blended into the SignalEngine's confidence score.

Training is triggered automatically after every N closed trades.
"""

import json
import logging
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Feature names must match the keys used in TradeLogger.log_indicator_snapshot
FEATURE_NAMES = [
    "rsi",
    "macd_hist",
    "bb_pct_b",
    "bb_bandwidth",
    "ema10_vs_ema20",     # 1 if ema10 > ema20 else 0
    "ema20_vs_sma50",
    "atr_pct",
    "momentum",
    "bb_squeeze",         # 1/0
    "trend_strong_up",    # One-hot
    "trend_up",
    "trend_sideways",
    "trend_down",
    "trend_strong_down",
    "has_bull_pattern",
    "has_bear_pattern",
    "volume_increasing",
]

TREND_MAP = {
    "STRONG_UP":   [1, 0, 0, 0, 0],
    "UP":          [0, 1, 0, 0, 0],
    "SIDEWAYS":    [0, 0, 1, 0, 0],
    "DOWN":        [0, 0, 0, 1, 0],
    "STRONG_DOWN": [0, 0, 0, 0, 1],
}


def extract_features(raw: dict) -> Optional[np.ndarray]:
    """
    Convert a raw_json indicator snapshot (from DB) into a feature vector.
    Returns None if the snapshot is missing critical fields.
    """
    try:
        trend_vec = TREND_MAP.get(raw.get("trend", "SIDEWAYS"), [0, 0, 1, 0, 0])
        patterns = raw.get("patterns", [])

        features = [
            float(raw.get("rsi", 50)) / 100,                  # Normalize 0–1
            float(raw.get("macd_hist", 0)),
            float(raw.get("bb_pct_b", 0.5)),
            float(raw.get("bb_bandwidth", 0.02)),
            1.0 if float(raw.get("ema10", 0)) > float(raw.get("ema20", 0)) else 0.0,
            1.0 if float(raw.get("ema20", 0)) > float(raw.get("sma50", 0)) else 0.0,
            float(raw.get("atr_pct", 0.5)) / 5,               # Normalize ~0–1
            float(raw.get("momentum", 0)),
            1.0 if raw.get("bb_squeeze") else 0.0,
            *trend_vec,
            1.0 if any(p in patterns for p in ["HAMMER", "MORNING_STAR", "BULL_ENGULFING",
                                                 "THREE_WHITE_SOLDIERS", "MARUBOZU"]) else 0.0,
            1.0 if any(p in patterns for p in ["SHOOTING_STAR", "EVENING_STAR", "BEAR_ENGULFING",
                                                "THREE_BLACK_CROWS"]) else 0.0,
            1.0 if raw.get("volume_trend") == "INCREASING" else 0.0,
        ]
        return np.array(features, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Feature extraction failed: {e}")
        return None


class SignalModel:
    """
    Gradient-boosted binary classifier for trade outcome prediction.
    
    Uses scikit-learn GradientBoostingClassifier as the primary model,
    with a simple LogisticRegression fallback when data is sparse.
    
    The model is retrained periodically from the trade database.
    Between retrains, it uses the last saved model from disk.
    """

    def __init__(self, model_path: str = "models/signal_model.pkl",
                 min_samples: int = 200):
        self.model_path = model_path
        self.min_samples = min_samples
        self._model = None
        self._scaler = None
        self._is_trained = False
        self._last_trained = 0.0
        self._training_samples = 0
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # ----------------------------------------------------------------
    # INFERENCE
    # ----------------------------------------------------------------

    def predict_proba(self, raw_features: dict) -> Optional[float]:
        """
        Predict win probability for a trade signal.
        Returns float in [0, 1] or None if model not available.
        """
        if not self._is_trained:
            self._load_model()

        if self._model is None:
            return None

        features = extract_features(raw_features)
        if features is None:
            return None

        try:
            if self._scaler is not None:
                features = self._scaler.transform(features.reshape(1, -1))
            else:
                features = features.reshape(1, -1)

            proba = self._model.predict_proba(features)[0]
            return float(proba[1])   # P(win)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None

    def predict_from_indicators(self, indicators) -> Optional[float]:
        """Convenience wrapper — accepts AllIndicators object directly."""
        if indicators is None:
            return None
        raw = {
            "rsi": indicators.rsi.value,
            "macd_hist": indicators.macd.histogram,
            "bb_pct_b": indicators.bb.percent_b,
            "bb_bandwidth": indicators.bb.bandwidth,
            "bb_squeeze": indicators.bb.squeeze,
            "ema10": indicators.ma.ema10,
            "ema20": indicators.ma.ema20,
            "sma50": indicators.ma.sma50,
            "atr_pct": indicators.atr.atr_pct,
            "momentum": indicators.momentum,
            "trend": indicators.ma.trend,
            "patterns": [p.name for p in (indicators.patterns or [])],
            "volume_trend": indicators.volume_trend,
        }
        return self.predict_proba(raw)

    # ----------------------------------------------------------------
    # TRAINING
    # ----------------------------------------------------------------

    def train(self, training_data: List[dict]) -> dict:
        """
        Train the model on historical trade snapshots.
        
        Args:
            training_data: List of {"features": raw_dict, "label": 0/1}
        
        Returns:
            Training metrics dict.
        """
        if len(training_data) < self.min_samples:
            logger.warning(f"Not enough training data: {len(training_data)} < {self.min_samples}")
            return {"error": "insufficient_data", "samples": len(training_data)}

        logger.info(f"Training ML model on {len(training_data)} samples...")

        # Build feature matrix
        X, y = [], []
        for item in training_data:
            vec = extract_features(item["features"])
            if vec is not None:
                X.append(vec)
                y.append(int(item["label"]))

        if len(X) < self.min_samples:
            return {"error": "feature_extraction_failed", "samples": len(X)}

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import accuracy_score, roc_auc_score

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train GBM
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42,
            )
            model.fit(X_scaled, y)

            # Cross-validated accuracy
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="roc_auc")
            auc = float(np.mean(cv_scores))

            train_preds = model.predict(X_scaled)
            accuracy = float(accuracy_score(y, train_preds))

            self._model = model
            self._scaler = scaler
            self._is_trained = True
            self._last_trained = time.time()
            self._training_samples = len(X)

            self._save_model()

            metrics = {
                "samples": len(X),
                "accuracy": round(accuracy, 4),
                "roc_auc": round(auc, 4),
                "win_rate_base": round(sum(y) / len(y), 4),
                "feature_importances": dict(zip(
                    FEATURE_NAMES[:len(model.feature_importances_)],
                    [round(float(f), 4) for f in model.feature_importances_]
                )),
            }
            logger.info(f"Model trained: accuracy={accuracy:.3f} AUC={auc:.3f} "
                        f"samples={len(X)}")
            return metrics

        except ImportError:
            logger.warning("scikit-learn not installed — using logistic regression fallback")
            return self._train_fallback(X, y)
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return {"error": str(e)}

    def _train_fallback(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Pure-numpy logistic regression fallback.
        Used when scikit-learn is not installed.
        """
        # Normalize
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X_norm = (X - mean) / std

        # Logistic regression via gradient descent
        n_features = X_norm.shape[1]
        W = np.zeros(n_features)
        b = 0.0
        lr = 0.01

        for _ in range(1000):
            logits = X_norm @ W + b
            proba = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))
            error = proba - y
            W -= lr * (X_norm.T @ error) / len(y)
            b -= lr * error.mean()

        self._model = {"W": W, "b": b, "mean": mean, "std": std}
        self._is_trained = True

        # Accuracy
        preds = ((X_norm @ W + b) > 0).astype(int)
        accuracy = (preds == y).mean()
        logger.info(f"Fallback model trained: accuracy={accuracy:.3f}")
        return {"samples": len(X), "accuracy": round(float(accuracy), 4), "mode": "fallback"}

    # ----------------------------------------------------------------
    # PERSISTENCE
    # ----------------------------------------------------------------

    def _save_model(self):
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump({"model": self._model, "scaler": self._scaler}, f)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.info("No saved model found — will use rule-based scoring only")
            return
        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            self._model = data.get("model")
            self._scaler = data.get("scaler")
            self._is_trained = True
            logger.info(f"Loaded saved model from {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")

    @property
    def info(self) -> dict:
        return {
            "trained": self._is_trained,
            "samples": self._training_samples,
            "last_trained": self._last_trained,
            "model_path": self.model_path,
        }
