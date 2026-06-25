"""Inference wrapper for the Poker44 bot-detection model.

Loads the trained artifact (feature selector + estimator ensemble + calibrated
decision threshold) and turns a miner-visible chunk into a risk score in
``[0, 1]``. The deployed score is remapped so that the validator's fixed 0.5
decision boundary sits at the human-safe operating point chosen at train time.

If the artifact or scientific stack is unavailable, callers should fall back to
the conservative heuristic in :mod:`neurons.miner`.
"""
from __future__ import annotations

import os
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

from poker44.miner.features import FEATURE_NAMES, extract_chunk_features

ARTIFACT_VERSION = 2

_DEFAULT_ARTIFACT = os.path.join(
    os.path.dirname(__file__), "artifacts", "bot_detector.joblib"
)


def remap_score(p: Any, threshold: float) -> np.ndarray:
    """Monotonic piecewise-linear map sending ``threshold`` -> 0.5.

    Preserves ranking (so average precision is unchanged) while placing the
    validator's 0.5 cut at the trained low-FPR operating point.
    """
    arr = np.asarray(p, dtype=float)
    t = float(min(max(threshold, 1e-6), 1.0 - 1e-6))
    out = np.where(
        arr <= t,
        0.5 * (arr / t),
        0.5 + 0.5 * ((arr - t) / (1.0 - t)),
    )
    return np.clip(out, 0.0, 1.0)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def heuristic_chunk_score(chunk: Sequence[Mapping[str, Any]]) -> float:
    """Conservative fallback score used when the trained model is unavailable.

    Uses miner-visible behavioral signals (lower action entropy / lower
    aggression / more regular play / shorter relative stack lean bot, per the
    benchmark audit) but is clamped below the validator's 0.5 decision boundary
    so it never false-positives a human while degraded.
    """
    if not chunk:
        return 0.1
    f = extract_chunk_features(chunk)
    low_entropy = _clamp(1.0 - f["hero_action_entropy"], 0.0, 1.0)
    low_aggr = _clamp(1.0 - f["hero_aggression_rate"] / 0.5, 0.0, 1.0)
    regularity = _clamp(1.0 - f["hero_aggr_ratio_cv"], 0.0, 1.0)
    short_stack = _clamp(1.0 - f["hero_stack_rel_mean"], 0.0, 1.0)
    raw = 0.35 * low_entropy + 0.30 * low_aggr + 0.20 * regularity + 0.15 * short_stack
    return round(_clamp(raw, 0.05, 0.45) * 0.9, 6)


class BotDetectorModel:
    def __init__(self, artifact: Mapping[str, Any]):
        self.feature_names: List[str] = list(artifact["feature_names"])
        self.selected_idx: np.ndarray = np.asarray(artifact["selected_idx"], dtype=int)
        self.estimators: List[Any] = list(artifact["estimators"])
        self.threshold: float = float(artifact["threshold"])
        self.meta: Mapping[str, Any] = dict(artifact.get("meta", {}))
        self.version = int(artifact.get("version", 0))
        if self.feature_names != list(FEATURE_NAMES):
            # Feature code and artifact disagree: refuse to load rather than
            # silently extract a mismatched vector.
            raise ValueError(
                "Artifact feature_names do not match poker44.miner.features.FEATURE_NAMES; "
                "retrain the model after changing features."
            )

    @classmethod
    def load(cls, path: Optional[str] = None) -> Optional["BotDetectorModel"]:
        target = path or _DEFAULT_ARTIFACT
        if not os.path.exists(target):
            return None
        try:
            import joblib

            artifact = joblib.load(target)
            return cls(artifact)
        except Exception:
            return None

    def _matrix(self, chunks: Sequence[Sequence[Mapping[str, Any]]]) -> np.ndarray:
        rows = []
        for chunk in chunks:
            feats = extract_chunk_features(chunk)
            rows.append([feats[n] for n in self.feature_names])
        X = np.asarray(rows, dtype=float) if rows else np.zeros((0, len(self.feature_names)))
        return X[:, self.selected_idx] if X.size else X

    def predict_proba(self, chunks: Sequence[Sequence[Mapping[str, Any]]]) -> np.ndarray:
        """Raw ensemble bot-probability per chunk (pre-remap)."""
        X = self._matrix(chunks)
        if X.shape[0] == 0:
            return np.zeros(0, dtype=float)
        preds = [est.predict_proba(X)[:, 1] for est in self.estimators]
        return np.mean(preds, axis=0)

    # Degenerate chunks (no hands / no usable actions) carry no behavioral
    # signal; default to a conservative human-leaning score to protect humans.
    SAFE_EMPTY_SCORE = 0.1

    def score_chunks(self, chunks: Sequence[Sequence[Mapping[str, Any]]]) -> List[float]:
        """Deployed, human-safety-calibrated risk score per chunk."""
        if not chunks:
            return []
        usable = [bool(chunk) and any(isinstance(h, Mapping) for h in chunk) for chunk in chunks]
        proba = self.predict_proba(chunks)
        mapped = remap_score(proba, self.threshold)
        return [
            round(float(v), 6) if ok else self.SAFE_EMPTY_SCORE
            for v, ok in zip(mapped, usable)
        ]

    def score_chunk(self, chunk: Sequence[Mapping[str, Any]]) -> float:
        scores = self.score_chunks([chunk])
        return scores[0] if scores else 0.0
