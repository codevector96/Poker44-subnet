"""Build a feature matrix from cached Poker44 benchmark data.

Each cached date file holds chunk-groups; each group has parallel
``chunks`` (list of batches, a batch is a list of raw hands) and
``groundTruth`` (1=bot, 0=human). We project every hand through the exact
production payload view, extract chunk features, and emit (X, y, dates, splits).
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from poker44.miner.features import FEATURE_NAMES, extract_chunk_features
from poker44.validator.payload_view import build_miner_payload_hand


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray          # release date per row (group key)
    splits: np.ndarray         # 'train' / 'validation' per row, when provided
    feature_names: List[str]


def _project_chunk(batch) -> list:
    out = []
    for hand in batch:
        if isinstance(hand, dict):
            try:
                out.append(build_miner_payload_hand(hand))
            except Exception:
                out.append(hand)
    return out


def build_dataset(cache_dir: str = "data/benchmark", project: bool = True) -> Dataset:
    rows_X: List[np.ndarray] = []
    rows_y: List[int] = []
    rows_date: List[str] = []
    rows_split: List[str] = []

    files = sorted(glob.glob(os.path.join(cache_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No cached benchmark files in {cache_dir}/")

    for path in files:
        with open(path) as fh:
            record = json.load(fh)
        date = str(record.get("sourceDate") or os.path.basename(path)[:-5])
        for group in record.get("chunk_groups", []):
            batches = group.get("chunks") or []
            labels = group.get("groundTruth") or []
            split = str(group.get("split") or "")
            for batch, label in zip(batches, labels):
                if not isinstance(batch, list) or not batch:
                    continue
                chunk = _project_chunk(batch) if project else batch
                feats = extract_chunk_features(chunk)
                rows_X.append(np.asarray([feats[n] for n in FEATURE_NAMES], dtype=float))
                rows_y.append(int(label))
                rows_date.append(date)
                rows_split.append(split)

    return Dataset(
        X=np.vstack(rows_X),
        y=np.asarray(rows_y, dtype=int),
        dates=np.asarray(rows_date),
        splits=np.asarray(rows_split),
        feature_names=list(FEATURE_NAMES),
    )
