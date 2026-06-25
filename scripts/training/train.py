#!/usr/bin/env python3
"""Train the Poker44 bot-detection model from cached benchmark data.

Pipeline:
  1. Build projected chunk features (zero train/serve skew).
  2. Grouped (by release date) out-of-fold predictions for an honest estimate
     of average precision / AUC and, crucially, of the human false-positive
     rate at any decision threshold.
  3. Place the deployed decision boundary: the validator scores at a fixed 0.5
     threshold with a hard human-safety kill at FPR >= 10%. We pick the raw
     probability threshold t* that maximises the validator reward subject to a
     conservative FPR cap, derived from OOF (honest) predictions, then remap so
     t* -> 0.5.
  4. Refit the ensemble + feature selector on ALL data and save the artifact.

Usage:
    python scripts/training/train.py --cache data/benchmark \
        --out poker44/miner/artifacts/bot_detector.joblib
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import joblib
import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from poker44.miner.model import ARTIFACT_VERSION, remap_score
from poker44.score.scoring import reward
from scripts.training.dataset import build_dataset

SELECT_K = 30
N_SPLITS = 6
# Target human false-positive rate for threshold placement. The validator hard-
# kills reward at FPR >= 0.10; a conservative 0.02 target on out-of-fold human
# scores transfers to unseen future dates with a wide safety margin (verified by
# walk-forward temporal tests).
TARGET_FPR = 0.02


def make_estimators() -> List:
    return [
        HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300,
            l2_regularization=1.0, random_state=0,
        ),
        RandomForestClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=8, random_state=0,
        ),
        GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=0,
        ),
    ]


def _ensemble_oof(ds, k: int) -> np.ndarray:
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(ds.y))
    for tr, te in gkf.split(ds.X, ds.y, groups=ds.dates):
        sel = SelectKBest(f_classif, k=min(k, ds.X.shape[1])).fit(ds.X[tr], ds.y[tr])
        Xtr, Xte = sel.transform(ds.X[tr]), sel.transform(ds.X[te])
        preds = []
        for est in make_estimators():
            est.fit(Xtr, ds.y[tr])
            preds.append(est.predict_proba(Xte)[:, 1])
        oof[te] = np.mean(preds, axis=0)
    return oof


def _choose_threshold(oof: np.ndarray, y: np.ndarray, target_fpr: float = TARGET_FPR) -> float:
    """Decision threshold = the OOF human-score quantile at ``target_fpr``.

    Placing the boundary by the human-score distribution (rather than by
    maximising in-sample reward) is what transfers across time: only a
    ``target_fpr`` fraction of *human* chunks sit above it, so the deployed 0.5
    cut keeps the false-positive rate well under the validator's 0.10 hard kill
    even on unseen future dates.
    """
    human = oof[y == 0]
    if human.size == 0:
        return float(np.quantile(oof, 0.95))
    t = float(np.quantile(human, 1.0 - target_fpr))
    return float(min(max(t, 1e-3), 1.0 - 1e-3))


def main() -> None:
    ap_arg = argparse.ArgumentParser()
    ap_arg.add_argument("--cache", default="data/benchmark")
    ap_arg.add_argument("--out", default="poker44/miner/artifacts/bot_detector.joblib")
    ap_arg.add_argument("--k", type=int, default=SELECT_K)
    ap_arg.add_argument("--target-fpr", type=float, default=TARGET_FPR)
    args = ap_arg.parse_args()

    ds = build_dataset(args.cache, project=True)
    print(f"Dataset: X={ds.X.shape}  bots={int(ds.y.sum())} humans={int((ds.y==0).sum())} "
          f"dates={len(set(ds.dates))}")

    # 1) Honest out-of-fold evaluation
    oof = _ensemble_oof(ds, args.k)
    ap = float(average_precision_score(ds.y, oof))
    auc = float(roc_auc_score(ds.y, oof))
    t_star = _choose_threshold(oof, ds.y, args.target_fpr)
    mapped = remap_score(oof, t_star)
    rew, res = reward(mapped, ds.y)
    print(f"\nOOF (grouped by date):  AP={ap:.3f}  AUC={auc:.3f}")
    print(f"Threshold t*={t_star:.4f} -> deployed reward={rew:.3f} "
          f"(fpr={res['fpr']:.3f} recall={res['bot_recall']:.3f})")

    # 2) Refit selector + ensemble on ALL data for deployment
    selector = SelectKBest(f_classif, k=min(args.k, ds.X.shape[1])).fit(ds.X, ds.y)
    Xsel = selector.transform(ds.X)
    estimators = make_estimators()
    for est in estimators:
        est.fit(Xsel, ds.y)
    selected_idx = np.where(selector.get_support())[0].astype(int)
    selected_names = [ds.feature_names[i] for i in selected_idx]

    artifact = {
        "version": ARTIFACT_VERSION,
        "feature_names": list(ds.feature_names),
        "selected_idx": selected_idx,
        "estimators": estimators,
        "threshold": float(t_star),
        "meta": {
            "select_k": int(args.k),
            "target_fpr": float(args.target_fpr),
            "oof_ap": ap,
            "oof_auc": auc,
            "oof_reward": float(rew),
            "oof_fpr": float(res["fpr"]),
            "oof_recall": float(res["bot_recall"]),
            "n_samples": int(len(ds.y)),
            "n_dates": int(len(set(ds.dates))),
            "selected_features": selected_names,
            "sklearn_estimators": [type(e).__name__ for e in estimators],
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(artifact, args.out)
    size_kb = os.path.getsize(args.out) / 1024.0
    print(f"\nSaved artifact -> {args.out} ({size_kb:.1f} KB)")
    print("Selected features:", ", ".join(selected_names))

    # also drop a human-readable sidecar
    with open(args.out + ".meta.json", "w") as fh:
        json.dump(artifact["meta"], fh, indent=2)


if __name__ == "__main__":
    main()
