#!/usr/bin/env python3
"""Held-out evaluation of the Poker44 bot detector.

Two views:
  * Temporal hold-out: train on older release dates, test on the newest ones.
    This is the closest offline proxy for production (future, unseen data).
  * Validator-window simulation: replay scored chunks through the validator's
    exact rolling-window reward to estimate the on-chain reward.

Also checks the conservative heuristic fallback never false-positives humans.
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from poker44.miner.model import heuristic_chunk_score, remap_score
from poker44.score.scoring import reward
from scripts.training.dataset import build_dataset
from scripts.training.train import _choose_threshold, make_estimators


def temporal_eval(ds, holdout_dates: int, k: int, target_fpr: float):
    uniq = sorted(set(ds.dates))
    test_dates = set(uniq[-holdout_dates:])
    tr = np.array([d not in test_dates for d in ds.dates])
    te = ~tr
    # Honest threshold: grouped-CV out-of-fold human-score quantile on PAST data.
    oof = np.zeros(int(tr.sum()))
    Xtr_all, ytr_all, dtr_all = ds.X[tr], ds.y[tr], ds.dates[tr]
    for a, b in GroupKFold(n_splits=5).split(Xtr_all, ytr_all, groups=dtr_all):
        s = SelectKBest(f_classif, k=min(k, ds.X.shape[1])).fit(Xtr_all[a], ytr_all[a])
        oof[b] = np.mean(
            [e.fit(s.transform(Xtr_all[a]), ytr_all[a]).predict_proba(s.transform(Xtr_all[b]))[:, 1]
             for e in make_estimators()], axis=0)
    t = _choose_threshold(oof, ytr_all, target_fpr)
    # Final model on all past, evaluate on unseen future.
    sel = SelectKBest(f_classif, k=min(k, ds.X.shape[1])).fit(ds.X[tr], ds.y[tr])
    Xtr, Xte = sel.transform(ds.X[tr]), sel.transform(ds.X[te])
    p_te = np.mean([e.fit(Xtr, ds.y[tr]).predict_proba(Xte)[:, 1] for e in make_estimators()], axis=0)
    y_te = ds.y[te]
    ap = average_precision_score(y_te, p_te)
    auc = roc_auc_score(y_te, p_te)
    rew, res = reward(remap_score(p_te, t), y_te)
    return ap, auc, rew, res, len(test_dates), int(te.sum())


def window_sim(scores: np.ndarray, labels: np.ndarray, window: int, rng_seed: int = 0):
    """Replay shuffled chunks through the validator's rolling-window reward."""
    idx = np.arange(len(scores))
    rs = np.random.RandomState(rng_seed)
    rs.shuffle(idx)
    rewards = []
    for end in range(window, len(idx) + 1):
        w = idx[end - window:end]
        rew, _ = reward(scores[w], labels[w])
        rewards.append(rew)
    return float(np.mean(rewards)), float(np.min(rewards)), float(np.max(rewards))


def main() -> None:
    ap_arg = argparse.ArgumentParser()
    ap_arg.add_argument("--cache", default="data/benchmark")
    ap_arg.add_argument("--holdout-dates", type=int, default=6)
    ap_arg.add_argument("--k", type=int, default=30)
    ap_arg.add_argument("--target-fpr", type=float, default=0.02)
    ap_arg.add_argument("--window", type=int, default=40)
    args = ap_arg.parse_args()

    ds = build_dataset(args.cache, project=True)

    print("=== Temporal hold-out (train on older dates, test on newest) ===")
    ap, auc, rew, res, nd, nrows = temporal_eval(ds, args.holdout_dates, args.k, args.target_fpr)
    print(f"  test: {nd} newest dates, {nrows} chunks")
    print(f"  AP={ap:.3f}  AUC={auc:.3f}  reward={rew:.3f} (fpr={res['fpr']:.3f} recall={res['bot_recall']:.3f})")

    print("\n=== Validator rolling-window reward simulation (deployed model) ===")
    from poker44.miner.model import BotDetectorModel
    import glob, json
    from poker44.validator.payload_view import build_miner_payload_hand
    model = BotDetectorModel.load()
    chunks, ys = [], []
    for path in sorted(glob.glob(args.cache + "/*.json")):
        rec = json.load(open(path))
        for g in rec["chunk_groups"]:
            for batch, label in zip(g["chunks"], g["groundTruth"]):
                if not batch:
                    continue
                chunks.append([build_miner_payload_hand(h) for h in batch if isinstance(h, dict)])
                ys.append(int(label))
    scores = np.asarray(model.score_chunks(chunks))
    ys = np.asarray(ys)
    mean_r, min_r, max_r = window_sim(scores, ys, args.window)
    print(f"  window={args.window}: mean reward={mean_r:.3f} (min={min_r:.3f} max={max_r:.3f})")
    print(f"  NOTE: deployed model scored in-sample here (optimistic); temporal hold-out above is the honest number.")

    print("\n=== Heuristic fallback human-safety check ===")
    h = np.asarray([heuristic_chunk_score(c) for c in chunks])
    print(f"  fallback score range [{h.min():.3f}, {h.max():.3f}] | any>=0.5 (would false-positive)? {(h>=0.5).any()}")
    hrew, hres = reward(h, ys)
    print(f"  fallback reward={hrew:.3f} (fpr={hres['fpr']:.3f} recall={hres['bot_recall']:.3f}) — safe, low recall by design")


if __name__ == "__main__":
    main()
