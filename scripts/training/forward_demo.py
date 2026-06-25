#!/usr/bin/env python3
"""Offline simulation of one validator->miner->scoring cycle.

Mimics what the live subnet does without needing a wallet, registration, or
bittensor:

  1. Take real benchmark hands and project them exactly as the validator does
     (``build_miner_payload_hand``) into a DetectionSynapse-shaped request.
  2. Run the miner's scoring path (the same call ``Miner.forward`` makes).
  3. Score the responses with the validator's reward function.

Useful as a local smoke test before registering on netuid 126.
"""
from __future__ import annotations

import argparse
import glob
import json
import random

import numpy as np

from poker44.miner.model import BotDetectorModel, heuristic_chunk_score
from poker44.score.scoring import reward
from poker44.validator.payload_view import build_miner_payload_hand


def load_request(cache_dir: str, n_chunks: int, seed: int):
    items = []
    for path in sorted(glob.glob(cache_dir + "/*.json")):
        rec = json.load(open(path))
        for g in rec["chunk_groups"]:
            for batch, label in zip(g["chunks"], g["groundTruth"]):
                if batch:
                    items.append((batch, int(label)))
    random.Random(seed).shuffle(items)
    items = items[:n_chunks]
    # validator-side projection -> these are the chunks the miner receives
    chunks = [[build_miner_payload_hand(h) for h in batch if isinstance(h, dict)]
              for batch, _ in items]
    labels = np.asarray([lab for _, lab in items])
    return chunks, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/benchmark")
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force-heuristic", action="store_true")
    args = ap.parse_args()

    chunks, labels = load_request(args.cache, args.chunks, args.seed)
    print(f"Validator sent DetectionSynapse(chunks=[{len(chunks)}]) "
          f"({int((labels==1).sum())} bot / {int((labels==0).sum())} human)\n")

    model = None if args.force_heuristic else BotDetectorModel.load()
    if model is not None:
        scores = model.score_chunks(chunks)
        mode = "trained model"
    else:
        scores = [heuristic_chunk_score(c) for c in chunks]
        mode = "heuristic fallback"

    preds = [s >= 0.5 for s in scores]
    print(f"Miner responded with risk_scores ({mode}):")
    print(f"{'idx':>3} {'truth':>6} {'score':>7} {'pred':>5} {'hit':>4}")
    for i, (s, p, y) in enumerate(zip(scores, preds, labels)):
        truth = "bot" if y == 1 else "human"
        hit = "OK" if int(p) == int(y) else ""
        print(f"{i:>3} {truth:>6} {s:>7.3f} {str(bool(p)):>5} {hit:>4}")

    rew, res = reward(np.asarray(scores), labels)
    print(f"\nValidator reward for this batch: {rew:.3f}")
    print(f"  fpr={res['fpr']:.3f} (humans flagged as bots)  "
          f"bot_recall={res['bot_recall']:.3f}  ap={res['ap_score']:.3f}")
    assert len(scores) == len(chunks), "contract: one score per chunk"
    assert all(0.0 <= s <= 1.0 for s in scores), "contract: scores in [0,1]"
    print("\nContract checks passed: one score per chunk, all in [0, 1].")


if __name__ == "__main__":
    main()
