# Poker44 Miner Model — `poker44-hero-behavior-gbm`

A trained bot detector that replaces the random-level reference heuristic. It
scores each validator chunk (≈30 miner-visible hands describing one classified
"hero" player) with one calibrated bot-risk score in `[0, 1]`.

## Why the reference miner scores ~0

The reference heuristic weights `outcome.showdown` and board/street depth, but
the validator **zeroes `outcome` and `board_cards`** before miners see them
(`poker44/validator/payload_view.py`). On projected data it is effectively
random (AUC ≈ 0.51) and catches **0%** of bots; its non-zero reward is the
`0.65·AP` floor the reward formula pays even at zero recall.

## What the miner actually sees

After `build_miner_payload_hand`, each hand exposes only: `metadata`
(`hero_seat`, `max_seats`, game/limit type), per-seat `starting_stack` (bb),
and a **sampled, bucket-coarsened** action list (`action_type`,
`normalized_amount_bb`, `pot_before/after`). No cards, no outcome, no timing.
The label applies to the **hero** across the chunk.

## Approach

* **Features** (`poker44/miner/features.py`): hero-centric, chunk-aggregated —
  action-mix rates, aggression rate/factor, action entropy, VPIP/PFR proxies,
  per-street aggression, bet-sizing shape, stack level + relative stack, and
  cross-hand regularity (bots play more consistently). These align with the
  benchmark audit's strongest signals (`mean_starting_stack`,
  `aggression_rate`, `action_entropy`).
* **No train/serve skew**: training hands are projected through the *exact*
  production `build_miner_payload_hand` once; at inference the validator has
  already projected, so hands are fed straight through.
* **Model**: an ensemble (HistGradientBoosting + RandomForest +
  GradientBoosting) over the top-30 `f_classif`-selected features.
* **Human-safety calibration**: the validator scores at a fixed 0.5 boundary
  and **hard-zeros reward at FPR ≥ 10%**. We place the boundary at the
  out-of-fold human-score quantile for a conservative **2% target FPR**, then
  remap monotonically so that point → 0.5 (preserving ranking / AP). This
  transfers to unseen future dates (temporal hold-out FPR ≈ 0).

## Measured performance (projected benchmark, 444 chunks / 31 dates)

| Miner | AUC | Catches bots | Reward |
|---|---|---|---|
| Reference heuristic | 0.51 | 0% | 0.335 (floor) |
| `poker44-hero-behavior-gbm` (grouped-CV) | 0.64 | yes | 0.50 |
| `poker44-hero-behavior-gbm` (temporal hold-out) | 0.69 | yes | 0.47 (FPR 0.00) |

The public benchmark is deliberately capped (combo-AP ≤ 0.76), so this is near
the achievable ceiling on obfuscated data.

## Reproduce

```bash
pip install -e .            # or: pip install -r requirements.txt
python scripts/training/download_benchmark.py --out data/benchmark
python scripts/training/train.py             # writes poker44/miner/artifacts/bot_detector.joblib
python scripts/training/evaluate.py          # temporal hold-out + window simulation
python -m unittest tests.test_miner_model    # regression tests (no bittensor needed)
```

Retrain regularly as new benchmark dates publish (daily ~00:05 UTC).

## Before you register / compete

The manifest is for transparency and does not change scoring, but high-scoring
miners are reviewed. To be `transparent` and pass review, publish your model
repo and set, before launching the miner:

```bash
export POKER44_MODEL_REPO_URL="https://github.com/<you>/<your-model-repo>"
export POKER44_MODEL_REPO_COMMIT="$(git rev-parse HEAD)"   # a real commit hash
```

Note: `repo_url` must point to **your** repo (not the reference repo) once
`model_name` is not the reference model, or compliance flags a policy violation.

## Versioning caveat

The artifact is a scikit-learn pickle; train and serve with the same
scikit-learn/numpy versions. If the artifact fails to load, the miner logs a
warning and falls back to a conservative heuristic that never crosses the 0.5
boundary (safe but low recall) — retrain in the deployment environment to
restore the model.
