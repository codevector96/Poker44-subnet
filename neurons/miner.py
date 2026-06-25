"""Poker44 miner: trained hero-behavior bot detector with a safe heuristic fallback."""

# from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.miner.model import BotDetectorModel, heuristic_chunk_score
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT_PATH = _REPO_ROOT / "poker44" / "miner" / "artifacts" / "bot_detector.joblib"
_FEATURES_PATH = _REPO_ROOT / "poker44" / "miner" / "features.py"
_MODEL_PATH = _REPO_ROOT / "poker44" / "miner" / "model.py"


class Miner(BaseMinerNeuron):
    """Poker44 miner.

    Scores each chunk (a list of miner-visible hands describing one classified
    "hero" player) with a trained gradient-boosted ensemble over hero-centric
    behavioral features. Scores are calibrated so the validator's fixed 0.5
    decision boundary sits at a low false-positive (human-safe) operating point.

    If the trained artifact cannot be loaded, the miner degrades to a
    conservative heuristic that never crosses the 0.5 bot threshold, protecting
    humans at the cost of bot recall until the model is restored.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🂡 Poker44 hero-behavior miner starting")

        self.model = BotDetectorModel.load(str(_ARTIFACT_PATH))
        if self.model is not None:
            m = self.model.meta
            bt.logging.info(
                "Loaded trained bot detector | "
                f"features={len(self.model.selected_idx)} threshold={self.model.threshold:.4f} "
                f"oof_ap={m.get('oof_ap'):.3f} oof_auc={m.get('oof_auc'):.3f} "
                f"oof_reward={m.get('oof_reward'):.3f} oof_fpr={m.get('oof_fpr'):.3f}"
            )
        else:
            bt.logging.warning(
                f"Trained artifact not found/loadable at {_ARTIFACT_PATH}; "
                "falling back to the conservative heuristic. "
                "Run scripts/training/train.py to (re)build the model."
            )

        impl_files = [Path(__file__).resolve(), _FEATURES_PATH, _MODEL_PATH]
        if _ARTIFACT_PATH.exists():
            impl_files.append(_ARTIFACT_PATH)
        self.model_manifest = build_local_model_manifest(
            repo_root=_REPO_ROOT,
            implementation_files=impl_files,
            defaults={
                "model_name": "poker44-hero-behavior-gbm",
                "model_version": "2",
                "framework": "scikit-learn",
                "license": "MIT",
                # Override with your own published repo before competing:
                #   POKER44_MODEL_REPO_URL / POKER44_MODEL_REPO_COMMIT
                "repo_url": "https://github.com/Poker44/Poker44-subnet",
                "notes": (
                    "Gradient-boosted ensemble over hero-centric behavioral features "
                    "extracted from miner-visible chunks; calibrated for human safety."
                ),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained on the public Poker44 training benchmark "
                    "(api.poker44.net/api/v1/benchmark), grouped-by-date cross-validated. "
                    "No validator-only evaluation data is used."
                ),
                "training_data_sources": ["poker44-public-training-benchmark"],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data; "
                    "only the public benchmark and runtime chunk features are used."
                ),
                "data_attestation": (
                    "Features are derived solely from miner-visible hand/action fields."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup()
        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self) -> None:
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']} "
            f"policy_violations={self.manifest_compliance['policy_violations']})"
        )
        bt.logging.info(
            f"Manifest | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"framework={self.model_manifest.get('framework', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '') or '<unset>'} "
            f"digest={self.manifest_digest}"
        )
        if self.manifest_compliance["status"] != "transparent":
            bt.logging.warning(
                "Manifest is not 'transparent'. Before competing, publish your model repo "
                "and set POKER44_MODEL_REPO_URL and POKER44_MODEL_REPO_COMMIT "
                "(a real git commit) so high-scoring submissions pass review."
            )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one bot-risk score per chunk."""
        chunks = synapse.chunks or []
        if self.model is not None:
            scores = self.model.score_chunks(chunks)
        else:
            scores = [heuristic_chunk_score(chunk) for chunk in chunks]

        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        n_bot = sum(1 for s in scores if s >= 0.5)
        bt.logging.info(
            f"Scored {len(chunks)} chunks | flagged_bot={n_bot} "
            f"score[min/mean/max]="
            f"{(min(scores) if scores else 0):.3f}/"
            f"{(sum(scores)/len(scores) if scores else 0):.3f}/"
            f"{(max(scores) if scores else 0):.3f}"
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 miner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
