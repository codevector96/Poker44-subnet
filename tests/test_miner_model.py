"""Regression tests for the trained miner model and feature pipeline.

These run without ``bittensor`` so they exercise the scoring path the miner's
``forward`` delegates to (model load -> feature extraction -> calibrated score).
"""
import unittest

from poker44.miner.features import (
    FEATURE_NAMES,
    chunk_to_vector,
    extract_chunk_features,
)
from poker44.miner.model import BotDetectorModel, heuristic_chunk_score, remap_score


def _hand(hero_seat=2, actions=None, stacks=None):
    stacks = stacks or [4.0, 4.0, 4.0]
    players = [
        {"player_uid": f"seat_{i+1}", "seat": i + 1, "starting_stack": s,
         "hole_cards": None, "showed_hand": False}
        for i, s in enumerate(stacks)
    ]
    return {
        "metadata": {"game_type": "Hold'em", "limit_type": "No Limit",
                     "max_seats": len(stacks), "hero_seat": hero_seat,
                     "sb": 0.01, "bb": 0.02, "ante": 0.0},
        "players": players,
        "streets": [{"street": "flop", "board_cards": []}],
        "actions": actions or [],
        "outcome": {"winners": [], "payouts": {}, "total_pot": 0.0,
                    "rake": 0.0, "result_reason": "", "showdown": False},
    }


def _action(seat, atype, street="preflop", amt=0.0, bb=0.0, pb=0.0, pa=0.0):
    return {"action_id": "1", "street": street, "actor_seat": seat,
            "action_type": atype, "amount": amt, "raise_to": None, "call_to": None,
            "normalized_amount_bb": bb, "pot_before": pb, "pot_after": pa}


class FeatureTests(unittest.TestCase):
    def test_vector_length_matches_feature_names(self):
        chunk = [_hand(actions=[_action(2, "raise", bb=22.0, amt=0.45, pb=0.3, pa=0.7)])]
        feats = extract_chunk_features(chunk)
        self.assertEqual(set(feats.keys()), set(FEATURE_NAMES))
        self.assertEqual(len(chunk_to_vector(chunk)), len(FEATURE_NAMES))

    def test_empty_chunk_is_all_zero_vector(self):
        feats = extract_chunk_features([])
        self.assertTrue(all(v == 0.0 for v in feats.values()))

    def test_hero_actions_resolved_against_hero_seat(self):
        # Hero (seat 2) folds; another seat raises. Hero fold-rate should be 1.
        chunk = [
            _hand(hero_seat=2, actions=[
                _action(1, "raise", bb=20.0, amt=0.4, pb=0.3, pa=0.7),
                _action(2, "fold"),
            ])
            for _ in range(5)
        ]
        feats = extract_chunk_features(chunk)
        self.assertAlmostEqual(feats["hero_fold_rate"], 1.0, places=6)
        self.assertAlmostEqual(feats["hero_aggression_rate"], 0.0, places=6)


class RemapTests(unittest.TestCase):
    def test_threshold_maps_to_half_and_is_monotonic(self):
        self.assertAlmostEqual(float(remap_score(0.706, 0.706)), 0.5, places=6)
        self.assertLess(float(remap_score(0.2, 0.706)), 0.5)
        self.assertGreater(float(remap_score(0.9, 0.706)), 0.5)
        # monotonic
        a, b = float(remap_score(0.4, 0.706)), float(remap_score(0.5, 0.706))
        self.assertLessEqual(a, b)


class FallbackTests(unittest.TestCase):
    def test_fallback_never_false_positives(self):
        # Conservative heuristic must never cross the 0.5 bot boundary.
        for atype in ("fold", "check", "call", "bet", "raise"):
            chunk = [_hand(actions=[_action(2, atype, bb=10.0, amt=0.2, pb=0.2, pa=0.4)])
                     for _ in range(10)]
            self.assertLess(heuristic_chunk_score(chunk), 0.5)
        self.assertEqual(heuristic_chunk_score([]), 0.1)


class ModelArtifactTests(unittest.TestCase):
    def setUp(self):
        self.model = BotDetectorModel.load()

    def test_artifact_loads_with_expected_features(self):
        if self.model is None:
            self.skipTest("trained artifact not present")
        self.assertEqual(self.model.feature_names, list(FEATURE_NAMES))
        self.assertTrue(0.0 < self.model.threshold < 1.0)
        self.assertGreater(self.model.meta.get("oof_ap", 0.0), 0.6)

    def test_scores_in_unit_interval_and_one_per_chunk(self):
        if self.model is None:
            self.skipTest("trained artifact not present")
        chunks = [
            [_hand(actions=[_action(2, "raise", bb=22.0, amt=0.45, pb=0.3, pa=0.7),
                            _action(1, "call", bb=22.0, amt=0.45, pb=0.7, pa=1.1)])
             for _ in range(8)],
            [_hand(hero_seat=3, actions=[_action(3, "fold")]) for _ in range(8)],
        ]
        scores = self.model.score_chunks(chunks)
        self.assertEqual(len(scores), len(chunks))
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))

    def test_empty_chunk_gets_safe_low_score(self):
        if self.model is None:
            self.skipTest("trained artifact not present")
        self.assertLess(self.model.score_chunks([[]])[0], 0.5)


if __name__ == "__main__":
    unittest.main()
