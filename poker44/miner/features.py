"""Chunk-level feature extraction for Poker44 bot detection.

A *chunk* is a list of miner-visible poker hands describing one classified
player ("hero"). The validator labels the whole chunk bot (1) or human (0), and
expects one risk score per chunk. This module turns a chunk into a fixed,
ordered feature vector.

Design constraints (derived from the production payload contract):

* Operates only on fields that survive ``build_miner_payload_hand``:
  ``metadata.hero_seat``, per-seat ``starting_stack`` (bb units), and a
  *sampled, coarsened* action list (``action_type``, ``normalized_amount_bb``,
  ``pot_before``/``pot_after``). Cards, outcomes and timing are never present.
* No identifiers, dates, hashes or ordering are used as features.
* The hero sits at different seats across hands, so every per-hand signal is
  resolved against that hand's own ``hero_seat`` and then aggregated.

To avoid train/serve skew, training data must be projected through
``poker44.validator.payload_view.build_miner_payload_hand`` exactly once before
calling :func:`extract_chunk_features`; in production the validator has already
applied that projection, so hands are fed straight through.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

_AGGRO = ("bet", "raise")
_PASSIVE = ("call", "check")
_VOLUNTARY = ("call", "bet", "raise")  # voluntary chips in (vpip-style)
_ACTION_TYPES = ("fold", "check", "call", "bet", "raise")
_STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
_EPS = 1e-9


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _entropy(counts: Sequence[float]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p + _EPS)
    # normalise to [0, 1] over the number of categories
    k = sum(1 for c in counts if c > 0)
    max_ent = math.log(len(counts)) if len(counts) > 1 else 1.0
    return ent / max_ent if max_ent > 0 else 0.0


def _safe_div(num: float, den: float) -> float:
    return num / den if den > _EPS else 0.0


def _agg(values: Sequence[float]) -> Dict[str, float]:
    """Mean, std and coefficient of variation for a list of per-hand values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0}
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())
    cv = _safe_div(std, abs(mean)) if abs(mean) > _EPS else 0.0
    return {"mean": mean, "std": std, "cv": cv}


def _hand_signals(hand: Mapping[str, Any]) -> Dict[str, Any]:
    """Per-hand hero-centric + table-level signals."""
    meta = hand.get("metadata") or {}
    hero_seat = int(_f(meta.get("hero_seat"), 0))
    players = hand.get("players") or []
    actions = hand.get("actions") or []
    streets = hand.get("streets") or []

    hero_stack = 0.0
    table_stacks: List[float] = []
    for p in players:
        if not isinstance(p, Mapping):
            continue
        stack = _f(p.get("starting_stack"))
        table_stacks.append(stack)
        if int(_f(p.get("seat"), -1)) == hero_seat:
            hero_stack = stack

    counts = {t: 0 for t in _ACTION_TYPES}
    hero_counts = {t: 0 for t in _ACTION_TYPES}
    hero_bet_pot_ratios: List[float] = []
    hero_bet_sizes_bb: List[float] = []
    hero_voluntary_pf = 0
    hero_raise_pf = 0
    hero_acted = 0
    hero_max_street = -1
    table_actions = 0
    hero_pf_aggro = hero_pf_actions = 0
    hero_post_aggro = hero_post_actions = 0
    hero_bet_small = hero_bet_mid = hero_bet_large = 0  # by pot fraction

    for a in actions:
        if not isinstance(a, Mapping):
            continue
        atype = str(a.get("action_type") or "").lower()
        if atype not in counts:
            continue
        counts[atype] += 1
        table_actions += 1
        is_hero = int(_f(a.get("actor_seat"), -1)) == hero_seat
        if not is_hero:
            continue
        hero_counts[atype] += 1
        hero_acted += 1
        street = str(a.get("street") or "preflop").lower()
        s_idx = _STREET_ORDER.get(street, 0)
        hero_max_street = max(hero_max_street, s_idx)
        is_aggro = atype in _AGGRO
        if street == "preflop":
            hero_pf_actions += 1
            hero_pf_aggro += 1 if is_aggro else 0
            if atype in _VOLUNTARY:
                hero_voluntary_pf = 1
            if atype == "raise":
                hero_raise_pf = 1
        else:
            hero_post_actions += 1
            hero_post_aggro += 1 if is_aggro else 0
        if is_aggro:
            amt = _f(a.get("normalized_amount_bb"))
            pot_before = _f(a.get("pot_before"))
            if amt > 0:
                hero_bet_sizes_bb.append(amt)
            ratio = _safe_div(_f(a.get("amount")), pot_before)
            if ratio > 0:
                hero_bet_pot_ratios.append(min(ratio, 5.0))
                if ratio < 0.5:
                    hero_bet_small += 1
                elif ratio <= 1.0:
                    hero_bet_mid += 1
                else:
                    hero_bet_large += 1

    hero_total = sum(hero_counts.values())
    hero_aggro = hero_counts["bet"] + hero_counts["raise"]
    hero_passive = hero_counts["call"] + hero_counts["check"]
    pot_after_vals = [_f(a.get("pot_after")) for a in actions if isinstance(a, Mapping)]
    max_pot = max(pot_after_vals) if pot_after_vals else 0.0
    table_stack_mean = float(np.mean(table_stacks)) if table_stacks else 0.0

    return {
        "hero_stack": hero_stack,
        "hero_stack_rel": _safe_div(hero_stack, table_stack_mean),
        "table_stacks": table_stacks,
        "n_players": len([p for p in players if isinstance(p, Mapping)]),
        "n_actions": table_actions,
        "n_streets": len(streets),
        "max_pot": max_pot,
        "counts": counts,
        "hero_counts": hero_counts,
        "hero_total": hero_total,
        "hero_acted": 1 if hero_acted else 0,
        "hero_aggro": hero_aggro,
        "hero_passive": hero_passive,
        "hero_fold": hero_counts["fold"],
        "hero_voluntary_pf": hero_voluntary_pf,
        "hero_raise_pf": hero_raise_pf,
        "hero_aggr_ratio": _safe_div(hero_aggro, hero_total),
        "hero_pf_aggro": hero_pf_aggro,
        "hero_pf_actions": hero_pf_actions,
        "hero_post_aggro": hero_post_aggro,
        "hero_post_actions": hero_post_actions,
        "hero_bet_small": hero_bet_small,
        "hero_bet_mid": hero_bet_mid,
        "hero_bet_large": hero_bet_large,
        "hero_bet_pot_ratios": hero_bet_pot_ratios,
        "hero_bet_sizes_bb": hero_bet_sizes_bb,
        "hero_max_street": hero_max_street,
        "hero_reached_flop": 1 if hero_max_street >= 1 else 0,
        "hero_reached_river": 1 if hero_max_street >= 3 else 0,
    }


# Ordered, stable list of feature names. The miner and trainer must agree on it.
FEATURE_NAMES: List[str] = [
    # hero action-mix rates (fraction of hero actions)
    "hero_fold_rate",
    "hero_check_rate",
    "hero_call_rate",
    "hero_bet_rate",
    "hero_raise_rate",
    # hero aggregate style
    "hero_aggression_rate",          # (bet+raise)/all hero actions  [audit feature]
    "hero_aggression_factor",        # (bet+raise)/calls
    "hero_action_entropy",           # entropy of hero action mix    [audit feature]
    "hero_vpip_rate",                # fraction of hands hero voluntarily entered preflop
    "hero_pfr_rate",                 # fraction of hands hero raised preflop
    "hero_actions_per_hand_mean",
    "hero_actions_per_hand_std",
    "hero_acted_rate",
    # hero continuation / showdown reach
    "hero_reach_flop_rate",
    "hero_reach_river_rate",
    "hero_max_street_mean",
    # hero bet sizing (survives coarsening as coarse signal)
    "hero_bet_size_bb_mean",
    "hero_bet_size_bb_std",
    "hero_bet_pot_ratio_mean",
    "hero_bet_pot_ratio_std",
    # hero stack (audit's strongest single feature)
    "hero_stack_mean",
    "hero_stack_std",
    "hero_stack_cv",
    # consistency / regularity across hands (bots are more regular)
    "hero_aggr_ratio_std",
    "hero_aggr_ratio_cv",
    "hero_vpip_consistency",         # |0.5 - vpip_rate|*2 -> 1 means very decided
    # table-level context
    "table_players_per_hand_mean",
    "table_actions_per_hand_mean",
    "table_streets_per_hand_mean",
    "table_action_entropy",
    "table_aggression_rate",
    "table_pot_mean",
    "table_pot_std",
    "table_stack_mean",
    "table_stack_std",
    # per-street hero aggression (bots tend to rigid street strategies)
    "hero_preflop_aggression_rate",
    "hero_postflop_aggression_rate",
    # hero bet-sizing shape by pot fraction (bots cluster on fixed sizings)
    "hero_bet_small_frac",
    "hero_bet_mid_frac",
    "hero_bet_large_frac",
    # hero stack relative to the table
    "hero_stack_rel_mean",
    "hero_stack_rel_std",
    # chunk shape
    "chunk_hand_count",
]


def extract_chunk_features(chunk: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """Return a name->value feature dict for one miner-visible chunk."""
    hands = [h for h in (chunk or []) if isinstance(h, Mapping)]
    if not hands:
        return {name: 0.0 for name in FEATURE_NAMES}

    sigs = [_hand_signals(h) for h in hands]
    n = float(len(sigs))

    # hero action totals across the chunk
    hero_tot_counts = {t: sum(s["hero_counts"][t] for s in sigs) for t in _ACTION_TYPES}
    hero_total_actions = sum(hero_tot_counts.values())
    hero_aggro = hero_tot_counts["bet"] + hero_tot_counts["raise"]
    hero_calls = hero_tot_counts["call"]

    table_tot_counts = {t: sum(s["counts"][t] for s in sigs) for t in _ACTION_TYPES}
    table_total_actions = sum(table_tot_counts.values())
    table_aggro = table_tot_counts["bet"] + table_tot_counts["raise"]

    hero_actions_per_hand = [float(s["hero_total"]) for s in sigs]
    hero_aggr_ratios = [s["hero_aggr_ratio"] for s in sigs]
    hero_stacks = [s["hero_stack"] for s in sigs if s["hero_stack"] > 0]
    table_stacks: List[float] = [v for s in sigs for v in s["table_stacks"] if v > 0]
    hero_bet_sizes = [v for s in sigs for v in s["hero_bet_sizes_bb"]]
    hero_bet_pot = [v for s in sigs for v in s["hero_bet_pot_ratios"]]
    max_pots = [s["max_pot"] for s in sigs]
    hero_stack_rels = [s["hero_stack_rel"] for s in sigs if s["hero_stack_rel"] > 0]

    pf_aggro = sum(s["hero_pf_aggro"] for s in sigs)
    pf_actions = sum(s["hero_pf_actions"] for s in sigs)
    post_aggro = sum(s["hero_post_aggro"] for s in sigs)
    post_actions = sum(s["hero_post_actions"] for s in sigs)
    bet_small = sum(s["hero_bet_small"] for s in sigs)
    bet_mid = sum(s["hero_bet_mid"] for s in sigs)
    bet_large = sum(s["hero_bet_large"] for s in sigs)
    bet_total = max(bet_small + bet_mid + bet_large, 0)

    vpip_rate = float(np.mean([s["hero_voluntary_pf"] for s in sigs]))
    stack_rel_agg = _agg(hero_stack_rels)

    stack_agg = _agg(hero_stacks)
    aggr_ratio_agg = _agg(hero_aggr_ratios)
    apph_agg = _agg(hero_actions_per_hand)
    bet_size_agg = _agg(hero_bet_sizes)
    bet_pot_agg = _agg(hero_bet_pot)
    pot_agg = _agg(max_pots)
    table_stack_agg = _agg(table_stacks)

    feats: Dict[str, float] = {
        "hero_fold_rate": _safe_div(hero_tot_counts["fold"], hero_total_actions),
        "hero_check_rate": _safe_div(hero_tot_counts["check"], hero_total_actions),
        "hero_call_rate": _safe_div(hero_tot_counts["call"], hero_total_actions),
        "hero_bet_rate": _safe_div(hero_tot_counts["bet"], hero_total_actions),
        "hero_raise_rate": _safe_div(hero_tot_counts["raise"], hero_total_actions),
        "hero_aggression_rate": _safe_div(hero_aggro, hero_total_actions),
        "hero_aggression_factor": _safe_div(hero_aggro, hero_calls),
        "hero_action_entropy": _entropy([hero_tot_counts[t] for t in _ACTION_TYPES]),
        "hero_vpip_rate": vpip_rate,
        "hero_pfr_rate": float(np.mean([s["hero_raise_pf"] for s in sigs])),
        "hero_actions_per_hand_mean": apph_agg["mean"],
        "hero_actions_per_hand_std": apph_agg["std"],
        "hero_acted_rate": float(np.mean([s["hero_acted"] for s in sigs])),
        "hero_reach_flop_rate": float(np.mean([s["hero_reached_flop"] for s in sigs])),
        "hero_reach_river_rate": float(np.mean([s["hero_reached_river"] for s in sigs])),
        "hero_max_street_mean": float(
            np.mean([max(0, s["hero_max_street"]) for s in sigs])
        ),
        "hero_bet_size_bb_mean": bet_size_agg["mean"],
        "hero_bet_size_bb_std": bet_size_agg["std"],
        "hero_bet_pot_ratio_mean": bet_pot_agg["mean"],
        "hero_bet_pot_ratio_std": bet_pot_agg["std"],
        "hero_stack_mean": stack_agg["mean"],
        "hero_stack_std": stack_agg["std"],
        "hero_stack_cv": stack_agg["cv"],
        "hero_aggr_ratio_std": aggr_ratio_agg["std"],
        "hero_aggr_ratio_cv": aggr_ratio_agg["cv"],
        "hero_vpip_consistency": abs(0.5 - vpip_rate) * 2.0,
        "table_players_per_hand_mean": float(np.mean([s["n_players"] for s in sigs])),
        "table_actions_per_hand_mean": float(np.mean([s["n_actions"] for s in sigs])),
        "table_streets_per_hand_mean": float(np.mean([s["n_streets"] for s in sigs])),
        "table_action_entropy": _entropy([table_tot_counts[t] for t in _ACTION_TYPES]),
        "table_aggression_rate": _safe_div(table_aggro, table_total_actions),
        "table_pot_mean": pot_agg["mean"],
        "table_pot_std": pot_agg["std"],
        "table_stack_mean": table_stack_agg["mean"],
        "table_stack_std": table_stack_agg["std"],
        "hero_preflop_aggression_rate": _safe_div(pf_aggro, pf_actions),
        "hero_postflop_aggression_rate": _safe_div(post_aggro, post_actions),
        "hero_bet_small_frac": _safe_div(bet_small, bet_total),
        "hero_bet_mid_frac": _safe_div(bet_mid, bet_total),
        "hero_bet_large_frac": _safe_div(bet_large, bet_total),
        "hero_stack_rel_mean": stack_rel_agg["mean"],
        "hero_stack_rel_std": stack_rel_agg["std"],
        "chunk_hand_count": n,
    }
    # guarantee key order / completeness
    return {name: float(feats.get(name, 0.0)) for name in FEATURE_NAMES}


def features_to_vector(feats: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(feats.get(name, 0.0)) for name in FEATURE_NAMES], dtype=float)


def chunk_to_vector(chunk: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return features_to_vector(extract_chunk_features(chunk))
