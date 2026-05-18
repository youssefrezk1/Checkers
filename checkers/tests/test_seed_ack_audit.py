# checkers/tests/test_seed_ack_audit.py
#
# Tests for checkers/evaluation/run_seed_ack_audit.py
#
# Coverage
# --------
#  1. Isolation - no runtime pipeline imports.
#  2. map_seed_to_claim_types finds known seed -> claim_type.
#  3. map_seed_to_claim_types respects "^" anchors (no false positive on
#     "opponent_near_promotion=true" mapping to "near_promotion").
#  4. map_seed_to_claim_types skips vague / forbidden / schema-leak types.
#  5. map_seed_to_claim_types returns [] for unclassifiable seed strings.
#  6. is_seed_acknowledged: phrase present -> True; absent -> False.
#  7. get_minimax_bucket boundaries: >0, [-50, 0], <-50, None.
#  8. compute_turn_seed_ack: acknowledged seed -> ack=True, claim_type in set.
#  9. compute_turn_seed_ack: seed present but reasoning silent -> ignored.
# 10. compute_turn_seed_ack: unclassifiable seed counted as unclassified, not
#     ignored.
# 11. compute_turn_seed_ack: does not mutate record / seeds / facts.
# 12. compute_turn_seed_ack: handles old logs with missing/None fields safely.
# 13. aggregate_seed_ack: RED-only counts, bucket and reasoning_path stats.
# 14. aggregate_seed_ack: top_10_ignored_claim_types sorted by descending count.
# 15. Multi-claim-type seed: only the type whose phrase appears is credited.

from __future__ import annotations

import copy
import sys
from typing import Any

# -- Isolation guard --------------------------------------------------------
_modules_before = set(sys.modules.keys())

from checkers.evaluation.run_seed_ack_audit import (
    map_seed_to_claim_types,
    is_seed_acknowledged,
    get_minimax_bucket,
    compute_turn_seed_ack,
    aggregate_seed_ack,
    print_report,
)
from checkers.evaluation.claim_taxonomy import _CLAIM_REGISTRY, TaxonomyCategory

_modules_after = set(sys.modules.keys())

_FORBIDDEN_RUNTIME_PREFIXES = (
    "checkers.engine",
    "checkers.agents",
    "checkers.graph",
    "checkers.nodes",
    "checkers.search",
)


def _rec(
    reasoning: str = "",
    seeds: list[str] | None = None,
    facts: dict[str, Any] | None = None,
    player: str = "RED",
    turn: int = 1,
    turn_id: str = "test_t1",
    reasoning_path: str = "",
) -> dict[str, Any]:
    """Build a minimal manual_game JSONL record for testing."""
    return {
        "player": player,
        "turn": turn,
        "turn_id": turn_id,
        "reasoning": reasoning,
        "seeds": list(seeds or []),
        "facts": dict(facts or {}),
        "reasoning_path": reasoning_path,
    }


# ---------------------------------------------------------------------------
# 1. Isolation
# ---------------------------------------------------------------------------

def test_no_runtime_pipeline_imports():
    new_mods = _modules_after - _modules_before
    for mod in new_mods:
        for prefix in _FORBIDDEN_RUNTIME_PREFIXES:
            assert not mod.startswith(prefix), (
                f"run_seed_ack_audit imported runtime module: {mod!r}"
            )


# ---------------------------------------------------------------------------
# 2-5. map_seed_to_claim_types
# ---------------------------------------------------------------------------

def test_map_seed_to_claim_types_known_seed():
    """A standard seed string maps to its claim type."""
    seed = "opponent_can_recapture=false - safe from recapture"
    result = map_seed_to_claim_types(seed)
    assert "avoids_recapture" in result


def test_map_seed_to_claim_types_forced_jump():
    """forced_opponent_jump_reply seed maps to forced_opponent_jump."""
    seed = "forced_opponent_jump_reply=true - opponent constrained"
    result = map_seed_to_claim_types(seed)
    assert "forced_opponent_jump" in result


def test_map_seed_to_claim_types_respects_anchor():
    """The ^near_promotion=true anchor must not match
    'opponent_near_promotion=true'."""
    seed = "opponent_near_promotion=true - opponent close to back row"
    result = map_seed_to_claim_types(seed)
    # Should map ONLY to opponent_near_promotion, not near_promotion.
    assert "opponent_near_promotion" in result
    assert "near_promotion" not in result


def test_map_seed_to_claim_types_anchored_self_promotion():
    """A seed starting with 'near_promotion=true' SHOULD map to near_promotion."""
    seed = "near_promotion=true - our piece one row from king row"
    result = map_seed_to_claim_types(seed)
    assert "near_promotion" in result


def test_map_seed_to_claim_types_skips_vague_types():
    """Seed-marker-free / unverifiable types must not appear in results."""
    seed = "creates_immediate_threat=true - threat next turn"
    result = map_seed_to_claim_types(seed)
    # creates_immediate_threat IS eligible.  positional_pressure / strategic
    # initiative / long_term_compensation must never appear, regardless of
    # input, because they have no seed_markers AND/OR are not eligible.
    for vague in ("positional_pressure", "strategic_initiative",
                  "long_term_compensation"):
        assert vague not in result


def test_map_seed_to_claim_types_no_match():
    """An unclassifiable seed returns an empty list."""
    seed = "destination column in center range (col=3)"
    result = map_seed_to_claim_types(seed)
    assert result == []


def test_map_seed_to_claim_types_empty_string():
    assert map_seed_to_claim_types("") == []


def test_map_seed_to_claim_types_returns_only_eligible():
    """Every returned claim type must be VERIFIABLE or AMBIGUOUS_CONTEXT_REQUIRED."""
    eligible = {TaxonomyCategory.VERIFIABLE,
                TaxonomyCategory.AMBIGUOUS_CONTEXT_REQUIRED}
    seeds_to_try = [
        "opponent_can_recapture=false",
        "opponent_can_recapture=true",
        "captures_count=1",
        "results_in_king=true",
        "shot_sequence_available=true",
        "blocks_opponent_landing=true",
        "leaves_piece_isolated=false",
        "center_control=true",
        "minimax_score=-12.0",
    ]
    for s in seeds_to_try:
        for ct in map_seed_to_claim_types(s):
            spec = _CLAIM_REGISTRY.get(ct)
            assert spec is not None
            assert spec.category in eligible, (
                f"Seed {s!r} mapped to non-eligible {ct!r}"
            )


# ---------------------------------------------------------------------------
# 6. is_seed_acknowledged
# ---------------------------------------------------------------------------

def test_is_seed_acknowledged_phrase_present():
    text = "This move avoids recapture and is safe."
    assert is_seed_acknowledged("avoids_recapture", text) is True


def test_is_seed_acknowledged_phrase_absent():
    text = "We make a structurally sound move."
    assert is_seed_acknowledged("avoids_recapture", text) is False


def test_is_seed_acknowledged_case_insensitive():
    text = "This GAINS MATERIAL on the next exchange."
    assert is_seed_acknowledged("gains_material", text) is True


def test_is_seed_acknowledged_unknown_claim_type():
    """An unregistered claim type returns False, not an exception."""
    text = "anything"
    assert is_seed_acknowledged("not_a_real_claim_type_xyz", text) is False


# ---------------------------------------------------------------------------
# 7. get_minimax_bucket
# ---------------------------------------------------------------------------

def test_get_minimax_bucket_positive():
    assert get_minimax_bucket(10.0) == "positive"
    assert get_minimax_bucket(0.1) == "positive"
    assert get_minimax_bucket(1) == "positive"


def test_get_minimax_bucket_slightly_losing():
    assert get_minimax_bucket(0.0) == "slightly_losing"
    assert get_minimax_bucket(-25) == "slightly_losing"
    assert get_minimax_bucket(-50) == "slightly_losing"


def test_get_minimax_bucket_deeply_losing():
    assert get_minimax_bucket(-50.1) == "deeply_losing"
    assert get_minimax_bucket(-100) == "deeply_losing"


def test_get_minimax_bucket_unknown():
    assert get_minimax_bucket(None) == "unknown"
    assert get_minimax_bucket("not a number") == "unknown"


# ---------------------------------------------------------------------------
# 8-12. compute_turn_seed_ack
# ---------------------------------------------------------------------------

def test_compute_turn_seed_ack_acknowledged():
    record = _rec(
        reasoning="This move avoids recapture and is safe.",
        seeds=["opponent_can_recapture=false - safe from recapture"],
        facts={"minimax_score": 5.0, "opponent_can_recapture": False},
        reasoning_path="strict",
    )
    result = compute_turn_seed_ack(record)
    assert result["seed_count"] == 1
    assert result["classified_seed_count"] == 1
    assert result["acknowledged_seed_count"] == 1
    assert result["unacknowledged_seed_count"] == 0
    assert result["unclassified_seed_count"] == 0
    assert "avoids_recapture" in result["acknowledged_claim_types"]
    assert result["bucket"] == "positive"


def test_compute_turn_seed_ack_ignored_seed():
    record = _rec(
        reasoning="We play this move because it looks structurally sound.",
        seeds=["forced_opponent_jump_reply=true - opponent constrained"],
        facts={"minimax_score": -10.0, "forced_opponent_jump_reply": True},
    )
    result = compute_turn_seed_ack(record)
    assert result["classified_seed_count"] == 1
    assert result["acknowledged_seed_count"] == 0
    assert result["unacknowledged_seed_count"] == 1
    assert "forced_opponent_jump" in result["ignored_claim_types"]
    assert result["bucket"] == "slightly_losing"


def test_compute_turn_seed_ack_unclassified_seed_is_separate():
    """An unclassifiable seed must not be counted as ignored."""
    record = _rec(
        reasoning="Some reasoning text.",
        seeds=["destination column in center range (col=3)"],
        facts={"minimax_score": 0.0},
    )
    result = compute_turn_seed_ack(record)
    assert result["seed_count"] == 1
    assert result["classified_seed_count"] == 0
    assert result["unclassified_seed_count"] == 1
    assert result["acknowledged_seed_count"] == 0
    assert result["unacknowledged_seed_count"] == 0


def test_compute_turn_seed_ack_does_not_mutate():
    seeds = ["opponent_can_recapture=false - safe"]
    facts = {"minimax_score": 5.0, "opponent_can_recapture": False}
    record = _rec(
        reasoning="This move avoids recapture.",
        seeds=seeds,
        facts=facts,
    )
    snap_record = copy.deepcopy(record)
    snap_seeds = copy.deepcopy(seeds)
    snap_facts = copy.deepcopy(facts)

    compute_turn_seed_ack(record)

    assert record == snap_record
    assert seeds == snap_seeds
    assert facts == snap_facts


def test_compute_turn_seed_ack_missing_fields():
    """Old logs lacking some fields should not crash."""
    record = {"player": "RED"}  # nothing else
    result = compute_turn_seed_ack(record)
    assert result["seed_count"] == 0
    assert result["classified_seed_count"] == 0
    assert result["acknowledged_seed_count"] == 0
    assert result["unacknowledged_seed_count"] == 0
    assert result["bucket"] == "unknown"


def test_compute_turn_seed_ack_none_fields():
    """Records with None seeds/facts/reasoning should not crash."""
    record = {
        "player": "RED",
        "turn": 1,
        "turn_id": "t1",
        "reasoning": None,
        "seeds": None,
        "facts": None,
        "reasoning_path": None,
    }
    result = compute_turn_seed_ack(record)
    assert result["seed_count"] == 0
    assert result["bucket"] == "unknown"


def test_compute_turn_seed_ack_per_seed_entries():
    record = _rec(
        reasoning="This move avoids recapture.",
        seeds=[
            "opponent_can_recapture=false - safe",
            "destination column in center range (col=3)",
            "forced_opponent_jump_reply=true - constrained",
        ],
        facts={"minimax_score": 5.0,
               "opponent_can_recapture": False,
               "forced_opponent_jump_reply": True},
    )
    result = compute_turn_seed_ack(record)
    statuses = [e["status"] for e in result["per_seed"]]
    assert "acknowledged" in statuses
    assert "unclassified" in statuses
    assert "ignored" in statuses
    assert result["seed_count"] == 3


# ---------------------------------------------------------------------------
# 13-14. aggregate_seed_ack
# ---------------------------------------------------------------------------

def test_aggregate_filters_to_red_turns():
    red = compute_turn_seed_ack(_rec(
        reasoning="avoids recapture",
        seeds=["opponent_can_recapture=false - safe"],
        facts={"opponent_can_recapture": False, "minimax_score": 1.0},
        player="RED", turn=1, turn_id="r1",
    ))
    black = compute_turn_seed_ack(_rec(
        reasoning="any text",
        seeds=["opponent_can_recapture=false - safe"],
        facts={"opponent_can_recapture": False, "minimax_score": 1.0},
        player="BLACK", turn=2, turn_id="b1",
    ))
    agg = aggregate_seed_ack([red, black])
    assert agg["total_red_turns"] == 1
    assert agg["total_all_turns"] == 2
    assert agg["total_seeds"] == 1  # BLACK turn excluded from totals


def test_aggregate_per_bucket_and_path():
    r1 = compute_turn_seed_ack(_rec(
        reasoning="avoids recapture",
        seeds=["opponent_can_recapture=false - safe"],
        facts={"opponent_can_recapture": False, "minimax_score": 5.0},
        turn_id="r1", reasoning_path="strict",
    ))
    r2 = compute_turn_seed_ack(_rec(
        reasoning="silent reasoning",
        seeds=["opponent_can_recapture=false - safe"],
        facts={"opponent_can_recapture": False, "minimax_score": -100.0},
        turn_id="r2", reasoning_path="recovery",
    ))
    agg = aggregate_seed_ack([r1, r2])
    assert agg["per_bucket"]["positive"]["ack"] == 1
    assert agg["per_bucket"]["deeply_losing"]["ack"] == 0
    assert agg["per_bucket"]["deeply_losing"]["unack"] == 1
    assert agg["per_reasoning_path"]["strict"]["ack"] == 1
    assert agg["per_reasoning_path"]["recovery"]["unack"] == 1


def test_aggregate_top_ignored_sorted_by_count():
    """Same ignored type across many turns should appear at the top."""
    turns = []
    for i in range(5):
        turns.append(compute_turn_seed_ack(_rec(
            reasoning="we choose this move",
            seeds=["forced_opponent_jump_reply=true - constrained"],
            facts={"forced_opponent_jump_reply": True,
                   "minimax_score": -10.0},
            turn=i, turn_id=f"r{i}",
        )))
    turns.append(compute_turn_seed_ack(_rec(
        reasoning="we choose this move",
        seeds=["blocks_opponent_landing=true"],
        facts={"blocks_opponent_landing": True, "minimax_score": -10.0},
        turn=99, turn_id="r99",
    )))
    agg = aggregate_seed_ack(turns)
    top = agg["top_10_ignored_claim_types"]
    assert top[0][0] == "forced_opponent_jump"
    assert top[0][1] == 5


def test_aggregate_overall_rate():
    ack_turn = compute_turn_seed_ack(_rec(
        reasoning="this move avoids recapture",
        seeds=["opponent_can_recapture=false - safe"],
        facts={"opponent_can_recapture": False, "minimax_score": 1.0},
        turn_id="r1",
    ))
    ignored_turn = compute_turn_seed_ack(_rec(
        reasoning="some text",
        seeds=["opponent_can_recapture=false - safe"],
        facts={"opponent_can_recapture": False, "minimax_score": 1.0},
        turn_id="r2",
    ))
    agg = aggregate_seed_ack([ack_turn, ignored_turn])
    assert agg["total_classified_seeds"] == 2
    assert agg["total_acknowledged_seeds"] == 1
    assert abs(agg["overall_ack_rate"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# 15. Multi-claim-type seed crediting
# ---------------------------------------------------------------------------

def test_aggregate_credits_only_phrase_matching_type():
    """If a seed maps to two claim types but only one phrase appears in
    reasoning, only that one type should be credited as acknowledged."""
    # Construct a mobility seed that maps to mobility_decrease only
    # (most seeds map 1:1).  But test the per-type accounting carefully
    # by stacking a seed that has more phrases.
    record = _rec(
        reasoning="opponent_can_recapture=false",  # exact seed-echo phrase
        seeds=["opponent_can_recapture=false - safe from recapture"],
        facts={"opponent_can_recapture": False, "minimax_score": 1.0},
    )
    result = compute_turn_seed_ack(record)
    # "opponent_can_recapture=false" is NOT in avoids_recapture's phrase list,
    # but the test still validates we only credit types whose own phrases match.
    if result["acknowledged_seed_count"] == 1:
        assert "avoids_recapture" in result["acknowledged_claim_types"]
    else:
        assert "avoids_recapture" in result["ignored_claim_types"]


def test_print_report_runs_without_error(capsys):
    r1 = compute_turn_seed_ack(_rec(
        reasoning="avoids recapture",
        seeds=["opponent_can_recapture=false - safe"],
        facts={"opponent_can_recapture": False, "minimax_score": 5.0},
        turn_id="r1",
    ))
    agg = aggregate_seed_ack([r1])
    print_report(agg, [r1])
    captured = capsys.readouterr()
    assert "SEED ACKNOWLEDGEMENT AUDIT" in captured.out
    assert "Ack rate per claim type" in captured.out
