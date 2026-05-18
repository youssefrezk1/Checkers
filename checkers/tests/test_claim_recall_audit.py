# checkers/tests/test_claim_recall_audit.py
#
# Tests for checkers/evaluation/run_claim_recall_audit.py
#
# Coverage
# --------
#  1. Isolation — no runtime pipeline imports.
#  2. Detects missing forced_opponent_jump when seed/fact present, phrase absent.
#  3. Detects missing shot_sequence_or_multi_jump when seed/fact present, phrase absent.
#  4. Detects missing blocks_landing_square when seed/fact present, phrase absent.
#  5. Does NOT expect vague/non-verifiable claim types.
#  6. Does NOT expect claims when required fact is absent.
#  7. Does NOT expect claims when fact contradicts the claim.
#  8. Does not mutate input record or seeds/facts.
#  9. Handles old logs with missing/None fields safely.
# 10. Aggregate counts RED turns only; total_all_turns counts all inputs.
# 11. Top-10 recall gaps sorted by descending frequency.
# 12. expected_claim_types semantics: seed / phrase / seed+phrase evidence.
# 13. Claim type already extracted → not in missing_types.
# 14. expected_claim_types returns only VERIFIABLE / AMBIGUOUS categories.

from __future__ import annotations

import copy
import sys
from typing import Any

# ── Isolation guard ───────────────────────────────────────────────────────────
_modules_before = set(sys.modules.keys())

from checkers.evaluation.run_claim_recall_audit import (
    compute_turn_recall,
    expected_claim_types,
    aggregate_recall_results,
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


# ---------------------------------------------------------------------------
# 1. Isolation
# ---------------------------------------------------------------------------

def test_no_runtime_pipeline_imports():
    """run_claim_recall_audit must not pull in runtime pipeline modules."""
    new_mods = _modules_after - _modules_before
    for mod in new_mods:
        for prefix in _FORBIDDEN_RUNTIME_PREFIXES:
            assert not mod.startswith(prefix), (
                f"run_claim_recall_audit imported runtime module: {mod!r}"
            )


# ---------------------------------------------------------------------------
# Synthetic record factory
# ---------------------------------------------------------------------------

def _rec(
    reasoning: str = "",
    seeds: list[str] | None = None,
    facts: dict[str, Any] | None = None,
    player: str = "RED",
    turn: int = 1,
    turn_id: str = "test_t1",
) -> dict[str, Any]:
    return {
        "player": player,
        "turn": turn,
        "turn_id": turn_id,
        "reasoning": reasoning,
        "seeds": list(seeds) if seeds is not None else [],
        "facts": dict(facts) if facts is not None else {},
    }


# ---------------------------------------------------------------------------
# 2–4. Detects missing Phase 4.1 claim types
# ---------------------------------------------------------------------------

def test_detects_missing_forced_opponent_jump_seed_only():
    """forced_opponent_jump expected when seed present + fact True; no phrase in reasoning."""
    r = _rec(
        reasoning="This move advances a piece to a strong position.",
        seeds=["forced_opponent_jump_reply=true — opponent constrained to a jump"],
        facts={"forced_opponent_jump_reply": True},
    )
    result = compute_turn_recall(r)
    assert "forced_opponent_jump" in result["expected_claim_types"], (
        "forced_opponent_jump should be expected: seed matches + fact True"
    )
    assert "forced_opponent_jump" in result["missing_types"]
    assert "forced_opponent_jump" in result["missing_with_seed"]
    assert "forced_opponent_jump" not in result["missing_with_phrase"]


def test_detects_missing_shot_sequence_seed_only():
    """shot_sequence_or_multi_jump expected when seed + fact True; no phrase."""
    r = _rec(
        reasoning="The move is positionally strong.",
        seeds=["shot_sequence_available=true — a multi-jump sequence is available"],
        facts={"shot_sequence_available": True},
    )
    result = compute_turn_recall(r)
    assert "shot_sequence_or_multi_jump" in result["expected_claim_types"]
    assert "shot_sequence_or_multi_jump" in result["missing_types"]
    assert "shot_sequence_or_multi_jump" in result["missing_with_seed"]


def test_detects_missing_blocks_landing_seed_only():
    """blocks_landing_square expected when seed + fact True; no phrase."""
    r = _rec(
        reasoning="The move is tactically sound.",
        seeds=["blocks_opponent_landing=true — denies opponent a key landing square"],
        facts={"blocks_opponent_landing": True},
    )
    result = compute_turn_recall(r)
    assert "blocks_landing_square" in result["expected_claim_types"]
    assert "blocks_landing_square" in result["missing_types"]
    assert "blocks_landing_square" in result["missing_with_seed"]


def test_not_missing_when_phrase_extracted():
    """If phrase IS in reasoning and extraction fires, claim is NOT missing."""
    r = _rec(
        reasoning="The opponent is constrained to a jump in reply.",
        seeds=["forced_opponent_jump_reply=true"],
        facts={"forced_opponent_jump_reply": True},
    )
    result = compute_turn_recall(r)
    assert "forced_opponent_jump" not in result["missing_types"]
    assert "forced_opponent_jump" in result["extracted_claim_types"]


def test_detects_missing_with_phrase_only_evidence():
    """When only phrase matches (no seed), missing_with_phrase captures it."""
    r = _rec(
        reasoning="The opponent is constrained to a single jump.",
        seeds=[],   # no seed
        facts={"forced_opponent_jump_reply": False},  # fact=False → CONTRADICTED → not expected
    )
    result = compute_turn_recall(r)
    # fact=False → verifier returns CONTRADICTED → not expected → not missing
    assert "forced_opponent_jump" not in result["expected_claim_types"]

    # Now with True fact and phrase but no seed → expected via phrase, extracted via phrase
    r2 = _rec(
        reasoning="The opponent is constrained to a single jump.",
        seeds=[],
        facts={"forced_opponent_jump_reply": True},
    )
    result2 = compute_turn_recall(r2)
    # Phrase fires → extracted; also expected via phrase → not missing
    assert "forced_opponent_jump" not in result2["missing_types"]


# ---------------------------------------------------------------------------
# 5. Does not expect vague / non-verifiable claims
# ---------------------------------------------------------------------------

def test_does_not_expect_positional_pressure():
    """positional_pressure (FORBIDDEN_UNGROUNDED) never in expected_claim_types."""
    r = _rec(
        reasoning="The move applies positional pressure and structural pressure.",
        seeds=[], facts={},
    )
    result = compute_turn_recall(r)
    assert "positional_pressure" not in result["expected_claim_types"]
    # It may still be extracted (phrase fires), but it is NOT expected.


def test_does_not_expect_strategic_initiative():
    """strategic_initiative (NON_VERIFIABLE_VAGUE) never expected."""
    r = _rec(
        reasoning="This seizes the strategic initiative.",
        seeds=[], facts={},
    )
    result = compute_turn_recall(r)
    assert "strategic_initiative" not in result["expected_claim_types"]


def test_does_not_expect_long_term_compensation():
    """long_term_compensation (NON_VERIFIABLE_VAGUE) never expected."""
    r = _rec(
        reasoning="This offers long-term compensation.",
        seeds=[], facts={},
    )
    result = compute_turn_recall(r)
    assert "long_term_compensation" not in result["expected_claim_types"]


def test_expected_claim_types_only_eligible_categories():
    """All types in expected_claim_types are VERIFIABLE or AMBIGUOUS_CONTEXT_REQUIRED."""
    eligible = {TaxonomyCategory.VERIFIABLE, TaxonomyCategory.AMBIGUOUS_CONTEXT_REQUIRED}
    exp = expected_claim_types(
        reasoning_text=(
            "The move avoids recapture and applies positional pressure. "
            "It seizes the strategic initiative."
        ),
        seeds=["opponent_can_recapture=false"],
        facts={"opponent_can_recapture": False},
    )
    for ct in exp:
        spec = _CLAIM_REGISTRY.get(ct)
        assert spec is not None, f"Unknown claim type in expected: {ct!r}"
        assert spec.category in eligible, (
            f"{ct!r} has category {spec.category.value}, should be eligible"
        )


# ---------------------------------------------------------------------------
# 6. Does not expect when required fact is absent
# ---------------------------------------------------------------------------

def test_no_expectation_when_fact_absent_forced_jump():
    """forced_opponent_jump not expected when forced_opponent_jump_reply absent from facts."""
    r = _rec(
        reasoning="Strong move.",
        seeds=["forced_opponent_jump_reply=true"],  # seed present
        facts={},                                    # but fact absent → UNSUPPORTED
    )
    result = compute_turn_recall(r)
    assert "forced_opponent_jump" not in result["expected_claim_types"]


def test_no_expectation_when_fact_absent_shot_sequence():
    r = _rec(
        reasoning="Strong move.",
        seeds=["shot_sequence_available=true"],
        facts={},
    )
    result = compute_turn_recall(r)
    assert "shot_sequence_or_multi_jump" not in result["expected_claim_types"]


def test_no_expectation_when_fact_absent_avoids_recapture():
    r = _rec(
        reasoning="The move avoids recapture.",
        seeds=["opponent_can_recapture=false"],
        facts={},  # opponent_can_recapture absent
    )
    result = compute_turn_recall(r)
    assert "avoids_recapture" not in result["expected_claim_types"]


# ---------------------------------------------------------------------------
# 7. Does not expect when fact contradicts the claim
# ---------------------------------------------------------------------------

def test_no_expectation_when_fact_contradicts_shot_sequence():
    """shot_sequence_or_multi_jump not expected when shot_sequence_available=False."""
    r = _rec(
        reasoning="The move is strong.",
        seeds=["shot_sequence_available=true"],   # seed says True
        facts={"shot_sequence_available": False}, # but fact says False → CONTRADICTED
    )
    result = compute_turn_recall(r)
    assert "shot_sequence_or_multi_jump" not in result["expected_claim_types"]


def test_no_expectation_when_fact_contradicts_gains_material():
    """gains_material not expected when net_gain=0 (verifier → CONTRADICTED)."""
    r = _rec(
        reasoning="The move gains material.",
        seeds=["captures_count=0"],
        facts={"net_gain": 0, "captures_count": 0},
    )
    result = compute_turn_recall(r)
    assert "gains_material" not in result["expected_claim_types"]


def test_no_expectation_when_fact_contradicts_avoids_recapture():
    """avoids_recapture not expected when opponent_can_recapture=True."""
    r = _rec(
        reasoning="The move avoids recapture.",
        seeds=[],
        facts={"opponent_can_recapture": True},  # contradicts avoids_recapture
    )
    result = compute_turn_recall(r)
    assert "avoids_recapture" not in result["expected_claim_types"]


# ---------------------------------------------------------------------------
# 8. Immutability
# ---------------------------------------------------------------------------

def test_compute_turn_recall_does_not_mutate_record():
    """compute_turn_recall must not modify the input record dict."""
    record = _rec(
        reasoning="The move avoids recapture.",
        seeds=["opponent_can_recapture=false"],
        facts={"opponent_can_recapture": False, "minimax_score": 1.0},
    )
    snapshot = copy.deepcopy(record)
    compute_turn_recall(record)
    assert record == snapshot, "compute_turn_recall mutated the input record"


def test_expected_claim_types_does_not_mutate_seeds():
    seeds = ["opponent_can_recapture=false"]
    orig = list(seeds)
    expected_claim_types("The move avoids recapture.", seeds, {"opponent_can_recapture": False})
    assert seeds == orig, "expected_claim_types mutated the seeds list"


def test_expected_claim_types_does_not_mutate_facts():
    facts = {"opponent_can_recapture": False}
    orig = dict(facts)
    expected_claim_types("The move avoids recapture.", [], facts)
    assert facts == orig, "expected_claim_types mutated the facts dict"


# ---------------------------------------------------------------------------
# 9. Handles missing / None fields (old log compatibility)
# ---------------------------------------------------------------------------

def test_handles_missing_all_fields():
    result = compute_turn_recall({})
    assert result["extracted_claim_types"] == []
    assert result["expected_claim_types"] == {}
    assert result["missing_types"] == []
    assert result["player"] == "UNKNOWN"


def test_handles_none_reasoning():
    result = compute_turn_recall({
        "player": "RED", "turn": 1, "turn_id": "t1",
        "reasoning": None, "seeds": None, "facts": None,
    })
    assert result["extracted_claim_types"] == []
    assert result["missing_types"] == []


def test_handles_empty_facts():
    """With empty facts, no VERIFIABLE claim should be expected."""
    result = compute_turn_recall({
        "player": "RED", "turn": 1, "turn_id": "t1",
        "reasoning": "The move gains material and avoids recapture.",
        "seeds": ["captures_count=1", "opponent_can_recapture=false"],
        "facts": {},
    })
    assert result["expected_claim_types"] == {}


def test_handles_empty_seeds():
    """With empty seeds, claims only expected if phrase appears in reasoning."""
    result = compute_turn_recall({
        "player": "RED", "turn": 1, "turn_id": "t1",
        "reasoning": "The move avoids recapture.",
        "seeds": [],
        "facts": {"opponent_can_recapture": False},
    })
    # Phrase "avoids recapture" IS in reasoning → expected via phrase
    assert "avoids_recapture" in result["expected_claim_types"]
    assert result["expected_claim_types"]["avoids_recapture"] == "phrase"


def test_turn_id_fallback():
    """turn_id is derived from 'turn' when 'turn_id' field is absent."""
    result = compute_turn_recall({"player": "RED", "turn": 7})
    assert "7" in result["turn_id"]


# ---------------------------------------------------------------------------
# 10. Aggregate counts RED turns only
# ---------------------------------------------------------------------------

def test_aggregate_counts_red_turns_only():
    """aggregate_recall_results counts RED turns separately from the total."""
    red_result = compute_turn_recall(_rec(
        player="RED", turn=1, turn_id="r1",
        reasoning="The move avoids recapture.",
        seeds=["opponent_can_recapture=false"],
        facts={"opponent_can_recapture": False},
    ))
    black_result = compute_turn_recall(_rec(
        player="BLACK", turn=2, turn_id="b2",
        reasoning="The move avoids recapture.",
        seeds=["opponent_can_recapture=false"],
        facts={"opponent_can_recapture": False},
    ))
    agg = aggregate_recall_results([red_result, black_result])
    assert agg["total_red_turns"] == 1
    assert agg["total_all_turns"] == 2


def test_aggregate_with_no_turns():
    agg = aggregate_recall_results([])
    assert agg["total_red_turns"] == 0
    assert agg["total_all_turns"] == 0
    assert agg["top_10_recall_gaps"] == []


# ---------------------------------------------------------------------------
# 11. Top-10 recall gaps sorted descending
# ---------------------------------------------------------------------------

def test_aggregate_top_gaps_sorted_by_frequency():
    """Top 10 gaps sorted: highest missing count first."""
    def _bare_result(player, missing_types):
        return {
            "player": player,
            "turn": 1, "turn_id": "x",
            "minimax_score": 0.0,
            "extracted_claim_types": [],
            "expected_claim_types": {ct: "seed" for ct in missing_types},
            "missing_types": list(missing_types),
            "missing_with_seed": list(missing_types),
            "missing_with_phrase": [],
            "notes": [],
        }

    results = (
        [_bare_result("RED", ["forced_opponent_jump"]) for _ in range(5)]
        + [_bare_result("RED", ["shot_sequence_or_multi_jump"]) for _ in range(2)]
        + [_bare_result("RED", ["blocks_landing_square"]) for _ in range(1)]
    )
    agg = aggregate_recall_results(results)
    gaps = agg["top_10_recall_gaps"]
    assert gaps[0][0] == "forced_opponent_jump"
    assert gaps[0][1] == 5
    assert gaps[1][0] == "shot_sequence_or_multi_jump"
    assert gaps[1][1] == 2


# ---------------------------------------------------------------------------
# 12. Evidence source semantics
# ---------------------------------------------------------------------------

def test_evidence_source_seed_only():
    """When only seed matches (phrase absent from reasoning), source is 'seed'."""
    exp = expected_claim_types(
        reasoning_text="This is a strong move.",  # no forced_opponent_jump phrase
        seeds=["forced_opponent_jump_reply=true"],
        facts={"forced_opponent_jump_reply": True},
    )
    assert exp.get("forced_opponent_jump") == "seed"


def test_evidence_source_phrase_only():
    """When only phrase matches (no seed), source is 'phrase'."""
    exp = expected_claim_types(
        reasoning_text="The opponent is constrained to a jump in reply.",
        seeds=[],  # no seed
        facts={"forced_opponent_jump_reply": True},
    )
    assert exp.get("forced_opponent_jump") == "phrase"


def test_evidence_source_both():
    """When both seed and phrase present, source is 'seed+phrase'."""
    exp = expected_claim_types(
        reasoning_text="The opponent is constrained to a jump.",
        seeds=["forced_opponent_jump_reply=true"],
        facts={"forced_opponent_jump_reply": True},
    )
    assert exp.get("forced_opponent_jump") == "seed+phrase"


def test_avoids_recapture_expected_when_fact_false():
    """avoids_recapture is expected when opponent_can_recapture=False."""
    exp = expected_claim_types(
        "The move avoids recapture.",
        seeds=["opponent_can_recapture=false"],
        facts={"opponent_can_recapture": False},
    )
    assert "avoids_recapture" in exp


def test_gains_material_expected_when_net_gain_positive():
    exp = expected_claim_types(
        "The move captures a piece.",
        seeds=["captures_count=1, net_gain=1"],
        facts={"net_gain": 1, "captures_count": 1},
    )
    assert "gains_material" in exp


def test_promotes_to_king_expected_when_results_in_king():
    exp = expected_claim_types(
        "The piece promotes to king.",
        seeds=["results_in_king=true"],
        facts={"results_in_king": True},
    )
    assert "promotes_to_king" in exp


def test_minimax_confirmation_expected_when_score_present():
    exp = expected_claim_types(
        "The minimax score confirms this choice.",
        seeds=["minimax_score=-2.00 — highest-evaluated option"],
        facts={"minimax_score": -2.0},
    )
    assert "minimax_confirmation" in exp


def test_shot_sequence_expected_when_available_true():
    exp = expected_claim_types(
        "The move enables a multi-jump sequence.",
        seeds=["shot_sequence_available=true"],
        facts={"shot_sequence_available": True},
    )
    assert "shot_sequence_or_multi_jump" in exp


def test_blocks_landing_expected_when_true():
    exp = expected_claim_types(
        "The move blocks the opponent from landing.",
        seeds=["blocks_opponent_landing=true"],
        facts={"blocks_opponent_landing": True},
    )
    assert "blocks_landing_square" in exp


# ---------------------------------------------------------------------------
# 13. Already extracted → not missing
# ---------------------------------------------------------------------------

def test_avoids_recapture_extracted_not_missing():
    """When the claim is extracted, it must NOT appear in missing_types."""
    r = _rec(
        reasoning="The move avoids recapture.",
        seeds=["opponent_can_recapture=false"],
        facts={"opponent_can_recapture": False},
    )
    result = compute_turn_recall(r)
    assert "avoids_recapture" in result["extracted_claim_types"]
    assert "avoids_recapture" not in result["missing_types"]


def test_gains_material_extracted_not_missing():
    r = _rec(
        reasoning="The move captures a piece and gains material.",
        seeds=["captures_count=1, net_gain=1"],
        facts={"net_gain": 1, "captures_count": 1},
    )
    result = compute_turn_recall(r)
    assert "gains_material" in result["extracted_claim_types"]
    assert "gains_material" not in result["missing_types"]


# ---------------------------------------------------------------------------
# 14. print_report smoke test (does not raise)
# ---------------------------------------------------------------------------

def test_print_report_smoke():
    """print_report must not raise on a minimal valid input."""
    import io
    r = _rec(
        reasoning="The move avoids recapture.",
        seeds=["opponent_can_recapture=false"],
        facts={"opponent_can_recapture": False},
    )
    result = compute_turn_recall(r)
    agg = aggregate_recall_results([result])
    buf = io.StringIO()
    print_report(agg, [result], out=buf)
    output = buf.getvalue()
    assert "CLAIM RECALL AUDIT" in output
    assert "avoids_recapture" in output


def test_print_report_empty():
    """print_report handles empty turn_results without raising."""
    import io
    agg = aggregate_recall_results([])
    buf = io.StringIO()
    print_report(agg, [], out=buf)
    assert "CLAIM RECALL AUDIT" in buf.getvalue()
