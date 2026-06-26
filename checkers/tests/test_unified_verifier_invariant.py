# checkers/tests/test_unified_verifier_invariant.py
#
# Hard invariant for the E.1 unification: the runtime refinement loop and the
# evaluator metric layer MUST agree on whether a reasoning string contains a
# CONTRADICTED claim.
#
# Concretely, for every test reasoning:
#   _check_reasoning_truthfulness(...) returns [] (empty list)
#     ⇔
#   unified_verifier.contradictions_only(...) returns [] (empty list)
#
# Asymmetry between these two MUST fail this test.  When runtime/evaluator
# drift again in the future, the test fires immediately and points at the
# specific reasoning string where they diverge.

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from checkers.agents.explainer_agent import _check_reasoning_truthfulness
from checkers.evaluation.unified_verifier import (
    assert_runtime_evaluator_agreement,
    RuntimeEvaluatorDisagreement,
    contradictions_only,
)


# ── Synthetic cases ─────────────────────────────────────────────────────────
# (label, reasoning, seeds, facts) — chosen to span the audit failure modes
# (E.2 negation, E.3 numeric, E.4 schema leak) plus the easy positives.
SYNTHETIC_CASES = [
    # Clean reasoning — both sides should be empty.
    (
        "clean_positive",
        "The move advances a piece to the center without capturing. "
        "The opponent cannot recapture next turn.",
        [],
        {
            "captures_count": 0, "net_gain": 0,
            "opponent_can_recapture": False, "center_control": True,
            "creates_immediate_threat": False, "leaves_piece_isolated": False,
            "minimax_score": -2.0,
        },
    ),
    # E.2 false-positive killers — must NOT flag a contradiction either side.
    (
        "e2_but_not_here",
        "the opponent can recapture after the other move but not here",
        [],
        {"opponent_can_recapture": False},
    ),
    (
        "e2_over_material",
        "without capturing, prioritizing piece placement over material gain",
        [],
        {"captures_count": 0, "net_gain": 0},
    ),
    # Real factual contradictions — BOTH sides must flag.
    (
        "real_recapture_contra",
        "The move avoids recapture and stays safe.",
        ["opponent_can_recapture=true — opponent can recapture next turn"],
        {"opponent_can_recapture": True},
    ),
    (
        "real_capture_contra",
        "The move captures a piece for a net gain of 1.",
        [],
        {"captures_count": 0, "net_gain": 0},
    ),
    # E.3 numeric fabrication — must flag.
    (
        "e3_numeric_capture_count",
        "The move captures three pieces for a net gain of three.",
        [],
        {"captures_count": 1, "net_gain": 1},
    ),
    (
        "e3_numeric_minimax",
        "The minimax_score=999 confirms this move.",
        [],
        {"minimax_score": -2.0},
    ),
    (
        "e3_numeric_mobility_transition",
        "Our mobility increases from 10 to 11 after the move.",
        [],
        {
            "our_mobility_before": 7, "our_mobility_after": 7,
            "opponent_mobility_before": 9, "opponent_mobility_after": 9,
        },
    ),
    # E.4 schema-leak with disagreement — must flag.
    (
        "e4_schema_leak_contra",
        "The position creates_immediate_threat=true after this move.",
        [],
        {"creates_immediate_threat": False, "shot_sequence_available": False},
    ),
    # E.4 schema-leak with agreement — raw schema string is an instruction
    # violation regardless of whether the value agrees with the fact.
    # Both runtime and evaluator must report a contradiction.
    (
        "e4_schema_leak_agree",
        "The fact opponent_can_recapture=false implies safety.",
        [],
        {"opponent_can_recapture": False},
    ),
]


@pytest.mark.parametrize("label,reasoning,seeds,facts", SYNTHETIC_CASES)
def test_runtime_evaluator_agree(label, reasoning, seeds, facts):
    """Runtime and evaluator must agree on whether `reasoning` contradicts."""
    runtime = _check_reasoning_truthfulness(reasoning, facts, seeds=seeds)
    # If either side flags, both must.
    assert_runtime_evaluator_agreement(
        runtime, reasoning, reasoning_seeds=seeds, facts=facts,
    )


def test_invariant_fires_on_drift():
    """Sanity: a hand-crafted disagreement raises RuntimeEvaluatorDisagreement."""
    with pytest.raises(RuntimeEvaluatorDisagreement):
        assert_runtime_evaluator_agreement(
            runtime_contradictions=[],
            reasoning_text="The move captures three pieces for a net gain of three.",
            facts={"captures_count": 1, "net_gain": 1},
        )


# ── Real-data agreement scan ────────────────────────────────────────────────
# Optional pass over the latest ablation logs (skipped if absent).  This
# proves the invariant holds on actual production traces, not just the
# synthetic vector.

REAL_LOG_DIRS = [
    Path("logs/ablation_debug/evaluation_source/seed_on"),
    Path("logs/ablation_debug/evaluation_source/seed_off"),
    Path("logs/ablation_run_001/evaluation_source/seed_on"),
    Path("logs/ablation_run_001/evaluation_source/seed_off"),
    Path("logs/ablation_no_fallback/evaluation_source/seed_on"),
    Path("logs/ablation_no_fallback/evaluation_source/seed_off"),
]


def _iter_records():
    import json
    for d in REAL_LOG_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def test_runtime_evaluator_agree_on_real_logs():
    """Agreement check over every available evaluation_source record."""
    n_checked = 0
    for rec in _iter_records():
        reasoning = rec.get("last_move_reasoning") or ""
        diag = rec.get("ranker_diagnostics") or {}
        seeds = [s for s in (diag.get("reasoning_seeds") or []) if isinstance(s, str)]
        facts = rec.get("chosen_move_facts") or {}
        if not isinstance(reasoning, str) or not reasoning.strip():
            continue
        runtime = _check_reasoning_truthfulness(reasoning, facts, seeds=seeds)
        try:
            assert_runtime_evaluator_agreement(
                runtime, reasoning, reasoning_seeds=seeds, facts=facts,
            )
        except RuntimeEvaluatorDisagreement as e:
            pytest.fail(f"{rec.get('turn_id', '?')}: {e}")
        n_checked += 1
    if n_checked == 0:
        pytest.skip("no real evaluation_source records found")
    print(f"\n[invariant] checked {n_checked} real records — all agreed", file=sys.stderr)


# ── Metric-layer ↔ unified-verifier agreement (Fix 2) ───────────────────────
# These tests catch the bug class that allowed pre_post_repair.py and
# zero_claim.py to silently bypass the unified verifier and undercount
# contradictions by ~8 pp.  When any metric module switches back to
# extract_claims+verify_claims, these tests fail loudly.

from checkers.evaluation.unified_verifier import verify_all
from checkers.evaluation.metrics.pre_post_repair import _count_statuses as _ppr_count
from checkers.evaluation.metrics.zero_claim import evaluate_zero_claim
from checkers.evaluation.reasoning_taxonomy import ClaimStatus


def _verify_all_counts(reasoning, seeds, facts, context):
    """Reference counts taken straight from verify_all."""
    if not isinstance(reasoning, str) or not reasoning.strip():
        return (0, 0, 0, 0, 0)
    cs = verify_all(reasoning, reasoning_seeds=seeds, facts=facts, context=context)
    return (
        len(cs),
        sum(1 for c in cs if c.claim_status == ClaimStatus.SUPPORTED),
        sum(1 for c in cs if c.claim_status == ClaimStatus.CONTRADICTED),
        sum(1 for c in cs if c.claim_status == ClaimStatus.UNSUPPORTED),
        sum(1 for c in cs if c.claim_status == ClaimStatus.VAGUE),
    )


# --- pre_post_repair._count_statuses must equal verify_all counts ---
@pytest.mark.parametrize("label,reasoning,seeds,facts", SYNTHETIC_CASES)
def test_metric_pre_post_repair_matches_verify_all(label, reasoning, seeds, facts):
    nb = None
    ctx = {"next_best_minimax_score": nb} if isinstance(nb, (int, float)) else None
    ref = _verify_all_counts(reasoning, seeds, facts, ctx)
    got = _ppr_count(reasoning, seeds or [], facts or {}, ctx)
    assert (got.total, got.supported, got.contradicted, got.unsupported, got.vague) == ref, (
        f"pre_post_repair drift on {label!r}: got={got}, ref={ref}"
    )


def test_metric_pre_post_repair_matches_verify_all_on_real_logs():
    """Every real record's pre_post_repair counts must match verify_all."""
    n = 0
    for rec in _iter_records():
        reasoning = rec.get("last_move_reasoning") or ""
        if not reasoning.strip():
            continue
        diag = rec.get("ranker_diagnostics") or {}
        seeds = [s for s in (diag.get("reasoning_seeds") or []) if isinstance(s, str)]
        facts = rec.get("chosen_move_facts") or {}
        nb = diag.get("next_best_minimax_score")
        ctx = {"next_best_minimax_score": nb} if isinstance(nb, (int, float)) else None
        ref = _verify_all_counts(reasoning, seeds, facts, ctx)
        got = _ppr_count(reasoning, seeds, facts, ctx)
        if (got.total, got.supported, got.contradicted, got.unsupported, got.vague) != ref:
            pytest.fail(
                f"pre_post_repair counts drift from verify_all on "
                f"{rec.get('turn_id', '?')}: got={got}, ref={ref}"
            )
        n += 1
    if n == 0:
        pytest.skip("no real records")


# --- zero_claim per-turn aggregate must reconcile with verify_all over sentences ---
def test_metric_zero_claim_matches_verify_all_on_real_logs():
    """zero_claim's per-sentence claim counts must equal verify_all summed."""
    import re as _re
    n = 0
    for rec in _iter_records():
        reasoning = rec.get("last_move_reasoning") or ""
        if not reasoning.strip():
            continue
        diag = rec.get("ranker_diagnostics") or {}
        seeds = [s for s in (diag.get("reasoning_seeds") or []) if isinstance(s, str)]
        facts = rec.get("chosen_move_facts") or {}
        nb = diag.get("next_best_minimax_score")
        ctx = {"next_best_minimax_score": nb} if isinstance(nb, (int, float)) else None

        zc = evaluate_zero_claim(rec)
        # Reference: sum verify_all over the same sentence split zc uses.
        parts = [p.strip() for p in _re.split(r"(?<=[.!?])\s+", reasoning.strip())
                 if len(p.strip()) >= 3]
        ref_total = sum(
            len(verify_all(p, reasoning_seeds=seeds, facts=facts, context=ctx))
            for p in parts
        )
        if zc.total_claims != ref_total:
            pytest.fail(
                f"zero_claim total_claims drift from verify_all sentence sum "
                f"on {rec.get('turn_id', '?')}: zc={zc.total_claims}, ref={ref_total}"
            )
        n += 1
    if n == 0:
        pytest.skip("no real records")
