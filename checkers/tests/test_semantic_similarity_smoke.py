# checkers/tests/test_semantic_similarity_smoke.py
#
# Smoke tests for the semantic-similarity metric layer.
#
# Two test tiers:
#
#   LIGHT (default, always runs):
#     Tests the per-turn / aggregator wiring, the in-memory caches, empty
#     and scoring-failed handling, dataclass shapes, and report-field
#     presence USING STUB scoring functions.  No transformer weights are
#     loaded.  These tests guarantee the rest of the pipeline keeps
#     working even when neither bert_score nor bleurt-pytorch is installed
#     on the runner.
#
#   HEAVY (gated behind `pytest -m semantic`):
#     Loads `bert_score` and `bleurt-pytorch` for real and checks
#     determinism, identical-input saturation, and chess-domain
#     calibration.  Backends are skipped INDIVIDUALLY if their library
#     is missing — BERTScore tests still run with BLEURT uninstalled and
#     vice-versa.
#
# All tests are deterministic and isolated.  Nothing touches factuality
# metrics, the runtime refinement loop, or any other evaluator module.

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from checkers.evaluation.metrics import semantic_similarity as ss


# ─────────────────────────────────────────────────────────────────────────────
# LIGHT tests — no transformer weights
# ─────────────────────────────────────────────────────────────────────────────

def _record(
    *,
    turn_id: str,
    pre: Optional[str],
    post: Optional[str],
    refinement_attempts: int = 0,
    contradiction_detected: bool = False,
) -> Dict[str, Any]:
    diag: Dict[str, Any] = {
        "raw_llm_reasoning_pre_refinement":    pre,
        "reasoning_refinement_retry_count":    refinement_attempts,
        "reasoning_contradiction_detected":    contradiction_detected,
    }
    return {
        "turn_id":             turn_id,
        "last_move_reasoning": post,
        "ranker_diagnostics":  diag,
        "chosen_move_facts":   {},
    }


def _stub_score(returns: Optional[float]):
    """Deterministic stub for the LIGHT tier."""
    calls: List[Any] = []

    def _fn(a: str, b: str):
        calls.append((a, b))
        return returns

    _fn.calls = calls  # type: ignore[attr-defined]
    return _fn


# ── eligibility gates ──────────────────────────────────────────────────────

def test_skip_when_pre_missing():
    rec = _record(turn_id="t1", pre=None, post="abc", refinement_attempts=1,
                  contradiction_detected=True)
    t = ss.evaluate_semantic(rec, bertscore_fn=_stub_score(0.9), bleurt_fn=_stub_score(0.4))
    assert t.refined is False and t.skipped_empty is True
    assert t.bertscore_f1 is None and t.bleurt is None
    assert t.scoring_failed is False


def test_skip_when_post_missing():
    rec = _record(turn_id="t2", pre="abc", post="", refinement_attempts=1,
                  contradiction_detected=True)
    t = ss.evaluate_semantic(rec, bertscore_fn=_stub_score(0.9), bleurt_fn=_stub_score(0.4))
    assert t.skipped_empty is True


def test_skip_when_refinement_did_not_run():
    rec = _record(turn_id="t3", pre="aaa", post="bbb", refinement_attempts=0,
                  contradiction_detected=False)
    t = ss.evaluate_semantic(rec, bertscore_fn=_stub_score(0.9), bleurt_fn=_stub_score(0.4))
    assert t.refined is False
    assert t.skipped_empty is False
    assert t.scoring_failed is False


def test_skip_when_pre_equals_post():
    rec = _record(turn_id="t4", pre="same", post="same",
                  refinement_attempts=1, contradiction_detected=True)
    t = ss.evaluate_semantic(rec, bertscore_fn=_stub_score(0.9), bleurt_fn=_stub_score(0.4))
    assert t.refined is False
    assert t.scoring_failed is False


# ── scoring on eligible turns ──────────────────────────────────────────────

def test_score_counted_when_refined_and_changed():
    rec = _record(turn_id="t5", pre="original text",
                  post="rewritten text", refinement_attempts=1,
                  contradiction_detected=True)
    t = ss.evaluate_semantic(
        rec, bertscore_fn=_stub_score(0.87), bleurt_fn=_stub_score(0.42),
    )
    assert t.refined is True and t.skipped_empty is False and t.scoring_failed is False
    assert t.bertscore_f1 == pytest.approx(0.87)
    assert t.bleurt       == pytest.approx(0.42)


def test_scoring_failed_when_both_backends_return_none():
    """Eligible turn but every backend returned None — must be flagged."""
    rec = _record(turn_id="t6", pre="aa", post="bb",
                  refinement_attempts=1, contradiction_detected=True)
    t = ss.evaluate_semantic(
        rec, bertscore_fn=_stub_score(None), bleurt_fn=_stub_score(None),
    )
    assert t.refined is False
    assert t.skipped_empty is False
    assert t.scoring_failed is True
    assert t.bertscore_f1 is None and t.bleurt is None


def test_partial_score_still_refined():
    """If one backend returns a value and the other returns None, the turn
    counts as refined and its available score contributes to its bucket."""
    rec = _record(turn_id="t7", pre="aa", post="bb",
                  refinement_attempts=1, contradiction_detected=True)
    t = ss.evaluate_semantic(
        rec, bertscore_fn=_stub_score(0.91), bleurt_fn=_stub_score(None),
    )
    assert t.refined is True
    assert t.scoring_failed is False
    assert t.bertscore_f1 == pytest.approx(0.91)
    assert t.bleurt is None


# ── aggregator counters and stats ──────────────────────────────────────────

def test_aggregate_counters_disjoint():
    turns = [
        ss.SemanticPairTurn(turn_id="a", refined=True,  skipped_empty=False, bertscore_f1=0.90, bleurt=0.40),
        ss.SemanticPairTurn(turn_id="b", refined=True,  skipped_empty=False, bertscore_f1=0.80, bleurt=0.30),
        ss.SemanticPairTurn(turn_id="c", refined=True,  skipped_empty=False, bertscore_f1=0.70, bleurt=0.20),
        ss.SemanticPairTurn(turn_id="d", refined=False, skipped_empty=False),                                      # unchanged
        ss.SemanticPairTurn(turn_id="e", refined=False, skipped_empty=True),                                       # empty
        ss.SemanticPairTurn(turn_id="f", refined=False, skipped_empty=False, scoring_failed=True),                 # failed
    ]
    s = ss.aggregate_semantic(turns)
    assert s.n_turns_total     == 6
    assert s.n_turns_refined   == 3
    assert s.n_turns_unchanged == 1
    assert s.n_skipped_empty   == 1
    assert s.n_scoring_failed  == 1
    # Counters partition the total.
    assert (s.n_turns_refined + s.n_turns_unchanged
            + s.n_skipped_empty + s.n_scoring_failed) == s.n_turns_total
    # BERTScore stats over the 3 refined turns.
    assert s.bertscore_f1.n == 3
    assert s.bertscore_f1.mean == pytest.approx(0.80)
    # BLEURT stats over the same 3 refined turns.
    assert s.bleurt.n == 3
    assert s.bleurt.mean == pytest.approx(0.30)


def test_aggregate_per_backend_independence():
    """If one backend is unavailable for half the corpus, its n shrinks
    independently of the other backend's n."""
    turns = [
        ss.SemanticPairTurn(turn_id="a", refined=True, skipped_empty=False, bertscore_f1=0.90, bleurt=None),
        ss.SemanticPairTurn(turn_id="b", refined=True, skipped_empty=False, bertscore_f1=0.80, bleurt=0.30),
    ]
    s = ss.aggregate_semantic(turns)
    assert s.n_turns_refined == 2
    assert s.bertscore_f1.n == 2
    assert s.bleurt.n == 1
    assert s.bleurt.mean == pytest.approx(0.30)


def test_aggregate_empty_corpus_returns_safe_summary():
    s = ss.aggregate_semantic([])
    assert s.n_turns_total == 0
    for k in ("mean", "median", "p10", "iqr"):
        assert getattr(s.bertscore_f1, k) is None
        assert getattr(s.bleurt, k) is None
    # model_versions always present (even when backends missing).
    assert "bertscore_model" in s.model_versions
    assert "bertscore_lib"   in s.model_versions
    assert "bleurt_model"    in s.model_versions
    assert "bleurt_lib"      in s.model_versions


def test_model_versions_always_present():
    versions = ss.model_versions()
    assert versions["bertscore_model"] == ss.BERTSCORE_MODEL
    assert versions["bertscore_lib"].startswith("bert_score==")
    assert versions["bleurt_model"]    == ss.BLEURT_CHECKPOINT
    assert versions["bleurt_lib"].startswith("bleurt-pytorch==")


# ── score_pair input validation ────────────────────────────────────────────

def test_score_pair_bertscore_handles_non_string_inputs():
    assert ss.score_pair_bertscore(None, "abc") is None  # type: ignore[arg-type]
    assert ss.score_pair_bertscore("abc", None) is None  # type: ignore[arg-type]
    assert ss.score_pair_bertscore("", "abc") is None
    assert ss.score_pair_bertscore("abc", "  ") is None


def test_score_pair_bleurt_handles_non_string_inputs():
    assert ss.score_pair_bleurt(None, "abc") is None  # type: ignore[arg-type]
    assert ss.score_pair_bleurt("abc", None) is None  # type: ignore[arg-type]
    assert ss.score_pair_bleurt("", "abc") is None
    assert ss.score_pair_bleurt("abc", "  ") is None


# ── end-to-end LIGHT report block shape ────────────────────────────────────

def test_report_json_block_shape_via_stub():
    """Simulate run_batch building a `semantic` block."""
    recs = [
        _record(turn_id="t1", pre="aa", post="bb", refinement_attempts=1, contradiction_detected=True),
        _record(turn_id="t2", pre="x",  post="x",  refinement_attempts=1, contradiction_detected=True),
        _record(turn_id="t3", pre=None, post="zz", refinement_attempts=1, contradiction_detected=True),
    ]
    _turns, summary = ss.evaluate_records(
        recs, bertscore_fn=_stub_score(0.92), bleurt_fn=_stub_score(0.38),
    )
    assert len(_turns) == 3
    block = summary.to_dict()
    for key in (
        "n_turns_total", "n_turns_refined", "n_turns_unchanged",
        "n_skipped_empty", "n_scoring_failed",
        "bertscore_f1", "bleurt", "model_versions", "backend_errors",
    ):
        assert key in block, f"missing semantic-block key: {key}"
    for stats_key in ("bertscore_f1", "bleurt"):
        for sub in ("n", "mean", "median", "p10", "iqr"):
            assert sub in block[stats_key]
    for vkey in ("bertscore_model", "bertscore_lib", "bleurt_model", "bleurt_lib"):
        assert vkey in block["model_versions"]


# ─────────────────────────────────────────────────────────────────────────────
# HEAVY tests — require backends; run with `pytest -m semantic`
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def bertscore_backend():
    """Skip BERTScore-only heavy tests cleanly when bert_score is missing."""
    try:
        ss._resolve_bert_score()
    except ss.SemanticDependencyMissing as e:
        pytest.skip(f"bert_score not installed: {e}")
    ss._BS_CACHE.clear()
    return True


@pytest.fixture(scope="session")
def bleurt_backend():
    """Skip BLEURT-only heavy tests cleanly when bleurt-pytorch is missing."""
    try:
        ss._resolve_bleurt()
    except ss.SemanticDependencyMissing as e:
        pytest.skip(f"bleurt-pytorch not installed: {e}")
    ss._BL_CACHE.clear()
    return True


# ── BERTScore heavy tier (preserved from previous calibration) ─────────────

@pytest.mark.semantic
def test_bertscore_identical_input_saturates(bertscore_backend):
    text = "The move captures one piece with no recapture risk."
    f1 = ss.score_pair_bertscore(text, text)
    assert f1 is not None and f1 > 0.999


@pytest.mark.semantic
def test_bertscore_paraphrase_above_baseline(bertscore_backend):
    """
    Chess-prose paraphrases score WELL ABOVE the empirical baseline but FAR
    BELOW saturation when scored with `rescale_with_baseline=True`.  Bound
    is intentionally modest (> 0.20) — it detects a broken backend, not a
    paraphrase-quality bar the metric cannot meet on chess prose.
    """
    a = "The move captures one piece without exposing our pieces."
    b = "The move wins a piece while keeping our side safe."
    f1 = ss.score_pair_bertscore(a, b)
    assert f1 is not None and f1 > 0.20


@pytest.mark.semantic
def test_bertscore_unrelated_near_baseline(bertscore_backend):
    """Unrelated text scores near the rescaled baseline.  Bound < 0.30."""
    a = "The move captures one piece and improves mobility."
    b = "Bicycles are an efficient way to commute in dense cities."
    f1 = ss.score_pair_bertscore(a, b)
    assert f1 is not None and f1 < 0.30


@pytest.mark.semantic
def test_bertscore_paraphrase_beats_unrelated(bertscore_backend):
    """Relative-ordering check — domain-robust signal."""
    pa = "The move captures one piece without exposing our pieces."
    pb = "The move wins a piece while keeping our side safe."
    ua = "The move captures one piece and improves mobility."
    ub = "Bicycles are an efficient way to commute in dense cities."
    para = ss.score_pair_bertscore(pa, pb)
    unrel = ss.score_pair_bertscore(ua, ub)
    assert para is not None and unrel is not None
    assert para > unrel + 0.10


@pytest.mark.semantic
def test_bertscore_determinism(bertscore_backend):
    a = "The piece advances to the centre without capturing."
    b = "The piece moves forward to a central square without taking material."
    ss._BS_CACHE.clear()
    s1 = ss.score_pair_bertscore(a, b)
    ss._BS_CACHE.clear()
    s2 = ss.score_pair_bertscore(a, b)
    assert s1 == s2


@pytest.mark.semantic
def test_bertscore_cache_hits(bertscore_backend):
    a = "Cache hit BERTScore sentence one."
    b = "Cache hit BERTScore sentence two."
    ss._BS_CACHE.clear()
    _ = ss.score_pair_bertscore(a, b)
    n = len(ss._BS_CACHE)
    _ = ss.score_pair_bertscore(a, b)
    assert len(ss._BS_CACHE) == n


# ── BLEURT heavy tier ──────────────────────────────────────────────────────

@pytest.mark.semantic
def test_bleurt_identical_input_high(bleurt_backend):
    """
    BLEURT-20-D12 is a regression model; it does NOT saturate at exactly 1.0
    even on identical input.  Empirically scores around 0.9–1.05 on
    English sentences and slightly higher than ANY non-identical pair.
    Threshold > 0.80 is conservative for chess prose.
    """
    text = "The move captures one piece with no recapture risk."
    s = ss.score_pair_bleurt(text, text)
    assert s is not None, "BLEURT returned None on identical input"
    assert s > 0.80, f"BLEURT identical score={s!r} unexpectedly low"


@pytest.mark.semantic
def test_bleurt_paraphrase_above_unrelated(bleurt_backend):
    """
    BLEURT is out-of-domain for chess prose, so absolute scores are noisy.
    The domain-robust check is RELATIVE ORDERING: a chess paraphrase pair
    must score above an unrelated cross-domain pair.
    """
    pa = "The move captures one piece without exposing our pieces."
    pb = "The move wins a piece while keeping our side safe."
    ua = "The move captures one piece and improves mobility."
    ub = "Bicycles are an efficient way to commute in dense cities."
    p = ss.score_pair_bleurt(pa, pb)
    u = ss.score_pair_bleurt(ua, ub)
    assert p is not None and u is not None
    assert p > u, (
        f"BLEURT failed relative ordering — paraphrase={p!r}, unrelated={u!r}"
    )


@pytest.mark.semantic
def test_bleurt_determinism(bleurt_backend):
    a = "The piece advances to the centre without capturing."
    b = "The piece moves forward to a central square without taking material."
    ss._BL_CACHE.clear()
    s1 = ss.score_pair_bleurt(a, b)
    ss._BL_CACHE.clear()
    s2 = ss.score_pair_bleurt(a, b)
    assert s1 == s2


@pytest.mark.semantic
def test_bleurt_cache_hits(bleurt_backend):
    a = "Cache hit BLEURT sentence one."
    b = "Cache hit BLEURT sentence two."
    ss._BL_CACHE.clear()
    _ = ss.score_pair_bleurt(a, b)
    n = len(ss._BL_CACHE)
    _ = ss.score_pair_bleurt(a, b)
    assert len(ss._BL_CACHE) == n
