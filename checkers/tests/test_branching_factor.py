# checkers/tests/test_branching_factor.py
#
# Regression tests for the branching-factor analysis extension.
# Covers:
#   - _bf_bucket()                  — label assignment for all boundary values
#   - _build_branching_factor_breakdown() — correctness of all per-bucket fields
#   - summarize_results()           — branching_factor_breakdown key present
#   - filtering_meta                — structure only (main() internals)
#   - edge cases: missing legal_move_count, all-None coverage, 0 positions
#   - bucket ordering is deterministic (1 < 2 < ... < 11+)
#   - empty results → empty breakdown (no crash)
#   - graceful KR absence (kingsrow_best_coverage_pct = None)
#   - all existing summary keys unchanged by new additions
#   - no mutation side-effects on input results
#
# Fully offline — no LLM, no DLL, no network.

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from checkers.eval.proposal_seperation_eval import (
    _bf_bucket,
    _build_branching_factor_breakdown,
    summarize_results,
)


# ════════════════════════════════════════════════════════════════════════════════
# Minimal result-dict factory
# ════════════════════════════════════════════════════════════════════════════════

def _r(
    *,
    legal_count: int = 3,
    classification: str = "perfect",
    scanner_correct: bool = True,
    contains_engine: bool | None = True,
    contains_kr: bool | None = None,      # None simulates absent annotations
    category: str = "crowded_board",
    difficulty: str = "easy",
    api_failure: bool = False,
    parse_failure: bool = False,
) -> dict:
    return {
        "legal_move_count": legal_count,
        "classification": classification,
        "quadrant": (
            "scanner_correct_proposal_correct" if scanner_correct and classification == "perfect"
            else "scanner_wrong_proposal_wrong"
        ),
        "scanner_eval": {"scanner_correct": scanner_correct},
        "ground_truth_has_captures": False,
        "contains_engine_best": contains_engine,
        "contains_kingsrow_best": contains_kr,
        "category": category,
        "difficulty": difficulty,
        "api_failure": api_failure,
        "parse_failure": parse_failure,
        "proposal_classification": {
            "classification": classification,
            "legal_count": legal_count,
            "proposed_count": legal_count,
            "legal_proposed": legal_count,
            "illegal_proposed": 0,
            "missing_legal": 0,
        },
        "failure_taxonomy": {
            "duplicate_moves_generated": 0,
            "partial_jump_sequences": 0,
            "illegal_geometry_moves": 0,
            "out_of_bounds_coordinates": 0,
            "missing_legal_moves": 0,
            "parse_failures": 0,
            "wrong_branch_called": 0,
            "api_failures": 0,
        },
        "branch_mismatch": False,
        "scanner_api_ok": True,
        "proposal_api_ok": True,
        "scanner_parse_failure": False,
        "proposal_parse_failure": parse_failure,
        "scanner_raw": "",
        "side_to_move": "RED",
        "elapsed_s": 1.0,
        "engine_best_move": None,
        "kingsrow_best_move": None,
    }


# ════════════════════════════════════════════════════════════════════════════════
# _bf_bucket
# ════════════════════════════════════════════════════════════════════════════════

class TestBfBucket:

    @pytest.mark.parametrize("n,expected", [
        (1, "1"), (2, "2"), (3, "3"),
        (9, "9"), (10, "10"),
        (11, "11+"), (12, "11+"), (20, "11+"), (100, "11+"),
    ])
    def test_exact_and_overflow(self, n, expected):
        assert _bf_bucket(n) == expected

    def test_zero_maps_to_zero_bucket(self):
        # 0 is unusual but should not crash; maps to "0" (≤10)
        assert _bf_bucket(0) == "0"

    def test_boundary_10_is_exact(self):
        assert _bf_bucket(10) == "10"

    def test_boundary_11_is_overflow(self):
        assert _bf_bucket(11) == "11+"


# ════════════════════════════════════════════════════════════════════════════════
# _build_branching_factor_breakdown
# ════════════════════════════════════════════════════════════════════════════════

class TestBuildBranchingFactorBreakdown:

    def test_empty_results_returns_empty_dict(self):
        result = _build_branching_factor_breakdown([])
        assert result == {}

    def test_single_bucket_populated(self):
        results = [_r(legal_count=3), _r(legal_count=3, classification="wrong")]
        bd = _build_branching_factor_breakdown(results)
        assert "3" in bd
        assert bd["3"]["n_positions"] == 2

    def test_empty_buckets_omitted(self):
        results = [_r(legal_count=1)]
        bd = _build_branching_factor_breakdown(results)
        assert "1" in bd
        assert "2" not in bd   # empty bucket omitted

    def test_overflow_bucket(self):
        results = [_r(legal_count=12), _r(legal_count=11)]
        bd = _build_branching_factor_breakdown(results)
        assert "11+" in bd
        assert bd["11+"]["n_positions"] == 2
        assert "12" not in bd  # no individual label for 12

    def test_bucket_ordering_deterministic(self):
        # Keys must come out in order: "1", "2", ..., "10", "11+"
        results = [
            _r(legal_count=5),
            _r(legal_count=2),
            _r(legal_count=11),
            _r(legal_count=1),
        ]
        bd = _build_branching_factor_breakdown(results)
        keys = list(bd.keys())
        expected_order = ["1", "2", "5", "11+"]
        assert keys == expected_order

    def test_perfect_proposal_pct_correct(self):
        results = [
            _r(legal_count=4, classification="perfect"),
            _r(legal_count=4, classification="wrong"),
            _r(legal_count=4, classification="perfect"),
        ]
        bd = _build_branching_factor_breakdown(results)
        assert bd["4"]["n_positions"] == 3
        assert bd["4"]["perfect_proposal_pct"] == pytest.approx(66.7, abs=0.1)

    def test_scanner_correct_pct_correct(self):
        results = [
            _r(legal_count=2, scanner_correct=True),
            _r(legal_count=2, scanner_correct=True),
            _r(legal_count=2, scanner_correct=False),
        ]
        bd = _build_branching_factor_breakdown(results)
        assert bd["2"]["scanner_correct_pct"] == pytest.approx(66.7, abs=0.1)

    def test_engine_coverage_correct(self):
        results = [
            _r(legal_count=3, contains_engine=True),
            _r(legal_count=3, contains_engine=False),
            _r(legal_count=3, contains_engine=True),
        ]
        bd = _build_branching_factor_breakdown(results)
        assert bd["3"]["engine_best_coverage_pct"] == pytest.approx(66.7, abs=0.1)

    def test_engine_coverage_none_when_all_none(self):
        """When every result has contains_engine_best=None, pct must be None."""
        results = [
            _r(legal_count=3, contains_engine=None),
            _r(legal_count=3, contains_engine=None),
        ]
        bd = _build_branching_factor_breakdown(results)
        assert bd["3"]["engine_best_coverage_pct"] is None

    def test_kr_coverage_none_when_absent(self):
        """KR coverage is None when no annotations were provided (all None)."""
        results = [
            _r(legal_count=4, contains_kr=None),
            _r(legal_count=4, contains_kr=None),
        ]
        bd = _build_branching_factor_breakdown(results)
        assert bd["4"]["kingsrow_best_coverage_pct"] is None

    def test_kr_coverage_correct_when_present(self):
        results = [
            _r(legal_count=5, contains_kr=True),
            _r(legal_count=5, contains_kr=False),
        ]
        bd = _build_branching_factor_breakdown(results)
        assert bd["5"]["kingsrow_best_coverage_pct"] == 50.0

    def test_missing_legal_move_count_skipped(self):
        results = [
            _r(legal_count=2),
            {"legal_move_count": None, "classification": "perfect"},   # missing
        ]
        bd = _build_branching_factor_breakdown(results)
        # Only the valid one counts
        assert bd["2"]["n_positions"] == 1

    def test_bucket_keys_present(self):
        results = [_r(legal_count=7)]
        bd = _build_branching_factor_breakdown(results)
        bv = bd["7"]
        for key in (
            "n_positions",
            "perfect_proposal_pct",
            "scanner_correct_pct",
            "engine_best_coverage_pct",
            "kingsrow_best_coverage_pct",
        ):
            assert key in bv, f"Missing key '{key}' in bucket"

    def test_all_100_perfect(self):
        results = [_r(legal_count=1, classification="perfect") for _ in range(5)]
        bd = _build_branching_factor_breakdown(results)
        assert bd["1"]["perfect_proposal_pct"] == 100.0

    def test_all_0_perfect(self):
        results = [_r(legal_count=6, classification="wrong") for _ in range(3)]
        bd = _build_branching_factor_breakdown(results)
        assert bd["6"]["perfect_proposal_pct"] == 0.0

    def test_mixed_engine_and_none_excludes_none_from_denominator(self):
        """Positions with contains_engine_best=None must not count in denominator."""
        results = [
            _r(legal_count=3, contains_engine=True),
            _r(legal_count=3, contains_engine=None),   # excluded
            _r(legal_count=3, contains_engine=False),
        ]
        bd = _build_branching_factor_breakdown(results)
        # 1 True out of 2 evaluated (None excluded)
        assert bd["3"]["engine_best_coverage_pct"] == 50.0

    def test_multiple_buckets_independent(self):
        results = [
            _r(legal_count=1, classification="perfect"),
            _r(legal_count=2, classification="wrong"),
            _r(legal_count=1, classification="wrong"),
        ]
        bd = _build_branching_factor_breakdown(results)
        assert bd["1"]["n_positions"] == 2
        assert bd["1"]["perfect_proposal_pct"] == 50.0
        assert bd["2"]["n_positions"] == 1
        assert bd["2"]["perfect_proposal_pct"] == 0.0


# ════════════════════════════════════════════════════════════════════════════════
# summarize_results integration
# ════════════════════════════════════════════════════════════════════════════════

class TestSummarizeResultsBranchingFactor:

    def test_branching_factor_breakdown_key_present(self):
        results = [_r(legal_count=3)]
        summary = summarize_results(results)
        assert "branching_factor_breakdown" in summary

    def test_branching_factor_values_consistent(self):
        results = [
            _r(legal_count=2, classification="perfect", scanner_correct=True),
            _r(legal_count=2, classification="wrong",   scanner_correct=False),
            _r(legal_count=5, classification="perfect", scanner_correct=True),
        ]
        summary = summarize_results(results)
        bd = summary["branching_factor_breakdown"]
        assert bd["2"]["n_positions"] == 2
        assert bd["5"]["n_positions"] == 1

    def test_existing_keys_unaffected(self):
        results = [_r()]
        summary = summarize_results(results)
        for key in (
            "total_positions", "quadrants", "classifications",
            "failure_taxonomy", "scanner_accuracy", "branch_routing",
            "proposal_accuracy", "proposal_accuracy_isolated",
            "by_category", "by_difficulty", "best_move_coverage",
        ):
            assert key in summary, f"Key missing after branching-factor addition: {key}"

    def test_empty_results_no_crash(self):
        # summarize_results([]) fast-exits with {"total": 0} — existing behavior.
        # The important contract is: no crash, and the helper returns {}.
        summary = summarize_results([])
        assert isinstance(summary, dict)  # no crash
        # Direct helper test: _build_branching_factor_breakdown([]) == {}
        assert _build_branching_factor_breakdown([]) == {}

    def test_no_mutation_of_results(self):
        results = [_r(legal_count=4)]
        snapshot = json.dumps(results, sort_keys=True)
        summarize_results(results)
        assert json.dumps(results, sort_keys=True) == snapshot


# ════════════════════════════════════════════════════════════════════════════════
# Filtering metadata structure (unit test — no main() invocation)
# ════════════════════════════════════════════════════════════════════════════════

class TestFilteringMetaStructure:
    """
    Verify the shape of the _filtering_meta dict that main() builds and
    attaches to summary["filtering"].  We test this by simulating the
    filtering logic directly rather than invoking main() (which would
    require LLM calls).
    """

    def _apply_filter(self, dataset, min_lm, max_lm):
        """Mirror the filtering logic from main()."""
        active = (min_lm is not None) or (max_lm is not None)
        before = len(dataset)
        if active:
            dataset = [
                e for e in dataset
                if (
                    (min_lm is None or len(e.get("hidden_legal_moves", [])) >= min_lm)
                    and (max_lm is None or len(e.get("hidden_legal_moves", [])) <= max_lm)
                )
            ]
        after = len(dataset)
        return dataset, {
            "min_legal_moves": min_lm,
            "max_legal_moves": max_lm,
            "active": active,
            "positions_before": before,
            "positions_after": after,
            "positions_removed": before - after,
        }

    def _entry(self, n_moves: int) -> dict:
        return {
            "hidden_legal_moves": [{}] * n_moves,
            "board": [],
            "side_to_move": "RED",
            "scenario_id": f"pos_{n_moves}",
        }

    def test_no_filter_active_false(self):
        dataset = [self._entry(3)]
        _, meta = self._apply_filter(dataset, None, None)
        assert meta["active"] is False
        assert meta["positions_removed"] == 0

    def test_min_filter_removes_small(self):
        dataset = [self._entry(1), self._entry(2), self._entry(5)]
        result, meta = self._apply_filter(dataset, min_lm=3, max_lm=None)
        assert meta["active"] is True
        assert meta["positions_after"] == 1
        assert meta["positions_removed"] == 2
        assert len(result) == 1

    def test_max_filter_removes_large(self):
        dataset = [self._entry(2), self._entry(4), self._entry(8)]
        result, meta = self._apply_filter(dataset, min_lm=None, max_lm=4)
        assert meta["positions_after"] == 2
        assert meta["positions_removed"] == 1

    def test_range_filter(self):
        dataset = [self._entry(i) for i in range(1, 10)]  # 1..9
        result, meta = self._apply_filter(dataset, min_lm=3, max_lm=6)
        # keeps 3,4,5,6 → 4 entries
        assert meta["positions_after"] == 4
        assert meta["positions_removed"] == 5

    def test_meta_required_keys_present(self):
        _, meta = self._apply_filter([self._entry(3)], None, None)
        for key in (
            "min_legal_moves", "max_legal_moves", "active",
            "positions_before", "positions_after", "positions_removed",
        ):
            assert key in meta, f"Missing key: {key}"

    def test_filter_exact_boundary_inclusive(self):
        """min/max boundaries are inclusive."""
        dataset = [self._entry(3), self._entry(5), self._entry(7)]
        result, meta = self._apply_filter(dataset, min_lm=3, max_lm=7)
        assert meta["positions_after"] == 3  # all included

    def test_filter_excludes_outside_boundaries(self):
        dataset = [self._entry(2), self._entry(3), self._entry(7), self._entry(8)]
        result, meta = self._apply_filter(dataset, min_lm=3, max_lm=7)
        assert meta["positions_after"] == 2   # 3 and 7 kept; 2 and 8 excluded
