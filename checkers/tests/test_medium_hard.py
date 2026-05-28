# checkers/tests/test_medium_hard.py
#
# Focused regression tests for medium_hard mode and difficulty distribution.
# Fully offline — no LLM, no DLL, no network.

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from checkers.eval.proposal_seperation_eval import (
    _VALID_MODES,
    filter_by_mode,
    summarize_results,
)


def _entry(difficulty: str) -> dict:
    return {
        "difficulty": difficulty,
        "scenario_id": f"scen_{difficulty}",
        "hidden_legal_moves": [{}],
    }


def _r(*, difficulty: str = "easy", classification: str = "perfect") -> dict:
    # Minimal result-dict helper
    return {
        "difficulty": difficulty,
        "classification": classification,
        "quadrant": "scanner_correct_proposal_correct",
        "scanner_eval": {"scanner_correct": True},
        "ground_truth_has_captures": False,
        "category": "category_1",
        "api_failure": False,
        "parse_failure": False,
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
    }


# ════════════════════════════════════════════════════════════════════════════════
# Mode validation & filter
# ════════════════════════════════════════════════════════════════════════════════

class TestMediumHardMode:

    def test_mode_registered(self):
        assert "medium_hard" in _VALID_MODES

    def test_filter_keeps_only_medium_and_hard(self):
        dataset = [
            _entry("easy"),
            _entry("medium"),
            _entry("hard"),
            _entry("unknown"),
        ]
        filtered = filter_by_mode(dataset, "medium_hard")
        
        difficulties = [e["difficulty"] for e in filtered]
        assert len(filtered) == 2
        assert "medium" in difficulties
        assert "hard" in difficulties
        assert "easy" not in difficulties
        assert "unknown" not in difficulties

    def test_filter_empty_dataset(self):
        filtered = filter_by_mode([], "medium_hard")
        assert filtered == []


# ════════════════════════════════════════════════════════════════════════════════
# Difficulty distribution
# ════════════════════════════════════════════════════════════════════════════════

class TestDifficultyDistribution:

    def test_distribution_computed_correctly(self):
        results = [
            _r(difficulty="medium"),
            _r(difficulty="medium"),
            _r(difficulty="hard"),
        ]
        summary = summarize_results(results)
        assert "difficulty_distribution" in summary
        dist = summary["difficulty_distribution"]
        assert dist["medium"] == 2
        assert dist["hard"] == 1
        assert "easy" not in dist

    def test_distribution_all_difficulties(self):
        results = [
            _r(difficulty="easy"),
            _r(difficulty="medium"),
            _r(difficulty="hard"),
        ]
        summary = summarize_results(results)
        dist = summary["difficulty_distribution"]
        assert dist["easy"] == 1
        assert dist["medium"] == 1
        assert dist["hard"] == 1

    def test_distribution_empty_results(self):
        # Empty results should degrade gracefully returning {"total": 0}
        summary = summarize_results([])
        assert "difficulty_distribution" not in summary
        assert summary == {"total": 0}
