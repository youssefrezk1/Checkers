# checkers/tests/test_bestmove_coverage.py
#
# Regression tests for the best-move coverage extension.
# Covers:
#   - load_bestmove_annotations()  — happy path, missing file, bad JSON, empty
#   - _path_in_proposals()         — all True/False/None cases, normalisation
#   - _coverage_block()            — correctness of all breakdown dimensions
#   - _build_coverage_stats()      — top-level structure
#   - evaluate_position() integration — 4 new fields present on every exit path
#   - summarize_results()          — best_move_coverage key present + correct
#   - no mutation of proposal_moves or result dicts
#   - graceful degradation when annotation file absent
#
# These tests are OFFLINE only — no LLM calls, no KingsRow DLL.
# All network-dependent components are fully mocked.

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── make the project root importable ────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from checkers.eval.proposal_seperation_eval import (
    load_bestmove_annotations,
    _path_in_proposals,
    _is_top1_match,
    _coverage_block,
    _build_coverage_stats,
    summarize_results,
    evaluate_position,
)


# ════════════════════════════════════════════════════════════════════════════════
# FIXTURES — minimal synthetic data
# ════════════════════════════════════════════════════════════════════════════════

def _ann_entry(scenario_id: str, kr_path=None, engine_path=None) -> dict:
    return {
        "scenario_id": scenario_id,
        "kr_path": kr_path,
        "kr_score": 10.0 if kr_path else None,
        "engine_best_path": engine_path,
        "engine_best_score": 5.0 if engine_path else None,
        "engine_ok": engine_path is not None,
        "kr_available": kr_path is not None,
    }


def _make_result(
    *,
    scenario_id: str = "s1",
    category: str = "crowded_board",
    difficulty: str = "easy",
    classification: str = "perfect",
    quadrant: str = "scanner_correct_proposal_correct",
    scanner_correct: bool = True,
    ground_truth_has_captures: bool = False,
    contains_engine_best: bool | None = True,
    contains_kingsrow_best: bool | None = True,
    proposed_moves: list | None = None,
    api_failure: bool = False,
    parse_failure: bool = False,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "category": category,
        "difficulty": difficulty,
        "classification": classification,
        "quadrant": quadrant,
        "scanner_eval": {"scanner_correct": scanner_correct},
        "ground_truth_has_captures": ground_truth_has_captures,
        "contains_engine_best": contains_engine_best,
        "contains_kingsrow_best": contains_kingsrow_best,
        "api_failure": api_failure,
        "parse_failure": parse_failure,
        "proposal_classification": {
            "classification": classification,
            "legal_count": 3,
            "proposed_count": 3,
            "legal_proposed": 3,
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
        "legal_move_count": 3,
        "elapsed_s": 1.0,
        "engine_best_move": [[5, 2], [4, 3]],
        "kingsrow_best_move": [[5, 2], [4, 3]],
        "branch_mismatch": False,
        "scanner_api_ok": True,
        "proposal_api_ok": True,
        "scanner_parse_failure": False,
        "proposal_parse_failure": parse_failure,
        "scanner_raw": "",
        "side_to_move": "RED",
    }


# ════════════════════════════════════════════════════════════════════════════════
# load_bestmove_annotations
# ════════════════════════════════════════════════════════════════════════════════

class TestLoadBestmoveAnnotations:

    def test_none_path_returns_empty_dict(self):
        result = load_bestmove_annotations(None)
        assert result == {}

    def test_missing_file_returns_empty_dict(self):
        result = load_bestmove_annotations("/nonexistent/path/annotations.json")
        assert result == {}

    def test_valid_file_returns_mapping(self, tmp_path):
        data = [
            {"scenario_id": "s1", "kr_path": [[1, 2], [3, 4]]},
            {"scenario_id": "s2", "kr_path": None},
        ]
        p = tmp_path / "ann.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = load_bestmove_annotations(str(p))
        assert set(result.keys()) == {"s1", "s2"}
        assert result["s1"]["kr_path"] == [[1, 2], [3, 4]]

    def test_entries_without_scenario_id_are_skipped(self, tmp_path):
        data = [
            {"kr_path": [[1, 2]]},           # no scenario_id → skipped
            {"scenario_id": "s1", "kr_path": [[3, 4]]},
        ]
        p = tmp_path / "ann.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = load_bestmove_annotations(str(p))
        assert list(result.keys()) == ["s1"]

    def test_malformed_json_returns_empty_dict(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("THIS IS NOT JSON", encoding="utf-8")
        result = load_bestmove_annotations(str(p))
        assert result == {}

    def test_non_list_json_returns_empty_dict(self, tmp_path):
        p = tmp_path / "obj.json"
        p.write_text(json.dumps({"scenario_id": "s1"}), encoding="utf-8")
        result = load_bestmove_annotations(str(p))
        assert result == {}

    def test_empty_list_returns_empty_dict(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text("[]", encoding="utf-8")
        result = load_bestmove_annotations(str(p))
        assert result == {}


# ════════════════════════════════════════════════════════════════════════════════
# _path_in_proposals
# ════════════════════════════════════════════════════════════════════════════════

class TestPathInProposals:

    def _move(self, path):
        return {"type": "simple", "path": path, "captured": []}

    def test_target_none_returns_none(self):
        proposals = [self._move([[1, 2], [3, 4]])]
        assert _path_in_proposals(None, proposals) is None

    def test_empty_proposals_returns_false(self):
        assert _path_in_proposals([[1, 2], [3, 4]], []) is False

    def test_none_proposals_returns_false(self):
        assert _path_in_proposals([[1, 2], [3, 4]], None) is False

    def test_exact_match_returns_true(self):
        path = [[5, 2], [4, 3]]
        proposals = [self._move([[1, 0], [2, 1]]), self._move(path)]
        assert _path_in_proposals(path, proposals) is True

    def test_no_match_returns_false(self):
        path = [[5, 2], [4, 3]]
        proposals = [self._move([[1, 0], [2, 1]]), self._move([[3, 2], [4, 3]])]
        assert _path_in_proposals(path, proposals) is False

    def test_int_float_normalisation(self):
        # target uses int coords; proposal uses floats — should still match
        target = [[5, 2], [4, 3]]
        proposals = [{"type": "simple", "path": [[5.0, 2.0], [4.0, 3.0]], "captured": []}]
        assert _path_in_proposals(target, proposals) is True

    def test_single_square_path(self):
        assert _path_in_proposals([[3, 4]], [self._move([[3, 4]])]) is True

    def test_multi_hop_match(self):
        path = [[2, 1], [4, 3], [6, 5]]
        proposals = [self._move(path)]
        assert _path_in_proposals(path, proposals) is True

    def test_partial_path_does_not_match(self):
        target = [[2, 1], [4, 3]]
        full   = [[2, 1], [4, 3], [6, 5]]
        proposals = [self._move(full)]
        assert _path_in_proposals(target, proposals) is False

    def test_non_dict_in_proposals_does_not_crash(self):
        target = [[1, 2], [3, 4]]
        # Proposals list containing a list, a string, None, and a correct dict
        proposals = [
            [[1, 2], [3, 4]],  # list instead of dict (raises AttributeError without fix)
            "simple_move_string",
            None,
            self._move(target),
        ]
        assert _path_in_proposals(target, proposals) is True

    def test_non_dict_only_no_match(self):
        target = [[1, 2], [3, 4]]
        proposals = [
            [[1, 2], [3, 4]],
            "simple_move_string",
            None,
        ]
        assert _path_in_proposals(target, proposals) is False


# ════════════════════════════════════════════════════════════════════════════════
# _coverage_block  /  _build_coverage_stats
# ════════════════════════════════════════════════════════════════════════════════

class TestCoverageBlock:

    def _results(self, specs: list[tuple]) -> list[dict]:
        """specs: list of (contains_engine_best, scanner_correct, tactical, cat, diff)"""
        out = []
        for ce, sc, is_tac, cat, diff in specs:
            out.append(_make_result(
                contains_engine_best=ce,
                contains_kingsrow_best=ce,
                scanner_correct=sc,
                ground_truth_has_captures=is_tac,
                category=cat,
                difficulty=diff,
            ))
        return out

    def test_all_true(self):
        results = self._results([
            (True, True, False, "cat_a", "easy"),
            (True, True, True,  "cat_b", "hard"),
        ])
        block = _coverage_block(results, "contains_engine_best")
        assert block["evaluated"] == 2
        assert block["contains"] == 2
        assert block["coverage_pct"] == 100.0

    def test_all_false(self):
        results = self._results([
            (False, True, False, "cat_a", "easy"),
            (False, False, True, "cat_b", "medium"),
        ])
        block = _coverage_block(results, "contains_engine_best")
        assert block["coverage_pct"] == 0.0
        assert block["contains"] == 0

    def test_mixed_coverage(self):
        results = self._results([
            (True,  True, False, "c", "easy"),
            (False, True, True,  "c", "easy"),
            (True,  False, False, "c", "hard"),
            (False, False, True,  "c", "hard"),
        ])
        block = _coverage_block(results, "contains_engine_best")
        assert block["evaluated"] == 4
        assert block["contains"] == 2
        assert block["coverage_pct"] == 50.0

    def test_none_values_excluded_from_denominator(self):
        results = [
            _make_result(contains_engine_best=True),
            _make_result(contains_engine_best=None),   # should be excluded
            _make_result(contains_engine_best=False),
        ]
        block = _coverage_block(results, "contains_engine_best")
        assert block["evaluated"] == 2  # None excluded
        assert block["contains"] == 1

    def test_scanner_correct_split(self):
        results = self._results([
            (True,  True, False, "c", "easy"),  # scanner correct, contains
            (False, True, False, "c", "easy"),  # scanner correct, miss
            (True,  False, False, "c", "easy"), # scanner wrong, contains
        ])
        block = _coverage_block(results, "contains_engine_best")
        assert block["scanner_correct"]["n"] == 2
        assert block["scanner_correct"]["contains"] == 1
        assert block["scanner_correct"]["coverage_pct"] == 50.0
        assert block["scanner_wrong"]["n"] == 1
        assert block["scanner_wrong"]["contains"] == 1
        assert block["scanner_wrong"]["coverage_pct"] == 100.0

    def test_quiet_tactical_split(self):
        results = self._results([
            (True,  True, False, "c", "easy"),   # quiet, contains
            (False, True, False, "c", "easy"),   # quiet, miss
            (True,  True, True,  "c", "easy"),   # tactical, contains
        ])
        block = _coverage_block(results, "contains_engine_best")
        assert block["quiet"]["n"] == 2
        assert block["quiet"]["contains"] == 1
        assert block["tactical"]["n"] == 1
        assert block["tactical"]["contains"] == 1

    def test_by_category_keys(self):
        results = self._results([
            (True,  True, False, "cat_x", "easy"),
            (False, True, False, "cat_y", "easy"),
        ])
        block = _coverage_block(results, "contains_engine_best")
        assert "cat_x" in block["by_category"]
        assert "cat_y" in block["by_category"]
        assert block["by_category"]["cat_x"]["coverage_pct"] == 100.0
        assert block["by_category"]["cat_y"]["coverage_pct"] == 0.0

    def test_by_difficulty_keys(self):
        results = self._results([
            (True,  True, False, "c", "easy"),
            (True,  True, False, "c", "hard"),
        ])
        block = _coverage_block(results, "contains_engine_best")
        assert "easy" in block["by_difficulty"]
        assert "hard" in block["by_difficulty"]

    def test_empty_results_returns_none_pct(self):
        block = _coverage_block([], "contains_engine_best")
        assert block["evaluated"] == 0
        assert block["coverage_pct"] is None

    def test_build_coverage_stats_structure(self):
        results = [_make_result()]
        stats = _build_coverage_stats(results)
        assert "engine" in stats
        assert "kingsrow" in stats
        for key in ("evaluated", "contains", "coverage_pct",
                    "scanner_correct", "scanner_wrong",
                    "quiet", "tactical", "by_category", "by_difficulty"):
            assert key in stats["engine"], f"Missing key in engine block: {key}"
            assert key in stats["kingsrow"], f"Missing key in kingsrow block: {key}"


# ════════════════════════════════════════════════════════════════════════════════
# summarize_results — best_move_coverage present
# ════════════════════════════════════════════════════════════════════════════════

class TestSummarizeResultsCoverage:

    def test_best_move_coverage_key_present(self):
        results = [_make_result(), _make_result(scenario_id="s2")]
        summary = summarize_results(results)
        assert "best_move_coverage" in summary

    def test_engine_coverage_correct(self):
        results = [
            _make_result(contains_engine_best=True),
            _make_result(scenario_id="s2", contains_engine_best=False),
        ]
        summary = summarize_results(results)
        eng = summary["best_move_coverage"]["engine"]
        assert eng["evaluated"] == 2
        assert eng["contains"] == 1
        assert eng["coverage_pct"] == 50.0

    def test_kingsrow_coverage_correct(self):
        results = [
            _make_result(contains_kingsrow_best=True),
            _make_result(scenario_id="s2", contains_kingsrow_best=True),
            _make_result(scenario_id="s3", contains_kingsrow_best=False),
        ]
        summary = summarize_results(results)
        kr = summary["best_move_coverage"]["kingsrow"]
        assert kr["evaluated"] == 3
        assert kr["contains"] == 2
        assert round(kr["coverage_pct"], 1) == 66.7

    def test_none_coverage_excluded(self):
        """API/parse failure positions with None coverage should not count."""
        results = [
            _make_result(contains_engine_best=True),
            _make_result(scenario_id="s2", contains_engine_best=None, api_failure=True),
        ]
        summary = summarize_results(results)
        eng = summary["best_move_coverage"]["engine"]
        assert eng["evaluated"] == 1  # None excluded

    def test_existing_summary_keys_unchanged(self):
        """Adding coverage must not remove or alter any existing summary keys."""
        results = [_make_result()]
        summary = summarize_results(results)
        for key in (
            "total_positions", "quadrants", "classifications",
            "failure_taxonomy", "scanner_accuracy", "branch_routing",
            "proposal_accuracy", "proposal_accuracy_isolated",
            "by_category", "by_difficulty",
        ):
            assert key in summary, f"Existing key missing: {key}"


# ════════════════════════════════════════════════════════════════════════════════
# evaluate_position integration — 4 new fields on every exit path
# ════════════════════════════════════════════════════════════════════════════════

# Minimal 8×8 empty board with two pieces so legal moves exist
def _make_board():
    board = [[0] * 8 for _ in range(8)]
    board[5][2] = 1   # RED man
    board[6][1] = 2   # BLACK man
    return board


def _mock_pipeline_result(api_failure=False, scanner_parse_failure=False,
                          proposal_parse_failure=False, branch="quiet"):
    return {
        "scanner_raw": '{"capture_available": false}',
        "scanner_prediction": False,
        "scanner_api_ok": not api_failure,
        "proposal_raw": '{"moves": [{"type": "simple", "path": [[5,2],[4,3]], "captured": []}]}',
        "proposal_moves": None if proposal_parse_failure else [
            {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []}
        ],
        "proposal_api_ok": not api_failure,
        "proposal_branch": branch,
        "api_failure": api_failure,
        "parse_failure": scanner_parse_failure or proposal_parse_failure,
        "scanner_parse_failure": scanner_parse_failure,
        "proposal_parse_failure": proposal_parse_failure,
    }


_COVERAGE_FIELDS = (
    "engine_best_move",
    "kingsrow_best_move",
    "contains_engine_best",
    "contains_kingsrow_best",
)


class TestEvaluatePositionCoverageFields:
    """All four coverage fields must appear on every code path."""

    _ann = {"test_pos": {"kr_path": [[5, 2], [4, 3]], "kr_score": 10.0}}

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_happy_path_has_all_fields(self, mock_sbm, mock_sal, mock_rps):
        from checkers.engine.board import RED
        mock_rps.return_value = _mock_pipeline_result()
        mock_sal.return_value = (
            [{"type": "simple", "path": [[5, 2], [4, 3]], "captured": [],
              "facts": {"minimax_score": 1.0, "symbolic_rank": 1}}],
            1.0, None, 0.0,
        )
        mock_sbm.return_value = (
            {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []},
            1.0, [], {},
        )
        result = evaluate_position(
            board=_make_board(), current_player=RED,
            scenario_id="test_pos",
            bestmove_annotations=self._ann,
        )
        for f in _COVERAGE_FIELDS:
            assert f in result, f"Field missing on happy path: {f}"

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_api_failure_path_has_all_fields(self, mock_sbm, mock_sal, mock_rps):
        from checkers.engine.board import RED
        mock_rps.return_value = _mock_pipeline_result(api_failure=True)
        mock_sal.return_value = ([], 0.0, None, 0.0)
        mock_sbm.return_value = ({}, 0.0, [], {})
        result = evaluate_position(
            board=_make_board(), current_player=RED,
            scenario_id="test_pos",
            bestmove_annotations=self._ann,
        )
        for f in _COVERAGE_FIELDS:
            assert f in result, f"Field missing on api_failure path: {f}"
        # No proposals → contains_* must be None
        assert result["contains_engine_best"] is None
        assert result["contains_kingsrow_best"] is None

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_scanner_parse_failure_path_has_all_fields(self, mock_sbm, mock_sal, mock_rps):
        from checkers.engine.board import RED
        mock_rps.return_value = _mock_pipeline_result(scanner_parse_failure=True)
        mock_sal.return_value = ([], 0.0, None, 0.0)
        mock_sbm.return_value = ({}, 0.0, [], {})
        result = evaluate_position(
            board=_make_board(), current_player=RED,
            scenario_id="test_pos",
            bestmove_annotations=self._ann,
        )
        for f in _COVERAGE_FIELDS:
            assert f in result, f"Field missing on scanner_parse_failure path: {f}"

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_no_annotation_file_degrades_gracefully(self, mock_sbm, mock_sal, mock_rps):
        """Without annotations file, coverage fields are present but kr_ fields are None."""
        from checkers.engine.board import RED
        mock_rps.return_value = _mock_pipeline_result()
        mock_sal.return_value = (
            [{"type": "simple", "path": [[5, 2], [4, 3]], "captured": [],
              "facts": {"minimax_score": 1.0, "symbolic_rank": 1}}],
            1.0, None, 0.0,
        )
        mock_sbm.return_value = (
            {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []},
            1.0, [], {},
        )
        result = evaluate_position(
            board=_make_board(), current_player=RED,
            bestmove_annotations={},   # empty — simulates missing file
        )
        for f in _COVERAGE_FIELDS:
            assert f in result
        assert result["kingsrow_best_move"] is None
        assert result["contains_kingsrow_best"] is None

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_engine_scorer_error_degrades_gracefully(self, mock_sbm, mock_sal, mock_rps):
        """If score_all_legal_moves raises, engine_best_move is None, no crash."""
        from checkers.engine.board import RED
        mock_rps.return_value = _mock_pipeline_result()
        mock_sal.side_effect = RuntimeError("scorer exploded")
        result = evaluate_position(
            board=_make_board(), current_player=RED,
            bestmove_annotations={},
        )
        for f in _COVERAGE_FIELDS:
            assert f in result
        assert result["engine_best_move"] is None
        assert result["contains_engine_best"] is None

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_coverage_match_detected(self, mock_sbm, mock_sal, mock_rps):
        """contains_engine_best=True when proposed path matches engine best."""
        from checkers.engine.board import RED
        best_path = [[5, 2], [4, 3]]
        mock_rps.return_value = _mock_pipeline_result()
        mock_sal.return_value = (
            [{"type": "simple", "path": best_path, "captured": [],
              "facts": {"minimax_score": 1.0, "symbolic_rank": 1}}],
            1.0, None, 0.0,
        )
        mock_sbm.return_value = (
            {"type": "simple", "path": best_path, "captured": []},
            1.0, [], {},
        )
        result = evaluate_position(
            board=_make_board(), current_player=RED,
            bestmove_annotations={},
        )
        assert result["contains_engine_best"] is True

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_coverage_miss_detected(self, mock_sbm, mock_sal, mock_rps):
        """contains_engine_best=False when engine best not in proposals."""
        from checkers.engine.board import RED
        engine_path = [[5, 2], [4, 1]]   # different from what LLM proposed
        mock_rps.return_value = _mock_pipeline_result()
        mock_sal.return_value = (
            [{"type": "simple", "path": engine_path, "captured": [],
              "facts": {"minimax_score": 1.0, "symbolic_rank": 1}}],
            1.0, None, 0.0,
        )
        mock_sbm.return_value = (
            {"type": "simple", "path": engine_path, "captured": []},
            1.0, [], {},
        )
        result = evaluate_position(
            board=_make_board(), current_player=RED,
            bestmove_annotations={},
        )
        # LLM proposed [[5,2],[4,3]] but engine best is [[5,2],[4,1]]
        assert result["contains_engine_best"] is False


# ════════════════════════════════════════════════════════════════════════════════
# No mutation side-effects
# ════════════════════════════════════════════════════════════════════════════════

class TestNoMutationSideEffects:

    def test_result_dict_not_mutated_by_summarize(self):
        """summarize_results must not alter the result dicts it receives."""
        results = [_make_result(), _make_result(scenario_id="s2")]
        snapshots = [json.dumps(r, sort_keys=True) for r in results]
        summarize_results(results)
        for r, snap in zip(results, snapshots):
            assert json.dumps(r, sort_keys=True) == snap

    def test_annotation_dict_not_mutated_by_load(self, tmp_path):
        data = [{"scenario_id": "s1", "kr_path": [[1, 2]]}]
        p = tmp_path / "ann.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = load_bestmove_annotations(str(p))
        # Mutate returned dict
        result["s1"]["injected"] = True
        # Reload — original file unaffected (not in-place mutation of file)
        result2 = load_bestmove_annotations(str(p))
        assert "injected" not in result2["s1"]


# ════════════════════════════════════════════════════════════════════════════════
# _is_top1_match
# ════════════════════════════════════════════════════════════════════════════════

class TestIsTop1Match:

    def test_target_none_returns_none(self):
        assert _is_top1_match(None, [{"path": [[1, 2], [3, 4]]}]) is None

    def test_empty_proposals_returns_false(self):
        assert _is_top1_match([[1, 2], [3, 4]], []) is False
        assert _is_top1_match([[1, 2], [3, 4]], None) is False

    def test_first_move_matches_returns_true(self):
        target = [[1, 2], [3, 4]]
        proposals = [
            {"path": [[1.0, 2.0], [3.0, 4.0]]},  # float normalization
            {"path": [[5, 6], [7, 8]]},
        ]
        assert _is_top1_match(target, proposals) is True

    def test_first_move_does_not_match_returns_false(self):
        target = [[1, 2], [3, 4]]
        proposals = [
            {"path": [[5, 6], [7, 8]]},         # first move is different
            {"path": [[1, 2], [3, 4]]},         # second move matches
        ]
        assert _is_top1_match(target, proposals) is False

    def test_malformed_move_path_returns_false(self):
        target = [[1, 2], [3, 4]]
        assert _is_top1_match(target, [{"path": "not a list"}]) is False
        assert _is_top1_match(target, [{}]) is False


# ════════════════════════════════════════════════════════════════════════════════
# Top-1 match summary metrics
# ════════════════════════════════════════════════════════════════════════════════

class TestTop1MatchSummaryMetrics:

    def test_summary_aggregates_top1_match_percentages(self):
        results = [
            # 2 engine matches, 1 engine miss
            _make_result(scenario_id="s1"),
            _make_result(scenario_id="s2"),
            _make_result(scenario_id="s3"),
        ]
        # Set engine & kingsrow top1 matches explicitly
        results[0]["top1_engine_match"] = True
        results[0]["top1_kingsrow_match"] = True

        results[1]["top1_engine_match"] = True
        results[1]["top1_kingsrow_match"] = False

        results[2]["top1_engine_match"] = False
        results[2]["top1_kingsrow_match"] = False

        summary = summarize_results(results)
        assert summary["top1_engine_match_pct"] == 66.7
        assert summary["top1_kingsrow_match_pct"] == 33.3

    def test_summary_graceful_on_none_values(self):
        results = [
            _make_result(scenario_id="s1"),
            _make_result(scenario_id="s2"),
        ]
        results[0]["top1_engine_match"] = True
        results[0]["top1_kingsrow_match"] = None  # None should be excluded

        results[1]["top1_engine_match"] = None  # None should be excluded
        results[1]["top1_kingsrow_match"] = False

        summary = summarize_results(results)
        assert summary["top1_engine_match_pct"] == 100.0
        assert summary["top1_kingsrow_match_pct"] == 0.0

