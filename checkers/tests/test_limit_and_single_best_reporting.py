# checkers/tests/test_limit_and_single_best_reporting.py
#
# Regression tests for:
#   1. --limit correctness: exactly N scenarios evaluated, counts/metrics reflect limited subset
#   2. Single-best reporting isolation: no missing_legal_moves in taxonomy or summary

from __future__ import annotations

import io
import unittest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(scenario_id: str, side: str = "red") -> dict:
    board = [[0] * 8 for _ in range(8)]
    board[5][2] = 1
    board[2][5] = 2
    return {
        "scenario_id": scenario_id,
        "side_to_move": side,
        "board": board,
        "hidden_legal_moves": [{"type": "simple", "path": [[5, 2], [4, 3]], "captured": []}],
        "category": "test",
        "difficulty": "easy",
    }


def _make_eval_result(scenario_id: str, single_best: bool = False) -> dict:
    classification = "perfect"
    failure_taxonomy = {
        "duplicate_moves_generated": 0,
        "partial_jump_sequences": 0,
        "illegal_geometry_moves": 0,
        "out_of_bounds_coordinates": 0,
        "parse_failures": 0,
        "wrong_branch_called": 0,
        "api_failures": 0,
    }
    if not single_best:
        failure_taxonomy["missing_legal_moves"] = 0

    return {
        "scenario_id": scenario_id,
        "classification": classification,
        "quadrant": "scanner_correct_proposal_correct",
        "elapsed_s": 0.1,
        "scanner_correct": True,
        "proposal_branch": "quiet",
        "api_failure": False,
        "parse_failure": False,
        "proposal_classification": {
            "classification": "perfect",
            "legal_count": 1,
            "proposed_count": 1,
            "legal_proposed": 1,
            "illegal_proposed": 0,
            "missing_legal": 0,
        },
        "failure_taxonomy": failure_taxonomy,
        "contains_engine_best": True,
        "top1_engine_match": True,
        "contains_kingsrow_best": None,
        "top1_kingsrow_match": None,
    }


# ---------------------------------------------------------------------------
# 1. --limit correctness
# ---------------------------------------------------------------------------

class TestLimitCorrectness(unittest.TestCase):

    def _run_main_with_args(self, dataset: list[dict], extra_args: list[str]) -> tuple[int, str]:
        """Patch load_dataset to return dataset, run main(), capture stdout."""
        from checkers.eval.proposal_seperation_eval import main

        def fake_evaluate(**kwargs):
            return _make_eval_result(kwargs.get("scenario_id", ""), single_best=kwargs.get("single_best", False))

        buf = io.StringIO()
        with (
            patch("checkers.eval.proposal_seperation_eval.load_dataset", return_value=dataset),
            patch("checkers.eval.proposal_seperation_eval.load_bestmove_annotations", return_value={}),
            patch("checkers.eval.proposal_seperation_eval.evaluate_position", side_effect=fake_evaluate),
            patch("sys.stdout", buf),
        ):
            rc = main(["--dataset", "fake.json", "--out", "/tmp/test_out.json"] + extra_args)
        return rc, buf.getvalue()

    def test_limit_restricts_to_exactly_n(self):
        dataset = [_make_entry(f"pos_{i}") for i in range(20)]
        _, output = self._run_main_with_args(dataset, ["--limit", "5"])
        # Exactly 5 row lines with position numbers 1..5
        self.assertIn("     5  ", output)
        self.assertNotIn("     6  ", output)

    def test_limit_positions_header_matches_actual(self):
        dataset = [_make_entry(f"pos_{i}") for i in range(20)]
        _, output = self._run_main_with_args(dataset, ["--limit", "7"])
        self.assertIn("Positions : 7", output)

    def test_limit_summary_total_matches_limit(self):
        dataset = [_make_entry(f"pos_{i}") for i in range(20)]
        _, output = self._run_main_with_args(dataset, ["--limit", "3"])
        self.assertIn("Total positions       : 3", output)

    def test_limit_larger_than_dataset_uses_full_dataset(self):
        dataset = [_make_entry(f"pos_{i}") for i in range(5)]
        _, output = self._run_main_with_args(dataset, ["--limit", "100"])
        self.assertIn("Total positions       : 5", output)

    def test_limit_zero_produces_empty_result(self):
        dataset = [_make_entry(f"pos_{i}") for i in range(10)]
        _, output = self._run_main_with_args(dataset, ["--limit", "0"])
        self.assertIn("Total positions       : 0", output)

    def test_limit_applied_after_bf_filter(self):
        # BF filter and limit both active: limit is applied last
        # All entries have 1 legal move; --min-legal-moves 1 keeps all
        dataset = [_make_entry(f"pos_{i}") for i in range(20)]
        _, output = self._run_main_with_args(
            dataset, ["--min-legal-moves", "1", "--limit", "6"]
        )
        self.assertIn("Total positions       : 6", output)


# ---------------------------------------------------------------------------
# 2. Single-best reporting isolation
# ---------------------------------------------------------------------------

class TestSingleBestReportingIsolation(unittest.TestCase):

    def _summarize(self, results: list[dict], single_best: bool) -> dict:
        from checkers.eval.proposal_seperation_eval import summarize_results
        return summarize_results(results, single_best=single_best)

    def test_missing_legal_moves_absent_from_taxonomy_when_single_best(self):
        results = [_make_eval_result(f"p{i}", single_best=True) for i in range(5)]
        summary = self._summarize(results, single_best=True)
        self.assertNotIn("missing_legal_moves", summary["failure_taxonomy"])

    def test_missing_legal_moves_absent_avg_when_single_best(self):
        results = [_make_eval_result(f"p{i}", single_best=True) for i in range(5)]
        summary = self._summarize(results, single_best=True)
        self.assertNotIn("missing_legal_moves_avg", summary["failure_taxonomy"])

    def test_missing_legal_moves_present_in_multi_mode(self):
        results = [_make_eval_result(f"p{i}", single_best=False) for i in range(5)]
        summary = self._summarize(results, single_best=False)
        self.assertIn("missing_legal_moves", summary["failure_taxonomy"])

    def test_single_best_violations_counter_present_when_single_best(self):
        results = [_make_eval_result(f"p{i}", single_best=True) for i in range(3)]
        summary = self._summarize(results, single_best=True)
        self.assertIn("single_best_violations", summary)

    def test_single_best_violations_zero_in_multi_mode(self):
        # Key may be present but must be 0 — there can be no violations in multi-mode
        results = [_make_eval_result(f"p{i}", single_best=False) for i in range(3)]
        summary = self._summarize(results, single_best=False)
        self.assertEqual(summary.get("single_best_violations", 0), 0)

    def test_perfect_count_correct_in_single_best(self):
        results = [_make_eval_result(f"p{i}", single_best=True) for i in range(4)]
        summary = self._summarize(results, single_best=True)
        self.assertEqual(summary["proposal_accuracy"]["perfect"], 4)
        self.assertEqual(summary["proposal_accuracy"]["total"], 4)

    def test_classification_breakdown_no_legal_but_incomplete_when_single_best(self):
        # A single-best result should never have "legal_but_incomplete" classification
        results = [_make_eval_result(f"p{i}", single_best=True) for i in range(3)]
        for r in results:
            self.assertNotEqual(r["classification"], "legal_but_incomplete")


# ---------------------------------------------------------------------------
# 3. evaluate_position never returns legal_but_incomplete in single_best mode
# ---------------------------------------------------------------------------

class TestEvaluatePositionSingleBest(unittest.TestCase):

    def setUp(self):
        self.board = [[0] * 8 for _ in range(8)]
        self.board[5][2] = 1  # RED man

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_single_legal_move_yields_perfect_not_incomplete(self, mock_sbm, mock_sal, mock_rps):
        from checkers.eval.proposal_seperation_eval import evaluate_position
        from checkers.engine.board import RED

        best_path = [[5, 2], [4, 3]]
        mock_rps.return_value = {
            "scanner_raw": '{"capture_available": false}',
            "scanner_prediction": False,
            "scanner_api_ok": True,
            "proposal_raw": '{"moves": [{"type": "simple", "path": [[5,2],[4,3]], "captured": []}]}',
            "proposal_moves": [{"type": "simple", "path": best_path, "captured": []}],
            "proposal_api_ok": True,
            "proposal_branch": "quiet",
            "api_failure": False,
            "parse_failure": False,
            "scanner_parse_failure": False,
            "proposal_parse_failure": False,
            "original_proposal_moves_len": 1,
        }
        mock_sal.return_value = (
            [{"type": "simple", "path": best_path, "captured": [],
              "facts": {"minimax_score": 0.5, "symbolic_rank": 1}}],
            0.5, None, 0.0,
        )
        mock_sbm.return_value = (
            {"type": "simple", "path": best_path, "captured": []},
            0.5, [], {},
        )

        result = evaluate_position(board=self.board, current_player=RED, single_best=True)
        self.assertEqual(result["classification"], "perfect")
        self.assertNotEqual(result["classification"], "legal_but_incomplete")

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_failure_taxonomy_missing_legal_zero_when_single_best(self, mock_sbm, mock_sal, mock_rps):
        from checkers.eval.proposal_seperation_eval import evaluate_position
        from checkers.engine.board import RED

        best_path = [[5, 2], [4, 3]]
        mock_rps.return_value = {
            "scanner_raw": '{"capture_available": false}',
            "scanner_prediction": False,
            "scanner_api_ok": True,
            "proposal_raw": '{"moves": [{"type": "simple", "path": [[5,2],[4,3]], "captured": []}]}',
            "proposal_moves": [{"type": "simple", "path": best_path, "captured": []}],
            "proposal_api_ok": True,
            "proposal_branch": "quiet",
            "api_failure": False,
            "parse_failure": False,
            "scanner_parse_failure": False,
            "proposal_parse_failure": False,
            "original_proposal_moves_len": 1,
        }
        mock_sal.return_value = (
            [{"type": "simple", "path": best_path, "captured": [],
              "facts": {"minimax_score": 0.5, "symbolic_rank": 1}}],
            0.5, None, 0.0,
        )
        mock_sbm.return_value = (
            {"type": "simple", "path": best_path, "captured": []},
            0.5, [], {},
        )

        result = evaluate_position(board=self.board, current_player=RED, single_best=True)
        self.assertEqual(result["failure_taxonomy"]["missing_legal_moves"], 0)

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_violation_coverage_fields_are_null(self, mock_sbm, mock_sal, mock_rps):
        # When LLM produces more than 1 move (violation), coverage fields must be None.
        # The truncated first move is an artifact — crediting it inflates coverage stats.
        from checkers.eval.proposal_seperation_eval import evaluate_position
        from checkers.engine.board import RED

        best_path = [[5, 2], [4, 3]]
        mock_rps.return_value = {
            "scanner_raw": '{"capture_available": false}',
            "scanner_prediction": False,
            "scanner_api_ok": True,
            "proposal_raw": '{"moves": [...]}',
            # Truncated to first; original_len=2 triggers violation
            "proposal_moves": [{"type": "simple", "path": best_path, "captured": []}],
            "proposal_api_ok": True,
            "proposal_branch": "quiet",
            "api_failure": False,
            "parse_failure": False,
            "scanner_parse_failure": False,
            "proposal_parse_failure": False,
            "original_proposal_moves_len": 2,  # LLM produced 2 moves → violation
        }
        mock_sal.return_value = (
            [{"type": "simple", "path": best_path, "captured": [],
              "facts": {"minimax_score": 0.5, "symbolic_rank": 1}}],
            0.5, None, 0.0,
        )
        mock_sbm.return_value = (
            {"type": "simple", "path": best_path, "captured": []},
            0.5, [], {},
        )

        result = evaluate_position(board=self.board, current_player=RED, single_best=True)
        self.assertEqual(result["classification"], "single_best_violation")
        # Coverage fields must be None — first move is a truncation artifact, not a real selection
        self.assertIsNone(result["contains_engine_best"])
        self.assertIsNone(result["contains_kingsrow_best"])
        self.assertIsNone(result["top1_engine_match"])
        self.assertIsNone(result["top1_kingsrow_match"])

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_non_violation_coverage_fields_are_bool(self, mock_sbm, mock_sal, mock_rps):
        # When LLM produces exactly 1 move (no violation), coverage fields must be bool.
        from checkers.eval.proposal_seperation_eval import evaluate_position
        from checkers.engine.board import RED

        best_path = [[5, 2], [4, 3]]
        mock_rps.return_value = {
            "scanner_raw": '{"capture_available": false}',
            "scanner_prediction": False,
            "scanner_api_ok": True,
            "proposal_raw": '{"moves": [{"type": "simple", "path": [[5,2],[4,3]], "captured": []}]}',
            "proposal_moves": [{"type": "simple", "path": best_path, "captured": []}],
            "proposal_api_ok": True,
            "proposal_branch": "quiet",
            "api_failure": False,
            "parse_failure": False,
            "scanner_parse_failure": False,
            "proposal_parse_failure": False,
            "original_proposal_moves_len": 1,
        }
        mock_sal.return_value = (
            [{"type": "simple", "path": best_path, "captured": [],
              "facts": {"minimax_score": 0.5, "symbolic_rank": 1}}],
            0.5, None, 0.0,
        )
        mock_sbm.return_value = (
            {"type": "simple", "path": best_path, "captured": []},
            0.5, [], {},
        )

        result = evaluate_position(board=self.board, current_player=RED, single_best=True)
        self.assertNotEqual(result["classification"], "single_best_violation")
        self.assertIsInstance(result["contains_engine_best"], bool)
        self.assertIsInstance(result["top1_engine_match"], bool)


if __name__ == "__main__":
    unittest.main()
