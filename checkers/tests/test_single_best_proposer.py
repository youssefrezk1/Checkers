# checkers/tests/test_single_best_proposer.py
#
# Regression tests for single_best proposer mode.
# Verify that:
#   - quiet_single_best_prompt and jump_single_best_prompt are correct
#   - build_quiet_single_best_prompt and build_jump_single_best_prompt format correctly
#   - run_proposal_seperation and run_proposer_only propagate single_best parameter
#   - evaluator truncation and strict validation work as expected
#

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
from checkers.engine.board import RED, BLACK
from checkers.agents.proposal_seperation import (
    quiet_single_best_prompt,
    jump_single_best_prompt,
    build_quiet_single_best_prompt,
    build_jump_single_best_prompt,
    run_proposal_seperation,
    run_proposer_only,
)
from checkers.eval.proposal_seperation_eval import evaluate_position


class TestSingleBestProposerPrompts(unittest.TestCase):

    def setUp(self):
        # 8x8 checkers board representation
        self.board = [[0] * 8 for _ in range(8)]
        self.board[5][2] = 1  # RED man
        self.board[6][1] = 2  # BLACK man

    def test_single_best_prompt_content(self):
        # Verify the prompt texts contain instruction for single strategically best move
        self.assertIn("output ONLY the SINGLE STRONGEST legal capture sequence", jump_single_best_prompt)
        self.assertIn("Choose the single strategically best legal capture sequence", jump_single_best_prompt)
        self.assertIn("output ONLY the SINGLE STRONGEST legal simple move", quiet_single_best_prompt)
        self.assertIn("Choose the single strategically best legal simple move", quiet_single_best_prompt)

    def test_build_jump_single_best_prompt(self):
        sys_p, usr_p = build_jump_single_best_prompt(self.board, RED)
        self.assertEqual(sys_p, jump_single_best_prompt)
        self.assertIn("Current player: RED", usr_p)
        self.assertIn("Choose the single strategically best legal capture sequence", usr_p)

    def test_build_quiet_single_best_prompt(self):
        sys_p, usr_p = build_quiet_single_best_prompt(self.board, RED)
        self.assertEqual(sys_p, quiet_single_best_prompt)
        self.assertIn("Current player: RED", usr_p)
        self.assertIn("Choose the single strategically best legal non-capturing (simple) move", usr_p)

    @patch("checkers.agents.proposal_seperation._call_with_infra_retry")
    def test_run_proposal_seperation_single_best_true(self, mock_call):
        mock_call.side_effect = [
            # Scanner call: return capture_available = False
            ('{"capture_available": false}', True),
            # Quiet single best proposer call
            ('{"moves": [{"type": "simple", "path": [[5, 2], [4, 3]], "captured": []}]}', True),
        ]

        result = run_proposal_seperation(self.board, RED, single_best=True)
        
        self.assertEqual(mock_call.call_count, 2)
        # Check scanner call arguments
        scanner_args = mock_call.call_args_list[0]
        self.assertIn("capture_available", scanner_args.kwargs["system"])

        # Check quiet single best proposer call arguments
        proposer_args = mock_call.call_args_list[1]
        self.assertEqual(proposer_args.kwargs["system"], quiet_single_best_prompt)
        self.assertIn("Choose the single strategically best", proposer_args.kwargs["user"])

        self.assertIsNotNone(result["proposal_moves"])
        self.assertEqual(len(result["proposal_moves"]), 1)
        self.assertEqual(result["proposal_moves"][0]["path"], [[5, 2], [4, 3]])

    @patch("checkers.agents.proposal_seperation._call_with_infra_retry")
    def test_run_proposer_only_single_best_true(self, mock_call):
        mock_call.return_value = (
            '{"moves": [{"type": "jump", "path": [[5, 2], [3, 4]], "captured": [[4, 3]]}]}',
            True
        )

        result = run_proposer_only(self.board, RED, branch="jump", single_best=True)

        self.assertEqual(mock_call.call_count, 1)
        proposer_args = mock_call.call_args_list[0]
        self.assertEqual(proposer_args.kwargs["system"], jump_single_best_prompt)
        self.assertIn("Choose the single strategically best", proposer_args.kwargs["user"])

        self.assertIsNotNone(result["proposal_moves"])
        self.assertEqual(len(result["proposal_moves"]), 1)
        self.assertEqual(result["proposal_moves"][0]["path"], [[5, 2], [3, 4]])


class TestSingleBestHardConstraints(unittest.TestCase):

    def setUp(self):
        self.board = [[0] * 8 for _ in range(8)]
        self.board[5][2] = 1   # RED man
        self.board[6][1] = 2   # BLACK man

    @patch("checkers.agents.proposal_seperation._call_with_infra_retry")
    def test_evaluator_never_receives_greater_than_one_move(self, mock_call):
        # Even if the LLM model proposes 2 moves, the output must be truncated to exactly 1
        mock_call.side_effect = [
            ('{"capture_available": false}', True),
            ('{"moves": [{"type": "simple", "path": [[5, 2], [4, 3]], "captured": []}, {"type": "simple", "path": [[5, 2], [4, 1]], "captured": []}]}', True),
        ]

        result = run_proposal_seperation(self.board, RED, single_best=True)
        self.assertEqual(len(result["proposal_moves"]), 1)
        self.assertEqual(result["original_proposal_moves_len"], 2)

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_strict_validation_single_best_violation(self, mock_sbm, mock_sal, mock_rps):
        mock_rps.return_value = {
            "scanner_raw": '{"capture_available": false}',
            "scanner_prediction": False,
            "scanner_api_ok": True,
            "proposal_raw": '{"moves": [{"type": "simple", "path": [[5,2],[4,3]], "captured": []}, {"type": "simple", "path": [[5,2],[4,1]], "captured": []}]}',
            "proposal_moves": [{"type": "simple", "path": [[5, 2], [4, 3]], "captured": []}], # truncated
            "proposal_api_ok": True,
            "proposal_branch": "quiet",
            "api_failure": False,
            "parse_failure": False,
            "scanner_parse_failure": False,
            "proposal_parse_failure": False,
            "original_proposal_moves_len": 2, # violation: > 1
        }
        mock_sal.return_value = ([], 0.0, None, 0.0)
        mock_sbm.return_value = ({}, 0.0, [], {})

        result = evaluate_position(
            board=self.board,
            current_player=RED,
            single_best=True,
        )

        self.assertEqual(result["classification"], "single_best_violation")
        # Proposed count shows the original (pre-truncated) count
        self.assertEqual(result["proposal_classification"]["proposed_count"], 2)

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_top1_match_equals_coverage_in_single_best(self, mock_sbm, mock_sal, mock_rps):
        # Test Case 1: Match
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
              "facts": {"minimax_score": 1.0, "symbolic_rank": 1}}],
            1.0, None, 0.0,
        )
        mock_sbm.return_value = (
            {"type": "simple", "path": best_path, "captured": []},
            1.0, [], {},
        )

        result_match = evaluate_position(
            board=self.board,
            current_player=RED,
            single_best=True,
        )

        self.assertEqual(result_match["contains_engine_best"], True)
        self.assertEqual(result_match["top1_engine_match"], True)
        self.assertEqual(result_match["contains_engine_best"], result_match["top1_engine_match"])

        # Test Case 2: Miss
        mock_rps.return_value = {
            "scanner_raw": '{"capture_available": false}',
            "scanner_prediction": False,
            "scanner_api_ok": True,
            "proposal_raw": '{"moves": [{"type": "simple", "path": [[5,2],[4,1]], "captured": []}]}',
            "proposal_moves": [{"type": "simple", "path": [[5, 2], [4, 1]], "captured": []}],
            "proposal_api_ok": True,
            "proposal_branch": "quiet",
            "api_failure": False,
            "parse_failure": False,
            "scanner_parse_failure": False,
            "proposal_parse_failure": False,
            "original_proposal_moves_len": 1,
        }

        result_miss = evaluate_position(
            board=self.board,
            current_player=RED,
            single_best=True,
        )

        self.assertEqual(result_miss["contains_engine_best"], False)
        self.assertEqual(result_miss["top1_engine_match"], False)
        self.assertEqual(result_miss["contains_engine_best"], result_miss["top1_engine_match"])
