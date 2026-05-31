# checkers/tests/test_proposal_separation_single_mode.py
#
# Focused regression tests for single-move mode parity with multi-move mode.
#
# Tests:
#   1. Prompt parity — single-move prompts inherit all legality constraints
#   2. No invented moves — illegal move classified as proposal_illegal
#   3. Exact formatting — single legal move classified as perfect
#   4. Legality parity — classify_proposal_single mirrors multi-move gate semantics
#   5. Deterministic outputs — build functions are stable across calls
#   6. Evaluator compatibility — evaluate_position routes through correct classifier
#   7. Violation detection — LLM returning >1 move is flagged (not silently accepted)

from __future__ import annotations

import unittest
from unittest.mock import patch

from checkers.engine.board import RED, BLACK
from checkers.agents.proposal_seperation import (
    JUMP_SINGLE_SYSTEM_PROMPT,
    JUMP_SYSTEM_PROMPT,
    QUIET_SINGLE_SYSTEM_PROMPT,
    QUIET_SYSTEM_PROMPT,
    build_jump_single_best_prompt,
    build_jump_prompt,
    build_quiet_single_best_prompt,
    build_quiet_prompt,
)
from checkers.eval.proposal_seperation_eval import (
    classify_proposal,
    classify_proposal_single,
    evaluate_position,
)


# ── Shared fixtures ──────────────────────────────────────────────────────────

def _empty_board() -> list[list[int]]:
    return [[0] * 8 for _ in range(8)]


def _board_with_quiet_moves() -> list[list[int]]:
    """RED man at [5,2]; two forward diagonals [4,1] and [4,3] are empty."""
    board = _empty_board()
    board[5][2] = 1  # RED man
    return board


def _board_with_jump() -> list[list[int]]:
    """RED man at [5,2] can jump BLACK man at [4,3] to land at [3,4]."""
    board = _empty_board()
    board[5][2] = 1  # RED man
    board[4][3] = 2  # BLACK man (capturable)
    return board


def _legal_quiet_moves() -> list[dict]:
    return [
        {"type": "simple", "path": [[5, 2], [4, 1]], "captured": []},
        {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []},
    ]


def _legal_jump_move() -> dict:
    return {"type": "jump", "path": [[5, 2], [3, 4]], "captured": [[4, 3]]}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Prompt parity — single-move prompts must contain all legality machinery
# ══════════════════════════════════════════════════════════════════════════════

class TestPromptParity(unittest.TestCase):
    """
    Single-move prompts must inherit all legality constraints from multi-move.
    Only the task objective line and output count should differ.
    """

    # Jump prompts use Phase 1/2/3 structure with GATE-1/2/3.
    # Quiet prompts use PROCEDURE structure with GATE-A/GATE-B.
    # These are the correct legality markers for each prompt family.

    _JUMP_LEGALITY_MARKERS = [
        "PHASE 1", "PHASE 2", "PHASE 3",
        "GATE-1", "GATE-2", "GATE-3",
        "Do not stop early",
        "in-bounds",
        "VERIFY",
    ]
    _QUIET_LEGALITY_MARKERS = [
        "GATE-A", "GATE-B",
        "Do NOT stop after finding the first valid direction",
        "Do NOT move to the next piece until all its directions are checked",
        "in-bounds",
    ]

    def test_jump_single_inherits_all_legality_constraints(self):
        for marker in self._JUMP_LEGALITY_MARKERS:
            self.assertIn(
                marker, JUMP_SINGLE_SYSTEM_PROMPT,
                f"[JUMP_SINGLE] missing legality marker: {marker!r}",
            )

    def test_quiet_single_inherits_all_legality_constraints(self):
        for marker in self._QUIET_LEGALITY_MARKERS:
            self.assertIn(
                marker, QUIET_SINGLE_SYSTEM_PROMPT,
                f"[QUIET_SINGLE] missing legality marker: {marker!r}",
            )

    def test_jump_single_task_objective_only_changes_count(self):
        # Multi-move: "output ALL"
        # Single-move: "output exactly ONE"  — no "STRONGEST", no strategic pressure
        self.assertIn("output exactly ONE complete legal capture sequence", JUMP_SINGLE_SYSTEM_PROMPT)
        self.assertNotIn("STRONGEST", JUMP_SINGLE_SYSTEM_PROMPT)
        self.assertNotIn("strategically best move", JUMP_SINGLE_SYSTEM_PROMPT)

    def test_quiet_single_task_objective_only_changes_count(self):
        self.assertIn("output exactly ONE complete legal simple move", QUIET_SINGLE_SYSTEM_PROMPT)
        self.assertNotIn("STRONGEST", QUIET_SINGLE_SYSTEM_PROMPT)
        self.assertNotIn("strategically best move", QUIET_SINGLE_SYSTEM_PROMPT)

    def test_jump_single_output_count_instruction(self):
        self.assertIn("Output exactly ONE move in the", JUMP_SINGLE_SYSTEM_PROMPT)
        self.assertIn("Never output more than one", JUMP_SINGLE_SYSTEM_PROMPT)

    def test_quiet_single_output_count_instruction(self):
        self.assertIn("Output exactly ONE move in the", QUIET_SINGLE_SYSTEM_PROMPT)
        self.assertIn("Never output more than one", QUIET_SINGLE_SYSTEM_PROMPT)

    def test_multi_move_still_says_all(self):
        # Guard: ensure multi-move prompts were not accidentally changed
        self.assertIn("output ALL legal capture sequences", JUMP_SYSTEM_PROMPT)
        self.assertIn("Output ALL legal simple moves", QUIET_SYSTEM_PROMPT)


# ══════════════════════════════════════════════════════════════════════════════
# 2 & 3. classify_proposal_single — legality and formatting
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifyProposalSingle(unittest.TestCase):
    """
    classify_proposal_single must mirror classify_proposal gate semantics
    but treat 1 legal move as "perfect" (not "legal_but_incomplete").
    """

    def setUp(self):
        self.legal_moves = _legal_quiet_moves()

    # ── Correctness: 1 legal move = "perfect" ──────────────────────────────

    def test_one_legal_move_is_perfect(self):
        proposed = [{"type": "simple", "path": [[5, 2], [4, 1]], "captured": []}]
        result = classify_proposal_single(proposed, self.legal_moves)
        self.assertEqual(result["classification"], "perfect")
        self.assertEqual(result["legal_proposed"], 1)
        self.assertEqual(result["illegal_proposed"], 0)
        self.assertEqual(result["missing_legal"], 0)

    def test_other_legal_move_also_perfect(self):
        proposed = [{"type": "simple", "path": [[5, 2], [4, 3]], "captured": []}]
        result = classify_proposal_single(proposed, self.legal_moves)
        self.assertEqual(result["classification"], "perfect")

    # ── No invented moves: illegal move = "proposal_illegal" ───────────────

    def test_invented_move_is_proposal_illegal(self):
        # [5,2] → [3,0] is not a legal simple move (2-square jump step)
        proposed = [{"type": "simple", "path": [[5, 2], [3, 0]], "captured": []}]
        result = classify_proposal_single(proposed, self.legal_moves)
        self.assertEqual(result["classification"], "proposal_illegal")
        self.assertEqual(result["illegal_proposed"], 1)
        self.assertEqual(result["legal_proposed"], 0)

    def test_out_of_bounds_move_is_proposal_illegal(self):
        proposed = [{"type": "simple", "path": [[5, 2], [4, 9]], "captured": []}]
        result = classify_proposal_single(proposed, self.legal_moves)
        self.assertEqual(result["classification"], "proposal_illegal")

    def test_wrong_type_is_proposal_illegal(self):
        # Jump when only simples are legal
        proposed = [{"type": "jump", "path": [[5, 2], [3, 0]], "captured": [[4, 1]]}]
        result = classify_proposal_single(proposed, self.legal_moves)
        self.assertEqual(result["classification"], "proposal_illegal")

    # ── Empty proposal ──────────────────────────────────────────────────────

    def test_empty_proposal_is_empty_proposal(self):
        result = classify_proposal_single([], self.legal_moves)
        self.assertEqual(result["classification"], "empty_proposal")
        self.assertEqual(result["proposed_count"], 0)

    # ── Parity: single legal must NOT produce "legal_but_incomplete" ────────

    def test_not_legal_but_incomplete_for_single_legal_move(self):
        """
        classify_proposal([one_legal_move], [n_legal_moves]) returns
        "legal_but_incomplete" — the whole reason classify_proposal_single exists.
        """
        proposed = [{"type": "simple", "path": [[5, 2], [4, 1]], "captured": []}]
        multi_result = classify_proposal(proposed, self.legal_moves)
        single_result = classify_proposal_single(proposed, self.legal_moves)
        self.assertEqual(multi_result["classification"], "legal_but_incomplete")
        self.assertEqual(single_result["classification"], "perfect")

    # ── Jump position ───────────────────────────────────────────────────────

    def test_single_legal_jump_is_perfect(self):
        legal = [_legal_jump_move()]
        proposed = [_legal_jump_move()]
        result = classify_proposal_single(proposed, legal)
        self.assertEqual(result["classification"], "perfect")

    def test_invented_jump_is_proposal_illegal(self):
        legal = [_legal_jump_move()]
        # Non-existent jump path
        proposed = [{"type": "jump", "path": [[5, 2], [3, 0]], "captured": [[4, 1]]}]
        result = classify_proposal_single(proposed, legal)
        self.assertEqual(result["classification"], "proposal_illegal")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Legality parity — gate semantics match multi-move mode
# ══════════════════════════════════════════════════════════════════════════════

class TestLegalityParity(unittest.TestCase):
    """
    A move classified as legal by classify_proposal must also be legal by
    classify_proposal_single and vice versa. The classification logic differs
    only in what constitutes "correct" output count, not in legality judgement.
    """

    def _make_legal_set(self) -> list[dict]:
        return [
            {"type": "simple", "path": [[5, 2], [4, 1]], "captured": []},
            {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []},
            {"type": "simple", "path": [[3, 4], [2, 3]], "captured": []},
        ]

    def test_legal_move_legal_in_both_classifiers(self):
        legal = self._make_legal_set()
        proposed = [legal[0]]
        mr = classify_proposal(proposed, legal)
        sr = classify_proposal_single(proposed, legal)
        self.assertEqual(mr["legal_proposed"], 1)
        self.assertEqual(sr["legal_proposed"], 1)

    def test_illegal_move_illegal_in_both_classifiers(self):
        legal = self._make_legal_set()
        illegal_move = {"type": "simple", "path": [[0, 0], [1, 1]], "captured": []}
        proposed = [illegal_move]
        mr = classify_proposal(proposed, legal)
        sr = classify_proposal_single(proposed, legal)
        self.assertEqual(mr["illegal_proposed"], 1)
        self.assertEqual(sr["illegal_proposed"], 1)
        self.assertEqual(sr["classification"], "proposal_illegal")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Deterministic outputs — build functions stable across identical inputs
# ══════════════════════════════════════════════════════════════════════════════

class TestDeterministicPromptBuilding(unittest.TestCase):

    def test_build_jump_single_deterministic(self):
        board = _board_with_jump()
        sys1, usr1 = build_jump_single_best_prompt(board, RED)
        sys2, usr2 = build_jump_single_best_prompt(board, RED)
        self.assertEqual(sys1, sys2)
        self.assertEqual(usr1, usr2)

    def test_build_quiet_single_deterministic(self):
        board = _board_with_quiet_moves()
        sys1, usr1 = build_quiet_single_best_prompt(board, RED)
        sys2, usr2 = build_quiet_single_best_prompt(board, RED)
        self.assertEqual(sys1, sys2)
        self.assertEqual(usr1, usr2)

    @staticmethod
    def _extract_board_grid(user_prompt: str) -> str:
        """Return only the rendered board grid (between 'BOARD:\\n' and the
        following blank line).  The final instruction lines differ between
        single and multi-move user prompts and must NOT be compared here."""
        after_board = user_prompt.split("BOARD:\n", 1)[1]
        # Board grid ends at the first blank line (empty line before instructions)
        grid, _, _ = after_board.partition("\n\n")
        return grid

    def test_single_and_multi_prompts_share_same_board_rendering(self):
        """Board grid in user prompt must be identical between modes."""
        board = _board_with_quiet_moves()
        _, usr_single = build_quiet_single_best_prompt(board, RED)
        _, usr_multi  = build_quiet_prompt(board, RED)
        self.assertIn("BOARD:", usr_single)
        self.assertIn("BOARD:", usr_multi)
        self.assertEqual(
            self._extract_board_grid(usr_single),
            self._extract_board_grid(usr_multi),
        )

    def test_single_and_multi_jump_share_same_board_rendering(self):
        board = _board_with_jump()
        _, usr_single = build_jump_single_best_prompt(board, RED)
        _, usr_multi  = build_jump_prompt(board, RED)
        self.assertEqual(
            self._extract_board_grid(usr_single),
            self._extract_board_grid(usr_multi),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6. Evaluator compatibility — evaluate_position single_best routing
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluatorCompatibilitySingleBest(unittest.TestCase):

    def _mock_pipeline(
        self,
        proposal_moves: list[dict],
        original_len: int,
        capture_available: bool = False,
    ) -> dict:
        return {
            "scanner_raw": '{"capture_available": false}',
            "scanner_prediction": capture_available,
            "scanner_api_ok": True,
            "proposal_raw": "{}",
            "proposal_moves": proposal_moves,
            "proposal_api_ok": True,
            "proposal_branch": "jump" if capture_available else "quiet",
            "api_failure": False,
            "parse_failure": False,
            "scanner_parse_failure": False,
            "proposal_parse_failure": False,
            "original_proposal_moves_len": original_len,
        }

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_single_legal_move_classified_perfect(self, mock_sbm, mock_sal, mock_rps):
        board = _board_with_quiet_moves()
        legal_move = {"type": "simple", "path": [[5, 2], [4, 1]], "captured": []}
        mock_rps.return_value = self._mock_pipeline([legal_move], original_len=1)
        mock_sal.return_value = ([], 0.0, None, 0.0)
        mock_sbm.return_value = ({}, 0.0, [], {})

        result = evaluate_position(board=board, current_player=RED, single_best=True)

        self.assertEqual(result["classification"], "perfect")
        self.assertIn(result["quadrant"], (
            "scanner_correct_proposal_correct",
            "scanner_wrong_proposal_correct",
        ))

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_single_illegal_move_classified_proposal_illegal(self, mock_sbm, mock_sal, mock_rps):
        board = _board_with_quiet_moves()
        illegal_move = {"type": "simple", "path": [[5, 2], [3, 0]], "captured": []}
        mock_rps.return_value = self._mock_pipeline([illegal_move], original_len=1)
        mock_sal.return_value = ([], 0.0, None, 0.0)
        mock_sbm.return_value = ({}, 0.0, [], {})

        result = evaluate_position(board=board, current_player=RED, single_best=True)

        self.assertEqual(result["classification"], "proposal_illegal")

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_single_legal_move_never_classified_legal_but_incomplete(self, mock_sbm, mock_sal, mock_rps):
        """
        Core regression: classify_proposal would return "legal_but_incomplete"
        for 1 move out of N legal moves. classify_proposal_single must return
        "perfect" instead.
        """
        board = _board_with_quiet_moves()
        legal_move = {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []}
        mock_rps.return_value = self._mock_pipeline([legal_move], original_len=1)
        mock_sal.return_value = ([], 0.0, None, 0.0)
        mock_sbm.return_value = ({}, 0.0, [], {})

        result = evaluate_position(board=board, current_player=RED, single_best=True)

        self.assertNotEqual(result["classification"], "legal_but_incomplete")
        self.assertEqual(result["classification"], "perfect")

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_multi_mode_still_uses_classify_proposal(self, mock_sbm, mock_sal, mock_rps):
        """
        Multi-move mode must not be affected — classify_proposal still applies.
        1 legal move out of N should remain "legal_but_incomplete".
        """
        board = _board_with_quiet_moves()
        one_legal = {"type": "simple", "path": [[5, 2], [4, 1]], "captured": []}
        mock_rps.return_value = self._mock_pipeline([one_legal], original_len=1)
        mock_sal.return_value = ([], 0.0, None, 0.0)
        mock_sbm.return_value = ({}, 0.0, [], {})

        result = evaluate_position(board=board, current_player=RED, single_best=False)

        self.assertEqual(result["classification"], "legal_but_incomplete")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Violation detection — LLM returning >1 move is flagged
# ══════════════════════════════════════════════════════════════════════════════

class TestSingleModeViolationDetection(unittest.TestCase):

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_two_moves_is_violation(self, mock_sbm, mock_sal, mock_rps):
        board = _board_with_quiet_moves()
        move1 = {"type": "simple", "path": [[5, 2], [4, 1]], "captured": []}
        # Proposer truncates internally; original_len captures the true count
        mock_rps.return_value = {
            "scanner_raw": '{"capture_available": false}',
            "scanner_prediction": False,
            "scanner_api_ok": True,
            "proposal_raw": "{}",
            "proposal_moves": [move1],  # already truncated to first
            "proposal_api_ok": True,
            "proposal_branch": "quiet",
            "api_failure": False,
            "parse_failure": False,
            "scanner_parse_failure": False,
            "proposal_parse_failure": False,
            "original_proposal_moves_len": 2,  # LLM returned 2
        }
        mock_sal.return_value = ([], 0.0, None, 0.0)
        mock_sbm.return_value = ({}, 0.0, [], {})

        result = evaluate_position(board=board, current_player=RED, single_best=True)
        self.assertEqual(result["classification"], "single_best_violation")

    @patch("checkers.eval.proposal_seperation_eval.run_proposal_seperation")
    @patch("checkers.eval.proposal_seperation_eval.score_all_legal_moves")
    @patch("checkers.eval.proposal_seperation_eval.select_best_move")
    def test_exactly_one_move_is_not_violation(self, mock_sbm, mock_sal, mock_rps):
        board = _board_with_quiet_moves()
        legal_move = {"type": "simple", "path": [[5, 2], [4, 1]], "captured": []}
        mock_rps.return_value = {
            "scanner_raw": '{"capture_available": false}',
            "scanner_prediction": False,
            "scanner_api_ok": True,
            "proposal_raw": "{}",
            "proposal_moves": [legal_move],
            "proposal_api_ok": True,
            "proposal_branch": "quiet",
            "api_failure": False,
            "parse_failure": False,
            "scanner_parse_failure": False,
            "proposal_parse_failure": False,
            "original_proposal_moves_len": 1,
        }
        mock_sal.return_value = ([], 0.0, None, 0.0)
        mock_sbm.return_value = ({}, 0.0, [], {})

        result = evaluate_position(board=board, current_player=RED, single_best=True)
        self.assertNotEqual(result["classification"], "single_best_violation")


if __name__ == "__main__":
    import unittest
    unittest.main()
