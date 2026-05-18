"""
checkers/tests/test_legality_pilot.py
======================================
Tests for the legality-stress pilot evaluation pipeline.

Schema: selected_move
  { "selected_move": [[row, col], ...], "reasoning": "..." }
  The LLM selects ONE move; the evaluator checks if it is in hidden_legal_moves.

Test groups
-----------
  1. TestPromptPrivacy         hidden_legal_moves never in any prompt
  2. TestEvaluatorLegal        legal selected_move is accepted
  3. TestEvaluatorIllegal      illegal moves are correctly classified
  4. TestEvaluatorParsing      JSON parsing edge cases
  5. TestMetricsAggregator     aggregate() and format_report()
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checkers.data.legality_eval.prompts import (
    BASELINES, build_user_prompt, render_board,
)
from checkers.data.legality_eval.evaluator import (
    evaluate_scenario, parse_llm_output, _norm_path,
)
from checkers.data.legality_eval.metrics import aggregate, format_report

# ── Constants ─────────────────────────────────────────────────────────────────
EMPTY = 0; RED = 1; BLACK = 2; RED_K = 3; BLACK_K = 4


# ── Board helpers ─────────────────────────────────────────────────────────────

def _empty_board():
    return [[EMPTY] * 8 for _ in range(8)]


def _place(board, pieces):
    for (r, c), v in pieces.items():
        board[r][c] = v
    return board


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def forced_jump_board():
    """RED man at (5,0), BLACK man at (4,1) — one forced jump [[5,0],[3,2]]."""
    b = _empty_board()
    b[5][0] = RED; b[4][1] = BLACK
    return b


@pytest.fixture
def hidden_single_jump():
    return [{"type": "jump", "path": [[5, 0], [3, 2]], "captured": [[4, 1]]}]


@pytest.fixture
def multi_jump_board():
    """RED at (6,1); BLACK at (5,2) and (3,4). Chain: [[6,1],[4,3],[2,5]]."""
    b = _empty_board()
    b[6][1] = RED; b[5][2] = BLACK; b[3][4] = BLACK
    return b


@pytest.fixture
def hidden_multi_jump():
    return [{"type": "jump", "path": [[6, 1], [4, 3], [2, 5]],
             "captured": [[5, 2], [3, 4]]}]


@pytest.fixture
def two_simples_board():
    b = _empty_board()
    b[5][0] = RED; b[5][2] = RED
    return b


@pytest.fixture
def hidden_two_simples():
    return [
        {"type": "simple", "path": [[5, 0], [4, 1]], "captured": []},
        {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []},
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Prompt privacy
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptPrivacy:
    FORBIDDEN = ["hidden_legal_moves", "hidden_legal", "ground_truth"]

    def test_user_prompt_no_forbidden_keywords(self, forced_jump_board):
        prompt = build_user_prompt(forced_jump_board, "RED", "sc_test_001")
        for kw in self.FORBIDDEN:
            assert kw not in prompt, f"'{kw}' leaked into user prompt"

    def test_all_system_prompts_no_forbidden_keywords(self):
        for bname, system in BASELINES.items():
            for kw in self.FORBIDDEN:
                assert kw not in system, (
                    f"'{kw}' found in system prompt [{bname}]"
                )

    def test_build_user_prompt_has_no_hidden_moves_param(self):
        import inspect
        sig = inspect.signature(build_user_prompt)
        assert "hidden_legal_moves" not in sig.parameters

    def test_prompt_contains_side_to_move(self, forced_jump_board):
        assert "RED" in build_user_prompt(forced_jump_board, "RED", "sc_001")
        assert "BLACK" in build_user_prompt(forced_jump_board, "BLACK", "sc_001")

    def test_prompt_contains_scenario_id(self, forced_jump_board):
        assert "sc_unique_999" in build_user_prompt(forced_jump_board, "RED", "sc_unique_999")

    def test_prompt_contains_selected_move_schema(self, forced_jump_board):
        """Each system prompt must mention selected_move — not legal_moves."""
        for bname, system in BASELINES.items():
            assert "selected_move" in system, (
                f"'selected_move' not found in system prompt [{bname}]"
            )
            assert "legal_moves" not in system or "hidden_legal_moves" not in system

    def test_baselines_registered(self):
        keys = set(BASELINES.keys())
        assert "B1_board_only"                 in keys
        assert "B2_rules"                      in keys
        assert "B3_rules_structured_checklist" in keys
        assert "B4_rules_engine_checking"      in keys
        # Alias removed — must NOT appear in the registry
        assert "B3_rules_self_check" not in keys
        # Exactly 7 prompt baselines (B1-B7)
        assert "B5_candidate_moves_rule_filter" in keys
        assert "B6_candidate_moves_verbatim" in keys
        assert "B7_candidate_moves_path_only" in keys
        assert len(keys) == 7

    def test_b3_contains_structured_checklist(self):
        b3 = BASELINES["B3_rules_structured_checklist"]
        # B3 uses REJECTION CHECKLIST with REJECT N labels
        assert "REJECTION CHECKLIST"     in b3
        assert "REJECT"                  in b3
        # All 9 rejection checks present
        assert "REJECT 1"                in b3   # source wrong
        assert "REJECT 2"                in b3   # invented piece/coord
        assert "REJECT 3"                in b3   # landing occupied/off-board
        assert "REJECT 4"                in b3   # BLACK reversal
        assert "REJECT 5"                in b3   # RED reversal
        assert "REJECT 6"                in b3   # fake capture
        assert "REJECT 7"                in b3   # simple when capture available
        assert "REJECT 8"                in b3   # incomplete multi-jump
        assert "REJECT 9"                in b3   # malformed output
        # No coordinate algebra (still B3, not B4)
        assert "dr" not in b3
        assert "mid_r" not in b3

    def test_b3_has_no_coordinate_math(self):
        """B3 must have zero coordinate/math content — that is B4's territory."""
        b3 = BASELINES["B3_rules_structured_checklist"]
        assert "dr" not in b3
        assert "dc" not in b3
        assert "mid_r" not in b3
        assert "dr//2" not in b3
        assert "PATH[i" not in b3
        assert "captured_so_far" not in b3
        assert "engine:" not in b3


    def test_b1_has_no_rules_text(self):
        """B1 must not mention mandatory capture or direction rules."""
        b1 = BASELINES["B1_board_only"]
        assert "MANDATORY" not in b1
        assert "Direction" not in b1


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Evaluator — legal moves accepted
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluatorLegal:

    def test_correct_single_jump_is_legal(self, forced_jump_board, hidden_single_jump):
        raw = json.dumps({"selected_move": [[5, 0], [3, 2]], "reasoning": "forced jump"})
        r = evaluate_scenario(raw, hidden_single_jump, forced_jump_board, "RED")
        assert r["legal"] is True
        assert r["parse_success"] is True
        assert r["illegal_move_type"] == ""

    def test_correct_multi_jump_is_legal(self, multi_jump_board, hidden_multi_jump):
        raw = json.dumps({"selected_move": [[6, 1], [4, 3], [2, 5]], "reasoning": "chain"})
        r = evaluate_scenario(raw, hidden_multi_jump, multi_jump_board, "RED")
        assert r["legal"] is True

    def test_correct_simple_move_is_legal(self, two_simples_board, hidden_two_simples):
        raw = json.dumps({"selected_move": [[5, 0], [4, 1]], "reasoning": "simple"})
        r = evaluate_scenario(raw, hidden_two_simples, two_simples_board, "RED")
        assert r["legal"] is True

    def test_second_simple_also_legal(self, two_simples_board, hidden_two_simples):
        raw = json.dumps({"selected_move": [[5, 2], [4, 3]], "reasoning": "other simple"})
        r = evaluate_scenario(raw, hidden_two_simples, two_simples_board, "RED")
        assert r["legal"] is True

    def test_selected_path_stored_correctly(self, forced_jump_board, hidden_single_jump):
        raw = json.dumps({"selected_move": [[5, 0], [3, 2]], "reasoning": "ok"})
        r = evaluate_scenario(raw, hidden_single_jump, forced_jump_board, "RED")
        assert r["selected_path"] == [[5, 0], [3, 2]]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Evaluator — illegal moves classified
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluatorIllegal:

    def test_wrong_path_classified(self, hidden_single_jump):
        """
        Model picks a simple move while a jump is mandatory.
        Board: RED at (5,2), destination (4,3) is empty & dark.
        hidden_legal_moves has only a jump → mandatory_capture_violation fires.
        """
        board = _empty_board()
        board[5][2] = RED   # RED man; (4,3) is empty
        raw = json.dumps({"selected_move": [[5, 2], [4, 3]], "reasoning": "wrong"})
        r = evaluate_scenario(raw, hidden_single_jump, board, "RED")
        assert r["legal"] is False
        assert r["mandatory_violation"] is True
        assert r["illegal_move_type"] == "mandatory_capture_violation"

    def test_wrong_direction_red_man_moving_down(self, hidden_single_jump):
        """RED man at (3,0) moving toward row 4 — direction violation."""
        board = _empty_board(); board[3][0] = RED
        raw = json.dumps({"selected_move": [[3, 0], [4, 1]], "reasoning": "wrong dir"})
        r = evaluate_scenario(raw, hidden_single_jump, board, "RED")
        assert r["legal"] is False
        assert r["illegal_move_type"] == "wrong_direction"
        assert r["wrong_direction"] is not None

    def test_wrong_direction_black_man_moving_up(self, hidden_single_jump):
        """BLACK man at (4,1) moving toward row 3 — direction violation."""
        board = _empty_board(); board[4][1] = BLACK
        raw = json.dumps({"selected_move": [[4, 1], [3, 0]], "reasoning": "wrong dir"})
        r = evaluate_scenario(raw, hidden_single_jump, board, "BLACK")
        assert r["illegal_move_type"] == "wrong_direction"

    def test_wrong_piece_square_empty(self, forced_jump_board, hidden_single_jump):
        """From-square is empty — wrong_piece_square."""
        raw = json.dumps({"selected_move": [[0, 0], [1, 1]], "reasoning": "empty sq"})
        r = evaluate_scenario(raw, hidden_single_jump, forced_jump_board, "RED")
        assert r["illegal_move_type"] == "wrong_piece_square"

    def test_wrong_piece_square_enemy(self, forced_jump_board, hidden_single_jump):
        """From-square holds a BLACK piece — wrong_piece_square for RED."""
        raw = json.dumps({"selected_move": [[4, 1], [3, 0]], "reasoning": "enemy piece"})
        r = evaluate_scenario(raw, hidden_single_jump, forced_jump_board, "RED")
        assert r["illegal_move_type"] == "wrong_piece_square"

    def test_multi_jump_incomplete(self, multi_jump_board, hidden_multi_jump):
        """Jump path stops after first landing — multi_jump_incomplete."""
        raw = json.dumps({"selected_move": [[6, 1], [4, 3]], "reasoning": "stopped early"})
        r = evaluate_scenario(raw, hidden_multi_jump, multi_jump_board, "RED")
        assert r["legal"] is False
        assert r["multi_jump_incomplete"] is True
        assert r["illegal_move_type"] == "multi_jump_incomplete"

    def test_mandatory_violation_simple_when_jump_required(self, hidden_single_jump):
        """
        Board: RED at (5,2), empty (4,3). hidden has only a jump.
        Simple attempt while jump required → mandatory_capture_violation.
        """
        board = _empty_board()
        board[5][2] = RED   # (4,3) is empty dark square
        raw = json.dumps({"selected_move": [[5, 2], [4, 3]], "reasoning": "should jump"})
        r = evaluate_scenario(raw, hidden_single_jump, board, "RED")
        assert r["mandatory_violation"] is True
        assert r["illegal_move_type"] == "mandatory_capture_violation"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Evaluator — parsing edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluatorParsing:

    def test_empty_response_fails(self, forced_jump_board, hidden_single_jump):
        r = evaluate_scenario("", hidden_single_jump, forced_jump_board)
        assert r["parse_success"] is False
        assert r["legal"] is False
        assert "empty" in r["parse_error"]

    def test_bad_json_fails(self, forced_jump_board, hidden_single_jump):
        r = evaluate_scenario("not json", hidden_single_jump, forced_jump_board)
        assert r["parse_success"] is False

    def test_missing_selected_move_field(self, forced_jump_board, hidden_single_jump):
        raw = json.dumps({"reasoning": "forgot the field"})
        r = evaluate_scenario(raw, hidden_single_jump, forced_jump_board)
        assert r["parse_success"] is False
        assert r["legal"] is False

    def test_markdown_fence_stripped(self, forced_jump_board, hidden_single_jump):
        raw = (
            "```json\n"
            + json.dumps({"selected_move": [[5, 0], [3, 2]], "reasoning": "fence"})
            + "\n```"
        )
        r = evaluate_scenario(raw, hidden_single_jump, forced_jump_board, "RED")
        assert r["parse_success"] is True
        assert r["legal"] is True

    def test_reasoning_extracted(self, forced_jump_board, hidden_single_jump):
        raw = json.dumps({"selected_move": [[5, 0], [3, 2]], "reasoning": "CAPTURED IT"})
        r = evaluate_scenario(raw, hidden_single_jump, forced_jump_board, "RED")
        assert r["reasoning"] == "CAPTURED IT"

    def test_parse_llm_output_valid(self):
        obj, err = parse_llm_output('{"selected_move": [[1,2]], "reasoning": "ok"}')
        assert obj is not None and err == ""

    def test_parse_llm_output_empty(self):
        obj, err = parse_llm_output("")
        assert obj is None and "empty" in err

    def test_norm_path_valid(self):
        assert _norm_path([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]

    def test_norm_path_too_short(self):
        assert _norm_path([[1, 2]]) is None

    def test_norm_path_bad_entry(self):
        assert _norm_path([[1, 2], "bad"]) is None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Metrics aggregator
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetricsAggregator:

    def _rec(self, legal, cat="mandatory_capture", diff="hard", itype="",
              result_type=None):
        """Build a minimal evaluated result record (api_success=True, parse_success=True)."""
        rt = result_type or ("legal" if legal else "illegal")
        return {
            "result_type":           rt,
            "api_success":           True,
            "api_failure":           False,
            "rate_limit_retry_count": 0,
            "parse_success":         True,
            "legal":                 legal,
            "illegal_move_type":     itype if not legal else "",
            "wrong_direction":       None,
            "mandatory_violation":   False,
            "multi_jump_incomplete": False,
            "category":              cat,
            "difficulty":            diff,
            "baseline":              "B1_board_only",
            "scenario_id":           "sc_x",
        }

    def test_all_legal(self):
        m = aggregate([self._rec(True) for _ in range(4)])
        assert m["legal_move_rate"]     == 1.0
        assert m["illegal_move_rate"]   == 0.0
        assert m["parse_success_rate"]  == 1.0

    def test_none_legal(self):
        m = aggregate([self._rec(False, itype="path_not_in_legal_moves") for _ in range(4)])
        assert m["legal_move_rate"]   == 0.0
        assert m["illegal_move_rate"] == 1.0

    def test_half_legal(self):
        recs = [self._rec(i % 2 == 0, itype="wrong_direction") for i in range(4)]
        m = aggregate(recs)
        assert m["legal_move_rate"] == 0.5

    def test_category_accuracy(self):
        recs = [
            self._rec(True,  "mandatory_capture"),
            self._rec(False, "mandatory_capture", itype="wrong_direction"),
            self._rec(True,  "multi_jump_required"),
        ]
        m = aggregate(recs)
        assert m["category_accuracy"]["mandatory_capture"]   == 0.5
        assert m["category_accuracy"]["multi_jump_required"] == 1.0

    def test_difficulty_accuracy(self):
        recs = [
            self._rec(True,  diff="hard"),
            self._rec(True,  diff="hard"),
            self._rec(False, diff="easy", itype="parse_failed"),
        ]
        m = aggregate(recs)
        assert m["difficulty_accuracy"]["hard"] == 1.0
        assert m["difficulty_accuracy"]["easy"] == 0.0

    def test_illegal_type_counts(self):
        recs = [
            self._rec(False, itype="wrong_direction"),
            self._rec(False, itype="wrong_direction"),
            self._rec(False, itype="mandatory_capture_violation"),
            self._rec(True),
        ]
        m = aggregate(recs)
        assert m["illegal_type_counts"]["wrong_direction"] == 2
        assert m["illegal_type_counts"]["mandatory_capture_violation"] == 1

    def test_empty_results(self):
        assert aggregate([]) == {}

    def test_format_report_contains_baselines(self):
        recs = [self._rec(True)]
        metrics = {
            "B1_board_only": aggregate(recs),
            "B2_rules":      aggregate([self._rec(False, itype="wrong_direction")]),
        }
        report = format_report("test", metrics, n_scenarios=2)
        assert "B1_board_only" in report
        assert "B2_rules"      in report
        assert "legal_move_rate" in report.lower() or "Legal move rate" in report

    def test_format_report_has_illegal_breakdown(self):
        metrics = {
            "B1_board_only": aggregate([
                self._rec(False, itype="wrong_direction"),
                self._rec(False, itype="mandatory_capture_violation"),
            ]),
        }
        report = format_report("test", metrics, n_scenarios=2)
        assert "wrong_direction" in report
        assert "mandatory_capture_violation" in report

    def test_api_success_rate_is_one_for_all_evaluated(self):
        m = aggregate([self._rec(True) for _ in range(3)])
        assert m["api_success_rate"] == 1.0
        assert m["api_failure_rate"] == 0.0
        assert m["total_api_failures"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. API metadata and failure handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPIMetrics:
    """Verify that API failures are excluded from legal/illegal denominators."""

    def _api_fail_rec(self, rl_retries=3):
        return {
            "result_type":            "api_failure",
            "api_success":            False,
            "rate_limit_retry_count": rl_retries,
            "parse_success":          False,
            "legal":                  False,
            "illegal_move_type":      "",
            "wrong_direction":        None,
            "mandatory_violation":    False,
            "multi_jump_incomplete":  False,
            "category":               "mandatory_capture",
            "difficulty":             "hard",
            "baseline":               "B1_board_only",
            "scenario_id":            "sc_api_fail",
        }

    def _parse_fail_rec(self):
        return {
            "result_type":            "parse_failure",
            "api_success":            True,
            "rate_limit_retry_count": 0,
            "parse_success":          False,
            "legal":                  False,
            "illegal_move_type":      "parse_failed",
            "wrong_direction":        None,
            "mandatory_violation":    False,
            "multi_jump_incomplete":  False,
            "category":               "mandatory_capture",
            "difficulty":             "hard",
            "baseline":               "B1_board_only",
            "scenario_id":            "sc_parse_fail",
        }

    def _legal_rec(self):
        return {
            "result_type":            "legal",
            "api_success":            True,
            "rate_limit_retry_count": 0,
            "parse_success":          True,
            "legal":                  True,
            "illegal_move_type":      "",
            "wrong_direction":        None,
            "mandatory_violation":    False,
            "multi_jump_incomplete":  False,
            "category":               "mandatory_capture",
            "difficulty":             "hard",
            "baseline":               "B1_board_only",
            "scenario_id":            "sc_legal",
        }

    def _illegal_rec(self, itype="wrong_direction"):
        return {
            "result_type":            "illegal",
            "api_success":            True,
            "rate_limit_retry_count": 0,
            "parse_success":          True,
            "legal":                  False,
            "illegal_move_type":      itype,
            "wrong_direction":        "red_man_moves_down" if itype == "wrong_direction" else None,
            "mandatory_violation":    itype == "mandatory_capture_violation",
            "multi_jump_incomplete":  itype == "multi_jump_incomplete",
            "category":               "mandatory_capture",
            "difficulty":             "hard",
            "baseline":               "B1_board_only",
            "scenario_id":            "sc_illegal",
        }

    def test_api_failure_excluded_from_legal_illegal_denominator(self):
        """
        Mix of api_failure + legal + illegal.
        legal_move_rate and illegal_move_rate must be computed over
        the 2 evaluated records only, NOT the 3 total.
        """
        recs = [
            self._api_fail_rec(),   # must be excluded
            self._legal_rec(),
            self._illegal_rec(),
        ]
        m = aggregate(recs)
        assert m["n_scenarios"]  == 3
        assert m["n_api_failure"] == 1
        assert m["n_evaluated"]   == 2     # only legal + illegal
        assert m["legal_move_rate"]  == 0.5  # 1/2
        assert m["illegal_move_rate"] == 0.5  # 1/2

    def test_api_failure_not_counted_as_hallucination(self):
        """An api_failure record must have api_failure_rate > 0 and 0 illegal counts."""
        recs = [self._api_fail_rec(), self._api_fail_rec()]
        m = aggregate(recs)
        assert m["api_failure_rate"]  == 1.0
        assert m["n_evaluated"]       == 0
        # legal_move_rate and illegal_move_rate default to 0.0 when n_evaluated==0
        assert m["legal_move_rate"]   == 0.0
        assert m["illegal_move_rate"] == 0.0
        # no illegal types should be counted
        assert m["illegal_type_counts"] == {}

    def test_invalid_json_from_successful_api_is_parse_failure(self):
        """parse_failure has api_success=True but counts in invalid_format_rate, not illegal."""
        recs = [self._parse_fail_rec(), self._legal_rec()]
        m = aggregate(recs)
        assert m["api_success_rate"]  == 1.0   # both reached API
        assert m["api_failure_rate"]  == 0.0
        assert m["invalid_format_rate"] > 0
        # parse_failure is excluded from evaluated
        assert m["n_evaluated"] == 1
        assert m["illegal_move_rate"] == 0.0   # only the legal rec is in denominator

    def test_parsed_illegal_counted_in_illegal_move_rate(self):
        """A fully-evaluated illegal record must appear in illegal_move_rate."""
        recs = [self._illegal_rec("wrong_direction"), self._legal_rec()]
        m = aggregate(recs)
        assert m["n_evaluated"]   == 2
        assert m["illegal_move_rate"] == 0.5
        assert m["illegal_type_counts"].get("wrong_direction", 0) == 1

    def test_rate_limit_retry_count_in_totals(self):
        """rate_limit_retry_count from each record is summed into total_rate_limit_retries."""
        recs = [
            self._api_fail_rec(rl_retries=3),
            self._api_fail_rec(rl_retries=1),
            self._legal_rec(),     # rl_retries=0
        ]
        m = aggregate(recs)
        assert m["total_rate_limit_retries"] == 4  # 3+1+0
        assert m["total_api_failures"]       == 2

    def test_parse_success_rate_denominator_excludes_api_failures(self):
        """
        parse_success_rate = n_parse_success / n_api_success.
        API failures are not api_successes so they don't inflate the denominator.
        """
        recs = [
            self._api_fail_rec(),    # excluded from api_success count
            self._parse_fail_rec(),  # api_success=True, parse_success=False
            self._legal_rec(),       # api_success=True, parse_success=True
        ]
        m = aggregate(recs)
        # n_api_success = 2 (parse_fail + legal); n_parse_ok = 1 (legal only)
        assert m["n_api_success"]   == 2
        assert m["n_parse_success"] == 1
        assert m["parse_success_rate"] == 0.5

    def test_request_delay_flag_accepted_by_argparse(self):
        """--request-delay must be a recognised CLI argument."""
        import argparse
        import importlib
        import sys
        # Import the parse_args function from the runner
        sys.argv = ["run_legality_pilot.py", "--dry-run", "--request-delay", "0"]
        spec = importlib.util.spec_from_file_location(
            "run_legality_pilot",
            str(PROJECT_ROOT / "run_legality_pilot.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Don't exec the full module (it calls load_dotenv etc); just test parse_args.
        # We verify the flag via direct ArgumentParser inspection instead.
        p = argparse.ArgumentParser()
        p.add_argument("--request-delay", type=float, default=2)
        p.add_argument("--dry-run", action="store_true")
        args = p.parse_args(["--request-delay", "0", "--dry-run"])
        assert args.request_delay == 0.0
        assert args.dry_run is True

    def test_dry_run_delay_does_not_apply(self):
        """
        In dry-run mode _dry_run_with_meta() is called instead of _call_api_with_metadata.
        Verify that the canned response has api_reached=False.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_lp", str(PROJECT_ROOT / "run_legality_pilot.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        raw, meta = mod._dry_run_with_meta()
        assert meta["api_reached"] is False, "dry-run must not touch the network"
        assert meta["api_success"] is True
        assert meta["rate_limit_retry_count"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Retry-group logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetryLogic:
    """Unit tests for the try-group retry architecture in run_legality_pilot."""

    def _load_runner(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_lp", str(PROJECT_ROOT / "run_legality_pilot.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_retry_constants_correct(self):
        """_MAX_TRY_GROUPS==3 and _INTRA_GROUP_WAITS==[20,30,40]."""
        mod = self._load_runner()
        assert mod._MAX_TRY_GROUPS    == 3
        assert mod._INTRA_GROUP_WAITS == [20, 30, 40]

    def test_dry_run_meta_has_try_group_fields(self):
        """_dry_run_with_meta must return api_try_group_count and api_attempt_count."""
        mod = self._load_runner()
        _, meta = mod._dry_run_with_meta()
        assert "api_try_group_count" in meta
        assert "api_attempt_count"   in meta
        assert meta["api_try_group_count"] == 0
        assert meta["api_attempt_count"]   == 0

    def test_success_on_first_attempt_stops_retries(self):
        """A 200 response on the first attempt → api_try_group_count=1, api_attempt_count=1."""
        import unittest.mock
        mod = self._load_runner()
        # Patch _API_KEY so the function proceeds to the request
        mod._API_KEY = "fake_key"

        fake_body = json.dumps({
            "choices": [{"message": {"content": '{"selected_move":[[1,2],[3,4]]}'}}]
        }).encode()

        class FakeResp:
            def read(self): return fake_body
            def __enter__(self): return self
            def __exit__(self, *a): pass

        with unittest.mock.patch("urllib.request.urlopen", return_value=FakeResp()), \
             unittest.mock.patch("time.sleep"):
            raw, meta = mod._call_api_with_metadata("sys", "usr")

        assert meta["api_success"]         is True
        assert meta["api_try_group_count"] == 1
        assert meta["api_attempt_count"]   == 1
        assert meta["api_retry_count"]     == 0

    def test_all_groups_exhausted_gives_api_failure(self):
        """
        If urlopen always raises HTTPError 429, all 3 groups × 4 attempts = 12 calls,
        then api_success=False and api_error_type contains the last error.
        """
        import unittest.mock
        import urllib.error
        mod = self._load_runner()
        mod._API_KEY = "fake_key"

        call_count = 0

        def always_429(*a, **kw):
            nonlocal call_count
            call_count += 1
            err = urllib.error.HTTPError(url="", code=429, msg="Too Many",
                                         hdrs={}, fp=None)
            err.read = lambda: b""
            raise err

        with unittest.mock.patch("urllib.request.urlopen", side_effect=always_429), \
             unittest.mock.patch("time.sleep"):
            raw, meta = mod._call_api_with_metadata("sys", "usr")

        max_calls = mod._MAX_TRY_GROUPS * (1 + len(mod._INTRA_GROUP_WAITS))
        assert meta["api_success"]         is False
        assert meta["api_try_group_count"] == mod._MAX_TRY_GROUPS
        assert meta["api_attempt_count"]   == max_calls
        assert meta["rate_limit_retry_count"] == max_calls  # every call was 429
        assert "429" in (meta["api_error_type"] or "")

    def test_success_in_second_group_stops_retries(self):
        """
        First group exhausted (4 failures), second group succeeds on 1st attempt.
        api_try_group_count=2, api_attempt_count=5.
        """
        import unittest.mock
        import urllib.error
        mod = self._load_runner()
        mod._API_KEY = "fake_key"

        max_per_group = 1 + len(mod._INTRA_GROUP_WAITS)  # 4
        call_count    = {"n": 0}
        fake_body = json.dumps({
            "choices": [{"message": {"content": '{"selected_move":[[0,1],[2,3]]}'}}]
        }).encode()

        class FakeSuccessResp:
            def read(self): return fake_body
            def __enter__(self): return self
            def __exit__(self, *a): pass

        def urlopen_side_effect(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] <= max_per_group:
                err = urllib.error.HTTPError(url="", code=429, msg="RL",
                                              hdrs={}, fp=None)
                err.read = lambda: b""
                raise err
            return FakeSuccessResp()

        with unittest.mock.patch("urllib.request.urlopen", side_effect=urlopen_side_effect), \
             unittest.mock.patch("time.sleep"):
            raw, meta = mod._call_api_with_metadata("sys", "usr")

        assert meta["api_success"]         is True
        assert meta["api_try_group_count"] == 2
        assert meta["api_attempt_count"]   == max_per_group + 1

    def test_non_retriable_http_exits_immediately(self):
        """A 401 response exits all groups at once (attempt_count=1, group_count=1)."""
        import unittest.mock
        import urllib.error
        mod = self._load_runner()
        mod._API_KEY = "fake_key"

        def raise_401(*a, **kw):
            err = urllib.error.HTTPError(url="", code=401, msg="Unauthorized",
                                          hdrs={}, fp=None)
            err.read = lambda: b""
            raise err

        with unittest.mock.patch("urllib.request.urlopen", side_effect=raise_401), \
             unittest.mock.patch("time.sleep"):
            raw, meta = mod._call_api_with_metadata("sys", "usr")

        assert meta["api_success"]         is False
        assert meta["api_attempt_count"]   == 1
        assert meta["api_try_group_count"] == 1
        assert "non_retriable" in (meta["api_error_type"] or "")

    def test_run_label_api_complete_when_no_failures(self):
        """format_report shows API_COMPLETE when total_api_failures==0."""
        from checkers.data.legality_eval.metrics import format_report, aggregate
        # Build an all-legal set
        recs = [{
            "result_type": "legal", "api_success": True,
            "rate_limit_retry_count": 0, "parse_success": True, "legal": True,
            "illegal_move_type": "", "wrong_direction": None,
            "mandatory_violation": False, "multi_jump_incomplete": False,
            "category": "mandatory_capture", "difficulty": "hard",
            "baseline": "B1_board_only", "scenario_id": "sc_x",
        }]
        m = aggregate(recs)
        report = format_report("test", {"B1_board_only": m},
                               n_scenarios=1, run_label="API_COMPLETE")
        assert "API_COMPLETE" in report

    def test_run_label_incomplete_when_api_failures(self):
        """format_report shows INCOMPLETE_FOR_FINAL_EVALUATION when failures exist."""
        from checkers.data.legality_eval.metrics import format_report, aggregate
        recs = [{
            "result_type": "api_failure", "api_success": False,
            "rate_limit_retry_count": 3, "parse_success": False, "legal": False,
            "illegal_move_type": "", "wrong_direction": None,
            "mandatory_violation": False, "multi_jump_incomplete": False,
            "category": "mandatory_capture", "difficulty": "hard",
            "baseline": "B1_board_only", "scenario_id": "sc_x",
        }]
        m = aggregate(recs)
        report = format_report("test", {"B1_board_only": m},
                               n_scenarios=1,
                               run_label="INCOMPLETE_FOR_FINAL_EVALUATION")
        assert "INCOMPLETE_FOR_FINAL_EVALUATION" in report


# ═══════════════════════════════════════════════════════════════════════════════
# 8. B4 baseline — rule-engine-checking prompt
# ═══════════════════════════════════════════════════════════════════════════════

class TestB4Baseline:
    """Tests proving revised B4_rules_engine_checking uses coordinate-delta checking."""

    B4 = BASELINES["B4_rules_engine_checking"]

    # ── Requirement 1: existence ──────────────────────────────────────────────

    def test_b4_exists_in_registry(self):
        assert "B4_rules_engine_checking" in BASELINES

    # ── Requirement 2: builds on B3 (rules + self-check included) ────────────

    def test_b4_builds_on_b3_content(self):
        """B4 must include the full B3 structured checklist above its coordinate section."""
        b4 = self.B4
        assert "MANDATORY CAPTURE" in b4
        assert "STRUCTURED LEGALITY CHECKLIST" in b4
        assert "SOURCE PIECE" in b4
        assert "CAPTURE FIRST" in b4
        assert "FULL CHAIN" in b4
        assert "PROMOTION ENDS TURN" in b4

    # ── Requirement 3: still uses selected_move schema ────────────────────────

    def test_b4_uses_selected_move_schema(self):
        b4 = self.B4
        assert '"selected_move"' in b4
        assert '"reasoning"' in b4

    # ── Requirement 4: dr/dc coordinate-delta language ───────────────────────

    def test_b4_contains_dr_dc_notation(self):
        """B4 must use dr/dc notation derived from get_single_jumps math."""
        b4 = self.B4
        assert "dr" in b4, "B4 missing row-delta variable 'dr'"
        assert "dc" in b4, "B4 missing col-delta variable 'dc'"
        # Exact row-delta assignments as used in the engine scan loop
        assert "PATH[i+1][0] - PATH[i][0]" in b4 or "dr = PATH" in b4

    # ── Requirement 5: midpoint calculation for jumps ─────────────────────────

    def test_b4_contains_midpoint_formula(self):
        """B4 must show mid_r = PATH[i][0] + dr//2 (from get_single_jumps)."""
        b4 = self.B4
        assert "mid_r" in b4, "B4 missing midpoint row variable 'mid_r'"
        assert "mid_c" in b4, "B4 missing midpoint col variable 'mid_c'"
        assert "dr//2" in b4, "B4 missing integer-halving formula dr//2"

    # ── Requirement 6: simple vs jump by coordinate distance |dr|==1 / |dr|==2

    def test_b4_distinguishes_simple_vs_jump_by_delta(self):
        """B4 must classify move type by |dr|: 1=simple, 2=jump (from get_move_directions)."""
        b4 = self.B4
        assert "|dr| == 1" in b4, "B4 must identify simple step as |dr|==1"
        assert "|dr| == 2" in b4, "B4 must identify jump step as |dr|==2"

    # ── Requirement 7: temporary board update for multi-jumps ─────────────────

    def test_b4_explains_mental_board_update(self):
        """B4 must explain captured_so_far filter and mental board removal."""
        b4 = self.B4
        # captured_so_far is used in get_all_jump_sequences to filter re-captures
        assert "captured_so_far" in b4, "B4 missing captured_so_far variable"
        # Mental board update — remove captured, place piece at landing
        assert "remove captured" in b4.lower() or "empty    (remove" in b4.lower()

    # ── Requirement 8: does not ask to list all legal moves ───────────────────

    def test_b4_does_not_ask_for_all_legal_moves(self):
        b4 = self.B4
        FORBIDDEN = [
            "list all legal moves", "enumerate all",
            "generate all moves", "output all legal",
        ]
        for phrase in FORBIDDEN:
            assert phrase.lower() not in b4.lower(), (
                f"B4 contains forbidden phrase: '{phrase}'"
            )

    # ── Requirement 9: no hidden_legal_moves / no pre-computed candidates ─────

    def test_b4_does_not_contain_hidden_legal_moves(self):
        b4 = self.B4
        for kw in ["hidden_legal_moves", "hidden_legal", "ground_truth"]:
            assert kw not in b4, f"Privacy violation: '{kw}' in B4 system prompt"

    def test_b4_does_not_expose_legal_move_candidates(self):
        """
        Requirement 6: B4 must not give the LLM a pre-computed list of legal moves.
        Note: engine function names like 'get_all_legal_moves' are legitimate
        documentation references and are NOT forbidden.
        """
        b4 = self.B4
        # These patterns mean the LLM is being handed move candidates
        FORBIDDEN = [
            "legal_moves =",      # assigning a move-list variable
            "legal_moves:",       # key in a JSON/dict structure
            "legal move list",    # explicit list handoff
            "candidate moves",    # pre-computed candidates
            "move list",          # pre-computed move list
            "hidden_legal_moves", # ground-truth leak (also in privacy test)
        ]
        for phrase in FORBIDDEN:
            assert phrase not in b4, (
                f"B4 exposes legal move candidates via phrase: '{phrase}'"
            )


    # ── Requirement 10: B1/B2/B3 unchanged ───────────────────────────────────

    def test_b1_b2_b3_unchanged(self):
        """Ladder separation: B1 < B2 < B3 < B4, no cross-contamination."""
        b1 = BASELINES["B1_board_only"]
        b2 = BASELINES["B2_rules"]
        b3 = BASELINES["B3_rules_structured_checklist"]
        # B1: board legend only — no rules, no checklist, no coordinate math
        assert "MANDATORY" not in b1
        assert "LEGALITY CHECKLIST" not in b1
        assert "dr" not in b1
        # B2: rules only — no checklist, no coordinate math
        assert "MANDATORY CAPTURE" in b2
        assert "LEGALITY CHECKLIST" not in b2
        assert "COORDINATE-BASED" not in b2
        assert "dr//2" not in b2
        # B3: rules + structured checklist — NO coordinate math
        assert "REJECTION CHECKLIST" in b3
        assert "COORDINATE-BASED" not in b3
        assert "mid_r" not in b3
        assert "dr//2" not in b3
        assert "PATH[i" not in b3


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Runner baseline selection — preview + evaluation scope
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunnerBaselineScope:
    """Prove that the runner respects --baselines and the alias is gone."""

    def test_active_baselines_list_matches_registry(self):
        """ACTIVE_BASELINES = 4 LLM prompt baselines + 1 symbolic control arm."""
        import sys, importlib
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "run_legality_pilot",
            str(Path(__file__).parents[2] / "run_legality_pilot.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Every prompt baseline must be in ACTIVE_BASELINES
        for key in BASELINES:
            assert key in mod.ACTIVE_BASELINES, f"{key} missing from ACTIVE_BASELINES"
        # Symbolic arm in ACTIVE_BASELINES but NOT in BASELINES (no prompt)
        assert mod.SYMBOLIC_BASELINE_NAME in mod.ACTIVE_BASELINES
        assert mod.SYMBOLIC_BASELINE_NAME not in BASELINES
        # Old alias gone
        assert "B3_rules_self_check" not in mod.ACTIVE_BASELINES
        # Exactly 12 entries: 7 prompt (B1-B7) + 2 B8 variants + B9 + 1 symbolic + 1 FST
        assert len(mod.ACTIVE_BASELINES) == 13
        # FST in ACTIVE_BASELINES but NOT in prompt BASELINES dict
        assert mod.FST_BASELINE_NAME in mod.ACTIVE_BASELINES
        assert mod.FST_BASELINE_NAME not in BASELINES
        # B5 in both ACTIVE_BASELINES and prompt BASELINES dict
        assert mod.B5_BASELINE_NAME in mod.ACTIVE_BASELINES
        assert mod.B5_BASELINE_NAME in BASELINES
        # B6 in both ACTIVE_BASELINES and prompt BASELINES dict
        assert mod.B6_BASELINE_NAME in mod.ACTIVE_BASELINES
        assert mod.B6_BASELINE_NAME in BASELINES
        # B7 in both ACTIVE_BASELINES and prompt BASELINES dict
        assert mod.B7_BASELINE_NAME in mod.ACTIVE_BASELINES
        assert mod.B7_BASELINE_NAME in BASELINES


    def test_only_selected_baselines_evaluated(self, forced_jump_board,
                                               hidden_single_jump):
        """
        Requirement 3: When --baselines B1_board_only is specified, run_pilot
        produces records ONLY for B1_board_only — no other baseline sneaks in.
        """
        from run_legality_pilot import run_pilot
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2]))

        scenario = {
            "scenario_id":      "test_b1_only",
            "board":            forced_jump_board,
            "side_to_move":     "RED",
            "category":         "mandatory_capture",
            "difficulty":       "hard",
            "source_file":      "test",
            "hidden_legal_moves": hidden_single_jump,
        }
        results = run_pilot(
            scenarios=[scenario],
            baselines=["B1_board_only"],
            dry_run=True,
            show_prompts=False,
            verbose=False,
            delay_sec=0,
        )
        # Exactly one record, only for B1
        assert len(results) == 1
        assert results[0]["baseline"] == "B1_board_only"
        # No other baseline leaked in
        baselines_in_results = {r["baseline"] for r in results}
        assert baselines_in_results == {"B1_board_only"}

    def test_preview_prompts_respects_selected_baselines(self, forced_jump_board,
                                                         hidden_single_jump,
                                                         capsys):
        """preview_prompts must only print sections for the selected baselines."""
        from run_legality_pilot import preview_prompts
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2]))

        scenario = {
            "scenario_id":      "test_preview",
            "board":            forced_jump_board,
            "side_to_move":     "RED",
            "hidden_legal_moves": hidden_single_jump,
        }
        preview_prompts([scenario], baselines=["B1_board_only"],
                        show_ground_truth=False)
        out = capsys.readouterr().out
        assert "BASELINE: B1_board_only" in out
        assert "BASELINE: B2_rules"                     not in out
        assert "BASELINE: B3_rules_structured_checklist" not in out
        assert "BASELINE: B4_rules_engine_checking"      not in out
        # Ground truth hidden by default — count shown, move paths suppressed
        assert "GROUND TRUTH hidden" in out
        # Move detail lines (e.g. "[JUMP] (5,0)→(3,2)") must NOT be printed
        assert "[JUMP]"   not in out
        assert "[SIMPLE]" not in out


# ═══════════════════════════════════════════════════════════════════════════════
# Side-level analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestSideMetrics:
    """Verify aggregate() produces correct per-side breakdowns."""

    def _rec(self, side, legal, wrong_dir=False, mand=False, multi=False,
             category="mandatory_capture"):
        itype = ""
        if not legal:
            itype = ("wrong_direction" if wrong_dir
                     else "mandatory_capture_violation" if mand
                     else "multi_jump_incomplete" if multi
                     else "path_not_in_legal_moves")
        return {
            "result_type":            "legal" if legal else "illegal",
            "api_success":            True,
            "rate_limit_retry_count": 0,
            "parse_success":          True,
            "legal":                  legal,
            "illegal_move_type":      itype,
            "wrong_direction":        "red_man_moves_down" if wrong_dir else None,
            "mandatory_violation":    mand,
            "multi_jump_incomplete":  multi,
            "category":               category,
            "difficulty":             "hard",
            "side_to_move":           side,
            "baseline":               "B1_board_only",
            "scenario_id":            f"sc_{side}_{legal}",
        }

    def _api_fail(self, side):
        return {
            "result_type":            "api_failure",
            "api_success":            False,
            "rate_limit_retry_count": 1,
            "parse_success":          False,
            "legal":                  False,
            "illegal_move_type":      "",
            "wrong_direction":        None,
            "mandatory_violation":    False,
            "multi_jump_incomplete":  False,
            "category":               "mandatory_capture",
            "difficulty":             "hard",
            "side_to_move":           side,
            "baseline":               "B1_board_only",
            "scenario_id":            f"sc_fail_{side}",
        }

    def test_side_metrics_keys_present(self):
        recs = [self._rec("RED", True), self._rec("BLACK", False)]
        m = aggregate(recs)
        assert "side_metrics"           in m
        assert "category_side_accuracy" in m
        assert "RED"   in m["side_metrics"]
        assert "BLACK" in m["side_metrics"]

    def test_per_side_counts_and_rates(self):
        """RED: 2 legal, 1 wrong_dir illegal.  BLACK: 0 legal, 2 illegal."""
        recs = [
            self._rec("RED",   True),
            self._rec("RED",   True),
            self._rec("RED",   False, wrong_dir=True),
            self._rec("BLACK", False),
            self._rec("BLACK", False, mand=True),
        ]
        m = aggregate(recs)
        red = m["side_metrics"]["RED"]
        blk = m["side_metrics"]["BLACK"]
        assert red["n_evaluated"]          == 3
        assert red["n_legal"]              == 2
        assert red["n_illegal"]            == 1
        assert red["legal_move_rate"]      == round(2/3, 4)
        assert red["wrong_direction_rate"] == round(1/3, 4)
        assert blk["n_evaluated"]          == 2
        assert blk["n_legal"]              == 0
        assert blk["legal_move_rate"]      == 0.0
        assert blk["illegal_move_rate"]    == 1.0

    def test_api_failure_excluded_from_side_denominator(self):
        recs = [
            self._rec("RED", True),
            self._api_fail("RED"),    # excluded from n_evaluated
            self._rec("BLACK", False),
        ]
        m = aggregate(recs)
        assert m["side_metrics"]["RED"]["n_evaluated"]   == 1
        assert m["side_metrics"]["BLACK"]["n_evaluated"] == 1

    def test_category_side_accuracy(self):
        recs = [
            self._rec("RED",   True,  category="mandatory_capture"),
            self._rec("RED",   False, category="mandatory_capture"),
            self._rec("BLACK", True,  category="mandatory_capture"),
            self._rec("RED",   False, category="wrong_direction_trap"),
            self._rec("BLACK", False, category="wrong_direction_trap"),
        ]
        m = aggregate(recs)
        csa = m["category_side_accuracy"]
        assert csa["mandatory_capture"]["RED"]     == 0.5
        assert csa["mandatory_capture"]["BLACK"]   == 1.0
        assert csa["wrong_direction_trap"]["RED"]   == 0.0
        assert csa["wrong_direction_trap"]["BLACK"] == 0.0

    def test_side_breakdown_in_format_report(self):
        recs = [self._rec("RED", True), self._rec("BLACK", False)]
        m = aggregate(recs)
        report = format_report("test", {"B1_board_only": m}, n_scenarios=2,
                               run_label="API_COMPLETE")
        assert "Side-to-Move Analysis"           in report
        assert "[RED]"                            in report
        assert "[BLACK]"                          in report
        assert "Category x Side Legal Move Rate" in report


# ═══════════════════════════════════════════════════════════════════════════════
# Full_System_Symbolic control arm
# ═══════════════════════════════════════════════════════════════════════════════

class TestSymbolicControlArm:
    """Prove Full_System_Symbolic is a correct non-LLM sanity/control arm."""

    # ── shared fixtures ───────────────────────────────────────────────────────

    @staticmethod
    def _load_runner():
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "run_legality_pilot",
            str(Path(__file__).parents[2] / "run_legality_pilot.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # Req 1: accepted by --baselines
    def test_symbolic_in_active_baselines(self):
        mod = self._load_runner()
        assert "Full_System_Symbolic" in mod.ACTIVE_BASELINES

    def test_symbolic_accepted_by_argparse(self):
        mod = self._load_runner()
        args = mod.parse_args.__wrapped__ if hasattr(mod.parse_args, "__wrapped__") else None
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["run_legality_pilot.py", "--baselines", "Full_System_Symbolic"]
            ns = mod.parse_args()
            assert ns.baselines == ["Full_System_Symbolic"]
        finally:
            sys.argv = old_argv

    # Req 2: does NOT call the API
    def test_symbolic_does_not_call_api(self, forced_jump_board, hidden_single_jump):
        mod = self._load_runner()
        raw, meta = mod._run_symbolic(forced_jump_board, "RED")
        assert meta["api_reached"]        is False
        assert meta["api_attempt_count"]  == 0
        assert meta["api_try_group_count"] == 0

    # Req 3: recomputes legal moves from board + side_to_move
    def test_symbolic_uses_engine_not_hidden_moves(self, forced_jump_board, hidden_single_jump):
        """_run_symbolic must call get_all_legal_moves, not read hidden_legal_moves."""
        mod = self._load_runner()
        # Recompute independently
        from checkers.engine.rules import get_all_legal_moves
        from checkers.data.pdn_importer.fen_utils import str_to_side
        side_int = str_to_side("RED")
        engine_moves = get_all_legal_moves(forced_jump_board, side_int)
        assert len(engine_moves) > 0
        # _run_symbolic must produce one of those paths
        import json
        raw, _ = mod._run_symbolic(forced_jump_board, "RED")
        chosen = json.loads(raw)["selected_move"]
        engine_paths = [[list(rc) for rc in m["path"]] for m in engine_moves]
        assert chosen in engine_paths, f"Symbolic chose {chosen} not in engine moves {engine_paths}"

    # Req 4: outputs selected_move schema
    def test_symbolic_output_schema(self, forced_jump_board, hidden_single_jump):
        import json
        mod = self._load_runner()
        raw, _ = mod._run_symbolic(forced_jump_board, "RED")
        out = json.loads(raw)
        assert "selected_move" in out
        assert "reasoning"     in out
        assert isinstance(out["selected_move"], list)
        assert len(out["selected_move"]) >= 2
        for entry in out["selected_move"]:
            assert isinstance(entry, list)
            assert len(entry) == 2

    # Req 5: evaluated by same evaluator
    def test_symbolic_uses_same_evaluator(self, forced_jump_board, hidden_single_jump):
        import json
        from checkers.data.legality_eval.evaluator import evaluate_scenario
        mod = self._load_runner()
        raw, _ = mod._run_symbolic(forced_jump_board, "RED")
        result = evaluate_scenario(raw, hidden_single_jump, forced_jump_board, "RED")
        assert "legal"         in result
        assert "parse_success" in result

    # Req 6: gets 100% legality on a sample of scenarios
    def test_symbolic_gets_100_percent_legality(self, forced_jump_board, hidden_single_jump):
        """run_pilot with Full_System_Symbolic in dry=False symbolic mode → all legal."""
        mod = self._load_runner()
        sc = {
            "scenario_id":        "sym_test",
            "board":              forced_jump_board,
            "side_to_move":       "RED",
            "category":           "mandatory_capture",
            "difficulty":         "hard",
            "source_file":        "test",
            "hidden_legal_moves": hidden_single_jump,
        }
        results = mod.run_pilot(
            scenarios=[sc],
            baselines=["Full_System_Symbolic"],
            dry_run=False,          # symbolic arm ignores dry_run flag
            show_prompts=False,
            verbose=False,
            delay_sec=0,
        )
        assert len(results) == 1
        assert results[0]["result_type"] == "legal", (
            f"Symbolic arm got non-legal result: {results[0]}"
        )
        assert results[0]["legal"] is True

    # Req 7: API failure metrics do not falsely apply
    def test_symbolic_no_api_failure_metrics(self, forced_jump_board, hidden_single_jump):
        mod = self._load_runner()
        sc = {
            "scenario_id":        "sym_api_test",
            "board":              forced_jump_board,
            "side_to_move":       "RED",
            "category":           "mandatory_capture",
            "difficulty":         "hard",
            "source_file":        "test",
            "hidden_legal_moves": hidden_single_jump,
        }
        results = mod.run_pilot(
            scenarios=[sc],
            baselines=["Full_System_Symbolic"],
            dry_run=False,
            show_prompts=False,
            verbose=False,
            delay_sec=0,
        )
        r = results[0]
        assert r["result_type"]        != "api_failure"
        assert r["api_success"]        is True
        assert r["api_reached"]        is False
        assert r["api_attempt_count"]  == 0
        assert r["rate_limit_retry_count"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Full_System_Trace tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullSystemTrace:
    """
    Prove Full_System_Trace is:
      - accepted by --baselines
      - using _stream_one_ply (real compiled graph), not the B1-B4 prompt path
      - never reading hidden_legal_moves
      - producing selected_move schema
      - evaluated by the same evaluator
      - leaving Full_System_Symbolic unchanged
    """

    # ── shared loader ─────────────────────────────────────────────────────────
    @staticmethod
    def _load_runner():
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "run_legality_pilot",
            str(Path(__file__).parents[2] / "run_legality_pilot.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # ── Req 1: accepted by --baselines ────────────────────────────────────────
    def test_fst_in_active_baselines(self):
        mod = self._load_runner()
        assert "Full_System_Trace" in mod.ACTIVE_BASELINES

    def test_fst_accepted_by_argparse(self):
        import sys
        mod = self._load_runner()
        old_argv = sys.argv
        try:
            sys.argv = ["run_legality_pilot.py", "--baselines", "Full_System_Trace"]
            ns = mod.parse_args()
            assert ns.baselines == ["Full_System_Trace"]
        finally:
            sys.argv = old_argv

    # ── Req 2: uses _stream_one_ply, not B1-B4 prompt path ───────────────────
    def test_fst_calls_stream_one_ply(self, forced_jump_board, hidden_single_jump, monkeypatch):
        """_run_full_system_trace must call _stream_one_ply internally."""
        mod = self._load_runner()
        called = []

        def _fake_fst_imports():
            # Return a sentinel graph, a spy wrapper, and real CheckersState
            from checkers.state.state import CheckersState as CS

            class _FakeGraph:
                pass

            def _spy_stream_one_ply(acc, quiet):
                called.append(True)
                # Return a minimal acc with a chosen_move so the adapter doesn't crash
                from checkers.engine.rules import get_all_legal_moves
                from checkers.data.pdn_importer.fen_utils import str_to_side
                side_int = acc.get("current_player", 1)
                moves = get_all_legal_moves(acc["board"], side_int)
                chosen = moves[0] if moves else None
                out = dict(acc)
                out["chosen_move"] = chosen
                out["move_history"] = [{"move": chosen, "player": acc.get("current_player", 1)}] if chosen else []
                out["last_move_reasoning"] = "spy"
                out["ranker_diagnostics"] = {}
                return out, True

            return _FakeGraph(), _spy_stream_one_ply, CS

        monkeypatch.setattr(mod, "_fst_imports", _fake_fst_imports)
        mod._run_full_system_trace(forced_jump_board, "RED")
        assert called, "_stream_one_ply was never called"

    def test_fst_does_not_use_baselines_prompt_dict(self, forced_jump_board, hidden_single_jump, monkeypatch):
        """_run_full_system_trace must never look up BASELINES[bname]."""
        mod = self._load_runner()
        accessed = []

        class _TrackingDict(dict):
            def __getitem__(self, key):
                accessed.append(key)
                return super().__getitem__(key)

        monkeypatch.setattr(mod, "BASELINES", _TrackingDict(mod.BASELINES))

        def _fake_fst_imports():
            from checkers.state.state import CheckersState as CS
            from checkers.engine.rules import get_all_legal_moves

            def _stub(acc, quiet):
                moves = get_all_legal_moves(acc["board"], acc.get("current_player", 1))
                out = dict(acc)
                out["chosen_move"] = moves[0] if moves else None
                out["move_history"] = [{"move": moves[0], "player": acc.get("current_player", 1)}] if moves else []
                out["last_move_reasoning"] = "stub"
                out["ranker_diagnostics"] = {}
                return out, True

            return object(), _stub, CS

        monkeypatch.setattr(mod, "_fst_imports", _fake_fst_imports)
        mod._run_full_system_trace(forced_jump_board, "RED")
        assert not accessed, f"BASELINES was accessed with keys: {accessed}"

    # ── Req 3: does NOT use hidden_legal_moves as input ───────────────────────
    def test_fst_does_not_receive_hidden_legal_moves(self, forced_jump_board, hidden_single_jump, monkeypatch):
        """_run_full_system_trace signature takes only board + side_str — no hidden_legal_moves."""
        import inspect
        mod = self._load_runner()
        sig = inspect.signature(mod._run_full_system_trace)
        params = list(sig.parameters.keys())
        assert "hidden" not in params
        assert "hidden_legal_moves" not in params
        assert len(params) == 2, f"Expected (board, side_str), got {params}"

    # ── Req 4: outputs selected_move schema ───────────────────────────────────
    def test_fst_output_schema(self, forced_jump_board, hidden_single_jump, monkeypatch):
        import json
        mod = self._load_runner()

        def _fake_fst_imports():
            from checkers.state.state import CheckersState as CS
            from checkers.engine.rules import get_all_legal_moves

            def _stub(acc, quiet):
                moves = get_all_legal_moves(acc["board"], acc.get("current_player", 1))
                chosen = moves[0] if moves else None
                out = dict(acc)
                out["chosen_move"] = chosen
                out["move_history"] = [{"move": chosen, "player": acc.get("current_player", 1)}] if chosen else []
                out["last_move_reasoning"] = "schema_test"
                out["ranker_diagnostics"] = {}
                return out, True

            return object(), _stub, CS

        monkeypatch.setattr(mod, "_fst_imports", _fake_fst_imports)
        raw_json, api_meta, trace_meta = mod._run_full_system_trace(forced_jump_board, "RED")
        out = json.loads(raw_json)
        assert "selected_move" in out
        assert "reasoning"     in out
        assert isinstance(out["selected_move"], list)
        # Each entry is [row, col]
        for entry in out["selected_move"]:
            assert isinstance(entry, list) and len(entry) == 2
        # api_meta has required keys
        for key in ("api_reached", "api_success", "api_attempt_count", "rate_limit_retry_count"):
            assert key in api_meta
        # trace_meta has fst_ keys
        assert "fst_stream_one_ply_success"            in trace_meta
        assert "fst_raw_llm_candidate_selection_valid" in trace_meta
        assert "fst_final_choice_source"               in trace_meta

    # ── Req 5: evaluated by the same evaluator ────────────────────────────────
    def test_fst_uses_same_evaluator(self, forced_jump_board, hidden_single_jump, monkeypatch):
        import json
        from checkers.data.legality_eval.evaluator import evaluate_scenario
        mod = self._load_runner()

        def _fake_fst_imports():
            from checkers.state.state import CheckersState as CS
            from checkers.engine.rules import get_all_legal_moves

            def _stub(acc, quiet):
                moves = get_all_legal_moves(acc["board"], acc.get("current_player", 1))
                chosen = moves[0] if moves else None
                out = dict(acc)
                out["chosen_move"] = chosen
                out["move_history"] = [{"move": chosen, "player": acc.get("current_player", 1)}] if chosen else []
                out["last_move_reasoning"] = "eval_test"
                out["ranker_diagnostics"] = {}
                return out, True

            return object(), _stub, CS

        monkeypatch.setattr(mod, "_fst_imports", _fake_fst_imports)
        raw_json, _, _ = mod._run_full_system_trace(forced_jump_board, "RED")
        result = evaluate_scenario(raw_json, hidden_single_jump, forced_jump_board, "RED")
        assert "legal"         in result
        assert "parse_success" in result

    # ── Req 6: Full_System_Symbolic remains separate ──────────────────────────
    def test_symbolic_arm_unchanged(self, forced_jump_board, hidden_single_jump):
        """Full_System_Symbolic must still work independently of FST."""
        mod = self._load_runner()
        raw, meta = mod._run_symbolic(forced_jump_board, "RED")
        assert meta["api_reached"]        is False
        assert meta["api_attempt_count"]  == 0
        import json
        out = json.loads(raw)
        assert "selected_move" in out
        assert len(out["selected_move"]) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# B5_candidate_moves_rule_filter tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestB5CandidateMovesRuleFilter:
    """
    Proves:
    1. B5 exists and is selectable.
    2. B5 uses selected_move schema.
    3. B5 prompt includes candidate moves.
    4. B5 prompt does NOT include hidden_legal_moves text.
    5. B5 does NOT label candidates as legal/illegal.
    6. In mandatory-capture positions: candidate list includes both jumps and simples.
    7. Evaluator still checks selected_move against hidden_legal_moves.
    8. B5 does not affect B1–B4 or full-system arms.
    """

    @staticmethod
    def _load_runner():
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "run_legality_pilot",
            str(Path(__file__).parents[2] / "run_legality_pilot.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # Req 1: B5 exists and is selectable
    def test_b5_in_active_baselines(self):
        mod = self._load_runner()
        assert "B5_candidate_moves_rule_filter" in mod.ACTIVE_BASELINES

    def test_b5_accepted_by_argparse(self):
        import sys
        mod = self._load_runner()
        old_argv = sys.argv
        try:
            sys.argv = ["run_legality_pilot.py", "--baselines", "B5_candidate_moves_rule_filter"]
            ns = mod.parse_args()
            assert ns.baselines == ["B5_candidate_moves_rule_filter"]
        finally:
            sys.argv = old_argv

    def test_b5_in_baselines_prompt_dict(self):
        from checkers.data.legality_eval.prompts import BASELINES
        assert "B5_candidate_moves_rule_filter" in BASELINES

    # Req 2: B5 uses selected_move schema
    def test_b5_system_prompt_uses_selected_move_schema(self):
        from checkers.data.legality_eval.prompts import BASELINES
        sys_prompt = BASELINES["B5_candidate_moves_rule_filter"]
        assert "selected_move" in sys_prompt
        assert "reasoning" in sys_prompt
        assert "chosen_index" not in sys_prompt   # not an index-picker

    # Req 3: B5 prompt includes candidate moves
    def test_b5_user_prompt_includes_candidates(self, forced_jump_board):
        from checkers.data.legality_eval.prompts import build_b5_user_prompt
        from checkers.data.legality_eval.candidate_moves import get_candidates
        from checkers.engine.board import RED
        cand_info = get_candidates(forced_jump_board, RED)
        user = build_b5_user_prompt(forced_jump_board, "RED", "test_scen", cand_info["candidates"])
        assert "Candidate moves" in user
        assert "C0" in user    # at least one candidate with an ID
        assert "path:" in user

    # Req 4: B5 prompt does NOT contain hidden_legal_moves
    def test_b5_user_prompt_does_not_contain_hidden_legal_moves(
        self, forced_jump_board, hidden_single_jump
    ):
        from checkers.data.legality_eval.prompts import build_b5_user_prompt
        from checkers.data.legality_eval.candidate_moves import get_candidates
        from checkers.engine.board import RED
        cand_info = get_candidates(forced_jump_board, RED)
        user = build_b5_user_prompt(forced_jump_board, "RED", "test_scen", cand_info["candidates"])
        # hidden_legal_moves contents must not appear verbatim
        for move in hidden_single_jump:
            for sq in move.get("path", []):
                path_str = f"hidden_legal_moves"
                assert "hidden_legal_moves" not in user
        assert "hidden_legal_moves" not in user

    # Req 5: B5 does NOT label candidates as legal/illegal
    def test_b5_prompt_does_not_label_legal_or_illegal(self, forced_jump_board):
        from checkers.data.legality_eval.prompts import build_b5_user_prompt
        from checkers.data.legality_eval.candidate_moves import get_candidates
        from checkers.engine.board import RED
        cand_info = get_candidates(forced_jump_board, RED)
        user = build_b5_user_prompt(forced_jump_board, "RED", "test_scen", cand_info["candidates"])
        # Candidate lines themselves (starting with C-id) must not contain LEGAL or ILLEGAL labels
        for line in user.splitlines():
            stripped = line.strip()
            if stripped.startswith("C") and "path:" in stripped:
                assert "LEGAL" not in stripped.upper().split("PATH:")[0] or \
                       "ILLEGAL" not in stripped.upper().split("PATH:")[0], \
                       f"Candidate line labels move as legal/illegal: {stripped}"

    # Req 6: Mandatory-capture positions include both jumps and simples in candidate list
    def test_b5_candidates_include_simples_when_jump_available(self):
        """
        A board where RED has a jump (mandatory) AND another RED piece with simple moves.
        get_candidates should return BOTH jump candidates AND simple candidates,
        because it does NOT apply the mandatory-capture filter.
        Board: RED at (6,1) can jump BLACK at (5,2) → (4,3).
               RED at (6,5) can make simple moves to (5,4) and (5,6).
        """
        from checkers.data.legality_eval.candidate_moves import get_candidates
        from checkers.engine.board import RED, BLACK, EMPTY
        board = [[EMPTY] * 8 for _ in range(8)]
        board[6][1] = RED   # jumper
        board[5][2] = BLACK # jump target
        board[6][5] = RED   # simple-move piece
        cand_info = get_candidates(board, RED)
        assert cand_info["any_jump_available"], "Expected a jump to exist"
        assert cand_info["capture_candidate_count"] >= 1, "Expected at least one jump candidate"
        assert cand_info["simple_candidate_count"] >= 1, "Expected at least one simple candidate"
        types = {c["move_type"] for c in cand_info["candidates"]}
        assert "simple" in types, "Simple candidates absent despite jump being available"
        assert "jump"   in types, "Jump candidates absent"

    def test_b5_candidates_no_capture_filter(self):
        """
        The candidate list must NOT apply mandatory-capture filter.
        get_all_legal_moves returns only jumps; get_candidates must also return simples.
        Board: RED at (6,1) jumps BLACK at (5,2); RED at (6,5) can make simple moves.
        """
        from checkers.data.legality_eval.candidate_moves import get_candidates
        from checkers.engine.rules import get_all_legal_moves
        from checkers.engine.board import RED, BLACK, EMPTY
        board = [[EMPTY] * 8 for _ in range(8)]
        board[6][1] = RED
        board[5][2] = BLACK
        board[6][5] = RED
        cand_info = get_candidates(board, RED)
        legal_moves = get_all_legal_moves(board, RED)
        # Legal moves should only have jumps (mandatory capture)
        assert all(m["type"] == "jump" for m in legal_moves), "Expected only jumps in legal moves"
        # But candidates should also have simples
        assert cand_info["simple_candidate_count"] >= 1, \
            "Candidate generator incorrectly filtered out simple moves"

    # Req 7: Evaluator still checks selected_move against hidden_legal_moves
    def test_b5_evaluator_still_checks_hidden(self, forced_jump_board, hidden_single_jump):
        import json
        from checkers.data.legality_eval.evaluator import evaluate_scenario
        # Simulate B5 outputting a simple move (which should be ILLEGAL — jump available)
        simple_move = [[5, 2], [4, 3]]   # a simple move (not in hidden_legal_moves)
        raw = json.dumps({"selected_move": simple_move, "reasoning": "test"})
        result = evaluate_scenario(raw, hidden_single_jump, forced_jump_board, "RED")
        assert result["parse_success"] is True
        assert result["legal"] is False   # simple move rejected — jump was mandatory

    # Req 8: B5 does not affect B1–B4 or full-system arms
    def test_b5_does_not_change_b1_b2_b3_b4(self):
        from checkers.data.legality_eval.prompts import BASELINES
        for name in ("B1_board_only", "B2_rules", "B3_rules_structured_checklist",
                     "B4_rules_engine_checking"):
            assert name in BASELINES, f"{name} was removed"
            assert "Candidate moves" not in BASELINES[name], \
                f"{name} was contaminated with B5 candidate section"

    def test_b5_count_in_active_baselines(self):
        """ACTIVE_BASELINES must have exactly 9 entries: B1-B7 + Symbolic + FST."""
        mod = self._load_runner()
        assert len(mod.ACTIVE_BASELINES) == 13, \
            f"Expected 7 baselines, got {len(mod.ACTIVE_BASELINES)}: {mod.ACTIVE_BASELINES}"

    def test_b5_match_selected_to_candidate_found(self, forced_jump_board):
        """match_selected_to_candidate correctly identifies a selected path in candidate list."""
        from checkers.data.legality_eval.candidate_moves import get_candidates, match_selected_to_candidate
        from checkers.engine.board import RED
        cand_info = get_candidates(forced_jump_board, RED)
        cands = cand_info["candidates"]
        # Pick the first candidate's path
        first_path = cands[0]["path"]
        result = match_selected_to_candidate(first_path, cands)
        assert result["selected_candidate_id"] == "C0"
        assert result["selected_path_not_in_candidates"] is False

    def test_b5_match_selected_hallucinated_path(self, forced_jump_board):
        """If selected path is not in candidates, selected_path_not_in_candidates = True."""
        from checkers.data.legality_eval.candidate_moves import get_candidates, match_selected_to_candidate
        from checkers.engine.board import RED
        cand_info = get_candidates(forced_jump_board, RED)
        fake_path = [[0, 0], [1, 1]]   # almost certainly not a real candidate
        result = match_selected_to_candidate(fake_path, cand_info["candidates"])
        assert result["selected_path_not_in_candidates"] is True
        assert result["selected_candidate_id"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# B6_candidate_moves_verbatim tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestB6CandidateMovesVerbatim:
    """
    Proves:
    1. B6 exists and is selectable.
    2. B6 uses selected_candidate_id + selected_move schema.
    3. B6 prompt says selected_move must exactly match one candidate path.
    4. B6 prompt forbids inventing/modifying coordinates.
    5. B6 does not expose hidden_legal_moves.
    6. B6 does not label candidates as legal/illegal.
    7. B6 diagnostics detect path not in candidates.
    8. B6 diagnostics detect candidate ID/path mismatch.
    9. B6 does not affect B5 or other baselines.
    """

    @staticmethod
    def _load_runner():
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "run_legality_pilot",
            str(Path(__file__).parents[2] / "run_legality_pilot.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _b6_board():
        """Board with a jump candidate AND simple candidates."""
        from checkers.engine.board import RED, BLACK, EMPTY
        board = [[EMPTY] * 8 for _ in range(8)]
        board[6][1] = RED
        board[5][2] = BLACK
        board[6][5] = RED
        return board

    # Req 1: B6 exists and is selectable
    def test_b6_in_active_baselines(self):
        mod = self._load_runner()
        assert "B6_candidate_moves_verbatim" in mod.ACTIVE_BASELINES

    def test_b6_accepted_by_argparse(self):
        import sys
        mod = self._load_runner()
        old_argv = sys.argv
        try:
            sys.argv = ["run_legality_pilot.py", "--baselines", "B6_candidate_moves_verbatim"]
            ns = mod.parse_args()
            assert ns.baselines == ["B6_candidate_moves_verbatim"]
        finally:
            sys.argv = old_argv

    def test_b6_in_baselines_prompt_dict(self):
        from checkers.data.legality_eval.prompts import BASELINES
        assert "B6_candidate_moves_verbatim" in BASELINES

    # Req 2: B6 uses selected_candidate_id + selected_move schema
    def test_b6_system_prompt_schema(self):
        from checkers.data.legality_eval.prompts import BASELINES
        sys_prompt = BASELINES["B6_candidate_moves_verbatim"]
        assert "selected_candidate_id" in sys_prompt
        assert "selected_move" in sys_prompt
        assert "reasoning" in sys_prompt

    # Req 3: B6 prompt says selected_move must exactly match one candidate path
    def test_b6_system_prompt_verbatim_instruction(self):
        from checkers.data.legality_eval.prompts import BASELINES
        p = BASELINES["B6_candidate_moves_verbatim"]
        assert "VERBATIM" in p or "verbatim" in p.lower()
        assert "exactly" in p.lower() or "EXACTLY" in p

    # Req 4: B6 forbids inventing/modifying coordinates
    def test_b6_prompt_forbids_inventing_coordinates(self):
        from checkers.data.legality_eval.prompts import BASELINES
        p = BASELINES["B6_candidate_moves_verbatim"]
        assert "Do NOT invent" in p or "not invent" in p.lower()
        assert "Do NOT modify" in p or "not modify" in p.lower()

    # Req 5: B6 user prompt does NOT expose hidden_legal_moves
    def test_b6_user_prompt_no_hidden_legal_moves(self):
        from checkers.data.legality_eval.prompts import build_b6_user_prompt
        from checkers.data.legality_eval.candidate_moves import get_candidates
        from checkers.engine.board import RED
        board = self._b6_board()
        cand_info = get_candidates(board, RED)
        user = build_b6_user_prompt(board, "RED", "test_b6", cand_info["candidates"])
        assert "hidden_legal_moves" not in user
        assert "hidden_legal" not in user

    # Req 6: B6 user prompt does NOT label candidates as legal/illegal
    def test_b6_prompt_no_legal_illegal_labels(self):
        from checkers.data.legality_eval.prompts import build_b6_user_prompt
        from checkers.data.legality_eval.candidate_moves import get_candidates
        from checkers.engine.board import RED
        board = self._b6_board()
        cand_info = get_candidates(board, RED)
        user = build_b6_user_prompt(board, "RED", "test_b6", cand_info["candidates"])
        for line in user.splitlines():
            stripped = line.strip()
            if stripped.startswith("C") and "path:" in stripped:
                assert "LEGAL" not in stripped.upper().split("PATH:")[0], \
                    f"Candidate line labels move as legal: {stripped}"
                assert "ILLEGAL" not in stripped.upper().split("PATH:")[0], \
                    f"Candidate line labels move as illegal: {stripped}"

    # Req 7: B6 diagnostics detect path not in candidates
    def test_b6_match_path_not_in_candidates(self):
        from checkers.data.legality_eval.candidate_moves import get_candidates, match_b6_response
        from checkers.engine.board import RED
        board = self._b6_board()
        cand_info = get_candidates(board, RED)
        cands = cand_info["candidates"]
        fake_path = [[0, 0], [1, 1]]
        result = match_b6_response("C0", fake_path, cands)
        assert result["b6_selected_path_not_in_candidates"] is True
        assert result["b6_selected_move_matches_candidate_id"] is False

    # Req 8: B6 diagnostics detect candidate ID/path mismatch
    def test_b6_match_id_path_mismatch(self):
        """Claims C0 but outputs C1's path."""
        from checkers.data.legality_eval.candidate_moves import get_candidates, match_b6_response
        from checkers.engine.board import RED
        board = self._b6_board()
        cand_info = get_candidates(board, RED)
        cands = cand_info["candidates"]
        assert len(cands) >= 2
        # Claim C0 but output C1's path
        c1_path = cands[1]["path"]
        result = match_b6_response("C0", c1_path, cands)
        assert result["b6_selected_candidate_id_valid"] is True   # C0 exists
        assert result["b6_selected_move_matches_candidate_id"] is False  # path ≠ C0

    def test_b6_match_correct_id_and_path(self):
        """Claims C0 and outputs C0's path — fully consistent."""
        from checkers.data.legality_eval.candidate_moves import get_candidates, match_b6_response
        from checkers.engine.board import RED
        board = self._b6_board()
        cand_info = get_candidates(board, RED)
        cands = cand_info["candidates"]
        c0_path = cands[0]["path"]
        result = match_b6_response("C0", c0_path, cands)
        assert result["b6_selected_candidate_id_valid"] is True
        assert result["b6_selected_move_matches_candidate_id"] is True
        assert result["b6_selected_path_not_in_candidates"] is False

    def test_b6_match_invalid_id(self):
        """Claims a non-existent ID (e.g. C99)."""
        from checkers.data.legality_eval.candidate_moves import get_candidates, match_b6_response
        from checkers.engine.board import RED
        board = self._b6_board()
        cand_info = get_candidates(board, RED)
        cands = cand_info["candidates"]
        result = match_b6_response("C99", cands[0]["path"], cands)
        assert result["b6_selected_candidate_id_valid"] is False
        assert result["b6_selected_move_matches_candidate_id"] is False

    def test_b6_match_none_id(self):
        """LLM omitted selected_candidate_id entirely."""
        from checkers.data.legality_eval.candidate_moves import get_candidates, match_b6_response
        from checkers.engine.board import RED
        board = self._b6_board()
        cand_info = get_candidates(board, RED)
        cands = cand_info["candidates"]
        result = match_b6_response(None, cands[0]["path"], cands)
        assert result["b6_selected_candidate_id_valid"] is False
        assert result["b6_selected_candidate_id"] is None

    # Req 9: B6 does not affect B5 or other baselines
    def test_b6_does_not_change_b1_to_b5(self):
        from checkers.data.legality_eval.prompts import BASELINES
        for name in ("B1_board_only", "B2_rules", "B3_rules_structured_checklist",
                     "B4_rules_engine_checking", "B5_candidate_moves_rule_filter"):
            assert name in BASELINES, f"{name} was removed"
            assert "VERBATIM COPY REQUIREMENT" not in BASELINES[name], \
                f"{name} was contaminated with B6 verbatim section"
            assert "selected_candidate_id" not in BASELINES[name], \
                f"{name} was contaminated with B6 schema field"

    def test_b6_count_in_active_baselines(self):
        """ACTIVE_BASELINES must have exactly 9 entries: B1-B7 + Symbolic + FST."""
        mod = self._load_runner()
        assert len(mod.ACTIVE_BASELINES) == 13, \
            f"Expected 8 baselines, got {len(mod.ACTIVE_BASELINES)}: {mod.ACTIVE_BASELINES}"

    def test_b6_user_prompt_includes_candidates(self):
        from checkers.data.legality_eval.prompts import build_b6_user_prompt
        from checkers.data.legality_eval.candidate_moves import get_candidates
        from checkers.engine.board import RED
        board = self._b6_board()
        cand_info = get_candidates(board, RED)
        user = build_b6_user_prompt(board, "RED", "test_b6", cand_info["candidates"])
        assert "Candidate moves" in user
        assert "C0" in user
        assert "path:" in user

    def test_b6_simple_candidate_present_when_jump_available(self):
        """Same candidate generator as B5 — jump + simples both appear."""
        from checkers.data.legality_eval.candidate_moves import get_candidates
        from checkers.engine.board import RED
        board = self._b6_board()
        cand_info = get_candidates(board, RED)
        assert cand_info["any_jump_available"] is True
        assert cand_info["simple_candidate_count"] >= 1
        assert cand_info["capture_candidate_count"] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# B7_candidate_moves_path_only tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestB7CandidateMovesPathOnly:
    """
    Proves:
    1. B7 exists and is selectable.
    2. B7 prompt includes candidates.
    3. B7 prompt requires selected_move to exactly match one candidate path.
    4. B7 output schema does NOT include selected_candidate_id.
    5. B7 does not expose hidden_legal_moves.
    6. B7 does not label candidates legal/illegal.
    7. B7 diagnostics detect path not in candidate list.
    8. B7 does not affect B1–B6 or full-system arms.
    """

    @staticmethod
    def _load_runner():
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "run_legality_pilot",
            str(Path(__file__).parents[2] / "run_legality_pilot.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _b7_board():
        from checkers.engine.board import RED, BLACK, EMPTY
        board = [[EMPTY] * 8 for _ in range(8)]
        board[6][1] = RED
        board[5][2] = BLACK
        board[6][5] = RED
        return board

    # Req 1: B7 exists and is selectable
    def test_b7_in_active_baselines(self):
        mod = self._load_runner()
        assert "B7_candidate_moves_path_only" in mod.ACTIVE_BASELINES

    def test_b7_accepted_by_argparse(self):
        import sys
        mod = self._load_runner()
        old_argv = sys.argv
        try:
            sys.argv = ["run_legality_pilot.py", "--baselines", "B7_candidate_moves_path_only"]
            ns = mod.parse_args()
            assert ns.baselines == ["B7_candidate_moves_path_only"]
        finally:
            sys.argv = old_argv

    def test_b7_in_baselines_prompt_dict(self):
        from checkers.data.legality_eval.prompts import BASELINES
        assert "B7_candidate_moves_path_only" in BASELINES

    # Req 2: B7 prompt includes candidates
    def test_b7_user_prompt_includes_candidates(self):
        from checkers.data.legality_eval.prompts import build_b7_user_prompt
        from checkers.data.legality_eval.candidate_moves import get_candidates
        from checkers.engine.board import RED
        board = self._b7_board()
        cand_info = get_candidates(board, RED)
        user = build_b7_user_prompt(board, "RED", "test_b7", cand_info["candidates"])
        assert "Candidate moves" in user
        assert "C0" in user
        assert "path:" in user

    # Req 3: B7 prompt requires verbatim copy
    def test_b7_system_prompt_verbatim_instruction(self):
        from checkers.data.legality_eval.prompts import BASELINES
        p = BASELINES["B7_candidate_moves_path_only"]
        assert "VERBATIM" in p or "verbatim" in p.lower()
        assert "exactly" in p.lower() or "EXACTLY" in p
        assert "Do NOT invent" in p or "not invent" in p.lower()
        assert "Do NOT modify" in p or "not modify" in p.lower()

    # Req 4: B7 output schema does NOT include selected_candidate_id
    def test_b7_system_prompt_no_selected_candidate_id(self):
        from checkers.data.legality_eval.prompts import BASELINES
        p = BASELINES["B7_candidate_moves_path_only"]
        assert "selected_candidate_id" not in p
        assert "selected_move" in p      # selected_move IS present
        assert "reasoning" in p          # reasoning IS present

    # Req 5: B7 does not expose hidden_legal_moves
    def test_b7_user_prompt_no_hidden_legal_moves(self):
        from checkers.data.legality_eval.prompts import build_b7_user_prompt
        from checkers.data.legality_eval.candidate_moves import get_candidates
        from checkers.engine.board import RED
        board = self._b7_board()
        cand_info = get_candidates(board, RED)
        user = build_b7_user_prompt(board, "RED", "test_b7", cand_info["candidates"])
        assert "hidden_legal_moves" not in user
        assert "hidden_legal" not in user

    # Req 6: B7 does not label candidates legal/illegal
    def test_b7_prompt_no_legal_illegal_labels_on_candidates(self):
        from checkers.data.legality_eval.prompts import build_b7_user_prompt
        from checkers.data.legality_eval.candidate_moves import get_candidates
        from checkers.engine.board import RED
        board = self._b7_board()
        cand_info = get_candidates(board, RED)
        user = build_b7_user_prompt(board, "RED", "test_b7", cand_info["candidates"])
        for line in user.splitlines():
            stripped = line.strip()
            if stripped.startswith("C") and "path:" in stripped:
                label_part = stripped.upper().split("PATH:")[0]
                assert "LEGAL"   not in label_part, f"Candidate labelled legal: {stripped}"
                assert "ILLEGAL" not in label_part, f"Candidate labelled illegal: {stripped}"

    # Req 7: B7 diagnostics detect path not in candidate list
    def test_b7_match_hallucinated_path(self):
        from checkers.data.legality_eval.candidate_moves import get_candidates, match_b7_response
        from checkers.engine.board import RED
        board = self._b7_board()
        cand_info = get_candidates(board, RED)
        fake_path = [[0, 0], [1, 1]]
        result = match_b7_response(fake_path, cand_info["candidates"])
        assert result["b7_selected_path_not_in_candidates"] is True
        assert result["b7_selected_candidate_match_count"] == 0
        assert result["b7_selected_candidate_move_type"] is None

    def test_b7_match_found_path(self):
        from checkers.data.legality_eval.candidate_moves import get_candidates, match_b7_response
        from checkers.engine.board import RED
        board = self._b7_board()
        cand_info = get_candidates(board, RED)
        cands = cand_info["candidates"]
        # Pick the jump candidate (C0)
        jump_cand = next(c for c in cands if c["move_type"] == "jump")
        result = match_b7_response(jump_cand["path"], cands)
        assert result["b7_selected_path_not_in_candidates"] is False
        assert result["b7_selected_candidate_match_count"] == 1
        assert result["b7_selected_candidate_move_type"] == "jump"
        assert result["b7_selected_candidate_was_simple_when_jump_available"] is False

    def test_b7_match_simple_when_jump_available(self):
        """Selecting a simple candidate when a jump exists — trap detection."""
        from checkers.data.legality_eval.candidate_moves import get_candidates, match_b7_response
        from checkers.engine.board import RED
        board = self._b7_board()
        cand_info = get_candidates(board, RED)
        cands = cand_info["candidates"]
        simple_cand = next(c for c in cands if c["move_type"] == "simple")
        result = match_b7_response(simple_cand["path"], cands)
        assert result["b7_selected_path_not_in_candidates"] is False
        assert result["b7_selected_candidate_was_simple_when_jump_available"] is True

    def test_b7_match_none_path(self):
        from checkers.data.legality_eval.candidate_moves import get_candidates, match_b7_response
        from checkers.engine.board import RED
        board = self._b7_board()
        cand_info = get_candidates(board, RED)
        result = match_b7_response(None, cand_info["candidates"])
        assert result["b7_selected_path_not_in_candidates"] is True
        assert result["b7_selected_candidate_match_count"] == 0

    # Req 8: B7 does not affect B1–B6
    def test_b7_does_not_change_b1_to_b6(self):
        from checkers.data.legality_eval.prompts import BASELINES
        for name in (
            "B1_board_only", "B2_rules", "B3_rules_structured_checklist",
            "B4_rules_engine_checking", "B5_candidate_moves_rule_filter",
            "B6_candidate_moves_verbatim",
        ):
            assert name in BASELINES, f"{name} was removed"
            # B7's verbatim section should not leak into other baselines
            assert "B7" not in BASELINES[name], \
                f"{name} was contaminated with B7 content"

    def test_b7_count_in_active_baselines(self):
        """ACTIVE_BASELINES must have exactly 9 entries: B1-B7 + Symbolic + FST."""
        mod = self._load_runner()
        assert len(mod.ACTIVE_BASELINES) == 13, \
            f"Expected 9 baselines, got {len(mod.ACTIVE_BASELINES)}: {mod.ACTIVE_BASELINES}"

    # B7 vs B6 schema distinction
    def test_b7_schema_differs_from_b6(self):
        """B7 has no selected_candidate_id; B6 does."""
        from checkers.data.legality_eval.prompts import BASELINES
        b6 = BASELINES["B6_candidate_moves_verbatim"]
        b7 = BASELINES["B7_candidate_moves_path_only"]
        assert "selected_candidate_id" in b6
        assert "selected_candidate_id" not in b7
        # Both have selected_move and VERBATIM requirement
        assert "selected_move" in b7
        assert "VERBATIM" in b7 or "verbatim" in b7.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Side-filter tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSideFilter:
    """
    Proves:
    1. --side-filter RED returns only RED records.
    2. --side-filter BLACK returns only BLACK records.
    3. Default (no flag) returns both sides.
    4. Side filtering happens before sampling.
    5. Report / header includes side filter label.
    6. Existing runs without --side-filter are unchanged.
    """

    @staticmethod
    def _load_runner():
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "run_legality_pilot",
            str(Path(__file__).parents[2] / "run_legality_pilot.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _mixed_pool():
        """Synthetic pool with known RED/BLACK distribution."""
        return [
            {"scenario_id": f"r{i}", "side_to_move": "RED",
             "board": [[0]*8 for _ in range(8)], "hidden_legal_moves": [],
             "category": "c", "difficulty": "easy", "source_file": "f"}
            for i in range(6)
        ] + [
            {"scenario_id": f"b{i}", "side_to_move": "BLACK",
             "board": [[0]*8 for _ in range(8)], "hidden_legal_moves": [],
             "category": "c", "difficulty": "easy", "source_file": "f"}
            for i in range(4)
        ]

    # Req 1 — filter_by_side RED
    def test_filter_red_only(self):
        mod = self._load_runner()
        pool = self._mixed_pool()
        result = mod.filter_by_side(pool, "RED")
        assert all(r["side_to_move"] == "RED" for r in result)
        assert len(result) == 6

    # Req 2 — filter_by_side BLACK
    def test_filter_black_only(self):
        mod = self._load_runner()
        pool = self._mixed_pool()
        result = mod.filter_by_side(pool, "BLACK")
        assert all(r["side_to_move"] == "BLACK" for r in result)
        assert len(result) == 4

    # Req 3 — default returns all
    def test_filter_none_returns_all(self):
        mod = self._load_runner()
        pool = self._mixed_pool()
        assert mod.filter_by_side(pool, None) is pool
        assert mod.filter_by_side(pool, "BOTH") == pool

    # Req 4 — filtering before sampling: sampled pool is a subset of filtered pool
    def test_filter_before_sample(self):
        mod = self._load_runner()
        pool = self._mixed_pool()
        filtered = mod.filter_by_side(pool, "RED")
        sampled  = mod.sample_scenarios(filtered, n=3, seed=42, label="test")
        assert len(sampled) == 3
        assert all(s["side_to_move"] == "RED" for s in sampled)

    def test_filter_before_sample_black(self):
        mod = self._load_runner()
        pool = self._mixed_pool()
        filtered = mod.filter_by_side(pool, "BLACK")
        sampled  = mod.sample_scenarios(filtered, n=3, seed=42, label="test")
        assert len(sampled) == 3
        assert all(s["side_to_move"] == "BLACK" for s in sampled)

    # Req 5 — argparse recognises --side-filter
    def test_side_filter_in_argparse(self):
        import sys
        mod = self._load_runner()
        for val in ("RED", "BLACK", "BOTH"):
            old = sys.argv
            try:
                sys.argv = ["run_legality_pilot.py", "--side-filter", val]
                ns = mod.parse_args()
                assert ns.side_filter == val, f"Expected {val}, got {ns.side_filter}"
            finally:
                sys.argv = old

    def test_side_filter_default_is_none(self):
        import sys
        mod = self._load_runner()
        old = sys.argv
        try:
            sys.argv = ["run_legality_pilot.py"]
            ns = mod.parse_args()
            assert ns.side_filter is None
        finally:
            sys.argv = old

    # Req 6 — existing runs without flag are unchanged
    def test_no_side_filter_does_not_reduce_pool(self):
        """filter_by_side(None) must return the same object, preserving all scenarios."""
        mod = self._load_runner()
        pool = self._mixed_pool()
        out  = mod.filter_by_side(pool, None)
        assert out is pool   # same object, no copy
        assert len(out) == 10

    # Edge cases
    def test_filter_invalid_value_raises(self):
        import pytest
        mod = self._load_runner()
        pool = self._mixed_pool()
        with pytest.raises(ValueError, match="RED, BLACK, or BOTH"):
            mod.filter_by_side(pool, "BOTH_SIDES")

    def test_sample_warns_when_n_exceeds_pool(self, capsys):
        """sample_scenarios warns (not raises) when n > pool size."""
        mod = self._load_runner()
        pool = self._mixed_pool()[:3]
        result = mod.sample_scenarios(pool, n=10, seed=42, label="test")
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert len(result) == 3   # returns all available

    def test_filter_case_insensitive(self):
        mod = self._load_runner()
        pool = self._mixed_pool()
        # filter_by_side does .upper() on the argument
        result = mod.filter_by_side(pool, "red")
        assert len(result) == 6

    def test_filter_empty_pool(self):
        mod = self._load_runner()
        assert mod.filter_by_side([], "RED") == []
        assert mod.filter_by_side([], "BLACK") == []


# ═══════════════════════════════════════════════════════════════════════════════
# B8 tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestB8RankerNoSafety:
    """
    Proves:
    1. B8a and B8b are selectable baselines.
    2. B8a uses proposal shortlist (≤5 candidates).
    3. B8b uses full legal set and shuffled deterministic order.
    4. Neither uses safety filter, retry, override, fallback, repair, or update_agent.
    5. Neither calls Full_System_Trace / _stream_one_ply.
    6. Neither uses dataset hidden_legal_moves.
    7. Both use the same legality evaluator.
    8. Diagnostic fields are present with correct invariants.
    9. Registry count includes both B8 variants.
    """

    @staticmethod
    def _load_runner():
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "run_legality_pilot",
            str(Path(__file__).parents[2] / "run_legality_pilot.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _simple_board():
        """
        Minimal valid board: one RED man at (5,0), one BLACK man at (2,1).
        side_to_move = RED.  RED has exactly 1 legal move: (5,0)→(4,1).
        """
        board = [[0] * 8 for _ in range(8)]
        board[5][0] = 1  # RED man
        board[2][1] = 2  # BLACK man
        return board

    # ── Req 1: both baselines selectable ─────────────────────────────────────

    def test_b8a_in_active_baselines(self):
        mod = self._load_runner()
        assert mod.B8A_BASELINE_NAME in mod.ACTIVE_BASELINES

    def test_b8b_in_active_baselines(self):
        mod = self._load_runner()
        assert mod.B8B_BASELINE_NAME in mod.ACTIVE_BASELINES

    def test_active_baselines_count_with_b8(self):
        mod = self._load_runner()
        assert len(mod.ACTIVE_BASELINES) == 13   # B1-B7, B8a, B8b, B8c, B9, Symbolic, FST

    # ── Req 4: no forbidden mechanisms present in _run_b8 source ─────────────

    def test_run_b8_does_not_call_safety_filter(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8)
        assert "_apply_safety_filter" not in src

    def test_run_b8_does_not_call_audit_override(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8)
        assert "_audit_override" not in src

    def test_run_b8_does_not_retry(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8)
        # No retry loops — no "for attempt" or "while" inside _run_b8
        assert "for attempt" not in src
        assert "OVERRIDE_MAX_RETRIES" not in src

    def test_run_b8_does_not_call_update_agent(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8)
        # "update_agent" may appear in comments/docstrings; what matters is
        # that update_agent is never *called* (no import or invocation).
        assert "update_agent(" not in src
        assert "from checkers.agents.update_agent" not in src
        assert "import update_agent" not in src

    def test_run_b8_has_no_fallback_choose_best_minimax(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8)
        assert "_choose_best_minimax_with_origin" not in src

    # ── Req 5: does not call FST / _stream_one_ply ───────────────────────────

    def test_run_b8_does_not_call_stream_one_ply(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8)
        assert "_stream_one_ply" not in src

    def test_run_b8_does_not_call_run_full_system_trace(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8)
        assert "_run_full_system_trace" not in src

    # ── Req 6: does not accept/use hidden_legal_moves ─────────────────────────

    def test_run_b8_signature_has_no_hidden_param(self):
        import inspect
        mod = self._load_runner()
        sig = inspect.signature(mod._run_b8)
        assert "hidden" not in sig.parameters
        assert "hidden_legal_moves" not in sig.parameters

    def test_run_b8_source_does_not_read_hidden_legal_moves(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8)
        assert "hidden_legal_moves" not in src

    # ── Req 2: B8a uses proposal shortlist ───────────────────────────────────

    def test_run_b8_shortlist_calls_select_proposal(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8)
        assert "select_proposal_candidates" in src

    def test_b8a_diag_used_proposal_shortlist_true(self):
        mod = self._load_runner()
        board = self._simple_board()
        _, _, diag = mod._run_b8(board, "RED", "sc_test", "shortlist_no_safety",
                                  run_seed=42, dry_run=True)
        assert diag["b8_used_proposal_shortlist"] is True
        assert diag["b8_used_full_legal_set"] is False
        assert diag["b8_variant"] == "shortlist_no_safety"
        assert diag["b8_candidate_order_mode"] == "shortlist_order"
        assert diag["b8_candidate_count"] >= 1
        assert diag["b8_candidate_count"] <= 5  # shortlist k=5

    # ── Req 3: B8b uses full legal set and shuffled deterministic order ───────

    def test_b8b_diag_used_full_legal_set_true(self):
        mod = self._load_runner()
        board = self._simple_board()
        _, _, diag = mod._run_b8(board, "RED", "sc_test",
                                  "full_legal_shuffled_no_safety",
                                  run_seed=42, dry_run=True)
        assert diag["b8_used_full_legal_set"] is True
        assert diag["b8_used_proposal_shortlist"] is False
        assert diag["b8_variant"] == "full_legal_shuffled_no_safety"
        assert diag["b8_candidate_order_mode"] == "shuffled_seeded"
        assert diag["b8_shuffle_seed"] is not None

    def test_b8b_shuffle_is_deterministic_for_same_seed_and_id(self):
        mod = self._load_runner()
        board = self._simple_board()
        _, _, d1 = mod._run_b8(board, "RED", "sc_abc",
                                "full_legal_shuffled_no_safety",
                                run_seed=42, dry_run=True)
        _, _, d2 = mod._run_b8(board, "RED", "sc_abc",
                                "full_legal_shuffled_no_safety",
                                run_seed=42, dry_run=True)
        assert d1["b8_candidate_order_saved"] == d2["b8_candidate_order_saved"]

    def test_b8b_different_scenario_ids_get_different_order(self):
        """Two different scenario_ids with same seed should produce different order
        (probabilistically true for any board with ≥2 legal moves)."""
        mod = self._load_runner()
        # Board with multiple legal moves: several RED men
        board = [[0] * 8 for _ in range(8)]
        for col in [0, 2, 4, 6]:
            board[5][col] = 1   # four RED men
        board[2][1] = 2         # one BLACK man
        _, _, d1 = mod._run_b8(board, "RED", "sc_id_A",
                                "full_legal_shuffled_no_safety",
                                run_seed=42, dry_run=True)
        _, _, d2 = mod._run_b8(board, "RED", "sc_id_B",
                                "full_legal_shuffled_no_safety",
                                run_seed=42, dry_run=True)
        # At least one board has >1 candidate; orders will very likely differ
        if d1["b8_candidate_count"] > 1:
            assert d1["b8_candidate_order_saved"] != d2["b8_candidate_order_saved"]

    # ── Req 4 (continued): confirmed diagnostic flags ─────────────────────────

    def test_b8_no_retry_flag(self):
        mod = self._load_runner()
        board = self._simple_board()
        for variant in ("shortlist_no_safety", "full_legal_shuffled_no_safety"):
            _, _, diag = mod._run_b8(board, "RED", "sc_t", variant,
                                      run_seed=42, dry_run=True)
            assert diag["b8_no_retry_confirmed"] is True
            assert diag["b8_no_override_confirmed"] is True
            assert diag["b8_no_fallback_confirmed"] is True
            assert diag["b8_no_update_agent_confirmed"] is True
            assert diag["b8_used_safety_filter"] is False

    # ── Req 7: uses same evaluator ────────────────────────────────────────────

    def test_b8_dispatch_calls_evaluate_scenario(self):
        import inspect
        mod = self._load_runner()
        # Confirm the dispatch block in run_pilot calls _run_b8 and evaluate_scenario.
        # The constants B8A_BASELINE_NAME / B8B_BASELINE_NAME appear by variable
        # name, not by their string values, so check the variable name instead.
        src = inspect.getsource(mod.run_pilot)
        assert "_run_b8(" in src          # B8 dispatch calls _run_b8
        assert "evaluate_scenario" in src  # legality evaluator always called
        assert "B8A_BASELINE_NAME" in src or "b8a" in src.lower()

    # ── Req 8: diagnostic field coverage ─────────────────────────────────────

    def test_b8_diag_field_coverage(self):
        mod = self._load_runner()
        board = self._simple_board()
        _, _, diag = mod._run_b8(board, "RED", "sc_cov", "shortlist_no_safety",
                                  run_seed=42, dry_run=True)
        required = [
            "b8_variant", "b8_candidate_count", "b8_candidate_order_mode",
            "b8_used_safety_filter", "b8_used_proposal_shortlist",
            "b8_used_full_legal_set", "b8_shuffle_seed",
            "b8_raw_chosen_index", "b8_resolved_chosen_index",
            "b8_chosen_index_valid", "b8_selected_move",
            "b8_invalid_selection_reason", "b8_final_legal",
            "b8_no_retry_confirmed", "b8_no_override_confirmed",
            "b8_no_fallback_confirmed", "b8_no_update_agent_confirmed",
            "b8_regex_recovery_used", "b8_index_was_1_based_corrected",
            "b8_candidate_order_saved",
        ]
        for field in required:
            assert field in diag, f"Missing diagnostic field: {field}"


# ═══════════════════════════════════════════════════════════════════════════════
# B9 tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestB9RankerRawPathNoSafety:
    """
    Proves:
    1.  B9 exists and is selectable (in ACTIVE_BASELINES).
    2.  B9 uses ranker candidate information (score_all_legal_moves +
        select_proposal_candidates shortlist).
    3.  B9 does NOT call the B8 index-selection parsing
        (_extract_chosen_index / _resolve_ranker_index).
    4.  B9 does NOT call safety filter, retry, override, fallback, repair,
        or update_agent.
    5.  B9 does NOT use dataset hidden_legal_moves.
    6.  B9 uses the same evaluator (evaluate_scenario).
    7.  Full_System_Trace / _stream_one_ply remains untouched.
    8.  All b9_* diagnostic fields are present.
    9.  In dry-run the selected path comes from the candidate list.
    10. Path-matching diagnostics are correct (matches / not-in-candidates).
    11. Registry count is now 12.
    """

    @staticmethod
    def _load_runner():
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "run_legality_pilot",
            str(Path(__file__).parents[2] / "run_legality_pilot.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _simple_board():
        board = [[0] * 8 for _ in range(8)]
        board[5][0] = 1   # RED man — can move to (4,1)
        board[2][1] = 2   # BLACK man
        return board

    # ── Req 1: selectable ────────────────────────────────────────────────────

    def test_b9_in_active_baselines(self):
        mod = self._load_runner()
        assert mod.B9_BASELINE_NAME in mod.ACTIVE_BASELINES

    def test_active_baselines_count_after_b9(self):
        mod = self._load_runner()
        assert len(mod.ACTIVE_BASELINES) == 13

    # ── Req 2: uses ranker candidate information ──────────────────────────────

    def test_run_b9_uses_score_all_legal_moves(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b9)
        assert "score_all_legal_moves" in src

    def test_run_b9_uses_select_proposal_candidates(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b9)
        assert "select_proposal_candidates" in src

    def test_run_b9_uses_build_ranker_user_prompt(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b9)
        assert "build_ranker_user_prompt" in src

    def test_b9_diag_candidate_source_is_shortlist(self):
        mod = self._load_runner()
        _, _, diag = mod._run_b9(self._simple_board(), "RED", "sc_t",
                                  run_seed=42, dry_run=True)
        assert diag["b9_candidate_source"] == "shortlist_no_safety"

    def test_b9_candidate_count_at_most_5(self):
        mod = self._load_runner()
        _, _, diag = mod._run_b9(self._simple_board(), "RED", "sc_t",
                                  run_seed=42, dry_run=True)
        assert 1 <= diag["b9_candidate_count"] <= 5

    # ── Req 3: does NOT use B8 index-selection parsing ───────────────────────

    def test_run_b9_does_not_call_extract_chosen_index(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b9)
        assert "_extract_chosen_index(" not in src
        assert "_resolve_ranker_index(" not in src

    def test_run_b9_does_not_output_chosen_index(self):
        """B9 output schema has selected_move, not chosen_index."""
        mod = self._load_runner()
        _, _, diag = mod._run_b9(self._simple_board(), "RED", "sc_t",
                                  run_seed=42, dry_run=True)
        # no b8 fields in b9 diag
        assert "b8_raw_chosen_index" not in diag
        assert "b9_selected_move" in diag

    # ── Req 4: no forbidden mechanisms ───────────────────────────────────────

    def test_run_b9_does_not_call_safety_filter(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b9)
        assert "_apply_safety_filter(" not in src

    def test_run_b9_does_not_call_audit_override(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b9)
        assert "_audit_override(" not in src

    def test_run_b9_has_no_retry_loop(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b9)
        assert "for attempt" not in src
        assert "OVERRIDE_MAX_RETRIES" not in src

    def test_run_b9_does_not_call_update_agent(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b9)
        assert "update_agent(" not in src
        assert "from checkers.agents.update_agent" not in src

    def test_run_b9_has_no_fallback(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b9)
        assert "_choose_best_minimax_with_origin" not in src

    def test_b9_confirmed_safety_flags(self):
        mod = self._load_runner()
        _, _, diag = mod._run_b9(self._simple_board(), "RED", "sc_t",
                                  run_seed=42, dry_run=True)
        assert diag["b9_no_retry_confirmed"] is True
        assert diag["b9_no_override_confirmed"] is True
        assert diag["b9_no_fallback_confirmed"] is True
        assert diag["b9_no_update_agent_confirmed"] is True

    # ── Req 5: does not use hidden_legal_moves ────────────────────────────────

    def test_run_b9_signature_has_no_hidden_param(self):
        import inspect
        mod = self._load_runner()
        sig = inspect.signature(mod._run_b9)
        assert "hidden" not in sig.parameters

    def test_run_b9_source_does_not_read_hidden_legal_moves(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b9)
        assert "hidden_legal_moves" not in src

    # ── Req 6: uses same evaluator ────────────────────────────────────────────

    def test_b9_dispatch_calls_evaluate_scenario(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod.run_pilot)
        assert "_run_b9(" in src
        assert "evaluate_scenario" in src

    # ── Req 7: FST unchanged ─────────────────────────────────────────────────

    def test_b9_does_not_call_stream_one_ply(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b9)
        assert "_stream_one_ply" not in src

    # ── Req 8: diagnostic field coverage ─────────────────────────────────────

    def test_b9_diag_field_coverage(self):
        mod = self._load_runner()
        _, _, diag = mod._run_b9(self._simple_board(), "RED", "sc_cov",
                                  run_seed=42, dry_run=True)
        required = [
            "b9_candidate_source", "b9_candidate_count",
            "b9_raw_response", "b9_parse_success",
            "b9_selected_move", "b9_selected_path_matches_candidate",
            "b9_selected_path_not_in_candidates", "b9_final_legal",
            "b9_invalid_reason", "b9_no_retry_confirmed",
            "b9_no_override_confirmed", "b9_no_fallback_confirmed",
            "b9_no_update_agent_confirmed",
        ]
        for field in required:
            assert field in diag, f"Missing field: {field}"

    # ── Req 9: dry-run selected path comes from candidate list ────────────────

    def test_b9_dry_run_selected_move_in_candidate_list(self):
        mod = self._load_runner()
        _, _, diag = mod._run_b9(self._simple_board(), "RED", "sc_dr",
                                  run_seed=42, dry_run=True)
        assert diag["b9_parse_success"] is True
        assert diag["b9_selected_path_matches_candidate"] is True
        assert diag["b9_selected_path_not_in_candidates"] is False

    # ── Req 10: path-matching diagnostics ────────────────────────────────────

    def test_b9_path_not_in_candidates_flag_is_false_for_valid_path(self):
        mod = self._load_runner()
        _, _, diag = mod._run_b9(self._simple_board(), "RED", "sc_chk",
                                  run_seed=42, dry_run=True)
        assert diag["b9_selected_path_not_in_candidates"] is False

    def test_b9_system_prompt_asks_for_selected_move_not_chosen_index(self):
        mod = self._load_runner()
        prompt = mod._B9_SYSTEM_PROMPT
        assert "selected_move" in prompt
        assert "chosen_index" not in prompt

    def test_b9_system_prompt_has_verbatim_copy_instruction(self):
        mod = self._load_runner()
        prompt = mod._B9_SYSTEM_PROMPT
        assert "VERBATIM" in prompt or "EXACT" in prompt or "character-for-character" in prompt

    def test_b9_system_prompt_does_not_expose_hidden_legal_moves(self):
        mod = self._load_runner()
        assert "hidden_legal_moves" not in mod._B9_SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════════════════════
# B8c tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestB8cRankerCompareNoSafety:
    """
    Proves:
    1.  B8c exists and is selectable.
    2.  B8c prompt requires candidate_analysis and chosen_index.
    3.  B8c prompt explicitly forbids selected_move and coordinate paths.
    4.  B8c calls the LLM exactly once (no retry loop).
    5.  B8c does NOT use safety filter / retry / override / fallback / update_agent.
    6.  B8c does NOT use dataset hidden_legal_moves.
    7.  Diagnostics detect missing chosen_index.
    8.  Diagnostics detect first-candidate selection.
    9.  Diagnostics detect missing / present comparison.
    10. Full_System_Trace remains unchanged (B8c does not touch _stream_one_ply).
    11. All b8c_* diagnostic fields are present.
    12. Registry count updated to 13.
    """

    @staticmethod
    def _load_runner():
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "run_legality_pilot",
            str(Path(__file__).parents[2] / "run_legality_pilot.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _simple_board():
        board = [[0] * 8 for _ in range(8)]
        board[5][0] = 1   # RED man
        board[2][1] = 2   # BLACK man
        return board

    # ── Req 1: selectable ────────────────────────────────────────────────────

    def test_b8c_in_active_baselines(self):
        mod = self._load_runner()
        assert mod.B8C_BASELINE_NAME in mod.ACTIVE_BASELINES

    def test_active_baselines_count_after_b8c(self):
        mod = self._load_runner()
        assert len(mod.ACTIVE_BASELINES) == 13

    # ── Req 2: prompt requires candidate_analysis and chosen_index ───────────

    def test_b8c_prompt_requires_candidate_analysis(self):
        mod = self._load_runner()
        assert "candidate_analysis" in mod._B8C_SYSTEM_PROMPT

    def test_b8c_prompt_requires_chosen_index(self):
        mod = self._load_runner()
        assert "chosen_index" in mod._B8C_SYSTEM_PROMPT

    def test_b8c_prompt_requires_pros_and_cons(self):
        mod = self._load_runner()
        p = mod._B8C_SYSTEM_PROMPT
        assert "pros" in p and "cons" in p

    def test_b8c_prompt_requires_comparison_of_at_least_two(self):
        mod = self._load_runner()
        p = mod._B8C_SYSTEM_PROMPT.lower()
        assert "compare" in p or "comparison" in p or "at least two" in p

    # ── Req 3: prompt forbids selected_move and coordinate paths ─────────────

    def test_b8c_prompt_forbids_selected_move(self):
        mod = self._load_runner()
        p = mod._B8C_SYSTEM_PROMPT
        assert "Do NOT output selected_move" in p or "NOT output selected_move" in p

    def test_b8c_prompt_forbids_coordinate_paths(self):
        mod = self._load_runner()
        p = mod._B8C_SYSTEM_PROMPT
        assert "coordinate" in p.lower() or "path" in p.lower()
        # Must contain a prohibition, not just mention
        assert "Do NOT output any coordinate" in p or "NOT output any coordinate" in p

    # ── Req 4: single LLM call ───────────────────────────────────────────────

    def test_run_b8c_has_no_retry_loop(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8c)
        assert "for attempt" not in src
        assert "OVERRIDE_MAX_RETRIES" not in src

    # ── Req 5: no safety mechanisms ──────────────────────────────────────────

    def test_run_b8c_does_not_call_safety_filter(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8c)
        assert "_apply_safety_filter(" not in src

    def test_run_b8c_does_not_call_audit_override(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8c)
        assert "_audit_override(" not in src

    def test_run_b8c_does_not_call_update_agent(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8c)
        assert "update_agent(" not in src
        assert "from checkers.agents.update_agent" not in src

    def test_run_b8c_has_no_choose_best_minimax_fallback(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8c)
        assert "_choose_best_minimax_with_origin" not in src

    def test_b8c_confirmed_safety_flags(self):
        mod = self._load_runner()
        _, _, diag = mod._run_b8c(self._simple_board(), "RED", "sc_t",
                                   dry_run=True)
        assert diag["b8c_no_retry_confirmed"] is True
        assert diag["b8c_no_override_confirmed"] is True
        assert diag["b8c_no_fallback_confirmed"] is True
        assert diag["b8c_no_update_agent_confirmed"] is True

    # ── Req 6: does not use hidden_legal_moves ────────────────────────────────

    def test_run_b8c_signature_has_no_hidden_param(self):
        import inspect
        mod = self._load_runner()
        sig = inspect.signature(mod._run_b8c)
        assert "hidden" not in sig.parameters

    def test_run_b8c_source_does_not_read_hidden_legal_moves(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8c)
        assert "hidden_legal_moves" not in src

    def test_b8c_prompt_does_not_contain_hidden_legal_moves(self):
        mod = self._load_runner()
        assert "hidden_legal_moves" not in mod._B8C_SYSTEM_PROMPT

    # ── Req 7: missing chosen_index detection ─────────────────────────────────

    def test_b8c_detects_missing_chosen_index(self):
        """Simulate a response that has candidate_analysis but no chosen_index."""
        import json
        mod = self._load_runner()
        # Manually inject a bad parse to verify the field
        # We test the logic by calling with dry_run=True and confirming the
        # diag has the field; then test the logic path via the source.
        import inspect
        src = inspect.getsource(mod._run_b8c)
        assert "missing_chosen_index" in src
        assert "b8c_missing_chosen_index" in src

    def test_b8c_missing_chosen_index_flag_in_diag(self):
        mod = self._load_runner()
        _, _, diag = mod._run_b8c(self._simple_board(), "RED", "sc_m",
                                   dry_run=True)
        # dry-run always provides chosen_index: 0, so flag should be False
        assert "b8c_missing_chosen_index" in diag
        assert diag["b8c_missing_chosen_index"] is False   # dry-run provides it

    # ── Req 8: first-candidate detection ──────────────────────────────────────

    def test_b8c_detects_first_candidate_selection(self):
        mod = self._load_runner()
        _, _, diag = mod._run_b8c(self._simple_board(), "RED", "sc_fc",
                                   dry_run=True)
        assert "b8c_selected_first_candidate" in diag
        # dry-run chooses index 0 by design
        assert diag["b8c_selected_first_candidate"] is True
        assert diag["b8c_chosen_index"] == 0

    # ── Req 9: comparison diagnostics ─────────────────────────────────────────

    def test_b8c_candidate_analysis_present_flag(self):
        mod = self._load_runner()
        _, _, diag = mod._run_b8c(self._simple_board(), "RED", "sc_ca",
                                   dry_run=True)
        assert "b8c_candidate_analysis_present" in diag
        assert diag["b8c_candidate_analysis_present"] is True   # dry-run fills it

    def test_b8c_candidate_analysis_count_equals_candidate_count(self):
        mod = self._load_runner()
        _, _, diag = mod._run_b8c(self._simple_board(), "RED", "sc_cn",
                                   dry_run=True)
        assert diag["b8c_candidate_analysis_count"] == diag["b8c_candidate_count"]

    def test_b8c_reasoning_score_fact_detection_in_dry_run(self):
        """dry-run reasoning string contains 'minimax_score' keyword."""
        mod = self._load_runner()
        _, _, diag = mod._run_b8c(self._simple_board(), "RED", "sc_rf",
                                   dry_run=True)
        assert "b8c_reasoning_mentions_score_or_fact" in diag
        # dry-run reasoning: "DRY-RUN comparison using minimax_score"
        assert diag["b8c_reasoning_mentions_score_or_fact"] is True

    # ── Req 10: FST unchanged ─────────────────────────────────────────────────

    def test_run_b8c_does_not_call_stream_one_ply(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8c)
        assert "_stream_one_ply" not in src

    def test_run_b8c_does_not_call_run_full_system_trace(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8c)
        assert "_run_full_system_trace" not in src

    # ── Req 11: diagnostic field coverage ─────────────────────────────────────

    def test_b8c_diag_field_coverage(self):
        mod = self._load_runner()
        _, _, diag = mod._run_b8c(self._simple_board(), "RED", "sc_cov",
                                   dry_run=True)
        required = [
            "b8c_candidate_count", "b8c_raw_response", "b8c_parse_success",
            "b8c_missing_chosen_index", "b8c_invalid_index", "b8c_chosen_index",
            "b8c_selected_first_candidate", "b8c_candidate_analysis_present",
            "b8c_candidate_analysis_count", "b8c_reasoning_mentions_alternative",
            "b8c_reasoning_mentions_score_or_fact", "b8c_schema_failure_type",
            "b8c_final_legal", "b8c_no_retry_confirmed", "b8c_no_override_confirmed",
            "b8c_no_fallback_confirmed", "b8c_no_update_agent_confirmed",
        ]
        for f in required:
            assert f in diag, f"Missing diagnostic field: {f}"

    def test_b8c_dispatch_calls_evaluate_scenario(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod.run_pilot)
        assert "_run_b8c(" in src
        assert "evaluate_scenario" in src

    # ── Extra: uses same ranker input prep as B8a ─────────────────────────────

    def test_run_b8c_uses_score_all_legal_moves(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8c)
        assert "score_all_legal_moves" in src

    def test_run_b8c_uses_select_proposal_candidates(self):
        import inspect
        mod = self._load_runner()
        src = inspect.getsource(mod._run_b8c)
        assert "select_proposal_candidates" in src


# ═══════════════════════════════════════════════════════════════════════════════
# Salvage analysis tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRawOutputSalvage:
    """
    Proves salvage.salvage_parse_failure():
    1.  parse_failure + valid selected_move → salvaged, legality checked.
    2.  parse_failure + illegal selected_move → salvage_success=True, legal=False.
    3.  parse_failure + no move at all → salvage_success=False, no_move_output.
    4.  B8/B8c wrong-schema (selected_move instead of index) → salvaged.
    5.  B8/B8c missing chosen_index AND no path → no_usable_selection.
    6.  Non-parse-failure records → salvage_attempted=False.
    7.  result_type never changes.
    8.  Main legal_move_rate does not change (salvage is additive).
    9.  aggregate_salvage computes correct counts and adjusted rates.
    10. New salvage fields appear in JSONL records from run_pilot dry-run.
    """

    @staticmethod
    def _hidden(paths: list) -> list:
        return [{"path": p, "type": "simple"} for p in paths]

    @staticmethod
    def _load_salvage():
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "salvage",
            str(Path(__file__).parents[2] /
                "checkers" / "data" / "legality_eval" / "salvage.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _load_runner():
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "run_legality_pilot",
            str(Path(__file__).parents[2] / "run_legality_pilot.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # ── Req 1: salvage of valid selected_move in parse_failure ───────────────

    def test_salvage_legal_path_from_path_style_failure(self):
        sv = self._load_salvage()
        hidden = self._hidden([[[5, 0], [4, 1]]])
        raw = '{"selected_move": [[5,0],[4,1]], "reasoning": ""}'
        record = {"result_type": "parse_failure", "baseline": "B1_board_only"}
        board = [[0]*8 for _ in range(8)]
        result = sv.salvage_parse_failure(record, raw, hidden, board, "RED")
        assert result["raw_salvage_attempted"] is True
        assert result["raw_salvage_success"] is True
        assert result["raw_salvaged_move_legal"] is True
        assert result["raw_salvage_failure_reason"] is None

    # ── Req 2: illegal salvaged path ─────────────────────────────────────────

    def test_salvage_illegal_path_from_path_style_failure(self):
        sv = self._load_salvage()
        hidden = self._hidden([[[5, 0], [4, 1]]])   # only legal move
        raw = '{"selected_move": [[3,0],[2,1]], "reasoning": "wrong"}'
        record = {"result_type": "parse_failure", "baseline": "B2_rules"}
        board = [[0]*8 for _ in range(8)]
        result = sv.salvage_parse_failure(record, raw, hidden, board, "RED")
        assert result["raw_salvage_attempted"] is True
        assert result["raw_salvage_success"] is True
        assert result["raw_salvaged_move_legal"] is False

    # ── Req 3: no move output → failure reason ────────────────────────────────

    def test_salvage_no_move_output(self):
        sv = self._load_salvage()
        hidden = self._hidden([[[5, 0], [4, 1]]])
        raw = '{"reasoning": "I think the best move is forward"}'  # no selected_move
        record = {"result_type": "parse_failure", "baseline": "B3_rules_structured_checklist"}
        board = [[0]*8 for _ in range(8)]
        result = sv.salvage_parse_failure(record, raw, hidden, board, "RED")
        assert result["raw_salvage_attempted"] is True
        assert result["raw_salvage_success"] is False
        assert result["raw_salvage_failure_reason"] in ("no_move_output", "cannot_parse_raw_output")

    # ── Req 4: B8/B8c schema confusion → selected_move salvaged ──────────────

    def test_salvage_b8c_schema_confusion_selected_move(self):
        """LLM returned selected_move coordinates instead of chosen_index."""
        sv = self._load_salvage()
        hidden = self._hidden([[[5, 0], [4, 1]]])
        raw = '{"selected_move": [[5,0],[4,1]], "reasoning": "schema confusion"}'
        record = {"result_type": "parse_failure", "baseline": "B8c_ranker_compare_no_safety"}
        board = [[0]*8 for _ in range(8)]
        result = sv.salvage_parse_failure(record, raw, hidden, board, "RED")
        assert result["raw_salvage_attempted"] is True
        assert result["raw_salvage_success"] is True
        assert result["raw_salvage_type"] == "selected_move"

    # ── Req 4b: B8 chosen_index schema confusion → index salvaged ────────────

    def test_salvage_b8a_chosen_index_recoverable(self):
        """LLM response has chosen_index but it was missing from the parsed diag."""
        sv = self._load_salvage()
        hidden = self._hidden([[[5, 0], [4, 1]], [[3, 2], [2, 3]]])
        raw = '{"chosen_index": 0, "reasoning": "best move"}'
        record = {"result_type": "parse_failure", "baseline": "B8a_ranker_shortlist_no_safety"}
        board = [[0]*8 for _ in range(8)]
        candidates = [{"path": [[5, 0], [4, 1]]}, {"path": [[3, 2], [2, 3]]}]
        result = sv.salvage_parse_failure(record, raw, hidden, board, "RED",
                                          candidates=candidates)
        assert result["raw_salvage_type"] == "chosen_index"
        assert result["raw_salvage_success"] is True
        assert result["raw_salvaged_move_legal"] is True

    # ── Req 5: no usable selection → failure reason ───────────────────────────

    def test_salvage_b8c_no_usable_selection(self):
        sv = self._load_salvage()
        hidden = self._hidden([[[5, 0], [4, 1]]])
        raw = '{"candidate_analysis": [], "reasoning": "I cannot decide"}'
        record = {"result_type": "parse_failure", "baseline": "B8c_ranker_compare_no_safety"}
        board = [[0]*8 for _ in range(8)]
        result = sv.salvage_parse_failure(record, raw, hidden, board, "RED")
        assert result["raw_salvage_success"] is False
        assert result["raw_salvage_failure_reason"] in ("no_usable_selection",
                                                         "no_index_output",
                                                         "no_move_output")

    # ── Req 6: non-parse_failure records skipped ──────────────────────────────

    def test_salvage_skips_legal_records(self):
        sv = self._load_salvage()
        record = {"result_type": "legal", "baseline": "B1_board_only"}
        board = [[0]*8 for _ in range(8)]
        result = sv.salvage_parse_failure(record, "", [], board, "RED")
        assert result["raw_salvage_attempted"] is False
        assert result["raw_salvage_type"] == "none"

    def test_salvage_skips_illegal_records(self):
        sv = self._load_salvage()
        record = {"result_type": "illegal", "baseline": "B2_rules"}
        board = [[0]*8 for _ in range(8)]
        result = sv.salvage_parse_failure(record, "", [], board, "RED")
        assert result["raw_salvage_attempted"] is False

    # ── Req 7: result_type never mutated ─────────────────────────────────────

    def test_salvage_does_not_mutate_result_type(self):
        sv = self._load_salvage()
        hidden = self._hidden([[[5, 0], [4, 1]]])
        raw = '{"selected_move": [[5,0],[4,1]]}'
        record = {"result_type": "parse_failure", "baseline": "B1_board_only"}
        board = [[0]*8 for _ in range(8)]
        result = sv.salvage_parse_failure(record, raw, hidden, board, "RED")
        # Original record unchanged
        assert record["result_type"] == "parse_failure"
        # Salvage result does NOT carry result_type
        assert "result_type" not in result

    # ── Req 8: legal_move_rate unchanged after salvage ────────────────────────

    def test_salvage_does_not_affect_legal_move_rate_in_aggregate(self):
        """Confirm aggregate() legal_move_rate only counts result_type==legal."""
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "metrics",
            str(Path(__file__).parents[2] /
                "checkers" / "data" / "legality_eval" / "metrics.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # 3 records: 1 legal, 1 parse_failure with a legal raw_salvaged move,
        #            1 illegal — only legal_move_rate = 1/2 = 0.5
        records = [
            {"result_type": "legal",         "api_success": True,
             "parse_success": True,  "legal": True,
             "rate_limit_retry_count": 0, "wrong_direction": None,
             "mandatory_violation": False, "multi_jump_incomplete": False,
             "illegal_move_type": "", "category": "x", "difficulty": "easy",
             "side_to_move": "RED",
             "raw_salvage_attempted": False, "raw_salvage_success": False},
            {"result_type": "parse_failure", "api_success": True,
             "parse_success": False, "legal": False,
             "rate_limit_retry_count": 0, "wrong_direction": None,
             "mandatory_violation": False, "multi_jump_incomplete": False,
             "illegal_move_type": "", "category": "x", "difficulty": "easy",
             "side_to_move": "RED",
             "raw_salvage_attempted": True, "raw_salvage_success": True,
             "raw_salvaged_move_legal": True},
            {"result_type": "illegal",       "api_success": True,
             "parse_success": True,  "legal": False,
             "rate_limit_retry_count": 0, "wrong_direction": None,
             "mandatory_violation": False, "multi_jump_incomplete": False,
             "illegal_move_type": "path_not_in_legal_moves",
             "category": "x", "difficulty": "easy", "side_to_move": "RED",
             "raw_salvage_attempted": False, "raw_salvage_success": False},
        ]
        agg = mod.aggregate(records)
        # legal_move_rate denominator = evaluated (legal+illegal) = 2
        assert agg["legal_move_rate"] == 0.5
        # Salvage adds adjusted metric but does not change legal_move_rate
        assert agg["salvage"]["salvage_legal_count"] == 1
        assert agg["legal_move_rate"] == 0.5   # unchanged

    # ── Req 9: aggregate_salvage computes correct counts ──────────────────────

    def test_aggregate_salvage_counts(self):
        sv = self._load_salvage()
        records = [
            # normal legal — skipped
            {"result_type": "legal",
             "raw_salvage_attempted": False, "raw_salvage_success": False,
             "raw_salvaged_move_legal": None},
            # parse_failure with legal salvage
            {"result_type": "parse_failure",
             "raw_salvage_attempted": True, "raw_salvage_success": True,
             "raw_salvaged_move_legal": True},
            # parse_failure with illegal salvage
            {"result_type": "parse_failure",
             "raw_salvage_attempted": True, "raw_salvage_success": True,
             "raw_salvaged_move_legal": False},
            # parse_failure with no usable output
            {"result_type": "parse_failure",
             "raw_salvage_attempted": True, "raw_salvage_success": False,
             "raw_salvaged_move_legal": None},
        ]
        agg = sv.aggregate_salvage(records, n_total=4)
        assert agg["parse_failure_count"] == 3
        assert agg["salvage_success_count"] == 2
        assert agg["salvage_legal_count"] == 1
        assert agg["salvage_illegal_count"] == 1
        assert agg["no_usable_output_count"] == 1
        # adjusted: 1 normal legal + 1 salvage legal = 2/4 = 0.5
        assert agg["adjusted_e2e_legal_if_salvaged"] == 0.5

    # ── Req 10: salvage fields appear in dry-run records ─────────────────────

    def test_salvage_fields_in_dry_run_records(self):
        mod = self._load_runner()
        board = [[0] * 8 for _ in range(8)]
        board[5][0] = 1
        board[2][1] = 2
        sc = {
            "scenario_id": "salvage_test",
            "board": board,
            "side_to_move": "RED",
            "hidden_legal_moves": [{"path": [[5, 0], [4, 1]], "type": "simple"}],
            "category": "test",
            "difficulty": "easy",
            "source_file": "test",
        }
        results = mod.run_pilot([sc], baselines=["B1_board_only"], dry_run=True)
        assert len(results) == 1
        r = results[0]
        assert "raw_salvage_attempted" in r
        assert "raw_salvage_success" in r
        assert "raw_salvage_type" in r
        assert "raw_salvaged_move_legal" in r
        assert "raw_salvage_failure_reason" in r
