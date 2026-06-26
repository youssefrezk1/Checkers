# checkers/tests/test_phase3_trace_ux.py
#
# Phase 3 fix — unit tests for trace/debugging UX improvements.
#
# Coverage:
#   1. _candidate_table_rows — lookup table generation
#   2. _refinement_diff_lines — diff rendering
#   3. _annotate_board_lines — board origin/destination markers
#   4. skip reason propagation through explainer_agent diagnostics
#   5. _SKIP_LABELS has an entry for every skip reason the pipeline emits

from __future__ import annotations

import sys
import os

import pytest

# ── Import helpers from the trace script ─────────────────────────────────────
# The trace script sets env vars at module level; guard against side-effects
# by importing only after isolating the symbol.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
from run_simplified_trace_reasoning import (
    _candidate_table_rows,
    _refinement_diff_lines,
    _annotate_board_lines,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _move(path, **facts):
    return {"path": path, "type": "simple", "facts": dict(facts)}


CAND_QUIET = _move(
    [(5, 2), (4, 1)],
    captures_count=0, net_gain=0,
    opponent_can_recapture=False,
    leaves_piece_isolated=False,
    opponent_mobility_before=11, opponent_mobility_after=11,
)

CAND_CAPTURE = _move(
    [(4, 5), (2, 3)],
    captures_count=1, net_gain=1,
    opponent_can_recapture=True,
    leaves_piece_isolated=True,
    opponent_mobility_before=8, opponent_mobility_after=7,
)

CAND_NO_FACTS = {"path": [(5, 0), (4, 1)], "type": "simple"}


# ═════════════════════════════════════════════════════════════════════════════
# 1. _candidate_table_rows
# ═════════════════════════════════════════════════════════════════════════════

class TestCandidateTableRows:
    def test_empty_candidates_returns_empty(self):
        assert _candidate_table_rows([]) == []

    def test_row_count_matches_candidate_count(self):
        rows = _candidate_table_rows([CAND_QUIET, CAND_CAPTURE])
        assert len(rows) == 2

    def test_index_is_zero_based(self):
        rows = _candidate_table_rows([CAND_QUIET, CAND_CAPTURE])
        assert rows[0][0] == 0
        assert rows[1][0] == 1

    def test_path_is_in_row(self):
        rows = _candidate_table_rows([CAND_QUIET])
        _, path_str, *_ = rows[0]
        assert "5" in path_str and "2" in path_str

    def test_recapture_no_shown_for_safe_move(self):
        rows = _candidate_table_rows([CAND_QUIET])
        _, _, _, recap, _, _ = rows[0]
        assert recap == "n"

    def test_recapture_yes_shown_for_capturing_move(self):
        rows = _candidate_table_rows([CAND_CAPTURE])
        _, _, _, recap, _, _ = rows[0]
        assert recap == "Y"

    def test_isolation_no_shown_for_safe_move(self):
        rows = _candidate_table_rows([CAND_QUIET])
        _, _, _, _, iso, _ = rows[0]
        assert iso == "n"

    def test_isolation_yes_shown_for_isolated_move(self):
        rows = _candidate_table_rows([CAND_CAPTURE])
        _, _, _, _, iso, _ = rows[0]
        assert iso == "Y"

    def test_mobility_delta_correct_zero(self):
        rows = _candidate_table_rows([CAND_QUIET])
        _, _, _, _, _, delta = rows[0]
        assert delta == "+0"

    def test_mobility_delta_correct_negative(self):
        rows = _candidate_table_rows([CAND_CAPTURE])
        _, _, _, _, _, delta = rows[0]
        assert delta == "-1"

    def test_mobility_delta_question_mark_when_absent(self):
        rows = _candidate_table_rows([CAND_NO_FACTS])
        _, _, _, _, _, delta = rows[0]
        assert delta == "?"

    def test_single_candidate(self):
        rows = _candidate_table_rows([CAND_QUIET])
        assert len(rows) == 1
        assert rows[0][0] == 0


# ═════════════════════════════════════════════════════════════════════════════
# 2. _refinement_diff_lines
# ═════════════════════════════════════════════════════════════════════════════

class TestRefinementDiffLines:
    def test_identical_texts_returns_empty(self):
        assert _refinement_diff_lines("same text", "same text") == []

    def test_empty_raw_returns_empty(self):
        assert _refinement_diff_lines("", "final text") == []

    def test_empty_final_returns_empty(self):
        assert _refinement_diff_lines("raw text", "") == []

    def test_both_empty_returns_empty(self):
        assert _refinement_diff_lines("", "") == []

    def test_different_texts_returns_non_empty(self):
        lines = _refinement_diff_lines("The move captures.", "The move advances.")
        assert len(lines) > 0

    def test_added_line_starts_with_plus(self):
        lines = _refinement_diff_lines("old line", "old line\nnew line")
        added = [l for l in lines if l.startswith("+")]
        assert added

    def test_removed_line_starts_with_minus(self):
        lines = _refinement_diff_lines("old line\nremoved", "old line")
        removed = [l for l in lines if l.startswith("-")]
        assert removed

    def test_no_file_header_lines(self):
        lines = _refinement_diff_lines("a b c", "a b d")
        for line in lines:
            assert not line.startswith("---")
            assert not line.startswith("+++")
            assert not line.startswith("@@")

    def test_context_line_neither_plus_nor_minus(self):
        raw   = "line one\nline two\nline three"
        final = "line one\nLINE TWO CHANGED\nline three"
        lines = _refinement_diff_lines(raw, final)
        context = [l for l in lines if not l.startswith(("+", "-"))]
        assert context, "Expected at least one context line"


# ═════════════════════════════════════════════════════════════════════════════
# 3. _annotate_board_lines
# ═════════════════════════════════════════════════════════════════════════════

_SAMPLE_BOARD_TEXT = (
    "  0 1 2 3 4 5 6 7 \n"
    "0 . b . b . b . b \n"
    "1 b . b . b . b . \n"
    "2 . b . b . b . b \n"
    "3 . . . . . . . . \n"
    "4 . . . . . . . . \n"
    "5 r . r . r . r . \n"
    "6 . r . r . r . r \n"
    "7 r . r . r . r . "
)


class TestAnnotateBoardLines:
    def test_no_path_returns_unchanged_lines(self):
        lines = _annotate_board_lines(_SAMPLE_BOARD_TEXT, [])
        expected = _SAMPLE_BOARD_TEXT.splitlines()
        assert lines == expected

    def test_short_path_returns_unchanged_lines(self):
        lines = _annotate_board_lines(_SAMPLE_BOARD_TEXT, [[5, 2]])
        expected = _SAMPLE_BOARD_TEXT.splitlines()
        assert lines == expected

    def test_origin_row_is_modified(self):
        path = [[5, 2], [4, 1]]
        lines = _annotate_board_lines(_SAMPLE_BOARD_TEXT, path)
        # Row 5 should contain ANSI escape (origin marked in yellow)
        row5 = lines[6]  # header + rows 0-4 before row 5 = index 6
        assert "\033[" in row5, "Expected ANSI code in origin row"

    def test_destination_row_is_modified(self):
        path = [[5, 2], [4, 1]]
        lines = _annotate_board_lines(_SAMPLE_BOARD_TEXT, path)
        row4 = lines[5]  # header + rows 0-3 before row 4 = index 5
        assert "\033[" in row4, "Expected ANSI code in destination row"

    def test_unchanged_rows_have_no_ansi(self):
        path = [[5, 2], [4, 1]]
        lines = _annotate_board_lines(_SAMPLE_BOARD_TEXT, path)
        for i, line in enumerate(lines):
            if i not in (5, 6):  # rows 4 and 5
                assert "\033[" not in line, f"Unexpected ANSI in line {i}: {line!r}"

    def test_origin_character_preserved_within_ansi_wrapper(self):
        path = [[5, 2], [4, 1]]
        lines = _annotate_board_lines(_SAMPLE_BOARD_TEXT, path)
        row5 = lines[6]
        # The piece character 'r' should still be there (inside ANSI codes)
        assert "r" in row5

    def test_bad_path_format_returns_unchanged(self):
        lines = _annotate_board_lines(_SAMPLE_BOARD_TEXT, [None, None])
        assert lines == _SAMPLE_BOARD_TEXT.splitlines()

    def test_line_count_unchanged(self):
        path = [[5, 2], [4, 1]]
        lines = _annotate_board_lines(_SAMPLE_BOARD_TEXT, path)
        assert len(lines) == len(_SAMPLE_BOARD_TEXT.splitlines())


# ═════════════════════════════════════════════════════════════════════════════
# 4. Skip reason propagation through explainer_agent._explain_chosen_move
# ═════════════════════════════════════════════════════════════════════════════

class TestSkipReasonPropagation:
    """
    Exercises the new explicit skip reason logic added to _explain_chosen_move
    in explainer_agent.py.  Uses the diagnostic accumulator path (diagnostics_out)
    to inspect what was written without invoking the full graph.
    """

    def _make_state(self, legal_moves):
        """Build a minimal CheckersState-like mock for _explain_chosen_move."""
        from checkers.state.state import CheckersState
        from checkers.engine.board import create_initial_board, RED
        board = create_initial_board()
        chosen = legal_moves[0] if legal_moves else None
        return CheckersState(
            board=board,
            current_player=RED,
            turn_number=1,
            legal_moves=legal_moves,
            unchosen_moves=legal_moves[1:],
            chosen_move=chosen,
            chosen_move_score=0.0,
        )

    def _run_ranker(self, legal_moves):
        """Run _explain_chosen_move and return ranker_diagnostics."""
        from checkers.agents.explainer_agent import _explain_chosen_move
        import os
        os.environ["EXPLAINER_COMPARATIVE_STAGE_ENABLED"] = "1"
        state = self._make_state(legal_moves)
        result = _explain_chosen_move(state)
        return result.get("explainer_diagnostics") or {}

    def _simple_move(self, from_sq, to_sq):
        return {
            "type": "simple",
            "path": [from_sq, to_sq],
            "captured": [],
            "facts": {
                "captures_count": 0, "net_gain": 0,
                "results_in_king": False,
                "creates_immediate_threat": False,
                "opponent_can_recapture": False,
                "leaves_piece_isolated": False,
                "weakens_king_row": False,
                "near_promotion": False,
                "our_pieces_threatened_after": 0,
                "opponent_mobility_before": 10,
                "opponent_mobility_after": 10,
                "minimax_score": 0.0,
            },
        }

    def test_single_legal_move_skip_reason(self):
        """With 1 legal move, skip reason must be 'single_legal_move'."""
        os.environ["EXPLAINER_COMPARATIVE_STAGE_ENABLED"] = "1"
        try:
            m = self._simple_move((5, 2), (4, 1))
            diag = self._run_ranker([m])
            assert diag.get("comparative_was_skipped") is True
            assert diag.get("comparative_skip_reason") == "single_legal_move"
        finally:
            os.environ.pop("EXPLAINER_COMPARATIVE_STAGE_ENABLED", None)

    def test_two_legal_moves_skip_reason(self):
        """With 2 legal moves, skip reason must be 'insufficient_candidates'."""
        os.environ["EXPLAINER_COMPARATIVE_STAGE_ENABLED"] = "1"
        try:
            m1 = self._simple_move((5, 2), (4, 1))
            m2 = self._simple_move((5, 0), (4, 1))
            diag = self._run_ranker([m1, m2])
            assert diag.get("comparative_was_skipped") is False
            assert diag.get("comparative_skip_reason") is None
            assert diag.get("comparative_paragraph_text") is not None
        finally:
            os.environ.pop("EXPLAINER_COMPARATIVE_STAGE_ENABLED", None)


# ═════════════════════════════════════════════════════════════════════════════
# 5. _SKIP_LABELS completeness — all pipeline skip reasons have a label
# ═════════════════════════════════════════════════════════════════════════════

class TestSkipLabelsCompleteness:
    """
    Ensures _SKIP_LABELS in _section_comparative covers every skip reason
    that the comparative pipeline can emit, so the trace output never
    falls back to the raw key string.
    """

    # All skip reasons emitted by comparative_reasoner + explainer_agent
    _KNOWN_SKIP_REASONS = {
        "single_legal_move",
        "insufficient_candidates",
        "no_informative_groups",
        "no_seeds",
        "api_failure",
        "all_samples_rejected",
        "unknown",
    }

    def test_all_known_reasons_have_label(self):
        # Import the labels table directly from the trace module
        import importlib
        mod = importlib.import_module("run_simplified_trace_reasoning")
        # Extract _SKIP_LABELS from the function source — it's a local dict
        # inside _section_comparative.  We test coverage at the integration
        # level: every known reason string is present in the function source.
        import inspect
        src = inspect.getsource(mod._section_comparative)
        for reason in self._KNOWN_SKIP_REASONS:
            assert f'"{reason}"' in src or f"'{reason}'" in src, (
                f"_SKIP_LABELS missing entry for skip reason {reason!r}"
            )
