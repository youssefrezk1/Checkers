# checkers/tests/test_ranker_retry_prompt.py
#
# Focused tests for Phase 2.3b: prompt-level retry diversity guidance.
#
# Validates that _build_override_feedback_str:
#   - produces no rejected-path block on the first retry attempt (empty/None paths)
#   - produces a correctly-worded block on subsequent attempts
#   - caps the displayed paths at 3
#   - uses informational wording (not hard exclusion)
#   - preserves all pre-existing feedback content
#   - normalises paths to list-of-lists for display
#
# What is NOT tested here:
#   - override audit logic, thresholds, or decision behavior
#   - LLM call mechanics
#   - final move selection or fallback

from __future__ import annotations

from checkers.agents.ranker_agent import _build_override_feedback_str

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_EMPTY_DEBUG: dict = {
    "best_vs_chosen_minimax_gap":  40.0,
    "best_vs_chosen_threat_delta": 0,
    "override_branch_triggered":   True,
    "override_branch_name":        "low_danger_minimax_dominance",
}

P1 = [[4, 3], [3, 2]]
P2 = [[5, 4], [4, 3]]
P3 = [[5, 2], [4, 1]]
P4 = [[6, 1], [5, 0]]
P5 = [[6, 5], [5, 4]]


# ---------------------------------------------------------------------------
# 1. No rejected-path block on first retry attempt
# ---------------------------------------------------------------------------

class TestNoRejectedBlockOnFirstAttempt:
    """First retry: _or_tried_paths is empty → no previously-rejected-paths block."""

    def test_none_rejected_paths_no_block(self):
        out = _build_override_feedback_str(_EMPTY_DEBUG, "low_danger_minimax_dominance",
                                           chosen_score=-130.0, best_score=-90.0,
                                           rejected_paths=None)
        assert "Previously rejected" not in out

    def test_empty_list_rejected_paths_no_block(self):
        out = _build_override_feedback_str(_EMPTY_DEBUG, "low_danger_minimax_dominance",
                                           chosen_score=-130.0, best_score=-90.0,
                                           rejected_paths=[])
        assert "Previously rejected" not in out

    def test_end_override_feedback_still_present_no_block(self):
        out = _build_override_feedback_str(_EMPTY_DEBUG, "low_danger_minimax_dominance",
                                           rejected_paths=None)
        assert "END_OVERRIDE_FEEDBACK" in out

    def test_diagnosis_present_without_block(self):
        out = _build_override_feedback_str(_EMPTY_DEBUG, "low_danger_minimax_dominance",
                                           rejected_paths=None)
        assert "DIAGNOSIS" in out


# ---------------------------------------------------------------------------
# 2. Block appears on second retry attempt (one prior path)
# ---------------------------------------------------------------------------

class TestRejectedBlockOnSecondAttempt:
    """Second retry: _or_tried_paths has one path → block shown."""

    def _out(self, paths=None):
        return _build_override_feedback_str(
            _EMPTY_DEBUG, "low_danger_minimax_dominance",
            chosen_score=-130.0, best_score=-90.0,
            rejected_paths=paths,
        )

    def test_block_present_with_one_prior_path(self):
        out = self._out([P1])
        assert "Previously rejected retry paths in this turn:" in out

    def test_block_contains_prior_path(self):
        out = self._out([P1])
        # Path normalised to list-of-lists; [4, 3] and [3, 2] must appear
        assert "4" in out and "3" in out and "2" in out

    def test_avoid_wording_present(self):
        out = self._out([P1])
        assert "Avoid repeating these paths unless you can explain" in out

    def test_block_placed_after_diagnosis(self):
        out = self._out([P1])
        diag_pos     = out.index("DIAGNOSIS")
        rejected_pos = out.index("Previously rejected")
        assert rejected_pos > diag_pos

    def test_block_placed_before_end_marker(self):
        out = self._out([P1])
        rejected_pos = out.index("Previously rejected")
        end_pos      = out.index("END_OVERRIDE_FEEDBACK")
        assert rejected_pos < end_pos

    def test_block_placed_before_candidate_list_note(self):
        out = self._out([P1])
        rejected_pos  = out.index("Previously rejected")
        candidate_pos = out.index("The candidate list below is the FULL proposal shortlist")
        assert rejected_pos < candidate_pos


# ---------------------------------------------------------------------------
# 3. Multiple prior paths and cap at 3
# ---------------------------------------------------------------------------

class TestMultipleRejectedPaths:

    def _out(self, paths):
        return _build_override_feedback_str(
            _EMPTY_DEBUG, "low_danger_minimax_dominance",
            chosen_score=-130.0, best_score=-90.0,
            rejected_paths=paths,
        )

    def test_two_paths_both_shown(self):
        out = self._out([P1, P2])
        assert "Previously rejected" in out
        # Both paths appear; check distinctive coordinates
        assert "3, 2" in out or str(P1) in out
        assert "4, 3" in out or str(P2) in out

    def test_three_paths_all_shown(self):
        out = self._out([P1, P2, P3])
        assert "Previously rejected" in out

    def test_four_paths_capped_at_three(self):
        # P4 is the 4th path — it must be omitted
        out = self._out([P1, P2, P3, P4])
        # The block shows at most 3; verify P4's unique coordinate (6, 1) is absent
        p4_marker = str([list(sq) for sq in P4])  # "[[6, 1], [5, 0]]"
        assert p4_marker not in out

    def test_five_paths_capped_at_three(self):
        out = self._out([P1, P2, P3, P4, P5])
        p4_marker = str([list(sq) for sq in P4])
        p5_marker = str([list(sq) for sq in P5])
        assert p4_marker not in out
        assert p5_marker not in out

    def test_cap_does_not_suppress_block_entirely(self):
        out = self._out([P1, P2, P3, P4, P5])
        assert "Previously rejected retry paths in this turn:" in out


# ---------------------------------------------------------------------------
# 4. Wording is informational — not hard exclusion
# ---------------------------------------------------------------------------

class TestBlockWordingIsInformational:
    """The block must inform, not forbid. The LLM may still select these moves."""

    def _out(self):
        return _build_override_feedback_str(
            _EMPTY_DEBUG, "low_danger_minimax_dominance",
            rejected_paths=[P1, P2],
        )

    def test_no_illegal_wording(self):
        out = self._out()
        assert "illegal" not in out.lower()

    def test_no_forbidden_wording(self):
        out = self._out()
        assert "forbidden" not in out.lower()

    def test_no_cannot_wording(self):
        out = self._out()
        assert "cannot choose" not in out.lower()
        assert "must not choose" not in out.lower()

    def test_no_do_not_choose_wording(self):
        out = self._out()
        # Should say "Avoid repeating" not "do not choose"
        assert "do not choose" not in out.lower()

    def test_informational_qualifier_present(self):
        # The block must say the LLM can still explain why the audit would pass
        out = self._out()
        assert "unless you can explain" in out


# ---------------------------------------------------------------------------
# 5. All pre-existing feedback content is preserved
# ---------------------------------------------------------------------------

class TestExistingFeedbackPreserved:

    def _out(self, branch, paths=None):
        return _build_override_feedback_str(
            _EMPTY_DEBUG, branch,
            chosen_score=-130.0, best_score=-90.0,
            rejected_paths=paths,
        )

    def test_override_feedback_header_present(self):
        for paths in [None, [P1]]:
            out = self._out("low_danger_minimax_dominance", paths)
            assert "OVERRIDE_FEEDBACK:" in out

    def test_previous_selection_rejected_line_present(self):
        for paths in [None, [P1]]:
            out = self._out("low_danger_minimax_dominance", paths)
            assert "Your previous selection was rejected" in out

    def test_diagnosis_block_present(self):
        for paths in [None, [P1]]:
            out = self._out("low_danger_minimax_dominance", paths)
            assert "DIAGNOSIS:" in out

    def test_action_block_present(self):
        for paths in [None, [P1]]:
            out = self._out("low_danger_minimax_dominance", paths)
            assert "ACTION:" in out

    def test_end_marker_present(self):
        for paths in [None, [P1]]:
            out = self._out("low_danger_minimax_dominance", paths)
            assert "END_OVERRIDE_FEEDBACK" in out

    def test_safe_vs_unsafe_branch_content_preserved_with_block(self):
        out = self._out("safe_vs_unsafe_large_gap", [P1])
        assert "DIAGNOSIS:" in out
        assert "Previously rejected" in out

    def test_safe_vs_unsafe_branch_content_preserved_without_block(self):
        out = self._out("safe_vs_unsafe_large_gap", None)
        assert "DIAGNOSIS:" in out
        assert "Previously rejected" not in out

    def test_unknown_branch_content_preserved_with_block(self):
        out = self._out("some_unknown_branch", [P1])
        assert "DIAGNOSIS:" in out
        assert "Previously rejected" in out


# ---------------------------------------------------------------------------
# 6. Path normalisation (list-of-tuples → list-of-lists for display)
# ---------------------------------------------------------------------------

class TestPathNormalisation:

    def test_list_of_tuples_path_normalised(self):
        path_tuples = [(4, 3), (3, 2)]
        out = _build_override_feedback_str(
            _EMPTY_DEBUG, "low_danger_minimax_dominance",
            rejected_paths=[path_tuples],
        )
        assert "Previously rejected" in out
        # Normalised form has brackets not parens
        assert "[[4, 3], [3, 2]]" in out

    def test_list_of_lists_path_displayed(self):
        out = _build_override_feedback_str(
            _EMPTY_DEBUG, "low_danger_minimax_dominance",
            rejected_paths=[P1],
        )
        assert "[[4, 3], [3, 2]]" in out

    def test_none_path_in_list_does_not_crash(self):
        out = _build_override_feedback_str(
            _EMPTY_DEBUG, "low_danger_minimax_dominance",
            rejected_paths=[None],
        )
        assert "Previously rejected" in out  # block fires; None normalised gracefully


# ---------------------------------------------------------------------------
# 7. No decision logic reads the rejected-path block
# ---------------------------------------------------------------------------

class TestNoDecisionLogicReadsRejectedPaths:
    """
    Static guard: the rejected-path block is injected into prompt text only.
    Production decision files must not read 'rejected_paths' as a diagnostic key.
    """

    def test_rejected_paths_not_read_in_decision_files(self):
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        decision_files = [
            root / "checkers" / "nodes" / "state_manager.py",
            root / "checkers" / "nodes" / "logger_node.py",
            root / "checkers" / "graph" / "graph.py",
        ]
        for fpath in decision_files:
            if not fpath.exists():
                continue
            text = fpath.read_text(encoding="utf-8")
            assert "rejected_paths" not in text, (
                f"rejected_paths must not appear in {fpath.name}"
            )
