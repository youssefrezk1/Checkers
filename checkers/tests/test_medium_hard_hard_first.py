# checkers/tests/test_medium_hard_hard_first.py
#
# Regression tests for medium_hard hard-first ordering.
# Required invariants:
#   1. All hard positions appear before all medium positions
#   2. Within each group, original dataset order is preserved
#   3. --limit always preserves hard positions first
#   4. No behavior change for any other mode

from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from checkers.eval.proposal_seperation_eval import filter_by_mode


def _entries(*difficulties: str) -> list[dict]:
    """Build minimal dataset entries with the given difficulty sequence."""
    return [
        {"scenario_id": f"p{i}", "difficulty": d, "board": [[0]*8 for _ in range(8)], "side_to_move": "red"}
        for i, d in enumerate(difficulties)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Ordering invariants
# ──────────────────────────────────────────────────────────────────────────────

class TestMediumHardOrdering(unittest.TestCase):

    def test_all_hard_before_all_medium(self):
        entries = _entries("medium", "hard", "medium", "hard", "medium")
        result = filter_by_mode(entries, "medium_hard")
        diffs = [e["difficulty"] for e in result]
        hard_indices   = [i for i, d in enumerate(diffs) if d == "hard"]
        medium_indices = [i for i, d in enumerate(diffs) if d == "medium"]
        self.assertTrue(hard_indices, "expected hard positions in result")
        self.assertTrue(medium_indices, "expected medium positions in result")
        self.assertGreater(min(medium_indices), max(hard_indices))

    def test_hard_group_preserves_original_order(self):
        # Hard positions appear at dataset indices 1 and 3; order must stay [h_idx1, h_idx3]
        entries = _entries("medium", "hard", "medium", "hard", "easy")
        entries[1]["scenario_id"] = "h_first"
        entries[3]["scenario_id"] = "h_second"
        result = filter_by_mode(entries, "medium_hard")
        hard_ids = [e["scenario_id"] for e in result if e["difficulty"] == "hard"]
        self.assertEqual(hard_ids, ["h_first", "h_second"])

    def test_medium_group_preserves_original_order(self):
        entries = _entries("hard", "medium", "hard", "medium", "hard")
        entries[1]["scenario_id"] = "m_first"
        entries[3]["scenario_id"] = "m_second"
        result = filter_by_mode(entries, "medium_hard")
        med_ids = [e["scenario_id"] for e in result if e["difficulty"] == "medium"]
        self.assertEqual(med_ids, ["m_first", "m_second"])

    def test_only_hard_positions_when_no_medium(self):
        entries = _entries("hard", "easy", "hard")
        result = filter_by_mode(entries, "medium_hard")
        diffs = [e["difficulty"] for e in result]
        self.assertEqual(diffs, ["hard", "hard"])

    def test_only_medium_positions_when_no_hard(self):
        entries = _entries("medium", "easy", "medium")
        result = filter_by_mode(entries, "medium_hard")
        diffs = [e["difficulty"] for e in result]
        self.assertEqual(diffs, ["medium", "medium"])

    def test_total_count_equals_hard_plus_medium(self):
        entries = _entries("medium", "hard", "easy", "medium", "hard", "medium")
        result = filter_by_mode(entries, "medium_hard")
        self.assertEqual(len(result), 5)  # 2 hard + 3 medium


# ──────────────────────────────────────────────────────────────────────────────
# 2. Limit interaction
# ──────────────────────────────────────────────────────────────────────────────

class TestMediumHardLimitInteraction(unittest.TestCase):
    """Simulate the --limit slice applied in main() after filter_by_mode."""

    def _filtered_limited(self, difficulties: list[str], limit: int) -> list[dict]:
        entries = _entries(*difficulties)
        return filter_by_mode(entries, "medium_hard")[:limit]

    def test_all_hard_preserved_when_hard_lt_limit(self):
        # hard=2, medium=3, limit=4 → both hard + 2 medium
        result = self._filtered_limited(["medium","hard","medium","hard","medium"], 4)
        diffs = [e["difficulty"] for e in result]
        self.assertEqual(diffs.count("hard"), 2)
        self.assertEqual(diffs.count("medium"), 2)

    def test_all_hard_preserved_when_hard_equals_limit(self):
        # hard=3, limit=3 → all hard, no medium
        result = self._filtered_limited(["hard","medium","hard","medium","hard"], 3)
        diffs = [e["difficulty"] for e in result]
        self.assertEqual(diffs, ["hard", "hard", "hard"])

    def test_only_hard_when_limit_lt_hard_count(self):
        # hard=4, limit=2 → first 2 hard only
        entries = _entries("hard", "hard", "medium", "hard", "hard")
        entries[0]["scenario_id"] = "h0"
        entries[1]["scenario_id"] = "h1"
        result = filter_by_mode(entries, "medium_hard")[:2]
        result_ids = [e["scenario_id"] for e in result]
        self.assertEqual(result_ids, ["h0", "h1"])

    def test_medium_fills_remaining_quota_in_original_order(self):
        # hard=2, medium=5, limit=5 → 2 hard + first 3 medium (in dataset order)
        entries = []
        for i in range(3):
            entries.append({"scenario_id": f"m{i}", "difficulty": "medium",
                            "board": [[0]*8 for _ in range(8)], "side_to_move": "red"})
        entries.append({"scenario_id": "h0", "difficulty": "hard",
                        "board": [[0]*8 for _ in range(8)], "side_to_move": "red"})
        for i in range(3, 5):
            entries.append({"scenario_id": f"m{i}", "difficulty": "medium",
                            "board": [[0]*8 for _ in range(8)], "side_to_move": "red"})
        entries.append({"scenario_id": "h1", "difficulty": "hard",
                        "board": [[0]*8 for _ in range(8)], "side_to_move": "red"})

        result = filter_by_mode(entries, "medium_hard")[:5]
        result_ids = [e["scenario_id"] for e in result]
        # hard first (h0, h1), then medium in original order (m0, m1, m2)
        self.assertEqual(result_ids, ["h0", "h1", "m0", "m1", "m2"])

    def test_limit_larger_than_total_returns_all(self):
        result = self._filtered_limited(["hard","medium","hard"], 100)
        diffs = [e["difficulty"] for e in result]
        self.assertEqual(diffs.count("hard"), 2)
        self.assertEqual(diffs.count("medium"), 1)

    def test_example_from_spec(self):
        # Spec example: hard=163, limit=500 → 163 hard + 337 medium
        hard_entries   = [{"scenario_id": f"h{i}", "difficulty": "hard",
                           "board": [[0]*8 for _ in range(8)], "side_to_move": "red"}
                          for i in range(163)]
        medium_entries = [{"scenario_id": f"m{i}", "difficulty": "medium",
                           "board": [[0]*8 for _ in range(8)], "side_to_move": "red"}
                          for i in range(500)]  # more than needed
        # Interleave them (worst case for old behavior)
        interleaved = []
        for i in range(max(len(hard_entries), len(medium_entries))):
            if i < len(medium_entries):
                interleaved.append(medium_entries[i])
            if i < len(hard_entries):
                interleaved.append(hard_entries[i])

        result = filter_by_mode(interleaved, "medium_hard")[:500]
        hard_count   = sum(1 for e in result if e["difficulty"] == "hard")
        medium_count = sum(1 for e in result if e["difficulty"] == "medium")
        self.assertEqual(hard_count, 163)
        self.assertEqual(medium_count, 337)
        # Verify hard-before-medium: every hard index < every medium index
        diffs = [e["difficulty"] for e in result]
        last_hard = max(i for i, d in enumerate(diffs) if d == "hard")
        first_med  = min(i for i, d in enumerate(diffs) if d == "medium")
        self.assertLess(last_hard, first_med)


# ──────────────────────────────────────────────────────────────────────────────
# 3. No behavior change for other modes
# ──────────────────────────────────────────────────────────────────────────────

class TestOtherModesUnchanged(unittest.TestCase):

    def test_full_mode_returns_dataset_as_is(self):
        entries = _entries("medium", "hard", "easy", "hard", "medium")
        result = filter_by_mode(entries, "full")
        self.assertEqual([e["difficulty"] for e in result],
                         ["medium", "hard", "easy", "hard", "medium"])

    def test_hard_mode_returns_hard_only_in_order(self):
        entries = _entries("medium", "hard", "easy", "hard")
        result = filter_by_mode(entries, "hard")
        self.assertEqual([e["difficulty"] for e in result], ["hard", "hard"])

    def test_medium_mode_returns_medium_only_in_order(self):
        entries = _entries("medium", "hard", "easy", "medium")
        result = filter_by_mode(entries, "medium")
        self.assertEqual([e["difficulty"] for e in result], ["medium", "medium"])

    def test_easy_mode_returns_easy_only(self):
        entries = _entries("easy", "hard", "medium", "easy")
        result = filter_by_mode(entries, "easy")
        self.assertEqual([e["difficulty"] for e in result], ["easy", "easy"])

    def test_jump_only_unaffected(self):
        entries = [
            {"scenario_id": "j1", "difficulty": "hard",
             "hidden_legal_moves": [{"type": "jump"}],
             "board": [], "side_to_move": "red"},
            {"scenario_id": "q1", "difficulty": "medium",
             "hidden_legal_moves": [{"type": "simple"}],
             "board": [], "side_to_move": "red"},
        ]
        result = filter_by_mode(entries, "jump_only")
        self.assertEqual([e["scenario_id"] for e in result], ["j1"])

    def test_quiet_only_unaffected(self):
        entries = [
            {"scenario_id": "j1", "hidden_legal_moves": [{"type": "jump"}],
             "board": [], "side_to_move": "red"},
            {"scenario_id": "q1", "hidden_legal_moves": [{"type": "simple"}],
             "board": [], "side_to_move": "red"},
        ]
        result = filter_by_mode(entries, "quiet_only")
        self.assertEqual([e["scenario_id"] for e in result], ["q1"])


# ──────────────────────────────────────────────────────────────────────────────
# 4. Header note printed when mode=medium_hard and limit is active
# ──────────────────────────────────────────────────────────────────────────────

class TestMediumHardHeaderNote(unittest.TestCase):

    def _run_main_limited(self, difficulties: list[str], limit: int) -> str:
        from checkers.eval.proposal_seperation_eval import main

        entries = _entries(*difficulties)

        def fake_evaluate(**kwargs):
            return {
                "scenario_id": kwargs.get("scenario_id", ""),
                "classification": "perfect",
                "quadrant": "scanner_correct_proposal_correct",
                "elapsed_s": 0.1,
                "scanner_correct": True,
                "proposal_branch": "quiet",
                "api_failure": False,
                "parse_failure": False,
                "proposal_classification": {"classification": "perfect", "legal_count": 1,
                                            "proposed_count": 1, "legal_proposed": 1,
                                            "illegal_proposed": 0, "missing_legal": 0},
                "failure_taxonomy": {"duplicate_moves_generated": 0, "partial_jump_sequences": 0,
                                     "illegal_geometry_moves": 0, "out_of_bounds_coordinates": 0,
                                     "parse_failures": 0, "wrong_branch_called": 0,
                                     "api_failures": 0, "missing_legal_moves": 0},
                "contains_engine_best": True,
                "top1_engine_match": True,
                "contains_kingsrow_best": None,
                "top1_kingsrow_match": None,
                "difficulty": "hard",
                "category": "test",
            }

        buf = io.StringIO()
        with (
            patch("checkers.eval.proposal_seperation_eval.load_dataset", return_value=entries),
            patch("checkers.eval.proposal_seperation_eval.load_bestmove_annotations", return_value={}),
            patch("checkers.eval.proposal_seperation_eval.evaluate_position", side_effect=fake_evaluate),
            patch("sys.stdout", buf),
        ):
            main(["--dataset", "fake.json", "--out", "/tmp/test_mh.json",
                  "--mode", "medium_hard", "--limit", str(limit)])
        return buf.getvalue()

    def test_hard_first_line_printed_when_medium_hard_with_limit(self):
        output = self._run_main_limited(["hard", "medium", "hard", "medium"], 3)
        self.assertIn("Hard-first", output)

    def test_hard_first_line_not_printed_for_other_modes(self):
        from checkers.eval.proposal_seperation_eval import main

        entries = _entries("hard", "medium")
        buf = io.StringIO()
        with (
            patch("checkers.eval.proposal_seperation_eval.load_dataset", return_value=entries),
            patch("checkers.eval.proposal_seperation_eval.load_bestmove_annotations", return_value={}),
            patch("checkers.eval.proposal_seperation_eval.evaluate_position",
                  side_effect=lambda **kw: {
                      "scenario_id": kw.get("scenario_id",""), "classification": "perfect",
                      "quadrant": "scanner_correct_proposal_correct", "elapsed_s": 0.1,
                      "api_failure": False, "parse_failure": False,
                      "proposal_classification": {"classification":"perfect","legal_count":1,
                          "proposed_count":1,"legal_proposed":1,"illegal_proposed":0,"missing_legal":0},
                      "failure_taxonomy": {"duplicate_moves_generated":0,"partial_jump_sequences":0,
                          "illegal_geometry_moves":0,"out_of_bounds_coordinates":0,"parse_failures":0,
                          "wrong_branch_called":0,"api_failures":0,"missing_legal_moves":0},
                      "scanner_correct": True, "proposal_branch": "quiet",
                      "contains_engine_best": None, "top1_engine_match": None,
                      "contains_kingsrow_best": None, "top1_kingsrow_match": None,
                      "difficulty":"hard", "category":"test",
                  }),
            patch("sys.stdout", buf),
        ):
            main(["--dataset", "fake.json", "--out", "/tmp/test_mh2.json",
                  "--mode", "full", "--limit", "1"])
        self.assertNotIn("Hard-first", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
