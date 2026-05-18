# checkers/tests/test_replay_evaluator.py
#
# Tests for checkers/evaluation/replay_evaluator.py
#
# Coverage:
#   1. Isolation — no runtime pipeline imports.
#   2. Missing source file → zero summary, no crash.
#   3. Empty source file → zero summary.
#   4. Single valid record → output JSONL created, summary correct.
#   5. Multiple records → all evaluated, summary counts correct.
#   6. Malformed JSON → ValueError with line number.
#   7. Missing reasoning text fields → evaluates as empty, no crash.
#   8. Missing ranker_diagnostics → handled gracefully.
#   9. Missing facts fields → handled gracefully.
#  10. turn_id from source record preserved.
#  11. Fallback turn_id generated when absent.
#  12. last_move_reasoning preferred over reasoning_text.
#  13. chosen_move_facts preferred over facts.
#  14. Output is append-safe (existing records not overwritten).
#  15. Summary has all required keys.
#  16. reasoning_seeds extracted from ranker_diagnostics.

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ── Isolation guard ───────────────────────────────────────────────────────────
_modules_before = set(sys.modules.keys())

from checkers.evaluation.replay_evaluator import replay_evaluate_file

_modules_after = set(sys.modules.keys())

_FORBIDDEN = (
    "checkers.engine",
    "checkers.agents",
    "checkers.graph",
    "checkers.state",
    "checkers.nodes",
    "checkers.search",
)

_SUMMARY_KEYS = {
    "total_turns",
    "total_claims",
    "supported_claims",
    "contradicted_claims",
    "unsupported_claims",
    "vague_claims",
    "turns_with_contradiction",
    "turns_with_unsupported",
    "turns_with_vague",
    "reasoning_path_counts",
    "trajectory_event_counts",
}


def test_no_runtime_pipeline_imports():
    new = _modules_after - _modules_before
    for mod in new:
        for prefix in _FORBIDDEN:
            assert not mod.startswith(prefix), (
                f"replay_evaluator pulled in runtime module: {mod!r}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_source(path: Path, records: List[Dict[str, Any]]) -> Path:
    """Write a list of dicts to a JSONL file and return the path."""
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


def _base_record(
    turn_id: str = "t1",
    reasoning: str = "This move avoids recapture.",
    facts: Optional[Dict[str, Any]] = None,
    seeds: Optional[List[str]] = None,
    diag: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Minimal source record with all optional fields."""
    if facts is None:
        facts = {"opponent_can_recapture": False}
    if seeds is None:
        seeds = []
    if diag is None:
        diag = {
            "reasoning_seeds": seeds,
            "reasoning_is_seed_fallback": False,
            "reasoning_has_unresolved_contradiction": False,
            "reasoning_refinement_retry_count": 0,
            "api_call_failure_count": 0,
            "override_retry_attempts": 0,
            "override_retry_resolved": False,
            "override_fallback_applied": False,
            "override_branch_name": None,
        }
    return {
        "turn_id": turn_id,
        "last_move_reasoning": reasoning,
        "chosen_move_facts": facts,
        "ranker_diagnostics": diag,
    }


# ---------------------------------------------------------------------------
# Missing / empty source
# ---------------------------------------------------------------------------

class TestMissingAndEmptySource:

    def test_missing_source_file_returns_zero_summary(self, tmp_path):
        src = tmp_path / "nonexistent.jsonl"
        out = tmp_path / "eval.jsonl"
        summary = replay_evaluate_file(src, out)
        assert summary["total_turns"] == 0
        assert not out.exists()

    def test_missing_source_has_all_summary_keys(self, tmp_path):
        summary = replay_evaluate_file(
            tmp_path / "ghost.jsonl",
            tmp_path / "out.jsonl",
        )
        assert _SUMMARY_KEYS == set(summary.keys())

    def test_empty_source_file_returns_zero_summary(self, tmp_path):
        src = tmp_path / "empty.jsonl"
        src.write_text("", encoding="utf-8")
        out = tmp_path / "eval.jsonl"
        summary = replay_evaluate_file(src, out)
        assert summary["total_turns"] == 0

    def test_blank_lines_only_returns_zero_summary(self, tmp_path):
        src = tmp_path / "blanks.jsonl"
        src.write_text("\n\n\n", encoding="utf-8")
        out = tmp_path / "eval.jsonl"
        summary = replay_evaluate_file(src, out)
        assert summary["total_turns"] == 0


# ---------------------------------------------------------------------------
# Single valid record
# ---------------------------------------------------------------------------

class TestSingleRecord:

    def test_output_file_created(self, tmp_path):
        src = _write_source(tmp_path / "src.jsonl", [_base_record()])
        out = tmp_path / "eval.jsonl"
        replay_evaluate_file(src, out)
        assert out.exists()

    def test_output_has_one_line(self, tmp_path):
        src = _write_source(tmp_path / "src.jsonl", [_base_record()])
        out = tmp_path / "eval.jsonl"
        replay_evaluate_file(src, out)
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == 1

    def test_summary_total_turns_one(self, tmp_path):
        src = _write_source(tmp_path / "src.jsonl", [_base_record()])
        out = tmp_path / "eval.jsonl"
        summary = replay_evaluate_file(src, out)
        assert summary["total_turns"] == 1

    def test_summary_has_all_keys(self, tmp_path):
        src = _write_source(tmp_path / "src.jsonl", [_base_record()])
        out = tmp_path / "eval.jsonl"
        summary = replay_evaluate_file(src, out)
        assert _SUMMARY_KEYS == set(summary.keys())

    def test_turn_id_from_source_preserved(self, tmp_path):
        src = _write_source(tmp_path / "src.jsonl", [_base_record("my_turn")])
        out = tmp_path / "eval.jsonl"
        replay_evaluate_file(src, out)
        obj = json.loads(out.read_text().strip())
        assert obj["turn_id"] == "my_turn"


# ---------------------------------------------------------------------------
# Multiple records
# ---------------------------------------------------------------------------

class TestMultipleRecords:

    def test_all_records_evaluated(self, tmp_path):
        src = _write_source(
            tmp_path / "src.jsonl",
            [_base_record(f"t{i}") for i in range(5)],
        )
        out = tmp_path / "eval.jsonl"
        summary = replay_evaluate_file(src, out)
        assert summary["total_turns"] == 5

    def test_output_line_count_matches(self, tmp_path):
        n = 4
        src = _write_source(
            tmp_path / "src.jsonl",
            [_base_record(f"t{i}") for i in range(n)],
        )
        out = tmp_path / "eval.jsonl"
        replay_evaluate_file(src, out)
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == n

    def test_summary_counts_consistent(self, tmp_path):
        src = _write_source(
            tmp_path / "src.jsonl",
            [_base_record(f"t{i}") for i in range(3)],
        )
        out = tmp_path / "eval.jsonl"
        summary = replay_evaluate_file(src, out)
        claim_total = (
            summary["supported_claims"]
            + summary["contradicted_claims"]
            + summary["unsupported_claims"]
            + summary["vague_claims"]
        )
        assert claim_total == summary["total_claims"]

    def test_supported_claims_nonzero_when_facts_present(self, tmp_path):
        records = [
            _base_record(f"t{i}", facts={"opponent_can_recapture": False})
            for i in range(3)
        ]
        src = _write_source(tmp_path / "src.jsonl", records)
        out = tmp_path / "eval.jsonl"
        summary = replay_evaluate_file(src, out)
        assert summary["supported_claims"] > 0


# ---------------------------------------------------------------------------
# Malformed JSON
# ---------------------------------------------------------------------------

class TestMalformedJSON:

    def test_raises_value_error(self, tmp_path):
        src = tmp_path / "bad.jsonl"
        src.write_text(
            json.dumps(_base_record("ok")) + "\nnot json\n",
            encoding="utf-8",
        )
        out = tmp_path / "eval.jsonl"
        with pytest.raises(ValueError, match="Malformed JSON"):
            replay_evaluate_file(src, out)

    def test_error_includes_line_number(self, tmp_path):
        src = tmp_path / "bad2.jsonl"
        src.write_text(
            json.dumps(_base_record("ok1")) + "\n"
            + json.dumps(_base_record("ok2")) + "\n"
            + "bad line\n",
            encoding="utf-8",
        )
        out = tmp_path / "eval.jsonl"
        with pytest.raises(ValueError) as exc_info:
            replay_evaluate_file(src, out)
        assert "3" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Missing / optional fields
# ---------------------------------------------------------------------------

class TestMissingOptionalFields:

    def test_missing_reasoning_evaluates_as_empty(self, tmp_path):
        """Record with no reasoning fields → empty reasoning, no crash."""
        rec = {"chosen_move_facts": {"opponent_can_recapture": False}}
        src = _write_source(tmp_path / "src.jsonl", [rec])
        out = tmp_path / "eval.jsonl"
        summary = replay_evaluate_file(src, out)
        assert summary["total_turns"] == 1

    def test_missing_ranker_diagnostics_no_crash(self, tmp_path):
        rec = {"last_move_reasoning": "This move avoids recapture."}
        src = _write_source(tmp_path / "src.jsonl", [rec])
        out = tmp_path / "eval.jsonl"
        summary = replay_evaluate_file(src, out)
        assert summary["total_turns"] == 1

    def test_missing_facts_no_crash(self, tmp_path):
        rec = {"last_move_reasoning": "This move avoids recapture."}
        src = _write_source(tmp_path / "src.jsonl", [rec])
        out = tmp_path / "eval.jsonl"
        summary = replay_evaluate_file(src, out)
        assert summary["total_turns"] == 1

    def test_missing_turn_id_generates_fallback(self, tmp_path):
        rec = {"last_move_reasoning": "avoids recapture"}
        src = _write_source(tmp_path / "src.jsonl", [rec])
        out = tmp_path / "eval.jsonl"
        replay_evaluate_file(src, out)
        obj = json.loads(out.read_text().strip())
        assert obj["turn_id"].startswith("turn_")

    def test_reasoning_text_fallback_used(self, tmp_path):
        """reasoning_text is used when last_move_reasoning is absent."""
        rec = {
            "reasoning_text": "This move avoids recapture.",
            "chosen_move_facts": {"opponent_can_recapture": False},
        }
        src = _write_source(tmp_path / "src.jsonl", [rec])
        out = tmp_path / "eval.jsonl"
        summary = replay_evaluate_file(src, out)
        assert summary["total_turns"] == 1

    def test_last_move_reasoning_preferred_over_reasoning_text(self, tmp_path):
        """When both keys present, last_move_reasoning wins."""
        rec = {
            "last_move_reasoning": "This move avoids recapture.",
            "reasoning_text": "SHOULD_NOT_BE_USED",
            "chosen_move_facts": {"opponent_can_recapture": False},
        }
        src = _write_source(tmp_path / "src.jsonl", [rec])
        out = tmp_path / "eval.jsonl"
        replay_evaluate_file(src, out)
        # If the right field was used, avoids_recapture will be SUPPORTED
        obj = json.loads(out.read_text().strip())
        recap_claims = [
            c for c in obj["claims"] if c["claim_type"] == "avoids_recapture"
        ]
        assert any(c["claim_status"] == "supported" for c in recap_claims)

    def test_facts_fallback_used(self, tmp_path):
        """'facts' key is used when 'chosen_move_facts' is absent."""
        rec = {
            "last_move_reasoning": "This move avoids recapture.",
            "facts": {"opponent_can_recapture": False},
        }
        src = _write_source(tmp_path / "src.jsonl", [rec])
        out = tmp_path / "eval.jsonl"
        replay_evaluate_file(src, out)
        obj = json.loads(out.read_text().strip())
        recap = [c for c in obj["claims"] if c["claim_type"] == "avoids_recapture"]
        assert any(c["claim_status"] == "supported" for c in recap)

    def test_chosen_move_facts_preferred_over_facts(self, tmp_path):
        """When both keys present, chosen_move_facts wins."""
        rec = {
            "last_move_reasoning": "This move avoids recapture.",
            "chosen_move_facts": {"opponent_can_recapture": False},
            "facts": {"opponent_can_recapture": True},  # would cause CONTRADICTED
        }
        src = _write_source(tmp_path / "src.jsonl", [rec])
        out = tmp_path / "eval.jsonl"
        replay_evaluate_file(src, out)
        obj = json.loads(out.read_text().strip())
        recap = [c for c in obj["claims"] if c["claim_type"] == "avoids_recapture"]
        # chosen_move_facts says False → SUPPORTED (not CONTRADICTED from facts)
        assert any(c["claim_status"] == "supported" for c in recap)


# ---------------------------------------------------------------------------
# Seeds extraction
# ---------------------------------------------------------------------------

class TestSeedsExtraction:

    def test_seeds_extracted_from_diagnostics(self, tmp_path):
        """reasoning_seeds in ranker_diagnostics must reach evaluate_turn."""
        seeds = ["opponent_can_recapture=false — safety confirmed"]
        diag = {"reasoning_seeds": seeds}
        rec = {
            "last_move_reasoning": "This move avoids recapture.",
            "chosen_move_facts": {"opponent_can_recapture": False},
            "ranker_diagnostics": diag,
        }
        src = _write_source(tmp_path / "src.jsonl", [rec])
        out = tmp_path / "eval.jsonl"
        replay_evaluate_file(src, out)
        obj = json.loads(out.read_text().strip())
        recap = [c for c in obj["claims"] if c["claim_type"] == "avoids_recapture"]
        assert len(recap) == 1
        assert recap[0]["claim_status"] == "supported"


# ---------------------------------------------------------------------------
# Append-safety
# ---------------------------------------------------------------------------

class TestAppendSafety:

    def test_existing_output_not_overwritten(self, tmp_path):
        """Running replay twice appends, not overwrites."""
        src_a = _write_source(tmp_path / "src_a.jsonl", [_base_record("first")])
        src_b = _write_source(tmp_path / "src_b.jsonl", [_base_record("second")])
        out   = tmp_path / "shared_eval.jsonl"

        replay_evaluate_file(src_a, out)
        replay_evaluate_file(src_b, out)

        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        ids = [json.loads(l)["turn_id"] for l in lines]
        assert "first" in ids
        assert "second" in ids

    def test_summary_reflects_all_appended_records(self, tmp_path):
        src_a = _write_source(tmp_path / "src_c.jsonl", [_base_record("c1")])
        src_b = _write_source(tmp_path / "src_d.jsonl", [_base_record("c2")])
        out   = tmp_path / "shared_eval2.jsonl"

        replay_evaluate_file(src_a, out)
        summary = replay_evaluate_file(src_b, out)
        assert summary["total_turns"] == 2
