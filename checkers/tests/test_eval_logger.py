# checkers/tests/test_eval_logger.py
#
# Tests for checkers/evaluation/eval_logger.py
#
# Coverage:
#   1. Isolation — no runtime pipeline imports.
#   2. record_to_dict — all fields present and JSON-serialisable.
#   3. record_to_dict — enum fields serialised to string values.
#   4. record_to_dict — claims list serialised recursively.
#   5. append_turn_record — creates file if absent.
#   6. append_turn_record — append-safe (does not overwrite prior records).
#   7. append_turn_record — creates parent directories.
#   8. append_turn_record — raises TypeError on wrong input type.
#   9. load_eval_records — roundtrip (appended == loaded).
#  10. load_eval_records — returns [] for missing file.
#  11. load_eval_records — skips empty lines.
#  12. load_eval_records — raises ValueError on malformed JSON.
#  13. summarize_records — empty input returns zero summary.
#  14. summarize_records — counts aggregated correctly.
#  15. summarize_records — reasoning_path_counts and trajectory_event_counts.
#  16. summarize_records — all required keys present.

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# ── Isolation guard ───────────────────────────────────────────────────────────
_modules_before = set(sys.modules.keys())

from checkers.evaluation.eval_logger import (
    record_to_dict,
    append_turn_record,
    load_eval_records,
    summarize_records,
)
from checkers.evaluation.turn_evaluator import evaluate_turn, TurnEvaluationRecord
from checkers.evaluation.reasoning_taxonomy import ClaimStatus

_modules_after = set(sys.modules.keys())

_FORBIDDEN = (
    "checkers.engine",
    "checkers.agents",
    "checkers.graph",
    "checkers.state",
    "checkers.nodes",
    "checkers.search",
)


def test_no_runtime_pipeline_imports():
    new = _modules_after - _modules_before
    for mod in new:
        for prefix in _FORBIDDEN:
            assert not mod.startswith(prefix), (
                f"eval_logger pulled in runtime module: {mod!r}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_diag(**overrides) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "reasoning_seeds": [],
        "reasoning_is_seed_fallback": False,
        "reasoning_has_unresolved_contradiction": False,
        "reasoning_refinement_retry_count": 0,
        "api_call_failure_count": 0,
        "ranker_selected_valid_candidate": True,
        "override_retry_attempts": 0,
        "override_retry_resolved": False,
        "override_fallback_applied": False,
        "override_branch_name": None,
    }
    base.update(overrides)
    return base


def _make_record(
    turn_id: str = "t1",
    text: str = "This move avoids recapture. The engine confirms minimax_score=3.00.",
    facts: "Optional[Dict[str, Any]]" = None,
    **diag_overrides,
) -> TurnEvaluationRecord:
    if facts is None:
        facts = {"opponent_can_recapture": False, "minimax_score": 3.0}
    diag = _clean_diag(**diag_overrides)
    return evaluate_turn(text, facts=facts, ranker_diagnostics=diag, turn_id=turn_id)


# ---------------------------------------------------------------------------
# record_to_dict
# ---------------------------------------------------------------------------

class TestRecordToDict:

    def test_all_required_keys_present(self):
        d = record_to_dict(_make_record())
        required = {
            "turn_id", "total_claims", "supported_count", "contradicted_count",
            "unsupported_count", "vague_count", "has_contradiction",
            "has_unsupported", "has_vague", "reasoning_path",
            "trajectory_events", "claims",
        }
        assert required == set(d.keys()), (
            f"Missing: {required - set(d.keys())}\n"
            f"Extra:   {set(d.keys()) - required}"
        )

    def test_turn_id_preserved(self):
        d = record_to_dict(_make_record("my_turn"))
        assert d["turn_id"] == "my_turn"

    def test_counts_are_integers(self):
        d = record_to_dict(_make_record())
        for key in ("total_claims", "supported_count", "contradicted_count",
                    "unsupported_count", "vague_count"):
            assert isinstance(d[key], int), f"{key} should be int"

    def test_flags_are_booleans(self):
        d = record_to_dict(_make_record())
        for key in ("has_contradiction", "has_unsupported", "has_vague"):
            assert isinstance(d[key], bool), f"{key} should be bool"

    def test_reasoning_path_is_string(self):
        d = record_to_dict(_make_record())
        assert isinstance(d["reasoning_path"], str)

    def test_trajectory_events_is_list(self):
        d = record_to_dict(_make_record())
        assert isinstance(d["trajectory_events"], list)

    def test_claims_is_list(self):
        d = record_to_dict(_make_record())
        assert isinstance(d["claims"], list)

    def test_claim_enum_fields_are_strings(self):
        d = record_to_dict(_make_record())
        for claim in d["claims"]:
            assert isinstance(claim["claim_status"], str), (
                "claim_status should be serialised to string"
            )
            assert isinstance(claim["claim_verifiability"], str), (
                "claim_verifiability should be serialised to string"
            )

    def test_claim_optional_enum_none_preserved(self):
        """Optional enum fields that are None must remain None (not "None")."""
        rec = _make_record(
            text="This move avoids recapture.",
            facts={"opponent_can_recapture": False},
        )
        d = record_to_dict(rec)
        for claim in d["claims"]:
            if claim["seed_risk_type"] is not None:
                assert isinstance(claim["seed_risk_type"], str)
            if claim["hallucination_type"] is not None:
                assert isinstance(claim["hallucination_type"], str)

    def test_output_is_json_serialisable(self):
        d = record_to_dict(_make_record())
        dumped = json.dumps(d)
        loaded = json.loads(dumped)
        assert loaded["turn_id"] == "t1"

    def test_deterministic(self):
        rec = _make_record("det")
        d1 = record_to_dict(rec)
        d2 = record_to_dict(rec)
        assert d1 == d2


# ---------------------------------------------------------------------------
# append_turn_record
# ---------------------------------------------------------------------------

class TestAppendTurnRecord:

    def test_creates_file(self, tmp_path):
        p = tmp_path / "eval.jsonl"
        assert not p.exists()
        append_turn_record(_make_record("a1"), p)
        assert p.exists()

    def test_file_has_one_line(self, tmp_path):
        p = tmp_path / "eval.jsonl"
        append_turn_record(_make_record("a2"), p)
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        assert len(lines) == 1

    def test_line_is_valid_json(self, tmp_path):
        p = tmp_path / "eval.jsonl"
        append_turn_record(_make_record("a3"), p)
        obj = json.loads(p.read_text().strip())
        assert obj["turn_id"] == "a3"

    def test_append_does_not_overwrite(self, tmp_path):
        p = tmp_path / "eval.jsonl"
        append_turn_record(_make_record("first"), p)
        append_turn_record(_make_record("second"), p)
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        ids = [json.loads(l)["turn_id"] for l in lines]
        assert ids == ["first", "second"]

    def test_creates_parent_directories(self, tmp_path):
        p = tmp_path / "nested" / "deep" / "eval.jsonl"
        append_turn_record(_make_record("deep"), p)
        assert p.exists()

    def test_raises_type_error_on_non_record(self, tmp_path):
        with pytest.raises(TypeError):
            append_turn_record({"not": "a record"}, tmp_path / "x.jsonl")  # type: ignore

    def test_raises_type_error_on_dict(self, tmp_path):
        with pytest.raises(TypeError):
            append_turn_record(None, tmp_path / "x.jsonl")  # type: ignore


# ---------------------------------------------------------------------------
# load_eval_records
# ---------------------------------------------------------------------------

class TestLoadEvalRecords:

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_eval_records(tmp_path / "nope.jsonl") == []

    def test_roundtrip_single_record(self, tmp_path):
        p = tmp_path / "rt.jsonl"
        rec = _make_record("rt1")
        append_turn_record(rec, p)
        loaded = load_eval_records(p)
        assert len(loaded) == 1
        assert loaded[0]["turn_id"] == "rt1"
        assert loaded[0]["total_claims"] == rec.total_claims
        assert loaded[0]["supported_count"] == rec.supported_count

    def test_roundtrip_multiple_records_order_preserved(self, tmp_path):
        p = tmp_path / "multi.jsonl"
        for i in range(4):
            append_turn_record(_make_record(f"r{i}"), p)
        loaded = load_eval_records(p)
        assert len(loaded) == 4
        assert [r["turn_id"] for r in loaded] == ["r0", "r1", "r2", "r3"]

    def test_skips_empty_lines(self, tmp_path):
        p = tmp_path / "empty_lines.jsonl"
        p.write_text('\n\n{"turn_id": "x", "total_claims": 0}\n\n')
        loaded = load_eval_records(p)
        assert len(loaded) == 1
        assert loaded[0]["turn_id"] == "x"

    def test_raises_value_error_on_malformed_json(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"turn_id": "ok"}\nnot valid json\n')
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_eval_records(p)

    def test_raises_value_error_includes_line_number(self, tmp_path):
        p = tmp_path / "bad2.jsonl"
        p.write_text('{"turn_id": "ok"}\n{"turn_id": "ok2"}\nbad line\n')
        with pytest.raises(ValueError) as exc_info:
            load_eval_records(p)
        assert "3" in str(exc_info.value)  # line number in message

    def test_all_claim_fields_present_after_roundtrip(self, tmp_path):
        p = tmp_path / "claims.jsonl"
        rec = _make_record("claimstest")
        append_turn_record(rec, p)
        loaded = load_eval_records(p)
        for claim in loaded[0]["claims"]:
            for key in ("claim_type", "claim_status", "claim_verifiability",
                        "source", "matched_phrase", "matched_seed"):
                assert key in claim, f"Missing claim field: {key}"


# ---------------------------------------------------------------------------
# summarize_records
# ---------------------------------------------------------------------------

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


class TestSummarizeRecords:

    def test_empty_input_returns_zero_summary(self):
        s = summarize_records([])
        assert s["total_turns"] == 0
        assert s["total_claims"] == 0
        assert s["supported_claims"] == 0
        assert s["reasoning_path_counts"] == {}
        assert s["trajectory_event_counts"] == {}

    def test_all_required_keys_present(self):
        s = summarize_records([record_to_dict(_make_record())])
        missing = _SUMMARY_KEYS - set(s.keys())
        assert not missing, f"Missing summary keys: {missing}"

    def test_total_turns_correct(self):
        records = [record_to_dict(_make_record(f"t{i}")) for i in range(5)]
        s = summarize_records(records)
        assert s["total_turns"] == 5

    def test_total_claims_summed(self):
        records = [record_to_dict(_make_record(f"t{i}")) for i in range(3)]
        expected = sum(r["total_claims"] for r in records)
        s = summarize_records(records)
        assert s["total_claims"] == expected

    def test_supported_claims_summed(self):
        records = [record_to_dict(_make_record(f"t{i}")) for i in range(3)]
        expected = sum(r["supported_count"] for r in records)
        s = summarize_records(records)
        assert s["supported_claims"] == expected

    def test_turns_with_contradiction_counted(self):
        r1 = record_to_dict(_make_record("c1"))
        r2 = record_to_dict(_make_record("c2"))
        r1["has_contradiction"] = True
        r1["contradicted_count"] = 1
        r2["has_contradiction"] = False
        s = summarize_records([r1, r2])
        assert s["turns_with_contradiction"] == 1

    def test_turns_with_unsupported_counted(self):
        r1 = record_to_dict(_make_record("u1"))
        r2 = record_to_dict(_make_record("u2"))
        r1["has_unsupported"] = True
        r2["has_unsupported"] = True
        s = summarize_records([r1, r2])
        assert s["turns_with_unsupported"] == 2

    def test_turns_with_vague_counted(self):
        r = record_to_dict(evaluate_turn(
            "This creates positional pressure.",
            facts={},
            turn_id="vague_test",
        ))
        s = summarize_records([r])
        assert s["turns_with_vague"] == 1

    def test_reasoning_path_counts(self):
        records = []
        for i in range(3):
            d = record_to_dict(_make_record(f"sl{i}"))
            d["reasoning_path"] = "seeded_llm"
            records.append(d)
        d2 = record_to_dict(_make_record("sf"))
        d2["reasoning_path"] = "seed_fallback"
        records.append(d2)
        s = summarize_records(records)
        assert s["reasoning_path_counts"]["seeded_llm"] == 3
        assert s["reasoning_path_counts"]["seed_fallback"] == 1

    def test_trajectory_event_counts(self):
        r1 = record_to_dict(_make_record("ev1"))
        r2 = record_to_dict(_make_record("ev2"))
        r1["trajectory_events"] = ["api_failure", "retry_used"]
        r2["trajectory_events"] = ["api_failure"]
        s = summarize_records([r1, r2])
        assert s["trajectory_event_counts"]["api_failure"] == 2
        assert s["trajectory_event_counts"]["retry_used"] == 1

    def test_no_trajectory_events_gives_empty_dict(self):
        records = [record_to_dict(_make_record(f"t{i}")) for i in range(3)]
        for r in records:
            r["trajectory_events"] = []
        s = summarize_records(records)
        assert s["trajectory_event_counts"] == {}

    def test_summary_is_json_serialisable(self):
        records = [record_to_dict(_make_record(f"j{i}")) for i in range(3)]
        s = summarize_records(records)
        dumped = json.dumps(s)
        loaded = json.loads(dumped)
        assert loaded["total_turns"] == 3

    def test_counts_sum_correctly(self):
        """supported + contradicted + unsupported + vague == total_claims."""
        records = [record_to_dict(_make_record(f"sum{i}")) for i in range(4)]
        s = summarize_records(records)
        claim_total = (
            s["supported_claims"]
            + s["contradicted_claims"]
            + s["unsupported_claims"]
            + s["vague_claims"]
        )
        assert claim_total == s["total_claims"]
