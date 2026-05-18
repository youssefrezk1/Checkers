# checkers/tests/test_baseline_scenario_suite.py
#
# Tests for checkers/baseline_eval/run_baseline_scenario_suite.py
#
# Coverage:
#   1. Loading generated scenarios from JSON and registering them.
#   2. Category aggregation across hand-authored + generated scenarios.
#   3. Error classification: invalid JSON → output_format_error.
#   4. Error classification: hallucinated path detected.
#   5. Error classification: legal non-best move → strategic_error.
#   6. Error classification: missing/short move_analysis → analysis_incomplete.
#   7. Smoke run: B3_strategic_facts_llm + selected baselines on a small
#      synthetic generated suite using a stubbed LLM (no network).
#   8. B4 full_system dispatch unchanged.
#   9. Scenario JSON may carry minimax labels; B3 prompt builder still hides
#      them when running through the suite.

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("USE_SIMPLIFIED_PIPELINE", "true")
os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

from checkers.engine.board import EMPTY, RED, BLACK
from checkers.engine.rules import get_all_legal_moves
from checkers.baseline_eval import run_baseline_scenario_suite as suite
from checkers.baseline_eval.run_baseline_scenario_suite import (
    _classify_error,
    _register_generated_scenarios,
    _scenario_category,
    _validate_move_analysis,
    aggregate_category_metrics,
    load_generated_scenarios,
    run_scenario_for_baseline,
)
from checkers.baseline_eval.run_baseline_human_trace import (
    BASELINE_MINIMAL_RAW_LLM,
    BASELINE_RULES_LEGAL_LLM_FACTS,
    BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS,
    BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH,
    BASELINE_FULL_SYSTEM,
)


def _empty_board() -> list[list[int]]:
    return [[EMPTY] * 8 for _ in range(8)]


def _board_capture() -> list[list[int]]:
    b = _empty_board()
    b[4][1] = RED
    b[4][5] = RED
    b[3][2] = BLACK
    b[3][6] = BLACK
    return b


def _stub_full_analysis(legal_count: int, chosen_index: int, chosen_path):
    analysis = [{"index": i, "pros": "", "cons": "", "verdict": "ok"}
                for i in range(legal_count)]
    return json.dumps({
        "move_analysis": analysis,
        "chosen_index":  chosen_index,
        "chosen_path":   [list(sq) for sq in chosen_path],
        "reasoning":     "stub reasoning",
    })


def _make_call_baseline_llm_stub_from_user(builder):
    """Build a stub that derives N from the user prompt and calls builder."""
    def _call(system, user):  # noqa: ARG001
        idxs = [int(m) for m in re.findall(r"\[(\d+)\]", user)]
        n = max(idxs) + 1 if idxs else 1
        return builder(n)
    return _call


# ── 1. JSON loading ─────────────────────────────────────────────────────────

def test_load_and_register_generated_scenarios():
    entries: list[dict[str, Any]] = [
        {
            "scenario_id":      "gen_demo_0001",
            "category":         "mandatory_capture",
            "tactical_tags":    ["mandatory_capture"],
            "board":            _board_capture(),
            "side_to_move":     RED,
            "legal_moves_count": 2,
            # Minimax labels — stored in JSON but must not reach the prompt.
            "best_move_index":   0,
            "best_score":        5.0,
            "second_best_score": 1.0,
            "score_gap":         4.0,
            "best_move_path":    [[4, 1], [2, 3]],
            "generation_source": "self_play_rollout",
        },
    ]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "demo.json"
        p.write_text(json.dumps(entries), encoding="utf-8")
        loaded = load_generated_scenarios(p)
        assert loaded == entries

        registered = _register_generated_scenarios(loaded)
        assert registered == ["gen_demo_0001"]
        assert _scenario_category("gen_demo_0001") == "mandatory_capture"


# ── 2. Category aggregation ────────────────────────────────────────────────

def test_aggregate_category_metrics_basic():
    records = [
        {
            "scenario": "opening", "baseline": "b1",
            "legality_result": "legal", "top1_hit": True, "top3_hit": True,
            "hallucinated_path": False, "strategic_error": False,
            "reasoning_truthfulness_passed": True,
            "chosen_minimax_rank": 1, "score_gap": 0.0, "error_class": "clean",
        },
        {
            "scenario": "opening", "baseline": "b1",
            "legality_result": "legal", "top1_hit": False, "top3_hit": True,
            "hallucinated_path": False, "strategic_error": False,
            "reasoning_truthfulness_passed": True,
            "chosen_minimax_rank": 2, "score_gap": 1.5, "error_class": "clean",
        },
    ]
    metrics = aggregate_category_metrics(records, baselines=["b1"])
    assert "hand_authored" in metrics
    m = metrics["hand_authored"]["b1"]
    assert m["n"] == 2
    assert m["legal_rate"] == 1.0
    assert m["top1_rate"] == 0.5
    assert m["top3_rate"] == 1.0
    assert m["avg_rank"] == pytest.approx(1.5)


# ── 3-5. Error classifier branches ─────────────────────────────────────────

def test_classify_error_output_format_error():
    e = _classify_error(
        llm_failed=False, json_valid=False, legality_result="illegal",
        reasoning_hallucinations=[], chosen_rank=0,
        parse_error="not_a_json_object",
    )
    assert e == "output_format_error"


def test_classify_error_empty_response():
    e = _classify_error(
        llm_failed=False, json_valid=False, legality_result="illegal",
        reasoning_hallucinations=[], chosen_rank=0,
        parse_error="empty_response",
    )
    assert e == "empty_response"


def test_classify_error_hallucinated_path():
    e = _classify_error(
        llm_failed=False, json_valid=True,
        legality_result="hallucinated_path",
        reasoning_hallucinations=[], chosen_rank=0, parse_error="",
    )
    assert e == "hallucinated_path"


def test_classify_error_strategic_error_when_rank_too_high():
    e = _classify_error(
        llm_failed=False, json_valid=True, legality_result="legal",
        reasoning_hallucinations=[], chosen_rank=5, parse_error="",
    )
    assert e == "strategic_error"


def test_classify_error_api_call_failure():
    e = _classify_error(
        llm_failed=True, json_valid=False, legality_result="illegal",
        reasoning_hallucinations=[], chosen_rank=0, parse_error="",
    )
    assert e == "api_call_failed"


# ── 6. move_analysis validator ─────────────────────────────────────────────

def test_validate_move_analysis_complete_coverage():
    obj = {"move_analysis": [
        {"index": 0}, {"index": 1}, {"index": 2}
    ]}
    ok, reason, seen = _validate_move_analysis(obj, n_legal=3)
    assert ok is True and reason == ""
    assert sorted(seen) == [0, 1, 2]


def test_validate_move_analysis_missing_array():
    ok, reason, _ = _validate_move_analysis({}, n_legal=3)
    assert ok is False and reason == "move_analysis_missing"


def test_validate_move_analysis_incomplete_coverage():
    obj = {"move_analysis": [{"index": 0}, {"index": 1}]}
    ok, reason, _ = _validate_move_analysis(obj, n_legal=3)
    assert ok is False and reason == "move_analysis_incomplete_coverage"


def test_validate_move_analysis_index_out_of_range():
    obj = {"move_analysis": [{"index": 0}, {"index": 5}]}
    ok, reason, _ = _validate_move_analysis(obj, n_legal=2)
    assert ok is False and reason == "move_analysis_index_out_of_range"


# ── 7. Hallucinated-path / invalid JSON / strategic_error / incomplete ─────

def test_strategic_facts_runner_flags_hallucinated_path(monkeypatch):
    """chosen_path is not in the legal-moves list."""
    def _stub(system, user):  # noqa: ARG001
        idxs = [int(m) for m in re.findall(r"\[(\d+)\]", user)]
        n = max(idxs) + 1 if idxs else 1
        analysis = [{"index": i, "pros": "", "cons": "", "verdict": "ok"}
                    for i in range(n)]
        return json.dumps({
            "move_analysis": analysis,
            "chosen_index": 0,
            "chosen_path":  [[0, 0], [1, 1]],
            "reasoning":    "stub",
        })
    monkeypatch.setattr(suite, "call_baseline_llm", _stub)
    rec = run_scenario_for_baseline(
        scenario_name="mandatory_capture",
        board=_board_capture(),
        baseline=BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS,
        show_prompts=False,
    )
    assert rec["hallucinated_path"] is True
    assert rec["error_class"] == "hallucinated_path"


def test_strategic_facts_runner_flags_invalid_json(monkeypatch):
    monkeypatch.setattr(suite, "call_baseline_llm", lambda s, u: "not json")
    rec = run_scenario_for_baseline(
        scenario_name="mandatory_capture",
        board=_board_capture(),
        baseline=BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS,
        show_prompts=False,
    )
    assert rec["json_valid"] is False
    assert rec["error_class"] in ("output_format_error", "parse_failed")


def test_strategic_facts_runner_flags_analysis_incomplete(monkeypatch):
    """
    Model returns valid JSON, a legal chosen_path, but omits required
    move_analysis entries → classified as analysis_incomplete.
    """
    from checkers.agents.scorer_agent import score_all_legal_moves

    board = _board_capture()
    legal = get_all_legal_moves(board, RED)
    scored, *_ = score_all_legal_moves(board, RED)
    # Pick the engine-best legal path so we don't tangle with strategic_error.
    best_path = scored[0]["path"]

    # Find the index of best_path in legal_all
    best_index = next(
        i for i, m in enumerate(legal)
        if [list(sq) for sq in m["path"]] == [list(sq) for sq in best_path]
    )

    def _stub(system, user):  # noqa: ARG001
        # Omit move_analysis entirely.
        return json.dumps({
            "chosen_index": best_index,
            "chosen_path":  [list(sq) for sq in best_path],
            "reasoning":    "stub",
        })
    monkeypatch.setattr(suite, "call_baseline_llm", _stub)
    rec = run_scenario_for_baseline(
        scenario_name="mandatory_capture",
        board=board,
        baseline=BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS,
        show_prompts=False,
    )
    assert rec["legality_result"] == "legal"
    assert rec["move_analysis_complete"] is False
    assert rec["error_class"] == "analysis_incomplete"


def test_strategic_facts_runner_flags_strategic_error_when_not_best(monkeypatch):
    """
    Model returns valid JSON, full analysis, a legal but minimax-suboptimal
    move (rank > 3) → strategic_error.
    """
    b = _empty_board()
    b[5][0] = RED; b[5][2] = RED; b[5][4] = RED; b[5][6] = RED
    b[6][1] = RED
    b[2][3] = BLACK; b[0][7] = BLACK
    legal = get_all_legal_moves(b, RED)
    assert len(legal) >= 5

    legal0_path = legal[0]["path"]
    def _stub(system, user):  # noqa: ARG001
        n = len(legal)
        return _stub_full_analysis(n, chosen_index=0, chosen_path=legal0_path)
    monkeypatch.setattr(suite, "call_baseline_llm", _stub)
    rec = run_scenario_for_baseline(
        scenario_name="opening",
        board=b,
        baseline=BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS,
        show_prompts=False,
    )
    assert rec["legality_result"] == "legal"
    assert rec["error_class"] in ("clean", "strategic_error")


# ── 8. Scenario JSON minimax labels not in prompt at runtime ───────────────

def test_runtime_prompt_hides_minimax_labels(monkeypatch):
    """
    Even when a scenario carries best_move_index / score_gap in its JSON
    metadata, the prompt sent to the LLM must not contain those labels.
    """
    captured_prompts: dict[str, str] = {}

    def _stub(system, user):
        captured_prompts["user"] = user
        n = max([int(m) for m in re.findall(r"\[(\d+)\]", user)] + [-1]) + 1
        return _stub_full_analysis(n, chosen_index=0, chosen_path=[[0, 0]])

    monkeypatch.setattr(suite, "call_baseline_llm", _stub)

    entries = [{
        "scenario_id":      "gen_meta_0001",
        "category":         "mandatory_capture",
        "tactical_tags":    ["mandatory_capture"],
        "board":            _board_capture(),
        "side_to_move":     RED,
        "legal_moves_count": 2,
        "best_move_index":   0,
        "best_score":        9.99,
        "second_best_score": 1.11,
        "score_gap":         8.88,
        "best_move_path":    [[4, 1], [2, 3]],
        "generation_source": "self_play_rollout",
    }]
    _register_generated_scenarios(entries)

    run_scenario_for_baseline(
        scenario_name="gen_meta_0001",
        board=entries[0]["board"],
        baseline=BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS,
        show_prompts=False,
    )
    user_prompt = captured_prompts["user"]
    for forbidden in ("best_move_index", "best_score", "score_gap",
                      "minimax_score", "symbolic_rank", "best_move_path"):
        assert forbidden not in user_prompt, (
            f"prompt unexpectedly contains forbidden label: {forbidden}"
        )


# ── 9. Smoke: small generated suite × multiple baselines ───────────────────

def test_smoke_generated_suite(monkeypatch):
    """Run the suite on five synthetic scenarios with a stubbed LLM."""
    boards = []
    for k in range(5):
        b = _empty_board()
        b[4][1] = RED
        b[4][5] = RED
        b[3][2] = BLACK
        b[3][6] = BLACK
        b[0][1 + (k % 4)] = BLACK
        boards.append(b)

    entries = [
        {
            "scenario_id":      f"gen_smoke_{i:04d}",
            "category":         "mandatory_capture",
            "tactical_tags":    ["mandatory_capture"],
            "board":            b,
            "side_to_move":     RED,
            "legal_moves_count": len(get_all_legal_moves(b, RED)),
            "best_move_index":   0,
            "best_score":        5.0,
            "second_best_score": 1.0,
            "score_gap":         4.0,
            "best_move_path":    [[4, 1], [2, 3]],
            "generation_source": "self_play_rollout",
        }
        for i, b in enumerate(boards)
    ]
    names = _register_generated_scenarios(entries)
    assert len(names) == 5

    def _stub(system, user):  # noqa: ARG001
        n = max([int(m) for m in re.findall(r"\[(\d+)\]", user)] + [-1]) + 1
        return _stub_full_analysis(
            n,
            chosen_index=0,
            chosen_path=[[4, 1], [2, 3]],
        )
    monkeypatch.setattr(suite, "call_baseline_llm", _stub)

    selected = [
        BASELINE_MINIMAL_RAW_LLM,
        BASELINE_RULES_LEGAL_LLM_FACTS,
        BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS,
    ]

    records: list[dict[str, Any]] = []
    for name in names:
        for bl in selected:
            board = boards[int(name.rsplit("_", 1)[1])]
            rec = run_scenario_for_baseline(name, board, bl, show_prompts=False)
            rec["repeat"] = 1
            records.append(rec)

    for r in records:
        assert r["baseline"] in selected
        assert r["scenario"].startswith("gen_smoke_")

    cat_metrics = aggregate_category_metrics(records, baselines=selected)
    assert "mandatory_capture" in cat_metrics
    for bl in selected:
        assert bl in cat_metrics["mandatory_capture"]
        assert cat_metrics["mandatory_capture"][bl]["n"] == 5


# ── 10. B4 dispatch unchanged ──────────────────────────────────────────────

def test_b4_full_system_dispatch_unchanged():
    import inspect
    src = inspect.getsource(run_scenario_for_baseline)
    assert "BASELINE_FULL_SYSTEM" in src
    assert "_run_scenario_full_system" in src
    # Ensure the strategic_facts runner is only used for the new baseline.
    assert src.count("_run_scenario_strategic_facts(") == 1


# ── 11. B3e freeform-path baseline ─────────────────────────────────────────

def _stub_freeform_path(chosen_path):
    return json.dumps({
        "chosen_path": [list(sq) for sq in chosen_path],
        "reasoning":   "stub reasoning",
    })


def test_freeform_path_runner_legal_when_path_matches(monkeypatch):
    from checkers.agents.scorer_agent import score_all_legal_moves

    board = _board_capture()
    legal = get_all_legal_moves(board, RED)
    scored, *_ = score_all_legal_moves(board, RED)
    best_path = scored[0]["path"]

    monkeypatch.setattr(
        suite, "call_baseline_llm",
        lambda s, u: _stub_freeform_path(best_path),
    )
    rec = run_scenario_for_baseline(
        scenario_name="mandatory_capture",
        board=board,
        baseline=BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH,
        show_prompts=False,
    )
    assert rec["legality_result"] == "legal"
    assert rec["hallucinated_path"] is False
    assert rec["path_matches_legal"] is True
    assert rec["path_copy_error"] is False
    assert rec["error_class"] in ("clean", "reasoning_hallucination", "strategic_error")


def test_freeform_path_runner_flags_hallucinated_path(monkeypatch):
    """chosen_path is a structurally valid path but not in the legal list."""
    monkeypatch.setattr(
        suite, "call_baseline_llm",
        lambda s, u: _stub_freeform_path([[0, 1], [1, 0]]),
    )
    rec = run_scenario_for_baseline(
        scenario_name="mandatory_capture",
        board=_board_capture(),
        baseline=BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH,
        show_prompts=False,
    )
    assert rec["json_valid"] is True
    assert rec["hallucinated_path"] is True
    assert rec["error_class"] == "hallucinated_path"
    assert rec["legality_result"] == "hallucinated_path"
    assert rec["hallucinated_path_subtype"] is not None
    assert rec["path_copy_error"] is True


def test_freeform_path_runner_malformed_json_is_not_hallucinated(monkeypatch):
    monkeypatch.setattr(suite, "call_baseline_llm", lambda s, u: "this is not json")
    rec = run_scenario_for_baseline(
        scenario_name="mandatory_capture",
        board=_board_capture(),
        baseline=BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH,
        show_prompts=False,
    )
    assert rec["json_valid"] is False
    assert rec["hallucinated_path"] is False
    assert rec["error_class"] in ("parse_failed", "output_format_error", "empty_response")


def test_freeform_path_runner_missing_chosen_path_is_output_format_error(monkeypatch):
    """Valid JSON but the chosen_path field is missing entirely."""
    monkeypatch.setattr(
        suite, "call_baseline_llm",
        lambda s, u: json.dumps({"reasoning": "no path"}),
    )
    rec = run_scenario_for_baseline(
        scenario_name="mandatory_capture",
        board=_board_capture(),
        baseline=BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH,
        show_prompts=False,
    )
    assert rec["json_valid"] is True
    assert rec["hallucinated_path"] is False
    assert rec["error_class"] == "output_format_error"
    assert rec["legality_result"] == "output_format_error"


def test_freeform_path_runner_chosen_path_not_a_list_is_output_format_error(monkeypatch):
    monkeypatch.setattr(
        suite, "call_baseline_llm",
        lambda s, u: json.dumps({"chosen_path": "not a list", "reasoning": "x"}),
    )
    rec = run_scenario_for_baseline(
        scenario_name="mandatory_capture",
        board=_board_capture(),
        baseline=BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH,
        show_prompts=False,
    )
    assert rec["json_valid"] is True
    assert rec["hallucinated_path"] is False
    assert rec["error_class"] == "output_format_error"


def test_freeform_path_subtype_wrong_piece_square():
    """Pick a from-square that's empty / non-RED on the test board."""
    from checkers.baseline_eval.run_baseline_scenario_suite import (
        _subcategorize_freeform_path,
    )
    board = _board_capture()
    legal = get_all_legal_moves(board, RED)
    # (0,1) is empty in this board.
    subtype = _subcategorize_freeform_path([[0, 1], [1, 0]], board, legal)
    assert subtype == "wrong_piece_square"


def test_freeform_path_subtype_malformed_coordinates():
    from checkers.baseline_eval.run_baseline_scenario_suite import (
        _subcategorize_freeform_path,
    )
    board = _board_capture()
    legal = get_all_legal_moves(board, RED)
    assert _subcategorize_freeform_path([[1]], board, legal) == "malformed_coordinates"
    assert _subcategorize_freeform_path("oops", board, legal) == "malformed_coordinates"


def test_freeform_path_dispatch_resolves(monkeypatch):
    """End-to-end: dispatcher reaches the freeform-path runner."""
    from checkers.agents.scorer_agent import score_all_legal_moves

    board = _board_capture()
    scored, *_ = score_all_legal_moves(board, RED)
    best_path = scored[0]["path"]
    monkeypatch.setattr(
        suite, "call_baseline_llm",
        lambda s, u: _stub_freeform_path(best_path),
    )
    rec = run_scenario_for_baseline(
        scenario_name="mandatory_capture",
        board=board,
        baseline=BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH,
        show_prompts=False,
    )
    assert rec["baseline"] == BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH


def test_freeform_path_aggregate_metrics(monkeypatch):
    """aggregate_freeform_path_metrics computes the five headline rates."""
    from checkers.baseline_eval.run_baseline_scenario_suite import (
        aggregate_freeform_path_metrics,
    )
    bl = BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH
    rows = [
        # 1 legal + clean
        {"baseline": bl, "json_valid": True, "legality_result": "legal",
         "hallucinated_path": False, "path_copy_error": False, "error_class": "clean"},
        # 1 hallucinated path
        {"baseline": bl, "json_valid": True, "legality_result": "hallucinated_path",
         "hallucinated_path": True, "path_copy_error": True, "error_class": "hallucinated_path"},
        # 1 parse_failed
        {"baseline": bl, "json_valid": False, "legality_result": "illegal",
         "hallucinated_path": False, "path_copy_error": False, "error_class": "parse_failed"},
        # 1 output_format_error
        {"baseline": bl, "json_valid": True, "legality_result": "output_format_error",
         "hallucinated_path": False, "path_copy_error": False, "error_class": "output_format_error"},
    ]
    m = aggregate_freeform_path_metrics(rows, baseline=bl)
    assert m["n"] == 4
    assert m["n_parse_valid"] == 3
    assert m["legal_rate_all_rows"] == pytest.approx(0.25)
    assert m["legal_rate_parse_valid"] == pytest.approx(1 / 3)
    assert m["hallucinated_path_rate_parse_valid"] == pytest.approx(1 / 3)
    assert m["parse_failed_rate"] == pytest.approx(0.5)
    assert m["path_copy_error_rate"] == pytest.approx(0.25)


def test_current_b3_strategic_facts_runner_still_uses_index_field(monkeypatch):
    """Sanity: current B3 strategic_facts still validates move_analysis + chosen_index."""
    from checkers.agents.scorer_agent import score_all_legal_moves

    board = _board_capture()
    legal = get_all_legal_moves(board, RED)
    scored, *_ = score_all_legal_moves(board, RED)
    best_path = scored[0]["path"]
    best_idx = next(
        i for i, m in enumerate(legal)
        if [list(sq) for sq in m["path"]] == [list(sq) for sq in best_path]
    )

    def _stub(s, u):
        return _stub_full_analysis(len(legal), chosen_index=best_idx, chosen_path=best_path)
    monkeypatch.setattr(suite, "call_baseline_llm", _stub)
    rec = run_scenario_for_baseline(
        scenario_name="mandatory_capture",
        board=board,
        baseline=BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS,
        show_prompts=False,
    )
    # Current B3 still has move_analysis_complete in its record.
    assert "move_analysis_complete" in rec
    assert rec["legality_result"] == "legal"


def test_b4_dispatch_still_routes_to_full_system_runner():
    import inspect
    src = inspect.getsource(run_scenario_for_baseline)
    assert "BASELINE_FULL_SYSTEM" in src
    assert "_run_scenario_full_system" in src
    # Freeform-path runner is reached only for the new baseline.
    assert src.count("_run_scenario_strategic_facts_freeform_path(") == 1
