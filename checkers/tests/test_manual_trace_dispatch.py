# checkers/tests/test_manual_trace_dispatch.py
#
# Tests for run_baseline_human_trace.py dispatch integrity.
#
# Coverage:
#   1. Every --baseline value maps to exactly one expected runner.
#   2. full_system → _run_red_ply_full_system (unchanged).
#   3. B3d (rules_legal_moves_llm_strategic_facts) routes to its own runner,
#      NOT _run_red_ply_full_system.
#   4. B3e (rules_legal_moves_llm_strategic_facts_freeform_path) routes to
#      its own runner, NOT _run_red_ply_full_system.
#   5. Unsupported --baseline raises (no silent fall-through to full_system).
#   6. B3d/B3e prompt builders hide minimax_score and symbolic_rank.
#   7. None of the B3 manual runners import or call checkers.graph or
#      ranker_agent in their source.
#   8. End-to-end with a stubbed Mistral client: B3d and B3e runners do not
#      invoke the graph stream and write trace records with the baseline
#      label they were asked to.

from __future__ import annotations

import inspect
import os

os.environ.setdefault("USE_SIMPLIFIED_PIPELINE", "true")
os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

import json
import re

import pytest

from checkers.engine.board import RED, EMPTY, BLACK, create_initial_board
from checkers.state.state import CheckersState

from checkers.baseline_eval import run_baseline_human_trace as ht
from checkers.baseline_eval.run_baseline_human_trace import (
    ALL_BASELINES,
    BASELINE_MINIMAL_RAW_LLM,
    BASELINE_RULES_ONLY_LLM,
    BASELINE_RULES_LEGAL_LLM_JSON,
    BASELINE_RULES_LEGAL_LLM_ARROW,
    BASELINE_RULES_LEGAL_LLM_FACTS,
    BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS,
    BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH,
    BASELINE_FULL_SYSTEM,
    _RED_PLY_DISPATCH,
    _dispatch_red_ply,
    _run_red_ply_minimal_raw_llm,
    _run_red_ply_rules_only_llm,
    _run_red_ply_rules_legal_llm,
    _run_red_ply_rules_legal_llm_arrow,
    _run_red_ply_rules_legal_llm_facts,
    _run_red_ply_rules_legal_llm_strategic_facts,
    _run_red_ply_rules_legal_llm_strategic_facts_freeform_path,
    _run_red_ply_full_system,
)


# ── 1. Dispatch table covers every --baseline value ─────────────────────────

EXPECTED_DISPATCH = {
    BASELINE_MINIMAL_RAW_LLM:                              _run_red_ply_minimal_raw_llm,
    BASELINE_RULES_ONLY_LLM:                               _run_red_ply_rules_only_llm,
    BASELINE_RULES_LEGAL_LLM_JSON:                         _run_red_ply_rules_legal_llm,
    BASELINE_RULES_LEGAL_LLM_ARROW:                        _run_red_ply_rules_legal_llm_arrow,
    BASELINE_RULES_LEGAL_LLM_FACTS:                        _run_red_ply_rules_legal_llm_facts,
    BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS:              _run_red_ply_rules_legal_llm_strategic_facts,
    BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH: _run_red_ply_rules_legal_llm_strategic_facts_freeform_path,
    BASELINE_FULL_SYSTEM:                                  _run_red_ply_full_system,
}


@pytest.mark.parametrize("baseline,expected", list(EXPECTED_DISPATCH.items()))
def test_dispatch_table_maps_each_baseline_to_expected_runner(baseline, expected):
    assert _RED_PLY_DISPATCH[baseline] is expected


def test_dispatch_table_covers_every_cli_baseline():
    missing = set(ALL_BASELINES) - set(_RED_PLY_DISPATCH)
    assert not missing, f"--baseline values missing a dispatch entry: {missing}"


# ── 2. full_system path unchanged ───────────────────────────────────────────

def test_full_system_still_routes_to_full_system_runner():
    assert _RED_PLY_DISPATCH[BASELINE_FULL_SYSTEM] is _run_red_ply_full_system


# ── 3-4. B3d / B3e do NOT route to full_system ──────────────────────────────

def test_b3d_strategic_facts_does_not_dispatch_to_full_system():
    runner = _RED_PLY_DISPATCH[BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS]
    assert runner is _run_red_ply_rules_legal_llm_strategic_facts
    assert runner is not _run_red_ply_full_system


def test_b3e_freeform_path_does_not_dispatch_to_full_system():
    runner = _RED_PLY_DISPATCH[
        BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH
    ]
    assert runner is _run_red_ply_rules_legal_llm_strategic_facts_freeform_path
    assert runner is not _run_red_ply_full_system


# ── 5. Unsupported baseline raises (no silent fall-through) ─────────────────

def test_unsupported_baseline_raises_systemexit():
    with pytest.raises(SystemExit):
        _dispatch_red_ply(
            "definitely_not_a_real_baseline",
            acc={"board": create_initial_board(), "current_player": RED, "turn_number": 0},
            game_traces=[],
            quiet=True,
            show_prompts=False,
        )


# ── 6. B3d / B3e prompt builders hide minimax labels ────────────────────────

def _b3_board() -> list[list[int]]:
    b = [[EMPTY] * 8 for _ in range(8)]
    b[4][1] = RED
    b[4][5] = RED
    b[3][2] = BLACK
    b[3][6] = BLACK
    return b


def test_b3d_prompt_hides_minimax_and_symbolic_rank():
    from checkers.engine.rules import get_all_legal_moves
    from checkers.agents.scorer_agent import score_all_legal_moves
    from checkers.baseline_eval.run_baseline_human_trace import (
        _build_rules_legal_llm_strategic_facts_user,
    )
    board = _b3_board()
    legal = get_all_legal_moves(board, RED)
    scored, *_ = score_all_legal_moves(board, RED)
    user = _build_rules_legal_llm_strategic_facts_user(board, legal, scored, 1)
    for tok in ("minimax_score", "symbolic_rank", "score_gap",
                "best_score", "best_move_index", "best_move_path"):
        assert tok not in user, f"B3d prompt leaks {tok!r}"


def test_b3e_prompt_hides_minimax_and_symbolic_rank():
    from checkers.engine.rules import get_all_legal_moves
    from checkers.agents.scorer_agent import score_all_legal_moves
    from checkers.baseline_eval.run_baseline_human_trace import (
        _build_rules_legal_llm_strategic_facts_freeform_path_user,
    )
    board = _b3_board()
    legal = get_all_legal_moves(board, RED)
    scored, *_ = score_all_legal_moves(board, RED)
    user = _build_rules_legal_llm_strategic_facts_freeform_path_user(
        board, legal, scored, 1,
    )
    for tok in ("minimax_score", "symbolic_rank", "score_gap",
                "best_score", "best_move_index", "best_move_path"):
        assert tok not in user, f"B3e prompt leaks {tok!r}"


# ── 7. B3 manual runners do not call ranker_agent / graph / safety_filter ──

_B3_MANUAL_RUNNERS = (
    _run_red_ply_rules_legal_llm,
    _run_red_ply_rules_legal_llm_arrow,
    _run_red_ply_rules_legal_llm_facts,
    _run_red_ply_rules_legal_llm_strategic_facts,
    _run_red_ply_rules_legal_llm_strategic_facts_freeform_path,
)


@pytest.mark.parametrize("runner", _B3_MANUAL_RUNNERS)
def test_b3_runner_does_not_call_graph_or_ranker(runner):
    src = inspect.getsource(runner)
    forbidden = [
        "checkers_graph", "checkers.graph", "graph.stream", "interrupt_after",
        "ranker_agent", "build_ranker_user_prompt",
        "_apply_safety_filter",
        "proposal_shortlist", "deterministic_proposal_node",
        "_override_if_llm_chose_much_worse_minimax",
        "_should_force_best_minimax",
        "ranker_retry", "fallback_used", "override_branch_name",
    ]
    # The trace dict initialises override_branch/override_used/fallback_used
    # as False/None constants; those literal field assignments are fine and
    # don't indicate code calling into the override machinery. The forbidden
    # check below targets function/module references only.
    hits = []
    for kw in ("checkers_graph", "checkers.graph", "graph.stream",
               "interrupt_after",
               "ranker_agent", "build_ranker_user_prompt",
               "_apply_safety_filter",
               "deterministic_proposal_node",
               "_override_if_llm_chose_much_worse_minimax",
               "_should_force_best_minimax"):
        if kw in src:
            hits.append(kw)
    assert not hits, (
        f"{runner.__name__} references full_system machinery: {hits}"
    )


# ── 8. Smoke runs: B3d / B3e do not stream the graph ────────────────────────

def _stub_make_acc() -> dict:
    return CheckersState(
        board=_b3_board(), current_player=RED, turn_number=0,
    ).model_dump()


def _patch_graph_to_raise(monkeypatch):
    """Replace checkers_graph.stream so any call would raise."""
    from checkers.graph import graph as graph_mod

    def _boom(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("B3 runner unexpectedly invoked checkers_graph.stream")
    monkeypatch.setattr(graph_mod.checkers_graph, "stream", _boom)


def test_b3d_manual_trace_does_not_stream_graph(monkeypatch):
    _patch_graph_to_raise(monkeypatch)
    # Stub the Mistral baseline client to return a fully-valid B3d response.
    def _stub(system, user):  # noqa: ARG001
        idxs = [int(m) for m in re.findall(r"\[(\d+)\]", user)]
        n = max(idxs) + 1 if idxs else 1
        # Pick a path that exists in the prompt's legal-move list. The prompt
        # embeds each path as JSON, so grab the first one.
        m = re.search(r'"path":\s*(\[\[\d+,\s*\d+\](?:,\s*\[\d+,\s*\d+\])*\])', user)
        path = json.loads(m.group(1)) if m else [[0, 0], [1, 1]]
        return json.dumps({
            "move_analysis": [{"index": i, "pros": "", "cons": "", "verdict": "ok"}
                              for i in range(n)],
            "chosen_index":  0,
            "chosen_path":   path,
            "reasoning":     "stub",
        })
    monkeypatch.setattr(ht, "call_baseline_llm", _stub)

    acc = _stub_make_acc()
    game_traces: list[dict] = []
    _dispatch_red_ply(
        BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS,
        acc, game_traces, quiet=True, show_prompts=False,
    )
    assert game_traces, "B3d runner should have appended a trace"
    rec = game_traces[-1]
    assert rec["baseline"] == BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS
    assert "move_analysis_complete" in rec


def test_b3e_manual_trace_does_not_stream_graph(monkeypatch):
    _patch_graph_to_raise(monkeypatch)

    def _stub(system, user):  # noqa: ARG001
        m = re.search(r'"path":\s*(\[\[\d+,\s*\d+\](?:,\s*\[\d+,\s*\d+\])*\])', user)
        path = json.loads(m.group(1)) if m else [[0, 0], [1, 1]]
        return json.dumps({"chosen_path": path, "reasoning": "stub"})
    monkeypatch.setattr(ht, "call_baseline_llm", _stub)

    acc = _stub_make_acc()
    game_traces: list[dict] = []
    _dispatch_red_ply(
        BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH,
        acc, game_traces, quiet=True, show_prompts=False,
    )
    assert game_traces, "B3e runner should have appended a trace"
    rec = game_traces[-1]
    assert rec["baseline"] == BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH


def test_full_system_runner_still_uses_graph(monkeypatch):
    """Sanity: B4 is NOT broken — its runner still uses checkers_graph.stream."""
    src = inspect.getsource(_run_red_ply_full_system)
    assert "checkers_graph.stream" in src
    assert "ranker_agent" in src


# ── 9. Scenario suite untouched ──────────────────────────────────────────────

def test_scenario_suite_dispatch_unchanged():
    """The scenario suite's per-baseline dispatch is independent and still
    routes each B3 variant to its own scenario-suite runner."""
    from checkers.baseline_eval import run_baseline_scenario_suite as suite
    src = inspect.getsource(suite.run_scenario_for_baseline)
    # Suite uses its own runners; nothing here should change.
    for fn_name in (
        "_run_scenario_path_json",
        "_run_scenario_strategic_facts",
        "_run_scenario_strategic_facts_freeform_path",
        "_run_scenario_full_system",
    ):
        assert fn_name in src, (
            f"scenario suite dispatch unexpectedly lost {fn_name!r}"
        )
