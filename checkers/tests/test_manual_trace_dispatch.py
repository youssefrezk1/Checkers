# checkers/tests/test_manual_trace_dispatch.py
#
# Tests for run_baseline_human_trace.py dispatch integrity.
#
# Coverage:
#   1. Every --baseline value maps to exactly one expected runner.
#   2. full_system → _run_red_ply_full_system (unchanged).
#   3. Unsupported --baseline raises (no silent fall-through to full_system).
#   4. The scenario suite's per-baseline dispatch is independent.

from __future__ import annotations

import inspect
import os

os.environ.setdefault("USE_SIMPLIFIED_PIPELINE", "true")
os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

import pytest

from checkers.engine.board import RED, create_initial_board
from checkers.state.state import CheckersState

from checkers.baseline_eval import run_baseline_human_trace as ht
from checkers.baseline_eval.run_baseline_human_trace import (
    ALL_BASELINES,
    BASELINE_MINIMAL_RAW_LLM,
    BASELINE_RULES_ONLY_LLM,
    BASELINE_FULL_SYSTEM,
    _RED_PLY_DISPATCH,
    _dispatch_red_ply,
    _run_red_ply_minimal_raw_llm,
    _run_red_ply_rules_only_llm,
    _run_red_ply_full_system,
)


# ── 1. Dispatch table covers every --baseline value ─────────────────────────

EXPECTED_DISPATCH = {
    BASELINE_MINIMAL_RAW_LLM: _run_red_ply_minimal_raw_llm,
    BASELINE_RULES_ONLY_LLM:  _run_red_ply_rules_only_llm,
    BASELINE_FULL_SYSTEM:     _run_red_ply_full_system,
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


def test_full_system_runner_still_uses_graph():
    """Sanity: B4 is NOT broken — its runner still uses checkers_graph.stream."""
    src = inspect.getsource(_run_red_ply_full_system)
    assert "checkers_graph.stream" in src


# ── 3. Unsupported baseline raises (no silent fall-through) ─────────────────

def test_unsupported_baseline_raises_systemexit():
    with pytest.raises(SystemExit):
        _dispatch_red_ply(
            "definitely_not_a_real_baseline",
            acc={"board": create_initial_board(), "current_player": RED, "turn_number": 0},
            game_traces=[],
            quiet=True,
            show_prompts=False,
        )


# ── 4. Scenario suite untouched ──────────────────────────────────────────────

def test_scenario_suite_dispatch_unchanged():
    """The scenario suite's per-baseline dispatch is independent and still
    routes each B3 variant to its own scenario-suite runner."""
    from checkers.baseline_eval import run_baseline_scenario_suite as suite
    src = inspect.getsource(suite.run_scenario_for_baseline)
    for fn_name in (
        "_run_scenario_path_json",
        "_run_scenario_strategic_facts",
        "_run_scenario_strategic_facts_freeform_path",
        "_run_scenario_full_system",
    ):
        assert fn_name in src, (
            f"scenario suite dispatch unexpectedly lost {fn_name!r}"
        )
