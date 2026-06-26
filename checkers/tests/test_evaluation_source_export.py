"""
checkers/tests/test_evaluation_source_export.py

End-to-end integration test for the evaluation-source JSONL export.

PURPOSE
-------
After Phase-5 evaluation export was added, the lifecycle of `chosen_move_facts`
became:
    explainer_agent sets state.chosen_move_facts
    state_manager   clears it (Phase A)       ← happens BEFORE logger_node
    updater_agent   snapshots it and restores it on a log-only state copy
    logger_node   writes evaluation_source/<game_log_id>.jsonl

This sequence is brittle: any reordering of state_manager → logger_node or
any future graph rewrite that re-introduces state_manager between the
snapshot and the logger call would silently drop the export.

This test exercises one full `updater_agent` invocation and asserts the
JSONL artifact contains the required non-null fields.  It also confirms that
the final merged dict does NOT carry `chosen_move_facts` forward into the next turn.
"""

from __future__ import annotations

import json
import os
import tempfile
import importlib
from pathlib import Path

import pytest

from checkers.engine.board import EMPTY, RED, BLACK
from checkers.engine.rules import get_all_legal_moves
from checkers.engine.move_facts import compute_move_facts
from checkers.state.state import CheckersState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_board() -> list[list[int]]:
    return [[EMPTY] * 8 for _ in range(8)]


def _simple_red_board() -> list[list[int]]:
    """A minimal but legal mid-game position with at least one quiet RED move."""
    board = _empty_board()
    board[5][2] = RED
    board[5][4] = RED
    board[2][3] = BLACK
    board[2][5] = BLACK
    return board


def _pick_red_simple_move(board: list[list[int]]) -> dict:
    """Return a simple (non-capture) RED move enriched with engine facts."""
    legal = get_all_legal_moves(board, RED)
    for m in legal:
        if m.get("type") == "simple":
            facts = compute_move_facts(board, m, RED)
            return {**m, "facts": dict(facts)}
    raise RuntimeError("expected at least one simple RED move on this board")


def _make_state_with_chosen(tmp_log_dir: Path) -> tuple[CheckersState, dict]:
    """
    Build a CheckersState that imitates the post-explainer state:
      - chosen_move set
      - chosen_move_facts populated (as explainer_agent does)
      - last_move_reasoning set
      - ranker_diagnostics set with reasoning_seeds list
      - game_log_id pinned so the test knows which JSONL file to inspect
    """
    board = _simple_red_board()
    chosen = _pick_red_simple_move(board)
    seeds = [
        "opponent_can_recapture=false — no immediate recapture",
        "captures_count=0 — positional move focused on improving piece placement",
        f"minimax_score={chosen['facts'].get('minimax_score', 0.0):.2f} — highest-evaluated option",
    ]
    diagnostics = {
        "reasoning_seeds": seeds,
        "api_call_failure_count": 0,
        "override_branch_name": None,
        "best_score_tie_count": 1,
        "minimax_best_path": chosen["path"],
        "minimax_best_score": chosen["facts"].get("minimax_score"),
        "tied_candidate_paths": [chosen["path"]],
        "tie_break_reason": None,
        "retry_all_paths": [],
        "retry_rejection_reasons": [],
        "reasoning_initial_contradictions": [],
        "reasoning_final_contradictions": [],
        "reasoning_has_unresolved_contradiction": False,
        "reasoning_refinement_retry_count": 0,
        "reasoning_is_seed_fallback": False,
        "raw_llm_reasoning_pre_refinement": "Test reasoning.",
        "final_chosen_path": chosen["path"],
        "final_choice_source": "raw_llm",
    }
    state = CheckersState(
        board=board,
        current_player=RED,
        turn_number=4,
        legal_moves=get_all_legal_moves(board, RED),
        chosen_move=chosen,
        last_move_reasoning="Test reasoning: quiet move with no recapture risk.",
        chosen_move_facts=chosen["facts"],
        explainer_diagnostics=diagnostics,
        game_log_id="game_test_eval_export",
        strategic_context={"score_state": "EQUAL", "game_phase": "MIDGAME"},
    )
    return state, chosen


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_log_dir(tmp_path, monkeypatch):
    """Redirect logger_node output to a temp dir and disable terminal printing.
    The logger_node module reads LOG_DIR at *call time* via os.environ in some
    code paths, but resolves the module-level constant once at import.  We
    monkeypatch the module attribute directly to be safe.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Disable terminal prints to keep test output clean.
    monkeypatch.setenv("CHECKERS_LOGGER_PRINT", "false")
    monkeypatch.setenv("CHECKERS_LOG_DIR", str(log_dir))

    # Force-reload logger_node so its module-level LOG_DIR/PRINT constants
    # pick up the new env vars.  update_agent imports logger_node at import
    # time but calls the function each turn, so reloading is sufficient.
    import checkers.nodes.logger_node as logger_node_mod
    importlib.reload(logger_node_mod)

    # update_agent caches the original logger_node symbol — reload it too.
    import checkers.agents.updater_agent as update_agent_mod
    importlib.reload(update_agent_mod)

    yield log_dir

    # Best-effort restore so subsequent tests see fresh modules.
    importlib.reload(logger_node_mod)
    importlib.reload(update_agent_mod)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvaluationSourceJSONLExport:
    """The update_agent → state_manager → logger_node pipeline must produce a
    JSONL line with non-null chosen_move_facts and non-empty reasoning_seeds.
    """

    def _run_one_ply(self, log_dir: Path):
        from checkers.agents.updater_agent import updater_agent as update_agent
        state, chosen = _make_state_with_chosen(log_dir)
        merged = update_agent(state)
        return state, chosen, merged

    def _read_eval_jsonl(self, log_dir: Path, game_log_id: str) -> list[dict]:
        eval_path = log_dir / "evaluation_source" / f"{game_log_id}.jsonl"
        assert eval_path.exists(), (
            f"evaluation_source JSONL missing: {eval_path}"
        )
        with eval_path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    # ── core: artifact exists and has the required fields ────────────────────

    def test_evaluation_source_file_created(self, isolated_log_dir):
        state, chosen, merged = self._run_one_ply(isolated_log_dir)
        eval_path = isolated_log_dir / "evaluation_source" / f"{state.game_log_id}.jsonl"
        assert eval_path.exists(), "evaluation_source/*.jsonl was not created"

    def test_chosen_move_facts_is_non_null(self, isolated_log_dir):
        state, chosen, _ = self._run_one_ply(isolated_log_dir)
        records = self._read_eval_jsonl(isolated_log_dir, state.game_log_id)
        assert len(records) == 1
        rec = records[0]
        assert rec.get("chosen_move_facts") is not None, (
            "chosen_move_facts is null in evaluation_source JSONL — "
            "state_manager-before-logger reordering bug"
        )
        # compute_move_facts returns a substantive dict; minimax_score is
        # added by the scorer layer, not by compute_move_facts itself, so
        # we only assert representative fact keys here.
        exported = rec["chosen_move_facts"]
        assert isinstance(exported, dict) and len(exported) > 0
        for representative_key in ("captures_count", "net_gain"):
            assert representative_key in exported, (
                f"expected key {representative_key!r} in exported facts: "
                f"{sorted(exported.keys())}"
            )

    def test_reasoning_seeds_present_and_non_empty(self, isolated_log_dir):
        state, _, _ = self._run_one_ply(isolated_log_dir)
        records = self._read_eval_jsonl(isolated_log_dir, state.game_log_id)
        rec = records[0]
        diag = rec.get("explainer_diagnostics") or rec.get("ranker_diagnostics") or {}
        seeds = diag.get("reasoning_seeds")
        assert isinstance(seeds, list) and len(seeds) > 0, (
            f"reasoning_seeds missing or empty in JSONL: {seeds!r}"
        )

    def test_turn_id_uses_game_log_id_and_turn_number(self, isolated_log_dir):
        state, _, _ = self._run_one_ply(isolated_log_dir)
        records = self._read_eval_jsonl(isolated_log_dir, state.game_log_id)
        rec = records[0]
        # turn_number is incremented by state_manager before logger_node sees it.
        expected = f"{state.game_log_id}_t{state.turn_number + 1}"
        assert rec.get("turn_id") == expected, (
            f"turn_id wrong: got {rec.get('turn_id')!r} expected {expected!r}"
        )

    def test_last_move_reasoning_propagated_from_move_history(self, isolated_log_dir):
        state, _, _ = self._run_one_ply(isolated_log_dir)
        records = self._read_eval_jsonl(isolated_log_dir, state.game_log_id)
        rec = records[0]
        # state_manager clears state.last_move_reasoning but copies it into
        # move_history[-1].  logger_node must read from move_history.
        assert rec.get("last_move_reasoning"), (
            f"last_move_reasoning missing in JSONL: {rec.get('last_move_reasoning')!r}"
        )

    # ── lifecycle: facts must NOT leak into the next turn ────────────────────

    def test_chosen_move_facts_cleared_in_merged_dict(self, isolated_log_dir):
        """Snapshot-and-restore must be log-only.  The merged dict returned to
        LangGraph for the next turn must carry chosen_move_facts=None so the
        next explainer_agent call starts clean."""
        _, _, merged = self._run_one_ply(isolated_log_dir)
        assert merged.get("chosen_move_facts") is None, (
            "chosen_move_facts leaked into the merged update_agent output — "
            "evaluation-only snapshot was promoted to runtime state"
        )
