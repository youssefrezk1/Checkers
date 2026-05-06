"""
checkers/tests/test_simplified_pipeline.py

Tests for the USE_SIMPLIFIED_PIPELINE experimental routing and the two new nodes.

What is NOT tested:
  - ranker_agent LLM call (network)
  - full game execution (requires LLM)

What IS tested:
  Group 1 — Routing (no LLM, no graph invocation)
    - Old pipeline routing unchanged when USE_SIMPLIFIED_PIPELINE=false/absent
    - Simplified routing correct when USE_SIMPLIFIED_PIPELINE=true
    - No old-pipeline nodes appear in the simplified path

  Group 2 — scorer_node (no LLM)
    - Produces non-empty legal_moves on a normal board
    - Every move has type, path, captured, facts, minimax_score, symbolic_rank
    - symbolic_best_score, symbolic_second_best_score, symbolic_gap, symbolic_best_move set
    - last_completed_node set to "scorer_node"
    - Empty board returns empty legal_moves

  Group 3 — deterministic_proposal_node (no LLM)
    - Reduces legal_moves to at most 5
    - All moves come from scorer_node output
    - last_completed_node set to "deterministic_proposal_node"
    - Works on single-move positions

  Group 4 — Pre-ranker pipeline (no LLM)
    - scorer_node → deterministic_proposal_node produces ranker-ready legal_moves
    - _apply_safety_filter runs without error on the output
    - No format_checker or validator node is ever reached in simplified routing

  Group 5 — Graph compilation
    - build_graph() compiles without error in both modes

Run:
    pytest checkers/tests/test_simplified_pipeline.py -v
"""
from __future__ import annotations

import math
import os

import pytest

from checkers.engine.board import RED, BLACK, RED_KING, BLACK_KING
from checkers.engine.rules import get_all_legal_moves
from checkers.graph.graph import _orchestrator_routing
from checkers.nodes.scorer_node import scorer_node
from checkers.nodes.deterministic_proposal_node import deterministic_proposal_node
from checkers.agents.ranker_agent import _apply_safety_filter, _get_minimax_score
from checkers.state.state import CheckersState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _start_board() -> list[list[int]]:
    b = [[0] * 8 for _ in range(8)]
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r][c] = RED
    for r in range(0, 3):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r][c] = BLACK
    return b


def _single_move_board() -> list[list[int]]:
    b = [[0] * 8 for _ in range(8)]
    b[5][0] = RED
    b[0][7] = BLACK
    return b


def _state(**kwargs) -> CheckersState:
    return CheckersState(**kwargs)


def _state_with_board(board=None, player=RED, **kwargs) -> CheckersState:
    return CheckersState(board=board or _start_board(), current_player=player, **kwargs)


# ── Group 1: Routing ──────────────────────────────────────────────────────────

class TestOldPipelineRoutingUnchanged:
    """USE_SIMPLIFIED_PIPELINE absent or false → old pipeline must be identical."""

    def test_inter_turn_memory_routes_to_symbolic_decision(self, monkeypatch):
        monkeypatch.delenv("USE_SIMPLIFIED_PIPELINE", raising=False)
        state = _state(last_completed_node="inter_turn_memory")
        assert _orchestrator_routing(state) == "symbolic_decision"

    def test_inter_turn_memory_routes_to_symbolic_decision_when_false(self, monkeypatch):
        monkeypatch.setenv("USE_SIMPLIFIED_PIPELINE", "false")
        state = _state(last_completed_node="inter_turn_memory")
        assert _orchestrator_routing(state) == "symbolic_decision"

    def test_symbolic_decision_routes_to_proposal_agent(self, monkeypatch):
        monkeypatch.delenv("USE_SIMPLIFIED_PIPELINE", raising=False)
        state = _state(last_completed_node="symbolic_decision")
        assert _orchestrator_routing(state) == "proposal_agent"

    def test_proposal_agent_routes_to_format_checker(self, monkeypatch):
        monkeypatch.delenv("USE_SIMPLIFIED_PIPELINE", raising=False)
        state = _state(last_completed_node="proposal_agent")
        assert _orchestrator_routing(state) == "format_checker"

    def test_minimax_scorer_routes_to_ranker_agent(self, monkeypatch):
        monkeypatch.delenv("USE_SIMPLIFIED_PIPELINE", raising=False)
        state = _state(last_completed_node="minimax_scorer")
        assert _orchestrator_routing(state) == "ranker_agent"

    def test_ranker_with_chosen_move_routes_to_state_manager(self, monkeypatch):
        monkeypatch.delenv("USE_SIMPLIFIED_PIPELINE", raising=False)
        state = _state(
            last_completed_node="ranker_agent",
            chosen_move={"type": "simple", "path": [[5, 0], [4, 1]], "captured": []},
        )
        assert _orchestrator_routing(state) == "state_manager"

    def test_state_manager_routes_to_win_condition(self, monkeypatch):
        monkeypatch.delenv("USE_SIMPLIFIED_PIPELINE", raising=False)
        state = _state(last_completed_node="state_manager")
        assert _orchestrator_routing(state) == "win_condition"


class TestSimplifiedPipelineRouting:
    """USE_SIMPLIFIED_PIPELINE=true → new simplified path."""

    def test_turn_start_routes_to_scorer_node(self, monkeypatch):
        """Orchestrator entry (last_completed_node=None) goes straight to scorer_node."""
        monkeypatch.setenv("USE_SIMPLIFIED_PIPELINE", "true")
        state = _state(last_completed_node=None)
        assert _orchestrator_routing(state) == "scorer_node"

    def test_scorer_node_routes_to_deterministic_proposal(self, monkeypatch):
        monkeypatch.setenv("USE_SIMPLIFIED_PIPELINE", "true")
        state = _state(last_completed_node="scorer_node")
        assert _orchestrator_routing(state) == "deterministic_proposal_node"

    def test_deterministic_proposal_routes_to_ranker_agent(self, monkeypatch):
        monkeypatch.setenv("USE_SIMPLIFIED_PIPELINE", "true")
        state = _state(last_completed_node="deterministic_proposal_node")
        assert _orchestrator_routing(state) == "ranker_agent"

    def test_ranker_routes_to_update_agent(self, monkeypatch):
        """In simplified mode ranker_agent always routes to update_agent."""
        monkeypatch.setenv("USE_SIMPLIFIED_PIPELINE", "true")
        state = _state(
            last_completed_node="ranker_agent",
            chosen_move={"type": "simple", "path": [[5, 0], [4, 1]], "captured": []},
        )
        assert _orchestrator_routing(state) == "update_agent"

    def test_simplified_path_never_reaches_symbolic_decision(self, monkeypatch):
        monkeypatch.setenv("USE_SIMPLIFIED_PIPELINE", "true")
        # Walk the full simplified turn sequence from orchestrator entry onward.
        # inter_turn_memory is no longer part of this path.
        sequence = [None, "scorer_node", "deterministic_proposal_node"]
        destinations = set()
        for node in sequence:
            dest = _orchestrator_routing(_state(last_completed_node=node))
            destinations.add(dest)
        forbidden = {"symbolic_decision", "proposal_agent", "format_checker",
                     "validator", "minimax_scorer", "inter_turn_memory"}
        assert destinations.isdisjoint(forbidden), (
            f"Simplified path reached old-pipeline node(s): "
            f"{destinations & forbidden}"
        )

    def test_format_checker_validator_unreachable_in_simplified_routing(self, monkeypatch):
        monkeypatch.setenv("USE_SIMPLIFIED_PIPELINE", "true")
        # Walk from orchestrator entry through the simplified turn sequence.
        # Neither format_checker nor validator should ever be a destination.
        for node in (None, "scorer_node", "deterministic_proposal_node"):
            dest = _orchestrator_routing(_state(last_completed_node=node))
            assert dest not in ("format_checker", "validator"), (
                f"Node last_completed_node={node!r} routed to '{dest}' in simplified mode"
            )

    def test_flag_true_routes_turn_start_to_scorer_node(self, monkeypatch):
        """USE_SIMPLIFIED_PIPELINE=true: orchestrator entry goes to scorer_node."""
        monkeypatch.setenv("USE_SIMPLIFIED_PIPELINE", "true")
        state = _state(last_completed_node=None)
        assert _orchestrator_routing(state) == "scorer_node"

    def test_flag_false_routes_turn_start_to_inter_turn_memory(self, monkeypatch):
        """USE_SIMPLIFIED_PIPELINE=false: orchestrator entry still goes to inter_turn_memory."""
        monkeypatch.setenv("USE_SIMPLIFIED_PIPELINE", "false")
        state = _state(last_completed_node=None)
        assert _orchestrator_routing(state) == "inter_turn_memory"


# ── Group 2: scorer_node ──────────────────────────────────────────────────────

class TestScorerNode:

    def test_produces_non_empty_legal_moves(self):
        state = _state_with_board()
        result = scorer_node(state)
        assert result["legal_moves"], "scorer_node must produce non-empty legal_moves"

    def test_legal_moves_count_matches_engine(self):
        board = _start_board()
        state = _state_with_board(board)
        result = scorer_node(state)
        expected = len(get_all_legal_moves(board, RED))
        assert len(result["legal_moves"]) == expected

    def test_every_move_has_type_path_captured_facts(self):
        state = _state_with_board()
        result = scorer_node(state)
        for m in result["legal_moves"]:
            assert "type"     in m, f"missing 'type' on {m.get('path')}"
            assert "path"     in m, "missing 'path'"
            assert "captured" in m, f"missing 'captured' on {m.get('path')}"
            assert "facts"    in m, f"missing 'facts' on {m.get('path')}"

    def test_every_move_has_minimax_score_and_rank(self):
        state = _state_with_board()
        result = scorer_node(state)
        for m in result["legal_moves"]:
            facts = m["facts"]
            score = facts.get("minimax_score")
            rank  = facts.get("symbolic_rank")
            assert score is not None and math.isfinite(score), (
                f"minimax_score invalid: {score} on {m.get('path')}"
            )
            assert isinstance(rank, int) and rank >= 1, (
                f"symbolic_rank invalid: {rank} on {m.get('path')}"
            )

    def test_symbolic_summary_fields_set(self):
        state = _state_with_board()
        result = scorer_node(state)
        assert math.isfinite(result["symbolic_best_score"])
        # second_best_score may be None only if there is exactly one legal move
        if len(result["legal_moves"]) > 1:
            assert result["symbolic_second_best_score"] is not None
            assert math.isfinite(result["symbolic_second_best_score"])
        # gap is always finite: scorer_node converts inf → round(best - LOSS_SCORE, 2)
        assert math.isfinite(result["symbolic_gap"]), (
            f"symbolic_gap must be finite, got {result['symbolic_gap']}"
        )

    def test_last_completed_node_set_correctly(self):
        state = _state_with_board()
        result = scorer_node(state)
        assert result["last_completed_node"] == "scorer_node"

    def test_empty_board_returns_empty_legal_moves(self):
        empty = [[0] * 8 for _ in range(8)]
        state = _state_with_board(empty)
        result = scorer_node(state)
        assert result["legal_moves"] == []
        assert result["symbolic_best_move"] is None

    def test_single_move_board(self):
        state = _state_with_board(_single_move_board())
        result = scorer_node(state)
        assert len(result["legal_moves"]) == 1

    def test_sorted_best_first(self):
        state = _state_with_board()
        result = scorer_node(state)
        scores = [m["facts"]["minimax_score"] for m in result["legal_moves"]]
        assert scores == sorted(scores, reverse=True), (
            "legal_moves must be sorted best-first by minimax_score"
        )


# ── Group 2b: symbolic_decision-compatibility ─────────────────────────────────

class TestScorerNodeSymbolicCompat:
    """scorer_node must write symbolic_scored_moves in the same format as
    symbolic_decision so logging, diagnostics, and evaluation scripts work
    without modification in both pipeline modes."""

    def _result(self, player=RED):
        state = _state_with_board(_start_board(), player)
        return scorer_node(state)

    def _result_single(self):
        state = _state_with_board(_single_move_board())
        return scorer_node(state)

    # ── symbolic_scored_moves structure ───────────────────────────────────────

    def test_symbolic_scored_moves_present(self):
        result = self._result()
        assert "symbolic_scored_moves" in result
        assert isinstance(result["symbolic_scored_moves"], list)
        assert len(result["symbolic_scored_moves"]) > 0

    def test_symbolic_scored_moves_count_equals_legal_moves(self):
        result = self._result()
        assert len(result["symbolic_scored_moves"]) == len(result["legal_moves"])

    def test_each_entry_has_move_score_rank(self):
        result = self._result()
        for entry in result["symbolic_scored_moves"]:
            assert "move"          in entry, f"missing 'move' in entry {entry}"
            assert "minimax_score" in entry, f"missing 'minimax_score' in entry {entry}"
            assert "rank"          in entry, f"missing 'rank' in entry {entry}"

    def test_move_is_slim_no_facts(self):
        """'move' inside each entry must not contain 'facts' — matching symbolic_decision."""
        result = self._result()
        for entry in result["symbolic_scored_moves"]:
            m = entry["move"]
            assert "type"     in m
            assert "path"     in m
            assert "captured" in m
            assert "facts" not in m, (
                f"entry move must not carry facts, got keys: {list(m.keys())}"
            )

    def test_ranks_are_one_based_and_sequential(self):
        result = self._result()
        ranks = [e["rank"] for e in result["symbolic_scored_moves"]]
        assert ranks[0] == 1, "rank-1 entry must be first"
        assert ranks == list(range(1, len(ranks) + 1)), (
            f"ranks must be 1..N sequential, got {ranks}"
        )

    def test_scores_match_legal_moves_facts(self):
        """minimax_score in each symbolic_scored_moves entry must equal the
        score in the corresponding legal_moves[i].facts entry."""
        result = self._result()
        for ssm, lm in zip(result["symbolic_scored_moves"], result["legal_moves"]):
            assert ssm["minimax_score"] == lm["facts"]["minimax_score"], (
                f"score mismatch: ssm={ssm['minimax_score']} "
                f"lm={lm['facts']['minimax_score']}"
            )

    def test_paths_match_legal_moves(self):
        """Paths in symbolic_scored_moves must match legal_moves in order."""
        result = self._result()
        for ssm, lm in zip(result["symbolic_scored_moves"], result["legal_moves"]):
            assert ssm["move"]["path"] == lm["path"], (
                f"path mismatch: {ssm['move']['path']} vs {lm['path']}"
            )

    # ── Score field precision ─────────────────────────────────────────────────

    def test_scores_rounded_to_two_dp(self):
        result = self._result()
        for entry in result["symbolic_scored_moves"]:
            score = entry["minimax_score"]
            assert round(score, 2) == score, (
                f"minimax_score not rounded to 2 dp: {score}"
            )
        assert round(result["symbolic_best_score"], 2) == result["symbolic_best_score"]
        sb2 = result["symbolic_second_best_score"]
        if sb2 is not None:
            assert round(sb2, 2) == sb2

    def test_gap_is_finite_on_single_move_board(self):
        """With only one legal move scorer_agent returns inf gap.
        scorer_node must convert it to round(best - LOSS_SCORE, 2)."""
        result = self._result_single()
        assert result["symbolic_second_best_score"] is None
        assert math.isfinite(result["symbolic_gap"]), (
            f"gap must be finite on single-move board, got {result['symbolic_gap']}"
        )

    def test_gap_equals_best_minus_second_on_multi_move_board(self):
        result = self._result()
        if result["symbolic_second_best_score"] is not None:
            expected = round(
                result["symbolic_best_score"] - result["symbolic_second_best_score"], 2
            )
            assert abs(result["symbolic_gap"] - expected) < 1e-6, (
                f"gap={result['symbolic_gap']} expected≈{expected}"
            )

    # ── Best-move field ───────────────────────────────────────────────────────

    def test_symbolic_best_move_matches_rank1_entry(self):
        result = self._result()
        rank1 = result["symbolic_scored_moves"][0]["move"]
        best  = result["symbolic_best_move"]
        assert best["path"]     == rank1["path"]
        assert best["type"]     == rank1["type"]
        assert best["captured"] == rank1["captured"]


# ── Group 3: deterministic_proposal_node ─────────────────────────────────────

class TestDeterministicProposalNode:

    def _run_scorer(self, board=None, player=RED) -> CheckersState:
        """Run scorer_node and return updated state."""
        b = board or _start_board()
        state = _state_with_board(b, player)
        result = scorer_node(state)
        return state.model_copy(update=result)

    def test_reduces_to_at_most_five(self):
        state = self._run_scorer()
        assert len(state.legal_moves) >= 5, "need >= 5 for this test"
        result = deterministic_proposal_node(state)
        assert len(result["legal_moves"]) <= 5

    def test_produces_exactly_five_when_enough_moves(self):
        state = self._run_scorer()
        assert len(state.legal_moves) >= 5
        result = deterministic_proposal_node(state)
        assert len(result["legal_moves"]) == 5

    def test_all_moves_come_from_scorer_output(self):
        state = self._run_scorer()
        source_paths = {
            tuple(tuple(sq) for sq in m["path"]) for m in state.legal_moves
        }
        result = deterministic_proposal_node(state)
        for m in result["legal_moves"]:
            pk = tuple(tuple(sq) for sq in m["path"])
            assert pk in source_paths, (
                f"deterministic_proposal_node returned path {m['path']} "
                "not in scorer_node output"
            )

    def test_no_duplicates_in_output(self):
        state = self._run_scorer()
        result = deterministic_proposal_node(state)
        paths = [tuple(tuple(sq) for sq in m["path"]) for m in result["legal_moves"]]
        assert len(paths) == len(set(paths)), "output contains duplicate moves"

    def test_last_completed_node_set_correctly(self):
        state = self._run_scorer()
        result = deterministic_proposal_node(state)
        assert result["last_completed_node"] == "deterministic_proposal_node"

    def test_single_move_position(self):
        state = self._run_scorer(_single_move_board())
        result = deterministic_proposal_node(state)
        assert len(result["legal_moves"]) == 1

    def test_empty_legal_moves_returns_empty(self):
        state = _state_with_board([[0] * 8 for _ in range(8)])
        result = deterministic_proposal_node(state)
        assert result["legal_moves"] == []

    def test_output_preserves_facts(self):
        state = self._run_scorer()
        result = deterministic_proposal_node(state)
        for m in result["legal_moves"]:
            assert "facts" in m
            assert "minimax_score" in m["facts"]
            assert "symbolic_rank" in m["facts"]

    def test_shortlist_is_in_legal_moves_not_proposed_moves(self):
        """
        deterministic_proposal_node overwrites legal_moves with the shortlist.
        It does NOT write to proposed_moves.
        This is the field contract that run_simplified_trace.py must read.
        The display bug (printing 0 candidates) was caused by reading
        proposed_moves instead of legal_moves after this node ran.
        """
        state = self._run_scorer()
        result = deterministic_proposal_node(state)
        # shortlist is in legal_moves
        assert "legal_moves" in result
        assert len(result["legal_moves"]) > 0, "shortlist must be non-empty"
        # proposed_moves is NOT set by this node
        assert "proposed_moves" not in result, (
            "deterministic_proposal_node must not write proposed_moves; "
            "runners that display candidates must read legal_moves"
        )


# ── Group 4: Pre-ranker pipeline (no LLM) ────────────────────────────────────

class TestPreRankerPipeline:

    def _run_simplified_turn_prep(self, board=None, player=RED, ctx=None):
        """Run scorer_node → deterministic_proposal_node, return final state."""
        b = board or _start_board()
        s0 = CheckersState(board=b, current_player=player, strategic_context=ctx)
        r1 = scorer_node(s0)
        s1 = s0.model_copy(update=r1)
        r2 = deterministic_proposal_node(s1)
        s2 = s1.model_copy(update=r2)
        return s2

    def test_legal_moves_non_empty_after_full_prep(self):
        state = self._run_simplified_turn_prep()
        assert state.legal_moves, "legal_moves must be non-empty after full prep"

    def test_legal_moves_at_most_five(self):
        state = self._run_simplified_turn_prep()
        assert len(state.legal_moves) <= 5

    def test_safety_filter_runs_on_prep_output(self):
        state = self._run_simplified_turn_prep()
        filtered, index_map = _apply_safety_filter(state.legal_moves)
        assert filtered, "safety filter must keep at least one move"
        assert isinstance(index_map, list)

    def test_minimax_scores_finite_after_prep(self):
        state = self._run_simplified_turn_prep()
        for m in state.legal_moves:
            score = _get_minimax_score(m)
            assert math.isfinite(score), (
                f"non-finite minimax_score after prep: {score} on {m.get('path')}"
            )

    def test_last_completed_node_is_deterministic_proposal(self):
        state = self._run_simplified_turn_prep()
        assert state.last_completed_node == "deterministic_proposal_node"

    def test_format_checker_never_called_in_simplified_prep(self):
        """
        Verify no format_checker or validator node is invoked.
        Since we call the node functions directly (not through LangGraph),
        the routing test above already verifies this. Here we confirm the
        node outputs contain no retry/format-error fields written by those nodes.
        """
        state = self._run_simplified_turn_prep()
        # format_checker increments format_error_count; it must stay at 0.
        assert state.format_error_count == 0
        # validator writes feedback on failure; it must stay None.
        assert state.feedback is None

    def test_pipeline_with_strategic_context(self):
        ctx = {
            "score_state": "SLIGHTLY_WINNING",
            "game_phase": "MIDGAME",
            "strategic_priorities": ["CONVERT_ADVANTAGE"],
        }
        state = self._run_simplified_turn_prep(ctx=ctx)
        assert state.legal_moves, "non-empty legal_moves with strategic context"
        assert len(state.legal_moves) <= 5

    def test_pipeline_with_black_player(self):
        state = self._run_simplified_turn_prep(player=BLACK)
        assert state.legal_moves, "non-empty legal_moves for BLACK"

    def test_proposed_moves_field_untouched(self):
        """proposed_moves is not written by the simplified nodes; must stay empty."""
        state = self._run_simplified_turn_prep()
        assert state.proposed_moves == [] or state.proposed_moves == ""


# ── Group 5: Graph compilation ────────────────────────────────────────────────

class TestGraphCompilation:

    def test_graph_compiles_with_old_pipeline(self, monkeypatch):
        monkeypatch.delenv("USE_SIMPLIFIED_PIPELINE", raising=False)
        from checkers.graph.graph import build_graph
        g = build_graph()
        assert g is not None

    def test_graph_compiles_with_simplified_pipeline_flag(self, monkeypatch):
        monkeypatch.setenv("USE_SIMPLIFIED_PIPELINE", "true")
        from checkers.graph.graph import build_graph
        g = build_graph()
        assert g is not None

    def test_graph_module_imports_without_error(self):
        import checkers.graph.graph as gg
        assert hasattr(gg, "_orchestrator_routing")
        assert hasattr(gg, "build_graph")
        assert hasattr(gg, "checkers_graph")
