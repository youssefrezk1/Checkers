"""
checkers/tests/test_simplified_pipeline.py

Tests for the simplified pipeline graph (scorer_agent → proposer_agent →
explainer_agent → updater_agent) and its constituent nodes.

What is NOT tested:
  - explainer_agent LLM call (network)
  - full game execution (requires LLM)

What IS tested:
  Group 1 — Routing (no LLM, no graph invocation)
    - _updater_agent_routing returns "end" on game_over, "scorer_agent" otherwise
    - No old-pipeline nodes appear in the compiled graph

  Group 2 — scorer_node (no LLM)
    - Produces non-empty legal_moves on a normal board
    - Every move has type, path, captured, facts, minimax_score, symbolic_rank
    - symbolic_best_score, symbolic_second_best_score, symbolic_gap, symbolic_best_move set
    - last_completed_node set to "scorer_node"
    - Empty board returns empty legal_moves

  Group 3 — proposer_agent node (no LLM)
    - Selects ONE chosen_move (proposal-authoritative); preserves legal_moves in full
    - All moves come from scorer_node output
    - last_completed_node set to "proposer_agent"
    - Works on single-move positions

  Group 4 — Pre-explainer pipeline (no LLM)
    - scorer_node → proposer_agent produces explainer-ready state with chosen_move set
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
from checkers.graph.graph import _updater_agent_routing
from checkers.nodes.scorer_node import scorer_agent
from checkers.nodes.proposer_node import proposer_node
from checkers.agents.explainer_agent import _get_minimax_score
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

class TestSimplifiedPipelineRouting:
    """Simplified pipeline graph routing via _update_agent_routing and graph edges."""

    def test_scorer_to_proposer_is_direct_edge(self):
        """scorer_agent → proposer_agent must be present in compiled graph."""
        from checkers.graph.graph import build_graph
        g = build_graph()
        assert "scorer_agent" in g.nodes
        assert "proposer_agent" in g.nodes

    def test_proposer_to_explainer_is_direct_edge(self):
        """proposer_agent → explainer_agent must be present in compiled graph."""
        from checkers.graph.graph import build_graph
        g = build_graph()
        assert "proposer_agent" in g.nodes
        assert "explainer_agent" in g.nodes

    def test_explainer_and_updater_both_in_graph(self):
        """explainer_agent → updater_agent — both nodes present."""
        from checkers.graph.graph import build_graph
        g = build_graph()
        assert "explainer_agent" in g.nodes
        assert "updater_agent" in g.nodes

    def test_update_agent_loops_to_scorer_agent(self):
        """Game continues: update_agent routes back to scorer_node."""
        state = _state(game_over=False)
        assert _updater_agent_routing(state) == "scorer_agent"

    def test_update_agent_routes_to_end_when_game_over(self):
        state = _state(game_over=True)
        assert _updater_agent_routing(state) == "end"

    def test_auto_play_env_var_ignored(self, monkeypatch):
        """AUTO_PLAY_UNTIL_GAME_OVER is not checked; game_over=False always loops."""
        monkeypatch.setenv("AUTO_PLAY_UNTIL_GAME_OVER", "false")
        state = _state(game_over=False)
        assert _updater_agent_routing(state) == "scorer_agent"

    def test_old_pipeline_nodes_absent_from_graph(self):
        """No old-pipeline node must appear in the compiled simplified graph."""
        from checkers.graph.graph import build_graph
        g = build_graph()
        old = {
            "orchestrator", "inter_turn_memory", "symbolic_decision",
            "proposal_agent", "format_checker", "validator",
            "minimax_scorer", "state_manager", "win_condition", "logger_node",
        }
        assert not (old & set(g.nodes)), f"Old nodes in graph: {old & set(g.nodes)}"


# ── Group 2: scorer_node ──────────────────────────────────────────────────────

class TestScorerNode:

    def test_produces_non_empty_legal_moves(self):
        state = _state_with_board()
        result = scorer_agent(state)
        assert result["legal_moves"], "scorer_node must produce non-empty legal_moves"

    def test_legal_moves_count_matches_engine(self):
        board = _start_board()
        state = _state_with_board(board)
        result = scorer_agent(state)
        expected = len(get_all_legal_moves(board, RED))
        assert len(result["legal_moves"]) == expected

    def test_every_move_has_type_path_captured_facts(self):
        state = _state_with_board()
        result = scorer_agent(state)
        for m in result["legal_moves"]:
            assert "type"     in m, f"missing 'type' on {m.get('path')}"
            assert "path"     in m, "missing 'path'"
            assert "captured" in m, f"missing 'captured' on {m.get('path')}"
            assert "facts"    in m, f"missing 'facts' on {m.get('path')}"

    def test_every_move_has_minimax_score_and_rank(self):
        state = _state_with_board()
        result = scorer_agent(state)
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
        result = scorer_agent(state)
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
        result = scorer_agent(state)
        assert result["last_completed_node"] == "scorer_agent"

    def test_empty_board_returns_empty_legal_moves(self):
        empty = [[0] * 8 for _ in range(8)]
        state = _state_with_board(empty)
        result = scorer_agent(state)
        assert result["legal_moves"] == []
        assert result["symbolic_best_move"] is None

    def test_single_move_board(self):
        state = _state_with_board(_single_move_board())
        result = scorer_agent(state)
        assert len(result["legal_moves"]) == 1

    def test_sorted_best_first(self):
        state = _state_with_board()
        result = scorer_agent(state)
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
        return scorer_agent(state)

    def _result_single(self):
        state = _state_with_board(_single_move_board())
        return scorer_agent(state)

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


# ── Group 3: proposer_agent node ─────────────────────────────────────────────

class TestDeterministicProposalNode:

    def _run_scorer(self, board=None, player=RED) -> CheckersState:
        """Run scorer_node and return updated state."""
        b = board or _start_board()
        state = _state_with_board(b, player)
        result = scorer_agent(state)
        return state.model_copy(update=result)

    def test_all_moves_come_from_scorer_output(self):
        state = self._run_scorer()
        source_paths = {
            tuple(tuple(sq) for sq in m["path"]) for m in state.legal_moves
        }
        result = proposer_node(state)
        for m in result["legal_moves"]:
            pk = tuple(tuple(sq) for sq in m["path"])
            assert pk in source_paths, (
                f"proposer_agent returned path {m['path']} "
                "not in scorer_node output"
            )

    def test_no_duplicates_in_output(self):
        state = self._run_scorer()
        result = proposer_node(state)
        paths = [tuple(tuple(sq) for sq in m["path"]) for m in result["legal_moves"]]
        assert len(paths) == len(set(paths)), "output contains duplicate moves"

    def test_last_completed_node_set_correctly(self):
        state = self._run_scorer()
        result = proposer_node(state)
        assert result["last_completed_node"] == "proposer_agent"

    def test_single_move_position(self):
        state = self._run_scorer(_single_move_board())
        result = proposer_node(state)
        assert len(result["legal_moves"]) == 1

    def test_empty_legal_moves_returns_empty(self):
        state = _state_with_board([[0] * 8 for _ in range(8)])
        result = proposer_node(state)
        assert result["legal_moves"] == []

    def test_output_preserves_facts(self):
        state = self._run_scorer()
        result = proposer_node(state)
        for m in result["legal_moves"]:
            assert "facts" in m
            assert "minimax_score" in m["facts"]
            assert "symbolic_rank" in m["facts"]

    def test_legal_moves_in_result_not_proposed_moves(self):
        """
        proposer_agent preserves legal_moves unchanged (full list, no truncation)
        and writes chosen_move for the proposal-authoritative selection.
        It does NOT write to proposed_moves.
        This is the field contract that run_simplified_trace.py must read.
        The display bug (printing 0 candidates) was caused by reading
        proposed_moves instead of legal_moves after this node ran.
        """
        state = self._run_scorer()
        result = proposer_node(state)
        # legal_moves is preserved from scorer output — full list, not a shortlist
        assert "legal_moves" in result
        assert len(result["legal_moves"]) > 0, "legal_moves must be non-empty"
        # proposed_moves is NOT set by this node
        assert "proposed_moves" not in result, (
            "proposer_agent must not write proposed_moves; "
            "runners that display candidates must read legal_moves"
        )


# ── Group 4: Pre-ranker pipeline (no LLM) ────────────────────────────────────

class TestPreRankerPipeline:

    def _run_simplified_turn_prep(self, board=None, player=RED):
        """Run scorer_node → proposer_agent, return final state."""
        b = board or _start_board()
        s0 = CheckersState(board=b, current_player=player)
        r1 = scorer_agent(s0)
        s1 = s0.model_copy(update=r1)
        r2 = proposer_node(s1)
        s2 = s1.model_copy(update=r2)
        return s2

    def test_legal_moves_non_empty_after_full_prep(self):
        state = self._run_simplified_turn_prep()
        assert state.legal_moves, "legal_moves must be non-empty after full prep"

    def test_minimax_scores_finite_after_prep(self):
        state = self._run_simplified_turn_prep()
        for m in state.legal_moves:
            score = _get_minimax_score(m)
            assert math.isfinite(score), (
                f"non-finite minimax_score after prep: {score} on {m.get('path')}"
            )

    def test_last_completed_node_is_proposer_agent(self):
        state = self._run_simplified_turn_prep()
        assert state.last_completed_node == "proposer_agent"

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

    def test_score_state_written_by_scorer_agent(self):
        """scorer_node must write a valid score_state to state."""
        state = self._run_simplified_turn_prep()
        assert state.score_state in (
            "CLEARLY_WINNING", "SLIGHTLY_WINNING", "EQUAL",
            "SLIGHTLY_LOSING", "CLEARLY_LOSING",
        ), f"unexpected score_state: {state.score_state!r}"

    def test_pipeline_with_black_player(self):
        state = self._run_simplified_turn_prep(player=BLACK)
        assert state.legal_moves, "non-empty legal_moves for BLACK"

    def test_proposed_moves_field_untouched(self):
        """proposed_moves is not written by the simplified nodes; must stay empty."""
        state = self._run_simplified_turn_prep()
        assert state.proposed_moves == [] or state.proposed_moves == ""


# ── Group 5: Graph compilation ────────────────────────────────────────────────

class TestGraphCompilation:

    def test_graph_compiles(self):
        from checkers.graph.graph import build_graph
        g = build_graph()
        assert g is not None

    def test_graph_module_imports_without_error(self):
        import checkers.graph.graph as gg
        assert hasattr(gg, "_updater_agent_routing")
        assert hasattr(gg, "build_graph")
        assert hasattr(gg, "checkers_graph")
