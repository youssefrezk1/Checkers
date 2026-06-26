"""
checkers/tests/test_update_agent.py

Unit and routing tests for the simplified-pipeline update_agent node.

What is tested
──────────────
  Group 1 — Routing
    - explainer_agent → updater_agent in simplified mode (with or without chosen_move)
    - updater_agent → scorer_node when game continues
    - updater_agent → end when game is over

  Group 2 — Normal turn (move applied)
    - Board is updated after update_agent runs
    - current_player is switched
    - turn_number is incremented
    - move_history gains one record
    - Per-turn fields cleared (proposed_moves, legal_moves, chosen_move, etc.)
    - last_completed_node is "update_agent"
    - game_over is False on a normal mid-game board

  Group 3 — Strategic context (inter_turn_memory phase)
    - strategic_context is set when game continues
    - strategic_context is computed for the NEXT player (post-switch)
    - strategic_context is NOT recomputed when game_over is True

  Group 4 — Terminal case (chosen_move is None)
    - update_agent does not crash
    - game_over is True (stuck player loses)
    - No move_history entry is added in the terminal branch

  Group 5 — Logging
    - game_log_id is populated after a normal turn
    - game_log_id is populated in the terminal case

Run:
    pytest checkers/tests/test_update_agent.py -v
"""
from __future__ import annotations

from checkers.engine.board import RED, BLACK
from checkers.engine.rules import get_all_legal_moves
from checkers.graph.graph import _updater_agent_routing
from checkers.agents.updater_agent import updater_agent
from checkers.state.state import CheckersState


# ── Board helpers ─────────────────────────────────────────────────────────────

def _start_board() -> list[list[int]]:
    """Standard 12-vs-12 starting position."""
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



def _legal_move(board: list[list[int]], player: int) -> dict:
    """Return the first legal move for *player* on *board*."""
    moves = get_all_legal_moves(board, player)
    assert moves, "helper _legal_move: no legal moves found"
    m = moves[0]
    return {"type": m["type"], "path": m["path"], "captured": m.get("captured", [])}


def _state(**kwargs) -> CheckersState:
    return CheckersState(**kwargs)


def _normal_state(player: int = RED) -> CheckersState:
    """Ready-to-update state: board set, chosen_move pre-selected."""
    board = _start_board()
    move = _legal_move(board, player)
    return CheckersState(
        board=board,
        current_player=player,
        chosen_move=move,
        turn_number=1,
    )


# ── Group 1: Routing ──────────────────────────────────────────────────────────

class TestRouting:
    # The simplified graph uses _update_agent_routing as the only conditional edge.

    def test_update_agent_routes_to_scorer_node_when_continuing(self):
        """Game continues: update_agent always loops to scorer_node."""
        state = _state(last_completed_node="updater_agent", game_over=False)
        assert _updater_agent_routing(state) == "scorer_agent"

    def test_update_agent_routes_to_end_when_game_over(self):
        state = _state(last_completed_node="updater_agent", game_over=True)
        assert _updater_agent_routing(state) == "end"


# ── Group 2: Normal turn ──────────────────────────────────────────────────────

class TestNormalTurn:

    def setup_method(self):
        self.state = _normal_state(RED)
        self.result = updater_agent(self.state)

    def test_board_is_updated(self):
        """The returned board must differ from the original."""
        assert self.result["board"] != self.state.board

    def test_current_player_is_switched(self):
        """After RED moves, current_player must be BLACK."""
        assert self.result["current_player"] == BLACK

    def test_turn_number_incremented(self):
        assert self.result["turn_number"] == self.state.turn_number + 1

    def test_move_history_grows_by_one(self):
        assert len(self.result["move_history"]) == len(self.state.move_history) + 1

    def test_move_record_contains_expected_keys(self):
        record = self.result["move_history"][-1]
        for key in ("turn", "player", "move", "promotion", "zobrist_before", "zobrist_after"):
            assert key in record, f"move_record missing key '{key}'"

    def test_intra_turn_fields_cleared(self):
        """All per-turn scratch fields must be reset to defaults."""
        assert self.result["proposed_moves"] == []
        assert self.result["legal_moves"] == []
        assert self.result["chosen_move"] is None
        assert self.result["last_move_reasoning"] is None
        assert self.result["explainer_retry_count"] == 0
        assert self.result["retry_count"] == 0
        assert self.result["insufficient_proposals"] is False
        assert self.result["feedback"] is None

    def test_symbolic_fields_cleared(self):
        assert self.result["symbolic_scored_moves"] == []
        assert self.result["symbolic_best_move"] is None
        assert self.result["symbolic_best_score"] == 0.0
        assert self.result["symbolic_bypass"] is False
        assert self.result["llm_invoked"] is False

    def test_game_not_over_on_normal_board(self):
        assert self.result["game_over"] is False
        assert self.result["winner"] is None
        assert self.result["draw"] is False

    def test_last_completed_node_is_updater_agent(self):
        assert self.result["last_completed_node"] == "updater_agent"

    def test_position_history_grows(self):
        assert len(self.result["position_history"]) == len(self.state.position_history) + 1

    def test_black_player_normal_turn(self):
        """Same checks hold when BLACK is the mover."""
        state = _normal_state(BLACK)
        result = updater_agent(state)
        assert result["current_player"] == RED
        assert result["game_over"] is False
        assert result["last_completed_node"] == "updater_agent"


# ── Group 3: Strategic context ────────────────────────────────────────────────
class TestTerminalCase:

    def _terminal_state(self, player: int = RED) -> CheckersState:
        """State where the current player has no legal moves."""
        # Board with only an opponent piece (current player has nothing → no moves).
        board = [[0] * 8 for _ in range(8)]
        if player == RED:
            board[2][1] = BLACK   # only a BLACK piece; RED has nothing
        else:
            board[5][0] = RED     # only a RED piece; BLACK has nothing
        return CheckersState(
            board=board,
            current_player=player,
            chosen_move=None,
            turn_number=3,
        )

    def test_does_not_crash(self):
        state = self._terminal_state(RED)
        result = updater_agent(state)  # must not raise
        assert result is not None

    def test_game_over_is_true(self):
        state = self._terminal_state(RED)
        result = updater_agent(state)
        assert result["game_over"] is True

    def test_winner_is_opponent(self):
        """When RED has no pieces, BLACK wins."""
        state = self._terminal_state(RED)
        result = updater_agent(state)
        assert result["winner"] == BLACK

    def test_no_move_history_entry_added(self):
        """Phase A is skipped so move_history must not grow."""
        state = self._terminal_state(RED)
        original_len = len(state.move_history)
        result = updater_agent(state)
        assert len(result.get("move_history", state.move_history)) == original_len
class TestLogging:

    def test_game_log_id_populated_normal_turn(self):
        state = _normal_state(RED)
        result = updater_agent(state)
        assert result.get("game_log_id") is not None
        assert isinstance(result["game_log_id"], str)
        assert result["game_log_id"].startswith("game_")

    def test_game_log_id_preserved_across_turns(self):
        """A game_log_id set on the state must not be regenerated."""
        state = _normal_state(RED)
        state = state.model_copy(update={"game_log_id": "game_test_fixed_id"})
        result = updater_agent(state)
        assert result["game_log_id"] == "game_test_fixed_id"

    def test_game_log_id_populated_terminal_case(self):
        board = [[0] * 8 for _ in range(8)]
        board[2][1] = BLACK
        state = CheckersState(
            board=board,
            current_player=RED,
            chosen_move=None,
            turn_number=1,
        )
        result = updater_agent(state)
        assert result.get("game_log_id") is not None


# ── Group 6: Simplified pipeline routing ──────────────────────────────────────

class TestSimplifiedPipelineRouting:
    """The simplified pipeline: scorer_node is the entry point, update_agent loops."""

    def test_update_agent_loops_to_scorer_node(self):
        """Game continues: update_agent always routes back to scorer_node."""
        state = _state(last_completed_node="updater_agent", game_over=False)
        assert _updater_agent_routing(state) == "scorer_agent"

    def test_auto_play_env_var_has_no_effect(self, monkeypatch):
        """AUTO_PLAY_UNTIL_GAME_OVER is ignored; game_over=False always loops to scorer_node."""
        monkeypatch.setenv("AUTO_PLAY_UNTIL_GAME_OVER", "false")
        state = _state(last_completed_node="updater_agent", game_over=False)
        assert _updater_agent_routing(state) == "scorer_agent"

    def test_simplified_full_turn_sequence_loops_back_to_scorer_node(self):
        """update_agent → scorer_node when game continues (direct edge loop)."""
        # scorer_node → proposer_agent → explainer_agent → updater_agent
        # are all direct edges verified by graph compilation tests.

        # update_agent → scorer_node (via _update_agent_routing)
        assert _updater_agent_routing(_state(
            last_completed_node="updater_agent",
            game_over=False,
        )) == "scorer_agent"

    def test_game_over_routes_to_end(self):
        """game_over=True always routes to end."""
        state = _state(last_completed_node="updater_agent", game_over=True)
        assert _updater_agent_routing(state) == "end"

    def test_simplified_pipeline_loops_continuously(self):
        """update_agent loops to scorer_node when game continues, to END when over."""
        after_turn = _state(last_completed_node="updater_agent", game_over=False)
        assert _updater_agent_routing(after_turn) == "scorer_agent"

        game_over = _state(last_completed_node="updater_agent", game_over=True)
        assert _updater_agent_routing(game_over) == "end"


# ── Group 7: scorer_node first-turn context injection ─────────────────────────

class TestScorerNodeFirstTurnContext:
    """scorer_node injects a minimal neutral strategic_context on the first turn."""

    def _start_board(self) -> list[list[int]]:
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
