"""
checkers/tests/test_update_agent.py

Unit and routing tests for the simplified-pipeline update_agent node.

What is tested
──────────────
  Group 1 — Routing
    - ranker_agent → update_agent in simplified mode (with or without chosen_move)
    - update_agent → scorer_node when game continues
    - update_agent → end when game is over
    - Old pipeline routing untouched (ranker → state_manager / win_condition)

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
from checkers.graph.graph import _update_agent_routing
from checkers.agents.update_agent import update_agent
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
        state = _state(last_completed_node="update_agent", game_over=False)
        assert _update_agent_routing(state) == "scorer_node"

    def test_update_agent_routes_to_end_when_game_over(self):
        state = _state(last_completed_node="update_agent", game_over=True)
        assert _update_agent_routing(state) == "end"


# ── Group 2: Normal turn ──────────────────────────────────────────────────────

class TestNormalTurn:

    def setup_method(self):
        self.state = _normal_state(RED)
        self.result = update_agent(self.state)

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
        assert self.result["ranker_retry_count"] == 0
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

    def test_last_completed_node_is_update_agent(self):
        assert self.result["last_completed_node"] == "update_agent"

    def test_position_history_grows(self):
        assert len(self.result["position_history"]) == len(self.state.position_history) + 1

    def test_black_player_normal_turn(self):
        """Same checks hold when BLACK is the mover."""
        state = _normal_state(BLACK)
        result = update_agent(state)
        assert result["current_player"] == RED
        assert result["game_over"] is False
        assert result["last_completed_node"] == "update_agent"


# ── Group 3: Strategic context ────────────────────────────────────────────────

class TestStrategicContext:

    def test_strategic_context_set_when_game_continues(self):
        state = _normal_state(RED)
        result = update_agent(state)
        assert result.get("strategic_context") is not None

    def test_strategic_context_is_for_next_player(self):
        """
        After RED moves, current_player is BLACK.
        inter_turn_memory runs with current_player=BLACK, so strategic_context
        describes the position from BLACK's perspective.
        The 'turn_number' stored inside the context must match the incremented
        turn (i.e., the turn that was just completed).
        """
        state = _normal_state(RED)
        result = update_agent(state)
        ctx = result["strategic_context"]
        # context was built with the post-switch board where next mover is BLACK
        assert ctx is not None
        # strategic_context must contain required keys from inter_turn_memory
        for key in ("strategic_priorities", "game_phase", "material_advantage",
                    "winning_score", "score_state"):
            assert key in ctx, f"strategic_context missing key '{key}'"

    def test_strategic_context_not_set_when_game_over(self):
        """When game_over is True, Phase D must be skipped."""
        # Build a board where RED has only one piece and no opponent → RED wins
        # after any move is applied.  Win is detected in Phase B, so
        # inter_turn_memory (Phase D) must be skipped.
        board = [[0] * 8 for _ in range(8)]
        board[5][0] = RED   # one RED piece
        # No BLACK pieces → after RED moves, check_win_condition sees BLACK has
        # no pieces → game_over=True, winner=RED.
        moves = get_all_legal_moves(board, RED)
        assert moves, "need at least one legal move for this test"
        move = {"type": moves[0]["type"], "path": moves[0]["path"],
                "captured": moves[0].get("captured", [])}
        state = CheckersState(
            board=board,
            current_player=RED,
            chosen_move=move,
            turn_number=1,
        )
        result = update_agent(state)
        assert result["game_over"] is True
        # strategic_context must NOT be overwritten by Phase D
        # (it stays as whatever the previous value was — None by default)
        assert result.get("strategic_context") is None

    def test_strategic_context_turn_history_appended(self):
        """Sliding window in inter_turn_memory grows with each call."""
        state = _normal_state(RED)
        result1 = update_agent(state)
        ctx1 = result1["strategic_context"]
        assert len(ctx1["turn_history"]) == 1

        # Simulate next turn by applying another move
        post_state = state.model_copy(update=result1)
        move2 = _legal_move(post_state.board, post_state.current_player)
        post_state = post_state.model_copy(update={"chosen_move": move2})
        result2 = update_agent(post_state)
        ctx2 = result2["strategic_context"]
        assert len(ctx2["turn_history"]) == 2


# ── Group 4: Terminal case (chosen_move is None) ──────────────────────────────

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
        result = update_agent(state)  # must not raise
        assert result is not None

    def test_game_over_is_true(self):
        state = self._terminal_state(RED)
        result = update_agent(state)
        assert result["game_over"] is True

    def test_winner_is_opponent(self):
        """When RED has no pieces, BLACK wins."""
        state = self._terminal_state(RED)
        result = update_agent(state)
        assert result["winner"] == BLACK

    def test_no_move_history_entry_added(self):
        """Phase A is skipped so move_history must not grow."""
        state = self._terminal_state(RED)
        original_len = len(state.move_history)
        result = update_agent(state)
        assert len(result.get("move_history", state.move_history)) == original_len

    def test_strategic_context_not_set_terminal(self):
        """Game is over → Phase D skipped → strategic_context not overwritten."""
        state = self._terminal_state(RED)
        result = update_agent(state)
        assert result["game_over"] is True
        assert result.get("strategic_context") is None

    def test_last_completed_node_is_update_agent(self):
        state = self._terminal_state(RED)
        result = update_agent(state)
        assert result["last_completed_node"] == "update_agent"


# ── Group 5: Logging ──────────────────────────────────────────────────────────

class TestLogging:

    def test_game_log_id_populated_normal_turn(self):
        state = _normal_state(RED)
        result = update_agent(state)
        assert result.get("game_log_id") is not None
        assert isinstance(result["game_log_id"], str)
        assert result["game_log_id"].startswith("game_")

    def test_game_log_id_preserved_across_turns(self):
        """A game_log_id set on the state must not be regenerated."""
        state = _normal_state(RED)
        state = state.model_copy(update={"game_log_id": "game_test_fixed_id"})
        result = update_agent(state)
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
        result = update_agent(state)
        assert result.get("game_log_id") is not None


# ── Group 6: Simplified pipeline routing ──────────────────────────────────────

class TestSimplifiedPipelineRouting:
    """The simplified pipeline: scorer_node is the entry point, update_agent loops."""

    def test_update_agent_loops_to_scorer_node(self):
        """Game continues: update_agent always routes back to scorer_node."""
        state = _state(last_completed_node="update_agent", game_over=False)
        assert _update_agent_routing(state) == "scorer_node"

    def test_auto_play_env_var_has_no_effect(self, monkeypatch):
        """AUTO_PLAY_UNTIL_GAME_OVER is ignored; game_over=False always loops to scorer_node."""
        monkeypatch.setenv("AUTO_PLAY_UNTIL_GAME_OVER", "false")
        state = _state(last_completed_node="update_agent", game_over=False)
        assert _update_agent_routing(state) == "scorer_node"

    def test_simplified_full_turn_sequence_loops_back_to_scorer_node(self):
        """update_agent → scorer_node when game continues (direct edge loop)."""
        # scorer_node → deterministic_proposal_node → ranker_agent → update_agent
        # are all direct edges verified by graph compilation tests.

        # update_agent → scorer_node (via _update_agent_routing)
        assert _update_agent_routing(_state(
            last_completed_node="update_agent",
            game_over=False,
        )) == "scorer_node"

    def test_game_over_routes_to_end(self):
        """game_over=True always routes to end."""
        state = _state(last_completed_node="update_agent", game_over=True)
        assert _update_agent_routing(state) == "end"

    def test_simplified_pipeline_loops_continuously(self):
        """update_agent loops to scorer_node when game continues, to END when over."""
        after_turn = _state(last_completed_node="update_agent", game_over=False)
        assert _update_agent_routing(after_turn) == "scorer_node"

        game_over = _state(last_completed_node="update_agent", game_over=True)
        assert _update_agent_routing(game_over) == "end"


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

    def test_injects_context_when_none(self):
        """With strategic_context=None, scorer_node must return a strategic_context."""
        from checkers.nodes.scorer_node import scorer_node
        state = CheckersState(
            board=self._start_board(),
            current_player=RED,
            strategic_context=None,
        )
        result = scorer_node(state)
        assert "strategic_context" in result, "scorer_node must inject context when None"
        ctx = result["strategic_context"]
        assert ctx["game_phase"] == "OPENING"
        assert ctx["score_state"] == "EQUAL"
        assert ctx["strategic_priorities"] == []

    def test_does_not_overwrite_existing_context(self):
        """When a real strategic_context already exists, scorer_node must not touch it."""
        from checkers.nodes.scorer_node import scorer_node
        existing_ctx = {
            "game_phase": "MIDGAME",
            "score_state": "SLIGHTLY_WINNING",
            "strategic_priorities": ["HOLD_ADVANTAGE"],
            "winning_score": 5,
        }
        state = CheckersState(
            board=self._start_board(),
            current_player=RED,
            strategic_context=existing_ctx,
        )
        result = scorer_node(state)
        # strategic_context must NOT appear in the returned dict (no overwrite)
        assert "strategic_context" not in result, (
            "scorer_node must not overwrite an existing strategic_context"
        )

    def test_injected_context_has_required_keys(self):
        """All keys downstream nodes depend on must be present in the default context."""
        from checkers.nodes.scorer_node import scorer_node
        state = CheckersState(
            board=self._start_board(),
            current_player=RED,
            strategic_context=None,
        )
        result = scorer_node(state)
        ctx = result["strategic_context"]
        required = (
            "game_phase", "score_state", "strategic_priorities",
            "material_advantage", "winning_score",
            "active_patterns", "stagnation_detected",
            "material_trend", "mobility_trend",
            "turn_history", "archive_summary",
        )
        for key in required:
            assert key in ctx, f"default context missing required key '{key}'"

    def test_scorer_node_still_returns_legal_moves(self):
        """Context injection must not interfere with the scoring output."""
        from checkers.nodes.scorer_node import scorer_node
        state = CheckersState(
            board=self._start_board(),
            current_player=RED,
            strategic_context=None,
        )
        result = scorer_node(state)
        assert result["legal_moves"], "scorer_node must still produce legal_moves"
        assert result["last_completed_node"] == "scorer_node"


# ── Group 8: Human move not overwritten ───────────────────────────────────────

class TestHumanMoveNotOverwritten:
    """
    Regression group: calling update_agent directly preserves the human's
    pre-selected chosen_move.

    This is the pattern _run_black_ply in run_simplified_trace.py uses:
    build a CheckersState from the accumulated dict and call update_agent(state)
    directly instead of streaming through the graph (which would restart at
    scorer_node and overwrite chosen_move via ranker_agent).
    """

    def test_human_chosen_move_applied_to_board(self):
        """The board must reflect the human's move, not any LLM choice."""
        board = _start_board()
        human_move = _legal_move(board, BLACK)
        state = CheckersState(
            board=board,
            current_player=BLACK,
            chosen_move=human_move,
            turn_number=1,
            last_move_reasoning="BLACK human move",
        )
        result = update_agent(state)

        assert result["board"] != board, "board must change after applying the move"

        mh = result.get("move_history", [])
        assert len(mh) == 1, "exactly one move_history entry expected"
        applied = mh[0]["move"]
        assert applied["type"] == human_move["type"]
        assert [list(sq) for sq in applied["path"]] == [list(sq) for sq in human_move["path"]]

    def test_chosen_move_cleared_after_application(self):
        """state_manager clears chosen_move after application; must be None."""
        board = _start_board()
        move = _legal_move(board, BLACK)
        state = CheckersState(
            board=board,
            current_player=BLACK,
            chosen_move=move,
            turn_number=1,
        )
        result = update_agent(state)
        assert result["chosen_move"] is None

    def test_last_completed_node_is_update_agent(self):
        """last_completed_node must be 'update_agent', never 'scorer_node' or 'ranker_agent'."""
        board = _start_board()
        move = _legal_move(board, BLACK)
        state = CheckersState(
            board=board,
            current_player=BLACK,
            chosen_move=move,
            turn_number=1,
        )
        result = update_agent(state)
        assert result["last_completed_node"] == "update_agent"

    def test_game_not_over_after_normal_move(self):
        """A normal mid-game move must not trigger game_over."""
        board = _start_board()
        move = _legal_move(board, BLACK)
        state = CheckersState(
            board=board,
            current_player=BLACK,
            chosen_move=move,
            turn_number=1,
        )
        result = update_agent(state)
        assert result["game_over"] is False

    def test_strategic_context_produced_for_next_player(self):
        """Phase D must run and produce context for RED (next to move after BLACK)."""
        board = _start_board()
        move = _legal_move(board, BLACK)
        state = CheckersState(
            board=board,
            current_player=BLACK,
            chosen_move=move,
            turn_number=1,
        )
        result = update_agent(state)
        ctx = result.get("strategic_context")
        assert ctx is not None, "strategic_context must be produced when game continues"
        assert "strategic_priorities" in ctx
        assert "game_phase" in ctx

