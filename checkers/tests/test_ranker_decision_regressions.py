from __future__ import annotations

import json

import checkers.agents.ranker_agent as ranker_module
from checkers.engine.board import create_initial_board
from checkers.engine.board import BLACK, EMPTY, RED
from checkers.engine.move_facts import compute_move_facts
from checkers.engine.rules import apply_move, get_all_legal_moves
from checkers.nodes.minimax_scorer import minimax_scorer
from checkers.state.state import CheckersState


def _empty_board() -> list[list[int]]:
    return [[EMPTY for _ in range(8)] for _ in range(8)]


def _state_with_opening_candidates(legal_moves: list[dict], turn_number: int = 5) -> CheckersState:
    board = _empty_board()
    board[0][1] = BLACK
    board[0][3] = BLACK
    board[1][2] = BLACK
    board[2][1] = BLACK
    board[5][0] = RED
    board[5][2] = RED
    board[6][1] = RED
    board[6][3] = RED
    return CheckersState(
        board=board,
        current_player=RED,
        turn_number=turn_number,
        legal_moves=legal_moves,
        strategic_context={
            "game_phase": "OPENING",
            "score_state": "EQUAL",
            "strategic_priorities": ["DEVELOP_PIECES", "CONTROL_CENTER", "MAINTAIN_BACK_ROW"],
        },
    )


def _patch_ranker_choice(monkeypatch, chosen_index: int) -> None:
    payload = json.dumps({"chosen_index": chosen_index, "reasoning": "regression fixture"})

    def _fake_call_ranker(system: str, user: str) -> str:
        return payload

    monkeypatch.setattr(ranker_module, "call_ranker", _fake_call_ranker)


def _index_for_path(legal_moves: list[dict], path: list[tuple[int, int]]) -> int:
    for i, move in enumerate(legal_moves):
        if move.get("path") == path:
            return i
    raise AssertionError(f"path {path} not found in legal_moves")


def _assert_decision(
    monkeypatch,
    capsys,
    legal_moves: list[dict],
    llm_idx: int,
    expected_idx: int,
    turn_number: int,
    expected_override_triggered: bool,
) -> None:
    _patch_ranker_choice(monkeypatch, chosen_index=llm_idx)
    state = _state_with_opening_candidates(legal_moves, turn_number=turn_number)
    patch = ranker_module.ranker_agent(state)

    chosen_path = patch["chosen_move"]["path"]
    chosen_idx = _index_for_path(legal_moves, chosen_path)
    assert chosen_idx == expected_idx

    out = capsys.readouterr().out
    expected_flag = (
        "override_triggered=True"
        if expected_override_triggered
        else "override_triggered=False"
    )
    assert expected_flag in out


def _load_trace_entries(log_path: str) -> list[dict]:
    entries: list[dict] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _reconstruct_board_before_turn(log_path: str, turn: int) -> tuple[list[list[int]], dict]:
    entries = _load_trace_entries(log_path)
    target = next(e for e in entries if e.get("turn") == turn)
    board = create_initial_board()
    for entry in entries:
        if entry.get("turn", 0) >= turn:
            break
        move = {
            "type": entry["move_type"],
            "path": [tuple(p) for p in entry["path"]],
            "captured": [tuple(c) for c in entry.get("captured", [])],
        }
        board = apply_move(board, move)
    return board, target


def _state_from_real_pipeline(
    board: list[list[int]],
    current_player: int,
    turn_number: int,
    strategic_priorities: list[str],
) -> CheckersState:
    legal = get_all_legal_moves(board, current_player)
    enriched = []
    for move in legal:
        enriched.append(
            {
                "type": move["type"],
                "path": move["path"],
                "captured": move["captured"],
                "facts": compute_move_facts(board, move, current_player),
            }
        )

    context = {
        "game_phase": "OPENING" if turn_number <= 20 else "MIDGAME",
        "score_state": "EQUAL",
        "strategic_priorities": strategic_priorities or [],
    }
    state = CheckersState(
        board=board,
        current_player=current_player,
        turn_number=turn_number,
        legal_moves=enriched,
        strategic_context=context,
    )
    scored = minimax_scorer(state)
    return CheckersState(
        board=board,
        current_player=current_player,
        turn_number=turn_number,
        legal_moves=scored["legal_moves"],
        strategic_context=context,
    )


def _pick_wrong_llm_index(state: CheckersState, expected_path: list[tuple[int, int]]) -> int:
    expected_idx = _index_for_path(state.legal_moves, expected_path)
    candidates = [i for i in range(len(state.legal_moves)) if i != expected_idx]
    assert candidates, "Need at least 2 legal moves for wrong-choice simulation."
    # Pick the lowest minimax alternative to force a wrong LLM choice.
    return min(candidates, key=lambda i: state.legal_moves[i].get("facts", {}).get("minimax_score", 0.0))


def test_opening_regression_case_1_prefers_best_low_danger_minimax(monkeypatch):
    legal_moves = [
        {
            "type": "simple",
            "path": [(5, 2), (4, 3)],
            "captured": [],
            "facts": {
                "captures_count": 0,
                "creates_immediate_threat": False,
                "shot_sequence_available": False,
                "blocks_opponent_landing": False,
                "opponent_can_recapture": False,
                "moved_piece_is_threatened": False,
                "our_pieces_threatened_after": 1,
                "net_gain": 0,
                "minimax_score": 5.5,
                "quiet_move_role": "DEVELOPMENT",
            },
        },
        {
            "type": "simple",
            "path": [(6, 1), (5, 0)],
            "captured": [],
            "facts": {
                "captures_count": 0,
                "creates_immediate_threat": False,
                "shot_sequence_available": False,
                "blocks_opponent_landing": False,
                "opponent_can_recapture": False,
                "moved_piece_is_threatened": False,
                "our_pieces_threatened_after": 0,
                "net_gain": 0,
                "minimax_score": -9.0,
                "quiet_move_role": "STRUCTURAL_RESTRICTION",
            },
        },
    ]
    _patch_ranker_choice(monkeypatch, chosen_index=1)
    state = _state_with_opening_candidates(legal_moves, turn_number=5)

    patch = ranker_module.ranker_agent(state)
    assert patch["chosen_move"]["path"] == [(5, 2), (4, 3)]


def test_opening_regression_case_2_prefers_best_low_danger_minimax(monkeypatch):
    legal_moves = [
        {
            "type": "simple",
            "path": [(5, 0), (4, 1)],
            "captured": [],
            "facts": {
                "captures_count": 0,
                "creates_immediate_threat": False,
                "shot_sequence_available": False,
                "blocks_opponent_landing": False,
                "opponent_can_recapture": False,
                "moved_piece_is_threatened": False,
                "our_pieces_threatened_after": 1,
                "net_gain": 0,
                "minimax_score": 6.0,
                "quiet_move_role": "DEVELOPMENT",
            },
        },
        {
            "type": "simple",
            "path": [(6, 3), (5, 4)],
            "captured": [],
            "facts": {
                "captures_count": 0,
                "creates_immediate_threat": False,
                "shot_sequence_available": False,
                "blocks_opponent_landing": False,
                "opponent_can_recapture": False,
                "moved_piece_is_threatened": False,
                "our_pieces_threatened_after": 0,
                "net_gain": 0,
                "minimax_score": -8.5,
                "quiet_move_role": "KING_ACTIVATION",
            },
        },
    ]
    _patch_ranker_choice(monkeypatch, chosen_index=1)
    state = _state_with_opening_candidates(legal_moves, turn_number=7)

    patch = ranker_module.ranker_agent(state)
    assert patch["chosen_move"]["path"] == [(5, 0), (4, 1)]


def test_opening_control_case_keeps_already_best_choice(monkeypatch):
    legal_moves = [
        {
            "type": "simple",
            "path": [(5, 2), (4, 3)],
            "captured": [],
            "facts": {
                "captures_count": 0,
                "creates_immediate_threat": False,
                "shot_sequence_available": False,
                "blocks_opponent_landing": False,
                "opponent_can_recapture": False,
                "moved_piece_is_threatened": False,
                "our_pieces_threatened_after": 1,
                "net_gain": 0,
                "minimax_score": 4.5,
                "quiet_move_role": "DEVELOPMENT",
            },
        },
        {
            "type": "simple",
            "path": [(6, 1), (5, 0)],
            "captured": [],
            "facts": {
                "captures_count": 0,
                "creates_immediate_threat": False,
                "shot_sequence_available": False,
                "blocks_opponent_landing": False,
                "opponent_can_recapture": False,
                "moved_piece_is_threatened": False,
                "our_pieces_threatened_after": 0,
                "net_gain": 0,
                "minimax_score": 1.0,
                "quiet_move_role": "STRUCTURAL_RESTRICTION",
            },
        },
    ]
    _patch_ranker_choice(monkeypatch, chosen_index=0)
    state = _state_with_opening_candidates(legal_moves, turn_number=9)

    patch = ranker_module.ranker_agent(state)
    assert patch["chosen_move"]["path"] == [(5, 2), (4, 3)]


def test_override_not_blocked_by_fake_immediate_danger(monkeypatch, capsys):
    legal_moves = [
        {
            "type": "simple",
            "path": [(5, 0), (4, 1)],
            "captured": [],
            "facts": {
                "captures_count": 0,
                "creates_immediate_threat": False,
                "shot_sequence_available": False,
                "blocks_opponent_landing": False,
                "opponent_can_recapture": False,
                "moved_piece_is_threatened": False,
                "our_pieces_threatened_after": 0,
                "net_gain": 0,
                "minimax_score": 10.2,
                "quiet_move_role": "DEVELOPMENT",
            },
        },
        {
            "type": "simple",
            "path": [(4, 5), (3, 6)],
            "captured": [],
            "facts": {
                "captures_count": 0,
                "creates_immediate_threat": True,
                "shot_sequence_available": True,
                "blocks_opponent_landing": False,
                "opponent_can_recapture": False,
                "moved_piece_is_threatened": False,
                "our_pieces_threatened_after": 0,
                "forced_opponent_jump_reply": False,
                "max_opponent_jump_captures": 0,
                "opponent_jump_count": 0,
                "net_gain": 0,
                "minimax_score": 1.9,
                "quiet_move_role": "TACTICAL_PRESSURE",
            },
        },
    ]
    _assert_decision(
        monkeypatch=monkeypatch,
        capsys=capsys,
        legal_moves=legal_moves,
        llm_idx=1,
        expected_idx=0,
        turn_number=11,
        expected_override_triggered=True,
    )


def test_tactical_pressure_does_not_override_minimax_when_safe(monkeypatch, capsys):
    legal_moves = [
        {
            "type": "simple",
            "path": [(5, 2), (4, 3)],
            "captured": [],
            "facts": {
                "captures_count": 0,
                "creates_immediate_threat": False,
                "shot_sequence_available": False,
                "blocks_opponent_landing": False,
                "opponent_can_recapture": False,
                "moved_piece_is_threatened": False,
                "our_pieces_threatened_after": 0,
                "net_gain": 0,
                "minimax_score": 20.0,
                "quiet_move_role": "DEVELOPMENT",
            },
        },
        {
            "type": "simple",
            "path": [(4, 5), (3, 6)],
            "captured": [],
            "facts": {
                "captures_count": 0,
                "creates_immediate_threat": True,
                "shot_sequence_available": False,
                "blocks_opponent_landing": True,
                "opponent_can_recapture": False,
                "moved_piece_is_threatened": False,
                "our_pieces_threatened_after": 0,
                "forced_opponent_jump_reply": False,
                "max_opponent_jump_captures": 0,
                "opponent_jump_count": 0,
                "net_gain": 0,
                "minimax_score": 11.5,
                "quiet_move_role": "TACTICAL_PRESSURE",
            },
        },
    ]
    _assert_decision(
        monkeypatch=monkeypatch,
        capsys=capsys,
        legal_moves=legal_moves,
        llm_idx=1,
        expected_idx=0,
        turn_number=13,
        expected_override_triggered=True,
    )


def test_threat_zero_does_not_beat_low_danger_with_better_minimax(monkeypatch, capsys):
    legal_moves = [
        {
            "type": "simple",
            "path": [(5, 2), (4, 3)],
            "captured": [],
            "facts": {
                "captures_count": 0,
                "creates_immediate_threat": False,
                "shot_sequence_available": False,
                "blocks_opponent_landing": False,
                "opponent_can_recapture": False,
                "moved_piece_is_threatened": False,
                "our_pieces_threatened_after": 1,
                "net_gain": 0,
                "minimax_score": 8.0,
                "quiet_move_role": "DEVELOPMENT",
            },
        },
        {
            "type": "simple",
            "path": [(6, 1), (5, 0)],
            "captured": [],
            "facts": {
                "captures_count": 0,
                "creates_immediate_threat": False,
                "shot_sequence_available": False,
                "blocks_opponent_landing": False,
                "opponent_can_recapture": False,
                "moved_piece_is_threatened": False,
                "our_pieces_threatened_after": 0,
                "net_gain": 0,
                "minimax_score": -2.0,
                "quiet_move_role": "STRUCTURAL_RESTRICTION",
            },
        },
    ]
    _assert_decision(
        monkeypatch=monkeypatch,
        capsys=capsys,
        legal_moves=legal_moves,
        llm_idx=1,
        expected_idx=0,
        turn_number=15,
        expected_override_triggered=True,
    )


def test_passive_structural_loses_to_low_danger_active(monkeypatch, capsys):
    legal_moves = [
        {
            "type": "simple",
            "path": [(5, 2), (4, 3)],
            "captured": [],
            "facts": {
                "captures_count": 0,
                "creates_immediate_threat": True,
                "shot_sequence_available": False,
                "blocks_opponent_landing": True,
                "opponent_can_recapture": False,
                "moved_piece_is_threatened": False,
                "our_pieces_threatened_after": 1,
                "forced_opponent_jump_reply": False,
                "max_opponent_jump_captures": 1,
                "opponent_jump_count": 1,
                "net_gain": 0,
                "minimax_score": 9.0,
                "quiet_move_role": "TACTICAL_PRESSURE",
            },
        },
        {
            "type": "simple",
            "path": [(6, 3), (5, 4)],
            "captured": [],
            "facts": {
                "captures_count": 0,
                "creates_immediate_threat": False,
                "shot_sequence_available": False,
                "blocks_opponent_landing": False,
                "opponent_can_recapture": False,
                "moved_piece_is_threatened": False,
                "our_pieces_threatened_after": 0,
                "net_gain": 0,
                "minimax_score": -1.0,
                "quiet_move_role": "STRUCTURAL_RESTRICTION",
            },
        },
    ]
    _assert_decision(
        monkeypatch=monkeypatch,
        capsys=capsys,
        legal_moves=legal_moves,
        llm_idx=1,
        expected_idx=0,
        turn_number=17,
        expected_override_triggered=True,
    )


def test_no_override_when_llm_already_best(monkeypatch, capsys):
    legal_moves = [
        {
            "type": "simple",
            "path": [(5, 0), (4, 1)],
            "captured": [],
            "facts": {
                "captures_count": 0,
                "creates_immediate_threat": False,
                "shot_sequence_available": False,
                "blocks_opponent_landing": False,
                "opponent_can_recapture": False,
                "moved_piece_is_threatened": False,
                "our_pieces_threatened_after": 1,
                "net_gain": 0,
                "minimax_score": 7.5,
                "quiet_move_role": "DEVELOPMENT",
            },
        },
        {
            "type": "simple",
            "path": [(6, 1), (5, 0)],
            "captured": [],
            "facts": {
                "captures_count": 0,
                "creates_immediate_threat": True,
                "shot_sequence_available": False,
                "blocks_opponent_landing": False,
                "opponent_can_recapture": False,
                "moved_piece_is_threatened": False,
                "our_pieces_threatened_after": 1,
                "net_gain": 0,
                "minimax_score": 2.5,
                "quiet_move_role": "TACTICAL_PRESSURE",
            },
        },
    ]
    _assert_decision(
        monkeypatch=monkeypatch,
        capsys=capsys,
        legal_moves=legal_moves,
        llm_idx=0,
        expected_idx=0,
        turn_number=19,
        expected_override_triggered=False,
    )


def test_integration_trace_turn5_opening_override(monkeypatch, capsys):
    log_path = "/Users/youssefrezk/Desktop/bachelor_project/logs/game_20260419_191038_610610.jsonl"
    board, turn_entry = _reconstruct_board_before_turn(log_path, turn=5)
    current_player = RED if turn_entry["player_who_moved"] == 1 else BLACK
    state = _state_from_real_pipeline(
        board=board,
        current_player=current_player,
        turn_number=5,
        strategic_priorities=turn_entry.get("strategic_priorities", []),
    )
    best_path = max(
        state.legal_moves,
        key=lambda m: m.get("facts", {}).get("minimax_score", float("-inf")),
    )["path"]
    wrong_idx = _pick_wrong_llm_index(state, best_path)

    _patch_ranker_choice(monkeypatch, chosen_index=wrong_idx)
    patch = ranker_module.ranker_agent(state)
    chosen_path = patch["chosen_move"]["path"]

    out = capsys.readouterr().out
    assert "override_triggered=True" in out

    best_score = max(
        m.get("facts", {}).get("minimax_score", float("-inf"))
        for m in state.legal_moves
    )
    chosen_score = next(
        (
            m.get("facts", {}).get("minimax_score", float("-inf"))
            for m in state.legal_moves
            if m["path"] == chosen_path
        ),
        float("-inf"),
    )
    assert chosen_score == best_score, (
        f"Override should pick best minimax. "
        f"chosen_score={chosen_score:.1f}, best={best_score:.1f}, path={chosen_path}"
    )


def test_integration_trace_turn59_hard_cap_override(monkeypatch, capsys):
    # Phase 8 NOTE: Originally pinned to a depth-2 historical path [(6,7),(5,6)].
    # After upgrading PIPELINE_SCORER_DEPTH to 3, the engine evaluates
    # [(6,1),(7,0)] as the stronger move.  The meaningful invariants are:
    #   (a) the override fires, and
    #   (b) the chosen move has the best minimax score in the candidate pool.
    log_path = "/Users/youssefrezk/Desktop/bachelor_project/logs/game_20260419_191038_610610.jsonl"
    board, turn_entry = _reconstruct_board_before_turn(log_path, turn=59)
    current_player = RED if turn_entry["player_who_moved"] == 1 else BLACK
    state = _state_from_real_pipeline(
        board=board,
        current_player=current_player,
        turn_number=59,
        strategic_priorities=turn_entry.get("strategic_priorities", []),
    )
    wrong_idx = _pick_wrong_llm_index(state, [tuple(p) for p in turn_entry["path"]])

    _patch_ranker_choice(monkeypatch, chosen_index=wrong_idx)
    patch = ranker_module.ranker_agent(state)
    chosen_path = patch["chosen_move"]["path"]

    # Invariant 1: override must fire regardless of depth.
    out = capsys.readouterr().out
    assert "override_triggered=True" in out

    # Invariant 2: chosen move must be best minimax in the pool.
    # (At depth=3 this is [(6,1),(7,0)], not the depth-2 historical path.)
    best_score = max(
        m.get("facts", {}).get("minimax_score", float("-inf"))
        for m in state.legal_moves
    )
    chosen_score = next(
        (
            m.get("facts", {}).get("minimax_score", float("-inf"))
            for m in state.legal_moves
            if m["path"] == chosen_path
        ),
        float("-inf"),
    )
    assert chosen_score == best_score, (
        f"Override should pick best minimax. "
        f"chosen_score={chosen_score:.1f}, best={best_score:.1f}, path={chosen_path}"
    )
