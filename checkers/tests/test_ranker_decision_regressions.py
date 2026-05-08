from __future__ import annotations

import json
from pathlib import Path

import pytest

import checkers.agents.ranker_agent as ranker_module
from checkers.engine.board import create_initial_board
from checkers.engine.board import BLACK, EMPTY, RED
from checkers.engine.rules import apply_move
from checkers.agents.scorer_agent import score_all_legal_moves
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
    enriched, *_ = score_all_legal_moves(board, current_player)
    context = {
        "game_phase": "OPENING" if turn_number <= 20 else "MIDGAME",
        "score_state": "EQUAL",
        "strategic_priorities": strategic_priorities or [],
    }
    return CheckersState(
        board=board,
        current_player=current_player,
        turn_number=turn_number,
        legal_moves=enriched,
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
    _log_file = (
        Path(__file__).resolve().parent.parent.parent
        / "logs"
        / "game_20260419_191038_610610.jsonl"
    )
    if not _log_file.exists():
        pytest.skip(f"fixture log not found: {_log_file}")
    log_path = str(_log_file)
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
    _log_file = (
        Path(__file__).resolve().parent.parent.parent
        / "logs"
        / "game_20260419_191038_610610.jsonl"
    )
    if not _log_file.exists():
        pytest.skip(f"fixture log not found: {_log_file}")
    log_path = str(_log_file)
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


def test_unsafe_vs_unsafe_midgap_minimax_uses_retry_not_fallback(monkeypatch, capsys):
    # Both moves are unsafe (opponent_can_recapture=True).
    # Same our_pieces_threatened_after=2 on both → _midgap condition fires (bt==ct).
    # Gap = 45 >= UNSAFE_VS_UNSAFE_MIDGAP_GAP (40).
    #
    # Expected:
    #   - First LLM call picks the worse move → midgap branch triggers.
    #   - One retry LLM call fires (retry_used_full_proposal=True).
    #   - Retry picks the best move → audit passes → override_retry_resolved=True.
    #   - No fallback applied.
    #   - Final chosen move is the best minimax.
    move_best = {
        "type": "simple",
        "path": [(5, 0), (4, 1)],
        "captured": [],
        "facts": {
            "captures_count": 0,
            "creates_immediate_threat": False,
            "shot_sequence_available": False,
            "blocks_opponent_landing": False,
            "opponent_can_recapture": True,
            "moved_piece_is_threatened": False,
            "our_pieces_threatened_after": 2,
            "net_gain": 0,
            "minimax_score": 50.0,
            "quiet_move_role": "DEVELOPMENT",
        },
    }
    move_chosen = {
        "type": "simple",
        "path": [(5, 2), (4, 3)],
        "captured": [],
        "facts": {
            "captures_count": 0,
            "creates_immediate_threat": False,
            "shot_sequence_available": False,
            "blocks_opponent_landing": False,
            "opponent_can_recapture": True,
            "moved_piece_is_threatened": False,
            "our_pieces_threatened_after": 2,
            "net_gain": 0,
            "minimax_score": 5.0,
            "quiet_move_role": "DEVELOPMENT",
        },
    }
    legal_moves = [move_best, move_chosen]

    # First call: LLM picks the worse move (index 1 in filtered=legal space).
    # Retry call: LLM corrects to the best move (index 0 in legal space).
    _call_count = [0]

    def _fake_call_ranker(system: str, user: str) -> str:
        _call_count[0] += 1
        if _call_count[0] == 1:
            return json.dumps({"chosen_index": 1, "reasoning": "midgap test initial call"})
        return json.dumps({"chosen_index": 0, "reasoning": "midgap test retry corrected"})

    monkeypatch.setattr(ranker_module, "call_ranker", _fake_call_ranker)

    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED

    from checkers.state.state import CheckersState as _CS
    state = _CS(
        board=board,
        current_player=RED,
        turn_number=25,
        legal_moves=legal_moves,
        strategic_context={
            "game_phase": "MIDGAME",
            "score_state": "EQUAL",
            "strategic_priorities": [],
        },
    )

    patch = ranker_module.ranker_agent(state)

    assert patch["chosen_move"]["path"] == [(5, 0), (4, 1)], (
        f"Expected best move [(5,0),(4,1)], got {patch['chosen_move']['path']}"
    )

    diag = patch["ranker_diagnostics"]
    assert diag["override_branch_name"] == "unsafe_vs_unsafe_midgap_minimax", (
        f"Expected midgap branch, got {diag['override_branch_name']}"
    )
    assert diag["override_retry_resolved"] is True, (
        "Retry should have resolved the override (not fallback)"
    )
    assert diag["override_fallback_applied"] is False, (
        "Fallback must not fire when retry resolves"
    )
    assert diag["override_retry_attempts"] == 1, (
        f"Expected exactly 1 retry attempt, got {diag['override_retry_attempts']}"
    )

    out = capsys.readouterr().out
    assert "override_retry_resolved=True" in out
    assert "override_retry_attempts=1" in out


def test_tier1_fires_lower_threat_smaller_gap(monkeypatch, capsys):
    # TIER-1: best has fewer threatened pieces (bt=1 < ct=2) AND gap=37 >= 35.
    # After the "threat_after" → "our_pieces_threatened_after" fix, TIER-1 is live.
    # First LLM call picks the worse move; retry corrects to best.
    move_best = {
        "type": "simple",
        "path": [(5, 0), (4, 1)],
        "captured": [],
        "facts": {
            "captures_count": 0,
            "creates_immediate_threat": False,
            "shot_sequence_available": False,
            "blocks_opponent_landing": False,
            "opponent_can_recapture": True,
            "moved_piece_is_threatened": False,
            "our_pieces_threatened_after": 1,
            "net_gain": 0,
            "minimax_score": 50.0,
            "quiet_move_role": "DEVELOPMENT",
        },
    }
    move_chosen = {
        "type": "simple",
        "path": [(5, 2), (4, 3)],
        "captured": [],
        "facts": {
            "captures_count": 0,
            "creates_immediate_threat": False,
            "shot_sequence_available": False,
            "blocks_opponent_landing": False,
            "opponent_can_recapture": True,
            "moved_piece_is_threatened": False,
            "our_pieces_threatened_after": 2,
            "net_gain": 0,
            "minimax_score": 13.0,
            "quiet_move_role": "DEVELOPMENT",
        },
    }
    legal_moves = [move_best, move_chosen]

    _call_count = [0]

    def _fake_call_ranker(system: str, user: str) -> str:
        _call_count[0] += 1
        if _call_count[0] == 1:
            return json.dumps({"chosen_index": 1, "reasoning": "tier1 test initial call"})
        return json.dumps({"chosen_index": 0, "reasoning": "tier1 test retry corrected"})

    monkeypatch.setattr(ranker_module, "call_ranker", _fake_call_ranker)

    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED

    from checkers.state.state import CheckersState as _CS
    state = _CS(
        board=board,
        current_player=RED,
        turn_number=25,
        legal_moves=legal_moves,
        strategic_context={
            "game_phase": "MIDGAME",
            "score_state": "EQUAL",
            "strategic_priorities": [],
        },
    )

    patch = ranker_module.ranker_agent(state)

    assert patch["chosen_move"]["path"] == [(5, 0), (4, 1)], (
        f"Expected best move [(5,0),(4,1)], got {patch['chosen_move']['path']}"
    )

    diag = patch["ranker_diagnostics"]
    assert diag["override_branch_name"] == "unsafe_vs_unsafe_fallback_tier1_lower_threat", (
        f"Expected tier1 branch, got {diag['override_branch_name']}"
    )
    assert diag["override_retry_resolved"] is True
    assert diag["override_fallback_applied"] is False
    assert diag["override_retry_attempts"] == 1


def test_tier2_does_not_fire_below_threshold(monkeypatch, capsys):
    # TIER-2 non-fire: bt=2 > ct=1 (best has MORE threatened pieces), gap=60 < 150.
    # After the fix, bt > ct and gap < UNSAFE_VS_UNSAFE_FALLBACK_GAP → no tier fires.
    # The LLM's choice must be preserved (no override, no retry).
    move_best = {
        "type": "simple",
        "path": [(5, 0), (4, 1)],
        "captured": [],
        "facts": {
            "captures_count": 0,
            "creates_immediate_threat": False,
            "shot_sequence_available": False,
            "blocks_opponent_landing": False,
            "opponent_can_recapture": True,
            "moved_piece_is_threatened": False,
            "our_pieces_threatened_after": 2,
            "net_gain": 0,
            "minimax_score": 70.0,
            "quiet_move_role": "DEVELOPMENT",
        },
    }
    move_chosen = {
        "type": "simple",
        "path": [(5, 2), (4, 3)],
        "captured": [],
        "facts": {
            "captures_count": 0,
            "creates_immediate_threat": False,
            "shot_sequence_available": False,
            "blocks_opponent_landing": False,
            "opponent_can_recapture": True,
            "moved_piece_is_threatened": False,
            "our_pieces_threatened_after": 1,
            "net_gain": 0,
            "minimax_score": 10.0,
            "quiet_move_role": "DEVELOPMENT",
        },
    }
    legal_moves = [move_best, move_chosen]

    _call_count = [0]

    def _fake_call_ranker(system: str, user: str) -> str:
        _call_count[0] += 1
        return json.dumps({"chosen_index": 1, "reasoning": "tier2 non-fire test"})

    monkeypatch.setattr(ranker_module, "call_ranker", _fake_call_ranker)

    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED

    from checkers.state.state import CheckersState as _CS
    state = _CS(
        board=board,
        current_player=RED,
        turn_number=25,
        legal_moves=legal_moves,
        strategic_context={
            "game_phase": "MIDGAME",
            "score_state": "EQUAL",
            "strategic_priorities": [],
        },
    )

    patch = ranker_module.ranker_agent(state)

    assert patch["chosen_move"]["path"] == [(5, 2), (4, 3)], (
        f"LLM choice should be kept; got {patch['chosen_move']['path']}"
    )

    diag = patch["ranker_diagnostics"]
    assert diag["override_branch_name"] is None, (
        f"No unsafe-vs-unsafe tier should fire; got branch={diag['override_branch_name']}"
    )
    assert diag["override_retry_attempts"] == 0, (
        f"No retry expected; got {diag['override_retry_attempts']}"
    )


def test_turn5_unsafe_best_safe_chosen_gap_below_threshold_no_override(monkeypatch, capsys):
    # Turn 5 threshold behavior (pinned — do not lower thresholds without updating this test).
    #
    # Scenario: best move is unsafe (opponent_can_recapture=True, score=13.0),
    # chosen move is safe (score=0.0), gap=13.0.
    #
    # Safety filter (single-safe-move branch): rank1 passes _unsafe_qualifies because
    # gap=13.0 >= MINIMAX_ALL_UNSAFE_MARGIN (3.0) — both moves reach the LLM.
    # LLM prefers safe rank2 (safety-first bias). Override check:
    #   safe_vs_unsafe_large_gap: gap=13.0 < SAFE_VS_UNSAFE_OVERRIDE_GAP (15.0) → does NOT fire.
    #   All other branches need best_safe=True — blocked.
    # Result: override_triggered=False, LLM choice preserved (rank2, 0.0).
    #
    # This is intended gate behavior, not a bug. The 2-point margin below the threshold
    # is deliberate: depth-6 minimax scores can be noisy; 13.0 is not unambiguous enough
    # to override an explicit safety preference. Threshold changes must update this test.
    move_unsafe_best = {
        "type": "simple",
        "path": [(5, 0), (4, 1)],
        "captured": [],
        "facts": {
            "captures_count": 0,
            "creates_immediate_threat": False,
            "shot_sequence_available": False,
            "blocks_opponent_landing": False,
            "opponent_can_recapture": True,
            "moved_piece_is_threatened": False,
            "our_pieces_threatened_after": 1,
            "forced_opponent_jump_reply": False,
            "max_opponent_jump_captures": 1,
            "opponent_jump_count": 1,
            "net_gain": 0,
            "leaves_piece_isolated": False,
            "weakens_king_row": False,
            "center_control": False,
            "minimax_score": 13.0,
            "quiet_move_role": "QUIET_DEFAULT",
            "counterplay_score": 0,
            "king_activity_score": 0,
            "simplification_value": 0,
        },
    }
    move_safe_chosen = {
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
            "forced_opponent_jump_reply": False,
            "max_opponent_jump_captures": 0,
            "opponent_jump_count": 0,
            "net_gain": 0,
            "leaves_piece_isolated": False,
            "weakens_king_row": False,
            "center_control": False,
            "minimax_score": 0.0,
            "quiet_move_role": "QUIET_DEFAULT",
            "counterplay_score": 0,
            "king_activity_score": 0,
            "simplification_value": 0,
        },
    }
    legal_moves = [move_unsafe_best, move_safe_chosen]

    # LLM picks index 1 (the safe move, second in the filtered list).
    # After the safety filter (single-safe-move path): rank1 passes _unsafe_qualifies
    # because gap=13.0 >= MINIMAX_ALL_UNSAFE_MARGIN (3.0), so filtered = [rank1, rank2].
    _patch_ranker_choice(monkeypatch, chosen_index=1)

    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[0][1] = BLACK

    from checkers.state.state import CheckersState as _CS
    state = _CS(
        board=board,
        current_player=RED,
        turn_number=5,
        legal_moves=legal_moves,
        strategic_context={
            "game_phase": "OPENING",
            "score_state": "EQUAL",
            "strategic_priorities": ["DEVELOP_PIECES", "CONTROL_CENTER"],
        },
    )

    patch = ranker_module.ranker_agent(state)

    assert patch["chosen_move"]["path"] == [(5, 2), (4, 3)], (
        f"LLM safe choice should be preserved when gap ({13.0}) < "
        f"SAFE_VS_UNSAFE_OVERRIDE_GAP ({ranker_module.SAFE_VS_UNSAFE_OVERRIDE_GAP}); "
        f"got {patch['chosen_move']['path']}"
    )

    out = capsys.readouterr().out
    assert "override_triggered=False" in out, (
        f"No override should fire at gap=13.0 < {ranker_module.SAFE_VS_UNSAFE_OVERRIDE_GAP}; "
        f"output: {out}"
    )


def test_unsafe_best_safe_chosen_gap_at_threshold_fires_override(monkeypatch, capsys):
    # Complement to the Turn 5 test: verify safe_vs_unsafe_large_gap fires when
    # gap >= SAFE_VS_UNSAFE_OVERRIDE_GAP (15.0). Uses gap=16.0 to be clearly above.
    move_unsafe_best = {
        "type": "simple",
        "path": [(5, 0), (4, 1)],
        "captured": [],
        "facts": {
            "captures_count": 0,
            "creates_immediate_threat": False,
            "shot_sequence_available": False,
            "blocks_opponent_landing": False,
            "opponent_can_recapture": True,
            "moved_piece_is_threatened": False,
            "our_pieces_threatened_after": 1,
            "forced_opponent_jump_reply": False,
            "max_opponent_jump_captures": 1,
            "opponent_jump_count": 1,
            "net_gain": 0,
            "leaves_piece_isolated": False,
            "weakens_king_row": False,
            "center_control": False,
            "minimax_score": 16.0,
            "quiet_move_role": "QUIET_DEFAULT",
            "counterplay_score": 0,
            "king_activity_score": 0,
            "simplification_value": 0,
        },
    }
    move_safe_chosen = {
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
            "forced_opponent_jump_reply": False,
            "max_opponent_jump_captures": 0,
            "opponent_jump_count": 0,
            "net_gain": 0,
            "leaves_piece_isolated": False,
            "weakens_king_row": False,
            "center_control": False,
            "minimax_score": 0.0,
            "quiet_move_role": "QUIET_DEFAULT",
            "counterplay_score": 0,
            "king_activity_score": 0,
            "simplification_value": 0,
        },
    }
    legal_moves = [move_unsafe_best, move_safe_chosen]

    _patch_ranker_choice(monkeypatch, chosen_index=1)

    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[0][1] = BLACK

    from checkers.state.state import CheckersState as _CS
    state = _CS(
        board=board,
        current_player=RED,
        turn_number=5,
        legal_moves=legal_moves,
        strategic_context={
            "game_phase": "OPENING",
            "score_state": "EQUAL",
            "strategic_priorities": ["DEVELOP_PIECES", "CONTROL_CENTER"],
        },
    )

    patch = ranker_module.ranker_agent(state)

    assert patch["chosen_move"]["path"] == [(5, 0), (4, 1)], (
        f"Override should fire at gap=16.0 >= SAFE_VS_UNSAFE_OVERRIDE_GAP "
        f"({ranker_module.SAFE_VS_UNSAFE_OVERRIDE_GAP}); got {patch['chosen_move']['path']}"
    )

    diag = patch["ranker_diagnostics"]
    assert diag["override_branch_name"] == "safe_vs_unsafe_large_gap", (
        f"Expected safe_vs_unsafe_large_gap branch, got {diag['override_branch_name']}"
    )

    out = capsys.readouterr().out
    assert "override_triggered=True" in out


def test_tier2_fires_above_threshold(monkeypatch, capsys):
    # TIER-2 fire: bt=2 > ct=1 (best has MORE threatened pieces), gap=160 >= 150.
    # After the fix, TIER-2 fires when gap is extreme despite counter-intuitive threat direction.
    # First LLM call picks the worse move; retry corrects to best.
    move_best = {
        "type": "simple",
        "path": [(5, 0), (4, 1)],
        "captured": [],
        "facts": {
            "captures_count": 0,
            "creates_immediate_threat": False,
            "shot_sequence_available": False,
            "blocks_opponent_landing": False,
            "opponent_can_recapture": True,
            "moved_piece_is_threatened": False,
            "our_pieces_threatened_after": 2,
            "net_gain": 0,
            "minimax_score": 170.0,
            "quiet_move_role": "DEVELOPMENT",
        },
    }
    move_chosen = {
        "type": "simple",
        "path": [(5, 2), (4, 3)],
        "captured": [],
        "facts": {
            "captures_count": 0,
            "creates_immediate_threat": False,
            "shot_sequence_available": False,
            "blocks_opponent_landing": False,
            "opponent_can_recapture": True,
            "moved_piece_is_threatened": False,
            "our_pieces_threatened_after": 1,
            "net_gain": 0,
            "minimax_score": 10.0,
            "quiet_move_role": "DEVELOPMENT",
        },
    }
    legal_moves = [move_best, move_chosen]

    _call_count = [0]

    def _fake_call_ranker(system: str, user: str) -> str:
        _call_count[0] += 1
        if _call_count[0] == 1:
            return json.dumps({"chosen_index": 1, "reasoning": "tier2 test initial call"})
        return json.dumps({"chosen_index": 0, "reasoning": "tier2 test retry corrected"})

    monkeypatch.setattr(ranker_module, "call_ranker", _fake_call_ranker)

    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED

    from checkers.state.state import CheckersState as _CS
    state = _CS(
        board=board,
        current_player=RED,
        turn_number=25,
        legal_moves=legal_moves,
        strategic_context={
            "game_phase": "MIDGAME",
            "score_state": "EQUAL",
            "strategic_priorities": [],
        },
    )

    patch = ranker_module.ranker_agent(state)

    assert patch["chosen_move"]["path"] == [(5, 0), (4, 1)], (
        f"Expected best move [(5,0),(4,1)], got {patch['chosen_move']['path']}"
    )

    diag = patch["ranker_diagnostics"]
    assert diag["override_branch_name"] == "unsafe_vs_unsafe_fallback_tier2_higher_threat", (
        f"Expected tier2 branch, got {diag['override_branch_name']}"
    )
    assert diag["override_retry_resolved"] is True
    assert diag["override_fallback_applied"] is False
    assert diag["override_retry_attempts"] == 1
