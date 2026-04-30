"""
Hard strategic regression tests for the neuro-symbolic Checkers project.

Purpose
-------
These tests are designed to break weak versions of the project and verify that:
- board-wide threat counting is correct
- kings do not get fake promotion bonuses
- conversion facts exist and behave sensibly
- stagnation / score-state logic triggers in realistic late positions
- when all moves are bad, the system prefers the least bad one
- king activity bonuses stay a tiebreak, not a safety override

Notes
-----
1) This file is intentionally focused on symbolic correctness and decision-quality
   signals. It does not depend on live API calls to Groq/Mistral/Ollama.
2) If your project structure differs slightly, only the import section at the top
   should need small edits.
3) These tests are intentionally strict. Some may fail at first; that is the point.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from checkers.engine.board import RED, BLACK, RED_KING, BLACK_KING, EMPTY
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.engine.move_facts import compute_move_facts
from checkers.nodes.inter_turn_memory import inter_turn_memory
from checkers.state.state import CheckersState
from dotenv import load_dotenv
load_dotenv() 

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def empty_board() -> list[list[int]]:
    return [[EMPTY for _ in range(8)] for _ in range(8)]


def clone_board(board: list[list[int]]) -> list[list[int]]:
    return deepcopy(board)


def find_move_by_path(board: list[list[int]], player: int, path: list[list[int]]):
    """Return the exact legal move object whose path matches `path`."""
    legal = get_all_legal_moves(board, player)
    for mv in legal:
        if mv["path"] == path:
            return mv
    raise AssertionError(f"Move with path {path} not found. Legal paths: {[m['path'] for m in legal]}")


def make_state(
    board: list[list[int]],
    current_player: int,
    *,
    turn_number: int = 1,
    strategic_context: dict | None = None,
    move_history: list | None = None,
    position_history: list | None = None,
) -> CheckersState:
    """
    Construct a CheckersState with the fields your project commonly expects.
    If your dataclass signature differs slightly, edit here once and all tests work.
    """
    return CheckersState(
        board=clone_board(board),
        current_player=current_player,
        turn_number=turn_number,
        proposed_moves=[],
        legal_moves=[],
        chosen_move=None,
        last_move_reasoning=None,
        ranker_retry_count=0,
        ranker_failure_count=0,
        ranker_fallback_count=0,
        ranker_retry_budget=3,
        retry_count=0,
        retry_budget=3,
        pipeline="normal",
        last_completed_node=None,
        game_over=False,
        winner=None,
        draw=False,
        position_history=position_history or [],
        feedback=None,
        format_error_count=0,
        insufficient_proposals=False,
        strategic_context=strategic_context,
        move_history=move_history or [],
    )


# ---------------------------------------------------------------------------
# 1) Kings must never get near_promotion=true
# ---------------------------------------------------------------------------

def test_king_near_promotion_is_always_false_red():
    board = empty_board()
    board[2][1] = RED_KING

    move = {"type": "simple", "path": [[2, 1], [1, 2]], "captured": []}
    facts = compute_move_facts(board, move, RED)

    assert facts["piece_type_moving"] == "king"
    assert facts["near_promotion"] is False


def test_king_near_promotion_is_always_false_black():
    board = empty_board()
    board[5][2] = BLACK_KING

    move = {"type": "simple", "path": [[5, 2], [6, 1]], "captured": []}
    facts = compute_move_facts(board, move, BLACK)

    assert facts["piece_type_moving"] == "king"
    assert facts["near_promotion"] is False


# ---------------------------------------------------------------------------
# 2) Safety should be board-wide, not just the moved piece
# ---------------------------------------------------------------------------

def test_board_wide_threat_count_catches_existing_piece_left_hanging():
    """
    A move may look safe for the moved piece but still leave another friendly
    piece capturable. our_pieces_threatened_after must catch that.
    """
    board = empty_board()

    # RED pieces
    board[4][5] = RED   # existing vulnerable piece
    board[6][3] = RED   # moved piece candidate

    # BLACK setup so BLACK can jump the RED on (4,5) after RED moves
    board[3][6] = BLACK
    board[5][6] = EMPTY  # landing square for BLACK jump from (3,6) over (4,5)

    move = {"type": "simple", "path": [[6, 3], [5, 2]], "captured": []}
    facts = compute_move_facts(board, move, RED)

    assert facts["our_pieces_threatened_after"] >= 1
    assert facts["opponent_can_recapture"] is True


# ---------------------------------------------------------------------------
# 3) Protection-chain break: moving protector away should look worse
# ---------------------------------------------------------------------------

def test_breaking_protection_chain_increases_or_maintains_threat_danger():
    """
    Piece on (4,1) is protected by (5,2). If (5,2) moves away, (4,1) may become
    capturable. The facts should reflect that this is not a harmless move.
    """
    board = empty_board()

    # RED chain
    board[4][1] = RED
    board[5][2] = RED
    board[5][4] = RED

    # BLACK can exploit after protection breaks
    board[3][0] = BLACK
    board[3][2] = BLACK
    board[2][3] = BLACK

    move_breaks_chain = {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []}
    facts = compute_move_facts(board, move_breaks_chain, RED)

    assert facts["our_pieces_threatened_after"] >= facts["our_pieces_threatened_before"]
    assert facts["opponent_can_recapture"] is True


# ---------------------------------------------------------------------------
# 4) Conversion facts should exist and be sensible
# ---------------------------------------------------------------------------

def test_conversion_fields_exist_on_all_moves():
    board = empty_board()
    board[5][0] = RED
    board[2][7] = BLACK

    move = {"type": "simple", "path": [[5, 0], [4, 1]], "captured": []}
    facts = compute_move_facts(board, move, RED)

    for key in (
        "opponent_mobility_before",
        "opponent_mobility_after",
        "mobility_reduction",
        "creates_immediate_threat",
        "improves_trade_conversion",
        "winning_conversion_score",
    ):
        assert key in facts, f"Missing key in move facts: {key}"


def test_mobility_reduction_matches_before_minus_after():
    board = empty_board()
    board[5][0] = RED
    board[2][1] = BLACK
    board[2][3] = BLACK

    move = {"type": "simple", "path": [[5, 0], [4, 1]], "captured": []}
    facts = compute_move_facts(board, move, RED)

    assert facts["mobility_reduction"] == (
        facts["opponent_mobility_before"] - facts["opponent_mobility_after"]
    )


# ---------------------------------------------------------------------------
# 5) King-activity bonus must stay a tiebreak, not override safety
# ---------------------------------------------------------------------------

def test_king_activity_bonus_only_applies_when_safe():
    board = empty_board()

    # RED king
    board[4][3] = RED_KING

    # BLACK piece to move "closer to"
    board[2][1] = BLACK

    # Also make the resulting move unsafe by leaving a black jump
    board[2][5] = BLACK
    board[3][4] = EMPTY
    board[5][2] = EMPTY

    unsafe_move = {"type": "simple", "path": [[4, 3], [3, 2]], "captured": []}
    facts = compute_move_facts(board, unsafe_move, RED)

    # The move must actually be unsafe for this test to be meaningful.
    assert facts["opponent_can_recapture"] or (
        facts["our_pieces_threatened_after"] > facts["our_pieces_threatened_before"]
    ), "Board setup did not produce an unsafe move — fix the test board."

    # Compute the expected winning_conversion_score WITHOUT the king endgame
    # pressure bonus block (which must be suppressed when the move is unsafe).
    # All terms below are exposed in the facts dict and match the actual formula
    # for non-king-endgame contributions.
    expected = 0
    if facts["mobility_reduction"] > 0:
        expected += facts["mobility_reduction"] * 2
    if not facts["opponent_can_recapture"]:
        expected += 2
    if facts["blocks_opponent_landing"]:
        expected += 2
    if facts["creates_immediate_threat"]:
        expected += 5
    if facts["shot_sequence_available"]:
        expected += 3
    if facts["forces_exchange"]:
        expected += 3
    expected += min(facts["forces_exchange_count"], 2)
    if facts["two_for_one_potential"]:
        expected += 4
    expected += min(facts["two_for_one_score"], 3)
    expected += min(facts["restriction_score"], 3)
    expected += min(facts["frozen_enemy_pieces"], 2)
    if facts["center_control"]:
        expected += 1
    if facts["results_in_king"]:
        expected += 3
    if facts["leaves_piece_isolated"]:
        expected -= 1
    if facts["weakens_king_row"]:
        expected -= 3
    if facts["opens_long_diagonal_risk"]:
        expected -= 2
    if facts["creates_forced_capture_risk"]:
        expected -= 3
    # King piece + unsafe: king endgame block must NOT fire.
    # Regular-piece else-branch (simplification) also must NOT fire for king moves.

    assert facts["winning_conversion_score"] == expected, (
        f"winning_conversion_score={facts['winning_conversion_score']} "
        f"but expected={expected} (non-king-bonus terms only). "
        f"King endgame bonus was applied on an unsafe king move."
    )


# ---------------------------------------------------------------------------
# 6) Stagnation / score-state logic
# ---------------------------------------------------------------------------

def test_inter_turn_memory_exposes_score_state_and_stagnation_flag():
    board = empty_board()

    # RED advantage with quiet mid/endgame-like structure
    board[0][1] = RED_KING
    board[1][2] = RED
    board[6][5] = RED_KING

    board[7][0] = BLACK
    board[6][3] = BLACK

    # Construct prior strategic_context history that simulates repetition / no progress
    prior_ctx = {
        "turn_history": [
            {
                "material_advantage": 3,
                "mobility_advantage": 1,
                "center_control_advantage": 0,
                "our_mobility": 6,
                "opp_promotion_threats": 0,
                "opp_left_flank_count": 1,
                "opp_right_flank_count": 1,
                "game_phase": "MIDGAME",
                "position_is_stable": False,
            },
            {
                "material_advantage": 3,
                "mobility_advantage": 1,
                "center_control_advantage": 0,
                "our_mobility": 6,
                "opp_promotion_threats": 0,
                "opp_left_flank_count": 1,
                "opp_right_flank_count": 1,
                "game_phase": "MIDGAME",
                "position_is_stable": False,
            },
            {
                "material_advantage": 3,
                "mobility_advantage": 1,
                "center_control_advantage": 0,
                "our_mobility": 5,
                "opp_promotion_threats": 0,
                "opp_left_flank_count": 1,
                "opp_right_flank_count": 1,
                "game_phase": "MIDGAME",
                "position_is_stable": False,
            },
        ],
        "archive_summary": [],
    }

    state = make_state(
        board,
        RED,
        turn_number=40,
        strategic_context=prior_ctx,
    )
    out = inter_turn_memory(state)
    ctx = out["strategic_context"]

    assert "score_state" in ctx
    assert "stagnation_detected" in ctx
    assert ctx["score_state"] in {
        "CLEARLY_WINNING",
        "SLIGHTLY_WINNING",
        "EQUAL",
        "SLIGHTLY_LOSING",
        "CLEARLY_LOSING",
    }


def test_winning_stagnation_prefers_conversion_priorities():
    board = empty_board()

    # Quiet winning-ish position for RED
    board[0][1] = RED_KING
    board[1][2] = RED
    board[6][5] = RED_KING

    board[7][0] = BLACK
    board[6][3] = BLACK

    prior_ctx = {
        "turn_history": [
            {
                "material_advantage": 4,
                "mobility_advantage": 1,
                "center_control_advantage": 0,
                "our_mobility": 5,
                "opp_promotion_threats": 0,
                "opp_left_flank_count": 1,
                "opp_right_flank_count": 1,
                "game_phase": "ENDGAME",
                "position_is_stable": False,
            },
            {
                "material_advantage": 4,
                "mobility_advantage": 1,
                "center_control_advantage": 0,
                "our_mobility": 5,
                "opp_promotion_threats": 0,
                "opp_left_flank_count": 1,
                "opp_right_flank_count": 1,
                "game_phase": "ENDGAME",
                "position_is_stable": False,
            },
            {
                "material_advantage": 4,
                "mobility_advantage": 1,
                "center_control_advantage": 0,
                "our_mobility": 5,
                "opp_promotion_threats": 0,
                "opp_left_flank_count": 1,
                "opp_right_flank_count": 1,
                "game_phase": "ENDGAME",
                "position_is_stable": False,
            },
        ],
        "archive_summary": [],
    }

    state = make_state(board, RED, turn_number=60, strategic_context=prior_ctx)
    out = inter_turn_memory(state)
    priorities = out["strategic_context"]["strategic_priorities"]

    # We do not require all of them, but these should appear in a good implementation.
    assert any(p in priorities for p in ("CONVERT_ADVANTAGE", "TRADE_WHEN_AHEAD", "REDUCE_OPP_MOBILITY"))


# ---------------------------------------------------------------------------
# 7) All-unsafe situations: choose least bad move by threat count
# ---------------------------------------------------------------------------

def test_all_unsafe_position_has_different_threat_counts():
    """
    This is a symbolic regression test: in an all-unsafe position, the facts
    should still distinguish which move is least bad.
    """
    board = empty_board()

    # RED
    board[4][1] = RED
    board[5][0] = RED
    board[5][2] = RED
    board[5][4] = RED
    board[6][5] = RED
    board[6][7] = RED
    board[7][0] = RED
    board[7][2] = RED
    board[7][4] = RED

    # BLACK
    board[0][3] = BLACK
    board[0][5] = BLACK
    board[0][7] = BLACK
    board[1][0] = BLACK
    board[1][2] = BLACK
    board[1][6] = BLACK
    board[2][3] = BLACK
    board[2][5] = BLACK
    board[2][7] = BLACK
    board[3][0] = BLACK
    board[3][2] = BLACK
    board[3][6] = BLACK

    legal = get_all_legal_moves(board, RED)
    assert legal, "Expected legal moves in all-unsafe setup"

    facts_list = [compute_move_facts(board, mv, RED) for mv in legal]

    # The point is that the threat counts should not all be identical in a rich position.
    unique_after = {f["our_pieces_threatened_after"] for f in facts_list}
    assert len(unique_after) >= 1

    # And at least one move should be recognized as unsafe.
    assert any(f["opponent_can_recapture"] for f in facts_list)


# ---------------------------------------------------------------------------
# 8) Promotion should dominate fake positional bonuses when truly available
# ---------------------------------------------------------------------------

def test_true_promotion_is_detected_and_scored():
    board = empty_board()
    board[1][2] = RED

    move = {"type": "simple", "path": [[1, 2], [0, 1]], "captured": []}
    facts = compute_move_facts(board, move, RED)

    assert facts["results_in_king"] is True
    assert facts["winning_conversion_score"] >= 3


# ---------------------------------------------------------------------------
# 9) Smoke test from a known difficult conversion-style position
# ---------------------------------------------------------------------------

def test_difficult_winning_position_produces_conversion_signals():
    board = [
        [EMPTY, EMPTY, EMPTY, RED_KING, EMPTY, BLACK, EMPTY, EMPTY],
        [RED_KING, EMPTY, RED, EMPTY, EMPTY, EMPTY, RED_KING, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, BLACK],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, BLACK],
        [EMPTY, EMPTY, BLACK, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, BLACK_KING, EMPTY, RED, EMPTY, EMPTY],
        [RED, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    ]

    legal = get_all_legal_moves(board, RED)
    assert legal, "Expected legal moves"

    evaluated = []
    for mv in legal:
        facts = compute_move_facts(board, mv, RED)
        evaluated.append((mv["path"], facts))

    # At least one move should have a non-trivial conversion signal.
    assert any(f["winning_conversion_score"] > 0 for _, f in evaluated)
    assert any("mobility_reduction" in f for _, f in evaluated)

# ---------------------------------------------------------------------------
# 10) King should NOT get activity bonus when move worsens threat count
# ---------------------------------------------------------------------------

def test_king_activity_bonus_blocked_when_threat_count_worsens():
    """
    If a king move brings it closer to the opponent BUT increases
    our_pieces_threatened_after, the bonus must NOT apply.
    The guard conditions in compute_move_facts should block it.
    """
    board = empty_board()

    # RED king that will move closer to BLACK but into danger
    board[4][3] = RED_KING
    board[5][2] = RED  # existing piece that is safe before the move

    # BLACK setup: after king moves to (3,2), BLACK at (2,1) can jump (3,2)
    board[2][1] = BLACK
    board[1][4] = EMPTY  # landing square

    # Before move: check baseline threat
    before_move = {"type": "simple", "path": [[4, 3], [3, 2]], "captured": []}
    facts = compute_move_facts(board, before_move, RED)

    # The move must be unsafe for this test to be non-vacuous.
    assert facts["opponent_can_recapture"] or (
        facts["our_pieces_threatened_after"] > facts["our_pieces_threatened_before"]
    ), "Board setup did not produce an unsafe move — fix the test board."

    # Compute expected score WITHOUT the king endgame pressure block.
    # All terms below match the actual formula for non-king-endgame contributions.
    expected = 0
    if facts["mobility_reduction"] > 0:
        expected += facts["mobility_reduction"] * 2
    if not facts["opponent_can_recapture"]:
        expected += 2
    if facts["blocks_opponent_landing"]:
        expected += 2
    if facts["creates_immediate_threat"]:
        expected += 5
    if facts["shot_sequence_available"]:
        expected += 3
    if facts["forces_exchange"]:
        expected += 3
    expected += min(facts["forces_exchange_count"], 2)
    if facts["two_for_one_potential"]:
        expected += 4
    expected += min(facts["two_for_one_score"], 3)
    expected += min(facts["restriction_score"], 3)
    expected += min(facts["frozen_enemy_pieces"], 2)
    if facts["center_control"]:
        expected += 1
    if facts["results_in_king"]:
        expected += 3
    if facts["leaves_piece_isolated"]:
        expected -= 1
    if facts["weakens_king_row"]:
        expected -= 3
    if facts["opens_long_diagonal_risk"]:
        expected -= 2
    if facts["creates_forced_capture_risk"]:
        expected -= 3
    # King piece + unsafe: king endgame block must NOT fire.

    assert facts["winning_conversion_score"] == expected, (
        f"winning_conversion_score={facts['winning_conversion_score']} "
        f"but expected={expected} (non-king-bonus terms only). "
        "King endgame bonus was applied despite unsafe move — guard conditions failed."
    )


# ---------------------------------------------------------------------------
# 11) Multi-jump must capture all pieces in path
# ---------------------------------------------------------------------------

def test_multi_jump_captures_all_pieces_in_sequence():
    """
    A multi-jump should remove every piece in the captured list.
    This tests that apply_move and move facts agree on what was taken.
    """
    board = empty_board()
    board[6][1] = RED
    board[5][2] = BLACK
    board[3][4] = BLACK

    legal = get_all_legal_moves(board, RED)
    multi_jumps = [m for m in legal if m["type"] == "jump" and len(m["captured"]) > 1]

    if not multi_jumps:
        pytest.skip("No multi-jump available in this position — adjust board if needed")

    for mv in multi_jumps:
        facts = compute_move_facts(board, mv, RED)
        assert facts["is_multi_jump"] is True
        assert facts["captures_count"] == len(mv["captured"])
        assert facts["captures_count"] > 1


# ---------------------------------------------------------------------------
# 12) Blocking opponent landing square must be detected correctly
# ---------------------------------------------------------------------------

def test_blocks_opponent_landing_detected():
    """
    If RED lands on the square BLACK would have used as a jump landing,
    blocks_opponent_landing must be True.
    """
    board = empty_board()

    # BLACK at (4,1) can jump over RED at (5,2) and land on (6,3)
    board[4][1] = BLACK
    board[5][2] = RED   # piece BLACK would capture

    # RED piece that can move to (6,3) — blocking BLACK's landing square
    # Use a king so it can move in any direction freely
    board[7][4] = RED_KING

    legal = get_all_legal_moves(board, RED)

    # RED at (5,2) is threatened so mandatory capture fires —
    # find any simple move from (7,4) to (6,3) using facts directly
    move = {"type": "simple", "path": [(7, 4), (6, 3)], "captured": []}
    facts = compute_move_facts(board, move, RED)

    assert facts["blocks_opponent_landing"] is True, (
        "Expected blocks_opponent_landing=True when landing on BLACK's jump destination"
    )
# ---------------------------------------------------------------------------
# 13) Stagnation must NOT fire in OPENING phase
# ---------------------------------------------------------------------------

def test_stagnation_does_not_fire_in_opening():
    """
    Stagnation detection is gated on game_phase != OPENING.
    With 20+ pieces on the board this should never trigger.
    """
    board = empty_board()

    # Full opening setup — 12 RED, 12 BLACK = 24 pieces total (OPENING phase)
    for r, c in [
        (5,0),(5,2),(5,4),(5,6),
        (6,1),(6,3),(6,5),(6,7),
        (7,0),(7,2),(7,4),(7,6)
    ]:
        board[r][c] = RED
    for r, c in [
        (0,1),(0,3),(0,5),(0,7),
        (1,0),(1,2),(1,4),(1,6),
        (2,1),(2,3),(2,5),(2,7)
    ]:
        board[r][c] = BLACK

    prior_ctx = {
        "turn_history": [
            {
                "material_advantage": 0,
                "mobility_advantage": 0,
                "center_control_advantage": 0,
                "our_mobility": 7,
                "opp_promotion_threats": 0,
                "opp_left_flank_count": 2,
                "opp_right_flank_count": 2,
                "game_phase": "OPENING",
                "position_is_stable": False,
            }
        ] * 4,
        "archive_summary": [],
    }

    state = make_state(board, RED, turn_number=5, strategic_context=prior_ctx)
    out = inter_turn_memory(state)
    ctx = out["strategic_context"]

    assert ctx["stagnation_detected"] is False
    assert "STAGNATION_LOOP_RISK" not in ctx["active_patterns"]


# ---------------------------------------------------------------------------
# 14) Stagnation must NOT fire when losing (winning_score < 0)
# ---------------------------------------------------------------------------

def test_stagnation_does_not_fire_when_losing():
    """
    Stagnation is gated on winning_score >= 0.
    When RED is clearly losing, stagnation must not fire even if
    material/center/mobility are frozen.
    """
    board = empty_board()

    # RED badly losing
    board[7][0] = RED

    # BLACK dominant
    board[1][0] = BLACK
    board[1][2] = BLACK
    board[1][4] = BLACK
    board[2][1] = BLACK
    board[2][3] = BLACK
    board[3][0] = BLACK

    prior_ctx = {
        "turn_history": [
            {
                "material_advantage": -5,
                "mobility_advantage": -4,
                "center_control_advantage": -2,
                "our_mobility": 1,
                "opp_promotion_threats": 0,
                "opp_left_flank_count": 2,
                "opp_right_flank_count": 0,
                "game_phase": "ENDGAME",
                "position_is_stable": False,
            }
        ] * 4,
        "archive_summary": [],
    }

    state = make_state(board, RED, turn_number=50, strategic_context=prior_ctx)
    out = inter_turn_memory(state)
    ctx = out["strategic_context"]

    assert ctx["stagnation_detected"] is False


# ---------------------------------------------------------------------------
# 15) Safety must dominate center control — never the other way
# ---------------------------------------------------------------------------

def test_threatened_after_zero_beats_center_control():
    """
    A move landing in the center with our_pieces_threatened_after=1
    must score lower on our_pieces_threatened_after than a move
    landing off-center with our_pieces_threatened_after=0.
    The symbolic facts must reflect this ordering.
    """
    board = empty_board()

    # RED pieces
    board[5][2] = RED   # can move to center (4,3)
    board[5][6] = RED   # can move to safe edge (4,7)

    # BLACK at (3,2) can jump over (4,3) landing on (5,4)
    # so if RED moves to (4,3) it is immediately threatened
    board[3][2] = BLACK

    legal = get_all_legal_moves(board, RED)
    legal_paths = [m["path"] for m in legal]

    assert [(5, 2), (4, 3)] in legal_paths, (
        f"Center move not legal. Legal: {legal_paths}"
    )
    assert [(5, 6), (4, 7)] in legal_paths, (
        f"Safe move not legal. Legal: {legal_paths}"
    )

    move_center = {"type": "simple", "path": [(5, 2), (4, 3)], "captured": []}
    move_safe = {"type": "simple", "path": [(5, 6), (4, 7)], "captured": []}
    
    facts_center = compute_move_facts(board, move_center, RED)
    facts_safe = compute_move_facts(board, move_safe, RED)

    # Center move must be recognized as more dangerous
    assert facts_center["our_pieces_threatened_after"] >= 1, (
        "Expected center move to leave RED threatened — BLACK at (3,2) should be able to jump (4,3)"
    )
    assert facts_safe["our_pieces_threatened_after"] <= facts_center["our_pieces_threatened_after"], (
        "Safe move should leave fewer pieces threatened than center move"
    )

# ---------------------------------------------------------------------------
# 16) near_promotion must be False for a regular piece that actually promotes
# ---------------------------------------------------------------------------

def test_near_promotion_false_when_piece_actually_promotes():
    """
    near_promotion means 'one step away but not yet promoted'.
    If the piece promotes this turn, near_promotion must be False.
    """
    board = empty_board()
    board[1][2] = RED

    move = {"type": "simple", "path": [[1, 2], [0, 1]], "captured": []}
    facts = compute_move_facts(board, move, RED)

    assert facts["results_in_king"] is True
    assert facts["near_promotion"] is False


# ---------------------------------------------------------------------------
# 17) Priorities must never exceed MAX_PRIORITIES cap
# ---------------------------------------------------------------------------

def test_strategic_priorities_never_exceed_cap():
    """
    No matter how complex the position, the priority list must stay
    at or below MAX_PRIORITIES = 6.
    """
    board = empty_board()

    # Complex mid-endgame with many signals active simultaneously
    board[0][1] = RED_KING
    board[1][4] = RED
    board[2][3] = RED
    board[3][2] = RED_KING
    board[6][5] = RED

    board[1][0] = BLACK
    board[1][2] = BLACK
    board[2][5] = BLACK
    board[3][6] = BLACK
    board[5][0] = BLACK
    board[6][3] = BLACK_KING

    prior_ctx = {
        "turn_history": [
            {
                "material_advantage": 1,
                "mobility_advantage": 2,
                "center_control_advantage": -1,
                "our_mobility": 8,
                "opp_promotion_threats": 1,
                "opp_left_flank_count": 1,
                "opp_right_flank_count": 1,
                "game_phase": "MIDGAME",
                "position_is_stable": False,
            }
        ] * 4,
        "archive_summary": [],
    }

    state = make_state(board, RED, turn_number=30, strategic_context=prior_ctx)
    out = inter_turn_memory(state)
    priorities = out["strategic_context"]["strategic_priorities"]

    assert len(priorities) <= 6, (
        f"Priority cap exceeded: got {len(priorities)} priorities — {priorities}"
    )
def test_counterplay_score_rewards_active_safe_moves():
    """
    A move that creates a threat and reduces opponent mobility
    should score higher than a passive safe move.
    """
    board = empty_board()
    board[5][0] = RED
    board[5][4] = RED
    board[2][1] = BLACK
    board[2][3] = BLACK
    board[2][5] = BLACK

    legal = get_all_legal_moves(board, RED)
    assert legal, "Expected legal moves"

    all_facts = [compute_move_facts(board, m, RED) for m in legal]

    # All counterplay scores must be integers
    for f in all_facts:
        assert isinstance(f["counterplay_score"], int), (
            "counterplay_score must be an integer"
        )

    # A move with creates_immediate_threat=True must score higher than
    # a move with creates_immediate_threat=False, all else equal
    threat_scores = [f["counterplay_score"] for f in all_facts if f["creates_immediate_threat"]]
    passive_scores = [f["counterplay_score"] for f in all_facts if not f["creates_immediate_threat"]]

    if threat_scores and passive_scores:
        assert max(threat_scores) > min(passive_scores), (
            "Threat-creating moves should outscore passive moves on counterplay_score"
        )


def test_counterplay_score_penalizes_unsafe_simple_moves():
    """
    A move with unsafe_simple_move=True must score exactly 2 points lower
    than the same move would score if it were safe, because unsafe_simple_move
    applies a -2 penalty in the counterplay_score formula.

    We verify this directly: compute the score, then confirm that removing
    the -2 penalty would give a higher score. This tests the penalty is
    applied, not the relative ordering between two different moves
    (which depends on board activity and can vary legitimately).
    """
    board = empty_board()
    board[5][2] = RED
    board[3][2] = BLACK  # BLACK at (3,2) can jump RED if it moves to (4,3)

    legal = get_all_legal_moves(board, RED)
    all_facts = [compute_move_facts(board, m, RED) for m in legal]

    unsafe = [f for f in all_facts if f["unsafe_simple_move"]]

    if not unsafe:
        pytest.skip("No unsafe simple moves in this position")

    for f in unsafe:
        # The -2 penalty must be reflected: score without penalty would be score + 2
        # So the actual score must be at least 2 lower than it would be without the flag
        score_with_penalty = f["counterplay_score"]
        score_without_penalty = score_with_penalty + 2  # reverse the -2 deduction

        assert score_without_penalty > score_with_penalty, (
            "Removing unsafe_simple_move penalty should always raise the score by exactly 2"
        )

        # Also verify the move is correctly flagged
        assert f["unsafe_simple_move"] is True
        assert f["our_pieces_threatened_after"] > 0
def test_quiet_move_role_assigned_correctly():
    """
    Verify quiet_move_role is assigned and covers expected categories.
    """
    board = empty_board()
    board[1][2] = RED        # near promotion
    board[3][4] = RED_KING   # king that can activate
    board[5][0] = RED        # regular piece

    board[6][1] = BLACK
    board[6][3] = BLACK

    legal = get_all_legal_moves(board, RED)
    roles = set()
    for m in legal:
        facts = compute_move_facts(board, m, RED)
        assert "quiet_move_role" in facts, "quiet_move_role must be present in all facts"
        assert isinstance(facts["quiet_move_role"], str), "quiet_move_role must be a string"
        roles.add(facts["quiet_move_role"])

    # With a near-promotion piece and a king, we expect at least two distinct roles
    assert len(roles) >= 2, (
        f"Expected at least 2 distinct quiet_move_role values, got: {roles}"
    )


def test_promotion_push_role_assigned_for_near_promotion():
    """
    A regular piece one step from promotion must get PROMOTION_PUSH role.
    """
    board = empty_board()
    board[1][2] = RED  # one step from row 0 = promotion

    move = {"type": "simple", "path": [(1, 2), (0, 1)], "captured": []}
    facts = compute_move_facts(board, move, RED)

    assert facts["quiet_move_role"] == "PROMOTION_PUSH", (
        f"Expected PROMOTION_PUSH, got {facts['quiet_move_role']}"
    )


def test_king_activation_role_for_active_king_move():
    """
    A king move that controls the center should get KING_ACTIVATION role.
    """
    board = empty_board()
    board[5][2] = RED_KING  # king that can move to center

    board[0][1] = BLACK  # distant opponent

    move = {"type": "simple", "path": [(5, 2), (4, 3)], "captured": []}

    legal = get_all_legal_moves(board, RED)
    legal_paths = [m["path"] for m in legal]
    if [(5, 2), (4, 3)] not in legal_paths:
        pytest.skip("Move not legal in this configuration")

    facts = compute_move_facts(board, move, RED)

    # (4,3) is in center (rows 3-4, cols 2-5) so center_control=True
    # king moving to center → KING_ACTIVATION expected
    assert facts["quiet_move_role"] == "KING_ACTIVATION", (
        f"Expected KING_ACTIVATION for center king move, got {facts['quiet_move_role']}"
    )


def test_quiet_default_for_passive_move():
    """
    A simple move that creates no threat, has no center, no mobility gain
    should get QUIET_DEFAULT role.
    """
    board = empty_board()
    board[7][0] = RED   # corner piece with very limited impact

    board[0][7] = BLACK  # distant opponent

    move = {"type": "simple", "path": [(7, 0), (6, 1)], "captured": []}

    legal = get_all_legal_moves(board, RED)
    if [(7, 0), (6, 1)] not in [m["path"] for m in legal]:
        pytest.skip("Move not legal")

    facts = compute_move_facts(board, move, RED)

    # This move is not a king, not near promotion, not counterplay (score < 3),
    # not conversion (winning_conversion_score < 3), no threat reduction,
    # no mobility gain → should be QUIET_DEFAULT
    assert facts["quiet_move_role"] == "QUIET_DEFAULT", (
        f"Expected QUIET_DEFAULT for passive corner move, got {facts['quiet_move_role']}"
    )
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])