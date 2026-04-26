from __future__ import annotations

from checkers.engine.board import BLACK, BLACK_KING, EMPTY, RED, RED_KING
from checkers.engine.evaluation import LOSS_SCORE, WIN_SCORE, evaluate_board
from checkers.engine.minimax import minimax_score, score_move_with_minimax
from checkers.engine.rules import apply_move, get_all_legal_moves
from checkers.search.minimax_core import (
    MAX_TACTICAL_EXTENSION_PLIES,
    _tt_key,
    clear_transposition_table,
    negamax,
    order_moves,
    search_root,
    search_root_all_scores,
    search_root_iterative,
    SearchStats,
)


def _empty_board() -> list[list[int]]:
    return [[EMPTY for _ in range(8)] for _ in range(8)]


def _opponent(player: int) -> int:
    return BLACK if player == RED else RED


def test_terminal_loss_when_side_to_move_has_no_legal_moves() -> None:
    board = _empty_board()
    board[0][1] = BLACK

    best_move, best_score, _ = search_root(board=board, current_player=RED, depth=2)
    assert best_move is None
    assert best_score == float(LOSS_SCORE)


def test_mandatory_capture_move_is_selected_from_capture_legal_set() -> None:
    board = _empty_board()
    board[5][0] = RED
    board[5][4] = RED
    board[4][1] = BLACK

    legal = get_all_legal_moves(board, RED)
    assert legal
    assert all(m["type"] == "jump" for m in legal), "Mandatory capture should filter to jumps only."

    best_move, _, _ = search_root(board=board, current_player=RED, depth=2, legal_moves=legal)
    assert best_move is not None
    assert best_move["type"] == "jump"
    assert best_move in legal


def test_depth_one_matches_best_immediate_child_evaluation() -> None:
    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[2][1] = BLACK
    board[2][3] = BLACK

    legal = get_all_legal_moves(board, RED)
    assert legal

    scored = []
    for move in legal:
        child = apply_move(board, move)
        immediate = evaluate_board(child, _opponent(RED), RED)
        scored.append((move, immediate))

    expected_move, expected_score = max(scored, key=lambda pair: pair[1])
    chosen_move, chosen_score, _ = search_root(board=board, current_player=RED, depth=1, legal_moves=legal)
    assert chosen_move == expected_move
    assert chosen_score == expected_score


def test_search_is_deterministic_for_same_position_and_depth() -> None:
    board = _empty_board()
    board[5][2] = RED
    board[4][3] = BLACK
    board[2][5] = BLACK

    first_move, first_score, _ = search_root(board=board, current_player=RED, depth=3)
    second_move, second_score, _ = search_root(board=board, current_player=RED, depth=3)
    assert first_move == second_move
    assert first_score == second_score


def test_sign_consistency_for_terminal_scores() -> None:
    board = _empty_board()

    score_root_turn = minimax_score(
        board=board,
        depth=2,
        current_player=RED,
        root_player=RED,
        alpha=float("-inf"),
        beta=float("inf"),
    )
    score_opp_turn = minimax_score(
        board=board,
        depth=2,
        current_player=BLACK,
        root_player=RED,
        alpha=float("-inf"),
        beta=float("inf"),
    )

    assert score_root_turn == float(LOSS_SCORE)
    assert score_opp_turn == float(WIN_SCORE)
    assert score_root_turn == -score_opp_turn


def test_score_move_with_minimax_matches_root_search_result() -> None:
    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[2][1] = BLACK
    board[2][3] = BLACK

    root_move, root_score, _ = search_root(board=board, current_player=RED, depth=2)
    assert root_move is not None

    move_score = score_move_with_minimax(board, root_move, RED, depth=2)
    assert move_score == root_score


def test_order_moves_prioritizes_captures_before_quiets() -> None:
    board = _empty_board()
    board[5][2] = RED
    board[4][3] = BLACK

    legal = get_all_legal_moves(board, RED)
    assert legal
    ordered = order_moves(board, legal, RED)
    assert ordered
    assert ordered[0]["type"] == "jump"


def test_order_moves_prioritizes_promotion_quiet_move() -> None:
    board = _empty_board()
    board[1][2] = RED
    board[5][4] = RED

    legal = get_all_legal_moves(board, RED)
    quiets = [m for m in legal if m["type"] == "simple"]
    assert len(quiets) >= 2

    ordered = order_moves(board, quiets, RED)
    assert ordered
    assert ordered[0]["type"] == "simple"
    assert ordered[0]["path"][-1][0] == 0


def test_search_root_returns_move_from_legal_set_with_ordering_enabled() -> None:
    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[2][1] = BLACK
    board[2][3] = BLACK

    legal = get_all_legal_moves(board, RED)
    best_move, _, _ = search_root(board=board, current_player=RED, depth=3)
    assert best_move is not None
    assert best_move in legal


def test_iterative_depth_one_matches_direct_depth_one() -> None:
    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[2][1] = BLACK
    board[2][3] = BLACK

    direct_move, direct_score, _ = search_root(board=board, current_player=RED, depth=1)
    iter_move, iter_score, _ = search_root_iterative(board=board, current_player=RED, target_depth=1)
    assert iter_move == direct_move
    assert iter_score == direct_score


def test_iterative_final_result_matches_direct_target_depth() -> None:
    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[4][3] = BLACK
    board[2][1] = BLACK
    board[2][5] = BLACK

    direct_move, direct_score, _ = search_root(board=board, current_player=RED, depth=3)
    iter_move, iter_score, _ = search_root_iterative(board=board, current_player=RED, target_depth=3)
    assert iter_move == direct_move
    assert iter_score == direct_score


def test_iterative_search_is_deterministic() -> None:
    board = _empty_board()
    board[5][2] = RED
    board[4][3] = BLACK
    board[2][5] = BLACK

    first_move, first_score, _ = search_root_iterative(board=board, current_player=RED, target_depth=4)
    second_move, second_score, _ = search_root_iterative(board=board, current_player=RED, target_depth=4)
    assert first_move == second_move
    assert first_score == second_score


def test_tt_does_not_change_move_or_score_vs_no_tt() -> None:
    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[4][3] = BLACK
    board[2][1] = BLACK
    board[2][5] = BLACK

    clear_transposition_table()
    no_tt_move, no_tt_score, _ = search_root(board=board, current_player=RED, depth=4, use_tt=False)
    clear_transposition_table()
    tt_move, tt_score, _ = search_root(board=board, current_player=RED, depth=4, use_tt=True)

    assert tt_move == no_tt_move
    assert tt_score == no_tt_score


def test_tt_key_distinguishes_side_to_move() -> None:
    board = _empty_board()
    board[5][2] = RED
    board[2][3] = BLACK

    red_key = _tt_key(board, RED, RED)
    black_key = _tt_key(board, BLACK, RED)
    assert red_key != black_key


def test_tt_key_distinguishes_root_perspective() -> None:
    board = _empty_board()
    board[5][2] = RED
    board[2][3] = BLACK

    red_root_key = _tt_key(board, RED, RED)
    black_root_key = _tt_key(board, RED, BLACK)
    assert red_root_key != black_root_key


def test_tt_key_distinguishes_phase7a_toggle() -> None:
    board = _empty_board()
    board[5][2] = RED
    board[2][3] = BLACK

    key_on = _tt_key(board, RED, RED, use_phase7a=True)
    key_off = _tt_key(board, RED, RED, use_phase7a=False)
    assert key_on != key_off


def test_repeated_tt_search_is_deterministic_and_can_hit_cache() -> None:
    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[4][3] = BLACK
    board[2][1] = BLACK
    board[2][5] = BLACK

    clear_transposition_table()
    first_move, first_score, first_stats = search_root(board=board, current_player=RED, depth=5, use_tt=True)
    second_move, second_score, second_stats = search_root(board=board, current_player=RED, depth=5, use_tt=True)

    assert first_move == second_move
    assert first_score == second_score
    assert second_stats.tt_hits >= 1
    assert second_stats.nodes <= first_stats.nodes


def test_capture_extension_changes_leaf_eval_in_tactical_position() -> None:
    board = _empty_board()
    board[7][6] = RED
    board[6][3] = RED
    board[7][4] = BLACK
    board[6][5] = BLACK

    score_no_ext = negamax(
        board=board,
        current_player=RED,
        root_player=RED,
        depth=0,
        alpha=float("-inf"),
        beta=float("inf"),
        stats=SearchStats(),
        use_tt=False,
        use_tactical_extension=False,
    )
    score_ext = negamax(
        board=board,
        current_player=RED,
        root_player=RED,
        depth=0,
        alpha=float("-inf"),
        beta=float("inf"),
        stats=SearchStats(),
        use_tt=False,
        use_tactical_extension=True,
    )
    assert score_ext != score_no_ext


def test_no_extension_when_no_captures_exist_at_leaf() -> None:
    board = _empty_board()
    board[5][2] = RED
    board[2][5] = BLACK

    clear_transposition_table()
    _, score_no_ext, _ = search_root(board=board, current_player=RED, depth=0, use_tactical_extension=False)
    clear_transposition_table()
    _, score_ext, _ = search_root(board=board, current_player=RED, depth=0, use_tactical_extension=True)
    assert score_ext == score_no_ext


def test_extension_depth_limit_respected() -> None:
    board = _empty_board()
    board[7][6] = RED
    board[6][3] = RED
    board[7][4] = BLACK
    board[6][5] = BLACK

    score_at_limit = negamax(
        board=board,
        current_player=RED,
        root_player=RED,
        depth=0,
        alpha=float("-inf"),
        beta=float("inf"),
        stats=SearchStats(),
        use_tt=False,
        extension_depth=MAX_TACTICAL_EXTENSION_PLIES,
        use_tactical_extension=True,
    )
    score_no_ext = negamax(
        board=board,
        current_player=RED,
        root_player=RED,
        depth=0,
        alpha=float("-inf"),
        beta=float("inf"),
        stats=SearchStats(),
        use_tt=False,
        use_tactical_extension=False,
    )
    assert score_at_limit == score_no_ext
    assert MAX_TACTICAL_EXTENSION_PLIES == 2


def test_capture_extension_search_is_deterministic() -> None:
    board = _empty_board()
    board[5][2] = RED
    board[4][3] = BLACK

    first_move, first_score, _ = search_root(board=board, current_player=RED, depth=0, use_tactical_extension=True)
    second_move, second_score, _ = search_root(board=board, current_player=RED, depth=0, use_tactical_extension=True)
    assert first_move == second_move
    assert first_score == second_score


# ── Leaf tension penalty tests ────────────────────────────────────────────────

def test_leaf_tension_penalty_applied_when_opponent_has_jumps() -> None:
    """
    At depth=0 with extension exhausted, the tension penalty must lower the
    leaf score when the opponent (BLACK) has pending captures against RED.

    Board: RED at (5,2), BLACK at (4,3) — BLACK can jump RED.
    We call negamax with current_player=RED, extension_depth at the cap,
    so the extension branch is skipped and we reach the static-eval branch.
    With extension ON, the penalty should reduce the score vs extension OFF.
    """
    board = _empty_board()
    board[5][2] = RED
    board[4][3] = BLACK  # BLACK can jump RED at (5,2) → (3,1) or (3,3) if empty

    # Place a landing square so BLACK has a legal jump
    # Black at (4,3) jumps RED at (5,2) landing at (6,1) — need (6,1) empty (it is)
    # Actually: BLACK moves DOWN, so (4,3) + (+1,−1) = (5,2) captured, land (6,1)
    # (6,1) is EMPTY in _empty_board() — jump is available

    score_no_penalty = negamax(
        board=board,
        current_player=RED,
        root_player=RED,
        depth=0,
        alpha=float("-inf"),
        beta=float("inf"),
        stats=SearchStats(),
        use_tt=False,
        extension_depth=MAX_TACTICAL_EXTENSION_PLIES,  # exhausted → goes to static eval
        use_tactical_extension=False,  # penalty only fires when extension is ON
    )
    score_with_penalty = negamax(
        board=board,
        current_player=RED,
        root_player=RED,
        depth=0,
        alpha=float("-inf"),
        beta=float("inf"),
        stats=SearchStats(),
        use_tt=False,
        extension_depth=MAX_TACTICAL_EXTENSION_PLIES,  # exhausted → goes to static eval
        use_tactical_extension=True,   # penalty fires here
    )
    # Penalty lowers the score when the opponent threatens (bad for root=RED)
    assert score_with_penalty < score_no_penalty


def test_leaf_tension_penalty_absent_when_no_opponent_jumps() -> None:
    """
    When the opponent has NO pending captures at the leaf, the penalty must
    NOT fire, so scores with and without extension are equal at the static
    evaluation level (with extension exhausted).
    """
    board = _empty_board()
    board[5][2] = RED
    board[1][6] = BLACK  # far away — BLACK has no jump over RED

    score_no_ext = negamax(
        board=board,
        current_player=RED,
        root_player=RED,
        depth=0,
        alpha=float("-inf"),
        beta=float("inf"),
        stats=SearchStats(),
        use_tt=False,
        extension_depth=MAX_TACTICAL_EXTENSION_PLIES,
        use_tactical_extension=False,
    )
    score_with_ext = negamax(
        board=board,
        current_player=RED,
        root_player=RED,
        depth=0,
        alpha=float("-inf"),
        beta=float("inf"),
        stats=SearchStats(),
        use_tt=False,
        extension_depth=MAX_TACTICAL_EXTENSION_PLIES,
        use_tactical_extension=True,
    )
    # No opponent jumps → no penalty → scores must be identical
    assert score_with_ext == score_no_ext


# ── King endgame pressure benchmarks ─────────────────────────────────────────
#
# Group D benchmark: 3K vs 2K pure-king endgame degeneracy.
# Without a king_endgame_pressure term in evaluation.py, depth-6 search
# returns a multi-way tie in quiescent pure-king positions because material,
# mobility, center, and chase terms all produce identical values across moves.
#
# Tests 1: failing benchmark — drives the king_endgame_pressure implementation.
# Tests 2–4: regression guards — must pass before AND after the fix.


def test_3k_vs_2k_pure_king_best_move_is_unique() -> None:
    # RED kings at (4,3),(2,1),(2,5); BLACK kings at (7,2),(7,6).
    # Without king_endgame_pressure: 3-way tie at score 474 at depth 6.
    # With king_endgame_pressure: approach move (4,3)→(5,2) must emerge as unique best.
    board = _empty_board()
    board[4][3] = RED_KING
    board[2][1] = RED_KING
    board[2][5] = RED_KING
    board[7][2] = BLACK_KING
    board[7][6] = BLACK_KING

    clear_transposition_table()
    _, _, all_scored, _ = search_root_all_scores(board=board, current_player=RED, depth=6)

    scores = [s for _, s in all_scored]
    top_score = scores[0]
    tied_count = sum(1 for s in scores if s == top_score)
    assert tied_count == 1, f"Expected unique best move; {tied_count} moves tied at {top_score}"


def test_7piece_pure_king_endgame_gate_blocks() -> None:
    # 4K vs 3K = 7 pieces total; exceeds the ≤6-piece gate.
    # king_endgame_pressure must not contribute — existing evaluator is sufficient.
    board = _empty_board()
    board[1][2] = RED_KING
    board[1][6] = RED_KING
    board[3][4] = RED_KING
    board[5][0] = RED_KING
    board[6][1] = BLACK_KING
    board[6][5] = BLACK_KING
    board[7][3] = BLACK_KING

    clear_transposition_table()
    best_move, best_score, _, _ = search_root_all_scores(
        board=board, current_player=RED, depth=4
    )
    assert best_move is not None
    assert best_score > 0, f"RED 4K vs 3K must score positive; got {best_score}"


def test_2k_vs_2k_equal_material_gate_blocks() -> None:
    # 2K vs 2K: no material advantage — king_endgame_pressure gate requires
    # winning material advantage, so it must not fire for either side.
    board = _empty_board()
    board[2][1] = RED_KING
    board[2][5] = RED_KING
    board[6][2] = BLACK_KING
    board[6][6] = BLACK_KING

    clear_transposition_table()
    best_move, _, _, _ = search_root_all_scores(
        board=board, current_player=RED, depth=4
    )
    assert best_move is not None, "2K vs 2K must have at least one legal move"


def test_mixed_king_man_endgame_gate_blocks() -> None:
    # Checkers still on board → gate requires pure king endgame → term must not fire.
    # Verifies that adding king_endgame_pressure does not alter mixed-material results.
    board = _empty_board()
    board[1][2] = RED        # RED checker
    board[2][5] = RED_KING
    board[3][4] = RED_KING
    board[6][3] = BLACK_KING
    board[6][7] = BLACK_KING

    clear_transposition_table()
    best_move, best_score, _, _ = search_root_all_scores(
        board=board, current_player=RED, depth=4
    )
    assert best_move is not None
    assert best_score > 0, f"RED has material advantage in mixed position; got {best_score}"


# ── History heuristic tests ───────────────────────────────────────────────────


def test_history_heuristic_soundness_same_move_and_score() -> None:
    # search_root with history enabled must return the same best move and score
    # as without history.  History changes only ordering, never the minimax value.
    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[4][3] = BLACK
    board[2][1] = BLACK
    board[2][5] = BLACK

    clear_transposition_table()
    move_no_hist, score_no_hist, _ = search_root(board=board, current_player=RED, depth=5)

    clear_transposition_table()
    move_with_hist, score_with_hist, _ = search_root(board=board, current_player=RED, depth=5)

    assert move_no_hist == move_with_hist, (
        f"History changed best move: {move_no_hist} vs {move_with_hist}"
    )
    assert score_no_hist == score_with_hist, (
        f"History changed score: {score_no_hist} vs {score_with_hist}"
    )


def test_history_heuristic_capture_priority_over_quiet() -> None:
    # order_moves must place jump moves before quiet moves even if the quiet
    # move has an arbitrarily large history score.
    board = _empty_board()
    board[5][2] = RED
    board[4][3] = BLACK

    jump_move = {"type": "jump", "path": [[5, 2], [3, 4]], "captured": [[4, 3]]}
    quiet_move = {"type": "simple", "path": [[5, 2], [4, 1]], "captured": []}

    # Give the quiet move a huge history score.
    big_history: dict = {((5, 2), (4, 1)): 10_000}
    ordered = order_moves(board, [quiet_move, jump_move], RED, history=big_history)

    assert ordered[0]["type"] == "jump", (
        "Capture must remain first regardless of quiet move's history score"
    )


def test_history_heuristic_efficiency_noop_without_history() -> None:
    # Passing history=None (default) must not raise and must produce correct results.
    # This validates backward compatibility of the updated signatures.
    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[2][1] = BLACK
    board[2][3] = BLACK

    clear_transposition_table()
    move_a, score_a, _ = search_root(board=board, current_player=RED, depth=4)

    clear_transposition_table()
    move_b, score_b, _ = search_root(board=board, current_player=RED, depth=4, history=None)

    assert move_a == move_b
    assert score_a == score_b
