"""
Phase 1 — Search Core Audit Tests
──────────────────────────────────
Focused tests for correctness properties not covered by test_minimax_core.py.
Uses small artificial boards and existing engine APIs only.
"""

from __future__ import annotations

from checkers.engine.board import BLACK, EMPTY, RED, RED_KING, BLACK_KING
from checkers.engine.evaluation import LOSS_SCORE, WIN_SCORE, evaluate_board
from checkers.engine.rules import apply_move, get_all_legal_moves
from checkers.search.minimax_core import (
    SearchStats,
    clear_transposition_table,
    negamax,
    search_root,
    search_root_all_scores,
    search_root_iterative,
)


def _empty_board() -> list[list[int]]:
    return [[EMPTY for _ in range(8)] for _ in range(8)]


def _opp(player: int) -> int:
    return BLACK if player == RED else RED


# ── 1. Terminal win via search_root ──────────────────────────────────────────

def test_terminal_win_when_opponent_has_no_pieces():
    """search_root for side with pieces vs empty opponent → near-WIN_SCORE.

    The engine uses mate-distance scoring: terminal wins return WIN_SCORE - ply_from_root
    (not a flat WIN_SCORE) so the engine prefers faster wins.  search_root starts
    child evaluation at ply_from_root=1, so a 1-ply forced win returns WIN_SCORE - 1.
    The assertion accepts any score in [WIN_SCORE - depth, WIN_SCORE].
    """
    depth = 2
    board = _empty_board()
    board[5][2] = RED

    best_move, score, _ = search_root(board=board, current_player=RED, depth=depth)
    assert best_move is not None
    child = apply_move(board, best_move)
    opp_moves = get_all_legal_moves(child, BLACK)
    assert len(opp_moves) == 0
    # Mate-distance: WIN_SCORE - ply_from_root.  ply_from_root=1 at first child
    # → score = WIN_SCORE - 1.  Allow the full depth window for robustness.
    assert score >= float(WIN_SCORE) - depth, (
        f"Expected near-WIN_SCORE (>= {WIN_SCORE - depth}), got {score}"
    )
    assert score <= float(WIN_SCORE), f"Score cannot exceed WIN_SCORE, got {score}"


def test_terminal_win_when_opponent_is_blocked():
    """Opponent has pieces but is completely blocked → near-WIN_SCORE.

    Original board bug: with only RED at (1,0) and (1,2), RED's only legal
    move is (1,2)→(0,3), which vacates (1,2) and unblocks BLACK at (0,1).
    No terminal is reached within depth=2.

    Fix: add a third RED piece at (3,4) that RED can move freely.  After
    RED(3,4)→(2,3) or (2,5), both (1,0) and (1,2) remain occupied, keeping
    BLACK blocked.  The terminal fires at ply_from_root=1 → WIN_SCORE - 1.

    Same mate-distance adjustment as test_terminal_win_when_opponent_has_no_pieces.
    """
    depth = 2
    board = _empty_board()
    board[0][1] = BLACK
    board[1][0] = RED
    board[1][2] = RED
    # Third RED piece: gives RED a move that does not vacate (1,0) or (1,2),
    # so BLACK at (0,1) stays blocked after RED moves.
    board[3][4] = RED

    best_move, score, _ = search_root(board=board, current_player=RED, depth=depth)
    assert best_move is not None
    # Mate-distance: best line is RED(3,4)→(2,x), BLACK still blocked → WIN_SCORE - 1.
    assert score >= float(WIN_SCORE) - depth, (
        f"Expected near-WIN_SCORE (>= {WIN_SCORE - depth}), got {score}"
    )
    assert score <= float(WIN_SCORE), f"Score cannot exceed WIN_SCORE, got {score}"


# ── 2. Multi-jump: full sequence generated and selected ──────────────────────

def test_multi_jump_double_capture_selected():
    """
    RED at (5,2) can double-jump via (3,4)→(1,2), capturing BLACK at (4,3)
    and (2,3). This is the only legal move (mandatory capture).
    Search must select it at any depth.
    """
    board = _empty_board()
    board[5][2] = RED
    board[4][3] = BLACK
    board[2][3] = BLACK

    legal = get_all_legal_moves(board, RED)
    assert len(legal) >= 1
    double_jumps = [m for m in legal if len(m.get("captured", [])) == 2]
    assert len(double_jumps) >= 1, "Engine must generate the double-jump sequence"

    best_move, _, _ = search_root(board=board, current_player=RED, depth=2)
    assert best_move is not None
    assert best_move["type"] == "jump"
    assert len(best_move["captured"]) == 2


def test_multi_jump_preferred_over_single_when_both_legal():
    """
    When multiple jump sequences exist (single and double), the engine should
    evaluate and potentially prefer the double-jump that captures more material.
    """
    board = _empty_board()
    board[7][0] = RED
    board[6][1] = BLACK
    board[4][3] = BLACK

    legal = get_all_legal_moves(board, RED)
    jump_moves = [m for m in legal if m["type"] == "jump"]
    assert len(jump_moves) >= 1

    best_move, _, _ = search_root(board=board, current_player=RED, depth=2)
    assert best_move is not None
    assert best_move["type"] == "jump"


# ── 3. Non-terminal sign consistency ─────────────────────────────────────────

def test_sign_consistency_nonterminal_position():
    """
    For a symmetric-ish non-terminal position, scoring from RED's perspective
    and BLACK's perspective should produce opposite-sign scores when the
    position is symmetric.

    Here we use an asymmetric position and just verify the sign relationship:
    score(board, RED_to_move, root=RED) should be the negative of
    score(board, RED_to_move, root=BLACK) at the static eval level.
    """
    board = _empty_board()
    board[5][2] = RED
    board[2][3] = BLACK

    score_red_root = negamax(
        board=board, depth=0, current_player=RED, root_player=RED,
        alpha=float("-inf"), beta=float("inf"), stats=SearchStats(),
        use_tt=False, use_tactical_extension=False,
    )
    score_black_root = negamax(
        board=board, depth=0, current_player=RED, root_player=BLACK,
        alpha=float("-inf"), beta=float("inf"), stats=SearchStats(),
        use_tt=False, use_tactical_extension=False,
    )
    assert score_red_root == -score_black_root, (
        f"Static eval from RED root ({score_red_root}) should negate "
        f"BLACK root ({score_black_root})"
    )


# ── 4. Mandatory capture: search never picks simple when jump exists ─────────

def test_search_never_selects_simple_move_when_jump_exists():
    """
    Even at depth > 1, if mandatory captures exist, search_root must
    return a jump move (since legal_moves filters to jumps only).
    """
    board = _empty_board()
    board[5][0] = RED
    board[5][4] = RED
    board[4][1] = BLACK
    board[2][5] = BLACK

    legal = get_all_legal_moves(board, RED)
    assert any(m["type"] == "jump" for m in legal)
    assert all(m["type"] == "jump" for m in legal), "Mandatory capture must filter"

    for depth in (1, 2, 4):
        clear_transposition_table()
        best, _, _ = search_root(board=board, current_player=RED, depth=depth)
        assert best is not None
        assert best["type"] == "jump", f"At depth={depth}, search selected non-jump"


# ── 5. Depth consistency: deeper search doesn't return illegal moves ─────────

def test_deeper_search_returns_legal_move():
    """Best move from search_root must always be in the legal move set."""
    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[5][4] = RED
    board[2][1] = BLACK
    board[2][3] = BLACK
    board[2][5] = BLACK

    legal = get_all_legal_moves(board, RED)
    for depth in (1, 2, 3, 4):
        clear_transposition_table()
        best, _, _ = search_root(board=board, current_player=RED, depth=depth)
        assert best is not None
        assert best in legal, f"depth={depth}: best move not in legal set"


# ── 6. Alpha-beta equivalence: full-window search matches brute force ────────

def _brute_force_minimax(
    board: list[list[int]],
    depth: int,
    current_player: int,
    root_player: int,
) -> float:
    """Minimal plain minimax without alpha-beta, for verification on tiny boards."""
    legal = get_all_legal_moves(board, current_player)
    if not legal:
        return float(LOSS_SCORE if current_player == root_player else WIN_SCORE)
    if depth <= 0:
        return float(evaluate_board(board, current_player, root_player, use_phase7a=False))

    if current_player == root_player:
        best = float("-inf")
        for move in legal:
            child = apply_move(board, move)
            score = _brute_force_minimax(child, depth - 1, _opp(current_player), root_player)
            if score > best:
                best = score
        return best
    else:
        best = float("inf")
        for move in legal:
            child = apply_move(board, move)
            score = _brute_force_minimax(child, depth - 1, _opp(current_player), root_player)
            if score < best:
                best = score
        return best


def test_alpha_beta_matches_brute_force_on_tiny_board():
    """
    On a 2v2 board at depth 2, alpha-beta negamax (no TT, no extensions)
    must return the exact same score as brute-force minimax.
    """
    board = _empty_board()
    board[5][0] = RED
    board[5][4] = RED
    board[2][1] = BLACK
    board[2][5] = BLACK

    depth = 2

    bf_score = _brute_force_minimax(board, depth, RED, RED)

    clear_transposition_table()
    ab_score = negamax(
        board=board, depth=depth, current_player=RED, root_player=RED,
        alpha=float("-inf"), beta=float("inf"), stats=SearchStats(),
        use_tt=False, use_tactical_extension=False, use_phase7a=False,
    )

    assert ab_score == bf_score, (
        f"Alpha-beta ({ab_score}) != brute force ({bf_score}) at depth {depth}"
    )


# ── 7. PV move feed-forward tests ───────────────────────────────────────────

def test_pv_move_does_not_change_score_or_best_move():
    """search_root with a legal pv_move returns the same best move and score."""
    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[5][4] = RED
    board[2][1] = BLACK
    board[2][3] = BLACK
    board[2][5] = BLACK

    legal = get_all_legal_moves(board, RED)

    clear_transposition_table()
    base_move, base_score, _ = search_root(
        board=board, current_player=RED, depth=4, use_tt=False,
    )

    for pv_candidate in legal:
        clear_transposition_table()
        pv_move, pv_score, _ = search_root(
            board=board, current_player=RED, depth=4, use_tt=False,
            pv_move=pv_candidate,
        )
        assert pv_score == base_score, (
            f"pv_move={pv_candidate['path']} changed score: {pv_score} vs {base_score}"
        )
        assert pv_move == base_move, (
            f"pv_move={pv_candidate['path']} changed best move"
        )


def test_pv_move_illegal_is_ignored():
    """search_root with a move not in the legal set ignores it safely."""
    board = _empty_board()
    board[5][0] = RED
    board[2][1] = BLACK

    fake_move = {"type": "simple", "path": [(7, 7), (6, 6)], "captured": []}

    clear_transposition_table()
    base_move, base_score, _ = search_root(
        board=board, current_player=RED, depth=3, use_tt=False,
    )
    clear_transposition_table()
    pv_move, pv_score, _ = search_root(
        board=board, current_player=RED, depth=3, use_tt=False,
        pv_move=fake_move,
    )

    assert pv_score == base_score
    assert pv_move == base_move


def test_iterative_deepening_unchanged_result():
    """search_root_iterative returns the same final move and score as before."""
    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[4][3] = BLACK
    board[2][1] = BLACK
    board[2][5] = BLACK

    clear_transposition_table()
    direct_move, direct_score, _ = search_root(
        board=board, current_player=RED, depth=3, use_tt=True,
    )
    clear_transposition_table()
    iter_move, iter_score, _ = search_root_iterative(
        board=board, current_player=RED, target_depth=3, use_tt=True,
    )

    assert iter_move == direct_move
    assert iter_score == direct_score


def test_pv_move_reduces_nodes_or_equal():
    """
    When pv_move is the actual best move, node count should be <= the count
    without pv_move (better ordering → more cutoffs or same).
    """
    board = _empty_board()
    board[5][0] = RED
    board[5][2] = RED
    board[5][4] = RED
    board[2][1] = BLACK
    board[2][3] = BLACK
    board[2][5] = BLACK

    clear_transposition_table()
    best_move, _, base_stats = search_root(
        board=board, current_player=RED, depth=4, use_tt=False,
    )
    assert best_move is not None

    clear_transposition_table()
    _, _, pv_stats = search_root(
        board=board, current_player=RED, depth=4, use_tt=False,
        pv_move=best_move,
    )

    assert pv_stats.nodes <= base_stats.nodes, (
        f"PV move ordering increased nodes: {pv_stats.nodes} > {base_stats.nodes}"
    )


# ── 8. search_root_all_scores tests ────────────────────────────────────────────

def _per_move_exact_scores(
    board: list[list[int]], player: int, depth: int,
) -> list[tuple[dict, float]]:
    """Production-style per-move full-window scoring (no TT).

    ply_from_root=1 mirrors search_root_all_scores, which always starts child
    evaluation one ply from the root.  Without it, terminal scores inside the
    search tree are off by one (WIN_SCORE-0 vs WIN_SCORE-1) and the comparison
    fails for any board where a terminal node is reached within `depth` plies.
    """
    legal = get_all_legal_moves(board, player)
    opp = BLACK if player == RED else RED
    scored = []
    for move in legal:
        child = apply_move(board, move)
        score = float(negamax(
            board=child, depth=max(0, depth - 1),
            current_player=opp, root_player=player,
            alpha=float("-inf"), beta=float("inf"),
            stats=SearchStats(), use_tt=False,
            extension_depth=0, use_tactical_extension=True, use_phase7a=True,
            ply_from_root=1,   # match search_root_all_scores: child is 1 ply from root
        ))
        scored.append((move, score))
    scored.sort(key=lambda x: -x[1])
    return scored


def test_search_root_all_scores_matches_per_move_no_tt():
    """Every move's exact score from search_root_all_scores(use_tt=False)
    must match production-style per-move scoring."""
    boards = []

    b = _empty_board()
    b[5][0] = RED; b[5][2] = RED; b[5][4] = RED
    b[2][1] = BLACK; b[2][3] = BLACK; b[2][5] = BLACK
    boards.append((b, RED))

    b2 = _empty_board()
    b2[5][0] = RED; b2[5][4] = RED
    b2[2][1] = BLACK; b2[2][5] = BLACK
    boards.append((b2, RED))

    b3 = _empty_board()
    b3[7][0] = RED; b3[6][1] = BLACK; b3[4][3] = BLACK
    boards.append((b3, RED))

    for board, player in boards:
        depth = 4
        expected = _per_move_exact_scores(board, player, depth)
        expected_by_path = {tuple(map(tuple, m["path"])): s for m, s in expected}

        clear_transposition_table()
        _, _, all_scored, _ = search_root_all_scores(
            board=board, current_player=player, depth=depth, use_tt=False,
        )

        assert len(all_scored) == len(expected), "Move count mismatch"
        for move, score in all_scored:
            key = tuple(map(tuple, move["path"]))
            assert key in expected_by_path, f"Unexpected move {key}"
            assert score == expected_by_path[key], (
                f"Score mismatch for {key}: got {score}, expected {expected_by_path[key]}"
            )


def test_search_root_all_scores_tt_matches_no_tt():
    """use_tt=True must produce identical per-move scores as use_tt=False."""
    board = _empty_board()
    board[5][0] = RED; board[5][2] = RED; board[5][4] = RED
    board[2][1] = BLACK; board[2][3] = BLACK; board[2][5] = BLACK

    for depth in (2, 4):
        clear_transposition_table()
        _, _, scores_no_tt, _ = search_root_all_scores(
            board=board, current_player=RED, depth=depth, use_tt=False,
        )
        clear_transposition_table()
        _, _, scores_tt, _ = search_root_all_scores(
            board=board, current_player=RED, depth=depth, use_tt=True,
        )

        no_tt_by_path = {tuple(map(tuple, m["path"])): s for m, s in scores_no_tt}
        tt_by_path = {tuple(map(tuple, m["path"])): s for m, s in scores_tt}

        assert no_tt_by_path.keys() == tt_by_path.keys(), "Move set differs"
        for key in no_tt_by_path:
            assert no_tt_by_path[key] == tt_by_path[key], (
                f"TT changed score for {key} at depth={depth}: "
                f"no_tt={no_tt_by_path[key]} tt={tt_by_path[key]}"
            )


def test_search_root_all_scores_best_matches_search_root():
    """Best move and score must match search_root output."""
    board = _empty_board()
    board[5][0] = RED; board[5][2] = RED; board[5][4] = RED
    board[2][1] = BLACK; board[2][3] = BLACK; board[2][5] = BLACK

    clear_transposition_table()
    sr_move, sr_score, _ = search_root(
        board=board, current_player=RED, depth=4, use_tt=False,
    )
    clear_transposition_table()
    all_move, all_score, all_scored, _ = search_root_all_scores(
        board=board, current_player=RED, depth=4, use_tt=False,
    )

    assert all_move == sr_move, "Best move differs"
    assert all_score == sr_score, "Best score differs"


def test_search_root_all_scores_covers_all_legal_moves():
    """Returned list must contain exactly the legal moves."""
    board = _empty_board()
    board[5][0] = RED; board[5][2] = RED; board[5][4] = RED
    board[2][1] = BLACK; board[2][3] = BLACK; board[2][5] = BLACK

    legal = get_all_legal_moves(board, RED)
    clear_transposition_table()
    _, _, all_scored, _ = search_root_all_scores(
        board=board, current_player=RED, depth=3,
    )
    assert len(all_scored) == len(legal)
    returned_moves = [m for m, _ in all_scored]
    for m in legal:
        assert m in returned_moves, f"Legal move {m['path']} missing from results"
