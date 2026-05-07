# checkers/tests/test_proposal_agent.py
#
# Unit tests for proposal_agent pre-LLM reordering, post-LLM cleanup,
# and MINIMAX_RANK marker formatting.
# No LLM calls — all tests use engine state and synthetic score_by_path.
#
# Run with: venv/bin/pytest checkers/tests/test_proposal_agent.py -v

from __future__ import annotations

from checkers.engine.board import RED, BLACK
from checkers.engine.rules import get_all_legal_moves
from checkers.oldfiles.proposal_agent import (
    _build_legal_moves_with_facts,
    _postprocess_llm_selection,
    _role_pin_moves,
)


# ── Board helpers ─────────────────────────────────────────────────────────────

def _empty_board() -> list[list[int]]:
    return [[0] * 8 for _ in range(8)]


def _standard_start() -> list[list[int]]:
    """Standard starting position (12 RED rows 5-7, 12 BLACK rows 0-2)."""
    board = _empty_board()
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                board[r][c] = RED
    for r in range(0, 3):
        for c in range(8):
            if (r + c) % 2 == 1:
                board[r][c] = BLACK
    return board


def _path_key(move: dict) -> tuple:
    return tuple(tuple(sq) for sq in move.get("path", []))


# ── Test 1: mm_pin uses score_by_path (not legacy scorer) ─────────────────────

def test_mm_pin_uses_score_by_path_not_legacy_scorer():
    """
    mm_pin must reorder using score_by_path, not score_move_with_minimax.

    Standard start gives 7 RED legal moves (n_slots=5).  The last move in the
    natural sort order (positions 5 or 6) is given the highest score via
    score_by_path.  After _build_legal_moves_with_facts, that move must be
    inside the window (position < n_slots) and mm_pinned_pres_idx must be set.
    """
    board = _standard_start()

    # Step 1: get natural order without scores (mm_pin guard skips)
    n, _, moves_no_score, _, mm_idx_no_score = _build_legal_moves_with_facts(
        board, RED, score_by_path=None
    )
    n_slots = min(5, n)

    assert n > n_slots, (
        f"standard start must have >5 RED legal moves for mm_pin to matter; got {n}"
    )
    assert mm_idx_no_score is None, (
        "score_by_path=None must prevent mm_pin from firing"
    )

    # Step 2: pick the last move (definitely outside the window)
    target_move = moves_no_score[-1][0]
    pk = _path_key(target_move)
    assert moves_no_score[-1][0]["path"] == target_move["path"]

    # Step 3: assign it the highest score
    score_by_path = {pk: 10000.0}
    n2, _, moves_with_score, _, mm_idx_with_score = _build_legal_moves_with_facts(
        board, RED, score_by_path=score_by_path
    )

    path_list = [_path_key(m) for m, _ in moves_with_score]
    target_pos = path_list.index(pk)

    assert n2 == n, "move count must not change after mm_pin"
    assert target_pos < n_slots, (
        f"mm_pin must bring score_by_path best into window: "
        f"target at pos {target_pos}, n_slots={n_slots}"
    )
    assert mm_idx_with_score is not None, "mm_pinned_pres_idx must be set when pin fires"
    assert mm_idx_with_score == n_slots - 1, (
        f"pin inserts at n_slots-1={n_slots - 1}, got {mm_idx_with_score}"
    )


# ── Test 2: mm_pin skips without score_by_path ────────────────────────────────

def test_mm_pin_skips_when_no_score_by_path():
    """
    When score_by_path is absent (None or empty), mm_pin must not fire.
    mm_pinned_pres_idx must be None regardless of board complexity.
    """
    board = _standard_start()

    for sbp in (None, {}):
        _, _, _, _, mm_idx = _build_legal_moves_with_facts(
            board, RED, score_by_path=sbp
        )
        assert mm_idx is None, (
            f"mm_pin must not fire with score_by_path={sbp!r}; "
            f"got mm_pinned_pres_idx={mm_idx}"
        )


# ── Test 3: postprocess never injects unselected moves ────────────────────────

def test_postprocess_never_injects_unselected_moves():
    """
    _postprocess_llm_selection must only validate/dedup/trim.
    It must NOT pad to quota when fewer indices are provided.
    Output can only contain indices the caller passed in.
    """
    # Two valid indices in [0..9] — below the quota of 5 for n_moves=10
    result = _postprocess_llm_selection([0, 2], n_moves=10)
    assert set(result).issubset({0, 2}), (
        f"output must be a subset of input; got {result}"
    )
    assert result == [0, 2], f"should preserve LLM selection unchanged; got {result}"

    # Single index well below quota — must NOT pad
    result_one = _postprocess_llm_selection([3], n_moves=8)
    assert result_one == [3], (
        f"postprocess must not pad to quota; got {result_one}"
    )
    assert len(result_one) == 1

    # Empty input — must return empty, not pick anything
    result_empty = _postprocess_llm_selection([], n_moves=6)
    assert result_empty == [], f"empty input must yield empty output; got {result_empty}"


# ── Test 4: postprocess removes duplicates, drops OOB, trims, preserves order ─

def test_postprocess_dedup_invalid_and_trim():
    """
    _postprocess_llm_selection correctness for dedup, OOB drop, trim, and
    LLM-order preservation.
    """
    # Duplicates removed, first occurrence wins
    result = _postprocess_llm_selection([0, 0, 1, 2], n_moves=5)
    assert result == [0, 1, 2], f"duplicates must be removed; got {result}"

    # OOB index 9 dropped for n_moves=5
    result = _postprocess_llm_selection([0, 9, 1, 3], n_moves=5)
    assert 9 not in result, "OOB index 9 must be dropped"
    assert all(0 <= i < 5 for i in result), f"all kept indices must be in range; got {result}"

    # Negative index dropped
    result = _postprocess_llm_selection([-1, 0, 1], n_moves=5)
    assert -1 not in result, "negative index must be dropped"

    # Trim to quota (5) when more are given
    result = _postprocess_llm_selection([0, 1, 2, 3, 4, 5], n_moves=10)
    assert len(result) <= 5, f"output must not exceed quota of 5; got {result}"
    assert result == [0, 1, 2, 3, 4], f"trim must keep first 5 LLM choices; got {result}"

    # LLM order preserved
    result = _postprocess_llm_selection([3, 1, 0], n_moves=5)
    assert result == [3, 1, 0], f"LLM selection order must be preserved; got {result}"

    # All-invalid input
    result = _postprocess_llm_selection([99, 100, -5], n_moves=5)
    assert result == [], f"all-invalid input must yield empty; got {result}"


# ── Test 5: role_pin and mm_pin preserve the exact move set ───────────────────

def test_pin_mechanisms_preserve_exact_move_set():
    """
    Both _role_pin_moves and the mm_pin block only reorder existing moves.
    The set of (path_key) must be identical before and after, with and without
    score_by_path.  Move count must be unchanged.
    """
    board = _standard_start()

    # Run without scores (role_pin only)
    n_base, _, moves_base, _, _ = _build_legal_moves_with_facts(
        board, RED, score_by_path=None
    )
    paths_base = frozenset(_path_key(m) for m, _ in moves_base)

    # Run with scores that promote the last move (role_pin + mm_pin)
    last_pk = _path_key(moves_base[-1][0])
    score_by_path = {last_pk: 9999.0}
    n_pin, _, moves_pin, _, mm_idx = _build_legal_moves_with_facts(
        board, RED, score_by_path=score_by_path
    )
    paths_pin = frozenset(_path_key(m) for m, _ in moves_pin)

    assert n_pin == n_base, f"move count must be unchanged: {n_base} → {n_pin}"
    assert paths_pin == paths_base, (
        "pin mechanisms must not create or remove moves; "
        f"added={paths_pin - paths_base}, removed={paths_base - paths_pin}"
    )
    # Verify order DID change (pin fired)
    assert mm_idx is not None, "mm_pin must have fired with a high score_by_path entry"
    pinned_path = _path_key(moves_pin[mm_idx][0])
    assert pinned_path == last_pk, (
        f"pinned move at mm_idx={mm_idx} must be the score_by_path best"
    )


# ── Test 6: role_pin preserves count by construction ─────────────────────────

def test_role_pin_preserves_count():
    """
    _role_pin_moves returns the same number of moves it receives.
    The assert inside _role_pin_moves already enforces this; this test
    makes the contract explicit and catchable at the test level.
    """
    board = _standard_start()
    legal = get_all_legal_moves(board, RED)

    from checkers.engine.move_facts import compute_move_facts as _cmf

    # Build minimal moves_with_facts for _role_pin_moves
    moves_with_facts = []
    for m in legal:
        facts = _cmf(board, m, RED)
        moves_with_facts.append((m, facts))

    n_before = len(moves_with_facts)
    n_slots = min(5, n_before)

    for score_state in ("EQUAL", "CLEARLY_WINNING", "CLEARLY_LOSING"):
        result, _ = _role_pin_moves(moves_with_facts, score_state, n_slots)
        assert len(result) == n_before, (
            f"_role_pin_moves({score_state}) changed count: "
            f"{n_before} → {len(result)}"
        )


# ── Test 7: MINIMAX_RANK markers placed at correct presentation indices ────────

def test_minimax_rank_markers_correct_positions():
    """
    _build_legal_moves_with_facts must annotate the top-3 scoring moves with
    MINIMAX_RANK_1/2/3 beside their actual move lines, reflecting final
    presentation order (after all sorts and pins).
    """
    board = _standard_start()
    legal = get_all_legal_moves(board, RED)
    assert len(legal) >= 3

    # Assign distinct known scores to three moves by their paths.
    pk_best   = _path_key(legal[0])
    pk_second = _path_key(legal[1])
    pk_third  = _path_key(legal[2])
    score_by_path = {
        pk_best:   5000.0,
        pk_second: 3000.0,
        pk_third:  1000.0,
    }

    _, block, moves_ranked, _, _ = _build_legal_moves_with_facts(
        board, RED, score_by_path=score_by_path
    )

    final_paths = [_path_key(m) for m, _ in moves_ranked]
    pos_best   = final_paths.index(pk_best)
    pos_second = final_paths.index(pk_second)
    pos_third  = final_paths.index(pk_third)

    assert f"[{pos_best}] MINIMAX_RANK_1" in block, (
        f"RANK_1 must be beside index {pos_best}; block:\n{block}"
    )
    assert f"[{pos_second}] MINIMAX_RANK_2" in block, (
        f"RANK_2 must be beside index {pos_second}; block:\n{block}"
    )
    assert f"[{pos_third}] MINIMAX_RANK_3" in block, (
        f"RANK_3 must be beside index {pos_third}; block:\n{block}"
    )
    # Count markers on move lines only.  The legend line uses "[MINIMAX_RANK_1/2/3"
    # (different prefix); move lines always have "] MINIMAX_RANK_".
    assert block.count("] MINIMAX_RANK_") == 3, (
        f"Expected exactly 3 move-line MINIMAX_RANK_ markers; "
        f"got {block.count('] MINIMAX_RANK_')}"
    )


# ── Test 8: MINIMAX_RANK markers do not change move count or set ──────────────

def test_minimax_rank_markers_do_not_affect_move_set():
    """
    MINIMAX_RANK markers are pure display annotations.  Providing score_by_path
    that triggers rank marking must not add, remove, or duplicate moves.
    """
    board = _standard_start()

    n_base, _, moves_base, _, _ = _build_legal_moves_with_facts(
        board, RED, score_by_path=None
    )
    paths_base = frozenset(_path_key(m) for m, _ in moves_base)

    # Give every move a unique score so all 3 ranks (and mm_pin if applicable)
    # are exercised simultaneously.
    score_by_path = {_path_key(m): float(100 - i) for i, (m, _) in enumerate(moves_base)}
    n_ranked, block_ranked, moves_ranked, _, _ = _build_legal_moves_with_facts(
        board, RED, score_by_path=score_by_path
    )
    paths_ranked = frozenset(_path_key(m) for m, _ in moves_ranked)

    assert n_base == n_ranked, f"move count changed: {n_base} → {n_ranked}"
    assert paths_base == paths_ranked, (
        f"move set changed: added={paths_ranked - paths_base}, "
        f"removed={paths_base - paths_ranked}"
    )
    assert "MINIMAX_RANK_1" in block_ranked
    assert "MINIMAX_RANK_2" in block_ranked
    assert "MINIMAX_RANK_3" in block_ranked


# ── Test 9: partial scores — only available ranks are marked ──────────────────

def test_minimax_rank_markers_partial_scores():
    """
    When score_by_path covers fewer than 3 moves, only the available rank
    labels must appear.  Absent ranks must not be fabricated.
    """
    board = _standard_start()
    legal = get_all_legal_moves(board, RED)
    pk0 = _path_key(legal[0])
    pk1 = _path_key(legal[1])

    # 1 scored move → only RANK_1
    _, block1, _, _, _ = _build_legal_moves_with_facts(
        board, RED, score_by_path={pk0: 100.0}
    )
    assert "MINIMAX_RANK_1" in block1,     "RANK_1 must appear when 1 score provided"
    assert "MINIMAX_RANK_2" not in block1, "RANK_2 must not appear with only 1 score"
    assert "MINIMAX_RANK_3" not in block1, "RANK_3 must not appear with only 1 score"

    # 2 scored moves → RANK_1 and RANK_2 only
    _, block2, _, _, _ = _build_legal_moves_with_facts(
        board, RED, score_by_path={pk0: 200.0, pk1: 100.0}
    )
    assert "MINIMAX_RANK_1" in block2
    assert "MINIMAX_RANK_2" in block2
    assert "MINIMAX_RANK_3" not in block2

    # No scores → no markers
    _, block0, _, _, _ = _build_legal_moves_with_facts(board, RED, score_by_path=None)
    assert "MINIMAX_RANK_" not in block0, "No markers when score_by_path is None"

    # Empty score dict → no markers
    _, block_empty, _, _, _ = _build_legal_moves_with_facts(
        board, RED, score_by_path={}
    )
    assert "MINIMAX_RANK_" not in block_empty, "No markers when score_by_path is empty"


# ── Test 10: rank markers use score_by_path, not score_move_with_minimax ──────

def test_minimax_rank_markers_use_score_by_path_not_legacy():
    """
    MINIMAX_RANK markers must be derived exclusively from the score_by_path
    argument.  score_move_with_minimax (legacy per-move scorer) must not be
    called — it is not imported in proposal_agent after the mm_pin fix.
    """
    import checkers.oldfiles.proposal_agent as _pa

    # Confirm the legacy scorer is not reachable from the module namespace.
    assert not hasattr(_pa, "score_move_with_minimax"), (
        "score_move_with_minimax must not be imported in proposal_agent"
    )

    board = _standard_start()
    legal = get_all_legal_moves(board, RED)
    pk0   = _path_key(legal[0])

    # Provide an artificially high score for one move via score_by_path.
    _, block, moves, _, _ = _build_legal_moves_with_facts(
        board, RED, score_by_path={pk0: 9999.0}
    )
    pos0 = [_path_key(m) for m, _ in moves].index(pk0)
    assert f"[{pos0}] MINIMAX_RANK_1" in block, (
        "RANK_1 must reflect the score_by_path value, not a computed score"
    )
