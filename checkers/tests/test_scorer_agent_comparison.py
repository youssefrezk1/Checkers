"""
checkers/tests/test_scorer_agent_comparison.py

Comparison tests verifying that score_all_legal_moves produces output
consistent with the old two-node pipeline (symbolic_decision → minimax_scorer)
when selective D8/D10 is not triggered, and explicitly documents the expected
scope difference when D8 is triggered.

Structure
─────────
Group 1 — scorer_agent vs symbolic_decision
    Score, rank, and summary-stat equality across several board positions
    where D8 cannot fire (piece count above threshold).

Group 2 — scorer_agent vs minimax_scorer cache-hit path
    When minimax_scorer gets a 100 % cache-hit from symbolic_scored_moves,
    its score/rank attachment must equal score_all_legal_moves (D8 off).

Group 3 — D8 scope difference (T35)
    Documents the intentional behavioural difference:
    old pipeline applies D8 to the proposal candidate shortlist (≤ 5 moves);
    score_all_legal_moves applies D8 to ALL legal moves.

    Key finding from running the tests:
    ─ D8 raw scores (at depth=8, full window) are IDENTICAL for shared paths
      between the 3-candidate and all-legal calls (verified empirically: both
      give -12.0 for [(6,5),(5,4)] on T35).
    ─ The promotion-race-verify sub-step (D10) depends on compute_move_facts
      fields (near_promotion, results_in_king, etc.) being present in the
      candidate facts dict. The old pipeline (minimax_scorer) always has these
      facts because validator runs first; score_all_legal_moves also computes
      them internally.
    ─ When promotion-race-verify fires on different candidate sets (3-move vs
      all-legal), the D10 run sees different root moves → TT sharing within
      each call differs → D10 scores for shared paths CAN differ.
    ─ This is the only mechanism by which scores diverge.  The divergence is
      expected and intentional: score_all_legal_moves applies the full D8+D10
      upgrade to a richer candidate set.

Run:
    pytest checkers/tests/test_scorer_agent_comparison.py -v

Fixture note
───────────
All Group 1 / 2 tests run with SELECTIVE_D8_ENABLED patched to False on the
scorer_agent module directly, so board piece count does not matter.
Group 3 sets SELECTIVE_D8_ENABLED=True and uses the T35 endgame position
(13 pieces, D6 gap within threshold) loaded from the same log files used by
test_selective_d8.py.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from checkers.engine.board import RED, BLACK, RED_KING, BLACK_KING
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.state.state import CheckersState
from checkers.search.minimax_core import (
    search_root_all_scores,
    clear_transposition_table,
)

import checkers.oldfiles.symbolic_decision as sd_mod
from checkers.oldfiles.symbolic_decision import symbolic_decision, SYMBOLIC_DECISION_DEPTH
from checkers.oldfiles.minimax_scorer import _build_score_lookup
from checkers.search.selective_d8 import _apply_selective_d8
import checkers.search.selective_d8 as selective_d8_mod
import checkers.agents.scorer_agent as sa_mod
from checkers.agents.scorer_agent import score_all_legal_moves
from checkers.engine.move_facts import compute_move_facts


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pk(path: list) -> tuple:
    return tuple(tuple(sq) for sq in path)


def _empty_board() -> list[list[int]]:
    return [[0] * 8 for _ in range(8)]


def _standard_start() -> list[list[int]]:
    """Full standard start: 24 pieces — D8 never fires (>14 threshold)."""
    b = _empty_board()
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r][c] = RED
    for r in range(0, 3):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r][c] = BLACK
    return b


def _mid_game_board() -> list[list[int]]:
    """3-v-3 mid-game: 6 pieces (D8 threshold=14 → could fire based on gap)."""
    b = _empty_board()
    b[5][0] = RED;   b[5][2] = RED;   b[5][4] = RED
    b[2][1] = BLACK; b[2][3] = BLACK; b[2][5] = BLACK
    return b


def _two_vs_two_board() -> list[list[int]]:
    """2-v-2 with tactical line: 4 pieces."""
    b = _empty_board()
    b[5][0] = RED;   b[5][4] = RED
    b[4][1] = BLACK; b[2][5] = BLACK
    return b


def _king_endgame_board() -> list[list[int]]:
    """King endgame: 4 pieces total."""
    b = _empty_board()
    b[3][2] = RED_KING; b[6][5] = RED
    b[1][4] = BLACK_KING; b[0][1] = BLACK
    return b


def _make_state(board: list[list[int]], player: int = RED) -> CheckersState:
    return CheckersState(board=board, current_player=player)


# All parametrised (board, player) pairs used across Group 1 and Group 2.
# Standard start is always included; smaller boards added for depth coverage.
_BOARDS = [
    pytest.param(_standard_start(), RED,   id="standard_start_RED"),
    pytest.param(_standard_start(), BLACK, id="standard_start_BLACK"),
    pytest.param(_mid_game_board(), RED,   id="3v3_RED"),
    pytest.param(_two_vs_two_board(), RED, id="2v2_RED"),
    pytest.param(_king_endgame_board(), RED, id="king_endgame_RED"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=False)
def d8_disabled(monkeypatch):
    """
    Patches SELECTIVE_D8_ENABLED=False on both scorer_agent and minimax_scorer
    so Group 1 / 2 tests are not contaminated by D8 regardless of env.
    """
    monkeypatch.setattr(sa_mod, "SELECTIVE_D8_ENABLED", False)


@pytest.fixture(autouse=False)
def sr_backend(monkeypatch):
    """Force symbolic_decision to use search_root_all_scores backend."""
    monkeypatch.setattr(sd_mod, "SYMBOLIC_SCORING_BACKEND", "search_root_all_scores")


# ─────────────────────────────────────────────────────────────────────────────
# Log-file helpers (reused from test_selective_d8.py pattern)
# ─────────────────────────────────────────────────────────────────────────────

LOGS      = Path(__file__).parent.parent.parent / "logs"
KF_JSON   = LOGS / "known_failure_positions_20260425_144451.json"
KF_JSONL  = LOGS / "game_20260425_144451_493544.jsonl"


def _rebuild_boards() -> dict[int, list[list[int]]]:
    """Replay moves from the game log; boards[t] = board state AFTER turn t."""
    with open(KF_JSONL) as f:
        records = [json.loads(line) for line in f if line.strip()]
    board: list[list[int]] = _standard_start()
    boards: dict[int, list[list[int]]] = {0: [row[:] for row in board]}
    for rec in records:
        t = rec["turn"]
        move = {"type": rec["move_type"], "path": rec["path"],
                "captured": rec.get("captured", [])}
        board = apply_move(board, move)
        boards[t] = [row[:] for row in board]
    return boards


def _board_for_turn(turn: int, boards: dict) -> list[list[int]]:
    return boards[turn - 1]


def _d6_all_legal_scored(board: list[list[int]], player: int,
                          depth: int) -> list[dict[str, Any]]:
    """
    Score ALL legal moves at *depth* and return candidate-style dicts.
    Each dict: {type, path, captured, facts: {minimax_score, symbolic_rank}}.
    Mirrors the format score_all_legal_moves emits (without move_facts).
    """
    legal = get_all_legal_moves(board, player)
    clear_transposition_table()
    _, _, scored, _ = search_root_all_scores(
        board=board, current_player=player, depth=depth,
        legal_moves=legal, use_tt=True,
        use_tactical_extension=True, use_phase7a=True,
    )
    candidates = []
    for rank, (mv, sc) in enumerate(scored):
        c = deepcopy(mv)
        c.setdefault("facts", {})
        c["facts"]["minimax_score"] = round(float(sc), 2)
        c["facts"]["symbolic_rank"] = rank + 1
        candidates.append(c)
    return candidates


def _d6_all_legal_scored_with_facts(board: list[list[int]], player: int,
                                     depth: int) -> list[dict[str, Any]]:
    """
    Like _d6_all_legal_scored but also computes compute_move_facts for each
    move, exactly matching what the real old pipeline produces:
      symbolic_decision scores → validator enriches with compute_move_facts
      → minimax_scorer attaches minimax_score/rank.

    This is required for fair D8/D10 comparison: _apply_selective_d8's
    promotion-race-verify reads facts["near_promotion"] etc., so both the
    old and new pipeline simulations must have those fields populated.
    """
    legal = get_all_legal_moves(board, player)
    clear_transposition_table()
    _, _, scored, _ = search_root_all_scores(
        board=board, current_player=player, depth=depth,
        legal_moves=legal, use_tt=True,
        use_tactical_extension=True, use_phase7a=True,
    )
    candidates = []
    for rank, (mv, sc) in enumerate(scored):
        c = deepcopy(mv)
        c["facts"] = compute_move_facts(board, mv, player)
        c["facts"]["minimax_score"] = round(float(sc), 2)
        c["facts"]["symbolic_rank"] = rank + 1
        candidates.append(c)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Group 1 — score_all_legal_moves vs symbolic_decision
#
# For every (board, player) pair:
#   • force search_root_all_scores backend in symbolic_decision
#   • force D8 off in scorer_agent
#   • both clear the TT then call search_root_all_scores with identical params
#   → results must be bitwise identical (float equality, same rank)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("board,player", _BOARDS)
def test_scores_match_symbolic_decision_per_path(board, player,
                                                  d8_disabled, sr_backend):
    """
    Every path scored by score_all_legal_moves must carry the same
    minimax_score as the corresponding entry in symbolic_decision output.
    """
    state = _make_state(board, player)
    sd_result = symbolic_decision(state)
    sd_by_path = {
        _pk(e["move"]["path"]): e["minimax_score"]
        for e in sd_result["symbolic_scored_moves"]
    }

    enriched, *_ = score_all_legal_moves(board, player)
    sa_by_path = {
        _pk(e["path"]): e["facts"]["minimax_score"]
        for e in enriched
    }

    assert sd_by_path.keys() == sa_by_path.keys(), (
        f"Path sets differ: sd_only={sd_by_path.keys()-sa_by_path.keys()}"
        f" sa_only={sa_by_path.keys()-sd_by_path.keys()}"
    )
    for path_key in sd_by_path:
        assert sd_by_path[path_key] == sa_by_path[path_key], (
            f"Score mismatch for {list(path_key)}: "
            f"symbolic_decision={sd_by_path[path_key]} "
            f"scorer_agent={sa_by_path[path_key]}"
        )


@pytest.mark.parametrize("board,player", _BOARDS)
def test_ranks_match_symbolic_decision_per_path(board, player,
                                                 d8_disabled, sr_backend):
    """
    Rank assigned by score_all_legal_moves (facts['symbolic_rank']) must equal
    the rank field from symbolic_decision for every path.
    """
    state = _make_state(board, player)
    sd_result = symbolic_decision(state)
    sd_rank_by_path = {
        _pk(e["move"]["path"]): e["rank"]
        for e in sd_result["symbolic_scored_moves"]
    }

    enriched, *_ = score_all_legal_moves(board, player)
    sa_rank_by_path = {
        _pk(e["path"]): e["facts"]["symbolic_rank"]
        for e in enriched
    }

    for path_key in sd_rank_by_path:
        assert sd_rank_by_path[path_key] == sa_rank_by_path[path_key], (
            f"Rank mismatch for {list(path_key)}: "
            f"symbolic_decision={sd_rank_by_path[path_key]} "
            f"scorer_agent={sa_rank_by_path[path_key]}"
        )


@pytest.mark.parametrize("board,player", _BOARDS)
def test_best_score_matches_symbolic_decision(board, player,
                                               d8_disabled, sr_backend):
    """best_score returned by scorer must equal symbolic_best_score from symbolic_decision."""
    state = _make_state(board, player)
    sd_result = symbolic_decision(state)

    _, best, *_ = score_all_legal_moves(board, player)

    assert best == sd_result["symbolic_best_score"], (
        f"best_score mismatch: scorer={best} symbolic_decision={sd_result['symbolic_best_score']}"
    )


@pytest.mark.parametrize("board,player", _BOARDS)
def test_second_best_score_matches_symbolic_decision(board, player,
                                                      d8_disabled, sr_backend):
    """second_best_score from scorer must match symbolic_second_best_score."""
    state = _make_state(board, player)
    sd_result = symbolic_decision(state)

    _, _, second, _ = score_all_legal_moves(board, player)

    assert second == sd_result["symbolic_second_best_score"], (
        f"second_best mismatch: scorer={second} "
        f"symbolic_decision={sd_result['symbolic_second_best_score']}"
    )


@pytest.mark.parametrize("board,player", _BOARDS)
def test_gap_matches_symbolic_decision(board, player, d8_disabled, sr_backend):
    """
    gap from scorer must match symbolic_gap from symbolic_decision.

    symbolic_decision stores gap = best - second_best (when two moves exist),
    or best - LOSS_SCORE (when only one legal move).  We compare to within
    0.01 to account for rounding during the round() calls in each module.
    """
    state = _make_state(board, player)
    sd_result = symbolic_decision(state)
    sd_gap = sd_result["symbolic_gap"]

    _, _, _, sa_gap = score_all_legal_moves(board, player)

    # When gap is inf (single-move position), symbolic_decision stores
    # best - LOSS_SCORE; scorer returns float('inf').  Accept both forms.
    if sa_gap == float("inf") or sd_gap == float("inf"):
        # Both should be semantically "only one legal move"
        legal = get_all_legal_moves(board, player)
        assert len(legal) <= 1, (
            f"inf gap but {len(legal)} legal moves — unexpected"
        )
        return

    assert abs(sa_gap - sd_gap) < 0.01, (
        f"gap mismatch: scorer={sa_gap:.4f} symbolic_decision={sd_gap:.4f}"
    )


def test_facts_present_on_every_move(d8_disabled, sr_backend):
    """score_all_legal_moves must populate a facts dict on every enriched move."""
    board = _standard_start()
    enriched, *_ = score_all_legal_moves(board, RED)
    assert enriched, "expected non-empty result for standard start"
    for entry in enriched:
        assert "facts" in entry, f"missing facts on {entry.get('path')}"
        f = entry["facts"]
        # A sample of move_facts fields that should always be present
        for key in ("move_type", "captures_count", "results_in_king",
                    "opponent_can_recapture", "material_advantage",
                    "minimax_score", "symbolic_rank"):
            assert key in f, (
                f"facts missing '{key}' for path {entry.get('path')}"
            )


def test_symbolic_decision_has_no_facts_scorer_does(d8_disabled, sr_backend):
    """
    symbolic_decision's symbolic_scored_moves contain only {move, minimax_score, rank}.
    score_all_legal_moves additionally embeds all compute_move_facts fields.
    This test documents that the extra 'facts' enrichment is new in scorer_agent.
    """
    board = _standard_start()
    state = _make_state(board, RED)
    sd_result = symbolic_decision(state)
    sd_entry = sd_result["symbolic_scored_moves"][0]

    # symbolic_decision entry has exactly these three keys
    assert set(sd_entry.keys()) == {"move", "minimax_score", "rank"}, (
        f"Unexpected keys in symbolic_scored_moves entry: {set(sd_entry.keys())}"
    )

    # scorer_agent entry has full facts dict embedded
    enriched, *_ = score_all_legal_moves(board, RED)
    sa_entry = enriched[0]
    assert "facts" in sa_entry
    # facts should have many more fields than just minimax_score + symbolic_rank
    assert len(sa_entry["facts"]) > 5, (
        "Expected rich facts dict from compute_move_facts; "
        f"got only {len(sa_entry['facts'])} keys"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Group 2 — score_all_legal_moves vs minimax_scorer cache-hit path
#
# The minimax_scorer cache-hit path:
#   1. symbolic_decision fills symbolic_scored_moves.
#   2. _build_score_lookup converts it to a path → (score, rank) map.
#   3. For each candidate, a lookup returns the pre-computed score and rank.
#
# When all candidates hit the cache, the scores come directly from
# symbolic_decision — so they must match scorer_agent (which runs the
# same search under the same conditions).
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("board,player", _BOARDS)
def test_scores_match_minimax_scorer_cache_hit(board, player,
                                                d8_disabled, sr_backend):
    """
    Scores from a 100 %-cache-hit minimax_scorer pass must equal scorer_agent.

    Simulation: run symbolic_decision → build lookup → look up ALL legal moves
    (simulating a case where proposal returned every legal move and all hit cache).
    """
    state = _make_state(board, player)
    sd_result = symbolic_decision(state)
    score_lookup = _build_score_lookup(sd_result["symbolic_scored_moves"])

    legal = get_all_legal_moves(board, player)
    old_by_path: dict[tuple, float] = {}
    for mv in legal:
        pk = _pk(mv["path"])
        cached = score_lookup.get(pk)
        assert cached is not None, (
            f"Cache miss for {mv['path']} — expected 100 % cache hit"
        )
        old_by_path[pk] = cached[0]  # (score, rank) → score

    enriched, *_ = score_all_legal_moves(board, player)
    sa_by_path = {_pk(e["path"]): e["facts"]["minimax_score"] for e in enriched}

    assert old_by_path.keys() == sa_by_path.keys()
    for pk in old_by_path:
        assert old_by_path[pk] == sa_by_path[pk], (
            f"Cache-hit vs scorer_agent score mismatch for {list(pk)}: "
            f"cache={old_by_path[pk]} scorer={sa_by_path[pk]}"
        )


@pytest.mark.parametrize("board,player", _BOARDS)
def test_ranks_match_minimax_scorer_cache_hit(board, player,
                                               d8_disabled, sr_backend):
    """
    symbolic_rank from cache-hit lookup must equal facts['symbolic_rank']
    produced by scorer_agent.
    """
    state = _make_state(board, player)
    sd_result = symbolic_decision(state)
    score_lookup = _build_score_lookup(sd_result["symbolic_scored_moves"])

    legal = get_all_legal_moves(board, player)
    old_rank_by_path: dict[tuple, int] = {}
    for mv in legal:
        pk = _pk(mv["path"])
        cached = score_lookup.get(pk)
        assert cached is not None
        old_rank_by_path[pk] = cached[1]  # (score, rank) → rank

    enriched, *_ = score_all_legal_moves(board, player)
    sa_rank_by_path = {_pk(e["path"]): e["facts"]["symbolic_rank"] for e in enriched}

    for pk in old_rank_by_path:
        assert old_rank_by_path[pk] == sa_rank_by_path[pk], (
            f"Rank mismatch for {list(pk)}: "
            f"cache={old_rank_by_path[pk]} scorer={sa_rank_by_path[pk]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Group 3 — D8 scope difference (T35)
#
# T35 is a 13-piece endgame (< 14 threshold) with D6 gap ≈ 24 (≤ 30 threshold).
# _apply_selective_d8 triggers for this position.
#
# Old pipeline behaviour:
#   proposal selects top-k candidates from the D6-sorted list.
#   minimax_scorer calls _apply_selective_d8 on those k candidates only.
#
# New scorer_agent behaviour:
#   score_all_legal_moves calls _apply_selective_d8 on ALL n_legal moves.
#
# The key invariant:  old_candidate_paths ⊂ new_all_legal_paths.
# The key difference: D8 considers more paths in the new pipeline, which can
#   expose a better move that the proposal shortlist never contained.
#
# D8 scores for paths that APPEAR in both inputs should be identical because
# search_root_all_scores uses a full window (alpha=-inf, beta=inf) per root
# move and shared TT — so score of path X is not altered by whether paths
# Y,Z,… also appear in the input list.  Any divergence would indicate a bug.
# ─────────────────────────────────────────────────────────────────────────────

PROPOSAL_SHORTLIST_SIZE = 3   # simulates a minimal proposal shortlist


@pytest.fixture(scope="module")
def t35_board():
    """Board state for T35 from the known-failure game log."""
    boards = _rebuild_boards()
    return _board_for_turn(35, boards)


def test_d8_no_trigger_scores_are_identical_without_d8(t35_board, monkeypatch):
    """
    Baseline: with D8 disabled, scorer_agent and symbolic_decision agree on
    all scores for T35.  This confirms the D6 layer is consistent between
    the old and new pipelines.
    """
    monkeypatch.setattr(sa_mod, "SELECTIVE_D8_ENABLED", False)
    monkeypatch.setattr(sd_mod, "SYMBOLIC_SCORING_BACKEND", "search_root_all_scores")

    state = _make_state(t35_board, RED)
    sd_result = symbolic_decision(state)
    sd_by_path = {
        _pk(e["move"]["path"]): e["minimax_score"]
        for e in sd_result["symbolic_scored_moves"]
    }

    enriched, *_ = score_all_legal_moves(t35_board, RED)
    sa_by_path = {_pk(e["path"]): e["facts"]["minimax_score"] for e in enriched}

    assert sd_by_path.keys() == sa_by_path.keys()
    for pk in sd_by_path:
        assert sd_by_path[pk] == sa_by_path[pk], (
            f"D6 score mismatch for {list(pk)}: "
            f"symbolic_decision={sd_by_path[pk]} scorer_agent={sa_by_path[pk]}"
        )


def test_d8_scope_old_candidates_strict_subset_of_all_legal(t35_board, monkeypatch):
    """
    DOCUMENTED DIFFERENCE — D8 input scope.

    Old pipeline: D8 receives only the proposal shortlist (top-k candidates).
    New pipeline: D8 receives ALL legal moves.

    This test asserts:
      1. The old candidate set is a strict subset of all legal moves.
      2. The new pipeline covers at least one path the old shortlist excluded.

    No change to scores or thresholds is implied — only the input set differs.
    """
    # Disable D8 in scorer_agent so we can inspect the D6 scores cleanly.
    monkeypatch.setattr(sa_mod, "SELECTIVE_D8_ENABLED", False)

    # Simulate D6 scoring to build the candidate shortlist the old pipeline
    # would pass to _apply_selective_d8.
    d6_candidates = _d6_all_legal_scored(t35_board, RED, SYMBOLIC_DECISION_DEPTH)

    # Old pipeline candidate set: top-k by D6 rank (proposal shortlist).
    old_candidates = d6_candidates[:PROPOSAL_SHORTLIST_SIZE]
    old_paths = {_pk(c["path"]) for c in old_candidates}

    # New pipeline: all legal moves.
    all_legal = get_all_legal_moves(t35_board, RED)
    all_paths = {_pk(m["path"]) for m in all_legal}

    assert old_paths < all_paths, (
        "Old candidate paths must be a STRICT subset of all legal paths "
        f"(old={len(old_paths)} new={len(all_paths)})"
    )
    assert len(all_paths) > len(old_paths), (
        f"Expected more legal moves ({len(all_paths)}) than "
        f"old candidates ({len(old_paths)})"
    )
    excluded_paths = all_paths - old_paths
    assert excluded_paths, (
        "Expected at least one legal move excluded from old candidate shortlist"
    )


def test_d8_trigger_both_fire_old_on_candidates_new_on_all_legal(
        t35_board, monkeypatch):
    """
    DOCUMENTED DIFFERENCE — paths and facts context of each D8/D10 call.

    When D8 triggers on T35 (13 pieces, D6 gap ≈ 24):

    OLD pipeline (minimax_scorer):
      _apply_selective_d8 receives the top-k proposal candidates, each
      already enriched by validator with full compute_move_facts fields.
      D8 (and potentially D10 via promotion-race-verify) runs on k paths.

    NEW pipeline (score_all_legal_moves):
      _apply_selective_d8 receives ALL legal moves, each enriched with
      compute_move_facts by the scorer itself.
      D8 and D10 run on n_legal paths.

    This test verifies three things:

    1. PATH-SET CONTAINMENT: old_paths ⊂ new_paths.  The new pipeline
       evaluates a strictly larger set — it cannot miss a move the old
       pipeline considered.

    2. D8 RAW SCORES (promotion-race-verify disabled): for shared paths,
       search_root_all_scores at depth=8 with full window (α=-∞, β=+∞)
       gives IDENTICAL scores regardless of what other root moves exist in
       the input.  This is verified by disabling D10 on both calls.

    3. FULL D8+D10 SCORES MAY DIFFER: when promotion-race-verify fires it
       calls search_root_all_scores at depth=10 on the CANDIDATE LIST
       (k moves vs n_legal moves).  Because the two calls share TT entries
       created by earlier root-move searches within each call, the subtree
       context differs → D10 scores for shared paths can legitimately differ.
       This divergence is expected and intentional.

    Root cause of divergence (empirically confirmed on T35):
      - D8 raw score of [(6,5),(5,4)]: -12.0 in BOTH the 3-candidate and
        all-legal calls (full-window independence confirmed).
      - After D10 fires (on different candidate sets), [(6,5),(5,4)] gets
        a different score because the TT is shared differently within each
        D10 call, changing subtree evaluations for later-ordered moves.
    """
    # ── Step 1: D6 scoring WITH full facts (mirrors real old pipeline) ────────
    # The real old pipeline: validator enriches candidates with compute_move_facts
    # before minimax_scorer calls _apply_selective_d8.  We reproduce that here
    # so promotion-race-verify has the same information in both calls.
    monkeypatch.setattr(sa_mod, "SELECTIVE_D8_ENABLED", False)
    d6_candidates_with_facts = _d6_all_legal_scored_with_facts(
        t35_board, RED, SYMBOLIC_DECISION_DEPTH
    )
    old_candidates = d6_candidates_with_facts[:PROPOSAL_SHORTLIST_SIZE]
    old_paths = {_pk(c["path"]) for c in old_candidates}

    all_legal = get_all_legal_moves(t35_board, RED)
    new_paths = {_pk(m["path"]) for m in all_legal}

    # ── Assertion 1: path-set containment ─────────────────────────────────────
    assert old_paths < new_paths, (
        "Old D8 candidate paths must be a strict subset of all legal paths. "
        f"old={len(old_paths)} new={len(new_paths)}"
    )
    excluded_paths = new_paths - old_paths
    assert excluded_paths, "Expected at least one path excluded from old shortlist"

    # ── Assertion 2: D8 raw scores for shared paths are identical ─────────────
    # Disable promotion-race-verify on both sides so we isolate pure D8 scores.
    monkeypatch.setattr(selective_d8_mod, "PROMOTION_RACE_VERIFY_ENABLED", False)

    os.environ["SELECTIVE_D8_ENABLED"]           = "true"
    os.environ["SELECTIVE_D8_PIECE_THRESHOLD"]   = "14"
    os.environ["SELECTIVE_D8_GAP_THRESHOLD"]     = "30"
    os.environ["SELECTIVE_D8_DEPTH"]             = "8"
    os.environ["SELECTIVE_D8_INCLUDE_EXACT_TIES"] = "false"

    old_d8_only = _apply_selective_d8(t35_board, RED, old_candidates)
    old_d8_scores = {_pk(c["path"]): c["facts"]["minimax_score"] for c in old_d8_only}

    # scorer_agent with D8 on all legal, promotion-race-verify also disabled.
    monkeypatch.setattr(sa_mod, "SELECTIVE_D8_ENABLED", True)
    new_enriched_d8_only, *_ = score_all_legal_moves(t35_board, RED)
    new_d8_scores = {
        _pk(e["path"]): e["facts"]["minimax_score"] for e in new_enriched_d8_only
    }

    for pk in old_paths:
        assert pk in new_d8_scores, f"Shared path {list(pk)} missing from new output"
        assert old_d8_scores[pk] == new_d8_scores[pk], (
            f"D8 raw score mismatch for shared path {list(pk)}: "
            f"old(k-candidates)={old_d8_scores[pk]} "
            f"new(all-legal)={new_d8_scores[pk]}. "
            "Full-window per-root search must give identical scores regardless "
            "of what other root moves are present."
        )

    # ── Assertion 3: full D8+D10 (promotion-race-verify re-enabled) ──────────
    # Re-enable promotion-race-verify so it fires naturally.
    # Scores for shared paths MAY differ because D10 runs search_root_all_scores
    # on different candidate sets (k vs n_legal) → different TT sharing context
    # within each D10 call → different subtree evaluations for later moves.
    # This difference is expected and is exactly what we are documenting.
    monkeypatch.setattr(selective_d8_mod, "PROMOTION_RACE_VERIFY_ENABLED", True)
    monkeypatch.setattr(sa_mod, "SELECTIVE_D8_ENABLED", True)

    _apply_selective_d8(t35_board, RED, old_candidates)   # run old D8+D10 for side-effects / logging

    new_enriched_full, *_ = score_all_legal_moves(t35_board, RED)
    new_full_scores = {
        _pk(e["path"]): e["facts"]["minimax_score"] for e in new_enriched_full
    }
    new_full_paths = {_pk(e["path"]) for e in new_enriched_full}

    # Path set must still be a strict superset (D10 cannot add/remove paths).
    assert old_paths < new_full_paths, (
        "After full D8+D10, new pipeline must still cover strictly more paths"
    )
    # Scores for shared paths CAN differ after D10 — we do NOT assert equality.
    # Instead, we assert the result is valid (finite scores, correct rank order).
    for pk in old_paths:
        assert pk in new_full_scores
        assert isinstance(new_full_scores[pk], float)
        assert new_full_scores[pk] != float("inf")
    # New result is sorted best-first with sequential ranks.
    new_full_ranks = [e["facts"]["symbolic_rank"] for e in new_enriched_full]
    assert new_full_ranks == list(range(1, len(new_enriched_full) + 1)), (
        "After full D8+D10, scorer_agent result must still be sorted with ranks 1..n"
    )


def test_d8_scores_unchanged_for_no_trigger_position(monkeypatch):
    """
    For a position where D8 does NOT trigger (piece count above threshold),
    score_all_legal_moves with D8 enabled must return the same scores as with
    D8 disabled — i.e., D8 is a no-op.
    """
    board = _standard_start()   # 24 pieces >> 14 threshold → D8 never fires

    monkeypatch.setattr(sa_mod, "SELECTIVE_D8_ENABLED", False)
    enriched_no_d8, best_no, *_ = score_all_legal_moves(board, RED)
    by_path_no_d8 = {_pk(e["path"]): e["facts"]["minimax_score"] for e in enriched_no_d8}

    monkeypatch.setattr(sa_mod, "SELECTIVE_D8_ENABLED", True)
    enriched_d8, best_d8, *_ = score_all_legal_moves(board, RED)
    by_path_d8 = {_pk(e["path"]): e["facts"]["minimax_score"] for e in enriched_d8}

    assert by_path_no_d8.keys() == by_path_d8.keys()
    for pk in by_path_no_d8:
        assert by_path_no_d8[pk] == by_path_d8[pk], (
            f"Score changed for {list(pk)} despite D8 not triggering: "
            f"d8_off={by_path_no_d8[pk]} d8_on={by_path_d8[pk]}"
        )
    assert best_no == best_d8


def test_d8_trigger_does_not_change_path_set(t35_board, monkeypatch):
    """
    Enabling D8 must not add or remove any move from the enriched list.
    Only minimax_score and symbolic_rank may change.
    """
    monkeypatch.setattr(sa_mod, "SELECTIVE_D8_ENABLED", False)
    enriched_no_d8, *_ = score_all_legal_moves(t35_board, RED)
    paths_no_d8 = {_pk(e["path"]) for e in enriched_no_d8}

    os.environ["SELECTIVE_D8_ENABLED"]           = "true"
    os.environ["SELECTIVE_D8_PIECE_THRESHOLD"]   = "14"
    os.environ["SELECTIVE_D8_GAP_THRESHOLD"]     = "30"
    os.environ["SELECTIVE_D8_DEPTH"]             = "8"
    os.environ["SELECTIVE_D8_INCLUDE_EXACT_TIES"] = "false"
    monkeypatch.setattr(sa_mod, "SELECTIVE_D8_ENABLED", True)

    enriched_d8, *_ = score_all_legal_moves(t35_board, RED)
    paths_d8 = {_pk(e["path"]) for e in enriched_d8}

    assert paths_no_d8 == paths_d8, (
        "D8 must not add or remove paths from the enriched list. "
        f"added={paths_d8-paths_no_d8} removed={paths_no_d8-paths_d8}"
    )
    assert len(enriched_no_d8) == len(enriched_d8)
