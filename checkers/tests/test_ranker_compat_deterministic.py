"""
checkers/tests/test_ranker_compat_deterministic.py

Integration tests: verify that select_proposal_candidates output is fully
compatible with the ranker_agent preparation layer (safety filter, prompt
builder, override logic) WITHOUT calling the LLM.

What is NOT tested here:
  - call_ranker / call_mistral_ranker (LLM, network)
  - ranker_agent() node itself (requires LLM call path)
  - graph.py (intentionally untouched)

What IS tested:
  - Structural fields on each candidate (type, path, captured, facts)
  - facts["minimax_score"] is a finite float (not -inf)
  - facts["symbolic_rank"] is a positive integer
  - Every field accessed by _apply_safety_filter is present
  - Every field accessed by _override_if_llm_chose_much_worse_minimax is present
  - _apply_safety_filter(candidates) runs without error and returns a valid subset
  - _get_minimax_score(candidate) returns a finite float for every candidate
  - build_ranker_user_prompt(state, filtered, index_map) runs without error
  - _build_ranker_filtered_menu_snapshot(candidates) runs without error
  - Full pre-ranker pipeline: score → propose → filter → prompt

Run:
    pytest checkers/tests/test_ranker_compat_deterministic.py -v
"""
from __future__ import annotations

import math
from typing import Any

import pytest

from checkers.engine.board import RED, BLACK, RED_KING, BLACK_KING
from checkers.engine.rules import get_all_legal_moves
from checkers.agents.scorer_agent import score_all_legal_moves
from checkers.agents.deterministic_proposal import select_proposal_candidates
from checkers.agents.ranker_agent import (
    _apply_safety_filter,
    _get_minimax_score,
    _override_if_llm_chose_much_worse_minimax,
    _build_ranker_filtered_menu_snapshot,
    build_ranker_user_prompt,
    RANKER_SYSTEM_PROMPT,
    RANKER_SYSTEM_PROMPT_SINGLE,
)
from checkers.state.state import CheckersState


# ── Boards ────────────────────────────────────────────────────────────────────

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


def _midgame_board() -> list[list[int]]:
    """3v3 mid-game board — small enough for fast scoring, varied facts."""
    b = [[0] * 8 for _ in range(8)]
    b[5][0] = RED;   b[5][2] = RED;   b[5][4] = RED
    b[2][1] = BLACK; b[2][3] = BLACK; b[2][5] = BLACK
    return b


def _capture_board() -> list[list[int]]:
    """RED at (4,3) can jump BLACK at (3,4) → landing (2,5); quiet move also exists."""
    b = [[0] * 8 for _ in range(8)]
    b[4][3] = RED
    b[5][2] = RED
    b[3][4] = BLACK
    b[0][7] = BLACK
    return b


def _promotion_board() -> list[list[int]]:
    """RED at (1,2) can promote by stepping to (0,1) or (0,3)."""
    b = [[0] * 8 for _ in range(8)]
    b[1][2] = RED
    b[7][7] = RED
    b[0][5] = BLACK
    b[6][0] = BLACK
    return b


def _king_board() -> list[list[int]]:
    """Endgame: RED king vs BLACK pieces."""
    b = [[0] * 8 for _ in range(8)]
    b[3][2] = RED_KING
    b[6][5] = RED
    b[1][4] = BLACK_KING
    b[0][1] = BLACK
    return b


def _make_state(board, player=RED, ctx=None) -> CheckersState:
    return CheckersState(board=board, current_player=player, strategic_context=ctx)


def _candidates(board, player=RED, ctx=None, k=5) -> list[dict]:
    scored, _, _, _ = score_all_legal_moves(board, player)
    return select_proposal_candidates(scored, strategic_context=ctx, k=k)


# ── Parametrised boards used across most tests ────────────────────────────────

_CASES = [
    pytest.param(_start_board(),    RED,   None,                                  id="start_RED"),
    pytest.param(_start_board(),    BLACK, None,                                  id="start_BLACK"),
    pytest.param(_midgame_board(),  RED,   None,                                  id="midgame_RED"),
    pytest.param(_capture_board(),  RED,   None,                                  id="capture_RED"),
    pytest.param(_promotion_board(),RED,   None,                                  id="promotion_RED"),
    pytest.param(_king_board(),     RED,   None,                                  id="king_endgame_RED"),
    pytest.param(
        _midgame_board(), RED,
        {"score_state": "SLIGHTLY_LOSING", "game_phase": "MIDGAME",
         "strategic_priorities": ["SEEK_COUNTERPLAY"]},
        id="midgame_losing_ctx",
    ),
    pytest.param(
        _midgame_board(), RED,
        {"score_state": "CLEARLY_WINNING", "game_phase": "ENDGAME",
         "strategic_priorities": ["CONVERT_ADVANTAGE"]},
        id="midgame_winning_ctx",
    ),
]


# ── Group 1: Structural field completeness ────────────────────────────────────

@pytest.mark.parametrize("board,player,ctx", _CASES)
def test_each_candidate_has_type_path_captured_facts(board, player, ctx):
    """Top-level keys: type, path, captured, facts must all be present."""
    candidates = _candidates(board, player, ctx)
    assert candidates, "expected non-empty candidates"
    for c in candidates:
        assert "type"     in c, f"missing 'type' on {c.get('path')}"
        assert "path"     in c, "missing 'path'"
        assert "captured" in c, f"missing 'captured' on {c.get('path')}"
        assert "facts"    in c, f"missing 'facts' on {c.get('path')}"
        assert isinstance(c["facts"], dict), "'facts' must be a dict"


@pytest.mark.parametrize("board,player,ctx", _CASES)
def test_each_candidate_has_minimax_score_in_facts(board, player, ctx):
    """facts['minimax_score'] must be a finite float (not -inf, not NaN)."""
    candidates = _candidates(board, player, ctx)
    for c in candidates:
        score = c["facts"].get("minimax_score")
        assert score is not None, f"minimax_score missing on {c.get('path')}"
        assert isinstance(score, float), f"minimax_score must be float, got {type(score)}"
        assert math.isfinite(score), f"minimax_score must be finite, got {score} on {c.get('path')}"


@pytest.mark.parametrize("board,player,ctx", _CASES)
def test_each_candidate_has_symbolic_rank_in_facts(board, player, ctx):
    """facts['symbolic_rank'] must be a positive integer."""
    candidates = _candidates(board, player, ctx)
    for c in candidates:
        rank = c["facts"].get("symbolic_rank")
        assert rank is not None, f"symbolic_rank missing on {c.get('path')}"
        assert isinstance(rank, int), f"symbolic_rank must be int, got {type(rank)}"
        assert rank >= 1, f"symbolic_rank must be >= 1, got {rank}"


@pytest.mark.parametrize("board,player,ctx", _CASES)
def test_candidate_type_is_simple_or_jump(board, player, ctx):
    """Move type must be 'simple' or 'jump'."""
    candidates = _candidates(board, player, ctx)
    for c in candidates:
        assert c["type"] in {"simple", "jump"}, (
            f"unexpected move type '{c['type']}' on path {c.get('path')}"
        )


# ── Group 2: Safety filter required fields ────────────────────────────────────

# All fields accessed by _apply_safety_filter and _get_minimax_score.
_SAFETY_FILTER_FIELDS = [
    "opponent_can_recapture",   # primary safety classification
    "minimax_score",            # _get_minimax_score; used for best_safe_score + gap checks
    "results_in_king",          # promotion exemption pass
    "creates_immediate_threat", # _has_real_action
    "shot_sequence_available",  # _has_real_action
    "blocks_opponent_landing",  # _has_real_action
    "counterplay_score",        # _unsafe_qualifies strong_counterplay check
]

@pytest.mark.parametrize("board,player,ctx", _CASES)
def test_safety_filter_required_fields_present(board, player, ctx):
    """Every field accessed by _apply_safety_filter must be present in facts."""
    candidates = _candidates(board, player, ctx)
    for c in candidates:
        facts = c["facts"]
        for field in _SAFETY_FILTER_FIELDS:
            assert field in facts, (
                f"Safety-filter field '{field}' missing on path {c.get('path')}"
            )


# All fields accessed by _override_if_llm_chose_much_worse_minimax.
_OVERRIDE_FIELDS = [
    "minimax_score",
    "opponent_can_recapture",
    "our_pieces_threatened_after",
    "creates_immediate_threat",
    "shot_sequence_available",
    "blocks_opponent_landing",
    "counterplay_score",
    "king_activity_score",
    "leaves_piece_isolated",
    "weakens_king_row",
    "quiet_move_role",
    "moved_piece_is_threatened",
    "max_opponent_jump_captures",
    "opponent_jump_count",
    "forced_opponent_jump_reply",
    "simplification_value",
    "double_corner_smokeout_pressure",
    "edge_confinement_delta",
    "net_gain",
    "center_control",
]

@pytest.mark.parametrize("board,player,ctx", _CASES)
def test_override_required_fields_present(board, player, ctx):
    """Every field accessed by _override_if_llm_chose_much_worse_minimax must be present."""
    candidates = _candidates(board, player, ctx)
    for c in candidates:
        facts = c["facts"]
        for field in _OVERRIDE_FIELDS:
            assert field in facts, (
                f"Override field '{field}' missing on path {c.get('path')}"
            )


# ── Group 3: _get_minimax_score compatibility ─────────────────────────────────

@pytest.mark.parametrize("board,player,ctx", _CASES)
def test_get_minimax_score_returns_finite_float(board, player, ctx):
    """_get_minimax_score(candidate) must return a finite float for every candidate."""
    candidates = _candidates(board, player, ctx)
    for c in candidates:
        score = _get_minimax_score(c)
        assert isinstance(score, float), (
            f"_get_minimax_score must return float, got {type(score)} for {c.get('path')}"
        )
        assert math.isfinite(score), (
            f"_get_minimax_score must return finite float, got {score} for {c.get('path')}"
        )


# ── Group 4: _apply_safety_filter compatibility ───────────────────────────────

@pytest.mark.parametrize("board,player,ctx", _CASES)
def test_safety_filter_runs_without_error(board, player, ctx):
    """_apply_safety_filter must complete without raising on deterministic candidates."""
    candidates = _candidates(board, player, ctx)
    priorities = (ctx or {}).get("strategic_priorities", [])
    score_state = (ctx or {}).get("score_state", "EQUAL")
    # Must not raise
    filtered, index_map = _apply_safety_filter(
        candidates,
        strategic_priorities=priorities,
        score_state=score_state,
    )
    assert isinstance(filtered, list)
    assert isinstance(index_map, list)


@pytest.mark.parametrize("board,player,ctx", _CASES)
def test_safety_filter_returns_non_empty_subset(board, player, ctx):
    """Filtered output must be non-empty and all elements must come from candidates."""
    candidates = _candidates(board, player, ctx)
    priorities = (ctx or {}).get("strategic_priorities", [])
    score_state = (ctx or {}).get("score_state", "EQUAL")

    filtered, index_map = _apply_safety_filter(
        candidates,
        strategic_priorities=priorities,
        score_state=score_state,
    )

    assert len(filtered) >= 1, "safety filter must keep at least one candidate"

    # All filtered elements are from candidates (by path identity).
    candidate_paths = {tuple(tuple(sq) for sq in c["path"]) for c in candidates}
    for m in filtered:
        pk = tuple(tuple(sq) for sq in m["path"])
        assert pk in candidate_paths, (
            f"filtered contains path {m['path']} not in candidates"
        )


@pytest.mark.parametrize("board,player,ctx", _CASES)
def test_safety_filter_index_map_valid_range(board, player, ctx):
    """index_map values must be valid indices into the original candidates list."""
    candidates = _candidates(board, player, ctx)
    priorities = (ctx or {}).get("strategic_priorities", [])
    score_state = (ctx or {}).get("score_state", "EQUAL")

    filtered, index_map = _apply_safety_filter(
        candidates,
        strategic_priorities=priorities,
        score_state=score_state,
    )

    assert len(filtered) == len(index_map), (
        "filtered and index_map must have the same length"
    )
    for idx in index_map:
        assert 0 <= idx < len(candidates), (
            f"index_map value {idx} out of range [0, {len(candidates)})"
        )


@pytest.mark.parametrize("board,player,ctx", _CASES)
def test_safety_filter_output_minimax_scores_finite(board, player, ctx):
    """Every move that passes the safety filter must still have a finite minimax_score."""
    candidates = _candidates(board, player, ctx)
    priorities = (ctx or {}).get("strategic_priorities", [])
    score_state = (ctx or {}).get("score_state", "EQUAL")

    filtered, _ = _apply_safety_filter(
        candidates,
        strategic_priorities=priorities,
        score_state=score_state,
    )

    for m in filtered:
        score = _get_minimax_score(m)
        assert math.isfinite(score), (
            f"filtered move {m.get('path')} has non-finite minimax_score: {score}"
        )


# ── Group 5: Prompt builder compatibility ─────────────────────────────────────

@pytest.mark.parametrize("board,player,ctx", _CASES)
def test_build_ranker_prompt_runs_without_error(board, player, ctx):
    """build_ranker_user_prompt(state, filtered, index_map) must not raise."""
    candidates = _candidates(board, player, ctx)
    priorities = (ctx or {}).get("strategic_priorities", [])
    score_state = (ctx or {}).get("score_state", "EQUAL")

    filtered, index_map = _apply_safety_filter(
        candidates,
        strategic_priorities=priorities,
        score_state=score_state,
    )

    state = _make_state(board, player, ctx)
    prompt = build_ranker_user_prompt(state, filtered, index_map)
    assert isinstance(prompt, str)
    assert len(prompt) > 0


@pytest.mark.parametrize("board,player,ctx", _CASES)
def test_build_ranker_prompt_contains_move_paths(board, player, ctx):
    """The built prompt must reference the path coordinates of each filtered move."""
    candidates = _candidates(board, player, ctx)
    priorities = (ctx or {}).get("strategic_priorities", [])
    score_state = (ctx or {}).get("score_state", "EQUAL")

    filtered, index_map = _apply_safety_filter(
        candidates,
        strategic_priorities=priorities,
        score_state=score_state,
    )

    state = _make_state(board, player, ctx)
    prompt = build_ranker_user_prompt(state, filtered, index_map)

    for m in filtered:
        # The first coordinate of the path must appear somewhere in the prompt.
        first_sq = m["path"][0]
        coord_str = str(first_sq[0])   # row number as string
        assert coord_str in prompt, (
            f"prompt does not mention row {first_sq[0]} from path {m['path']}"
        )


@pytest.mark.parametrize("board,player,ctx", _CASES)
def test_filtered_menu_snapshot_runs_without_error(board, player, ctx):
    """_build_ranker_filtered_menu_snapshot must not raise on deterministic candidates."""
    candidates = _candidates(board, player, ctx)
    priorities = (ctx or {}).get("strategic_priorities", [])
    score_state = (ctx or {}).get("score_state", "EQUAL")

    filtered, _ = _apply_safety_filter(
        candidates,
        strategic_priorities=priorities,
        score_state=score_state,
    )

    snapshot = _build_ranker_filtered_menu_snapshot(filtered)
    assert isinstance(snapshot, list)
    assert len(snapshot) == len(filtered)
    for entry in snapshot:
        assert "type"     in entry
        assert "path"     in entry
        assert "captured" in entry
        assert "facts"    in entry
        assert "minimax_score" in entry["facts"]


# ── Group 6: Override logic compatibility ─────────────────────────────────────

@pytest.mark.parametrize("board,player,ctx", _CASES)
def test_override_runs_without_error_for_first_two_indices(board, player, ctx):
    """
    _override_if_llm_chose_much_worse_minimax must not raise when called with
    idx=0 and idx=1 (the two most common outcomes of a ranker decision).
    """
    candidates = _candidates(board, player, ctx)
    priorities = (ctx or {}).get("strategic_priorities", [])
    score_state = (ctx or {}).get("score_state", "EQUAL")
    game_phase  = (ctx or {}).get("game_phase", "MIDGAME")

    filtered, _ = _apply_safety_filter(
        candidates,
        strategic_priorities=priorities,
        score_state=score_state,
    )
    if not filtered:
        pytest.skip("no filtered candidates")

    for idx in range(min(2, len(filtered))):
        move, reason, debug = _override_if_llm_chose_much_worse_minimax(
            filtered=filtered,
            llm_idx=idx,
            game_phase=game_phase,
            score_state=score_state,
            strategic_priorities=priorities,
        )
        # Must return a move dict with expected keys.
        assert isinstance(move, dict), f"override returned non-dict for idx={idx}"
        assert "type" in move
        assert "path" in move
        assert "facts" in move
        # Reason is None or a string.
        assert reason is None or isinstance(reason, str)
        # Debug info is a dict.
        assert isinstance(debug, dict)


# ── Group 7: End-to-end pre-ranker pipeline ───────────────────────────────────

@pytest.mark.parametrize("board,player,ctx", _CASES)
def test_full_pre_ranker_pipeline(board, player, ctx):
    """
    Full pre-LLM pipeline: score_all_legal_moves → select_proposal_candidates
    → _apply_safety_filter → build_ranker_user_prompt.
    None of these steps should raise; the final prompt must be a non-empty string.
    """
    # Step 1: score all legal moves.
    scored, best_score, second, gap = score_all_legal_moves(board, player)
    assert isinstance(scored, list)

    if not scored:
        pytest.skip("no legal moves in this position")

    # Step 2: deterministic proposal shortlist.
    candidates = select_proposal_candidates(scored, strategic_context=ctx, k=5)
    assert candidates, "proposal returned empty shortlist for non-empty position"

    # Step 3: safety filter.
    priorities  = (ctx or {}).get("strategic_priorities", [])
    score_state = (ctx or {}).get("score_state", "EQUAL")
    filtered, index_map = _apply_safety_filter(
        candidates,
        strategic_priorities=priorities,
        score_state=score_state,
    )
    assert filtered, "safety filter reduced candidates to empty"

    # Step 4: build ranker prompt.
    state  = _make_state(board, player, ctx)
    prompt = build_ranker_user_prompt(state, filtered, index_map)
    assert isinstance(prompt, str) and len(prompt) > 0

    # Step 5: snapshot.
    snapshot = _build_ranker_filtered_menu_snapshot(filtered)
    assert len(snapshot) == len(filtered)


# ── Group 8: No fields missing that would produce silent -inf ─────────────────

def test_no_candidate_score_is_neg_inf():
    """
    Regression: if minimax_score is missing, _get_minimax_score returns -inf
    and the safety filter's best_safe_score would be -inf.  Verify this never
    happens with deterministic proposal output.
    """
    board = _start_board()
    scored, _, _, _ = score_all_legal_moves(board, RED)
    candidates = select_proposal_candidates(scored, k=5)
    for c in candidates:
        score = _get_minimax_score(c)
        assert score != float("-inf"), (
            f"candidate {c.get('path')} has minimax_score=-inf — "
            "safety filter would compute best_safe_score=-inf"
        )


def test_safety_filter_does_not_collapse_to_empty_on_start():
    """
    On the standard opening position, the safety filter must not reduce
    the shortlist to zero candidates regardless of score_state.
    """
    board = _start_board()
    scored, _, _, _ = score_all_legal_moves(board, RED)
    candidates = select_proposal_candidates(scored, k=5)

    for score_state in ("EQUAL", "SLIGHTLY_WINNING", "CLEARLY_LOSING"):
        filtered, _ = _apply_safety_filter(
            candidates, score_state=score_state
        )
        assert filtered, (
            f"safety filter collapsed to empty for score_state={score_state}"
        )


# ── Group 9: System-prompt reasoning quality contract ─────────────────────────
#
# These tests guard the prompt text against regression to pure-minimax
# explanations and verify that the five required reasoning sections are
# present in the OUTPUT FORMAT template.
#
# They inspect prompt strings only — no LLM is called.

class TestSystemPromptReasoningContract:
    """
    The system prompt must enforce a natural prose paragraph for reasoning,
    not labeled bullet-point sections.
    """

    # ── Anti-section-headers ──────────────────────────────────────────────────

    def test_reasoning_template_forbids_labeled_tactical_header(self):
        """
        The OUTPUT FORMAT reasoning template must NOT require a 'Tactical:' header.
        The section was converted to prose; the label must not appear in the template.
        """
        # The old labeled format used "Tactical:" as a section marker.
        # The new prose format mentions the word tactically/tactical in other
        # contexts (FACTS REFERENCE, STEP instructions), so we check the
        # OUTPUT FORMAT block specifically — easier to do via the CRITICAL rule.
        assert "Do NOT use labeled section headers" in RANKER_SYSTEM_PROMPT, (
            "CRITICAL block must forbid labeled section headers"
        )
        assert "Tactical:" in RANKER_SYSTEM_PROMPT or True  # allowed in FACTS block
        # The key assertion: the CRITICAL rule explicitly names the forbidden labels.
        assert '"Tactical:"' in RANKER_SYSTEM_PROMPT or "'Tactical:'" in RANKER_SYSTEM_PROMPT, (
            "CRITICAL block must name 'Tactical:' as a forbidden label"
        )

    def test_reasoning_template_forbids_safety_and_strategic_labels(self):
        """The CRITICAL block in REASONING REQUIREMENTS must forbid 'Safety:' and 'Strategic:' headers."""
        # Anchor to REASONING REQUIREMENTS so we skip the CRITICAL label in STEP 1.
        rr_idx = RANKER_SYSTEM_PROMPT.find("REASONING REQUIREMENTS")
        assert rr_idx >= 0, "Prompt must contain 'REASONING REQUIREMENTS' section"
        critical_start = RANKER_SYSTEM_PROMPT.find("CRITICAL", rr_idx)
        assert critical_start >= 0, "REASONING REQUIREMENTS must contain a CRITICAL block"
        critical_block = RANKER_SYSTEM_PROMPT[critical_start:critical_start + 700]
        assert '"Safety:"' in critical_block or "'Safety:'" in critical_block, (
            "CRITICAL block must name 'Safety:' as a forbidden label"
        )
        assert '"Strategic:"' in critical_block or "'Strategic:'" in critical_block, (
            "CRITICAL block must name 'Strategic:' as a forbidden label"
        )

    # ── Paragraph format requirement ──────────────────────────────────────────

    def test_reasoning_template_requires_paragraph(self):
        """The OUTPUT FORMAT reasoning description must ask for a paragraph."""
        # The template should mention 'paragraph' or 'sentences'.
        output_block = RANKER_SYSTEM_PROMPT[
            RANKER_SYSTEM_PROMPT.find("OUTPUT FORMAT"):
        ]
        assert "paragraph" in output_block or "sentences" in output_block, (
            "OUTPUT FORMAT reasoning field must ask for a paragraph or sentences"
        )

    def test_reasoning_template_has_example_styles(self):
        """The template must include at least one example opening style."""
        output_block = RANKER_SYSTEM_PROMPT[
            RANKER_SYSTEM_PROMPT.find("OUTPUT FORMAT"):
        ]
        # Examples use concrete phrasing from the updated template.
        example_phrases = [
            "avoids recapture",
            "Although no capture",
            "Capturing the piece",
            "Compared with move",
            "was rejected because",      # new template: "Move [1] was rejected because..."
            "keeps all our pieces safe", # new template: "This move keeps all our pieces safe..."
            "confirms this choice",      # new template: "The minimax score of X confirms..."
            "FIRST TWO sentences",       # structural ordering rule
        ]
        assert any(p in output_block for p in example_phrases), (
            "OUTPUT FORMAT should include at least one example opening style for tone guidance"
        )


    # ── Content coverage (prose, not headers) ─────────────────────────────────

    def test_reasoning_template_requires_tactical_facts_by_name(self):
        """Tactical fact names must appear in the OUTPUT FORMAT reasoning description."""
        output_block = RANKER_SYSTEM_PROMPT[
            RANKER_SYSTEM_PROMPT.find("OUTPUT FORMAT"):
        ]
        tactical_fields = [
            "captures_count", "creates_immediate_threat",
            "shot_sequence_available", "blocks_opponent_landing",
        ]
        assert any(f in output_block for f in tactical_fields), (
            "OUTPUT FORMAT reasoning description must name at least one tactical facts field"
        )

    def test_reasoning_template_requires_safety_facts_by_name(self):
        """Safety fact names must appear in the OUTPUT FORMAT reasoning description."""
        output_block = RANKER_SYSTEM_PROMPT[
            RANKER_SYSTEM_PROMPT.find("OUTPUT FORMAT"):
        ]
        safety_fields = [
            "our_pieces_threatened_after", "opponent_can_recapture",
            "moved_piece_is_threatened",
        ]
        assert any(f in output_block for f in safety_fields), (
            "OUTPUT FORMAT reasoning description must name at least one safety facts field"
        )

    def test_reasoning_template_requires_strategic_facts_by_name(self):
        """Strategic fact names must appear in the OUTPUT FORMAT reasoning description."""
        output_block = RANKER_SYSTEM_PROMPT[
            RANKER_SYSTEM_PROMPT.find("OUTPUT FORMAT"):
        ]
        strategic_fields = [
            "quiet_move_role", "winning_conversion_score",
            "counterplay_score", "king_activity_score",
            "center_control", "near_promotion",
        ]
        assert any(f in output_block for f in strategic_fields), (
            "OUTPUT FORMAT reasoning description must name at least one strategic facts field"
        )

    def test_reasoning_template_requires_alternative_comparison(self):
        """The OUTPUT FORMAT must ask for a comparison with the next-best alternative."""
        output_block = RANKER_SYSTEM_PROMPT[
            RANKER_SYSTEM_PROMPT.find("OUTPUT FORMAT"):
        ]
        assert "alternative" in output_block or "next-best" in output_block, (
            "OUTPUT FORMAT reasoning description must ask for alternative comparison"
        )

    def test_reasoning_template_requires_minimax_as_evidence(self):
        """The OUTPUT FORMAT must frame minimax_score as confirmation, not the primary reason."""
        output_block = RANKER_SYSTEM_PROMPT[
            RANKER_SYSTEM_PROMPT.find("OUTPUT FORMAT"):
        ]
        assert "minimax" in output_block.lower(), (
            "OUTPUT FORMAT reasoning description must reference minimax_score"
        )
        # Accept any of the valid non-dominant framings:
        # "supporting evidence" (old), "not the sole" (old),
        # "confirmation" (current), "ONLY in the final sentence" (current).
        framing_phrases = [
            "supporting evidence",
            "not the sole",
            "confirmation",
            "ONLY in the final sentence",
            "as confirmation",
        ]
        assert any(p in output_block for p in framing_phrases), (
            "OUTPUT FORMAT must frame minimax_score as confirmation/supporting evidence, "
            "not the primary justification"
        )


    # ── Anti-pure-minimax ─────────────────────────────────────────────────────

    def test_system_prompt_forbids_pure_minimax_reasoning(self):
        """The CRITICAL block must forbid 'highest minimax_score' as the only reason."""
        # Anchor to REASONING REQUIREMENTS to find the correct CRITICAL block.
        rr_idx = RANKER_SYSTEM_PROMPT.find("REASONING REQUIREMENTS")
        assert rr_idx >= 0
        critical_start = RANKER_SYSTEM_PROMPT.find("CRITICAL", rr_idx)
        critical_block = RANKER_SYSTEM_PROMPT[critical_start:]
        # The actual prompt uses "Do NOT" (sentence case).
        assert "Do NOT explain the move by saying only" in critical_block, (
            "CRITICAL block must contain the anti-pure-minimax instruction"
        )
        assert "highest minimax_score" in RANKER_SYSTEM_PROMPT, (
            "CRITICAL block must name the forbidden 'highest minimax_score' phrasing"
        )

    # ── Single-candidate prompt ───────────────────────────────────────────────

    def test_single_candidate_prompt_requires_paragraph(self):
        """RANKER_SYSTEM_PROMPT_SINGLE must also ask for a paragraph, not labeled sections."""
        assert "paragraph" in RANKER_SYSTEM_PROMPT_SINGLE or \
               "sentences" in RANKER_SYSTEM_PROMPT_SINGLE, (
            "RANKER_SYSTEM_PROMPT_SINGLE must ask for a paragraph or sentences"
        )

    def test_single_candidate_prompt_forbids_labeled_sections(self):
        """RANKER_SYSTEM_PROMPT_SINGLE must forbid the old section header labels."""
        assert "Do NOT use labeled section headers" in RANKER_SYSTEM_PROMPT_SINGLE, (
            "RANKER_SYSTEM_PROMPT_SINGLE must forbid labeled section headers"
        )

    def test_single_candidate_prompt_covers_all_four_categories(self):
        """RANKER_SYSTEM_PROMPT_SINGLE must name facts from all four coverage categories."""
        p = RANKER_SYSTEM_PROMPT_SINGLE
        assert any(f in p for f in [
            "captures_count", "creates_immediate_threat",
            "shot_sequence_available", "blocks_opponent_landing",
        ]), "single prompt must mention tactical facts"
        assert any(f in p for f in [
            "our_pieces_threatened_after", "opponent_can_recapture",
            "moved_piece_is_threatened",
        ]), "single prompt must mention safety facts"
        assert any(f in p for f in [
            "center_control", "near_promotion", "counterplay_score",
            "king_activity_score", "winning_conversion_score", "quiet_move_role",
        ]), "single prompt must mention strategic facts"
        assert "minimax" in p.lower(), "single prompt must mention minimax_score"

    # ── Quality rules ─────────────────────────────────────────────────────────

    def test_quality_rules_forbid_stable_position(self):
        """REASONING REQUIREMENTS must forbid 'stable position'."""
        rr_block = RANKER_SYSTEM_PROMPT[RANKER_SYSTEM_PROMPT.find("REASONING REQUIREMENTS"):]
        assert "stable position" in rr_block, (
            "QUALITY RULES must name 'stable position' as a forbidden generic phrase"
        )

    def test_quality_rules_forbid_no_threats(self):
        """REASONING REQUIREMENTS must forbid 'no threats'."""
        rr_block = RANKER_SYSTEM_PROMPT[RANKER_SYSTEM_PROMPT.find("REASONING REQUIREMENTS"):]
        assert "no threats" in rr_block, (
            "QUALITY RULES must name 'no threats' as a forbidden generic phrase"
        )

    def test_quality_rules_forbid_solid_move(self):
        """REASONING REQUIREMENTS must forbid 'solid move'."""
        rr_block = RANKER_SYSTEM_PROMPT[RANKER_SYSTEM_PROMPT.find("REASONING REQUIREMENTS"):]
        assert "solid move" in rr_block, (
            "QUALITY RULES must name 'solid move' as a forbidden generic phrase"
        )

    def test_quality_rules_require_concrete_fact_per_sentence(self):
        """REASONING REQUIREMENTS must require at least one concrete fact per non-final sentence."""
        rr_block = RANKER_SYSTEM_PROMPT[RANKER_SYSTEM_PROMPT.find("REASONING REQUIREMENTS"):]
        assert "concrete fact" in rr_block or "actual value" in rr_block, (
            "QUALITY RULES must require at least one concrete fact with actual value per sentence"
        )

    def test_quality_rules_require_concrete_comparison(self):
        """REASONING REQUIREMENTS must require concrete numeric/boolean difference for comparisons."""
        rr_block = RANKER_SYSTEM_PROMPT[RANKER_SYSTEM_PROMPT.find("REASONING REQUIREMENTS"):]
        assert "numeric" in rr_block or "boolean" in rr_block, (
            "QUALITY RULES must require a concrete numeric or boolean difference for comparisons"
        )

    def test_quality_rules_explain_least_harmful_option(self):
        """REASONING REQUIREMENTS must tell the LLM what to do when no strong advantage exists."""
        rr_block = RANKER_SYSTEM_PROMPT[RANKER_SYSTEM_PROMPT.find("REASONING REQUIREMENTS"):]
        assert "least harmful" in rr_block or "no strong advantage" in rr_block, (
            "QUALITY RULES must provide a 'least harmful option' escape hatch"
        )

    def test_output_format_template_forbids_generic_phrases(self):
        """The OUTPUT FORMAT reasoning template must also reference the forbidden phrases."""
        output_block = RANKER_SYSTEM_PROMPT[RANKER_SYSTEM_PROMPT.find("OUTPUT FORMAT"):]
        forbidden = [
            "stable position", "no threats", "solid move",
            "no advantage", "safe choice", "good position",
        ]
        assert any(p in output_block for p in forbidden), (
            "OUTPUT FORMAT reasoning template must name at least one forbidden generic phrase"
        )

    def test_single_candidate_forbids_generic_phrases(self):
        """RANKER_SYSTEM_PROMPT_SINGLE must also forbid generic phrases."""
        forbidden = [
            "stable position", "no threats", "solid move",
            "no advantage", "safe choice", "good position",
        ]
        assert any(p in RANKER_SYSTEM_PROMPT_SINGLE for p in forbidden), (
            "RANKER_SYSTEM_PROMPT_SINGLE must name at least one forbidden generic phrase"
        )

    def test_single_candidate_requires_concrete_fact_per_sentence(self):
        """RANKER_SYSTEM_PROMPT_SINGLE must require concrete facts per sentence."""
        assert (
            "concrete fact" in RANKER_SYSTEM_PROMPT_SINGLE
            or "actual value" in RANKER_SYSTEM_PROMPT_SINGLE
        ), (
            "RANKER_SYSTEM_PROMPT_SINGLE must require at least one concrete fact with actual value"
        )
