# checkers/tests/test_board_proposal_standalone.py
#
# Standalone tests for board_proposal_agent — no graph, no current pipeline.
# Run with:
#   python -m pytest checkers/tests/test_board_proposal_standalone.py -v
# or directly:
#   python checkers/tests/test_board_proposal_standalone.py
#
# The LLM integration test (test_full_proposal_*) is skipped when
# MISTRAL_API_KEY is absent so unit tests always pass in CI.

from __future__ import annotations

import json
import os
import sys

# Allow running as a script from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional — env vars may already be set

from checkers.engine.board import (
    create_initial_board, RED, BLACK, RED_KING, BLACK_KING, EMPTY,
)
from checkers.state.state import CheckersState
from checkers.agents.board_proposal_agent import (
    board_proposal_agent,
    build_board_proposal_prompt,
    call_board_proposal_llm,
    parse_proposal_output,
    normalize_candidate,
    _render_board,
    _list_pieces,
    _list_empty_dark_squares,
    _count_simple_unbacked,
    _count_missed_valid_simples,
    _count_grounding_failures,
    _detect_contradictions,
    _build_retry_user_prompt,
    _filter_contradictory_proposals,
    _list_simple_geometry_targets,
    _count_missing_simple_checks_for_geometry,
    _extract_final_json_text,
    _project_missing_simples,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_state(board=None, player=BLACK, turn=1, ctx=None):
    b = board if board is not None else create_initial_board()
    return CheckersState(
        board=b,
        current_player=player,
        turn_number=turn,
        strategic_context=ctx,
    )


# ── render tests ───────────────────────────────────────────────────────────────

def test_render_board_contains_headers():
    board = create_initial_board()
    rendered = _render_board(board, BLACK)
    assert "r0" in rendered and "r7" in rendered, "Row headers missing"
    assert "c0" in rendered and "c7" in rendered, "Col headers missing"
    print("PASS test_render_board_contains_headers")


def test_render_board_has_pieces_and_empty():
    board = create_initial_board()
    rendered = _render_board(board, BLACK)
    assert " r" in rendered, "Current player pieces (r) not rendered"
    assert " b" in rendered, "Opponent pieces (b) not rendered"
    assert " ." in rendered, "Empty dark squares (.) not rendered"
    assert " #" in rendered, "Light squares (#) not rendered"
    print("PASS test_render_board_has_pieces_and_empty")


def test_render_board_perspective_flips():
    """When player is RED, their pieces should show as 'r', opponent as 'b'."""
    board = create_initial_board()
    rendered_red   = _render_board(board, RED)
    rendered_black = _render_board(board, BLACK)
    # Both renders should contain 'r' and 'b'
    assert " r" in rendered_red   and " b" in rendered_red
    assert " r" in rendered_black and " b" in rendered_black
    print("PASS test_render_board_perspective_flips")


def test_list_pieces_initial_board():
    board = create_initial_board()
    summary = _list_pieces(board, BLACK)
    assert "BLACK" in summary
    assert "RED" in summary
    assert "YOUR" in summary and "OPP" in summary
    print("PASS test_list_pieces_initial_board")


def test_list_pieces_empty_board():
    board = [[EMPTY] * 8 for _ in range(8)]
    summary = _list_pieces(board, RED)
    assert "—" in summary  # _fmt returns em-dash for empty piece lists
    print("PASS test_list_pieces_empty_board")


# ── build_board_proposal_prompt tests ─────────────────────────────────────────

def test_prompt_contains_no_legal_move_list():
    """The prompt must not contain engine-enumerated legal move indexes."""
    board = create_initial_board()
    system, user = build_board_proposal_prompt(board, BLACK)
    # Should not contain any pattern like '[0]', '[1]' that signals a pre-indexed list
    combined = system + user
    assert "MINIMAX_RANK" not in combined, "Minimax rank labels leaked into proposal prompt"
    assert "symbolic_rank" not in combined, "symbolic_rank leaked into proposal prompt"
    assert "minimax_score" not in combined, "minimax_score leaked into proposal prompt"
    assert "legal_moves:" not in combined, "legal_moves list leaked into proposal prompt"
    print("PASS test_prompt_contains_no_legal_move_list")


def test_prompt_contains_board_and_rules():
    board = create_initial_board()
    system, user = build_board_proposal_prompt(board, RED)
    assert "r0" in user,               "Board grid missing from user prompt"
    assert "Mandatory capture" in system, "Mandatory capture rule missing from system prompt"
    assert "final_proposed_moves" in system, "Output schema missing from system prompt"
    assert "Promotion row" in user,    "Promotion row hint missing from user prompt"
    print("PASS test_prompt_contains_board_and_rules")


def test_prompt_strategic_context_restricted():
    """Only game_phase and score_state are allowed in the prompt — no move hints."""
    board = create_initial_board()
    ctx = {
        "game_phase":           "MIDGAME",
        "score_state":          "EQUAL",
        # These fields must NOT appear in the prompt:
        "our_promotion_threats": 3,
        "opp_promotion_threats": 1,
        "material_advantage":    2,
        "our_vulnerable_pieces": 1,
        "strategic_priorities":  ["PROMOTE", "DEFEND_PIECES"],
    }
    system, user = build_board_proposal_prompt(board, BLACK, strategic_context=ctx)
    combined = system + user
    assert "our_promotion_threats" not in combined, "promotion_threats leaked into prompt"
    assert "our_vulnerable_pieces" not in combined, "vulnerable_pieces leaked into prompt"
    assert "material_advantage"    not in combined, "material_advantage leaked into prompt"
    assert "strategic_priorities"  not in combined, "strategic_priorities leaked into prompt"
    assert "MIDGAME" in combined,  "game_phase not present in prompt"
    assert "EQUAL"   in combined,  "score_state not present in prompt"
    print("PASS test_prompt_strategic_context_restricted")


# ── normalize_candidate tests ──────────────────────────────────────────────────

def test_normalize_valid_simple():
    raw = {"type": "simple", "path": [[5, 0], [4, 1]], "captured": []}
    result = normalize_candidate(raw)
    assert result is not None
    assert result["type"] == "simple"
    assert result["path"] == [[5, 0], [4, 1]]
    assert result["captured"] == []
    print("PASS test_normalize_valid_simple")


def test_normalize_valid_jump():
    raw = {"type": "jump", "path": [[5, 0], [3, 2]], "captured": [[4, 1]]}
    result = normalize_candidate(raw)
    assert result is not None
    assert result["type"] == "jump"
    assert result["path"] == [[5, 0], [3, 2]]
    assert result["captured"] == [[4, 1]]
    print("PASS test_normalize_valid_jump")


def test_normalize_valid_multijump():
    raw = {"type": "jump", "path": [[5, 0], [3, 2], [1, 4]], "captured": [[4, 1], [2, 3]]}
    result = normalize_candidate(raw)
    assert result is not None
    assert result["path"] == [[5, 0], [3, 2], [1, 4]]
    assert result["captured"] == [[4, 1], [2, 3]]
    print("PASS test_normalize_valid_multijump")


def test_normalize_drops_out_of_range_path():
    raw = {"type": "simple", "path": [[5, 0], [8, 1]], "captured": []}
    assert normalize_candidate(raw) is None
    print("PASS test_normalize_drops_out_of_range_path")


def test_normalize_drops_out_of_range_captured():
    raw = {"type": "jump", "path": [[5, 0], [3, 2]], "captured": [[4, 9]]}
    assert normalize_candidate(raw) is None
    print("PASS test_normalize_drops_out_of_range_captured")


def test_normalize_drops_short_path():
    raw = {"type": "simple", "path": [[5, 0]], "captured": []}
    assert normalize_candidate(raw) is None
    print("PASS test_normalize_drops_short_path")


def test_normalize_drops_non_dict():
    assert normalize_candidate("simple") is None
    assert normalize_candidate(42) is None
    assert normalize_candidate(None) is None
    assert normalize_candidate([1, 2, 3]) is None
    print("PASS test_normalize_drops_non_dict")


def test_normalize_converts_float_coords():
    """Floats like 5.0 should be accepted and converted to int."""
    raw = {"type": "simple", "path": [[5.0, 0.0], [4.0, 1.0]], "captured": []}
    result = normalize_candidate(raw)
    assert result is not None
    assert result["path"] == [[5, 0], [4, 1]]
    print("PASS test_normalize_converts_float_coords")


def test_normalize_unknown_type_becomes_simple():
    raw = {"type": "diagonal", "path": [[5, 0], [4, 1]], "captured": []}
    result = normalize_candidate(raw)
    assert result is not None
    assert result["type"] == "simple"
    print("PASS test_normalize_unknown_type_becomes_simple")


# ── parse_proposal_output tests ────────────────────────────────────────────────

def test_parse_valid_json():
    raw = json.dumps({
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 0], [4, 1]], "captured": []},
            {"type": "jump",   "path": [[5, 2], [3, 4]], "captured": [[4, 3]]},
        ]
    })
    result = parse_proposal_output(raw)
    assert len(result) == 2
    assert result[0]["type"] == "simple"
    assert result[1]["type"] == "jump"
    print("PASS test_parse_valid_json")


def test_parse_accepts_alias_key():
    """'candidates' should be accepted as alias for 'final_proposed_moves'."""
    raw = json.dumps({
        "candidates": [{"type": "simple", "path": [[5, 0], [4, 1]], "captured": []}]
    })
    result = parse_proposal_output(raw)
    assert len(result) == 1
    print("PASS test_parse_accepts_alias_key")


def test_parse_strips_markdown_fence():
    inner = json.dumps({
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 0], [4, 1]], "captured": []}
        ]
    })
    raw = f"```json\n{inner}\n```"
    result = parse_proposal_output(raw)
    assert len(result) == 1
    print("PASS test_parse_strips_markdown_fence")


def test_parse_deduplicates_by_path():
    raw = json.dumps({
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 0], [4, 1]], "captured": []},
            {"type": "simple", "path": [[5, 0], [4, 1]], "captured": []},  # duplicate
        ]
    })
    result = parse_proposal_output(raw)
    assert len(result) == 1
    print("PASS test_parse_deduplicates_by_path")


def test_parse_drops_invalid_entries_keeps_valid():
    raw = json.dumps({
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 0], [4, 1]], "captured": []},  # valid
            {"type": "simple", "path": [[5, 0], [8, 1]], "captured": []},  # out of range
            {"type": "simple", "path": [[5, 0]],          "captured": []},  # too short
            "not_a_dict",                                                   # wrong type
        ]
    })
    result = parse_proposal_output(raw)
    assert len(result) == 1
    assert result[0]["path"] == [[5, 0], [4, 1]]
    print("PASS test_parse_drops_invalid_entries_keeps_valid")


def test_parse_empty_string_returns_empty():
    assert parse_proposal_output("") == []
    print("PASS test_parse_empty_string_returns_empty")


def test_parse_broken_json_returns_empty():
    assert parse_proposal_output("not json at all") == []
    print("PASS test_parse_broken_json_returns_empty")


def test_parse_caps_at_max_candidates():
    from checkers.agents.board_proposal_agent import PROPOSAL_MAX_CANDIDATES
    moves = [
        {"type": "simple", "path": [[5, i * 2 % 8], [4, (i * 2 + 1) % 8]], "captured": []}
        for i in range(PROPOSAL_MAX_CANDIDATES + 5)
    ]
    # Remove duplicates that would be deduplicated anyway
    seen = set()
    unique_moves = []
    for m in moves:
        key = json.dumps(m["path"])
        if key not in seen:
            seen.add(key)
            unique_moves.append(m)
    raw = json.dumps({"final_proposed_moves": unique_moves})
    result = parse_proposal_output(raw)
    assert len(result) <= PROPOSAL_MAX_CANDIDATES
    print("PASS test_parse_caps_at_max_candidates")


# ── board_proposal_agent state output tests ────────────────────────────────────

def test_agent_returns_required_keys():
    """board_proposal_agent must always return the three expected state keys."""
    state = _make_state()
    # Mock the LLM call so this doesn't require API access
    import unittest.mock as mock
    mock_raw = json.dumps({
        "final_proposed_moves": [
            {"type": "simple", "path": [[0, 1], [1, 2]], "captured": []},
        ]
    })
    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        return_value=mock_raw,
    ):
        result = board_proposal_agent(state)

    assert "board_proposal_moves"       in result
    assert "board_proposal_raw"         in result
    assert "board_proposal_diagnostics" in result
    assert result["last_completed_node"] == "board_proposal_agent"
    print("PASS test_agent_returns_required_keys")


def test_agent_api_failure_returns_empty():
    """When the API fails, board_proposal_moves must be [] — not an exception."""
    import unittest.mock as mock
    state = _make_state()
    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        side_effect=ValueError("simulated API failure"),
    ):
        result = board_proposal_agent(state)

    assert result["board_proposal_moves"] == []
    assert result["board_proposal_diagnostics"]["api_call_succeeded"] is False
    assert result["board_proposal_diagnostics"]["fallback_reason"] is not None
    print("PASS test_agent_api_failure_returns_empty")


def test_agent_does_not_read_legal_moves_from_state():
    """
    Structural contract: board_proposal_agent must never touch these state fields.
    This test verifies the agent ignores engine-computed fields by injecting
    sentinel values and confirming they don't appear in the output.
    """
    board = create_initial_board()
    state = CheckersState(
        board=board,
        current_player=BLACK,
        turn_number=1,
        # Inject sentinel values into fields that must NOT be read
        legal_moves=[{"type": "SENTINEL_LEGAL", "path": [], "captured": []}],
        symbolic_best_move={"type": "SENTINEL_BEST", "path": [], "captured": []},
        symbolic_scored_moves=[{"move": {"type": "SENTINEL_SCORED"}, "minimax_score": 999.0}],
    )

    import unittest.mock as mock
    captured_prompts: list[str] = []

    def _capture_call(system: str, user: str) -> str:
        captured_prompts.append(system + "\n" + user)
        return json.dumps({"final_proposed_moves": []})

    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        side_effect=_capture_call,
    ):
        board_proposal_agent(state)

    assert captured_prompts, "LLM was never called"
    combined = captured_prompts[0]
    assert "SENTINEL_LEGAL"  not in combined, "legal_moves leaked into prompt"
    assert "SENTINEL_BEST"   not in combined, "symbolic_best_move leaked into prompt"
    assert "SENTINEL_SCORED" not in combined, "symbolic_scored_moves leaked into prompt"
    assert "999.0"           not in combined, "minimax_score leaked into prompt"
    print("PASS test_agent_does_not_read_legal_moves_from_state")


# ── LLM integration test (skipped without API key) ────────────────────────────

def _has_real_api_key() -> bool:
    """Returns True only when MISTRAL_API_KEY looks like a real key (>= 20 chars)."""
    key = os.environ.get("MISTRAL_API_KEY", "")
    return len(key) >= 20


def test_full_proposal_initial_board_llm():
    """
    Integration test: calls the real Mistral API.
    Skipped when MISTRAL_API_KEY is absent or clearly a placeholder.
    Verifies the LLM returns at least one parseable candidate for the opening position.
    """
    if not _has_real_api_key():
        print("SKIP test_full_proposal_initial_board_llm (MISTRAL_API_KEY not set)")
        return

    board = create_initial_board()
    state = _make_state(board=board, player=BLACK, turn=1)
    result = board_proposal_agent(state)

    print(f"\n[LLM integration] n_candidates={len(result['board_proposal_moves'])}")
    for i, m in enumerate(result["board_proposal_moves"]):
        print(f"  [{i}] {m}")

    assert isinstance(result["board_proposal_moves"], list)
    diag = result["board_proposal_diagnostics"]
    assert diag["api_call_succeeded"], f"API call failed: {diag.get('fallback_reason')}"
    assert diag["parse_succeeded"]
    # Opening has 7 legal moves; LLM should produce at least 2 parseable proposals
    assert diag["n_normalized"] >= 1, (
        f"Expected at least 1 proposal, got {diag['n_normalized']}"
    )
    print("PASS test_full_proposal_initial_board_llm")


def test_full_proposal_midgame_position_llm():
    """
    Integration test: calls the real Mistral API with a midgame board.
    Skipped when MISTRAL_API_KEY is absent or clearly a placeholder.
    """
    if not _has_real_api_key():
        print("SKIP test_full_proposal_midgame_position_llm (MISTRAL_API_KEY not set)")
        return

    # A simple midgame board: RED has 3 pieces, BLACK has 3 pieces
    board = [[EMPTY] * 8 for _ in range(8)]
    board[2][1] = BLACK
    board[2][3] = BLACK
    board[3][4] = BLACK
    board[5][0] = RED
    board[5][2] = RED
    board[6][5] = RED

    state = _make_state(
        board=board,
        player=RED,
        turn=15,
        ctx={"game_phase": "MIDGAME", "score_state": "EQUAL"},
    )
    result = board_proposal_agent(state)

    print(f"\n[LLM integration midgame] n_candidates={len(result['board_proposal_moves'])}")
    for i, m in enumerate(result["board_proposal_moves"]):
        print(f"  [{i}] {m}")

    assert isinstance(result["board_proposal_moves"], list)
    print("PASS test_full_proposal_midgame_position_llm")


# ── empty dark squares Phase-2 tests ─────────────────────────────────────────

def test_empty_dark_squares_initial_board():
    """Initial board: only rows 3-4 are empty → exactly 8 empty dark squares."""
    board = create_initial_board()
    result = _list_empty_dark_squares(board)
    # Parse out all [r,c] tokens
    import re
    coords = re.findall(r'\[(\d+),(\d+)\]', result)
    assert len(coords) == 8, f"Expected 8 empty dark squares, got {len(coords)}: {result}"
    rows = {int(r) for r, _ in coords}
    assert rows == {3, 4}, f"Empty squares should be in rows 3 and 4 only, got rows {rows}"
    print("PASS test_empty_dark_squares_initial_board")


def test_empty_dark_squares_excludes_occupied_and_light():
    """Occupied and light squares must never appear in the empty-squares list."""
    board = [[EMPTY] * 8 for _ in range(8)]
    board[3][0] = RED      # occupies dark square [3,0] (3+0=3 odd)
    board[4][1] = BLACK    # occupies dark square [4,1] (4+1=5 odd)
    result = _list_empty_dark_squares(board)
    import re
    coords = {(int(r), int(c)) for r, c in re.findall(r'\[(\d+),(\d+)\]', result)}
    # Occupied squares must not appear
    assert (3, 0) not in coords, "[3,0] is occupied by RED — must not be in empty list"
    assert (4, 1) not in coords, "[4,1] is occupied by BLACK — must not be in empty list"
    # Light squares (row+col even) must never appear
    for r, c in coords:
        assert (r + c) % 2 == 1, f"[{r},{c}] is a light square (r+c even) — must not be in empty list"
    print("PASS test_empty_dark_squares_excludes_occupied_and_light")


def test_empty_dark_squares_all_empty_board():
    """A fully empty board has 32 dark squares; cap shows 24 + remainder note."""
    board = [[EMPTY] * 8 for _ in range(8)]
    result = _list_empty_dark_squares(board)
    import re
    shown = re.findall(r'\[(\d+),(\d+)\]', result)
    assert len(shown) == 24, f"Expected 24 shown (cap), got {len(shown)}"
    assert "(+8 more)" in result, f"Expected '(+8 more)' remainder note, got: {result!r}"
    print("PASS test_empty_dark_squares_all_empty_board")


def test_empty_dark_squares_single_piece():
    """Board with one RED piece: 31 empty dark squares, cap shows 24 + 7 more."""
    board = [[EMPTY] * 8 for _ in range(8)]
    board[0][1] = RED    # [0,1] is dark (0+1=1 odd); now occupied
    result = _list_empty_dark_squares(board)
    import re
    shown = re.findall(r'\[(\d+),(\d+)\]', result)
    assert len(shown) == 24, f"Expected 24 shown entries, got {len(shown)}"
    assert "(+7 more)" in result, f"Expected '(+7 more)', got: {result!r}"
    assert "[0,1]" not in result.split("(")[0], "[0,1] is occupied — must not be in shown list"
    print("PASS test_empty_dark_squares_single_piece")


def test_prompt_contains_empty_dark_squares_section():
    """User prompt must contain the EMPTY PLAYABLE DARK SQUARES section."""
    board = create_initial_board()
    _system, user = build_board_proposal_prompt(board, BLACK)
    assert "EMPTY PLAYABLE DARK SQUARES" in user, \
        "EMPTY PLAYABLE DARK SQUARES section missing from user prompt"
    assert "8 squares" in user, \
        "Expected '8 squares' count in prompt for initial board"
    print("PASS test_prompt_contains_empty_dark_squares_section")


def test_prompt_empty_squares_jump_cross_check_instruction():
    """Step A must instruct using EMPTY PLAYABLE DARK SQUARES for landing cross-check."""
    board = create_initial_board()
    _system, user = build_board_proposal_prompt(board, RED)
    assert "landing cross-check" in user, \
        "'landing cross-check' instruction missing from step A"
    assert "EMPTY PLAYABLE DARK SQUARES" in user, \
        "Cross-check must reference EMPTY PLAYABLE DARK SQUARES"
    print("PASS test_prompt_empty_squares_jump_cross_check_instruction")


def test_prompt_empty_squares_simple_cross_check_instruction():
    """Step C must instruct using EMPTY PLAYABLE DARK SQUARES for simple target cross-check."""
    board = create_initial_board()
    _system, user = build_board_proposal_prompt(board, BLACK)
    assert "Target cross-check" in user, \
        "'Target cross-check' instruction missing from step C"
    print("PASS test_prompt_empty_squares_simple_cross_check_instruction")


# ── simple_checks Phase-1 tests ───────────────────────────────────────────────

def test_prompt_contains_simple_checks_schema():
    """System prompt must mention the simple_checks field in the JSON schema."""
    board = create_initial_board()
    system, user = build_board_proposal_prompt(board, RED)
    assert "simple_checks" in system, "simple_checks schema missing from system prompt"
    assert '"to_val"' in system,      "to_val field missing from simple_checks schema"
    print("PASS test_prompt_contains_simple_checks_schema")


def test_prompt_contains_simple_gate_rule():
    """System prompt must contain the SIMPLE-GATE invariant rule."""
    board = create_initial_board()
    system, _user = build_board_proposal_prompt(board, BLACK)
    assert "SIMPLE-GATE" in system, "SIMPLE-GATE invariant missing from system prompt"
    assert "valid=true" in system,  "valid=true condition missing from system prompt"
    print("PASS test_prompt_contains_simple_gate_rule")


def test_user_prompt_references_simple_checks_procedure():
    """User prompt procedure step C must reference recording simple_checks."""
    board = create_initial_board()
    _system, user = build_board_proposal_prompt(board, RED)
    assert "simple_check" in user,   "simple_check procedure missing from user prompt step C"
    assert "SIMPLE-GATE" in user,    "SIMPLE-GATE missing from user prompt step C"
    print("PASS test_user_prompt_references_simple_checks_procedure")


def test_parse_accepts_output_with_simple_checks():
    """parse_proposal_output must not choke on JSON containing a simple_checks array."""
    raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": False,
        "simple_checks": [
            {"dir": "NE", "from": [5, 2], "to": [4, 3], "to_val": ".", "valid": True,
             "reason": "target empty"},
            {"dir": "NW", "from": [5, 2], "to": [4, 1], "to_val": "b", "valid": False,
             "reason": "target occupied"},
        ],
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []},
        ],
    })
    result = parse_proposal_output(raw)
    assert len(result) == 1
    assert result[0]["type"] == "simple"
    assert result[0]["path"] == [[5, 2], [4, 3]]
    print("PASS test_parse_accepts_output_with_simple_checks")


def test_count_simple_unbacked_all_backed():
    """_count_simple_unbacked returns 0 when every simple has a matching valid=true entry."""
    proposals = [
        {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []},
        {"type": "simple", "path": [[5, 4], [4, 5]], "captured": []},
    ]
    simple_checks = [
        {"dir": "NE", "from": [5, 2], "to": [4, 3], "to_val": ".", "valid": True},
        {"dir": "NE", "from": [5, 4], "to": [4, 5], "to_val": ".", "valid": True},
    ]
    assert _count_simple_unbacked(proposals, simple_checks) == 0
    print("PASS test_count_simple_unbacked_all_backed")


def test_count_simple_unbacked_partial():
    """_count_simple_unbacked counts simples with no backing valid=true entry."""
    proposals = [
        {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []},   # backed
        {"type": "simple", "path": [[5, 4], [4, 5]], "captured": []},   # not in checks
        {"type": "jump",   "path": [[5, 6], [3, 4]], "captured": [[4, 5]]},  # ignored
    ]
    simple_checks = [
        {"dir": "NE", "from": [5, 2], "to": [4, 3], "to_val": ".", "valid": True},
        # [5,4]→[4,5] has no valid=true entry
        {"dir": "NE", "from": [5, 4], "to": [4, 5], "to_val": "b", "valid": False},
    ]
    assert _count_simple_unbacked(proposals, simple_checks) == 1
    print("PASS test_count_simple_unbacked_partial")


def test_count_simple_unbacked_no_checks_field():
    """When simple_checks is absent/None, every simple counts as unbacked."""
    proposals = [
        {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []},
        {"type": "simple", "path": [[5, 4], [4, 5]], "captured": []},
    ]
    assert _count_simple_unbacked(proposals, None)      == 2
    assert _count_simple_unbacked(proposals, "missing") == 2
    assert _count_simple_unbacked(proposals, [])        == 2
    print("PASS test_count_simple_unbacked_no_checks_field")


def test_agent_diagnostics_includes_simple_unbacked():
    """board_proposal_agent diagnostics must contain n_simple_unbacked key."""
    import unittest.mock as mock
    raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": False,
        "simple_checks": [
            {"dir": "NE", "from": [5, 2], "to": [4, 3], "to_val": ".", "valid": True,
             "reason": "target empty"},
        ],
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []},
        ],
    })
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    state = _make_state(board=board, player=RED)
    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        return_value=raw,
    ):
        result = board_proposal_agent(state)

    diag = result["board_proposal_diagnostics"]
    assert "n_simple_unbacked" in diag, "n_simple_unbacked missing from diagnostics"
    assert diag["n_simple_unbacked"] == 0, (
        f"Expected 0 unbacked simples, got {diag['n_simple_unbacked']}"
    )
    print("PASS test_agent_diagnostics_includes_simple_unbacked")


# ── source_check_id / grounding Phase-4 tests ────────────────────────────────

def test_prompt_contains_source_check_id_schema():
    """System prompt schema must include source_check_id for simple moves."""
    board = create_initial_board()
    system, _user = build_board_proposal_prompt(board, RED)
    assert "source_check_id" in system, "source_check_id missing from system prompt schema"
    print("PASS test_prompt_contains_source_check_id_schema")


def test_prompt_contains_source_check_ids_schema():
    """System prompt schema must include source_check_ids for jump moves."""
    board = create_initial_board()
    system, _user = build_board_proposal_prompt(board, BLACK)
    assert "source_check_ids" in system, "source_check_ids missing from system prompt schema"
    print("PASS test_prompt_contains_source_check_ids_schema")


def test_prompt_contains_partial_path_gate():
    """System prompt constraints must include the PARTIAL-PATH GATE rule."""
    board = create_initial_board()
    system, _user = build_board_proposal_prompt(board, RED)
    assert "PARTIAL-PATH GATE" in system, "PARTIAL-PATH GATE rule missing from system prompt"
    assert "PARTIAL" in system, "PARTIAL keyword missing from system prompt"
    print("PASS test_prompt_contains_partial_path_gate")


def test_prompt_contains_source_check_link_invariant():
    """System prompt constraints must include SOURCE-CHECK LINK INVARIANT."""
    board = create_initial_board()
    system, _user = build_board_proposal_prompt(board, RED)
    assert "SOURCE-CHECK LINK" in system, "SOURCE-CHECK LINK INVARIANT missing from system prompt"
    print("PASS test_prompt_contains_source_check_link_invariant")


def test_parse_does_not_error_on_source_ids():
    """parse_proposal_output must handle moves with source_check_ids without error.
    normalize_candidate strips extra fields — source ids are not in the output,
    but the parser must not raise on their presence."""
    raw = json.dumps({
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 2], [4, 3]], "captured": [],
             "source_check_id": "S_5_2_NE"},
            {"type": "jump",   "path": [[5, 2], [3, 4]], "captured": [[4, 3]],
             "source_check_ids": ["J_5_2_NE"]},
        ]
    })
    result = parse_proposal_output(raw)
    assert len(result) == 2
    # normalize_candidate strips source ids — they must not appear in output
    assert "source_check_id"  not in result[0]
    assert "source_check_ids" not in result[1]
    print("PASS test_parse_does_not_error_on_source_ids")


def test_count_grounding_failures_unlinked_jump():
    """Jump with no source_check_ids → unlinked_jump_count=1, bad_source=0."""
    raw_finals = [
        {"type": "jump", "path": [[5, 2], [3, 4]], "captured": [[4, 3]]},
    ]
    scan = [
        {"piece": [5, 2], "piece_type": "RED_MAN", "jump_checks": [
            {"id": "J_5_2_NE", "dir": "NE", "mid": [4, 3], "mid_val": "b",
             "land": [3, 4], "land_val": ".", "valid": True},
        ]}
    ]
    result = _count_grounding_failures(raw_finals, scan, [])
    assert result["unlinked_jump_count"]   == 1, f"Expected 1 unlinked jump, got {result}"
    assert result["bad_source_jump_count"] == 0
    print("PASS test_count_grounding_failures_unlinked_jump")


def test_count_grounding_failures_bad_source_jump():
    """Jump citing a non-existent or valid=false id → bad_source_jump_count=1."""
    raw_finals = [
        {"type": "jump", "path": [[5, 2], [3, 4]], "captured": [[4, 3]],
         "source_check_ids": ["J_5_2_WRONG"]},   # id not in valid set
    ]
    scan = [
        {"piece": [5, 2], "piece_type": "RED_MAN", "jump_checks": [
            {"id": "J_5_2_NE", "dir": "NE", "mid": [4, 3], "mid_val": "b",
             "land": [3, 4], "land_val": ".", "valid": True},
        ]}
    ]
    result = _count_grounding_failures(raw_finals, scan, [])
    assert result["unlinked_jump_count"]   == 0
    assert result["bad_source_jump_count"] == 1, f"Expected 1 bad source jump, got {result}"
    print("PASS test_count_grounding_failures_bad_source_jump")


def test_count_grounding_failures_valid_jump():
    """Properly linked jump with correct id → both jump counts = 0."""
    raw_finals = [
        {"type": "jump", "path": [[5, 2], [3, 4]], "captured": [[4, 3]],
         "source_check_ids": ["J_5_2_NE"]},
    ]
    scan = [
        {"piece": [5, 2], "piece_type": "RED_MAN", "jump_checks": [
            {"id": "J_5_2_NE", "dir": "NE", "mid": [4, 3], "mid_val": "b",
             "land": [3, 4], "land_val": ".", "valid": True},
        ]}
    ]
    result = _count_grounding_failures(raw_finals, scan, [])
    assert result["unlinked_jump_count"]   == 0
    assert result["bad_source_jump_count"] == 0, f"Expected 0 bad source jumps, got {result}"
    print("PASS test_count_grounding_failures_valid_jump")


def test_count_grounding_failures_unlinked_simple():
    """Simple with no source_check_id → unlinked_simple_count=1."""
    raw_finals = [
        {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []},
    ]
    simple_checks = [
        {"id": "S_5_2_NE", "dir": "NE", "from": [5, 2], "to": [4, 3],
         "to_val": ".", "valid": True},
    ]
    result = _count_grounding_failures(raw_finals, None, simple_checks)
    assert result["unlinked_simple_count"]   == 1, f"Expected 1 unlinked simple, got {result}"
    assert result["bad_source_simple_count"] == 0
    print("PASS test_count_grounding_failures_unlinked_simple")


def test_count_grounding_failures_bad_source_count_mismatch():
    """Jump with source_check_ids length ≠ len(captured) → bad_source_jump_count=1."""
    raw_finals = [
        {"type": "jump", "path": [[7, 0], [5, 2], [3, 4]], "captured": [[6, 1], [4, 3]],
         "source_check_ids": ["J_7_0_NE"]},   # 1 id for 2-leg jump → wrong length
    ]
    scan = [
        {"piece": [7, 0], "piece_type": "RED_MAN", "jump_checks": [
            {"id": "J_7_0_NE", "dir": "NE", "mid": [6, 1], "mid_val": "b",
             "land": [5, 2], "land_val": ".", "valid": True},
        ], "continuation_checks": [
            {"id": "C_5_2_NE_step2", "from": [5, 2], "dir": "NE", "mid": [4, 3], "mid_val": "b",
             "land": [3, 4], "land_val": ".", "valid": True},
        ]}
    ]
    result = _count_grounding_failures(raw_finals, scan, [])
    assert result["unlinked_jump_count"]   == 0
    assert result["bad_source_jump_count"] == 1, f"Expected 1 bad source (length mismatch), got {result}"
    print("PASS test_count_grounding_failures_bad_source_count_mismatch")


def test_count_grounding_failures_valid_multijump():
    """Properly linked 2-leg jump with correct ids for both legs → both counts = 0."""
    raw_finals = [
        {"type": "jump", "path": [[7, 0], [5, 2], [3, 4]], "captured": [[6, 1], [4, 3]],
         "source_check_ids": ["J_7_0_NE", "C_5_2_NE_step2"]},
    ]
    scan = [
        {"piece": [7, 0], "piece_type": "RED_MAN", "jump_checks": [
            {"id": "J_7_0_NE", "dir": "NE", "mid": [6, 1], "mid_val": "b",
             "land": [5, 2], "land_val": ".", "valid": True},
        ], "continuation_checks": [
            {"id": "C_5_2_NE_step2", "from": [5, 2], "dir": "NE", "mid": [4, 3], "mid_val": "b",
             "land": [3, 4], "land_val": ".", "valid": True},
        ]}
    ]
    result = _count_grounding_failures(raw_finals, scan, [])
    assert result["unlinked_jump_count"]   == 0
    assert result["bad_source_jump_count"] == 0, f"Expected 0 bad source for multi-jump, got {result}"
    print("PASS test_count_grounding_failures_valid_multijump")


def test_agent_diagnostics_has_grounding_fields():
    """board_proposal_agent diagnostics must contain all four grounding counters."""
    import unittest.mock as mock
    raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": False,
        "simple_checks": [
            {"id": "S_5_2_NE", "dir": "NE", "from": [5, 2], "to": [4, 3],
             "to_val": ".", "valid": True, "reason": "target empty"},
        ],
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 2], [4, 3]], "captured": [],
             "source_check_id": "S_5_2_NE"},
        ],
    })
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    state = _make_state(board=board, player=RED)
    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        return_value=raw,
    ):
        result = board_proposal_agent(state)

    diag = result["board_proposal_diagnostics"]
    assert "n_unlinked_jumps"    in diag, "n_unlinked_jumps missing from diagnostics"
    assert "n_bad_source_jumps"  in diag, "n_bad_source_jumps missing from diagnostics"
    assert "n_unlinked_simples"  in diag, "n_unlinked_simples missing from diagnostics"
    assert "n_bad_source_simples"in diag, "n_bad_source_simples missing from diagnostics"
    assert diag["n_unlinked_simples"]   == 0, "properly linked simple flagged as unlinked"
    assert diag["n_bad_source_simples"] == 0, "valid source id flagged as bad"
    print("PASS test_agent_diagnostics_has_grounding_fields")


# ── Phase-5: contradiction retry tests ───────────────────────────────────────

def test_detect_contradictions_scan_gate_violated():
    """_detect_contradictions fires when n_valid_scan_jumps=0 but jumps proposed."""
    dbg = {"capture_available_estimate": True}
    grounding = {"unlinked_jump_count": 0, "bad_source_jump_count": 0,
                 "unlinked_simple_count": 0, "bad_source_simple_count": 0}
    reasons = _detect_contradictions(dbg, n_valid_scan_jumps=0, grounding=grounding, n_jump_proposals=1)
    assert len(reasons) >= 1, f"Expected ≥1 contradiction reason, got {reasons}"
    assert any("N_VALID=0" in r or "scan" in r.lower() for r in reasons)
    print("PASS test_detect_contradictions_scan_gate_violated")


def test_detect_contradictions_no_fire_when_clean():
    """_detect_contradictions returns empty list for clean simple-only output."""
    dbg = {"capture_available_estimate": False}
    grounding = {"unlinked_jump_count": 0, "bad_source_jump_count": 0,
                 "unlinked_simple_count": 0, "bad_source_simple_count": 0}
    reasons = _detect_contradictions(dbg, n_valid_scan_jumps=0, grounding=grounding, n_jump_proposals=0)
    assert reasons == [], f"Expected no contradiction reasons, got {reasons}"
    print("PASS test_detect_contradictions_no_fire_when_clean")


def test_detect_contradictions_unlinked_jump():
    """_detect_contradictions fires for unlinked_jump_count > 0."""
    dbg = {"capture_available_estimate": True}
    grounding = {"unlinked_jump_count": 1, "bad_source_jump_count": 0,
                 "unlinked_simple_count": 0, "bad_source_simple_count": 0}
    # Even with n_valid_scan_jumps=1 (valid scan) an unlinked jump is a contradiction
    reasons = _detect_contradictions(dbg, n_valid_scan_jumps=1, grounding=grounding, n_jump_proposals=1)
    assert any("source_check_ids" in r or "unlink" in r.lower() for r in reasons), (
        f"Expected source_check_ids mention in reasons, got {reasons}"
    )
    print("PASS test_detect_contradictions_unlinked_jump")


def test_retry_triggered_when_jumps_without_scan_support():
    """First LLM response has jumps with N_VALID=0; second is clean simples → retry_count=1."""
    import unittest.mock as mock

    first_raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": True,
        "scan": [{"piece": [5, 2], "piece_type": "RED_MAN", "jump_checks": [
            {"id": "J_5_2_NE", "dir": "NE", "mid": [4, 3], "mid_val": "r",
             "land": [3, 4], "land_val": "r", "valid": False},
        ]}],
        "final_proposed_moves": [
            {"type": "jump", "path": [[5, 2], [3, 4]], "captured": [[4, 3]],
             "source_check_ids": ["J_5_2_NE"]},
        ],
    })
    second_raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": False,
        "simple_checks": [
            {"id": "S_5_2_NE", "dir": "NE", "from": [5, 2], "to": [4, 3],
             "to_val": ".", "valid": True, "reason": "target empty"},
        ],
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 2], [4, 3]], "captured": [],
             "source_check_id": "S_5_2_NE"},
        ],
    })

    call_sequence = [first_raw, second_raw]
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    state = _make_state(board=board, player=RED)

    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        side_effect=call_sequence,
    ):
        result = board_proposal_agent(state)

    diag = result["board_proposal_diagnostics"]
    assert diag["contradiction_retry_count"] == 1, (
        f"Expected contradiction_retry_count=1, got {diag['contradiction_retry_count']}"
    )
    assert isinstance(diag["contradiction_reasons"], list)
    assert len(diag["contradiction_reasons"]) >= 1
    # After retry the proposals should be the clean simples
    assert result["board_proposal_moves"][0]["type"] == "simple"
    print("PASS test_retry_triggered_when_jumps_without_scan_support")


def test_no_retry_when_output_clean():
    """Clean LLM output (simples, valid scan, N_VALID=0) → contradiction_retry_count=0."""
    import unittest.mock as mock

    clean_raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": False,
        "simple_checks": [
            {"id": "S_5_2_NE", "dir": "NE", "from": [5, 2], "to": [4, 3],
             "to_val": ".", "valid": True, "reason": "target empty"},
        ],
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 2], [4, 3]], "captured": [],
             "source_check_id": "S_5_2_NE"},
        ],
    })
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    state = _make_state(board=board, player=RED)

    call_count = [0]
    def single_call(system, user):
        call_count[0] += 1
        return clean_raw

    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        side_effect=single_call,
    ):
        result = board_proposal_agent(state)

    diag = result["board_proposal_diagnostics"]
    assert diag["contradiction_retry_count"] == 0, (
        f"Expected no retry for clean output, got retry_count={diag['contradiction_retry_count']}"
    )
    assert call_count[0] == 1, f"Expected exactly 1 LLM call, got {call_count[0]}"
    print("PASS test_no_retry_when_output_clean")


def test_retry_prompt_excludes_legal_moves():
    """The retry prompt must not contain engine legal moves or sentinel values."""
    import unittest.mock as mock

    contradiction_raw = json.dumps({
        "side_to_move": "BLACK",
        "capture_available_estimate": True,
        "scan": [{"piece": [2, 1], "piece_type": "BLACK_MAN", "jump_checks": [
            {"id": "J_2_1_SE", "dir": "SE", "mid": [3, 2], "mid_val": ".",
             "land": [4, 3], "land_val": ".", "valid": False},
        ]}],
        "final_proposed_moves": [
            {"type": "jump", "path": [[2, 1], [4, 3]], "captured": [[3, 2]],
             "source_check_ids": []},   # empty → unlinked
        ],
    })

    captured_retry_user: list[str] = []
    call_count = [0]

    def capture_call(system, user):
        call_count[0] += 1
        if call_count[0] == 1:
            return contradiction_raw
        captured_retry_user.append(user)
        return json.dumps({"final_proposed_moves": []})

    state = _make_state(player=BLACK)
    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        side_effect=capture_call,
    ):
        board_proposal_agent(state)

    assert call_count[0] == 2, f"Expected 2 LLM calls (initial + retry), got {call_count[0]}"
    assert captured_retry_user, "Retry user prompt was not captured"
    retry_prompt = captured_retry_user[0]

    # Must NOT contain legal move markers
    assert "SENTINEL_LEGAL"  not in retry_prompt
    assert "legal_moves:"    not in retry_prompt
    assert "minimax_score"   not in retry_prompt
    assert "get_all_legal"   not in retry_prompt

    # Must contain the correction feedback header
    assert "CORRECTION FEEDBACK" in retry_prompt
    print("PASS test_retry_prompt_excludes_legal_moves")


def test_retry_prompt_instructs_rebuild_simples():
    """Retry prompt must instruct LLM to rebuild simple_checks when N_VALID=0."""
    reasons = ["Your final_proposed_moves contains 2 jump(s) but N_VALID=0."]
    prompt = _build_retry_user_prompt("ORIGINAL USER PROMPT", reasons)
    assert "simple_checks" in prompt, "Retry prompt must mention rebuilding simple_checks"
    assert "NO-CAPTURE" in prompt or "no-capture" in prompt.lower(), (
        "Retry prompt must reference the no-capture branch"
    )
    assert "CORRECTION FEEDBACK" in prompt
    print("PASS test_retry_prompt_instructs_rebuild_simples")


def test_retry_prompt_warns_against_empty_output():
    """Retry prompt must explicitly forbid empty final_proposed_moves when valid checks exist."""
    reasons = ["1 jump has source_check_ids that are invalid."]
    prompt = _build_retry_user_prompt("ORIGINAL USER PROMPT", reasons)
    assert "empty" in prompt.lower(), (
        "Retry prompt must warn against empty final_proposed_moves"
    )
    assert "valid=true" in prompt, (
        "Retry prompt must reference valid=true as the gate for output"
    )
    print("PASS test_retry_prompt_warns_against_empty_output")


def test_retry_api_failure_safe_filter_still_runs():
    """
    When the retry API call fails, Phase 5b safe filter still runs on the original
    (bad) proposals and drops unverified jumps.  The original bad jump is NOT passed through.
    """
    import unittest.mock as mock

    # First response: jump with valid=False scan → N_VALID=0 AND jump proposed
    contradiction_raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": True,
        "scan": [{"piece": [5, 2], "piece_type": "RED_MAN", "jump_checks": [
            {"id": "J_5_2_NE", "dir": "NE", "mid": [4, 3], "mid_val": "r",
             "land": [3, 4], "land_val": "r", "valid": False},
        ]}],
        "final_proposed_moves": [
            {"type": "jump", "path": [[5, 2], [3, 4]], "captured": [[4, 3]],
             "source_check_ids": ["J_5_2_NE"]},
        ],
    })

    call_count = [0]
    def side_effect(system, user):
        call_count[0] += 1
        if call_count[0] == 1:
            return contradiction_raw
        raise ValueError("simulated retry failure")

    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    state = _make_state(board=board, player=RED)

    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        side_effect=side_effect,
    ):
        result = board_proposal_agent(state)

    diag = result["board_proposal_diagnostics"]
    # Retry was attempted
    assert diag["contradiction_retry_count"] == 1, (
        f"Expected retry_count=1, got {diag['contradiction_retry_count']}"
    )
    # Safe filter ran and dropped the bad jump
    assert diag["post_retry_still_contradictory"] is True, (
        "Expected post_retry_still_contradictory=True when original was bad"
    )
    assert diag["safe_rejection_count"] > 0, (
        f"Expected safe_rejection_count>0, got {diag['safe_rejection_count']}"
    )
    # No unverified jumps pass through
    returned_jumps = [m for m in result["board_proposal_moves"] if m.get("type") == "jump"]
    assert returned_jumps == [], (
        f"Expected no jumps after safe filter, got {returned_jumps}"
    )
    print("PASS test_retry_api_failure_safe_filter_still_runs")


# ── Phase-5b: post-retry safety filter tests ─────────────────────────────────

def test_filter_drops_all_jumps_when_n_valid_zero():
    """_filter_contradictory_proposals drops ALL jumps when n_valid_scan_jumps=0."""
    candidates = [
        {"type": "jump", "path": [[5, 2], [3, 4]], "captured": [[4, 3]]},
        {"type": "jump", "path": [[5, 6], [3, 4]], "captured": [[4, 5]]},
    ]
    dbg = {
        "scan": [{"piece": [5, 2], "piece_type": "RED_MAN", "jump_checks": [
            {"id": "J_5_2_NE", "dir": "NE", "mid": [4, 3], "mid_val": "r",
             "land": [3, 4], "land_val": "r", "valid": False},
        ]}],
        "simple_checks": None,
        "llm_final_raw": [
            {"type": "jump", "path": [[5, 2], [3, 4]], "captured": [[4, 3]],
             "source_check_ids": ["J_5_2_NE"]},
        ],
    }
    filtered, stats = _filter_contradictory_proposals(candidates, dbg, n_valid_scan_jumps=0)
    assert filtered == [], f"Expected empty after filter, got {filtered}"
    assert stats["dropped_unverified"] == 2
    assert stats["dropped_bad_source"] == 0
    print("PASS test_filter_drops_all_jumps_when_n_valid_zero")


def test_filter_keeps_valid_source_jump():
    """_filter_contradictory_proposals keeps a jump whose source_check_ids are valid."""
    candidates = [
        {"type": "jump", "path": [[5, 2], [3, 4]], "captured": [[4, 3]]},
    ]
    dbg = {
        "scan": [{"piece": [5, 2], "piece_type": "RED_MAN", "jump_checks": [
            {"id": "J_5_2_NE", "dir": "NE", "mid": [4, 3], "mid_val": "b",
             "land": [3, 4], "land_val": ".", "valid": True},
        ]}],
        "simple_checks": None,
        "llm_final_raw": [
            {"type": "jump", "path": [[5, 2], [3, 4]], "captured": [[4, 3]],
             "source_check_ids": ["J_5_2_NE"]},
        ],
    }
    filtered, stats = _filter_contradictory_proposals(candidates, dbg, n_valid_scan_jumps=1)
    assert len(filtered) == 1, f"Expected jump kept, got {filtered}"
    assert stats["dropped_unverified"] == 0
    assert stats["dropped_bad_source"] == 0
    print("PASS test_filter_keeps_valid_source_jump")


def test_safe_rejection_after_retry_still_contradictory():
    """
    First output bad; retry output still contains jumps with N_VALID=0.
    Agent must return no jumps and diagnostics show post_retry_still_contradictory=True.
    """
    import unittest.mock as mock

    bad_raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": True,
        "scan": [{"piece": [5, 2], "piece_type": "RED_MAN", "jump_checks": [
            {"id": "J_5_2_NE", "dir": "NE", "mid": [4, 3], "mid_val": "r",
             "land": [3, 4], "land_val": "r", "valid": False},
        ]}],
        "final_proposed_moves": [
            {"type": "jump", "path": [[5, 2], [3, 4]], "captured": [[4, 3]],
             "source_check_ids": ["J_5_2_NE"]},
        ],
    })

    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    state = _make_state(board=board, player=RED)

    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        side_effect=[bad_raw, bad_raw],   # both calls return same bad output
    ):
        result = board_proposal_agent(state)

    diag = result["board_proposal_diagnostics"]
    assert diag["contradiction_retry_count"] == 1
    assert diag["post_retry_still_contradictory"] is True, (
        "post_retry_still_contradictory should be True"
    )
    assert diag["safe_rejection_count"] > 0, (
        f"Expected safe_rejection_count>0, got {diag['safe_rejection_count']}"
    )
    # No jumps must pass through
    returned_jumps = [m for m in result["board_proposal_moves"] if m.get("type") == "jump"]
    assert returned_jumps == [], f"Unexpected jumps in output: {returned_jumps}"
    print("PASS test_safe_rejection_after_retry_still_contradictory")


def test_safe_rejection_not_triggered_when_retry_clean():
    """
    First output bad (jumps); retry output is clean simples.
    safe_rejection_count=0, post_retry_still_contradictory=False, simples returned.
    """
    import unittest.mock as mock

    bad_raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": True,
        "scan": [{"piece": [5, 2], "piece_type": "RED_MAN", "jump_checks": [
            {"id": "J_5_2_NE", "dir": "NE", "mid": [4, 3], "mid_val": "r",
             "land": [3, 4], "land_val": "r", "valid": False},
        ]}],
        "final_proposed_moves": [
            {"type": "jump", "path": [[5, 2], [3, 4]], "captured": [[4, 3]],
             "source_check_ids": ["J_5_2_NE"]},
        ],
    })
    clean_retry_raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": False,
        "scan": [],
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []},
        ],
    })

    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    state = _make_state(board=board, player=RED)

    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        side_effect=[bad_raw, clean_retry_raw],
    ):
        result = board_proposal_agent(state)

    diag = result["board_proposal_diagnostics"]
    assert diag["contradiction_retry_count"] == 1
    assert diag["post_retry_still_contradictory"] is False, (
        "post_retry_still_contradictory should be False for clean retry"
    )
    assert diag["safe_rejection_count"] == 0
    # The clean simple should pass through
    assert len(result["board_proposal_moves"]) == 1
    assert result["board_proposal_moves"][0]["type"] == "simple"
    print("PASS test_safe_rejection_not_triggered_when_retry_clean")


def test_no_filter_when_no_retry():
    """No retry triggered → post_retry_still_contradictory=False, safe_rejection_count=0."""
    import unittest.mock as mock

    clean_raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": False,
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []},
        ],
    })

    call_count = [0]
    def single_call(system, user):
        call_count[0] += 1
        return clean_raw

    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    state = _make_state(board=board, player=RED)

    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        side_effect=single_call,
    ):
        result = board_proposal_agent(state)

    diag = result["board_proposal_diagnostics"]
    assert diag["contradiction_retry_count"] == 0
    assert diag["post_retry_still_contradictory"] is False
    assert diag["safe_rejection_count"] == 0
    assert call_count[0] == 1, "Expected exactly 1 LLM call when no retry needed"
    assert len(result["board_proposal_moves"]) == 1
    print("PASS test_no_filter_when_no_retry")


# ── Phase-6: missed valid simples + no-capture empty output diagnostics ──────

def test_count_missed_valid_simples_all_present():
    """_count_missed_valid_simples returns 0 when all valid checks appear in proposals."""
    proposals = [
        {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []},
        {"type": "simple", "path": [[5, 4], [4, 5]], "captured": []},
    ]
    simple_checks = [
        {"id": "S_5_2_NE", "from": [5, 2], "to": [4, 3], "valid": True},
        {"id": "S_5_4_NE", "from": [5, 4], "to": [4, 5], "valid": True},
        {"id": "S_5_2_NW", "from": [5, 2], "to": [4, 1], "valid": False},  # invalid — not counted
    ]
    assert _count_missed_valid_simples(proposals, simple_checks) == 0
    print("PASS test_count_missed_valid_simples_all_present")


def test_count_missed_valid_simples_some_missing():
    """_count_missed_valid_simples counts valid checks absent from proposals."""
    proposals = [
        {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []},
        # [5,4]→[4,5] is in valid checks but NOT in proposals
    ]
    simple_checks = [
        {"id": "S_5_2_NE", "from": [5, 2], "to": [4, 3], "valid": True},
        {"id": "S_5_4_NE", "from": [5, 4], "to": [4, 5], "valid": True},
    ]
    assert _count_missed_valid_simples(proposals, simple_checks) == 1
    print("PASS test_count_missed_valid_simples_some_missing")


def test_count_missed_valid_simples_no_checks():
    """_count_missed_valid_simples returns 0 when simple_checks is None or empty."""
    proposals = [{"type": "simple", "path": [[5, 2], [4, 3]], "captured": []}]
    assert _count_missed_valid_simples(proposals, None)      == 0
    assert _count_missed_valid_simples(proposals, [])        == 0
    assert _count_missed_valid_simples(proposals, "invalid") == 0
    print("PASS test_count_missed_valid_simples_no_checks")


def test_agent_diagnostics_n_missed_valid_simples():
    """board_proposal_agent diagnostics must expose n_missed_valid_simples."""
    import unittest.mock as mock
    # LLM has 2 valid simple_checks but only includes 1 in final_proposed_moves
    raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": False,
        "simple_checks": [
            {"id": "S_5_2_NE", "dir": "NE", "from": [5, 2], "to": [4, 3],
             "to_val": ".", "valid": True, "reason": "target empty"},
            {"id": "S_5_2_NW", "dir": "NW", "from": [5, 2], "to": [4, 1],
             "to_val": ".", "valid": True, "reason": "target empty"},
        ],
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 2], [4, 3]], "captured": [],
             "source_check_id": "S_5_2_NE"},
            # S_5_2_NW valid but omitted → n_missed_valid_simples should be 1
        ],
    })
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    state = _make_state(board=board, player=RED)
    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        return_value=raw,
    ):
        result = board_proposal_agent(state)

    diag = result["board_proposal_diagnostics"]
    assert "n_missed_valid_simples" in diag, "n_missed_valid_simples missing from diagnostics"
    assert diag["n_missed_valid_simples"] == 1, (
        f"Expected n_missed_valid_simples=1, got {diag['n_missed_valid_simples']}"
    )
    print("PASS test_agent_diagnostics_n_missed_valid_simples")


def test_agent_diagnostics_no_capture_empty_output_flag():
    """no_capture_empty_output=True when capture_est=False, valid simples exist, but output empty."""
    import unittest.mock as mock
    raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": False,
        "simple_checks": [
            {"id": "S_5_2_NE", "dir": "NE", "from": [5, 2], "to": [4, 3],
             "to_val": ".", "valid": True, "reason": "target empty"},
        ],
        "final_proposed_moves": [],  # empty despite valid checks — the failure mode we track
    })
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    state = _make_state(board=board, player=RED)
    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        return_value=raw,
    ):
        result = board_proposal_agent(state)

    diag = result["board_proposal_diagnostics"]
    assert "no_capture_empty_output" in diag, "no_capture_empty_output missing from diagnostics"
    assert diag["no_capture_empty_output"] is True, (
        f"Expected no_capture_empty_output=True, got {diag['no_capture_empty_output']}"
    )
    print("PASS test_agent_diagnostics_no_capture_empty_output_flag")


def test_agent_diagnostics_no_capture_empty_output_false_when_has_simples():
    """no_capture_empty_output=False when final_proposed_moves has simples."""
    import unittest.mock as mock
    raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": False,
        "simple_checks": [
            {"id": "S_5_2_NE", "dir": "NE", "from": [5, 2], "to": [4, 3],
             "to_val": ".", "valid": True, "reason": "target empty"},
        ],
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 2], [4, 3]], "captured": [],
             "source_check_id": "S_5_2_NE"},
        ],
    })
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    state = _make_state(board=board, player=RED)
    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        return_value=raw,
    ):
        result = board_proposal_agent(state)

    diag = result["board_proposal_diagnostics"]
    assert diag.get("no_capture_empty_output") is False, (
        f"Expected no_capture_empty_output=False when simples present, got {diag.get('no_capture_empty_output')}"
    )
    print("PASS test_agent_diagnostics_no_capture_empty_output_false_when_has_simples")


def test_system_prompt_king_four_directions_simples():
    """System prompt must include KING SIMPLE COMPLETENESS rule for no-capture branch."""
    board = create_initial_board()
    system, _user = build_board_proposal_prompt(board, RED)
    assert "KING SIMPLE COMPLETENESS" in system, (
        "System prompt must contain KING SIMPLE COMPLETENESS rule"
    )
    # The rule should reference all 4 diagonals
    assert "NW" in system and "NE" in system and "SW" in system and "SE" in system, (
        "System prompt must list all 4 diagonal directions for king simple checks"
    )
    print("PASS test_system_prompt_king_four_directions_simples")


def test_system_prompt_no_capture_output_completeness():
    """System prompt must forbid empty final_proposed_moves when valid simples exist."""
    board = create_initial_board()
    system, _user = build_board_proposal_prompt(board, BLACK)
    assert "NO-CAPTURE OUTPUT COMPLETENESS" in system, (
        "System prompt must contain NO-CAPTURE OUTPUT COMPLETENESS rule"
    )
    assert "FORBIDDEN" in system, (
        "System prompt must use FORBIDDEN to stress the no-empty rule"
    )
    print("PASS test_system_prompt_no_capture_output_completeness")


def test_retry_prompt_specifies_king_four_directions():
    """Retry prompt no-capture branch must name all 4 diagonal directions for kings."""
    reasons = ["1 jump proposed but N_VALID=0."]
    prompt = _build_retry_user_prompt("ORIGINAL USER PROMPT", reasons)
    assert "KING" in prompt.upper(), "Retry prompt must reference KING direction requirement"
    assert "NW" in prompt, "Retry prompt must specify NW direction"
    assert "NE" in prompt, "Retry prompt must specify NE direction"
    assert "SW" in prompt, "Retry prompt must specify SW direction"
    assert "SE" in prompt, "Retry prompt must specify SE direction"
    print("PASS test_retry_prompt_specifies_king_four_directions")


def test_user_prompt_b5_king_completeness():
    """User prompt B.5 must instruct king 4-diagonal completeness after N_VALID=0."""
    board = create_initial_board()
    _system, user = build_board_proposal_prompt(board, RED)
    assert "KING SIMPLE COMPLETENESS" in user, (
        "User prompt B.5 must instruct king 4-diagonal completeness"
    )
    assert "OUTPUT COMPLETENESS" in user, (
        "User prompt B.5 must instruct output completeness (no empty output)"
    )
    assert "N_VALID_SIMPLE" in user, (
        "User prompt B.5 must define N_VALID_SIMPLE as a gate"
    )
    print("PASS test_user_prompt_b5_king_completeness")


# ── Phase-7: simple geometry targets tests ───────────────────────────────────

def test_geometry_red_man_has_nw_ne_only():
    """RED man at an interior square should produce NW and NE targets only."""
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED  # interior dark square
    result = _list_simple_geometry_targets(board, RED)
    assert "NW→[4,1]" in result, f"Expected NW→[4,1] for RED man at [5,2], got: {result}"
    assert "NE→[4,3]" in result, f"Expected NE→[4,3] for RED man at [5,2], got: {result}"
    assert "SW" not in result,   "RED man must NOT have SW direction"
    assert "SE" not in result,   "RED man must NOT have SE direction"
    print("PASS test_geometry_red_man_has_nw_ne_only")


def test_geometry_black_man_has_sw_se_only():
    """BLACK man at an interior square should produce SW and SE targets only."""
    board = [[EMPTY] * 8 for _ in range(8)]
    board[2][3] = BLACK  # interior dark square
    result = _list_simple_geometry_targets(board, BLACK)
    assert "SW→[3,2]" in result, f"Expected SW→[3,2] for BLACK man at [2,3], got: {result}"
    assert "SE→[3,4]" in result, f"Expected SE→[3,4] for BLACK man at [2,3], got: {result}"
    assert "NW" not in result,   "BLACK man must NOT have NW direction"
    assert "NE" not in result,   "BLACK man must NOT have NE direction"
    print("PASS test_geometry_black_man_has_sw_se_only")


def test_geometry_king_has_all_four_directions():
    """A king at an interior square should produce all 4 diagonal targets."""
    board = [[EMPTY] * 8 for _ in range(8)]
    board[4][3] = RED_KING  # interior dark square
    result = _list_simple_geometry_targets(board, RED)
    assert "NW→[3,2]" in result, f"Missing NW for king at [4,3]: {result}"
    assert "NE→[3,4]" in result, f"Missing NE for king at [4,3]: {result}"
    assert "SW→[5,2]" in result, f"Missing SW for king at [4,3]: {result}"
    assert "SE→[5,4]" in result, f"Missing SE for king at [4,3]: {result}"
    print("PASS test_geometry_king_has_all_four_directions")


def test_geometry_skips_out_of_bounds():
    """A piece at row 0 must not produce NW/NE targets (row -1 is out of bounds)."""
    board = [[EMPTY] * 8 for _ in range(8)]
    board[0][1] = RED  # top-left area — NW→[-1,0] and NE→[-1,2] are OOB
    result = _list_simple_geometry_targets(board, RED)
    assert "[-1," not in result, f"Out-of-bounds row -1 should not appear: {result}"
    # The piece at [0,1] has no forward (NW/NE for RED man) in-bounds targets
    assert "[0,1]" not in result or "NW" not in result.split("[0,1]")[1].split("\n")[0], (
        f"RED man at row 0 should have no NW/NE targets: {result}"
    )
    print("PASS test_geometry_skips_out_of_bounds")


def test_geometry_no_pieces_returns_empty_marker():
    """Empty board should return the no-pieces marker string."""
    board = [[EMPTY] * 8 for _ in range(8)]
    result = _list_simple_geometry_targets(board, RED)
    assert "no own pieces" in result.lower(), f"Expected no-pieces marker, got: {result}"
    print("PASS test_geometry_no_pieces_returns_empty_marker")


def test_geometry_excludes_opponent_pieces():
    """Geometry targets must only list own pieces, not opponent pieces."""
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED    # own piece
    board[2][3] = BLACK  # opponent piece
    result = _list_simple_geometry_targets(board, RED)
    assert "[5,2]" in result,  "Own RED piece must appear in geometry"
    assert "[2,3]" not in result, "Opponent BLACK piece must NOT appear in geometry"
    print("PASS test_geometry_excludes_opponent_pieces")


def test_prompt_contains_simple_geometry_targets():
    """User prompt must contain the SIMPLE GEOMETRY TARGETS section."""
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    _system, user = build_board_proposal_prompt(board, RED)
    assert "SIMPLE GEOMETRY TARGETS" in user, (
        "SIMPLE GEOMETRY TARGETS section missing from user prompt"
    )
    # Should list the RED man's targets
    assert "NW→[4,1]" in user or "NE→[4,3]" in user, (
        "Geometry targets for RED man at [5,2] missing from prompt"
    )
    print("PASS test_prompt_contains_simple_geometry_targets")


def test_prompt_geometry_targets_not_legal_moves():
    """Geometry targets section must state these are NOT legal moves."""
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    _system, user = build_board_proposal_prompt(board, RED)
    # Find the geometry section
    assert "NOT legal moves" in user, (
        "Prompt must clarify geometry targets are NOT legal moves"
    )
    print("PASS test_prompt_geometry_targets_not_legal_moves")


def test_prompt_geometry_targets_require_to_val_check():
    """Geometry section must instruct LLM to read to_val and require '.' for valid=true."""
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    _system, user = build_board_proposal_prompt(board, RED)
    assert "to_val" in user, "Prompt must reference to_val check for geometry targets"
    assert "valid=true" in user, "Prompt must state valid=true requires to_val='.'"
    print("PASS test_prompt_geometry_targets_require_to_val_check")


def test_count_missing_geometry_checks_all_present():
    """Returns 0 when all geometry target pairs appear in simple_checks."""
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED  # geometry targets: NW→[4,1], NE→[4,3]
    simple_checks = [
        {"from": [5, 2], "to": [4, 1], "valid": False, "to_val": "r"},  # present but invalid
        {"from": [5, 2], "to": [4, 3], "valid": True,  "to_val": "."},  # present and valid
    ]
    assert _count_missing_simple_checks_for_geometry(board, RED, simple_checks) == 0
    print("PASS test_count_missing_geometry_checks_all_present")


def test_count_missing_geometry_checks_some_missing():
    """Returns count of geometry targets with no corresponding simple_checks entry."""
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED  # geometry: NW→[4,1], NE→[4,3]
    simple_checks = [
        {"from": [5, 2], "to": [4, 3], "valid": True, "to_val": "."},
        # [5,2]→[4,1] is missing → count=1
    ]
    assert _count_missing_simple_checks_for_geometry(board, RED, simple_checks) == 1
    print("PASS test_count_missing_geometry_checks_some_missing")


def test_count_missing_geometry_checks_no_simple_checks():
    """Returns 0 when simple_checks is None (not yet produced by LLM)."""
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    assert _count_missing_simple_checks_for_geometry(board, RED, None) == 0
    assert _count_missing_simple_checks_for_geometry(board, RED, [])   == 0
    print("PASS test_count_missing_geometry_checks_no_simple_checks")


def test_agent_diagnostics_geometry_counts():
    """board_proposal_agent must include simple_geometry_targets_count and
    missing_simple_checks_for_geometry_count in diagnostics."""
    import unittest.mock as mock
    raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": False,
        "simple_checks": [
            {"id": "S_5_2_NE", "dir": "NE", "from": [5, 2], "to": [4, 3],
             "to_val": ".", "valid": True, "reason": "target empty"},
            # NW→[4,1] is missing from simple_checks → missing_count=1
        ],
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 2], [4, 3]], "captured": [],
             "source_check_id": "S_5_2_NE"},
        ],
    })
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    state = _make_state(board=board, player=RED)
    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        return_value=raw,
    ):
        result = board_proposal_agent(state)

    diag = result["board_proposal_diagnostics"]
    assert "simple_geometry_targets_count" in diag, (
        "simple_geometry_targets_count missing from diagnostics"
    )
    assert "missing_simple_checks_for_geometry_count" in diag, (
        "missing_simple_checks_for_geometry_count missing from diagnostics"
    )
    # RED man at [5,2]: 2 geometry targets (NW→[4,1], NE→[4,3])
    assert diag["simple_geometry_targets_count"] == 2, (
        f"Expected 2 geometry targets for one RED man, got {diag['simple_geometry_targets_count']}"
    )
    # NW→[4,1] is missing from simple_checks
    assert diag["missing_simple_checks_for_geometry_count"] == 1, (
        f"Expected 1 missing geometry check, got {diag['missing_simple_checks_for_geometry_count']}"
    )
    print("PASS test_agent_diagnostics_geometry_counts")


def test_retry_prompt_references_geometry_targets():
    """Retry prompt must reference SIMPLE GEOMETRY TARGETS checklist."""
    reasons = ["1 jump proposed but N_VALID=0."]
    prompt = _build_retry_user_prompt("ORIGINAL USER PROMPT", reasons)
    assert "SIMPLE GEOMETRY TARGETS" in prompt, (
        "Retry prompt must reference SIMPLE GEOMETRY TARGETS"
    )
    print("PASS test_retry_prompt_references_geometry_targets")


# ── Phase-6 reason-first output format tests ─────────────────────────────────

def test_extract_json_after_marker():
    """_extract_final_json_text extracts JSON text after <FINAL_JSON> marker."""
    raw = ('DRAFT_BOARD_REASONING:\n  N_VALID=0, no captures.\n\n'
           '<FINAL_JSON>\n{"final_proposed_moves": []}')
    text, marker_found, draft_present = _extract_final_json_text(raw)
    assert marker_found is True
    assert draft_present is True
    assert text.startswith("{")
    assert "final_proposed_moves" in text
    print("PASS test_extract_json_after_marker")


def test_extract_json_ignores_draft_before_marker():
    """_extract_final_json_text does not include draft text in returned json_text."""
    raw = 'DRAFT_BOARD_REASONING:\n  reasoning here\n<FINAL_JSON>\n{"final_proposed_moves": []}'
    text, marker_found, draft_present = _extract_final_json_text(raw)
    assert marker_found is True
    assert "reasoning here" not in text
    assert "DRAFT_BOARD_REASONING" not in text
    print("PASS test_extract_json_ignores_draft_before_marker")


def test_extract_json_strips_fences_after_marker():
    """_extract_final_json_text strips markdown fences after the marker."""
    raw = '<FINAL_JSON>\n```json\n{"final_proposed_moves": []}\n```'
    text, marker_found, draft_present = _extract_final_json_text(raw)
    assert marker_found is True
    assert "```" not in text
    assert text.startswith("{")
    print("PASS test_extract_json_strips_fences_after_marker")


def test_extract_json_no_marker_fallback():
    """_extract_final_json_text returns full text when no marker (backward compat)."""
    raw = '{"final_proposed_moves": []}'
    text, marker_found, draft_present = _extract_final_json_text(raw)
    assert marker_found is False
    assert draft_present is False
    assert text == raw.strip()
    print("PASS test_extract_json_no_marker_fallback")


def test_extract_json_marker_found_flag():
    """marker_found flag correctly reports presence/absence of <FINAL_JSON>."""
    _, found_with,    _ = _extract_final_json_text('<FINAL_JSON>\n{}')
    _, found_without, _ = _extract_final_json_text('{}')
    assert found_with    is True
    assert found_without is False
    print("PASS test_extract_json_marker_found_flag")


def test_extract_json_draft_present_flag():
    """draft_present flag correctly detects DRAFT_BOARD_REASONING prefix."""
    _, _, draft_with    = _extract_final_json_text('DRAFT_BOARD_REASONING:\n  stuff\n<FINAL_JSON>\n{}')
    _, _, draft_without = _extract_final_json_text('<FINAL_JSON>\n{}')
    assert draft_with    is True
    assert draft_without is False
    print("PASS test_extract_json_draft_present_flag")


def test_parse_extracts_after_marker():
    """parse_proposal_output correctly parses JSON after <FINAL_JSON> marker."""
    raw = ('DRAFT_BOARD_REASONING:\n  N_VALID=0, no captures.\n<FINAL_JSON>\n'
           + json.dumps({
               "final_proposed_moves": [
                   {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []}
               ]
           }))
    result = parse_proposal_output(raw)
    assert len(result) == 1
    assert result[0]["type"] == "simple"
    print("PASS test_parse_extracts_after_marker")


def test_parse_fallback_still_works():
    """parse_proposal_output still parses old pure-JSON output without any marker."""
    raw = json.dumps({
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 2], [4, 3]], "captured": []}
        ]
    })
    result = parse_proposal_output(raw)
    assert len(result) == 1
    assert result[0]["type"] == "simple"
    print("PASS test_parse_fallback_still_works")


def test_parse_malformed_final_json_returns_empty():
    """parse_proposal_output returns [] when JSON after <FINAL_JSON> is malformed."""
    raw = '<FINAL_JSON>\nnot valid json at all'
    result = parse_proposal_output(raw)
    assert result == [], f"Expected [] for malformed JSON after marker, got {result}"
    print("PASS test_parse_malformed_final_json_returns_empty")


def test_prompt_reason_first_contains_draft_reasoning():
    """When reason_first=True, system prompt instructs LLM to write DRAFT_BOARD_REASONING."""
    board = create_initial_board()
    system, _user = build_board_proposal_prompt(board, RED, reason_first=True)
    assert "DRAFT_BOARD_REASONING" in system, (
        "System prompt must contain DRAFT_BOARD_REASONING in reason-first mode"
    )
    print("PASS test_prompt_reason_first_contains_draft_reasoning")


def test_prompt_reason_first_contains_final_json_marker():
    """When reason_first=True, system prompt requires <FINAL_JSON> tag."""
    board = create_initial_board()
    system, user = build_board_proposal_prompt(board, RED, reason_first=True)
    assert "<FINAL_JSON>" in system, (
        "System prompt must contain <FINAL_JSON> instruction in reason-first mode"
    )
    assert "<FINAL_JSON>" in user, (
        "User prompt must contain <FINAL_JSON> reminder in reason-first mode"
    )
    print("PASS test_prompt_reason_first_contains_final_json_marker")


def test_agent_diagnostics_has_simple_completeness_fields():
    """board_proposal_agent must include the three simple-completeness diagnostic fields.

    After projection: the missing valid simple_check is added → completeness=1.0.
    The field values reflect the post-projection state.
    """
    import unittest.mock as mock
    # LLM has 2 valid simple_checks; only 1 appears in final_proposed_moves
    raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": False,
        "simple_checks": [
            {"id": "S_5_2_NW", "dir": "NW", "from": [5, 2], "to": [4, 1],
             "to_val": ".", "valid": True, "reason": "target empty"},
            {"id": "S_5_2_NE", "dir": "NE", "from": [5, 2], "to": [4, 3],
             "to_val": ".", "valid": True, "reason": "target empty"},
        ],
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 2], [4, 1]], "captured": [],
             "source_check_id": "S_5_2_NW"},
            # S_5_2_NE omitted by LLM → projection adds it
        ],
    })
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    state = _make_state(board=board, player=RED)
    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        return_value=raw,
    ):
        result = board_proposal_agent(state)

    diag = result["board_proposal_diagnostics"]
    assert "valid_simple_checks_count" in diag, "valid_simple_checks_count missing"
    assert "missing_final_moves_from_valid_simple_checks_count" in diag, (
        "missing_final_moves_from_valid_simple_checks_count missing"
    )
    assert "final_simple_completeness_rate" in diag, "final_simple_completeness_rate missing"
    assert "projected_missing_simple_count" in diag, "projected_missing_simple_count missing"
    assert "simple_projection_applied" in diag, "simple_projection_applied missing"
    assert diag["valid_simple_checks_count"] == 2, (
        f"Expected 2 valid simple checks, got {diag['valid_simple_checks_count']}"
    )
    # Projection adds the missing S_5_2_NE → post-projection completeness is full
    assert diag["projected_missing_simple_count"] == 1, (
        f"Expected 1 projected, got {diag['projected_missing_simple_count']}"
    )
    assert diag["simple_projection_applied"] is True, "simple_projection_applied should be True"
    assert diag["missing_final_moves_from_valid_simple_checks_count"] == 0, (
        f"After projection: expected 0 missing, got "
        f"{diag['missing_final_moves_from_valid_simple_checks_count']}"
    )
    assert diag["final_simple_completeness_rate"] == 1.0, (
        f"After projection: expected rate=1.0, got {diag['final_simple_completeness_rate']}"
    )
    # Projected move appears in output
    output_paths = [m["path"] for m in result["board_proposal_moves"]]
    assert [[5, 2], [4, 3]] in output_paths, (
        f"Projected move [[5,2],[4,3]] not in output: {output_paths}"
    )
    print("PASS test_agent_diagnostics_has_simple_completeness_fields")


def test_agent_diagnostics_full_completeness_rate():
    """final_simple_completeness_rate=1.0 when all valid simple_checks appear in output."""
    import unittest.mock as mock
    raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": False,
        "simple_checks": [
            {"id": "S_5_2_NW", "dir": "NW", "from": [5, 2], "to": [4, 1],
             "to_val": ".", "valid": True, "reason": "target empty"},
            {"id": "S_5_2_NE", "dir": "NE", "from": [5, 2], "to": [4, 3],
             "to_val": ".", "valid": True, "reason": "target empty"},
        ],
        "final_proposed_moves": [
            {"type": "simple", "path": [[5, 2], [4, 1]], "captured": [],
             "source_check_id": "S_5_2_NW"},
            {"type": "simple", "path": [[5, 2], [4, 3]], "captured": [],
             "source_check_id": "S_5_2_NE"},
        ],
    })
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    state = _make_state(board=board, player=RED)
    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        return_value=raw,
    ):
        result = board_proposal_agent(state)

    diag = result["board_proposal_diagnostics"]
    assert diag["final_simple_completeness_rate"] == 1.0, (
        f"Expected rate=1.0, got {diag['final_simple_completeness_rate']}"
    )
    assert diag["missing_final_moves_from_valid_simple_checks_count"] == 0
    print("PASS test_agent_diagnostics_full_completeness_rate")


def test_prompt_n_valid_simple_completeness_rule():
    """User prompt B.5 must describe the N_VALID_SIMPLE enumeration + verification step."""
    board = create_initial_board()
    _system, user = build_board_proposal_prompt(board, RED)
    assert "N_VALID_SIMPLE" in user, "Prompt must define N_VALID_SIMPLE"
    assert "valid_pairs" in user or "valid=true pair" in user, (
        "Prompt must instruct explicit enumeration of valid pairs"
    )
    assert "All N_VALID_SIMPLE pairs covered" in user or "all covered? yes" in user.lower(), (
        "Prompt must include a final completeness confirmation step"
    )
    print("PASS test_prompt_n_valid_simple_completeness_rule")


def test_prompt_king_nw_ne_sw_se_order():
    """User prompt must explicitly state KING direction order NW → NE → SW → SE."""
    board = create_initial_board()
    _system, user = build_board_proposal_prompt(board, RED)
    assert "NW → NE → SW → SE" in user, (
        "User prompt must specify KING direction order NW → NE → SW → SE"
    )
    print("PASS test_prompt_king_nw_ne_sw_se_order")


def test_system_prompt_no_capture_completeness_per_entry():
    """System prompt must forbid omitting any single valid=true simple_check from output."""
    board = create_initial_board()
    system, _user = build_board_proposal_prompt(board, RED)
    assert "Omitting ANY valid=true simple_check" in system or "FORBIDDEN" in system, (
        "System prompt must explicitly forbid omitting any valid=true simple_check"
    )
    assert "SW" in system and "SE" in system, (
        "System prompt NO-CAPTURE rule must mention SW and SE directions"
    )
    print("PASS test_system_prompt_no_capture_completeness_per_entry")


def test_reason_first_draft_reasoning_has_completeness_check():
    """When reason_first=True, system prompt must include COMPLETENESS CHECK step."""
    board = create_initial_board()
    system, _user = build_board_proposal_prompt(board, RED, reason_first=True)
    assert "COMPLETENESS CHECK" in system, (
        "System prompt must include COMPLETENESS CHECK step in reason-first DRAFT_BOARD_REASONING"
    )
    assert "N_VALID_SIMPLE" in system, (
        "COMPLETENESS CHECK must reference N_VALID_SIMPLE"
    )
    print("PASS test_reason_first_draft_reasoning_has_completeness_check")


def test_agent_diagnostics_has_parse_fields():
    """board_proposal_agent diagnostics must include the four reason-first parse fields."""
    import unittest.mock as mock
    raw = json.dumps({
        "side_to_move": "RED",
        "capture_available_estimate": False,
        "simple_checks": [],
        "final_proposed_moves": [],
    })
    board = [[EMPTY] * 8 for _ in range(8)]
    board[5][2] = RED
    state = _make_state(board=board, player=RED)
    with mock.patch(
        "checkers.agents.board_proposal_agent.call_board_proposal_llm",
        return_value=raw,
    ):
        result = board_proposal_agent(state)

    diag = result["board_proposal_diagnostics"]
    assert "final_json_marker_found" in diag, "final_json_marker_found missing from diagnostics"
    assert "draft_reasoning_present" in diag, "draft_reasoning_present missing from diagnostics"
    assert "json_extraction_used"    in diag, "json_extraction_used missing from diagnostics"
    assert "parse_fallback_used"     in diag, "parse_fallback_used missing from diagnostics"
    # Old-format response → no marker → fallback path
    assert diag["final_json_marker_found"] is False
    assert diag["parse_fallback_used"]     is True
    assert diag["json_extraction_used"]    is False
    print("PASS test_agent_diagnostics_has_parse_fields")


# ── Phase-9: simple projection tests ─────────────────────────────────────────

def test_project_missing_simples_adds_missing():
    """Projection adds moves for valid=true simple_checks not yet in candidates.

    Mirrors the all_kings_endgame_no_capture failure: 12 valid checks,
    10 in output — projection must supply the 2 missing ones.
    """
    # Build 12 valid simple_checks (3 kings × 4 directions)
    simple_checks = []
    pieces = [(1, 2), (3, 4), (5, 6)]
    dirs = [("NW", -1, -1), ("NE", -1, +1), ("SW", +1, -1), ("SE", +1, +1)]
    for pr, pc in pieces:
        for dname, dr, dc in dirs:
            simple_checks.append({
                "id":    f"S_{pr}_{pc}_{dname}",
                "dir":   dname,
                "from":  [pr, pc],
                "to":    [pr + dr, pc + dc],
                "to_val": ".",
                "valid": True,
            })

    # Candidates: 10 of the 12 — missing the SW and SE from king at [5,6]
    candidates = [
        {"type": "simple", "path": [[pr, pc], [pr + dr, pc + dc]], "captured": []}
        for pr, pc in pieces
        for dname, dr, dc in dirs
        if not (pr == 5 and pc == 6 and dname in ("SW", "SE"))
    ]
    assert len(candidates) == 10, f"Expected 10 pre-projection candidates, got {len(candidates)}"

    result, n_added = _project_missing_simples(candidates, simple_checks, 0, False)
    assert n_added == 2, f"Expected 2 projected, got {n_added}"
    assert len(result) == 12, f"Expected 12 total, got {len(result)}"
    output_paths = [m["path"] for m in result]
    assert [[5, 6], [6, 5]] in output_paths, "Missing projected SW from [5,6]"
    assert [[5, 6], [6, 7]] in output_paths, "Missing projected SE from [5,6]"
    # Projected moves carry source_check_id from the check
    proj_sw = next(m for m in result if m["path"] == [[5, 6], [6, 5]])
    assert proj_sw.get("source_check_id") == "S_5_6_SW", (
        f"Expected source_check_id='S_5_6_SW', got {proj_sw.get('source_check_id')}"
    )
    print("PASS test_project_missing_simples_adds_missing")


def test_project_missing_simples_no_add_when_all_present():
    """Projection adds nothing when all valid=true checks are already in candidates."""
    simple_checks = [
        {"id": "S_5_6_SW", "from": [5, 6], "to": [6, 5], "to_val": ".", "valid": True},
        {"id": "S_5_6_SE", "from": [5, 6], "to": [6, 7], "to_val": ".", "valid": True},
    ]
    candidates = [
        {"type": "simple", "path": [[5, 6], [6, 5]], "captured": []},
        {"type": "simple", "path": [[5, 6], [6, 7]], "captured": []},
    ]
    result, n_added = _project_missing_simples(candidates, simple_checks, 0, False)
    assert n_added == 0, f"Expected 0 added, got {n_added}"
    assert len(result) == 2, f"Expected 2 unchanged, got {len(result)}"
    print("PASS test_project_missing_simples_no_add_when_all_present")


def test_project_missing_simples_skips_capture_branch():
    """Projection must not run when the scan has valid jumps or capture_est=True."""
    simple_checks = [
        {"id": "S_5_6_SW", "from": [5, 6], "to": [6, 5], "to_val": ".", "valid": True},
    ]
    candidates: list = []

    # Case A: n_valid_scan_jumps > 0
    result_a, n_a = _project_missing_simples(candidates, simple_checks, 1, False)
    assert n_a == 0, f"Projection must not run when n_valid_scan_jumps=1, got n_added={n_a}"
    assert result_a == [], "Candidates must be unchanged"

    # Case B: capture_est=True
    result_b, n_b = _project_missing_simples(candidates, simple_checks, 0, True)
    assert n_b == 0, f"Projection must not run when capture_est=True, got n_added={n_b}"
    assert result_b == [], "Candidates must be unchanged"

    # Case C: jump already in candidates
    jump_cands = [{"type": "jump", "path": [[3, 4], [1, 6]], "captured": [[2, 5]]}]
    result_c, n_c = _project_missing_simples(jump_cands, simple_checks, 0, False)
    assert n_c == 0, f"Projection must not run when jump in candidates, got n_added={n_c}"
    assert result_c == jump_cands, "Candidates with jump must be returned unchanged"
    print("PASS test_project_missing_simples_skips_capture_branch")


def test_project_missing_simples_no_engine_access():
    """_project_missing_simples must not accept board or engine parameters (isolation check)."""
    import inspect
    sig = inspect.signature(_project_missing_simples)
    param_names = list(sig.parameters.keys())
    assert "board" not in param_names, (
        f"_project_missing_simples must not take a 'board' parameter, got {param_names}"
    )
    assert "get_all_legal_moves" not in param_names, (
        "_project_missing_simples must not take an engine parameter"
    )
    # Verify the function only projects from its explicit simple_checks argument
    simple_checks = [
        {"id": "S_5_6_SW", "from": [5, 6], "to": [6, 5], "to_val": ".", "valid": True},
    ]
    # Passes no board — projection succeeds from simple_checks alone
    result, n_added = _project_missing_simples([], simple_checks, 0, False)
    assert n_added == 1, "Should project 1 move from simple_checks without needing a board"
    assert result[0]["path"] == [[5, 6], [6, 5]]
    print("PASS test_project_missing_simples_no_engine_access")


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("board_proposal_agent — unit tests")
    print("=" * 60)

    # Render tests
    test_render_board_contains_headers()
    test_render_board_has_pieces_and_empty()
    test_render_board_perspective_flips()
    test_list_pieces_initial_board()
    test_list_pieces_empty_board()

    # Prompt builder tests
    test_prompt_contains_no_legal_move_list()
    test_prompt_contains_board_and_rules()
    test_prompt_strategic_context_restricted()

    # normalize_candidate tests
    test_normalize_valid_simple()
    test_normalize_valid_jump()
    test_normalize_valid_multijump()
    test_normalize_drops_out_of_range_path()
    test_normalize_drops_out_of_range_captured()
    test_normalize_drops_short_path()
    test_normalize_drops_non_dict()
    test_normalize_converts_float_coords()
    test_normalize_unknown_type_becomes_simple()

    # parse_proposal_output tests
    test_parse_valid_json()
    test_parse_accepts_alias_key()
    test_parse_strips_markdown_fence()
    test_parse_deduplicates_by_path()
    test_parse_drops_invalid_entries_keeps_valid()
    test_parse_empty_string_returns_empty()
    test_parse_broken_json_returns_empty()
    test_parse_caps_at_max_candidates()

    # board_proposal_agent state tests (use mocks — no API needed)
    test_agent_returns_required_keys()
    test_agent_api_failure_returns_empty()
    test_agent_does_not_read_legal_moves_from_state()

    # Phase-2 empty dark squares tests
    test_empty_dark_squares_initial_board()
    test_empty_dark_squares_excludes_occupied_and_light()
    test_empty_dark_squares_all_empty_board()
    test_empty_dark_squares_single_piece()
    test_prompt_contains_empty_dark_squares_section()
    test_prompt_empty_squares_jump_cross_check_instruction()
    test_prompt_empty_squares_simple_cross_check_instruction()

    # Phase-1 simple_checks tests
    test_prompt_contains_simple_checks_schema()
    test_prompt_contains_simple_gate_rule()
    test_user_prompt_references_simple_checks_procedure()
    test_parse_accepts_output_with_simple_checks()
    test_count_simple_unbacked_all_backed()
    test_count_simple_unbacked_partial()
    test_count_simple_unbacked_no_checks_field()
    test_agent_diagnostics_includes_simple_unbacked()

    # Phase-4 source_check_id / grounding tests
    test_prompt_contains_source_check_id_schema()
    test_prompt_contains_source_check_ids_schema()
    test_prompt_contains_partial_path_gate()
    test_prompt_contains_source_check_link_invariant()
    test_parse_does_not_error_on_source_ids()
    test_count_grounding_failures_unlinked_jump()
    test_count_grounding_failures_bad_source_jump()
    test_count_grounding_failures_valid_jump()
    test_count_grounding_failures_unlinked_simple()
    test_count_grounding_failures_bad_source_count_mismatch()
    test_count_grounding_failures_valid_multijump()
    test_agent_diagnostics_has_grounding_fields()

    # Phase-5 contradiction retry tests
    test_detect_contradictions_scan_gate_violated()
    test_detect_contradictions_no_fire_when_clean()
    test_detect_contradictions_unlinked_jump()
    test_retry_triggered_when_jumps_without_scan_support()
    test_no_retry_when_output_clean()
    test_retry_prompt_excludes_legal_moves()
    test_retry_prompt_instructs_rebuild_simples()
    test_retry_prompt_warns_against_empty_output()
    test_retry_api_failure_safe_filter_still_runs()

    # Phase-5b post-retry safety filter tests
    test_filter_drops_all_jumps_when_n_valid_zero()
    test_filter_keeps_valid_source_jump()
    test_safe_rejection_after_retry_still_contradictory()
    test_safe_rejection_not_triggered_when_retry_clean()
    test_no_filter_when_no_retry()

    # Phase-6 missed valid simples + no-capture empty output diagnostics
    test_count_missed_valid_simples_all_present()
    test_count_missed_valid_simples_some_missing()
    test_count_missed_valid_simples_no_checks()
    test_agent_diagnostics_n_missed_valid_simples()
    test_agent_diagnostics_no_capture_empty_output_flag()
    test_agent_diagnostics_no_capture_empty_output_false_when_has_simples()
    test_system_prompt_king_four_directions_simples()
    test_system_prompt_no_capture_output_completeness()
    test_retry_prompt_specifies_king_four_directions()
    test_user_prompt_b5_king_completeness()

    # Phase-7 simple geometry targets tests
    test_geometry_red_man_has_nw_ne_only()
    test_geometry_black_man_has_sw_se_only()
    test_geometry_king_has_all_four_directions()
    test_geometry_skips_out_of_bounds()
    test_geometry_no_pieces_returns_empty_marker()
    test_geometry_excludes_opponent_pieces()
    test_prompt_contains_simple_geometry_targets()
    test_prompt_geometry_targets_not_legal_moves()
    test_prompt_geometry_targets_require_to_val_check()
    test_count_missing_geometry_checks_all_present()
    test_count_missing_geometry_checks_some_missing()
    test_count_missing_geometry_checks_no_simple_checks()
    test_agent_diagnostics_geometry_counts()
    test_retry_prompt_references_geometry_targets()

    # Phase-8 simple completeness diagnostics + prompt strengthening
    test_agent_diagnostics_has_simple_completeness_fields()
    test_agent_diagnostics_full_completeness_rate()
    test_prompt_n_valid_simple_completeness_rule()
    test_prompt_king_nw_ne_sw_se_order()
    test_system_prompt_no_capture_completeness_per_entry()
    test_reason_first_draft_reasoning_has_completeness_check()

    # Phase-6 reason-first output format tests
    test_extract_json_after_marker()
    test_extract_json_ignores_draft_before_marker()
    test_extract_json_strips_fences_after_marker()
    test_extract_json_no_marker_fallback()
    test_extract_json_marker_found_flag()
    test_extract_json_draft_present_flag()
    test_parse_extracts_after_marker()
    test_parse_fallback_still_works()
    test_parse_malformed_final_json_returns_empty()
    test_prompt_reason_first_contains_draft_reasoning()
    test_prompt_reason_first_contains_final_json_marker()
    test_agent_diagnostics_has_parse_fields()

    # Phase-9 simple projection tests
    test_project_missing_simples_adds_missing()
    test_project_missing_simples_no_add_when_all_present()
    test_project_missing_simples_skips_capture_branch()
    test_project_missing_simples_no_engine_access()

    print("\n" + "=" * 60)
    print("Unit tests complete.")
    print("=" * 60)

    # LLM integration tests (require MISTRAL_API_KEY)
    print("\nLLM integration tests:")
    test_full_proposal_initial_board_llm()
    test_full_proposal_midgame_position_llm()
