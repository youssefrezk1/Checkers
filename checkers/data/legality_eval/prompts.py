"""
checkers/data/legality_eval/prompts.py
======================================
Prompt templates for the three legality-stress baselines.

B1 — board + side + legend only.
B2 — B1 + full American Checkers ruleset (7 rules).
B3 — B2 + structured natural-language legality checklist (8 items, no math).

Output schema (all three baselines)
-------------------------------------
The LLM selects ONE move and returns:

  {
    "selected_move": [[row, col], [row, col], ...],
    "reasoning": "brief explanation"
  }

"selected_move" is a path (list of [row, col] pairs).
  Simple move : 2 entries  [[r_from, c_from], [r_to, c_to]]
  Single jump : 2 entries  [[r_from, c_from], [r_landing, c_landing]]
  Multi-jump  : 3+ entries [[r_from, c_from], [land1], [land2], ...]

Privacy guarantee
-----------------
  hidden_legal_moves MUST NOT appear in any prompt string.
  The evaluator loads them from JSONL AFTER the LLM responds and
  checks whether selected_move is in the legal set.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Board renderer
# ---------------------------------------------------------------------------

_EMPTY    = 0
_RED      = 1
_BLACK    = 2
_RED_KING = 3
_BLACK_K  = 4

_SYMBOLS = {_EMPTY: ".", _RED: "r", _BLACK: "b", _RED_KING: "R", _BLACK_K: "B"}


def render_board(board: list[list[int]]) -> str:
    """Compact text board suitable for LLM prompts."""
    lines = ["  " + " ".join(str(c) for c in range(8))]
    for row in range(8):
        line = str(row) + " " + " ".join(
            _SYMBOLS.get(board[row][col], "?") for col in range(8)
        )
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_OUTPUT_SPEC = """\
OUTPUT — respond with valid JSON only, no prose before or after:
{
  "selected_move": [[row, col], [row, col], ...],
  "reasoning": "brief explanation"
}

Path encoding:
  Simple move : [[r_from, c_from], [r_to,      c_to     ]]  (2 entries)
  Single jump : [[r_from, c_from], [r_landing,  c_landing]]  (2 entries)
  Multi-jump  : [[r_from, c_from], [land1], [land2], ...]    (3+ entries)
  The captured piece lies at the midpoint between consecutive path entries.
"""


# Shared opening used by all four baselines.
# Replaces the ambiguous "You are playing as the indicated side" wording.
_INTRO = (
    "You are selecting exactly ONE move for the CURRENT SIDE TO MOVE.\n"
    "The side to move is stated in the user prompt (RED or BLACK).\n\n"
    "PIECE LEGEND:\n"
    "  r = RED man    R = RED king    b = BLACK man    B = BLACK king    . = empty\n\n"
    "BOARD ORIENTATION:\n"
    "  Row 0 = top edge.  Row 7 = bottom edge.\n"
    "  Column 0 = left edge.  Column 7 = right edge.\n"
    "  RED pieces move toward lower row numbers (toward row 0).\n"
    "  BLACK pieces move toward higher row numbers (toward row 7).\n\n"
)

SYSTEM_B1 = (
    _INTRO
    + "TASK: Select ONE move for the side to move.\n\n"
    "MUST / NEVER:\n"
    "  MUST  move only pieces that belong to the side to move.\n"
    "  MUST  verify the source square visibly contains a piece of the side to move.\n"
    "  MUST  output exactly one move — no lists, no alternatives.\n"
    "  NEVER move an opponent piece or an empty square.\n"
    "  NEVER invent a piece, coordinate, or capture not visible on the board.\n"
    "  NEVER output a square outside rows 0-7, columns 0-7.\n\n"
    + _OUTPUT_SPEC
    + "\nOutput JSON only."
)

SYSTEM_B2 = (
    _INTRO
    + "AMERICAN CHECKERS RULES:\n\n"
    "1. PIECE OWNERSHIP — MUST move only the side-to-move's own pieces.\n"
    "   NEVER move an opponent piece or an empty square.\n\n"
    "2. DARK SQUARES ONLY — every square in the path must have (row + col) ODD.\n"
    "   NEVER land on or depart from a light square.\n\n"
    "3. DIRECTION (MEN) — apply exactly:\n"
    "   RED man (r): row MUST DECREASE with each step. Moving toward row 7 is ILLEGAL.\n"
    "   BLACK man (b): row MUST INCREASE with each step. Moving toward row 0 is ILLEGAL.\n"
    "   King (R or B): may move in any diagonal direction.\n\n"
    "4. SIMPLE MOVE — one diagonal step to an adjacent EMPTY dark square.\n\n"
    "5. CAPTURE — jump over one adjacent ENEMY piece to the EMPTY square beyond.\n"
    "   The jumped-over square MUST visibly contain an enemy piece.\n"
    "   The landing square MUST visibly be empty.\n"
    "   NEVER claim a capture unless both conditions are true on the actual board.\n\n"
    "6. MANDATORY CAPTURE — if ANY capture exists for ANY of your pieces, you MUST jump.\n"
    "   A simple move is ILLEGAL when a capture is available.\n\n"
    "7. MULTI-JUMP — after landing, if the same piece can jump again it MUST continue.\n"
    "   Include every landing square. Stop only when no further jump is available.\n\n"
    "8. PROMOTION — RED man landing on row 0 becomes king; BLACK man on row 7 becomes king.\n"
    "   The turn ends immediately at promotion. Do NOT extend the path past that square.\n\n"
    "Choose ONE legal move for the side to move.\n\n"
    + _OUTPUT_SPEC
    + "\nOutput JSON only."
)

SYSTEM_B3 = (
    _INTRO
    + "AMERICAN CHECKERS RULES:\n\n"
    "1. PIECE OWNERSHIP — MUST move only the side-to-move's own pieces.\n"
    "   NEVER move an opponent piece or an empty square.\n\n"
    "2. DARK SQUARES ONLY — every square in the path must have (row + col) ODD.\n"
    "   NEVER land on or depart from a light square.\n\n"
    "3. DIRECTION (MEN) — apply exactly:\n"
    "   RED man (r): row MUST DECREASE with each step. Moving toward row 7 is ILLEGAL.\n"
    "   BLACK man (b): row MUST INCREASE with each step. Moving toward row 0 is ILLEGAL.\n"
    "   King (R or B): may move in any diagonal direction.\n\n"
    "4. SIMPLE MOVE — one diagonal step to an adjacent EMPTY dark square.\n\n"
    "5. CAPTURE — jump over one adjacent ENEMY piece to the EMPTY square beyond.\n"
    "   The jumped-over square MUST visibly contain an enemy piece.\n"
    "   The landing square MUST visibly be empty.\n"
    "   NEVER claim a capture unless both conditions are true on the actual board.\n\n"
    "6. MANDATORY CAPTURE — if ANY capture exists for ANY of your pieces, you MUST jump.\n"
    "   A simple move is ILLEGAL when a capture is available.\n\n"
    "7. MULTI-JUMP — after landing, if the same piece can jump again it MUST continue.\n"
    "   Include every landing square. Stop only when no further jump is available.\n\n"
    "8. PROMOTION — RED man landing on row 0 becomes king; BLACK man on row 7 becomes king.\n"
    "   The turn ends immediately at promotion. Do NOT extend the path past that square.\n\n"
    "REJECTION CHECKLIST\n"
    "===================\n"
    "Apply every check in order before finalising. If any check fails, REJECT the move\n"
    "and try a different one.\n\n"
    "  REJECT 1 — Source wrong\n"
    "    REJECT if the first square in the path is empty or holds an opponent piece.\n\n"
    "  REJECT 2 — Invented piece or coordinate\n"
    "    REJECT if the source piece does not visibly appear on the board,\n"
    "    or any coordinate is outside rows 0-7 / columns 0-7.\n\n"
    "  REJECT 3 — Landing square occupied or off-board\n"
    "    REJECT if any landing square is occupied by any piece, or off the board.\n\n"
    "  REJECT 4 — BLACK man going the wrong way\n"
    "    REJECT if a BLACK man (b) moves toward a smaller row number.\n"
    "    BLACK men MUST move toward larger row numbers only.\n\n"
    "  REJECT 5 — RED man going the wrong way\n"
    "    REJECT if a RED man (r) moves toward a larger row number.\n"
    "    RED men MUST move toward smaller row numbers only.\n\n"
    "  REJECT 6 — Fake capture\n"
    "    REJECT if you claim a capture but the jumped-over square does not visibly\n"
    "    contain an enemy piece, or the landing square is not visibly empty.\n\n"
    "  REJECT 7 — Simple move when capture available\n"
    "    REJECT any simple move if a real capture is available anywhere on the board.\n\n"
    "  REJECT 8 — Incomplete multi-jump\n"
    "    REJECT if the piece can jump again after landing but the path stops.\n\n"
    "  REJECT 9 — Malformed output\n"
    "    REJECT any response that is not valid JSON with a single selected_move path.\n\n"
    "Choose ONE legal move for the side to move.\n\n"
    + _OUTPUT_SPEC
    + "\nOutput JSON only."
)


SYSTEM_B4 = (
    _INTRO
    + "AMERICAN CHECKERS RULES:\n"
    "1. Dark squares only: a square is playable only when (row + col) is ODD.\n"
    "2. Direction:\n"
    "     RED men (r) move diagonally UP — toward lower row numbers (row decreases).\n"
    "     BLACK men (b) move diagonally DOWN — toward higher row numbers (row increases).\n"
    "     Kings (R or B) move diagonally in all four directions.\n"
    "3. Simple move: one diagonal step to an adjacent empty dark square.\n"
    "4. Capture (jump): jump over an adjacent enemy piece onto the empty square\n"
    "   diagonally beyond it (2 diagonal steps). The jumped piece is removed.\n"
    "5. MANDATORY CAPTURE: If any jump is available, the moving side MUST jump.\n"
    "   Simple moves are NOT legal when a jump exists.\n"
    "6. Multi-jump: After landing, if the same piece can jump again it MUST\n"
    "   continue in the same turn. The path lists every landing square.\n"
    "7. Promotion: A RED man (r) reaching row 0 becomes a RED king (R).\n"
    "   A BLACK man (b) reaching row 7 becomes a BLACK king (B).\n"
    "   Promotion ends the turn; a newly crowned king does NOT continue jumping.\n\n"
    "STRUCTURED LEGALITY CHECKLIST\n"
    "Before finalising your answer, verify each item below in order.\n"
    "If any item fails, discard your candidate and choose a different move.\n\n"
    "  1. SOURCE PIECE: The first square in your path must contain a piece\n"
    "     that belongs to the side to move. Do not move an empty square or\n"
    "     an opponent piece.\n\n"
    "  2. DESTINATION IS EMPTY: Every landing square in your path (all squares\n"
    "     after the first) must be empty on the board as it stands at that point\n"
    "     in the sequence.\n\n"
    "  3. DESTINATION IS DARK: Every square in your path must be a dark square\n"
    "     (row + col is odd). Light squares are never legal.\n\n"
    "  4. DIRECTION IS LEGAL: Verify that the direction of movement matches the\n"
    "     piece type. A RED man may only move toward lower row numbers. A BLACK\n"
    "     man may only move toward higher row numbers. A king may move in any\n"
    "     diagonal direction.\n\n"
    "  5. CAPTURE FIRST: If any enemy piece can be jumped by any of your pieces,\n"
    "     your selected_move must be a jump. A simple move is only legal when\n"
    "     no capture is available anywhere on the board.\n\n"
    "  6. FULL CHAIN: If your piece can continue jumping after a landing, the\n"
    "     path must include that continuation. A multi-jump that stops early\n"
    "     when another jump is still available is not legal.\n\n"
    "  7. PROMOTION ENDS TURN: If a man reaches the promotion row (row 0 for RED,\n"
    "     row 7 for BLACK), it is crowned king and the turn ends immediately.\n"
    "     The path must end at that square even if a further jump seems possible.\n\n"
    "  8. ONE MOVE ONLY: Output exactly one selected_move path. Do not output\n"
    "     multiple candidates or a list of all legal moves.\n\n"
    "COORDINATE-BASED LEGALITY CHECK\n"
    "================================\n"
    "Let PATH = selected_move  (a list of [row, col] waypoints).\n"
    "Apply each check below to your candidate. If any check fails, discard\n"
    "that candidate and choose a different move.\n\n"
    "CHECK 1 — In-bounds (engine: in_bounds)\n"
    "  For every [r, c] in PATH:\n"
    "    Require: 0 <= r <= 7  and  0 <= c <= 7\n\n"
    "CHECK 2 — Dark square (engine: is_dark_square: (r + c) % 2 == 1)\n"
    "  For every [r, c] in PATH:\n"
    "    Require: (r + c) % 2 == 1\n"
    "  (Light squares are never legal origins or destinations.)\n\n"
    "CHECK 3 — Source piece (engine: is_own_piece)\n"
    "  At PATH[0] = [r0, c0], read board[r0][c0]:\n"
    "    If side is RED:   piece must be r (=1) or R (=3)\n"
    "    If side is BLACK: piece must be b (=2) or B (=4)\n"
    "  If the square is empty or holds the opponent, discard.\n\n"
    "CHECK 4 — Delta and step type (engine: get_move_directions + step size)\n"
    "  For each consecutive leg i: PATH[i] \u2192 PATH[i+1]\n"
    "    dr = PATH[i+1][0] - PATH[i][0]\n"
    "    dc = PATH[i+1][1] - PATH[i][1]\n"
    "  Require |dc| == |dr|  (must be diagonal; no horizontal or vertical moves).\n"
    "  |dr| == 1  \u2192 this leg is a SIMPLE step\n"
    "  |dr| == 2  \u2192 this leg is a JUMP step\n"
    "  Any other |dr|  \u2192 ILLEGAL, discard.\n\n"
    "CHECK 5 — Direction validity per piece type\n"
    "  (engine: get_move_directions returns exact direction tuples)\n"
    "  Using dr of the FIRST leg:\n"
    "    r  (RED man):    allowed direction tuples: (-1,-1) and (-1,+1)  \u21d2  dr must be -1\n"
    "    b  (BLACK man):  allowed direction tuples: (+1,-1) and (+1,+1)  \u21d2  dr must be +1\n"
    "    R or B (king):   allowed direction tuples: (-1,-1),(-1,+1),(+1,-1),(+1,+1)  \u21d2  dr = \u00b11\n"
    "  For JUMP legs |dr|==2: use dr//2 and dc//2 to get the underlying unit step,\n"
    "  then apply the same direction rule above.\n\n"
    "CHECK 6 — Mandatory capture scan (engine: get_all_legal_moves)\n"
    "  If CHECK 4 says this is a SIMPLE move, you must first scan for any jump:\n"
    "    For each square [pr, pc] where board[pr][pc] is your own piece:\n"
    "      Compute its allowed direction tuples (per CHECK 5 rules above).\n"
    "      For each direction (ddr, ddc):\n"
    "        mid_r = pr + ddr,    mid_c = pc + ddc\n"
    "        land_r = pr + 2*ddr, land_c = pc + 2*ddc\n"
    "        If 0<=land_r<=7 and 0<=land_c<=7:\n"
    "          If board[mid_r][mid_c] is an opponent piece\n"
    "             AND board[land_r][land_c] == empty (=0):\n"
    "            \u2192 A jump exists. SIMPLE moves are ILLEGAL.\n"
    "              Discard your candidate and choose a jump path instead.\n\n"
    "CHECK 7 — Jump midpoint and landing (engine: get_single_jumps)\n"
    "  For each JUMP leg PATH[i] \u2192 PATH[i+1]:\n"
    "    dr = PATH[i+1][0] - PATH[i][0]  (\u00b12)\n"
    "    dc = PATH[i+1][1] - PATH[i][1]  (\u00b12)\n"
    "    mid_r = PATH[i][0] + dr//2\n"
    "    mid_c = PATH[i][1] + dc//2\n"
    "    Require:\n"
    "      board[mid_r][mid_c] is an opponent piece\n"
    "      board[PATH[i+1][0]][PATH[i+1][1]] == empty (=0)\n"
    "      (mid_r, mid_c) is NOT in the set of squares already captured\n"
    "        earlier in this path (captured_so_far \u2014 no re-capture allowed)\n\n"
    "CHECK 8 — Promotion termination (engine: apply_jump_on_board)\n"
    "  After each jump leg landing at PATH[i+1] = [rl, cl]:\n"
    "    If piece was r (RED man) and rl == 0:\n"
    "      Piece becomes RED king. Turn ENDS. PATH must stop at PATH[i+1].\n"
    "    If piece was b (BLACK man) and rl == 7:\n"
    "      Piece becomes BLACK king. Turn ENDS. PATH must stop at PATH[i+1].\n"
    "    If your PATH has more waypoints after a promotion square, DISCARD.\n\n"
    "CHECK 9 — Multi-jump continuation (engine: get_all_jump_sequences)\n"
    "  After each non-promotion jump leg, update your mental board:\n"
    "    board[mid_r][mid_c] = empty    (remove captured piece)\n"
    "    board[PATH[i+1][0]][PATH[i+1][1]] = piece  (place piece at landing)\n"
    "    board[PATH[i][0]][PATH[i][1]]   = empty    (clear origin)\n"
    "    Add (mid_r, mid_c) to captured_so_far.\n"
    "  Then re-apply CHECK 6 for only this piece at its new position.\n"
    "  If another jump is available:\n"
    "    Your PATH must include another leg. If it does not, DISCARD and extend.\n"
    "  If no further jump is available, the path ends correctly here.\n\n"
    "If all checks pass \u2192 output selected_move.\n"
    "If any check fails \u2192 pick a different candidate and re-run from CHECK 1.\n\n"
    "Choose ONE legal move for the side to move.\n\n"
    + _OUTPUT_SPEC
    + "\nIMPORTANT: Only move pieces belonging to the side to move. Output JSON only."
)


# ---------------------------------------------------------------------------
# User prompt builder  (shared by B1 / B2 / B3 / B4)
# ---------------------------------------------------------------------------

def build_user_prompt(
    board: list[list[int]],
    side_to_move: str,
    scenario_id: str,
) -> str:
    """
    Build the user-turn message for a legality-stress scenario.

    Parameters
    ----------
    board        : 8x8 int array (engine format).
    side_to_move : "RED" or "BLACK".
    scenario_id  : Identifier included for log traceability.

    Privacy guarantee
    -----------------
    This function does NOT receive or emit hidden_legal_moves.
    """
    return (
        f"Scenario: {scenario_id}\n"
        f"Side to move: {side_to_move}\n\n"
        f"Board:\n{render_board(board)}\n\n"
        f"Choose ONE legal move for {side_to_move}. "
        "Respond with JSON only."
    )


# ---------------------------------------------------------------------------
# Prompt registry
# ---------------------------------------------------------------------------

BASELINES: dict[str, str] = {
    "B1_board_only":                 SYSTEM_B1,
    "B2_rules":                      SYSTEM_B2,
    "B3_rules_structured_checklist": SYSTEM_B3,
    "B4_rules_engine_checking":      SYSTEM_B4,
}

# ---------------------------------------------------------------------------
# B5: Candidate-assisted rule filter
# ---------------------------------------------------------------------------

SYSTEM_B5 = (
    "You are playing American Checkers (8x8).\n\n"
    "PIECES:\n"
    "  r = RED man    R = RED king    b = BLACK man    B = BLACK king    . = empty\n\n"
    "RULES:\n"
    "1. Dark squares only (row + col is odd). Never move to a light square.\n"
    "2. Direction:\n"
    "     RED men (r) move diagonally UP — toward lower row numbers (row decreases).\n"
    "     BLACK men (b) move diagonally DOWN — toward higher row numbers (row increases).\n"
    "     Kings (R or B) move diagonally in all four directions.\n"
    "3. Simple move: one diagonal step to an adjacent empty dark square.\n"
    "4. Capture (jump): jump over an adjacent enemy piece onto the empty square\n"
    "   diagonally beyond it (2 diagonal steps). The jumped piece is removed.\n"
    "5. MANDATORY CAPTURE: If any jump is available, the moving side MUST jump.\n"
    "   Simple moves are NOT legal when a jump exists.\n"
    "6. Multi-jump: After landing, if the same piece can jump again it MUST\n"
    "   continue in the same turn. The path lists every landing square.\n"
    "7. Promotion: A RED man (r) reaching row 0 becomes a RED king (R).\n"
    "   A BLACK man (b) reaching row 7 becomes a BLACK king (B).\n"
    "   Promotion ends the turn; a newly crowned king does NOT continue jumping.\n\n"
    "STRUCTURED LEGALITY CHECKLIST\n"
    "Before finalising your answer, verify each item below in order.\n"
    "If any item fails, discard your candidate and choose a different move.\n\n"
    "  1. SOURCE PIECE: The first square in your path must contain a piece\n"
    "     that belongs to the side to move.\n\n"
    "  2. DESTINATION IS EMPTY: Every landing square in your path must be empty.\n\n"
    "  3. DESTINATION IS DARK: Every square in your path must satisfy (row+col) % 2 == 1.\n\n"
    "  4. DIRECTION IS LEGAL: Direction must match piece type (see Rule 2 above).\n\n"
    "  5. CAPTURE FIRST: If any jump is available anywhere on the board, you MUST\n"
    "     select a jump. Simple moves are ILLEGAL when any jump exists.\n\n"
    "  6. FULL CHAIN: If the piece can continue jumping after landing, the path\n"
    "     must include that continuation.\n\n"
    "  7. PROMOTION ENDS TURN: If a man reaches the promotion row, the turn ends.\n\n"
    "  8. ONE MOVE ONLY: Output exactly one selected_move path.\n\n"
    "CANDIDATE MOVES\n"
    "===============\n"
    "The list below shows candidate moves derived from the board geometry.\n"
    "IMPORTANT: These candidates are NOT guaranteed to all be legal.\n"
    "  - Some simple-move candidates may be ILLEGAL because a jump exists elsewhere\n"
    "    (mandatory capture rule).\n"
    "  - Some jump candidates may be ILLEGAL if the path is incomplete (multi-jump\n"
    "    continuation was available but not included).\n"
    "You must apply the rules above to decide which candidate — if any — is legal,\n"
    "and output exactly one legal selected_move.\n"
    "Do NOT label candidates as legal or illegal in your reasoning.\n"
    "Simply apply the rules and output the correct move.\n\n"
    + _OUTPUT_SPEC
    + "\nIMPORTANT: Only move pieces belonging to the side to move. Output JSON only."
)


def build_b5_user_prompt(
    board: list,
    side_to_move: str,
    scenario_id: str,
    candidates: list[dict],
) -> str:
    """
    User prompt for B5_candidate_moves_rule_filter.

    Shows:
      - board + scenario header (same as B1-B4)
      - candidate moves list (physically possible, no mandatory-capture filter)
      - reminder that candidates may include illegal moves

    Does NOT show hidden_legal_moves.
    Does NOT label any candidate as legal or illegal.
    """
    from checkers.data.legality_eval.prompts import render_board

    lines = [
        f"Scenario: {scenario_id}",
        f"Side to move: {side_to_move}",
        "",
        "Board:",
        render_board(board),
        "",
        f"Candidate moves ({len(candidates)} total — apply rules to determine legality):",
    ]

    for c in candidates:
        path_str = ", ".join(f"[{r},{col}]" for r, col in c["path"])
        cap_str  = ""
        if c["captured"]:
            cap_str = "  captures: " + ", ".join(f"[{r},{col}]" for r, col in c["captured"])
        lines.append(
            f"  {c['id']} ({c['move_type']})  path: [{path_str}]{cap_str}"
        )

    lines += [
        "",
        "WARNING: The candidate list above may include simple moves that are illegal",
        "because a jump is available (mandatory capture). Inspect all candidates,",
        "apply the CAPTURE FIRST rule, and output exactly ONE legal selected_move.",
        "",
        "Choose ONE legal move. Respond with JSON only.",
    ]

    return "\n".join(lines)


# Register B5 in the BASELINES dict so run_legality_pilot.py picks it up
BASELINES["B5_candidate_moves_rule_filter"] = SYSTEM_B5

# ---------------------------------------------------------------------------
# B6: Candidate-verbatim copy
# ---------------------------------------------------------------------------

SYSTEM_B6 = (
    "You are playing American Checkers (8x8).\n\n"
    "PIECES:\n"
    "  r = RED man    R = RED king    b = BLACK man    B = BLACK king    . = empty\n\n"
    "RULES:\n"
    "1. Dark squares only (row + col is odd). Never move to a light square.\n"
    "2. Direction:\n"
    "     RED men (r) move diagonally UP — toward lower row numbers (row decreases).\n"
    "     BLACK men (b) move diagonally DOWN — toward higher row numbers (row increases).\n"
    "     Kings (R or B) move diagonally in all four directions.\n"
    "3. Simple move: one diagonal step to an adjacent empty dark square.\n"
    "4. Capture (jump): jump over an adjacent enemy piece onto the empty square\n"
    "   diagonally beyond it (2 diagonal steps). The jumped piece is removed.\n"
    "5. MANDATORY CAPTURE: If any jump is available, the moving side MUST jump.\n"
    "   Simple moves are NOT legal when a jump exists.\n"
    "6. Multi-jump: After landing, if the same piece can jump again it MUST\n"
    "   continue in the same turn. The path lists every landing square.\n"
    "7. Promotion: A RED man (r) reaching row 0 becomes a RED king (R).\n"
    "   A BLACK man (b) reaching row 7 becomes a BLACK king (B).\n"
    "   Promotion ends the turn; a newly crowned king does NOT continue jumping.\n\n"
    "STRUCTURED LEGALITY CHECKLIST\n"
    "Before finalising your answer, verify each item below in order.\n"
    "If any item fails, discard your candidate and choose a different one.\n\n"
    "  1. SOURCE PIECE: The first square in the path must contain a piece\n"
    "     that belongs to the side to move.\n\n"
    "  2. DESTINATION IS EMPTY: Every landing square in the path must be empty.\n\n"
    "  3. DESTINATION IS DARK: Every square must satisfy (row+col) % 2 == 1.\n\n"
    "  4. DIRECTION IS LEGAL: Direction must match piece type (see Rule 2 above).\n\n"
    "  5. CAPTURE FIRST: If any jump is available anywhere on the board,\n"
    "     you MUST select a jump candidate. Simple candidates are ILLEGAL\n"
    "     when any jump candidate exists in the list.\n\n"
    "  6. FULL CHAIN: If the piece can continue jumping after landing, the path\n"
    "     must include that continuation.\n\n"
    "  7. PROMOTION ENDS TURN: If a man reaches the promotion row, the turn ends.\n\n"
    "  8. ONE MOVE ONLY: Output exactly one selected_move path.\n\n"
    "CANDIDATE MOVES\n"
    "===============\n"
    "The list below shows candidate moves derived from the board geometry.\n"
    "IMPORTANT: These candidates are NOT guaranteed to all be legal.\n"
    "  - Simple candidates may be ILLEGAL if a jump exists (mandatory capture).\n"
    "  - Jump candidates may be ILLEGAL if a continuation was missed.\n"
    "Apply the rules above to decide which candidate is legal.\n\n"
    "VERBATIM COPY REQUIREMENT\n"
    "=========================\n"
    "Your selected_move MUST be copied EXACTLY from one displayed candidate path.\n\n"
    "STRICT RULES:\n"
    "  - Choose one candidate ID (e.g., C2).\n"
    "  - Copy its path coordinate-for-coordinate into selected_move.\n"
    "  - Do NOT invent new coordinates.\n"
    "  - Do NOT modify, extend, shorten, re-order, or 'improve' any candidate path.\n"
    "  - Do NOT combine two candidates.\n"
    "  - Do NOT output a path that does not appear in the candidate list.\n"
    "  - If selected_candidate_id is C2, then selected_move must equal C2's path exactly.\n\n"
    "IMPORTANT: The evaluator will reject any move not matching a displayed candidate path.\n\n"
    "OUTPUT — respond with valid JSON only, no prose before or after:\n"
    "{\n"
    '  "selected_candidate_id": "<ID from candidate list, e.g. C0>",\n'
    '  "selected_move": [[row, col], [row, col], ...],\n'
    '  "reasoning": "brief explanation"\n'
    "}\n\n"
    "IMPORTANT: selected_move must be copied verbatim from the candidate you chose.\n"
    "Output JSON only."
)


def build_b6_user_prompt(
    board: list,
    side_to_move: str,
    scenario_id: str,
    candidates: list[dict],
) -> str:
    """
    User prompt for B6_candidate_moves_verbatim.

    Identical candidate list as B5, but with explicit verbatim-copy instruction.
    Does NOT show hidden_legal_moves.
    Does NOT label candidates as legal or illegal.
    """
    from checkers.data.legality_eval.prompts import render_board

    lines = [
        f"Scenario: {scenario_id}",
        f"Side to move: {side_to_move}",
        "",
        "Board:",
        render_board(board),
        "",
        f"Candidate moves ({len(candidates)} total):",
    ]

    for c in candidates:
        path_str = ", ".join(f"[{r},{col}]" for r, col in c["path"])
        cap_str  = ""
        if c["captured"]:
            cap_str = "  captures: " + ", ".join(f"[{r},{col}]" for r, col in c["captured"])
        lines.append(
            f"  {c['id']} ({c['move_type']})  path: [{path_str}]{cap_str}"
        )

    lines += [
        "",
        "INSTRUCTION: Apply the rules (especially CAPTURE FIRST) to decide which",
        "candidate is legal. Then copy its path EXACTLY into selected_move.",
        "Do NOT invent coordinates. Do NOT modify the path. Output JSON only.",
    ]

    return "\n".join(lines)


# Register B6 in the BASELINES dict
BASELINES["B6_candidate_moves_verbatim"] = SYSTEM_B6

# ---------------------------------------------------------------------------
# B7: Candidate-verbatim, path-only output (no selected_candidate_id)
# ---------------------------------------------------------------------------

SYSTEM_B7 = (
    "You are playing American Checkers (8x8).\n\n"
    "PIECES:\n"
    "  r = RED man    R = RED king    b = BLACK man    B = BLACK king    . = empty\n\n"
    "RULES:\n"
    "1. DARK SQUARES ONLY: A square is playable only when (row + col) is ODD.\n"
    "   Never move to or from a light square (where row + col is even).\n\n"
    "2. PIECE OWNERSHIP: You may only move pieces belonging to the side to move.\n"
    "   Moving an opponent piece or an empty square is not legal.\n\n"
    "3. DIRECTION:\n"
    "   RED men (r): move diagonally toward lower row numbers only (row DECREASES).\n"
    "   BLACK men (b): move diagonally toward higher row numbers only (row INCREASES).\n"
    "   Kings (R or B): move diagonally in ANY of the four diagonal directions.\n\n"
    "4. SIMPLE MOVE: Move one diagonal step to an adjacent empty dark square.\n"
    "   The source square must contain your piece. The destination must be empty.\n\n"
    "5. CAPTURE (JUMP): Jump over one adjacent enemy piece onto the empty square\n"
    "   diagonally beyond it (2 steps). The jumped enemy piece is removed.\n"
    "   The destination must be empty.\n\n"
    "6. MANDATORY CAPTURE: If ANY capture is available to ANY of your pieces,\n"
    "   you MUST jump. Choosing a simple move when a capture exists is ILLEGAL.\n\n"
    "7. MULTI-JUMP: After landing from a jump, if the SAME piece can jump again\n"
    "   it MUST continue jumping in the same turn.\n"
    "   The path must include every landing square in sequence.\n"
    "   The turn ends only when no further jump is available.\n\n"
    "8. PROMOTION: A RED man (r) landing on row 0 becomes a RED king (R).\n"
    "   A BLACK man (b) landing on row 7 becomes a BLACK king (B).\n"
    "   Promotion ends the turn immediately — the newly crowned king does NOT\n"
    "   continue jumping even if another jump is geometrically available.\n\n"
    "LEGALITY CHECKLIST\n"
    "==================\n"
    "Before finalising your answer, apply every check below in order.\n"
    "If a check FAILS, REJECT that move and choose a different one.\n\n"
    "  CHECK 1 — SOURCE IS YOURS\n"
    "    The first square in your path must contain a piece belonging to the\n"
    "    side to move.\n"
    "    REJECT if that square is empty or holds an opponent piece.\n\n"
    "  CHECK 2 — DESTINATION IS EMPTY\n"
    "    Every square after the first in your path must be empty.\n"
    "    REJECT if any landing square is occupied (by any piece).\n\n"
    "  CHECK 3 — ALL SQUARES ARE DARK\n"
    "    Every square in your path must satisfy (row + col) % 2 == 1.\n"
    "    REJECT if any square in the path is a light square.\n\n"
    "  CHECK 4 — DIRECTION MATCHES PIECE TYPE\n"
    "    A RED man (r) must move toward lower row numbers (row decreases).\n"
    "    A BLACK man (b) must move toward higher row numbers (row increases).\n"
    "    A king (R or B) may move in any diagonal direction.\n"
    "    REJECT if a man moves in the wrong direction.\n\n"
    "  CHECK 5 — CAPTURE FIRST (mandatory capture)\n"
    "    Scan the entire board: can ANY of your pieces jump over ANY enemy piece\n"
    "    onto an empty square?\n"
    "    If YES — you must jump. REJECT any simple move.\n"
    "    If NO  — a simple move is legal.\n\n"
    "  CHECK 6 — COMPLETE THE MULTI-JUMP\n"
    "    After each jump landing: can the SAME piece jump again?\n"
    "    If YES — the path must continue. REJECT if the path stops early.\n"
    "    If NO  — the path ends correctly here.\n\n"
    "  CHECK 7 — PROMOTION ENDS THE TURN\n"
    "    If a man lands on the promotion row (row 0 for RED, row 7 for BLACK),\n"
    "    the turn ends immediately. REJECT if the path continues past that square.\n\n"
    "  CHECK 8 — OUTPUT EXACTLY ONE MOVE\n"
    "    REJECT any response containing multiple move paths or a list of options.\n\n"
    "CANDIDATE MOVES\n"
    "===============\n"
    "The list below shows candidate moves derived from the board geometry.\n"
    "IMPORTANT: These candidates are NOT guaranteed to all be legal.\n"
    "  - Simple candidates may be ILLEGAL if a jump exists (mandatory capture).\n"
    "  - Jump candidates may be ILLEGAL if a continuation was missed.\n"
    "Apply the rules and checklist above to decide which candidate is legal.\n\n"
    "VERBATIM COPY REQUIREMENT\n"
    "=========================\n"
    "Your selected_move MUST be copied EXACTLY from one displayed candidate path.\n\n"
    "STRICT RULES:\n"
    "  - Choose one candidate from the list.\n"
    "  - Copy its path coordinate-for-coordinate into selected_move.\n"
    "  - Do NOT invent new coordinates.\n"
    "  - Do NOT modify, extend, shorten, re-order, or 'improve' any candidate path.\n"
    "  - Do NOT combine two candidates.\n"
    "  - Do NOT output a path that does not appear in the candidate list.\n\n"
    "IMPORTANT: The evaluator will reject any move not matching a displayed candidate path.\n\n"
    + _OUTPUT_SPEC
    + "\nIMPORTANT: selected_move must be copied verbatim from one candidate path.\n"
    "Output JSON only."
)


def build_b7_user_prompt(
    board: list,
    side_to_move: str,
    scenario_id: str,
    candidates: list[dict],
) -> str:
    """
    User prompt for B7_candidate_moves_path_only.

    Identical candidate list as B5/B6, same verbatim-copy instruction as B6,
    but output schema is selected_move only (no selected_candidate_id).
    Does NOT show hidden_legal_moves.
    Does NOT label candidates as legal or illegal.
    """
    from checkers.data.legality_eval.prompts import render_board

    lines = [
        f"Scenario: {scenario_id}",
        f"Side to move: {side_to_move}",
        "",
        "Board:",
        render_board(board),
        "",
        f"Candidate moves ({len(candidates)} total):",
    ]

    for c in candidates:
        path_str = ", ".join(f"[{r},{col}]" for r, col in c["path"])
        cap_str  = ""
        if c["captured"]:
            cap_str = "  captures: " + ", ".join(f"[{r},{col}]" for r, col in c["captured"])
        lines.append(
            f"  {c['id']} ({c['move_type']})  path: [{path_str}]{cap_str}"
        )

    lines += [
        "",
        "INSTRUCTION: Apply the rules and checklist (especially CAPTURE FIRST) to decide",
        "which candidate is legal. Then copy its path EXACTLY into selected_move.",
        "Do NOT invent coordinates. Do NOT modify the path. Output JSON only.",
    ]

    return "\n".join(lines)


# Register B7 in the BASELINES dict
BASELINES["B7_candidate_moves_path_only"] = SYSTEM_B7
