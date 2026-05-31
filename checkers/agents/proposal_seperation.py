# checkers/agents/proposal_seperation.py
#
# PURE RAW LLM BASELINE — Separated Scanner + Proposal
#
# ══════════════════════════════════════════════════════════════════════════════
# PURPOSE
# ══════════════════════════════════════════════════════════════════════════════
# Measures RAW LLM proposal quality WITHOUT hidden assistance.
#
# This is NOT a gameplay engine.
# This is NOT a minimax system.
# This is NOT a verifier-assisted architecture.
#
# The implementation is intentionally:
#   simple, transparent, auditable, minimally assisted, scientifically honest.
#
# ══════════════════════════════════════════════════════════════════════════════
# FORBIDDEN OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
# This module must NEVER:
#   - repair malformed proposals
#   - reconstruct intended moves
#   - infer missing continuations
#   - semantically reinterpret outputs
#   - generate helper geometry
#   - provide hidden legality hints
#   - inflate benchmark results
#
# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
# 1. SCANNER   — receives board + player + pieces → returns {capture_available}
# 2. PROPOSER  — receives board + player + pieces → returns all legal moves
#    - If scanner says capture_available=true  → jump proposer
#    - If scanner says capture_available=false → quiet proposer
#    - Only ONE proposal call per position (jump OR quiet, never both)
#
# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
# Scanner API:  SCANNER_MISTRAL_API_KEY, SCANNER_MISTRAL_URL, SCANNER_MISTRAL_MODEL
# Proposal API: JUMP_QUIET_MISTRAL_API_KEY, JUMP_QUIET_MISTRAL_URL, JUMP_QUIET_MISTRAL_MODEL
#

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Optional

try:
    from dotenv import load_dotenv as _load_dotenv
    _ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    _load_dotenv(os.path.abspath(_ENV_PATH), override=True)
except ImportError:
    pass

from checkers.engine.board import (
    EMPTY, RED, BLACK, RED_KING, BLACK_KING, BOARD_SIZE,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

_MISTRAL_DEFAULT_URL   = "https://api.mistral.ai/v1/chat/completions"
_MISTRAL_DEFAULT_MODEL = "mistral-large-latest"

# Scanner API — uses SCANNER_MISTRAL_* env vars from .env
SCANNER_API_KEY  = os.environ.get("SCANNER_MISTRAL_API_KEY", "").strip()
SCANNER_BASE_URL = os.environ.get("SCANNER_MISTRAL_URL", _MISTRAL_DEFAULT_URL).strip()
SCANNER_MODEL    = os.environ.get("SCANNER_MISTRAL_MODEL", _MISTRAL_DEFAULT_MODEL).strip()

# Proposal API (shared by jump + quiet — only one call per position)
# Uses JUMP_QUIET_MISTRAL_* env vars from .env
PROPOSAL_API_KEY  = os.environ.get("JUMP_QUIET_MISTRAL_API_KEY", "").strip()
PROPOSAL_BASE_URL = os.environ.get("JUMP_QUIET_MISTRAL_URL", _MISTRAL_DEFAULT_URL).strip()
PROPOSAL_MODEL    = os.environ.get("JUMP_QUIET_MISTRAL_MODEL", _MISTRAL_DEFAULT_MODEL).strip()

PROPOSAL_TEMPERATURE = float(os.environ.get("PROPOSAL_TEMPERATURE", "0.2"))


# ══════════════════════════════════════════════════════════════════════════════
# BOARD RENDERING — MINIMAL, NO HELPER GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

_SYM = {
    RED:        " r",
    RED_KING:   " R",
    BLACK:      " b",
    BLACK_KING: " B",
    EMPTY:      " .",
}


def render_board(board: list[list[int]]) -> str:
    """
    Render the 8×8 board with absolute symbols.
    No helper geometry, no empty-square lists, no diagonal targets.
    """
    col_header = "        " + "  ".join(f"c{c}" for c in range(BOARD_SIZE))
    separator  = "        " + "+--" * BOARD_SIZE + "+"
    lines      = [col_header, separator]

    for row in range(BOARD_SIZE):
        cells: list[str] = []
        for col in range(BOARD_SIZE):
            if (row + col) % 2 == 0:
                cells.append(" #")
            else:
                cells.append(_SYM.get(board[row][col], " ?"))
        lines.append(f"  r{row}  |" + "|".join(cells) + "|")
        lines.append(separator)

    return "\n".join(lines)


def list_pieces(board: list[list[int]], current_player: int) -> str:
    """
    Compact piece summary. No movable-pieces hints, no candidate lists.
    Only: piece type + coordinates.
    """
    red_men:     list[str] = []
    red_kings:   list[str] = []
    black_men:   list[str] = []
    black_kings: list[str] = []

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            p  = board[row][col]
            sq = f"[{row},{col}]"
            if   p == RED:        red_men.append(sq)
            elif p == RED_KING:   red_kings.append(sq)
            elif p == BLACK:      black_men.append(sq)
            elif p == BLACK_KING: black_kings.append(sq)

    def _fmt(lst: list[str]) -> str:
        return " ".join(lst) if lst else "—"

    if current_player == RED:
        return "\n".join([
            "YOUR pieces (RED):",
            f"  RED_MAN  (r): {_fmt(red_men)}",
            f"  RED_KING (R): {_fmt(red_kings)}",
            "OPPONENT pieces (BLACK):",
            f"  BLACK_MAN  (b): {_fmt(black_men)}",
            f"  BLACK_KING (B): {_fmt(black_kings)}",
        ])
    else:
        return "\n".join([
            "YOUR pieces (BLACK):",
            f"  BLACK_MAN  (b): {_fmt(black_men)}",
            f"  BLACK_KING (B): {_fmt(black_kings)}",
            "OPPONENT pieces (RED):",
            f"  RED_MAN  (r): {_fmt(red_men)}",
            f"  RED_KING (R): {_fmt(red_kings)}",
        ])


# ══════════════════════════════════════════════════════════════════════════════
# LLM API CALL — RAW HTTP (no extra dependencies)
# ══════════════════════════════════════════════════════════════════════════════


class _ConfigError(Exception):
    """Raised for configuration errors that must NOT be retried."""
    pass


def _call_llm_raw(
    system: str,
    user: str,
    api_key: str,
    api_url: str,
    model: str,
    temperature: float,
    json_mode: bool = True,
) -> str:
    """
    Raw HTTP POST to an OpenAI-compatible chat completions endpoint.
    Returns the raw content string. No repair, no post-processing.

    json_mode=True  → sends response_format:{type:json_object} (proposers).
    json_mode=False → omits response_format entirely (scanner scan-log format).

    Raises _ConfigError for missing credentials (must NOT be retried).
    Raises other exceptions for transport/HTTP failures (may be retried).
    """
    if not api_key:
        raise _ConfigError(f"API key not set for model={model} at url={api_url}")
    if not model:
        raise _ConfigError(f"Model not set for url={api_url}")

    payload: dict[str, Any] = {
        "model":       model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        api_url,
        data=body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept":        "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=90.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError(f"Response content is not a string: {type(content)}")
    return content


# ══════════════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE RETRY — transport failures ONLY
# ══════════════════════════════════════════════════════════════════════════════
# Retries are ONLY for: timeout, connection failure, rate limit, empty response,
# transport failure.
#
# Retries are NOT for: malformed reasoning, illegal moves, bad outputs, parse
# errors, incomplete jumps, hallucinations.
#
# Schedule: 2 groups × 3 attempts (20s, 30s, 40s each)
# If all fail → api_failure

_INFRA_RETRY_DELAYS = [20, 30, 40]
_INFRA_RETRY_GROUPS = 2


def _call_with_infra_retry(
    system: str,
    user: str,
    api_key: str,
    api_url: str,
    model: str,
    temperature: float,
    stage: str,
    json_mode: bool = True,
) -> tuple[str, bool]:
    """
    Call the LLM with infrastructure-only retry.

    Configuration errors (missing API key, missing model) fail IMMEDIATELY
    without any retry — they are not transient.

    Transport errors (timeout, connection, rate limit, empty response)
    are retried: 2 groups × 3 attempts (20s, 30s, 40s).

    json_mode is threaded through to _call_llm_raw.

    Returns (raw_output, api_ok).
    If api_ok is False, raw_output is "" and the caller should record api_failure.
    This entire process counts as ONE logical proposal attempt.
    """
    last_error = ""

    # Initial call
    try:
        raw = _call_llm_raw(system, user, api_key, api_url, model, temperature, json_mode)
        if raw and raw.strip():
            return raw, True
        last_error = "EMPTY_RESPONSE"
    except _ConfigError as exc:
        # Configuration errors must NOT be retried.
        logger.error("[%s] config error (no retry): %s", stage, exc)
        return "", False
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {str(exc)[:200]}"

    logger.warning("[%s] initial call failed: %s", stage, last_error)

    # Retry groups — transport/infrastructure failures only
    for group in range(1, _INFRA_RETRY_GROUPS + 1):
        for i, delay in enumerate(_INFRA_RETRY_DELAYS, 1):
            logger.warning(
                "[%s] infra retry group %d/%d attempt %d/%d in %ds: %s",
                stage, group, _INFRA_RETRY_GROUPS, i, len(_INFRA_RETRY_DELAYS),
                delay, last_error,
            )
            time.sleep(delay)
            try:
                raw = _call_llm_raw(system, user, api_key, api_url, model, temperature, json_mode)
                if raw and raw.strip():
                    logger.info(
                        "[%s] recovered: group %d attempt %d", stage, group, i,
                    )
                    return raw, True
                last_error = "EMPTY_RESPONSE"
            except _ConfigError as exc:
                # Should not happen mid-retry, but guard anyway.
                logger.error("[%s] config error during retry (aborting): %s", stage, exc)
                return "", False
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:200]}"

    logger.error("[%s] all infra retries exhausted", stage)
    return "", False


# ══════════════════════════════════════════════════════════════════════════════
# SCANNER PROMPT
# ══════════════════════════════════════════════════════════════════════════════
# The scanner receives ONLY: board grid, current player, current player's pieces.
# It must NOT receive: legal moves, movable pieces, candidate jumps, geometry
# hints, anchor hints, continuation hints, validator output.
#
# Its ONLY task: determine whether at least one legal capture exists.

SCANNER_SYSTEM_PROMPT = """\
You are an American Checkers capture detector.

Your ONLY task: determine whether the active player has at least one legal
capture (jump) available on the given board.

══ BOARD SYMBOLS (absolute — same regardless of which side plays) ══
  r = RED man    R = RED king    b = BLACK man    B = BLACK king
  . = empty dark square    # = light square (never playable, ignore)
Row 0 = TOP.  Row 7 = BOTTOM.  Col 0 = LEFT.  Col 7 = RIGHT.
Dark square: (row + col) is ODD. No piece ever sits on a light (#) square.

══ DIAGONAL DIRECTIONS (coordinate formulas) ══
  NW = [row-1, col-1]    NE = [row-1, col+1]
  SW = [row+1, col-1]    SE = [row+1, col+1]
A jump lands 2 diagonal steps from start:
  NW jump: mid=[row-1, col-1], land=[row-2, col-2]
  NE jump: mid=[row-1, col+1], land=[row-2, col+2]
  SW jump: mid=[row+1, col-1], land=[row+2, col-2]
  SE jump: mid=[row+1, col+1], land=[row+2, col+2]

══ CAPTURE VALIDITY — ALL THREE gates must pass ══
For a piece at [r,c] jumping in direction D:
  GATE-1: board[mid]  = OPPONENT piece  (not '.', not your own piece, not '#')
  GATE-2: board[land] = '.'             (empty dark square)
  GATE-3: land is in-bounds             (both row and col in 0–7)
If ANY gate fails → NOT a valid capture for that direction.

══ DIRECTION RESTRICTIONS ══
  RED men:   forward only → NW and NE only  (row DECREASES)
  BLACK men: forward only → SW and SE only  (row INCREASES)
  Kings (R or B): all 4 directions → NW, NE, SW, SE

══ MANDATORY SCAN LOG ══
You MUST produce a SCAN_LOG before the final JSON verdict.
This is not optional. The scan log is your working evidence.

For EVERY piece belonging to the current player, write:

  piece=[r,c] type=man|king  dirs=<applicable directions>
    dir=NW  mid=[r-1,c-1] mid_val=<symbol from board>  land=[r-2,c-2] land_val=<symbol from board>  GATE-1=pass|fail  GATE-2=pass|fail  GATE-3=pass|fail  valid=true|false
    dir=NE  mid=[r-1,c+1] mid_val=<symbol from board>  land=[r-2,c+2] land_val=<symbol from board>  GATE-1=pass|fail  GATE-2=pass|fail  GATE-3=pass|fail  valid=true|false
    (only the applicable directions for this piece type)

Rules for writing the scan log:
  - mid_val and land_val MUST be read from the actual board grid, not assumed.
  - GATE-1=pass only if mid_val is an OPPONENT symbol.
  - GATE-2=pass only if land_val = '.' exactly.
  - GATE-3=pass only if the landing row and col are both in 0–7.
  - If the landing square is out-of-bounds (GATE-3 fails), write land_val=OOB.
  - valid=true only if ALL THREE gates pass. Otherwise valid=false.
  - Omit a direction entirely if GATE-3 obviously fails (coord < 0 or > 7).

After scanning ALL pieces, write:

  VERDICT: capture_available=true|false
  Reason: [one sentence — which piece and direction was valid, or why none were]

Consistency rule (MANDATORY):
  If ANY scan entry has valid=true  → VERDICT must be capture_available=true.
  If NO  scan entry has valid=true  → VERDICT must be capture_available=false.
  The VERDICT must always match the scan log. Never override your own evidence.

══ COMMON TRAPS — check these explicitly in your scan log ══

TRAP 1 — ADJACENCY ≠ CAPTURE:
  An opponent piece at mid does NOT guarantee valid=true.
  GATE-2 (land_val='.') must ALSO pass. Read land_val from the grid.

TRAP 2 — CROWDED BOARDS:
  When many pieces are present, landing squares are often occupied.
  A crowded board means FEWER valid captures, not more.
  Never skip writing land_val. Read it from the actual grid.

TRAP 3 — KINGS CHECK ALL 4 DIRS INDIVIDUALLY:
  A king's 4 directions are independent. Each must pass all three gates.
  Do not assume any direction is valid without checking.

TRAP 4 — OWN PIECE AT MIDPOINT:
  If mid_val is your OWN symbol → GATE-1=fail. Write it explicitly.

TRAP 5 — OCCUPIED LANDING:
  If land_val is ANY piece symbol (r, R, b, B) → GATE-2=fail.
  Write the actual symbol you read. Do NOT write '.' if the square is occupied.

══ OUTPUT FORMAT ══
Write the SCAN_LOG first (plain text, one line per piece/direction).
Then write the VERDICT line.
Then write exactly one JSON object on its own line at the end:

{"capture_available": true}
  or
{"capture_available": false}

The JSON must be the LAST thing you write. It must match the VERDICT.
"""


def build_scanner_prompt(
    board: list[list[int]],
    current_player: int,
) -> tuple[str, str]:
    """Build (system, user) prompts for the scanner. No hints, no geometry."""
    if current_player == RED:
        player_label = "RED"
        own_symbols  = "r (man) and R (king)"
        opp_symbols  = "b (man) and B (king)"
        man_dirs     = "NW [row-1,col-1] and NE [row-1,col+1] — forward only (row decreases)"
    else:
        player_label = "BLACK"
        own_symbols  = "b (man) and B (king)"
        opp_symbols  = "r (man) and R (king)"
        man_dirs     = "SW [row+1,col-1] and SE [row+1,col+1] — forward only (row increases)"

    board_grid = render_board(board)
    piece_list = list_pieces(board, current_player)

    user_prompt = "\n".join([
        f"Current player: {player_label}",
        "",
        "━━ SIDE-SPECIFIC RULES FOR THIS CALL ━━",
        f"  Your symbols:     {own_symbols}",
        f"  Opponent symbols: {opp_symbols}",
        f"  Man jump dirs:    {man_dirs}",
        "  King jump dirs:   all 4 diagonals: NW, NE, SW, SE",
        "",
        piece_list,
        "",
        "BOARD:",
        board_grid,
        "",
        "LOOKUP: to read square [row,col], find the row labeled 'r{row}' and the column 'c{col}'.",
        "Always read mid_val and land_val from the board above before writing each scan log entry.",
        "",
        "Write your SCAN_LOG, then VERDICT, then the JSON verdict on the last line.",
        "The JSON must match the VERDICT.",
    ])

    # json_mode=False: scan-log output is not JSON; parse_scanner_output
    # uses regex to extract the JSON verdict from anywhere in the raw text.
    return SCANNER_SYSTEM_PROMPT, user_prompt


# Scanner json_mode flag — False so scan-log text is allowed before the JSON
_SCANNER_JSON_MODE = False


# ══════════════════════════════════════════════════════════════════════════════
# JUMP PROPOSER PROMPT
# ══════════════════════════════════════════════════════════════════════════════
# Receives ONLY: board grid, current player, current player's pieces.
# Must NOT receive: movable pieces, legal moves, candidate anchors, jump hints,
# continuation hints, validator hints, scanner geometry.
# Must independently reason from scratch.

JUMP_SYSTEM_PROMPT = """\
You are an American Checkers legal jump-move generator.
Task: output ALL legal capture sequences for the active player.

SYMBOLS: r=RED man  R=RED king  b=BLACK man  B=BLACK king  .=empty  #=light(ignore)
Row 0=TOP  Row 7=BOTTOM  Col 0=LEFT  Col 7=RIGHT  Dark=(row+col) ODD

JUMP COORDINATES: for direction (dr,dc) from [r,c]:
  mid  = [r+dr,  c+dc ]   (opponent piece here)
  land = [r+2dr, c+2dc]   (must be empty)

DIRECTIONS:
  RED men:   NW(dr=-1,dc=-1)  NE(dr=-1,dc=+1)   [forward only, row decreases]
  BLACK men: SW(dr=+1,dc=-1)  SE(dr=+1,dc=+1)   [forward only, row increases]
  Kings:     NW  NE  SW  SE   [all four; kings may change direction between captures]

══ PHASE 1 — SCAN all pieces ══
For EVERY own piece, EVERY applicable direction:
  GATE-1: land in-bounds (both row and col in 0–7)?            Skip if no.
  GATE-2: board[mid] = OPPONENT symbol?                         Skip if no.
  GATE-3: board[land] = '.'?                                    Skip if no.
  All 3 pass → valid first jump. Record it.
❖ Scan ALL pieces and ALL directions before deciding output. Do not stop early.

══ PHASE 2 — CONTINUATION from each landing ══
After landing at [land_r, land_c]:
  Apply the same 3 gates from the new position in all applicable directions.
  GATE-2 exception: mid2 must not already appear in this path’s captured list.
  STOP rule: a man that lands on its promotion row (RED→row 0, BLACK→row 7) stops — no further jumps.
  Any direction passes → MUST extend the path (stopping early = illegal partial jump).
  No direction passes → path is complete; record it.
  Recurse from each new landing until complete.

══ PHASE 3 — VERIFY before writing output ══
For each jump sequence:
  □ len(path) = len(captured) + 1
  □ captured[i] is the midpoint between path[i] and path[i+1]
  □ No square appears twice in path; no square appears twice in captured
  If any check fails → discard that move.

FORMAT — path = squares the piece STANDS ON; captured = squares jumped OVER:
  CORRECT single:  {"type":"jump","path":[[3,2],[1,4]],"captured":[[2,3]]}
                   → len(path)=2, len(captured)=1 ✓
  WRONG:           {"path":[[3,2],[2,3],[1,4]],...}  ← [2,3] is midpoint, NOT in path

  CORRECT multi:   {"type":"jump","path":[[2,5],[4,3],[6,1]],"captured":[[3,4],[5,2]]}
                   → len(path)=3, len(captured)=2 ✓
  WRONG (partial): {"path":[[2,5],[4,3]],...}  ← illegal if continuation from [4,3] exists

OUTPUT — strict JSON, no text before or after:
{"moves":[{"type":"jump","path":[...],"captured":[...]},...]
}

Output ALL legal jump sequences. Output ONLY complete sequences.
"""


def build_jump_prompt(
    board: list[list[int]],
    current_player: int,
) -> tuple[str, str]:
    """Build (system, user) prompts for the jump proposer. No hints."""
    player_label = "RED" if current_player == RED else "BLACK"
    board_grid   = render_board(board)
    piece_list   = list_pieces(board, current_player)

    user_prompt = "\n".join([
        f"Current player: {player_label}",
        "",
        piece_list,
        "",
        "BOARD:",
        board_grid,
        "",
        "Generate ALL legal capture sequences for the current player.",
        "Return STRICT JSON ONLY.",
    ])

    return JUMP_SYSTEM_PROMPT, user_prompt


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-MOVE JUMP PROPOSER PROMPT
# ══════════════════════════════════════════════════════════════════════════════
# Architecturally parallel to JUMP_SYSTEM_PROMPT.
# Inherits: all legality gates, board grounding, coordinate discipline,
#   Phase 1/2/3 procedure, format rules, anti-hallucination instructions.
# Only difference from JUMP_SYSTEM_PROMPT: task objective changes from
#   "output ALL legal capture sequences" → "output exactly ONE legal capture sequence".

JUMP_SINGLE_SYSTEM_PROMPT = """\
You are an American Checkers legal jump-move generator.
Task: output exactly ONE complete legal capture sequence for the active player.

SYMBOLS: r=RED man  R=RED king  b=BLACK man  B=BLACK king  .=empty  #=light(ignore)
Row 0=TOP  Row 7=BOTTOM  Col 0=LEFT  Col 7=RIGHT  Dark=(row+col) ODD

JUMP COORDINATES: for direction (dr,dc) from [r,c]:
  mid  = [r+dr,  c+dc ]   (opponent piece here)
  land = [r+2dr, c+2dc]   (must be empty)

DIRECTIONS:
  RED men:   NW(dr=-1,dc=-1)  NE(dr=-1,dc=+1)   [forward only, row decreases]
  BLACK men: SW(dr=+1,dc=-1)  SE(dr=+1,dc=+1)   [forward only, row increases]
  Kings:     NW  NE  SW  SE   [all four; kings may change direction between captures]

══ PHASE 1 — SCAN all pieces ══
For EVERY own piece, EVERY applicable direction:
  GATE-1: land in-bounds (both row and col in 0–7)?            Skip if no.
  GATE-2: board[mid] = OPPONENT symbol?                         Skip if no.
  GATE-3: board[land] = ‘.’?                                    Skip if no.
  All 3 pass → valid first jump. Record it.
❖ Scan ALL pieces and ALL directions before deciding output. Do not stop early.

══ PHASE 2 — CONTINUATION from each landing ══
After landing at [land_r, land_c]:
  Apply the same 3 gates from the new position in all applicable directions.
  GATE-2 exception: mid2 must not already appear in this path’s captured list.
  STOP rule: a man that lands on its promotion row (RED→row 0, BLACK→row 7) stops — no further jumps.
  Any direction passes → MUST extend the path (stopping early = illegal partial jump).
  No direction passes → path is complete; record it.
  Recurse from each new landing until complete.

══ PHASE 3 — VERIFY before writing output ══
For each jump sequence:
  □ len(path) = len(captured) + 1
  □ captured[i] is the midpoint between path[i] and path[i+1]
  □ No square appears twice in path; no square appears twice in captured
  If any check fails → discard that move.

FORMAT — path = squares the piece STANDS ON; captured = squares jumped OVER:
  CORRECT single:  {"type":"jump","path":[[3,2],[1,4]],"captured":[[2,3]]}
                   → len(path)=2, len(captured)=1 ✓
  WRONG:           {"path":[[3,2],[2,3],[1,4]],...}  ← [2,3] is midpoint, NOT in path

  CORRECT multi:   {"type":"jump","path":[[2,5],[4,3],[6,1]],"captured":[[3,4],[5,2]]}
                   → len(path)=3, len(captured)=2 ✓
  WRONG (partial): {"path":[[2,5],[4,3]],...}  ← illegal if continuation from [4,3] exists

OUTPUT — strict JSON, no text before or after:
{"moves":[{"type":"jump","path":[...],"captured":[...]}]
}

Output exactly ONE complete legal capture sequence from your verified scan results.
Output exactly ONE move in the "moves" list. Never output more than one. Do not explain.
"""

# Backward-compatible alias (deprecated — use JUMP_SINGLE_SYSTEM_PROMPT)
jump_single_best_prompt = JUMP_SINGLE_SYSTEM_PROMPT


def build_jump_single_best_prompt(
    board: list[list[int]],
    current_player: int,
) -> tuple[str, str]:
    """Build (system, user) prompts for the single-move jump proposer. No hints."""
    player_label = "RED" if current_player == RED else "BLACK"
    board_grid   = render_board(board)
    piece_list   = list_pieces(board, current_player)

    user_prompt = "\n".join([
        f"Current player: {player_label}",
        "",
        piece_list,
        "",
        "BOARD:",
        board_grid,
        "",
        "Generate exactly one legal capture sequence for the current player.",
        "Return STRICT JSON ONLY.",
    ])

    return JUMP_SINGLE_SYSTEM_PROMPT, user_prompt



# ══════════════════════════════════════════════════════════════════════════════
# QUIET PROPOSER PROMPT
# ══════════════════════════════════════════════════════════════════════════════
# Receives ONLY: board grid, current player, current player's pieces.
# Must NOT receive: movable pieces, legal moves, candidate destinations,
# helper geometry, validator hints.

QUIET_SYSTEM_PROMPT = """\
You are an American Checkers non-capturing move generator.
Task: output ALL legal simple moves for the active player.

SYMBOLS: r=RED man  R=RED king  b=BLACK man  B=BLACK king  .=empty  #=light(ignore)
Row 0=TOP  Row 7=BOTTOM  Col 0=LEFT  Col 7=RIGHT  Dark=(row+col) ODD

DIRECTIONS:
  RED men:   NW=[row-1,col-1]  NE=[row-1,col+1]
  BLACK men: SW=[row+1,col-1]  SE=[row+1,col+1]
  Kings:     NW  NE  SW  SE  (all four)

GATES — BOTH must pass to emit a move:
  GATE-A: target in-bounds: 0 ≤ row ≤ 7  AND  0 ≤ col ≤ 7
  GATE-B: board[target] = ‘.’  (read the grid — any piece symbol means occupied → skip)

PROCEDURE:
  For every own piece, in order:
    Men  → check NW first, then NE.    Emit every direction that passes both gates.
    Kings → check NW, NE, SW, SE.     Emit every direction that passes both gates.
  ❖ Do NOT stop after finding the first valid direction — continue through all.
  ❖ Do NOT move to the next piece until all its directions are checked.

COMMON MISTAKES TO AVOID:
  • Men have 2 directions — both must be checked. Found one valid? Check the other.
  • Kings have 4 directions — all four must be checked.
  • board[target] = r/R/b/B means OCCUPIED → GATE-B fails → do NOT propose it.
  • Each piece may contribute 0, 1, or 2+ moves to the output.

OUTPUT — strict JSON, no text before or after:
{"moves":[{"type":"simple","path":[[from_r,from_c],[to_r,to_c]],"captured":[]},...]}

Output ALL legal simple moves.
"""


def build_quiet_prompt(
    board: list[list[int]],
    current_player: int,
) -> tuple[str, str]:
    """Build (system, user) prompts for the quiet proposer. No hints."""
    player_label = "RED" if current_player == RED else "BLACK"
    board_grid   = render_board(board)
    piece_list   = list_pieces(board, current_player)

    user_prompt = "\n".join([
        f"Current player: {player_label}",
        "",
        piece_list,
        "",
        "BOARD:",
        board_grid,
        "",
        "Generate ALL legal non-capturing (simple) moves for the current player.",
        "Return STRICT JSON ONLY.",
    ])

    return QUIET_SYSTEM_PROMPT, user_prompt


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-MOVE QUIET PROPOSER PROMPT
# ══════════════════════════════════════════════════════════════════════════════
# Architecturally parallel to QUIET_SYSTEM_PROMPT.
# Inherits: all legality gates, board grounding, coordinate discipline,
#   procedure, format rules, anti-hallucination instructions.
# Only difference from QUIET_SYSTEM_PROMPT: task objective changes from
#   "output ALL legal simple moves" → "output exactly ONE legal simple move".

QUIET_SINGLE_SYSTEM_PROMPT = """\
You are an American Checkers non-capturing move generator.
Task: output exactly ONE complete legal simple move for the active player.

SYMBOLS: r=RED man  R=RED king  b=BLACK man  B=BLACK king  .=empty  #=light(ignore)
Row 0=TOP  Row 7=BOTTOM  Col 0=LEFT  Col 7=RIGHT  Dark=(row+col) ODD

DIRECTIONS:
  RED men:   NW=[row-1,col-1]  NE=[row-1,col+1]
  BLACK men: SW=[row+1,col-1]  SE=[row+1,col+1]
  Kings:     NW  NE  SW  SE  (all four)

GATES — BOTH must pass to emit a move:
  GATE-A: target in-bounds: 0 ≤ row ≤ 7  AND  0 ≤ col ≤ 7
  GATE-B: board[target] = ‘.’  (read the grid — any piece symbol means occupied → skip)

PROCEDURE:
  For every own piece, in order:
    Men  → check NW first, then NE.    Emit every direction that passes both gates.
    Kings → check NW, NE, SW, SE.     Emit every direction that passes both gates.
  ❖ Do NOT stop after finding the first valid direction — continue through all.
  ❖ Do NOT move to the next piece until all its directions are checked.

COMMON MISTAKES TO AVOID:
  • Men have 2 directions — both must be checked. Found one valid? Check the other.
  • Kings have 4 directions — all four must be checked.
  • board[target] = r/R/b/B means OCCUPIED → GATE-B fails → do NOT propose it.
  • Each piece may contribute 0, 1, or 2+ moves to the output.

OUTPUT — strict JSON, no text before or after:
{"moves":[{"type":"simple","path":[[from_r,from_c],[to_r,to_c]],"captured":[]}]}

Output exactly ONE complete legal simple move from your verified scan results.
Output exactly ONE move in the "moves" list. Never output more than one. Do not explain.
"""

# Backward-compatible alias (deprecated — use QUIET_SINGLE_SYSTEM_PROMPT)
quiet_single_best_prompt = QUIET_SINGLE_SYSTEM_PROMPT


def build_quiet_single_best_prompt(
    board: list[list[int]],
    current_player: int,
) -> tuple[str, str]:
    """Build (system, user) prompts for the single-move quiet proposer. No hints."""
    player_label = "RED" if current_player == RED else "BLACK"
    board_grid   = render_board(board)
    piece_list   = list_pieces(board, current_player)

    user_prompt = "\n".join([
        f"Current player: {player_label}",
        "",
        piece_list,
        "",
        "BOARD:",
        board_grid,
        "",
        "Generate exactly one legal non-capturing (simple) move for the current player.",
        "Return STRICT JSON ONLY.",
    ])

    return QUIET_SINGLE_SYSTEM_PROMPT, user_prompt


# ══════════════════════════════════════════════════════════════════════════════
# STRICT OUTPUT PARSING — NO REPAIR
# ══════════════════════════════════════════════════════════════════════════════
# Allowed: strict JSON parsing, exact schema parsing.
# Forbidden: regex move reconstruction, semantic reinterpretation,
# continuation completion, reasoning extraction, malformed move repair,
# inferred geometry.

import re as _re

_SCANNER_CAPTURE_RE = _re.compile(
    r'\{\s*"capture_available"\s*:\s*(true|false)\s*\}',
    _re.IGNORECASE,
)


def parse_scanner_output(raw: str) -> Optional[bool]:
    """
    Parse scanner output. Returns True/False for capture_available,
    or None on parse failure. No repair.

    Handles two formats:
      1. Bare JSON: {"capture_available": true}  (old format, json_mode=True)
      2. Scan-log + verdict + JSON (new scan-log format, json_mode=False):
         The JSON verdict appears on the last line; regex finds it anywhere.

    The regex is intentionally minimal — it only extracts the verdict JSON,
    not any intermediate scan log values.
    """
    if not raw or not raw.strip():
        return None

    # Try fast path: bare JSON
    try:
        data = json.loads(raw.strip())
        if isinstance(data, dict) and "capture_available" in data:
            val = data["capture_available"]
            if isinstance(val, bool):
                return val
    except (json.JSONDecodeError, TypeError):
        pass

    # Regex path: extract from scan-log mixed output
    # Take the LAST match — the verdict JSON is always written last
    matches = _SCANNER_CAPTURE_RE.findall(raw)
    if matches:
        return matches[-1].lower() == "true"

    return None


def parse_proposal_output(raw: str) -> Optional[list[dict[str, Any]]]:
    """
    Parse proposal (jump or quiet) output. Returns list of move dicts,
    or None on parse failure. No repair, no reconstruction.

    Each move must have: type, path, captured.
    Path and captured must be lists of [row, col] pairs with integer coords.
    """
    try:
        data = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    moves_raw = data.get("moves")
    if not isinstance(moves_raw, list):
        return None

    moves: list[dict[str, Any]] = []
    for entry in moves_raw:
        if not isinstance(entry, dict):
            return None  # Entire output is malformed

        move_type = entry.get("type")
        if move_type not in ("simple", "jump"):
            return None

        path_raw = entry.get("path")
        captured_raw = entry.get("captured")

        if not isinstance(path_raw, list) or not isinstance(captured_raw, list):
            return None

        # Validate path: must be list of [int, int]
        path: list[list[int]] = []
        for coord in path_raw:
            if not isinstance(coord, list) or len(coord) != 2:
                return None
            try:
                r, c = int(coord[0]), int(coord[1])
            except (ValueError, TypeError):
                return None
            path.append([r, c])

        # Validate captured: must be list of [int, int]
        captured: list[list[int]] = []
        for coord in captured_raw:
            if not isinstance(coord, list) or len(coord) != 2:
                return None
            try:
                r, c = int(coord[0]), int(coord[1])
            except (ValueError, TypeError):
                return None
            captured.append([r, c])

        moves.append({
            "type": move_type,
            "path": path,
            "captured": captured,
        })

    return moves


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT — run_proposal_seperation
# ══════════════════════════════════════════════════════════════════════════════

def run_proposal_seperation(
    board: list[list[int]],
    current_player: int,
    single_best: bool = False,
) -> dict[str, Any]:
    """
    Run the full separated proposal pipeline for one position.

    Returns a result dict containing:
      - scanner_raw:           raw scanner LLM output
      - scanner_prediction:    True/False/None
      - scanner_api_ok:        bool
      - proposal_raw:          raw proposal LLM output
      - proposal_moves:        parsed moves or None
      - proposal_api_ok:       bool
      - proposal_branch:       "jump" | "quiet"
      - api_failure:           bool (True if any API call failed completely)
      - parse_failure:         bool (True if any output was malformed)
      - scanner_parse_failure: bool
      - proposal_parse_failure: bool
    """
    result: dict[str, Any] = {
        "scanner_raw": "",
        "scanner_prediction": None,
        "scanner_api_ok": False,
        "proposal_raw": "",
        "proposal_moves": None,
        "proposal_api_ok": False,
        "proposal_branch": "",
        "api_failure": False,
        "parse_failure": False,
        "scanner_parse_failure": False,
        "proposal_parse_failure": False,
    }

    # ── Step 1: Scanner ──────────────────────────────────────────────────────
    scanner_sys, scanner_usr = build_scanner_prompt(board, current_player)

    scanner_raw, scanner_api_ok = _call_with_infra_retry(
        system=scanner_sys,
        user=scanner_usr,
        api_key=SCANNER_API_KEY,
        api_url=SCANNER_BASE_URL,
        model=SCANNER_MODEL,
        temperature=PROPOSAL_TEMPERATURE,
        stage="scanner",
        json_mode=_SCANNER_JSON_MODE,
    )

    result["scanner_raw"]    = scanner_raw
    result["scanner_api_ok"] = scanner_api_ok

    if not scanner_api_ok:
        result["api_failure"] = True
        logger.error("Scanner API failure — marking as api_failure")
        return result

    scanner_prediction = parse_scanner_output(scanner_raw)
    result["scanner_prediction"] = scanner_prediction

    if scanner_prediction is None:
        result["scanner_parse_failure"] = True
        result["parse_failure"] = True
        logger.warning("Scanner output malformed — marking parse_failure")
        # We still proceed: we use scanner's prediction as-is (None → no routing)
        # The evaluator will handle this case.
        return result

    # ── Step 2: Proposal (jump OR quiet, never both) ─────────────────────────
    if scanner_prediction is True:
        proposal_branch = "jump"
        if single_best:
            proposal_sys, proposal_usr = build_jump_single_best_prompt(board, current_player)
        else:
            proposal_sys, proposal_usr = build_jump_prompt(board, current_player)
    else:
        proposal_branch = "quiet"
        if single_best:
            proposal_sys, proposal_usr = build_quiet_single_best_prompt(board, current_player)
        else:
            proposal_sys, proposal_usr = build_quiet_prompt(board, current_player)

    result["proposal_branch"] = proposal_branch

    proposal_raw, proposal_api_ok = _call_with_infra_retry(
        system=proposal_sys,
        user=proposal_usr,
        api_key=PROPOSAL_API_KEY,
        api_url=PROPOSAL_BASE_URL,
        model=PROPOSAL_MODEL,
        temperature=PROPOSAL_TEMPERATURE,
        stage=f"proposal_{proposal_branch}",
    )

    result["proposal_raw"]    = proposal_raw
    result["proposal_api_ok"] = proposal_api_ok

    if not proposal_api_ok:
        result["api_failure"] = True
        logger.error("Proposal API failure — marking as api_failure")
        return result

    proposal_moves = parse_proposal_output(proposal_raw)
    
    # Store original length before truncation
    original_len = len(proposal_moves) if proposal_moves is not None else 0
    result["original_proposal_moves_len"] = original_len

    parsed_moves = proposal_moves
    if single_best:
        if parsed_moves:
            parsed_moves = [parsed_moves[0]]
            
    result["proposal_moves"] = parsed_moves

    if parsed_moves is None:
        result["proposal_parse_failure"] = True
        result["parse_failure"] = True
        logger.warning("Proposal output malformed — marking parse_failure")

    return result



# ══════════════════════════════════════════════════════════════════════════════
# run_proposer_only — ground-truth branch call
# ══════════════════════════════════════════════════════════════════════════════
# Used by the evaluator when the scanner routed to the wrong branch.
# Calls ONLY the proposer for the specified branch (no scanner involved).
# The scanner is still recorded as wrong; this gives the proposer a fair
# evaluation on the correct task.

def run_proposer_only(
    board: list[list[int]],
    current_player: int,
    branch: str,
    single_best: bool = False,
) -> dict[str, Any]:
    """
    Call only the proposal LLM for the specified branch ("jump" or "quiet").

    Returns:
      proposal_raw:           raw LLM output
      proposal_api_ok:        bool
      proposal_moves:         parsed moves or None
      proposal_branch:        the branch that was called
      proposal_parse_failure: bool
      api_failure:            bool
    """
    if branch == "jump":
        if single_best:
            proposal_sys, proposal_usr = build_jump_single_best_prompt(board, current_player)
        else:
            proposal_sys, proposal_usr = build_jump_prompt(board, current_player)
    else:
        if single_best:
            proposal_sys, proposal_usr = build_quiet_single_best_prompt(board, current_player)
        else:
            proposal_sys, proposal_usr = build_quiet_prompt(board, current_player)

    proposal_raw, proposal_api_ok = _call_with_infra_retry(
        system=proposal_sys,
        user=proposal_usr,
        api_key=PROPOSAL_API_KEY,
        api_url=PROPOSAL_BASE_URL,
        model=PROPOSAL_MODEL,
        temperature=PROPOSAL_TEMPERATURE,
        stage=f"proposal_{branch}_gt",
    )

    if not proposal_api_ok:
        return {
            "proposal_raw": proposal_raw,
            "proposal_api_ok": False,
            "proposal_moves": None,
            "proposal_branch": branch,
            "proposal_parse_failure": False,
            "api_failure": True,
        }

    proposal_moves = parse_proposal_output(proposal_raw)
    parse_failure  = proposal_moves is None

    original_proposal_moves_len = len(proposal_moves) if proposal_moves is not None else 0

    parsed_moves = proposal_moves
    if single_best:
        if parsed_moves:
            parsed_moves = [parsed_moves[0]]

    if parse_failure:
        logger.warning(
            "[proposal_%s_gt] output malformed — marking parse_failure", branch
        )

    return {
        "proposal_raw": proposal_raw,
        "proposal_api_ok": True,
        "proposal_moves": parsed_moves,
        "proposal_branch": branch,
        "proposal_parse_failure": parse_failure,
        "api_failure": False,
        "original_proposal_moves_len": original_proposal_moves_len,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIC BEST-MOVE SELECTION — Mode B
# ══════════════════════════════════════════════════════════════════════════════
# Completely separate from the exhaustive proposal pipeline (Mode A).
# No scanner. No move enumeration. One LLM call → one chosen move path.
#
# Output format: {"best_move": [[r,c],[r,c],...]}
# NOT "moves":[...]. This is a different schema on purpose.

STRATEGIC_BEST_MOVE_PROMPT = """\
You are an expert American Checkers strategist.

Your ONLY task: choose the single strongest legal move for the active player.

SYMBOLS: r=RED man  R=RED king  b=BLACK man  B=BLACK king  .=empty  #=light(ignore)
Row 0=TOP  Row 7=BOTTOM  Col 0=LEFT  Col 7=RIGHT  Dark=(row+col) ODD

══ MANDATORY CAPTURE RULE ══
If ANY legal capture (jump) exists, you MUST choose a capture.
You may NOT choose a simple move when a capture is available.

══ SIMPLE MOVE ══
A piece slides one diagonal step to an adjacent empty dark square.
  RED men:   NW=[row-1,col-1]  NE=[row-1,col+1]  (row DECREASES — forward only)
  BLACK men: SW=[row+1,col-1]  SE=[row+1,col+1]  (row INCREASES — forward only)
  Kings:     NW  NE  SW  SE   (all four directions)

══ CAPTURE (JUMP) ══
A piece jumps over an adjacent opponent piece to the empty square beyond.
  mid =[r+dr, c+dc]   must be an OPPONENT piece
  land=[r+2dr,c+2dc]  must be empty and in-bounds
Multi-jumps: if another capture is available from the landing square, it is mandatory.

══ OUTPUT ══
Path = sequence of squares the piece STANDS ON during the move:
  Simple move:   2 squares — [[from_r,from_c],[to_r,to_c]]
  Single jump:   2 squares — [[from_r,from_c],[land_r,land_c]]
  Multi-jump:   3+ squares — [[from],[land1],[land2],...]

Choose the move that gives the best strategic advantage.
Respond with ONLY this JSON. No text before or after. No alternatives. No explanation.

{"best_move": [[r,c],[r,c],...]}
"""


def build_strategic_best_move_prompt(
    board: list[list[int]],
    current_player: int,
) -> tuple[str, str]:
    """
    Build (system, user) prompts for the strategic best-move selector.
    No scanner involved. Handles jump and quiet positions directly.
    """
    player_label = "RED" if current_player == RED else "BLACK"
    board_grid   = render_board(board)
    piece_list   = list_pieces(board, current_player)

    user_prompt = "\n".join([
        f"Current player: {player_label}",
        "",
        piece_list,
        "",
        "BOARD:",
        board_grid,
        "",
        "Choose the single strongest legal move for the current player.",
        'Respond ONLY with: {"best_move": [[r,c],[r,c],...]}',
    ])

    return STRATEGIC_BEST_MOVE_PROMPT, user_prompt


def parse_best_move_output(raw: str) -> Optional[list[list[int]]]:
    """
    Parse strategic best-move output.

    Expected format: {"best_move": [[r,c],[r,c],...]}

    Returns path as list of [row, col] pairs (minimum length 2),
    or None on any parse failure.
    No repair, no reconstruction. Does NOT touch parse_proposal_output.
    """
    if not raw or not raw.strip():
        return None
    try:
        data = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    path_raw = data.get("best_move")
    if not isinstance(path_raw, list) or len(path_raw) < 2:
        return None

    path: list[list[int]] = []
    for coord in path_raw:
        if not isinstance(coord, list) or len(coord) != 2:
            return None
        try:
            r, c = int(coord[0]), int(coord[1])
        except (ValueError, TypeError):
            return None
        path.append([r, c])

    return path


def run_strategic_selection(
    board: list[list[int]],
    current_player: int,
) -> dict[str, Any]:
    """
    Run strategic best-move selection for one position.

    No scanner. One single LLM call.
    The model directly chooses the single strongest legal move.

    Returns dict with keys:
      strategic_raw:           raw LLM output
      strategic_best_move:     parsed path (list of [r,c]) or None
      strategic_api_ok:        bool
      strategic_parse_failure: bool
      api_failure:             bool
    """
    sys_prompt, usr_prompt = build_strategic_best_move_prompt(board, current_player)

    raw, api_ok = _call_with_infra_retry(
        system=sys_prompt,
        user=usr_prompt,
        api_key=PROPOSAL_API_KEY,
        api_url=PROPOSAL_BASE_URL,
        model=PROPOSAL_MODEL,
        temperature=PROPOSAL_TEMPERATURE,
        stage="strategic_best_move",
    )

    if not api_ok:
        return {
            "strategic_raw": raw,
            "strategic_best_move": None,
            "strategic_api_ok": False,
            "strategic_parse_failure": False,
            "api_failure": True,
        }

    parsed_path = parse_best_move_output(raw)
    parse_failure = parsed_path is None

    if parse_failure:
        logger.warning("[strategic_best_move] output malformed — marking parse_failure")

    return {
        "strategic_raw": raw,
        "strategic_best_move": parsed_path,
        "strategic_api_ok": True,
        "strategic_parse_failure": parse_failure,
        "api_failure": False,
    }

