# checkers/agents/board_proposal_agent.py
#
# LLM-based board-grounded move proposal agent  (v7)
#
# ── ISOLATION CONTRACT ─────────────────────────────────────────────────────────
# Allowed reads from state:
#   board, current_player, strategic_context (game_phase + score_state ONLY)
#
# Forbidden reads (would leak engine knowledge):
#   legal_moves, symbolic_scored_moves, symbolic_best_move, symbolic_best_score,
#   proposed_moves, ranker_diagnostics, chosen_move_facts, any minimax output.
#
# The parser (parse_proposal_output / normalize_candidate) is format-only:
#   it may strip fences, parse JSON, normalize int types, drop unreadable entries.
#   It must NOT check legality, call get_all_legal_moves(), add/repair moves.
#
# ── Provider selection ─────────────────────────────────────────────────────────
# PROPOSAL_PROVIDER must be set explicitly in .env. Supported values:
#
#   github_models      OpenAI-compatible endpoint via GitHub Models
#                      Requires: GITHUB_MODELS_API_KEY
#                      Optional: GITHUB_MODELS_BASE_URL (default: https://models.github.ai/inference)
#
#   openai             Official OpenAI API
#                      Requires: OPENAI_API_KEY
#
#   mistral            Mistral AI via raw urllib POST (no extra deps)
#                      Requires: MISTRAL_API_KEY
#                      Optional: MISTRAL_API_URL (default: https://api.mistral.ai/v1/chat/completions)
#
#   openai_compatible  Any other OpenAI-compatible endpoint
#                      Requires: PROPOSAL_OPENAI_COMPATIBLE_API_KEY
#                                PROPOSAL_OPENAI_COMPATIBLE_BASE_URL
#
# Model / temperature (shared across all providers):
#   PROPOSAL_MODEL        — model name string (required; backward compat: PROPOSAL_MISTRAL_MODEL)
#   PROPOSAL_TEMPERATURE  — float, default 0.2

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Optional

# ── Load .env with override=True so .env always wins over stale shell exports ──
# This ensures switching PROPOSAL_PROVIDER in .env takes effect immediately,
# even if the shell session has old env vars already exported.
# python-dotenv is optional — skipped silently if not installed.
try:
    from dotenv import load_dotenv as _load_dotenv
    _ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    _load_dotenv(os.path.abspath(_ENV_PATH), override=True)
except ImportError:
    pass  # python-dotenv not installed — env vars must be set by caller


from checkers.engine.board import (
    EMPTY, RED, BLACK, RED_KING, BLACK_KING, BOARD_SIZE,
)
from checkers.state.state import CheckersState

logger = logging.getLogger(__name__)

# ── Provider / model configuration ────────────────────────────────────────────
# PROPOSAL_PROVIDER must be set explicitly in .env — no silent default.
_raw_provider = os.environ.get("PROPOSAL_PROVIDER", "").strip().lower()
if not _raw_provider:
    raise EnvironmentError(
        "PROPOSAL_PROVIDER is not set. "
        "Add it to your .env file. "
        "Supported values: github_models | openai | mistral | openai_compatible"
    )
PROPOSAL_PROVIDER = _raw_provider

# Model: PROPOSAL_MODEL takes precedence; PROPOSAL_MISTRAL_MODEL kept for backward compat.
PROPOSAL_MODEL = (
    os.environ.get("PROPOSAL_MODEL", "").strip()
    or os.environ.get("PROPOSAL_MISTRAL_MODEL", "").strip()
)
if not PROPOSAL_MODEL:
    raise EnvironmentError(
        "PROPOSAL_MODEL is not set. "
        "Add it to your .env file (e.g. PROPOSAL_MODEL=openai/gpt-4.1)."
    )

PROPOSAL_TEMPERATURE    = float(os.environ.get("PROPOSAL_TEMPERATURE",    "0.2"))
# LEGACY — used only by parse_proposal_output / the old board_proposal_agent() path.
# DO NOT import into benchmark code (silently caps n_proposed).
PROPOSAL_MAX_CANDIDATES = int(os.environ.get("PROPOSAL_MAX_CANDIDATES",   "10"))
PROPOSAL_REASON_FIRST   = os.environ.get("PROPOSAL_REASON_FIRST", "false").strip().lower() == "true"

# ── Per-provider credentials ───────────────────────────────────────────────────

# github_models
GITHUB_MODELS_API_KEY  = os.environ.get("GITHUB_MODELS_API_KEY",  "")
GITHUB_MODELS_BASE_URL = os.environ.get(
    "GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference"
)

# openai
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# mistral
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_API_URL = os.environ.get(
    "MISTRAL_API_URL", "https://api.mistral.ai/v1/chat/completions"
)

# openai_compatible  (generic third-party OpenAI-compatible endpoint)
OPENAI_COMPATIBLE_API_KEY  = os.environ.get("PROPOSAL_OPENAI_COMPATIBLE_API_KEY",  "")
OPENAI_COMPATIBLE_BASE_URL = os.environ.get("PROPOSAL_OPENAI_COMPATIBLE_BASE_URL", "")

# ── Per-stage Proposal credentials (new pipeline) ─────────────────────────────
# Stage 1 — Scanner
SCANNER_MISTRAL_API_KEY = os.environ.get("SCANNER_MISTRAL_API_KEY", "")
SCANNER_MISTRAL_MODEL   = os.environ.get("SCANNER_MISTRAL_MODEL",   "mistral-large-latest")
SCANNER_MISTRAL_URL     = os.environ.get("SCANNER_MISTRAL_URL",
                                          "https://api.mistral.ai/v1/chat/completions")

# Stage 2 — Jump + Quiet (shared)
JUMP_QUIET_MISTRAL_API_KEY = os.environ.get("JUMP_QUIET_MISTRAL_API_KEY", "")
JUMP_QUIET_MISTRAL_MODEL   = os.environ.get("JUMP_QUIET_MISTRAL_MODEL",   "mistral-large-latest")
JUMP_QUIET_MISTRAL_URL     = os.environ.get("JUMP_QUIET_MISTRAL_URL",
                                             "https://api.mistral.ai/v1/chat/completions")

# Stage 3 — Final JSON Builder (reserved; pure Python today)
FINAL_JSON_MISTRAL_API_KEY = os.environ.get("FINAL_JSON_MISTRAL_API_KEY", "")
FINAL_JSON_MISTRAL_MODEL   = os.environ.get("FINAL_JSON_MISTRAL_MODEL",   "mistral-large-latest")
FINAL_JSON_MISTRAL_URL     = os.environ.get("FINAL_JSON_MISTRAL_URL",
                                             "https://api.mistral.ai/v1/chat/completions")

# Computed label for traces (includes base_url when relevant)
def _provider_base_url() -> Optional[str]:
    """Return the base_url in use for the current provider, or None."""
    if PROPOSAL_PROVIDER == "github_models":
        return GITHUB_MODELS_BASE_URL
    if PROPOSAL_PROVIDER == "openai_compatible":
        return OPENAI_COMPATIBLE_BASE_URL or None
    return None


# ── Output-format header constants ────────────────────────────────────────────

_OUTPUT_HEADER_JSON_ONLY = (
    "══ OUTPUT — compact JSON, no text before or after ════════════════════"
)

_REASON_FIRST_OUTPUT_HEADER = """\
══ OUTPUT FORMAT — reason-first ════════════════════════════════════
Step 1 — Write your chain-of-thought reasoning under EXACTLY this header (required):

DRAFT_BOARD_REASONING:
  1. BOARD FACTS: for each own piece, state [row,col], type, and applicable directions.
     KING direction order: NW [r-1,c-1], NE [r-1,c+1], SW [r+1,c-1], SE [r+1,c+1] — always all four.
  2. JUMP SCAN RESULTS: for each piece/direction: piece[r,c] dir: mid=[r,c] mid_val=X  land=[r,c] land_val=X  valid=true|false
  3. N_VALID: count total valid=true jump_checks. Write "N_VALID=N".
  4. BRANCH DECISION:
     • "BRANCH: NO-CAPTURE (N_VALID=0)" → For every SIMPLE GEOMETRY TARGET, read to_val, state valid=true/false.
       For each KING: process NW → NE → SW → SE without stopping early.
     • "BRANCH: JUMP (N_VALID≥1)" → list valid jumps, continuation scan, complete paths.
  5. PLANNED OUTPUT: list each entry for final_proposed_moves.
  6. COMPLETENESS CHECK (N_VALID=0 branch only):
     List every valid=true simple_check as: "(from_r,from_c)→(to_r,to_c)".
     Count: N_VALID_SIMPLE = <number>.
     Confirm planned output covers all N_VALID_SIMPLE pairs.
     Write: "COMPLETENESS: N_VALID_SIMPLE=N, final_proposed_moves covers N — all covered? yes/no."
     If the answer is "no", add the missing entries to PLANNED OUTPUT before proceeding.

Step 2 — Write exactly this tag alone on its own line:
<FINAL_JSON>

Step 3 — Write the JSON object immediately after (schema:)"""


# ── System prompt (compact, ~42 lines) ────────────────────────────────────────

BOARD_PROPOSAL_SYSTEM_PROMPT = """\
You are an American Checkers legal-move generator.
Task: enumerate ALL legal candidate moves for the active player, working ONLY from
the board grid and the rules below. This is move DISCOVERY, not move selection.

══ BOARD SYMBOLS (absolute — same regardless of which side you play) ══
  r = RED man    R = RED king    b = BLACK man    B = BLACK king
  . = empty dark square    # = light square (never playable, ignore)
Row 0 = TOP.  Row 7 = BOTTOM.  Col 0 = LEFT.  Col 7 = RIGHT.
Dark square: (row + col) is ODD. No piece is ever on a light (#) square.

══ JUMP FORMAT — PATH vs CAPTURED (read before writing any jump) ══════
path     = squares where the piece STANDS: [start, landing_1, landing_2, ...]
captured = squares jumped OVER (midpoints): [mid_1, mid_2, ...]
INVARIANT: len(path) = len(captured) + 1.   ALWAYS.
  • DO NOT put midpoint squares inside path.
  • DO NOT revisit any square in path.
  • DO NOT extend path after a man promotes at the landing square.

EXAMPLE A — RED man at [5,2] jumps over [4,3], lands at [3,4]:
  CORRECT: {"type":"jump","path":[[5,2],[3,4]],"captured":[[4,3]]}
  WRONG:   {"type":"jump","path":[[5,2],[4,3],[3,4]],...}  ← [4,3] is midpoint, NOT in path

EXAMPLE B — BLACK man at [2,3] jumps over [3,4], lands at [4,5]:
  CORRECT: {"type":"jump","path":[[2,3],[4,5]],"captured":[[3,4]]}

EXAMPLE C — RED multi-jump [5,2]→[3,4]→[1,6] (captures [4,3] then [2,5]):
  CORRECT: {"type":"jump","path":[[5,2],[3,4],[1,6]],"captured":[[4,3],[2,5]]}
  len(path)=3, len(captured)=2 →  3 = 2+1 ✓
  WRONG (partial): {"type":"jump","path":[[5,2],[3,4]],"captured":[[4,3]]}  ← INVALID if [1,6]='.' and [2,5]=BLACK
  Rule: after landing at [3,4], check [2,5]→[1,6]; if [2,5] is BLACK and [1,6]='.' → MUST continue.
  Partial paths are ALWAYS INVALID when another jump is available from the landing square.

EXAMPLE F — Continuation scan walkthrough for RED [5,2]→[3,4]→[1,6]:
  Step 1 — initial jump from [5,2]:
    NE dir: mid=[4,3] (b), land=[3,4] (.) → valid=true. Land at [3,4].
  Step 2 — CONTINUATION SCAN from landing [3,4] (RED man, forward = row-1):
    NW dir: mid=[2,3], land=[1,2]  (check board values)
    NE dir: mid=[2,5] (b), land=[1,6] (.) → continuation valid! EXTEND path.
  Step 3 — no further jumps from [1,6] (row 1 ≠ promotion row 0; check NW/NE from [1,6]):
    NW: mid=[0,5], land=[-1,4] → out-of-bounds, skip.
    NE: mid=[0,7], land=[-1,8] → out-of-bounds, skip.
  Final: {"type":"jump","path":[[5,2],[3,4],[1,6]],"captured":[[4,3],[2,5]]}
  ★ The continuation scan MUST happen before finalizing any jump path.

EXAMPLE D — BLACK multi-jump [2,5]→[4,3]→[6,1] (captures [3,4] then [5,2]):
  CORRECT: {"type":"jump","path":[[2,5],[4,3],[6,1]],"captured":[[3,4],[5,2]]}
  len(path)=3, len(captured)=2 → 3 = 2+1 ✓
  WRONG (partial): {"type":"jump","path":[[2,5],[4,3]],"captured":[[3,4]]}  ← INVALID if [6,1] is empty
  Rule: after landing at [4,3], check [5,2]→[6,1]; if [5,2] is RED and [6,1]='.' → MUST continue.
  Partial paths are ALWAYS INVALID when another jump is available from the landing square.

EXAMPLE E — Multiple RED pieces each have a valid capture:
  Board: RED at [5,0] and [5,4].  BLACK at [4,1] and [4,5].  [3,2] and [3,6] are empty.
  Scan: [5,0] SE → mid=[4,1] (b) land=[3,2] (.) → valid=true
        [5,4] SE → mid=[4,5] (b) land=[3,6] (.) → valid=true
  capture_available_estimate = true  (≥1 valid jump found)
  CORRECT output — BOTH jumps, zero simples:
    {"type":"jump","path":[[5,0],[3,2]],"captured":[[4,1]]}
    {"type":"jump","path":[[5,4],[3,6]],"captured":[[4,5]]}
  WRONG: proposing only one jump, or mixing simples with jumps.

══ JUMP CONDITIONS — all three must hold for EACH leg ═════════════════
  ① board[start]    = YOUR piece
  ② board[midpoint] = OPPONENT piece  (not '.', not your own symbol)
  ③ board[landing]  = '.'  AND  (landing_row + landing_col) is ODD

══ SCAN — read the board before proposing any jump ════════════════════
Directions: NW=[r-1,c-1]  NE=[r-1,c+1]  SW=[r+1,c-1]  SE=[r+1,c+1]
For each direction you check, read the actual symbols from the board grid:
  mid_val  = symbol at board[mid_row][mid_col]
  land_val = symbol at board[landing_row][landing_col]
  valid = true ONLY when mid_val = OPPONENT symbol AND land_val = '.'
NEVER mark valid=true if:
  • mid_val is '.' (empty) — no opponent to capture.
  • mid_val is your OWN symbol (r/R for RED, b/B for BLACK) — cannot capture own piece.
  • land_val is r, R, b, B, or '#' — landing square is occupied or a light square.
  • You are not certain of the actual symbol — mark valid=false and skip that jump.
SCAN ALL OWN PIECES — do not stop after the first piece or first valid jump.
  Iterate over every own piece on the board and check its jump directions.
  capture_available_estimate = true if ANY piece has ANY valid jump_check.
  If true, every piece that has a valid jump MUST contribute a jump entry to final_proposed_moves.
EDGE-OF-BOARD: coordinates outside 0–7 are out-of-bounds. Do NOT include those jump_check entries.
  An edge piece may have only 1 or 2 valid diagonal directions; that is normal — list only in-bounds ones.
  Same applies to simples: if a diagonal target is out-of-bounds, skip it silently.
OPENING ANTI-HALLUCINATION — standard opening board (RED rows 5–7, BLACK rows 0–2):
  Every piece is surrounded by own pieces in adjacent diagonals — mid_val = OWN symbol.
  Landing squares (2 steps) are also occupied by own pieces — land_val ≠ '.'.
  CONSEQUENCE: in a pure opening position, ALL jump_checks are valid=false.
    capture_available_estimate MUST be false. Output simples ONLY.
  Do NOT infer a capture from memory or assumption. Read mid_val and land_val from the board grid.
  If mid_val ≠ opponent symbol OR land_val ≠ '.', the jump is INVALID — do not propose it.

══ RULES ══════════════════════════════════════════════════════════════
R1. Simple: one diagonal step to an adjacent EMPTY (.) dark square.
      Men: forward direction only.  Kings: any of 4 diagonals.
      ▶ BEFORE listing any simple move, verify board[target] == '.'.
        Do NOT propose a simple move if target contains r, R, b, B, or #.
        RED man   at [r,c]: targets [r-1,c-1] and [r-1,c+1] — only if board[target]='.'.
        BLACK man at [r,c]: targets [r+1,c-1] and [r+1,c+1] — only if board[target]='.'.
        King      at [r,c]: targets all 4 diagonals      — only if board[target]='.'.
R2. Jump: conditions ①②③ per leg. Landing = 2 diagonal steps from start.
      Men: forward direction only.  Kings: any of 4 diagonals.
R3. Mandatory capture: if ANY valid jump exists → ONLY jumps in output.
R4. Multi-jump — MANDATORY CONTINUATION ALGORITHM:
      After each successful jump landing, before finalizing the path:
      (a) SCAN from the landing square in all forward directions (men) or all 4 (kings).
      (b) Skip any direction whose midpoint is already in the captured list.
      (c) If mid_val=OPPONENT and land_val='.' and land is in-bounds:
            → continuation exists — APPEND landing to path, APPEND mid to captured.
            → repeat from (a) at the new landing until no continuation found.
      (d) Stop ONLY when no continuation is found OR man reaches promotion row.
      A path is PARTIAL (INVALID) if any continuation was possible but not taken.
      ▶ For RED: forward = row-1. From [r,c] check NW=[r-1,c-1] and NE=[r-1,c+1].
      ▶ For BLACK: forward = row+1. From [r,c] check SW=[r+1,c-1] and SE=[r+1,c+1].
      See EXAMPLE F for the explicit step-by-step walkthrough.
R5. Promotion stop: man landing on promotion row → end path immediately.
R6. Own piece at midpoint → condition ② fails → NOT a jump.

══ OUTPUT — compact JSON, no text before or after ════════════════════
{
  "side_to_move": "RED|BLACK",
  "capture_available_estimate": true|false,
  "scan": [
    {
      "piece": [r,c],
      "piece_type": "RED_MAN|RED_KING|BLACK_MAN|BLACK_KING",
      "jump_checks": [
        {"id":"J_r_c_dir","dir":"NW|NE|SW|SE","mid":[r,c],"mid_val":"r|R|b|B|.","land":[r,c],"land_val":"r|R|b|B|.","valid":true|false}
      ],
      "continuation_checks": [
        {"id":"C_r_c_dir_stepN","from":[r,c],"dir":"NW|NE|SW|SE","mid":[r,c],"mid_val":"r|R|b|B|.","land":[r,c],"land_val":"r|R|b|B|.","valid":true|false}
      ]
    }
  ],
  "simple_checks": [
    {"id":"S_r_c_dir","dir":"NW|NE|SW|SE","from":[r,c],"to":[r,c],"to_val":".|r|R|b|B|#|OUT","valid":true|false,"reason":"target empty|target occupied|out of bounds|wrong direction for man"}
  ],
  "final_proposed_moves": [
    {"type":"simple","path":[[r,c],[r,c]],"captured":[],"source_check_id":"S_r_c_dir"},
    {"type":"jump","path":[[r,c],[r,c]],"captured":[[r,c]],"source_check_ids":["J_r_c_dir"]},
    {"type":"jump","path":[[r,c],[r,c],[r,c]],"captured":[[r,c],[r,c]],"source_check_ids":["J_r_c_dir","C_r_c_dir_step2"]}
  ]
}
Constraints:
  ● SCAN-GATE INVARIANT (hard rule, never violate):
    Every type="jump" in final_proposed_moves MUST be backed by a scan jump_check entry where:
      piece = path[0],  mid = captured[0],  land = path[1],  valid = true.
    If no such matching scan entry exists, the jump MUST NOT appear in final_proposed_moves.
    If capture_available_estimate=true but no valid=true scan entry exists, the output is malformed.
    When uncertain about any board square value — set valid=false and omit that jump.
  - SCAN ALL pieces before deciding capture_available_estimate.
  - capture_available_estimate=true ↔ at least one jump_check has valid=true in ANY scan entry.
  - capture_available_estimate=true  → ALL moves type="jump", zero simples.
    Include a jump entry for EVERY distinct starting piece that has a valid=true jump_check.
  - capture_available_estimate=false → ALL moves type="simple", zero jumps.
  - scan: include ALL pieces with ≥1 checked direction; omit pieces with zero jump directions. Cap 8 entries.
  - scan: omit out-of-bounds jump_check entries (any coord outside 0–7).
  - continuation_checks: OPTIONAL field per scan entry. Add it for any piece where a jump_check is valid=true.
    Shows the re-scan from the landing square. Cap at 4 entries per piece. Omit if no first jump.
  - Every jump's captured[0] must be backed by a valid=true jump_check in scan.
  - Propose 1–8 moves (exactly the count of legal moves when that count is < 3).
  - path[0]: YOUR piece.  path[-1]: board shows '.'.
  - Jump: len(path)=len(captured)+1. Consecutive path entries: 2 diagonal steps apart.
  - All coords: integers 0–7. Every path entry: (row+col) is ODD. No duplicate paths.
  - Out-of-bounds targets (row or col < 0 or > 7) are illegal — never propose them.
  ● SIMPLE-GATE INVARIANT (hard rule, applies when capture_available_estimate=false):
    Every type="simple" in final_proposed_moves MUST be backed by a simple_checks entry where:
      from = path[0],  to = path[1],  valid = true,  to_val = ".".
    Never propose a simple from visual guesswork — record the check first, then propose.
  - simple_checks: omit entirely when capture_available_estimate=true (no simples in that case).
  - simple_checks: when capture_available_estimate=false, one entry per own piece per checked
    direction (men: ≤2 dirs; kings: ≤4 dirs). Cap 24 total entries across all pieces.
  ● KING SIMPLE COMPLETENESS (hard rule): When capture_available_estimate=false, every KING
    MUST have simple_checks entries for ALL 4 diagonals: NW [r-1,c-1], NE [r-1,c+1],
    SW [r+1,c-1], SE [r+1,c+1]. Mark out-of-bounds directions valid=false with to_val='OUT'.
    Omitting ANY in-bounds king diagonal from simple_checks is a SIMPLE-GATE violation.
  ● NO-CAPTURE OUTPUT COMPLETENESS (hard rule): When capture_available_estimate=false:
    (a) Count N_VALID_SIMPLE = number of simple_checks where valid=true.
    (b) final_proposed_moves MUST contain EXACTLY N_VALID_SIMPLE type='simple' entries (up to candidate cap).
    (c) For EVERY valid=true simple_check there MUST be a matching entry in final_proposed_moves.
    (d) Omitting ANY valid=true simple_check is FORBIDDEN — including the last SW/SE direction of a king.
    (e) KING direction order: NW → NE → SW → SE. Do NOT stop after NW/NE. Do NOT skip SW or SE.
    Producing fewer than N_VALID_SIMPLE entries when all valid=true checks are present is an error.
  ● SOURCE-CHECK LINK INVARIANT (hard rule):
    Assign a unique stable id to every check entry before writing final_proposed_moves:
      jump_check id:        "J_{r}_{c}_{dir}"          where [r,c]=piece, dir=NW|NE|SW|SE
      continuation id:      "C_{r}_{c}_{dir}_{stepN}"  where [r,c]=from, N=leg number (2,3,...)
      simple_check id:      "S_{r}_{c}_{dir}"          where [r,c]=from, dir=NW|NE|SW|SE
    Every type="simple" MUST carry "source_check_id": the id of its backing simple_checks entry
      (valid=true, from=path[0], to=path[1]).
    Every type="jump" MUST carry "source_check_ids": a list with len=len(captured), one id per leg:
      source_check_ids[0]  = id of the jump_checks entry   (valid=true, piece=path[0], land=path[1]).
      source_check_ids[k]  = id of the continuation_checks entry (valid=true) for leg k+1.
    A move with missing source id, wrong-length list, or any id with valid=false MUST be removed.
  ● PARTIAL-PATH GATE (hard rule):
    A jump is PARTIAL if any in-bounds continuation_checks entry from path[-1] has valid=true
    and its mid is not already in captured. PARTIAL jumps MUST NOT appear in final_proposed_moves.
    Extend the path fully first. Only write terminal (complete) jump chains.
"""


# ── Board renderer (absolute symbols, unchanged from v2) ──────────────────────

def _render_board(board: list[list[int]], current_player: int) -> str:
    """
    Renders the 8×8 board with absolute symbols.
    current_player is accepted for API compatibility but does not affect symbols.
    """
    _SYM: dict[int, str] = {
        RED:        " r",
        RED_KING:   " R",
        BLACK:      " b",
        BLACK_KING: " B",
        EMPTY:      " .",
    }

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


# ── Piece list (updated: explicit type labels) ────────────────────────────────

def _list_pieces(board: list[list[int]], current_player: int) -> str:
    """
    Compact piece summary with explicit RED_MAN / RED_KING / BLACK_MAN / BLACK_KING
    labels so the LLM knows exactly which pieces have which movement rules.
    """
    red_men:     list[str] = []
    red_kings:   list[str] = []
    black_men:   list[str] = []
    black_kings: list[str] = []

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            p  = board[row][col]
            sq = f"[{row},{col}]"
            if   p == RED:          red_men.append(sq)
            elif p == RED_KING:     red_kings.append(sq)
            elif p == BLACK:        black_men.append(sq)
            elif p == BLACK_KING:   black_kings.append(sq)

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


# ── Empty playable dark squares ──────────────────────────────────────────────

def _list_empty_dark_squares(board: list[list[int]]) -> str:
    """
    Returns all empty playable dark squares as "[r,c] [r,c] ..." in row-major order.

    Dark square: (row + col) is ODD — identical convention to _render_board.
    Only squares where board[r][c] == EMPTY are included.
    Light squares (#) and occupied squares are never included.

    Soft cap: 24 entries shown; any remainder is reported as "(+N more)" so
    the prompt stays bounded while the true count remains visible.
    """
    squares: list[str] = [
        f"[{row},{col}]"
        for row in range(BOARD_SIZE)
        for col in range(BOARD_SIZE)
        if (row + col) % 2 == 1 and board[row][col] == EMPTY
    ]
    _CAP = 24
    if len(squares) <= _CAP:
        return " ".join(squares) if squares else "(none)"
    return " ".join(squares[:_CAP]) + f"  (+{len(squares) - _CAP} more)"


# ── Prompt builder (dynamic side rules per current_player) ────────────────────

def build_board_proposal_prompt(
    board: list[list[int]],
    current_player: int,
    strategic_context: Optional[dict[str, Any]] = None,
    reason_first: Optional[bool] = None,
) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt).

    Side-specific rules (direction, jump formula, promotion row) are derived
    entirely from current_player — no engine output, no legal_moves, no scores.
    strategic_context is restricted to game_phase + score_state only.
    reason_first overrides PROPOSAL_REASON_FIRST when provided explicitly.
    """
    _reason_first = reason_first if reason_first is not None else PROPOSAL_REASON_FIRST
    if current_player == RED:
        player_label = "RED"
        own_man      = "r (man)"
        own_king     = "R (king)"
        opp_man      = "b (man)"
        opp_king     = "B (king)"
        man_move     = "row -1 (upward).    from [r,c]: [r-1, c-1] and [r-1, c+1]"
        man_jump     = "from [r,c]: mid=[r-1, c±1]  land=[r-2, c±2]"
        promo_row    = 0
    else:
        player_label = "BLACK"
        own_man      = "b (man)"
        own_king     = "B (king)"
        opp_man      = "r (man)"
        opp_king     = "R (king)"
        man_move     = "row +1 (downward).  from [r,c]: [r+1, c-1] and [r+1, c+1]"
        man_jump     = "from [r,c]: mid=[r+1, c±1]  land=[r+2, c±2]"
        promo_row    = 7

    ctx         = strategic_context or {}
    phase       = ctx.get("game_phase", "MIDGAME")
    score_state = ctx.get("score_state", "EQUAL")

    board_grid = _render_board(board, current_player)
    piece_list = _list_pieces(board, current_player)
    # true count computed separately so the prompt header is accurate even when
    # _list_empty_dark_squares truncates to _CAP entries
    n_empty    = sum(
        1 for r in range(BOARD_SIZE)
        for c in range(BOARD_SIZE)
        if (r + c) % 2 == 1 and board[r][c] == EMPTY
    )
    empty_sq     = _list_empty_dark_squares(board)
    geometry_str = _list_simple_geometry_targets(board, current_player)

    user_lines: list[str] = [
        f"YOU PLAY {player_label}.",
        "",
        "━━ SIDE-SPECIFIC RULES (apply these for this call) ━━━━━━━━━━━━━━━",
        f"  Your pieces:     {own_man},  {own_king}",
        f"  Opponent pieces: {opp_man},  {opp_king}",
        f"  Man simple:      {man_move}",
        f"  Man jump:        {man_jump}",
        f"  King:            moves and jumps in ALL 4 diagonals — [r-1,c-1] [r-1,c+1] [r+1,c-1] [r+1,c+1]",
        f"  Promotion row:   {promo_row}  (your man reaching row {promo_row} becomes a king — end multi-jump there)",
        f"  Phase / score:   {phase} / {score_state}",
        "",
        piece_list,
        "",
        f"EMPTY PLAYABLE DARK SQUARES ({n_empty} squares — every square where board[r][c]='.'):",
        empty_sq,
        "  ▶ Any target or landing [r,c] NOT in this list is occupied or light — mark valid=false immediately.",
        "",
        "━━ SIMPLE GEOMETRY TARGETS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "In-bounds diagonal targets for simple moves (NOT legal moves — read to_val from board below):",
        "When capture_available_estimate=false, EVERY listed target MUST appear in simple_checks.",
        "Set valid=true ONLY if to_val='.' — read the actual board symbol for each target.",
        geometry_str,
        "",
        "BOARD  (r=RED man, R=RED king, b=BLACK man, B=BLACK king, .=empty dark, #=light/skip):",
        board_grid,
        "",
        "LOOKUP: to read square [row,col], find row labeled 'r{row}' and column 'c{col}'.",
        "",
        "━━ JUMP REMINDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  • path  = [start, landing_1, ...]  — squares where the piece STANDS.",
        "  • captured = [mid_1, ...]  — jumped-over squares, NEVER in path.",
        f"  • Scan first: mid_val must be {opp_man[0]} or {opp_king[0]} (OPPONENT) and land_val must be '.'.",
        "  • Do NOT add any jump to final_proposed_moves unless its scan entry has valid=true.",
        f"  • Do NOT extend path after piece reaches row {promo_row} (promotion stop).",
        "",
        "━━ PROCEDURE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "A) SCAN ALL OWN PIECES (do not stop early):",
        "   For EVERY own piece on the board, check jump directions (men: forward only; kings: all 4).",
        "   Skip any direction where mid or land is out-of-bounds (row/col < 0 or > 7).",
        "   Edge pieces naturally have fewer valid directions — that is fine, check only in-bounds ones.",
        "   For each direction, look up mid_val and land_val in the board grid above.",
        "   Before setting valid=true, verify BOTH conditions from the grid:",
        "     • mid_val is exactly the opponent symbol (not '.', not your own symbol, not '#')",
        "     • land_val is exactly '.' (not r/R/b/B/#)",
        "     • landing cross-check: the landing [r,c] MUST appear in EMPTY PLAYABLE DARK SQUARES.",
        "       If it is absent from that list, it is occupied — set land_val to its actual symbol, valid=false.",
        "   If any condition fails → valid=false for that direction. Do NOT propose that jump.",
        "   Opening-board reminder: in starting positions, your adjacent pieces fill the mid squares",
        "   and your own pieces fill the landing squares → all valid=false → capture_available_estimate=false.",
        "   Continue scanning EVERY piece even after finding the first valid jump.",
        "B) capture_available_estimate = true iff ANY jump_check across ALL scan entries has valid=true.",
        "   ▶ SCAN-GATE: if capture_available_estimate=true, every proposed jump MUST match a valid=true",
        "     scan entry (same piece, same mid, same landing). If you cannot find such an entry in your",
        "     scan, set capture_available_estimate=false and output simples instead.",
        "B.5) ══ MANDATORY GATE CHECK — execute this before writing ANY final_proposed_moves ══",
        "   Count N_VALID = total number of jump_check entries across ALL scan entries where valid=true.",
        "   • IF N_VALID = 0:",
        "       → capture_available_estimate = false.",
        "       → final_proposed_moves must contain ONLY type='simple' entries.",
        "       → Do NOT write any type='jump' entry. STOP jump reasoning here.",
        "       → KING SIMPLE COMPLETENESS: for each KING piece, simple_checks MUST include",
        "         entries for ALL 4 diagonals in order NW → NE → SW → SE.",
        "         Mark out-of-bounds directions valid=false (to_val='OUT'). Do NOT skip SW or SE.",
        "       → OUTPUT COMPLETENESS (strict): follow these steps in order:",
        "         Step 1: After recording all simple_checks, enumerate every valid=true pair:",
        "                 'valid_pairs = [(from1,to1), (from2,to2), ...]'. Write them out explicitly.",
        "         Step 2: N_VALID_SIMPLE = len(valid_pairs). Write 'N_VALID_SIMPLE=N'.",
        "         Step 3: final_proposed_moves MUST contain EXACTLY N_VALID_SIMPLE entries.",
        "         Step 4: For EACH pair in valid_pairs, add the corresponding simple move.",
        "                 Do NOT stop early. Do NOT omit a pair because it's SW or SE.",
        "         Step 5: Write 'All N_VALID_SIMPLE pairs covered in final_proposed_moves? yes'.",
        "                 If 'no', add the missing entries before writing <FINAL_JSON>.",
        "         Omitting ANY valid=true simple_check = silent completeness error = FORBIDDEN.",
        "   • IF N_VALID ≥ 1:",
        "       → capture_available_estimate = true.",
        "       → Proceed to step C to build jump entries.",
        "   This gate is NON-NEGOTIABLE. The scan is the only source of truth.",
        "   Opening boards always produce N_VALID=0 → always output simples.",
        "   After confirming N_VALID: every jump in final_proposed_moves requires source_check_ids",
        "   (one id per leg, each id valid=true). A jump with missing or invalid source_check_ids",
        "   must NOT appear in final_proposed_moves.",
        "C) Build final_proposed_moves:",
        "   If true:  ONLY type='jump'. For EACH piece that has ≥1 valid=true jump_check, build its jump.",
        "             Do not skip pieces with valid captures — include all of them.",
        "",
        "   ★ CONTINUATION SCAN (mandatory before finalizing any jump):",
        "     After the first valid jump leg lands at square L:",
        "       (1) Check all forward directions from L (men: 2 dirs; kings: 4 dirs).",
        "       (2) Skip any direction whose mid is already in captured list.",
        f"       (3) If mid_val=OPPONENT and land_val='.' and in-bounds → EXTEND path with L2, EXTEND captured.",
        "       (4) Repeat from (1) at the new landing. Stop only when no extension found.",
        "       (5) Record these checks in continuation_checks for transparency.",
        "       (6) Assign id='C_{from_r}_{from_c}_{dir}_step{N}' to each continuation_checks entry (N≥2).",
        f"     For {player_label} men: forward = {'row-1 (NW/NE)' if player_label == 'RED' else 'row+1 (SW/SE)'}.",
        "     PARTIAL paths are INVALID. A path is final only when no continuation exists.",
        "     See EXAMPLE C and EXAMPLE F in the system prompt.",
        "",
        f"   Multi-jump reminder ({player_label}): after each landing, verify board[mid]=OPPONENT",
        "             and board[landing]='.' for every subsequent leg — extend if valid.",
        "   If false: ONLY type='simple'. Use the SIMPLE GEOMETRY TARGETS listed above as your checklist.",
        "             For EVERY listed target [to_r,to_c], record a simple_checks entry:",
        "               • Look up to_val = symbol at board[to_r][to_c] in the grid above.",
        "               • Target cross-check: [to_r,to_c] MUST appear in EMPTY PLAYABLE DARK SQUARES above.",
        "                 If absent → it is occupied; set to_val to its actual symbol, valid=false.",
        "               • valid=true ONLY if to_val='.' (empty dark square, in-bounds).",
        "               • valid=false if to_val is r/R/b/B (occupied), '#' (light), or out-of-bounds.",
        "               • reason = 'target empty' | 'target occupied' | 'out of bounds' | 'wrong direction for man'.",
        "             KING DIRECTION ORDER: for each KING, process NW → NE → SW → SE without stopping early.",
        "             EVERY geometry target MUST have a simple_checks entry — do not skip any listed target.",
        "             COMPLETENESS STEP — after recording all simple_checks:",
        "               (i)  Enumerate every valid=true pair: 'valid_pairs=[(from1,to1),(from2,to2),...]'.",
        "               (ii) N_VALID_SIMPLE = len(valid_pairs).",
        "               (iii) final_proposed_moves = one entry per pair in valid_pairs (preserve order).",
        "               (iv) Confirm: 'All N_VALID_SIMPLE pairs in final_proposed_moves? yes.'",
        "                    If no: add the missing entries before writing final JSON.",
        "             SIMPLE-GATE: add to final_proposed_moves ONLY if its simple_check has valid=true.",
        "             Do NOT propose any simple move without a backing valid=true simple_check entry.",
        "D) Re-verify every entry:",
        "   • Simples: source_check_id present? id valid=true in simple_checks (from=path[0], to=path[1])?",
        "             (reject if source_check_id missing, id not found, or to_val≠'.')",
        "   • All coords in [0,7]? (reject out-of-bounds targets)",
        "   • board[start]=YOUR piece?  board[path[-1]]='.'?",
        "   • Jumps: source_check_ids present? len(source_check_ids)=len(captured)?",
        "     Each id valid=true in scan (jump_checks or continuation_checks)?",
        "     PARTIAL check: any valid=true continuation_checks from path[-1] (mid not in captured)?",
        "     → If yes: path is PARTIAL — extend fully before writing. Never output partial jumps.",
        "     (also: len(path)=len(captured)+1? No midpoint in path? Steps exactly 2 diagonal apart?)",
        "   • Remove any entry that fails any check above.",
    ]

    if _reason_first:
        user_lines += [
            "",
            "━━ OUTPUT FORMAT REMINDER (REQUIRED) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "Respond with:",
            "  1. DRAFT_BOARD_REASONING: [your reasoning — board facts, scan, branch, plan]",
            "  2. <FINAL_JSON>  (this exact tag, alone on its own line)",
            "  3. The complete JSON object immediately after <FINAL_JSON>.",
            "Do NOT put any text between <FINAL_JSON> and the opening brace of the JSON.",
        ]
        system_prompt = BOARD_PROPOSAL_SYSTEM_PROMPT.replace(
            _OUTPUT_HEADER_JSON_ONLY, _REASON_FIRST_OUTPUT_HEADER
        )
    else:
        system_prompt = BOARD_PROPOSAL_SYSTEM_PROMPT

    return system_prompt, "\n".join(user_lines)


# ── LLM API call ───────────────────────────────────────────────────────────────

def _call_openai_compatible(
    system: str,
    user: str,
    api_key: str,
    base_url: Optional[str],
    model: str,
) -> str:
    """
    Calls any OpenAI-compatible endpoint using langchain_openai.ChatOpenAI
    with JSON mode enforced via response_format.

    Used by: github_models, openai, openai_compatible providers.
    Raises ImportError if langchain_openai is not installed.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain_openai is required for this provider. "
            "Install it with:  pip install langchain-openai"
        ) from exc

    kwargs: dict[str, Any] = {
        "model":        model,
        "api_key":      api_key,
        "temperature":  PROPOSAL_TEMPERATURE,
    }
    if not PROPOSAL_REASON_FIRST:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    if base_url:
        kwargs["base_url"] = base_url

    llm = ChatOpenAI(**kwargs)

    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [SystemMessage(content=system), HumanMessage(content=user)]

    response = llm.invoke(messages)
    content  = response.content if hasattr(response, "content") else str(response)
    if not isinstance(content, str):
        raise ValueError(
            f"Unexpected response content type {type(content)}: {str(content)[:300]}"
        )
    return content


def _call_mistral_raw(system: str, user: str) -> str:
    """
    Calls Mistral AI via a raw urllib POST (no extra dependencies).
    Uses MISTRAL_API_KEY and MISTRAL_API_URL from env.
    Raises ValueError on non-200 or missing content.
    """
    if not MISTRAL_API_KEY:
        raise ValueError(
            "MISTRAL_API_KEY is not set. "
            "Add MISTRAL_API_KEY=<key> to your .env file."
        )

    payload: dict[str, Any] = {
        "model":       PROPOSAL_MODEL,
        "temperature": PROPOSAL_TEMPERATURE,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    if not PROPOSAL_REASON_FIRST:
        payload["response_format"] = {"type": "json_object"}

    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        MISTRAL_API_URL,
        data=body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Accept":        "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        if e.code == 429:
            time.sleep(15)
        raise ValueError(f"Mistral API HTTP {e.code}: {body_text[:300]}") from e

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ValueError(
            f"Unexpected Mistral response structure: {str(data)[:300]}"
        ) from exc

    if not isinstance(content, str):
        raise ValueError(f"Mistral content is not a string: {type(content)}")
    return content


def _call_mistral_raw_for(
    system: str,
    user: str,
    api_key: str,
    api_url: str,
    model: str,
    temperature: float,
    reason_first: bool,
) -> str:
    """
    Parameterized Mistral raw POST call.
    Used by per-stage callers (scanner, jump/quiet, final-json).
    """
    if not api_key:
        raise ValueError(
            f"Mistral API key not set for this stage. "
            f"Check SCANNER_MISTRAL_API_KEY / JUMP_QUIET_MISTRAL_API_KEY / "
            f"FINAL_JSON_MISTRAL_API_KEY in your .env file."
        )

    payload: dict[str, Any] = {
        "model":       model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    if not reason_first:
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

    try:
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        if e.code == 429:
            time.sleep(15)
        raise ValueError(f"Mistral API HTTP {e.code}: {body_text[:300]}") from e

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ValueError(
            f"Unexpected Mistral response structure: {str(data)[:300]}"
        ) from exc

    if not isinstance(content, str):
        raise ValueError(f"Mistral content is not a string: {type(content)}")
    return content


def call_scanner_llm(system: str, user: str) -> str:
    """LLM call for Stage 1 — Scanner. Uses SCANNER_MISTRAL_* env vars."""
    return _call_mistral_raw_for(
        system=system, user=user,
        api_key=SCANNER_MISTRAL_API_KEY,
        api_url=SCANNER_MISTRAL_URL,
        model=SCANNER_MISTRAL_MODEL,
        temperature=PROPOSAL_TEMPERATURE,
        reason_first=PROPOSAL_REASON_FIRST,
    )


def call_jump_quiet_llm(system: str, user: str) -> str:
    """LLM call for Stage 2 — Jump + Quiet. Uses JUMP_QUIET_MISTRAL_* env vars."""
    return _call_mistral_raw_for(
        system=system, user=user,
        api_key=JUMP_QUIET_MISTRAL_API_KEY,
        api_url=JUMP_QUIET_MISTRAL_URL,
        model=JUMP_QUIET_MISTRAL_MODEL,
        temperature=PROPOSAL_TEMPERATURE,
        reason_first=PROPOSAL_REASON_FIRST,
    )


# ── Infrastructure retry helper ───────────────────────────────────────────────

_INFRA_RETRY_CYCLES = 2             # total retry cycles after the initial call
_INFRA_RETRY_DELAYS = [20, 30, 40]  # wait seconds before each retry within a cycle

_infra_logger = logging.getLogger(__name__ + ".infra_retry")


def call_with_infra_retry(
    llm_fn: Any,
    sys_p: str,
    usr_p: str,
    stage: str,
) -> tuple[str, bool, int]:
    """
    Call llm_fn with infrastructure-only progressive backoff retry.

    Retry policy
    ------------
      2 retry cycles, each with 3 attempts (20 s → 30 s → 40 s wait before each).
      Retries on ANY exception or empty/blank response (transport / availability).

    NEVER retries for:
      parse failures, semantic errors, reasoning failures, or illegal proposals.
      Those are content issues — surfacing them raw is intentional.

    Returns (raw_output, api_ok, n_infra_retries).
    """
    n_retries = 0

    # ── Initial call ──────────────────────────────────────────────────────────
    try:
        raw = llm_fn(sys_p, usr_p)
        if raw and raw.strip():
            return raw, True, 0
        error_desc = "EMPTY_RESPONSE"
    except Exception as exc:
        error_desc = str(exc)[:200]

    _infra_logger.warning("[%s] initial call failed: %s", stage, error_desc)

    # ── Retry cycles (2 cycles × 3 delays each) ───────────────────────────────
    for cycle in range(1, _INFRA_RETRY_CYCLES + 1):
        for i, delay in enumerate(_INFRA_RETRY_DELAYS, 1):
            _infra_logger.warning(
                "[%s] infra retry → cycle %d/%d attempt %d/%d in %ds: %s",
                stage, cycle, _INFRA_RETRY_CYCLES, i, len(_INFRA_RETRY_DELAYS),
                delay, error_desc,
            )
            time.sleep(delay)
            n_retries += 1
            try:
                raw = llm_fn(sys_p, usr_p)
                if raw and raw.strip():
                    _infra_logger.info(
                        "[%s] infra retry recovered: cycle %d attempt %d (%d total retries)",
                        stage, cycle, i, n_retries,
                    )
                    return raw, True, n_retries
                error_desc = "EMPTY_RESPONSE"
            except Exception as exc:
                error_desc = str(exc)[:200]
            _infra_logger.warning(
                "[%s] retry cycle %d attempt %d failed: %s",
                stage, cycle, i, error_desc,
            )

    _infra_logger.error(
        "[%s] all infra retry cycles exhausted (%d retries attempted)", stage, n_retries,
    )
    return "", False, n_retries


def call_final_json_llm(system: str, user: str) -> str:
    """LLM call for Stage 3 — Final JSON Builder. Uses FINAL_JSON_MISTRAL_* env vars."""
    return _call_mistral_raw_for(
        system=system, user=user,
        api_key=FINAL_JSON_MISTRAL_API_KEY,
        api_url=FINAL_JSON_MISTRAL_URL,
        model=FINAL_JSON_MISTRAL_MODEL,
        temperature=PROPOSAL_TEMPERATURE,
        reason_first=PROPOSAL_REASON_FIRST,
    )


def call_board_proposal_llm(system: str, user: str) -> str:
    """
    Dispatches to the provider configured by PROPOSAL_PROVIDER in .env.

    Supported providers:
      github_models     OpenAI-compatible via GitHub Models
                        Requires: GITHUB_MODELS_API_KEY
                        Optional: GITHUB_MODELS_BASE_URL

      openai            Official OpenAI API
                        Requires: OPENAI_API_KEY

      mistral           Mistral AI (raw urllib, no extra deps)
                        Requires: MISTRAL_API_KEY
                        Optional: MISTRAL_API_URL

      openai_compatible Any other OpenAI-compatible endpoint
                        Requires: PROPOSAL_OPENAI_COMPATIBLE_API_KEY
                                  PROPOSAL_OPENAI_COMPATIBLE_BASE_URL

    Returns raw content string.
    Raises ValueError on missing credentials or unknown provider.
    """
    if PROPOSAL_PROVIDER == "github_models":
        if not GITHUB_MODELS_API_KEY:
            raise ValueError(
                "PROPOSAL_PROVIDER=github_models requires GITHUB_MODELS_API_KEY. "
                "Add it to your .env file."
            )
        return _call_openai_compatible(
            system=system, user=user,
            api_key=GITHUB_MODELS_API_KEY,
            base_url=GITHUB_MODELS_BASE_URL,
            model=PROPOSAL_MODEL,
        )

    if PROPOSAL_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError(
                "PROPOSAL_PROVIDER=openai requires OPENAI_API_KEY. "
                "Add it to your .env file."
            )
        return _call_openai_compatible(
            system=system, user=user,
            api_key=OPENAI_API_KEY,
            base_url=None,
            model=PROPOSAL_MODEL,
        )

    if PROPOSAL_PROVIDER == "mistral":
        return _call_mistral_raw(system, user)

    if PROPOSAL_PROVIDER == "openai_compatible":
        if not OPENAI_COMPATIBLE_API_KEY:
            raise ValueError(
                "PROPOSAL_PROVIDER=openai_compatible requires PROPOSAL_OPENAI_COMPATIBLE_API_KEY. "
                "Add it to your .env file."
            )
        if not OPENAI_COMPATIBLE_BASE_URL:
            raise ValueError(
                "PROPOSAL_PROVIDER=openai_compatible requires PROPOSAL_OPENAI_COMPATIBLE_BASE_URL. "
                "Add it to your .env file."
            )
        return _call_openai_compatible(
            system=system, user=user,
            api_key=OPENAI_COMPATIBLE_API_KEY,
            base_url=OPENAI_COMPATIBLE_BASE_URL,
            model=PROPOSAL_MODEL,
        )

    raise ValueError(
        f"Unknown PROPOSAL_PROVIDER={PROPOSAL_PROVIDER!r}. "
        "Supported values: github_models | openai | mistral | openai_compatible"
    )


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY SINGLE-STAGE PIPELINE — NOT USED BY STAGED BENCHMARK
#
# The three items below (normalize_candidate, parse_proposal_output,
# PROPOSAL_MAX_CANDIDATES) belong exclusively to the old monolithic
# board_proposal_agent() code path (lines ~1940+).
#
# They perform repair and normalization that MUST NOT appear in the staged
# pipeline (scan → jump/quiet) or its benchmarks.  Importing any of these
# into benchmark code constitutes silent contamination of the forensic signal.
#
# Specifically forbidden in benchmark imports:
#   normalize_candidate     — silently discards and repairs malformed entries
#   parse_proposal_output   — regex fallback + dedup + candidate cap
#   PROPOSAL_MAX_CANDIDATES — silently caps n_proposed, masking over-generation
#
# These remain here only to keep board_proposal_agent() functional.
# They will be removed when the legacy pipeline is retired.
# ══════════════════════════════════════════════════════════════════════════════

# ── Candidate normalizer (format-only, unchanged from v2) ─────────────────────

def normalize_candidate(raw_move: Any) -> Optional[dict[str, Any]]:
    """
    Format-only normalization of a single raw candidate dict.

    ALLOWED: convert path/captured to list[list[int]], normalize type string,
             drop entries with out-of-range coords, drop entries with path < 2.
    FORBIDDEN: legality checking, calling get_all_legal_moves(), adding or
               repairing moves, completing partial jump paths.

    Returns None → caller should discard this entry.
    """
    if not isinstance(raw_move, dict):
        return None

    raw_type = str(raw_move.get("type", "simple")).strip().lower()
    if raw_type not in ("simple", "jump"):
        raw_type = "simple"

    raw_path = raw_move.get("path", [])
    if not isinstance(raw_path, list):
        return None

    path: list[list[int]] = []
    try:
        for sq in raw_path:
            if isinstance(sq, (list, tuple)) and len(sq) == 2:
                r, c = int(sq[0]), int(sq[1])
                if not (0 <= r <= 7 and 0 <= c <= 7):
                    return None
                path.append([r, c])
            else:
                return None
    except (TypeError, ValueError):
        return None

    if len(path) < 2:
        return None

    raw_cap = raw_move.get("captured", [])
    if not isinstance(raw_cap, list):
        raw_cap = []

    captured: list[list[int]] = []
    try:
        for sq in raw_cap:
            if isinstance(sq, (list, tuple)) and len(sq) == 2:
                r, c = int(sq[0]), int(sq[1])
                if not (0 <= r <= 7 and 0 <= c <= 7):
                    return None
                captured.append([r, c])
            else:
                return None
    except (TypeError, ValueError):
        return None

    return {"type": raw_type, "path": path, "captured": captured}


# ── Reason-first JSON extractor ───────────────────────────────────────────────

def _extract_final_json_text(raw: str) -> tuple[str, bool, bool]:
    """
    Extracts the JSON text to parse from a raw LLM response.

    Returns (json_text, marker_found, draft_present):
      json_text     — text to pass to json.loads (after marker, or full text)
      marker_found  — True if <FINAL_JSON> marker was present
      draft_present — True if DRAFT_BOARD_REASONING section was present

    If <FINAL_JSON> is present: extracts everything after it (ignores draft).
    Otherwise: returns the full stripped text (backward-compat fallback).
    Markdown fences are stripped from the extracted text in both cases.
    """
    _MARKER = "<FINAL_JSON>"
    marker_found  = _MARKER in raw
    draft_present = "DRAFT_BOARD_REASONING" in raw

    if marker_found:
        idx  = raw.index(_MARKER) + len(_MARKER)
        text = raw[idx:].strip()
    else:
        text = raw.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text  = "\n".join(lines).strip()

    return text, marker_found, draft_present


# ── Output parser (format-only, unchanged from v2) ────────────────────────────

def parse_proposal_output(raw: str) -> list[dict[str, Any]]:
    """
    Parses raw LLM JSON and returns normalized final_proposed_moves.

    ALLOWED: strip fences, JSON parse, regex fallback, normalize via
             normalize_candidate(), deduplicate by path, cap at PROPOSAL_MAX_CANDIDATES.
    FORBIDDEN: legality checking, get_all_legal_moves(), adding/repairing moves.

    Returns [] on any unrecoverable failure.
    """
    text, _marker_found, _draft_present = _extract_final_json_text(raw)

    obj: Optional[dict] = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    if not isinstance(obj, dict):
        logger.warning("[board_proposal] parse_proposal_output: no valid JSON object found")
        return []

    raw_candidates = obj.get("final_proposed_moves")
    if raw_candidates is None:
        raw_candidates = obj.get("candidates", [])

    if not isinstance(raw_candidates, list):
        logger.warning("[board_proposal] 'final_proposed_moves' is not a list")
        return []

    normalized: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for raw_cand in raw_candidates:
        norm = normalize_candidate(raw_cand)
        if norm is None:
            continue
        path_key = json.dumps(norm["path"], separators=(",", ":"))
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        normalized.append(norm)
        if len(normalized) >= PROPOSAL_MAX_CANDIDATES:
            break

    return normalized


# ── Debug field extractor (simplified for compact schema) ─────────────────────

def _extract_debug_fields(raw: str) -> dict[str, Any]:
    """
    Extracts top-level debug fields from the JSON schema, including the scan array.
    Never raises. Returns {} on any failure.
    """
    try:
        text, _marker_found, _draft_present = _extract_final_json_text(raw)
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return {}
        scan          = obj.get("scan")
        simple_checks = obj.get("simple_checks")
        llm_final_raw = obj.get("final_proposed_moves")
        return {
            "capture_available_estimate": obj.get("capture_available_estimate"),
            "side_to_move":               obj.get("side_to_move"),
            "scan":                       scan          if isinstance(scan, list) else None,
            "simple_checks":              simple_checks if isinstance(simple_checks, list) else None,
            "llm_final_raw":              llm_final_raw if isinstance(llm_final_raw, list) else None,
        }
    except Exception:
        return {}


# ── Simple-gate unbacked counter ──────────────────────────────────────────────

def _count_simple_unbacked(proposals: list[dict], simple_checks: Any) -> int:
    """
    Counts type='simple' proposals that have no backing valid=true simple_checks entry
    (from == path[0], to == path[1]).  Mirrors _count_unverified_jumps for simples.
    Returns 0 when every simple is properly backed.  Goal is 0.
    """
    if not isinstance(simple_checks, list):
        # No simple_checks field at all — every simple is unbacked
        return sum(1 for p in proposals if p.get("type") == "simple")

    valid_pairs: set[tuple] = set()
    for chk in simple_checks:
        if not isinstance(chk, dict) or chk.get("valid") is not True:
            continue
        frm = chk.get("from")
        to  = chk.get("to")
        if (isinstance(frm, (list, tuple)) and len(frm) == 2
                and isinstance(to, (list, tuple)) and len(to) == 2):
            valid_pairs.add(((int(frm[0]), int(frm[1])), (int(to[0]), int(to[1]))))

    count = 0
    for prop in proposals:
        if prop.get("type") != "simple":
            continue
        path = prop.get("path", [])
        if len(path) < 2:
            count += 1
            continue
        try:
            frm_key = (int(path[0][0]), int(path[0][1]))
            to_key  = (int(path[1][0]), int(path[1][1]))
        except (IndexError, TypeError, ValueError):
            count += 1
            continue
        if (frm_key, to_key) not in valid_pairs:
            count += 1
    return count


# ── Missed valid simples diagnostic ──────────────────────────────────────────

def _count_missed_valid_simples(
    proposals: list[dict],
    simple_checks: Any,
) -> int:
    """
    Counts valid=true simple_checks entries (from the LLM's own output) that are
    NOT represented as a type='simple' in proposals (path[0]==from, path[1]==to).

    Pure post-hoc diagnostic — never modifies anything.
    Returns 0 when simple_checks is absent/empty.  Goal = 0.

    A non-zero value means the LLM prepared the evidence but omitted moves from output.
    """
    if not isinstance(simple_checks, list):
        return 0

    valid_pairs: list[tuple] = []
    for chk in simple_checks:
        if not isinstance(chk, dict) or chk.get("valid") is not True:
            continue
        frm = chk.get("from")
        to  = chk.get("to")
        if (isinstance(frm, (list, tuple)) and len(frm) == 2
                and isinstance(to, (list, tuple)) and len(to) == 2):
            try:
                valid_pairs.append(
                    ((int(frm[0]), int(frm[1])), (int(to[0]), int(to[1])))
                )
            except (TypeError, ValueError):
                continue

    if not valid_pairs:
        return 0

    proposed: set[tuple] = set()
    for prop in proposals:
        if prop.get("type") != "simple":
            continue
        path = prop.get("path", [])
        if len(path) >= 2:
            try:
                proposed.add(
                    ((int(path[0][0]), int(path[0][1])), (int(path[1][0]), int(path[1][1])))
                )
            except (IndexError, TypeError, ValueError):
                continue

    return sum(1 for pair in valid_pairs if pair not in proposed)


# ── Simple geometry scaffolding ───────────────────────────────────────────────

_DIRS_MAN_RED   = [("NW", -1, -1), ("NE", -1, +1)]
_DIRS_MAN_BLACK = [("SW", +1, -1), ("SE", +1, +1)]
_DIRS_KING      = [("NW", -1, -1), ("NE", -1, +1), ("SW", +1, -1), ("SE", +1, +1)]


def _simple_geometry_pairs(
    board: list[list[int]],
    current_player: int,
) -> list[tuple]:
    """
    Returns all in-bounds simple-move target tuples for each own piece:
      ((from_r, from_c), (to_r, to_c), dir_name, piece_label)

    Derived from piece positions and movement geometry only — no occupancy check.
    Direction rules:
      RED man:   NW and NE
      BLACK man: SW and SE
      KING:      NW, NE, SW, SE
    """
    if current_player == RED:
        piece_dir_map = [
            (RED,       "RED_MAN",   _DIRS_MAN_RED),
            (RED_KING,  "RED_KING",  _DIRS_KING),
        ]
    else:
        piece_dir_map = [
            (BLACK,      "BLACK_MAN",  _DIRS_MAN_BLACK),
            (BLACK_KING, "BLACK_KING", _DIRS_KING),
        ]

    result: list[tuple] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            p = board[row][col]
            for piece_val, piece_label, dirs in piece_dir_map:
                if p != piece_val:
                    continue
                for dir_name, dr, dc in dirs:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr <= 7 and 0 <= nc <= 7:
                        result.append(((row, col), (nr, nc), dir_name, piece_label))
    return result


def _list_simple_geometry_targets(
    board: list[list[int]],
    current_player: int,
) -> str:
    """
    Returns a formatted string listing in-bounds simple target coordinates for
    each own piece, derived from geometry only (no occupancy or legality check).

    Intended for the user prompt. The LLM must still read to_val from the board
    and set valid=true only if to_val='.'.
    """
    pairs = _simple_geometry_pairs(board, current_player)
    if not pairs:
        return "(no own pieces on board)"

    by_piece: dict[tuple, list[str]] = {}
    piece_labels: dict[tuple, str]   = {}
    for (fr, fc), (tr, tc), dir_name, piece_label in pairs:
        key = (fr, fc)
        if key not in by_piece:
            by_piece[key]      = []
            piece_labels[key]  = piece_label
        by_piece[key].append(f"{dir_name}→[{tr},{tc}]")

    lines: list[str] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            key = (row, col)
            if key in by_piece:
                label = piece_labels[key]
                targets_str = "  ".join(by_piece[key])
                lines.append(f"  [{row},{col}] {label}:  {targets_str}")
    return "\n".join(lines)


def _count_missing_simple_checks_for_geometry(
    board: list[list[int]],
    current_player: int,
    simple_checks: Any,
) -> int:
    """
    Counts geometry target (from, to) pairs that have NO corresponding entry
    in the LLM's simple_checks output (regardless of valid/invalid).

    A non-zero value means the LLM skipped some geometry targets entirely.
    Returns 0 if simple_checks is absent/empty.  Goal = 0.
    """
    pairs = _simple_geometry_pairs(board, current_player)
    # Return 0 when simple_checks is absent or empty — we can only diagnose
    # missing entries when the LLM has produced at least one check entry.
    if not pairs or not simple_checks or not isinstance(simple_checks, list):
        return 0

    checked: set[tuple] = set()
    for chk in simple_checks:
        if not isinstance(chk, dict):
            continue
        frm = chk.get("from")
        to  = chk.get("to")
        if (isinstance(frm, (list, tuple)) and len(frm) == 2
                and isinstance(to, (list, tuple)) and len(to) == 2):
            try:
                checked.add(((int(frm[0]), int(frm[1])), (int(to[0]), int(to[1]))))
            except (TypeError, ValueError):
                continue

    return sum(1 for (from_sq, to_sq, _, _) in pairs if (from_sq, to_sq) not in checked)


# ── Source-check grounding diagnostics ────────────────────────────────────────

def _count_grounding_failures(
    raw_finals: Any,
    scan: Any,
    simple_checks: Any,
) -> dict[str, int]:
    """
    Post-hoc grounding check: are every jump/simple's cited source IDs valid?

    Inspects source_check_ids (jumps) and source_check_id (simples) in the
    raw final_proposed_moves list — extracted before normalize_candidate strips
    those extra fields.  Never modifies proposals; purely diagnostic.

    Returns four counters (goal=0 for all):
      unlinked_jump_count     — jumps with no source_check_ids field (or empty list)
      bad_source_jump_count   — jumps with ids present but any id invalid / wrong count
      unlinked_simple_count   — simples with no source_check_id field
      bad_source_simple_count — simples with source_check_id not in valid simple_checks
    """
    result = {
        "unlinked_jump_count":     0,
        "bad_source_jump_count":   0,
        "unlinked_simple_count":   0,
        "bad_source_simple_count": 0,
    }
    if not isinstance(raw_finals, list):
        return result

    # Collect all valid=true ids from scan (jump_checks + continuation_checks)
    valid_jump_ids: set[str] = set()
    if isinstance(scan, list):
        for entry in scan:
            if not isinstance(entry, dict):
                continue
            for chk in entry.get("jump_checks", []):
                if isinstance(chk, dict) and chk.get("valid") is True:
                    cid = chk.get("id")
                    if isinstance(cid, str) and cid:
                        valid_jump_ids.add(cid)
            for chk in entry.get("continuation_checks", []):
                if isinstance(chk, dict) and chk.get("valid") is True:
                    cid = chk.get("id")
                    if isinstance(cid, str) and cid:
                        valid_jump_ids.add(cid)

    # Collect all valid=true ids from simple_checks
    valid_simple_ids: set[str] = set()
    if isinstance(simple_checks, list):
        for chk in simple_checks:
            if isinstance(chk, dict) and chk.get("valid") is True:
                cid = chk.get("id")
                if isinstance(cid, str) and cid:
                    valid_simple_ids.add(cid)

    for move in raw_finals:
        if not isinstance(move, dict):
            continue
        mtype = str(move.get("type", "simple")).strip().lower()
        if mtype not in ("simple", "jump"):
            mtype = "simple"

        if mtype == "jump":
            src_ids = move.get("source_check_ids")
            if not isinstance(src_ids, list) or not src_ids:
                result["unlinked_jump_count"] += 1
            else:
                captured = move.get("captured", [])
                bad = (len(src_ids) != len(captured)) or any(
                    not isinstance(sid, str) or sid not in valid_jump_ids
                    for sid in src_ids
                )
                if bad:
                    result["bad_source_jump_count"] += 1
        else:
            src_id = move.get("source_check_id")
            if not isinstance(src_id, str) or not src_id:
                result["unlinked_simple_count"] += 1
            else:
                if src_id not in valid_simple_ids:
                    result["bad_source_simple_count"] += 1

    return result


# ── Contradiction detector (uses only LLM's own output fields) ───────────────

def _detect_contradictions(
    dbg: dict[str, Any],
    n_valid_scan_jumps: int,
    grounding: dict[str, int],
    n_jump_proposals: int,
) -> list[str]:
    """
    Inspects the LLM's own output fields to find internal contradictions.
    Never reads engine legal moves — only compares scan/check/source-id fields
    against final_proposed_moves.

    Returns a list of human-readable reason strings.  Empty list → no retry.

    Triggers:
      A) n_valid_scan_jumps == 0 AND jumps proposed   (scan-gate violated)
      B) capture_available_estimate=false AND jumps proposed
      C) unlinked_jump_count > 0  (missing source_check_ids on a jump)
      D) bad_source_jump_count > 0  (source id invalid / wrong length)
    """
    reasons: list[str] = []
    capture_est = dbg.get("capture_available_estimate")

    # A — scan gate: no valid scan entry but jumps proposed
    if n_valid_scan_jumps == 0 and n_jump_proposals > 0:
        reasons.append(
            f"Your final_proposed_moves contains {n_jump_proposals} jump(s) but your "
            f"scan has N_VALID=0 (zero valid=true jump_check entries). "
            f"Rule B.5: N_VALID=0 → output only type='simple'. "
            f"Remove all type='jump' entries or fix your scan so valid=true entries exist."
        )

    # B — estimate contradicts proposals
    if capture_est is False and n_jump_proposals > 0:
        reasons.append(
            f"capture_available_estimate=false but final_proposed_moves contains "
            f"{n_jump_proposals} jump(s). When capture_available_estimate=false, "
            f"ONLY type='simple' moves are allowed — remove all type='jump' entries."
        )

    # C — SOURCE-CHECK LINK: missing source_check_ids on a jump
    n_unlinked_j = grounding["unlinked_jump_count"]
    if n_unlinked_j > 0:
        reasons.append(
            f"{n_unlinked_j} jump(s) in final_proposed_moves are missing source_check_ids. "
            f"The SOURCE-CHECK LINK INVARIANT requires every type='jump' to carry "
            f"source_check_ids with len=len(captured), each id valid=true in scan. "
            f"Assign ids to every jump or remove that jump."
        )

    # D — SOURCE-CHECK LINK: source_check_ids present but invalid / wrong count
    n_bad_j = grounding["bad_source_jump_count"]
    if n_bad_j > 0:
        reasons.append(
            f"{n_bad_j} jump(s) have source_check_ids that are invalid: "
            f"id not found in scan, valid=false in scan, or len(source_check_ids)≠len(captured). "
            f"Each id must refer to a jump_check or continuation_checks entry with valid=true, "
            f"and len(source_check_ids) must equal len(captured)."
        )

    return reasons


def _build_retry_user_prompt(original_user: str, reasons: list[str]) -> str:
    """
    Appends a CORRECTION FEEDBACK section to the original user prompt.
    References only the LLM's own output fields — never engine legal moves.
    The system prompt is unchanged; only the user turn is extended.
    """
    lines = [
        original_user,
        "",
        "━━ CORRECTION FEEDBACK — your previous output violated these invariants ━━",
        f"  {len(reasons)} issue(s) detected:",
    ]
    for i, reason in enumerate(reasons, 1):
        lines.append(f"  [{i}] {reason}")
    lines += [
        "",
        "  Regenerate the complete JSON output from scratch.",
        "  Core rules:",
        "    • Re-read every board square from the grid above before claiming any value.",
        "    • Do NOT copy any move from your previous output — re-derive from the board.",
        "    • Do NOT use engine-provided or external legal move lists.",
        "",
        "  If your rescan has N_VALID=0 (no valid=true jump_check entries):",
        "    → Set capture_available_estimate=false.",
        "    → Enter the NO-CAPTURE BRANCH:",
        "        (1) Rebuild simple_checks using the SIMPLE GEOMETRY TARGETS listed above as your checklist.",
        "            Every listed target [to_r,to_c] MUST produce one simple_checks entry:",
        "            • RED man:   NW [r-1,c-1] and NE [r-1,c+1] only.",
        "            • BLACK man: SW [r+1,c-1] and SE [r+1,c+1] only.",
        "            • KING:      ALL 4 diagonals — NW [r-1,c-1], NE [r-1,c+1], SW [r+1,c-1], SE [r+1,c+1].",
        "            For each target: read to_val from the board grid. valid=true ONLY if to_val='.'.",
        "            Do NOT skip any target listed in SIMPLE GEOMETRY TARGETS — missing entries are errors.",
        "            KING COMPLETENESS: every KING must produce exactly 4 simple_checks entries.",
        "        (2) Build final_proposed_moves from simple_checks: add ONLY entries where",
        "            valid=true. Assign source_check_id to each simple move.",
        "        (3) Do NOT return an empty final_proposed_moves list unless every own piece",
        "            has zero valid simple_check entries (to_val='.' for no direction).",
        "            If any simple_check has valid=true, that move MUST appear in the output.",
        "        (4) Do NOT include any type='jump' entries in this branch.",
        "",
        "  If your rescan has N_VALID≥1 (valid jumps exist):",
        "    → Set capture_available_estimate=true.",
        "    → Output ONLY type='jump' entries, each with source_check_ids (len=len(captured)).",
        "    → Every id in source_check_ids must be valid=true in jump_checks or continuation_checks.",
    ]
    return "\n".join(lines)


# ── Post-retry safety filter ─────────────────────────────────────────────────

def _filter_contradictory_proposals(
    candidates: list[dict[str, Any]],
    dbg: dict[str, Any],
    n_valid_scan_jumps: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Post-retry safety filter: removes proposals that contradict the LLM's own evidence.
    Called only when contradiction_retry_count==1 AND contradictions persist.

    Never adds replacement moves, never calls the engine.

    Drop rules:
      Jumps — scan-gate violation (n_valid_scan_jumps==0): drop ALL jumps.
      Jumps — missing source_check_ids in raw output: drop → counted as unverified.
      Jumps — source_check_ids present but id not found / wrong count: drop → bad_source.
      Simples — only dropped when simple_checks field is present AND the simple has no
                valid=true backing entry (from, to) in that field.
                If simple_checks is absent, simples are trusted.

    Returns (filtered_candidates, stats) where stats has:
      dropped_unverified  — jumps with no valid scan backing
      dropped_bad_source  — jumps whose source ids are present but invalid/mismatched
      dropped_simple      — simples dropped via the simple_checks gate
    """
    stats: dict[str, int] = {
        "dropped_unverified": 0,
        "dropped_bad_source": 0,
        "dropped_simple":     0,
    }

    scan          = dbg.get("scan") or []
    simple_checks = dbg.get("simple_checks")
    raw_finals    = dbg.get("llm_final_raw")

    # ── Collect valid scan ids (jump_checks + continuation_checks, valid=true) ─
    valid_scan_ids: set[str] = set()
    for entry in scan:
        if not isinstance(entry, dict):
            continue
        for chk in entry.get("jump_checks", []):
            if isinstance(chk, dict) and chk.get("valid") is True:
                cid = chk.get("id")
                if isinstance(cid, str) and cid:
                    valid_scan_ids.add(cid)
        for chk in entry.get("continuation_checks", []):
            if isinstance(chk, dict) and chk.get("valid") is True:
                cid = chk.get("id")
                if isinstance(cid, str) and cid:
                    valid_scan_ids.add(cid)

    # ── Build path-key → raw jump move map (pre-normalisation) ───────────────
    raw_jump_map: dict[tuple, dict] = {}
    if isinstance(raw_finals, list):
        for raw_move in raw_finals:
            if not isinstance(raw_move, dict):
                continue
            if str(raw_move.get("type", "")).strip().lower() != "jump":
                continue
            path = raw_move.get("path")
            if not path:
                continue
            try:
                pkey = tuple((int(sq[0]), int(sq[1])) for sq in path)
                raw_jump_map[pkey] = raw_move
            except (TypeError, ValueError, IndexError):
                continue

    # ── Build set of jumps whose source_check_ids are fully valid ────────────
    valid_jump_pkeys: set[tuple] = set()
    if n_valid_scan_jumps > 0:
        for pkey, raw_move in raw_jump_map.items():
            src_ids  = raw_move.get("source_check_ids")
            captured = raw_move.get("captured", [])
            if (isinstance(src_ids, list) and src_ids
                    and len(src_ids) == len(captured)
                    and all(isinstance(sid, str) and sid in valid_scan_ids
                            for sid in src_ids)):
                valid_jump_pkeys.add(pkey)

    # ── Build valid simple (from, to) pairs when simple_checks is present ────
    valid_simple_pairs: Optional[set[tuple]] = None
    if isinstance(simple_checks, list):
        valid_simple_pairs = set()
        for chk in simple_checks:
            if not isinstance(chk, dict) or chk.get("valid") is not True:
                continue
            frm = chk.get("from")
            to  = chk.get("to")
            if (isinstance(frm, (list, tuple)) and len(frm) == 2
                    and isinstance(to, (list, tuple)) and len(to) == 2):
                valid_simple_pairs.add(
                    ((int(frm[0]), int(frm[1])), (int(to[0]), int(to[1])))
                )

    # ── Filter ────────────────────────────────────────────────────────────────
    filtered: list[dict[str, Any]] = []
    for cand in candidates:
        mtype = cand.get("type", "simple")
        path  = cand.get("path", [])

        if mtype == "jump":
            try:
                pkey = tuple((int(sq[0]), int(sq[1])) for sq in path)
            except (TypeError, ValueError, IndexError):
                stats["dropped_bad_source"] += 1
                continue

            if n_valid_scan_jumps == 0:
                # Scan gate: zero valid scan entries → every jump is unverified
                stats["dropped_unverified"] += 1
            elif pkey in valid_jump_pkeys:
                filtered.append(cand)
            else:
                # Source ids present but invalid, or raw entry missing → bad_source
                raw_m   = raw_jump_map.get(pkey)
                src_ids = raw_m.get("source_check_ids") if raw_m else None
                if not isinstance(src_ids, list) or not src_ids:
                    stats["dropped_unverified"] += 1
                else:
                    stats["dropped_bad_source"] += 1

        else:  # simple
            if valid_simple_pairs is not None:
                # simple_checks was provided — apply the gate
                if len(path) >= 2:
                    try:
                        frm_key = (int(path[0][0]), int(path[0][1]))
                        to_key  = (int(path[1][0]), int(path[1][1]))
                        if (frm_key, to_key) in valid_simple_pairs:
                            filtered.append(cand)
                        else:
                            stats["dropped_simple"] += 1
                    except (IndexError, TypeError, ValueError):
                        stats["dropped_simple"] += 1
                else:
                    stats["dropped_simple"] += 1
            else:
                # No simple_checks — trust the simple move
                filtered.append(cand)

    return filtered, stats


# ── Simple projection (no-capture branch completeness fix) ────────────────────

def _project_missing_simples(
    candidates: list[dict[str, Any]],
    simple_checks: Any,
    n_valid_scan_jumps: int,
    capture_est: Any,
) -> tuple[list[dict[str, Any]], int]:
    """
    Projects missing simple moves from the LLM's own valid=true simple_checks.

    Guard conditions — projection is skipped when:
      • n_valid_scan_jumps > 0  (capture branch active by scan evidence)
      • capture_est is True     (LLM's own estimate says capture is available)
      • any type='jump' present in candidates (already in capture branch)
      • simple_checks is absent, empty, or not a list

    For each valid=true simple_check whose (from, to) pair is not already
    represented in candidates, creates and appends:
      {"type":"simple","path":[from,to],"captured":[],"source_check_id":<id>}

    source_check_id is taken from the LLM's own check entry id field.
    If the id field is absent or not a string, a stable fallback id is generated:
      "PROJ_{fr}_{fc}_{tr}_{tc}"

    Never calls get_all_legal_moves(), never reads board geometry independently,
    never infers a move that is not explicitly backed by a valid=true simple_check.

    Returns (augmented_candidates, n_added).
    """
    if n_valid_scan_jumps > 0 or capture_est is True:
        return candidates, 0
    if any(c.get("type") == "jump" for c in candidates):
        return candidates, 0
    if not isinstance(simple_checks, list) or not simple_checks:
        return candidates, 0

    existing: set[tuple] = set()
    for cand in candidates:
        if cand.get("type") != "simple":
            continue
        path = cand.get("path", [])
        if len(path) >= 2:
            try:
                existing.add(
                    ((int(path[0][0]), int(path[0][1])),
                     (int(path[1][0]), int(path[1][1])))
                )
            except (IndexError, TypeError, ValueError):
                continue

    projected = list(candidates)
    n_added   = 0

    for chk in simple_checks:
        if not isinstance(chk, dict) or chk.get("valid") is not True:
            continue
        frm = chk.get("from")
        to  = chk.get("to")
        if not (isinstance(frm, (list, tuple)) and len(frm) == 2
                and isinstance(to, (list, tuple)) and len(to) == 2):
            continue
        try:
            fr, fc = int(frm[0]), int(frm[1])
            tr, tc = int(to[0]),  int(to[1])
        except (TypeError, ValueError):
            continue
        if not (0 <= fr <= 7 and 0 <= fc <= 7 and 0 <= tr <= 7 and 0 <= tc <= 7):
            continue
        pair = ((fr, fc), (tr, tc))
        if pair in existing:
            continue
        check_id = chk.get("id")
        if not isinstance(check_id, str) or not check_id:
            check_id = f"PROJ_{fr}_{fc}_{tr}_{tc}"
        projected.append({
            "type":            "simple",
            "path":            [[fr, fc], [tr, tc]],
            "captured":        [],
            "source_check_id": check_id,
        })
        existing.add(pair)
        n_added += 1

    return projected, n_added


# ── Rate-limit header inspector ────────────────────────────────────────────────

_RL_HEADERS = [
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "retry-after",
]


def _log_rate_limit_headers(exc: Exception, attempt: int) -> None:
    """
    Inspect a caught API exception for HTTP rate-limit headers and print them.

    The openai SDK (used by langchain_openai) attaches the raw response to the
    exception in different places depending on the SDK version:
      exc.response          — httpx.Response  (openai >= 1.x)
      exc.http_response     — alternative attribute name
      exc.headers           — direct headers dict (older SDK or wrapped errors)
      exc.args[0].response  — sometimes nested inside the first argument

    Headers of interest:
      x-ratelimit-limit           — max requests per window
      x-ratelimit-remaining       — requests left in current window
      x-ratelimit-reset           — UTC timestamp when window resets (or seconds)
      x-ratelimit-limit-tokens    — token quota per window
      x-ratelimit-remaining-tokens— tokens left in current window
      retry-after                 — seconds to wait before retrying

    Never raises — all paths are guarded with try/except.
    """
    prefix = f"[board_proposal][rl-headers][attempt={attempt+1}]"

    # ── Locate the raw headers dict ───────────────────────────────────────────
    headers: Any = None
    source: str  = "unknown"

    candidates = [
        # (attribute_path_description, getter)
        ("exc.response.headers",          lambda e: e.response.headers),
        ("exc.http_response.headers",     lambda e: e.http_response.headers),
        ("exc.headers",                   lambda e: e.headers),
        ("exc.args[0].response.headers",  lambda e: e.args[0].response.headers),
        ("exc.args[0].headers",           lambda e: e.args[0].headers),
    ]

    for desc, getter in candidates:
        try:
            h = getter(exc)
            if h is not None:
                headers = h
                source  = desc
                break
        except Exception:
            continue

    if headers is None:
        print(
            f"{prefix} No HTTP response headers accessible on "
            f"{type(exc).__name__} — cannot determine limit type. "
            f"Tried: {', '.join(d for d, _ in candidates)}"
        )
        return

    print(f"{prefix} Headers found via '{source}' on {type(exc).__name__}:")
    for h in _RL_HEADERS:
        # Headers objects from httpx/requests/urllib3 all support case-insensitive .get()
        try:
            val = headers.get(h)
        except Exception:
            val = None
        status = str(val) if val is not None else "unavailable"
        print(f"{prefix}   {h}: {status}")
    # Full dump — catches any non-standard / GitHub-specific header names
    try:
        all_keys = list(headers.keys()) if hasattr(headers, "keys") else list(headers)
        print(f"{prefix} All response headers ({len(all_keys)} total):")
        for k in all_keys:
            try:
                print(f"{prefix}   {k}: {headers.get(k)}")
            except Exception:
                print(f"{prefix}   {k}: <error reading value>")
    except Exception as dump_exc:
        print(f"{prefix} Could not enumerate all headers: {dump_exc}")


# ── Main node ──────────────────────────────────────────────────────────────────


def board_proposal_agent(state: CheckersState) -> dict[str, Any]:
    """
    LLM-based board-grounded move proposal node.

    Reads:   state.board, state.current_player,
             state.strategic_context (game_phase + score_state only).
    Writes:  board_proposal_moves, board_proposal_raw, board_proposal_diagnostics.
    """
    player_label = "RED" if state.current_player == RED else "BLACK"

    system, user = build_board_proposal_prompt(
        board=state.board,
        current_player=state.current_player,
        strategic_context=state.strategic_context,
    )

    _base_url = _provider_base_url()
    diagnostics: dict[str, Any] = {
        "provider":            PROPOSAL_PROVIDER,
        "model":               PROPOSAL_MODEL,
        "temperature":         PROPOSAL_TEMPERATURE,
        "base_url":            _base_url,
        "player":              player_label,
        "turn_number":         state.turn_number,
        "api_call_succeeded":  False,
        "api_attempts":        0,
        "parse_succeeded":     False,
        "n_normalized":        0,
        "n_simple_proposals":  0,
        "n_jump_proposals":    0,
        "fallback_reason":     None,
        "raw_response_length": 0,
        "llm_capture_estimate": None,
        "llm_side_to_move":    None,
        "n_simple_unbacked":   0,
        "n_unlinked_jumps":         0,
        "n_bad_source_jumps":       0,
        "n_unlinked_simples":       0,
        "n_bad_source_simples":     0,
        "contradiction_retry_count":           0,
        "contradiction_reasons":              [],
        "post_retry_still_contradictory":     False,
        "dropped_unverified_after_retry_count": 0,
        "dropped_bad_source_after_retry_count": 0,
        "safe_rejection_count":               0,
        "n_missed_valid_simples":                    0,
        "no_capture_empty_output":                   False,
        "simple_geometry_targets_count":             0,
        "missing_simple_checks_for_geometry_count":  0,
        "final_json_marker_found":                   False,
        "draft_reasoning_present":                   False,
        "json_extraction_used":                      False,
        "parse_fallback_used":                       False,
        "valid_simple_checks_count":                 0,
        "missing_final_moves_from_valid_simple_checks_count": 0,
        "final_simple_completeness_rate":            1.0,
        "projected_missing_simple_count":            0,
        "simple_projection_applied":                 False,
    }
    # ── Trace header — always printed so every probe run is self-documenting ──
    _bu_label = f"  base_url={_base_url}" if _base_url else ""
    print(
        f"[board_proposal] Proposal provider: {PROPOSAL_PROVIDER}  "
        f"model: {PROPOSAL_MODEL}  temperature: {PROPOSAL_TEMPERATURE}"
        + _bu_label
    )

    raw:        Optional[str] = None
    last_error: Optional[str] = None

    # Fixed retry delays (seconds): attempt 1 fail→20s, attempt 2 fail→30s,
    # attempt 3+ fail→40s.  These are intentionally longer than the old
    # exponential back-off to be kinder to per-minute rate limits.
    # A daily quota error (x-ratelimit-type: UserByModelByDay) cannot be solved
    # by waiting — it will still be logged and all attempts will fail.
    _MAX_ATTEMPTS  = 3
    _RETRY_DELAYS  = [20, 30, 40]   # index = attempt number (0-based)

    for attempt in range(_MAX_ATTEMPTS):
        diagnostics["api_attempts"] = attempt + 1
        try:
            raw = call_board_proposal_llm(system, user)
            diagnostics["api_call_succeeded"] = True
            break
        except Exception as exc:  # noqa: BLE001 — catches SDK-specific errors (e.g. openai.RateLimitError)
            last_error = str(exc)[:300]
            logger.warning(
                "[board_proposal] API attempt %d/%d failed (%s): %s",
                attempt + 1, _MAX_ATTEMPTS, type(exc).__name__, last_error,
            )
            _log_rate_limit_headers(exc, attempt)
            if attempt < _MAX_ATTEMPTS - 1:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                logger.info(
                    "[board_proposal] waiting %ds before retry %d/%d",
                    delay, attempt + 2, _MAX_ATTEMPTS,
                )
                print(
                    f"[board_proposal] waiting {delay}s before retry "
                    f"{attempt + 2}/{_MAX_ATTEMPTS}"
                )
                time.sleep(delay)

    if raw is None:
        diagnostics["fallback_reason"] = last_error or "api_failed_all_attempts"
        logger.error(
            "[board_proposal] All API attempts failed — returning [] player=%s turn=%d",
            player_label, state.turn_number,
        )
        return {
            "board_proposal_raw":         "",
            "board_proposal_moves":       [],
            "board_proposal_diagnostics": diagnostics,
            "last_completed_node":        "board_proposal_agent",
        }

    diagnostics["raw_response_length"] = len(raw)

    _json_text_d, _marker_found_d, _draft_d = _extract_final_json_text(raw)
    diagnostics["final_json_marker_found"] = _marker_found_d
    diagnostics["draft_reasoning_present"] = _draft_d
    diagnostics["json_extraction_used"]    = _marker_found_d
    diagnostics["parse_fallback_used"]     = not _marker_found_d

    _geom_pairs = _simple_geometry_pairs(state.board, state.current_player)
    diagnostics["simple_geometry_targets_count"] = len(_geom_pairs)

    dbg = _extract_debug_fields(raw)
    diagnostics["llm_capture_estimate"] = dbg.get("capture_available_estimate")
    diagnostics["llm_side_to_move"]     = dbg.get("side_to_move")
    diagnostics["llm_scan"]             = dbg.get("scan")
    diagnostics["llm_simple_checks"]    = dbg.get("simple_checks")
    diagnostics["llm_final_raw"]        = dbg.get("llm_final_raw")

    candidates = parse_proposal_output(raw)
    diagnostics["parse_succeeded"]    = True
    diagnostics["n_normalized"]       = len(candidates)
    n_jump   = sum(1 for m in candidates if m.get("type") == "jump")
    n_simple = len(candidates) - n_jump
    diagnostics["n_simple_proposals"] = n_simple
    diagnostics["n_jump_proposals"]   = n_jump

    scan      = dbg.get("scan") or []
    n_scan_e  = len(scan)
    n_valid_s = sum(
        1 for entry in scan if isinstance(entry, dict)
        for chk in entry.get("jump_checks", [])
        if isinstance(chk, dict) and chk.get("valid") is True
    )

    simple_checks_raw          = dbg.get("simple_checks") or []
    n_simple_unbacked          = _count_simple_unbacked(candidates, simple_checks_raw)
    diagnostics["n_simple_unbacked"] = n_simple_unbacked
    n_missed_valid_simples     = _count_missed_valid_simples(candidates, simple_checks_raw)
    diagnostics["n_missed_valid_simples"] = n_missed_valid_simples
    diagnostics["missing_simple_checks_for_geometry_count"] = (
        _count_missing_simple_checks_for_geometry(
            state.board, state.current_player, dbg.get("simple_checks")
        )
    )
    _n_valid_sc = sum(
        1 for chk in (simple_checks_raw if isinstance(simple_checks_raw, list) else [])
        if isinstance(chk, dict) and chk.get("valid") is True
    )
    diagnostics["valid_simple_checks_count"] = _n_valid_sc
    diagnostics["missing_final_moves_from_valid_simple_checks_count"] = n_missed_valid_simples
    diagnostics["final_simple_completeness_rate"] = (
        round((_n_valid_sc - n_missed_valid_simples) / _n_valid_sc, 3)
        if _n_valid_sc > 0 else 1.0
    )

    grounding      = _count_grounding_failures(
        dbg.get("llm_final_raw"),
        dbg.get("scan"),
        dbg.get("simple_checks"),
    )
    n_unlinked_j   = grounding["unlinked_jump_count"]
    n_bad_source_j = grounding["bad_source_jump_count"]
    n_unlinked_s   = grounding["unlinked_simple_count"]
    n_bad_source_s = grounding["bad_source_simple_count"]
    diagnostics["n_unlinked_jumps"]     = n_unlinked_j
    diagnostics["n_bad_source_jumps"]   = n_bad_source_j
    diagnostics["n_unlinked_simples"]   = n_unlinked_s
    diagnostics["n_bad_source_simples"] = n_bad_source_s

    # ── Contradiction check → exactly one retry if LLM contradicts its own scan ─
    contradiction_reasons = _detect_contradictions(dbg, n_valid_s, grounding, n_jump)
    diagnostics["contradiction_reasons"] = contradiction_reasons

    if contradiction_reasons:
        diagnostics["contradiction_retry_count"] = 1
        _reasons_log = "; ".join(r[:80] for r in contradiction_reasons)
        print(
            f"[board_proposal] contradiction_retry: "
            f"{len(contradiction_reasons)} issue(s): {_reasons_log}"
        )
        retry_user = _build_retry_user_prompt(user, contradiction_reasons)
        retry_raw: Optional[str] = None
        try:
            retry_raw = call_board_proposal_llm(system, retry_user)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[board_proposal] retry API call failed "
                f"({type(exc).__name__}): {str(exc)[:200]}"
            )

        if retry_raw is not None:
            # Adopt the retry response — recompute all metrics from it
            raw      = retry_raw
            _json_text_d, _marker_found_d, _draft_d = _extract_final_json_text(retry_raw)
            diagnostics["final_json_marker_found"] = _marker_found_d
            diagnostics["draft_reasoning_present"] = _draft_d
            diagnostics["json_extraction_used"]    = _marker_found_d
            diagnostics["parse_fallback_used"]     = not _marker_found_d
            dbg      = _extract_debug_fields(retry_raw)
            diagnostics["llm_capture_estimate"] = dbg.get("capture_available_estimate")
            diagnostics["llm_side_to_move"]     = dbg.get("side_to_move")
            diagnostics["llm_scan"]             = dbg.get("scan")
            diagnostics["llm_simple_checks"]    = dbg.get("simple_checks")
            diagnostics["llm_final_raw"]        = dbg.get("llm_final_raw")

            candidates    = parse_proposal_output(retry_raw)
            n_jump        = sum(1 for m in candidates if m.get("type") == "jump")
            n_simple      = len(candidates) - n_jump
            diagnostics["n_normalized"]        = len(candidates)
            diagnostics["n_simple_proposals"]  = n_simple
            diagnostics["n_jump_proposals"]    = n_jump
            diagnostics["raw_response_length"] = len(raw)

            scan          = dbg.get("scan") or []
            n_scan_e      = len(scan)
            n_valid_s     = sum(
                1 for entry in scan if isinstance(entry, dict)
                for chk in entry.get("jump_checks", [])
                if isinstance(chk, dict) and chk.get("valid") is True
            )

            simple_checks_raw = dbg.get("simple_checks") or []
            n_simple_unbacked = _count_simple_unbacked(candidates, simple_checks_raw)
            diagnostics["n_simple_unbacked"] = n_simple_unbacked
            n_missed_valid_simples = _count_missed_valid_simples(candidates, simple_checks_raw)
            diagnostics["n_missed_valid_simples"] = n_missed_valid_simples
            diagnostics["missing_simple_checks_for_geometry_count"] = (
                _count_missing_simple_checks_for_geometry(
                    state.board, state.current_player, dbg.get("simple_checks")
                )
            )
            _n_valid_sc = sum(
                1 for chk in (simple_checks_raw if isinstance(simple_checks_raw, list) else [])
                if isinstance(chk, dict) and chk.get("valid") is True
            )
            diagnostics["valid_simple_checks_count"] = _n_valid_sc
            diagnostics["missing_final_moves_from_valid_simple_checks_count"] = n_missed_valid_simples
            diagnostics["final_simple_completeness_rate"] = (
                round((_n_valid_sc - n_missed_valid_simples) / _n_valid_sc, 3)
                if _n_valid_sc > 0 else 1.0
            )

            grounding      = _count_grounding_failures(
                dbg.get("llm_final_raw"),
                dbg.get("scan"),
                dbg.get("simple_checks"),
            )
            n_unlinked_j   = grounding["unlinked_jump_count"]
            n_bad_source_j = grounding["bad_source_jump_count"]
            n_unlinked_s   = grounding["unlinked_simple_count"]
            n_bad_source_s = grounding["bad_source_simple_count"]
            diagnostics["n_unlinked_jumps"]     = n_unlinked_j
            diagnostics["n_bad_source_jumps"]   = n_bad_source_j
            diagnostics["n_unlinked_simples"]   = n_unlinked_s
            diagnostics["n_bad_source_simples"] = n_bad_source_s

    # ── No-capture empty output diagnostic ──────────────────────────────────────
    # Flags when LLM said no captures, had valid simple_checks, but produced nothing.
    _capture_est_diag = dbg.get("capture_available_estimate")
    _n_valid_sc_diag  = sum(
        1 for chk in (simple_checks_raw if isinstance(simple_checks_raw, list) else [])
        if isinstance(chk, dict) and chk.get("valid") is True
    )
    diagnostics["no_capture_empty_output"] = (
        _capture_est_diag is False and _n_valid_sc_diag > 0 and len(candidates) == 0
    )

    # ── Post-retry safety filter ───────────────────────────────────────────────
    # Only runs when a contradiction retry was attempted (retry_count==1).
    # Re-checks the current state (original if retry failed, retry output if succeeded).
    # Drops proposals that still violate scan-gate or source-check-id invariants.
    if diagnostics["contradiction_retry_count"] == 1:
        post_reasons = _detect_contradictions(dbg, n_valid_s, grounding, n_jump)
        still_bad    = len(post_reasons) > 0
        diagnostics["post_retry_still_contradictory"] = still_bad
        if still_bad:
            filtered_cands, f_stats = _filter_contradictory_proposals(
                candidates, dbg, n_valid_s
            )
            n_drop_u   = f_stats["dropped_unverified"]
            n_drop_b   = f_stats["dropped_bad_source"]
            n_drop_s   = f_stats["dropped_simple"]
            safe_count = n_drop_u + n_drop_b + n_drop_s
            diagnostics["dropped_unverified_after_retry_count"] = n_drop_u
            diagnostics["dropped_bad_source_after_retry_count"] = n_drop_b
            diagnostics["safe_rejection_count"]                 = safe_count
            if safe_count > 0:
                print(
                    f"[board_proposal] safe_rejection: {safe_count} proposal(s) dropped "
                    f"(unverified={n_drop_u} bad_source={n_drop_b} simple={n_drop_s})"
                )
                candidates = filtered_cands
                n_jump     = sum(1 for m in candidates if m.get("type") == "jump")
                n_simple   = len(candidates) - n_jump
                diagnostics["n_normalized"]       = len(candidates)
                diagnostics["n_simple_proposals"] = n_simple
                diagnostics["n_jump_proposals"]   = n_jump

    # ── Simple projection (no-capture branch completeness fix) ──────────────────
    # Projects valid=true simple_checks that the LLM omitted from final output.
    # Only runs in the no-capture branch; never uses engine moves.
    _capture_est_proj = dbg.get("capture_available_estimate")
    candidates, _n_projected = _project_missing_simples(
        candidates, simple_checks_raw, n_valid_s, _capture_est_proj
    )
    diagnostics["projected_missing_simple_count"] = _n_projected
    diagnostics["simple_projection_applied"]       = _n_projected > 0
    if _n_projected > 0:
        n_simple = len(candidates) - n_jump
        diagnostics["n_normalized"]       = len(candidates)
        diagnostics["n_simple_proposals"] = n_simple
        n_missed_post = _count_missed_valid_simples(candidates, simple_checks_raw)
        diagnostics["missing_final_moves_from_valid_simple_checks_count"] = n_missed_post
        diagnostics["final_simple_completeness_rate"] = (
            round((_n_valid_sc - n_missed_post) / _n_valid_sc, 3)
            if _n_valid_sc > 0 else 1.0
        )
        print(f"[board_proposal] simple_projection: added {_n_projected} missing simple(s)")

    print(
        f"[board_proposal] player={player_label} turn={state.turn_number} "
        f"candidates={len(candidates)} (simple={n_simple}, jump={n_jump}) "
        f"scan_entries={n_scan_e} valid_scan_jumps={n_valid_s} "
        f"simple_unbacked={n_simple_unbacked} "
        f"unlinked_jump={n_unlinked_j} bad_source_jump={n_bad_source_j} "
        f"unlinked_simple={n_unlinked_s} "
        f"contradiction_retries={diagnostics['contradiction_retry_count']} "
        f"safe_rejections={diagnostics['safe_rejection_count']} "
        f"projected_simples={_n_projected} "
        f"llm_capture_estimate={dbg.get('capture_available_estimate')} "
        f"llm_side_to_move={dbg.get('side_to_move')} "
        f"provider={PROPOSAL_PROVIDER} model={PROPOSAL_MODEL}"
    )
    for i, m in enumerate(candidates):
        print(f"  [{i}] type={m['type']} path={m['path']} captured={m['captured']}")

    return {
        "board_proposal_raw":         raw,
        "board_proposal_moves":       candidates,
        "board_proposal_diagnostics": diagnostics,
        "last_completed_node":        "board_proposal_agent",
    }
