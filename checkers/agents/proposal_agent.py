# agents/proposal_agent.py
#
# CHANGES FROM PREVIOUS VERSION:
#   - Groq API only (Ollama removed)
#   - compute_move_facts() called per legal move BEFORE sending to Groq
#     so the LLM sees 7 essential facts alongside each move
#   - System prompt has a clear priority-ordered selection algorithm
#     so Groq picks the best 3-5 candidates, not random ones
#   - Strategic context formatter extracts only what proposal needs
#     (not a raw JSON dump)
#   - Strict count rules: N>=5 → 5, N==4 → 4, N==3 → 3, N<3 → all
#   - Retry logic with exponential backoff (5s, 10s, 20s)
#   - Output format unchanged: {"selected_indices": [...]} — format_checker compatible

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from checkers.state.state import CheckersState
from checkers.engine.board import (
    BOARD_SIZE,
    EMPTY,
    RED,
    BLACK,
    RED_KING,
    BLACK_KING,
    is_own_piece,
)
from checkers.engine.rules import get_all_legal_moves
from checkers.engine.move_facts import compute_move_facts

_PIECE_NAMES = {
    EMPTY: "EMPTY",
    RED: "RED",
    BLACK: "BLACK",
    RED_KING: "RED_KING",
    BLACK_KING: "BLACK_KING",
}

# ── Groq settings ─────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_PROPOSAL_MODEL = os.environ.get("GROQ_PROPOSAL_MODEL", "llama-3.1-8b-instant")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
PROPOSAL_TEMPERATURE = float(os.environ.get("PROPOSAL_TEMPERATURE", "0.15"))


# ── Utility helpers ───────────────────────────────────────────────────────────

def _current_player_label(current_player: int) -> str:
    return "RED" if current_player == RED else "BLACK"


def _king_counts(board: list[list[int]], current_player: int) -> dict[str, int]:
    kings = 0
    regular = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            p = board[row][col]
            if is_own_piece(p, current_player):
                if p == RED_KING or p == BLACK_KING:
                    kings += 1
                else:
                    regular += 1
    return {"regular": regular, "kings": kings}


def _format_strategic_context_for_proposal(ctx: dict[str, Any] | None) -> str:
    """
    Extracts only what the proposal agent needs from strategic_context.
    Does NOT dump the full raw JSON — only fields relevant to shortlisting.
    """
    if not ctx:
        return "(no strategic context yet)"

    lines: list[str] = []

    phase = ctx.get("game_phase", "UNKNOWN")
    score = ctx.get("winning_score", 0)
    score_state = ctx.get("score_state", "UNKNOWN")
    mat = ctx.get("material_advantage", 0)
    king = ctx.get("king_advantage", 0)

    lines.append(
        f"game_phase: {phase} | score_state: {score_state} | winning_score: {score:+d} | "
        f"material_advantage: {mat:+d} | king_advantage: {king:+d}"
    )

    our_vuln = ctx.get("our_vulnerable_pieces", 0)
    opp_vuln = ctx.get("opp_vulnerable_pieces", 0)
    our_prom = ctx.get("our_promotion_threats", 0)
    opp_prom = ctx.get("opp_promotion_threats", 0)
    lines.append(
        f"our_vulnerable_pieces: {our_vuln} | opp_vulnerable_pieces: {opp_vuln} | "
        f"our_promotion_threats: {our_prom} | opp_promotion_threats: {opp_prom}"
    )

    patterns = ctx.get("active_patterns", [])
    if patterns:
        lines.append(f"active_patterns: {', '.join(patterns)}")

    priorities = ctx.get("strategic_priorities", [])
    if priorities:
        lines.append(f"strategic_priorities (apply in order): {', '.join(priorities)}")

    return "\n".join(lines)


def _proposal_sort_key(
    facts: dict,
    score_state: str,
    game_phase: str,
    strategic_priorities: list[str],
) -> tuple:
    """
    Symbolic pre-sort key for legal moves before LLM sees them.
    Lower tuple value = presented earlier (more likely to be shortlisted).

    Sort priority order:
      1. Safety first  — our_pieces_threatened_after (ascending)
      2. Unsafe flag   — unsafe_simple_move=True goes last
      3. Captures      — captures_count (descending)
      4. Promotion     — results_in_king or near_promotion
      5. Score-state   — winning_conversion_score (winning) or counterplay_score (losing)
      6. Role coverage — quiet_move_role ranking
      7. Tiebreakers   — center_control, isolation penalty
    """
    threatened_after = facts.get("our_pieces_threatened_after", 0)
    unsafe_simple = facts.get(
        "unsafe_simple_move",
        facts.get("move_type") == "simple" and threatened_after > 0
    )
    unsafe = 1 if unsafe_simple else 0
    captures         = facts.get("captures_count", 0)
    is_promotion     = 1 if (facts.get("results_in_king", False) or facts.get("near_promotion", False)) else 0
    conversion_score = facts.get("winning_conversion_score", 0)
    counterplay      = facts.get("counterplay_score", 0)
    center           = 1 if facts.get("center_control", False) else 0
    isolated         = 1 if facts.get("leaves_piece_isolated", False) else 0

    # Score-state-aware primary ranking signal
    losing_states  = ("CLEARLY_LOSING", "SLIGHTLY_LOSING")
    winning_states = ("CLEARLY_WINNING", "SLIGHTLY_WINNING")

    if score_state in winning_states:
        state_score = -conversion_score   # negate: higher is better, but tuple sorts ascending
    elif score_state in losing_states:
        state_score = -counterplay
    else:
        state_score = -max(conversion_score, counterplay)

    # Quiet move role ranking (lower = presented earlier)
    _ROLE_RANK = {
        "TACTICAL":                 0,
        "PROMOTION_PUSH":           1,
        "KING_ACTIVATION":          2,
        "COUNTERPLAY":              3,
        "CONVERSION":               4,
        "DEFENSIVE_STABILIZATION":  5,
        "MOBILITY_IMPROVEMENT":     6,
        "QUIET_DEFAULT":            7,
    }
    role = facts.get("quiet_move_role", "QUIET_DEFAULT")

    # Adjust role priority based on game phase and priorities
    if game_phase == "ENDGAME" and role == "KING_ACTIVATION":
        role_rank = 1   # boost king activation in endgame
    elif "SEEK_COUNTERPLAY" in strategic_priorities and role == "COUNTERPLAY":
        role_rank = 2
    elif "CONVERT_ADVANTAGE" in strategic_priorities and role == "CONVERSION":
        role_rank = 2
    else:
        role_rank = _ROLE_RANK.get(role, 7)

    return (
        threatened_after,          # 1st: safety (ascending — 0 best)
        unsafe,                    # 2nd: unsafe flag (0=safe first)
        -captures,                 # 3rd: captures (descending — more captures first)
        -is_promotion,             # 4th: promotion (1=yes first)
        state_score,               # 5th: score-state-aware signal
        role_rank,                 # 6th: quiet_move_role
        -center,                   # 7th: center control bonus
        isolated,                  # 8th: isolation penalty (0=not isolated first)
    )

def _build_legal_moves_with_facts(
    board: list[list[int]],
    current_player: int,
    strategic_context: dict | None = None,
) -> tuple[int, str, list[tuple[dict, dict]]]:
    """
    Gets all legal moves from the engine, computes facts per move,
    pre-sorts by symbolic quality, and returns:
      (n_moves, formatted_block_string, sorted_moves_with_facts)

    Pre-sort ensures the LLM sees the strongest candidates first,
    improving recall without removing any moves.
    """
    moves = get_all_legal_moves(board, current_player)

    ctx = strategic_context or {}
    score_state          = ctx.get("score_state", "EQUAL")
    game_phase           = ctx.get("game_phase", "MIDGAME")
    strategic_priorities = ctx.get("strategic_priorities", [])

    # Compute facts for every move first
    moves_with_facts: list[tuple[dict, dict]] = []
    for m in moves:
        try:
            facts = compute_move_facts(board, m, current_player)
        except Exception:
            facts = {}
        moves_with_facts.append((m, facts))

    # Pre-sort by symbolic quality key
    moves_with_facts.sort(
        key=lambda pair: _proposal_sort_key(
            pair[1], score_state, game_phase, strategic_priorities
        )
    )

    n = len(moves_with_facts)

    lines = [
        f"N = {n}  (legal moves indexed 0 .. {n - 1} when N > 0).",
        "",
        "LEGAL_MOVES (path and captured are engine ground truth — never modify them):",
        "",
    ]

    for i, (m, facts) in enumerate(moves_with_facts):
        essential = {
            "captures_count":              facts.get("captures_count", 0),
            "results_in_king":             facts.get("results_in_king", False),
            "near_promotion":              facts.get("near_promotion", False),
            "net_gain":                    facts.get("net_gain", 0),
            "our_pieces_threatened_after": facts.get("our_pieces_threatened_after", 0),
            "opponent_can_recapture":      facts.get("opponent_can_recapture", False),
            "unsafe_simple_move":          facts.get("unsafe_simple_move", False),
            "blocks_opponent_landing":     facts.get("blocks_opponent_landing", False),
            "center_control":              facts.get("center_control", False),
            "leaves_piece_isolated":       facts.get("leaves_piece_isolated", False),
            "mobility_reduction":          facts.get("mobility_reduction", 0),
            "creates_immediate_threat":    facts.get("creates_immediate_threat", False),
            "winning_conversion_score":    facts.get("winning_conversion_score", 0),
            "counterplay_score":           facts.get("counterplay_score", 0),
            "quiet_move_role":             facts.get("quiet_move_role", "QUIET_DEFAULT"),
        }

        payload = {
            "type":     m["type"],
            "path":     m["path"],
            "captured": m["captured"],
            "facts":    essential,
        }
        lines.append(f"  [{i}] {json.dumps(payload, ensure_ascii=False)}")

    return n, "\n".join(lines), moves_with_facts


def _apply_safety_net(
    selected_indices: list[int],
    moves_with_facts: list[tuple[dict, dict]],
    score_state: str,
    game_phase: str,
    strategic_priorities: list[str],
    active_patterns: list[str],
    n_moves: int,
) -> list[int]:
    """
    After LLM proposes selected_indices, inject missing important move classes
    using symbolic safety-net logic. Preserves LLM order, deduplicates,
    and enforces strict count rules.

    Protected move classes (checked in priority order):
      1. Best safe move  — lowest our_pieces_threatened_after
      2. Best promotion  — results_in_king or near_promotion
      3. Best conversion — winning_conversion_score (when winning)
      4. Best counterplay — counterplay_score (when losing)
      5. Best king activation — KING_ACTIVATION role (endgame)
      6. Best mobility reduction — when STAGNATION_LOOP_RISK or REDUCE_OPP_MOBILITY
    """
    if not moves_with_facts:
        return selected_indices

    # Determine target count from strict rules
    if n_moves >= 5:
        target = 5
    elif n_moves == 4:
        target = 4
    elif n_moves == 3:
        target = 3
    elif n_moves == 2:
        target = 2
    else:
        target = 1

    losing_states  = ("CLEARLY_LOSING", "SLIGHTLY_LOSING")
    winning_states = ("CLEARLY_WINNING", "SLIGHTLY_WINNING")

    def _best_index(condition_fn, score_fn) -> int | None:
        """Find index of best move matching condition, scored by score_fn."""
        best_idx   = None
        best_score = None
        for i, (m, f) in enumerate(moves_with_facts):
            if condition_fn(m, f):
                s = score_fn(f)
                if best_score is None or s > best_score:
                    best_score = s
                    best_idx   = i
        return best_idx

    # Build candidate injections — ordered by importance
    injections: list[int] = []

    # 1. Best safe move (our_pieces_threatened_after == 0)
    safe_idx = _best_index(
        lambda m, f: f.get("our_pieces_threatened_after", 99) == 0,
        lambda f: f.get("winning_conversion_score", 0) + f.get("counterplay_score", 0),
    )
    if safe_idx is not None:
        injections.append(safe_idx)

    # 2. Best promotion move
    promo_idx = _best_index(
        lambda m, f: f.get("results_in_king", False) or f.get("near_promotion", False),
        lambda f: (2 if f.get("results_in_king", False) else 1),
    )
    if promo_idx is not None:
        injections.append(promo_idx)

    # 3. Best conversion move (when winning)
    if score_state in winning_states:
        conv_idx = _best_index(
            lambda m, f: f.get("our_pieces_threatened_after", 99) == 0,
            lambda f: f.get("winning_conversion_score", 0),
        )
        if conv_idx is not None:
            injections.append(conv_idx)

    # 4. Best counterplay move (when losing)
    if score_state in losing_states:
        cp_idx = _best_index(
            lambda m, f: not f.get("unsafe_simple_move", False),
            lambda f: f.get("counterplay_score", 0),
        )
        if cp_idx is not None:
            injections.append(cp_idx)

    # 5. Best king activation (endgame or king-heavy positions)
    if game_phase == "ENDGAME" or "ACTIVATE_KINGS" in strategic_priorities:
        king_idx = _best_index(
            lambda m, f: (
                f.get("quiet_move_role", "") == "KING_ACTIVATION"
                and f.get("our_pieces_threatened_after", 99) == 0
            ),
            lambda f: f.get("winning_conversion_score", 0) + f.get("counterplay_score", 0),
        )
        if king_idx is not None:
            injections.append(king_idx)

    # 6. Best blocker move — prevents opponent capture landing
    blocker_idx = _best_index(
        lambda m, f: f.get("blocks_opponent_landing", False),
        lambda f: (
            1 if f.get("our_pieces_threatened_after", 99) == 0 else 0,
            -f.get("our_pieces_threatened_after", 99),
            f.get("winning_conversion_score", 0),
            f.get("counterplay_score", 0),
        ),
    )
    if blocker_idx is not None:
        injections.append(blocker_idx)
        
    # 6. Best mobility reduction
    # Include a mobility-reduction move not only in explicit stagnation cases,
    # but also whenever such a safe move exists and no stronger injected move
    # already covers that pressure type.
    mob_idx = _best_index(
        lambda m, f: (
            f.get("mobility_reduction", 0) > 0
            and f.get("our_pieces_threatened_after", 99) == 0
        ),
        lambda f: (
            f.get("mobility_reduction", 0),
            f.get("winning_conversion_score", 0),
            f.get("counterplay_score", 0),
        ),
    )
    if mob_idx is not None:
        injections.append(mob_idx)

    # Merge: critical injections first, then LLM selections, then remaining injections.
    # This ensures safety-net moves are never crowded out by LLM choices at trim time.
    seen:   set[int]  = set()
    merged: list[int] = []

    # Critical injections always go in first (blocker, safe move, promotion)
    critical = []
    if safe_idx is not None:
        critical.append(safe_idx)
    if promo_idx is not None:
        critical.append(promo_idx)
    if blocker_idx is not None:
        critical.append(blocker_idx)

    for idx in critical:
        if idx not in seen:
            seen.add(idx)
            merged.append(idx)

    # LLM selections next
    for idx in selected_indices:
        if 0 <= idx < n_moves and idx not in seen:
            seen.add(idx)
            merged.append(idx)

    # Remaining injections last
    for idx in injections:
        if idx not in seen:
            seen.add(idx)
            merged.append(idx)

    # Enforce target count
    if len(merged) > target:
        # Critical protections: safety + promotion + mobility reduction
        protected = set()

        for idx in injections:
            _, f = moves_with_facts[idx]

            # Always protect:
            if f.get("our_pieces_threatened_after", 99) == 0:
                protected.add(idx)

            if f.get("results_in_king", False) or f.get("near_promotion", False):
                protected.add(idx)

            if f.get("mobility_reduction", 0) > 0:
                protected.add(idx)

            if f.get("blocks_opponent_landing", False):
                protected.add(idx)

        # Fallback: if nothing protected, keep original behavior
        if not protected:
            protected = set(injections)

        kept: list[int] = []
        seen_keep: set[int] = set()

        for i in merged:
            if i in protected and i not in seen_keep:
                kept.append(i)
                seen_keep.add(i)

        for i in merged:
            if len(kept) >= target:
                break
            if i not in seen_keep:
                kept.append(i)
                seen_keep.add(i)

        merged = kept[:target]

    # Rebuild seen from current merged state before padding
    seen = set(merged)

    # If still under target, pad with best remaining moves
    if len(merged) < target:
        for i, (m, f) in enumerate(moves_with_facts):
            if len(merged) >= target:
                break
            if i not in seen:
                seen.add(i)
                merged.append(i)

    return merged[:target]


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a move proposer for American Checkers (8x8).

Your job: select the best candidate move indices from the LEGAL_MOVES list.
The ranker makes the final choice. Your job is to return a strong shortlist.

Rules:
- Never invent coordinates, paths, or captures.
- Every index must exist in LEGAL_MOVES (0..N-1).
- Never output duplicates.

SELECTION ORDER:

STEP 1 — SAFETY
Prefer moves with lower our_pieces_threatened_after.
Prefer opponent_can_recapture=false.
unsafe_simple_move=true should be ranked below safe simple moves.
Exclude clearly unsafe simple moves unless all moves are unsafe.

STEP 2 — CAPTURES
Always include jump moves when they exist.
Prefer higher captures_count.
Prefer jumps with better net_gain.

STEP 3 — PROMOTION
Always include results_in_king=true or near_promotion=true moves unless clearly unsafe.

STEP 4 — STRATEGIC PRIORITIES
Use strategic_priorities in order:
- CONTROL_CENTER → prefer center_control=true
- DEVELOP_PIECES → prefer leaves_piece_isolated=false
- PROMOTE → prefer near_promotion=true or results_in_king=true
- TRADE_WHEN_AHEAD / CONVERT_ADVANTAGE → prefer higher winning_conversion_score and mobility_reduction
- CREATE_THREATS / SEEK_COUNTERPLAY → prefer higher counterplay_score and creates_immediate_threat=true
- DEFEND / DEFEND_PIECES → prefer lower threat count and blocks_opponent_landing=true
- REDUCE_OPP_MOBILITY → prefer higher mobility_reduction
- ACTIVATE_KINGS → prefer quiet_move_role=KING_ACTIVATION
- INCREASE_MOBILITY → prefer quiet_move_role=MOBILITY_IMPROVEMENT
- otherwise use net_gain, counterplay_score, winning_conversion_score, then center_control

STEP 5 — QUIET POSITION COVERAGE
If no captures are available and several safe moves are available, avoid returning only passive duplicates.
Try to cover different quiet_move_role categories when possible:
PROMOTION_PUSH, KING_ACTIVATION, COUNTERPLAY, CONVERSION,
DEFENSIVE_STABILIZATION, MOBILITY_IMPROVEMENT.
Safety always dominates diversity.

COUNT RULES:
- N >= 5 → output exactly 5 indices
- N == 4 → output exactly 4 indices
- N == 3 → output exactly 3 indices
- N == 2 → output exactly 2 indices
- N == 1 → output [0]
- N == 0 → output []

Reply with ONLY this JSON object:
{"selected_indices": [<int>, ...]}
"""
# ── Prompt builder ────────────────────────────────────────────────────────────

def build_proposal_prompts(state: CheckersState) -> tuple[str, str, int, list[tuple[dict, dict]]]:
    """
    Returns (system_prompt, user_prompt, n_moves, moves_with_facts).
    moves_with_facts is the pre-sorted list used by _apply_safety_net.
    """
    board   = state.board
    current = state.current_player
    counts  = _king_counts(board, current)

    n_moves, legal_block, moves_with_facts = _build_legal_moves_with_facts(
        board, current, state.strategic_context
    )

    user_parts = [
        f"current_player: {_current_player_label(current)}",
        f"turn_number: {state.turn_number}",
        f"our pieces — regular: {counts['regular']}, kings: {counts['kings']}",
        "",
        legal_block,
        "",
        "strategic_context:",
        _format_strategic_context_for_proposal(state.strategic_context),
    ]

    required = n_moves if n_moves <= 5 else 5
    user_parts.extend([
        "",
        f"IMPORTANT: There are {n_moves} legal moves available (indices 0 to {n_moves - 1}).",
        f"You MUST select exactly {required} indices. Not {required - 1}, not {required + 1} — exactly {required}.",
        f'Output exactly {required} integers inside selected_indices.',
    ])

    if state.feedback:
        user_parts.extend([
            "",
            "Previous attempt failed — fix using this feedback:",
            state.feedback.strip(),
            "",
            'Reminder: reply with ONLY {"selected_indices": [int, ...]} '
            "using valid indices from the LEGAL_MOVES list above.",
        ])

    return _SYSTEM_PROMPT, "\n".join(user_parts), n_moves, moves_with_facts


# ── Groq API call ─────────────────────────────────────────────────────────────

def call_groq_proposal(system: str, user: str) -> str:
    """
    Calls llama-3.1-8b-instant via Groq REST API with native JSON mode.

    Raises:
        ValueError — API key missing, non-200 response, missing content
        OSError    — network-level failure
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add GROQ_API_KEY=your_key to your .env file."
        )

    payload: dict[str, Any] = {
        "model": GROQ_PROPOSAL_MODEL,
        "temperature": PROPOSAL_TEMPERATURE,
        "max_tokens": 64,                           # index list only — very short
        "response_format": {"type": "json_object"},  # native JSON mode
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        GROQ_API_URL,
        data=body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Accept":        "application/json",
            "User-Agent":    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",

        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        if e.code == 429:
            time.sleep(10)  # rate limit — wait before caller retries
        raise ValueError(f"Groq API HTTP {e.code}: {body_text[:300]}") from e

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(
            f"Unexpected Groq response structure: {str(data)[:300]}"
        ) from e

    if not isinstance(content, str):
        raise ValueError(f"Groq content is not a string: {type(content)}")

    return content


# ── Main node ─────────────────────────────────────────────────────────────────

def proposal_agent(state: CheckersState) -> dict:
    """
    Calls Groq to propose the best 3-5 candidate move indices.
    After LLM response, applies symbolic safety-net injection (Part B)
    and score-state-aware role coverage (Part C).
    Returns raw JSON string for format_checker to parse.
    Retries up to 3 times with exponential backoff on API failures.
    """
    system, user, n_moves, moves_with_facts = build_proposal_prompts(state)

    ctx                  = state.strategic_context or {}
    score_state          = ctx.get("score_state", "EQUAL")
    game_phase           = ctx.get("game_phase", "MIDGAME")
    strategic_priorities = ctx.get("strategic_priorities", [])
    active_patterns      = ctx.get("active_patterns", [])

    raw = None
    last_error_text = ""

    for attempt in range(3):
        try:
            raw = call_groq_proposal(system, user)
            break
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as e:
            last_error_text = str(e)

            if "tokens per day" in last_error_text.lower() or "tpd" in last_error_text.lower():
                print(f"[proposal_agent] daily quota exhausted: {e}")
                break

            wait = 2 ** attempt * 12   # 12s, 24s, 48s
            print(f"[proposal_agent] call failed (attempt {attempt + 1}): {e} — waiting {wait}s")
            time.sleep(wait)

    if raw is None:
        if n_moves >= 5:
            k = 5
        elif n_moves == 4:
            k = 4
        elif n_moves == 3:
            k = 3
        elif n_moves == 2:
            k = 2
        elif n_moves == 1:
            k = 1
        else:
            k = 0

        fallback_indices = list(range(k))
        final_raw = json.dumps({"selected_indices": fallback_indices})
        print(f"[proposal_agent] fallback selected indices: {fallback_indices}")

        return {
            "proposed_moves": final_raw,
            "last_completed_node": "proposal_agent",
        }

    # ── Parse LLM output ──────────────────────────────────────────────────────
    selected_indices: list[int] = []
    try:
        parsed = json.loads(raw.strip())
        raw_indices = parsed.get("selected_indices", [])
        # Validate: keep only integers within valid range
        selected_indices = [
            int(i) for i in raw_indices
            if isinstance(i, (int, float)) and 0 <= int(i) < n_moves
        ]
    except (json.JSONDecodeError, TypeError, ValueError):
        # format_checker will handle the error; pass through raw
        return {
            "proposed_moves": raw.strip(),
            "last_completed_node": "proposal_agent",
        }

    # ── Part B+C: symbolic safety-net + role coverage ─────────────────────────
    selected_indices = _apply_safety_net(
        selected_indices,
        moves_with_facts,
        score_state,
        game_phase,
        strategic_priorities,
        active_patterns,
        n_moves,
    )

    # Re-serialize for format_checker (same format as before)
    final_raw = json.dumps({"selected_indices": selected_indices})
    print(f"[proposal_agent] Raw output: '{raw.strip()}'")
    print(f"[proposal_agent] Selected indices after safety-net: {selected_indices}")

    return {
        "proposed_moves": final_raw,
        "last_completed_node": "proposal_agent",
    }