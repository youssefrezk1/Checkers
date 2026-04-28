# agents/proposal_agent.py
#
# Calls Groq (llama-3.1-8b-instant) to shortlist min(5, N) candidate move indices
# from the symbolic-pre-sorted legal move list.
#
# Pipeline responsibilities:
#   - Pre-LLM: sort by quality key, role-pin strategic representatives,
#     mm-pin minimax-best if outside window, annotate top-3 with MINIMAX_RANK.
#   - LLM: select exactly min(5, N) indices from the annotated list.
#   - Post-LLM: validate / deduplicate / trim only — never adds unselected moves.
#   - Retry: up to 9 attempts (3×10s/20s/40s) on transient Groq 429s.
#   - Output: {"selected_indices": [...]} in expansion_basis index space.

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
    RED,
    BLACK,
    RED_KING,
    BLACK_KING,
    is_own_piece,
)
from checkers.engine.rules import get_all_legal_moves
from checkers.engine.move_facts import compute_move_facts

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


def _role_pin_moves(
    sorted_moves: list[tuple[dict, dict]],
    score_state: str,
    n_slots: int,
) -> tuple[list[tuple[dict, dict]], frozenset[int]]:
    """
    Ensures strategic role diversity by pinning up to one protected move per
    critical role into the first n_slots positions after the symbolic pre-sort.

    Only acts when a protected move is currently OUTSIDE positions 0..n_slots-1.
    Position 0 (symbolic best) is never displaced.

    Protected roles (pinned in priority order after position 0):
      1. Best promotion / near-promotion move
      2. Best mobility-reduction move (mobility_reduction > 0)
      3. Best winning-conversion move  (only when score_state is winning)
      4. Best counterplay move         (only when score_state is not winning)

    Returns (reordered_moves, pinned_new_positions) where pinned_new_positions
    is the frozenset of positions in the returned list that were role-pinned.
    """
    n = len(sorted_moves)
    if n <= n_slots:
        return sorted_moves, frozenset()  # everything already in the visible window

    _WINNING = ("CLEARLY_WINNING", "SLIGHTLY_WINNING")

    def _find_best(score_fn, filter_fn) -> int | None:
        """Index of the best qualifying move in the full list (or None)."""
        best_i, best_s = None, float("-inf")
        for i, (m, f) in enumerate(sorted_moves):
            if not filter_fn(f):
                continue
            s = score_fn(f)
            if s > best_s:
                best_s, best_i = s, i
        return best_i

    to_pin_indices: set[int] = set()

    # ── Pin 1: best promotion / near-promotion ───────────────────────────
    i = _find_best(
        score_fn=lambda f: 2 if f.get("results_in_king") else 1,
        filter_fn=lambda f: f.get("results_in_king") or f.get("near_promotion"),
    )
    if i is not None and i >= n_slots:
        to_pin_indices.add(i)

    # ── Pin 2: best SAFE mobility-reduction move ─────────────────────────
    i = _find_best(
        score_fn=lambda f: f.get("mobility_reduction", 0),
        filter_fn=lambda f: (
            f.get("mobility_reduction", 0) > 0
            and not f.get("opponent_can_recapture", False)
        ),
    )
    if i is not None and i >= n_slots:
        to_pin_indices.add(i)

    # ── Pin 3: best SAFE conversion move (only when winning) ───────────────
    if score_state in _WINNING:
        i = _find_best(
            score_fn=lambda f: f.get("winning_conversion_score", 0),
            filter_fn=lambda f: (
                f.get("winning_conversion_score", 0) > 0
                and not f.get("opponent_can_recapture", False)
            ),
        )
        if i is not None and i >= n_slots:
            to_pin_indices.add(i)

    # ── Pin 4: best counterplay move (when not winning) ────────────
    # In losing states, the recapture guard is intentionally dropped:
    # accepting a recapture IS the definition of counterplay — we must
    # never silently exclude the best active counterplay move from the
    # shortlist just because the opponent can recapture.
    if score_state not in _WINNING:
        _LOSING = ("CLEARLY_LOSING", "SLIGHTLY_LOSING")
        if score_state in _LOSING:
            # No recapture guard: pin the single best counterplay move.
            i = _find_best(
                score_fn=lambda f: f.get("counterplay_score", 0),
                filter_fn=lambda f: f.get("counterplay_score", 0) > 0,
            )
        else:
            # EQUAL: keep the original safe-only guard.
            i = _find_best(
                score_fn=lambda f: f.get("counterplay_score", 0),
                filter_fn=lambda f: (
                    f.get("counterplay_score", 0) > 0
                    and not f.get("opponent_can_recapture", False)
                ),
            )
        if i is not None and i >= n_slots:
            to_pin_indices.add(i)

    if not to_pin_indices:
        return sorted_moves, frozenset()  # nothing needs pinning

    # Rebuild: position 0 preserved, pinned moves next, rest in original order.
    to_pin = [sorted_moves[i] for i in sorted(to_pin_indices)]
    rest   = [
        item
        for i, item in enumerate(sorted_moves)
        if i != 0 and i not in to_pin_indices
    ]
    result = [sorted_moves[0]] + to_pin + rest
    assert len(result) == n, "_role_pin_moves must not change the number of moves"

    # Pinned positions in the NEW list: 1..len(to_pin)
    pinned_new_positions = frozenset(range(1, 1 + len(to_pin)))
    return result, pinned_new_positions


def _build_legal_moves_with_facts(
    board: list[list[int]],
    current_player: int,
    strategic_context: dict | None = None,
    override_moves: list[dict] | None = None,
    score_by_path: dict[tuple, float] | None = None,
) -> tuple[int, str, list[tuple[dict, dict]]]:
    """
    Builds the sorted+pinned move presentation for the LLM.

    Returns a 5-tuple:
      (n_moves, formatted_block_string, sorted_moves_with_facts,
       path_to_basis_idx, mm_pinned_pres_idx)

    override_moves: skip get_all_legal_moves and use this list instead
      (e.g. state.symbolic_scored_moves already scored by symbolic_decision).

    score_by_path: path-key → minimax_score float. When provided, injects
      minimax_score into each move's facts payload for LLM visibility and
      drives mm_pin and MINIMAX_RANK markers. Preserves any pre-existing
      minimax_score (benchmark harness / minimax_scorer injection).
    """
    moves = override_moves if override_moves is not None else get_all_legal_moves(board, current_player)

    ctx = strategic_context or {}
    score_state          = ctx.get("score_state", "EQUAL")
    game_phase           = ctx.get("game_phase", "MIDGAME")
    strategic_priorities = ctx.get("strategic_priorities", [])

    # Compute facts for every move, then optionally inject minimax_score.
    moves_with_facts: list[tuple[dict, dict]] = []
    for m in moves:
        try:
            facts = compute_move_facts(board, m, current_player)
        except Exception:
            facts = {}
        # Resolve minimax_score from three sources (highest priority first):
        #   1. score_by_path (built from state.symbolic_scored_moves — real pipeline)
        #   2. m["facts"]["minimax_score"] (injected by benchmark harness or minimax_scorer)
        #   3. None (score not available; field omitted from payload)
        if "minimax_score" not in facts:
            path_key = tuple(tuple(sq) for sq in m["path"])
            sc: float | None = None
            if score_by_path is not None:
                sc = score_by_path.get(path_key)
            if sc is None:
                # Fallback: read from the move dict's own pre-computed facts.
                pre = m.get("facts", {})
                v = pre.get("minimax_score")
                if v is not None and v != float("-inf"):
                    sc = float(v)
            if sc is not None:
                facts["minimax_score"] = sc
        moves_with_facts.append((m, facts))

    # Record path -> original index BEFORE sorting so we can translate
    # LLM indices (sorted+pinned space) back to expansion_basis indices
    # (format_checker's index space) after the LLM makes its selection.
    path_to_basis_idx: dict[tuple, int] = {
        tuple(tuple(sq) for sq in m["path"]): i
        for i, (m, _) in enumerate(moves_with_facts)
    }

    # Pre-sort by symbolic quality key
    moves_with_facts.sort(
        key=lambda pair: _proposal_sort_key(
            pair[1], score_state, game_phase, strategic_priorities
        )
    )

    # Role-pin: after the symbolic sort, guarantee that moves representing
    # protected strategic roles (promotion, mobility-reduction, conversion,
    # counterplay) appear within the first n_slots positions so the LLM
    # always sees at least one representative per active role.
    n_slots = min(5, len(moves_with_facts))
    moves_with_facts, pinned_positions = _role_pin_moves(
        moves_with_facts, score_state, n_slots
    )

    # ── Minimax-best pin ────────────────────────────────────────────────────
    # The symbolic sort key ranks by centrality, safety and role — NOT by
    # minimax value.  Confirmed from trace: moves at edge columns or with low
    # centrality score (e.g. col-0/col-6 destinations) routinely fall to
    # positions 5+ even when they are the minimax-best by 94–192 points.
    # This pin reorders existing moves before the LLM sees them so the
    # canonical minimax-best is always inside n_slots. It does not create
    # new moves and does not modify LLM output after selection.
    # Requires score_by_path (built from search_root_all_scores); skipped
    # when scores are unavailable.
    mm_pinned_pres_idx: int | None = None
    if len(moves_with_facts) > n_slots and score_by_path:
        _mm_best_i = None
        _mm_best_s = float("-inf")
        for _i, (_mv, _) in enumerate(moves_with_facts):
            _pk = tuple(tuple(sq) for sq in _mv.get("path", []))
            _s = score_by_path.get(_pk, float("-inf"))
            if _s > _mm_best_s:
                _mm_best_s = _s
                _mm_best_i = _i
        # Only pin if the best minimax move is OUTSIDE the visible window
        # and is not already position 0 (symbolic best is never displaced).
        if _mm_best_i is not None and _mm_best_i >= n_slots:
            _pin_item = moves_with_facts.pop(_mm_best_i)
            # Insert at n_slots-1 so position 0 (symbolic sort top) is preserved.
            _insert_at = n_slots - 1
            moves_with_facts.insert(_insert_at, _pin_item)
            pinned_positions = pinned_positions | frozenset({_insert_at})
            mm_pinned_pres_idx = _insert_at
    # ── End minimax-best pin ────────────────────────────────────────────────

    n = len(moves_with_facts)

    # ── Minimax rank markers ────────────────────────────────────────────────
    # Identify the top-3 moves by minimax score in the FINAL presentation
    # order (after all sorts and pins), using the same score_by_path that
    # mm_pin uses — no new computation.  These markers appear beside each
    # move line so the LLM sees them without scanning individual JSON blobs.
    _mm_rank_for_idx: dict[int, int] = {}  # presentation index → rank 1/2/3
    if score_by_path:
        _scored_pres: list[tuple[float, int]] = []
        for _i, (_mv, _) in enumerate(moves_with_facts):
            _pk = tuple(tuple(sq) for sq in _mv.get("path", []))
            _s = score_by_path.get(_pk)
            if _s is not None and _s != float("-inf"):
                _scored_pres.append((_s, _i))
        # Sort descending by score; ties broken by lower presentation index.
        _scored_pres.sort(key=lambda t: (-t[0], t[1]))
        for _rank_0, (_, _i) in enumerate(_scored_pres[:3]):
            _mm_rank_for_idx[_i] = _rank_0 + 1  # 1, 2, or 3

    has_pins  = bool(pinned_positions)
    has_ranks = bool(_mm_rank_for_idx)
    lines = [
        f"N = {n}  (legal moves indexed 0 .. {n - 1} when N > 0).",
        "",
        "LEGAL_MOVES (path and captured are engine ground truth — never modify them):",
    ]
    if has_pins:
        lines.append(
            "  [★ = ROLE_SUGGESTION: this move covers a key strategic role that would "
            "otherwise be absent from the shortlist. Include it unless clearly unsafe "
            "or tactically dominated.]"
        )
    if has_ranks:
        lines.append(
            "  [MINIMAX_RANK_1/2/3 = top moves by symbolic minimax search — "
            "always include RANK_1; include RANK_2 and RANK_3 when selecting 5 moves]"
        )
    lines.append("")

    for i, (m, facts) in enumerate(moves_with_facts):
        mm_sc = facts.get("minimax_score")
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
        # Include minimax_score only when it is a finite numeric value.
        if mm_sc is not None and mm_sc != float("-inf"):
            essential["minimax_score"] = round(mm_sc, 1)

        payload = {
            "type":     m["type"],
            "path":     m["path"],
            "captured": m["captured"],
            "facts":    essential,
        }
        prefix   = "★ " if i in pinned_positions else "  "
        rank_tag = f" MINIMAX_RANK_{_mm_rank_for_idx[i]}" if i in _mm_rank_for_idx else ""
        lines.append(f"{prefix}[{i}]{rank_tag} {json.dumps(payload, ensure_ascii=False)}")

    return n, "\n".join(lines), moves_with_facts, path_to_basis_idx, mm_pinned_pres_idx


def _postprocess_llm_selection(
    selected_indices: list[int],
    n_moves: int,
) -> list[int]:
    """
    Post-LLM index cleanup — validates, deduplicates, and trims only.

    After the LLM returns selected_indices this function:
      1. Keeps only integer indices in [0, n_moves).
      2. Removes duplicates, preserving LLM order.
      3. If len > target: trims to target from the already-selected list.
      4. If len < target: does NOT pad with unselected moves.

    It does NOT add any move the LLM did not select.
    It does NOT inject safe moves, promotion moves, or any symbolic candidate.
    """
    target = min(5, max(1, n_moves))  # same count rule as before
    seen: set[int] = set()
    cleaned: list[int] = []
    for i in selected_indices:
        if isinstance(i, int) and 0 <= i < n_moves and i not in seen:
            seen.add(i)
            cleaned.append(i)
    return cleaned[:target]


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

STEP 0 — MINIMAX RANK (mandatory when present)
Moves labeled MINIMAX_RANK_1, MINIMAX_RANK_2, MINIMAX_RANK_3 are the top
moves by symbolic minimax search score — the strongest available tactical
signal for the current position.
When selecting candidates:
1. Always include MINIMAX_RANK_1 in your shortlist.
2. Include MINIMAX_RANK_2 and MINIMAX_RANK_3 whenever they appear in the list
    and you are selecting 5 moves.
3. Use remaining slots for useful tactical/strategic alternatives:
   - captures (captures_count > 0)
   - promotion/conversion (results_in_king=true, near_promotion=true)
   - safety/counterplay (counterplay_score, creates_immediate_threat=true)
   - mobility restriction (mobility_reduction > 0)
4. Do not skip MINIMAX_RANK_1/2/3 only for diversity.


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
MINIMAX_RANK also dominates diversity — do not drop RANK_1/2/3 to vary roles.

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

def build_proposal_prompts(
    state: CheckersState,
) -> tuple[str, str, int, list[tuple[dict, dict]], dict, int | None]:
    """
    Returns (system_prompt, user_prompt, n_moves, moves_with_facts, path_to_basis_idx,
             mm_pinned_pres_idx).

    moves_with_facts    — sorted+pinned presentation order used by the postprocessor.
    path_to_basis_idx   — maps each move's path tuple -> its index in the original
                          expansion_basis (legal_basis or scored_basis) so that
                          proposal_agent can translate LLM indices before handing
                          them to format_checker.
    mm_pinned_pres_idx  — presentation index where the minimax-best move was pinned
                          (None when no pin was needed).
    """
    board   = state.board
    current = state.current_player
    counts  = _king_counts(board, current)

    # ── Phase 8: use symbolic_scored_moves as candidate pool ────────────────────
    scored_override: list[dict] | None = None
    # Build score_by_path from symbolic_scored_moves so minimax_score is visible
    # to the LLM in the LEGAL_MOVES payload (before-LLM information improvement).
    # Entry schema from symbolic_decision: {"move": {...}, "minimax_score": float, "rank": int}
    score_by_path: dict[tuple, float] = {}
    if state.symbolic_scored_moves:
        scored_override = [entry["move"] for entry in state.symbolic_scored_moves]
        for entry in state.symbolic_scored_moves:
            # Accept both "minimax_score" (real pipeline) and "score" (legacy/benchmark).
            sc = entry.get("minimax_score") if "minimax_score" in entry else entry.get("score")
            if sc is not None:
                path_key = tuple(tuple(sq) for sq in entry["move"]["path"])
                score_by_path[path_key] = float(sc)
        print(
            f"[proposal_agent] using symbolic_scored_moves ({len(scored_override)} candidates, "
            f"best_score={state.symbolic_best_score:.1f}, gap={state.symbolic_gap:.1f})"
        )


    # Fallback: read minimax_score from state.legal_moves facts when available.
    # The benchmark harness (and minimax_scorer node) inject scores into those dicts.
    if not score_by_path:
        for lm in (getattr(state, 'legal_moves', None) or []):
            v = lm.get('facts', {}).get('minimax_score')
            if v is not None and v != float('-inf'):
                path_key = tuple(tuple(sq) for sq in lm['path'])
                score_by_path[path_key] = float(v)
    n_moves, legal_block, moves_with_facts, path_to_basis_idx, mm_pinned_pres_idx = (
        _build_legal_moves_with_facts(
            board, current, state.strategic_context,
            override_moves=scored_override,
            score_by_path=score_by_path if score_by_path else None,
        )
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

    return _SYSTEM_PROMPT, "\n".join(user_parts), n_moves, moves_with_facts, path_to_basis_idx, mm_pinned_pres_idx


def _translate_to_basis_indices(
    selected: list[int],
    moves_with_facts: list[tuple[dict, dict]],
    path_to_basis_idx: dict,
) -> list[int]:
    """
    Convert sorted+pinned presentation indices to expansion_basis indices.

    proposal_agent numbers moves 0..n-1 in sorted+pinned order; format_checker
    maps those same integers against the original (unsorted) expansion_basis.
    This function converts between the two spaces so format_checker picks the
    exact moves the LLM intended to select.
    """
    result: list[int] = []
    seen: set[int] = set()
    for i in selected:
        if 0 <= i < len(moves_with_facts):
            m, _ = moves_with_facts[i]
            key = tuple(tuple(sq) for sq in m["path"])
            j = path_to_basis_idx.get(key)
            if j is not None and j not in seen:
                seen.add(j)
                result.append(j)
    return result


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
    Calls Groq to propose the best candidate move indices.
    After the LLM responds, post-processing validates, deduplicates, and trims
    only the LLM-selected indices.  It does not add unselected moves.
    Retries up to 9 times (3×10s/20s/40s cycles) on transient 429s.
    """
    system, user, n_moves, moves_with_facts, path_to_basis_idx, mm_pinned_pres_idx = (
        build_proposal_prompts(state)
    )

    ctx                  = state.strategic_context or {}

    # ── ANSI colour helpers (safe on all POSIX terminals) ────────────────────
    _RED   = "\033[91m"
    _RESET = "\033[0m"

    # Hard-quota keywords — any match → fallback immediately, no retry
    _HARD_QUOTA_KEYWORDS = (
        "daily limit",
        "daily quota exhausted",
        "quota exhausted",
        "insufficient quota",
        "billing limit",
        "usage limit reached",
        "tokens per day",
        "tpd",
    )
    # Transient rate-limit keywords — any match → retry forever
    _TRANSIENT_RATE_LIMIT_KEYWORDS = (
        "rate limit",
        "too many requests",
        "resource exhausted",
        "429",
    )

    def _is_hard_quota(text: str) -> bool:
        t = text.lower()
        return any(kw in t for kw in _HARD_QUOTA_KEYWORDS)

    def _is_transient_rate_limit(text: str) -> bool:
        t = text.lower()
        return any(kw in t for kw in _TRANSIENT_RATE_LIMIT_KEYWORDS)

    # Retry waits: 3 full cycles of 10s→20s→40s = 9 attempts, then fallback
    _RETRY_WAITS = [10, 20, 40]
    _MAX_TRANSIENT_RETRIES = 9

    raw = None
    transient_attempt = 0

    while True:
        try:
            raw = call_groq_proposal(system, user)
            break  # success — zero delay
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as e:
            err_text = str(e)

            # ── Hard quota / daily limit → fallback immediately, no retry ─
            if _is_hard_quota(err_text):
                print(
                    f"{_RED}[proposal_agent] HARD QUOTA EXHAUSTED — "
                    f"falling back immediately (no retry). "
                    f"Error: {err_text[:300]}{_RESET}"
                )
                break

            # ── Transient 429 / rate limit → up to 3 retries ─────────────
            if _is_transient_rate_limit(err_text):
                if transient_attempt >= _MAX_TRANSIENT_RETRIES:
                    print(
                        f"{_RED}[proposal_agent] TRANSIENT RATE LIMIT — "
                        f"exhausted {_MAX_TRANSIENT_RETRIES} retries, falling back. "
                        f"Error: {err_text[:300]}{_RESET}"
                    )
                    break
                wait = _RETRY_WAITS[transient_attempt % len(_RETRY_WAITS)]
                transient_attempt += 1
                print(
                    f"{_RED}[proposal_agent] TRANSIENT RATE LIMIT "
                    f"(attempt {transient_attempt}/{_MAX_TRANSIENT_RETRIES}, waiting {wait}s). "
                    f"Error: {err_text[:300]}{_RESET}"
                )
                time.sleep(wait)
                continue

            # ── Any other error (network, timeout, …) → fallback ─────────
            print(
                f"{_RED}[proposal_agent] UNEXPECTED ERROR — "
                f"falling back. Error: {err_text[:300]}{_RESET}"
            )
            break

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
        print(f"{_RED}[proposal_agent] FALLBACK selected indices: {fallback_indices}{_RESET}")

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

    # ── Compute actual_best_pres_idx (for diagnostics only) ─────────────────
    # 1. Prefer state.symbolic_best_move["path"] if set.
    # 2. Else state.symbolic_scored_moves[0]["move"]["path"].
    # 3. Else None.
    _actual_best_path: list | None = None
    try:
        _sbm = getattr(state, "symbolic_best_move", None)
        if _sbm and isinstance(_sbm, dict) and "path" in _sbm:
            _actual_best_path = _sbm["path"]
    except Exception:
        pass
    if _actual_best_path is None and state.symbolic_scored_moves:
        try:
            _actual_best_path = state.symbolic_scored_moves[0]["move"]["path"]
        except Exception:
            pass

    _actual_best_pres_idx: int | None = None
    if _actual_best_path is not None:
        _best_key = tuple(tuple(sq) for sq in _actual_best_path)
        for _pi, (_mv, _) in enumerate(moves_with_facts):
            if tuple(tuple(sq) for sq in _mv["path"]) == _best_key:
                _actual_best_pres_idx = _pi
                break

    # ── Snapshot raw LLM presentation indices BEFORE postprocess ────────────
    _raw_pres_indices: list[int] = list(selected_indices)
    _raw_llm_selected_actual_best = (
        _actual_best_pres_idx is not None
        and _actual_best_pres_idx in _raw_pres_indices
    )

    # ── Post-LLM processing: validate / deduplicate / trim only ───────────
    selected_indices = _postprocess_llm_selection(selected_indices, n_moves)

    _final_pres_indices: list[int] = list(selected_indices)
    _final_contains_actual_best = (
        _actual_best_pres_idx is not None
        and _actual_best_pres_idx in _final_pres_indices
    )
    _dropped_by_postprocess = _raw_llm_selected_actual_best and not _final_contains_actual_best
    _added_after_llm = any(i not in _raw_pres_indices for i in _final_pres_indices)

    # ── Logging ─────────────────────────────────────────────────────
    print(f"[proposal_agent] Raw output: '{raw.strip()}'")
    print(f"[proposal_agent] actual_best_path={_actual_best_path}  actual_best_pres_idx={_actual_best_pres_idx}")
    print(f"[proposal_agent] mm_pinned_pres_idx={mm_pinned_pres_idx} (pin slot, not necessarily best)")
    print(f"[proposal_agent] Raw LLM presentation indices: {_raw_pres_indices}")
    print(
        f"[proposal_agent] "
        f"raw_llm_selected_actual_best={_raw_llm_selected_actual_best}  "
        f"final_contains_actual_best={_final_contains_actual_best}  "
        f"dropped_by_postprocess={_dropped_by_postprocess}  "
        f"added_after_llm={_added_after_llm}"
    )
    print(f"[proposal_agent] Post-postprocess presentation indices: {_final_pres_indices}")

    # ── Translate presentation indices -> expansion_basis indices ──────────
    selected_indices = _translate_to_basis_indices(
        selected_indices, moves_with_facts, path_to_basis_idx
    )

    final_raw = json.dumps({"selected_indices": selected_indices})
    print(f"[proposal_agent] Final basis indices (expansion_basis): {selected_indices}")
    _exp_basis = (
        [entry["move"] for entry in state.symbolic_scored_moves]
        if state.symbolic_scored_moves
        else get_all_legal_moves(state.board, state.current_player)
    )
    _final_paths = [_exp_basis[i]["path"] for i in selected_indices if 0 <= i < len(_exp_basis)]
    print(f"[proposal_agent] Final proposed paths: {_final_paths}")

    return {
        "proposed_moves": final_raw,
        "last_completed_node": "proposal_agent",
    }