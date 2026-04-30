# agents/ranker_agent.py
#
# LLM ranker for the American Checkers pipeline.
# Backend : Mistral API only (call_mistral_ranker / call_ranker).
# Model   : MISTRAL_RANKER_MODEL env var, default "mistral-small-latest".
# Key     : MISTRAL_API_KEY env var (console.mistral.ai).

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Optional

from checkers.state.state import CheckersState
from checkers.engine.board import RED

# ── Backend selection ─────────────────────────────────────────────────────────
# Ranker is fixed to Mistral API only.
RANKER_BACKEND = "mistral"

# ── Mistral settings ──────────────────────────────────────────────────────────
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_RANKER_MODEL = os.environ.get("MISTRAL_RANKER_MODEL", "mistral-small-latest")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


# ── Shared settings ───────────────────────────────────────────────────────────
RANKER_TEMPERATURE = float(os.environ.get("RANKER_TEMPERATURE", "0.2"))
_include = os.environ.get("RANKER_INCLUDE_STRATEGIC_CONTEXT", "true").lower()
RANKER_INCLUDE_STRATEGIC_CONTEXT = _include in ("1", "true", "yes", "on")

MINIMAX_ALL_UNSAFE_MARGIN = float(os.environ.get("MINIMAX_ALL_UNSAFE_MARGIN", "3.0"))
# Minimum minimax advantage over the best safe move required for an unsafe move
# to pass through the safety filter in non-losing (equal/winning) positions.
# Lowered from 20.0 → 14.0 to fix the T5-class bug: when a dominant unsafe move
# (gap=17pt) was rejected by the filter, the override logic never saw it.
# At 14.0, an unsafe move must still be meaningfully better (+14pt) to survive.
SAFETY_FILTER_LARGE_GAP = float(os.environ.get("SAFETY_FILTER_LARGE_GAP", "14.0"))
# ── Utility helpers ───────────────────────────────────────────────────────────

def _current_player_label(current_player: int) -> str:
    return "RED" if current_player == RED else "BLACK"


def _strip_markdown_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    return raw


def _parse_ranker_json(text: str) -> Optional[dict[str, Any]]:
    text = _strip_markdown_fences(text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start >= 0:
        end = text.rfind("}", start)
        while end > start:
            chunk = text[start : end + 1]
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            end = text.rfind("}", start, end)
        start = text.find("{", start + 1)
    return None


def _extract_chosen_index(obj: dict[str, Any]) -> Optional[int]:
    for key in (
        "chosen_index",
        "chosen_idx",
        "selected_index",
        "move_index",
        "index",
        "choice",
    ):
        if key not in obj:
            continue
        v = obj[key]
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if re.fullmatch(r"-?\d+", s):
                return int(s)
    return None


def _regex_extract_chosen_index(text: str) -> Optional[int]:
    patterns = (
        r'"chosen_index"\s*:\s*(\d+)',
        r"'chosen_index'\s*:\s*(\d+)",
        r'"selected_index"\s*:\s*(\d+)',
        r'"move_index"\s*:\s*(\d+)',
        r'chosen_index\s*[=:]\s*(\d+)',
    )
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _regex_extract_reasoning(text: str) -> Optional[str]:
    m = re.search(
        r'"reasoning"\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
        re.DOTALL,
    )
    if not m:
        return None
    inner = m.group(1).replace("\\n", " ").replace("\\\"", '"')
    inner = re.sub(r"\s+", " ", inner).strip()
    return inner or None


def _resolve_ranker_index(raw_idx: Optional[int], n: int) -> Optional[int]:
    if raw_idx is None or n <= 0:
        return None
    if 0 <= raw_idx < n:
        return raw_idx
    if 1 <= raw_idx <= n:
        return raw_idx - 1
    return None


# ── Safety filter ─────────────────────────────────────────────────────────────

def _apply_safety_filter(
    legal: list[dict[str, Any]],
    strategic_priorities: Optional[list[str]] = None,
    score_state: str = "EQUAL",
) -> tuple[list[dict[str, Any]], list[int]]:
    """
    Context-aware soft symbolic pre-filter.

    In equal/winning positions: keep safe moves, plus any unsafe move whose
    minimax_score exceeds the best safe score by SAFETY_FILTER_LARGE_GAP.
    This prevents a tactically dominant move from being invisible to the ranker
    just because it carries a small recapture risk.
    In losing/counterplay positions: also allow unsafe moves through
    if they have meaningfully better minimax_score AND real tactical action.
    This gives the ranker visibility into active counterplay continuations
    without opening the gate to every flashy unsafe move.

    PROMOTION EXEMPTION: Any move with results_in_king=True is always kept in
    the candidate set regardless of opponent_can_recapture. The King's enhanced
    mobility means a corner threat is rarely immediately fatal. The ranker and
    minimax override are trusted to evaluate the position correctly once the
    promotion move is visible.
    """
    priorities = strategic_priorities or []

    losing_mode = (
        score_state in ("CLEARLY_LOSING", "SLIGHTLY_LOSING")
        or "SEEK_COUNTERPLAY" in priorities
        or "COMPLICATE" in priorities
        or "CREATE_THREATS" in priorities
    )

    # ── Promotion exemption pass ──────────────────────────────────────────────
    # Identify promotion moves before any filtering. They will be added back
    # unconditionally at the end of each path so the ranker always sees them.
    promotion_moves: list[tuple[int, dict]] = [
        (i, m) for i, m in enumerate(legal)
        if (m.get("facts") or {}).get("results_in_king", False)
    ]
    if promotion_moves:
        promo_paths = [m.get("path") for _, m in promotion_moves]
        promo_scores = [_get_minimax_score(m) for _, m in promotion_moves]
        print(
            f"[SAFETY_FILTER][PROMOTION] {len(promotion_moves)} promotion move(s) detected: "
            f"paths={promo_paths} scores={promo_scores}. "
            "These will survive the safety filter unconditionally."
        )

    def _merge_promotions(
        kept: list[tuple[int, dict]],
    ) -> list[tuple[int, dict]]:
        """Add any promotion moves not already in kept, then sort by original index."""
        kept_indices = {idx for idx, _ in kept}
        for i, m in promotion_moves:
            if i not in kept_indices:
                kept.append((i, m))
                print(
                    f"[SAFETY_FILTER][PROMOTION] Promotion move {m.get('path')} "
                    f"(score={_get_minimax_score(m):.1f}) added back after safety filter."
                )
        kept.sort(key=lambda x: x[0])
        return kept
    # ─────────────────────────────────────────────────────────────────────────

    safe = [
        (i, m) for i, m in enumerate(legal)
        if not m.get("facts", {}).get("opponent_can_recapture", False)
    ]

    # If no safe moves exist, keep all.
    if not safe:
        return legal, list(range(len(legal)))

    best_safe_score = max(_get_minimax_score(m) for _, m in safe)

    def _has_real_action(facts: dict) -> bool:
        return (
            facts.get("creates_immediate_threat", False)
            or facts.get("shot_sequence_available", False)
            or facts.get("blocks_opponent_landing", False)
        )

    def _unsafe_qualifies(m: dict) -> bool:
        """Returns True if an unsafe move is strong enough to show the ranker."""
        facts = m.get("facts", {}) or {}
        score = _get_minimax_score(m)
        gap = score - best_safe_score
        has_action = _has_real_action(facts)
        strong_counterplay = facts.get("counterplay_score", 0) >= 12
        # Always allow through if minimax gap is large enough (normal rule)
        if gap >= MINIMAX_ALL_UNSAFE_MARGIN:
            return True
        # In losing/counterplay mode: allow if gap is meaningful AND
        # the move has concrete tactical justification
        if losing_mode and gap >= 2.0 and (has_action or strong_counterplay):
            return True
        return False

    pre_filter_size = len(legal)

    if len(safe) >= 2:
        if not losing_mode:
            # Normal mode: keep safe moves, plus any unsafe move that is
            # tactically dominant (minimax gap above SAFETY_FILTER_LARGE_GAP).
            kept = list(safe)
            for i, m in enumerate(legal):
                if any(i == si for si, _ in safe):
                    continue
                score = _get_minimax_score(m)
                if score > best_safe_score + SAFETY_FILTER_LARGE_GAP:
                    kept.append((i, m))
            kept = _merge_promotions(kept)
            post_filter_size = len(kept)
            print(
                f"[SAFETY_FILTER] pre={pre_filter_size} post={post_filter_size} "
                f"mode=normal safe_count={len(safe)}"
            )
            if len(kept) == len([k for k in kept if any(k[0] == si for si, _ in safe)]):
                # No dominant unsafe moves and no promotions added — fast path.
                return [m for _, m in kept], [i for i, _ in kept]
            kept.sort(key=lambda x: x[0])
            return [m for _, m in kept], [i for i, _ in kept]
        # Losing mode: keep safe moves + qualifying unsafe moves
        kept = list(safe)
        for i, m in enumerate(legal):
            if any(i == si for si, _ in safe):
                continue
            if _unsafe_qualifies(m):
                kept.append((i, m))
        kept = _merge_promotions(kept)
        post_filter_size = len(kept)
        print(
            f"[SAFETY_FILTER] pre={pre_filter_size} post={post_filter_size} "
            f"mode=losing safe_count={len(safe)}"
        )
        kept.sort(key=lambda x: x[0])
        return [m for _, m in kept], [i for i, _ in kept]

    # Only one safe move — keep it plus any qualifying unsafe moves
    safe_idx, safe_move = safe[0]
    kept: list[tuple[int, dict[str, Any]]] = [(safe_idx, safe_move)]
    for i, m in enumerate(legal):
        if i == safe_idx:
            continue
        if _unsafe_qualifies(m):
            kept.append((i, m))
    kept = _merge_promotions(kept)
    post_filter_size = len(kept)
    print(
        f"[SAFETY_FILTER] pre={pre_filter_size} post={post_filter_size} "
        f"mode={'losing' if losing_mode else 'normal'} safe_count=1"
    )
    if post_filter_size == 1 and not losing_mode:
        print(
            f"[WARNING][SAFETY_FILTER] Collapsed to 1 candidate in non-losing mode. "
            f"best_safe_score={best_safe_score:.1f}. "
            "Consider reviewing SAFETY_FILTER_LARGE_GAP."
        )
    return [m for _, m in kept], [i for i, _ in kept]

    


# ── Minimax dominance guardrail ──────────────────────────────────────────────

MINIMAX_DOMINANCE_MARGIN = float(os.environ.get("MINIMAX_DOMINANCE_MARGIN", "2.0"))
QUIET_MINIMAX_MARGIN = float(os.environ.get("QUIET_MINIMAX_MARGIN", "2.0"))
TACTICAL_MINIMAX_MAX_GAP = float(os.environ.get("TACTICAL_MINIMAX_MAX_GAP", "8.0"))
LOW_DANGER_ACTIVE_MINIMAX_GAP = float(os.environ.get("LOW_DANGER_ACTIVE_MINIMAX_GAP", "8.0"))
OPENING_LOW_DANGER_GAP = float(os.environ.get("OPENING_LOW_DANGER_GAP", "6.0"))
LOW_DANGER_MINIMAX_GAP = float(os.environ.get("LOW_DANGER_MINIMAX_GAP", "3.0"))
# Minimax gap above which structural exceptions (isolation, weakens_king_row)
# can no longer suppress an override.  When the minimax gap is this large,
# the search has already priced in the structural risk — letting isolation
# cancel a 15+ pt gap is wrong.  Previous value was 50.0 (too permissive),
# which caused the T5-class bug where a 17 pt gap was vetoed by isolation.
STRUCTURE_EXCEPTION_FLOOR = float(os.environ.get("STRUCTURE_EXCEPTION_FLOOR", "15.0"))
# Gap at which the override fires even when the best move is unsafe (threat_after=1)
# but the chosen move is fully safe. Prevents the LLM's safety-first bias from
# ignoring a tactically dominant move that was explicitly shown in the filtered menu.
# Lowered from 18.0 → 15.0: the T5-class gap is 17pt which was 1pt below threshold.
# At depth-4 minimax, 15pt already accounts for structural risk; letting the LLM
# absorb a 15+ pt penalty for safety bias is no longer acceptable.
SAFE_VS_UNSAFE_OVERRIDE_GAP = float(os.environ.get("SAFE_VS_UNSAFE_OVERRIDE_GAP", "15.0"))
# Fallback gap threshold for the unsafe-vs-unsafe blind spot.
# When BOTH chosen and best are unsafe (no branch covers them), this fires if
# the gap is large enough to be unambiguous.
# TWO TIERS:
#   Tier-1 (gap ≥ 35): fires when best_threat < chosen_threat — best is safer
#     despite both being "unsafe". Confirmed at T11 (gap=42, best_th=1 < chosen_th=2).
#   Tier-2 (gap ≥ 150): fires when threat exposure is equal — conservative to
#     avoid false positives from noise in deep-game scores.
UNSAFE_VS_UNSAFE_LOWER_THREAT_GAP = float(os.environ.get("UNSAFE_VS_UNSAFE_LOWER_THREAT_GAP", "35.0"))
UNSAFE_VS_UNSAFE_FALLBACK_GAP = float(os.environ.get("UNSAFE_VS_UNSAFE_FALLBACK_GAP", "150.0"))
# Midgap threshold: equal threat exposure on both moves, override purely by minimax.
# Slightly above tier-1 (35) because no threat advantage justifies the override —
# minimax must work alone. Confirmed T13-class miss: gap=56, best_threat=chosen_threat=1.
UNSAFE_VS_UNSAFE_MIDGAP_GAP = float(os.environ.get("UNSAFE_VS_UNSAFE_MIDGAP_GAP", "40.0"))


def _get_minimax_score(move: dict[str, Any]) -> float:
    facts = move.get("facts", {}) or {}
    v = facts.get("minimax_score", float("-inf"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("-inf")


def _best_and_second_best_minimax(moves: list[dict[str, Any]]) -> tuple[Optional[int], Optional[float], Optional[float]]:
    """
    Returns:
      (best_idx, best_score, second_best_score)
    based on minimax_score over the given move list.
    """
    if not moves:
        return None, None, None

    scored = [(i, _get_minimax_score(m)) for i, m in enumerate(moves)]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_idx, best_score = scored[0]
    second_best_score = scored[1][1] if len(scored) > 1 else None
    return best_idx, best_score, second_best_score


def _should_force_best_minimax(
    moves: list[dict[str, Any]],
) -> tuple[bool, Optional[int], str]:
    """
    Disabled pre-LLM forcing.

    We no longer let shallow minimax choose the move before the ranker.
    Minimax is now a strong verifier and post-LLM override signal only.
    """
    return False, None, ""


def _override_if_llm_chose_much_worse_minimax(
    filtered: list[dict[str, Any]],
    llm_idx: int,
    game_phase: str = "MIDGAME",
    score_state: str = "EQUAL",
    strategic_priorities: Optional[list[str]] = None,
    comparison_moves: Optional[list[dict[str, Any]]] = None,
) -> tuple[dict[str, Any], Optional[str], dict[str, Any]]:
    """
    Override the LLM only in a narrow case:
    - the LLM chose a clearly worse minimax move
    - the best minimax move is SAFE
    - the chosen move is unsafe or tactically much worse

    This prevents shallow minimax from hijacking strategy in normal positions,
    while still protecting against catastrophic blunders.
    """
    moves_for_best = comparison_moves if comparison_moves is not None else filtered

    best_idx, best_score, _ = _best_and_second_best_minimax(moves_for_best)
    if best_idx is None or best_score is None:
        return filtered[llm_idx], None, {}

    best_move = moves_for_best[best_idx]
    chosen_move = filtered[llm_idx]

    best_facts = best_move.get("facts", {}) or {}
    chosen_facts = chosen_move.get("facts", {}) or {}

    best_safe = not best_facts.get("opponent_can_recapture", False)
    chosen_safe = not chosen_facts.get("opponent_can_recapture", False)

    chosen_score = _get_minimax_score(chosen_move)
    gap = best_score - chosen_score

    def _is_quiet_non_tactical(facts: dict[str, Any]) -> bool:
        return (
            facts.get("captures_count", 0) == 0
            and not facts.get("creates_immediate_threat", False)
            and not facts.get("shot_sequence_available", False)
            and not facts.get("blocks_opponent_landing", False)
        )

    def _is_quiet_safe_opening_style(facts: dict[str, Any]) -> bool:
        return (
            not facts.get("opponent_can_recapture", False)
            and not facts.get("moved_piece_is_threatened", False)
            and facts.get("our_pieces_threatened_after", 99) <= 1
            and facts.get("net_gain", -999) >= 0
        )

    best_quiet = _is_quiet_non_tactical(best_facts)
    chosen_quiet = _is_quiet_non_tactical(chosen_facts)
    best_quiet_safe = _is_quiet_safe_opening_style(best_facts)
    chosen_quiet_safe = _is_quiet_safe_opening_style(chosen_facts)

    def _is_low_danger_active(facts: dict[str, Any]) -> bool:
        return (
            not facts.get("opponent_can_recapture", False)
            and not facts.get("moved_piece_is_threatened", False)
            and facts.get("our_pieces_threatened_after", 99) <= 1
        )

    def _is_passive_safe_structural(facts: dict[str, Any]) -> bool:
        role = str(facts.get("quiet_move_role", "") or "")
        return (
            role in {"STRUCTURAL_RESTRICTION", "KING_ACTIVATION", "PROMOTION_PUSH"}
            and not facts.get("opponent_can_recapture", False)
            and not facts.get("moved_piece_is_threatened", False)
            and facts.get("our_pieces_threatened_after", 99) == 0
        )

    def _is_acceptable_low_danger_candidate(facts: dict[str, Any]) -> bool:
        """
        Broader than strict low-danger active:
        allow tactical/forcing candidates into minimax comparison when they are
        still not materially dangerous in immediate reply terms.
        """
        if facts.get("opponent_can_recapture", False):
            return False
        if facts.get("moved_piece_is_threatened", False):
            return False

        threat_after = facts.get("our_pieces_threatened_after", 99)
        max_jump = facts.get("max_opponent_jump_captures", 99)
        jump_count = facts.get("opponent_jump_count", 99)
        forced_reply = bool(facts.get("forced_opponent_jump_reply", False))

        # Strict lane remains valid.
        if threat_after <= 1:
            return True

        # Broader lane for practical tactical candidates:
        # modest threat footprint and no heavy immediate jump punishment.
        return (
            threat_after <= 2
            and not forced_reply
            and max_jump <= 1
            and jump_count <= 1
        )

    best_threat = best_facts.get("our_pieces_threatened_after", 99)
    chosen_threat = chosen_facts.get("our_pieces_threatened_after", 99)

    debug_info: dict[str, Any] = {
        "is_passive_safe_structural": {
            "chosen": _is_passive_safe_structural(chosen_facts),
            "best": _is_passive_safe_structural(best_facts),
        },
        "is_low_danger_active": {
            "chosen": _is_low_danger_active(chosen_facts),
            "best": _is_low_danger_active(best_facts),
        },
        "override_branch_triggered": False,
        "override_branch_name": None,
        "override_block_reason": None,
        "best_move_rejected_reason": None,
        "best_vs_chosen_minimax_gap": round(gap, 2),
        "best_vs_chosen_threat_delta": (
            best_threat - chosen_threat
            if isinstance(best_threat, (int, float)) and isinstance(chosen_threat, (int, float))
            else None
        ),
    }

    def _b(x: Any) -> int:
        return 1 if bool(x) else 0

    def _f(x: Any) -> str:
        try:
            return f"{float(x):.1f}"
        except (TypeError, ValueError):
            return "None"

    def _gate_kv(**kwargs: Any) -> str:
        # Compact serialization: key=value pairs without spaces.
        return ",".join(f"{k}={v}" for k, v in kwargs.items())

    def _chosen_has_concrete_defensive_advantage_over_best() -> bool:
        """
        True only when chosen is materially safer on immediate reply danger metrics.
        Offensive / forcing style must NOT count as defensive advantage.

        Refined: do not veto low-danger dominance on marginal numeric edges (jump
        metrics require >=2 margin) or when chosen is globally worse on threat
        footprint than best. Boolean immediate-danger edges still count when they
        asymmetrically favor chosen over best.
        """
        c_th = chosen_facts.get("our_pieces_threatened_after")
        b_th = best_facts.get("our_pieces_threatened_after")
        if isinstance(c_th, (int, float)) and isinstance(b_th, (int, float)):
            if c_th > b_th:
                return False

        b_mpj = best_facts.get("max_opponent_jump_captures", 99)
        c_mpj = chosen_facts.get("max_opponent_jump_captures", 99)
        b_jc = best_facts.get("opponent_jump_count", 99)
        c_jc = chosen_facts.get("opponent_jump_count", 99)
        jump_captures_edge = (
            isinstance(b_mpj, (int, float))
            and isinstance(c_mpj, (int, float))
            and (b_mpj - c_mpj) >= 2
        )
        jump_count_edge = (
            isinstance(b_jc, (int, float))
            and isinstance(c_jc, (int, float))
            and (b_jc - c_jc) >= 2
        )

        moved_piece_defensive_edge = (
            not chosen_facts.get("moved_piece_is_threatened", False)
            and bool(best_facts.get("moved_piece_is_threatened", False))
        )
        forced_jump_defensive_edge = (
            not chosen_facts.get("forced_opponent_jump_reply", False)
            and bool(best_facts.get("forced_opponent_jump_reply", False))
        )

        return (
            moved_piece_defensive_edge
            or forced_jump_defensive_edge
            or jump_captures_edge
            or jump_count_edge
        )

    # Low-danger vs low-danger: minimax dominates when there is no concrete defensive
    # advantage for the weaker minimax choice. Applies even when both moves are
    # quiet-safe (quiet-safe alone must not suppress this guardrail).
    if (
        best_safe
        and chosen_safe
        and _is_low_danger_active(best_facts)
        and _is_low_danger_active(chosen_facts)
        and gap >= LOW_DANGER_MINIMAX_GAP
        and abs(best_score) < 1000.0
        and abs(chosen_score) < 1000.0
    ):
        best_isolated = bool(best_facts.get("leaves_piece_isolated", False))
        chosen_isolated = bool(chosen_facts.get("leaves_piece_isolated", False))
        best_weakens_back = bool(best_facts.get("weakens_king_row", False))
        chosen_weakens_back = bool(chosen_facts.get("weakens_king_row", False))
        best_uniquely_worse_structure = (
            gap < STRUCTURE_EXCEPTION_FLOOR
            and (
                (best_isolated and not chosen_isolated)
                or (best_weakens_back and not chosen_weakens_back)
            )
        )
        if not best_uniquely_worse_structure and not _chosen_has_concrete_defensive_advantage_over_best():
            debug_info["override_branch_triggered"] = True
            debug_info["override_branch_name"] = "low_danger_minimax_dominance"
            return best_move, (
                "LLM low-danger choice overridden by minimax dominance guardrail: "
                f"chosen={chosen_score:.1f}, best={best_score:.1f}, "
                f"gap={gap:.1f} (threshold={LOW_DANGER_MINIMAX_GAP:.1f})."
            ), debug_info
        if _chosen_has_concrete_defensive_advantage_over_best():
            debug_info["override_block_reason"] = "chosen_concrete_defensive_better"
            debug_info["best_move_rejected_reason"] = "chosen_concrete_defensive_better"

    # Quiet-safe vs quiet-safe: minimax is primary.
    # Structural exception is narrow: only block override if the higher-minimax move
    # uniquely worsens isolation or king-row discipline.
    if (
        best_safe
        and chosen_safe
        and best_quiet
        and chosen_quiet
        and best_quiet_safe
        and chosen_quiet_safe
        and gap >= QUIET_MINIMAX_MARGIN
    ):
        best_isolated = bool(best_facts.get("leaves_piece_isolated", False))
        chosen_isolated = bool(chosen_facts.get("leaves_piece_isolated", False))
        best_weakens_back = bool(best_facts.get("weakens_king_row", False))
        chosen_weakens_back = bool(chosen_facts.get("weakens_king_row", False))

        best_uniquely_worse_structure = (
            gap < STRUCTURE_EXCEPTION_FLOOR
            and (
                (best_isolated and not chosen_isolated)
                or (best_weakens_back and not chosen_weakens_back)
            )
        )

        if not best_uniquely_worse_structure:
            debug_info["override_branch_triggered"] = True
            debug_info["override_branch_name"] = "quiet_safe_minimax_primary"
            return best_move, (
                "LLM quiet-safe choice overridden by minimax guardrail: "
                f"chosen={chosen_score:.1f}, best={best_score:.1f}, "
                f"gap={gap:.1f} (threshold={QUIET_MINIMAX_MARGIN:.1f})."
            ), debug_info
        debug_info["override_block_reason"] = "best_uniquely_worse_structure"
        debug_info["best_move_rejected_reason"] = "best_uniquely_worse_structure"

    # Safe-chosen vs unsafe-best: override when the gap is large enough to
    # justify accepting the risk. This fires only when the best move was
    # explicitly shown to the LLM (survived _apply_safety_filter) but the LLM
    # still chose a safe-but-weaker move due to Step 1 safety bias.
    # Does NOT modify any existing branch.
    if (
        chosen_safe
        and not best_safe
        and gap >= SAFE_VS_UNSAFE_OVERRIDE_GAP
    ):
        debug_info["override_branch_triggered"] = True
        debug_info["override_branch_name"] = "safe_vs_unsafe_large_gap"
        return best_move, (
            "LLM safe choice overridden: best move is unsafe but minimax gap is large: "
            f"chosen={chosen_score:.1f} (safe), best={best_score:.1f} (unsafe), "
            f"gap={gap:.1f} (threshold={SAFE_VS_UNSAFE_OVERRIDE_GAP:.1f})."
        ), debug_info

    priorities = strategic_priorities or []

    if game_phase == "OPENING":
        dominance_threshold = 15.0
    elif game_phase == "MIDGAME":
        if score_state in ("CLEARLY_LOSING", "SLIGHTLY_LOSING"):
            dominance_threshold = 4.0
        else:
            dominance_threshold = 6.0
    else:  # ENDGAME
        if score_state in ("CLEARLY_LOSING", "SLIGHTLY_LOSING"):
            dominance_threshold = 4.0
        elif score_state in ("CLEARLY_WINNING", "SLIGHTLY_WINNING"):
            dominance_threshold = max(MINIMAX_DOMINANCE_MARGIN, 4.0)
        else:
            dominance_threshold = 6.0

    if game_phase != "OPENING" and "ACTIVATE_KINGS" in priorities:
        dominance_threshold += 3.0

    chosen_king_activity = chosen_facts.get("king_activity_score", 0)
    chosen_counterplay = chosen_facts.get("counterplay_score", 0)
    best_counterplay = best_facts.get("counterplay_score", 0)

    chosen_has_real_action = (
        chosen_facts.get("creates_immediate_threat", False)
        or chosen_facts.get("shot_sequence_available", False)
        or chosen_facts.get("blocks_opponent_landing", False)
    )

    losing_mode = (
        score_state in ("CLEARLY_LOSING", "SLIGHTLY_LOSING")
        or "SEEK_COUNTERPLAY" in priorities
        or "COMPLICATE" in priorities
    )

    chosen_active_counterplay = (
        chosen_safe
        and (
            chosen_has_real_action
            or chosen_counterplay >= max(10, best_counterplay + 5)
            or chosen_king_activity >= 4
        )
    )

    chosen_strict_safe_structural = _is_passive_safe_structural(chosen_facts)
    best_low_danger_active = _is_low_danger_active(best_facts)
    best_acceptable_low_danger = _is_acceptable_low_danger_candidate(best_facts)

    # ── Predicate-level gate instrumentation (no behavior change) ─────────────
    # This is only used when we would otherwise fall through to
    # best_move_rejected_reason=no_override_condition_met.
    best_isolated = bool(best_facts.get("leaves_piece_isolated", False))
    chosen_isolated = bool(chosen_facts.get("leaves_piece_isolated", False))
    best_weakens_back = bool(best_facts.get("weakens_king_row", False))
    chosen_weakens_back = bool(chosen_facts.get("weakens_king_row", False))
    best_uniquely_worse_structure = (
        gap < STRUCTURE_EXCEPTION_FLOOR
        and (
            (best_isolated and not chosen_isolated) or (best_weakens_back and not chosen_weakens_back)
        )
    )

    low_danger_minimax_dominance_gate = _gate_kv(
        bs=_b(best_safe),
        cs=_b(chosen_safe),
        ldB=_b(_is_low_danger_active(best_facts)),
        ldC=_b(_is_low_danger_active(chosen_facts)),
        g=_f(gap),
        th=_f(LOW_DANGER_MINIMAX_GAP),
        gOK=_b(gap >= LOW_DANGER_MINIMAX_GAP),
        cap=_b(abs(best_score) < 1000.0 and abs(chosen_score) < 1000.0),
        structBlock=_b(best_uniquely_worse_structure),
        defBlock=_b(_chosen_has_concrete_defensive_advantage_over_best()),
    )

    quiet_safe_minimax_primary_gate = _gate_kv(
        bs=_b(best_safe),
        cs=_b(chosen_safe),
        bq=_b(best_quiet),
        cq=_b(chosen_quiet),
        bqs=_b(best_quiet_safe),
        cqs=_b(chosen_quiet_safe),
        g=_f(gap),
        th=_f(QUIET_MINIMAX_MARGIN),
        gOK=_b(gap >= QUIET_MINIMAX_MARGIN),
        structBlock=_b(best_uniquely_worse_structure),
    )

    opening_low_danger_minimax_guardrail_gate = _gate_kv(
        op=_b(game_phase == "OPENING"),
        csss=_b(chosen_strict_safe_structural),
        bAccLD=_b(best_acceptable_low_danger),
        g=_f(gap),
        th=_f(OPENING_LOW_DANGER_GAP),
        gOK=_b(gap >= OPENING_LOW_DANGER_GAP),
        structBlock=_b(best_uniquely_worse_structure),
    )

    low_danger_active_minimax_guardrail_gate = _gate_kv(
        csss=_b(chosen_strict_safe_structural),
        bAccLD=_b(best_acceptable_low_danger),
        g=_f(gap),
        th=_f(LOW_DANGER_ACTIVE_MINIMAX_GAP),
        gOK=_b(gap >= LOW_DANGER_ACTIVE_MINIMAX_GAP),
        structBlock=_b(best_uniquely_worse_structure),
    )

    safe_vs_safe_dominance_gate = _gate_kv(
        bs=_b(best_safe),
        cs=_b(chosen_safe),
        g=_f(gap),
        th=_f(dominance_threshold),
        gOK=_b(gap >= dominance_threshold),
    )

    # OPENING-specific low-danger guardrail:
    # do not let a perfectly safe structural lane (threat_after=0) suppress a
    # low-danger active alternative (threat_after<=1) when minimax is clearly better.
    if (
        game_phase == "OPENING"
        and chosen_strict_safe_structural
        and best_acceptable_low_danger
        and gap >= OPENING_LOW_DANGER_GAP
    ):
        best_isolated = bool(best_facts.get("leaves_piece_isolated", False))
        chosen_isolated = bool(chosen_facts.get("leaves_piece_isolated", False))
        best_weakens_back = bool(best_facts.get("weakens_king_row", False))
        chosen_weakens_back = bool(chosen_facts.get("weakens_king_row", False))
        best_uniquely_worse_structure = (
            gap < STRUCTURE_EXCEPTION_FLOOR
            and (
                (best_isolated and not chosen_isolated)
                or (best_weakens_back and not chosen_weakens_back)
            )
        )
        if not best_uniquely_worse_structure:
            debug_info["override_branch_triggered"] = True
            debug_info["override_branch_name"] = "opening_low_danger_minimax_guardrail"
            return best_move, (
                "LLM opening structural-safe choice overridden by low-danger minimax guardrail: "
                f"chosen={chosen_score:.1f}, best={best_score:.1f}, "
                f"gap={gap:.1f} (threshold={OPENING_LOW_DANGER_GAP:.1f})."
            ), debug_info
        debug_info["override_block_reason"] = "best_uniquely_worse_structure"
        debug_info["best_move_rejected_reason"] = "best_uniquely_worse_structure"

    if (
        chosen_strict_safe_structural
        and best_acceptable_low_danger
        and gap >= LOW_DANGER_ACTIVE_MINIMAX_GAP
    ):
        best_isolated = bool(best_facts.get("leaves_piece_isolated", False))
        chosen_isolated = bool(chosen_facts.get("leaves_piece_isolated", False))
        best_weakens_back = bool(best_facts.get("weakens_king_row", False))
        chosen_weakens_back = bool(chosen_facts.get("weakens_king_row", False))
        best_uniquely_worse_structure = (
            gap < STRUCTURE_EXCEPTION_FLOOR
            and (
                (best_isolated and not chosen_isolated)
                or (best_weakens_back and not chosen_weakens_back)
            )
        )
        if not best_uniquely_worse_structure:
            debug_info["override_branch_triggered"] = True
            debug_info["override_branch_name"] = "low_danger_active_minimax_guardrail"
            return best_move, (
                "LLM structural-safe choice overridden by low-danger active minimax guardrail: "
                f"chosen={chosen_score:.1f}, best={best_score:.1f}, "
                f"gap={gap:.1f} (threshold={LOW_DANGER_ACTIVE_MINIMAX_GAP:.1f})."
            ), debug_info
        debug_info["override_block_reason"] = "best_uniquely_worse_structure"
        debug_info["best_move_rejected_reason"] = "best_uniquely_worse_structure"
            

    # Hard cap for tactical-pressure protection:
    # if a tactical/active move is dramatically worse than the best genuinely safe move,
    # do not preserve it unless it is uniquely better on immediate danger severity.
    best_safe_idx = None
    best_safe_score = None
    for i, m in enumerate(filtered):
        f = m.get("facts", {}) or {}
        if (
            not f.get("opponent_can_recapture", False)
            and not f.get("moved_piece_is_threatened", False)
        ):
            s = _get_minimax_score(m)
            if best_safe_score is None or s > best_safe_score:
                best_safe_score = s
                best_safe_idx = i

    if best_safe_idx is not None and chosen_active_counterplay:
        safe_move = filtered[best_safe_idx]
        safe_facts = safe_move.get("facts", {}) or {}
        safe_score = _get_minimax_score(safe_move)
        tactical_gap_vs_best_safe = safe_score - chosen_score

        chosen_threat_after = chosen_facts.get("our_pieces_threatened_after", 99)
        safe_threat_after = safe_facts.get("our_pieces_threatened_after", 99)
        both_genuinely_low_danger = (
            _is_low_danger_active(chosen_facts)
            and _is_low_danger_active(safe_facts)
        )
        equal_threat_footprint = (
            isinstance(chosen_threat_after, (int, float))
            and isinstance(safe_threat_after, (int, float))
            and chosen_threat_after == safe_threat_after
        )

        chosen_concrete_danger_better = (
            (
                not chosen_facts.get("moved_piece_is_threatened", False)
                and safe_facts.get("moved_piece_is_threatened", False)
            )
            or (
                not chosen_facts.get("forced_opponent_jump_reply", False)
                and safe_facts.get("forced_opponent_jump_reply", False)
            )
            or (
                chosen_facts.get("max_opponent_jump_captures", 99)
                < safe_facts.get("max_opponent_jump_captures", 99)
            )
            or (
                chosen_facts.get("opponent_jump_count", 99)
                < safe_facts.get("opponent_jump_count", 99)
            )
        )

        # Do not let tactical style block minimax override when both options are
        # genuinely low-danger and equally safe on threat footprint.
        if both_genuinely_low_danger and equal_threat_footprint:
            chosen_better_immediate_danger = False
        else:
            chosen_better_immediate_danger = chosen_concrete_danger_better

        if (
            tactical_gap_vs_best_safe >= TACTICAL_MINIMAX_MAX_GAP
            and not chosen_better_immediate_danger
        ):
            debug_info["override_branch_triggered"] = True
            debug_info["override_branch_name"] = "tactical_pressure_hard_cap"
            return safe_move, (
                "LLM tactical-pressure choice overridden by minimax hard cap: "
                f"chosen={chosen_score:.1f}, best_safe={safe_score:.1f}, "
                f"gap={tactical_gap_vs_best_safe:.1f} "
                f"(threshold={TACTICAL_MINIMAX_MAX_GAP:.1f})."
            ), debug_info
        if (
            tactical_gap_vs_best_safe >= TACTICAL_MINIMAX_MAX_GAP
            and chosen_better_immediate_danger
        ):
            debug_info["override_block_reason"] = "chosen_better_immediate_danger"
            debug_info["best_move_rejected_reason"] = "chosen_better_immediate_danger"

    small_safety_gap = abs(best_threat - chosen_threat) <= 1

    if best_safe and gap >= dominance_threshold:
        if not chosen_safe:
            debug_info["override_branch_triggered"] = True
            debug_info["override_branch_name"] = "unsafe_chosen_vs_safe_best"
            return best_move, (
                f"LLM choice overridden by minimax guardrail ({game_phase}): "
                f"chosen={chosen_score:.1f}, best={best_score:.1f}, "
                f"safe_best=true, chosen_safe=false, gap={gap:.1f} "
                f"(threshold={dominance_threshold:.1f})."
            ), debug_info

        if (
            score_state in ("CLEARLY_LOSING", "SLIGHTLY_LOSING")
            and small_safety_gap
        ):
            chosen_bad_danger = (
                chosen_facts.get("moved_piece_is_threatened", False)
                or chosen_facts.get("forced_opponent_jump_reply", False)
                or chosen_facts.get("max_opponent_jump_captures", 0)
                > best_facts.get("max_opponent_jump_captures", 0)
            )

            if not chosen_bad_danger:
                debug_info["override_branch_triggered"] = True
                debug_info["override_branch_name"] = "losing_mode_small_safety_gap"
                return best_move, (
                    f"LLM choice overridden in losing mode: "
                    f"chosen={chosen_score:.1f}, best={best_score:.1f}, "
                    f"gap={gap:.1f}, small_safety_gap=true."
                ), debug_info
            debug_info["override_block_reason"] = "chosen_bad_danger"
            debug_info["best_move_rejected_reason"] = "chosen_bad_danger"

        # In OPENING, usually do NOT override safe-vs-safe choices with shallow minimax.
        # But if the chosen move is only "safer" by a tiny margin and the best move is still
        # opening-safe, allow the better minimax move to win.
        if game_phase == "OPENING" and chosen_safe:
            best_opening_safe = (
                best_safe
                and not best_facts.get("moved_piece_is_threatened", False)
                and best_facts.get("net_gain", 0) >= 0
                and best_facts.get("our_pieces_threatened_after", 99) <= 1
            )
            chosen_opening_safe = (
                chosen_safe
                and not chosen_facts.get("moved_piece_is_threatened", False)
                and chosen_facts.get("net_gain", 0) >= 0
                and chosen_facts.get("our_pieces_threatened_after", 99) <= 1
            )

            if best_opening_safe and chosen_opening_safe:
                best_isolated = best_facts.get("leaves_piece_isolated", False)
                chosen_isolated = chosen_facts.get("leaves_piece_isolated", False)
                best_weakens_back = best_facts.get("weakens_king_row", False)
                chosen_weakens_back = chosen_facts.get("weakens_king_row", False)
                best_center = bool(best_facts.get("center_control", False))
                chosen_center = bool(chosen_facts.get("center_control", False))

                # Hard override when minimax difference is meaningful and the better move
                # does not lose on back-row discipline or isolation.
                if gap >= 2.0:
                    structure_blocks = (
                        gap < STRUCTURE_EXCEPTION_FLOOR
                        and (
                            (best_isolated and not chosen_isolated) or
                            (best_weakens_back and not chosen_weakens_back)
                        )
                    )
                    if not structure_blocks:
                        debug_info["override_branch_triggered"] = True
                        debug_info["override_branch_name"] = "opening_minimax_guardrail"
                        return best_move, (
                            f"LLM opening choice overridden by minimax guardrail: "
                            f"chosen={chosen_score:.1f}, best={best_score:.1f}, "
                            f"both_opening_safe=true, gap={gap:.1f}."
                        ), debug_info
                    debug_info["override_block_reason"] = "best_uniquely_worse_structure"
                    debug_info["best_move_rejected_reason"] = "best_uniquely_worse_structure"

                # If minimax is near-tied, prefer the move with better development.
                # Development here means center control without worsening isolation/back-row discipline.
                if gap < 2.0:
                    best_dev_better = (
                        (best_center and not chosen_center)
                        and not (best_isolated and not chosen_isolated)
                        and not (best_weakens_back and not chosen_weakens_back)
                    )
                    if best_dev_better and best_score >= chosen_score:
                        debug_info["override_branch_triggered"] = True
                        debug_info["override_branch_name"] = "opening_development_guardrail"
                        return best_move, (
                            f"LLM opening choice overridden by development guardrail: "
                            f"chosen={chosen_score:.1f}, best={best_score:.1f}, "
                            f"best_center=true, chosen_center=false."
                        ), debug_info

            debug_info["best_move_rejected_reason"] = (
                debug_info["best_move_rejected_reason"]
                or "opening_safe_lane_preserved"
            )
            return chosen_move, None, debug_info

        # Protect active king / counterplay moves from shallow override
        chosen_simplification = chosen_facts.get("simplification_value", 0)
        chosen_smokeout = chosen_facts.get("double_corner_smokeout_pressure", 0)
        chosen_edge = chosen_facts.get("edge_confinement_delta", 0)
        chosen_role = chosen_facts.get("quiet_move_role", "")

        # Never protect a king shuffle move
        if chosen_role == "KING_SHUFFLE" or chosen_facts.get("anti_shuffle_penalty", 0) < 0:
            pass  # fall through to override
        elif (
            chosen_safe
            and game_phase == "ENDGAME"
            and (
                chosen_king_activity >= 3
                or chosen_simplification >= 3
                or chosen_smokeout >= 2
                or chosen_edge >= 2
            )
            and gap < 10.0
            and chosen_facts.get("our_pieces_threatened_after", 0)
            <= best_facts.get("our_pieces_threatened_after", 0)
        ):
            debug_info["best_move_rejected_reason"] = "protect_endgame_activity"
            return chosen_move, None, debug_info
        elif (
            chosen_safe
            and "ACTIVATE_KINGS" in priorities
            and chosen_king_activity >= 3
            and chosen_counterplay >= best_counterplay
            and gap < 10.0
        ):
            debug_info["best_move_rejected_reason"] = "protect_king_activation"
            return chosen_move, None, debug_info
        elif (
            chosen_active_counterplay
            and game_phase != "OPENING"
            and losing_mode
            and gap < 12.0
        ):
            debug_info["best_move_rejected_reason"] = "protect_losing_mode_counterplay"
            return chosen_move, None, debug_info

        if chosen_safe:
            debug_info["override_branch_triggered"] = True
            debug_info["override_branch_name"] = "safe_vs_safe_dominance"
            return best_move, (
                f"LLM choice overridden by minimax guardrail ({game_phase}): "
                f"chosen={chosen_score:.1f}, best={best_score:.1f}, "
                f"both_safe=true, gap={gap:.1f} "
                f"(threshold={dominance_threshold:.1f})."
            ), debug_info

    debug_info["best_move_rejected_reason"] = (
        debug_info["best_move_rejected_reason"] or "no_override_condition_met"
    )
    if (
        debug_info.get("best_move_rejected_reason") == "no_override_condition_met"
        and not debug_info.get("override_block_reason")
    ):
        # Encode gate outcomes into a single token (no spaces) so the evaluator
        # can capture it without changing its parser.
        debug_info["override_block_reason"] = (
            "gates|"
            f"LDMD:{low_danger_minimax_dominance_gate};"
            f"OPNLD:{opening_low_danger_minimax_guardrail_gate};"
            f"LDAC:{low_danger_active_minimax_guardrail_gate};"
            f"QSAFE:{quiet_safe_minimax_primary_gate};"
            f"SVDOM:{safe_vs_safe_dominance_gate}"
        )

    # ── Unsafe-vs-unsafe override ────────────────────────────────────────────
    # All existing branches require at least one of {passive_safe, low_danger} to
    # be true on either the chosen or best move.  When BOTH moves are fully unsafe
    # (bs=0, cs=0), every prior branch is skipped regardless of gap size.
    # Three mutually exclusive tiers handle this blind spot:
    #
    # TIER-1 (unsafe_vs_unsafe_fallback_tier1_lower_threat):
    #   _best_threat < _chosen_threat AND gap >= UNSAFE_VS_UNSAFE_LOWER_THREAT_GAP (35)
    #   Best move leaves fewer pieces hanging AND is minimax-better. The combination
    #   of two advantages makes 35pt sufficient. Confirmed T11: gap=42, bt=1, ct=2.
    #
    # MIDGAP (unsafe_vs_unsafe_midgap_minimax):
    #   _best_threat == _chosen_threat AND gap >= UNSAFE_VS_UNSAFE_MIDGAP_GAP (40)
    #   Equal threat exposure on both moves. Override is justified purely by minimax
    #   dominance; no secondary threat advantage. Slightly higher threshold (40 > 35)
    #   because minimax must work alone. Confirmed T13: gap=56, bt=ct=1.
    #
    # TIER-2 (unsafe_vs_unsafe_fallback_tier2_higher_threat):
    #   _best_threat > _chosen_threat AND gap >= UNSAFE_VS_UNSAFE_FALLBACK_GAP (150)
    #   Best move leaves MORE pieces hanging but minimax gap is extreme. Very
    #   conservative threshold because the threat direction is counter-intuitive.
    #
    # Common guard (_both_fully_unsafe): not passive_safe, not low_danger,
    # |score| < 1000 (excludes forced-loss sentinels).
    _best_threat = best_facts.get("our_pieces_threatened_after", 0)
    _chosen_threat = chosen_facts.get("our_pieces_threatened_after", 0)
    _both_fully_unsafe = (
        not chosen_safe
        and not best_safe
        and not _is_low_danger_active(chosen_facts)
        and not _is_low_danger_active(best_facts)
        and abs(best_score) < 1000.0
        and abs(chosen_score) < 1000.0
    )
    if _both_fully_unsafe:
        _tier1  = _best_threat <  _chosen_threat and gap >= UNSAFE_VS_UNSAFE_LOWER_THREAT_GAP
        _midgap = _best_threat == _chosen_threat and gap >= UNSAFE_VS_UNSAFE_MIDGAP_GAP
        _tier2  = _best_threat >  _chosen_threat and gap >= UNSAFE_VS_UNSAFE_FALLBACK_GAP
        if _tier1 or _midgap or _tier2:
            if _tier1:
                _branch_name = "unsafe_vs_unsafe_fallback_tier1_lower_threat"
            elif _midgap:
                _branch_name = "unsafe_vs_unsafe_midgap_minimax"
            else:
                _branch_name = "unsafe_vs_unsafe_fallback_tier2_higher_threat"
            debug_info["override_branch_triggered"] = True
            debug_info["override_branch_name"] = _branch_name
            return best_move, (
                f"Unsafe-vs-unsafe override ({_branch_name}): "
                f"chosen={chosen_score:.1f} (threat={_chosen_threat}), "
                f"best={best_score:.1f} (threat={_best_threat}), "
                f"gap={gap:.1f}."
            ), debug_info


    return chosen_move, None, debug_info

# ── Strategic context formatter ───────────────────────────────────────────────

_PRIORITY_GUIDANCE: dict[str, str] = {
    "RESOLVE_TACTICS":            "a jump exists — prefer highest captures_count first",
    "CONTROL_CENTER": (
        "In OPENING, center_control is a late tiebreak only and must not override "
        "DEVELOP_PIECES, MAINTAIN_BACK_ROW, or safety. "
        "In MIDGAME/ENDGAME, prefer center_control=true only when it also brings "
        "concrete value such as threat creation, mobility reduction, or restriction."), 
    "DEVELOP_PIECES": "prefer moves that improve development without leaving the moved piece isolated",
    "MAINTAIN_BACK_ROW":          "avoid moving back-row pieces unless it gains material",
    "INCREASE_MOBILITY":          "prefer moves that keep our pieces connected; avoid isolated=true",
    "ACTIVATE_KINGS": (
        "prefer safe king moves with higher king_activity_score; among equally safe king moves, "
        "prefer the one that creates threats, has shot_sequence_available=true, reduces opponent mobility, "
        "increases restriction_score, freezes enemy pieces, blocks landing squares, or improves conversion quality; "
        "center control is only a minor tiebreak and should not decide the move by itself"
    ),
    "PROMOTE":                    "prefer near_promotion=true or results_in_king=true moves",
    "TRADE_WHEN_AHEAD":           "prefer jumps (captures_count>0) when net_gain>=0",
    "SIMPLIFY":                   "prefer jumps that reduce total pieces; net_gain>=0 required",
    "AVOID_TRADES":               "avoid jumps where opponent_can_recapture=true",
    "COMPLICATE":                 "prefer moves that create tactical threats; avoid simplifying",
    "CONVERT_ADVANTAGE": (
        "outside the OPENING, prefer higher winning_conversion_score, higher mobility_reduction, lower opponent_mobility_after, "
        "forces_exchange=true, restriction_score > 0, frozen_enemy_pieces > 0, or stronger king_activity_score; "
        "in OPENING, these are supportive only and must not justify a quiet move unless it also has real action "
        "(captures_count>0, creates_immediate_threat=true, shot_sequence_available=true, or blocks_opponent_landing=true)"
    ),
    "HOLD_ADVANTAGE": (
        "prefer safe moves that preserve material, keep our_pieces_threatened_after low, "
        "and avoid unnecessary complications"
    ),
    "CREATE_THREATS": (
        "prefer moves with creates_real_trap=true first; then creates_immediate_threat=true, "
        "shot_sequence_available=true, two_for_one_potential=true, or forces_exchange=true. "
        "Among such moves, prefer lower opponent_safe_reply_count, then higher counterplay_score "
        "and higher two_for_one_score. Use minimax_score to verify which threatening line survives "
        "the opponent reply better. A move that only blocks or restricts but still leaves many "
        "safe opponent replies is not a real trap."
    ),
    "SEEK_COUNTERPLAY": (
    "prefer moves with creates_real_trap=true or low opponent_safe_reply_count. "
    "If a move only blocks one square or raises restriction_score but the opponent still has many safe replies, "
    "it is not real counterplay. mobility_reduction, restriction_score, and frozen_enemy_pieces are supportive only. "
    "Among safe or nearly equally unsafe moves, use minimax_score to verify the best counterplay line."
    ),
    "MAINTAIN_EQUALITY":          "avoid opponent_can_recapture=true; keep net_gain=0",
    "AVOID_SIMPLIFICATION":       "prefer simple moves over jumps when behind in kings",
    "BLOCK_PROMOTION":            "prefer moves where opponent_near_promotion=false after move",
    "REDUCE_OPP_MOBILITY":        "prefer moves that restrict opponent mobility; use center_control only if it also improves real restriction or king pressure",
    "ATTACK_WEAK_PIECES":         "prefer captures_count>0 targeting opponent vulnerable squares",
    "KEEP_BACK_ROW":              "do not move last back-row piece unless it captures a king",
    "CENTRALIZE_KINGS":           "prefer king moves that improve king_activity_score, escape restriction, or pressure; center_control alone is not enough",
    "CONSOLIDATE_PIECES":         "prefer moves that reduce leaves_piece_isolated=true",
    "STRENGTHEN_PIECE_CONNECTIONS": "prefer moves where the piece lands adjacent to a friendly",
    "BREAK_OPPONENT_BACK_ROW":    "prefer moves that threaten promotion into opp back row",
    "PRESS_ADVANTAGE":            "prefer highest net_gain; accept recapture risk if gain>0",
    "PRESSURE_IN_DRAW":           "prefer center_control=true to restrict opponent options",
    "DEFEND_LEFT_FLANK":          "prefer moves on left side columns 0-3",
    "DEFEND_RIGHT_FLANK":         "prefer moves on right side columns 4-7",
    "PLAY_SAFE":                  "prefer moves where opponent_can_recapture=false",
    "CREATE_IMBALANCES":          "prefer moves that lead to asymmetric piece counts",
    "DEFEND_PIECES": (
        "prefer moves where our_pieces_threatened_after=0; "
        "if impossible, prefer moves with lowest our_pieces_threatened_after; "
        "a move where our_pieces_threatened_after < our_pieces_threatened_before "
        "is always better than one that maintains or increases the threat count"
    ),
    "DEFEND": (
        "prefer moves where our_pieces_threatened_after < our_pieces_threatened_before; "
        "also prefer blocks_opponent_landing=true to deny the opponent capture lanes"
    ),
}


def _format_ranker_context(ctx: Optional[dict[str, Any]]) -> str:
    if not ctx:
        return "(no strategic context available)"

    lines: list[str] = []

    phase = ctx.get("game_phase", "UNKNOWN")
    score = ctx.get("winning_score", 0)
    score_state = ctx.get("score_state", "UNKNOWN")
    lines.append(f"game_phase: {phase}  |  winning_score: {score:+d}  |  score_state: {score_state}")

    mat = ctx.get("material_advantage", 0)
    king = ctx.get("king_advantage", 0)
    mob = ctx.get("mobility_advantage", 0)
    center = ctx.get("center_control_advantage", 0)
    lines.append(
        f"material_advantage: {mat:+d}  |  king_advantage: {king:+d}  "
        f"|  mobility_advantage: {mob:+d}  |  center_control_advantage: {center:+d}"
    )

    our_prom = ctx.get("our_promotion_threats", 0)
    opp_prom = ctx.get("opp_promotion_threats", 0)
    lines.append(f"our_promotion_threats: {our_prom}  |  opp_promotion_threats: {opp_prom}")

    our_vuln = ctx.get("our_vulnerable_pieces", 0)
    opp_vuln = ctx.get("opp_vulnerable_pieces", 0)
    lines.append(f"our_vulnerable_pieces: {our_vuln}  |  opp_vulnerable_pieces: {opp_vuln}")

    mat_trend = ctx.get("material_trend")
    mob_trend = ctx.get("mobility_trend")
    center_trend = ctx.get("center_trend")
    if mat_trend is not None:
        lines.append(
            f"trends (last 4 turns): material {mat_trend:+d}  "
            f"mobility {mob_trend:+d}  center {center_trend:+d}"
        )

    patterns = ctx.get("active_patterns", [])
    if patterns:
        lines.append(f"active_patterns: {', '.join(patterns)}")

    stable = ctx.get("position_is_stable", False)
    stagnation = ctx.get("stagnation_detected", False)
    lines.append(f"position_is_stable: {stable}  |  stagnation_detected: {stagnation}")

    priorities = ctx.get("strategic_priorities", [])
    if priorities:
        lines.append("")
        lines.append("strategic_priorities (apply in order — each maps to facts fields):")
        for p in priorities:
            guidance = _PRIORITY_GUIDANCE.get(p, "no specific facts mapping defined")
            lines.append(f"  {p}: {guidance}")

    return "\n".join(lines)


# ── System prompts ────────────────────────────────────────────────────────────

RANKER_SYSTEM_PROMPT = """\
You are the move ranker for American Checkers (8×8). Pick the single best move
from the numbered list and explain it like an experienced checkers coach.

DECISION ALGORITHM — apply these steps internally in strict order.
Do NOT narrate the steps. Use them only to arrive at your choice.

STEP 1 — BOARD SAFETY
  Prefer moves with the lowest our_pieces_threatened_after.
  A move that leaves 0 of our pieces threatened is ideal.
  A move that leaves 2 pieces threatened is worse than one that leaves 1.
  
  This is the most important criterion in equal or winning positions — never let center_control,
  near_promotion, or any positional fact override a lower
  our_pieces_threatened_after value.

  HOWEVER:
  In clearly losing or counterplay-seeking positions, a move that is only slightly less safe
  may still be best if it has clearly better minimax_score and real tactical action
  (creates_immediate_threat=true, shot_sequence_available=true, or strong counterplay_score).

  HARD LOSING-MODE RULE:
  If score_state is CLEARLY_LOSING or SLIGHTLY_LOSING, and two candidate moves differ
  by only 0 or 1 in our_pieces_threatened_after, prefer the move with the better
  minimax_score unless:
    - opponent_can_recapture=true for that move and false for the alternative
    - net_gain is worse
    - or the moved piece is directly threatened while the alternative avoids that

  In losing or counterplay-seeking positions, do NOT override a clearly better minimax_score
  using restriction_score, frozen_enemy_pieces, or generic structural language.
  Do not automatically reject active counterplay just because it leaves 1 more threatened piece.


  CRITICAL:
  - If a simple move has unsafe_simple_move=true, it must score worse than any
    simple move with unsafe_simple_move=false.
  - If all remaining simple moves are unsafe, first minimize
    our_pieces_threatened_after.
  - If two unsafe moves differ by only 0 or 1 in our_pieces_threatened_after,
    first compare danger severity:
      * moved_piece_is_threatened
      * forced_opponent_jump_reply
      * max_opponent_jump_captures
      * opponent_jump_count
    Then use minimax_score BEFORE center_control, creates_immediate_threat,
    leaves_piece_isolated, or other positional bonuses.

  - A move that allows a forced opponent jump reply or a larger immediate jump
    capture is worse than another move with the same threat count unless minimax
    clearly shows the opposite by a meaningful margin.

  - In all-unsafe positions, minimax_score is the PRIMARY decision signal
    after basic safety comparison. A move with minimax_score=-17 must be
    chosen over a move with minimax_score=-23 when our_pieces_threatened_after
    is equal or differs by only 1.

  - In losing positions, apply the same rule even when not all moves are unsafe:
    if two moves differ by only 0 or 1 in our_pieces_threatened_after,
    the move with clearly better minimax_score must be preferred unless the
    worse-minimax move has a clearly better immediate tactical safety fact
    (moved_piece_is_threatened=false, forced_opponent_jump_reply=false,
    or much lower max_opponent_jump_captures).

  - Do not pick the lower-scoring minimax move based on restriction_score,
    frozen_enemy_pieces, vague mobility claims, or generic structure.

  If all moves have the same our_pieces_threatened_after, proceed to Step 2.
    When two moves have similar our_pieces_threatened_after, use these danger
  severity facts BEFORE positional reasoning:

  1. prefer moved_piece_is_threatened=false
  2. prefer forced_opponent_jump_reply=false
  3. prefer lower max_opponent_jump_captures
  4. prefer lower opponent_jump_count

  Example:
  - if two moves both leave 2 pieces threatened, but one move makes the moved
    piece directly capturable or allows a forced jump reply, that move is worse
    even if the raw threat count is equal.

  Do not treat two unsafe moves as equally bad just because
  our_pieces_threatened_after is the same.

  Special case: a jump that captures a piece and still leaves 1 threatened
  may still be better than a simple move that leaves 0 threatened — weigh
  captures_count against the threat count. This exception applies ONLY
  to jumps with captures_count >= 1, never to simple moves.

  CRITICAL: If a simple move has unsafe_simple_move=true, it must score
worse than any simple move with unsafe_simple_move=false. No other fact
— not creates_immediate_threat, not center_control, not near_promotion —
can override this for simple moves. Pressure or positional benefits do not
justify exposing pieces on a non-capturing move.
  
STEP 2 — CAPTURES
  Prefer the highest captures_count.
  If a jump exists among the remaining candidates, a simple move is never best.

STEP 3 — BLOCK OPPONENT THREATS
  Among remaining candidates, prefer moves where blocks_opponent_landing=true.
  This means our piece physically lands on the square the opponent would have
  used to complete a capture — it removes a threat without us having to jump.

MINIMAX DOMINANCE RULE (applies between Steps 3 and 4):
  If all remaining candidate moves are equally safe after applying Steps 1–3
  — meaning they share the same our_pieces_threatened_after value and none is
  disqualified by an immediate safety constraint (opponent_can_recapture,
  moved_piece_is_threatened, or forced_opponent_jump_reply) —
  AND one move has a minimax_score that is better than all other remaining
  moves by 15.0 or more, then that move MUST be chosen immediately.

  In this equal-safety, large-gap case:
    - minimax_score overrides center_control
    - minimax_score overrides quiet_move_role (STRUCTURAL_RESTRICTION,
      KING_ACTIVATION, TACTICAL_PRESSURE, etc.)
    - minimax_score overrides restriction_score and frozen_enemy_pieces
    - minimax_score overrides general strategic-priority preferences from Step 4

  Do NOT proceed to Step 4 if the minimax dominance condition is met.

  This rule applies only when safety is genuinely equal. If moves differ in
  our_pieces_threatened_after, or one has opponent_can_recapture=true while
  another does not, resolve that safety difference first using Step 1 before
  applying this rule.

  When the minimax gap is below 15.0, proceed normally to Step 4.

UNSAFE BEST EXCEPTION (applies alongside the MINIMAX DOMINANCE RULE):
  If the move with the highest minimax_score has opponent_can_recapture=true
  or our_pieces_threatened_after=1 (but NOT >= 2),
  AND its minimax_score exceeds all fully safe candidates by 25.0 or more,
  THEN safety alone is NOT sufficient reason to reject it.
  A 25+ point minimax advantage means the engine already accounts
  for the opponent's best recapture — the net result is still favorable.
  Prefer the higher-minimax move in this case.
  Do NOT apply this exception if our_pieces_threatened_after >= 2.

STEP 4 — STRATEGIC PRIORITIES
  Apply the strategic_priorities list in the order given.
  Each priority maps to specific facts fields — use exactly those mappings.
  Skip any priority that does not distinguish the remaining candidates.

STEP 5 — CONVERSION, COUNTERPLAY, AND TIEBREAKERS
  Prefer highest net_gain.

  If net_gain is equal, rank in this order:
    1. higher minimax_score
    2. higher king_activity_score
    3. better conversion signals (mobility_reduction, lower opponent_mobility_after, winning_conversion_score)
    4. leaves_piece_isolated=false

  Use center_control only as a final tiebreak.
  Do not choose a move mainly because it is central.

  If strategic_priorities includes CONVERT_ADVANTAGE or TRADE_WHEN_AHEAD,
then among safe moves:

  FIRST check if the move has real action:
    - captures_count > 0
    OR creates_immediate_threat = true
    OR shot_sequence_available = true
    OR blocks_opponent_landing = true

  If YES:
    prefer the move that:
      - has higher winning_conversion_score
      - has higher mobility_reduction
      - has lower opponent_mobility_after
      - keeps our_pieces_threatened_after low

  If NO (quiet move with no immediate action):

    In quiet positions, you MUST prefer measurable progress toward winning.
    Do NOT choose a move mainly because it gives generic restriction.

    Among quiet moves, rank in this exact order:
      1. higher minimax_score
      2. higher king_activity_score
      3. higher mobility_reduction OR lower opponent_mobility_after
      4. higher winning_conversion_score
      5. leaves_piece_isolated=false

    restriction_score and frozen_enemy_pieces are SUPPORTING signals only.
    They must NEVER be the main reason to choose a quiet move.

    If minimax_score is significantly worse than another safe move,
    restriction_score and frozen_enemy_pieces MUST be ignored.

    If two quiet moves are similarly safe, prefer the one that:
      - improves activity
      - improves king activation
      - improves mobility
      - or has the better minimax_score

    Do NOT justify a quiet move mainly by "restriction" or "freezing pieces"
    unless it also creates immediate tactical pressure or a real mobility collapse.

  IMPORTANT OPENING RULE:
  In game_phase=OPENING, winning_conversion_score, restriction_score,
  frozen_enemy_pieces, mobility_reduction, and opponent_mobility_after
  are weak secondary signals only.

  In OPENING, these signals matter only when the move also has at least one of:
    - captures_count > 0
    - blocks_opponent_landing=true
    - creates_immediate_threat=true
    - shot_sequence_available=true

  Otherwise they must NOT override cleaner opening structure,
  back-row preservation, or a clearly better safe move.

  Use counterplay_score as a strong tiebreak in favor of active play,
but do not let it override clearly better promotion, capture, or
safety outcomes. Never choose a move with unsafe_simple_move=true
over a safe alternative regardless of counterplay_score.
  

  REAL TRAP RULE:
  Do not treat a move as strong counterplay or tactical pressure merely because:
    - blocks_opponent_landing=true
    - restriction_score > 0
    - frozen_enemy_pieces > 0

  If opponent_safe_reply_count is still high, the move did not create a real trap.
  Prefer a move with creates_real_trap=true or lower opponent_safe_reply_count
  over a move that only gives generic restriction.

  A fake trap is:
    - blocks or restricts one square
    - but still leaves the opponent several comfortable safe replies

  A real trap is:
    - creates_real_trap=true
    - or sharply reduces opponent_safe_reply_count
    - or combines real action with low safe-reply count

If ACTIVATE_KINGS is in strategic_priorities, then among equally safe king moves,
prefer the move with the highest king_activity_score. For kings, center_control
is only a minor tiebreak — it matters only when it comes with concrete benefits
such as creates_immediate_threat=true, shot_sequence_available=true,
mobility_reduction > 0, higher restriction_score, frozen_enemy_pieces > 0,
or stronger conversion quality. Do not choose a king move mainly because it is central.

  
 KING ENDGAME ANTI-SHUFFLE RULE:
  If quiet_move_role == "KING_SHUFFLE" or anti_shuffle_penalty < 0,
  that move must lose to any other safe king move that has a better
  king_activity_score, winning_conversion_score, simplification_value,
  double_corner_smokeout_pressure, or edge_confinement_delta.
  A king move must NOT be selected just because it is safe and central
  when another safe king move creates real pressure or conversion progress.
  In endgame-like positions (few pieces or multiple kings), prefer moves with:
    1. higher king_activity_score
    2. higher winning_conversion_score
    3. higher simplification_value
    4. higher double_corner_smokeout_pressure
    5. higher edge_confinement_delta
  over a passive king move that merely occupies a central square.

  If stagnation_detected=true in strategic_context, break ties by preferring
  moves that improve mobility, reduce opponent mobility, create threats,
  improve king_activity_score, or improve conversion quality. Do not repeat
  a quiet move unless it has a concrete benefit.

  minimax_score is a strong tactical verifier, not an automatic winner.

    OPENING positions (game_phase=OPENING):
    Strategic development still matters, but minimax_score is NOT a weak signal.

    In OPENING:
      - Safety is important, but threat_after=0 is NOT an automatic winner.
      - If two moves are both low-danger (no recapture, moved piece not threatened)
        and differ by only 0 or 1 in our_pieces_threatened_after, minimax_score is
        a strong signal and should usually decide.
      - Strategic development (DEVELOP_PIECES / MAINTAIN_BACK_ROW / CONTROL_CENTER)
        remains important as tiebreak context, not a reason to ignore clearly better minimax.

    CRITICAL OPENING LOW-DANGER RULE:
    If two candidate moves differ by only 0 or 1 in our_pieces_threatened_after,
    and both have:
      - opponent_can_recapture=false
      - moved_piece_is_threatened=false
    then minimax_score must be treated as a strong decision signal.
    Do NOT choose a passive safe structural move over a low-danger active move
    when the active move has clearly better minimax_score.
    threat_after=0 does NOT automatically beat threat_after=1.

    For quiet opening moves:
      - center_control is only a late tiebreak
      - winning_conversion_score is weak
      - restriction_score and frozen_enemy_pieces are weak
      - mobility_reduction is weak

    IMPORTANT:
    In OPENING, quiet_move_role="TACTICAL_PRESSURE" does NOT automatically mean
    the move is worse than a STRUCTURAL_RESTRICTION move.

    If opponent_can_recapture=false, moved_piece_is_threatened=false, and
    our_pieces_threatened_after <= 1, treat that move as structurally acceptable.
    Then compare minimax_score and development before preferring a quieter move.

    If a move has:
  creates_immediate_threat=false
  and shot_sequence_available=false
  and captures_count=0
  and blocks_opponent_landing=false

then it is a quiet non-tactical opening move.

For such a move:
  - winning_conversion_score
  - restriction_score
  - frozen_enemy_pieces
  - mobility_reduction
  - opponent_mobility_after

MUST NOT appear in the reasoning for a quiet opening move unless the move already has a clearly stated structural justification such as center_control=true, leaves_piece_isolated=false, safer back-row discipline, or lower our_pieces_threatened_after.
If they are mentioned, they must be secondary and brief, never the decision basis.
  - cleaner development
  - better structure
  - better back-row discipline
  - safer piece coordination
  - center_control when relevant to opening priorities

If a quiet opening move is chosen, the explanation should focus on development, structure, safety, or center control — not on conversion or freezing enemy pieces.

    Among equally safe quiet opening moves, choose in this strict order:
      1. highest safe minimax_score
      2. stronger practical development:
         - center_control=true
         - better forward development from the starting structure
         - better piece activation without weakening back-row discipline
      3. lower isolation / better structure

    HARD OPENING RULE:
    If two quiet opening moves are both safe (opponent_can_recapture=false),
    and one has a clearly better minimax_score by 3.0 or more, choose the
    higher-minimax move unless it weakens back-row discipline or leaves the
    moved piece isolated when the better-minimax alternative does not.

    In a quiet non-tactical OPENING move, do NOT choose mainly because of:
      - blocks_opponent_landing
      - restriction_score
      - frozen_enemy_pieces
      - winning_conversion_score
      - mobility_reduction
      - opponent_mobility_after

    Those signals may be mentioned only if the move already wins on opening
    development or safety grounds. They must never be the main reason for
    selecting a quiet opening move.

    HARD OPENING SAFETY RULE:
    In OPENING, a move with:
      - opponent_can_recapture=false
      - net_gain >= 0
      - moved_piece_is_threatened=false
      - and our_pieces_threatened_after <= 1

    is still considered OPENING-SAFE for comparison purposes.

    Therefore, if two quiet opening moves are both OPENING-SAFE, and one has
    a better minimax_score by 2.0 or more, prefer the higher-minimax move.

    Do NOT reject an OPENING-SAFE move just because:
      - our_pieces_threatened_after is 1 instead of 0
      - quiet_move_role is TACTICAL_PRESSURE
      - center_control=true
      - creates_immediate_threat=false
      - shot_sequence_available=false

    If a move is OPENING-SAFE and has clearly better minimax_score, do not let
    a quieter STRUCTURAL_RESTRICTION move override it unless that alternative:
      - preserves back-row discipline while the better-minimax move weakens it, or
      - avoids isolation when the better-minimax move leaves the moved piece isolated.

    Use generic structure only after minimax_score and practical development.
    Do not let "cleaner structure" or vague restriction override a clearly better
    OPENING-SAFE developmental move.

  MIDGAME and ENDGAME positions:
    Use minimax_score heavily when safety is similar, especially among
    equally safe moves. Do not let shallow minimax override clearly better
    strategic safety, consolidation, or threat-prevention when the position
    is unstable.

    If score_state indicates losing, or strategic_priorities includes SEEK_COUNTERPLAY,
    COMPLICATE, or CREATE_THREATS, then do NOT automatically reject a move just because
    it leaves 1 more threatened piece than the safest option. In these cases, prefer
    active moves with clearly better minimax_score, creates_immediate_threat=true,
    shot_sequence_available=true, or much higher counterplay_score.

  Apply minimax_score like this:
  - if moves are equally safe, prefer higher minimax_score
  - if moves differ by only 0 or 1 in our_pieces_threatened_after,
    prefer higher minimax_score unless one move has a clearly better
    forced capture or promotion outcome
  - in all-unsafe positions, minimax_score should guide the choice among
    the least bad moves after basic safety comparison

  minimax_score should NOT override:
  - a forced capture with clearly better material outcome
  - a clearly better promotion move (results_in_king=true)
  - a large safety gap (for example, 0 threatened vs 2+ threatened)

  Do NOT treat minimax_score as a last-resort tie only.
  Use it as a strong decision signal after safety comparison,
  but not as an automatic override in every unstable position.

  If still tied after minimax_score, prefer the lower index.
  


FACTS REFERENCE (engine-computed — never recompute these yourself):
  our_pieces_threatened_after  int   — how many of our pieces can be captured
                                        next turn if we make this move (LOWER=BETTER)
  our_pieces_threatened_before int   — how many were threatened before this move
  opponent_can_recapture       bool  — true if our_pieces_threatened_after > 0
  recapturable_piece_is_king   bool  — true if the threatened piece is a king
  moved_piece_is_threatened    bool  — true if the moved piece itself is directly capturable
                                       after this move; worse than leaving only some other piece hanging
  max_opponent_jump_captures   int   — maximum number of our pieces the opponent can capture
                                       in one immediate reply after this move; lower is better
  forced_opponent_jump_reply   bool  — true if all legal opponent replies are jumps;
                                       this means we allowed a tactically forced capture sequence
  blocks_opponent_landing      bool  — true if we land on their jump landing square
  captures_count               int   — pieces we capture this turn
  net_gain                     int   — material change after this move
  results_in_king              bool  — promotes one of our pieces to king
  center_control               bool  — our piece lands in the strategic center
  leaves_piece_isolated        bool  — moved piece has no friendly diagonal neighbor
  any_piece_isolated           bool  — any of our pieces is isolated after the move
  near_promotion               bool  — our piece is one step from promotion
  kings_captured               int   — how many kings we captured
  piece_type_moving            str   — "regular" or "king"
  opponent_jump_count          int   — how many jumps opponent has after our move
  opponent_mobility_before     int   — opponent legal move count before our move
  opponent_mobility_after      int   — opponent legal move count after our move
  mobility_reduction           int   — how much our move reduces opponent mobility
  creates_immediate_threat     bool  — whether our resulting position creates a likely jump threat next turn
  shot_sequence_available      bool  — whether our resulting position already gives us a jump next turn
  forces_exchange              bool  — whether the move tends to force an exchange sequence
  forces_exchange_count        int   — how many forcing jump replies appear in that exchange profile
  two_for_one_potential        bool  — whether the move creates a likely 2-for-1 tactical opportunity
  two_for_one_score            int   — strength of that 2-for-1 tactical opportunity
  restriction_score            int   — how strongly the move restricts opponent structure or mobility
  frozen_enemy_pieces          int   — how many opponent pieces become immobile after the move
  improves_trade_conversion    bool  — whether this safely helps simplify when ahead
  winning_conversion_score     int   — symbolic score for how well this move converts an advantage
  unsafe_simple_move           bool  — true when a simple move leaves any of our pieces threatened;

                                       this is a major warning signal — a simple move with
                                       unsafe_simple_move=true must score worse than any simple
                                       move with unsafe_simple_move=false
  minimax_score                float — shallow search score (depth-limited minimax with alpha-beta)
                                       from current_player's perspective; higher = tactically better;
                                       use as a strong tiebreak among equally safe moves,
                                       especially for ACTIVATE_KINGS, CONVERT_ADVANTAGE,
                                       SEEK_COUNTERPLAY, CREATE_THREATS; do NOT let it override
                                       clearly better immediate safety or forced captures
  counterplay_score            int   — symbolic score for how actively a move creates pressure
                                       when behind; higher = more active; used only when
                                       CREATE_THREATS or SEEK_COUNTERPLAY are priorities;
                                       does NOT override safety — use as a strong tiebreak
                                       among equally safe moves, not as a dominant criterion
  king_activity_score          int   — symbolic score for how actively a king improves pressure,
                                       mobility restriction, or board control; used as a tiebreak
                                       for ACTIVATE_KINGS and endgame king play; higher = better

OUTPUT FORMAT — reply with ONLY this JSON object, no markdown, no extra text:

                                       {
  "chosen_index": <integer, 0-based>,
  "reasoning": "<1-2 sentences as a checkers coach explaining the strategic
                 benefit of this move and why the alternatives were weaker>"
}

REASONING REQUIREMENTS:
  - Never describe the position as MIDGAME, ENDGAME, winning, clearly ahead, or converting an advantage unless that exactly matches strategic_context.game_phase and strategic_context.score_state or material_advantage.
  - In OPENING, if the chosen move has creates_immediate_threat=false, shot_sequence_available=false, captures_count=0, and blocks_opponent_landing=false, do not justify it mainly using winning_conversion_score, restriction_score, frozen_enemy_pieces, mobility_reduction, or opponent_mobility_after.
  - In that quiet OPENING case, explain the move using minimax_score, development, center_control, back-row discipline, or structure only.
  - In OPENING, if the chosen move has our_pieces_threatened_after <= 1, opponent_can_recapture=false, and moved_piece_is_threatened=false, do not describe it as effectively unsafe or tactically inferior just because a different move had threat_after=0.
  - If a better OPENING-SAFE move existed with minimax_score higher by 2.0 or more, do NOT justify a lower-minimax structural move unless you name the exact structural reason:
      back-row discipline or isolation.
  - If a better safe move existed only because of restriction_score, frozen_enemy_pieces, or mobility_reduction, do NOT use those as the reason for choosing it.
  - Mention only facts explicitly present in the chosen move's facts or strategic_context.
  - If safety drove the choice: mention our_pieces_threatened_after value.
  - If danger severity drove the choice: mention moved_piece_is_threatened,
    forced_opponent_jump_reply, max_opponent_jump_captures, or opponent_jump_count.
    - If conversion drove the choice for a quiet move in MIDGAME/ENDGAME:
    mention minimax_score first, then winning_conversion_score,
    mobility_reduction, or opponent_mobility_after as supporting signals.
  - If conversion drove the choice for an active move:
    mention winning_conversion_score, mobility_reduction,
    opponent_mobility_after, or the tactical action that made the move active.
  - If pressure drove the choice: mention creates_immediate_threat=true.
  - If blocking drove the choice: mention blocks_opponent_landing=true.
  - If captures drove the choice: mention captures_count and net_gain.
  - Do not infer hidden ideas like "defends the flank" unless that is directly supported
    by a named priority or fact.
  - Do not say alternatives were weaker unless you name the exact compared fact.
  - Never say "no other moves existed" unless exactly one candidate was provided.
  - Never mention promotion unless results_in_king=true or near_promotion=true.
  - Never mention a priority unless it appears in strategic_priorities.
  - If you cannot justify a comparison exactly, omit it.
  - Do NOT justify a move mainly by restriction_score or frozen_enemy_pieces
    unless the move also creates immediate tactical pressure or mobility collapse.
  - In CLEARLY_LOSING or SLIGHTLY_LOSING positions, if minimax_score was the decisive
    reason, say so explicitly before mentioning any structural or mobility signal.
  - In losing positions, do not explain a lower-minimax move as better unless you name
    the exact immediate danger fact that forced the choice:
    moved_piece_is_threatened, forced_opponent_jump_reply, or max_opponent_jump_captures.
  - If you claim a move creates pressure, trap, or counterplay, mention creates_real_trap
    or opponent_safe_reply_count when available.
  - Do NOT describe a move as a trap mainly because of blocks_opponent_landing,
    restriction_score, or frozen_enemy_pieces if opponent_safe_reply_count remains high."""

RANKER_SYSTEM_PROMPT_SINGLE = """\
You are the move ranker for American Checkers (8×8).
Exactly ONE legal candidate is available — list index 0. Output chosen_index: 0.

Analyse the move using these facts (in order of importance):
  1. our_pieces_threatened_after — how many of our pieces remain at risk (lower is better)
  2. moved_piece_is_threatened   — whether the moved piece itself walks into direct danger
  3. forced_opponent_jump_reply  — whether opponent is tactically forced to capture
  4. max_opponent_jump_captures  — severity of the opponent's best immediate jump reply
  5. blocks_opponent_landing     — does this move physically block an opponent capture?
  6. captures_count / net_gain   — material gained
  7. results_in_king             — promotion achieved
  8. mobility_reduction          — whether the move reduces opponent options
  9. creates_immediate_threat    — whether the move creates pressure next turn
  10. center_control             — positional value

  
STRICT OPENING RULE (MANDATORY)

If a move has:
  creates_immediate_threat = false
  AND shot_sequence_available = false
  AND captures_count = 0
  AND blocks_opponent_landing = false

Then:

- This is a quiet non-tactical move.

- The following signals MUST NOT be used as justification:
    restriction_score
    frozen_enemy_pieces
    winning_conversion_score
    mobility_reduction
    opponent_mobility_after

- These signals MUST NOT appear in the reasoning as a reason to choose the move.

- If such signals are mentioned, they must be ignored in the decision.

Instead, the move can ONLY be justified by:
  - development (DEVELOP_PIECES)
  - center_control
  - structural integrity (no isolation)
  - back-row safety
  - overall safety

If none of these structural reasons are clearly better than alternatives,
then prefer the move with better minimax_score.

OUTPUT FORMAT — reply with ONLY this JSON object, no markdown, no extra text:
{
  "chosen_index": 0,
  "reasoning": "<1-2 sentences: what this move achieves strategically,
                 citing the most significant facts by name and value>"
}
"""


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_ranker_user_prompt(
    state: CheckersState,
    filtered: list[dict[str, Any]],
    index_map: list[int],
) -> str:
    lines: list[str] = [
        f"current_player: {_current_player_label(state.current_player)}",
        f"turn_number: {state.turn_number}",
        "",
        f"Choose exactly one index from 0 to {len(filtered) - 1} inclusive.",
        "",
        "legal_moves:",
        "Safety note: clearly unsafe moves may be removed before ranking,",
        "but if all legal moves are unsafe then the candidate list may still",
        "contain recapturable moves. Use the facts fields to judge safety.",]
    phase = (state.strategic_context or {}).get("game_phase", "MIDGAME")

    for i, move in enumerate(filtered):
        facts = dict(move.get("facts", {}) or {})

        is_quiet_opening_move = (
            phase == "OPENING"
            and facts.get("captures_count", 0) == 0
            and not facts.get("creates_immediate_threat", False)
            and not facts.get("shot_sequence_available", False)
            and not facts.get("blocks_opponent_landing", False)
        )

        if is_quiet_opening_move:
            for key in (
                "restriction_score",
                "frozen_enemy_pieces",
                "winning_conversion_score",
                "mobility_reduction",
                "opponent_mobility_after",
            ):
                facts.pop(key, None)

        payload = {
            "type": move.get("type"),
            "path": move.get("path"),
            "captured": move.get("captured", []),
            "facts": facts,
        }
        lines.append(f"  [{i}] {json.dumps(payload, ensure_ascii=False)}")

    if RANKER_INCLUDE_STRATEGIC_CONTEXT:
        lines.extend([
            "",
            "strategic_context:",
            _format_ranker_context(state.strategic_context),
        ])
    else:
        lines.append("")
        lines.append("(strategic_context omitted — rely on per-move facts only.)")



    return "\n".join(lines)


def build_ranker_user_prompt_single(
    state: CheckersState,
    move: dict[str, Any],
) -> str:
    payload = {
        "type": move.get("type"),
        "path": move.get("path"),
        "captured": move.get("captured", []),
        "facts": move.get("facts", {}),
    }
    recapture = move.get("facts", {}).get("opponent_can_recapture", False)
    safety_note = (
        "Note: this is a forced move — opponent can recapture, "
        "but it is the only legal option."
        if recapture else
        "This move is safe — no immediate recapture threat."
    )
    lines: list[str] = [
        f"current_player: {_current_player_label(state.current_player)}",
        f"turn_number: {state.turn_number}",
        "",
        "Exactly ONE legal candidate — index 0. Output chosen_index: 0.",
        safety_note,
        "",
        "legal_move [0]:",
        f"  {json.dumps(payload, ensure_ascii=False)}",
    ]
    if RANKER_INCLUDE_STRATEGIC_CONTEXT:
        lines.extend([
            "",
            "strategic_context:",
            _format_ranker_context(state.strategic_context),
        ])
    else:
        lines.append("")
        lines.append("(strategic_context omitted — rely on per-move facts only.)")

    return "\n".join(lines)
# ── Mistral API call ──────────────────────────────────────────────────────────

def call_mistral_ranker(system: str, user: str) -> str:
    """
    Calls mistral-small-latest via the Mistral REST API.
    Uses response_format: json_object for guaranteed JSON output —
    no regex fallback needed for the outer structure.

    Raises:
        ValueError  — API key missing, non-200 response, or missing content
        OSError     — network-level failure
    """
    if not MISTRAL_API_KEY:
        raise ValueError(
            "MISTRAL_API_KEY is not set. "
            "Run: export MISTRAL_API_KEY='your_key_from_console.mistral.ai'"
        )

    payload: dict[str, Any] = {
        "model": MISTRAL_RANKER_MODEL,
        "temperature": RANKER_TEMPERATURE,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},   # native JSON mode
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
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
            import time
            time.sleep(15)   # wait before retry
        raise ValueError(
            f"Mistral API HTTP {e.code}: {body_text[:300]}"
        ) from e

    # Mistral response shape:
    # {"choices": [{"message": {"content": "..."}}], ...}
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(
            f"Unexpected Mistral response structure: {str(data)[:300]}"
        ) from e

    if not isinstance(content, str):
        raise ValueError(f"Mistral content is not a string: {type(content)}")

    return content



# ── Unified call dispatcher ───────────────────────────────────────────────────

def call_ranker(system: str, user: str) -> str:
    return call_mistral_ranker(system, user)


# ── Response interpretation ───────────────────────────────────────────────────

def _interpret_ranker_response(raw: str, n: int) -> tuple[Optional[int], str]:
    parsed = _parse_ranker_json(raw)
    raw_idx: Optional[int] = None
    reasoning = ""

    if parsed:
        raw_idx = _extract_chosen_index(parsed)
        r = parsed.get("reasoning")
        if isinstance(r, str):
            reasoning = r.strip()
        elif r is not None:
            reasoning = str(r).strip()

    if raw_idx is None:
        raw_idx = _regex_extract_chosen_index(raw)
    if not reasoning:
        reasoning = _regex_extract_reasoning(raw) or ""

    idx = _resolve_ranker_index(raw_idx, n)
    return idx, reasoning


def _failure_patch(state: CheckersState) -> dict[str, Any]:
    return {
        "chosen_move": None,
        "last_move_reasoning": None,
        "ranker_retry_count": state.ranker_retry_count + 1,
        "ranker_failure_count": state.ranker_failure_count + 1,
        "last_completed_node": "ranker_agent",
    }


# ── Failure diagnostics / fallback helpers ───────────────────────────────────

def _candidate_signal_snapshot(m: dict[str, Any]) -> dict[str, Any]:
    facts = m.get("facts", {}) or {}
    return {
        "path": m.get("path"),
        "type": m.get("type"),
        "minimax_score": _get_minimax_score(m),
        "creates_immediate_threat": facts.get("creates_immediate_threat"),
        "shot_sequence_available": facts.get("shot_sequence_available"),
        "quiet_move_role": facts.get("quiet_move_role"),
        "counterplay_score": facts.get("counterplay_score"),
        "winning_conversion_score": facts.get("winning_conversion_score"),
    }


def _canonical_coord_list(value: Any) -> list[list[int]]:
    """
    Canonical JSON-safe coordinate list: list[list[int]].
    Best-effort normalization for tuple/list or numeric-string coordinates.
    """
    if not isinstance(value, (list, tuple)):
        return []
    out: list[list[int]] = []
    for step in value:
        if not isinstance(step, (list, tuple)) or len(step) < 2:
            continue
        try:
            out.append([int(step[0]), int(step[1])])
        except (TypeError, ValueError):
            continue
    return out


def _build_ranker_filtered_menu_snapshot(filtered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": m.get("type"),
            "path": _canonical_coord_list(m.get("path")),
            "captured": _canonical_coord_list(m.get("captured", [])),
            "facts": {"minimax_score": (m.get("facts", {}) or {}).get("minimax_score")},
        }
        for m in filtered
    ]


def _parse_diagnostics(raw: Optional[str], n_filtered: int) -> dict[str, Any]:
    if raw is None:
        return {
            "raw_llm_output_present": False,
            "raw_llm_output_excerpt": None,
            "parsed_chosen_index_raw": None,
            "parsed_chosen_index_resolved": None,
        }
    parsed = _parse_ranker_json(raw)
    raw_idx: Optional[int] = _extract_chosen_index(parsed) if parsed else None
    if raw_idx is None:
        raw_idx = _regex_extract_chosen_index(raw)
    return {
        "raw_llm_output_present": True,
        "raw_llm_output_excerpt": raw[:1200],
        "parsed_chosen_index_raw": raw_idx,
        "parsed_chosen_index_resolved": _resolve_ranker_index(raw_idx, n_filtered),
    }


def _choose_best_minimax_with_origin(
    legal: list[dict[str, Any]],
    filtered: list[dict[str, Any]],
    index_map: list[int],
) -> tuple[dict[str, Any], int, str]:
    """
    Deterministic fallback:
    1) best minimax in filtered candidates, if available
    2) else best minimax in legal list
    """
    if filtered:
        best_idx, _, _ = _best_and_second_best_minimax(filtered)
        if best_idx is not None and 0 <= best_idx < len(filtered):
            original_idx = index_map[best_idx] if 0 <= best_idx < len(index_map) else best_idx
            original_idx = max(0, min(original_idx, len(legal) - 1))
            return legal[original_idx], original_idx, "filtered_best_minimax"
    best_legal_idx, _, _ = _best_and_second_best_minimax(legal)
    if best_legal_idx is None:
        # legal is guaranteed non-empty at call sites, but keep safe fallback.
        return legal[0], 0, "legal_index_0_fallback"
    return legal[best_legal_idx], best_legal_idx, "legal_best_minimax"


def _log_no_chosen_move_failure(
    *,
    state: CheckersState,
    legal: list[dict[str, Any]],
    filtered: list[dict[str, Any]],
    raw: Optional[str],
    reason: str,
    fallback_path: str,
    fallback_original_idx: Optional[int] = None,
    override_path: Optional[str] = None,
) -> None:
    parse = _parse_diagnostics(raw, len(filtered))
    payload: dict[str, Any] = {
        "event": "ranker_no_chosen_move_hardened",
        "reason": reason,
        "turn_number": state.turn_number,
        "current_player": _current_player_label(state.current_player),
        "legal_move_count": len(legal),
        "filtered_candidate_count": len(filtered),
        "fallback_path": fallback_path,
        "fallback_original_idx": fallback_original_idx,
        "override_path": override_path,
        "filtered_candidate_signals": [
            _candidate_signal_snapshot(m) for m in filtered[:5]
        ],
    }
    payload.update(parse)
    print(f"[RANKER_FAILURE_DEBUG] {json.dumps(payload, ensure_ascii=False)}")


# ── Override audit helpers ───────────────────────────────────────────────────

OVERRIDE_MAX_RETRIES = int(os.environ.get("OVERRIDE_MAX_RETRIES", "3"))


def _audit_override(
    candidate_moves: list[dict[str, Any]],
    idx_in_candidates: int,
    game_phase: str,
    score_state: str,
    priorities: list[str],
    comparison_moves: list[dict[str, Any]],
) -> tuple[bool, Optional[str], Optional[dict[str, Any]], Optional[str], dict[str, Any]]:
    """
    Thin wrapper around _override_if_llm_chose_much_worse_minimax.
    Returns (triggered, branch_name, best_move, override_reason, override_debug).
    Does NOT apply the override — caller decides what to do.
    """
    result_move, override_reason, override_debug = _override_if_llm_chose_much_worse_minimax(
        candidate_moves,
        idx_in_candidates,
        game_phase,
        score_state,
        priorities,
        comparison_moves=comparison_moves,
    )
    triggered: bool = bool(override_debug.get("override_branch_triggered", False))
    branch_name: Optional[str] = override_debug.get("override_branch_name")
    # best_move: the move the override would choose (from comparison_moves)
    best_idx, _, _ = _best_and_second_best_minimax(comparison_moves)
    best_move = comparison_moves[best_idx] if best_idx is not None else None
    return triggered, branch_name, best_move, override_reason, override_debug


def _build_override_feedback_str(
    override_debug: dict[str, Any],
    branch_name: Optional[str],
    chosen_score: Optional[float] = None,
    best_score: Optional[float] = None,
) -> str:
    """
    Builds a branch-aware OVERRIDE_FEEDBACK block injected BEFORE the move list.
    Each branch explains specifically why the previous LLM reasoning failed.
    Does NOT reveal the exact index to choose.
    """
    gap = override_debug.get("best_vs_chosen_minimax_gap")
    gap_str = f"{gap:.1f}" if isinstance(gap, (int, float)) else "unknown"
    threat_delta = override_debug.get("best_vs_chosen_threat_delta")
    threat_str = f"{threat_delta:+d}" if isinstance(threat_delta, int) else "n/a"
    c_str = f"{chosen_score:.1f}" if isinstance(chosen_score, (int, float)) else "n/a"
    b_str = f"{best_score:.1f}" if isinstance(best_score, (int, float)) else "n/a"
    bn = branch_name or "unknown"

    if bn == "safe_vs_unsafe_large_gap":
        body = (
            f"  branch_triggered    : {bn}\n"
            f"  your_choice_score   : {c_str}\n"
            f"  best_available_score: {b_str}\n"
            f"  minimax_gap         : {gap_str}  (threshold={SAFE_VS_UNSAFE_OVERRIDE_GAP:.1f})\n"
            f"  threat_delta        : {threat_str}  (best.threat_after - your_choice.threat_after)\n"
            "\n"
            "DIAGNOSIS: You chose a lower-threat move, but the minimax engine evaluated\n"
            "all opponent replies and found the higher-scoring move is tactically dominant.\n"
            f"A {gap_str}-point minimax gap means the engine already priced in the\n"
            "opponent recapture risk. opponent_can_recapture=True does not mean the\n"
            "position becomes losing — minimax proves the best move recovers well after.\n"
            "Choosing the safe-but-weaker move permanently loses that positional value.\n"
            "\n"
            "ACTION: Re-rank all candidates by minimax_score. A move with threat_after=1\n"
            "or opponent_can_recapture=True is still correct when its minimax_score is\n"
            "significantly higher than the safe alternative."
        )
    elif bn == "low_danger_minimax_dominance":
        body = (
            f"  branch_triggered    : {bn}\n"
            f"  your_choice_score   : {c_str}\n"
            f"  best_available_score: {b_str}\n"
            f"  minimax_gap         : {gap_str}  (threshold={LOW_DANGER_MINIMAX_GAP:.1f})\n"
            f"  threat_delta        : {threat_str}\n"
            "\n"
            "DIAGNOSIS: Both your choice and the best move are low-danger. You likely\n"
            "over-weighted counterplay_score, creates_immediate_threat, or\n"
            "shot_sequence_available. These fields describe the current move only —\n"
            "they do NOT guarantee a better position after the opponent's best reply.\n"
            "A high counterplay_score with a lower minimax_score means the tactical\n"
            "pressure evaporates once the opponent responds optimally.\n"
            "\n"
            "ACTION: Among low-danger moves, treat minimax_score as the dominant\n"
            "criterion. Counterplay and threat signals are valid tiebreakers only when\n"
            f"minimax scores are within {LOW_DANGER_MINIMAX_GAP:.1f} points of each other."
        )
    elif bn and bn.startswith("unsafe_vs_unsafe"):
        body = (
            f"  branch_triggered    : {bn}\n"
            f"  your_choice_score   : {c_str}\n"
            f"  best_available_score: {b_str}\n"
            f"  minimax_gap         : {gap_str}\n"
            f"  threat_delta        : {threat_str}\n"
            "\n"
            "DIAGNOSIS: All available moves expose pieces. In this situation\n"
            "our_pieces_threatened_after and opponent_can_recapture are not useful\n"
            "discriminators — both moves carry similar immediate risk. The minimax\n"
            "engine evaluated the full sequence and found one move recovers\n"
            "significantly better after the opponent's optimal reply.\n"
            "\n"
            "ACTION: When all moves are risky, rank by minimax_score alone. Do not\n"
            "use threat_after or opponent_can_recapture to prefer one unsafe move."
        )
    else:
        body = (
            f"  branch_triggered    : {bn}\n"
            f"  your_choice_score   : {c_str}\n"
            f"  best_available_score: {b_str}\n"
            f"  minimax_gap         : {gap_str}\n"
            f"  threat_delta        : {threat_str}\n"
            "\n"
            "DIAGNOSIS: The minimax engine found a significantly better move. Minimax\n"
            "evaluates all opponent replies to depth and is more reliable than\n"
            "single-move heuristics (counterplay, safety, threat creation alone).\n"
            "\n"
            "ACTION: Re-examine the candidate list. Prefer the move with the highest\n"
            "minimax_score that does not introduce unacceptable immediate danger."
        )

    lines = [
        "OVERRIDE_FEEDBACK:",
        "Your previous selection was rejected by the tactical override guardrail.",
        "",
        body,
        "",
        "The candidate list below is the FULL proposal shortlist (no safety filter applied).",
        "END_OVERRIDE_FEEDBACK",
    ]
    return "\n".join(lines)


def _build_retry_user_prompt(
    state: "CheckersState",
    move_list: list[dict[str, Any]],
    index_map: list[int],
    feedback_str: str,
    system_prompt: str,
) -> str:
    """
    Builds a retry user prompt that places feedback_str BEFORE the move list.
    move_list is the full proposal shortlist (legal[]); index_map is identity.
    The structure mirrors build_ranker_user_prompt but inserts the feedback
    block between the header and the 'legal_moves:' section.
    """
    phase = (state.strategic_context or {}).get("game_phase", "MIDGAME")
    header_lines = [
        f"current_player: {_current_player_label(state.current_player)}",
        f"turn_number: {state.turn_number}",
        "",
        feedback_str,
        "",
        f"Choose exactly one index from 0 to {len(move_list) - 1} inclusive.",
        "",
        "legal_moves:",
        "Safety note: clearly unsafe moves may be removed before ranking,",
        "but if all legal moves are unsafe then the candidate list may still",
        "contain recapturable moves. Use the facts fields to judge safety.",
    ]
    move_lines: list[str] = []
    for i, move in enumerate(move_list):
        facts = dict(move.get("facts", {}) or {})
        is_quiet_opening_move = (
            phase == "OPENING"
            and facts.get("captures_count", 0) == 0
            and not facts.get("creates_immediate_threat", False)
            and not facts.get("shot_sequence_available", False)
            and not facts.get("blocks_opponent_landing", False)
        )
        if is_quiet_opening_move:
            for key in (
                "restriction_score",
                "frozen_enemy_pieces",
                "winning_conversion_score",
                "mobility_reduction",
                "opponent_mobility_after",
            ):
                facts.pop(key, None)
        payload = {
            "type": move.get("type"),
            "path": move.get("path"),
            "captured": move.get("captured", []),
            "facts": facts,
        }
        move_lines.append(f"  [{i}] {json.dumps(payload, ensure_ascii=False)}")
    ctx_lines: list[str] = []
    if RANKER_INCLUDE_STRATEGIC_CONTEXT:
        ctx_lines = [
            "",
            "strategic_context:",
            _format_ranker_context(state.strategic_context),
        ]
    else:
        ctx_lines = ["", "(strategic_context omitted — rely on per-move facts only.)"]
    return "\n".join(header_lines + move_lines + ctx_lines)


# ── Main node ─────────────────────────────────────────────────────────────────

def ranker_agent(state: CheckersState) -> dict:
    legal = state.legal_moves
    if not legal:
        return _failure_patch(state)

    n = len(legal)

    if n == 1:
        ranker_filtered_menu_snapshot = _build_ranker_filtered_menu_snapshot(legal)
        system = RANKER_SYSTEM_PROMPT_SINGLE
        user = build_ranker_user_prompt_single(state, legal[0])
        import time
        raw = None
        for attempt in range(3):
            try:
                raw = call_ranker(system, user)
                break
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as e:
                wait = 2 ** attempt * 10   # 10s, 20s ,40s
                print(f"[ranker_agent] call failed (attempt {attempt+1}): {e} — waiting {wait}s")
                time.sleep(wait)
        if raw is None:
            chosen = legal[0]
            reasoning = "Fallback: ranker call failed; choosing only legal candidate."
            _log_no_chosen_move_failure(
                state=state,
                legal=legal,
                filtered=legal,
                raw=None,
                reason="llm_call_failed_after_retries_single",
                fallback_path="single_candidate_only_legal",
                fallback_original_idx=0,
                override_path=None,
            )
            return {
                "chosen_move": chosen,
                "last_move_reasoning": reasoning,
                "ranker_retry_count": state.ranker_retry_count + 1,
                "ranker_failure_count": state.ranker_failure_count + 1,
                "ranker_filtered_menu": ranker_filtered_menu_snapshot,
                "last_completed_node": "ranker_agent",
            }
        _, reasoning = _interpret_ranker_response(raw, 1)
        chosen = legal[0]

    else:
        ctx = state.strategic_context or {}
        game_phase = ctx.get("game_phase", "MIDGAME")
        score_state = ctx.get("score_state", "EQUAL")
        priorities = ctx.get("strategic_priorities", [])

        filtered, index_map = _apply_safety_filter(
            legal,
            strategic_priorities=priorities,
            score_state=score_state,
        )
        ranker_filtered_menu_snapshot = _build_ranker_filtered_menu_snapshot(filtered)
        all_unsafe = len(filtered) == len(legal) and all(
            m.get("facts", {}).get("opponent_can_recapture", False)
            for m in legal
        )

        system = RANKER_SYSTEM_PROMPT
        user = build_ranker_user_prompt(state, filtered, index_map)

        import time
        raw = None
        for attempt in range(3):
            try:
                raw = call_ranker(system, user)
                break
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as e:
                wait = 2 ** attempt * 10   # 10s, 20s, 40s
                print(f"[ranker_agent] call failed (attempt {attempt+1}): {e} — waiting {wait}s")
                time.sleep(wait)

        if raw is None:
            chosen, original_idx, fallback_path = _choose_best_minimax_with_origin(
                legal, filtered, index_map
            )
            reasoning = (
                "Fallback: ranker call failed after retries; selected deterministic "
                f"{fallback_path}."
            )
            _log_no_chosen_move_failure(
                state=state,
                legal=legal,
                filtered=filtered,
                raw=None,
                reason="llm_call_failed_after_retries",
                fallback_path=fallback_path,
                fallback_original_idx=original_idx,
                override_path=None,
            )
            return {
                "chosen_move": chosen,
                "last_move_reasoning": reasoning,
                "ranker_retry_count": state.ranker_retry_count + 1,
                "ranker_failure_count": state.ranker_failure_count + 1,
                "ranker_filtered_menu": ranker_filtered_menu_snapshot,
                "last_completed_node": "ranker_agent",
            }

        idx, reasoning = _interpret_ranker_response(raw, len(filtered))
        if idx is None:
            chosen, original_idx, fallback_path = _choose_best_minimax_with_origin(
                legal, filtered, index_map
            )
            reasoning = (
                "Fallback: ranker output had no valid chosen_index; selected deterministic "
                f"{fallback_path}."
            )
            _log_no_chosen_move_failure(
                state=state,
                legal=legal,
                filtered=filtered,
                raw=raw,
                reason="parsed_index_invalid_or_missing",
                fallback_path=fallback_path,
                fallback_original_idx=original_idx,
                override_path=None,
            )
            return {
                "chosen_move": chosen,
                "last_move_reasoning": reasoning,
                "ranker_retry_count": state.ranker_retry_count + 1,
                "ranker_failure_count": state.ranker_failure_count + 1,
                "ranker_filtered_menu": ranker_filtered_menu_snapshot,
                "last_completed_node": "ranker_agent",
            }

        # ── Post-LLM override: audit → retry loop → Python fallback ─────────
        #
        # Variables that are NEVER mutated after initial assignment:
        #   filtered    — safety-filter output (the menu the LLM saw on attempt 0)
        #   legal       — full proposal shortlist (state.legal_moves)
        #   index_map   — filtered→legal index translation
        #   idx         — raw parsed index from the first LLM call (filtered space)
        #
        # Working variables updated each retry iteration:
        #   attempt_moves      — filtered on attempt 0; legal on retries
        #   idx_in_attempt_moves — correct index into attempt_moves for this iteration
        #   idx_legal          — same choice expressed in legal[] space

        # Diagnostic accumulators
        _or_retry_attempts:   int           = 0
        _or_retry_resolved:   bool          = False
        _or_fallback_applied: bool          = False
        _or_branch_name:      Optional[str] = None
        _or_retry_full:       bool          = False
        _last_override_debug: dict[str, Any] = {}
        _last_override_reason: Optional[str] = None
        _retry_reasoning:     Optional[str] = None   # text from most-recent retry LLM call

        # Attempt-0 setup
        attempt_moves        = filtered                    # NEVER overwrite filtered
        idx_in_attempt_moves = idx                         # in filtered space
        idx_legal            = index_map[idx]              # same move in legal space

        _chosen_final: Optional[dict[str, Any]] = None

        while True:
            # ── Audit ────────────────────────────────────────────────────────
            # comparison_moves=legal is ALWAYS the full proposal shortlist.
            # candidate_moves=attempt_moves and idx_in_attempt_moves are
            # iteration-local — never bleed back into filtered or idx.
            (
                _triggered,
                _branch,
                _best_move,
                _override_reason,
                _override_debug,
            ) = _audit_override(
                candidate_moves   = attempt_moves,
                idx_in_candidates = idx_in_attempt_moves,
                game_phase        = game_phase,
                score_state       = score_state,
                priorities        = priorities,
                comparison_moves  = legal,
            )
            _last_override_debug  = _override_debug
            _last_override_reason = _override_reason

            if not _triggered:
                # Audit passed — accept this iteration's choice.
                _chosen_final = attempt_moves[idx_in_attempt_moves]
                if _or_retry_attempts > 0:
                    _or_retry_resolved = True
                break

            # Record branch on first trigger
            if _or_branch_name is None:
                _or_branch_name = _branch

            # ── C2: Log audit result for retry iterations (attempt >= 1) ──────
            if _or_retry_attempts > 0:
                print(
                    f"[override_retry] attempt={_or_retry_attempts}  "
                    f"audit_still_triggered=True  "
                    f"branch={_branch}  "
                    f"gap={_override_debug.get('best_vs_chosen_minimax_gap')}"
                )

            if _or_retry_attempts >= OVERRIDE_MAX_RETRIES:
                # ── C3: Fallback exit log ─────────────────────────────────────
                print(
                    f"[override_retry] FALLBACK  "
                    f"original_branch={_or_branch_name}  "
                    f"final_audit_branch={_branch}  "
                    f"attempts={_or_retry_attempts}  "
                    f"fallback_path={(_best_move or {}).get('path')}"
                )
                # All retries exhausted — Python fallback (best_move from last audit).
                _chosen_final = _best_move if _best_move is not None else attempt_moves[idx_in_attempt_moves]
                _or_fallback_applied = True
                _last_override_reason = _override_reason
                break

            # ── Build and fire retry call ─────────────────────────────────
            _or_retry_attempts += 1
            _or_retry_full = True

            _feedback_str  = _build_override_feedback_str(
                _override_debug,
                _branch,
                chosen_score=_get_minimax_score(attempt_moves[idx_in_attempt_moves]),
                best_score=_get_minimax_score(_best_move) if _best_move is not None else None,
            )
            _retry_user    = _build_retry_user_prompt(
                state       = state,
                move_list   = legal,                     # full proposal shortlist
                index_map   = list(range(len(legal))),  # identity map
                feedback_str = _feedback_str,
                system_prompt = system,
            )
            import time as _time
            _retry_raw: Optional[str] = None
            for _api_attempt in range(3):
                try:
                    _retry_raw = call_ranker(system, _retry_user)
                    break
                except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as _e:
                    _wait = 2 ** _api_attempt * 10
                    print(
                        f"[ranker_agent][override_retry] API call failed "
                        f"(attempt {_api_attempt+1}): {_e} — waiting {_wait}s"
                    )
                    _time.sleep(_wait)

            if _retry_raw is None:
                # API totally failed on this retry slot — burn slot, continue
                # (will fall back if _or_retry_attempts reaches OVERRIDE_MAX_RETRIES)
                continue

            _retry_idx_legal, _retry_reasoning = _interpret_ranker_response(_retry_raw, len(legal))

            if _retry_idx_legal is None or not (0 <= _retry_idx_legal < len(legal)):
                # Bad parse — burn retry slot
                print(
                    f"[ranker_agent][override_retry] retry {_or_retry_attempts}: "
                    f"bad parse or out-of-range index ({_retry_idx_legal}) — burning slot"
                )
                continue

            # ── C1: Per-retry chosen move log ─────────────────────────────
            _retry_chosen_move = legal[_retry_idx_legal]
            print(
                f"[override_retry] attempt={_or_retry_attempts}  "
                f"chosen_idx={_retry_idx_legal}  "
                f"chosen_path={_retry_chosen_move.get('path')}  "
                f"chosen_minimax={_get_minimax_score(_retry_chosen_move):.1f}  "
                f"reasoning_prefix=\"{(_retry_reasoning or '')[:80]}\""
            )

            # ── Set up next audit iteration ───────────────────────────────
            # Retries always audit against the full proposal list.
            # Do NOT overwrite attempt_moves with filtered or legal:
            # re-assign local working variables only.
            attempt_moves        = legal             # full proposal — not filtered
            idx_in_attempt_moves = _retry_idx_legal  # index into legal[] space
            idx_legal            = _retry_idx_legal  # keep aligned
            # Loop → audit runs again with attempt_moves=legal

        # attempt_moves and idx_in_attempt_moves resolved; _chosen_final is set.
        # filtered, legal, index_map, idx are all still their original values.

        # ── Emit DECISION_DEBUG (unchanged structure) ─────────────────────
        _override_debug  = _last_override_debug
        override_reason  = _last_override_reason
        if _or_retry_resolved and _retry_reasoning:
            # Retry LLM chose a corrected move — use its own reasoning.
            reasoning = _retry_reasoning
        elif override_reason and not _or_retry_resolved:
            # Override triggered but retries exhausted without resolving — use override message.
            reasoning = override_reason

        best_idx, _, _ = _best_and_second_best_minimax(legal)
        best_move = legal[best_idx] if best_idx is not None else None
        chosen_before_override = filtered[idx]          # original LLM choice (unchanged)
        legal_scores = [_get_minimax_score(m) for m in legal]
        legal_best_score = max(legal_scores) if legal_scores else None
        legal_best_idxs = (
            [i for i, s in enumerate(legal_scores)
             if legal_best_score is not None and s == legal_best_score]
            if legal_scores else []
        )
        best_idx_is_argmax = (
            best_idx is not None
            and legal_best_score is not None
            and _get_minimax_score(legal[best_idx]) == legal_best_score
        )
        chosen_path_internal = (_chosen_final or {}).get("path")
        best_path_internal = best_move.get("path") if best_move else None
        chosen_path_matches_llm_idx = (
            idx is not None
            and 0 <= idx < len(filtered)
            and filtered[idx].get("path") == chosen_before_override.get("path")
        )
        chosen_debug_facts = (_chosen_final or {}).get("facts", {}) or {}
        best_debug_facts = (best_move.get("facts", {}) if best_move else {}) or {}
        print(
            "[DECISION_DEBUG] "
            f"chosen={(_chosen_final or {}).get('path')} "
            f"best={(best_move.get('path') if best_move else None)} "
            f"gap={_override_debug.get('best_vs_chosen_minimax_gap')} "
            f"llm_idx={idx} "
            f"best_idx={best_idx} "
            f"filtered_menu_size={len(filtered)} "
            f"chosen_minimax_internal={_get_minimax_score(_chosen_final) if _chosen_final else None} "
            f"best_minimax_internal={_get_minimax_score(best_move) if best_move else None} "
            f"chosen_path_internal={chosen_path_internal} "
            f"best_path_internal={best_path_internal} "
            f"chosen_path_matches_llm_idx={chosen_path_matches_llm_idx} "
            f"best_idx_is_argmax={best_idx_is_argmax} "
            f"best_score_tie_count={len(legal_best_idxs)} "
            f"chosen_passive_safe={_override_debug.get('is_passive_safe_structural', {}).get('chosen')} "
            f"best_passive_safe={_override_debug.get('is_passive_safe_structural', {}).get('best')} "
            f"chosen_low_danger={_override_debug.get('is_low_danger_active', {}).get('chosen')} "
            f"best_low_danger={_override_debug.get('is_low_danger_active', {}).get('best')} "
            f"override_triggered={_override_debug.get('override_branch_triggered')} "
            f"override_branch_name={_override_debug.get('override_branch_name')} "
            f"override_block_reason={_override_debug.get('override_block_reason')} "
            f"best_move_rejected_reason={_override_debug.get('best_move_rejected_reason')} "
            f"threat_delta={_override_debug.get('best_vs_chosen_threat_delta')} "
            f"override_retry_attempts={_or_retry_attempts} "
            f"override_retry_resolved={_or_retry_resolved} "
            f"override_fallback_applied={_or_fallback_applied} "
            f"retry_used_full_proposal={_or_retry_full}"
        )
        print(
            "[DECISION_DEBUG] "
            f"chosen_move_facts={{"
            f"'path': {(_chosen_final or {}).get('path')}, "
            f"'is_passive_safe_structural': {_override_debug.get('is_passive_safe_structural', {}).get('chosen')}, "
            f"'is_low_danger_active': {_override_debug.get('is_low_danger_active', {}).get('chosen')}, "
            f"'threat_after': {chosen_debug_facts.get('our_pieces_threatened_after')}, "
            f"'minimax_score': {_get_minimax_score(_chosen_final) if _chosen_final else None}"
            f"}} "
            f"best_move_facts={{"
            f"'path': {(best_move.get('path') if best_move else None)}, "
            f"'is_passive_safe_structural': {_override_debug.get('is_passive_safe_structural', {}).get('best')}, "
            f"'is_low_danger_active': {_override_debug.get('is_low_danger_active', {}).get('best')}, "
            f"'threat_after': {best_debug_facts.get('our_pieces_threatened_after')}, "
            f"'minimax_score': {_get_minimax_score(best_move) if best_move else None}"
            f"}} "
            f"llm_choice_path={chosen_before_override.get('path')}"
        )

        chosen = _chosen_final

        # ── Promotion tie-break ───────────────────────────────────────────────
        # When scores are tied or near-tied and the chosen move is NOT an
        # actual promotion, but a real promotion exists in the full legal set,
        # prefer the promotion deterministically.
        #
        # Conditions (all must hold):
        #   1. chosen move does NOT actually crown a King
        #   2. a legal move with results_in_king=True exists (engine fact only)
        #   3. promotion_score >= chosen_score - PROMOTION_TIEBREAK_MARGIN
        #
        # Explicitly does NOT fire if the promotion is clearly worse.
        PROMOTION_TIEBREAK_MARGIN = float(
            os.environ.get("PROMOTION_TIEBREAK_MARGIN", "3.0")
        )
        chosen_facts_tb = chosen.get("facts", {}) or {}
        chosen_actually_promotes = chosen_facts_tb.get("results_in_king", False)
        if not chosen_actually_promotes:
            chosen_score_tb = _get_minimax_score(chosen)
            promo_candidate = None
            promo_score_tb = float("-inf")
            for _m in legal:
                _mf = _m.get("facts", {}) or {}
                if _mf.get("results_in_king", False):
                    _ms = _get_minimax_score(_m)
                    if _ms >= chosen_score_tb - PROMOTION_TIEBREAK_MARGIN:
                        if _ms > promo_score_tb:
                            promo_score_tb = _ms
                            promo_candidate = _m
            if promo_candidate is not None:
                print(
                    f"[TIE_BREAK][PROMOTION] "
                    f"chosen={chosen.get('path')} "
                    f"replacement={promo_candidate.get('path')} "
                    f"chosen_score={chosen_score_tb:.1f} "
                    f"promotion_score={promo_score_tb:.1f} "
                    f"gap={promo_score_tb - chosen_score_tb:.1f} "
                    f"reason=results_in_king_near_tie"
                )
                chosen = promo_candidate
                if not reasoning or reasoning.startswith("Fallback"):
                    reasoning = (
                        f"Promotion tie-break: chose {promo_candidate.get('path')} "
                        f"(score={promo_score_tb:.1f}) over "
                        f"{chosen.get('path')} (score={chosen_score_tb:.1f}) — "
                        f"immediate King creation preferred within tie window."
                    )
        # ── End promotion tie-break ───────────────────────────────────────────

        if all_unsafe and reasoning:
            reasoning = f"[All moves expose a piece] {reasoning}"

    # ── Fallback reasoning (model returned empty string) ──────────────
    if not reasoning:
        facts = chosen.get("facts") or {}
        path = chosen.get("path", [])
        dest = path[-1] if path else "unknown"
        captures = facts.get("captures_count", 0)
        net = facts.get("net_gain", 0)
        recapture = facts.get("opponent_can_recapture", False)
        center = facts.get("center_control", False)
        promotes = facts.get("results_in_king", False)
        creates_threat = facts.get("creates_immediate_threat", False)

        if captures and captures > 0:
            safety = "with no immediate recapture risk" if not recapture else "though opponent may recapture"
            reasoning = (
                f"Captures {captures} piece(s) for a net gain of {net:+d}, "
                f"landing on {dest} {safety}."
            )
        elif promotes:
            reasoning = (
                f"Advances to {dest} achieving promotion to king — "
                f"significant structural gain."
            )
        elif creates_threat:
            two_for_one = facts.get("two_for_one_potential", False)
            forces_exchange = facts.get("forces_exchange", False)
            restriction_score = facts.get("restriction_score", 0)
            extra = []
            if two_for_one:
                extra.append("creates 2-for-1 pressure")
            if forces_exchange:
                extra.append("forces exchanges")
            if restriction_score > 0:
                extra.append(f"adds supporting restriction ({restriction_score})")
            extra_text = ""
            if extra:
                extra_text = " and " + ", ".join(extra)
            reasoning = (
                f"Moves to {dest} safely and creates immediate tactical pressure"
                f"{extra_text} for the next turn."
            )
        elif center:
            reasoning = (
                f"Moves to {dest}, controlling the center safely "
                f"and maintaining piece development."
            )
        else:
            reasoning = (
                f"Advances to {dest} with no recapture risk, "
                f"maintaining piece connectivity and structure."
            )

    reasoning = re.sub(r"\s+", " ", reasoning).strip()
    if len(reasoning) > 400:
        reasoning = reasoning[:397] + "..."

    # ── Phase 8: thesis instrumentation ──────────────────────────────────────
    # Compare chosen path to the symbolic engine's top-1 path.
    llm_agreed: bool | None = None
    if state.symbolic_best_move is not None:
        sym_path = state.symbolic_best_move.get("path")
        chosen_path = chosen.get("path")
        if sym_path is not None and chosen_path is not None:
            llm_agreed = (sym_path == chosen_path)

    # ── Build structured override diagnostics ────────────────────────────────
    # These variables are set inside the multi-candidate branch (n >= 2).
    # For the single-candidate branch (n == 1) they default to None/False.
    _ranker_diagnostics: dict[str, Any] = {
        "override_retry_attempts":   locals().get("_or_retry_attempts",   0),
        "override_retry_resolved":   locals().get("_or_retry_resolved",   False),
        "override_fallback_applied": locals().get("_or_fallback_applied", False),
        "override_branch_name":      locals().get("_or_branch_name",      None),
        "retry_used_full_proposal":  locals().get("_or_retry_full",       False),
    }

    out = {
        "chosen_move": chosen,
        "last_move_reasoning": reasoning,
        "ranker_retry_count": 0,
        "last_completed_node": "ranker_agent",
        "ranker_filtered_menu": ranker_filtered_menu_snapshot,
        "llm_agreed_with_symbolic_best": llm_agreed,
        "ranker_diagnostics": _ranker_diagnostics,
    }
    return out
