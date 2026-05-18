"""
checkers/baseline_eval/reasoning_checker.py

Strengthened reasoning truthfulness checker for baseline_eval diagnostics.

Wraps ranker_agent._check_reasoning_truthfulness (confirmed pure function,
safe to import at 0.20 s) and adds baseline-eval-specific extensions:

  1. Full ranker-agent check suite — fact-dependent contradiction checks
     (mobility, recapture, isolation, center, capture/material, promotion,
     block-landing, forbidden vocabulary, numeric pattern detection).
     Called with seeds=None since baseline-eval LLMs never receive seeds.

     IMPORTANT: the ranker's context-forbidden-vocabulary check fires on any
     word that wasn't in the seeds.  With seeds=None this over-fires on normal
     checkers language ("diagonal", "traps", "escape") that the model may have
     used because those words were literally in its system/user prompt.
     We filter those warnings: if the flagged word appears in the prompts the
     model received, the warning is suppressed.

  2. Reverse-capture check: model claims "no capture / quiet / simple move"
     but the chosen move type is "jump" (captures_count > 0).

  3. Baseline-specific forbidden metric names: minimax_score, symbolic_rank,
     score_gap, counterplay_score, etc. — flagged for non-full_system baselines
     because those metrics are never exposed to those LLMs.

  4. Wrong-path reference (conservative): reasoning explicitly states a
     full bracket-notation path [[r,c],[r,c]] that matches a DIFFERENT legal
     move's path, not the chosen move's path.

reasoning_check_applicable flag
---------------------------------
Set to False when no facts are available from the scoring oracle (empty facts
dict).  In that case fact-dependent checks are skipped; only vocabulary and
metric-name checks run.  Callers should not treat "no reasoning warnings"
as "reasoning is correct" when reasoning_check_applicable=False.

Public API
----------
check_reasoning(
    reasoning, chosen_move, legal_all, scored, baseline,
    system_prompt="", user_prompt=""
) -> dict

Returned dict keys:
  reasoning_hallucinations       : list[str]  — all detected warnings
  reasoning_hallucination_count  : int        — len(warnings)
  reasoning_truthfulness_passed  : bool       — True when list is empty
  reasoning_check_applicable     : bool       — False when facts unavailable
"""

from __future__ import annotations

import re
from typing import Any, Optional

# Safe import: pure function, no side effects, no network calls.
from checkers.agents.ranker_agent import _check_reasoning_truthfulness

# ── Baseline-specific forbidden metric names ──────────────────────────────────
# Never exposed to any baseline-eval LLM prompt.
# full_system is exempt (its ranker prompt explicitly includes these terms).
_BASELINE_EXTRA_FORBIDDEN: list[str] = [
    "minimax_score",
    "minimax score",
    "symbolic_rank",
    "symbolic rank",
    "score_gap",
    "counterplay_score",
    "counterplay score",
    "winning_conversion_score",
    "conversion_score",
    "mobility_reduction",
    "shot_sequence_available",
    "near_promotion",           # internal fact field name
    "our_pieces_threatened",    # internal fact field name
    "opponent_mobility_after",  # internal fact field name
]

# ── Phrases indicating "no capture" or "simple / quiet" move ─────────────────
_NO_CAPTURE_PHRASES: list[str] = [
    "no capture",
    "without capturing",
    "doesn't capture",
    "does not capture",
    "non-capturing move",
    "quiet move",
    "simple move",
    "no piece taken",
    "no piece removed",
    "no pieces captured",
]

# ── Pattern to find "term 'X' used but not in seeds" warnings ────────────────
_CONTEXT_FORBIDDEN_RE = re.compile(r"term '([^']+)' used but not in seeds")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _norm_path(path: Any) -> list:
    if not isinstance(path, (list, tuple)):
        return []
    out: list = []
    for sq in path:
        if isinstance(sq, (list, tuple)) and len(sq) == 2:
            try:
                out.append([int(sq[0]), int(sq[1])])
            except (TypeError, ValueError):
                return []
        else:
            return []
    return out


def _enrich_facts(chosen_move: dict, scored: list[dict]) -> dict:
    """
    Return full move-facts dict from the scoring oracle if available.
    Falls back to facts embedded directly in chosen_move (full_system case).
    """
    if not chosen_move:
        return {}
    chosen_path = _norm_path(chosen_move.get("path"))
    for m in scored or []:
        if _norm_path(m.get("path")) == chosen_path:
            return m.get("facts") or {}
    # Fallback: full_system attaches facts directly to chosen_move
    return chosen_move.get("facts") or {}


def _path_tuples(move: dict) -> frozenset[tuple[int, int]]:
    return frozenset(
        (int(sq[0]), int(sq[1]))
        for sq in (move.get("path") or [])
        if isinstance(sq, (list, tuple)) and len(sq) == 2
    )


def _filter_prompt_vocab_warnings(
    warnings: list[str],
    prompt_text: str,
) -> list[str]:
    """
    Remove context-forbidden-vocabulary warnings where the flagged word is
    present in the system or user prompt the model received.

    Rationale: the ranker's context-forbidden check requires that any flagged
    word must have appeared in the seeds.  With seeds=None the check fires
    on ALL such words.  But if the word was in the model's prompt, the model
    could have picked it up from there — flagging it as a hallucination is
    a false positive.

    Only filters "term 'X' used but not in seeds" warnings.
    All other warnings are passed through unchanged.
    """
    if not prompt_text:
        return warnings
    filtered: list[str] = []
    for w in warnings:
        m = _CONTEXT_FORBIDDEN_RE.search(w)
        if m:
            word = m.group(1).lower()
            if word in prompt_text:
                continue  # word came from the prompt — not a hallucination
        filtered.append(w)
    return filtered


# ── Main public function ──────────────────────────────────────────────────────

def check_reasoning(
    reasoning: str,
    chosen_move: Optional[dict],
    legal_all: list[dict],
    scored: list[dict],
    baseline: str = "",
    system_prompt: str = "",
    user_prompt: str = "",
) -> dict[str, Any]:
    """
    Run all reasoning truthfulness checks against the chosen move's facts.

    Parameters
    ----------
    reasoning     : LLM-generated reasoning text for the chosen move.
    chosen_move   : The move dict as matched in legal_all (or from ranker).
    legal_all     : Full list of legal moves for the position.
    scored        : Enriched scored-moves list from score_all_legal_moves.
                    Used to retrieve move facts for checking.  May be empty.
    baseline      : Baseline name; controls metric-ban exemptions.
    system_prompt : Exact system prompt the model received.  Used to filter
                    false-positive vocabulary warnings.
    user_prompt   : Exact user prompt the model received.

    Returns
    -------
    dict with keys:
      reasoning_hallucinations       list[str]
      reasoning_hallucination_count  int
      reasoning_truthfulness_passed  bool
      reasoning_check_applicable     bool  — False when no facts available
    """
    if not reasoning or not chosen_move:
        return {
            "reasoning_hallucinations":      [],
            "reasoning_hallucination_count": 0,
            "reasoning_truthfulness_passed": True,
            "reasoning_check_applicable":    False,
        }

    facts = _enrich_facts(chosen_move, scored)
    facts_available = bool(facts)   # False → only vocabulary checks run

    text         = reasoning.lower()
    prompt_text  = (system_prompt + " " + user_prompt).lower()
    warnings: list[str] = []

    # ── 1. Full ranker-agent checker ──────────────────────────────────────────
    # seeds=None: context-forbidden check will fire on any word not in seeds.
    # We post-filter to remove false positives from prompt vocabulary.
    ranker_warnings = _check_reasoning_truthfulness(reasoning, facts, seeds=None)
    ranker_warnings = _filter_prompt_vocab_warnings(ranker_warnings, prompt_text)
    warnings.extend(ranker_warnings)

    # ── 2. Reverse-capture check ──────────────────────────────────────────────
    # The ranker checker catches "claims capture but captures_count=0".
    # This adds the reverse: "claims no capture but the move IS a jump".
    is_jump   = chosen_move.get("type") == "jump"
    cap_count = facts.get("captures_count", 0) if facts else 0
    if is_jump or cap_count > 0:
        if any(p in text for p in _NO_CAPTURE_PHRASES):
            warnings.append(
                f"REASONING_CONTRADICTION: claims no capture / quiet move but "
                f"chosen move type='{chosen_move.get('type')}' "
                f"(captures_count={cap_count})"
            )

    # ── 3. Baseline-specific forbidden metric names ───────────────────────────
    # full_system legitimately exposes these terms — exempt.
    if baseline and baseline != "full_system":
        for phrase in _BASELINE_EXTRA_FORBIDDEN:
            if phrase in text:
                warnings.append(
                    f"REASONING_CONTRADICTION: internal metric '{phrase}' mentioned "
                    f"but never exposed to {baseline!r} — unsupported reference"
                )

    # ── 4. Wrong-path reference (conservative) ────────────────────────────────
    # Flag only if reasoning contains an explicit full bracket path [[r,c],...]
    # that matches a DIFFERENT legal move's path, not the chosen move's path.
    chosen_path_tuples = _path_tuples(chosen_move)
    if chosen_path_tuples and legal_all:
        other_paths: dict[frozenset, dict] = {
            _path_tuples(lm): lm
            for lm in legal_all
            if _path_tuples(lm) != chosen_path_tuples
        }
        full_path_pattern = re.compile(
            r'\[\s*\[(\d+)\s*,\s*(\d+)\]\s*(?:,\s*\[\d+\s*,\s*\d+\]\s*)+\]'
        )
        for match in full_path_pattern.finditer(reasoning):
            coords = re.findall(r'\[(\d+)\s*,\s*(\d+)\]', match.group(0))
            coord_set = frozenset((int(r), int(c)) for r, c in coords)
            if coord_set in other_paths:
                other = other_paths[coord_set]
                warnings.append(
                    f"REASONING_CONTRADICTION: reasoning references path "
                    f"{other.get('path')} which belongs to a different legal move, "
                    f"not the chosen path {chosen_move.get('path')}"
                )

    return {
        "reasoning_hallucinations":      warnings,
        "reasoning_hallucination_count": len(warnings),
        "reasoning_truthfulness_passed": len(warnings) == 0,
        "reasoning_check_applicable":    facts_available,
    }
