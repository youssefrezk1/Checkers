# checkers/agents/comparative_reasoner.py
#
# Comparative-reasoning helpers for the ranker pipeline.
#
# CURRENT SCOPE (Step 1 of the Comparative Reasoning v2 roadmap):
#   Pure deterministic functions that cluster alternative moves by theme
#   and emit natural-language grouped seed strings. NO LLM calls. NO
#   mutation of inputs. NO I/O. ZERO runtime wiring — none of these
#   helpers are invoked from any production code path yet.
#
# LOCKED THEME TAXONOMY (do not extend without a roadmap revision):
#
#   AGGRESSIVE   alt.creates_immediate_threat=True
#                  OR alt.shot_sequence_available=True
#   MATERIAL     alt.captures_count > 0
#   DEFENSIVE    alt.opponent_can_recapture=False
#                  AND alt.our_pieces_threatened_after
#                       <= chosen.our_pieces_threatened_after
#   STRUCTURAL   alt.leaves_piece_isolated differs from chosen,
#                  OR alt.weakens_king_row differs from chosen
#   PROMOTION    alt.results_in_king=True OR alt.near_promotion=True
#   MOBILITY     alt.opponent_mobility_after < alt.opponent_mobility_before
#
# Multiple themes per alternative are permitted: the same alternative may
# appear in several theme groups. Empty groups are dropped from the
# returned dict. The chosen move itself is excluded from every group by
# path-equality comparison.
#
# OUTPUT STYLE (natural grouped prose, not robotic per-alt enumeration):
#
#   multi-member group:
#     "Aggressive alternatives [1], [3], and [5] create immediate threats
#      but allow recapture; chosen move forfeits initiative for safety."
#
#   single-member group (singular phrasing):
#     "Defensive alternative [2] avoids recapture but does not capture;
#      chosen move accepts exposure for material gain."
#
#   tradeoff seed (one per turn, compact bracket format):
#     "Chosen move tradeoff: forfeits aggressive options [1,3,5] in
#      favour of recapture safety."
#
# OUT OF SCOPE FOR STEP 1 (do not add here — see roadmap Steps 2-9):
#   - LLM prompts (system or user)
#   - A comparative verifier
#   - Refinement orchestration
#   - Integration into explainer_agent or _explain_chosen_move
#   - Telemetry fields under `comparative_*`
#   - Provider adapters
#
# Determinism: same inputs always produce the same outputs (same dict-key
# order, same list ordering inside groups, same seed strings). No LLM,
# no randomness, no I/O.

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from checkers.agents.llm_provider import call_mistral_once, ProviderHTTPError

__all__ = [
    "THEME_TAGS",
    "_cluster_alternatives_by_theme",
    "build_comparative_group_seeds",
    "build_comparative_tradeoff_seed",
    "EXPLAINER_COMPARATIVE_SYSTEM",
    "EXPLAINER_COMPARATIVE_REFINEMENT_SYSTEM",
    "build_comparative_user_prompt",
    "build_comparative_refinement_user_prompt",
    "ComparativeContradiction",
    "verify_comparative_reasoning",
    "sanitize_comparative_contradiction",
    # Step 4
    "RefinementCandidate",
    "_evaluate_refinement_candidate",
    "refine_comparative_reasoning",
    # Step 5
    "_COMPARATIVE_DIAGNOSTICS_KEYS",
    "generate_comparative_reasoning",
]


# Exact set of flat keys that generate_comparative_reasoning writes into
# diagnostics_out.  Tests use this tuple to assert completeness and absence
# of extraneous keys.
_COMPARATIVE_DIAGNOSTICS_KEYS: tuple[str, ...] = (
    "comparative_was_skipped",
    "comparative_skip_reason",
    "comparative_paragraph_text",
    "comparative_seeds",
    "comparative_groups",
    "comparative_generation_samples_used",
    "comparative_sample_contradiction_counts",
    "comparative_generation_short_circuited",
    "comparative_initial_contradictions",
    "comparative_final_contradictions",
    "comparative_refinement_attempts",
    "comparative_provider",
)


# Locked closed vocabulary. The Python order here is the canonical
# emission order: build_comparative_group_seeds iterates THEME_TAGS and
# emits one seed per non-empty group.
THEME_TAGS: tuple[str, ...] = (
    "AGGRESSIVE",
    "MATERIAL",
    "DEFENSIVE",
    "STRUCTURAL",
    "PROMOTION",
    "MOBILITY",
)


# ── Clustering ──────────────────────────────────────────────────────────────

def _cluster_alternatives_by_theme(
    all_candidates: list[dict],
    chosen_move: dict,
) -> dict[str, list[tuple[int, dict]]]:
    """
    Deterministically partition the alternatives in ``all_candidates`` into
    theme groups defined by the locked taxonomy above.

    Returns a dict keyed by theme tag (one of ``THEME_TAGS``). Each value
    is a list of ``(index, candidate)`` tuples, where ``index`` is the
    position of the alternative in ``all_candidates``. Members within a
    group are sorted in ascending index order; the dict's key insertion
    order follows ``THEME_TAGS``. Empty groups are omitted entirely.

    The chosen move is excluded from every group by ``path`` equality
    comparison. Multiple themes per alternative are permitted.

    Pure function. Never mutates inputs. Deterministic.
    """
    chosen_path = chosen_move.get("path")
    chosen_facts = chosen_move.get("facts") or {}

    chosen_pta = chosen_facts.get("our_pieces_threatened_after")
    chosen_isolated = chosen_facts.get("leaves_piece_isolated")
    chosen_weakens_back = chosen_facts.get("weakens_king_row")

    buckets: dict[str, list[tuple[int, dict]]] = {tag: [] for tag in THEME_TAGS}

    for idx, alt in enumerate(all_candidates):
        if alt.get("path") == chosen_path:
            continue
        facts = alt.get("facts") or {}

        # AGGRESSIVE
        if (
            facts.get("creates_immediate_threat") is True
            or facts.get("shot_sequence_available") is True
        ):
            buckets["AGGRESSIVE"].append((idx, alt))

        # MATERIAL
        cap = facts.get("captures_count", 0)
        if isinstance(cap, (int, float)) and cap > 0:
            buckets["MATERIAL"].append((idx, alt))

        # DEFENSIVE
        alt_pta = facts.get("our_pieces_threatened_after")
        if (
            facts.get("opponent_can_recapture") is False
            and isinstance(alt_pta, (int, float))
            and isinstance(chosen_pta, (int, float))
            and alt_pta <= chosen_pta
        ):
            buckets["DEFENSIVE"].append((idx, alt))

        # STRUCTURAL
        alt_isolated = facts.get("leaves_piece_isolated")
        alt_weakens_back = facts.get("weakens_king_row")
        differs = False
        if (
            isinstance(alt_isolated, bool)
            and isinstance(chosen_isolated, bool)
            and alt_isolated != chosen_isolated
        ):
            differs = True
        if (
            isinstance(alt_weakens_back, bool)
            and isinstance(chosen_weakens_back, bool)
            and alt_weakens_back != chosen_weakens_back
        ):
            differs = True
        if differs:
            buckets["STRUCTURAL"].append((idx, alt))

        # PROMOTION
        if (
            facts.get("results_in_king") is True
            or facts.get("near_promotion") is True
        ):
            buckets["PROMOTION"].append((idx, alt))

        # MOBILITY
        a_mb = facts.get("opponent_mobility_before")
        a_ma = facts.get("opponent_mobility_after")
        if (
            isinstance(a_mb, (int, float))
            and isinstance(a_ma, (int, float))
            and a_ma < a_mb
        ):
            buckets["MOBILITY"].append((idx, alt))

    # Drop empty groups; preserve THEME_TAGS order in the result.
    # The explicit sort is a safety net — enumerate already yields ascending
    # indices, but a future caller could pre-filter the candidate list out
    # of order, so we normalise here.
    result: dict[str, list[tuple[int, dict]]] = {}
    for tag in THEME_TAGS:
        members = sorted(buckets[tag], key=lambda im: im[0])
        if members:
            result[tag] = members
    return result


# ── Index-list rendering ────────────────────────────────────────────────────

def _format_index_list_natural(indices: list[int]) -> str:
    """
    Render indices for grouped seeds in natural English conjunction form:
      []          -> ""
      [1]         -> "[1]"
      [1, 2]      -> "[1] and [2]"
      [1, 3, 5]   -> "[1], [3], and [5]"
    """
    if not indices:
        return ""
    if len(indices) == 1:
        return f"[{indices[0]}]"
    if len(indices) == 2:
        return f"[{indices[0]}] and [{indices[1]}]"
    head = ", ".join(f"[{i}]" for i in indices[:-1])
    return f"{head}, and [{indices[-1]}]"


def _format_index_list_compact(indices: list[int]) -> str:
    """
    Render indices for the tradeoff seed in compact bracket form:
      [1, 3, 5]  -> "[1,3,5]"
    Matches the roadmap example verbatim.
    """
    return "[" + ",".join(str(i) for i in indices) + "]"


# ── Group-seed builder ──────────────────────────────────────────────────────

def build_comparative_group_seeds(
    groups: dict[str, list[tuple[int, dict]]],
    chosen_facts: dict,
) -> list[str]:
    """
    Render each non-empty theme group as a single natural-language seed
    string. Groups are emitted in ``THEME_TAGS`` order; alternatives within
    a group are listed in ascending index order.

    Single-member groups fold to singular phrasing ("alternative [2]"
    rather than "alternatives [2]"). Multi-member groups use plural
    phrasing with natural conjunctions.

    Each seed ends with a short chosen-move contrast clause when the
    contrast is unambiguous; otherwise the contrast is omitted (the prose
    still reads naturally because the theme itself is named).

    Pure function. Never mutates inputs. Deterministic.
    """
    seeds: list[str] = []
    cf = chosen_facts or {}

    for tag in THEME_TAGS:
        members = groups.get(tag) or []
        if not members:
            continue

        idxs = [i for i, _ in members]
        single = len(members) == 1
        article = "alternative" if single else "alternatives"
        idx_phrase = _format_index_list_natural(idxs)

        if tag == "AGGRESSIVE":
            verb = "creates" if single else "create"
            verb_allow = "allows" if single else "allow"
            all_allow_recapture = all(
                (m[1].get("facts") or {}).get("opponent_can_recapture") is True
                for m in members
            )
            drawback = (
                f" but {verb_allow} recapture"
                if all_allow_recapture else ""
            )
            chosen_aggressive = (
                cf.get("creates_immediate_threat") is True
                or cf.get("shot_sequence_available") is True
            )
            chosen_safer = cf.get("opponent_can_recapture") is False
            if not chosen_aggressive and chosen_safer:
                contrast = "; chosen move forfeits initiative for safety"
            elif not chosen_aggressive:
                contrast = "; chosen move does not create an immediate threat"
            else:
                contrast = ""
            seeds.append(
                f"Aggressive {article} {idx_phrase} {verb} immediate threats"
                f"{drawback}{contrast}."
            )

        elif tag == "MATERIAL":
            verb = "wins" if single else "win"
            chosen_caps = cf.get("captures_count", 0)
            chosen_captures = (
                isinstance(chosen_caps, (int, float)) and chosen_caps > 0
            )
            contrast = (
                "; chosen move does not capture"
                if not chosen_captures else ""
            )
            seeds.append(
                f"Material {article} {idx_phrase} {verb} material{contrast}."
            )

        elif tag == "DEFENSIVE":
            verb_avoid = "avoids" if single else "avoid"
            no_captures = all(
                ((m[1].get("facts") or {}).get("captures_count", 0) == 0)
                for m in members
            )
            verb_capture = "capture" if single else "capture"
            chosen_recapture = cf.get("opponent_can_recapture") is True
            chosen_caps = cf.get("captures_count", 0)
            chosen_captures = (
                isinstance(chosen_caps, (int, float)) and chosen_caps > 0
            )
            # "but do not capture" is only a meaningful contrast when the chosen
            # move itself captures; omit it when chosen also has captures_count=0.
            drawback = (
                f" but {'does' if single else 'do'} not {verb_capture}"
                if no_captures and chosen_captures else ""
            )
            if chosen_recapture and chosen_captures:
                contrast = (
                    "; chosen move accepts exposure for material gain"
                )
            elif chosen_recapture:
                contrast = "; chosen move accepts recapture risk"
            else:
                contrast = ""
            seeds.append(
                f"Defensive {article} {idx_phrase} {verb_avoid} recapture"
                f"{drawback}{contrast}."
            )

        elif tag == "STRUCTURAL":
            # Describe a uniform contrast when all members agree on the
            # isolation/king-row axis; otherwise emit a generic descriptor.
            first_facts = members[0][1].get("facts") or {}
            firsts_iso = first_facts.get("leaves_piece_isolated")
            firsts_back = first_facts.get("weakens_king_row")

            all_iso_match = all(
                (m[1].get("facts") or {}).get("leaves_piece_isolated") == firsts_iso
                for m in members
            )
            all_back_match = all(
                (m[1].get("facts") or {}).get("weakens_king_row") == firsts_back
                for m in members
            )

            phrases: list[str] = []
            if all_iso_match and isinstance(firsts_iso, bool):
                if firsts_iso is False and cf.get("leaves_piece_isolated") is True:
                    phrases.append(
                        "preserve piece coordination where the chosen move "
                        "leaves the moved piece isolated"
                    )
                elif firsts_iso is True and cf.get("leaves_piece_isolated") is False:
                    phrases.append(
                        "isolate the moved piece where the chosen move keeps "
                        "it connected"
                    )
            if all_back_match and isinstance(firsts_back, bool):
                if firsts_back is False and cf.get("weakens_king_row") is True:
                    phrases.append(
                        "preserve back-row defence where the chosen move "
                        "weakens it"
                    )
                elif firsts_back is True and cf.get("weakens_king_row") is False:
                    phrases.append(
                        "weaken the back row where the chosen move "
                        "preserves it"
                    )

            if not phrases:
                phrases.append("differ structurally from the chosen move")

            descriptor = " and ".join(phrases)
            seeds.append(
                f"Structural {article} {idx_phrase} {descriptor}."
            )

        elif tag == "PROMOTION":
            verb = "advances" if single else "advance"
            chosen_promotes = (
                cf.get("results_in_king") is True
                or cf.get("near_promotion") is True
            )
            contrast = (
                "; chosen move does not promote or near-promote"
                if not chosen_promotes else ""
            )
            seeds.append(
                f"Promotion {article} {idx_phrase} {verb} toward king status"
                f"{contrast}."
            )

        elif tag == "MOBILITY":
            verb = "reduces" if single else "reduce"
            chosen_mb = cf.get("opponent_mobility_before")
            chosen_ma = cf.get("opponent_mobility_after")
            chosen_reduces_mob = (
                isinstance(chosen_mb, (int, float))
                and isinstance(chosen_ma, (int, float))
                and chosen_ma < chosen_mb
            )
            contrast = (
                "; chosen move does not restrict the opponent"
                if not chosen_reduces_mob else ""
            )
            seeds.append(
                f"Mobility-restricting {article} {idx_phrase} {verb} "
                f"opponent mobility{contrast}."
            )

    return seeds


# ── Tradeoff seed ───────────────────────────────────────────────────────────

def build_comparative_tradeoff_seed(
    chosen_facts: dict,
    groups: dict[str, list[tuple[int, dict]]],
) -> Optional[str]:
    """
    Emit a single high-level tradeoff seed framing the chosen move as
    forfeiting one comparative theme in favour of another.

    Returns ``None`` when no meaningful tradeoff can be identified (e.g.,
    no groups exist, or the chosen move dominates the alternatives on
    every dimension covered by the heuristic).

    Priority cascade (first match wins):
      1. AGGRESSIVE alts present + chosen is safe + chosen not aggressive
         → "trades initiative for recapture safety"
      2. MATERIAL alts present + chosen does not capture
         → "trades material for positional advantage"
      3. AGGRESSIVE alts present + chosen is structurally sound +
         chosen not aggressive
         → "trades initiative for structural integrity"
      4. DEFENSIVE alts present + chosen captures + chosen recapture-risky
         → "accepts exposure to win material"

    Pure function. Never mutates inputs. Deterministic.
    """
    cf = chosen_facts or {}

    aggressive_idxs = [i for i, _ in (groups.get("AGGRESSIVE") or [])]
    material_idxs = [i for i, _ in (groups.get("MATERIAL") or [])]
    defensive_idxs = [i for i, _ in (groups.get("DEFENSIVE") or [])]

    chosen_aggressive = (
        cf.get("creates_immediate_threat") is True
        or cf.get("shot_sequence_available") is True
    )
    chosen_caps = cf.get("captures_count", 0)
    chosen_captures = (
        isinstance(chosen_caps, (int, float)) and chosen_caps > 0
    )
    chosen_safe = cf.get("opponent_can_recapture") is False
    chosen_well_structured = cf.get("leaves_piece_isolated") is False
    chosen_recapture_risk = cf.get("opponent_can_recapture") is True

    # Priority 1: chosen is defensive, aggressive options exist
    if aggressive_idxs and not chosen_aggressive and chosen_safe:
        return (
            f"Chosen move tradeoff: forfeits aggressive options "
            f"{_format_index_list_compact(aggressive_idxs)} in favour of "
            f"recapture safety."
        )

    # Priority 2: chosen does not capture, material options exist
    if material_idxs and not chosen_captures:
        return (
            f"Chosen move tradeoff: forfeits material captures "
            f"{_format_index_list_compact(material_idxs)} in favour of "
            f"positional advantage."
        )

    # Priority 3: chosen is structurally sound, aggressive options exist
    if aggressive_idxs and not chosen_aggressive and chosen_well_structured:
        return (
            f"Chosen move tradeoff: forfeits aggressive options "
            f"{_format_index_list_compact(aggressive_idxs)} in favour of "
            f"structural integrity."
        )

    # Priority 4: chosen accepts exposure to win material
    if defensive_idxs and chosen_captures and chosen_recapture_risk:
        return (
            f"Chosen move tradeoff: accepts exposure to win material, "
            f"where defensive options "
            f"{_format_index_list_compact(defensive_idxs)} would have "
            f"stayed safe."
        )

    return None


# ── Comparative system prompts (Step 2) ─────────────────────────────────────
#
# These are the system prompts the comparative LLM stage will eventually use.
# They are defined here in Step 2 with ZERO runtime wiring — no call site
# references them yet. Step 5 will wire them into the orchestrator; Step 6
# will gate the orchestrator behind an env flag in `_explain_chosen_move`.
#
# Locked design constraints:
#   - ≤30 lines per prompt (compactness)
#   - JSON-only output: {"comparative_reasoning": "..."}
#   - Explicit "describe alternatives only" contract
#   - Explicit "do NOT re-justify the chosen move" prohibition
#   - No "Alternative [N] [THEME]:" robotic template
#   - No verbatim forbidden-vocabulary enumeration (categorical guidance only)
#   - Natural-language prose

EXPLAINER_COMPARATIVE_SYSTEM: str = """\
You are a checkers coach writing a SHORT comparative paragraph that
describes the alternative moves NOT chosen and the tradeoff the chosen
move accepted. You do NOT re-justify the chosen move; another stage has
already explained why it was selected.

CONTRACT
  - Write a single fluent paragraph of 2-4 sentences.
  - Describe alternatives in natural prose grounded in the provided
    grouped seeds. Use the seeds' bracketed index references (e.g.,
    "alternatives [1] and [3]") and natural theme words such as
    "aggressive", "defensive", "material", "structural", "promotion",
    "mobility".
  - Do NOT use the literal "Alternative [N] [THEME]:" template form.
  - Do NOT re-state, re-justify, or re-praise the chosen move beyond
    the tradeoff framing already given in the seeds.
  - Do not invent strategic concepts, named scores, or jargon beyond
    what the seeds describe.
  - Do not state any number that does not appear in a seed.
  - Do not write raw field=value syntax in prose.

OUTPUT FORMAT — reply with ONLY this JSON, no markdown:
{"comparative_reasoning": "<your paragraph>"}
"""


EXPLAINER_COMPARATIVE_REFINEMENT_SYSTEM: str = """\
You are revising a previously-generated comparative paragraph that
contained unsupported claims. Produce a corrected paragraph that
preserves every sentence and clause that was NOT flagged as problematic.

CONTRACT
  - Describe alternatives only; do NOT re-justify the chosen move.
  - Preserve unchanged sentences verbatim.
  - For each flagged issue, change only the clause that contains it.
  - Do not add new sentences. If a flagged claim has no supported
    replacement, REMOVE the claim entirely rather than reword it.
  - Do not introduce new strategic terms, named scores, or jargon.
  - Do not state any number not present in the seeds.

OUTPUT FORMAT — reply with ONLY this JSON, no markdown:
{"comparative_reasoning": "<corrected paragraph>"}
"""


# ── Comparative user-prompt builders (Step 2) ───────────────────────────────
#
# Both builders are pure functions: deterministic, no I/O, no mutation. They
# return strings that downstream Step 5 orchestration will pass to the LLM
# client. No production code path references these builders yet.

def _path_to_notation(path: Any) -> Optional[str]:
    """
    Convert a candidate path (list of (row, col) tuples) to compact notation.

    Returns 'r,c→r,c' using the first and last positions in the path, or None
    when the path is absent, malformed, or too short to derive a move.
    Single-step and multi-jump paths are both handled: only start and end matter.
    """
    if not isinstance(path, (list, tuple)) or len(path) < 2:
        return None
    try:
        start, end = path[0], path[-1]
        r1, c1 = int(start[0]), int(start[1])
        r2, c2 = int(end[0]), int(end[1])
        return f"{r1},{c1}→{r2},{c2}"
    except (TypeError, ValueError, IndexError):
        return None


def _annotate_seed_indices(seed: str, idx_to_notation: dict) -> str:
    """
    Replace every [N] token in a seed string with [N] (r,c→r,c) when a
    notation entry for N exists in idx_to_notation.  Tokens with no entry
    are left unchanged.  Pure, deterministic, no side-effects.
    """
    def _sub(m: re.Match) -> str:
        n = int(m.group(1))
        notation = idx_to_notation.get(n)
        return f"[{n}] ({notation})" if notation else m.group(0)
    return re.sub(r"\[(\d+)\]", _sub, seed)


def build_comparative_user_prompt(
    seeds: list[str],
    chosen_path: Any,
    idx_to_notation: Optional[dict] = None,
) -> str:
    """
    Build the user prompt for the comparative-reasoning LLM call.

    Contents:
      1. The chosen-move path printed as reference context. The path is
         coordinate data, NOT the chosen-move reasoning paragraph; this
         keeps the comparative stage isolated from chosen-move prose.
      2. Every grouped comparative seed, listed verbatim.
      3. A short reminder of the JSON output contract.

    This function never reads or references the chosen-move reasoning
    paragraph, any chosen-move seed, any chosen-move diagnostic field, or
    any state outside its two arguments. Pure, deterministic.
    """
    lines: list[str] = [
        f"Chosen-move path (reference only — do NOT re-justify the move): "
        f"{chosen_path}",
        "",
        "Grouped comparative seeds — describe the alternatives and the",
        "tradeoff using ONLY these:",
    ]
    if seeds:
        for i, s in enumerate(seeds, 1):
            if idx_to_notation:
                s = _annotate_seed_indices(s, idx_to_notation)
            lines.append(f"  {i}. {s}")
    else:
        lines.append("  (no comparative seeds provided)")
    lines += [
        "",
        "Produce a single paragraph of 2-4 sentences that describes the",
        "alternative groups and the tradeoff in natural prose.",
        "",
        'Reply with ONLY: {"comparative_reasoning": "<your paragraph>"}',
    ]
    return "\n".join(lines)


def build_comparative_refinement_user_prompt(
    prev_text: str,
    sanitized_issues: list[str],
) -> str:
    """
    Build the user prompt for a comparative-refinement LLM call.

    Contents:
      1. The previous comparative paragraph (verbatim).
      2. A pre-sanitized list of issue descriptors. The caller is
         responsible for sanitising — this function does NOT scrub the
         strings. Passing raw verbatim forbidden phrases or fabricated
         numbers will pass them through to the LLM unchanged.
      3. A short reminder of the JSON output contract.

    Pure, deterministic. No reference to the chosen-move pipeline.
    """
    lines: list[str] = [
        "Previous comparative paragraph (contained unsupported claims):",
        f"  {prev_text}",
        "",
        "Issues to fix (sanitized — do NOT echo any verbatim phrase or "
        "number back into the prose):",
    ]
    if sanitized_issues:
        for issue in sanitized_issues:
            lines.append(f"  - {issue}")
    else:
        lines.append("  (no issues provided)")
    lines += [
        "",
        "Rewrite the paragraph removing each issue. Preserve correct text",
        "verbatim. Do not introduce new strategic terms or numbers.",
        "",
        'Reply with ONLY: {"comparative_reasoning": "<your paragraph>"}',
    ]
    return "\n".join(lines)


# ── Comparative verifier (Step 3) ───────────────────────────────────────────
#
# Validates a comparative-reasoning paragraph against the actual alternative-
# move facts. Locked scope (do not extend without a roadmap revision):
#
#   - index validity                  alt indices must be in range
#   - self-reference detection        no alt-index may resolve to chosen
#   - per-alternative fact consistency  claims about one alt match its facts
#   - grouped-claim consistency       group claims must hold for every member
#   - tradeoff consistency            chosen-move tradeoff prose matches
#                                       chosen-move facts
#
# Out of scope (the chosen-move verifier handles these on chosen-move prose
# in a separate, isolated path):
#   - forbidden vocabulary
#   - schema-leak detection
#   - chosen-move numeric fabrication
#   - chosen-move prose semantics
#
# Isolation invariant: this module imports nothing from
# `checkers.agents.explainer_agent`, `checkers.evaluation.unified_verifier`,
# `checkers.evaluation.claim_extractor`, or any other chosen-move verifier
# component. The comparative verifier is a stand-alone pure function.


@dataclass(frozen=True)
class ComparativeContradiction:
    """
    Lightweight record describing one comparative-prose contradiction.

    Deliberately NOT compatible with `ClaimRecord` from
    `checkers.evaluation.reasoning_taxonomy`: the comparative verifier has
    different semantics (index-attribution, group consistency, tradeoff
    consistency) than the chosen-move verifier (claim_verifiability,
    hallucination_type, seed_risk_type). Keeping the record types separate
    is what enforces the verifier-isolation invariant at the type level.

    Field semantics:
      type     One of:
                 "invalid_index"          alt index out of range
                 "self_reference"         alt-index resolves to chosen
                 "per_alt_mismatch"       single-alt claim wrong
                 "grouped_claim_partial"  group claim wrong for >=1 member
                 "invalid_tradeoff"       chosen-move tradeoff inconsistent
                 "chosen_move_factual"    direct/implied factual claim about
                                          the chosen move contradicts chosen_facts
      indices  Tuple of alt indices involved.
      fact_key Engine fact key the claim was about (or "" if unstructured).
      expected What the prose claimed (True/False/"POSITIVE"/"ZERO").
      actual   What the facts actually say (for per_alt_mismatch) or the
               failing-subset tuple (for grouped_claim_partial).
      clause   First ~120 chars of the offending clause (debug context).
    """
    type: str
    indices: tuple[int, ...] = ()
    fact_key: str = ""
    expected: Any = None
    actual: Any = None
    clause: str = ""


# Field name → semantic category. Private to this module — the chosen-move
# verifier has its own mapping in explainer_agent.py. Keeping these separate
# is intentional and enforces the verifier-isolation invariant.
_COMPARATIVE_FIELD_TO_CATEGORY: dict[str, str] = {
    "opponent_can_recapture":   "safety",
    "creates_immediate_threat": "threat",
    "shot_sequence_available":  "threat",
    "captures_count":           "material",
    "net_gain":                 "material",
    "leaves_piece_isolated":    "structure",
    "results_in_king":          "promotion",
    "_MOBILITY_REDUCED":        "mobility",
}


# Regex: "alternative [N]", "alternatives [a] and [b]",
# "alternatives [a], [b], and [c]" (also with "move" / "moves").
#
# Separator alternatives handled:
#   ", "         non-Oxford comma list:    "[1], [2]"
#   ", and "     Oxford-comma terminal:    "[1], [2], and [3]"
#   " and "      two-member conjunction:   "[1] and [2]"
_SEPARATOR_RE_FRAGMENT = r"(?:\s*,\s*(?:and\s+)?|\s+and\s+)"
_INDEX_REF_RE = re.compile(
    r"\b(?:alternative|move)s?\s+"
    r"(?P<indices>"
    r"\[\s*\d+\s*\]"
    r"(?:" + _SEPARATOR_RE_FRAGMENT + r"\[\s*\d+\s*\])*"
    r")",
    flags=re.IGNORECASE,
)
_BARE_IDX_RE = re.compile(r"\[\s*(\d+)\s*\]")


# Per-alt claim patterns. Order matters: NEGATED/FALSE patterns appear
# BEFORE the equivalent POSITIVE/TRUE patterns so a clause like
# "does not create a threat" attributes to the False claim only.
_PER_ALT_CLAIMS: tuple[tuple[Any, str, Any, str], ...] = (
    (re.compile(
        r"(?:do(?:es)?\s+not\s+create|no)\s+(?:an?\s+)?(?:immediate\s+)?threat",
        re.IGNORECASE,
    ), "creates_immediate_threat", False, "creates no threat"),
    (re.compile(
        r"(?:avoid(?:s|ing)?|cannot)\s+(?:opponent\s+)?recapture",
        re.IGNORECASE,
    ), "opponent_can_recapture", False, "avoids recapture"),
    (re.compile(
        r"do(?:es)?\s+not\s+capture|without\s+capturing|no\s+capture",
        re.IGNORECASE,
    ), "captures_count", "ZERO", "does not capture"),
    (re.compile(
        r"maintain(?:s)?\s+connectivity|stay(?:s)?\s+connected"
        r"|preserve(?:s)?\s+(?:piece\s+)?coordination",
        re.IGNORECASE,
    ), "leaves_piece_isolated", False, "maintains connectivity"),

    (re.compile(
        r"(?:create(?:s|d)?|creating)\s+(?:an?\s+)?(?:immediate\s+)?threats?",
        re.IGNORECASE,
    ), "creates_immediate_threat", True, "creates threat"),
    (re.compile(
        r"allow(?:s|ing)?\s+(?:opponent\s+)?recapture|recapture\s+risk",
        re.IGNORECASE,
    ), "opponent_can_recapture", True, "allows recapture"),
    (re.compile(
        r"(?:capture(?:s|d)?|capturing)\s+(?:a|an|the|one|two|three|four|five|"
        r"six|seven|eight|nine|ten|\d+)\b|win(?:s)?\s+material|gain(?:s)?\s+material",
        re.IGNORECASE,
    ), "captures_count", "POSITIVE", "captures"),
    (re.compile(
        r"(?:isolate(?:s|d)?|leaves?\s+(?:the\s+)?(?:moved\s+)?piece\s+isolated"
        r"|(?:is|are)\s+isolated)\b",
        re.IGNORECASE,
    ), "leaves_piece_isolated", True, "isolates"),
    (re.compile(
        r"(?:promote(?:s|d)?|crown(?:s|ed)?|become(?:s|ing)?\s+a\s+king"
        r"|advance(?:s|d)?\s+toward\s+king)",
        re.IGNORECASE,
    ), "results_in_king", True, "promotes"),
    (re.compile(r"shot\s+sequence", re.IGNORECASE),
     "shot_sequence_available", True, "offers shot sequence"),
    (re.compile(
        r"reduce(?:s|ing)?\s+(?:the\s+)?opponent\s+mobility"
        r"|restrict(?:s|ing)?\s+(?:the\s+)?opponent",
        re.IGNORECASE,
    ), "_MOBILITY_REDUCED", True, "reduces opponent mobility"),
)


# ── Pass 3: chosen-move factual claim patterns ───────────────────────────────
#
# These are checked ONLY on sentences that explicitly reference the chosen move
# (matched by _CHOSEN_MOVE_REF_RE), so they never fire on sentences that are
# purely about indexed alternatives.
#
# Tuple structure: (regex, fact_key, expected, label)
# Same semantics as _PER_ALT_CLAIMS.  "expected" is True/False/"POSITIVE"/"ZERO".

_CHOSEN_MOVE_REF_RE = re.compile(
    r"\b(?:chosen|selected)\s+(?:move|path)\b",
    re.IGNORECASE,
)

_CHOSEN_MOVE_CLAIM_PATTERNS: tuple[tuple[Any, str, Any, str], ...] = (
    # Positive material claims on the chosen move
    (re.compile(
        r"secure(?:s|d)?\s+(?:a\s+)?capture",
        re.IGNORECASE,
    ), "captures_count", "POSITIVE", "secures a capture"),
    (re.compile(
        r"win(?:s|ning)?\s+material|gain(?:s|ing)?\s+material"
        r"|net\s+(?:material\s+)?gain",
        re.IGNORECASE,
    ), "net_gain", "POSITIVE", "wins/gains material"),
    # Threat claim on the chosen move
    (re.compile(
        r"(?:create(?:s|d)?|open(?:s|ed)?)\s+(?:an?\s+)?(?:immediate\s+)?threats?",
        re.IGNORECASE,
    ), "creates_immediate_threat", True, "creates immediate threat"),
    # Promotion claim on the chosen move
    (re.compile(
        r"promote(?:s|d)?|become(?:s|ing)?\s+a\s+king",
        re.IGNORECASE,
    ), "results_in_king", True, "promotes to king"),
    # Recapture safety claim on the chosen move
    (re.compile(
        r"(?:avoid(?:s|ing)?|safe\s+from)\s+recapture|recapture\s+safety",
        re.IGNORECASE,
    ), "opponent_can_recapture", False, "avoids recapture"),
    # Recapture exposure claim on the chosen move
    (re.compile(
        r"(?:left\s+)?exposed\s+to\s+recapture|allows?\s+recapture",
        re.IGNORECASE,
    ), "opponent_can_recapture", True, "exposed to recapture"),
)

# Contrastive framing patterns that do NOT contain an explicit "chosen move"
# reference but imply the chosen move has a property by describing what
# alternatives "sacrificed" relative to it.
# Scanned on the full text, not per-sentence.
_CHOSEN_MOVE_IMPLIED_PATTERNS: tuple[tuple[Any, str, Any, str], ...] = (
    # "sacrificed the chance to capture" → chosen_facts["captures_count"] > 0 required
    (re.compile(
        r"sacrific(?:e(?:s|d)?|ing)\s+(?:the\s+)?(?:chance\s+to\s+capture"
        r"|capture\s+opportunit(?:y|ies))",
        re.IGNORECASE,
    ), "captures_count", "POSITIVE", "implies chosen move captures"),
)


def _resolve_actual_fact(facts: dict, fact_key: str) -> Any:
    """Read a fact value, synthesising the mobility-reduction boolean."""
    if fact_key == "_MOBILITY_REDUCED":
        mb = facts.get("opponent_mobility_before")
        ma = facts.get("opponent_mobility_after")
        if isinstance(mb, (int, float)) and isinstance(ma, (int, float)):
            return ma < mb
        return None
    return facts.get(fact_key)


def _indices_violating(
    idxs: list[int],
    all_candidates: list[dict],
    fact_key: str,
    expected: Any,
) -> list[int]:
    """Return the subset of indices whose facts do NOT satisfy expected."""
    violators: list[int] = []
    for idx in idxs:
        facts = all_candidates[idx].get("facts") or {}
        actual = _resolve_actual_fact(facts, fact_key)
        if expected == "POSITIVE":
            ok = isinstance(actual, (int, float)) and actual > 0
        elif expected == "ZERO":
            ok = isinstance(actual, (int, float)) and actual == 0
        else:
            ok = (actual == expected)
        if not ok:
            violators.append(idx)
    return violators


def verify_comparative_reasoning(
    text: str,
    all_candidates: list[dict],
    chosen_move: dict,
) -> list[ComparativeContradiction]:
    """
    Validate a comparative-reasoning paragraph.

    Locked scope (Step 3 of the Comparative Reasoning v2 roadmap):
      - index validity
      - self-reference
      - per-alternative fact consistency
      - grouped-claim consistency
      - tradeoff consistency

    Returns a (possibly empty) list of `ComparativeContradiction` records.

    Pure, deterministic. Never mutates inputs. Never raises. Never invokes
    the chosen-move verifier or any chosen-move-prose checker.
    """
    if not isinstance(text, str) or not text:
        return []
    if not isinstance(all_candidates, list) or not all_candidates:
        return []

    chosen_path = (chosen_move or {}).get("path") if chosen_move else None
    n_cand = len(all_candidates)

    results: list[ComparativeContradiction] = []

    # ── Pass 1: per-reference index + per-alt claim checks ──────────────
    for match in _INDEX_REF_RE.finditer(text):
        indices_text = match.group("indices")
        raw_idxs = [int(m) for m in _BARE_IDX_RE.findall(indices_text)]
        if not raw_idxs:
            continue

        # Clause = from match end to next sentence boundary (or EOF).
        clause_start = match.end()
        boundary = re.search(r"[.!?]", text[clause_start:])
        clause_end = (
            clause_start + boundary.start() if boundary else len(text)
        )
        clause = text[clause_start:clause_end].strip()

        # Index validity + self-reference per index.
        valid_idxs: list[int] = []
        for idx in raw_idxs:
            if idx < 0 or idx >= n_cand:
                results.append(ComparativeContradiction(
                    type="invalid_index",
                    indices=(idx,),
                    clause=clause[:120],
                ))
                continue
            if chosen_path is not None and \
                    all_candidates[idx].get("path") == chosen_path:
                results.append(ComparativeContradiction(
                    type="self_reference",
                    indices=(idx,),
                    clause=clause[:120],
                ))
                continue
            valid_idxs.append(idx)

        if not valid_idxs:
            continue

        # Per-alt / group claim checks against this clause.
        for pat, fact_key, expected, _label in _PER_ALT_CLAIMS:
            if not pat.search(clause):
                continue
            violators = _indices_violating(
                valid_idxs, all_candidates, fact_key, expected,
            )
            if not violators:
                continue
            if len(valid_idxs) == 1:
                idx = valid_idxs[0]
                actual = _resolve_actual_fact(
                    all_candidates[idx].get("facts") or {}, fact_key,
                )
                results.append(ComparativeContradiction(
                    type="per_alt_mismatch",
                    indices=(idx,),
                    fact_key=fact_key,
                    expected=expected,
                    actual=actual,
                    clause=clause[:120],
                ))
            else:
                results.append(ComparativeContradiction(
                    type="grouped_claim_partial",
                    indices=tuple(valid_idxs),
                    fact_key=fact_key,
                    expected=expected,
                    actual=tuple(violators),
                    clause=clause[:120],
                ))

    # ── Pass 2: tradeoff consistency ────────────────────────────────────
    if chosen_move is not None:
        text_lower = text.lower()
        cf = chosen_move.get("facts") or {}

        # A. forfeits ... in favour of recapture safety
        if re.search(
            r"forfeit(?:s|ed)?\s+.*?\bin\s+favou?r\s+of\s+recapture\s+safety",
            text_lower,
        ):
            if cf.get("opponent_can_recapture") is not False:
                results.append(ComparativeContradiction(
                    type="invalid_tradeoff",
                    fact_key="opponent_can_recapture",
                    expected=False,
                    actual=cf.get("opponent_can_recapture"),
                    clause="<tradeoff: recapture safety>",
                ))

        # B. forfeits ... in favour of structural integrity
        if re.search(
            r"forfeit(?:s|ed)?\s+.*?\bin\s+favou?r\s+of\s+structural\s+integrity",
            text_lower,
        ):
            if cf.get("leaves_piece_isolated") is not False:
                results.append(ComparativeContradiction(
                    type="invalid_tradeoff",
                    fact_key="leaves_piece_isolated",
                    expected=False,
                    actual=cf.get("leaves_piece_isolated"),
                    clause="<tradeoff: structural integrity>",
                ))

        # C. forfeits material captures ...
        if re.search(r"forfeit(?:s|ed)?\s+material\s+captures", text_lower):
            cap = cf.get("captures_count", 0)
            if isinstance(cap, (int, float)) and cap > 0:
                results.append(ComparativeContradiction(
                    type="invalid_tradeoff",
                    fact_key="captures_count",
                    expected="ZERO",
                    actual=cap,
                    clause="<tradeoff: forfeits material>",
                ))

        # D. chosen move ... accepts exposure or temporary vulnerability
        #    (implies: captures_count > 0 AND opponent_can_recapture=True)
        #    Extended to also match "accepts a temporary vulnerability" in
        #    addition to the original "accepts exposure" form.
        if re.search(
            r"chosen\s+move\s+.*?accepts?\s+(?:exposure|(?:a\s+)?temporary\s+vulnerability)",
            text_lower,
        ):
            cap = cf.get("captures_count", 0)
            if not (isinstance(cap, (int, float)) and cap > 0):
                results.append(ComparativeContradiction(
                    type="invalid_tradeoff",
                    fact_key="captures_count",
                    expected="POSITIVE",
                    actual=cap,
                    clause="<tradeoff: accepts exposure>",
                ))
            if cf.get("opponent_can_recapture") is not True:
                results.append(ComparativeContradiction(
                    type="invalid_tradeoff",
                    fact_key="opponent_can_recapture",
                    expected=True,
                    actual=cf.get("opponent_can_recapture"),
                    clause="<tradeoff: accepts exposure>",
                ))

        # E. chosen move ... to secure a gain / secures a gain
        #    (implies: net_gain > 0)
        #    Catches "accepts a temporary vulnerability to secure a gain"
        #    and similar phrasing where the chosen move is falsely credited
        #    with a material gain when net_gain=0.
        if re.search(
            r"chosen\s+move\s+.*?(?:to\s+secure|secures?)\s+(?:a\s+)?(?:material\s+)?gain",
            text_lower,
        ):
            ng = cf.get("net_gain", 0)
            if not (isinstance(ng, (int, float)) and ng > 0):
                results.append(ComparativeContradiction(
                    type="invalid_tradeoff",
                    fact_key="net_gain",
                    expected="POSITIVE",
                    actual=ng,
                    clause="<tradeoff: secure a gain>",
                ))

        # ── Pass 3: chosen-move factual claim scan ────────────────────────────
        # For each sentence that explicitly references the chosen move, check
        # every factual claim pattern against chosen_facts.
        # Skip mixed sentences that also contain alternative-index references
        # (e.g. "... alternative [1] creates a threat; chosen move forfeits ...").
        # In mixed sentences the claim may describe an alternative, not the chosen
        # move, so checking them would cause false positives.  Full-text implied
        # patterns (handled below) cover the indirect cases.
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if not _CHOSEN_MOVE_REF_RE.search(sentence):
                continue
            if _BARE_IDX_RE.search(sentence):
                continue
            for pat, fact_key, expected, _label in _CHOSEN_MOVE_CLAIM_PATTERNS:
                if not pat.search(sentence):
                    continue
                actual = _resolve_actual_fact(cf, fact_key)
                if expected == "POSITIVE":
                    ok = isinstance(actual, (int, float)) and actual > 0
                elif expected == "ZERO":
                    ok = isinstance(actual, (int, float)) and actual == 0
                else:
                    ok = (actual == expected)
                if not ok:
                    results.append(ComparativeContradiction(
                        type="chosen_move_factual",
                        fact_key=fact_key,
                        expected=expected,
                        actual=actual,
                        clause=sentence[:120],
                    ))

        for pat, fact_key, expected, _label in _CHOSEN_MOVE_IMPLIED_PATTERNS:
            if not pat.search(text):
                continue
            actual = _resolve_actual_fact(cf, fact_key)
            if expected == "POSITIVE":
                ok = isinstance(actual, (int, float)) and actual > 0
            elif expected == "ZERO":
                ok = isinstance(actual, (int, float)) and actual == 0
            else:
                ok = (actual == expected)
            if not ok:
                results.append(ComparativeContradiction(
                    type="chosen_move_factual",
                    fact_key=fact_key,
                    expected=expected,
                    actual=actual,
                    clause=f"<implied: {_label}>",
                ))

    return results


def sanitize_comparative_contradiction(
    record: ComparativeContradiction,
) -> str:
    """
    Convert a `ComparativeContradiction` into a non-priming hint string
    suitable for inclusion in a refinement user prompt.

    Sanitization properties (mirrored from the chosen-move sanitizer
    pattern but independently implemented — no cross-module imports):
      - Never echoes a verbatim forbidden phrase or numeric fact value.
      - Mentions the alternative INDEX (structural data, safe to echo).
      - Mentions a semantic CATEGORY label, not a raw field name.

    Pure function. Never raises.
    """
    t = record.type
    idxs = record.indices

    if t == "invalid_index":
        idx = idxs[0] if idxs else "?"
        return (
            f"sentence references alternative [{idx}] which does not exist "
            "in the candidate list"
        )
    if t == "self_reference":
        idx = idxs[0] if idxs else "?"
        return (
            f"sentence refers to alternative [{idx}] but that index is the "
            "chosen move itself"
        )
    if t == "per_alt_mismatch":
        idx = idxs[0] if idxs else "?"
        category = _COMPARATIVE_FIELD_TO_CATEGORY.get(
            record.fact_key, "factual",
        )
        return (
            f"sentence makes an incorrect {category} claim about "
            f"alternative [{idx}]"
        )
    if t == "grouped_claim_partial":
        return (
            "sentence makes a group claim that does not hold for every "
            "named alternative"
        )
    if t == "invalid_tradeoff":
        category = _COMPARATIVE_FIELD_TO_CATEGORY.get(
            record.fact_key, "factual",
        )
        return (
            f"sentence describes a {category} tradeoff that does not match "
            "the chosen move's facts"
        )
    if t == "chosen_move_factual":
        category = _COMPARATIVE_FIELD_TO_CATEGORY.get(
            record.fact_key, "factual",
        )
        return (
            f"sentence makes an incorrect {category} claim about the chosen move"
        )
    return "sentence contains an unsupported comparative claim"


# ── Step 4: Refinement primitives and orchestration ──────────────────────────
#
# SCOPE (Step 4 of the Comparative Reasoning v2 roadmap):
#   refine_comparative_reasoning — one-shot LLM refinement for the comparative
#   paragraph.  Uses Steps 2/3 prompts + verifier + sanitizer.  max_attempts=1
#   (hard cap).  No runtime wiring — not called from any production code path.
#
# RefinementCandidate and _evaluate_refinement_candidate are defined here so
# they are importable from comparative_reasoner without any dependency on
# explainer_agent.  Future chosen-move refactoring may import them from here.
#
# Isolation: no top-level or function-level import from explainer_agent, unified_verifier,
# claim_extractor, or any other chosen-move component.


@dataclass
class RefinementCandidate:
    """One parsed LLM refinement attempt with its contradiction count."""
    raw: str
    text: Optional[str]
    n_contradictions: int


def _evaluate_refinement_candidate(
    candidate: RefinementCandidate,
    baseline_count: int,
) -> bool:
    """
    Monotonic gate: accept this candidate only when it strictly reduces
    the contradiction count relative to baseline_count.

    Returns True when candidate.text is not None AND
    candidate.n_contradictions < baseline_count.

    This is the single authoritative implementation of the monotonic gate
    for comparative refinement.  Do not duplicate this predicate inline in
    refine_comparative_reasoning or elsewhere.
    """
    return (
        candidate.text is not None
        and candidate.n_contradictions < baseline_count
    )


def _extract_comparative_json(raw: str) -> Optional[str]:
    """
    Extract the "comparative_reasoning" string from a raw LLM JSON response.

    Strips markdown code fences if present, parses JSON, and returns the
    value of the "comparative_reasoning" key.  Falls back to a regex scan
    when strict parsing fails.  Returns None on malformed input.

    Pure function.  Never raises.
    """
    if not isinstance(raw, str) or not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            v = obj.get("comparative_reasoning")
            if isinstance(v, str) and v.strip():
                return v.strip()
    except (json.JSONDecodeError, Exception):
        pass
    # Regex fallback: handles partial JSON or surrounding prose.
    m = re.search(
        r'"comparative_reasoning"\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
        re.DOTALL,
    )
    if m:
        inner = m.group(1).replace("\\n", " ").replace('\\"', '"')
        inner = re.sub(r"\s+", " ", inner).strip()
        return inner or None
    return None


# Retry wait schedule (seconds) for API calls: 6 attempts, 5 inter-attempt sleeps.
# Cycle: 20 → 30 → 40 → 20 → 30 (second cycle truncated to 2 entries so total = 6).
_API_RETRY_WAITS: tuple[int, ...] = (20, 30, 40, 20, 30)


def _call_comparative_api(system: str, user: str) -> str:
    """
    Comparative-pipeline Mistral caller (Step 8: delegates to llm_provider).

    Reads MISTRAL_COMPARATIVE_API_KEY and MISTRAL_EXPLAINER_MODEL from the
    environment at call time (not at import time) so ablation env-var changes
    take effect immediately without re-importing the module.

    MISTRAL_COMPARATIVE_API_KEY is the ONLY key consulted.  There is no
    fallback to MISTRAL_API_KEY — the two paths are intentionally isolated.
    If the variable is absent or empty a ValueError is raised immediately so
    the operator knows exactly which key is missing.

    Retry policy: up to 6 attempts with waits [20, 30, 40, 20, 30]s.
    HTTP 429 adds an extra 15 s sleep before the scheduled wait.
    Raises the last exception when all attempts are exhausted.
    """
    api_key = os.environ.get("MISTRAL_COMPARATIVE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "MISTRAL_COMPARATIVE_API_KEY is not set. "
            "Export it before running the comparative reasoning path. "
            "This key is separate from MISTRAL_API_KEY (chosen-reasoning path) "
            "and must be set independently."
        )
    model       = os.environ.get("MISTRAL_EXPLAINER_MODEL", "mistral-small-latest")
    temperature = float(os.environ.get("EXPLAINER_TEMPERATURE", "0.2"))
    messages    = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    last_exc: Exception = RuntimeError("no attempts made")
    for api_try in range(6):
        try:
            return call_mistral_once(
                messages,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=512,
            )
        except ProviderHTTPError as e:
            last_exc = e
            if api_try < len(_API_RETRY_WAITS):
                if e.code == 429:
                    time.sleep(15)
                wait = _API_RETRY_WAITS[api_try]
                print(
                    f"[COMPARATIVE_API] http {e.code} "
                    f"(try={api_try + 1}/6) — waiting {wait}s"
                )
                time.sleep(wait)
        except Exception as e:
            last_exc = e
            if api_try < len(_API_RETRY_WAITS):
                wait = _API_RETRY_WAITS[api_try]
                print(
                    f"[COMPARATIVE_API] error (try={api_try + 1}/6): {e} "
                    f"— waiting {wait}s"
                )
                time.sleep(wait)

    raise last_exc


def refine_comparative_reasoning(
    text: str,
    all_candidates: list[dict],
    chosen_move: dict,
    seeds: list[str],
    *,
    _api_caller: Any = None,
) -> tuple[str, int, bool]:
    """
    One-shot comparative refinement (max_attempts = 1, hard cap).

    Uses the Step 2/3 comparative prompts + verifier + sanitizer to attempt
    one LLM rewrite of a comparative paragraph that contains contradictions.

    Arguments:
        text            The comparative paragraph to refine.
        all_candidates  Full candidate list (including chosen move).
        chosen_move     The move selected by the proposal node (read-only).
        seeds           Grouped comparative seeds from Step 1 (not used
                        directly in the one-shot refinement prompt, but
                        carried for future multi-attempt expansion).
        _api_caller     Injectable LLM caller(system, user) -> str.
                        Defaults to _call_comparative_api.  Pass a mock in
                        tests; never set to None in production if you want
                        live LLM behaviour.

    Returns:
        (final_text, retry_count, resolved)
        final_text   The refined paragraph if improvement was accepted;
                     the original text otherwise.
        retry_count  0 when no API attempt was made (clean input); 1 after
                     the single attempt (including failure cases).
        resolved     True only when final_text has zero contradictions.

    Preservation guarantees — original text is returned unchanged on:
        - API failure (exception from _api_caller)
        - Malformed JSON (response cannot be parsed)
        - Rejected refinement (candidate does not strictly improve)
        - No improvement (same or higher contradiction count)

    Invariants:
        - chosen_move is never mutated or re-evaluated.
        - _refine_reasoning (chosen-move path) is never called.
        - max_attempts = 1 is enforced by the function body, not a parameter.
    """
    caller = _api_caller if _api_caller is not None else _call_comparative_api

    # ── 1. Initial contradiction check ───────────────────────────────────────
    initial = verify_comparative_reasoning(text, all_candidates, chosen_move)
    if not initial:
        return text, 0, True

    # ── 2. Build refinement prompt ────────────────────────────────────────────
    sanitized = [sanitize_comparative_contradiction(c) for c in initial]
    user_prompt = build_comparative_refinement_user_prompt(text, sanitized)

    # ── 3. One API attempt (max_attempts = 1) ─────────────────────────────────
    retry_count = 1
    raw: Optional[str] = None
    try:
        raw = caller(EXPLAINER_COMPARATIVE_REFINEMENT_SYSTEM, user_prompt)
    except Exception:
        return text, retry_count, False

    # ── 4. Parse response ─────────────────────────────────────────────────────
    parsed_text = _extract_comparative_json(raw)
    if parsed_text is None:
        return text, retry_count, False

    # ── 5. Evaluate candidate via monotonic gate ──────────────────────────────
    new_contras = verify_comparative_reasoning(
        parsed_text, all_candidates, chosen_move,
    )
    candidate = RefinementCandidate(
        raw=raw,
        text=parsed_text,
        n_contradictions=len(new_contras),
    )
    if not _evaluate_refinement_candidate(candidate, len(initial)):
        return text, retry_count, False

    resolved = (candidate.n_contradictions == 0)
    return candidate.text, retry_count, resolved


# ── Step 5: Generation orchestrator ──────────────────────────────────────────
#
# SCOPE (Step 5 of the Comparative Reasoning v2 roadmap):
#   generate_comparative_reasoning — top-level orchestrator that wires Steps 1-4
#   into a single call.  NO runtime wiring — not called from any production code
#   path yet.  Step 6 will gate this behind an env flag inside _explain_chosen_move.
#
# Pipeline (invariants I1-I8 enforced):
#   I1  chosen_move is never mutated.
#   I2  No import from explainer_agent or any chosen-move verifier component.
#   I3  Runtime behavior of explainer_agent/_explain_chosen_move is byte-identical.
#   I4  No new env vars introduced.
#   I5  No provider split (always "mistral").
#   I6  Diagnostics written ONLY into comparative_* keys of diagnostics_out;
#       never touches ranker_diagnostics or any chosen-move diagnostic field.
#   I7  max_samples caps generation calls; refinement is one additional call (Step 4).
#   I8  Returns Optional[str]: the verified comparative paragraph or None.


def generate_comparative_reasoning(
    chosen_move: dict,
    all_candidates: list[dict],
    chosen_facts: dict,
    *,
    max_samples: int = 2,
    diagnostics_out: Optional[dict] = None,
    _api_caller: Any = None,
) -> Optional[str]:
    """
    Orchestrate full comparative-reasoning generation (Steps 1-4).

    Pipeline:
      1. Cluster alternatives by theme (Step 1) → groups.
         Return None if no groups exist.
      2. Build group seeds + tradeoff seed (Step 1) → all_seeds.
         Return None if no seeds can be built.
      3. Build the comparative user prompt (Step 2).
      4. Reject-sample loop (up to max_samples):
           • Call LLM with EXPLAINER_COMPARATIVE_SYSTEM.
           • Parse {"comparative_reasoning": "..."} from the response.
           • Verify via verify_comparative_reasoning (Step 3).
           • Short-circuit and return immediately on a zero-contradiction sample.
      5. If no valid samples were produced: return None.
      6. Pick the best sample (fewest contradictions).
      7. Refine via refine_comparative_reasoning (Step 4, max_attempts=1).
           • If refinement resolves to 0 contradictions: return refined text.
           • Otherwise: return None.

    Arguments:
        chosen_move      Proposal-selected move dict (read-only).
        all_candidates   Full candidate list including the chosen move (read-only).
        chosen_facts     Facts dict for the chosen move (read-only mirror).
        max_samples      Maximum generation attempts before refinement (default 2).
        diagnostics_out  Optional flat dict.  When provided, exactly the keys in
                         _COMPARATIVE_DIAGNOSTICS_KEYS are written into it.
                         No other key is touched.
        _api_caller      Injectable callable(system, user) -> str for testing.
                         Defaults to _call_comparative_api.  Internal test hook;
                         not part of the roadmap public signature.

    Returns None on:
        • No informative groups (no alternatives match any theme).
        • No seeds (defensive; should not occur when groups exist).
        • API failure across all generation samples.
        • All samples rejected (none could be parsed).
        • Refinement failure (best sample could not be resolved to 0 contradictions).
    """
    caller = _api_caller if _api_caller is not None else _call_comparative_api

    # ── Diagnostics accumulator ───────────────────────────────────────────────
    _d: dict[str, Any] = {
        "comparative_was_skipped":                 False,
        "comparative_skip_reason":                 None,
        "comparative_paragraph_text":              None,
        "comparative_seeds":                       [],
        "comparative_groups":                      {},
        "comparative_generation_samples_used":     0,
        "comparative_sample_contradiction_counts": [],
        "comparative_generation_short_circuited":  False,
        "comparative_initial_contradictions":      0,
        "comparative_final_contradictions":        0,
        "comparative_refinement_attempts":         0,
        "comparative_provider":                    "mistral",
    }

    def _flush() -> None:
        if diagnostics_out is not None:
            diagnostics_out.update(_d)

    # ── 1. Cluster alternatives by theme (Step 1) ─────────────────────────────
    cf: dict = chosen_facts or {}
    _candidates: list[dict] = all_candidates or []
    _chosen: dict = chosen_move or {}

    groups = _cluster_alternatives_by_theme(_candidates, _chosen)
    _d["comparative_groups"] = {
        tag: [idx for idx, _ in members]
        for tag, members in groups.items()
    }

    if not groups:
        _d["comparative_was_skipped"] = True
        _d["comparative_skip_reason"] = "no_informative_groups"
        _flush()
        return None

    # ── 2. Build seeds (Step 1) ───────────────────────────────────────────────
    group_seeds: list[str] = build_comparative_group_seeds(groups, cf)
    tradeoff_seed: Optional[str] = build_comparative_tradeoff_seed(cf, groups)
    all_seeds: list[str] = group_seeds + ([tradeoff_seed] if tradeoff_seed else [])
    _d["comparative_seeds"] = all_seeds

    if not all_seeds:
        _d["comparative_was_skipped"] = True
        _d["comparative_skip_reason"] = "no_seeds"
        _flush()
        return None

    # ── 3. Build generation user prompt (Step 2) ──────────────────────────────
    chosen_path = _chosen.get("path")
    _idx_to_notation: dict[int, str] = {}
    for _i, _cand in enumerate(_candidates):
        _n = _path_to_notation(_cand.get("path"))
        if _n:
            _idx_to_notation[_i] = _n
    user_prompt: str = build_comparative_user_prompt(
        all_seeds, chosen_path, _idx_to_notation or None,
    )

    # ── 4. Reject-sample loop ─────────────────────────────────────────────────
    valid_samples: list[tuple[str, int]] = []  # (text, n_contradictions)
    _loop_count = max(0, max_samples)
    _api_error_count = 0

    for _ in range(_loop_count):
        raw: Optional[str] = None
        try:
            raw = caller(EXPLAINER_COMPARATIVE_SYSTEM, user_prompt)
        except Exception:
            _api_error_count += 1
            continue

        parsed = _extract_comparative_json(raw)
        if parsed is None:
            continue

        n_contras = len(verify_comparative_reasoning(parsed, _candidates, _chosen))
        valid_samples.append((parsed, n_contras))
        _d["comparative_generation_samples_used"] += 1
        _d["comparative_sample_contradiction_counts"].append(n_contras)

        if n_contras == 0:
            # Perfect sample — return immediately without refinement.
            _d["comparative_generation_short_circuited"] = True
            _d["comparative_initial_contradictions"] = 0
            _d["comparative_final_contradictions"] = 0
            _d["comparative_paragraph_text"] = parsed
            _flush()
            return parsed

    # ── 5. No valid samples ───────────────────────────────────────────────────
    if not valid_samples:
        _d["comparative_was_skipped"] = True
        # Distinguish: every API call raised (api_failure) vs at least one
        # response was received but none could be parsed (all_samples_rejected).
        _d["comparative_skip_reason"] = (
            "api_failure"
            if _loop_count > 0 and _api_error_count == _loop_count
            else "all_samples_rejected"
        )
        _flush()
        return None

    # ── 6. Pick best sample (fewest contradictions) ───────────────────────────
    best_text, best_n = min(valid_samples, key=lambda x: x[1])
    _d["comparative_initial_contradictions"] = best_n

    # ── 7. Refine via Step 4 ──────────────────────────────────────────────────
    refined_text, refine_attempts, resolved = refine_comparative_reasoning(
        best_text, _candidates, _chosen, all_seeds,
        _api_caller=_api_caller,
    )
    _d["comparative_refinement_attempts"] = refine_attempts

    if resolved:
        _d["comparative_final_contradictions"] = 0
        _d["comparative_paragraph_text"] = refined_text
        _flush()
        return refined_text

    # Refinement did not resolve to 0 contradictions.
    # Record actual final count (may differ from best_n when partial improvement).
    final_n = len(verify_comparative_reasoning(refined_text, _candidates, _chosen))
    _d["comparative_final_contradictions"] = final_n
    _flush()
    return None
