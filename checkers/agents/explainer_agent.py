# agents/ranker_agent.py
#
# Reasoning-only explanation node for the American Checkers
# simplified proposal-authoritative pipeline.
#
# In this pipeline the move is selected ENTIRELY by
# deterministic_proposal_node. ranker_agent receives the already-chosen
# move plus the full candidate list and produces a grounded natural-
# language explanation. It MUST NOT — and structurally CANNOT — modify,
# re-score, re-rank, override, retry, or otherwise revisit the chosen
# move. unchosen_moves are read solely as comparative context for the
# explanation (i.e. why alternatives are weaker, expressed in symbolic
# facts), never as input to a selection decision.
#
# Backend : Mistral API only (call_mistral_explainer / call_explainer).
# Model   : MISTRAL_EXPLAINER_MODEL env var, default "mistral-small-latest".
# Key     : MISTRAL_API_KEY env var (console.mistral.ai).

from __future__ import annotations

import os
import re
from typing import Any, Optional

from checkers.state.state import CheckersState
from checkers.engine.board import RED
from checkers.agents.comparative_reasoner import generate_comparative_reasoning
from checkers.agents.llm_provider import call_mistral_once, ProviderHTTPError
from checkers.ontology.semantic_ontology import (
    FORBIDDEN_CONFLATION_PHRASES as _ONTOLOGY_FORBIDDEN_CONFLATION,
    GENERIC_FILLER_PHRASES as _ONTOLOGY_GENERIC_FILLER,
)
from checkers.evaluation.forbidden_vocab import (
    ABSOLUTE_FORBIDDEN_VOCAB as _FORBIDDEN_VOCAB,
    CONTEXT_FORBIDDEN_VOCAB as _CONTEXT_FORBIDDEN_VOCAB,
)

# ── Mistral settings ──────────────────────────────────────────────────────────
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_EXPLAINER_MODEL = os.environ.get("MISTRAL_EXPLAINER_MODEL", "mistral-small-latest")


# ── Shared settings ───────────────────────────────────────────────────────────
EXPLAINER_TEMPERATURE = float(os.environ.get("EXPLAINER_TEMPERATURE", "0.2"))


# ── Ablation toggle ──────────────────────────────────────────────────────────
# When EXPLAINER_SEEDS_DISABLED is truthy, the explanation path runs with an
# EMPTY reasoning_seeds list. The user prompt to the LLM is structurally
# identical (the same "Verified reasoning seeds (use ONLY these — …)" template
# is rendered), but no seed lines are included. The refinement loop and the
# symbolic truthfulness verifier remain active. The deterministic seed-derived
# fallbacks are deliberately suppressed in this mode so the metric layer can
# observe ungrounded reasoning behaviour.
#
# Purpose: produce reproducible evidence that symbolic seeds reduce
# contradictions and hallucinations. Read by _seeds_disabled() and
# _current_run_tag() below; not consulted anywhere else in the pipeline.

def _seeds_disabled() -> bool:
    return os.environ.get("EXPLAINER_SEEDS_DISABLED", "").lower() in (
        "1", "true", "yes", "on",
    )


def _current_run_tag() -> str:
    """Tag emitted into ranker_diagnostics and the eval-source record."""
    explicit = os.environ.get("EXPLAINER_RUN_TAG")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return "seed_off" if _seeds_disabled() else "seed_on"


def _comparative_stage_enabled() -> bool:
    """True by default (Step 7 cutover); disabled only by explicit opt-out.

    Set EXPLAINER_COMPARATIVE_STAGE_ENABLED to '0', 'false', 'no', or 'off' to
    disable the comparative-reasoning paragraph and restore pre-Step-7 behaviour
    for ablation or emergency rollback.
    """
    return os.environ.get("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1").lower() not in (
        "0", "false", "no", "off", "",
    )

# ── Utility helpers ───────────────────────────────────────────────────────────
def _get_minimax_score(move: dict[str, Any]) -> float:
    facts = move.get("facts", {}) or {}
    v = facts.get("minimax_score", float("-inf"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("-inf")
def call_mistral_explainer(system: str, user: str) -> str:
    """
    Calls the Mistral API via the provider layer (Step 8).

    Delegates the HTTP call to llm_provider.call_mistral_once so the chosen-
    reasoning path and the comparative-reasoning path share one HTTP primitive
    while remaining independently configured (provider-split boundary).

    HTTP 429 adds a 15 s sleep before re-raising so the retry loops in
    _generate_seeded_reasoning and _refine_reasoning get the scheduled delay.
    All HTTP errors are converted to ValueError for backward compatibility.

    Raises:
        ValueError  — API key missing, HTTP error, or unexpected response shape.
        OSError     — network-level failure propagated from call_mistral_once.
    """
    if not MISTRAL_API_KEY:
        raise ValueError(
            "MISTRAL_API_KEY is not set. "
            "Run: export MISTRAL_API_KEY='your_key_from_console.mistral.ai'"
        )
    import time as _t
    try:
        return call_mistral_once(
            [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            api_key=MISTRAL_API_KEY,
            model=MISTRAL_EXPLAINER_MODEL,
            temperature=EXPLAINER_TEMPERATURE,
            max_tokens=512,
        )
    except ProviderHTTPError as e:
        if e.code == 429:
            _t.sleep(15)
        raise ValueError(f"Mistral API HTTP {e.code}: {e.body}") from e


# ── Unified call dispatcher ───────────────────────────────────────────────────

def call_explainer(system: str, user: str) -> str:
    return call_mistral_explainer(system, user)


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


def _build_explainer_filtered_menu_snapshot(filtered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": m.get("type"),
            "path": _canonical_coord_list(m.get("path")),
            "captured": _canonical_coord_list(m.get("captured", [])),
            "facts": {"minimax_score": (m.get("facts", {}) or {}).get("minimax_score")},
        }
        for m in filtered
    ]


_LEGACY_CONTEXT_FORBIDDEN_NOTES: list[str] = [
    "central board presence",
    "new vulnerabilities",
]


def _merge_ontology_phrases() -> None:
    """Append any ontology phrases not already present in the runtime lists."""
    existing_abs = {p.lower() for p in _FORBIDDEN_VOCAB}
    for phrase in sorted(_ONTOLOGY_GENERIC_FILLER):
        if phrase.lower() not in existing_abs:
            _FORBIDDEN_VOCAB.append(phrase)
            existing_abs.add(phrase.lower())

    existing_ctx = {p.lower() for p in _CONTEXT_FORBIDDEN_VOCAB}
    for phrase in sorted(_ONTOLOGY_FORBIDDEN_CONFLATION):
        if phrase.lower() not in existing_ctx:
            _CONTEXT_FORBIDDEN_VOCAB.append(phrase)
            existing_ctx.add(phrase.lower())

    for phrase in _LEGACY_CONTEXT_FORBIDDEN_NOTES:
        if phrase.lower() not in existing_ctx:
            _CONTEXT_FORBIDDEN_VOCAB.append(phrase)
            existing_ctx.add(phrase.lower())


_merge_ontology_phrases()


def _ctx_phrase_negated(text: str, phrase: str) -> bool:
    """Return True if every occurrence of *phrase* in *text* is preceded by a
    negation marker within a 35-char window.  Returns False if any occurrence
    lacks a negation marker (warning should fire) or phrase not found.

    Recognised markers: 'no ', 'without ', 'not ', 'avoids ', 'avoid ',
    'prevents ', 'never '.  Covers forms like 'without creating new
    vulnerabilities' and 'avoids introducing new vulnerabilities' in addition
    to the original 'no new vulnerabilities' pattern.

    35 chars is enough for the longest expected negation phrase
    ('without introducing ' = 20 chars) while avoiding cross-sentence bleed.
    """
    _NEGATION_MARKERS = (
        "no ", "without ", "not ", "avoids ", "avoid ", "prevents ", "never ",
    )
    idx = text.find(phrase)
    if idx == -1:
        return False
    while idx != -1:
        window = text[max(0, idx - 35): idx]
        if not any(m in window for m in _NEGATION_MARKERS):
            return False
        idx = text.find(phrase, idx + 1)
    return True


def _check_reasoning_truthfulness(
    reasoning: str,
    facts: dict[str, Any],
    seeds: Optional[list[str]] = None,
) -> list[str]:
    """
    Post-hoc scan of LLM reasoning for claims that contradict engine-computed
    facts.  Returns a (possibly empty) list of human-readable warning strings.
    NEVER raises.  NEVER modifies the chosen move or reasoning text.

    Composition (post-E.1):
      1. Legacy in-file phrase/vocabulary/numeric checks (below).
      2. UNIFIED layer via checkers.evaluation.unified_verifier.contradiction_strings
         which runs extract_claims + verify_claims + numeric (E.3) + schema-leak
         (E.4) on the same text.  Any CONTRADICTED record produced there is
         appended as a warning string.  This guarantees that the runtime
         refinement loop and the evaluator metric layer see the same
         contradictions — the E.1 invariant.

    Parameters
    ----------
    reasoning : str
        The LLM-generated reasoning paragraph to check.
    facts : dict
        Engine-computed facts for the chosen move.
    seeds : list[str] or None
        The exact seed strings that were fed to the LLM.  If provided, the
        checker verifies that forbidden concepts are absent unless seeded.
    """
    if not reasoning:
        return []
    facts = facts or {}  # normalise; legacy guards use .get() which handle absent keys

    warnings: list[str] = []
    text = reasoning.lower()
    seeds_text = " ".join(s.lower() for s in (seeds or []))

    # ── Mobility ──────────────────────────────────────────────────────────────
    mob_after  = facts.get("opponent_mobility_after")
    mob_before = facts.get("opponent_mobility_before")
    if mob_after is not None and mob_before is not None:
        mobility_improvement_claimed = any(p in text for p in [
            "reduces mobility", "reducing mobility",
            "reduces opponent mobility", "reducing opponent mobility",
            "limits mobility", "limiting mobility", "limiting opponent",
            "restricts mobility", "restricts opponent",
            "fewer moves for", "cuts opponent moves",
            "narrows their options", "tightening their options",
            "constrains their options",
        ])
        if mobility_improvement_claimed and mob_after >= mob_before:
            warnings.append(
                f"REASONING_CONTRADICTION: claims mobility reduction but "
                f"opponent_mobility_after={mob_after} >= opponent_mobility_before={mob_before}"
            )

    # ── BUG-6: Mobility-disadvantage overclaim ────────────────────────────────
    # When opponent_mobility_after > our_mobility_after, the mobility
    # disadvantage persists.  Phrases that claim it is fully resolved are
    # factual contradictions.  "narrows the gap" is the correct alternative.
    _our_mob_after = facts.get("our_mobility_after")
    if (mob_after is not None and _our_mob_after is not None
            and mob_after > _our_mob_after):
        _mob_overclaim_phrases = [
            "solves the disadvantage",
            "addresses the disadvantage",
            "fixes the disadvantage",
            "eliminates the disadvantage",
            "solves the mobility disadvantage",
            "addresses the mobility disadvantage",
            "eliminates the mobility gap",
            "closes the gap entirely",
            "erases the mobility gap",
        ]
        for _mob_phrase in _mob_overclaim_phrases:
            if _mob_phrase in text:
                warnings.append(
                    f"REASONING_CONTRADICTION: '{_mob_phrase}' overclaims mobility "
                    f"resolution but opponent_mobility_after={mob_after} > "
                    f"our_mobility_after={_our_mob_after} "
                    f"(mobility disadvantage persists)"
                )

    # ── Recapture ─────────────────────────────────────────────────────────────
    recapture = facts.get("opponent_can_recapture")
    if recapture is True:
        safe_phrases = [
            "avoids recapture", "no recapture", "cannot recapture",
            "without recapture risk", "no recapture risk",
            "safe from recapture", "safe move",
        ]
        if any(p in text for p in safe_phrases):
            warnings.append(
                "REASONING_CONTRADICTION: claims avoids recapture but "
                "opponent_can_recapture=true"
            )

    # ── Isolation ─────────────────────────────────────────────────────────────
    isolated = facts.get("leaves_piece_isolated")
    if isolated is True:
        no_isolation_phrases = [
            "does not isolate", "no isolation", "maintains connectivity",
            "piece not isolated", "stays connected",
        ]
        if any(p in text for p in no_isolation_phrases):
            warnings.append(
                "REASONING_CONTRADICTION: claims no isolation but "
                "leaves_piece_isolated=true"
            )

    # ── Immediate threat ──────────────────────────────────────────────────────
    creates_threat = facts.get("creates_immediate_threat")
    if creates_threat is False:
        threat_phrases = [
            "creates a threat", "creates immediate threat",
            "creates an immediate threat",
            # "immediate threat" (bare) removed — fires on correct negations such as
            # "Although this move does not create an immediate threat..."
            "applies pressure next turn", "creates pressure next",
            "threatens opponent", "creates tactical threat",
        ]
        if any(p in text for p in threat_phrases):
            warnings.append(
                "REASONING_CONTRADICTION: claims creates_immediate_threat but "
                "creates_immediate_threat=false"
            )

    # ── BUG-3: Single-legal-move superlative ─────────────────────────────────
    if any("only legal move" in s.lower() for s in (seeds or [])):
        _slm_superlatives = ["strongest choice", "best move", "highest-ranked option"]
        for _slm_phrase in _slm_superlatives:
            if _slm_phrase in text:
                warnings.append(
                    f"REASONING_CONTRADICTION: '{_slm_phrase}' used but this is "
                    f"the only legal move (single_legal_move_context)"
                )
                break

    # ── Center control ────────────────────────────────────────────────────────
    center = facts.get("center_control")
    if center is False:
        center_phrases = [
            "controls the center", "controls center", "central control",
            "occupies the center", "center control=true",
            # Ontology guard: geometric/tactical conflation phrases are also
            # forbidden when center_control=False — they imply tactical control.
            "central board presence", "influence over central",
        ]
        if any(p in text for p in center_phrases):
            warnings.append(
                "REASONING_CONTRADICTION: claims center_control but "
                "center_control=false"
            )
        # BUG-4: "center of the board" as a strategic claim (seed-exempt).
        # The geometric seed "The destination is in the center of the board
        # (column X)" exempts the phrase when destination is in center columns.
        # The pure-geometry form "center of the board (column N)" is also
        # allowed even without seeds — the "(column" qualifier makes it
        # unambiguously geometric.  Only the bare strategic form is flagged.
        if "center of the board" in text and "center of the board" not in seeds_text:
            import re as _re_ctr
            _is_geometric = bool(
                _re_ctr.search(r"center of the board\s*\(\s*column", text)
            )
            if not _is_geometric:
                warnings.append(
                    "REASONING_CONTRADICTION: 'center of the board' used as strategic "
                    "claim but center_control=false and phrase not in seeds "
                    "(factual_contradiction)"
                )

    # ── Capture / material gain ───────────────────────────────────────────────
    captures_count = facts.get("captures_count", 0)
    if captures_count == 0:
        cap_phrases = [
            "captures a piece", "captures the piece", "captures an opponent",
            "captures opponent", "gaining a piece",
        ]
        if any(p in text for p in cap_phrases):
            warnings.append(
                "REASONING_CONTRADICTION: claims capture but captures_count=0"
            )

    # ── (removed: in-file gains_material check) ──────────────────────────────
    # gains_material contradiction detection is now provided by the unified
    # verifier (clause-level negation pre-pass + verify_claims).  Keeping a
    # second copy here would diverge from the evaluator on edge cases like
    # "does not result in a net material gain", breaking the E.1 invariant.
    # The unified-verifier merge below re-emits the equivalent warning string
    # whenever a contradiction is detected.

    # ── Opponent jump count ───────────────────────────────────────────────────
    # BUG-2 (semantic audit): reasoning can claim "single jump" even when the
    # opponent has multiple jump options.  forced_opponent_jump_reply=True only
    # means a jump is mandatory — it says nothing about how many distinct jump
    # moves exist.  Flag the claim when opponent_jump_count > 1.
    opp_jc = facts.get("opponent_jump_count")
    if isinstance(opp_jc, int) and opp_jc > 1:
        _single_jump_phrases = [
            "single jump",
            "one jump option",
            "only one jump",
            "limited to a single jump",
            "limited to one jump",
        ]
        if any(p in text for p in _single_jump_phrases):
            warnings.append(
                f"REASONING_CONTRADICTION: claims single opponent jump but "
                f"opponent_jump_count={opp_jc} (factual_contradiction)"
            )

    # ── Promotion ─────────────────────────────────────────────────────────────
    results_in_king = facts.get("results_in_king", False)
    if not results_in_king:
        promo_phrases = ["promotes to king", "promotes a piece", "crowns a piece", "becomes a king"]
        if any(p in text for p in promo_phrases):
            warnings.append(
                "REASONING_CONTRADICTION: claims promotion but results_in_king=false"
            )

    # ── Block opponent landing ────────────────────────────────────────────────
    blocks_landing = facts.get("blocks_opponent_landing")
    if blocks_landing is False:
        block_phrases = [
            "blocks opponent landing", "blocks the opponent from landing",
            "blocks_opponent_landing=true",
        ]
        if any(p in text for p in block_phrases):
            warnings.append(
                "REASONING_CONTRADICTION: claims blocks_opponent_landing but "
                "blocks_opponent_landing=false"
            )

    # ── Reverse recapture: opponent_can_recapture=False but claims recapture ──
    # The forward direction (recapture=True, claims safe) is handled above.
    # This check covers the reverse: recapture=False but reasoning invents it.
    # Fact-based (not seed-based) so it fires in seed_off ablation too.
    # Bypass: when the phrase is qualified "but not here" / "but not in this case",
    # the author is contrasting the alternative move — do NOT flag.
    if recapture is False:
        _false_recap_phrases = [
            "opponent can recapture",
            "can be recaptured next",
            "vulnerable to recapture",
            "allows the opponent to recapture",
            "opponent may recapture",
            # Hedged surface forms — same factual claim, softer wording.
            "exposed to recapture",
            "exposed to potential recapture",
            "potential recapture",
            "could be recaptured",
            "risk of recapture",
            "recapture risk",
            "recapture risks",
        ]
        import re as _re_recap
        _NEG_RECAP = _re_recap.compile(
            r"\b(no|not|without|cannot|never|nor|none|nothing|neither|"
            r"avoid(?:s|ing)?|prevent(?:s)?|eliminat(?:e|es|ing)|fails?\s+to)\b"
        )
        for _frp in _false_recap_phrases:
            _idx = text.find(_frp)
            if _idx < 0:
                continue
            _window = text[_idx : _idx + 80]
            if "but not" in _window or "not here" in _window:
                continue  # contrast qualifier — describes alternative, not chosen
            # Sentence-level negation guard — phrases like "exposed to recapture"
            # legitimately appear in "without recapture risk" / "not exposed to
            # recapture next turn" prose.  Skip when a negation marker precedes
            # the phrase inside the same sentence.
            _sent_start = max(text.rfind(".", 0, _idx),
                              text.rfind("!", 0, _idx),
                              text.rfind("?", 0, _idx))
            _sent_start = 0 if _sent_start < 0 else _sent_start + 1
            if _NEG_RECAP.search(text[_sent_start:_idx]):
                continue
            warnings.append(
                "REASONING_CONTRADICTION: claims opponent can recapture but "
                "opponent_can_recapture=false"
            )
            break

    # ── False forced-opponent-reply ────────────────────────────────────────────
    # When forced_opponent_jump_reply=False, prose phrases asserting the
    # opponent is forced to respond / has no choice are fabricated.  Mirrors
    # unified_verifier._check_false_forced_opp_reply for E.1 invariant.
    # Sentence-level negation filter: skip when a negation marker precedes
    # the phrase in the same sentence ("does not force the opponent",
    # "without forcing the opponent", etc.).
    _fjr = facts.get("forced_opponent_jump_reply")
    if _fjr is False:
        _false_force_phrases = [
            "forces the opponent",
            "forcing the opponent",
            "forcing them to respond",
            "force the opponent into",
            "opponent must respond",
            "opponent must reply",
            "opponent is forced",
            "forced reply",
            "forced response",
            "compelled to respond",
            "opponent compelled",
            "no choice but to respond",
            "opponent has no choice",
        ]
        import re as _re_fjr
        _NEG_RE = _re_fjr.compile(
            r"\b(no|not|without|cannot|never|nor|none|nothing|neither|"
            r"avoid(?:s|ing)?|prevent(?:s)?|eliminat(?:e|es|ing)|fails?\s+to)\b"
        )
        for _ffp in _false_force_phrases:
            _idx = text.find(_ffp)
            if _idx < 0:
                continue
            # Sentence-level negation: skip when preceded by a negation marker
            # within the same sentence.
            _sent_start = max(text.rfind(".", 0, _idx),
                              text.rfind("!", 0, _idx),
                              text.rfind("?", 0, _idx))
            _sent_start = 0 if _sent_start < 0 else _sent_start + 1
            if _NEG_RE.search(text[_sent_start:_idx]):
                continue
            warnings.append(
                "REASONING_CONTRADICTION: claims forced opponent reply but "
                "forced_opponent_jump_reply=false"
            )
            break

    # ── Near-promotion false claim ─────────────────────────────────────────────
    near_promo = facts.get("near_promotion")
    _results_in_king = facts.get("results_in_king")
    if near_promo is False and not _results_in_king:
        _near_promo_phrases = [
            "near promotion",
            "one step from promotion",
            "one square from promotion",
            "approaching promotion",
            "toward promotion",
            "closing in on promotion",
        ]
        for _npp in _near_promo_phrases:
            _npp_idx = text.find(_npp)
            if _npp_idx < 0:
                continue
            # Bypass if "opponent" appears within 30 chars before the phrase
            # (e.g. "the opponent remains one step from promotion" is correct
            # when opponent_near_promotion=true — it's not about OUR piece).
            _look_back = text[max(0, _npp_idx - 30) : _npp_idx]
            if "opponent" in _look_back:
                continue
            warnings.append(
                "REASONING_CONTRADICTION: claims near promotion but near_promotion=false"
            )
            break

    # ── Wrong capture count ────────────────────────────────────────────────────
    # The zero-capture check above catches false captures; this catches wrong counts
    # when captures_count > 0 (e.g., reasoning says "captures 1" but count is 2).
    if isinstance(captures_count, int) and captures_count > 0:
        import re as _re_cap
        for _cap_m in _re_cap.finditer(
            r'captures?\s+(?:only\s+)?(\d+)\s+piece'
            r'|(\d+)\s+piece[s]?\s+(?:are\s+)?captured'
            r'|capturing\s+(\d+)\s+piece',
            text,
        ):
            _claimed_cap = next(int(g) for g in _cap_m.groups() if g is not None)
            if _claimed_cap != captures_count:
                warnings.append(
                    f"REASONING_CONTRADICTION: claims {_claimed_cap} capture(s) but "
                    f"captures_count={captures_count}"
                )
                break

    # ── False 'only legal move' / forced-move claim ────────────────────────────
    # When seeds say "Multiple legal moves were available", the reasoning must
    # not claim the move was forced, the only option, or the only legal move.
    # Complements BUG-3 (which catches wrong superlatives on single-legal moves).
    if seeds is not None and any(
        "multiple legal moves were available" in s.lower() for s in seeds
    ):
        _false_forced = [
            "only legal move",
            "only available move",
            "only option available",
            "only viable option",
            "the only move",
            "no alternative",
            "forced move",
            "was forced",
        ]
        for _ffp in _false_forced:
            if _ffp in text:
                warnings.append(
                    f"REASONING_CONTRADICTION: '{_ffp}' claimed but multiple "
                    "legal moves were available (forced_move_for_us=false)"
                )
                break

    # ── Fabricated comparison values ───────────────────────────────────────────
    # Pattern: "N points better/stronger/over/above/ahead of" — requires the
    # exact number to appear in seeds.  Fires when the LLM invents comparison
    # magnitudes not backed by any seed.  The comparator group covers the
    # synonyms the audit found the LLM substituting ("over", "above",
    # "ahead of") to dodge the bare "better" form.
    import re as _re_cmp
    for _cmp_m in _re_cmp.finditer(
        r'(\d+\.?\d*)\s+points?\s+(?:better|stronger|over|above|ahead\s+of)', text
    ):
        _cmp_number = _cmp_m.group(1)
        if seeds is None or _cmp_number not in seeds_text:
            warnings.append(
                f"REASONING_CONTRADICTION: fabricated comparison value "
                f"'{_cmp_m.group(0).strip()}' — "
                "no matching comparison seed found"
            )
            break

    # ── Forbidden vocabulary (always prohibited) ───────────────────────────────
    # These phrases are invented concepts that never appear in any seed.
    for phrase in _FORBIDDEN_VOCAB:
        if phrase in text:
            warnings.append(
                f"REASONING_CONTRADICTION: forbidden term '{phrase}' used — "
                "not present in any reasoning seed"
            )

    # ── Context-forbidden vocabulary (prohibited unless explicitly seeded) ─────
    # These are allowed only if the seed list introduced them first.
    # Negation-aware: skip when every occurrence is preceded by a negation
    # marker within 60 chars (e.g. "no new vulnerabilities", "without creating
    # new vulnerabilities", "avoids introducing new vulnerabilities").
    for phrase in _CONTEXT_FORBIDDEN_VOCAB:
        if phrase in text and phrase not in seeds_text:
            if _ctx_phrase_negated(text, phrase):
                continue
            warnings.append(
                f"REASONING_CONTRADICTION: term '{phrase}' used but not in seeds"
            )

    # ── Unsupported numeric statements ────────────────────────────────────────
    # Detects "from X to Y" and "remains at X" / "unchanged" patterns where
    # the specific number cited is not found in the seeds.
    import re as _re

    def _matches_mobility_change(n1: str, n2: str) -> bool:
        """Allow 'from N to M' when (N, M) match a mobility before/after pair
        in the facts dict.  Keeps precision: wrong numbers still flag."""
        if not isinstance(facts, dict):
            return False
        try:
            ni, mi = int(n1), int(n2)
        except (TypeError, ValueError):
            return False
        for b_key, a_key in (
            ("our_mobility_before", "our_mobility_after"),
            ("opponent_mobility_before", "opponent_mobility_after"),
        ):
            b, a = facts.get(b_key), facts.get(a_key)
            if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                if int(b) == ni and int(a) == mi:
                    return True
        return False

    def _matches_mobility_stable(n: str) -> bool:
        """Allow 'remains at N' when N equals before==after for a mobility pair."""
        if not isinstance(facts, dict):
            return False
        try:
            ni = int(n)
        except (TypeError, ValueError):
            return False
        for b_key, a_key in (
            ("our_mobility_before", "our_mobility_after"),
            ("opponent_mobility_before", "opponent_mobility_after"),
        ):
            b, a = facts.get(b_key), facts.get(a_key)
            if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                if int(b) == ni and int(a) == ni:
                    return True
        return False

    # Pattern: "from N to M" where N and M are integers.
    # Kept in sync with unified_verifier._check_mobility_transition so the
    # runtime checker and the evaluator agree on which transitions are
    # legitimate.  A transition is allowed when (N, M) matches ANY known
    # mobility OR piece-count pair (total / regular / kings) in the facts.
    def _matches_piece_count_change(n1: str, n2: str) -> bool:
        if not isinstance(facts, dict):
            return False
        try:
            ni, mi = int(n1), int(n2)
        except (TypeError, ValueError):
            return False
        for prefix in ("opp_pieces", "our_pieces"):
            b = facts.get(f"{prefix}_before")
            a = facts.get(f"{prefix}_after")
            if isinstance(b, dict) and isinstance(a, dict):
                for k in ("total", "regular", "kings"):
                    if k in b and k in a:
                        try:
                            if int(b[k]) == ni and int(a[k]) == mi:
                                return True
                        except (TypeError, ValueError):
                            pass
        return False

    for m in _re.finditer(r'from\s+(\d+)\s+to\s+(\d+)', text):
        n1, n2 = m.group(1), m.group(2)
        # Both numbers must appear somewhere in the seeds OR match an actual
        # mobility OR piece-count before/after pair in the facts.
        if (
            (n1 not in seeds_text or n2 not in seeds_text)
            and not _matches_mobility_change(n1, n2)
            and not _matches_piece_count_change(n1, n2)
        ):
            warnings.append(
                f"REASONING_CONTRADICTION: unsupported numeric statement "
                f"'from {n1} to {n2}' — value(s) not found in seeds"
            )
    # Pattern: "remains at N" or "stays at N" (single number assertion).
    # Allow when N equals before==after for a mobility OR piece-count pair.
    def _matches_piece_count_stable(n: str) -> bool:
        if not isinstance(facts, dict):
            return False
        try:
            ni = int(n)
        except (TypeError, ValueError):
            return False
        for prefix in ("opp_pieces", "our_pieces"):
            b = facts.get(f"{prefix}_before")
            a = facts.get(f"{prefix}_after")
            if isinstance(b, dict) and isinstance(a, dict):
                for k in ("total", "regular", "kings"):
                    if k in b and k in a:
                        try:
                            if int(b[k]) == ni and int(a[k]) == ni:
                                return True
                        except (TypeError, ValueError):
                            pass
        return False

    for m in _re.finditer(r'(?:remains?\s+at|stays?\s+at|consistent\s+at|maintain(?:s|ing)?\s+(?:[^.,;]*?)\s+at)\s+(\d+)', text):
        n = m.group(1)
        if (
            n not in seeds_text
            and not _matches_mobility_stable(n)
            and not _matches_piece_count_stable(n)
        ):
            warnings.append(
                f"REASONING_CONTRADICTION: unsupported numeric assertion "
                f"'remains at {n}' — value not found in seeds"
            )

    # ── Unsupported absence claims ─────────────────────────────────────────────
    # Flags specific absence claims that require explicit seed support.
    absence_phrases: list[tuple[str, str]] = [
        ("no kings lost",       "no_kings_lost seed"),
        ("piece count unchanged", "piece_count seed"),
        ("pieces unchanged",    "piece_count seed"),
        ("no vulnerabilities",  "no_vulnerabilities seed"),
    ]
    for phrase, seed_required in absence_phrases:
        if phrase in text and phrase not in seeds_text:
            warnings.append(
                f"REASONING_CONTRADICTION: unsupported absence claim '{phrase}' — "
                f"requires explicit {seed_required}"
            )

    # ── Direct inversion detection (seed says X=true → reasoning says X=false) ─
    # Only fires when seeds are provided and the seed explicitly states a boolean.
    if seeds:
        # Comparison seeds (e.g. "Move [1] isolates the moved piece
        # (leaves_piece_isolated=true vs false here)") describe the ALTERNATIVE
        # move's facts, not the chosen move's facts.  Including them in seeds_text
        # causes false-positive inversions: the checker sees "leaves_piece_isolated=true"
        # and fires when the reasoning correctly says "stays connected" about the
        # chosen move (which has isolated=false).
        # Fix: for the inversion check only, exclude seeds that start with "Move [N]".
        # Exclude comparison seeds that describe the alternative move, not the chosen move.
        # Old format started with "move [N]"; new format starts with "unlike move [N]".
        _chosen_only_seeds_text = " ".join(
            s.lower() for s in seeds
            if not (s.lower().strip().startswith("move [")
                    or s.lower().strip().startswith("unlike move ["))
        )
        _INVERSION_PAIRS: list[tuple[str, str, str]] = [
            # (seed phrase indicating TRUE/FALSE state, reasoning phrase implying opposite, label)
            # Recapture
            ("opponent can recapture the moved piece",  "no recapture",           "opponent_can_recapture"),
            ("opponent can recapture the moved piece",  "avoids recapture",       "opponent_can_recapture"),
            ("opponent can recapture the moved piece",  "safe from recapture",    "opponent_can_recapture"),
            ("cannot be immediately recaptured",         "opponent can recapture", "opponent_can_recapture"),
            # Isolation
            ("left without adjacent support",    "no isolation",           "leaves_piece_isolated"),
            ("left without adjacent support",    "stays connected",        "leaves_piece_isolated"),
            ("left without adjacent support",    "maintains connectivity", "leaves_piece_isolated"),
            ("left without adjacent support",    "does not isolate",       "leaves_piece_isolated"),
            ("left without adjacent support",    "avoids isolation",       "leaves_piece_isolated"),
            ("left without adjacent support",    "remains connected",      "leaves_piece_isolated"),
            ("left without adjacent support",    "maintains structure",    "leaves_piece_isolated"),
            ("left without adjacent support",    "keeping connectivity",   "leaves_piece_isolated"),
            ("is not left isolated",  "isolates the piece",  "leaves_piece_isolated"),
            ("is not left isolated",  "piece is isolated",   "leaves_piece_isolated"),
            # Immediate threat (True only — False is caught by factual checker; no False seed emitted)
            ("forces the opponent to respond to an immediate threat",  "no immediate threat", "creates_immediate_threat"),
            # Moved piece threatened (True only — no False seed emitted)
            ("remains under immediate threat",  "piece is safe",  "moved_piece_is_threatened"),
            # King row weakening
            ("weakening the defensive structure",         "back-row discipline maintained", "weakens_king_row"),
            ("weakening the defensive structure",         "preserves back row",             "weakens_king_row"),
            ("weakens the back-row defensive structure",  "back-row discipline maintained", "weakens_king_row"),
            ("weakens the back-row defensive structure",  "preserves back row",             "weakens_king_row"),
            ("defensive structure remains intact",  "weakens the back row",  "weakens_king_row"),
            ("defensive structure remains intact",  "back-row weakened",     "weakens_king_row"),
        ]
        for seed_marker, contradiction_phrase, label in _INVERSION_PAIRS:
            seed_says_it = seed_marker in _chosen_only_seeds_text
            reasoning_contradicts = contradiction_phrase in text
            if seed_says_it and reasoning_contradicts:
                warnings.append(
                    f"REASONING_CONTRADICTION: inversion detected — "
                    f"seed states '{seed_marker}' but reasoning says '{contradiction_phrase}' "
                    f"(field: {label})"
                )

    # ── Semantic numeric / quantity leakage detection ────────────────────────────
    # Flags quantity claims that are not backed by verbatim numbers in the seeds.
    # Exact numbers that appear in seeds are allowed (mobility before/after, etc.).

    # Unchanged-mobility support: when facts confirm opp or our mobility did not
    # change (before == after), the four "unchanged mobility" patterns below are
    # valid claims and must NOT fire as false positives.  Seeds say "no change in
    # opponent mobility" — not "unchanged mobility" — so verbatim matching would
    # incorrectly flag them.  Gating on facts is precise and avoids weakening the
    # general hallucination check.
    _opp_mob_unchanged = (
        mob_before is not None and mob_after is not None and mob_before == mob_after
    )
    _our_mb = facts.get("our_mobility_before")
    _our_ma = facts.get("our_mobility_after")
    _our_mob_unchanged = (
        _our_mb is not None and _our_ma is not None and _our_mb == _our_ma
    )
    # Labels of the four patterns that should be skipped when mobility is confirmed
    # unchanged.  Explicitly listed so no other patterns are accidentally suppressed.
    _MOBILITY_UNCHANGED_LABELS = frozenset({
        "unsupported 'unchanged mobility' claim",
        "unsupported 'mobility unchanged' claim",
        "unsupported 'same number of' claim",
        "unsupported 'same move count' claim",
    })

    # ── Number-word → integer lookup (one–ten) ────────────────────────────────
    # Used exclusively by the before→after number-word checker below.
    # No impact on any decision, repair, or selection logic.
    _WORD_TO_INT: dict[str, int] = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    # Known before/after fact pairs that the LLM is allowed to express in
    # number-word form.  If (parsed_word1, parsed_word2) == (before, after)
    # for any pair, the "from X to Y" claim is factually grounded.
    _BEFORE_AFTER_PAIRS: list[tuple[str, str]] = [
        ("opponent_mobility_before", "opponent_mobility_after"),
        ("our_mobility_before",      "our_mobility_after"),
        ("our_pieces_threatened_before", "our_pieces_threatened_after"),
    ]

    _numeric_patterns: list[tuple[str, str]] = [
        # Before→after quantity narratives (number-word form only; digit form
        # is handled separately above by the r'from\s+(\d+)\s+to\s+(\d+)' loop)
        (
            r"from\s+(three|four|five|six|seven|eight|nine|ten)\s+to\s+"
            r"(one|two|three|four|five|six|seven|eight|nine)",
            "unsupported before→after numeric claim (e.g. 'from three to two')",
        ),
        # Specific reply-count claims
        (
            r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
            r"\s+safe\s+repl(?:y|ies)",
            "unsupported specific safe-reply count",
        ),
        # Mobility unchanged assertions
        (
            r"unchanged\s+mobility",
            "unsupported 'unchanged mobility' claim",
        ),
        (
            r"mobility\s+(?:remains|remain|stays|stays at)\s+unchanged",
            "unsupported 'mobility unchanged' claim",
        ),
        # Same-number assertions
        (
            r"same\s+number\s+of\s+(?:replies|moves|options|pieces)",
            "unsupported 'same number of' claim",
        ),
        (
            r"(?:opponent\s+)?(?:retains|maintains)\s+(?:the\s+)?"
            r"(?:same|equal)\s+(?:number|count)\s+of\s+(?:moves|replies|options)",
            "unsupported 'same move count' claim",
        ),
    ]
    import re as _re_num
    for _pat, _label in _numeric_patterns:
        # Skip unchanged-mobility patterns when facts explicitly confirm no change.
        # Use context cues in the text to identify which mobility type is claimed:
        #   "opponent"/"their"/"foe" → check opponent mobility
        #   "our"/"my"/"we"/"own"   → check our mobility
        #   ambiguous               → suppress when EITHER mobility is confirmed unchanged
        if _label in _MOBILITY_UNCHANGED_LABELS:
            _opp_ctx = any(w in text for w in ("opponent", "their", "enemy", "foe"))
            _our_ctx = any(w in text for w in (" our ", "my ", " we ", "own "))
            if _opp_ctx and not _our_ctx:
                if _opp_mob_unchanged:
                    continue
            elif _our_ctx and not _opp_ctx:
                if _our_mob_unchanged:
                    continue
            else:  # ambiguous: suppress when either is confirmed unchanged
                if _opp_mob_unchanged or _our_mob_unchanged:
                    continue
        _match = _re_num.search(_pat, text, _re_num.IGNORECASE)
        if not _match:
            continue
        _matched_text = _match.group(0)

        # ── Verbatim seed bypass (existing behaviour for all patterns) ────────
        if seeds and _matched_text.lower() in seeds_text:
            continue

        # ── Fact-grounded bypass for number-word before→after claims ─────────
        # If both captured groups are number words (groups 1 and 2 exist) and
        # the parsed integers match a known before/after fact pair, the claim is
        # factually supported — do NOT emit the warning.
        # This handles the common LLM surface form "from six to four" when
        # seeds only contain digit form ("opponent_mobility_before=6, …after=4").
        if (
            _label == "unsupported before→after numeric claim (e.g. 'from three to two')"
            and _match.lastindex is not None
            and _match.lastindex >= 2
        ):
            _w1 = _match.group(1).lower()
            _w2 = _match.group(2).lower()
            _i1 = _WORD_TO_INT.get(_w1)
            _i2 = _WORD_TO_INT.get(_w2)
            if _i1 is not None and _i2 is not None:
                _fact_grounded = any(
                    facts.get(_bf) == _i1 and facts.get(_af) == _i2
                    for _bf, _af in _BEFORE_AFTER_PAIRS
                )
                # Also allow piece-count transitions (total / regular / kings)
                # — kept in sync with unified_verifier._check_mobility_transition.
                if not _fact_grounded:
                    for _prefix in ("opp_pieces", "our_pieces"):
                        _b = facts.get(f"{_prefix}_before")
                        _a = facts.get(f"{_prefix}_after")
                        if isinstance(_b, dict) and isinstance(_a, dict):
                            for _k in ("total", "regular", "kings"):
                                if _k in _b and _k in _a:
                                    try:
                                        if int(_b[_k]) == _i1 and int(_a[_k]) == _i2:
                                            _fact_grounded = True
                                            break
                                    except (TypeError, ValueError):
                                        pass
                        if _fact_grounded:
                            break
                if _fact_grounded:
                    continue  # Factually grounded transition — suppress warning

        # ── Fact-grounded bypass for "N safe replies" claims ───────────────
        # Kept in sync with unified_verifier._check_safe_reply_count: when
        # the asserted count equals facts["opponent_safe_reply_count"], the
        # claim is supported — do not emit the warning.
        if _label == "unsupported specific safe-reply count":
            _osrc = facts.get("opponent_safe_reply_count")
            if isinstance(_osrc, (int, float)):
                _toks = _re_num.findall(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", _matched_text.lower())
                _asserted_int = None
                for _t in _toks:
                    try:
                        _asserted_int = int(_t)
                        break
                    except ValueError:
                        if _t in _WORD_TO_INT:
                            _asserted_int = _WORD_TO_INT[_t]
                            break
                if _asserted_int is not None and int(_osrc) == _asserted_int:
                    continue  # Asserted count matches fact — suppress warning

        warnings.append(
            f"REASONING_CONTRADICTION: {_label} — not found in seeds"
        )

    # ── Fix A2: Must-mention check for our-mobility decrease ─────────────────
    # If any seed explicitly states that our mobility decreases, the reasoning
    # must acknowledge it.  Omitting a seeded negative fact is misleading.
    _mob_decrease_seed = next(
        (s for s in (seeds or []) if "decreases our mobility by" in s.lower()),
        None,
    )
    if _mob_decrease_seed is not None:
        _mob_mention_phrases = [
            "decreases", "decrease", "our mobility drops", "our mobility falls",
            "reduces our mobility", "reducing our mobility",
            "our mobility decreases",
            "our mobility narrows", "our mobility shrinks",
            "our mobility contracts",
            "our mobility goes down", "losing mobility",
        ]
        if not any(p in text for p in _mob_mention_phrases):
            warnings.append(
                "REASONING_CONTRADICTION: our-mobility decrease seeded but "
                "omitted from reasoning (negative_fact_omission)"
            )

    # ── Fix A3: any_piece_isolated contradicts "no vulnerabilities" ──────────
    # any_piece_isolated=True means some ally piece is isolated after the move.
    # Claiming "no tactical vulnerabilities" when any piece is isolated is false.
    _any_iso = facts.get("any_piece_isolated")
    if _any_iso is True:
        _iso_vuln_phrases = [
            "no tactical vulnerabilities",
            "ensuring no tactical",
            "no vulnerabilities are created",
            "no tactical vulnerabilities are created",
        ]
        for _ivp in _iso_vuln_phrases:
            if _ivp in text:
                warnings.append(
                    f"REASONING_CONTRADICTION: 'any_piece_isolated=true' contradicts "
                    f"'{_ivp}' claim (factual_contradiction)"
                )
                break

    # ── Fix A4: "narrowing the gap" when our mobility >= opponent mobility ────
    # "Narrowing the gap" implies the opponent still leads.  When
    # our_mobility_after >= opponent_mobility_after the gap was matched or
    # reversed — "narrowing" is factually wrong.
    _our_mob_af  = facts.get("our_mobility_after")
    _opp_mob_af  = facts.get("opponent_mobility_after")
    if (
        isinstance(_our_mob_af, (int, float))
        and isinstance(_opp_mob_af, (int, float))
        and _our_mob_af >= _opp_mob_af
        and "narrowing the gap" in text
    ):
        warnings.append(
            f"REASONING_CONTRADICTION: 'narrowing the gap' is wrong when "
            f"our_mobility_after={int(_our_mob_af)} >= "
            f"opponent_mobility_after={int(_opp_mob_af)} "
            f"(gap was matched or reversed, not narrowed)"
        )

    # ── Phase G, Step 3: mobility-direction phrase verifier (mirror) ────────
    # Three exact phrases checked against engine mobility deltas.  Complementary
    # to the A4 'narrowing the gap' check above — different verb forms, no
    # overlap.  Maintains E.1 parity with unified_verifier.
    _ub = facts.get("our_mobility_before"); _ua = facts.get("our_mobility_after")
    _ob = facts.get("opponent_mobility_before"); _oa = facts.get("opponent_mobility_after")
    if all(isinstance(_x, (int, float)) for _x in (_ub, _ua, _ob, _oa)):
        import re as _re_mob
        _UNCHANGED_RE = _re_mob.compile(
            # Two syntactic forms; side-qualified phrases excluded on each arm.
            r"(?:"
            # (1) Noun-first: "mobility (verb) (complement)"
            r"(?<!our\s)(?<!opponent\s)"
            r"\bmobility\s+(?:remained?|stays?|is|does\s+not\s+(?:alter|change))\s+"
            r"(?:unchanged|the\s+same|intact|for\s+both\s+sides)"
            r"|"
            # (2) Verb-first: "(does/did not | doesn't/didn't) (alter|change|
            #     affect|modify) mobility"  — excludes 'our mobility' /
            #     'opponent mobility' via negative lookahead.
            r"\b(?:does\s+not|did\s+not|doesn't|didn't)\s+"
            r"(?:alter|change|affect|modify)\s+"
            r"(?!our\b)(?!opponent\b)(?:the\s+)?(?:overall\s+)?mobility"
            r")",
            _re_mob.IGNORECASE,
        )
        # Matches both verb orders:
        #   "gap narrows / narrowed / narrowing"       (noun → verb)
        #   "narrows / narrowed / narrowing the gap"   (verb → noun)
        _NARROW_RE = _re_mob.compile(
            r"\b(?:"
            r"(?:mobility\s+)?gap\s+narrow(?:s|ed|ing)"
            r"|narrow(?:s|ed|ing)\s+(?:the\s+)?(?:mobility\s+)?gap"
            r")\b",
            _re_mob.IGNORECASE,
        )
        _WIDEN_RE = _re_mob.compile(
            r"\b(?:"
            r"(?:mobility\s+)?gap\s+widen(?:s|ed|ing)"
            r"|widen(?:s|ed|ing)\s+(?:the\s+)?(?:mobility\s+)?gap"
            r")\b",
            _re_mob.IGNORECASE,
        )
        if _UNCHANGED_RE.search(reasoning):
            if int(_ub) != int(_ua) or int(_ob) != int(_oa):
                warnings.append(
                    f"REASONING_CONTRADICTION: claims mobility unchanged but "
                    f"our_mobility={int(_ub)}->{int(_ua)} and "
                    f"opponent_mobility={int(_ob)}->{int(_oa)} "
                    f"(mobility_unchanged_misclaim)"
                )
        _gap_b = abs(int(_ub) - int(_ob))
        _gap_a = abs(int(_ua) - int(_oa))
        if _NARROW_RE.search(reasoning) and not (_gap_a < _gap_b):
            warnings.append(
                f"REASONING_CONTRADICTION: claims 'gap narrowed' but "
                f"|gap_before|={_gap_b} and |gap_after|={_gap_a} "
                f"(gap_did_not_narrow)"
            )
        if _WIDEN_RE.search(reasoning) and not (_gap_a > _gap_b):
            warnings.append(
                f"REASONING_CONTRADICTION: claims 'gap widened' but "
                f"|gap_before|={_gap_b} and |gap_after|={_gap_a} "
                f"(gap_did_not_widen)"
            )

    # ── B1.1: Comparative recapture fabrication ──────────────────────────────
    # When the chosen move can be recaptured (opponent_can_recapture=True),
    # comparative-context phrases that claim recapture safety are fabricated.
    _b11_phrases = (
        "recapture safety", "avoiding recapture", "avoid recapture risk",
        "recapture-safe", "recapture safety edge",
    )
    if recapture is True:
        for _b11p in _b11_phrases:
            if _b11p in text:
                warnings.append(
                    f"COMPARATIVE_CONTRADICTION: '{_b11p}' claimed but "
                    f"opponent_can_recapture=true (fabricated_claim)"
                )
                break

    # ── B1.2: Tradeoff language requires numeric grounding ────────────────────
    # "outweighs", "compensates for", "offsets the" imply a quantitative trade.
    # If no explicit number appears in the same sentence the claim is unverifiable.
    import re as _re_b12
    _b12_tradeoff = ("outweighs", "compensates for", "offsets the")
    _b12_has_num = _re_b12.compile(
        r'\b\d+(?:\.\d+)?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b',
        _re_b12.IGNORECASE,
    )
    for _b12_sent in _re_b12.split(r'(?<=[.!?])\s+', text):
        _b12_has_tradeoff = any(p in _b12_sent for p in _b12_tradeoff)
        if _b12_has_tradeoff and not _b12_has_num.search(_b12_sent):
            _b12_which = next(p for p in _b12_tradeoff if p in _b12_sent)
            warnings.append(
                f"COMPARATIVE_CONTRADICTION: '{_b12_which}' used without numeric "
                f"grounding in same sentence (tradeoff_without_evidence)"
            )
            break

    # ── B1.3: Negative-score absolute advantage protection ────────────────────
    # When minimax_score < 0, absolute advantage phrases ("positional advantage",
    # "strongest option", etc.) are misleading unless paired with relative framing
    # ("best available", "least unfavorable", …).
    _b13_mm = facts.get("minimax_score")
    if isinstance(_b13_mm, (int, float)) and _b13_mm < 0:
        _b13_forbidden = (
            "positional advantage", "advantage gained",
            "strongest option", "decisive advantage",
        )
        _b13_relative = (
            "best available", "least unfavorable", "least harmful",
            "highest-evaluated", "relative to", "best of the",
            "only option", "best option available",
        )
        _b13_has_relative = any(rp in text for rp in _b13_relative)
        if not _b13_has_relative:
            for _b13p in _b13_forbidden:
                if _b13p in text:
                    warnings.append(
                        f"COMPARATIVE_CONTRADICTION: '{_b13p}' used when "
                        f"minimax_score={float(_b13_mm):.1f} < 0 without relative "
                        f"framing (misleading_advantage_claim)"
                    )
                    break

    # ── B2.1b: Deliberate-choice framing in forced-move context ──────────────
    # "drives the decision", "chosen for its", etc. imply voluntary selection
    # but the context is a forced move — these phrases are contradictory.
    _forced_move_seed = any("only legal move" in s.lower() for s in (seeds or []))
    if _forced_move_seed:
        _deliberate_phrases = (
            "drives the decision", "chosen for its", "was chosen for",
            "selected for its", "was preferred because",
        )
        for _dcp in _deliberate_phrases:
            if _dcp in text:
                warnings.append(
                    f"REASONING_CONTRADICTION: '{_dcp}' deliberate-choice framing "
                    f"in forced-move context (forced_move_deliberate_framing)"
                )
                break

    # ── B2.3: Geometric impossibility ────────────────────────────────────────
    # A legal move always moves a piece; these phrases are geometrically false.
    _geo_impossible_phrases = (
        "piece remains stationary",
        "no piece movement occurred",
        "piece did not move",
    )
    for _gip in _geo_impossible_phrases:
        if _gip in text:
            warnings.append(
                f"REASONING_CONTRADICTION: '{_gip}' is geometrically impossible "
                f"for a legal move (geometric_impossibility)"
            )
            break

    # ── B2.5: Our-mobility directional consistency ────────────────────────────
    # When our_mobility_after <= our_mobility_before, claiming an increase is false.
    _our_mb_b25 = facts.get("our_mobility_before")
    _our_ma_b25 = facts.get("our_mobility_after")
    if (isinstance(_our_mb_b25, (int, float)) and isinstance(_our_ma_b25, (int, float))
            and _our_ma_b25 <= _our_mb_b25):
        _our_mob_increase_phrases = (
            "increases our mobility", "improves our mobility",
            "our mobility increases", "our mobility improves",
            "our mobility grows", "expands our mobility",
        )
        for _omip in _our_mob_increase_phrases:
            if _omip in text:
                warnings.append(
                    f"REASONING_CONTRADICTION: claims our-mobility increase but "
                    f"our_mobility_after={int(_our_ma_b25)} <= "
                    f"our_mobility_before={int(_our_mb_b25)} (our_mobility_direction)"
                )
                break

    # ── B2.6: Tactical move defensive framing ─────────────────────────────────
    # When creates_immediate_threat=True, claiming "no pressure" is a contradiction.
    _creates_threat_b26 = facts.get("creates_immediate_threat")
    if _creates_threat_b26 is True:
        _no_pressure_phrases = (
            "no tactical pressure", "applies no pressure",
            "creates no pressure", "no immediate pressure",
        )
        for _npp in _no_pressure_phrases:
            if _npp in text:
                warnings.append(
                    f"REASONING_CONTRADICTION: '{_npp}' framing contradicts "
                    f"creates_immediate_threat=true (tactical_move_defensive_framing)"
                )
                break

    # ── E.1 unification: merge unified-verifier findings ─────────────────────
    # The unified verifier runs extract_claims + verify_claims (with the
    # clause-level negation pre-pass), the numeric verifier (E.3), and the
    # schema-leak detector (E.4) against the SAME reasoning text.  Any
    # CONTRADICTED claim found there is appended as a warning string in the
    # same "REASONING_CONTRADICTION: …" shape the refinement loop already
    # consumes.  After this merge, runtime contradictions ⊇ evaluator
    # contradictions, which is exactly what the E.1 invariant requires.
    try:
        from checkers.evaluation.unified_verifier import contradiction_strings as _unified_strings
        _unified = _unified_strings(
            reasoning,
            reasoning_seeds=list(seeds or []),
            facts=facts,
            context=None,
        )
        # Deduplicate against existing warnings (string equality).
        _existing = set(warnings)
        for _w in _unified:
            if _w not in _existing:
                warnings.append(_w)
                _existing.add(_w)
    except Exception as _e:
        # The unified verifier must never crash the runtime refinement loop.
        # If it raises, log and proceed with the legacy warnings only.
        print(f"[EXPLAINER_TRUTHFULNESS] unified verifier error (ignored): {_e}")

    return warnings


# ── Reasoning-only refinement loop ───────────────────────────────────────────

EXPLAINER_REASONING_REFINEMENT_SYSTEM: str = """\
You are a checkers move explainer. Your previous reasoning paragraph contained
false claims that contradict the engine-computed facts for the chosen move.

Your task: rewrite ONLY the reasoning paragraph.
Do NOT change the chosen move. Do NOT suggest a different move.

Rules:
  - Write a single coherent paragraph of 3-5 sentences.
  - Do NOT use labeled section headers.
  - Do NOT repeat any of the false claims listed in the feedback below.
  - Only state a claim if the facts explicitly support it.
  - minimax_score may appear ONLY in the final sentence as confirmation.

NATURALNESS AND CLARITY — improve quality while correcting errors:
  - Preserve and strengthen causal connections: explain WHY the move is preferred,
    not just WHAT it does.
  - Vary sentence openers — do NOT start consecutive sentences with the same word
    or phrase.
  - Do NOT open with "Despite", "Additionally", "Furthermore" more than once.
  - Do NOT close with filler ("This makes it the best choice.", "Overall, ...",
    "In summary, ...", "Therefore, this is the optimal move.").
  - Replace mechanical corrections ("The move does not capture") with natural
    explanations of the actual situation.

FORBIDDEN VOCABULARY — never use any of the following terms or phrases:
  conversion potential, winning conversion, trade conversion, conversion score,
  quiet_move_role, winning_conversion_score, king_activity_score,
  escape squares, escape routes, king escape, king distance,
  diagonal pressure, diagonal risks, long diagonal,
  strategic goal, positional adjustment, real trap, no new vulnerabilities,
  counterplay_score, coordination score, activity score, king activity score,
  quiet move role, regulars_captured, new vulnerabilities.

TRUTHFULNESS RULES — only state a claim if the facts explicitly support it:
  - Only say "reduces mobility" if opponent_mobility_after < opponent_mobility_before.
  - Only say "avoids recapture" / "no recapture risk" if opponent_can_recapture = false.
  - Only say "does not isolate" / "maintains connectivity" if leaves_piece_isolated = false.
  - Only say "creates a threat" / "applies pressure" if creates_immediate_threat = true.
  - Only say "controls the center" if center_control = true.
  - Only say "captures" / "gains material" if captures_count > 0.
  - Only say "promotes" / "crowns" if results_in_king = true.
  - INVERSION CHECK: if a seed says X=true, do NOT write that X is false or absent.
    If a seed says X=false, do NOT write that X is true or present.

OUTPUT FORMAT — reply with ONLY this JSON object, no markdown:
{"reasoning": "<rewritten paragraph>"}
"""


def _relevant_facts_summary(facts: dict) -> list[str]:
    """Return human-readable lines for the most decision-relevant move facts."""
    lines: list[str] = []
    cap = facts.get("captures_count", 0)
    net = facts.get("net_gain", 0)
    if cap > 0:
        lines.append(f"  captures {cap} piece(s), net gain {net}")
    else:
        lines.append("  no captures (positional move)")
    recap = facts.get("opponent_can_recapture")
    if recap is True:
        lines.append("  opponent CAN recapture next turn")
    elif recap is False:
        lines.append("  opponent cannot recapture next turn")
    pta = facts.get("our_pieces_threatened_after")
    if pta is not None:
        lines.append(f"  {pta} allied piece(s) threatened after the move")
    threat = facts.get("creates_immediate_threat")
    if threat is True:
        lines.append("  creates an immediate threat")
    elif threat is False:
        lines.append("  does not create an immediate threat")
    isolated = facts.get("leaves_piece_isolated")
    if isolated is True:
        lines.append("  moved piece is left unsupported")
    elif isolated is False:
        lines.append("  moved piece remains supported by adjacent allies")
    mob_b = facts.get("opponent_mobility_before")
    mob_a = facts.get("opponent_mobility_after")
    if mob_b is not None and mob_a is not None:
        if mob_a != mob_b:
            lines.append(f"  opponent mobility: {mob_b} → {mob_a}")
        else:
            lines.append(f"  opponent mobility unchanged ({mob_b})")
    mm = facts.get("minimax_score")
    if mm is not None:
        lines.append(f"  engine score: {mm:.1f}")
    return lines


def _build_refinement_prompt(
    chosen_move: dict,
    contradictions: list[str],
) -> str:
    """
    Build the user-prompt for a reasoning-only refinement call.
    Shows the LLM the chosen move path, its facts, and the exact
    contradiction list.  The LLM must NOT alter the chosen move.
    """
    path  = chosen_move.get("path", [])
    mtype = chosen_move.get("type", "simple")
    facts = chosen_move.get("facts") or {}

    lines: list[str] = [
        f"Chosen move: type={mtype}  path={path}",
        "",
        "Key facts about the chosen move (use these — do not invent other values):",
    ]
    lines.extend(_relevant_facts_summary(facts))

    lines += [
        "",
        "Your previous reasoning contained the following false claims:",
    ]
    for c in contradictions:
        claim = c.replace("REASONING_CONTRADICTION: ", "")
        lines.append(f"  - {claim}")

    # When any contradiction is a minimax mismatch, prepend an explicit
    # correction block before the generic instructions.  The LLM otherwise
    # tends to repeat the previously-written wrong number because the
    # contradiction message says "fact disagrees" without citing the right
    # value, and the facts summary labels the value as "engine score" while
    # the prose uses "minimax_score" — a vocabulary disconnect that this
    # block resolves explicitly.
    _mm_fact = facts.get("minimax_score")
    _has_minimax_mismatch = any(
        "minimax_score" in c.lower() and "mismatch" in c.lower()
        for c in contradictions
    )
    if _has_minimax_mismatch and isinstance(_mm_fact, (int, float)):
        lines += [
            "",
            "MINIMAX CORRECTION (mandatory):",
            f"  The ONLY correct value is minimax_score = {_mm_fact:.1f}.",
            "  In the rewrite, if you mention the engine's evaluation:",
            "    - Use exactly this value, written either as a bare number "
            f"({_mm_fact:.1f}) or as the phrase 'minimax score of {_mm_fact:.1f}'.",
            "    - Do NOT cite any other number for minimax_score.  Any other",
            "      value (including 0, 0.0, 0.1, 0.12, 0.45, etc.) is forbidden.",
            "    - Do NOT copy the wrong number from the previous reasoning.",
        ]

    lines += [
        "",
        "Instructions:",
        "  1. Rewrite the reasoning paragraph to remove every false claim above.",
        "  2. Keep the same chosen move — do NOT suggest a different move.",
        "  3. Do not mention any claim not supported by the facts listed above.",
        "  4. Minimax_score may appear ONLY in the final sentence as confirmation.",
        "     Do NOT write numeric score-gap claims (e.g., 'X points better than",
        "     alternatives' or 'by a margin of X') — cite only the engine score above.",
        "  5. Do NOT add new recapture claims, opponent-mobility claims, or numeric",
        "     score-comparison claims unless the corresponding fact is explicitly",
        "     listed in Key Facts above.",
        "  6. Prefer fixing only the specific false claim; do not rewrite surrounding",
        "     sentences unnecessarily.",
        "  7. Preserve all grounded seed-backed factual claims unless the fact",
        "     itself is the flagged contradiction. In particular, never remove",
        "     or omit:",
        "       - only-legal-move / forced-move disclosures",
        "       - must-capture disclosures",
        "       - explicit mobility transitions ('from N to M')",
        "       - immediate king-promotion facts",
        "       - grounded comparative anchors tied to specific alternatives",
        "     You may paraphrase these facts, but you may not silently drop them.",
        "  8. Forced-move framing rule (strict): IF AND ONLY IF the original",
        "     reasoning already contained a forced-move disclosure grounded in a",
        "     seed that explicitly states the move is the only legal move available",
        "     (e.g., 'only legal move', 'mandatory jump', 'must capture', or",
        "     equivalent forced-move wording), you must preserve that disclosure",
        "     in the rewrite. Otherwise, do NOT introduce or invent any claim that",
        "     the move is forced, only-legal, mandatory, or has no alternative. A",
        "     seed saying 'the opponent is forced to respond' describes the",
        "     OPPONENT's reply, not our choice, and does NOT make our move forced.",
        "  9. Write a single coherent paragraph of 3-5 sentences.",
        " 10. Do not use vague positional descriptors — 'board control', 'piece"
        "     coordination', 'board cohesion', 'connectivity' — that are not"
        "     directly supported by a fact listed above.",
        " 11. AUDIT-DRIVEN ANTI-HALLUCINATION RULES (must not appear in the rewrite",
        "     unless the corresponding fact in Key Facts above is True):",
        "       - 'creates an immediate threat' / 'forces the opponent' /",
        "         'creates pressure' — requires creates_immediate_threat=true",
        "         OR forced_opponent_jump_reply=true.",
        "       - 'only legal move' / 'forced move' / 'no alternative' /",
        "         'must play' — requires an explicit only-legal-move disclosure",
        "         in the original reasoning grounded in a seed.",
        "       - 'controls the center' / 'central control' / 'central board",
        "         presence' / 'centralizes' — requires center_control=true.",
        "       - 'reduces opponent mobility' / 'restricts opponent mobility' /",
        "         'narrows opponent options' — requires",
        "         opponent_mobility_after < opponent_mobility_before.",
        "       - 'N points better' / 'best by N' — requires an explicit",
        "         comparison number grounded in the seed list.",
        "       - 'near promotion' / 'one step from promotion' / 'toward",
        "         promotion' — requires near_promotion=true or results_in_king=true.",
        "     Replace any such phrase with the corresponding factual statement",
        "     (e.g., 'does not create an immediate threat',",
        "     'opponent mobility is unchanged at N'). Preserve the move's actual",
        "     factual rationale; do not introduce new strategic interpretation.",
        "",
        'Reply with ONLY: {"reasoning": "<your rewritten paragraph>"}',
    ]
    return "\n".join(lines)


def _extract_refinement_reasoning(raw: str) -> Optional[str]:
    """
    Parse the reasoning string from a refinement LLM response.
    Accepts {"reasoning": "..."} JSON.  Falls back to regex.
    Never raises.
    """
    import json as _json
    try:
        obj = _json.loads(raw.strip())
        r = obj.get("reasoning", "")
        if isinstance(r, str) and r.strip():
            return r.strip()
    except Exception:
        pass
    m = re.search(r'"reasoning"\s*:\s*"((?:\\.|[^"\\])*)"', raw, re.DOTALL)
    if m:
        inner = m.group(1).replace("\\n", " ").replace('\\"', '"')
        inner = re.sub(r"\s+", " ", inner).strip()
        return inner or None
    return None


# ── Targeted sentence repair helpers ─────────────────────────────────────────

def _split_reasoning_sentences(reasoning: str) -> list[str]:
    """Split a reasoning paragraph into individual sentences."""
    import re as _re
    parts = _re.split(r'(?<=[.!?])\s+(?=[A-Z])', reasoning.strip())
    return [s.strip() for s in parts if s.strip()]


def _extract_detecting_phrases(contradiction: str) -> list[str]:
    """
    Given a contradiction warning string, return the concrete phrases that
    caused it.  Used to identify which sentence(s) in the paragraph are bad.

    Extraction order:
      1. Universal patterns covering the verifier output formats
         (phrase='X' and "reasoning says 'X'").  These are tried first so
         every verifier-format contradiction routes through targeted
         sentence repair instead of full-paragraph fallback.
      2. Legacy runtime-format patterns (forbidden term, claims X, etc.)
         for older contradiction strings.
    """
    import re as _re
    text = contradiction

    # ── Universal verifier-format extractors (added to eliminate
    #    fallback-to-full-paragraph on ~90% of contradictions) ──────────────
    #
    # (A) Unified-verifier emits "(phrase='X', ...)" for every
    #     ClaimRecord-derived contradiction.  When present this is the
    #     canonical bad phrase — use it directly.
    m = _re.search(r"\(phrase='([^']+)'", text)
    if m:
        return [m.group(1).lower()]
    #
    # (B) The minimax-mismatch and other numeric checks emit
    #     "reasoning says 'X' but fact disagrees".  The legacy pattern
    #     below only matches "but reasoning says 'X'" (different word
    #     order), so add the forward form here.
    m = _re.search(r"reasoning says '([^']+)'", text, _re.IGNORECASE)
    if m:
        return [m.group(1).lower()]

    # forbidden term 'X'
    m = _re.search(r"forbidden term '([^']+)'", text, _re.IGNORECASE)
    if m:
        return [m.group(1).lower()]

    # term 'X' used but not in seeds
    m = _re.search(r"term '([^']+)' used but not in seeds", text, _re.IGNORECASE)
    if m:
        return [m.group(1).lower()]

    # inversion detected — ... but reasoning says 'X'
    m = _re.search(r"but reasoning says '([^']+)'", text, _re.IGNORECASE)
    if m:
        return [m.group(1).lower()]

    # unsupported numeric statement 'from N to M'
    m = _re.search(r"unsupported numeric statement '([^']+)'", text, _re.IGNORECASE)
    if m:
        return [m.group(1).lower()]

    # unsupported numeric assertion 'remains at N'
    m = _re.search(r"unsupported numeric assertion '([^']+)'", text, _re.IGNORECASE)
    if m:
        return [m.group(1).lower()]

    # unsupported absence claim 'X'
    m = _re.search(r"unsupported absence claim '([^']+)'", text, _re.IGNORECASE)
    if m:
        return [m.group(1).lower()]

    lower = text.lower()
    if "claims mobility reduction" in lower:
        return [
            "reduces mobility", "reducing mobility", "limits mobility",
            "limiting mobility", "restricts mobility", "restricts opponent",
            "fewer moves for", "cuts opponent moves",
            "reduces opponent mobility", "reducing opponent mobility",
            "limits opponent", "limiting opponent",
        ]
    if "claims avoids recapture" in lower:
        return [
            "avoids recapture", "no recapture", "cannot recapture",
            "without recapture risk", "no recapture risk",
            "safe from recapture", "safe move",
        ]
    if "claims no isolation" in lower:
        return [
            "does not isolate", "no isolation", "maintains connectivity",
            "piece not isolated", "stays connected",
        ]
    if "claims creates_immediate_threat" in lower:
        return [
            "creates a threat", "creates immediate threat",
            "applies pressure next turn", "creates pressure next",
            "threatens opponent", "creates tactical threat",
        ]
    if "claims center_control" in lower:
        return [
            "controls the center", "controls center",
            "central control", "occupies the center",
        ]
    if "claims capture but" in lower:
        return [
            "captures a piece", "captures the piece",
            "captures an opponent", "gaining a piece",
        ]
    if "claims material gain" in lower:
        return ["gains material", "material gain", "gains a piece"]
    if "claims promotion" in lower:
        return [
            "promotes to king", "promotes a piece",
            "crowns a piece", "becomes a king",
        ]
    if "claims blocks_opponent_landing" in lower:
        return [
            "blocks opponent landing",
            "blocks the opponent from landing",
        ]

    # ── Last-resort generic single-quote catch-all.  Any contradiction
    #    string that wraps its bad phrase in single quotes — e.g.
    #    "unsupported 'mobility unchanged' claim — not found in seeds" —
    #    falls through to here.  Extracts the first single-quoted span.
    #    Skipped if the captured token looks like a schema literal
    #    (contains '=' or a bare digit) so we never target a sentence by
    #    matching "0" or "captures_count=0" as a phrase.
    m = _re.search(r"'([^']+)'", text)
    if m:
        token = m.group(1).strip()
        if token and "=" not in token and not token.isdigit():
            return [token.lower()]

    return []


def _partition_sentences_by_contradiction(
    sentences: list[str],
    contradictions: list[str],
) -> tuple[list[int], list[int]]:
    """
    Return (bad_indices, good_indices).
    bad_indices  — sentences that contain at least one contradicting phrase.
    good_indices — sentences with no detected contradiction.
    When no bad sentence can be pinpointed, bad_indices is empty so the caller
    can fall back to full-paragraph regeneration.
    """
    bad: set[int] = set()
    for c in contradictions:
        phrases = _extract_detecting_phrases(c)
        for i, sent in enumerate(sentences):
            sl = sent.lower()
            if any(p in sl for p in phrases):
                bad.add(i)
    good = [i for i in range(len(sentences)) if i not in bad]
    return sorted(bad), good


def _build_targeted_refinement_prompt(
    chosen_move: dict,
    bad_sentences: list[str],
    contradictions: list[str],
) -> str:
    """
    Build a prompt that asks the LLM to replace ONLY the bad sentences.
    Each bad sentence gets exactly one replacement.
    """
    facts = chosen_move.get("facts") or {}
    path  = chosen_move.get("path", [])
    mtype = chosen_move.get("type", "simple")
    n     = len(bad_sentences)

    lines: list[str] = [
        f"Chosen move: type={mtype}  path={path}",
        "",
        "Key facts about the chosen move (use these — do not invent other values):",
    ]
    lines.extend(_relevant_facts_summary(facts))

    lines += [
        "",
        "The following sentences are INCORRECT and must each be replaced with "
        "ONE corrected sentence:",
    ]
    for i, sent in enumerate(bad_sentences):
        lines.append(f'  [{i}] "{sent}"')

    lines += ["", "False claims to fix:"]
    for c in contradictions:
        lines.append(f"  - {c.replace('REASONING_CONTRADICTION: ', '')}")

    # When any contradiction is a minimax mismatch, prepend an explicit
    # correction block.  Mirrors _build_refinement_prompt; the same vocabulary
    # disconnect ("engine score" vs "minimax_score") causes the LLM to repeat
    # the previously-written wrong number unless given the right value directly.
    _mm_fact = facts.get("minimax_score")
    _has_minimax_mismatch = any(
        "minimax_score" in c.lower() and "mismatch" in c.lower()
        for c in contradictions
    )
    if _has_minimax_mismatch and isinstance(_mm_fact, (int, float)):
        lines += [
            "",
            "MINIMAX CORRECTION (mandatory):",
            f"  The ONLY correct value is minimax_score = {_mm_fact:.1f}.",
            "  Any replacement sentence that cites the minimax score MUST use",
            f"  exactly this value ({_mm_fact:.1f}).  Do NOT copy the wrong number",
            "  from the original sentence.  Do NOT cite 0, 0.0, 0.1, 0.12, 0.45,",
            "  or any other value as the minimax score.",
        ]

    lines += [
        "",
        "Instructions:",
        "  1. Write exactly one replacement for each numbered sentence above.",
        "  2. Every replacement must be consistent with the move facts.",
        "  3. Do not reference any fact not listed above.",
        "  4. Do not use forbidden vocabulary.",
        "  5. Keep each replacement concise (one sentence).",
        "  6. Do NOT add new recapture claims, opponent-mobility claims, or numeric",
        "     score-comparison claims (e.g., 'X points better') in your replacements",
        "     unless those facts are explicitly listed in Key Facts above.",
        "  7. Preserve all grounded seed-backed factual claims unless the fact",
        "     itself is the flagged contradiction. In particular, never remove",
        "     or omit:",
        "       - only-legal-move / forced-move disclosures",
        "       - must-capture disclosures",
        "       - explicit mobility transitions ('from N to M')",
        "       - immediate king-promotion facts",
        "       - grounded comparative anchors tied to specific alternatives",
        "     You may paraphrase these facts, but you may not silently drop them.",
        "  8. Forced-move framing rule (strict): IF AND ONLY IF the original",
        "     reasoning already contained a forced-move disclosure grounded in a",
        "     seed that explicitly states the move is the only legal move available",
        "     (e.g., 'only legal move', 'mandatory jump', 'must capture', or",
        "     equivalent forced-move wording), you must preserve that disclosure",
        "     in your replacements. Otherwise, do NOT introduce or invent any",
        "     claim that the move is forced, only-legal, mandatory, or has no",
        "     alternative. A seed saying 'the opponent is forced to respond'",
        "     describes the OPPONENT's reply, not our choice, and does NOT make",
        "     our move forced.",
        "  9. Do not use vague positional descriptors — 'board control', 'piece"
        "     coordination', 'board cohesion', 'connectivity' — that are not"
        "     directly supported by a fact listed above.",
        " 10. AUDIT-DRIVEN ANTI-HALLUCINATION RULES — each replacement sentence",
        "     must NOT contain any of the following unless the supporting fact",
        "     in Key Facts above is True:",
        "       - 'creates an immediate threat' / 'forces the opponent' →",
        "         requires creates_immediate_threat=true OR",
        "         forced_opponent_jump_reply=true.",
        "       - 'only legal move' / 'forced move' / 'no alternative' →",
        "         requires an only-legal-move disclosure grounded in a seed.",
        "       - 'controls the center' / 'central control' / 'centralizes' →",
        "         requires center_control=true.",
        "       - 'reduces opponent mobility' / 'restricts opponent mobility' →",
        "         requires opponent_mobility_after < opponent_mobility_before.",
        "       - 'N points better' / 'best by N' → requires an explicit",
        "         comparison number from the seed list.",
        "       - 'near promotion' / 'one step from promotion' → requires",
        "         near_promotion=true or results_in_king=true.",
        "",
        f'Reply with ONLY this JSON (exactly {n} replacement(s)):',
        '{"replacements": ["<sentence 0>", "<sentence 1>", ...]}',
    ]
    return "\n".join(lines)


def _extract_targeted_repair_response(
    raw: str,
    expected_count: int,
) -> Optional[list[str]]:
    """Parse a targeted repair LLM response into a list of replacement sentences."""
    import json as _json
    import re as _re
    try:
        obj = _json.loads(raw.strip())
        reps = obj.get("replacements")
        if isinstance(reps, list) and len(reps) == expected_count and all(
            isinstance(r, str) for r in reps
        ):
            return reps
    except Exception:
        pass
    m = _re.search(r'"replacements"\s*:\s*(\[[^\]]+\])', raw, _re.DOTALL)
    if m:
        try:
            reps = _json.loads(m.group(1))
            if isinstance(reps, list) and len(reps) == expected_count:
                return reps
        except Exception:
            pass
    return None


_FORCED_FRAMING_RE = re.compile(
    r"\b(only legal move|only legal positional option|only available choice"
    r"|no alternative|no other option|no other choice|no choice but"
    r"|forced jump|forced capture|forced move|forced sequence"
    r"|must capture|must jump|must take|mandatory|compulsory"
    r"|rules require|cannot avoid|position cannot avoid)\b",
    re.IGNORECASE,
)

_PROMOTION_RE = re.compile(
    r"\b(promot\w+|king\b|crowns?)\b",
    re.IGNORECASE,
)

_FORCED_SEED_MARKERS = ("only legal move", "mandatory jump", "must capture")


def _has_forced_seed(seeds: Optional[list[str]]) -> bool:
    if not seeds:
        return False
    for s in seeds:
        sl = s.lower()
        for m in _FORCED_SEED_MARKERS:
            if m in sl:
                return True
    return False


def _validate_and_select(
    prior_text: str,
    prior_count: int,
    repair_text: Optional[str],
    repair_count: int,
    seeds: Optional[list[str]],
    facts: Optional[dict],
) -> tuple[str, int, str]:
    """
    Pre-commit gate for a repair attempt.

    Returns (selected_text, selected_count, decision_tag) where decision_tag is
    one of: 'accepted', 'rejected_parse', 'rejected_no_improvement',
    'rejected_forced_fabricated', 'rejected_forced_dropped',
    'rejected_promotion_dropped'.

    On any rejection the prior (best-so-far) text and count are returned
    unchanged, so the caller can simply assign back unconditionally.
    """
    if not repair_text:
        return prior_text, prior_count, "rejected_parse"

    # (1) Monotonicity — never accept equal-or-worse contradiction counts.
    if repair_count >= prior_count:
        return prior_text, prior_count, "rejected_no_improvement"

    # (2) Forced-framing symmetry.
    has_forced_output = bool(_FORCED_FRAMING_RE.search(repair_text))
    forced_seed_present = _has_forced_seed(seeds)
    if has_forced_output and not forced_seed_present:
        return prior_text, prior_count, "rejected_forced_fabricated"
    if forced_seed_present and not has_forced_output:
        return prior_text, prior_count, "rejected_forced_dropped"

    # (3) King-promotion preservation.
    if facts and facts.get("results_in_king") is True:
        if not _PROMOTION_RE.search(repair_text):
            return prior_text, prior_count, "rejected_promotion_dropped"

    return repair_text, repair_count, "accepted"


# ── Reasoning-only refinement loop ────────────────────────────────────────────

def _refine_reasoning(
    reasoning: str,
    chosen_move: dict,
    initial_contradictions: list[str],
    max_attempts: int = 2,
    seeds: Optional[list[str]] = None,
) -> tuple[str, int, bool]:
    """
    Post-hoc reasoning refinement loop.

    Uses targeted sentence repair: only the sentences that contain contradicting
    phrases are sent to the LLM for replacement.  All other sentences are
    preserved unchanged.  Guarantees:
      - chosen_move is never mutated or re-evaluated.
      - _apply_safety_filter is never called.
      - move-selection override / re-ranking never happen.
      - reasoning_retry_count is separate from the move-selection retry count.

    Returns:
        (final_reasoning, reasoning_retry_count, resolved)
    """
    import time as _time

    facts        = chosen_move.get("facts") or {}
    retry_count  = 0

    # Track best-so-far across attempts.  Repair candidates are committed only
    # when _validate_and_select accepts them (strict-improvement monotonicity
    # plus forced-framing symmetry plus king-promotion preservation).  Each
    # next attempt re-bases on the current best, so a rejected candidate is
    # discarded entirely rather than corrupting the baseline.
    best_text          = reasoning
    best_count         = len(initial_contradictions)
    best_contradictions = list(initial_contradictions)

    for attempt in range(1, max_attempts + 1):
        print(
            f"[EXPLAINER_REASONING_RETRY] attempt={attempt} "
            f"contradictions={best_contradictions}"
        )
        retry_count += 1

        sentences   = _split_reasoning_sentences(best_text)
        bad_indices, good_indices = _partition_sentences_by_contradiction(
            sentences, best_contradictions
        )

        use_targeted = bool(bad_indices)

        if use_targeted:
            bad_sentences = [sentences[i] for i in bad_indices]
            print(
                f"[EXPLAINER_REASONING_RETRY] targeted_repair: "
                f"bad_count={len(bad_sentences)} "
                f"preserved_count={len(good_indices)} "
                f"bad_sentences={bad_sentences}"
            )
            user_prompt = _build_targeted_refinement_prompt(
                chosen_move, bad_sentences, best_contradictions
            )
        else:
            print(
                f"[EXPLAINER_REASONING_RETRY] cannot isolate bad sentences; "
                "falling back to full-paragraph refinement"
            )
            user_prompt = _build_refinement_prompt(chosen_move, best_contradictions)

        raw: Optional[str] = None
        _refine_waits = (20, 30, 40, 20, 30)
        for api_try in range(6):
            try:
                raw = call_explainer(EXPLAINER_REASONING_REFINEMENT_SYSTEM, user_prompt)
                break
            except Exception as e:
                if api_try < len(_refine_waits):
                    wait = _refine_waits[api_try]
                    print(
                        f"[EXPLAINER_REASONING_RETRY] api error "
                        f"(attempt={attempt}, api_try={api_try + 1}): {e} "
                        f"— waiting {wait}s"
                    )
                    _time.sleep(wait)

        if raw is None:
            print(
                f"[EXPLAINER_REASONING_RETRY] api call failed on attempt {attempt}; "
                "keeping previous reasoning"
            )
            break

        # Build candidate paragraph from the LLM response.
        candidate: Optional[str] = None
        if use_targeted:
            replacements = _extract_targeted_repair_response(raw, len(bad_sentences))
            if replacements:
                repaired = list(sentences)
                for idx, new_sent in zip(bad_indices, replacements):
                    repaired[idx] = new_sent
                candidate = " ".join(repaired)
            else:
                # Targeted parse failed; try extracting a full paragraph instead.
                candidate = _extract_refinement_reasoning(raw)
        else:
            candidate = _extract_refinement_reasoning(raw)

        if not candidate:
            print(
                f"[EXPLAINER_REASONING_RETRY] could not parse refinement response "
                f"on attempt {attempt}; keeping previous reasoning"
            )
            break

        # Score candidate and route through the deterministic pre-commit gate.
        cand_contradictions = _check_reasoning_truthfulness(
            candidate, facts, seeds=seeds
        )
        cand_count = len(cand_contradictions)

        new_text, new_count, decision = _validate_and_select(
            best_text, best_count,
            candidate, cand_count,
            seeds, facts,
        )
        print(
            f"[EXPLAINER_TRUTHFULNESS] gate_decision={decision} "
            f"prior_count={best_count} candidate_count={cand_count}"
        )

        if decision == "accepted":
            best_text          = new_text
            best_count         = new_count
            best_contradictions = cand_contradictions
            if best_count == 0:
                print("[EXPLAINER_TRUTHFULNESS] intermediate_check_clean: breaking early")
                break
        # On any rejection the best-so-far is retained; the next attempt
        # re-runs from the same baseline so a bad candidate cannot poison
        # subsequent attempts.

    # Final full-paragraph validation always runs, regardless of how the loop ended.
    # This is the single authoritative validation point for the returned reasoning.
    final_contradictions = _check_reasoning_truthfulness(
        best_text, facts, seeds=seeds
    )
    resolved = len(final_contradictions) == 0
    print(f"[EXPLAINER_TRUTHFULNESS] reasoning_refinement_resolved={resolved}")
    if not resolved:
        print(
            f"[EXPLAINER_TRUTHFULNESS] reasoning_still_contradicts_after_"
            f"{retry_count}_attempt(s)={final_contradictions}"
        )
    return best_text, retry_count, resolved


# ── Grounded reasoning seeds ───────────────────────────────────────────────────

EXPLAINER_SEED_REASONING_SYSTEM: str = """\
You are a checkers move coach. You have been given a verified list of engine-computed
factual claims about the chosen move ("reasoning seeds").

Your task: write a single paragraph (3-5 sentences) that explains WHY this move was
chosen. Use ONLY the provided reasoning seeds as evidence. Each sentence must convey
a reason or consequence grounded in the seeds — do NOT paraphrase or mechanically
list them; synthesize them into a causal explanation.

STRICT RULES:
  - Use ONLY the provided reasoning seeds. Do NOT introduce any new strategic claims,
    positional assessments, or concepts not present in the seed list.
  - NEVER add phrases like: "structural pressure", "stable position", "limits options",
    "good position", "no advantage", "better structure", or any vague evaluation.
  - NEVER use variable names, schema keys, or key=value notation (e.g., do NOT write
    "opponent_can_recapture=false" — write it as a natural English statement instead).
  - Every sentence except the final minimax confirmation MUST be grounded in a concrete
    fact from the seed list.
  - minimax_score may appear ONLY in the final sentence as confirmation.
  - Do NOT use labeled section headers.
  - Write a single coherent paragraph only.
  - DRAWBACKS: If the seeds contain any drawback (e.g., the opponent can recapture,
    allied pieces remain under threat, the moved piece is isolated or threatened), you
    MUST acknowledge that drawback explicitly. Do NOT hide, omit, or downplay it.
  - PRIORITY: When safety/tactical seeds (recapture, threats, captures, immediate threat)
    and structural/positional seeds both exist, address safety/tactical seeds in the
    first 1–2 sentences. Structural seeds may appear only as supporting context.

  ANTI-TEMPLATE RULES — vary your language and structure:
    - Do NOT open more than one sentence with "Despite", "Additionally", "Furthermore",
      "Moreover", or "Also".
    - Do NOT close the paragraph with filler such as "This makes it the best choice.",
      "Overall, ...", "In summary, ...", or "Therefore, this is the optimal move."
    - Do NOT produce sentences with identical grammatical structure back-to-back.
    - The opening sentence must introduce the primary reason for the move — not restate
      the move path or announce a positional theme without evidence.

  DECISION-RELEVANT FACTS — keep the paragraph faithful to the seeds it ranks on.
    Use only the grounded facts that were provided. Within the 3–5 sentences,
    briefly mention the most decision-relevant verifiable facts present in the
    seeds, prioritised in this order when they appear:
      1. material change (captures_count, net_gain),
      2. mobility change (opponent_mobility_before/after, our_mobility_before/after,
         mobility_reduction),
      3. immediate threat (creates_immediate_threat),
      4. recapture safety or risk (opponent_can_recapture),
      5. forced opponent reply (forced_opponent_jump_reply, max_opponent_jump_captures),
      6. isolation or connectivity (leaves_piece_isolated),
      7. adversity / losing-position context when present
         (slightly_losing, clearly_losing, least_harmful, forced_choice).
    Do not invent unsupported strategic terms.
    Do not mechanically restate every seed — pick the few that actually drove
    the decision and weave them into natural prose.
    Do NOT use any of the following words or phrases under any circumstances:
      "initiative", "dominance", "control the game", "pressure", "strong position",
      "conversion potential", "winning conversion", "trade conversion", "conversion score",
      "quiet_move_role", "winning_conversion_score", "king_activity_score",
      "counterplay_score", "coordination score", "activity score",
      "escape squares", "escape routes", "king escape", "king distance",
      "diagonal pressure", "diagonal risks", "long diagonal",
      "strategic goal", "positional adjustment", "real traps",
      "regulars_captured",
      "central board presence", "central influence",
      "improves activity", "piece activity", "more active position",
      "maintains pressure",
      "tangible positional advantage", "improved position", "strong positional edge".
    Do NOT state any number ("from X to Y", "remains at N", "unchanged at N")
    unless that exact number appears verbatim in the seed list.
    Do NOT claim "no kings lost", "piece count unchanged", or "no vulnerabilities"
    unless an explicit seed states it.

  SINGLE-LEGAL-MOVE CONTEXT (BUG-3):
    If a seed states "This is the only legal move available", do NOT use
    "strongest choice", "best move", or "highest-ranked option". The move is
    not chosen for superiority — it is the only option. Use the factual
    wording from the seed instead.

  MULTIPLE-MOVES CONTEXT:
    If a seed states "Multiple legal moves were available", do NOT use phrases
    like "only legal move", "only available option", "only viable option",
    "forced move", "no alternative", or "no other option". Multiple options
    existed; the move was chosen by the engine, not forced.

  IMMEDIATE THREAT (audit-driven):
    Do NOT write "creates an immediate threat", "creates immediate pressure",
    "creates a threat", "threatens the opponent", or "forces the opponent"
    UNLESS a seed explicitly states "forces the opponent to respond to an
    immediate threat" or "The opponent is forced to respond with a jump".
    If a seed says "This move does not create an immediate threat", you MUST
    use that wording rather than the opposite.

  MOBILITY REDUCTION (audit-driven):
    Do NOT write "reduces opponent mobility", "restricts opponent mobility",
    "narrows opponent options", "cuts opponent moves", or "limits opponent
    replies" UNLESS a seed explicitly contains the phrase "reduces opponent
    mobility by N".  If a mobility seed says "remains at N" or "no change in
    opponent mobility" or "Opponent mobility is unchanged", the mobility did
    not change — do not assert any reduction.

  COMPARISON VALUES (anti-hallucination):
    NEVER write "N points better", "N points stronger", or similar numeric
    comparison phrases unless a seed explicitly provides the exact number in
    the form "The chosen move scores N.N points better than the next-best
    option". If no such seed exists, use qualitative language only (e.g.,
    "the engine evaluated this path more highly"). Inventing comparison
    magnitudes is a factual error.

  CENTER GEOMETRY vs TACTICAL CONTROL (BUG-4):
    A seed stating "The destination is in the center of the board (column X)"
    is a GEOMETRIC fact only — it does NOT imply tactical center control. Only
    write "controls the center" or "central control" if a seed explicitly states
    "The move gains central board control" or "The move claims central control".
    Do NOT draw strategic center-control conclusions from a geometry-only seed.

  TRADEOFF LANGUAGE (BUG-5):
    NEVER say "outweighs", "compensates for", "justifies the risk", or
    "balances out" unless a seed explicitly provides numeric evidence for both
    sides of the comparison (e.g., capture count AND recapture risk together).

  MOBILITY DISADVANTAGE (BUG-6):
    If opponent mobility is still higher than ours after the move, use
    "narrows the gap" rather than "solves", "addresses", "fixes", or
    "eliminates" the disadvantage. Do not overclaim resolution of a
    disadvantage that persists.

  COORDINATE REFERENCE (BUG-10):
    Every explanation paragraph must reference the move path coordinates at
    least once. Use the path from the "Chosen move:" line above.

  TACTICAL EXPOSURE CONTEXT:
    When a seed states that the opponent can recapture, allied pieces remain under
    threat, or the moved piece is threatened, acknowledge that drawback honestly.
    Then explain WHY the move is still chosen: e.g., material gain, threat creation,
    minimax advantage, or constrained opponent reply.
    Do NOT invert the seed: if a seed mentions pieces under threat, do NOT write
    "no threats remain". If a seed says the opponent can recapture, do NOT write
    "avoids recapture".

OUTPUT FORMAT — reply with ONLY this JSON, no markdown:
{"reasoning": "<your paragraph>"}
"""


def _find_comparison_seed(
    chosen_facts: dict,
    alt_facts: dict,
    alt_index: int,
) -> Optional[str]:
    """
    Return a concrete factual comparison seed vs one alternative, or None.
    Priority order: recapture > moved-threatened > pieces-at-risk >
                    captures > isolation > immediate-threat > center.
    NEVER uses vague words like 'worse', 'weaker', 'no advantage'.
    """
    # 1. Recapture safety
    if chosen_facts.get("opponent_can_recapture") is False \
            and alt_facts.get("opponent_can_recapture") is True:
        return f"Unlike move [{alt_index}], the chosen piece cannot be immediately recaptured."
    # 2. Moved piece threatened
    if chosen_facts.get("moved_piece_is_threatened") is False \
            and alt_facts.get("moved_piece_is_threatened") is True:
        return (
            f"Unlike move [{alt_index}], the moved piece is not left under immediate threat."
        )
    # 3. Pieces at risk count
    c_pta = chosen_facts.get("our_pieces_threatened_after")
    a_pta = alt_facts.get("our_pieces_threatened_after")
    if c_pta is not None and a_pta is not None and c_pta < a_pta:
        return (
            f"Unlike move [{alt_index}], fewer allied pieces are left under threat "
            f"({c_pta} vs {a_pta})."
        )
    # 4. Captures
    c_cap = chosen_facts.get("captures_count", 0)
    a_cap = alt_facts.get("captures_count", 0)
    if c_cap > a_cap:
        return (
            f"The chosen move captures {c_cap} piece(s); move [{alt_index}] captures only {a_cap}."
        )
    # 5. Isolation
    if chosen_facts.get("leaves_piece_isolated") is False \
            and alt_facts.get("leaves_piece_isolated") is True:
        return (
            f"Unlike move [{alt_index}], the moved piece remains supported by adjacent allies."
        )
    # 6. Immediate threat
    if chosen_facts.get("creates_immediate_threat") is True \
            and alt_facts.get("creates_immediate_threat") is False:
        return f"The chosen move creates an immediate threat; move [{alt_index}] does not."
    # 7. Center control
    if chosen_facts.get("center_control") is True \
            and alt_facts.get("center_control") is False:
        return f"The chosen move gains central board control; move [{alt_index}] does not."
    return None


# Semantic thresholds for minimax score wording.
# Below CLEARLY_LOSING the position is materially hopeless; below SLIGHTLY_LOSING
# it is noticeably unfavourable.  These are wording-layer constants only and have
# no effect on scoring, search, or move selection.
_MINIMAX_CLEARLY_LOSING: float = -100.0
_MINIMAX_SLIGHTLY_LOSING: float = -20.0


def _build_adversity_context_seeds(
    facts: dict,
    all_candidates: list,
    chosen_path,
) -> list[str]:
    """
    Return 0-5 fact-grounded adversity context seeds for a losing position.
    Called only when minimax_score < _MINIMAX_SLIGHTLY_LOSING.
    Seeds are prepended to the standard seed list so the LLM reads positional
    context before per-move safety facts.

    Rules:
    - Every seed contains the exact fact name and value it derives from.
    - No vague terms: counterplay, pressure, activity, balance, trap, initiative.
    - Does NOT claim the chosen move resolves threats that it cannot verify.
    - READ-ONLY: never mutates facts, all_candidates, or chosen_path.
    """
    seeds: list[str] = []

    # ── A. Score gap to next-best alternative ─────────────────────────────────
    chosen_mm = facts.get("minimax_score")
    if chosen_mm is not None:
        alternatives = [
            (i, m) for i, m in enumerate(all_candidates)
            if m.get("path") != chosen_path
        ]
        if alternatives:
            best_alt_idx, best_alt = max(
                alternatives,
                key=lambda im: (
                    _get_minimax_score(im[1])
                    if _get_minimax_score(im[1]) is not None
                    else float("-inf")
                ),
            )
            alt_mm = _get_minimax_score(best_alt)
            if alt_mm is not None and alt_mm != float("-inf"):
                gap = chosen_mm - alt_mm
                if gap > 20.0:
                    seeds.append(
                        f"The chosen move scores {gap:.1f} points better than "
                        f"the next-best option [move {best_alt_idx}] "
                        f"(engine scores: {chosen_mm:.1f} vs {alt_mm:.1f})."
                    )

    # ── B. Material deficit ───────────────────────────────────────────────────
    mat_adv = facts.get("material_advantage")
    if mat_adv is not None and mat_adv < 0:
        deficit = -mat_adv
        seeds.append(
            f"The position is behind by {deficit} piece(s) in material."
        )

    # ── C. Threat reduction ───────────────────────────────────────────────────
    pta_before = facts.get("our_pieces_threatened_before")
    pta_after  = facts.get("our_pieces_threatened_after")
    if (
        pta_before is not None
        and pta_after is not None
        and pta_before > 0
        and pta_after < pta_before
    ):
        seeds.append(
            f"This move reduces threatened allied pieces from {pta_before} to {pta_after}, "
            "improving immediate safety."
        )

    # ── D. Opponent near promotion ────────────────────────────────────────────
    # Board-state fact only — does NOT claim the chosen move blocks it.
    if facts.get("opponent_near_promotion") is True:
        seeds.append(
            "At least one opponent piece is one step from promotion."
        )

    # ── E. Mobility asymmetry ─────────────────────────────────────────────────
    opp_mob = facts.get("opponent_mobility_before")
    our_mob = facts.get("our_mobility_before")
    if opp_mob is not None and our_mob is not None and (opp_mob - our_mob) >= 3:
        seeds.append(
            f"The opponent has {opp_mob} available moves against our {our_mob} — "
            "a mobility disadvantage going into this turn."
        )

    return seeds


def _minimax_wording_label(mm: float) -> str:
    """
    Return a score-appropriate minimax confirmation label for the seed.
    Only adjusts wording for losing evaluations; non-losing positions
    keep the existing 'highest-evaluated option' phrasing unchanged.
    """
    if mm < _MINIMAX_CLEARLY_LOSING:
        return "least harmful available continuation"
    if mm < _MINIMAX_SLIGHTLY_LOSING:
        return "best available option in a difficult position"
    return "highest-evaluated option"


def _is_losing_score_state(score_state: Optional[str]) -> bool:
    """Return True iff state.score_state indicates the current mover is
    materially behind.  Conservative: an unknown / missing value returns
    False (callers fall back to the raw-minimax gate).
    """
    return isinstance(score_state, str) and score_state in (
        "CLEARLY_LOSING",
        "SLIGHTLY_LOSING",
    )


def _resolve_score_state_for_seeds(state: CheckersState) -> str:
    """
    Return score_state for the adversity-seed gate.

    Reads state.score_state, which scorer_node writes on every ply from
    compute_score_state(board, player).  Falls back to "EQUAL" for unit
    tests or harnesses that construct a CheckersState without running
    scorer_node (the Pydantic default is "EQUAL").
    """
    val = state.score_state
    if isinstance(val, str) and val.strip():
        return val
    return "EQUAL"


# ── Move-class routing for semantic grounding (Phase G, Step 1) ──────────────
#
# Audit finding: the existing seed-reasoning prompt asks the LLM to write a
# 3–5-sentence "causal explanation" for EVERY move.  On quiet positional moves
# (no captures, no immediate threat, no forced opponent reply, no king
# promotion, small mobility delta) the symbolic facts do not supply enough
# narrative volume to fill that prompt, so the LLM confabulates strategic
# content ("creates pressure", "restricts N pieces", "narrows the gap", etc.).
#
# This helper classifies a move as "tactical" or "quiet" so the prompt builder
# can route to a shorter, more factual variant for quiet moves.  Deterministic,
# reads only the move's `facts` dict.

_QUIET_MOBILITY_DELTA_THRESHOLD = 1   # |Δmob| <= 1 on each side qualifies as small


def _classify_move_intent(facts: Optional[dict]) -> str:
    """Return 'tactical' or 'quiet' based on grounded engine facts.

    A move is QUIET iff ALL of the following hold:
      - captures_count == 0
      - creates_immediate_threat is not True
      - forced_opponent_jump_reply is not True
      - results_in_king is not True
      - |our_mobility_after - our_mobility_before| <= 1
      - |opponent_mobility_after - opponent_mobility_before| <= 1

    Any other case is TACTICAL.  Missing fields are treated conservatively
    (absent fields do not trigger tactical classification on their own).
    """
    f = facts or {}
    if (f.get("captures_count") or 0) > 0:
        return "tactical"
    if f.get("creates_immediate_threat") is True:
        return "tactical"
    if f.get("forced_opponent_jump_reply") is True:
        return "tactical"
    if f.get("results_in_king") is True:
        return "tactical"
    ub = f.get("our_mobility_before")
    ua = f.get("our_mobility_after")
    if isinstance(ub, int) and isinstance(ua, int) and abs(ua - ub) > _QUIET_MOBILITY_DELTA_THRESHOLD:
        return "tactical"
    ob = f.get("opponent_mobility_before")
    oa = f.get("opponent_mobility_after")
    if isinstance(ob, int) and isinstance(oa, int) and abs(oa - ob) > _QUIET_MOBILITY_DELTA_THRESHOLD:
        return "tactical"
    return "quiet"


def _negative_grounding_seeds(
    facts: Optional[dict],
    n_candidates: int,
    existing_seeds: list[str],
) -> list[str]:
    """Emit explicit negative-fact seeds for the four predicates the human
    audit identified as the most commonly *fabricated* positives.

    Negatives are short, atomic, and never duplicate an existing seed.  When a
    fact is True (the positive direction) the corresponding positive seed is
    already emitted upstream; this helper only fires on the False direction.

    Targets:
      T2  creates_immediate_threat=False     → "does not create an immediate threat"
      T3  forced_opponent_jump_reply=False   → "opponent is not forced ... jump"
      T1  frozen_enemy_pieces == 0           → "does not restrict any opponent piece"
      forced_move_for_us=False (n>1)         → "multiple legal moves were available"
    """
    f = facts or {}
    out: list[str] = []
    haystack = " ".join(s.lower() for s in (existing_seeds or []))

    # T2 — fake immediate threat (cit=False)
    if f.get("creates_immediate_threat") is False \
            and "immediate threat" not in haystack \
            and "immediate pressure" not in haystack:
        out.append("This move does not create an immediate threat.")

    # T3 — fake forced opponent jump reply
    if f.get("forced_opponent_jump_reply") is False \
            and "forced to respond with a jump" not in haystack \
            and "opponent is forced" not in haystack:
        out.append("The opponent is not forced to respond with a jump.")

    # T1 — fake restriction effect (frozen_enemy_pieces == 0)
    fep = f.get("frozen_enemy_pieces")
    if isinstance(fep, int) and fep == 0 and "restrict" not in haystack:
        out.append("This move does not restrict any opponent piece's forward movement.")

    # Forced-move-for-us negative (audit's false-forced-only-legal class).
    # Suppressed when the single-candidate path has already emitted an
    # "only legal move" seed.
    if n_candidates > 1 and "only legal move" not in haystack:
        out.append("Multiple legal moves were available; the engine selected this option.")

    # Near-promotion negative: only when near_promotion is explicitly False.
    # Prevents the LLM from fabricating "one step from promotion" claims.
    # Use "promot" prefix to match both "promotion" and "promoted" in haystack.
    if f.get("near_promotion") is False and "promot" not in haystack:
        out.append("The piece is not near promotion after this move.")

    # Audit pattern: false center-control claim when center_control=False.
    # No positive geometric seed is emitted in this case (the geometric seed
    # is gated on center_control=True), so without this negative the LLM has
    # nothing pushing back against fabricated "central" / "controls center"
    # phrasing.  Keep wording aligned with the system-prompt CENTER GEOMETRY
    # rule so the negative naturally lands in the model's vocabulary.
    if f.get("center_control") is False \
            and "central board control" not in haystack \
            and "claims central control" not in haystack:
        out.append("The move does not gain central board control.")

    # Audit pattern: false mobility-reduction claim when opponent mobility is
    # unchanged.  The grounded mobility seed already states "remains at N —
    # no change in opponent mobility", but a short explicit negative makes the
    # constraint visible to the LLM without forcing it to parse the numeric
    # transition seed.  Only emit when before/after are both ints AND equal AND
    # no positive opponent-mobility-reduction seed is already present.
    _omb = f.get("opponent_mobility_before")
    _oma = f.get("opponent_mobility_after")
    if (
        isinstance(_omb, int)
        and isinstance(_oma, int)
        and _omb == _oma
        and "reduces opponent mobility" not in haystack
        and "opponent mobility is unchanged" not in haystack
    ):
        out.append("Opponent mobility is unchanged after this move; this move does not reduce opponent mobility.")

    return out


# ── Mobility-gap direction seed (Phase G, Step 3) ────────────────────────────
#
# Audit T4 finding: the seed list surfaces the raw mobility numbers
# ("our mobility changes from 8 to 9", "opponent mobility remains at 12") but
# does NOT emit a derived direction summary of the gap.  The LLM is therefore
# asked to compute the absolute-distance change itself, and sometimes mislabels
# the direction (claims "narrows the gap" when |gap| widened, claims "mobility
# unchanged" when individual mobilities decreased symmetrically, etc.).
#
# Step 3 emits a single grounded seed describing the engine's actual gap
# direction whenever the four mobility values are available.  Deterministic,
# factual-only, no strategic interpretation.

def _mobility_gap_seed(facts: Optional[dict]) -> Optional[str]:
    """Return a grounded seed describing the |our − opp| mobility-gap
    direction, or None when not computable.

      gap_after  <  gap_before  → 'The mobility gap narrowed by N.'
      gap_after  >  gap_before  → 'The mobility gap widened by N.'
      gap_after  == gap_before  → 'The mobility gap remained unchanged.'

    Pure function.  Reads only the four mobility fields.  No side effects.
    """
    f = facts or {}
    ub = f.get("our_mobility_before")
    ua = f.get("our_mobility_after")
    ob = f.get("opponent_mobility_before")
    oa = f.get("opponent_mobility_after")
    if not all(isinstance(x, int) for x in (ub, ua, ob, oa)):
        return None
    gap_before = abs(ub - ob)
    gap_after  = abs(ua - oa)
    if gap_after == gap_before:
        return "The mobility gap remained unchanged."
    delta = abs(gap_after - gap_before)
    if gap_after < gap_before:
        return f"The mobility gap narrowed by {delta}."
    return f"The mobility gap widened by {delta}."


def _build_grounded_reasoning_seeds(
    chosen_move: dict,
    all_candidates: list,
    player: int = 0,
    score_state: Optional[str] = None,
) -> list[str]:
    """
    Build a list of truthful, fact-derived reasoning seeds for chosen_move.
    NEVER emits claims that are not backed by actual fact values.
    NEVER uses forbidden vague language.
    chosen_move is READ-ONLY.

    player  INT constant (RED or BLACK from checkers.engine.board).
            When 0 (unknown), direction-sensitive seeds fall back to safe defaults.

    score_state  Optional state.score_state value (written by scorer_node).
                 When equal to "CLEARLY_LOSING"/"SLIGHTLY_LOSING", adversity
                 seeds fire regardless of minimax_score.  When None or missing,
                 the raw-minimax threshold (_MINIMAX_SLIGHTLY_LOSING) is used.
                 This avoids emitting losing-position language during forced-but-
                 winning lines where a single move has mm<-20 but the player is
                 actually ahead overall.
    """
    seeds: list[str] = []
    facts = chosen_move.get("facts") or {}
    chosen_path = chosen_move.get("path")

    # ── Adversity context gate ────────────────────────────────────────────────
    # Preferred gate: state.score_state (written by scorer_node), which reflects
    # whole-position material/king balance from the current mover's perspective.
    # Fallback gate: raw minimax_score < _MINIMAX_SLIGHTLY_LOSING, used only when
    # score_state is not provided (back-compat for unit tests and isolated calls).
    _mm = facts.get("minimax_score")
    if score_state is not None:
        _adversity_active = _is_losing_score_state(score_state)
    else:
        _adversity_active = _mm is not None and _mm < _MINIMAX_SLIGHTLY_LOSING

    if _adversity_active:
        seeds.extend(
            _build_adversity_context_seeds(facts, all_candidates, chosen_path)
        )

    # ── Safety ──────────────────────────────────────────────────────────────
    recapture = facts.get("opponent_can_recapture")
    if recapture is False:
        seeds.append("The moved piece cannot be immediately recaptured.")
    elif recapture is True:
        seeds.append("The opponent can recapture the moved piece next turn.")

    pta = facts.get("our_pieces_threatened_after")
    if pta is not None:
        if pta == 0:
            seeds.append("No allied pieces remain under attack after this move.")
        else:
            seeds.append(f"{pta} allied piece(s) remain under threat after the move.")

    mpt = facts.get("moved_piece_is_threatened")
    if mpt is True:
        seeds.append("The moved piece remains under immediate threat.")

    # ── Tactical ───────────────────────────────────────────────────────────
    cap = facts.get("captures_count", 0)
    net = facts.get("net_gain", 0)
    if cap > 0:
        seeds.append(f"The move captures {cap} piece(s), gaining a net advantage of {net}.")

    if facts.get("creates_immediate_threat") is True:
        seeds.append("This move forces the opponent to respond to an immediate threat.")

    if facts.get("shot_sequence_available") is True:
        seeds.append("A multi-jump sequence is available to continue the attack.")

    if facts.get("blocks_opponent_landing") is True:
        seeds.append("The move denies the opponent a key landing square.")

    # ── Restriction-count grounding (Phase G, Step 2) ──────────────────────
    # Audit T1 finding: when the engine knows a non-zero number of opponent
    # pieces become immobile after the move (`frozen_enemy_pieces > 0`) but
    # the seed list does not surface that count, the LLM tends to invent a
    # plausible-sounding integer ("restricts N opponent pieces from
    # advancing").  Emit the real count as a grounded positive seed, with
    # singular/plural grammar.  The Step 1 negative seed
    # ("does not restrict any opponent piece's forward movement") fires only
    # when the count is zero, so this positive and the Step 1 negative are
    # mutually exclusive by construction.
    fep = facts.get("frozen_enemy_pieces")
    if isinstance(fep, int) and fep > 0:
        _noun = "piece has" if fep == 1 else "pieces have"
        seeds.append(
            f"{fep} opponent {_noun} restricted forward movement after this move."
        )

    fjr = facts.get("forced_opponent_jump_reply")
    mjc = facts.get("max_opponent_jump_captures")
    if fjr is True and mjc is not None:
        jc = facts.get("opponent_jump_count")
        if isinstance(jc, int) and jc > 1:
            # Explicitly state the jump count so the LLM cannot claim "single jump"
            # when multiple jump options exist.  Conflating forced-jump mode with
            # a single legal jump option was the root cause of BUG-2 (audit).
            seeds.append(
                f"The opponent is forced to respond with a jump; "
                f"{jc} jump options are available "
                f"(each captures at most {mjc} piece(s))."
            )
        else:
            seeds.append(
                f"The opponent is forced to respond with a jump "
                f"(at most {mjc} piece(s) captured)."
            )

    # ── Structure ───────────────────────────────────────────────────────────
    isolated = facts.get("leaves_piece_isolated")
    if isolated is True:
        seeds.append("The moved piece is left without adjacent support.")
    elif isolated is False:
        seeds.append("The moved piece is not left isolated.")

    if facts.get("weakens_king_row") is True:
        seeds.append("The move weakens the back-row defensive structure.")

    # center_control ONLY if True (never claim it when False).
    # BUG-7: when frozen_enemy_pieces > 0, inject a causal seed tying
    # central control to the restriction count so the LLM explains WHY
    # the square matters instead of repeating the label.
    if facts.get("center_control") is True:
        _frozen_for_center = facts.get("frozen_enemy_pieces")
        if isinstance(_frozen_for_center, int) and _frozen_for_center > 0:
            seeds.append(
                f"The move claims central control; "
                f"this positioning restricts {_frozen_for_center} opponent "
                f"piece(s) from advancing."
            )
        else:
            seeds.append("The move gains central board control.")

    # ── Restriction / structural pressure ────────────────────────────────────
    restriction = facts.get("restriction_score")
    frozen = facts.get("frozen_enemy_pieces")
    role = facts.get("quiet_move_role") or ""
    if role == "STRUCTURAL_RESTRICTION" or (restriction and restriction > 0):
        if frozen and frozen > 0:
            seeds.append(
                f"After this move, {frozen} opponent piece(s) have restricted forward movement."
            )
        elif restriction and restriction > 0:
            seeds.append("The move constrains the opponent's available options.")

    # ── Promotion ───────────────────────────────────────────────────────────
    if facts.get("results_in_king") is True:
        seeds.append("The piece is immediately promoted to king.")
    elif facts.get("near_promotion") is True:
        seeds.append("The piece is now one step from promotion.")

    # ── Mobility — emit ONE natural-language seed per before/after pair ───────
    # Earlier versions emitted both a structured "key=value" seed and a
    # natural-language seed; that doubled up the same fact and pushed the LLM
    # toward mechanical seed restatement.  We keep the natural-language form,
    # which the truthfulness checker grounds against the facts dict directly
    # (digit / number-word bypass), so explicit key=value seeds are unnecessary.
    mob_after  = facts.get("opponent_mobility_after")
    mob_before = facts.get("opponent_mobility_before")
    if mob_after is not None and mob_before is not None:
        if mob_after < mob_before:
            delta = mob_before - mob_after
            seeds.append(
                f"opponent mobility changes from {mob_before} to {mob_after} — "
                f"reduces opponent mobility by {delta}, restricting available replies"
            )
        elif mob_after > mob_before:
            delta = mob_after - mob_before
            seeds.append(
                f"opponent mobility changes from {mob_before} to {mob_after} — "
                f"increases opponent mobility by {delta}"
            )
        else:
            seeds.append(
                f"opponent mobility remains at {mob_before} — "
                "no change in opponent mobility"
            )

    our_mob_before = facts.get("our_mobility_before")
    our_mob_after  = facts.get("our_mobility_after")
    if our_mob_before is not None and our_mob_after is not None:
        if our_mob_after > our_mob_before:
            delta = our_mob_after - our_mob_before
            seeds.append(
                f"our mobility changes from {our_mob_before} to {our_mob_after} — "
                f"increases our mobility by {delta}"
            )
        elif our_mob_after < our_mob_before:
            delta = our_mob_before - our_mob_after
            seeds.append(
                f"our mobility changes from {our_mob_before} to {our_mob_after} — "
                f"decreases our mobility by {delta}"
            )
        else:
            seeds.append(
                f"our mobility remains at {our_mob_before} — "
                "no change in our mobility"
            )

    # ── Mobility-gap direction (Phase G, Step 3) ─────────────────────────
    # Emit a single grounded direction summary so the LLM does not have to
    # compute the absolute-distance change itself.  Addresses T4 audit
    # residuals (gap-direction misclaims, false 'unchanged' on symmetric
    # decreases, restriction-implication when opp mobility increased).
    _gap_seed = _mobility_gap_seed(facts)
    if _gap_seed is not None:
        seeds.append(_gap_seed)

    # ── Strategic interpretation (LOW-STRENGTH, supporting context only) ────────────
    # Derived purely from path geometry and fact values. No risky assumptions.
    # These seeds must NEVER be the primary justification in the paragraph.
    _path    = chosen_path or []
    _mtype   = chosen_move.get("type", "")
    _src     = _path[0]  if len(_path) >= 1 else None
    _dst     = _path[-1] if len(_path) >= 2 else None
    _src_row = _src[0]   if isinstance(_src, (list, tuple)) and len(_src) >= 1 else None
    _dst_row = _dst[0]   if isinstance(_dst, (list, tuple)) and len(_dst) >= 1 else None
    _dst_col = _dst[1]   if isinstance(_dst, (list, tuple)) and len(_dst) >= 2 else None

    # (A) Development: simple non-capture move, color-aware forward direction.
    # RED moves toward lower row numbers; BLACK moves toward higher row numbers.
    # ONLY emitted when player is explicitly RED or BLACK — never when unknown (0).
    if _mtype == "simple" and cap == 0 and player != 0:
        if _src_row is not None and _dst_row is not None:
            _is_forward = (_dst_row < _src_row) if player == RED else (_dst_row > _src_row)
        else:
            _is_forward = False
        if _is_forward:
            seeds.append("The piece advances forward without capturing.")

    # (B) Back-row origin: color-aware back row detection.
    # RED back row = row 7; BLACK back row = row 0.
    # ONLY emitted when player is explicitly RED or BLACK — never when unknown (0).
    if _src_row is not None and player != 0:
        _is_back_row = (_src_row == 7) if player == RED else (_src_row == 0)
        if _is_back_row:
            # Fix 2C: condition seed text on the actual weakens_king_row fact.
            # The old template always said "slightly weakens" regardless of the
            # symbolic fact value, causing a seed-fact mismatch and a verifier
            # UNSUPPORTED verdict whenever weakens_king_row=False.
            _actually_weakens = bool(facts.get("weakens_king_row", False))
            if _actually_weakens:
                seeds.append("A back-row piece is moved, weakening the defensive structure.")
            else:
                seeds.append("A back-row piece is moved; the defensive structure remains intact.")


    # (C) Positional (quiet) move: no captures
    if cap == 0:
        seeds.append("The move improves piece placement without capturing.")

    # (D) Center direction: emit geometric column seed ONLY when the engine
    # confirms center_control=True.  Without this gate the seed was injected for
    # every move to columns 2-5 regardless of center_control, causing 19.9% of
    # turns to carry a false "center of the board" claim that also bypassed the
    # verifier's seed-exempt check.
    if facts.get("center_control") and _dst_col is not None:
        seeds.append(f"The destination is in the center of the board (column {_dst_col}).")

    # (E) Edge-awareness: destination on board edge limits diagonal flexibility.
    # Columns 0 and 7 are the board edges; only one diagonal direction is available.
    if _dst_col in {0, 7}:
        seeds.append(
            f"The destination is on the board edge (column {_dst_col}), "
            f"which limits diagonal flexibility to one direction only."
        )

    # ── BUG-3: Single-legal-move context ─────────────────────────────────
    # When there is only one legal move the chosen move is not "best" or
    # "strongest" — it is the only option.  Replace the comparison and
    # minimax confirmation seeds with a single explicit statement so the
    # LLM cannot use superlatives that imply a comparison set.
    if len(all_candidates) <= 1:
        # Insert negative-grounding seeds before the closing forced-move seed
        # so the LLM sees them in the body of the seed list.
        seeds.extend(_negative_grounding_seeds(facts, len(all_candidates), seeds))
        mm = _get_minimax_score(chosen_move)
        if mm is not None:
            seeds.append(
                f"This is the only legal move available; "
                f"the engine assigns it a minimax score of {mm:.1f}."
            )
        else:
            seeds.append("This is the only legal move available.")
        return seeds

    # ── Comparison vs next-best alternative ──────────────────────────────
    alternatives = [
        (i, m) for i, m in enumerate(all_candidates)
        if m.get("path") != chosen_path
    ]
    if alternatives:
        best_alt_idx, best_alt = max(
            alternatives,
            key=lambda im: _get_minimax_score(im[1]) if _get_minimax_score(im[1]) is not None else float("-inf"),
        )
        cmp = _find_comparison_seed(
            facts,
            best_alt.get("facts") or {},
            best_alt_idx,
        )
        if cmp:
            seeds.append(cmp)

    # ── Negative-grounding seeds (Phase G, Step 1) ──────────────────────
    # Emit explicit negatives for the predicates the human audit showed are
    # most commonly fabricated as positives.  Placed before the minimax
    # confirmation so the minimax line remains the final seed.
    seeds.extend(_negative_grounding_seeds(facts, len(all_candidates), seeds))

    # ── Minimax confirmation (always last) ──────────────────────────────
    mm = _get_minimax_score(chosen_move)
    if mm is not None:
        seeds.append(f"The engine scores this move {mm:.1f} — {_minimax_wording_label(mm)}.")

    return seeds


def _build_seed_reasoning_prompt(chosen_move: dict, seeds: list[str]) -> str:
    """Build the user prompt for a seed-based reasoning call.

    Routes by move class:
      - 'tactical' moves use the existing causal-explanation prompt (3-5 sentences).
      - 'quiet'    moves use a shorter factual prompt (2-3 sentences) that
                   suppresses strategic embellishment.  See _classify_move_intent.
    """
    path  = chosen_move.get("path", [])
    mtype = chosen_move.get("type", "simple")
    facts = chosen_move.get("facts") or {}
    move_class = _classify_move_intent(facts)

    lines = [
        f"Chosen move: type={mtype}  path={path}",
        f"Move class: {move_class}",
        "",
        "Verified reasoning seeds (use ONLY these — do not add unsupported claims):",
    ]
    for i, s in enumerate(seeds, 1):
        lines.append(f"  {i}. {s}")
    _coord_hint = f"REQUIRED: reference the move path {path} at least once in your paragraph."

    # Common forced-move framing rule used by both variants.
    _forced_rule = (
        "Forced-move framing rule (strict): IF AND ONLY IF one of the seeds above "
        "explicitly states the move is the only legal move available (e.g., "
        "'only legal move', 'mandatory jump', 'must capture', or equivalent "
        "forced-move wording), then the reasoning paragraph MUST open with that "
        "fact and not bury it in the closing sentence. "
        "Otherwise, do NOT assert that the move is forced, only-legal, mandatory, "
        "or has no alternative. A seed saying 'the opponent is forced to respond' "
        "describes the OPPONENT's reply, not our choice, and does NOT make our "
        "move forced."
    )

    if move_class == "quiet":
        # Quiet positional move: short, factual, no strategic storytelling.
        lines += [
            "",
            "This is a QUIET positional move (no captures, no immediate threat,",
            "no forced opponent jump, no promotion, and no large mobility shift).",
            "Write a SHORT factual paragraph of 2 OR 3 sentences ONLY.  Describe",
            "what the move literally does: starting and destination squares, any",
            "change in mobility, safety relative to recapture, and the engine score.",
            "Do NOT use the words: 'pressure', 'forces', 'forcing', 'control',",
            "'controls', 'controlling', 'influence', 'initiative', 'long-term',",
            "'structural', 'positional pressure', 'tactical pressure', 'dominance',",
            "'restricts', 'restricting', 'restriction', 'narrows the gap',",
            "'maintains initiative', 'creates an immediate threat', 'creates a threat',",
            "'central control', 'central board presence', 'centralizes', 'centralized',",
            "'only legal move', 'only viable option', 'no alternative', 'forced move',",
            "'reduces opponent mobility', 'restricts opponent mobility', 'N points better'.",
            "Do NOT invent strategic causality and do NOT claim consequences not",
            "present in the seed list.  If a seed states a negative fact (e.g.,",
            "'does not create an immediate threat'), use that wording rather than",
            "asserting the opposite.",
            "minimax_score must appear only in the final sentence as confirmation.",
            _forced_rule,
            _coord_hint,
            'Reply with ONLY: {"reasoning": "<your paragraph>"}',
        ]
    else:
        # Tactical move: 3-5 sentence causal explanation with anti-hallucination guards.
        lines += [
            "",
            "Write a single paragraph (3-5 sentences) that explains WHY this move was chosen.",
            "Use the facts above as evidence — each sentence should convey a reason or consequence.",
            "Do not mechanically list the seeds; synthesize them into a causal explanation.",
            "minimax_score must appear only in the final sentence as confirmation.",
            "ANTI-HALLUCINATION HARD RULES (audit-driven — read carefully):",
            "  IMMEDIATE THREAT:",
            "    - Do NOT write 'creates an immediate threat', 'creates immediate pressure',",
            "      'creates a threat', 'threatens the opponent', 'pressures the opponent',",
            "      or 'forces the opponent' UNLESS a seed explicitly states 'forces the",
            "      opponent to respond to an immediate threat'. If a seed states the move",
            "      'does not create an immediate threat', acknowledge that wording instead.",
            "  FORCED MOVE (our move):",
            "    - Do NOT write 'only legal move', 'only viable option', 'forced move',",
            "      'no alternative', 'no other option', or 'must play' UNLESS a seed",
            "      explicitly states 'This is the only legal move available'. A seed saying",
            "      'the opponent is forced to respond' is about the OPPONENT'S reply and",
            "      does NOT make our move forced.",
            "  COMPARISON NUMBERS:",
            "    - Do NOT write 'N points better', 'N points stronger', 'best by N', or",
            "      similar numeric magnitudes UNLESS a seed explicitly contains the exact",
            "      number in a 'points better' phrase. Use qualitative language otherwise.",
            "  CENTER CONTROL:",
            "    - Do NOT write 'controls the center', 'central control', 'central board",
            "      presence', 'centralizes', 'centralized', 'central influence', or any",
            "      strategic-center phrasing UNLESS a seed says 'The move gains central",
            "      board control' or 'The move claims central control'. A geometric column",
            "      number is NOT center control.",
            "  MOBILITY REDUCTION:",
            "    - Do NOT write 'reduces opponent mobility', 'restricts opponent mobility',",
            "      'narrows opponent options', 'cuts opponent moves', 'limits opponent",
            "      replies', or 'opponent mobility decreases' UNLESS a seed explicitly",
            "      states 'reduces opponent mobility by N'. If a mobility seed says",
            "      'remains at N' or 'no change in opponent mobility', the mobility is",
            "      unchanged — do not assert any reduction.",
            "  NEAR PROMOTION:",
            "    - Do NOT claim 'near promotion', 'one step from promotion', 'toward",
            "      promotion' UNLESS a seed states 'one step from promotion'.",
            "  VAGUE STRATEGIC FILLER (always forbidden):",
            "    - 'creates pressure', 'maintains pressure', 'improves control',",
            "      'forces defense', 'strong positional move', 'tactical pressure',",
            "      'strategic edge', 'dominates the position'.",
            _forced_rule,
            _coord_hint,
            'Reply with ONLY: {"reasoning": "<your paragraph>"}',
        ]
    return "\n".join(lines)


def _generate_seeded_reasoning(
    chosen_move: dict,
    all_candidates: list,
    player: int = 0,
    score_state: Optional[str] = None,
) -> tuple[Optional[str], list[str]]:
    """
    Build grounded seeds for chosen_move, then call the LLM once to turn them
    into a fluent paragraph.
    NEVER modifies chosen_move or any candidate.
    NEVER calls safety_filter, override, or scoring.
    player  INT constant (RED or BLACK); 0 = unknown (safe fallback).
    score_state  Optional state.score_state value.  Forwarded to
                 _build_grounded_reasoning_seeds so adversity seeds activate
                 by position state rather than the per-move minimax_score.
    Returns (reasoning_string_or_None, seeds_list).
    """
    import time as _time

    seeds = _build_grounded_reasoning_seeds(
        chosen_move, all_candidates, player=player, score_state=score_state,
    )

    # ── Ablation: force-empty seed list ────────────────────────────────────────
    # When EXPLAINER_SEEDS_DISABLED is on, we DROP every grounded seed but still
    # call the LLM with the same prompt skeleton.  The "use ONLY these" line
    # remains, which means the LLM must produce reasoning without any
    # symbolic guardrail — exactly the ungrounded baseline the experiment
    # needs.  No deterministic seed-derived fallback is taken in this mode.
    if _seeds_disabled():
        seeds = []
        print("[EXPLAINER_SEED_REASONING] ablation: seeds disabled (EXPLAINER_SEEDS_DISABLED=1)")
    elif not seeds:
        # Normal mode: no grounded seed produced ⇒ leave reasoning to the
        # deterministic fallback in _explain_chosen_move.
        return None, []

    user_prompt = _build_seed_reasoning_prompt(chosen_move, seeds)
    print(f"[EXPLAINER_SEED_REASONING] seeds={seeds}")

    raw: Optional[str] = None
    _seed_waits = (20, 30, 40, 20, 30)
    for api_try in range(6):
        try:
            raw = call_explainer(EXPLAINER_SEED_REASONING_SYSTEM, user_prompt)
            break
        except Exception as e:
            if api_try < len(_seed_waits):
                wait = _seed_waits[api_try]
                print(f"[EXPLAINER_SEED_REASONING] api error (try={api_try + 1}): {e} — waiting {wait}s")
                _time.sleep(wait)

    if raw is None:
        print("[EXPLAINER_SEED_REASONING] api call failed; keeping previous reasoning")
        return None, seeds

    result = _extract_refinement_reasoning(raw)  # reuse existing JSON parser
    if not result:
        print("[EXPLAINER_SEED_REASONING] could not parse response; keeping previous reasoning")
        return None, seeds

    return result, seeds


# ── Binary comparative fast-path (2 legal moves) ─────────────────────────────
# Called only when len(candidates) == 2 (one chosen move + one alternative).
# Builds a deterministic 1-2 sentence comparison without any LLM call and
# without the category-grouping template used by generate_comparative_reasoning.
# Never raises; returns None on any error so the chosen-move paragraph stands.

def _generate_binary_comparative(
    chosen: dict,
    candidates: list,
    chosen_facts: dict,
) -> Optional[str]:
    """Deterministic 2-candidate binary comparison (no LLM, no grouping)."""
    chosen_path = (chosen or {}).get("path")
    alt: Optional[dict] = None
    for c in (candidates or []):
        if c.get("path") != chosen_path:
            alt = c
            break
    if alt is None:
        return None

    alt_facts = alt.get("facts") or {}
    cf = chosen_facts or {}

    def _notation(path: Any) -> str:
        try:
            s, e = path[0], path[-1]
            return f"{int(s[0])},{int(s[1])}→{int(e[0])},{int(e[1])}"
        except Exception:
            return "?"

    chosen_score = cf.get("minimax_score")
    alt_score    = alt_facts.get("minimax_score")
    alt_notation = _notation(alt.get("path") or [])

    # Sentence 1: score margin (always present).
    if chosen_score is not None and alt_score is not None:
        gap = chosen_score - alt_score
        s1 = (
            f"The only alternative [1] ({alt_notation}) evaluates at "
            f"{alt_score:.1f} against the chosen move's {chosen_score:.1f} — "
            f"a margin of {gap:.1f} points in favour of the chosen move."
        )
    else:
        s1 = (
            f"The only alternative [1] ({alt_notation}) is rated lower by "
            f"the engine than the chosen move."
        )

    # Sentence 2: first salient factual difference between the two candidates.
    chosen_caps     = int(cf.get("captures_count", 0) or 0)
    alt_caps        = int(alt_facts.get("captures_count", 0) or 0)
    chosen_threatened = bool(cf.get("moved_piece_is_threatened", False))
    alt_threatened    = bool(alt_facts.get("moved_piece_is_threatened", False))
    chosen_recap    = bool(cf.get("opponent_can_recapture", False))
    alt_recap       = bool(alt_facts.get("opponent_can_recapture", False))
    chosen_isolated = bool(cf.get("leaves_piece_isolated", False))
    alt_isolated    = bool(alt_facts.get("leaves_piece_isolated", False))

    s2: Optional[str] = None
    if chosen_caps != alt_caps:
        if chosen_caps > alt_caps:
            s2 = (
                f"The chosen move captures {chosen_caps} piece(s) while [1] "
                f"captures only {alt_caps}, providing greater immediate material "
                f"gain despite the shared recapture risk."
            )
        else:
            s2 = (
                f"Move [1] captures {alt_caps} piece(s) against the chosen "
                f"move's {chosen_caps}, but the engine still favours the chosen "
                f"move based on downstream evaluation."
            )
    elif not chosen_threatened and alt_threatened:
        s2 = (
            "The chosen move does not leave the moved piece under immediate "
            "threat; [1] does, making the chosen path safer despite no captures."
        )
    elif not chosen_recap and alt_recap:
        s2 = (
            "The chosen move avoids immediate recapture while [1] allows it, "
            "giving the chosen move a clear safety edge."
        )
    elif not chosen_isolated and alt_isolated:
        s2 = (
            "Move [1] would leave the moved piece without adjacent support; "
            "the chosen move maintains connectivity."
        )

    return (s1 + " " + s2) if s2 else s1


# ── Proposal-authoritative explanation path ──────────────────────────────────
# Sole reasoning entry point in the simplified pipeline. The move was already
# selected by deterministic_proposal_node; this function only produces the
# natural-language explanation for it. It NEVER selects, re-scores, re-ranks,
# overrides, retries, tie-breaks, or mutates chosen_move.

def _explain_chosen_move(state: CheckersState) -> dict:
    """
    Generate a grounded explanation for the proposal-chosen move.

    Inputs read:
      - state.chosen_move          (move selected by proposal — IMMUTABLE)
      - state.chosen_move_score    (proposal's minimax score for that move)
      - state.legal_moves          (full menu, for seeded comparative context)
      - state.unchosen_moves       (alternatives, for comparative seeds only)
      - state.score_state          (for adversity / score-state seeds)
      - state.symbolic_best_move   (for the LLM-vs-symbolic agreement metric)

    Outputs returned (LangGraph state delta):
      - chosen_move                (PASS-THROUGH, unchanged)
      - last_move_reasoning        (the generated explanation)
      - ranker_diagnostics         (reasoning provenance + neutral legacy keys)
      - chosen_move_facts          (read-only mirror of chosen_move["facts"])
      - ranker_filtered_menu       (candidate-menu snapshot, evaluation only)
      - llm_agreed_with_symbolic_best (Boolean metric)


    Guarantees:
      - chosen_move is NEVER mutated, re-selected, or overridden.
      - Safety filter is NEVER called.
      - Override guardrail is NEVER called.
      - Decision-time retry loop is NEVER called.
      - The LLM is used ONLY for reasoning generation (seeds → prose) and
        for the reasoning-only truthfulness refinement loop.
      - chosen_move passes through unchanged: proposal → ranker → updater.
    """
    chosen = state.chosen_move
    legal = state.legal_moves or []

    print(
        f"[RANKER] proposal_authoritative_path: "
        f"chosen_path={chosen.get('path')} "
        f"chosen_score={state.chosen_move_score} "
        f"legal_count={len(legal)} "
        f"unchosen_count={len(state.unchosen_moves)}"
    )

    # ── 1. Build seed-based reasoning (LLM-authored) ─────────────────────────
    # The reasoning text returned here is the verbatim LLM seed-prose paragraph.
    # If the LLM call fails completely, `reasoning` is None / empty string — we
    # log it as-is rather than substitute Python-generated text.  The
    # deterministic seed-derived summary that previously back-filled this slot
    # has been permanently retired; the final logged reasoning is now strictly
    # LLM-authored (or empty when the LLM produced nothing).
    _score_state = _resolve_score_state_for_seeds(state)
    _candidates = legal if legal else [chosen]
    reasoning, _active_seeds = _generate_seeded_reasoning(
        chosen, _candidates,
        player=state.current_player,
        score_state=_score_state,
    )
    if not isinstance(reasoning, str):
        reasoning = ""

    # ── 2. Truthfulness check + refinement (LLM-authored sentence rewrite) ───
    # `_check_reasoning_truthfulness` is read-only.  `_refine_reasoning` may
    # ask the LLM to rewrite specific sentences, but every replacement word
    # comes from the LLM — Python only routes (picks which sentences to send
    # back for replacement).  If the LLM cannot resolve every contradiction,
    # the unrefined LLM text is kept.  Python never overwrites the paragraph.
    _chosen_facts = chosen.get("facts") or {}
    _initial_contradictions = (
        _check_reasoning_truthfulness(reasoning, _chosen_facts, seeds=_active_seeds)
        if reasoning else []
    )
    _reasoning_retry_count = 0
    # Always False going forward — the deterministic seed-summary fallback has
    # been retired from the pipeline.  Field is kept in diagnostics for log-
    # schema compatibility with older traces.
    _reasoning_is_seed_fallback = False

    # Snapshot the reasoning BEFORE the LLM-driven refinement loop modifies
    # any sentences.  Evaluation-only: enables pre/post repair metrics.
    _raw_reasoning_pre_refinement: Optional[str] = reasoning or None

    if _initial_contradictions and reasoning:
        for _w in _initial_contradictions:
            print(f"[EXPLAINER_TRUTHFULNESS] {_w}")
        reasoning, _reasoning_retry_count, _resolved = _refine_reasoning(
            reasoning=reasoning,
            chosen_move=chosen,
            initial_contradictions=_initial_contradictions,
            max_attempts=2,
            seeds=_active_seeds,
        )
        if not _resolved:
            # Refinement loop could not resolve every contradiction.  Per the
            # post-audit policy, we KEEP the unrefined LLM text exactly as the
            # LLM produced it — no Python-built summary, no fact injection.
            # The remaining contradictions stay visible in the metric layer so
            # the evaluation reflects true LLM behaviour.
            print(
                f"[EXPLAINER_TRUTHFULNESS] reasoning still contains contradictions "
                f"after {_reasoning_retry_count} refinement attempt(s); "
                "keeping unrefined LLM text (no deterministic fallback)"
            )

    # Normalize whitespace and cap length (format-only, no semantic change).
    reasoning = re.sub(r"\s+", " ", reasoning).strip() if reasoning else ""
    if len(reasoning) > 1500:
        reasoning = reasoning[:1497] + "..."

    # ── 3. Thesis instrumentation (reuses existing pattern) ───────────────────
    llm_agreed: bool | None = None
    if state.symbolic_best_move is not None:
        sym_path = state.symbolic_best_move.get("path")
        chosen_path = chosen.get("path")
        if sym_path is not None and chosen_path is not None:
            llm_agreed = (sym_path == chosen_path)

    # ── 4. Final contradiction check for diagnostics ──────────────────────────
    _reasoning_final_contradictions = _check_reasoning_truthfulness(
        reasoning, _chosen_facts, seeds=_active_seeds,
    )
    _reasoning_has_unresolved = bool(_reasoning_final_contradictions)

    # ── 5. Comparative stage (env-gated, default OFF) ─────────────────────────
    # The chosen-move pipeline above is byte-identical regardless of this flag.
    # A comparative failure must never block the chosen output: the try/except
    # ensures any unexpected exception from generate_comparative_reasoning is
    # swallowed and reasoning falls back to the chosen-move paragraph only.
    _comparative_diag: dict = {}
    _comparative_text: Optional[str] = None

    if _comparative_stage_enabled():
        if len(legal) >= 3:
            try:
                _comparative_text = generate_comparative_reasoning(
                    chosen,
                    legal,
                    _chosen_facts,
                    diagnostics_out=_comparative_diag,
                )
            except Exception:
                pass  # comparative failure must never block chosen output
        elif len(legal) == 2:
            # Exactly 2 legal moves: deterministic binary comparison.
            # Bypasses the LLM + category-grouping template; never blocks output.
            try:
                _binary_text = _generate_binary_comparative(chosen, legal, _chosen_facts)
            except Exception:
                _binary_text = None
            if _binary_text:
                _comparative_text = _binary_text
                _comparative_diag.update({
                    "comparative_was_skipped":                False,
                    "comparative_skip_reason":                None,
                    "comparative_paragraph_text":             _binary_text,
                    "comparative_seeds":                      [],
                    "comparative_groups":                     {},
                    "comparative_generation_samples_used":    0,
                    "comparative_sample_contradiction_counts": [],
                    "comparative_generation_short_circuited": False,
                    "comparative_initial_contradictions":     0,
                    "comparative_final_contradictions":       0,
                    "comparative_refinement_attempts":        0,
                    "comparative_provider":                   "deterministic_binary",
                })
            else:
                _comparative_diag["comparative_was_skipped"] = True
                _comparative_diag["comparative_skip_reason"] = "binary_comparative_failed"
        else:
            # Single legal move — nothing to compare against.
            _comparative_diag["comparative_was_skipped"] = True
            _comparative_diag["comparative_skip_reason"] = "single_legal_move"

    # Snapshot the chosen-only reasoning BEFORE appending the comparative
    # paragraph.  Evaluation-only: the pre_post_repair evaluator uses this
    # field as its "post" baseline so comparative-paragraph claims are never
    # scored against chosen_move_facts (Fix 1 — evaluator contamination).
    _chosen_reasoning: str = reasoning

    if _comparative_text:
        reasoning = reasoning + "\n\n" + _comparative_text

    # ── 6. Build explainer_diagnostics ──────────────────────────────────────────
    _explainer_diagnostics: dict[str, Any] = {
        # ── Override/retry state (always neutral — proposal_authoritative) ────
        "override_retry_attempts":      0,
        "override_retry_resolved":      False,
        "override_fallback_applied":    False,
        "override_branch_name":         None,
        # ── Attribution ──────────────────────────────────────────────────────
        "raw_llm_choice_path":          None,
        "final_chosen_idx":             next(
            (i for i, m in enumerate(legal) if m.get("path") == chosen.get("path")),
            None,
        ),
        "final_chosen_path":            chosen.get("path"),
        "final_choice_source":          "proposal_authoritative",
        # ── Evaluation logging ───────────────────────────────────────────────
        "api_call_failure_count":        0,
        "reasoning_seeds":               _active_seeds,
        "reasoning_final_contradictions": _reasoning_final_contradictions,
        "reasoning_has_unresolved_contradiction": _reasoning_has_unresolved,
        "reasoning_refinement_retry_count": _reasoning_retry_count,
        "reasoning_is_seed_fallback":    _reasoning_is_seed_fallback,
        # ── Pre-repair contradiction diagnostics ─────────────────────────────
        "reasoning_initial_contradictions": list(_initial_contradictions),
        "reasoning_contradiction_detected": bool(_initial_contradictions),
        "reasoning_contradiction_repaired": (
            bool(_initial_contradictions)
            and not _reasoning_is_seed_fallback
            and not _reasoning_final_contradictions
        ),
        "final_chosen_path_in_legal_moves": any(
            m.get("path") == chosen.get("path") for m in legal
        ),
        # ── Phase 2 provenance ───────────────────────────────────────────────
        "best_score_tie_count":          0,
        "minimax_best_path":             chosen.get("path"),
        "minimax_best_score":            state.chosen_move_score,
        "next_best_minimax_score":       (
            max(
                (_get_minimax_score(m) for m in state.unchosen_moves
                 if _get_minimax_score(m) != float("-inf")),
                default=None,
            ) if state.unchosen_moves else None
        ),
        "raw_llm_reasoning_pre_refinement": _raw_reasoning_pre_refinement,
        # Chosen-only reasoning (no comparative paragraph appended).
        # The pre_post_repair evaluator uses this as the post-repair baseline
        # so comparative-paragraph claims are not scored against chosen_move_facts.
        "chosen_reasoning":              _chosen_reasoning or None,
        # ── Ablation provenance ──────────────────────────────────────────────
        # Evaluation-only fields; never read by any decision logic.
        # run_tag is "seed_on" / "seed_off" by default and may be overridden
        # via the EXPLAINER_RUN_TAG env var for arbitrary experiment labels.
        "seeds_disabled":                _seeds_disabled(),
        "run_tag":                       _current_run_tag(),
        "retry_all_paths":               [],
        "retry_rejection_reasons":       [],
        "tie_break_reason":              None,
        "tied_candidate_paths":          [],
    }

    if _comparative_stage_enabled():
        _explainer_diagnostics.update(_comparative_diag)

    return {
        "chosen_move": chosen,                    # pass-through UNCHANGED
        "last_move_reasoning": reasoning,
        "last_completed_node": "explainer_agent",
        "explainer_filtered_menu": _build_explainer_filtered_menu_snapshot(legal),
        "llm_agreed_with_symbolic_best": llm_agreed,
        "explainer_diagnostics": _explainer_diagnostics,
        "chosen_move_facts": _chosen_facts or None,
    }


# ── Main node ─────────────────────────────────────────────────────────────────
#
# In the simplified proposal-authoritative pipeline, ranker_agent is a
# pure explanation node. It NEVER selects, re-scores, re-ranks, retries,
# or overrides the move chosen by deterministic_proposal_node.
#
# Invariants (enforced below):
#   • state.chosen_move is set by deterministic_proposal_node
#   • state.chosen_move_score is set by deterministic_proposal_node
#   • If either is None, the upstream graph is misconfigured and we raise
#     immediately rather than fall back to any decision-making code path.
#
# All legacy decision-making helpers defined above in this file
# (_apply_safety_filter, _override_if_llm_chose_much_worse_minimax,
# _audit_override, _choose_best_minimax_with_origin, retry/feedback
# builders, RANKER_SYSTEM_PROMPT, build_ranker_user_prompt, …) are NOT
# reachable from this entry point. They are retained at module scope only
# because external evaluation scripts and regression tests import them by
# name. None of them are invoked when ranker_agent runs inside the
# simplified pipeline.

def explainer_agent(state: CheckersState) -> dict:
    """
    Reasoning-only node for the simplified proposal-authoritative pipeline.

    Responsibilities:
      1. Read the move already chosen by deterministic_proposal_node
         (state.chosen_move + state.chosen_move_score).
      2. Generate a grounded natural-language explanation for that move,
         comparing it against state.unchosen_moves only as context for
         describing why alternatives are weaker.
      3. Emit diagnostics and the chosen_move_facts mirror for evaluation.

    Guarantees:
      • state.chosen_move is returned unchanged.
      • No safety filter, override audit, retry loop, tie-break, or
        fallback selector is ever invoked from this function.
    """
    if state.chosen_move is None or state.chosen_move_score is None:
        raise RuntimeError(
            "ranker_agent invariant violated: deterministic_proposal_node must "
            "set both chosen_move and chosen_move_score before ranker_agent runs. "
            "Got chosen_move=%r chosen_move_score=%r."
            % (state.chosen_move, state.chosen_move_score)
        )
    return _explain_chosen_move(state)
