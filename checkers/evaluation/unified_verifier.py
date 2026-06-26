# checkers/evaluation/unified_verifier.py
#
# Single source of truth for "what counts as a contradiction" in the
# proposal-authoritative reasoning evaluator.
#
# Used by BOTH:
#   • the runtime refinement loop (via _check_reasoning_truthfulness)
#   • the evaluator metric layer  (via metrics._record_helpers)
#
# Composed of three layers:
#   1. claim_extractor → claim_verifier (existing) — with the new
#      clause-level negation pre-pass already wired in claim_extractor.
#   2. numeric verifier (E.3) — flags fabricated number tokens that don't
#      match the engine-computed facts.
#   3. schema-leak verifier (E.4) — flags raw "field=value" tokens echoed
#      by the LLM as instruction-inconsistency, optionally upgrading to
#      CONTRADICTED when the asserted value disagrees with the fact.
#
# Output convention:
#   verify_all() returns a list[ClaimRecord] where every record produced by
#   the legacy extractor/verifier is preserved, and the numeric / schema-leak
#   findings are appended as synthetic ClaimRecord entries with distinct
#   claim_type prefixes ("numeric_*", "schema_leak_*").  CONTRADICTED records
#   carry hallucination_type appropriately:
#     numeric mismatch  → FABRICATED_CLAIM
#     schema leak       → INSTRUCTION_INCONSISTENCY (+ FACTUAL_CONTRADICTION
#                                                     when value contradicts fact)
#
# Determinism: same inputs → same outputs.  No LLM calls.  No randomness.
# No imports from checkers.agents / checkers.nodes / checkers.engine /
# checkers.graph — preserves evaluator/runtime layering.

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from checkers.evaluation.claim_extractor import ClaimRecord, extract_claims
from checkers.evaluation.claim_verifier import verify_claims
from checkers.evaluation.forbidden_vocab import (
    ABSOLUTE_FORBIDDEN_VOCAB,
    CONTEXT_FORBIDDEN_VOCAB,
)
from checkers.evaluation.reasoning_taxonomy import (
    ClaimStatus,
    ClaimVerifiability,
    HallucinationType,
)

# ---------------------------------------------------------------------------
# E.3 — Numeric contradiction verifier
# ---------------------------------------------------------------------------
#
# Detects natural-language numeric assertions that disagree with the
# engine-computed facts for the chosen move.  Conservative on purpose:
# only fires when a numeric value is explicitly tied to a specific field
# via one of the patterns below.  Free-floating digits in the prose are
# IGNORED to avoid penalising paraphrased counts (e.g. "captures one piece"
# vs facts.captures_count == 1).

# Word→int for number words 0–12 (small enough to be unambiguous in checkers
# explanations).  Extending past 12 risks confusion with column indices.
_WORD_TO_INT: Dict[str, int] = {
    "zero": 0, "no": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
}


def _to_int(token: str) -> Optional[int]:
    """Parse a token as an int.  Accepts digits and number words 0–12."""
    if token is None:
        return None
    t = token.strip().lower()
    if t in _WORD_TO_INT:
        return _WORD_TO_INT[t]
    try:
        return int(t)
    except ValueError:
        try:
            f = float(t)
            if f.is_integer():
                return int(f)
        except ValueError:
            pass
    return None


def _to_float(token: str) -> Optional[float]:
    if token is None:
        return None
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


# Each rule: (regex, field_name, parser, comparator).  comparator(asserted,
# fact_value) returns True iff CONTRADICTED.  Patterns are matched
# case-insensitively against the lowercased reasoning text.
_NUMERIC_RULES: List[tuple] = [
    # captures_count claims
    (
        re.compile(
            r"captures?\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"\s+(?:opponent\s+|enemy\s+)?pieces?",
            flags=re.IGNORECASE,
        ),
        "captures_count",
        _to_int,
        lambda asserted, fact: asserted is not None and asserted != fact,
    ),
    (
        re.compile(
            r"capture[_\s-]?count\s+of\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)",
            flags=re.IGNORECASE,
        ),
        "captures_count",
        _to_int,
        lambda asserted, fact: asserted is not None and asserted != fact,
    ),
    # net_gain claims
    (
        re.compile(
            r"net\s+gain\s+of\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)",
            flags=re.IGNORECASE,
        ),
        "net_gain",
        _to_int,
        lambda asserted, fact: asserted is not None and asserted != fact,
    ),
    # "from N to M" mobility narrative — match against any mobility pair.
    # This is checked specially below because it spans TWO numbers.
    # (handled in _check_mobility_transition, not in this rule list)

    # "minimax_score=N" / "minimax score of N" / "minimax_score is N"
    (
        re.compile(
            r"minimax[_\s]?score(?:\s+(?:is|of|=))?\s*(=?\s*-?\d+(?:\.\d+)?)",
            flags=re.IGNORECASE,
        ),
        "minimax_score",
        _to_float,
        lambda asserted, fact: (
            asserted is not None
            and isinstance(fact, (int, float))
            # tolerance: 0.5 on integer-ish, 0.01 on fractional
            and abs(asserted - float(fact)) > 0.5
        ),
    ),
]


def _check_mobility_transition(
    text_lower: str,
    facts: Dict[str, Any],
    seeds: Optional[List[str]] = None,
) -> List[ClaimRecord]:
    """
    Flag 'from N to M' mobility narratives whose (N, M) match no known
    before/after pair in the facts dict.

    To avoid false-positives on PIECE-COUNT narratives (the LLM frequently
    writes "from 11 to 9" when describing the opponent's piece count after
    a capture), this verifier now:

      • allows the transition when (N, M) matches ANY known before/after
        pair — mobility OR piece counts;
      • only fires CONTRADICTED when the surrounding 50-char window contains
        a MOBILITY anchor word ("mobility", "moves", "replies", "options").

    A "from N to M" with no mobility anchor and no matching pair is now
    marked UNSUPPORTED (instruction-level fabrication risk) rather than
    CONTRADICTED, so it stays visible without inflating contradiction rates.
    """
    out: List[ClaimRecord] = []
    pairs = [
        ("opponent_mobility_before",        "opponent_mobility_after"),
        ("our_mobility_before",             "our_mobility_after"),
        # Piece-count transitions — opp_pieces / our_pieces are dicts with
        # a "total" key produced by compute_move_facts().
    ]
    mobility_anchors = ("mobility", " moves", "replies", "options", "available")

    def _piece_count_pairs() -> List[tuple]:
        out: List[tuple] = []
        for prefix in ("opp_pieces", "our_pieces"):
            b = facts.get(f"{prefix}_before")
            a = facts.get(f"{prefix}_after")
            if isinstance(b, dict) and isinstance(a, dict):
                if "total" in b and "total" in a:
                    out.append((int(b["total"]), int(a["total"])))
                if "regular" in b and "regular" in a:
                    out.append((int(b["regular"]), int(a["regular"])))
                if "kings" in b and "kings" in a:
                    out.append((int(b["kings"]), int(a["kings"])))
        return out

    pc_pairs = _piece_count_pairs()
    seeds_text = " ".join(s.lower() for s in (seeds or []))

    for m in re.finditer(
        r"from\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"\s+to\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)",
        text_lower,
    ):
        a = _to_int(m.group(1))
        b = _to_int(m.group(2))
        if a is None or b is None:
            continue
        # ── Allow when both numbers appear in the seeds text ──
        if seeds_text and str(a) in seeds_text and str(b) in seeds_text:
            continue
        # ── Allow when (a, b) matches a known mobility pair ──
        matched_mobility = False
        for before_field, after_field in pairs:
            fb = facts.get(before_field)
            fa = facts.get(after_field)
            if isinstance(fb, (int, float)) and isinstance(fa, (int, float)):
                if int(fb) == a and int(fa) == b:
                    matched_mobility = True
                    break
        if matched_mobility:
            continue
        # ── Allow when (a, b) matches a known piece-count pair ──
        if any(fb == a and fa == b for fb, fa in pc_pairs):
            continue
        # ── Decide severity: only CONTRADICTED when there is a mobility
        #    anchor nearby (the LLM is unambiguously talking mobility).
        start, end = m.span()
        window = text_lower[max(0, start - 30): min(len(text_lower), end + 30)]
        has_mobility_anchor = any(w in window for w in mobility_anchors)
        if has_mobility_anchor:
            status = ClaimStatus.CONTRADICTED
            halluc = HallucinationType.FABRICATED_CLAIM
        else:
            # Ambiguous "from N to M" with no mobility anchor and no fact match:
            # log as UNSUPPORTED so it appears in diagnostics but does not
            # inflate the contradiction rate.
            status = ClaimStatus.UNSUPPORTED
            halluc = HallucinationType.FABRICATED_CLAIM
        out.append(ClaimRecord(
            claim_type="numeric_mobility_transition",
            claim_status=status,
            claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
            seed_risk_type=None,
            hallucination_type=halluc,
            matched_phrase=m.group(0),
            matched_seed=None,
            source="unsupported_phrase",
        ))
    return out


# ---------------------------------------------------------------------------
# Mobility-reduction claim verifier
# ---------------------------------------------------------------------------
#
# Mirrors the legacy runtime check in `_check_reasoning_truthfulness`:
# when the reasoning claims the move REDUCES opponent mobility but the
# fact `opponent_mobility_after >= opponent_mobility_before`, that is a
# direct factual contradiction.

_MOBILITY_REDUCTION_PHRASES: tuple = (
    "reduces mobility", "reducing mobility",
    "reduces opponent mobility", "reducing opponent mobility",
    "limits mobility", "limiting mobility", "limiting opponent",
    "restricts mobility", "restricts opponent",
    "fewer moves for", "cuts opponent moves",
    "narrows their options", "tightening their options",
    "constrains their options",
)


def _check_mobility_reduction_claim(
    text_lower: str,
    facts: Dict[str, Any],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    fb = facts.get("opponent_mobility_before")
    fa = facts.get("opponent_mobility_after")
    if not (isinstance(fb, (int, float)) and isinstance(fa, (int, float))):
        return []
    if fa < fb:
        return []  # claim would actually be supported
    out: List[ClaimRecord] = []
    seen: set = set()
    for phrase in _MOBILITY_REDUCTION_PHRASES:
        if phrase in text_lower and phrase not in seen:
            seen.add(phrase)
            out.append(ClaimRecord(
                claim_type="numeric_mobility_reduction",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.FACTUAL_CONTRADICTION,
                matched_phrase=phrase,
                matched_seed=None,
                source="unsupported_phrase",
            ))
            break  # one finding per turn is enough
    return out


# ---------------------------------------------------------------------------
# Safe-reply-count claim verifier
# ---------------------------------------------------------------------------
#
# Mirrors the legacy runtime check.  When the reasoning asserts a specific
# count of safe opponent replies ("five safe replies") and that exact number
# is not present in the seeds OR in opponent_safe_reply_count, the claim is
# fabricated.  Conservative: marked CONTRADICTED only when an explicit fact
# disagrees; UNSUPPORTED otherwise (visible but not contradiction-rate).

_SAFE_REPLY_COUNT_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"\s+safe\s+repl(?:y|ies)\b",
    flags=re.IGNORECASE,
)


def _check_safe_reply_count(
    text_lower: str,
    facts: Dict[str, Any],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    out: List[ClaimRecord] = []
    fact_value = facts.get("opponent_safe_reply_count")
    seen: set = set()
    for m in _SAFE_REPLY_COUNT_RE.finditer(text_lower):
        asserted = _to_int(m.group(1))
        if asserted is None:
            continue
        key = (m.group(0), asserted)
        if key in seen:
            continue
        seen.add(key)
        if isinstance(fact_value, (int, float)) and int(fact_value) == asserted:
            continue  # claim agrees with fact
        if isinstance(fact_value, (int, float)) and int(fact_value) != asserted:
            status = ClaimStatus.CONTRADICTED
            halluc = HallucinationType.FACTUAL_CONTRADICTION
        else:
            # No fact present — runtime calls this "unsupported specific
            # safe-reply count".  Surface it as UNSUPPORTED so it shows in
            # diagnostics but does not inflate the contradiction rate.
            status = ClaimStatus.UNSUPPORTED
            halluc = HallucinationType.FABRICATED_CLAIM
        out.append(ClaimRecord(
            claim_type="numeric_safe_reply_count",
            claim_status=status,
            claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
            seed_risk_type=None,
            hallucination_type=halluc,
            matched_phrase=m.group(0),
            matched_seed=None,
            source="unsupported_phrase",
        ))
    return out


def _check_numeric_claims(
    text_lower: str,
    facts: Dict[str, Any],
    seeds: Optional[List[str]] = None,
) -> List[ClaimRecord]:
    """Run E.3 numeric verification.  Returns synthetic ClaimRecords."""
    if not text_lower or not isinstance(facts, dict):
        return []

    records: List[ClaimRecord] = []

    # Field-specific numeric assertions.
    for regex, field, parser, comparator in _NUMERIC_RULES:
        fact_value = facts.get(field)
        if fact_value is None:
            continue  # cannot judge without the fact
        for m in regex.finditer(text_lower):
            asserted_raw = m.group(1).lstrip("=").strip()
            asserted = parser(asserted_raw)
            if asserted is None:
                continue
            if comparator(asserted, fact_value):
                records.append(ClaimRecord(
                    claim_type=f"numeric_{field}",
                    claim_status=ClaimStatus.CONTRADICTED,
                    claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
                    seed_risk_type=None,
                    hallucination_type=HallucinationType.FABRICATED_CLAIM,
                    matched_phrase=m.group(0),
                    matched_seed=None,
                    source="unsupported_phrase",
                ))

    # Cross-field mobility transition narrative.
    records.extend(_check_mobility_transition(text_lower, facts, seeds=seeds))

    # Deduplicate by (claim_type, matched_phrase) — defensive, in case the
    # same span fires two rules.
    seen: set = set()
    deduped: List[ClaimRecord] = []
    for r in records:
        key = (r.claim_type, r.matched_phrase)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


# ---------------------------------------------------------------------------
# E.4 — Schema-leak detector
# ---------------------------------------------------------------------------
#
# Detects raw schema markers like `creates_immediate_threat=true` echoed
# verbatim by the LLM.  Two outcomes:
#   • value disagrees with the engine fact  → CONTRADICTED
#                                              + FACTUAL_CONTRADICTION
#                                              (still also tagged
#                                              INSTRUCTION_INCONSISTENCY via
#                                              the matched_phrase shape)
#   • value matches the fact OR no fact exists for the field → UNSUPPORTED
#                                              + INSTRUCTION_INCONSISTENCY
#
# The hallucination_type field is constrained to a single enum, so we use
# FACTUAL_CONTRADICTION when the leak is also a contradiction (it carries
# more research signal); the claim_type prefix "schema_leak_" preserves
# the schema-leak provenance.

_SCHEMA_LEAK_RE = re.compile(
    r"\b([a-z_][a-z0-9_]*)\s*=\s*(true|false|-?\d+(?:\.\d+)?)\b",
    flags=re.IGNORECASE,
)


def _normalise_field_value(raw: str) -> Any:
    s = raw.strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return raw


def _values_disagree(asserted: Any, fact: Any) -> bool:
    if isinstance(asserted, bool) and isinstance(fact, bool):
        return asserted != fact
    if isinstance(asserted, (int, float)) and isinstance(fact, (int, float)):
        return abs(float(asserted) - float(fact)) > 0.5
    return False


def _check_schema_leaks(
    reasoning_text: str,
    facts: Dict[str, Any],
) -> List[ClaimRecord]:
    """
    Run E.4 schema-leak detection.  Returns synthetic ClaimRecord entries.
    The original-case reasoning_text is used for the matched_phrase so the
    record reads naturally in evaluator dumps; matching itself is
    case-insensitive.
    """
    if not isinstance(reasoning_text, str) or not reasoning_text:
        return []
    if not isinstance(facts, dict):
        facts = {}

    records: List[ClaimRecord] = []
    for m in _SCHEMA_LEAK_RE.finditer(reasoning_text):
        field = m.group(1).lower()
        asserted = _normalise_field_value(m.group(2))
        fact_value = facts.get(field)

        if fact_value is not None and _values_disagree(asserted, fact_value):
            records.append(ClaimRecord(
                claim_type=f"schema_leak_{field}",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.FACTUAL_CONTRADICTION,
                matched_phrase=m.group(0),
                matched_seed=None,
                source="unsupported_phrase",
            ))
        else:
            # Either the fact is absent OR the value agrees: either way the
            # raw schema string in prose is an instruction violation.
            # Mark CONTRADICTED so it flows through contradictions_only() and
            # contradiction_strings() into the refinement loop.
            records.append(ClaimRecord(
                claim_type=f"schema_leak_{field}",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.INSTRUCTION_INCONSISTENCY,
                matched_phrase=m.group(0),
                matched_seed=None,
                source="unsupported_phrase",
            ))

    # Deduplicate by (claim_type, matched_phrase) — same field=value may
    # repeat in the prose; one record per unique pair is enough.
    seen: set = set()
    out: List[ClaimRecord] = []
    for r in records:
        key = (r.claim_type, r.matched_phrase)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Forbidden-vocabulary detector (instruction-inconsistency layer)
# ---------------------------------------------------------------------------
#
# Same lists the runtime refinement loop consults — sourced from
# checkers.evaluation.forbidden_vocab so both sides agree on which phrases
# count as instruction violations.  Two tiers:
#   ABSOLUTE : never allowed.
#   CONTEXT  : only allowed when the seed list introduces them verbatim.
#
# A hit becomes a CONTRADICTED ClaimRecord with hallucination_type
# INSTRUCTION_INCONSISTENCY.  Treating these as contradictions (not just
# UNSUPPORTED) is what keeps runtime/evaluator in sync — the runtime
# refinement loop also treats them as contradictions.

def _ctx_phrase_negated(text: str, phrase: str) -> bool:
    """Mirror of explainer_agent._ctx_phrase_negated — kept in sync for E.1 parity.

    Return True if every occurrence of *phrase* in *text* is preceded by a
    negation marker within a 35-char window so the forbidden-vocab check should
    be suppressed.  Returns False if any occurrence lacks a negation marker.
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


def _check_forbidden_vocab(
    reasoning_text: str,
    text_lower: str,
    seeds: List[str],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    seeds_text = " ".join(s.lower() for s in (seeds or []))

    records: List[ClaimRecord] = []
    seen: set = set()

    for phrase in ABSOLUTE_FORBIDDEN_VOCAB:
        p_lower = phrase.lower()
        if p_lower in text_lower and ("abs", p_lower) not in seen:
            seen.add(("abs", p_lower))
            records.append(ClaimRecord(
                claim_type=f"forbidden_vocab:{phrase}",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.PARTIALLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.INSTRUCTION_INCONSISTENCY,
                matched_phrase=phrase,
                matched_seed=None,
                source="unsupported_phrase",
            ))

    for phrase in CONTEXT_FORBIDDEN_VOCAB:
        p_lower = phrase.lower()
        if p_lower in text_lower and p_lower not in seeds_text and ("ctx", p_lower) not in seen:
            # Negation-aware: mirror _ctx_phrase_negated from explainer_agent.
            if _ctx_phrase_negated(text_lower, p_lower):
                continue
            seen.add(("ctx", p_lower))
            records.append(ClaimRecord(
                claim_type=f"forbidden_vocab_ctx:{phrase}",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.PARTIALLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.INSTRUCTION_INCONSISTENCY,
                matched_phrase=phrase,
                matched_seed=None,
                source="unsupported_phrase",
            ))

    return records


# ---------------------------------------------------------------------------
# Absence-claim detector (instruction-inconsistency)
# ---------------------------------------------------------------------------
#
# Mirrors the legacy runtime "unsupported absence claim" check.  Certain
# absence assertions ("no kings lost", "pieces unchanged", "no
# vulnerabilities") are strong factual claims that the verifier cannot
# independently confirm — they must be authorised by an explicit seed.
# When the phrase appears in the reasoning AND no seed contains it, the
# claim is treated as a CONTRADICTED instruction-violation so that runtime
# and evaluator remain in sync.

_ABSENCE_CLAIM_PHRASES: tuple = (
    "no kings lost",
    "piece count unchanged",
    "pieces unchanged",
    "no vulnerabilities",
)


# ---------------------------------------------------------------------------
# Opponent single-jump-claim verifier (BUG-2 mirror)
# ---------------------------------------------------------------------------
#
# Mirrors the runtime check added in _check_reasoning_truthfulness().
# When the reasoning asserts a "single jump" for the opponent but
# opponent_jump_count > 1, the claim is a direct factual contradiction.

_SINGLE_JUMP_PHRASES: tuple = (
    "single jump",
    "one jump option",
    "only one jump",
    "limited to a single jump",
    "limited to one jump",
)


def _check_opponent_jump_count(
    text_lower: str,
    facts: Dict[str, Any],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    opp_jc = facts.get("opponent_jump_count")
    if not (isinstance(opp_jc, int) and opp_jc > 1):
        return []
    for phrase in _SINGLE_JUMP_PHRASES:
        if phrase in text_lower:
            return [ClaimRecord(
                claim_type="numeric_opponent_jump_count",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.FACTUAL_CONTRADICTION,
                matched_phrase=phrase,
                matched_seed=None,
                source="unsupported_phrase",
            )]
    return []


def _check_absence_claims(
    text_lower: str,
    seeds: List[str],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    seeds_text = " ".join(s.lower() for s in (seeds or []))
    out: List[ClaimRecord] = []
    for phrase in _ABSENCE_CLAIM_PHRASES:
        if phrase in text_lower and phrase not in seeds_text:
            out.append(ClaimRecord(
                claim_type=f"absence_claim:{phrase}",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.PARTIALLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.INSTRUCTION_INCONSISTENCY,
                matched_phrase=phrase,
                matched_seed=None,
                source="unsupported_phrase",
            ))
    return out


# ---------------------------------------------------------------------------
# Fix 4 — Strategic-claim blind-spot guard
# ---------------------------------------------------------------------------
#
# strategic_initiative / positional_pressure / long_term_compensation are in
# _ALWAYS_VAGUE (claim_verifier.py) and have seed_markers=[] so they can never
# be seed-grounded.  When seeds exist the LLM uses them as free-form prose
# decoration that passes the refinement loop undetected.  When seeds IS
# non-empty these claims are CONTRADICTED (not just VAGUE) because no seed
# can ever support them — activating the refinement loop to remove them.
# When seeds IS empty (no-seed baseline) the upgrade is suppressed to avoid
# false positives in a context where all claims are ungrounded by design.

_STRATEGIC_VAGUE_TYPES: frozenset[str] = frozenset({
    "strategic_initiative",
    "positional_pressure",
    "long_term_compensation",
})


def _upgrade_strategic_vague(
    records: List[ClaimRecord],
    seeds_nonempty: bool,
) -> List[ClaimRecord]:
    if not seeds_nonempty:
        return records
    out: List[ClaimRecord] = []
    for r in records:
        if r.claim_type in _STRATEGIC_VAGUE_TYPES and r.claim_status == ClaimStatus.VAGUE:
            out.append(ClaimRecord(
                claim_type=r.claim_type,
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=r.claim_verifiability,
                seed_risk_type=r.seed_risk_type,
                hallucination_type=HallucinationType.INSTRUCTION_INCONSISTENCY,
                matched_phrase=r.matched_phrase,
                matched_seed=r.matched_seed,
                source=r.source,
            ))
        else:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# BUG-3 — Single-legal-move superlative verifier
# ---------------------------------------------------------------------------
#
# When the seed list signals that this is the only legal move ("only legal
# move" is a substring of one seed), the LLM must not claim it is the
# "strongest choice", "best move", or "highest-ranked option" — there is
# no comparison set to draw that conclusion from.

_SINGLE_LEGAL_INDICATOR: str = "only legal move"
_SINGLE_LEGAL_SUPERLATIVES: tuple = (
    "strongest choice",
    "best move",
    "highest-ranked option",
)


def _check_single_legal_move_superlatives(
    text_lower: str,
    seeds: Optional[List[str]],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    seeds_text = " ".join(s.lower() for s in (seeds or []))
    if _SINGLE_LEGAL_INDICATOR not in seeds_text:
        return []
    for phrase in _SINGLE_LEGAL_SUPERLATIVES:
        if phrase in text_lower:
            return [ClaimRecord(
                claim_type="single_legal_move_superlative",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.FACTUAL_CONTRADICTION,
                matched_phrase=phrase,
                matched_seed=None,
                source="unsupported_phrase",
            )]
    return []


# ---------------------------------------------------------------------------
# BUG-4 — "center of the board" strategic-claim verifier
# ---------------------------------------------------------------------------
#
# When center_control=False and "center of the board" appears in reasoning
# without being introduced by a seed (seed-exempt), the LLM is drawing a
# strategic conclusion from geometry alone.  The geometric seed
# "The destination is in the center of the board (column X)" exempts the
# phrase when the destination column is in the center range — the LLM may
# reference geometry but must not draw tactical-control conclusions from it.

_CENTER_BOARD_PHRASE: str = "center of the board"


def _check_center_of_board_strategic(
    text_lower: str,
    facts: Dict[str, Any],
    seeds: Optional[List[str]],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    if facts.get("center_control") is not False:
        return []
    seeds_text = " ".join(s.lower() for s in (seeds or []))
    if _CENTER_BOARD_PHRASE in text_lower and _CENTER_BOARD_PHRASE not in seeds_text:
        # Allow pure-geometry form: "center of the board (column N)"
        if re.search(r"center of the board\s*\(\s*column", text_lower):
            return []
        return [ClaimRecord(
            claim_type="center_of_board_strategic",
            claim_status=ClaimStatus.CONTRADICTED,
            claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
            seed_risk_type=None,
            hallucination_type=HallucinationType.FACTUAL_CONTRADICTION,
            matched_phrase=_CENTER_BOARD_PHRASE,
            matched_seed=None,
            source="unsupported_phrase",
        )]
    return []


# ---------------------------------------------------------------------------
# BUG-6 — Mobility-disadvantage overclaim verifier
# ---------------------------------------------------------------------------
#
# When opponent_mobility_after > our_mobility_after (the opponent still has
# more moves than us after our move), the mobility disadvantage persists.
# The LLM must use "narrows the gap" rather than claiming the disadvantage
# is solved, addressed, fixed, or eliminated.

_MOBILITY_DISADVANTAGE_OVERCLAIM_PHRASES: tuple = (
    "solves the disadvantage",
    "addresses the disadvantage",
    "fixes the disadvantage",
    "eliminates the disadvantage",
    "solves the mobility disadvantage",
    "addresses the mobility disadvantage",
    "eliminates the mobility gap",
    "closes the gap entirely",
    "erases the mobility gap",
)


def _check_mobility_disadvantage_overclaim(
    text_lower: str,
    facts: Dict[str, Any],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    opp_mob_after = facts.get("opponent_mobility_after")
    our_mob_after = facts.get("our_mobility_after")
    if not (
        isinstance(opp_mob_after, (int, float))
        and isinstance(our_mob_after, (int, float))
        and opp_mob_after > our_mob_after
    ):
        return []
    out: List[ClaimRecord] = []
    for phrase in _MOBILITY_DISADVANTAGE_OVERCLAIM_PHRASES:
        if phrase in text_lower:
            out.append(ClaimRecord(
                claim_type="mobility_disadvantage_overclaim",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.FACTUAL_CONTRADICTION,
                matched_phrase=phrase,
                matched_seed=None,
                source="unsupported_phrase",
            ))
    return out


# ---------------------------------------------------------------------------
# Fix A2 — Our-mobility decrease must-mention verifier
# ---------------------------------------------------------------------------
#
# When a seed explicitly states "decreases our mobility by N", the LLM must
# acknowledge the decrease in its reasoning.  Silently omitting a seeded
# negative fact produces misleading reasoning that passes all claim-level
# checks but hides a material downside.

def _check_mobility_decrease_omission(
    text_lower: str,
    seeds: Optional[List[str]],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    seeds_lower = [s.lower() for s in (seeds or [])]
    decrease_seed = next(
        (s for s in seeds_lower if "decreases our mobility by" in s),
        None,
    )
    if decrease_seed is None:
        return []
    mention_phrases = (
        "decreases", "decrease", "our mobility drops", "our mobility falls",
        "reduces our mobility", "reducing our mobility",
        "our mobility decreases",
        "our mobility narrows", "our mobility shrinks",
        "our mobility contracts",
        "our mobility goes down", "losing mobility",
    )
    if any(p in text_lower for p in mention_phrases):
        return []
    return [ClaimRecord(
        claim_type="mobility_decrease_omission",
        claim_status=ClaimStatus.CONTRADICTED,
        claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
        seed_risk_type=None,
        hallucination_type=HallucinationType.FACTUAL_CONTRADICTION,
        matched_phrase="our-mobility decrease seed",
        matched_seed=decrease_seed,
        source="unsupported_phrase",
    )]


# ---------------------------------------------------------------------------
# Fix A3 — any_piece_isolated vs "no vulnerabilities" verifier
# ---------------------------------------------------------------------------
#
# any_piece_isolated=True means some ally piece is isolated after the move.
# Claiming "no tactical vulnerabilities" when that flag is set is a factual
# contradiction: an isolated piece IS a tactical vulnerability.

_ANY_ISO_VULN_PHRASES: tuple = (
    "no tactical vulnerabilities",
    "ensuring no tactical",
    "no vulnerabilities are created",
    "no tactical vulnerabilities are created",
)


def _check_any_piece_isolated_vulnerability(
    text_lower: str,
    facts: Dict[str, Any],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    if facts.get("any_piece_isolated") is not True:
        return []
    for phrase in _ANY_ISO_VULN_PHRASES:
        if phrase in text_lower:
            return [ClaimRecord(
                claim_type="any_piece_isolated_vulnerability",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.FACTUAL_CONTRADICTION,
                matched_phrase=phrase,
                matched_seed=None,
                source="unsupported_phrase",
            )]
    return []


# ---------------------------------------------------------------------------
# Fix A4 — "narrowing the gap" mobility-direction verifier
# ---------------------------------------------------------------------------
#
# "Narrowing the gap" implies the opponent still leads in mobility.  When
# our_mobility_after >= opponent_mobility_after, the gap was matched or
# reversed — not narrowed.  The phrase is a factual misrepresentation.

def _check_narrowing_gap_direction(
    text_lower: str,
    facts: Dict[str, Any],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    our_mob_af = facts.get("our_mobility_after")
    opp_mob_af = facts.get("opponent_mobility_after")
    if not (
        isinstance(our_mob_af, (int, float))
        and isinstance(opp_mob_af, (int, float))
        and our_mob_af >= opp_mob_af
        and "narrowing the gap" in text_lower
    ):
        return []
    return [ClaimRecord(
        claim_type="narrowing_gap_direction",
        claim_status=ClaimStatus.CONTRADICTED,
        claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
        seed_risk_type=None,
        hallucination_type=HallucinationType.FACTUAL_CONTRADICTION,
        matched_phrase="narrowing the gap",
        matched_seed=None,
        source="unsupported_phrase",
    )]


# ---------------------------------------------------------------------------
# Phase G, Step 3 — Mobility-direction phrase verifier
# ---------------------------------------------------------------------------
#
# Targeted defence-in-depth for three explicit phrasings the audit observed
# as T4 failures (mobility misinterpretation).  Each phrase is checked against
# the engine's exact mobility deltas:
#
#   "mobility remained unchanged" / "mobility is unchanged"
#       → both our_mobility_before==our_mobility_after AND
#                opponent_mobility_before==opponent_mobility_after must hold
#
#   "gap narrowed" / "gap narrows" / "gap narrowing"
#       → |our_after − opp_after| < |our_before − opp_before|
#
#   "gap widened" / "gap widens" / "gap widening"
#       → |our_after − opp_after| > |our_before − opp_before|
#
# The existing "narrowing the gap" check (legacy A4) covers a different
# syntactic form; this verifier is complementary and does not overlap.

import re as _re_dir

_MOB_UNCHANGED_EXACT_RE = _re_dir.compile(
    # Matches the GENERIC "both sides unchanged" claim in two syntactic forms.
    # Side-qualified phrases ("our mobility ...", "opponent mobility ...") are
    # excluded on each arm by negative lookbehind (noun-first) or negative
    # lookahead (verb-first), so one-sided claims do not trigger this check.
    r"(?:"
    # (1) Noun-first form:
    #     "mobility (remained|remains|stays|is|does not alter|does not change)
    #      (unchanged|the same|intact|for both sides)"
    r"(?<!our\s)(?<!opponent\s)"
    r"\bmobility\s+(?:remained?|stays?|is|does\s+not\s+(?:alter|change))\s+"
    r"(?:unchanged|the\s+same|intact|for\s+both\s+sides)"
    r"|"
    # (2) Verb-first form (Phase G surgical-fix mirror of the G1 gap-direction
    #     extension):
    #     "(does/did not | doesn't/didn't) (alter|change|affect|modify) mobility"
    #     — fires only when 'mobility' is NOT side-qualified (the negative
    #     lookahead `(?!our\b)(?!opponent\b)` excludes 'our mobility' and
    #     'opponent mobility').
    r"\b(?:does\s+not|did\s+not|doesn't|didn't)\s+"
    r"(?:alter|change|affect|modify)\s+"
    r"(?!our\b)(?!opponent\b)(?:the\s+)?(?:overall\s+)?mobility"
    r")",
    _re_dir.IGNORECASE,
)
_GAP_NARROWED_DIR_RE = _re_dir.compile(
    # Matches both verb orders:
    #   "gap narrows / narrowed / narrowing"       (noun → verb)
    #   "narrows / narrowed / narrowing the gap"   (verb → noun)
    r"\b(?:"
    r"(?:mobility\s+)?gap\s+narrow(?:s|ed|ing)"
    r"|narrow(?:s|ed|ing)\s+(?:the\s+)?(?:mobility\s+)?gap"
    r")\b",
    _re_dir.IGNORECASE,
)
_GAP_WIDENED_DIR_RE = _re_dir.compile(
    r"\b(?:"
    r"(?:mobility\s+)?gap\s+widen(?:s|ed|ing)"
    r"|widen(?:s|ed|ing)\s+(?:the\s+)?(?:mobility\s+)?gap"
    r")\b",
    _re_dir.IGNORECASE,
)


def _check_mobility_direction_phrases(
    text: Optional[str],
    facts: Optional[Dict[str, Any]],
) -> List[ClaimRecord]:
    """Verify three explicit mobility-direction phrases against engine deltas.

    Pure function.  Returns 0–3 ClaimRecord entries.  Never raises.
    """
    if not text or not facts:
        return []
    ub = facts.get("our_mobility_before")
    ua = facts.get("our_mobility_after")
    ob = facts.get("opponent_mobility_before")
    oa = facts.get("opponent_mobility_after")
    if not all(isinstance(x, (int, float)) for x in (ub, ua, ob, oa)):
        return []
    out: List[ClaimRecord] = []

    # (1) "mobility remained unchanged" — both sides must be unchanged
    if _MOB_UNCHANGED_EXACT_RE.search(text):
        if int(ub) != int(ua) or int(ob) != int(oa):
            out.append(ClaimRecord(
                claim_type="mobility_unchanged_misclaim",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.FACTUAL_CONTRADICTION,
                matched_phrase="mobility remained/stays unchanged",
                matched_seed=None,
                source="unsupported_phrase",
            ))

    gap_before = abs(int(ub) - int(ob))
    gap_after  = abs(int(ua) - int(oa))

    # (2) "gap narrowed" — |gap_after| < |gap_before| required
    if _GAP_NARROWED_DIR_RE.search(text):
        if not (gap_after < gap_before):
            out.append(ClaimRecord(
                claim_type="gap_did_not_narrow",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.FACTUAL_CONTRADICTION,
                matched_phrase="gap narrowed/narrows/narrowing",
                matched_seed=None,
                source="unsupported_phrase",
            ))

    # (3) "gap widened" — |gap_after| > |gap_before| required
    if _GAP_WIDENED_DIR_RE.search(text):
        if not (gap_after > gap_before):
            out.append(ClaimRecord(
                claim_type="gap_did_not_widen",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.FACTUAL_CONTRADICTION,
                matched_phrase="gap widened/widens/widening",
                matched_seed=None,
                source="unsupported_phrase",
            ))

    return out


# ---------------------------------------------------------------------------
# B1.1 — Comparative recapture fabrication verifier
# ---------------------------------------------------------------------------
#
# When the chosen move can be recaptured (opponent_can_recapture=True),
# comparative-context phrases that claim recapture safety are fabricated claims.

_B11_RECAPTURE_PHRASES: tuple = (
    "recapture safety", "avoiding recapture", "avoid recapture risk",
    "recapture-safe", "recapture safety edge",
)


def _check_comparative_recapture_fabrication(
    text_lower: str,
    facts: Dict[str, Any],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    if facts.get("opponent_can_recapture") is not True:
        return []
    for phrase in _B11_RECAPTURE_PHRASES:
        if phrase in text_lower:
            return [ClaimRecord(
                claim_type="comparative_recapture_fabrication",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.FABRICATED_CLAIM,
                matched_phrase=phrase,
                matched_seed=None,
                source="unsupported_phrase",
            )]
    return []


# ---------------------------------------------------------------------------
# Shared helper — sentence-level negation guard
# ---------------------------------------------------------------------------
# Used by reverse-recapture and false-forced-opp-reply checks below.  When a
# negation marker precedes the matched phrase inside the same sentence the
# prose is correctly asserting the opposite and the match must be skipped.

_NEG_SENTENCE_RE = re.compile(
    r"\b(no|not|without|cannot|never|nor|none|nothing|neither|"
    r"avoid(?:s|ing)?|prevent(?:s)?|eliminat(?:e|es|ing)|fails?\s+to)\b",
    re.IGNORECASE,
)


def _sentence_negated(text_lower: str, phrase: str) -> bool:
    """True if a negation marker appears in the same sentence, before phrase."""
    i = text_lower.find(phrase)
    if i < 0:
        return False
    sentence_start = max(
        text_lower.rfind(".", 0, i),
        text_lower.rfind("!", 0, i),
        text_lower.rfind("?", 0, i),
    )
    sentence_start = 0 if sentence_start < 0 else sentence_start + 1
    return bool(_NEG_SENTENCE_RE.search(text_lower[sentence_start:i]))


# ---------------------------------------------------------------------------
# B1.1b — Reverse recapture fabrication
# ---------------------------------------------------------------------------
#
# When opponent_can_recapture=False, reasoning that claims the opponent CAN
# recapture (or that the piece is vulnerable to recapture) is fabricated.
# The negation pre-pass in claim_extractor can mis-treat "if we fail to
# respond" as a polarity flip; this check runs on the raw text so such
# hallucinations are still caught. Mirrors the runtime check in
# explainer_agent._check_reasoning_truthfulness so E.1 invariant holds.

_B11B_FALSE_RECAPTURE_PHRASES: tuple = (
    "opponent can recapture",
    "can be recaptured next",
    "vulnerable to recapture",
    "allows the opponent to recapture",
    "opponent may recapture",
    # Hedged surface forms — same factual claim, softer wording.  Audit
    # showed the LLM falls back to these when forbidden from the direct
    # form, so they must be covered by the same fact-gated check.
    "exposed to recapture",
    "exposed to potential recapture",
    "potential recapture",
    "could be recaptured",
    "risk of recapture",
    "recapture risk",
    "recapture risks",
)


def _check_reverse_recapture_fabrication(
    text_lower: str,
    facts: Dict[str, Any],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    if facts.get("opponent_can_recapture") is not False:
        return []
    for phrase in _B11B_FALSE_RECAPTURE_PHRASES:
        idx = text_lower.find(phrase)
        if idx < 0:
            continue
        window = text_lower[idx : idx + 80]
        # Contrast qualifier: phrase describes the alternative, not chosen move
        if "but not" in window or "not here" in window:
            continue
        # Sentence-level negation guard — the expanded phrase list contains
        # forms like "exposed to recapture" / "recapture risk" that legitimately
        # appear in prose such as "without recapture risk" or "not exposed to
        # recapture next turn".  Skip when a negation marker precedes the
        # phrase inside the same sentence.
        if _sentence_negated(text_lower, phrase):
            continue
        return [ClaimRecord(
            claim_type="reverse_recapture_fabrication",
            claim_status=ClaimStatus.CONTRADICTED,
            claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
            seed_risk_type=None,
            hallucination_type=HallucinationType.FABRICATED_CLAIM,
            matched_phrase=phrase,
            matched_seed=None,
            source="unsupported_phrase",
        )]
    return []


# ---------------------------------------------------------------------------
# B1.1c — False forced-opponent-reply
# ---------------------------------------------------------------------------
#
# When facts.forced_opponent_jump_reply == False, prose phrases that assert
# the opponent is forced to respond / has no choice / must reply are
# fabricated.  Mirrors B1.1b (reverse_recapture_fabrication) in style and
# sentence-level negation handling.  Same E.1 invariant rationale: the
# runtime check in explainer_agent._check_reasoning_truthfulness must have
# an evaluator-side equivalent so refinement-loop diagnostics and metric
# layer stay in sync.

_B11C_FALSE_FORCED_OPP_REPLY_PHRASES: tuple = (
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
)

def _check_false_forced_opp_reply(
    text_lower: str,
    facts: Dict[str, Any],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    if facts.get("forced_opponent_jump_reply") is not False:
        return []
    for phrase in _B11C_FALSE_FORCED_OPP_REPLY_PHRASES:
        if phrase not in text_lower:
            continue
        if _sentence_negated(text_lower, phrase):
            continue
        return [ClaimRecord(
            claim_type="false_forced_opp_reply",
            claim_status=ClaimStatus.CONTRADICTED,
            claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
            seed_risk_type=None,
            hallucination_type=HallucinationType.FABRICATED_CLAIM,
            matched_phrase=phrase,
            matched_seed=None,
            source="unsupported_phrase",
        )]
    return []


# ---------------------------------------------------------------------------
# B1.2 — Tradeoff language without numeric grounding
# ---------------------------------------------------------------------------
#
# "outweighs", "compensates for", "offsets the" imply a quantitative trade.
# If the sentence containing the phrase has no explicit number, the tradeoff
# is asserted without evidence.

_B12_TRADEOFF_PHRASES: tuple = ("outweighs", "compensates for", "offsets the")
_B12_NUMBER_RE = re.compile(
    r'\b\d+(?:\.\d+)?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b',
    re.IGNORECASE,
)


def _check_outweighs_numeric_grounding(
    text_lower: str,
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    for sent in re.split(r'(?<=[.!?])\s+', text_lower):
        has_tradeoff = any(p in sent for p in _B12_TRADEOFF_PHRASES)
        if has_tradeoff and not _B12_NUMBER_RE.search(sent):
            which = next(p for p in _B12_TRADEOFF_PHRASES if p in sent)
            return [ClaimRecord(
                claim_type="tradeoff_without_numeric_grounding",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.PARTIALLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.FABRICATED_CLAIM,
                matched_phrase=which,
                matched_seed=None,
                source="unsupported_phrase",
            )]
    return []


# ---------------------------------------------------------------------------
# B1.3 — Negative-score absolute advantage protection
# ---------------------------------------------------------------------------
#
# When minimax_score < 0, absolute advantage phrases ("positional advantage",
# "strongest option", etc.) are misleading unless paired with relative framing
# ("best available", "least unfavorable", …).

_B13_FORBIDDEN_ADVANTAGE: tuple = (
    "positional advantage", "advantage gained",
    "strongest option", "decisive advantage",
)
_B13_RELATIVE_FRAMING: tuple = (
    "best available", "least unfavorable", "least harmful",
    "highest-evaluated", "relative to", "best of the",
    "only option", "best option available",
)


def _check_negative_score_advantage(
    text_lower: str,
    facts: Dict[str, Any],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    mm = facts.get("minimax_score")
    if not (isinstance(mm, (int, float)) and mm < 0):
        return []
    if any(rp in text_lower for rp in _B13_RELATIVE_FRAMING):
        return []
    for phrase in _B13_FORBIDDEN_ADVANTAGE:
        if phrase in text_lower:
            return [ClaimRecord(
                claim_type="negative_score_advantage_claim",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.FACTUAL_CONTRADICTION,
                matched_phrase=phrase,
                matched_seed=None,
                source="unsupported_phrase",
            )]
    return []


# ---------------------------------------------------------------------------
# B2.1b — Deliberate-choice framing in forced-move context
# ---------------------------------------------------------------------------
#
# "drives the decision", "chosen for its", etc. imply voluntary selection
# but the context is a forced move — these phrases are contradictory.
# B2.1a (first-sentence acknowledgment check) was removed: it produced
# false positives when single-candidate seed lists were used in other tests.

_FORCED_MOVE_INDICATOR: str = "only legal move"

_DELIBERATE_CHOICE_PHRASES: tuple = (
    "drives the decision", "chosen for its", "was chosen for",
    "selected for its", "was preferred because",
)


def _check_forced_move_framing(
    text_lower: str,
    seeds: Optional[List[str]],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    seeds_text = " ".join(s.lower() for s in (seeds or []))
    if _FORCED_MOVE_INDICATOR not in seeds_text:
        return []

    # B2.1b: deliberate-choice phrases forbidden — report first match only
    for phrase in _DELIBERATE_CHOICE_PHRASES:
        if phrase in text_lower:
            return [ClaimRecord(
                claim_type="forced_move_deliberate_framing",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.FACTUAL_CONTRADICTION,
                matched_phrase=phrase,
                matched_seed=None,
                source="deliberate_framing",
            )]
    return []


# ---------------------------------------------------------------------------
# B2.3 — Geometric impossibility
# ---------------------------------------------------------------------------
#
# A legal move always moves a piece.  These phrases are geometrically false.

_GEOMETRIC_IMPOSSIBILITY_PHRASES: tuple = (
    "piece remains stationary",
    "no piece movement occurred",
    "piece did not move",
)


def _check_geometric_impossibility(text_lower: str) -> List[ClaimRecord]:
    if not text_lower:
        return []
    for phrase in _GEOMETRIC_IMPOSSIBILITY_PHRASES:
        if phrase in text_lower:
            return [ClaimRecord(
                claim_type="geometric_impossibility",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.FACTUAL_CONTRADICTION,
                matched_phrase=phrase,
                matched_seed=None,
                source="geometric_impossibility",
            )]
    return []


# ---------------------------------------------------------------------------
# B2.5 — Our-mobility directional consistency
# ---------------------------------------------------------------------------
#
# When our_mobility_after <= our_mobility_before, claiming an increase is false.

_OUR_MOBILITY_INCREASE_PHRASES: tuple = (
    "increases our mobility",
    "improves our mobility",
    "our mobility increases",
    "our mobility improves",
    "our mobility grows",
    "expands our mobility",
)


def _check_our_mobility_direction(
    text_lower: str,
    facts: Dict[str, Any],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    our_mb = facts.get("our_mobility_before")
    our_ma = facts.get("our_mobility_after")
    if not (isinstance(our_mb, (int, float)) and isinstance(our_ma, (int, float))):
        return []
    if our_ma > our_mb:
        return []
    for phrase in _OUR_MOBILITY_INCREASE_PHRASES:
        if phrase in text_lower:
            return [ClaimRecord(
                claim_type="our_mobility_direction",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.FACTUAL_CONTRADICTION,
                matched_phrase=phrase,
                matched_seed=None,
                source="mobility_direction_mismatch",
            )]
    return []


# ---------------------------------------------------------------------------
# B2.6 — Tactical move defensive framing
# ---------------------------------------------------------------------------
#
# When creates_immediate_threat=True, "no pressure" framing is contradictory.

_TACTICAL_DEFENSIVE_PHRASES: tuple = (
    "no tactical pressure",
    "applies no pressure",
    "creates no pressure",
    "no immediate pressure",
)


def _check_tactical_move_framing(
    text_lower: str,
    facts: Dict[str, Any],
) -> List[ClaimRecord]:
    if not text_lower:
        return []
    if facts.get("creates_immediate_threat") is not True:
        return []
    for phrase in _TACTICAL_DEFENSIVE_PHRASES:
        if phrase in text_lower:
            return [ClaimRecord(
                claim_type="tactical_move_defensive_framing",
                claim_status=ClaimStatus.CONTRADICTED,
                claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
                seed_risk_type=None,
                hallucination_type=HallucinationType.FACTUAL_CONTRADICTION,
                matched_phrase=phrase,
                matched_seed=None,
                source="tactical_framing_mismatch",
            )]
    return []


# ---------------------------------------------------------------------------
# E.1 — Unified entry point
# ---------------------------------------------------------------------------

def verify_all(
    reasoning_text: str,
    *,
    reasoning_seeds: Optional[List[str]] = None,
    facts: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> List[ClaimRecord]:
    """
    Single-call verifier used by both the runtime refinement loop and the
    evaluation metric layer.

    Combines:
      1. extract_claims + verify_claims (legacy phrase-table verifier with
         the new clause-level negation pre-pass).
      2. _check_numeric_claims (E.3)
      3. _check_schema_leaks  (E.4)

    Returns
    -------
    list[ClaimRecord]
        All records merged, in stable order: legacy claims first
        (insertion order from the phrase table), then numeric findings,
        then schema-leak findings.  Empty list when the text is empty.

    Guarantees
    ----------
    • Deterministic.
    • No LLM.
    • Never mutates inputs.
    """
    if not isinstance(reasoning_text, str) or not reasoning_text:
        return []
    seeds_in = [s for s in (reasoning_seeds or []) if isinstance(s, str)]
    fact_dict = dict(facts) if isinstance(facts, dict) else {}

    raw_claims = extract_claims(reasoning_text, reasoning_seeds=seeds_in, facts=fact_dict)
    legacy = verify_claims(raw_claims, fact_dict, context=context)
    legacy = _upgrade_strategic_vague(legacy, bool(seeds_in))

    text_lower = reasoning_text.lower()
    numeric    = _check_numeric_claims(text_lower, fact_dict, seeds_in)
    schema     = _check_schema_leaks(reasoning_text, fact_dict)
    forbidden  = _check_forbidden_vocab(reasoning_text, text_lower, seeds_in)
    mob_red    = _check_mobility_reduction_claim(text_lower, fact_dict)
    safe_reply = _check_safe_reply_count(text_lower, fact_dict)
    absence    = _check_absence_claims(text_lower, seeds_in)
    opp_jump   = _check_opponent_jump_count(text_lower, fact_dict)
    single_leg = _check_single_legal_move_superlatives(text_lower, seeds_in)
    ctr_board  = _check_center_of_board_strategic(text_lower, fact_dict, seeds_in)
    mob_over   = _check_mobility_disadvantage_overclaim(text_lower, fact_dict)
    mob_dec    = _check_mobility_decrease_omission(text_lower, seeds_in)
    any_iso    = _check_any_piece_isolated_vulnerability(text_lower, fact_dict)
    narrow_gap = _check_narrowing_gap_direction(text_lower, fact_dict)
    mob_dir    = _check_mobility_direction_phrases(reasoning_text, fact_dict)
    comp_recap  = _check_comparative_recapture_fabrication(text_lower, fact_dict)
    rev_recap   = _check_reverse_recapture_fabrication(text_lower, fact_dict)
    false_force = _check_false_forced_opp_reply(text_lower, fact_dict)
    tradeoff_ng = _check_outweighs_numeric_grounding(text_lower)
    neg_score   = _check_negative_score_advantage(text_lower, fact_dict)
    forced_mv   = _check_forced_move_framing(text_lower, seeds_in)
    geo_imp     = _check_geometric_impossibility(text_lower)
    our_mob_dir = _check_our_mobility_direction(text_lower, fact_dict)
    tact_frame  = _check_tactical_move_framing(text_lower, fact_dict)

    return (
        legacy + numeric + schema + forbidden
        + mob_red + safe_reply + absence + opp_jump
        + single_leg + ctr_board + mob_over
        + mob_dec + any_iso + narrow_gap + mob_dir
        + comp_recap + rev_recap + false_force + tradeoff_ng + neg_score
        + forced_mv + geo_imp + our_mob_dir + tact_frame
    )


def contradictions_only(
    reasoning_text: str,
    *,
    reasoning_seeds: Optional[List[str]] = None,
    facts: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> List[ClaimRecord]:
    """Helper: subset of verify_all where claim_status == CONTRADICTED."""
    return [
        r for r in verify_all(
            reasoning_text,
            reasoning_seeds=reasoning_seeds,
            facts=facts,
            context=context,
        )
        if r.claim_status == ClaimStatus.CONTRADICTED
    ]


def contradiction_strings(
    reasoning_text: str,
    *,
    reasoning_seeds: Optional[List[str]] = None,
    facts: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Convert CONTRADICTED records into human-readable warning strings,
    in the same `REASONING_CONTRADICTION: …` format the legacy
    refinement loop already consumes.  Used by the runtime adapter so the
    refinement prompt template does not need to change.
    """
    out: List[str] = []
    for r in contradictions_only(
        reasoning_text,
        reasoning_seeds=reasoning_seeds,
        facts=facts,
        context=context,
    ):
        ct = r.claim_type
        phrase = r.matched_phrase or ""
        ht = r.hallucination_type.value if r.hallucination_type else "factual_contradiction"
        if ct == "numeric_mobility_reduction":
            out.append(
                f"REASONING_CONTRADICTION: claims mobility reduction but "
                f"opponent_mobility_after >= opponent_mobility_before "
                f"(phrase='{phrase}')"
            )
        elif ct == "numeric_safe_reply_count":
            out.append(
                f"REASONING_CONTRADICTION: specific safe-reply count not "
                f"matched by opponent_safe_reply_count (phrase='{phrase}')"
            )
        elif ct == "numeric_opponent_jump_count":
            fact_dict_local = dict(facts) if isinstance(facts, dict) else {}
            opp_jc = fact_dict_local.get("opponent_jump_count")
            out.append(
                f"REASONING_CONTRADICTION: claims single opponent jump but "
                f"opponent_jump_count={opp_jc} (factual_contradiction)"
            )
        elif ct == "single_legal_move_superlative":
            out.append(
                f"REASONING_CONTRADICTION: '{phrase}' used but this is the only "
                f"legal move (single_legal_move_context)"
            )
        elif ct == "center_of_board_strategic":
            out.append(
                "REASONING_CONTRADICTION: 'center of the board' used as strategic "
                "claim but center_control=false and phrase not in seeds "
                "(factual_contradiction)"
            )
        elif ct == "mobility_disadvantage_overclaim":
            fact_dict_local = dict(facts) if isinstance(facts, dict) else {}
            opp_after = fact_dict_local.get("opponent_mobility_after")
            our_after = fact_dict_local.get("our_mobility_after")
            out.append(
                f"REASONING_CONTRADICTION: '{phrase}' overclaims mobility "
                f"resolution but opponent_mobility_after={opp_after} > "
                f"our_mobility_after={our_after} (mobility disadvantage persists)"
            )
        elif ct == "mobility_decrease_omission":
            out.append(
                "REASONING_CONTRADICTION: our-mobility decrease seeded but "
                "omitted from reasoning (negative_fact_omission)"
            )
        elif ct == "any_piece_isolated_vulnerability":
            out.append(
                f"REASONING_CONTRADICTION: 'any_piece_isolated=true' contradicts "
                f"'{phrase}' claim (factual_contradiction)"
            )
        elif ct == "narrowing_gap_direction":
            fact_dict_local = dict(facts) if isinstance(facts, dict) else {}
            our_af = fact_dict_local.get("our_mobility_after")
            opp_af = fact_dict_local.get("opponent_mobility_after")
            out.append(
                f"REASONING_CONTRADICTION: 'narrowing the gap' is wrong when "
                f"our_mobility_after={our_af} >= "
                f"opponent_mobility_after={opp_af} "
                f"(gap was matched or reversed, not narrowed)"
            )
        elif ct == "mobility_unchanged_misclaim":
            fl = dict(facts) if isinstance(facts, dict) else {}
            out.append(
                f"REASONING_CONTRADICTION: claims mobility unchanged but "
                f"our_mobility={fl.get('our_mobility_before')}"
                f"->{fl.get('our_mobility_after')} and "
                f"opponent_mobility={fl.get('opponent_mobility_before')}"
                f"->{fl.get('opponent_mobility_after')} "
                f"(mobility_unchanged_misclaim)"
            )
        elif ct == "gap_did_not_narrow":
            fl = dict(facts) if isinstance(facts, dict) else {}
            ub = fl.get('our_mobility_before'); ua = fl.get('our_mobility_after')
            ob = fl.get('opponent_mobility_before'); oa = fl.get('opponent_mobility_after')
            try:
                gb = abs(int(ub) - int(ob)); ga = abs(int(ua) - int(oa))
            except Exception:
                gb = ga = None
            out.append(
                f"REASONING_CONTRADICTION: claims 'gap narrowed' but "
                f"|gap_before|={gb} and |gap_after|={ga} "
                f"(gap_did_not_narrow)"
            )
        elif ct == "gap_did_not_widen":
            fl = dict(facts) if isinstance(facts, dict) else {}
            ub = fl.get('our_mobility_before'); ua = fl.get('our_mobility_after')
            ob = fl.get('opponent_mobility_before'); oa = fl.get('opponent_mobility_after')
            try:
                gb = abs(int(ub) - int(ob)); ga = abs(int(ua) - int(oa))
            except Exception:
                gb = ga = None
            out.append(
                f"REASONING_CONTRADICTION: claims 'gap widened' but "
                f"|gap_before|={gb} and |gap_after|={ga} "
                f"(gap_did_not_widen)"
            )
        elif ct == "comparative_recapture_fabrication":
            out.append(
                f"COMPARATIVE_CONTRADICTION: '{phrase}' claimed but "
                f"opponent_can_recapture=true (fabricated_claim)"
            )
        elif ct == "tradeoff_without_numeric_grounding":
            out.append(
                f"COMPARATIVE_CONTRADICTION: '{phrase}' used without numeric "
                f"grounding in same sentence (tradeoff_without_evidence)"
            )
        elif ct == "negative_score_advantage_claim":
            fact_dict_local = dict(facts) if isinstance(facts, dict) else {}
            mm_val = fact_dict_local.get("minimax_score")
            mm_str = f"{float(mm_val):.1f}" if isinstance(mm_val, (int, float)) else str(mm_val)
            out.append(
                f"COMPARATIVE_CONTRADICTION: '{phrase}' used when "
                f"minimax_score={mm_str} < 0 without relative "
                f"framing (misleading_advantage_claim)"
            )
        elif ct == "forced_move_deliberate_framing":
            out.append(
                f"REASONING_CONTRADICTION: '{phrase}' deliberate-choice framing "
                f"in forced-move context (forced_move_deliberate_framing)"
            )
        elif ct == "geometric_impossibility":
            out.append(
                f"REASONING_CONTRADICTION: '{phrase}' is geometrically impossible "
                f"for a legal move (geometric_impossibility)"
            )
        elif ct == "our_mobility_direction":
            fact_dict_local = dict(facts) if isinstance(facts, dict) else {}
            our_mb = fact_dict_local.get("our_mobility_before")
            our_ma = fact_dict_local.get("our_mobility_after")
            out.append(
                f"REASONING_CONTRADICTION: claims our-mobility increase but "
                f"our_mobility_after={int(our_ma)} <= "
                f"our_mobility_before={int(our_mb)} (our_mobility_direction)"
            )
        elif ct == "tactical_move_defensive_framing":
            out.append(
                f"REASONING_CONTRADICTION: '{phrase}' framing contradicts "
                f"creates_immediate_threat=true (tactical_move_defensive_framing)"
            )
        elif ct.startswith("numeric_"):
            field = ct[len("numeric_"):]
            out.append(
                f"REASONING_CONTRADICTION: numeric mismatch on {field} — "
                f"reasoning says '{phrase}' but fact disagrees ({ht})"
            )
        elif ct.startswith("schema_leak_"):
            field = ct[len("schema_leak_"):]
            out.append(
                f"REASONING_CONTRADICTION: schema-leak on {field} — "
                f"reasoning contains raw assertion '{phrase}' contradicting fact ({ht})"
            )
        else:
            out.append(
                f"REASONING_CONTRADICTION: {ct} contradicted by facts "
                f"(phrase='{phrase}', {ht})"
            )
    return out


# ---------------------------------------------------------------------------
# E.1 invariant — runtime ↔ evaluator agreement assertion
# ---------------------------------------------------------------------------

class RuntimeEvaluatorDisagreement(AssertionError):
    """Raised when refinement-loop output disagrees with verify_all."""


def assert_runtime_evaluator_agreement(
    runtime_contradictions: List[str],
    reasoning_text: str,
    *,
    reasoning_seeds: Optional[List[str]] = None,
    facts: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Hard invariant for tests / CI:
      runtime_contradictions == []  ⇔  verify_all returns no CONTRADICTED
      claim against the same reasoning text.

    Symmetric: if either side flags zero contradictions while the other
    side flags one or more, raise RuntimeEvaluatorDisagreement.
    """
    rt_clean = (len(runtime_contradictions) == 0)
    ev_clean = (len(contradictions_only(
        reasoning_text,
        reasoning_seeds=reasoning_seeds,
        facts=facts,
        context=context,
    )) == 0)
    if rt_clean != ev_clean:
        raise RuntimeEvaluatorDisagreement(
            f"runtime checker reports {len(runtime_contradictions)} contradiction(s) "
            f"while evaluator verify_all reports "
            f"{0 if ev_clean else 'one or more'} contradiction(s) — "
            f"refinement loop and metric layer are out of sync.\n"
            f"  reasoning: {reasoning_text[:200]!r}"
        )
