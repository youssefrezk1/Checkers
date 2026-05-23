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
) -> List[ClaimRecord]:
    """
    Flag 'from N to M' mobility narratives whose (N, M) match no known
    before/after pair in the facts dict.  Always returns CONTRADICTED +
    FABRICATED_CLAIM when fired.
    """
    out: List[ClaimRecord] = []
    pairs = [
        ("opponent_mobility_before", "opponent_mobility_after"),
        ("our_mobility_before",      "our_mobility_after"),
    ]
    for m in re.finditer(
        r"from\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"\s+to\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)",
        text_lower,
    ):
        a = _to_int(m.group(1))
        b = _to_int(m.group(2))
        if a is None or b is None:
            continue
        # Match against any known transition pair.
        matched = False
        for before_field, after_field in pairs:
            fb = facts.get(before_field)
            fa = facts.get(after_field)
            if isinstance(fb, (int, float)) and isinstance(fa, (int, float)):
                if int(fb) == a and int(fa) == b:
                    matched = True
                    break
        if matched:
            continue
        # No pair matches → fabricated transition.
        out.append(ClaimRecord(
            claim_type="numeric_mobility_transition",
            claim_status=ClaimStatus.CONTRADICTED,
            claim_verifiability=ClaimVerifiability.FULLY_VERIFIABLE,
            seed_risk_type=None,
            hallucination_type=HallucinationType.FABRICATED_CLAIM,
            matched_phrase=m.group(0),
            matched_seed=None,
            source="unsupported_phrase",
        ))
    return out


def _check_numeric_claims(
    text_lower: str,
    facts: Dict[str, Any],
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
    records.extend(_check_mobility_transition(text_lower, facts))

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
            records.append(ClaimRecord(
                claim_type=f"schema_leak_{field}",
                claim_status=ClaimStatus.UNSUPPORTED,
                claim_verifiability=ClaimVerifiability.PARTIALLY_VERIFIABLE,
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
            # Negation-aware: the runtime checker skips "no <phrase>" — mirror that.
            if ("no " + p_lower) in text_lower:
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

    text_lower = reasoning_text.lower()
    numeric  = _check_numeric_claims(text_lower, fact_dict)
    schema   = _check_schema_leaks(reasoning_text, fact_dict)
    forbidden = _check_forbidden_vocab(reasoning_text, text_lower, seeds_in)

    return legacy + numeric + schema + forbidden


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
        if ct.startswith("numeric_"):
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
