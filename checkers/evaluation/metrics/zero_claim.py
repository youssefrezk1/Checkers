# checkers/evaluation/metrics/zero_claim.py
#
# Grounding metric — flags sentences whose verifiable-claim coverage is
# empty (or trivially weak). High uncovered_sentence_fraction indicates
# the LLM is padding the explanation with content that the deterministic
# claim verifier can neither support nor refute.
#
# Metrics (per turn, then aggregated):
#   - uncovered_sentence_count    — sentences yielding 0 claims
#   - uncovered_sentence_fraction — uncovered / total sentences
#   - claim_density               — total claims / total sentences
#   - filler_sentence_rate        — sentences with 0 SUPPORTED claims
#                                   (broader than uncovered: includes
#                                   sentences whose only claims are
#                                   UNSUPPORTED / VAGUE / CONTRADICTED)
#
# Definitions
#   - sentence  = punctuation-delimited span, see _SENT_SPLIT_RE.
#                 Empty / whitespace-only spans are dropped.
#                 Spans shorter than 3 characters are treated as
#                 punctuation noise and dropped.
#
# Determinism: identical inputs always produce identical outputs.
# No LLM calls.  No external state.  No imports from runtime pipeline.

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

# Use the unified verifier so per-sentence claim counts match the runtime
# refinement loop and every other metric module.  Bypassing it would skip
# the numeric (E.3), schema-leak (E.4), forbidden-vocab, mobility-reduction,
# and safe-reply contradictions that the unified verifier owns.
from checkers.evaluation.unified_verifier import verify_all
from checkers.evaluation.reasoning_taxonomy import ClaimStatus


# Sentence splitter.  Conservative: splits on '.', '!', '?' followed by
# whitespace OR end of string.  Does not attempt to handle ellipses, decimal
# numbers, or abbreviations specially — the ranker reasoning text contains
# very few of these (coordinates are written as "[r,c]" without periods).
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_MIN_SENTENCE_CHARS = 3


# ---------------------------------------------------------------------------
# Per-sentence and per-turn dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SentenceCoverage:
    """Claim coverage outcome for a single sentence."""
    text:               str
    claim_count:        int
    supported_count:    int
    contradicted_count: int
    unsupported_count:  int
    vague_count:        int
    is_uncovered:       bool   # True when claim_count == 0
    is_filler:          bool   # True when supported_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ZeroClaimTurn:
    """Per-turn aggregate built from SentenceCoverage objects."""
    turn_id:                     str
    total_sentences:             int
    total_claims:                int
    uncovered_sentence_count:    int
    filler_sentence_count:       int
    uncovered_sentence_fraction: Optional[float]   # None when no sentences
    filler_sentence_rate:        Optional[float]
    claim_density:               Optional[float]   # claims / sentences
    sentences:                   List[SentenceCoverage] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sentences"] = [s.to_dict() if hasattr(s, "to_dict") else s for s in self.sentences]
        return d


@dataclass
class ZeroClaimSummary:
    """Corpus-level aggregate over a batch of ZeroClaimTurn records."""
    n_turns:                     int = 0
    total_sentences:             int = 0
    total_claims:                int = 0
    total_uncovered_sentences:   int = 0
    total_filler_sentences:      int = 0
    # micro_* weight every sentence equally across the corpus
    uncovered_sentence_fraction_micro: Optional[float] = None
    filler_sentence_rate_micro:        Optional[float] = None
    claim_density_micro:               Optional[float] = None
    # macro_* weight every turn (with ≥1 sentence) equally
    uncovered_sentence_fraction_macro: Optional[float] = None
    filler_sentence_rate_macro:        Optional[float] = None
    claim_density_macro:               Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if len(p.strip()) >= _MIN_SENTENCE_CHARS]


def _safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


# ---------------------------------------------------------------------------
# Public API — per turn
# ---------------------------------------------------------------------------

def evaluate_zero_claim(
    record: Dict[str, Any],
    turn_id: Optional[str] = None,
) -> ZeroClaimTurn:
    """
    Compute sentence-level claim coverage for a single record.

    Reads:
      - last_move_reasoning             (the explanation text)
      - ranker_diagnostics.reasoning_seeds
      - chosen_move_facts
      - ranker_diagnostics.next_best_minimax_score (verifier context)
    """
    tid = (
        turn_id
        if isinstance(turn_id, str) and turn_id
        else (record.get("turn_id") if isinstance(record.get("turn_id"), str) else "unknown")
    )

    reasoning = record.get("last_move_reasoning") or ""
    if not isinstance(reasoning, str):
        reasoning = ""

    diag    = record.get("ranker_diagnostics") or {}
    facts   = record.get("chosen_move_facts") or {}
    seeds   = [s for s in (diag.get("reasoning_seeds") or []) if isinstance(s, str)]
    nb      = diag.get("next_best_minimax_score")
    ctx: Dict[str, Any] = {}
    if isinstance(nb, (int, float)):
        ctx["next_best_minimax_score"] = nb

    sentences = _split_sentences(reasoning)
    per_sent: List[SentenceCoverage] = []
    total_claims = 0
    uncovered    = 0
    filler       = 0

    for s in sentences:
        # Single source of truth — must match the runtime refinement loop
        # and every other metric module.  See the import-block comment.
        claims = verify_all(
            s, reasoning_seeds=seeds, facts=facts, context=ctx or None,
        )
        sup = sum(1 for c in claims if c.claim_status == ClaimStatus.SUPPORTED)
        con = sum(1 for c in claims if c.claim_status == ClaimStatus.CONTRADICTED)
        uns = sum(1 for c in claims if c.claim_status == ClaimStatus.UNSUPPORTED)
        vag = sum(1 for c in claims if c.claim_status == ClaimStatus.VAGUE)
        total_claims += len(claims)
        is_uncovered = (len(claims) == 0)
        is_filler    = (sup == 0)
        if is_uncovered:
            uncovered += 1
        if is_filler:
            filler += 1
        per_sent.append(SentenceCoverage(
            text=s,
            claim_count=len(claims),
            supported_count=sup,
            contradicted_count=con,
            unsupported_count=uns,
            vague_count=vag,
            is_uncovered=is_uncovered,
            is_filler=is_filler,
        ))

    n_sent = len(sentences)
    return ZeroClaimTurn(
        turn_id=tid,
        total_sentences=n_sent,
        total_claims=total_claims,
        uncovered_sentence_count=uncovered,
        filler_sentence_count=filler,
        uncovered_sentence_fraction=_safe_div(uncovered, n_sent),
        filler_sentence_rate=_safe_div(filler, n_sent),
        claim_density=_safe_div(total_claims, n_sent),
        sentences=per_sent,
    )


# ---------------------------------------------------------------------------
# Public API — corpus
# ---------------------------------------------------------------------------

def aggregate_zero_claim(turns: List[ZeroClaimTurn]) -> ZeroClaimSummary:
    """
    Aggregate per-turn ZeroClaimTurn records.

    Rates are None when the corpus contains 0 sentences (or 0 turns with
    sentences for the macro variant).
    """
    s = ZeroClaimSummary(n_turns=len(turns))
    s.total_sentences           = sum(t.total_sentences for t in turns)
    s.total_claims              = sum(t.total_claims for t in turns)
    s.total_uncovered_sentences = sum(t.uncovered_sentence_count for t in turns)
    s.total_filler_sentences    = sum(t.filler_sentence_count for t in turns)

    s.uncovered_sentence_fraction_micro = _safe_div(
        s.total_uncovered_sentences, s.total_sentences
    )
    s.filler_sentence_rate_micro = _safe_div(
        s.total_filler_sentences, s.total_sentences
    )
    s.claim_density_micro = _safe_div(s.total_claims, s.total_sentences)

    macro_turns = [t for t in turns if t.total_sentences > 0]
    if macro_turns:
        s.uncovered_sentence_fraction_macro = sum(
            (t.uncovered_sentence_fraction or 0.0) for t in macro_turns
        ) / len(macro_turns)
        s.filler_sentence_rate_macro = sum(
            (t.filler_sentence_rate or 0.0) for t in macro_turns
        ) / len(macro_turns)
        s.claim_density_macro = sum(
            (t.claim_density or 0.0) for t in macro_turns
        ) / len(macro_turns)

    return s
