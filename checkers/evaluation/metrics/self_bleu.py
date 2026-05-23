# checkers/evaluation/metrics/self_bleu.py
#
# Diversity metric — Self-BLEU at n = 2, 3, 4.
#
# Self-BLEU treats each hypothesis as the candidate and ALL OTHER
# hypotheses as the reference set, then averages over the corpus.
# LOWER Self-BLEU  ⇒  MORE diverse output (less template reuse).
# HIGHER Self-BLEU ⇒  template collapse / paraphrase reuse.
#
# Implementation
# --------------
# - Tokenisation: lowercase, whitespace + punctuation split.
# - Modified n-gram precision with clipping (Papineni et al. 2002).
# - Smoothing: add-one in numerator and denominator (Chen & Cherry 2014,
#   method 1) to avoid zero precisions on short hypotheses.
# - Brevity penalty over the closest reference length.
#
# Deterministic, no LLM calls, no external dependencies.
# (We deliberately do NOT use nltk so the metric is reproducible and
# free of nltk's tokenisation / smoothing-method drift.)

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class SelfBleuSummary:
    n_hypotheses: int
    bleu_2:       Optional[float]   # None when corpus has fewer than 2 hyps
    bleu_3:       Optional[float]
    bleu_4:       Optional[float]
    mean_token_length: Optional[float] = None
    # Per-hypothesis lists (same order as the input). Useful for histograms
    # / template-collapse heatmaps without re-running the metric.
    per_hyp_bleu_2: Optional[List[float]] = None
    per_hyp_bleu_3: Optional[List[float]] = None
    per_hyp_bleu_4: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# BLEU primitives
# ---------------------------------------------------------------------------

def _ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _modified_precision(
    hyp_tokens: Sequence[str],
    refs_tokens: Sequence[Sequence[str]],
    n: int,
) -> tuple:
    """Returns (numerator_with_clip, denominator). Both ints."""
    hyp_ngrams = Counter(_ngrams(hyp_tokens, n))
    if not hyp_ngrams:
        return 0, 0

    max_ref_counts: Counter = Counter()
    for ref in refs_tokens:
        ref_counts = Counter(_ngrams(ref, n))
        for ng, c in ref_counts.items():
            if c > max_ref_counts[ng]:
                max_ref_counts[ng] = c

    clipped = sum(min(c, max_ref_counts[ng]) for ng, c in hyp_ngrams.items())
    total   = sum(hyp_ngrams.values())
    return clipped, total


def _brevity_penalty(hyp_len: int, ref_lens: Sequence[int]) -> float:
    if hyp_len <= 0 or not ref_lens:
        return 0.0
    # closest reference length (ties → shorter, standard BLEU rule)
    r = min(ref_lens, key=lambda rl: (abs(rl - hyp_len), rl))
    if hyp_len > r:
        return 1.0
    if hyp_len == 0:
        return 0.0
    return math.exp(1.0 - r / hyp_len)


def _bleu_n(
    hyp_tokens: Sequence[str],
    refs_tokens: Sequence[Sequence[str]],
    max_n: int,
) -> float:
    """
    Single-hypothesis BLEU at order max_n with add-one smoothing on every
    n-gram precision (Chen & Cherry 2014 method 1).
    """
    if max_n < 1:
        return 0.0
    if not refs_tokens:
        return 0.0

    log_precisions: List[float] = []
    for n in range(1, max_n + 1):
        clipped, total = _modified_precision(hyp_tokens, refs_tokens, n)
        # Add-one smoothing to avoid log(0) and to be order-invariant.
        smoothed = (clipped + 1.0) / (total + 1.0)
        if smoothed <= 0.0:
            return 0.0
        log_precisions.append(math.log(smoothed))

    geo_mean = math.exp(sum(log_precisions) / max_n)
    bp = _brevity_penalty(len(hyp_tokens), [len(r) for r in refs_tokens])
    return bp * geo_mean


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_self_bleu(
    reasonings: Iterable[str],
    *,
    keep_per_hyp: bool = True,
) -> SelfBleuSummary:
    """
    Compute Self-BLEU 2/3/4 over a corpus of reasoning strings.

    Parameters
    ----------
    reasonings : iterable of str
        Each non-empty entry becomes one hypothesis. Empty strings, None,
        and non-strings are skipped silently.
    keep_per_hyp : bool, default True
        When True, the summary includes per-hypothesis BLEU lists (useful
        for finding cluster centroids / template-collapse hotspots).

    Returns
    -------
    SelfBleuSummary
        bleu_n is None when fewer than 2 valid hypotheses are available
        (Self-BLEU requires at least one reference).
    """
    tokenised: List[List[str]] = []
    for r in reasonings:
        if not isinstance(r, str):
            continue
        toks = _tokenize(r)
        if toks:
            tokenised.append(toks)

    n = len(tokenised)
    if n < 2:
        return SelfBleuSummary(
            n_hypotheses=n,
            bleu_2=None, bleu_3=None, bleu_4=None,
            mean_token_length=(sum(len(t) for t in tokenised) / n) if n else None,
            per_hyp_bleu_2=None, per_hyp_bleu_3=None, per_hyp_bleu_4=None,
        )

    per_bleu2: List[float] = []
    per_bleu3: List[float] = []
    per_bleu4: List[float] = []

    for i, hyp in enumerate(tokenised):
        refs = [tokenised[j] for j in range(n) if j != i]
        per_bleu2.append(_bleu_n(hyp, refs, 2))
        per_bleu3.append(_bleu_n(hyp, refs, 3))
        per_bleu4.append(_bleu_n(hyp, refs, 4))

    mean_len = sum(len(t) for t in tokenised) / n

    return SelfBleuSummary(
        n_hypotheses=n,
        bleu_2=sum(per_bleu2) / n,
        bleu_3=sum(per_bleu3) / n,
        bleu_4=sum(per_bleu4) / n,
        mean_token_length=mean_len,
        per_hyp_bleu_2=per_bleu2 if keep_per_hyp else None,
        per_hyp_bleu_3=per_bleu3 if keep_per_hyp else None,
        per_hyp_bleu_4=per_bleu4 if keep_per_hyp else None,
    )
