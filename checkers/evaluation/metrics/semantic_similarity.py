# checkers/evaluation/metrics/semantic_similarity.py
#
# Semantic-quality metric layer — Setup A (pre/post refinement within
# condition).  Computes BERTScore F1 (primary) and BLEURT (secondary,
# out-of-domain) on every turn where the refinement loop actually ran AND
# produced a different post text.
#
# Design rules (see also METHODOLOGY.md in this directory):
#
#   * Only turns where the refinement loop ran AND produced a different
#     post text contribute to aggregates.  Turns with pre == post are
#     counted in `n_turns_unchanged`; turns with missing pre/post in
#     `n_skipped_empty`; turns whose score returned None despite being
#     eligible in `n_scoring_failed`.  These three counters are disjoint.
#
#   * Both backends are lazy-loaded.  Importing this module does NOT
#     require `bert_score` or `bleurt-pytorch` to be installed.  Callers
#     without the libraries still get a valid (zeroed) SemanticSummary so
#     the rest of the metrics pipeline keeps running, plus a structured
#     `error` field in the per-metric report block.
#
#   * Scores are cached in-memory keyed by (model_name, sha256(a), sha256(b))
#     for each backend independently.
#
#   * Model names + library versions are pinned in `requirements.txt` and
#     surfaced in `semantic.model_versions` so a run is reproducible.
#
#   * Semantic scores never read from the factuality verifier; the only
#     coupling point is `_refinement_ran`, which gates *which* turns are
#     scored (not what score they receive).  See METHODOLOGY.md §4.

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


# ---------------------------------------------------------------------------
# Pinned configuration
# ---------------------------------------------------------------------------
# Keep these in sync with requirements.txt.

BERTSCORE_MODEL: str = "roberta-large"
BERTSCORE_LANG:  str = "en"

# BLEURT: distilled BLEURT-20 via the lucadiliello PyTorch port.  ~410 MB.
# Smaller variant (`Elron/bleurt-base-128`) is acceptable as a fallback if
# the distilled checkpoint is unavailable, but the port-specific tokenizer
# differs and reproducibility metadata would change.
BLEURT_CHECKPOINT: str = "lucadiliello/BLEURT-20-D12"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SemanticPairTurn:
    """Per-turn Setup-A record."""
    turn_id:        str
    refined:        bool          # True iff refinement ran AND text changed AND ≥1 score available
    skipped_empty:  bool          # True iff pre or post was empty/missing
    scoring_failed: bool = False  # True iff eligible but every backend returned None
    bertscore_f1:   Optional[float] = None
    bleurt:         Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _Stats:
    n:      int = 0
    mean:   Optional[float] = None
    median: Optional[float] = None
    p10:    Optional[float] = None
    iqr:    Optional[List[float]] = None   # [q25, q75]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticSummary:
    """Corpus-level Setup-A aggregate written into report.json."""
    n_turns_total:     int = 0
    n_turns_refined:   int = 0
    n_turns_unchanged: int = 0
    n_skipped_empty:   int = 0
    n_scoring_failed:  int = 0
    bertscore_f1:      _Stats = field(default_factory=_Stats)
    bleurt:            _Stats = field(default_factory=_Stats)
    model_versions:    Dict[str, str] = field(default_factory=dict)
    # When a backend is uninstalled, the corresponding error string is
    # recorded here.  The aggregate `_Stats` block for that backend stays
    # populated with `n=0, mean=None, ...` so report-schema consumers
    # don't have to special-case the absence.
    backend_errors:    Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bertscore_f1"] = self.bertscore_f1.to_dict()
        d["bleurt"]       = self.bleurt.to_dict()
        return d


# ---------------------------------------------------------------------------
# Lazy backend loaders + in-memory caches
# ---------------------------------------------------------------------------

class SemanticDependencyMissing(RuntimeError):
    """Raised lazily when a semantic backend is required but not installed."""


# BERTScore state.
_BS_SCORER = None
_BS_VERSION: Optional[str] = None
_BS_CACHE: Dict[Tuple[str, str, str], float] = {}

# BLEURT state.
_BL_BUNDLE = None  # tuple (tokenizer, model, torch_module)
_BL_VERSION: Optional[str] = None
_BL_CACHE: Dict[Tuple[str, str, str], float] = {}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_bert_score():
    """Lazy-import bert_score.  Returns (BERTScorer instance, lib_version)."""
    global _BS_SCORER, _BS_VERSION
    if _BS_SCORER is not None:
        return _BS_SCORER, _BS_VERSION
    try:
        import bert_score  # type: ignore
    except ImportError as exc:
        raise SemanticDependencyMissing(
            "bert_score is not installed.  Install it with "
            "`pip install bert_score==0.3.13` to enable BERTScore."
        ) from exc

    _BS_VERSION = getattr(bert_score, "__version__", "unknown")
    # Single scorer reused across calls; rescale_with_baseline=True subtracts
    # the empirical random-pair baseline.
    _BS_SCORER = bert_score.BERTScorer(
        model_type=BERTSCORE_MODEL,
        lang=BERTSCORE_LANG,
        rescale_with_baseline=True,
        idf=False,
    )
    return _BS_SCORER, _BS_VERSION


def _resolve_bleurt():
    """Lazy-import bleurt-pytorch.  Returns ((tok, model, torch), lib_version)."""
    global _BL_BUNDLE, _BL_VERSION
    if _BL_BUNDLE is not None:
        return _BL_BUNDLE, _BL_VERSION
    try:
        import bleurt_pytorch as bp  # type: ignore
        from bleurt_pytorch import (  # type: ignore
            BleurtForSequenceClassification,
        )
        # BLEURT-20 checkpoints (incl. the distilled D12 variant) require the
        # SentencePiece tokenizer, NOT the BERT-style `BleurtTokenizer`.
        # Picking the wrong one silently miscalibrates the score; the
        # transformers loader will emit a warning rather than fail.
        from bleurt_pytorch.bleurt.tokenization_bleurt_sp import (  # type: ignore
            BleurtSPTokenizer,
        )
        import torch  # type: ignore
    except ImportError as exc:
        raise SemanticDependencyMissing(
            "bleurt-pytorch is not installed.  Install it with "
            "`pip install bleurt-pytorch==0.0.1` to enable BLEURT."
        ) from exc

    _BL_VERSION = getattr(bp, "__version__", "unknown")
    tokenizer = BleurtSPTokenizer.from_pretrained(BLEURT_CHECKPOINT)
    model = BleurtForSequenceClassification.from_pretrained(BLEURT_CHECKPOINT)
    model.eval()
    _BL_BUNDLE = (tokenizer, model, torch)
    return _BL_BUNDLE, _BL_VERSION


def model_versions() -> Dict[str, str]:
    """Return pinned model + library version block for the report."""
    try:
        _, bs_ver = _resolve_bert_score()
    except SemanticDependencyMissing:
        bs_ver = "not_installed"
    try:
        _, bl_ver = _resolve_bleurt()
    except SemanticDependencyMissing:
        bl_ver = "not_installed"
    return {
        "bertscore_model": BERTSCORE_MODEL,
        "bertscore_lib":   f"bert_score=={bs_ver}",
        "bleurt_model":    BLEURT_CHECKPOINT,
        "bleurt_lib":      f"bleurt-pytorch=={bl_ver}",
    }


# ---------------------------------------------------------------------------
# Public scoring API
# ---------------------------------------------------------------------------

def _validate_text_pair(a: Any, b: Any) -> bool:
    return (
        isinstance(a, str) and isinstance(b, str)
        and bool(a.strip()) and bool(b.strip())
    )


def score_pair_bertscore(a: str, b: str) -> Optional[float]:
    """
    Return rescaled BERTScore F1 between `a` and `b`, or None when either
    input is empty / not a string.  Cached in-memory by content hash.

    Raises SemanticDependencyMissing on first call when `bert_score` is
    not installed.
    """
    if not _validate_text_pair(a, b):
        return None

    key = (BERTSCORE_MODEL, _hash(a), _hash(b))
    if key in _BS_CACHE:
        return _BS_CACHE[key]

    scorer, _ = _resolve_bert_score()
    _, _, f1 = scorer.score([a], [b])
    value = float(f1[0].item())
    if math.isnan(value):
        return None
    _BS_CACHE[key] = value
    return value


def score_pair_bleurt(a: str, b: str) -> Optional[float]:
    """
    Return BLEURT-20-D12 score for the reference `a` and candidate `b`,
    or None when either input is empty / not a string.  Cached in-memory.

    Raises SemanticDependencyMissing on first call when `bleurt-pytorch`
    is not installed.
    """
    if not _validate_text_pair(a, b):
        return None

    key = (BLEURT_CHECKPOINT, _hash(a), _hash(b))
    if key in _BL_CACHE:
        return _BL_CACHE[key]

    (tokenizer, model, torch), _ = _resolve_bleurt()
    with torch.no_grad():
        inputs = tokenizer(
            [a], [b],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        out = model(**inputs).logits
        # BleurtForSequenceClassification returns shape [1, 1] or [1]; both
        # collapse to a single scalar via .item().
        value = float(out.flatten()[0].item())
    if math.isnan(value):
        return None
    _BL_CACHE[key] = value
    return value


# ---------------------------------------------------------------------------
# Per-turn evaluation
# ---------------------------------------------------------------------------

def _post_text(record: Mapping[str, Any]) -> str:
    val = record.get("last_move_reasoning")
    return val if isinstance(val, str) else ""


def _pre_text(record: Mapping[str, Any]) -> Optional[str]:
    diag = record.get("explainer_diagnostics") or record.get("ranker_diagnostics") or {}
    val = diag.get("raw_llm_reasoning_pre_refinement")
    if isinstance(val, str) and val.strip():
        return val
    return None


def _refinement_ran(record: Mapping[str, Any]) -> bool:
    diag = record.get("explainer_diagnostics") or record.get("ranker_diagnostics") or {}
    attempts = diag.get("reasoning_refinement_retry_count")
    if isinstance(attempts, (int, float)) and attempts > 0:
        return True
    return bool(diag.get("reasoning_contradiction_detected"))


def evaluate_semantic(
    record: Mapping[str, Any],
    *,
    turn_id: Optional[str] = None,
    bertscore_fn=None,
    bleurt_fn=None,
) -> SemanticPairTurn:
    """
    Per-turn Setup-A evaluation.  Computes BERTScore F1 (primary) and
    BLEURT (secondary).  Either may be None if its backend is unavailable
    or returns NaN; aggregates track this in `n_scoring_failed`.

    Parameters
    ----------
    record : Mapping[str, Any]
        One line from logs/evaluation_source/<run_tag>/<game>.jsonl.
    turn_id : str | None
        Override the turn id; default uses record["turn_id"] or "unknown".
    bertscore_fn, bleurt_fn : callable | None
        Override the scoring backends (used by tests to stub out
        transformers).  Signature: (a: str, b: str) -> Optional[float].
    """
    tid = (
        turn_id
        if isinstance(turn_id, str) and turn_id
        else (record.get("turn_id") if isinstance(record.get("turn_id"), str) else "unknown")
    )
    bert_call   = bertscore_fn if bertscore_fn is not None else score_pair_bertscore
    bleurt_call = bleurt_fn    if bleurt_fn    is not None else score_pair_bleurt

    pre  = _pre_text(record)
    post = _post_text(record)

    if pre is None or not post:
        return SemanticPairTurn(turn_id=tid, refined=False, skipped_empty=True)

    if not _refinement_ran(record):
        return SemanticPairTurn(turn_id=tid, refined=False, skipped_empty=False)

    if pre == post:
        return SemanticPairTurn(turn_id=tid, refined=False, skipped_empty=False)

    # Eligible: try both backends independently.  Backend dependencies
    # raise SemanticDependencyMissing; we catch here so a missing BLEURT
    # never blocks BERTScore (and vice-versa).
    try:
        bs = bert_call(pre, post)
    except SemanticDependencyMissing:
        bs = None
    try:
        bl = bleurt_call(pre, post)
    except SemanticDependencyMissing:
        bl = None

    if bs is None and bl is None:
        return SemanticPairTurn(
            turn_id=tid, refined=False, skipped_empty=False, scoring_failed=True,
        )

    return SemanticPairTurn(
        turn_id=tid, refined=True, skipped_empty=False, scoring_failed=False,
        bertscore_f1=bs, bleurt=bl,
    )


# ---------------------------------------------------------------------------
# Corpus aggregator
# ---------------------------------------------------------------------------

def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _stats(values: List[float]) -> _Stats:
    if not values:
        return _Stats()
    return _Stats(
        n      = len(values),
        mean   = sum(values) / len(values),
        median = statistics.median(values),
        p10    = _percentile(values, 0.10),
        iqr    = [
            _percentile(values, 0.25) or 0.0,
            _percentile(values, 0.75) or 0.0,
        ],
    )


def aggregate_semantic(turns: List[SemanticPairTurn]) -> SemanticSummary:
    """
    Build the report-ready summary.

    Aggregates use only turns with `refined=True` AND a non-None metric
    value.  Each backend is aggregated independently because one may be
    available while the other is not.
    """
    summary = SemanticSummary(model_versions=model_versions())
    summary.n_turns_total    = len(turns)
    summary.n_turns_refined  = sum(1 for t in turns if t.refined)
    summary.n_skipped_empty  = sum(1 for t in turns if t.skipped_empty)
    summary.n_scoring_failed = sum(1 for t in turns if t.scoring_failed)
    summary.n_turns_unchanged = (
        summary.n_turns_total
        - summary.n_turns_refined
        - summary.n_skipped_empty
        - summary.n_scoring_failed
    )

    bs_vals = [t.bertscore_f1 for t in turns if t.refined and t.bertscore_f1 is not None]
    bl_vals = [t.bleurt       for t in turns if t.refined and t.bleurt       is not None]
    summary.bertscore_f1 = _stats(bs_vals)
    summary.bleurt       = _stats(bl_vals)

    # Record backend errors so consumers can distinguish "no eligible
    # turns" (n=0, no error) from "backend missing" (n=0, error present).
    if summary.bertscore_f1.n == 0 and summary.n_turns_refined > 0:
        try:
            _resolve_bert_score()
        except SemanticDependencyMissing as e:
            summary.backend_errors["bertscore"] = str(e)
    if summary.bleurt.n == 0 and summary.n_turns_refined > 0:
        try:
            _resolve_bleurt()
        except SemanticDependencyMissing as e:
            summary.backend_errors["bleurt"] = str(e)

    return summary


# ---------------------------------------------------------------------------
# Convenience: evaluate an iterable of records
# ---------------------------------------------------------------------------

def evaluate_records(
    records: Iterable[Mapping[str, Any]],
    *,
    bertscore_fn=None,
    bleurt_fn=None,
) -> Tuple[List[SemanticPairTurn], SemanticSummary]:
    """Evaluate every record and return (per-turn list, corpus summary)."""
    turns = [
        evaluate_semantic(r, bertscore_fn=bertscore_fn, bleurt_fn=bleurt_fn)
        for r in records
    ]
    return turns, aggregate_semantic(turns)
