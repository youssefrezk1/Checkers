# checkers/evaluation/eval_logger.py
#
# Minimal JSONL evaluation logger.
#
# PURPOSE
# -------
# Serialises TurnEvaluationRecord objects to a JSONL file (one JSON object
# per line) and reads them back as plain dicts for offline analysis.
# Also provides a lightweight summary aggregator for thesis experiments.
#
# DESIGN CONSTRAINTS
# ------------------
# - No imports from the runtime pipeline (engine, agents, graph, state).
# - No LLM calls.  Pure I/O and arithmetic only.
# - Deterministic: same records → same serialised output always.
# - Append-mode writes: existing records are never overwritten.
# - Inputs are never mutated.
# - All outputs are plain Python dicts (JSON-serialisable).
#
# USAGE
# -----
# This module is imported only by evaluation scripts and tests.
# It must NEVER be imported by the runtime pipeline.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from checkers.evaluation.turn_evaluator import TurnEvaluationRecord


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _serialise_claim(claim: Any) -> Dict[str, Any]:
    """
    Convert a ClaimRecord to a plain JSON-serialisable dict.
    Enum fields are converted to their .value strings.
    """
    return {
        "claim_type":         claim.claim_type,
        "claim_status":       claim.claim_status.value,
        "claim_verifiability": claim.claim_verifiability.value,
        "seed_risk_type":     (
            claim.seed_risk_type.value
            if claim.seed_risk_type is not None else None
        ),
        "hallucination_type": (
            claim.hallucination_type.value
            if claim.hallucination_type is not None else None
        ),
        "matched_phrase":     claim.matched_phrase,
        "matched_seed":       claim.matched_seed,
        "source":             claim.source,
    }


def record_to_dict(record: TurnEvaluationRecord) -> Dict[str, Any]:
    """
    Convert a TurnEvaluationRecord to a plain, JSON-serialisable dict.

    All enum values are converted to their string .value representations.
    The claims list is recursively serialised.

    Parameters
    ----------
    record : TurnEvaluationRecord

    Returns
    -------
    dict
        Flat dict suitable for json.dumps().
    """
    return {
        "turn_id":             record.turn_id,
        "total_claims":        record.total_claims,
        "supported_count":     record.supported_count,
        "contradicted_count":  record.contradicted_count,
        "unsupported_count":   record.unsupported_count,
        "vague_count":         record.vague_count,
        "has_contradiction":   record.has_contradiction,
        "has_unsupported":     record.has_unsupported,
        "has_vague":           record.has_vague,
        "reasoning_path":      record.reasoning_path,
        "trajectory_events":   list(record.trajectory_events),
        "claims":              [_serialise_claim(c) for c in record.claims],
    }


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def append_turn_record(
    record: TurnEvaluationRecord,
    path: str | Path,
) -> None:
    """
    Append one TurnEvaluationRecord to a JSONL file as a single JSON line.

    The file is created if it does not exist.  Parent directories are
    created automatically.  Existing records are never overwritten.

    Parameters
    ----------
    record : TurnEvaluationRecord
    path : str | Path
        Target JSONL file path.

    Raises
    ------
    TypeError
        If record is not a TurnEvaluationRecord.
    OSError
        If the file cannot be opened for writing.
    """
    if not isinstance(record, TurnEvaluationRecord):
        raise TypeError(
            f"Expected TurnEvaluationRecord, got {type(record).__name__}"
        )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record_to_dict(record), ensure_ascii=False)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def load_eval_records(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load all evaluation records from a JSONL file.

    Empty lines are silently skipped.  Malformed JSON lines raise ValueError.

    Parameters
    ----------
    path : str | Path
        JSONL file produced by append_turn_record().

    Returns
    -------
    list[dict]
        Records in file order.  Empty list if the file does not exist.

    Raises
    ------
    ValueError
        On the first malformed (non-empty) JSON line encountered.
    """
    p = Path(path)
    if not p.exists():
        return []

    records: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {lineno} of {p}: {exc}"
                ) from exc
            records.append(obj)
    return records


# ---------------------------------------------------------------------------
# Summary aggregator
# ---------------------------------------------------------------------------

def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate statistics over a list of plain evaluation dicts.

    Parameters
    ----------
    records : list[dict]
        As returned by load_eval_records() or built from record_to_dict().
        May be empty.

    Returns
    -------
    dict with keys:
        total_turns               : int
        total_claims              : int
        supported_claims          : int
        contradicted_claims       : int
        unsupported_claims        : int
        vague_claims              : int
        turns_with_contradiction  : int
        turns_with_unsupported    : int
        turns_with_vague          : int
        reasoning_path_counts     : dict[str, int]   path label → count
        trajectory_event_counts   : dict[str, int]   event label → count
    """
    n = len(records)
    total_claims       = 0
    supported_claims   = 0
    contradicted_claims = 0
    unsupported_claims = 0
    vague_claims       = 0
    turns_contradiction = 0
    turns_unsupported  = 0
    turns_vague        = 0
    path_counts: Dict[str, int] = {}
    event_counts: Dict[str, int] = {}

    for rec in records:
        total_claims        += int(rec.get("total_claims", 0))
        supported_claims    += int(rec.get("supported_count", 0))
        contradicted_claims += int(rec.get("contradicted_count", 0))
        unsupported_claims  += int(rec.get("unsupported_count", 0))
        vague_claims        += int(rec.get("vague_count", 0))

        if rec.get("has_contradiction"):
            turns_contradiction += 1
        if rec.get("has_unsupported"):
            turns_unsupported += 1
        if rec.get("has_vague"):
            turns_vague += 1

        path = rec.get("reasoning_path", "unknown")
        path_counts[path] = path_counts.get(path, 0) + 1

        for ev in rec.get("trajectory_events", []):
            event_counts[ev] = event_counts.get(ev, 0) + 1

    return {
        "total_turns":              n,
        "total_claims":             total_claims,
        "supported_claims":         supported_claims,
        "contradicted_claims":      contradicted_claims,
        "unsupported_claims":       unsupported_claims,
        "vague_claims":             vague_claims,
        "turns_with_contradiction": turns_contradiction,
        "turns_with_unsupported":   turns_unsupported,
        "turns_with_vague":         turns_vague,
        "reasoning_path_counts":    path_counts,
        "trajectory_event_counts":  event_counts,
    }
