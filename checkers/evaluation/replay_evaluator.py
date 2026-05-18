# checkers/evaluation/replay_evaluator.py
#
# Minimal replay evaluator for existing ranker output logs.
#
# PURPOSE
# -------
# Reads a JSONL source log produced by the runtime pipeline (one dict per
# turn), runs evaluate_turn() on each entry, appends TurnEvaluationRecord
# objects to an evaluation output file, and returns an aggregate summary.
#
# DESIGN CONSTRAINTS
# ------------------
# - No imports from the runtime pipeline (engine, agents, graph, state).
# - No LLM calls.
# - No plotting, markdown reports, or move quality inference.
# - Missing source file → zero summary, no crash.
# - Malformed JSON lines → ValueError with line number.
# - Missing optional fields → handled conservatively (empty string / {}).
# - Deterministic: same source log → same eval output always.
#
# USAGE
# -----
# This module is imported only by evaluation scripts and tests.
# It must NEVER be imported by the runtime pipeline.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from checkers.evaluation.turn_evaluator import evaluate_turn
from checkers.evaluation.eval_logger import (
    append_turn_record,
    load_eval_records,
    summarize_records,
)

# ---------------------------------------------------------------------------
# Source record field resolution helpers
# ---------------------------------------------------------------------------

def _extract_reasoning(record: Dict[str, Any]) -> str:
    """
    Return the reasoning text from a source record.
    Prefers 'last_move_reasoning', falls back to 'reasoning_text'.
    Returns empty string if neither key is present or value is not a string.
    """
    for key in ("last_move_reasoning", "reasoning_text"):
        val = record.get(key)
        if isinstance(val, str):
            return val
    return ""


def _extract_facts(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the move-facts dict from a source record.
    Prefers 'chosen_move_facts', falls back to 'facts'.
    Returns {} if neither key is present or value is not a dict.
    """
    for key in ("chosen_move_facts", "facts"):
        val = record.get(key)
        if isinstance(val, dict):
            return val
    return {}


def _extract_seeds(record: Dict[str, Any]) -> List[str]:
    """
    Extract reasoning_seeds from ranker_diagnostics["reasoning_seeds"].
    Returns [] if absent or not a list.
    """
    diag = record.get("ranker_diagnostics")
    if not isinstance(diag, dict):
        return []
    seeds = diag.get("reasoning_seeds")
    if isinstance(seeds, list):
        return [s for s in seeds if isinstance(s, str)]
    return []


def _extract_turn_id(record: Dict[str, Any], line_index: int) -> str:
    """
    Return the turn_id from the source record, or a generated fallback.
    """
    val = record.get("turn_id")
    if isinstance(val, str) and val.strip():
        return val.strip()
    return f"turn_{line_index}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def replay_evaluate_file(
    source_path: str | Path,
    output_path: str | Path,
) -> Dict[str, Any]:
    """
    Evaluate every turn in a JSONL source log and append results to output_path.

    Parameters
    ----------
    source_path : str | Path
        JSONL source log. Each line is a dict containing at least one of:
        'last_move_reasoning' or 'reasoning_text'.
        Optional: 'ranker_diagnostics', 'chosen_move_facts' or 'facts',
        'turn_id'.

    output_path : str | Path
        JSONL destination for TurnEvaluationRecord objects. Records are
        appended; existing records are never overwritten.

    Returns
    -------
    dict
        Aggregate summary from summarize_records(load_eval_records(output_path)).
        Returns a zero summary (from summarize_records([])) if source_path
        does not exist.

    Raises
    ------
    ValueError
        On the first malformed (non-empty, non-JSON) source line encountered.
    """
    src = Path(source_path)

    if not src.exists():
        return summarize_records([])

    turn_index = 0  # counts only non-empty lines processed

    with src.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue

            try:
                source_record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {lineno} of {src}: {exc}"
                ) from exc

            turn_index += 1

            reasoning_text   = _extract_reasoning(source_record)
            facts            = _extract_facts(source_record)
            seeds            = _extract_seeds(source_record)
            diag             = source_record.get("ranker_diagnostics") or {}
            turn_id          = _extract_turn_id(source_record, turn_index)

            eval_record = evaluate_turn(
                reasoning_text,
                reasoning_seeds=seeds,
                facts=facts,
                ranker_diagnostics=diag,
                turn_id=turn_id,
            )

            append_turn_record(eval_record, output_path)

    return summarize_records(load_eval_records(output_path))
