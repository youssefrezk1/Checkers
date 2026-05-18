# checkers/evaluation/run_seed_ack_audit.py
#
# Phase 4.3a: seed-acknowledgement audit on real manual_eval logs.
#
# PURPOSE
# -------
# Measures, per RED turn, how often the ranker LLM acknowledges seeded
# strategic facts in its reasoning.  For each seed string we reverse-map to
# the claim type(s) whose seed_markers it would match, then check whether
# the reasoning text contains any of the extractor phrases for that claim
# type.  Aggregates are produced overall, by minimax bucket, and by
# reasoning_path label.
#
# DESIGN CONSTRAINTS
# ------------------
# - No imports from the runtime pipeline (engine, agents, graph, state).
# - No LLM calls.  No NLI/semantic model calls.  Purely symbolic.
# - Inputs (log records, seeds, facts) are never mutated.
# - Deterministic: same log -> same report always.
# - Read-only on extractor/verifier internals.  No prompt or pipeline edits.
#
# USAGE
# -----
#   python -m checkers.evaluation.run_seed_ack_audit \
#       --log logs/manual_eval/manual_game_*.jsonl
#
#   python -m checkers.evaluation.run_seed_ack_audit \
#       --log logs/manual_eval/manual_game_20260512_175023.jsonl --json-out
#
# This module is evaluation-only.  It must NEVER be imported by the pipeline.

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Evaluation-only imports - none from the runtime pipeline.
from checkers.evaluation.claim_extractor import _PHRASE_TABLE
from checkers.evaluation.claim_taxonomy import _CLAIM_REGISTRY, TaxonomyCategory


# Only claim types in these categories are eligible for seed acknowledgement
# analysis.  NON_VERIFIABLE_VAGUE / FORBIDDEN_UNGROUNDED / SCHEMA_LEAK types
# are excluded by design.
_ELIGIBLE_CATEGORIES: frozenset[TaxonomyCategory] = frozenset({
    TaxonomyCategory.VERIFIABLE,
    TaxonomyCategory.AMBIGUOUS_CONTEXT_REQUIRED,
})


# ---------------------------------------------------------------------------
# Seed -> claim type reverse mapping
# ---------------------------------------------------------------------------

def map_seed_to_claim_types(seed_string: str) -> list[str]:
    """
    Reverse the extractor's seed-marker matching: given one seed string,
    return the list of claim_type names whose seed_markers would match it.

    Uses the same anchored ("^prefix") and substring matching semantics as
    claim_extractor._find_matching_seed, but checks ONE seed against ALL
    phrase-table entries (the inverse direction).

    Only returns claim types whose taxonomy category is VERIFIABLE or
    AMBIGUOUS_CONTEXT_REQUIRED; vague/forbidden/schema-leak types are
    filtered out because acknowledgement of those is not meaningful.

    Parameters
    ----------
    seed_string : str
        One seed line from a reasoning_seeds list, e.g.
        "opponent_can_recapture=false - safe from recapture".

    Returns
    -------
    list[str]
        Sorted list of matching claim types (deterministic).  Empty list if
        no eligible claim type matches.
    """
    if not seed_string:
        return []

    seed_lower = seed_string.lower()
    matched: set[str] = set()

    for entry in _PHRASE_TABLE:
        spec = _CLAIM_REGISTRY.get(entry.claim_type)
        if spec is None or spec.category not in _ELIGIBLE_CATEGORIES:
            continue
        if not entry.seed_markers:
            continue
        for marker in entry.seed_markers:
            if marker.startswith("^"):
                if seed_lower.startswith(marker[1:].lower()):
                    matched.add(entry.claim_type)
                    break
            elif marker.lower() in seed_lower:
                matched.add(entry.claim_type)
                break

    return sorted(matched)


def is_seed_acknowledged(claim_type: str, reasoning_text: str) -> bool:
    """
    Return True if the reasoning text contains any of the extractor phrases
    for the given claim_type (case-insensitive substring match).

    This uses the SAME phrase list the extractor uses, so an acknowledgement
    here corresponds exactly to a claim the extractor would have produced.
    """
    if not reasoning_text:
        return False
    text_lower = reasoning_text.lower()
    for entry in _PHRASE_TABLE:
        if entry.claim_type != claim_type:
            continue
        for phrase in entry.phrases:
            if phrase.lower() in text_lower:
                return True
        return False
    return False


# ---------------------------------------------------------------------------
# Minimax bucket
# ---------------------------------------------------------------------------

def get_minimax_bucket(score: Any) -> str:
    """
    Bucket a numeric minimax score for aggregation.

    Buckets
    -------
    "positive"        : score > 0
    "slightly_losing" : -50 <= score <= 0
    "deeply_losing"   : score < -50
    "unknown"         : score is None or not numeric

    The thresholds match the qualitative score-state semantics already used
    by the pipeline's inter-turn memory layer.
    """
    if not isinstance(score, (int, float)):
        return "unknown"
    if score > 0:
        return "positive"
    if score >= -50:
        return "slightly_losing"
    return "deeply_losing"


# ---------------------------------------------------------------------------
# Per-turn seed-ack computation
# ---------------------------------------------------------------------------

def compute_turn_seed_ack(record: dict[str, Any]) -> dict[str, Any]:
    """
    Compute seed acknowledgement results for one JSONL log record.

    For each seed in the record:
      1. Reverse-map to one or more claim types via map_seed_to_claim_types.
      2. If at least one eligible claim type matches, mark the seed
         "acknowledged" iff the reasoning contains an extractor phrase for
         ANY of the matched claim types.
      3. If no eligible claim type matches, mark the seed "unclassified"
         (the seed is not in the verifiable-claim taxonomy).

    Parameters
    ----------
    record : dict
        A single log record from a manual_game_*.jsonl file.
        Missing or None fields are handled safely.  Not mutated.

    Returns
    -------
    dict with keys:
        turn_id : str
        player : str
        turn : int
        minimax_score : float | None
        bucket : str            # one of "positive", "slightly_losing",
                                #         "deeply_losing", "unknown"
        reasoning_path : str    # from record; "" if absent
        seed_count : int
        classified_seed_count : int
        acknowledged_seed_count : int
        unacknowledged_seed_count : int
        unclassified_seed_count : int
        per_seed : list[dict]   # one entry per seed, with mapped claim_types,
                                #   acknowledged flag, and the seed string
        acknowledged_claim_types : list[str]   # union across all seeds
        ignored_claim_types : list[str]        # mapped but not acknowledged
    """
    reasoning: str = record.get("reasoning") or ""
    seeds: list[str] = list(record.get("seeds") or [])
    facts: dict[str, Any] = dict(record.get("facts") or {})

    turn_id: str = record.get("turn_id") or f"turn_{record.get('turn', '?')}"
    player: str = record.get("player") or "UNKNOWN"
    turn: int = record.get("turn") or 0
    reasoning_path: str = record.get("reasoning_path") or ""
    minimax_score = facts.get("minimax_score")
    bucket = get_minimax_bucket(minimax_score)

    per_seed: list[dict[str, Any]] = []
    acknowledged_types: set[str] = set()
    ignored_types: set[str] = set()
    n_classified = n_ack = n_unack = n_unclassified = 0

    for seed in seeds:
        mapped = map_seed_to_claim_types(seed)
        if not mapped:
            per_seed.append({
                "seed": seed,
                "claim_types": [],
                "acknowledged": False,
                "status": "unclassified",
            })
            n_unclassified += 1
            continue

        n_classified += 1
        ack = any(is_seed_acknowledged(ct, reasoning) for ct in mapped)
        per_seed.append({
            "seed": seed,
            "claim_types": mapped,
            "acknowledged": ack,
            "status": "acknowledged" if ack else "ignored",
        })
        if ack:
            n_ack += 1
            for ct in mapped:
                if is_seed_acknowledged(ct, reasoning):
                    acknowledged_types.add(ct)
                else:
                    ignored_types.add(ct)
        else:
            n_unack += 1
            for ct in mapped:
                ignored_types.add(ct)

    return {
        "turn_id": turn_id,
        "player": player,
        "turn": turn,
        "minimax_score": minimax_score,
        "bucket": bucket,
        "reasoning_path": reasoning_path,
        "seed_count": len(seeds),
        "classified_seed_count": n_classified,
        "acknowledged_seed_count": n_ack,
        "unacknowledged_seed_count": n_unack,
        "unclassified_seed_count": n_unclassified,
        "per_seed": per_seed,
        "acknowledged_claim_types": sorted(acknowledged_types),
        "ignored_claim_types": sorted(ignored_types - acknowledged_types),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_seed_ack(turn_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-turn seed-ack results across RED turns.

    Returns a dict with overall, per-bucket, and per-reasoning-path stats,
    plus per-claim-type ack rates and the top-ignored claim types.
    """
    red_turns = [t for t in turn_results if t.get("player") == "RED"]

    total_seeds = 0
    total_classified = 0
    total_ack = 0
    total_unack = 0
    total_unclassified = 0

    seeds_by_type: Counter = Counter()           # times each claim_type was seeded
    ack_by_type: Counter = Counter()             # times acknowledged
    ignored_by_type: Counter = Counter()         # times mapped but not acknowledged

    per_bucket: dict[str, dict[str, int]] = defaultdict(
        lambda: {"turns": 0, "seeds": 0, "classified": 0, "ack": 0, "unack": 0}
    )
    per_path: dict[str, dict[str, int]] = defaultdict(
        lambda: {"turns": 0, "seeds": 0, "classified": 0, "ack": 0, "unack": 0}
    )

    for t in red_turns:
        total_seeds += t["seed_count"]
        total_classified += t["classified_seed_count"]
        total_ack += t["acknowledged_seed_count"]
        total_unack += t["unacknowledged_seed_count"]
        total_unclassified += t["unclassified_seed_count"]

        b = t["bucket"]
        per_bucket[b]["turns"] += 1
        per_bucket[b]["seeds"] += t["seed_count"]
        per_bucket[b]["classified"] += t["classified_seed_count"]
        per_bucket[b]["ack"] += t["acknowledged_seed_count"]
        per_bucket[b]["unack"] += t["unacknowledged_seed_count"]

        p = t["reasoning_path"] or "unknown"
        per_path[p]["turns"] += 1
        per_path[p]["seeds"] += t["seed_count"]
        per_path[p]["classified"] += t["classified_seed_count"]
        per_path[p]["ack"] += t["acknowledged_seed_count"]
        per_path[p]["unack"] += t["unacknowledged_seed_count"]

        # Per-claim-type counts: count each claim-type appearance per seed.
        # A seed mapping to multiple claim types contributes to each one.
        for entry in t["per_seed"]:
            if entry["status"] == "unclassified":
                continue
            for ct in entry["claim_types"]:
                seeds_by_type[ct] += 1
                # Credit the claim_type only if its OWN phrases appeared in
                # the reasoning (not just a sibling claim_type from the same
                # seed).  acknowledged_claim_types holds that per-type set.
                if ct in t["acknowledged_claim_types"]:
                    ack_by_type[ct] += 1
                else:
                    ignored_by_type[ct] += 1

    ack_rate_by_type: dict[str, float] = {}
    for ct, seeded in seeds_by_type.items():
        if seeded:
            ack_rate_by_type[ct] = ack_by_type[ct] / seeded

    top_ignored = ignored_by_type.most_common(10)

    overall_rate = (total_ack / total_classified) if total_classified else 0.0

    return {
        "total_red_turns": len(red_turns),
        "total_all_turns": len(turn_results),
        "total_seeds": total_seeds,
        "total_classified_seeds": total_classified,
        "total_unclassified_seeds": total_unclassified,
        "total_acknowledged_seeds": total_ack,
        "total_unacknowledged_seeds": total_unack,
        "overall_ack_rate": overall_rate,
        "seeds_by_claim_type": dict(seeds_by_type),
        "ack_by_claim_type": dict(ack_by_type),
        "ignored_by_claim_type": dict(ignored_by_type),
        "ack_rate_by_claim_type": ack_rate_by_type,
        "top_10_ignored_claim_types": top_ignored,
        "per_bucket": {k: dict(v) for k, v in per_bucket.items()},
        "per_reasoning_path": {k: dict(v) for k, v in per_path.items()},
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(
    agg: dict[str, Any],
    turn_results: list[dict[str, Any]],
    out: Any = None,
) -> None:
    """Print the human-readable seed-ack report to `out` (default: stdout)."""
    if out is None:
        out = sys.stdout

    red_turns = [t for t in turn_results if t.get("player") == "RED"]

    print("=" * 72, file=out)
    print("SEED ACKNOWLEDGEMENT AUDIT - Phase 4.3a", file=out)
    print("=" * 72, file=out)
    print(f"  RED turns analyzed                  : {agg['total_red_turns']}", file=out)
    print(f"  Total seeds across RED turns        : {agg['total_seeds']}", file=out)
    print(f"  Classified seeds (eligible)         : {agg['total_classified_seeds']}", file=out)
    print(f"  Unclassified seeds (skipped)        : {agg['total_unclassified_seeds']}", file=out)
    print(f"  Acknowledged in reasoning           : {agg['total_acknowledged_seeds']}", file=out)
    print(f"  Ignored in reasoning                : {agg['total_unacknowledged_seeds']}", file=out)
    print(f"  Overall acknowledgement rate        : {agg['overall_ack_rate']:.1%}", file=out)
    print(file=out)

    print("-- Ack rate per claim type ---------------------------------------", file=out)
    rows = sorted(
        agg["ack_rate_by_claim_type"].items(),
        key=lambda kv: (-agg["seeds_by_claim_type"].get(kv[0], 0), kv[0]),
    )
    for ct, rate in rows:
        seeded = agg["seeds_by_claim_type"].get(ct, 0)
        acked = agg["ack_by_claim_type"].get(ct, 0)
        ignored = agg["ignored_by_claim_type"].get(ct, 0)
        print(
            f"  {ct:38s}  seeded={seeded:3d}  ack={acked:3d}  ignored={ignored:3d}  rate={rate:6.1%}",
            file=out,
        )
    print(file=out)

    print("-- Top 10 ignored claim types ------------------------------------", file=out)
    if agg["top_10_ignored_claim_types"]:
        for ct, cnt in agg["top_10_ignored_claim_types"]:
            seeded = agg["seeds_by_claim_type"].get(ct, 0)
            print(
                f"  {ct:38s}  ignored={cnt:3d}  seeded={seeded:3d}",
                file=out,
            )
    else:
        print("  (no ignored seeds)", file=out)
    print(file=out)

    print("-- By minimax bucket ---------------------------------------------", file=out)
    for bucket in ("positive", "slightly_losing", "deeply_losing", "unknown"):
        b = agg["per_bucket"].get(bucket)
        if not b:
            continue
        rate = (b["ack"] / b["classified"]) if b["classified"] else 0.0
        print(
            f"  {bucket:18s}  turns={b['turns']:3d}  seeds={b['seeds']:3d}"
            f"  classified={b['classified']:3d}  ack={b['ack']:3d}"
            f"  rate={rate:6.1%}",
            file=out,
        )
    print(file=out)

    print("-- By reasoning_path ---------------------------------------------", file=out)
    for path, p in sorted(agg["per_reasoning_path"].items()):
        rate = (p["ack"] / p["classified"]) if p["classified"] else 0.0
        print(
            f"  {path:30s}  turns={p['turns']:3d}  seeds={p['seeds']:3d}"
            f"  classified={p['classified']:3d}  ack={p['ack']:3d}"
            f"  rate={rate:6.1%}",
            file=out,
        )
    print(file=out)

    print("-- Sample ignored seeds (first 10) -------------------------------", file=out)
    examples_shown = 0
    for t in red_turns:
        for entry in t["per_seed"]:
            if entry["status"] != "ignored":
                continue
            print(
                f"  [{t['turn_id']}] ({t['bucket']}) {','.join(entry['claim_types'])}"
                f"  <-  {entry['seed'][:80]}",
                file=out,
            )
            examples_shown += 1
            if examples_shown >= 10:
                break
        if examples_shown >= 10:
            break
    if examples_shown == 0:
        print("  (no ignored seeds)", file=out)
    print(file=out)


# ---------------------------------------------------------------------------
# Log loader
# ---------------------------------------------------------------------------

def run_audit(log_paths: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Load RED-player turns from log files, compute per-turn seed-ack, and
    aggregate.

    Parameters
    ----------
    log_paths : list[str]
        Paths to manual_game_*.jsonl files.

    Returns
    -------
    (aggregate_dict, list_of_per_turn_dicts)
    """
    turn_results: list[dict[str, Any]] = []

    for path in log_paths:
        try:
            with open(path) as fh:
                for raw_line in fh:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("player") != "RED":
                        continue
                    if not record.get("reasoning"):
                        continue
                    result = compute_turn_seed_ack(record)
                    result["_source_file"] = Path(path).name
                    turn_results.append(result)
        except OSError as exc:
            print(f"WARNING: cannot open {path!r}: {exc}", file=sys.stderr)

    agg = aggregate_seed_ack(turn_results)
    return agg, turn_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4.3a: seed-acknowledgement audit on manual_eval JSONL logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m checkers.evaluation.run_seed_ack_audit \\\n"
            "      --log logs/manual_eval/manual_game_*.jsonl\n"
            "  python -m checkers.evaluation.run_seed_ack_audit \\\n"
            "      --log logs/manual_eval/manual_game_20260512_175023.jsonl --json-out\n"
        ),
    )
    parser.add_argument(
        "--log",
        nargs="+",
        required=True,
        help="Path(s) to manual_game_*.jsonl files; shell glob patterns are supported.",
    )
    parser.add_argument(
        "--json-out",
        dest="json_out",
        action="store_true",
        help="Emit aggregate statistics as JSON instead of the formatted text report.",
    )
    args = parser.parse_args(argv)

    log_paths: list[str] = []
    for pattern in args.log:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            log_paths.extend(expanded)
        else:
            log_paths.append(pattern)

    if not log_paths:
        print("ERROR: no log files found.", file=sys.stderr)
        sys.exit(1)

    agg, turn_results = run_audit(log_paths)

    if args.json_out:
        print(json.dumps(agg, indent=2, default=str))
    else:
        print_report(agg, turn_results)


if __name__ == "__main__":
    main()
