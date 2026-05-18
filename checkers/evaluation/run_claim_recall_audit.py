# checkers/evaluation/run_claim_recall_audit.py
#
# Phase 4.2: element-level recall audit on real manual_eval logs.
#
# PURPOSE
# -------
# Measures extractor recall gaps on real logged turns by comparing what the
# extractor actually detected against what it SHOULD have detected given the
# seeds and symbolic facts for each turn.
#
# DESIGN CONSTRAINTS
# ------------------
# - No imports from the runtime pipeline (engine, agents, graph, state).
# - No LLM calls. No NLI/semantic model calls. Purely symbolic.
# - Inputs (log records, seeds, facts) are never mutated.
# - Deterministic: same log → same report always.
#
# USAGE
# -----
#   python -m checkers.evaluation.run_claim_recall_audit \
#       --log logs/manual_eval/manual_game_*.jsonl
#
#   python -m checkers.evaluation.run_claim_recall_audit \
#       --log logs/manual_eval/manual_game_20260512_175023.jsonl --json-out
#
# This module is evaluation-only. It must NEVER be imported by the pipeline.

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Evaluation-only imports — none from the runtime pipeline.
from checkers.evaluation.claim_extractor import (
    extract_claims,
    _PHRASE_TABLE,
    _find_matching_seed,   # private helper; acceptable in evaluation-only code
)
from checkers.evaluation.claim_verifier import verify_claims, _VERIFICATION_RULES
from checkers.evaluation.claim_taxonomy import _CLAIM_REGISTRY, TaxonomyCategory
from checkers.evaluation.reasoning_taxonomy import ClaimStatus


# Claim categories eligible for the recall audit.
# NON_VERIFIABLE_VAGUE and FORBIDDEN_UNGROUNDED types are structurally
# unverifiable and are excluded from the "expected" set.
_ELIGIBLE_CATEGORIES: frozenset[TaxonomyCategory] = frozenset({
    TaxonomyCategory.VERIFIABLE,
    TaxonomyCategory.AMBIGUOUS_CONTEXT_REQUIRED,
})


# ---------------------------------------------------------------------------
# Core: expected claim type computation
# ---------------------------------------------------------------------------

def expected_claim_types(
    reasoning_text: str,
    seeds: list[str],
    facts: dict[str, Any],
) -> dict[str, str]:
    """
    Return the set of claim types that SHOULD have been extracted for this turn.

    Conservative criteria — all three must hold for a claim type to be expected:

    1. Taxonomy category is VERIFIABLE or AMBIGUOUS_CONTEXT_REQUIRED.
       (Unverifiable strategic and forbidden-vocabulary types are excluded.)

    2. The claim verifier returns SUPPORTED given the current facts dict.
       (Avoids flagging claim types whose fact values contradict the claim,
       or whose required facts are absent.  Uses the same symbolic rules as
       claim_verifier._VERIFICATION_RULES.)

    3. At least one of:
       a. A seed marker appears in the seeds list — the LLM was explicitly
          told about this fact and should have reflected it in reasoning.
       b. A phrase from the extractor's phrase table appears in the reasoning
          text — the LLM DID mention it, so the extractor should have fired.

    Parameters
    ----------
    reasoning_text : str
    seeds : list[str]
    facts : dict[str, Any]

    Returns
    -------
    dict[str, str]
        Maps claim_type → evidence_source, where evidence_source is one of:
        "seed"        — seed was present, phrase absent from reasoning
        "phrase"      — phrase found in reasoning, no seed matched
        "seed+phrase" — both seed and phrase present

    Guarantees
    ----------
    - seeds and facts are never mutated.
    - Only returns VERIFIABLE or AMBIGUOUS_CONTEXT_REQUIRED types.
    - Empty reasoning/seeds/facts → returns {} safely.
    """
    text_lower = reasoning_text.lower() if reasoning_text else ""
    _seeds: list[str] = seeds if seeds is not None else []
    _facts: dict[str, Any] = facts if facts is not None else {}

    result: dict[str, str] = {}

    for entry in _PHRASE_TABLE:
        spec = _CLAIM_REGISTRY.get(entry.claim_type)
        if spec is None or spec.category not in _ELIGIBLE_CATEGORIES:
            continue

        # Criterion 2: verifier confirms the claim is factually supported.
        rule = _VERIFICATION_RULES.get(entry.claim_type)
        if rule is None:
            continue
        if rule(_facts) != ClaimStatus.SUPPORTED:
            continue

        # Criterion 3: seed or phrase evidence.
        seed_match = bool(_find_matching_seed(entry.seed_markers, _seeds))
        phrase_match = any(p.lower() in text_lower for p in entry.phrases)

        if not seed_match and not phrase_match:
            continue

        if seed_match and phrase_match:
            result[entry.claim_type] = "seed+phrase"
        elif seed_match:
            result[entry.claim_type] = "seed"
        else:
            result[entry.claim_type] = "phrase"

    return result


# ---------------------------------------------------------------------------
# Per-turn recall computation
# ---------------------------------------------------------------------------

def compute_turn_recall(record: dict[str, Any]) -> dict[str, Any]:
    """
    Compute the recall report for one JSONL log record.

    Re-runs extract_claims() and verify_claims() fresh from the record fields,
    then computes expected_claim_types() and reports what is missing.

    Parameters
    ----------
    record : dict
        A single log record as loaded from a manual_game_*.jsonl file.
        Missing or None fields are handled safely.  The dict is not mutated.

    Returns
    -------
    dict with keys:
        turn_id : str
        player : str
        turn : int
        minimax_score : float | None
        extracted_claim_types : list[str]
        expected_claim_types : dict[str, str]   {type: evidence_source}
        missing_types : list[str]
            Claim types in expected but not in extracted.
        missing_with_seed : list[str]
            Subset of missing_types where the seed was present (but LLM did
            not use the phrase in reasoning).
        missing_with_phrase : list[str]
            Subset of missing_types where the phrase appeared in reasoning
            (extractor failed to fire — possible extractor bug or polarity
            suppression).
        notes : list[str]
            Human-readable notes for each missing entry.
    """
    reasoning: str = record.get("reasoning") or ""
    seeds: list[str] = list(record.get("seeds") or [])
    facts: dict[str, Any] = dict(record.get("facts") or {})

    turn_id: str = record.get("turn_id") or f"turn_{record.get('turn', '?')}"
    player: str = record.get("player") or "UNKNOWN"
    turn: int = record.get("turn") or 0
    minimax_score = facts.get("minimax_score")

    # Re-run the full extraction + verification pipeline from raw log data.
    raw_claims = extract_claims(reasoning, seeds, facts)
    verified_claims = verify_claims(raw_claims, facts)

    extracted_types: list[str] = [c.claim_type for c in verified_claims]
    extracted_set: set[str] = set(extracted_types)

    expected: dict[str, str] = expected_claim_types(reasoning, seeds, facts)
    missing: list[str] = [ct for ct in expected if ct not in extracted_set]

    missing_with_seed: list[str] = [
        ct for ct in missing if expected[ct] in ("seed", "seed+phrase")
    ]
    missing_with_phrase: list[str] = [
        ct for ct in missing if expected[ct] in ("phrase", "seed+phrase")
    ]

    notes: list[str] = []
    for ct in missing_with_phrase:
        notes.append(
            f"phrase matched for '{ct}' but extractor did not emit it"
        )
    for ct in missing_with_seed:
        if ct not in missing_with_phrase:
            notes.append(
                f"seed present for '{ct}' but claim absent from reasoning"
            )

    return {
        "turn_id": turn_id,
        "player": player,
        "turn": turn,
        "minimax_score": minimax_score,
        "extracted_claim_types": extracted_types,
        "expected_claim_types": expected,
        "missing_types": missing,
        "missing_with_seed": missing_with_seed,
        "missing_with_phrase": missing_with_phrase,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_recall_results(turn_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate a list of compute_turn_recall() results into summary statistics.

    Counts are computed separately for RED turns (pipeline turns) and for all
    turns combined.  The top-10 recall gaps are sorted by descending frequency.
    """
    red_turns = [t for t in turn_results if t.get("player") == "RED"]

    extracted_counts: Counter = Counter()
    expected_counts: Counter = Counter()
    missing_counts: Counter = Counter()
    missing_seed_counts: Counter = Counter()
    missing_phrase_counts: Counter = Counter()
    ambiguous_skip_count: int = 0

    for t in red_turns:
        for ct in t["extracted_claim_types"]:
            extracted_counts[ct] += 1
            spec = _CLAIM_REGISTRY.get(ct)
            if spec and spec.category not in _ELIGIBLE_CATEGORIES:
                ambiguous_skip_count += 1
        for ct in t["expected_claim_types"]:
            expected_counts[ct] += 1
        for ct in t["missing_types"]:
            missing_counts[ct] += 1
        for ct in t["missing_with_seed"]:
            missing_seed_counts[ct] += 1
        for ct in t["missing_with_phrase"]:
            missing_phrase_counts[ct] += 1

    # Types with a verifier that were never extracted in any RED turn.
    never_extracted_verifier: list[str] = [
        ct for ct, spec in _CLAIM_REGISTRY.items()
        if spec.verifier_exists and extracted_counts[ct] == 0
    ]

    top_gaps = missing_counts.most_common(10)

    return {
        "total_red_turns": len(red_turns),
        "total_all_turns": len(turn_results),
        "extracted_claim_type_counts": dict(extracted_counts),
        "expected_claim_type_counts": dict(expected_counts),
        "missing_claim_type_counts": dict(missing_counts),
        "missing_with_seed_counts": dict(missing_seed_counts),
        "missing_with_phrase_counts": dict(missing_phrase_counts),
        "never_extracted_verifier_types": never_extracted_verifier,
        "ambiguous_nonverifiable_skipped_count": ambiguous_skip_count,
        "top_10_recall_gaps": top_gaps,
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(
    agg: dict[str, Any],
    turn_results: list[dict[str, Any]],
    out: Any = None,
) -> None:
    """Print the human-readable audit report to `out` (default: stdout)."""
    if out is None:
        out = sys.stdout

    red_turns = [t for t in turn_results if t.get("player") == "RED"]

    print("=" * 72, file=out)
    print("CLAIM RECALL AUDIT — Phase 4.2", file=out)
    print("=" * 72, file=out)
    print(f"  RED turns analyzed                  : {agg['total_red_turns']}", file=out)
    print(
        f"  Ambiguous/non-verifiable instances  : {agg['ambiguous_nonverifiable_skipped_count']}",
        file=out,
    )
    print(file=out)

    print("── Extracted vs Expected counts (RED turns) ──────────────────────", file=out)
    all_types = sorted(
        set(agg["extracted_claim_type_counts"]) | set(agg["expected_claim_type_counts"])
    )
    for ct in all_types:
        ext_cnt = agg["extracted_claim_type_counts"].get(ct, 0)
        exp_cnt = agg["expected_claim_type_counts"].get(ct, 0)
        miss_cnt = agg["missing_claim_type_counts"].get(ct, 0)
        recall = f"{ext_cnt / exp_cnt:.0%}" if exp_cnt else "  n/a"
        print(
            f"  {ct:38s}  ext={ext_cnt:3d}  exp={exp_cnt:3d}  miss={miss_cnt:3d}  recall={recall}",
            file=out,
        )
    print(file=out)

    print("── Top 10 recall gaps (expected but not extracted) ───────────────", file=out)
    if agg["top_10_recall_gaps"]:
        for ct, cnt in agg["top_10_recall_gaps"]:
            seed_cnt = agg["missing_with_seed_counts"].get(ct, 0)
            phrase_cnt = agg["missing_with_phrase_counts"].get(ct, 0)
            exp_cnt = agg["expected_claim_type_counts"].get(ct, 0)
            recall = f"{(exp_cnt - cnt) / exp_cnt:.0%}" if exp_cnt else "n/a"
            print(
                f"  {ct:38s}  missing={cnt:3d}  seed={seed_cnt:3d}  phrase={phrase_cnt:3d}"
                f"  recall={recall}",
                file=out,
            )
    else:
        print("  (no recall gaps detected)", file=out)
    print(file=out)

    print("── Never-extracted types with a verifier ─────────────────────────", file=out)
    if agg["never_extracted_verifier_types"]:
        for ct in agg["never_extracted_verifier_types"]:
            print(f"  {ct}", file=out)
    else:
        print("  (all verifiable types extracted at least once)", file=out)
    print(file=out)

    print("── Per-turn recall table (RED turns) ─────────────────────────────", file=out)
    col = 44
    print(
        f"  {'turn_id':{col}}  {'score':>7}  {'ext':>3}  {'exp':>3}  {'miss':>4}  notes",
        file=out,
    )
    print("  " + "-" * (col + 34), file=out)
    for t in red_turns:
        ms = t["minimax_score"]
        score_str = f"{ms:7.2f}" if isinstance(ms, (int, float)) else "    n/a"
        n_ext = len(t["extracted_claim_types"])
        n_exp = len(t["expected_claim_types"])
        n_miss = len(t["missing_types"])
        notes_str = "; ".join(t["notes"]) if t["notes"] else ""
        tid = t["turn_id"]
        print(
            f"  {tid:{col}}  {score_str}  {n_ext:3d}  {n_exp:3d}  {n_miss:4d}  {notes_str}",
            file=out,
        )
    print(file=out)


# ---------------------------------------------------------------------------
# Log loader
# ---------------------------------------------------------------------------

def run_audit(log_paths: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Load RED-player turns from log files, compute per-turn recall, and aggregate.

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
                    # Only RED turns (LangGraph pipeline turns).
                    if record.get("player") != "RED":
                        continue
                    # Skip turns with no reasoning text.
                    if not record.get("reasoning"):
                        continue
                    result = compute_turn_recall(record)
                    result["_source_file"] = Path(path).name
                    turn_results.append(result)
        except OSError as exc:
            print(f"WARNING: cannot open {path!r}: {exc}", file=sys.stderr)

    agg = aggregate_recall_results(turn_results)
    return agg, turn_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4.2: claim recall audit on manual_eval JSONL logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m checkers.evaluation.run_claim_recall_audit \\\n"
            "      --log logs/manual_eval/manual_game_*.jsonl\n"
            "  python -m checkers.evaluation.run_claim_recall_audit \\\n"
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

    # Expand glob patterns supplied on the command line.
    log_paths: list[str] = []
    for pattern in args.log:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            log_paths.extend(expanded)
        else:
            log_paths.append(pattern)  # treat as literal path; error surfaces at open()

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
