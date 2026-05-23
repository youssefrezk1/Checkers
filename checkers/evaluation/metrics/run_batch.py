# checkers/evaluation/metrics/run_batch.py
#
# Batch runner that consumes one or more evaluation_source/*.jsonl files
# and prints aggregate metrics for the three first-layer evaluator modules:
#
#   factuality  → pre/post repair
#   grounding   → zero-claim / filler
#   diversity   → Self-BLEU 2/3/4
#
# Usage:
#   python -m checkers.evaluation.metrics.run_batch  PATH [PATH ...] [--out FILE]
#
# - PATH may be a single .jsonl file or a directory (every *.jsonl in it
#   is included).
# - --out writes the aggregate JSON to disk; without it the JSON is
#   printed to stdout.
#
# Deterministic; no LLM calls; safe to run repeatedly on the same files.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from checkers.evaluation.metrics.pre_post_repair import (
    evaluate_pre_post_repair,
    aggregate_pre_post_repair,
)
from checkers.evaluation.metrics.zero_claim import (
    evaluate_zero_claim,
    aggregate_zero_claim,
)
from checkers.evaluation.metrics.self_bleu import compute_self_bleu
from checkers.evaluation.metrics.by_claim_type import aggregate_by_claim_type
from checkers.evaluation.metrics.by_source import aggregate_by_source
from checkers.evaluation.metrics.compare import compare_summaries


def _iter_records(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        with p.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError as e:
                    print(
                        f"[run_batch] skipping malformed line {lineno} of {p}: {e}",
                        file=sys.stderr,
                    )


def _expand_paths(inputs: List[str]) -> List[Path]:
    out: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(p.glob("*.jsonl")))
        elif p.is_file():
            out.append(p)
        else:
            print(f"[run_batch] warning: {p} not found", file=sys.stderr)
    return out


def evaluate_batch(paths: List[Path]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = list(_iter_records(paths))

    pre_post_turns = [evaluate_pre_post_repair(r) for r in records]
    zero_claim_turns = [evaluate_zero_claim(r) for r in records]
    reasonings = [
        r.get("last_move_reasoning") for r in records
        if isinstance(r.get("last_move_reasoning"), str)
    ]

    pre_post_summary    = aggregate_pre_post_repair(pre_post_turns)
    zero_claim_summary  = aggregate_zero_claim(zero_claim_turns)
    diversity_summary   = compute_self_bleu(reasonings, keep_per_hyp=False)
    claim_type_summary  = aggregate_by_claim_type(records)
    claim_source_summary = aggregate_by_source(records)

    return {
        "n_files":   len(paths),
        "n_records": len(records),
        "factuality":      pre_post_summary.to_dict(),
        "grounding":       zero_claim_summary.to_dict(),
        "diversity":       diversity_summary.to_dict(),
        "by_claim_type":   claim_type_summary.to_dict(),
        "by_claim_source": claim_source_summary.to_dict(),
    }


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*",
        help="JSONL file(s) or directory of .jsonl files (single-condition mode)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Write aggregate JSON to this path instead of stdout",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("DIR_A", "DIR_B"), default=None,
        help=(
            "Comparative mode: run the batch evaluator on DIR_A and DIR_B "
            "(typically the seed_on / seed_off directories produced by "
            "run_ablation.py) and emit a paired delta report. "
            "Use --label-a / --label-b to override the top-level keys."
        ),
    )
    parser.add_argument(
        "--label-a", default="seed_on",
        help="Label for the first compared bundle (default: seed_on)",
    )
    parser.add_argument(
        "--label-b", default="seed_off",
        help="Label for the second compared bundle (default: seed_off)",
    )
    args = parser.parse_args(argv)

    # ── Comparative mode ─────────────────────────────────────────────────────
    if args.compare is not None:
        paths_a = _expand_paths([args.compare[0]])
        paths_b = _expand_paths([args.compare[1]])
        if not paths_a or not paths_b:
            print(
                f"[run_batch] --compare requires both sides to resolve to files; "
                f"got {len(paths_a)} / {len(paths_b)}",
                file=sys.stderr,
            )
            return 1
        agg_a = evaluate_batch(paths_a)
        agg_b = evaluate_batch(paths_b)
        report = compare_summaries(
            agg_a, agg_b, label_a=args.label_a, label_b=args.label_b,
        )
        blob = json.dumps(report, indent=2, ensure_ascii=False)
        if args.out:
            Path(args.out).write_text(blob, encoding="utf-8")
            print(f"[run_batch] wrote comparative report → {args.out}")
        else:
            print(blob)
        return 0

    # ── Single-condition mode ────────────────────────────────────────────────
    if not args.paths:
        parser.error("provide either positional PATHS or --compare DIR_A DIR_B")
    paths = _expand_paths(args.paths)
    if not paths:
        print("[run_batch] no input files resolved; nothing to do", file=sys.stderr)
        return 1

    aggregate = evaluate_batch(paths)
    blob = json.dumps(aggregate, indent=2, ensure_ascii=False)

    if args.out:
        Path(args.out).write_text(blob, encoding="utf-8")
        print(f"[run_batch] wrote aggregate metrics → {args.out}")
    else:
        print(blob)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
