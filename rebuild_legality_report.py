#!/usr/bin/env python3
"""
rebuild_legality_report.py
==========================
Re-aggregate existing results JSONL file(s) using the CURRENT metrics.py
and write a new report TXT next to each file.

No API calls are made. The original JSONL is never modified.

Usage
-----
    python rebuild_legality_report.py logs/legality_pilot/results_<ts>.jsonl
    python rebuild_legality_report.py logs/legality_pilot/results_*.jsonl
    python rebuild_legality_report.py logs/legality_pilot/results_*.jsonl --show-side-summary

Algorithm
---------
  1. Load all records from the JSONL.
  2. Group records by baseline name.
  3. Call aggregate() for each baseline group.
  4. Call format_report() with all baseline groups combined.
  5. Write report to <same-dir>/report_rebuilt_<ts>.txt
     (does NOT overwrite the original report_<ts>.txt).

Notes
-----
  - Records with result_type='?' (pre-result_type schema) are treated as
    'api_failure' so they are excluded from legal/illegal denominators.
  - Records with no side_to_move field default to 'UNKNOWN' in side_metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checkers.data.legality_eval.metrics import aggregate, format_report


# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalise_record(rec: dict) -> dict:
    """
    Back-compat: old runs stored result_type='?' before the field was defined.
    Treat those as api_failure so they are excluded from eval denominators.
    """
    r = dict(rec)
    if r.get("result_type", "?") == "?":
        r["result_type"]  = "api_failure"
        r["api_success"]  = False
        r.setdefault("rate_limit_retry_count", 0)
        r.setdefault("parse_success", False)
        r.setdefault("legal", False)
        r.setdefault("illegal_move_type", "")
        r.setdefault("wrong_direction", None)
        r.setdefault("mandatory_violation", False)
        r.setdefault("multi_jump_incomplete", False)
        r.setdefault("category", "unknown")
        r.setdefault("difficulty", "unknown")
    return r


def rebuild_one(path: Path, show_side_summary: bool) -> None:
    print(f"\n{'='*70}")
    print(f"Rebuilding: {path.name}")
    recs = [normalise_record(r) for r in load_jsonl(path)]
    if not recs:
        print("  [SKIP] empty file")
        return

    # Group by baseline
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        groups[r.get("baseline", "unknown")].append(r)

    baselines = sorted(groups.keys())
    n_scenarios = max(len(g) for g in groups.values()) if groups else 0

    print(f"  Records : {len(recs)}")
    print(f"  Baselines: {baselines}")

    baseline_metrics: dict[str, dict] = {}
    for bname in baselines:
        baseline_metrics[bname] = aggregate(groups[bname])

    any_api_failure = any(
        m.get("total_api_failures", 0) > 0
        for m in baseline_metrics.values()
    )
    run_label = "INCOMPLETE_FOR_FINAL_EVALUATION" if any_api_failure else "API_COMPLETE"

    # Derive a pilot name from the original filename
    pilot_name = path.stem.replace("results_", "pilot_") + " [REBUILT]"

    report = format_report(
        pilot_name=pilot_name,
        baseline_metrics=baseline_metrics,
        n_scenarios=n_scenarios,
        run_label=run_label,
    )

    # Write rebuilt report next to the original (never overwrites original report)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = path.stem.replace("results_", "")
    out    = path.parent / f"report_rebuilt_{suffix}.txt"
    out.write_text(report, encoding="utf-8")
    print(f"  Saved  → {out}")
    print(f"  Run label: {run_label}")

    # Optional side summary to stdout
    if show_side_summary:
        print(f"\n  ── Side-to-Move Summary ──")
        for bname in baselines:
            sm = baseline_metrics[bname].get("side_metrics", {})
            print(f"  [{bname}]")
            for side in ["RED", "BLACK"]:
                sd = sm.get(side, {})
                n_ev  = sd.get("n_evaluated", 0)
                lmr   = sd.get("legal_move_rate", "n/a")
                wdr   = sd.get("wrong_direction_rate", "n/a")
                mviol = sd.get("mandatory_capture_viol_rate", "n/a")
                print(
                    f"    {side:6s}  n_eval={n_ev:3d}  "
                    f"legal={lmr}  wrong_dir={wdr}  mand_viol={mviol}"
                )

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rebuild legality pilot reports from existing JSONL files (no API calls)."
    )
    p.add_argument(
        "files", nargs="+", type=Path,
        help="One or more results_*.jsonl files to rebuild reports for.",
    )
    p.add_argument(
        "--show-side-summary", action="store_true",
        help="Print a side-level summary table to stdout for each file.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(set(args.files))
    missing = [p for p in paths if not p.exists()]
    if missing:
        for m in missing:
            print(f"[ERROR] File not found: {m}", file=sys.stderr)
        sys.exit(1)

    for path in paths:
        rebuild_one(path, show_side_summary=args.show_side_summary)

    print(f"Done. {len(paths)} file(s) processed.")


if __name__ == "__main__":
    main()
