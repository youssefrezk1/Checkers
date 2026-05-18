#!/usr/bin/env python3
"""
build_eval_subset.py
====================
Builds checkers/data/legality_stress/eval_subset_balanced.jsonl
and checkers/data/legality_stress/eval_subset_report.txt

Usage (from project root, venv active):
    python build_eval_subset.py

Does NOT create or run any LLM baseline.
"""

import os
import sys
import logging

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from checkers.data.pdn_importer.subset_sampler import (
    load_scenarios, sample_balanced_subset,
    export_subset, build_report, TARGET_CATEGORIES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("build_eval_subset")

LEGALITY_DIR   = os.path.join(PROJECT_ROOT, "checkers", "data", "legality_stress")
INPUT_JSONL    = os.path.join(LEGALITY_DIR, "scenarios.jsonl")
OUTPUT_JSONL   = os.path.join(LEGALITY_DIR, "eval_subset_balanced.jsonl")
OUTPUT_REPORT  = os.path.join(LEGALITY_DIR, "eval_subset_report.txt")


def main():
    if not os.path.exists(INPUT_JSONL):
        log.error("Input not found: %s  (run build_legality_stress_dataset.py first)", INPUT_JSONL)
        sys.exit(1)

    log.info("Loading scenarios from %s …", INPUT_JSONL)
    scenarios = load_scenarios(INPUT_JSONL)
    log.info("Loaded %d total scenarios.", len(scenarios))

    log.info("Sampling balanced subset (up to 50 per category) …")
    result = sample_balanced_subset(scenarios)
    subset  = result["subset"]
    counts  = result["counts"]
    skipped = result["skipped"]

    log.info("Selected %d scenarios across %d categories.", len(subset), len(TARGET_CATEGORIES))

    export_subset(subset, OUTPUT_JSONL)
    log.info("Exported → %s", OUTPUT_JSONL)

    report = build_report(counts, skipped, subset, OUTPUT_REPORT, INPUT_JSONL)
    print(report)
    log.info("Report  → %s", OUTPUT_REPORT)


if __name__ == "__main__":
    main()
