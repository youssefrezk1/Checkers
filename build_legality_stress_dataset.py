#!/usr/bin/env python3
"""
build_legality_stress_dataset.py
=================================
Driver script for the PDN → legality-stress JSONL pipeline.
Auto-discovers ALL .pdn files in the raw_pdn/bob_newell/ directory.

Usage (from project root, venv active):
    python build_legality_stress_dataset.py

Outputs:
    checkers/data/legality_stress/scenarios.jsonl
    checkers/data/legality_stress/hard_subset_balanced.jsonl
    checkers/data/legality_stress/report.txt
"""

import os
import sys
import json
import random
import logging
import textwrap
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from checkers.data.pdn_importer.pdn_parser import parse_pdn_file
from checkers.data.pdn_importer.scenario_generator import (
    generate_scenarios, export_jsonl, category_counts, CATEGORIES
)

PREVIOUS_HARD_COUNT = 89   # baseline from the previous 3-file run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("build_dataset")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PDN_DIR       = os.path.join(PROJECT_ROOT, "checkers", "data", "raw_pdn", "bob_newell")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "checkers", "data", "legality_stress")
OUTPUT_JSONL  = os.path.join(OUTPUT_DIR, "scenarios.jsonl")
OUTPUT_HARD   = os.path.join(OUTPUT_DIR, "hard_subset_balanced.jsonl")
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "report.txt")

MAX_HARD_PER_CAT = 30   # cap per category in the balanced hard subset
HARD_SEED        = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_pdn_files(directory: str) -> list:
    """Return sorted list of all .pdn files in directory."""
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.lower().endswith(".pdn")
    ]


PIECE_SYMBOLS = {0: ".", 1: "r", 2: "b", 3: "R", 4: "B"}

def _board_str(board) -> str:
    lines = ["    0 1 2 3 4 5 6 7"]
    for r, row in enumerate(board):
        lines.append(f"  {r} " + " ".join(PIECE_SYMBOLS[c] for c in row))
    return "\n".join(lines)


def _move_str(mv: dict) -> str:
    path = "→".join(f"({r},{c})" for r, c in mv["path"])
    if mv["captured"]:
        caps = ", ".join(f"({r},{c})" for r, c in mv["captured"])
        return f"[{mv['type'].upper()}] {path}  captures={caps}"
    return f"[{mv['type'].upper()}] {path}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== PDN → Legality-Stress Dataset Builder ===")

    # 1. Auto-discover all .pdn files
    pdn_files = discover_pdn_files(PDN_DIR)
    log.info("Discovered %d PDN files in %s", len(pdn_files), PDN_DIR)
    for f in pdn_files:
        log.info("  %s", os.path.basename(f))

    # 2. Parse all files — track per-file raw position counts
    all_positions = []
    per_file_raw  = {}

    for fpath in pdn_files:
        basename = os.path.basename(fpath)
        log.info("Parsing %s …", basename)
        positions = parse_pdn_file(fpath)
        per_file_raw[basename] = len(positions)
        log.info("  → %d raw positions extracted", len(positions))
        all_positions.extend(positions)

    log.info("Total raw positions (before dedup+filter): %d", len(all_positions))

    # 3. Classify + deduplicate → scenarios
    log.info("Classifying and deduplicating …")
    scenarios = generate_scenarios(all_positions)
    log.info("Total scenarios after dedup/filter: %d", len(scenarios))

    if not scenarios:
        log.error("No scenarios generated — check PDN files and parser.")
        sys.exit(1)

    # 4. Export full JSONL
    export_jsonl(scenarios, OUTPUT_JSONL)
    log.info("Exported full dataset → %s", OUTPUT_JSONL)

    # 5. Export balanced hard subset  (cap MAX_HARD_PER_CAT per category)
    hard_scenarios = [sc for sc in scenarios if sc["difficulty"] == "hard"]
    hard_by_cat: dict = {}
    for sc in hard_scenarios:
        hard_by_cat.setdefault(sc["category"], []).append(sc)

    rng = random.Random(HARD_SEED)
    hard_subset        = []
    hard_subset_counts = {}
    for cat, pool in sorted(hard_by_cat.items()):
        rng.shuffle(pool)
        selected = pool[:MAX_HARD_PER_CAT]
        hard_subset.extend(selected)
        hard_subset_counts[cat] = len(selected)
    rng.shuffle(hard_subset)

    export_jsonl(hard_subset, OUTPUT_HARD)
    log.info("Exported balanced hard subset (%d) → %s", len(hard_subset), OUTPUT_HARD)

    # 6. Symbolic validation
    log.info("Running symbolic validation on all %d scenarios …", len(scenarios))
    from checkers.engine.rules import get_all_legal_moves
    from checkers.data.pdn_importer.fen_utils import str_to_side

    def _path_key(mv):
        return (mv["type"], tuple(tuple(p) for p in mv["path"]))

    all_errors = []
    for idx, sc in enumerate(scenarios):
        board  = sc["board"]
        side   = str_to_side(sc["side_to_move"])
        stored = sc["hidden_legal_moves"]
        fresh  = get_all_legal_moves(board, side)
        if len(stored) != len(fresh):
            all_errors.append(
                f"[{idx}] {sc['scenario_id']}: stored {len(stored)}, fresh {len(fresh)}"
            )
            continue
        fresh_keys = {_path_key(m) for m in fresh}
        for mv in stored:
            if _path_key(mv) not in fresh_keys:
                all_errors.append(
                    f"[{idx}] {sc['scenario_id']}: move {mv['path']} not in fresh"
                )

    if all_errors:
        log.error("VALIDATION FAILED — %d errors:", len(all_errors))
        for e in all_errors[:20]:
            log.error("  %s", e)
    else:
        log.info("✓ Symbolic validation PASSED — all hidden_legal_moves verified.")

    # 7. Build report
    counts      = category_counts(scenarios)
    diff_counts = Counter(sc["difficulty"] for sc in scenarios)
    new_hard    = diff_counts["hard"]
    delta       = new_hard - PREVIOUS_HARD_COUNT
    src_counts  = Counter(sc["source_file"] for sc in scenarios)

    L = []   # report lines

    L += [
        "=" * 64,
        "LEGALITY-STRESS DATASET REPORT  (full 8-file run)",
        "=" * 64,
        f"PDN directory      : {PDN_DIR}",
        f"Files discovered   : {len(pdn_files)}",
        f"Total scenarios    : {len(scenarios)}",
        f"Validation errors  : {len(all_errors)}",
        "",
        "--- Files discovered ---",
    ]
    for f in pdn_files:
        L.append(f"  {os.path.basename(f)}")

    L += ["", "--- Per-file raw positions extracted ---"]
    for basename, cnt in sorted(per_file_raw.items()):
        L.append(f"  {basename:<46} {cnt:>6} raw positions")

    L += ["", "--- Per-file scenarios in final dataset ---"]
    for src, cnt in sorted(src_counts.items()):
        L.append(f"  {src:<46} {cnt:>6} scenarios")

    L += ["", "--- Category counts ---"]
    for cat in CATEGORIES:
        L.append(f"  {cat:<35} {counts.get(cat, 0):>6}")

    L += [
        "",
        "--- Difficulty counts ---",
        f"  {'easy':<35} {diff_counts['easy']:>6}",
        f"  {'medium':<35} {diff_counts['medium']:>6}",
        f"  {'hard':<35} {new_hard:>6}",
        "",
        "--- Hard count comparison ---",
        f"  Previous (3-file run): {PREVIOUS_HARD_COUNT}",
        f"  Current  (8-file run): {new_hard}",
        f"  Delta                : {'+' if delta >= 0 else ''}{delta}",
        "",
        "--- Balanced hard subset (hard_subset_balanced.jsonl) ---",
        f"  Total hard scenarios  : {len(hard_scenarios)}",
        f"  Subset exported       : {len(hard_subset)}  (cap {MAX_HARD_PER_CAT}/category)",
    ]
    for cat, cnt in sorted(hard_subset_counts.items()):
        L.append(f"    {cat:<33} {cnt:>4}")

    L += ["", "--- 10 Sample Hard Scenarios ---",
          "    Board shown as-prompt; GROUND TRUTH printed separately below each board.", ""]

    sample_hard = hard_subset[:10]
    for i, sc in enumerate(sample_hard, 1):
        L.append(f"  [{i:02d}] {sc['scenario_id']}")
        L.append(f"       category   : {sc['category']}")
        L.append(f"       difficulty : {sc['difficulty']}")
        L.append(f"       side       : {sc['side_to_move']}")
        L.append(f"       source     : {sc['source_file']}"
                 f"  game={sc['game_index']} ply={sc['ply_index']}")
        L.append(f"       rule       : {textwrap.shorten(sc['expected_rule'], 72)}")
        L.append("")
        L.append("       Board (r=red man  R=red king  b=black man  B=black king):")
        for bl in _board_str(sc["board"]).splitlines():
            L.append("       " + bl)
        L.append("")
        n = len(sc["hidden_legal_moves"])
        L.append(f"       GROUND TRUTH — hidden_legal_moves ({n}) — NOT shown to LLM:")
        for mv in sc["hidden_legal_moves"]:
            L.append(f"         {_move_str(mv)}")
        L.append("")

    report_text = "\n".join(L)
    print(report_text)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write(report_text)
    log.info("Report saved → %s", OUTPUT_REPORT)

    if all_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
