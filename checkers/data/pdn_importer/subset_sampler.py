# checkers/data/pdn_importer/subset_sampler.py
"""
Balanced subset sampler for the legality-stress evaluation dataset.

Reads scenarios.jsonl and samples up to MAX_PER_CATEGORY scenarios per
targeted category, with deterministic shuffling for reproducibility.

Targeted categories (included):
    mandatory_capture, multi_jump_required, king_vs_man_confusion,
    promotion_state_update, wrong_direction_trap, crowded_board

Excluded (too few samples to be meaningful right now):
    occupied_destination_trap, wrong_player_piece_trap, state_update_after_capture

The sampled scenarios retain ALL fields including hidden_legal_moves
(for evaluator use only — prompt-generation strips them before any LLM sees them).
"""

import json
import random
import os
from typing import Optional

# Categories to include in the balanced eval subset
TARGET_CATEGORIES = [
    "mandatory_capture",
    "multi_jump_required",
    "king_vs_man_confusion",
    "promotion_state_update",
    "wrong_direction_trap",
    "crowded_board",
]

EXCLUDED_CATEGORIES = [
    "occupied_destination_trap",
    "wrong_player_piece_trap",
    "state_update_after_capture",
]

MAX_PER_CATEGORY = 50
DEFAULT_SEED = 42


def load_scenarios(jsonl_path: str) -> list:
    """Load all scenarios from a JSONL file."""
    scenarios = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))
    return scenarios


def sample_balanced_subset(
    scenarios: list,
    max_per_category: int = MAX_PER_CATEGORY,
    seed: int = DEFAULT_SEED,
    target_categories: Optional[list] = None,
) -> dict:
    """
    Sample a balanced subset from scenarios.

    Returns:
        {
          "subset":   list of selected scenario dicts,
          "counts":   dict {category: count},
          "skipped":  dict {category: total_available} for excluded categories,
        }
    """
    if target_categories is None:
        target_categories = TARGET_CATEGORIES

    rng = random.Random(seed)

    # Bucket by category
    buckets: dict[str, list] = {cat: [] for cat in target_categories}
    skipped: dict[str, int] = {}

    for sc in scenarios:
        cat = sc["category"]
        if cat in buckets:
            buckets[cat].append(sc)
        else:
            skipped[cat] = skipped.get(cat, 0) + 1

    subset = []
    counts = {}

    for cat in target_categories:
        pool = buckets[cat]
        rng.shuffle(pool)
        selected = pool[:max_per_category]
        subset.extend(selected)
        counts[cat] = len(selected)

    # Shuffle the final subset so categories are interleaved
    rng.shuffle(subset)

    return {"subset": subset, "counts": counts, "skipped": skipped}


def export_subset(subset: list, output_path: str) -> None:
    """Write subset to JSONL. All fields including hidden_legal_moves are kept."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sc in subset:
            f.write(json.dumps(sc) + "\n")


def build_report(
    counts: dict,
    skipped: dict,
    subset: list,
    output_path: str,
    jsonl_path: str,
) -> str:
    """Build and return a human-readable report string; also writes it to output_path."""
    total = sum(counts.values())
    lines = [
        "=" * 60,
        "LEGALITY-STRESS EVAL SUBSET REPORT",
        "=" * 60,
        f"Source:            {jsonl_path}",
        f"Total selected:    {total}",
        f"Max per category:  {MAX_PER_CATEGORY}",
        "",
        "--- Included category counts ---",
    ]
    for cat in TARGET_CATEGORIES:
        lines.append(f"  {cat:<35} {counts.get(cat, 0):>5}")

    lines += ["", "--- Excluded categories (not sampled) ---"]
    for cat in EXCLUDED_CATEGORIES:
        n = skipped.get(cat, 0)
        lines.append(f"  {cat:<35} {n:>5} available (excluded)")

    # Difficulty breakdown within subset
    diff = {"easy": 0, "medium": 0, "hard": 0}
    for sc in subset:
        diff[sc["difficulty"]] = diff.get(sc["difficulty"], 0) + 1

    lines += [
        "",
        "--- Difficulty breakdown ---",
        f"  {'easy':<35} {diff['easy']:>5}",
        f"  {'medium':<35} {diff['medium']:>5}",
        f"  {'hard':<35} {diff['hard']:>5}",
        "",
        "--- Source file breakdown ---",
    ]
    from collections import Counter
    src_counts = Counter(sc["source_file"] for sc in subset)
    for src, cnt in sorted(src_counts.items()):
        lines.append(f"  {src:<42} {cnt:>5}")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report
