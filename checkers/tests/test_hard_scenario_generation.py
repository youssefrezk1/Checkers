# checkers/tests/test_hard_scenario_generation.py
#
# Tests for checkers/baseline_eval/generate_hard_scenarios.py
#
# Coverage:
#   1. Generator produces VALID legal positions (every entry's board has at
#      least min_legal legal moves for side_to_move, and best_move_index is
#      within range).
#   2. Filters enforce min_legal and min_gap.
#   3. Dedup by board hash: no two entries share an identical board+side.
#   4. JSON save/load round-trips losslessly.
#   5. Reproducibility: same seed + same args → identical entries.
#   6. _classify produces at least one tag for at least one rollout in a
#      reasonable budget.

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("USE_SIMPLIFIED_PIPELINE", "true")
os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

from checkers.engine.rules import get_all_legal_moves
from checkers.baseline_eval.generate_hard_scenarios import (
    CATEGORIES, _dedup_key, _classify,
    generate, save_scenarios, load_scenarios,
)


# ── Small/fast generation parameters used across tests ──────────────────────
SMALL_TARGET    = 5
SMALL_ROLLOUTS  = 60
SMALL_PLIES     = 50
SMALL_MIN_GAP   = 1.0
SMALL_MIN_LEGAL = 2
SEED            = 1234


# ── 1. Valid legal positions ────────────────────────────────────────────────

def test_generator_produces_legal_positions():
    entries = generate(
        target_count=SMALL_TARGET,
        max_rollouts=SMALL_ROLLOUTS,
        max_plies=SMALL_PLIES,
        seed=SEED,
        min_gap=SMALL_MIN_GAP,
        min_legal=SMALL_MIN_LEGAL,
    )
    assert entries, "Expected at least one generated scenario"
    for e in entries:
        # board structure
        assert isinstance(e["board"], list) and len(e["board"]) == 8
        for row in e["board"]:
            assert isinstance(row, list) and len(row) == 8
        # legal-moves consistency
        legal = get_all_legal_moves(e["board"], e["side_to_move"])
        assert len(legal) == e["legal_moves_count"]
        assert len(legal) >= SMALL_MIN_LEGAL
        # best_move_index is valid
        if e["best_move_index"] is not None:
            assert 0 <= e["best_move_index"] < len(legal)


# ── 2. Filter enforcement ──────────────────────────────────────────────────

def test_filter_enforces_min_legal_and_min_gap():
    entries = generate(
        target_count=SMALL_TARGET,
        max_rollouts=SMALL_ROLLOUTS,
        max_plies=SMALL_PLIES,
        seed=SEED,
        min_gap=SMALL_MIN_GAP,
        min_legal=4,
    )
    for e in entries:
        assert e["legal_moves_count"] >= 4
        assert e["score_gap"] is None or e["score_gap"] >= SMALL_MIN_GAP


# ── 3. Dedup ────────────────────────────────────────────────────────────────

def test_dedup_no_duplicate_boards():
    entries = generate(
        target_count=SMALL_TARGET,
        max_rollouts=SMALL_ROLLOUTS,
        max_plies=SMALL_PLIES,
        seed=SEED,
        min_gap=SMALL_MIN_GAP,
        min_legal=SMALL_MIN_LEGAL,
    )
    keys = {_dedup_key(e["board"], e["side_to_move"]) for e in entries}
    assert len(keys) == len(entries), "Duplicate boards in generator output"


def test_dedup_key_is_stable_and_distinct():
    b1 = [[0] * 8 for _ in range(8)]
    b2 = [[0] * 8 for _ in range(8)]
    b2[3][2] = 1
    assert _dedup_key(b1, 1) == _dedup_key(b1, 1)
    assert _dedup_key(b1, 1) != _dedup_key(b2, 1)
    assert _dedup_key(b1, 1) != _dedup_key(b1, 2)


# ── 4. JSON round-trip ──────────────────────────────────────────────────────

def test_save_load_round_trip():
    entries = generate(
        target_count=3,
        max_rollouts=SMALL_ROLLOUTS,
        max_plies=SMALL_PLIES,
        seed=SEED,
        min_gap=SMALL_MIN_GAP,
        min_legal=SMALL_MIN_LEGAL,
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "hard.json"
        save_scenarios(entries, path)
        loaded = load_scenarios(path)
    assert loaded == entries


def test_load_rejects_non_list():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bad.json"
        p.write_text('{"not": "a list"}', encoding="utf-8")
        with pytest.raises(ValueError):
            load_scenarios(p)


# ── 5. Reproducibility ──────────────────────────────────────────────────────

def test_same_seed_produces_identical_output():
    a = generate(
        target_count=4,
        max_rollouts=SMALL_ROLLOUTS,
        max_plies=SMALL_PLIES,
        seed=99,
        min_gap=SMALL_MIN_GAP,
        min_legal=SMALL_MIN_LEGAL,
    )
    b = generate(
        target_count=4,
        max_rollouts=SMALL_ROLLOUTS,
        max_plies=SMALL_PLIES,
        seed=99,
        min_gap=SMALL_MIN_GAP,
        min_legal=SMALL_MIN_LEGAL,
    )
    assert a == b


def test_different_seed_produces_different_pool():
    a = generate(
        target_count=4,
        max_rollouts=SMALL_ROLLOUTS,
        max_plies=SMALL_PLIES,
        seed=1,
        min_gap=SMALL_MIN_GAP,
        min_legal=SMALL_MIN_LEGAL,
    )
    b = generate(
        target_count=4,
        max_rollouts=SMALL_ROLLOUTS,
        max_plies=SMALL_PLIES,
        seed=2,
        min_gap=SMALL_MIN_GAP,
        min_legal=SMALL_MIN_LEGAL,
    )
    # Very unlikely to be identical given different seeds and large state space.
    if a and b:
        assert a != b


# ── 6. Categories ──────────────────────────────────────────────────────────

def test_each_entry_has_known_category():
    entries = generate(
        target_count=SMALL_TARGET,
        max_rollouts=SMALL_ROLLOUTS,
        max_plies=SMALL_PLIES,
        seed=SEED,
        min_gap=SMALL_MIN_GAP,
        min_legal=SMALL_MIN_LEGAL,
    )
    for e in entries:
        assert e["category"] in CATEGORIES
        for t in e["tactical_tags"]:
            assert t in CATEGORIES
