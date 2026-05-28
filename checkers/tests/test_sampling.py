# checkers/tests/test_sampling.py
#
# Regression tests for --sample-size deterministic sampling.
# Fully offline — no LLM, no DLL, no network.

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ── mirror of the sampling logic from main() ─────────────────────────────────

_SEED = 42

def _apply_sampling(dataset: list, sample_size: int | None) -> tuple[list, dict]:
    """Mirror of the sampling block in main()."""
    before = len(dataset)
    active = sample_size is not None and sample_size < before
    if active:
        rng = random.Random(_SEED)
        dataset = rng.sample(dataset, sample_size)
    after = len(dataset)
    meta = {
        "active": active,
        "sample_size": sample_size,
        "positions_before_sampling": before,
        "positions_after_sampling":  after,
    }
    return dataset, meta


def _apply_limit(dataset: list, limit: int | None) -> list:
    if limit is not None:
        return dataset[:limit]
    return dataset


def _make_dataset(n: int) -> list[dict]:
    return [{"scenario_id": f"s{i}", "hidden_legal_moves": [{}] * (i % 7 + 1)} for i in range(n)]


# ════════════════════════════════════════════════════════════════════════════════
# Deterministic sampling
# ════════════════════════════════════════════════════════════════════════════════

class TestDeterministicSampling:

    def test_same_sample_every_run(self):
        ds = _make_dataset(100)
        result1, _ = _apply_sampling(ds, 20)
        result2, _ = _apply_sampling(ds, 20)
        ids1 = [r["scenario_id"] for r in result1]
        ids2 = [r["scenario_id"] for r in result2]
        assert ids1 == ids2

    def test_sample_size_correct(self):
        ds = _make_dataset(100)
        result, _ = _apply_sampling(ds, 30)
        assert len(result) == 30

    def test_sample_is_subset_of_original(self):
        ds = _make_dataset(50)
        result, _ = _apply_sampling(ds, 15)
        original_ids = {e["scenario_id"] for e in ds}
        for r in result:
            assert r["scenario_id"] in original_ids

    def test_no_duplicates_in_sample(self):
        ds = _make_dataset(80)
        result, _ = _apply_sampling(ds, 40)
        ids = [r["scenario_id"] for r in result]
        assert len(ids) == len(set(ids))

    def test_different_n_gives_different_sample(self):
        ds = _make_dataset(100)
        r10, _ = _apply_sampling(ds, 10)
        r20, _ = _apply_sampling(ds, 20)
        # The 10-sample should be a different subset (not necessarily a prefix of 20)
        ids10 = set(r["scenario_id"] for r in r10)
        ids20 = set(r["scenario_id"] for r in r20)
        # They won't be identical sets (with seed=42 and n=100 this is guaranteed)
        assert ids10 != ids20


# ════════════════════════════════════════════════════════════════════════════════
# sample_size >= dataset size → no-op
# ════════════════════════════════════════════════════════════════════════════════

class TestSampleSizeNoOp:

    def test_sample_equals_size_is_noop(self):
        ds = _make_dataset(20)
        result, meta = _apply_sampling(ds, 20)
        assert len(result) == 20
        assert meta["active"] is False

    def test_sample_greater_than_size_is_noop(self):
        ds = _make_dataset(10)
        result, meta = _apply_sampling(ds, 999)
        assert len(result) == 10
        assert meta["active"] is False

    def test_noop_preserves_original_order(self):
        ds = _make_dataset(15)
        result, meta = _apply_sampling(ds, 100)
        assert [r["scenario_id"] for r in result] == [e["scenario_id"] for e in ds]
        assert meta["active"] is False

    def test_no_sample_size_is_noop(self):
        ds = _make_dataset(50)
        result, meta = _apply_sampling(ds, None)
        assert len(result) == 50
        assert meta["active"] is False
        assert meta["sample_size"] is None


# ════════════════════════════════════════════════════════════════════════════════
# Sampling after filtering
# ════════════════════════════════════════════════════════════════════════════════

class TestSamplingAfterFiltering:
    """Sampling is applied to the already-filtered pool, not the original dataset."""

    def _bf_filter(self, dataset, min_lm=None, max_lm=None):
        return [
            e for e in dataset
            if (
                (min_lm is None or len(e.get("hidden_legal_moves", [])) >= min_lm)
                and (max_lm is None or len(e.get("hidden_legal_moves", [])) <= max_lm)
            )
        ]

    def test_sample_pool_is_post_filter(self):
        ds = _make_dataset(100)
        filtered = self._bf_filter(ds, min_lm=3)
        sampled, meta = _apply_sampling(filtered, 5)
        # All sampled entries must satisfy the filter criterion
        for e in sampled:
            assert len(e.get("hidden_legal_moves", [])) >= 3
        assert meta["positions_before_sampling"] == len(filtered)

    def test_filter_then_sample_size_correct(self):
        ds = _make_dataset(100)
        filtered = self._bf_filter(ds, max_lm=2)
        n_filtered = len(filtered)
        if n_filtered >= 5:
            sampled, meta = _apply_sampling(filtered, 5)
            assert len(sampled) == 5
            assert meta["active"] is True

    def test_filter_empties_pool_sample_is_noop(self):
        ds = _make_dataset(5)
        filtered = self._bf_filter(ds, min_lm=999)   # impossible criterion
        assert len(filtered) == 0
        sampled, meta = _apply_sampling(filtered, 3)
        assert len(sampled) == 0
        assert meta["active"] is False   # 3 >= 0, so no-op


# ════════════════════════════════════════════════════════════════════════════════
# Order: filtering → sampling → limit
# ════════════════════════════════════════════════════════════════════════════════

class TestOrderFilterSampleLimit:

    def test_limit_applied_after_sample(self):
        ds = _make_dataset(100)
        sampled, _ = _apply_sampling(ds, 30)
        limited = _apply_limit(sampled, 10)
        assert len(limited) == 10
        # All limited entries must come from the sampled set
        sampled_ids = {e["scenario_id"] for e in sampled}
        for e in limited:
            assert e["scenario_id"] in sampled_ids

    def test_limit_larger_than_sample_is_noop(self):
        ds = _make_dataset(50)
        sampled, _ = _apply_sampling(ds, 20)
        limited = _apply_limit(sampled, 100)
        assert len(limited) == 20

    def test_no_limit_no_change(self):
        ds = _make_dataset(50)
        sampled, _ = _apply_sampling(ds, 20)
        limited = _apply_limit(sampled, None)
        assert len(limited) == 20

    def test_full_pipeline_correct_count(self):
        """filter(max=3) → sample(10) → limit(5) must give exactly 5."""
        ds = _make_dataset(200)
        filtered = [e for e in ds if len(e.get("hidden_legal_moves", [])) <= 3]
        sampled, _ = _apply_sampling(filtered, 10)
        limited = _apply_limit(sampled, 5)
        assert len(limited) == 5

    def test_pipeline_deterministic_end_to_end(self):
        ds = _make_dataset(200)
        filtered = [e for e in ds if len(e.get("hidden_legal_moves", [])) <= 3]

        sampled1, _ = _apply_sampling(filtered, 15)
        limited1 = _apply_limit(sampled1, 8)

        sampled2, _ = _apply_sampling(filtered, 15)
        limited2 = _apply_limit(sampled2, 8)

        assert [e["scenario_id"] for e in limited1] == [e["scenario_id"] for e in limited2]


# ════════════════════════════════════════════════════════════════════════════════
# Metadata structure
# ════════════════════════════════════════════════════════════════════════════════

class TestSamplingMeta:

    def test_meta_keys_present(self):
        ds = _make_dataset(50)
        _, meta = _apply_sampling(ds, 10)
        for key in ("active", "sample_size", "positions_before_sampling", "positions_after_sampling"):
            assert key in meta

    def test_meta_active_true_when_sampling(self):
        ds = _make_dataset(50)
        _, meta = _apply_sampling(ds, 10)
        assert meta["active"] is True
        assert meta["sample_size"] == 10
        assert meta["positions_before_sampling"] == 50
        assert meta["positions_after_sampling"] == 10

    def test_meta_active_false_when_noop(self):
        ds = _make_dataset(10)
        _, meta = _apply_sampling(ds, 20)
        assert meta["active"] is False
        assert meta["positions_before_sampling"] == 10
        assert meta["positions_after_sampling"] == 10

    def test_meta_sample_size_none_preserved(self):
        ds = _make_dataset(20)
        _, meta = _apply_sampling(ds, None)
        assert meta["sample_size"] is None
        assert meta["active"] is False


# ════════════════════════════════════════════════════════════════════════════════
# Backward compatibility — no --sample-size means zero change
# ════════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:

    def test_no_sample_size_full_dataset_unchanged(self):
        ds = _make_dataset(100)
        result, meta = _apply_sampling(ds, None)
        assert result is ds   # same object, no copy
        assert meta["active"] is False

    def test_existing_limit_still_works_without_sampling(self):
        ds = _make_dataset(50)
        result, _ = _apply_sampling(ds, None)
        limited = _apply_limit(result, 10)
        assert len(limited) == 10
        assert [e["scenario_id"] for e in limited] == [e["scenario_id"] for e in ds[:10]]

    def test_original_dataset_not_mutated(self):
        ds = _make_dataset(30)
        original_ids = [e["scenario_id"] for e in ds]
        _apply_sampling(ds, 10)
        assert [e["scenario_id"] for e in ds] == original_ids
