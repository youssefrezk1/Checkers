# checkers/tests/test_eval_subset.py
"""
Tests for the balanced eval subset sampler and prompt-preview tool.

Verifies:
  1. Sampled subset contains only targeted categories
  2. Each category count is ≤ MAX_PER_CATEGORY
  3. promotion_state_update gets all available (41 < 50)
  4. Excluded categories are absent from the subset
  5. All scenarios in the subset pass symbolic legal-move re-validation
  6. Deterministic: same seed → same subset
  7. Prompt builder never exposes hidden_legal_moves
  8. Prompt contains all required fields: board, side, rules, output format
  9. Ground truth (hidden_legal_moves) is non-empty for every scenario
  10. Board field in every scenario is 8×8 list of ints 0-4
"""

import json
import os
import pytest
import random

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FULL_JSONL   = os.path.join(PROJECT_ROOT, "checkers", "data", "legality_stress", "scenarios.jsonl")
EVAL_JSONL   = os.path.join(PROJECT_ROOT, "checkers", "data", "legality_stress", "eval_subset_balanced.jsonl")


# ── fixtures ────────────────────────────────────────────────────────────────

def _load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


@pytest.fixture(scope="module")
def full_scenarios():
    if not os.path.exists(FULL_JSONL):
        pytest.skip(f"Full JSONL not found: {FULL_JSONL}")
    return _load_jsonl(FULL_JSONL)


@pytest.fixture(scope="module")
def eval_subset():
    if not os.path.exists(EVAL_JSONL):
        pytest.skip(f"Eval subset not found: {EVAL_JSONL}")
    return _load_jsonl(EVAL_JSONL)


# ── imports under test ──────────────────────────────────────────────────────

from checkers.data.pdn_importer.subset_sampler import (
    sample_balanced_subset,
    TARGET_CATEGORIES,
    EXCLUDED_CATEGORIES,
    MAX_PER_CATEGORY,
)
from checkers.data.pdn_importer.fen_utils import str_to_side
from checkers.engine.rules import get_all_legal_moves


# ===========================================================================
# 1-4 – Subset content and category constraints
# ===========================================================================

class TestSubsetCategories:

    def test_only_target_categories_present(self, eval_subset):
        """No excluded or unknown category should appear in the subset."""
        found = {sc["category"] for sc in eval_subset}
        unexpected = found - set(TARGET_CATEGORIES)
        assert not unexpected, f"Unexpected categories in subset: {unexpected}"

    def test_all_target_categories_present(self, eval_subset):
        """Every targeted category should have at least one scenario."""
        found = {sc["category"] for sc in eval_subset}
        missing = set(TARGET_CATEGORIES) - found
        # promotion_state_update has only 41 scenarios — still present
        assert not missing, f"Missing targeted categories: {missing}"

    def test_no_category_exceeds_max(self, eval_subset):
        """Each category count must be ≤ MAX_PER_CATEGORY (50)."""
        from collections import Counter
        counts = Counter(sc["category"] for sc in eval_subset)
        for cat, cnt in counts.items():
            assert cnt <= MAX_PER_CATEGORY, (
                f"Category {cat!r} has {cnt} > MAX_PER_CATEGORY ({MAX_PER_CATEGORY})"
            )

    def test_promotion_state_update_count(self, eval_subset):
        """promotion_state_update must be present and respect the cap."""
        from collections import Counter
        counts = Counter(sc["category"] for sc in eval_subset)
        n = counts["promotion_state_update"]
        assert n > 0, "promotion_state_update missing from subset"
        assert n <= MAX_PER_CATEGORY, (
            f"promotion_state_update exceeded cap: {n} > {MAX_PER_CATEGORY}"
        )

    def test_excluded_categories_absent(self, eval_subset):
        """Excluded categories must not appear in the subset."""
        present = {sc["category"] for sc in eval_subset}
        for cat in EXCLUDED_CATEGORIES:
            assert cat not in present, f"Excluded category {cat!r} found in subset"

    def test_full_set_categories_50_hit_max(self, full_scenarios):
        """Categories with > 50 in the full set should exactly hit the cap."""
        from collections import Counter
        full_counts = Counter(
            sc["category"] for sc in full_scenarios
            if sc["category"] in TARGET_CATEGORIES
        )
        result = sample_balanced_subset(full_scenarios)
        counts = result["counts"]
        for cat in TARGET_CATEGORIES:
            if full_counts[cat] >= MAX_PER_CATEGORY:
                assert counts[cat] == MAX_PER_CATEGORY, (
                    f"{cat}: expected {MAX_PER_CATEGORY}, got {counts[cat]}"
                )
            else:
                assert counts[cat] == full_counts[cat], (
                    f"{cat}: expected all {full_counts[cat]}, got {counts[cat]}"
                )


# ===========================================================================
# 5 – Symbolic legal-move re-validation
# ===========================================================================

class TestSymbolicValidation:

    def test_all_hidden_legal_moves_revalidate(self, eval_subset):
        """
        For every scenario, re-run get_all_legal_moves() and confirm the
        stored hidden_legal_moves exactly match the fresh computation.
        """
        errors = []
        for sc in eval_subset:
            board = sc["board"]
            side  = str_to_side(sc["side_to_move"])
            stored = sc["hidden_legal_moves"]
            fresh  = get_all_legal_moves(board, side)

            if len(stored) != len(fresh):
                errors.append(
                    f"{sc['scenario_id']}: stored {len(stored)} moves, "
                    f"fresh computation gives {len(fresh)}"
                )
                continue

            def _key(mv):
                return (mv["type"], tuple(tuple(p) for p in mv["path"]))

            fresh_keys = {_key(m) for m in fresh}
            for mv in stored:
                if _key(mv) not in fresh_keys:
                    errors.append(
                        f"{sc['scenario_id']}: stored move {mv['path']} "
                        f"not in fresh legal moves"
                    )

        assert not errors, "Symbolic validation failures:\n" + "\n".join(errors[:10])

    def test_no_scenario_has_zero_legal_moves(self, eval_subset):
        """Every scenario must have at least one legal move."""
        for sc in eval_subset:
            board = sc["board"]
            side  = str_to_side(sc["side_to_move"])
            assert get_all_legal_moves(board, side), (
                f"Scenario {sc['scenario_id']} has 0 legal moves — should have been filtered"
            )


# ===========================================================================
# 6 – Determinism
# ===========================================================================

class TestDeterminism:

    def test_same_seed_same_subset(self, full_scenarios):
        """Two calls with the same seed must produce identical subsets."""
        r1 = sample_balanced_subset(full_scenarios, seed=42)
        r2 = sample_balanced_subset(full_scenarios, seed=42)
        ids1 = [sc["scenario_id"] for sc in r1["subset"]]
        ids2 = [sc["scenario_id"] for sc in r2["subset"]]
        assert ids1 == ids2, "Same seed produced different subsets"

    def test_different_seeds_differ(self, full_scenarios):
        """Different seeds should (almost certainly) produce different orderings."""
        r1 = sample_balanced_subset(full_scenarios, seed=1)
        r2 = sample_balanced_subset(full_scenarios, seed=99)
        ids1 = [sc["scenario_id"] for sc in r1["subset"]]
        ids2 = [sc["scenario_id"] for sc in r2["subset"]]
        # Not guaranteed but astronomically unlikely to be equal
        assert ids1 != ids2, "Different seeds produced identical subsets"


# ===========================================================================
# 7-8 – Prompt builder: no leakage, required fields present
# ===========================================================================

class TestPromptBuilder:

    @pytest.fixture(scope="class")
    def sample_sc(self, eval_subset):
        return eval_subset[0]

    def _build(self, sc):
        # Import locally to avoid circular issues
        import sys
        sys.path.insert(0, PROJECT_ROOT)
        from preview_legality_prompts import build_prompt
        return build_prompt(sc)

    def test_hidden_legal_moves_not_in_prompt(self, eval_subset):
        """hidden_legal_moves must not appear anywhere in the prompt string."""
        import sys
        sys.path.insert(0, PROJECT_ROOT)
        from preview_legality_prompts import build_prompt

        for sc in eval_subset[:20]:   # check first 20
            prompt = build_prompt(sc)
            assert "hidden_legal_moves" not in prompt, (
                f"Prompt for {sc['scenario_id']} leaks the field name 'hidden_legal_moves'"
            )

    def test_prompt_contains_side_to_move(self, eval_subset):
        import sys; sys.path.insert(0, PROJECT_ROOT)
        from preview_legality_prompts import build_prompt
        for sc in eval_subset[:10]:
            prompt = build_prompt(sc)
            assert sc["side_to_move"] in prompt

    def test_prompt_contains_rules(self, eval_subset):
        import sys; sys.path.insert(0, PROJECT_ROOT)
        from preview_legality_prompts import build_prompt
        required_phrases = [
            "MANDATORY CAPTURE",
            "MULTI-JUMP",
            "PROMOTION",
            "move in ONE forward direction only",   # actual wording in RULES_TEXT
        ]
        for sc in eval_subset[:5]:
            prompt = build_prompt(sc)
            for phrase in required_phrases:
                assert phrase in prompt, (
                    f"Prompt for {sc['scenario_id']} missing rule phrase: {phrase!r}"
                )

    def test_prompt_contains_output_format(self, eval_subset):
        import sys; sys.path.insert(0, PROJECT_ROOT)
        from preview_legality_prompts import build_prompt
        for sc in eval_subset[:5]:
            prompt = build_prompt(sc)
            assert "OUTPUT FORMAT" in prompt
            assert "SIMPLE" in prompt
            assert "JUMP" in prompt

    def test_prompt_contains_board(self, eval_subset):
        import sys; sys.path.insert(0, PROJECT_ROOT)
        from preview_legality_prompts import build_prompt
        for sc in eval_subset[:5]:
            prompt = build_prompt(sc)
            assert "CURRENT BOARD STATE" in prompt
            assert "row 0" in prompt
            assert "row 7" in prompt

    def test_prompt_contains_scenario_id(self, eval_subset):
        import sys; sys.path.insert(0, PROJECT_ROOT)
        from preview_legality_prompts import build_prompt
        for sc in eval_subset[:5]:
            prompt = build_prompt(sc)
            assert sc["scenario_id"] in prompt


# ===========================================================================
# 9-10 – Schema validity
# ===========================================================================

class TestSchemaValidity:

    REQUIRED_FIELDS = {
        "scenario_id", "source_file", "game_index", "ply_index",
        "board", "side_to_move", "hidden_legal_moves",
        "category", "expected_rule", "difficulty",
    }

    def test_all_required_fields_present(self, eval_subset):
        for sc in eval_subset:
            missing = self.REQUIRED_FIELDS - set(sc.keys())
            assert not missing, f"{sc.get('scenario_id','?')} missing fields: {missing}"

    def test_hidden_legal_moves_non_empty(self, eval_subset):
        for sc in eval_subset:
            assert len(sc["hidden_legal_moves"]) > 0, (
                f"{sc['scenario_id']}: hidden_legal_moves is empty"
            )

    def test_board_is_8x8(self, eval_subset):
        for sc in eval_subset:
            board = sc["board"]
            assert len(board) == 8, f"{sc['scenario_id']}: board has {len(board)} rows"
            for r, row in enumerate(board):
                assert len(row) == 8, (
                    f"{sc['scenario_id']}: row {r} has {len(row)} cols"
                )

    def test_board_values_in_range(self, eval_subset):
        valid = {0, 1, 2, 3, 4}
        for sc in eval_subset:
            for r, row in enumerate(sc["board"]):
                for c, val in enumerate(row):
                    assert val in valid, (
                        f"{sc['scenario_id']}: illegal piece code {val} at ({r},{c})"
                    )

    def test_side_to_move_valid(self, eval_subset):
        for sc in eval_subset:
            assert sc["side_to_move"] in {"RED", "BLACK"}, (
                f"{sc['scenario_id']}: invalid side_to_move {sc['side_to_move']!r}"
            )

    def test_difficulty_valid(self, eval_subset):
        for sc in eval_subset:
            assert sc["difficulty"] in {"easy", "medium", "hard"}, (
                f"{sc['scenario_id']}: invalid difficulty {sc['difficulty']!r}"
            )

    def test_category_valid(self, eval_subset):
        all_cats = set(TARGET_CATEGORIES)
        for sc in eval_subset:
            assert sc["category"] in all_cats, (
                f"{sc['scenario_id']}: unknown category {sc['category']!r}"
            )
