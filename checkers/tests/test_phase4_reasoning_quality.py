"""
Phase 4 reasoning-quality tests.

Covers:
  1. No schema-key leakage in seeds (seeds must be natural-language only)
  2. Restriction-role seeds (quiet_move_role=STRUCTURAL_RESTRICTION / frozen_enemy_pieces)
  3. Anti-template constraints in system prompts (no template-abuse patterns)
  4. Explanation-over-paraphrase prompting (causal WHY, not seed listing)
  5. Refinement prompt no longer exposes raw schema keys
  6. No unsupported strategic wording in seeds
"""
import sys
import os
import re
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from checkers.agents.ranker_agent import (
    _build_grounded_reasoning_seeds,
    _build_seed_reasoning_prompt,
    _build_refinement_prompt,
    RANKER_SEED_REASONING_SYSTEM,
    RANKER_REASONING_REFINEMENT_SYSTEM,
)


# ── Shared move fixture ───────────────────────────────────────────────────────

_BASE_FACTS: dict = {
    "opponent_can_recapture": False,
    "our_pieces_threatened_after": 0,
    "captures_count": 0,
    "net_gain": 0,
    "creates_immediate_threat": False,
    "leaves_piece_isolated": False,
    "center_control": False,
    "results_in_king": False,
    "minimax_score": 1.5,
}

_BASE_MOVE: dict = {
    "type": "simple",
    "path": [[5, 4], [4, 3]],
    "captured": [],
    "facts": _BASE_FACTS,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — No schema-key leakage in seeds
# ═══════════════════════════════════════════════════════════════════════════════

_KEY_VALUE_PATTERN = re.compile(r'\b\w+=(?:true|false|-?\d+\.?\d*)\b', re.IGNORECASE)


class TestNoSchemaLeakageInSeeds:
    """Seeds must never contain raw key=value schema notation."""

    def _seeds(self, facts: Optional[dict] = None, candidates=None) -> list[str]:
        move = {**_BASE_MOVE, "facts": facts or _BASE_FACTS}
        if candidates is None:
            candidates = [move]
        return _build_grounded_reasoning_seeds(move, candidates)

    def _joined(self, facts: Optional[dict] = None) -> str:
        return " ".join(self._seeds(facts))

    def test_no_key_equals_value_in_any_seed(self):
        seeds = self._seeds()
        for seed in seeds:
            assert not _KEY_VALUE_PATTERN.search(seed), (
                f"Seed contains key=value notation: {seed!r}"
            )

    def test_no_key_equals_value_with_capture_facts(self):
        facts = {**_BASE_FACTS, "captures_count": 2, "net_gain": 2}
        seeds = self._seeds(facts)
        for seed in seeds:
            assert not _KEY_VALUE_PATTERN.search(seed), f"Schema leak in: {seed!r}"

    def test_no_key_equals_value_with_recapture_true(self):
        facts = {**_BASE_FACTS, "opponent_can_recapture": True}
        seeds = self._seeds(facts)
        for seed in seeds:
            assert not _KEY_VALUE_PATTERN.search(seed), f"Schema leak in: {seed!r}"

    def test_no_key_equals_value_with_isolation_true(self):
        facts = {**_BASE_FACTS, "leaves_piece_isolated": True}
        seeds = self._seeds(facts)
        for seed in seeds:
            assert not _KEY_VALUE_PATTERN.search(seed), f"Schema leak in: {seed!r}"

    def test_no_key_equals_value_with_threat_true(self):
        facts = {**_BASE_FACTS, "creates_immediate_threat": True}
        seeds = self._seeds(facts)
        for seed in seeds:
            assert not _KEY_VALUE_PATTERN.search(seed), f"Schema leak in: {seed!r}"

    def test_no_raw_field_names_as_standalone_words(self):
        text = self._joined()
        forbidden_keys = [
            "opponent_can_recapture", "captures_count", "net_gain",
            "center_control", "leaves_piece_isolated", "results_in_king",
            "creates_immediate_threat", "minimax_score",
        ]
        for key in forbidden_keys:
            assert key not in text, f"Raw field name '{key}' appears in seeds: {text!r}"

    def test_minimax_seed_is_natural_language(self):
        seeds = self._seeds()
        mm_seeds = [s for s in seeds if "engine scores" in s]
        assert len(mm_seeds) == 1
        assert not _KEY_VALUE_PATTERN.search(mm_seeds[0])

    def test_comparison_seed_is_natural_language(self):
        alt = {
            "type": "simple",
            "path": [[6, 3], [5, 2]],
            "captured": [],
            "facts": {**_BASE_FACTS, "opponent_can_recapture": True},
        }
        chosen = {**_BASE_MOVE, "facts": {**_BASE_FACTS, "opponent_can_recapture": False}}
        seeds = _build_grounded_reasoning_seeds(chosen, [chosen, alt])
        comparison_seeds = [s for s in seeds if "unlike move" in s.lower() or "chosen move" in s.lower()]
        for seed in comparison_seeds:
            assert not _KEY_VALUE_PATTERN.search(seed), f"Schema leak in comparison: {seed!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — Restriction-role seeds
# ═══════════════════════════════════════════════════════════════════════════════

class TestRestrictionRoleSeeds:
    """quiet_move_role=STRUCTURAL_RESTRICTION and frozen_enemy_pieces seeds."""

    def _seeds(self, **extra_facts) -> list[str]:
        facts = {**_BASE_FACTS, **extra_facts}
        move = {**_BASE_MOVE, "facts": facts}
        return _build_grounded_reasoning_seeds(move, [move])

    def test_frozen_pieces_seed_emitted_when_role_set(self):
        seeds = self._seeds(quiet_move_role="STRUCTURAL_RESTRICTION", frozen_enemy_pieces=3)
        assert any("3 opponent piece(s) have restricted forward movement" in s for s in seeds), (
            f"Expected frozen-piece restriction seed, got: {seeds}"
        )

    def test_frozen_pieces_seed_emitted_when_restriction_score_positive(self):
        seeds = self._seeds(restriction_score=0.5, frozen_enemy_pieces=2)
        assert any("2 opponent piece(s) have restricted forward movement" in s for s in seeds)

    def test_generic_constraint_seed_when_no_frozen_pieces(self):
        seeds = self._seeds(quiet_move_role="STRUCTURAL_RESTRICTION", restriction_score=0.5)
        assert any("constrains the opponent" in s for s in seeds), (
            f"Expected generic constraint seed when frozen=0, got: {seeds}"
        )

    def test_no_restriction_seed_when_role_not_set(self):
        seeds = self._seeds(restriction_score=0.0, frozen_enemy_pieces=0)
        assert not any("restricted forward movement" in s for s in seeds)
        assert not any("constrains the opponent" in s for s in seeds)

    def test_restriction_seed_contains_correct_count(self):
        seeds = self._seeds(quiet_move_role="STRUCTURAL_RESTRICTION", frozen_enemy_pieces=1)
        restriction_seeds = [s for s in seeds if "restricted forward movement" in s]
        assert restriction_seeds
        assert "1 opponent piece(s)" in restriction_seeds[0]

    def test_restriction_seed_is_natural_language(self):
        seeds = self._seeds(quiet_move_role="STRUCTURAL_RESTRICTION", frozen_enemy_pieces=2)
        restriction_seeds = [s for s in seeds if "restricted forward movement" in s]
        for seed in restriction_seeds:
            assert not _KEY_VALUE_PATTERN.search(seed), f"Schema leak in restriction: {seed!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3 — Anti-template constraints in seed reasoning system prompt
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeedReasoningSystemPromptAntiTemplate:
    """RANKER_SEED_REASONING_SYSTEM must enforce causal explanation over paraphrase."""

    PROMPT = RANKER_SEED_REASONING_SYSTEM

    def test_requires_causal_why_explanation(self):
        assert "why" in self.PROMPT.lower(), "Prompt must require WHY explanation"

    def test_forbids_seed_listing_or_paraphrase(self):
        lower = self.PROMPT.lower()
        assert any(p in lower for p in ("do not paraphrase", "do not list", "do not mechanically")), (
            "Prompt must explicitly forbid mechanical listing/paraphrasing"
        )

    def test_requires_synthesis(self):
        lower = self.PROMPT.lower()
        assert "synthesize" in lower or "causal" in lower, (
            "Prompt must require synthesis/causal reasoning"
        )

    def test_has_anti_template_rules_section(self):
        assert "ANTI-TEMPLATE" in self.PROMPT or "anti-template" in self.PROMPT.lower(), (
            "Prompt must have an anti-template rules block"
        )

    def test_forbids_key_value_notation(self):
        lower = self.PROMPT.lower()
        assert "key=value" in lower or "variable name" in lower or "schema key" in lower, (
            "Prompt must explicitly forbid key=value notation"
        )

    def test_forbidden_vocab_block_present(self):
        assert "Do NOT use any of the following" in self.PROMPT, (
            "Prompt must have explicit forbidden-vocab block"
        )

    def test_paragraph_length_guidance_present(self):
        assert "3-5 sentences" in self.PROMPT or "3–5 sentences" in self.PROMPT, (
            "Prompt must specify 3-5 sentence paragraph length"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Part 4 — Refinement prompt does not expose raw schema keys
# ═══════════════════════════════════════════════════════════════════════════════

class TestRefinementPromptNoSchemaKeys:
    """_build_refinement_prompt must produce human-readable fact lines, not key=value."""

    def _prompt(self, facts: Optional[dict] = None) -> str:
        move = {**_BASE_MOVE, "facts": facts or _BASE_FACTS}
        return _build_refinement_prompt(move, [])

    def test_no_raw_key_equals_value_in_prompt(self):
        prompt = self._prompt({**_BASE_FACTS, "opponent_can_recapture": True})
        assert "opponent_can_recapture=" not in prompt
        assert "center_control=" not in prompt

    def test_human_readable_recapture_line(self):
        prompt = self._prompt({**_BASE_FACTS, "opponent_can_recapture": True})
        assert "opponent CAN recapture" in prompt

    def test_human_readable_no_recapture_line(self):
        prompt = self._prompt({**_BASE_FACTS, "opponent_can_recapture": False})
        assert "cannot recapture" in prompt

    def test_human_readable_capture_line(self):
        facts = {**_BASE_FACTS, "captures_count": 2, "net_gain": 2}
        prompt = self._prompt(facts)
        assert "captures 2 piece(s)" in prompt
        assert "captures_count=" not in prompt

    def test_human_readable_positional_move_line(self):
        prompt = self._prompt()  # captures_count=0
        assert "no captures" in prompt or "positional" in prompt

    def test_minimax_score_shown_as_number(self):
        facts = {**_BASE_FACTS, "minimax_score": 3.5}
        prompt = self._prompt(facts)
        assert "3.5" in prompt
        assert "minimax_score=" not in prompt

    def test_isolation_shown_as_natural_language(self):
        facts = {**_BASE_FACTS, "leaves_piece_isolated": True}
        prompt = self._prompt(facts)
        assert "leaves_piece_isolated=" not in prompt
        assert "unsupported" in prompt or "isolated" in prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Part 5 — Refinement system prompt naturlness section
# ═══════════════════════════════════════════════════════════════════════════════

class TestRefinementSystemPromptNaturalness:
    """RANKER_REASONING_REFINEMENT_SYSTEM must contain naturlness/clarity guidance."""

    PROMPT = RANKER_REASONING_REFINEMENT_SYSTEM

    def test_naturalness_section_present(self):
        assert "NATURALNESS" in self.PROMPT or "naturalness" in self.PROMPT.lower()

    def test_forbids_filler_openers(self):
        lower = self.PROMPT.lower()
        assert "despite" in lower or "furthermore" in lower or "in summary" in lower, (
            "Prompt should list filler openers to avoid"
        )

    def test_requires_causal_connection(self):
        lower = self.PROMPT.lower()
        assert "causal" in lower or "why" in lower

    def test_forbids_repetitive_sentence_openers(self):
        lower = self.PROMPT.lower()
        assert "same word" in lower or "consecutive" in lower or "do not start" in lower, (
            "Prompt must warn against repeating sentence openers"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Part 6 — Seed prompt content: grounded formatting checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeedReasoningPromptContent:
    """_build_seed_reasoning_prompt must list natural-language seeds verbatim."""

    def _prompt(self, seeds: list[str]) -> str:
        return _build_seed_reasoning_prompt(_BASE_MOVE, seeds)

    def test_seeds_appear_verbatim_in_prompt(self):
        seeds = [
            "The moved piece cannot be immediately recaptured.",
            "The move captures 1 piece(s), gaining a net advantage of 1.",
            "The engine scores this move 2.5 — highest-evaluated option.",
        ]
        prompt = self._prompt(seeds)
        for seed in seeds:
            assert seed in prompt, f"Seed not found verbatim in prompt: {seed!r}"

    def test_move_path_included_in_prompt(self):
        prompt = self._prompt(["A seed."])
        assert "5, 4" in prompt or "[5, 4]" in prompt

    def test_prompt_uses_label_not_key_value(self):
        seeds = ["The moved piece cannot be immediately recaptured."]
        prompt = self._prompt(seeds)
        assert "opponent_can_recapture" not in prompt

    def test_prompt_includes_why_instruction(self):
        prompt = self._prompt(["A seed."])
        assert "why" in prompt.lower() or "reason" in prompt.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Part 7 — No unsupported strategic wording in seeds
# ═══════════════════════════════════════════════════════════════════════════════

_FORBIDDEN_SEED_PHRASES = [
    "central board presence",
    "positional advantage",
    "strategic goal",
    "positional adjustment",
    "neutral positional",
    "conversion potential",
    "escape route",
    "real trap",
    "structural restriction",  # label is fine but phrase itself is banned
    "winning conversion",
]


class TestNoUnsupportedStrategicWordingInSeeds:
    """Seeds must not introduce phrases the checker would flag as forbidden vocab."""

    def _all_seeds(self, extra_facts: Optional[dict] = None) -> list[str]:
        facts = {**_BASE_FACTS, **(extra_facts or {})}
        move = {**_BASE_MOVE, "facts": facts}
        return _build_grounded_reasoning_seeds(move, [move])

    def _joined(self, extra_facts: Optional[dict] = None) -> str:
        return " ".join(self._all_seeds(extra_facts)).lower()

    def test_no_central_board_presence(self):
        assert "central board presence" not in self._joined({"center_control": True})

    def test_no_positional_advantage_phrase(self):
        assert "positional advantage" not in self._joined()

    def test_no_strategic_goal_phrase(self):
        assert "strategic goal" not in self._joined()

    def test_no_conversion_potential_phrase(self):
        assert "conversion potential" not in self._joined()

    def test_no_escape_route_phrase(self):
        assert "escape route" not in self._joined()

    def test_no_real_trap_phrase(self):
        assert "real trap" not in self._joined()

    def test_no_positional_adjustment_phrase(self):
        assert "positional adjustment" not in self._joined()

    def test_center_seed_uses_safe_phrasing(self):
        """Center seed must say 'center of the board', not the conflated phrase."""
        seeds = self._all_seeds()
        center_seeds = [s for s in seeds if "center" in s.lower()]
        for seed in center_seeds:
            assert "central board presence" not in seed.lower()
            assert "central board control" not in seed.lower() or "gains central board control" in seed.lower()

    def test_restriction_role_seed_does_not_use_structural_restriction_phrase(self):
        """The structural_restriction seed must not literally write 'structural restriction role'."""
        seeds = self._all_seeds(
            {"quiet_move_role": "STRUCTURAL_RESTRICTION", "frozen_enemy_pieces": 2}
        )
        for seed in seeds:
            assert "structural restriction role" not in seed.lower(), (
                f"Seed uses forbidden phrase: {seed!r}"
            )
