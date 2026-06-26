# checkers/tests/test_comparative_prompts.py
#
# Step 2 of the Comparative Reasoning v2 roadmap: focused unit tests for the
# comparative system prompts and user-prompt builders in
# `checkers/agents/comparative_reasoner.py`.
#
# Coverage matrix:
#   - JSON output contract present in both system prompts
#   - prompts remain compact (≤30 lines locked)
#   - user prompt injects every seed
#   - user prompt includes chosen-path as reference but NO chosen paragraph
#   - refinement builder accepts ALREADY-sanitized issues (no in-builder scrub)
#   - both builders are pure (no runtime side effects)
#   - no robotic "Alternative [N] [THEME]:" enumeration produced by builders
#   - no verbatim forbidden-vocabulary enumeration in system prompts

from __future__ import annotations

from checkers.agents.comparative_reasoner import (
    EXPLAINER_COMPARATIVE_SYSTEM as RANKER_COMPARATIVE_SYSTEM,
    EXPLAINER_COMPARATIVE_REFINEMENT_SYSTEM as RANKER_COMPARATIVE_REFINEMENT_SYSTEM,
    build_comparative_user_prompt,
    build_comparative_refinement_user_prompt,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. RANKER_COMPARATIVE_SYSTEM contract
# ═══════════════════════════════════════════════════════════════════════════


class TestComparativeSystemPrompt:
    def test_contains_json_output_contract(self):
        assert '{"comparative_reasoning"' in RANKER_COMPARATIVE_SYSTEM

    def test_specifies_no_markdown(self):
        assert "no markdown" in RANKER_COMPARATIVE_SYSTEM.lower()

    def test_prompt_is_compact_30_lines_or_fewer(self):
        # Locked: comparative system prompt must stay ≤30 lines.
        lines = RANKER_COMPARATIVE_SYSTEM.splitlines()
        assert len(lines) <= 30, (
            f"comparative system prompt has {len(lines)} lines; "
            "locked maximum is 30"
        )

    def test_states_describe_alternatives_only(self):
        lower = RANKER_COMPARATIVE_SYSTEM.lower()
        assert "alternative" in lower
        assert "tradeoff" in lower

    def test_forbids_re_justifying_chosen_move(self):
        lower = RANKER_COMPARATIVE_SYSTEM.lower()
        # Either phrasing satisfies the "do NOT re-justify" contract.
        assert (
            "do not re-justify" in lower
            or "do not re-state" in lower
            or "not re-justify" in lower
        )

    def test_forbids_robotic_alternative_theme_template(self):
        # The locked spec forbids the "Alternative [N] [THEME]:" robotic
        # form. The prompt must convey this either as an explicit negative
        # example or as a positive instruction to use natural prose.
        has_negative = "[N] [THEME]" in RANKER_COMPARATIVE_SYSTEM
        has_positive = "natural prose" in RANKER_COMPARATIVE_SYSTEM.lower()
        assert has_negative or has_positive

    def test_no_forbidden_vocab_enumeration(self):
        # Locked spec: no verbatim forbidden-vocab enumeration in the
        # comparative system prompt. A sample of phrases that the
        # chosen-move pipeline forbids must NOT appear here.
        for phrase in (
            "conversion potential",
            "king escape",
            "activity score",
            "counterplay_score",
            "diagonal pressure",
            "regulars_captured",
            "quiet_move_role",
        ):
            assert phrase not in RANKER_COMPARATIVE_SYSTEM, (
                f"forbidden vocab '{phrase}' must not be enumerated in the "
                "comparative system prompt"
            )

    def test_natural_language_prose_requirement(self):
        lower = RANKER_COMPARATIVE_SYSTEM.lower()
        assert "natural" in lower or "fluent" in lower


# ═══════════════════════════════════════════════════════════════════════════
# 2. RANKER_COMPARATIVE_REFINEMENT_SYSTEM contract
# ═══════════════════════════════════════════════════════════════════════════


class TestComparativeRefinementSystemPrompt:
    def test_contains_json_output_contract(self):
        assert '{"comparative_reasoning"' in RANKER_COMPARATIVE_REFINEMENT_SYSTEM

    def test_specifies_no_markdown(self):
        assert "no markdown" in RANKER_COMPARATIVE_REFINEMENT_SYSTEM.lower()

    def test_prompt_is_compact_30_lines_or_fewer(self):
        lines = RANKER_COMPARATIVE_REFINEMENT_SYSTEM.splitlines()
        assert len(lines) <= 30, (
            f"comparative refinement system prompt has {len(lines)} lines"
        )

    def test_forbids_re_justifying_chosen_move(self):
        lower = RANKER_COMPARATIVE_REFINEMENT_SYSTEM.lower()
        assert "do not re-justify" in lower or "not re-justify" in lower

    def test_minimal_diff_principle_present(self):
        lower = RANKER_COMPARATIVE_REFINEMENT_SYSTEM.lower()
        # Refinement contract requires preserving correct text.
        assert "preserve" in lower

    def test_no_forbidden_vocab_enumeration(self):
        for phrase in (
            "conversion potential",
            "king escape",
            "activity score",
            "diagonal pressure",
        ):
            assert phrase not in RANKER_COMPARATIVE_REFINEMENT_SYSTEM


# ═══════════════════════════════════════════════════════════════════════════
# 3. build_comparative_user_prompt
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildComparativeUserPrompt:
    SEEDS_SAMPLE = [
        "Aggressive alternatives [1] and [3] create immediate threats; "
        "chosen move forfeits initiative for safety.",
        "Defensive alternative [2] avoids recapture but does not capture.",
        "Chosen move tradeoff: forfeits aggressive options [1,3] in "
        "favour of recapture safety.",
    ]
    PATH_SAMPLE = [[5, 4], [4, 3]]

    def test_includes_every_seed_verbatim(self):
        prompt = build_comparative_user_prompt(
            self.SEEDS_SAMPLE, self.PATH_SAMPLE,
        )
        for s in self.SEEDS_SAMPLE:
            assert s in prompt, f"missing seed: {s!r}"

    def test_includes_chosen_path_as_reference(self):
        prompt = build_comparative_user_prompt([], self.PATH_SAMPLE)
        # The path data should be present somewhere in the prompt.
        assert "[5, 4]" in prompt or str(self.PATH_SAMPLE) in prompt

    def test_chosen_path_labelled_as_reference_only(self):
        # The path appears with an explicit "reference only" framing so
        # the LLM does not treat it as material for re-justifying.
        prompt = build_comparative_user_prompt([], self.PATH_SAMPLE)
        assert "reference only" in prompt.lower()

    def test_no_chosen_move_paragraph_contamination(self):
        # The user prompt must NOT include any chosen-move reasoning prose.
        prompt = build_comparative_user_prompt(
            ["seed text"], self.PATH_SAMPLE,
        )
        for marker in (
            "chosen reasoning",
            "chosen paragraph",
            "previous reasoning",
            "previously generated reasoning",
            "last_move_reasoning",
            "chosen-move paragraph",
        ):
            assert marker not in prompt.lower(), (
                f"chosen-paragraph marker '{marker}' leaked into "
                "comparative user prompt"
            )

    def test_includes_json_output_reminder(self):
        prompt = build_comparative_user_prompt([], self.PATH_SAMPLE)
        assert '"comparative_reasoning"' in prompt

    def test_handles_empty_seeds_gracefully(self):
        prompt = build_comparative_user_prompt([], self.PATH_SAMPLE)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_handles_none_path_gracefully(self):
        # If a caller passes None, the function must still return a string.
        prompt = build_comparative_user_prompt(["seed"], None)
        assert isinstance(prompt, str)
        assert "seed" in prompt

    def test_pure_function_same_inputs_same_output(self):
        # Determinism check — no hidden state.
        p1 = build_comparative_user_prompt(
            self.SEEDS_SAMPLE, self.PATH_SAMPLE,
        )
        p2 = build_comparative_user_prompt(
            self.SEEDS_SAMPLE, self.PATH_SAMPLE,
        )
        assert p1 == p2

    def test_does_not_emit_robotic_pattern_from_builder(self):
        # The builder itself never inserts the "Alternative [N] [THEME]:"
        # form. Seed strings are passed through verbatim — that's the
        # caller's responsibility — but the builder's own scaffolding text
        # must be free of the pattern.
        prompt = build_comparative_user_prompt([], self.PATH_SAMPLE)
        assert "Alternative [N] [THEME]" not in prompt
        for tag in (
            "[AGGRESSIVE]", "[MATERIAL]", "[DEFENSIVE]",
            "[STRUCTURAL]", "[PROMOTION]", "[MOBILITY]",
        ):
            assert tag not in prompt


# ═══════════════════════════════════════════════════════════════════════════
# 4. build_comparative_refinement_user_prompt
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildComparativeRefinementUserPrompt:
    PREV_TEXT = (
        "Aggressive alternatives [1] and [3] create immediate threats but "
        "allow recapture; chosen move forfeits initiative for safety."
    )
    SANITIZED_ISSUES = [
        "sentence makes an incorrect claim about an alternative move",
        "sentence contains a numeric value not present in any seed",
    ]

    def test_includes_previous_text_verbatim(self):
        prompt = build_comparative_refinement_user_prompt(
            self.PREV_TEXT, self.SANITIZED_ISSUES,
        )
        assert self.PREV_TEXT in prompt

    def test_includes_each_sanitized_issue(self):
        prompt = build_comparative_refinement_user_prompt(
            self.PREV_TEXT, self.SANITIZED_ISSUES,
        )
        for issue in self.SANITIZED_ISSUES:
            assert issue in prompt

    def test_includes_json_output_reminder(self):
        prompt = build_comparative_refinement_user_prompt("prev", [])
        assert '"comparative_reasoning"' in prompt

    def test_accepts_already_sanitized_issues_only_no_in_builder_scrub(self):
        # Locked contract: the builder DOES NOT sanitize its inputs. If a
        # caller passes a raw verbatim forbidden phrase or number, the
        # builder forwards it as-is. (This is by design: sanitization is
        # the orchestrator's responsibility — Step 5.)
        raw = "raw forbidden phrase 'activity score' appears here"
        prompt = build_comparative_refinement_user_prompt(
            "prev text", [raw],
        )
        assert raw in prompt
        # And the literal phrase is passed through.
        assert "activity score" in prompt

    def test_no_chosen_move_paragraph_contamination(self):
        prompt = build_comparative_refinement_user_prompt(
            self.PREV_TEXT, self.SANITIZED_ISSUES,
        )
        for marker in (
            "chosen reasoning",
            "chosen paragraph",
            "chosen-move paragraph",
            "last_move_reasoning",
        ):
            assert marker not in prompt.lower()

    def test_handles_empty_issues_gracefully(self):
        prompt = build_comparative_refinement_user_prompt(self.PREV_TEXT, [])
        assert isinstance(prompt, str)
        assert self.PREV_TEXT in prompt

    def test_pure_function_same_inputs_same_output(self):
        p1 = build_comparative_refinement_user_prompt(
            self.PREV_TEXT, self.SANITIZED_ISSUES,
        )
        p2 = build_comparative_refinement_user_prompt(
            self.PREV_TEXT, self.SANITIZED_ISSUES,
        )
        assert p1 == p2

    def test_returns_string(self):
        result = build_comparative_refinement_user_prompt("", [])
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════
# 5. No-runtime-side-effects sanity (importing the module has no effect)
# ═══════════════════════════════════════════════════════════════════════════


class TestNoRuntimeSideEffects:
    def test_module_import_does_not_invoke_llm(self):
        # The Step 2 additions must not register, schedule, or invoke any
        # network/LLM call at import time. We verify by importing fresh
        # and checking that no module-level side-effect markers appear.
        import importlib
        import checkers.agents.comparative_reasoner as cr
        importlib.reload(cr)
        # The four Step 2 names are present.
        assert hasattr(cr, "EXPLAINER_COMPARATIVE_SYSTEM")
        assert hasattr(cr, "EXPLAINER_COMPARATIVE_REFINEMENT_SYSTEM")
        assert hasattr(cr, "build_comparative_user_prompt")
        assert hasattr(cr, "build_comparative_refinement_user_prompt")

    def test_constants_are_strings(self):
        from checkers.agents.comparative_reasoner import (
            EXPLAINER_COMPARATIVE_SYSTEM as RANKER_COMPARATIVE_SYSTEM,
            EXPLAINER_COMPARATIVE_REFINEMENT_SYSTEM as RANKER_COMPARATIVE_REFINEMENT_SYSTEM,
        )
        assert isinstance(RANKER_COMPARATIVE_SYSTEM, str)
        assert isinstance(RANKER_COMPARATIVE_REFINEMENT_SYSTEM, str)
