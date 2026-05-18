# checkers/tests/test_baseline_prompts.py
#
# Targeted tests for the B3_strategic_facts_llm baseline prompt.
#
# Coverage:
#   1. B3_strategic_facts_llm prompt exposes tactical/strategic facts
#      (captures, recapture, mobility, threats, isolation, promotion,
#      forced reply, geometry, …).
#   2. B3 prompt HIDES every minimax-answer-label field
#      (minimax_score, symbolic_rank, rank, score_gap, best_score,
#      second_best_score, best_move_index, best_move_path, is_best_move).
#   3. B3 prompt requires per-move analysis covering every legal move.
#   4. B3 system prompt does NOT copy the B4 ranker decision algorithm.
#   5. B3 runner has no verifier / retry / fallback / override hook.
#   6. Baseline name + dispatch are registered and reachable; LLM call is
#      stubbed so no network is used.

from __future__ import annotations

import os
os.environ.setdefault("USE_SIMPLIFIED_PIPELINE", "true")
os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

import inspect

import pytest

from checkers.engine.board import EMPTY, RED, BLACK
from checkers.baseline_eval.run_baseline_human_trace import (
    BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS,
    _RULES_LEGAL_LLM_STRATEGIC_FACTS_SYSTEM,
    _STRATEGIC_FACTS_KEYS,
    _STRATEGIC_FACTS_MINIMAX_DENY,
    _build_rules_legal_llm_strategic_facts_user,
)
from checkers.baseline_eval.run_baseline_scenario_suite import (
    _ALL_SUITE_BASELINES, _SUITE_BASELINES,
    run_scenario_for_baseline, _run_scenario_strategic_facts,
)


# ── Fixture position ─────────────────────────────────────────────────────────

def _board_with_capture() -> list[list[int]]:
    """Two RED pieces, mandatory captures available."""
    b = [[EMPTY] * 8 for _ in range(8)]
    b[4][1] = RED
    b[4][5] = RED
    b[3][2] = BLACK
    b[3][6] = BLACK
    return b


# ── 1. Tactical facts present ────────────────────────────────────────────────

def test_strategic_facts_prompt_contains_tactical_fact_fields():
    from checkers.engine.rules import get_all_legal_moves
    from checkers.agents.scorer_agent import score_all_legal_moves

    board = _board_with_capture()
    legal = get_all_legal_moves(board, RED)
    scored, *_ = score_all_legal_moves(board, RED)

    user = _build_rules_legal_llm_strategic_facts_user(board, legal, scored, 1)

    must_have = {
        "captures_count", "net_gain",
        "opponent_can_recapture",
        "moved_piece_is_threatened",
        "our_pieces_threatened_before", "our_pieces_threatened_after",
        "max_opponent_jump_captures", "forced_opponent_jump_reply",
        "our_mobility_after", "opponent_mobility_after",
        "results_in_king", "near_promotion",
        "any_piece_isolated",
    }
    missing = [k for k in must_have if k not in user]
    assert not missing, (
        f"B3_strategic_facts prompt is missing tactical fact fields: {missing}"
    )


# ── 2. Minimax-answer-label fields hidden ───────────────────────────────────

def test_strategic_facts_prompt_hides_minimax_labels():
    """
    The B3 prompt must NEVER include any minimax oracle answer signal.
    """
    from checkers.engine.rules import get_all_legal_moves
    from checkers.agents.scorer_agent import score_all_legal_moves

    board = _board_with_capture()
    legal = get_all_legal_moves(board, RED)
    scored, *_ = score_all_legal_moves(board, RED)
    # Sanity: scored entries DO have minimax_score / symbolic_rank populated.
    assert "minimax_score" in (scored[0].get("facts") or {})
    assert "symbolic_rank" in (scored[0].get("facts") or {})

    user = _build_rules_legal_llm_strategic_facts_user(board, legal, scored, 1)

    forbidden = [
        "minimax_score", "symbolic_rank", "score_gap", "best_score",
        "second_best_score", "best_move_index", "best_move_path",
        "is_best_move", "is_best",
        "highest evaluated", "least harmful",
    ]
    leaks = [tok for tok in forbidden if tok in user]
    assert not leaks, (
        f"B3 prompt leaks minimax-answer signal fields: {leaks}"
    )
    # Also: the allow-list itself must not contain any forbidden field.
    bad_keys = [k for k in _STRATEGIC_FACTS_KEYS
                if k in _STRATEGIC_FACTS_MINIMAX_DENY]
    assert not bad_keys, (
        f"_STRATEGIC_FACTS_KEYS leaks deny-list members: {bad_keys}"
    )


def test_strategic_facts_minimax_deny_lists_required_fields():
    required_deny = {
        "minimax_score", "symbolic_rank", "rank", "score_gap",
        "best_score", "second_best_score",
        "best_move_index", "best_move_path",
        "is_best_move",
    }
    missing = required_deny - _STRATEGIC_FACTS_MINIMAX_DENY
    assert not missing, (
        f"Deny-list must include: {missing}"
    )


def test_scenario_json_minimax_labels_are_not_in_prompt():
    """
    Scenario JSON may carry best_move_index / best_score / score_gap for
    evaluation metrics, but the prompt builder must NEVER expose them.
    """
    from checkers.engine.rules import get_all_legal_moves
    from checkers.agents.scorer_agent import score_all_legal_moves

    board = _board_with_capture()
    legal = get_all_legal_moves(board, RED)
    scored, best_score, second_best, gap = score_all_legal_moves(board, RED)
    # Simulate scenario-JSON metadata that DOES include minimax labels.
    scenario_meta = {
        "best_move_index": 0,
        "best_score":      float(best_score),
        "score_gap":       float(gap) if gap != float("inf") else 99.0,
        "best_move_path":  list(scored[0]["path"]),
    }

    user = _build_rules_legal_llm_strategic_facts_user(board, legal, scored, 1)
    for k, v in scenario_meta.items():
        # Field name must not appear in the prompt.
        assert k not in user, f"Prompt leaks scenario meta key {k!r}"
    # And the numeric values of minimax-only fields must not appear with the
    # forbidden field names.
    assert "minimax_score" not in user
    assert "symbolic_rank" not in user


# ── 3. Requires per-move analysis covering every index ──────────────────────

def test_strategic_facts_prompt_requires_per_move_analysis():
    text = _RULES_LEGAL_LLM_STRATEGIC_FACTS_SYSTEM
    # Must reference move_analysis and require every index.
    assert "move_analysis" in text
    assert "every" in text.lower() and "legal move" in text.lower()
    # Output schema includes move_analysis with index/pros/cons/verdict.
    for tok in ("pros", "cons", "verdict", "chosen_index", "chosen_path"):
        assert tok in text, f"system prompt missing schema token: {tok}"


def test_strategic_facts_user_prompt_lists_all_indices():
    from checkers.engine.rules import get_all_legal_moves
    from checkers.agents.scorer_agent import score_all_legal_moves

    board = _board_with_capture()
    legal = get_all_legal_moves(board, RED)
    scored, *_ = score_all_legal_moves(board, RED)
    user = _build_rules_legal_llm_strategic_facts_user(board, legal, scored, 1)
    n = len(legal)
    # Every index 0..n-1 must appear in the rendered move list.
    for i in range(n):
        assert f"[{i}]" in user, f"Prompt missing index marker [{i}]"
    # Phrasing must explicitly require coverage of every index.
    assert f"0 to {n - 1}" in user


# ── 4. Does NOT copy B4 ranker system prompt ────────────────────────────────

def test_strategic_facts_system_does_not_copy_b4_ranker_prompt():
    from checkers.agents.ranker_agent import RANKER_SYSTEM_PROMPT

    text = _RULES_LEGAL_LLM_STRATEGIC_FACTS_SYSTEM
    banned_phrases = [
        "STEP 1 — BOARD SAFETY",
        "ANTI-OVERDEFENSIVE",
        "HARD LOSING-MODE RULE",
        "TACTICAL EXPOSURE RULE",
        "DECISION ALGORITHM",
        "unsafe_simple_move",
        "counterplay_score",
    ]
    for phrase in banned_phrases:
        assert phrase not in text, (
            f"B3 system prompt unexpectedly contains B4 ranker phrase: "
            f"{phrase!r}"
        )

    ranker_lines = {
        ln.strip() for ln in RANKER_SYSTEM_PROMPT.splitlines()
        if len(ln.strip()) > 25
    }
    overlap = sum(1 for ln in ranker_lines if ln in text)
    overlap_ratio = overlap / max(1, len(ranker_lines))
    assert overlap_ratio < 0.20, (
        f"B3 prompt shares {overlap_ratio:.0%} of B4 ranker prompt's lines"
    )


# ── 5. No verifier/retry/fallback in runner ─────────────────────────────────

def test_strategic_facts_runner_has_no_verifier_retry_fallback():
    src = inspect.getsource(_run_scenario_strategic_facts)
    forbidden_keywords = [
        "ranker_agent",
        "ranker_retry",
        "build_ranker_user_prompt",
        "build_ranker_user_prompt_single",
        "build_retry_user_prompt",
        "_check_reasoning_truthfulness",
        "refinement_prompt",
        "_apply_safety_filter",
        "_override_if_llm_chose_much_worse_minimax",
        "_should_force_best_minimax",
        "checkers_graph",
        "CheckersState",
        "interrupt_after",
        "graph.stream",
    ]
    hits = [kw for kw in forbidden_keywords if kw in src]
    assert not hits, (
        f"_run_scenario_strategic_facts should not reference verifier/retry/"
        f"fallback machinery, but found: {hits}"
    )


# ── 5b. B4 fairness: symbolic_rank hidden from LLM prompt only ──────────────

def test_b4_ranker_prompt_includes_minimax_but_hides_symbolic_rank():
    """
    Phase 5.2 fairness:
      • B4 LLM prompt must still expose minimax_score per candidate.
      • B4 LLM prompt must NOT expose symbolic_rank.
      • The underlying move objects must STILL carry symbolic_rank so that
        fallback / diagnostics / evaluation can read it.
    """
    import json
    from checkers.engine.rules import get_all_legal_moves
    from checkers.agents.scorer_agent import score_all_legal_moves
    from checkers.agents.ranker_agent import (
        build_ranker_user_prompt, build_ranker_user_prompt_single,
        _build_retry_user_prompt,
    )
    from checkers.state.state import CheckersState

    board = _board_with_capture()
    legal = get_all_legal_moves(board, RED)
    scored, *_ = score_all_legal_moves(board, RED)

    # Sanity: scored entries still carry symbolic_rank.
    assert "symbolic_rank" in (scored[0].get("facts") or {})

    st = CheckersState(
        board=board, current_player=RED, turn_number=0,
        legal_moves=scored, symbolic_scored_moves=scored,
    )
    user = build_ranker_user_prompt(st, scored, list(range(len(scored))))

    assert "minimax_score" in user, "B4 prompt must still expose minimax_score"
    assert "symbolic_rank" not in user, "B4 prompt must NOT expose symbolic_rank"

    # The underlying move objects must NOT have been mutated.
    assert "symbolic_rank" in (scored[0].get("facts") or {}), (
        "build_ranker_user_prompt mutated the underlying move facts"
    )

    # The single-candidate variant must also hide symbolic_rank.
    user_single = build_ranker_user_prompt_single(st, scored[0])
    assert "minimax_score" in user_single
    assert "symbolic_rank" not in user_single
    assert "symbolic_rank" in (scored[0].get("facts") or {})

    # The retry-prompt builder must also hide symbolic_rank.
    user_retry = _build_retry_user_prompt(
        state=st,
        move_list=scored,
        index_map=list(range(len(scored))),
        feedback_str="(no feedback in this test)",
        system_prompt="",
    )
    assert "minimax_score" in user_retry
    assert "symbolic_rank" not in user_retry
    assert "symbolic_rank" in (scored[0].get("facts") or {})


def test_b4_selection_logic_paths_unchanged():
    """
    Sanity: the helpers that drive selection / override / fallback still
    look up symbolic_rank/minimax_score directly from move facts (NOT from
    the prompt), so hiding symbolic_rank in the prompt does not affect
    decision behaviour.
    """
    import inspect
    from checkers.agents import ranker_agent

    # Decision-relevant helpers continue to reference minimax_score via
    # _get_minimax_score, and _best_and_second_best_minimax compares those
    # scores — neither path consults the rendered prompt text.
    for fn_name in (
        "_get_minimax_score",
        "_best_and_second_best_minimax",
        "_choose_best_minimax_with_origin",
        "_override_if_llm_chose_much_worse_minimax",
        "_apply_safety_filter",
    ):
        fn = getattr(ranker_agent, fn_name)
        src = inspect.getsource(fn)
        # Each helper either reads facts directly or delegates — none of them
        # parses the LLM prompt text.
        assert "build_ranker_user_prompt" not in src, (
            f"{fn_name} must not depend on the rendered prompt text"
        )


# ── 6. Baseline is registered + dispatch reaches the new runner ─────────────

def test_strategic_facts_baseline_in_suite_constants():
    assert BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS in _SUITE_BASELINES
    assert BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS in _ALL_SUITE_BASELINES


# ── 7. B3e freeform-path baseline ───────────────────────────────────────────

def test_freeform_path_prompt_contains_legal_moves_and_facts():
    from checkers.engine.rules import get_all_legal_moves
    from checkers.agents.scorer_agent import score_all_legal_moves
    from checkers.baseline_eval.run_baseline_human_trace import (
        _build_rules_legal_llm_strategic_facts_freeform_path_user,
    )

    board = _board_with_capture()
    legal = get_all_legal_moves(board, RED)
    scored, *_ = score_all_legal_moves(board, RED)

    user = _build_rules_legal_llm_strategic_facts_freeform_path_user(
        board, legal, scored, 1,
    )
    # Every legal index marker appears.
    for i in range(len(legal)):
        assert f"[{i}]" in user, f"Prompt missing index marker [{i}]"
    # At least some core tactical fact fields are present.
    must_have = {"captures_count", "opponent_can_recapture", "results_in_king"}
    missing = [k for k in must_have if k not in user]
    assert not missing, (
        f"freeform-path prompt is missing tactical fact fields: {missing}"
    )


def test_freeform_path_prompt_hides_minimax_labels():
    from checkers.engine.rules import get_all_legal_moves
    from checkers.agents.scorer_agent import score_all_legal_moves
    from checkers.baseline_eval.run_baseline_human_trace import (
        _build_rules_legal_llm_strategic_facts_freeform_path_user,
    )

    board = _board_with_capture()
    legal = get_all_legal_moves(board, RED)
    scored, *_ = score_all_legal_moves(board, RED)

    user = _build_rules_legal_llm_strategic_facts_freeform_path_user(
        board, legal, scored, 1,
    )
    forbidden = [
        "minimax_score", "symbolic_rank", "rank", "score_gap",
        "best_score", "second_best_score",
        "best_move_index", "best_move_path",
    ]
    leaks = [tok for tok in forbidden if tok in user]
    assert not leaks, f"freeform-path prompt leaks minimax labels: {leaks}"


def test_freeform_path_prompt_does_not_request_chosen_index():
    from checkers.engine.rules import get_all_legal_moves
    from checkers.agents.scorer_agent import score_all_legal_moves
    from checkers.baseline_eval.run_baseline_human_trace import (
        _RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH_SYSTEM,
        _build_rules_legal_llm_strategic_facts_freeform_path_user,
    )

    board = _board_with_capture()
    legal = get_all_legal_moves(board, RED)
    scored, *_ = score_all_legal_moves(board, RED)
    user = _build_rules_legal_llm_strategic_facts_freeform_path_user(
        board, legal, scored, 1,
    )
    system = _RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH_SYSTEM
    # System and user must require chosen_path. chosen_index may only appear
    # as part of an explicit "Do NOT output chosen_index" instruction.
    assert "chosen_path" in user and "chosen_path" in system
    for src_name, src in (("system", system), ("user", user)):
        for line in src.splitlines():
            if "chosen_index" in line:
                assert "Do NOT output chosen_index" in line or \
                       "Do not output chosen_index" in line, (
                    f"{src_name} prompt references chosen_index outside of a "
                    f"prohibition: {line!r}"
                )
    # The user-prompt schema example must list only chosen_path (+ reasoning).
    schema_lines = [
        ln for ln in user.splitlines()
        if '"chosen_path"' in ln and '"chosen_index"' in ln
    ]
    assert not schema_lines, (
        "user-prompt schema example should not contain chosen_index"
    )
    # Neither prompt should *request* move_analysis (it may appear only inside
    # an explicit "Do NOT output ... move_analysis" prohibition).
    for src_name, src in (("system", system), ("user", user)):
        for line in src.splitlines():
            if "move_analysis" in line:
                assert "Do NOT" in line or "Do not" in line, (
                    f"{src_name} prompt requests move_analysis: {line!r}"
                )


def test_b3_strategic_facts_baseline_unchanged():
    """
    Current B3 (strategic_facts with chosen_index + chosen_path) must still
    request move_analysis and chosen_index.
    """
    from checkers.baseline_eval.run_baseline_human_trace import (
        _RULES_LEGAL_LLM_STRATEGIC_FACTS_SYSTEM,
    )
    text = _RULES_LEGAL_LLM_STRATEGIC_FACTS_SYSTEM
    assert "move_analysis" in text
    assert "chosen_index" in text
    assert "chosen_path" in text


def test_b4_full_system_baseline_constant_unchanged():
    """Sanity: B4 constant string is exactly 'full_system'."""
    from checkers.baseline_eval.run_baseline_human_trace import (
        BASELINE_FULL_SYSTEM,
    )
    assert BASELINE_FULL_SYSTEM == "full_system"


def test_freeform_path_baseline_in_suite_constants():
    from checkers.baseline_eval.run_baseline_human_trace import (
        BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH,
    )
    assert (
        BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH
        in _SUITE_BASELINES
    )
    assert (
        BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS_FREEFORM_PATH
        in _ALL_SUITE_BASELINES
    )


def test_strategic_facts_dispatch_resolves_without_calling_llm(monkeypatch):
    import json
    from checkers.baseline_eval import run_baseline_scenario_suite as rs

    # Stub the LLM with a fully-compliant response (covers every legal index).
    def _stub_call(system, user):
        # crude parse: count "[<n>] " markers to infer N legal moves.
        import re
        idxs = [int(m) for m in re.findall(r"\[(\d+)\]", user)]
        n = max(idxs) + 1 if idxs else 1
        analysis = [{"index": i, "pros": "", "cons": "", "verdict": "ok"}
                    for i in range(n)]
        return json.dumps({
            "move_analysis": analysis,
            "chosen_index":  0,
            "chosen_path":   [[0, 0]],
            "reasoning":     "stub",
        })
    monkeypatch.setattr(rs, "call_baseline_llm", _stub_call)

    board = _board_with_capture()
    rec = run_scenario_for_baseline(
        scenario_name="mandatory_capture",
        board=board,
        baseline=BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS,
        show_prompts=False,
    )
    assert rec["baseline"] == BASELINE_RULES_LEGAL_LLM_STRATEGIC_FACTS
    assert "error_class" in rec
    assert "move_analysis_complete" in rec
