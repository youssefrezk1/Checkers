# checkers/tests/test_explainer_agent_step6.py
#
# Step 6 of the Comparative Reasoning v2 roadmap: integration tests for the
# env-gated comparative stage in _explain_chosen_move / explainer_agent.
#
# Coverage matrix:
#   - flag OFF → byte-identical output vs baseline (generate never called)
#   - flag ON + comparative succeeds → chosen + "\n\n" + comparative
#   - flag ON + comparative returns None → chosen only, no extra blank line
#   - flag ON + len(legal) < 3 gate → comparative skipped, chosen only
#   - flag ON + generate raises unexpectedly → chosen still returned safely
#   - diagnostics namespace: comparative_* keys present only when flag ON
#   - diagnostics namespace: no comparative_* when flag OFF
#   - ranker_diagnostics still contains all legacy keys regardless of flag
#   - _comparative_stage_enabled reads env var correctly
#   - chosen_move passes through unchanged in both flag states

from __future__ import annotations

import os
from typing import Any

import pytest

import checkers.agents.explainer_agent as ranker_module
from checkers.agents.comparative_reasoner import _COMPARATIVE_DIAGNOSTICS_KEYS
from checkers.engine.board import RED
from checkers.state.state import CheckersState


# ── Fixtures ──────────────────────────────────────────────────────────────────

_CHOSEN_TEXT = "Chosen move text."
_COMPARATIVE_TEXT = "Comparative reasoning text."


def _make_move(path: list, score: float = 0.5) -> dict[str, Any]:
    return {
        "path": path,
        "type": "simple",
        "captured": [],
        "facts": {
            "minimax_score": score,
            "opponent_can_recapture": False,
            "creates_immediate_threat": False,
            "shot_sequence_available": False,
            "captures_count": 0,
            "leaves_piece_isolated": False,
            "weakens_king_row": False,
            "our_pieces_threatened_after": 0,
            "results_in_king": False,
            "near_promotion": False,
            "opponent_mobility_before": 5,
            "opponent_mobility_after": 5,
        },
    }


def _make_state(n_legal: int = 4) -> CheckersState:
    moves = [_make_move([[5, i * 2], [4, i * 2 + 1]]) for i in range(n_legal)]
    return CheckersState(
        board=[[0] * 8 for _ in range(8)],
        current_player=RED,
        turn_number=10,
        legal_moves=moves,
        chosen_move=moves[0],
        chosen_move_score=0.5,
        unchosen_moves=moves[1:],
        strategic_context={
            "game_phase": "MIDGAME",
            "score_state": "EQUAL",
            "strategic_priorities": [],
        },
    )


def _mock_chosen_pipeline(monkeypatch, reasoning: str = _CHOSEN_TEXT) -> None:
    """Bypass the LLM-dependent chosen-move generation and refinement."""
    monkeypatch.setattr(
        ranker_module, "_generate_seeded_reasoning",
        lambda *a, **kw: (reasoning, ["seed1"]),
    )
    monkeypatch.setattr(
        ranker_module, "_check_reasoning_truthfulness",
        lambda *a, **kw: [],
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. _comparative_stage_enabled reads the env var
# ═══════════════════════════════════════════════════════════════════════════


class TestComparativeStageEnabled:
    def test_default_on(self, monkeypatch):
        monkeypatch.delenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", raising=False)
        assert ranker_module._comparative_stage_enabled() is True

    def test_enabled_by_1(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        assert ranker_module._comparative_stage_enabled() is True

    def test_enabled_by_true(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "true")
        assert ranker_module._comparative_stage_enabled() is True

    def test_enabled_by_yes(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "yes")
        assert ranker_module._comparative_stage_enabled() is True

    def test_enabled_by_on(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "on")
        assert ranker_module._comparative_stage_enabled() is True

    def test_disabled_by_0(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "0")
        assert ranker_module._comparative_stage_enabled() is False

    def test_disabled_by_false(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "false")
        assert ranker_module._comparative_stage_enabled() is False

    def test_disabled_by_empty_string(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "")
        assert ranker_module._comparative_stage_enabled() is False

    def test_case_insensitive_TRUE(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "TRUE")
        assert ranker_module._comparative_stage_enabled() is True


# ═══════════════════════════════════════════════════════════════════════════
# 2. Flag OFF → byte-identical output
# ═══════════════════════════════════════════════════════════════════════════


class TestFlagOff:
    def test_generate_never_called_when_flag_off(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "0")
        _mock_chosen_pipeline(monkeypatch)

        called = []

        def _assert_not_called(*a, **kw):
            called.append(True)
            return None

        monkeypatch.setattr(
            ranker_module, "generate_comparative_reasoning", _assert_not_called,
        )
        ranker_module.explainer_agent(_make_state())
        assert called == [], "generate_comparative_reasoning must not be called when flag is OFF"

    def test_last_move_reasoning_is_chosen_only(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "0")
        _mock_chosen_pipeline(monkeypatch)
        monkeypatch.setattr(
            ranker_module, "generate_comparative_reasoning", lambda *a, **kw: _COMPARATIVE_TEXT,
        )
        result = ranker_module.explainer_agent(_make_state())
        assert result["last_move_reasoning"] == _CHOSEN_TEXT

    def test_no_comparative_keys_in_diagnostics_when_flag_off(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "0")
        _mock_chosen_pipeline(monkeypatch)
        result = ranker_module.explainer_agent(_make_state())
        diag = result["explainer_diagnostics"]
        comparative_keys = [k for k in diag if k.startswith("comparative_")]
        assert comparative_keys == []

    def test_chosen_move_passes_through_unchanged(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "0")
        _mock_chosen_pipeline(monkeypatch)
        state = _make_state()
        original_path = state.chosen_move["path"]
        result = ranker_module.explainer_agent(state)
        assert result["chosen_move"]["path"] == original_path

    def test_ranker_diagnostics_has_legacy_keys(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "0")
        _mock_chosen_pipeline(monkeypatch)
        result = ranker_module.explainer_agent(_make_state())
        diag = result["explainer_diagnostics"]
        for key in ("final_choice_source", "reasoning_seeds", "run_tag"):
            assert key in diag, f"legacy key {key!r} missing from ranker_diagnostics"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Flag ON + comparative succeeds
# ═══════════════════════════════════════════════════════════════════════════


class TestFlagOnSuccess:
    def test_output_is_chosen_newline_newline_comparative(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)
        monkeypatch.setattr(
            ranker_module, "generate_comparative_reasoning",
            lambda *a, **kw: _COMPARATIVE_TEXT,
        )
        result = ranker_module.explainer_agent(_make_state())
        assert result["last_move_reasoning"] == _CHOSEN_TEXT + "\n\n" + _COMPARATIVE_TEXT

    def test_separator_is_exactly_double_newline(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)
        monkeypatch.setattr(
            ranker_module, "generate_comparative_reasoning",
            lambda *a, **kw: _COMPARATIVE_TEXT,
        )
        result = ranker_module.explainer_agent(_make_state())
        reasoning = result["last_move_reasoning"]
        assert "\n\n" in reasoning
        parts = reasoning.split("\n\n", 1)
        assert parts[0] == _CHOSEN_TEXT
        assert parts[1] == _COMPARATIVE_TEXT

    def test_comparative_keys_present_in_diagnostics(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)

        def _mock_gen(chosen, candidates, facts, *, diagnostics_out=None, **kw):
            if diagnostics_out is not None:
                for key in _COMPARATIVE_DIAGNOSTICS_KEYS:
                    diagnostics_out[key] = None
            return _COMPARATIVE_TEXT

        monkeypatch.setattr(ranker_module, "generate_comparative_reasoning", _mock_gen)
        result = ranker_module.explainer_agent(_make_state())
        diag = result["explainer_diagnostics"]
        for key in _COMPARATIVE_DIAGNOSTICS_KEYS:
            assert key in diag, f"comparative key {key!r} missing when flag is ON"

    def test_chosen_move_passes_through_unchanged(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)
        monkeypatch.setattr(
            ranker_module, "generate_comparative_reasoning",
            lambda *a, **kw: _COMPARATIVE_TEXT,
        )
        state = _make_state()
        original_path = state.chosen_move["path"]
        result = ranker_module.explainer_agent(state)
        assert result["chosen_move"]["path"] == original_path

    def test_legacy_diagnostics_keys_still_present(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)
        monkeypatch.setattr(
            ranker_module, "generate_comparative_reasoning",
            lambda *a, **kw: _COMPARATIVE_TEXT,
        )
        result = ranker_module.explainer_agent(_make_state())
        diag = result["explainer_diagnostics"]
        for key in ("final_choice_source", "reasoning_seeds", "run_tag"):
            assert key in diag, f"legacy key {key!r} missing when comparative is ON"

    def test_generate_called_with_chosen_and_legal(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)
        state = _make_state(n_legal=4)
        captured: dict = {}

        def _mock_gen(chosen, candidates, facts, *, diagnostics_out=None, **kw):
            captured["chosen"] = chosen
            captured["candidates"] = candidates
            return _COMPARATIVE_TEXT

        monkeypatch.setattr(ranker_module, "generate_comparative_reasoning", _mock_gen)
        ranker_module.explainer_agent(state)
        assert captured["chosen"] is state.chosen_move
        assert captured["candidates"] is state.legal_moves


# ═══════════════════════════════════════════════════════════════════════════
# 4. Flag ON + comparative returns None
# ═══════════════════════════════════════════════════════════════════════════


class TestFlagOnComparativeNone:
    def test_output_is_chosen_only_no_extra_newline(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)
        monkeypatch.setattr(
            ranker_module, "generate_comparative_reasoning",
            lambda *a, **kw: None,
        )
        result = ranker_module.explainer_agent(_make_state())
        assert result["last_move_reasoning"] == _CHOSEN_TEXT

    def test_no_trailing_newline_in_output(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)
        monkeypatch.setattr(
            ranker_module, "generate_comparative_reasoning",
            lambda *a, **kw: None,
        )
        result = ranker_module.explainer_agent(_make_state())
        assert not result["last_move_reasoning"].endswith("\n")

    def test_chosen_move_unchanged(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)
        monkeypatch.setattr(
            ranker_module, "generate_comparative_reasoning",
            lambda *a, **kw: None,
        )
        state = _make_state()
        original_path = state.chosen_move["path"]
        result = ranker_module.explainer_agent(state)
        assert result["chosen_move"]["path"] == original_path


# ═══════════════════════════════════════════════════════════════════════════
# 5. len(legal) < 3 gate
# ═══════════════════════════════════════════════════════════════════════════


class TestLegalCountGate:
    def test_comparative_skipped_when_fewer_than_3_legal(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)
        called = []

        def _assert_not_called(*a, **kw):
            called.append(True)
            return _COMPARATIVE_TEXT

        monkeypatch.setattr(
            ranker_module, "generate_comparative_reasoning", _assert_not_called,
        )
        ranker_module.explainer_agent(_make_state(n_legal=2))
        assert called == []

    def test_output_is_chosen_only_when_gate_blocks(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)
        monkeypatch.setattr(
            ranker_module, "generate_comparative_reasoning",
            lambda *a, **kw: _COMPARATIVE_TEXT,
        )
        result = ranker_module.explainer_agent(_make_state(n_legal=2))
        assert result["last_move_reasoning"].startswith(_CHOSEN_TEXT)
        assert "The only alternative" in result["last_move_reasoning"]

    def test_comparative_runs_with_exactly_3_legal(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)
        called = []

        def _capture(*a, **kw):
            called.append(True)
            return _COMPARATIVE_TEXT

        monkeypatch.setattr(ranker_module, "generate_comparative_reasoning", _capture)
        ranker_module.explainer_agent(_make_state(n_legal=3))
        assert called == [True]


# ═══════════════════════════════════════════════════════════════════════════
# 6. Comparative failure never blocks chosen output
# ═══════════════════════════════════════════════════════════════════════════


class TestComparativeFailureIsolation:
    def test_unexpected_exception_does_not_propagate(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)

        def _raise(*a, **kw):
            raise RuntimeError("unexpected comparative failure")

        monkeypatch.setattr(ranker_module, "generate_comparative_reasoning", _raise)
        # Must not raise; chosen output still returned.
        result = ranker_module.explainer_agent(_make_state())
        assert result["last_move_reasoning"] == _CHOSEN_TEXT

    def test_chosen_move_returned_after_exception(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)

        def _raise(*a, **kw):
            raise ValueError("mock error")

        monkeypatch.setattr(ranker_module, "generate_comparative_reasoning", _raise)
        state = _make_state()
        result = ranker_module.explainer_agent(state)
        assert result["chosen_move"] is state.chosen_move

    def test_return_dict_complete_after_exception(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)
        monkeypatch.setattr(
            ranker_module, "generate_comparative_reasoning",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        result = ranker_module.explainer_agent(_make_state())
        for key in (
            "chosen_move", "last_move_reasoning", "explainer_diagnostics",
            "last_completed_node",
        ):
            assert key in result, f"return key {key!r} missing after comparative exception"


# ═══════════════════════════════════════════════════════════════════════════
# 7. Diagnostics namespace isolation
# ═══════════════════════════════════════════════════════════════════════════


class TestDiagnosticsNamespace:
    def test_flag_off_zero_comparative_keys(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "0")
        _mock_chosen_pipeline(monkeypatch)
        result = ranker_module.explainer_agent(_make_state())
        bad = [k for k in result["explainer_diagnostics"] if k.startswith("comparative_")]
        assert bad == []

    def test_flag_on_comparative_keys_present(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)

        def _populate_diag(chosen, candidates, facts, *, diagnostics_out=None, **kw):
            if diagnostics_out is not None:
                for key in _COMPARATIVE_DIAGNOSTICS_KEYS:
                    diagnostics_out[key] = f"value_{key}"
            return _COMPARATIVE_TEXT

        monkeypatch.setattr(ranker_module, "generate_comparative_reasoning", _populate_diag)
        result = ranker_module.explainer_agent(_make_state())
        diag = result["explainer_diagnostics"]
        missing = [k for k in _COMPARATIVE_DIAGNOSTICS_KEYS if k not in diag]
        assert missing == [], f"comparative keys missing: {missing}"

    def test_comparative_keys_do_not_overwrite_legacy_keys(self, monkeypatch):
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)

        def _populate_diag(chosen, candidates, facts, *, diagnostics_out=None, **kw):
            if diagnostics_out is not None:
                for key in _COMPARATIVE_DIAGNOSTICS_KEYS:
                    diagnostics_out[key] = None
            return _COMPARATIVE_TEXT

        monkeypatch.setattr(ranker_module, "generate_comparative_reasoning", _populate_diag)
        result = ranker_module.explainer_agent(_make_state())
        diag = result["explainer_diagnostics"]
        # Legacy keys must still be present and not wiped.
        assert diag["final_choice_source"] == "proposal_authoritative"
        assert diag["run_tag"] in ("seed_on", "seed_off")

    def test_ranker_diagnostics_always_has_legacy_count(self, monkeypatch):
        # With flag ON: should have both legacy AND comparative keys.
        monkeypatch.setenv("EXPLAINER_COMPARATIVE_STAGE_ENABLED", "1")
        _mock_chosen_pipeline(monkeypatch)

        def _populate_diag(chosen, candidates, facts, *, diagnostics_out=None, **kw):
            if diagnostics_out is not None:
                for key in _COMPARATIVE_DIAGNOSTICS_KEYS:
                    diagnostics_out[key] = None
            return _COMPARATIVE_TEXT

        monkeypatch.setattr(ranker_module, "generate_comparative_reasoning", _populate_diag)
        result = ranker_module.explainer_agent(_make_state())
        diag = result["explainer_diagnostics"]
        comparative_count = sum(1 for k in diag if k.startswith("comparative_"))
        assert comparative_count == len(_COMPARATIVE_DIAGNOSTICS_KEYS)
