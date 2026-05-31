# checkers/tests/test_phase5b_reasoning_quality.py
#
# Phase 5B regression tests — pure-deterministic, no LLM calls.
#
# Covers BUG-3 through BUG-10 from the Phase 5B semantic-quality fixes.
# Each class is self-contained and documents which bug it covers.
# All tests use only: _check_reasoning_truthfulness, _build_grounded_reasoning_seeds,
# _build_seed_reasoning_prompt, RANKER_SEED_REASONING_SYSTEM, verify_all /
# contradictions_only from unified_verifier, ABSOLUTE_FORBIDDEN_VOCAB.

from __future__ import annotations

from typing import Optional

import pytest

from checkers.agents.ranker_agent import (
    _build_grounded_reasoning_seeds,
    _build_seed_reasoning_prompt,
    _check_reasoning_truthfulness,
    RANKER_SEED_REASONING_SYSTEM,
)
from checkers.evaluation.forbidden_vocab import ABSOLUTE_FORBIDDEN_VOCAB
from checkers.evaluation.unified_verifier import (
    contradictions_only,
    contradiction_strings,
    verify_all,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

def _rt(text: str, facts: dict, seeds: Optional[list] = None) -> list[str]:
    """Shortcut: runtime checker."""
    return _check_reasoning_truthfulness(text, facts, seeds=seeds or [])


def _ev(text: str, facts: dict, seeds: Optional[list] = None) -> list[str]:
    """Shortcut: evaluator contradiction_strings."""
    return contradiction_strings(text, reasoning_seeds=seeds or [], facts=facts)


def _has(warnings: list[str], fragment: str) -> bool:
    return any(fragment in w for w in warnings)


def _clean(warnings: list[str]) -> bool:
    return not any("REASONING_CONTRADICTION" in w for w in warnings)


# ── BUG-3: Single-legal-move context ──────────────────────────────────────────

_SLM_FACTS = {
    "opponent_can_recapture": False,
    "captures_count": 0,
    "net_gain": 0,
    "minimax_score": -3.0,
}

_SLM_SEEDS = ["This is the only legal move available; the engine assigns it a minimax score of -3.0."]


class TestBug3SingleLegalMoveSuperlatives:
    """BUG-3: 'best move', 'strongest choice', 'highest-ranked option' must be
    flagged when the seeds signal a single-legal-move context."""

    def test_best_move_flagged_runtime(self):
        ws = _rt("This is the best move in this position.", _SLM_FACTS, _SLM_SEEDS)
        assert _has(ws, "single_legal_move_context")

    def test_strongest_choice_flagged_runtime(self):
        ws = _rt("The engine selected the strongest choice.", _SLM_FACTS, _SLM_SEEDS)
        assert _has(ws, "single_legal_move_context")

    def test_highest_ranked_flagged_runtime(self):
        ws = _rt("This is the highest-ranked option.", _SLM_FACTS, _SLM_SEEDS)
        assert _has(ws, "single_legal_move_context")

    def test_best_move_flagged_evaluator(self):
        ws = _ev("This is the best move in this position.", _SLM_FACTS, _SLM_SEEDS)
        assert _has(ws, "single_legal_move_context")

    def test_runtime_evaluator_agree_slm(self):
        text = "This is the best move available."
        rt = _rt(text, _SLM_FACTS, _SLM_SEEDS)
        ev = _ev(text, _SLM_FACTS, _SLM_SEEDS)
        assert bool(rt) == bool(ev), "E.1 invariant: both sides must agree"

    def test_no_slm_seed_no_flag(self):
        """Without the only-legal-move seed, 'best move' must NOT be flagged."""
        seeds = ["opponent cannot recapture next turn"]
        ws = _rt("This is the best move available.", _SLM_FACTS, seeds)
        assert not _has(ws, "single_legal_move_context")

    def test_seed_builder_injects_slm_seed_for_single_candidate(self):
        move = {"path": [[5, 4], [4, 3]], "type": "simple", "facts": _SLM_FACTS,
                "minimax_score": -3.0}
        seeds = _build_grounded_reasoning_seeds(move, [move])
        assert any("only legal move" in s.lower() for s in seeds)

    def test_seed_builder_no_slm_seed_for_multiple_candidates(self):
        move = {"path": [[5, 4], [4, 3]], "type": "simple", "facts": _SLM_FACTS,
                "minimax_score": -3.0}
        alt  = {"path": [[5, 2], [4, 1]], "type": "simple", "facts": _SLM_FACTS,
                "minimax_score": -5.0}
        seeds = _build_grounded_reasoning_seeds(move, [move, alt])
        assert not any("only legal move" in s.lower() for s in seeds)

    def test_seed_builder_no_standalone_minimax_seed_for_single_candidate(self):
        """The standalone 'The engine scores this move X' seed must NOT appear
        alongside the combined 'only legal move' seed."""
        move = {"path": [[5, 4], [4, 3]], "type": "simple", "facts": _SLM_FACTS,
                "minimax_score": -3.0}
        seeds = _build_grounded_reasoning_seeds(move, [move])
        standalone = [s for s in seeds if s.startswith("The engine scores this move")]
        assert standalone == [], "standalone minimax seed must be absent in SLM context"


# ── BUG-4: center_control=False → "center of the board" strategic claim ────────

_CTR_FACTS_FALSE = {
    "opponent_can_recapture": False,
    "captures_count": 0,
    "net_gain": 0,
    "center_control": False,
}

_CTR_FACTS_TRUE = {**_CTR_FACTS_FALSE, "center_control": True}


class TestBug4CenterOfBoardStrategic:
    """BUG-4: 'center of the board' as a strategic claim must be caught when
    center_control=False and the phrase is not in seeds."""

    def test_flagged_runtime_no_seed(self):
        ws = _rt("This move controls the center of the board.", _CTR_FACTS_FALSE, [])
        assert _has(ws, "center of the board")

    def test_flagged_evaluator_no_seed(self):
        ws = _ev("This move controls the center of the board.", _CTR_FACTS_FALSE, [])
        assert _has(ws, "center of the board")

    def test_runtime_evaluator_agree(self):
        text = "This advances to the center of the board."
        rt = _rt(text, _CTR_FACTS_FALSE, [])
        ev = _ev(text, _CTR_FACTS_FALSE, [])
        assert bool(rt) == bool(ev)

    def test_seed_exempt_when_geometric_seed_present(self):
        """When the geometric seed introduces 'center of the board', the phrase
        is allowed in the reasoning even when center_control=False."""
        seeds = ["The destination is in the center of the board (column 3)."]
        ws = _rt("The piece moves to the center of the board.", _CTR_FACTS_FALSE, seeds)
        assert not _has(ws, "center of the board")

    def test_not_flagged_when_center_control_true(self):
        ws = _rt("This move controls the center of the board.", _CTR_FACTS_TRUE, [])
        assert not _has(ws, "center of the board")

    def test_existing_phrases_still_flagged(self):
        """Existing center phrases ('controls the center') must still fire."""
        ws = _rt("This move controls the center.", _CTR_FACTS_FALSE, [])
        assert _has(ws, "center_control=false")


# ── BUG-5: Tradeoff weighting — prompt-only enforcement ───────────────────────

class TestBug5TradeoffPhrasesInPrompt:
    """BUG-5: System prompt must explicitly prohibit tradeoff-weighting phrases."""

    def test_outweighs_banned_in_prompt(self):
        assert "outweighs" in RANKER_SEED_REASONING_SYSTEM

    def test_compensates_banned_in_prompt(self):
        assert "compensates for" in RANKER_SEED_REASONING_SYSTEM

    def test_justifies_risk_banned_in_prompt(self):
        assert "justifies the risk" in RANKER_SEED_REASONING_SYSTEM

    def test_balances_out_banned_in_prompt(self):
        assert "balances out" in RANKER_SEED_REASONING_SYSTEM


# ── BUG-6: Mobility disadvantage overclaim ────────────────────────────────────

_MOB_DISADV_FACTS = {
    "opponent_can_recapture": False,
    "captures_count": 0,
    "net_gain": 0,
    "opponent_mobility_after": 10,
    "our_mobility_after": 6,
}

_MOB_EQUAL_FACTS = {**_MOB_DISADV_FACTS, "opponent_mobility_after": 6}


class TestBug6MobilityDisadvantageOverclaim:
    """BUG-6: when opponent_mobility_after > our_mobility_after, 'solves/addresses/
    fixes/eliminates the disadvantage' must be flagged by both sides."""

    def test_solves_disadvantage_flagged_runtime(self):
        ws = _rt("This move solves the disadvantage.", _MOB_DISADV_FACTS)
        assert _has(ws, "overclaims mobility")

    def test_addresses_disadvantage_flagged_runtime(self):
        ws = _rt("This addresses the disadvantage.", _MOB_DISADV_FACTS)
        assert _has(ws, "overclaims mobility")

    def test_eliminates_mobility_gap_flagged_runtime(self):
        ws = _rt("The move eliminates the mobility gap.", _MOB_DISADV_FACTS)
        assert _has(ws, "overclaims mobility")

    def test_solves_disadvantage_flagged_evaluator(self):
        ws = _ev("This move solves the disadvantage.", _MOB_DISADV_FACTS)
        assert _has(ws, "overclaims mobility")

    def test_runtime_evaluator_agree(self):
        text = "This fixes the disadvantage entirely."
        rt = _rt(text, _MOB_DISADV_FACTS)
        ev = _ev(text, _MOB_DISADV_FACTS)
        assert bool(rt) == bool(ev)

    def test_not_flagged_when_no_disadvantage(self):
        ws = _rt("This solves the disadvantage.", _MOB_EQUAL_FACTS)
        assert not _has(ws, "overclaims mobility")

    def test_narrows_the_gap_not_flagged(self):
        ws = _rt("This move narrows the mobility gap.", _MOB_DISADV_FACTS)
        assert not _has(ws, "overclaims mobility")


# ── BUG-7: Causal center-control seed ─────────────────────────────────────────

_BUG7_FACTS_WITH_FROZEN = {
    "opponent_can_recapture": False,
    "captures_count": 0,
    "net_gain": 0,
    "center_control": True,
    "frozen_enemy_pieces": 3,
}

_BUG7_FACTS_NO_FROZEN = {**_BUG7_FACTS_WITH_FROZEN, "frozen_enemy_pieces": 0}


class TestBug7CausalCenterSeed:
    """BUG-7: when center_control=True and frozen_enemy_pieces > 0, the center
    seed must be causal (tie central control to the restriction count)."""

    def _seeds(self, facts: dict) -> list[str]:
        move = {"path": [[5, 4], [4, 3]], "type": "simple",
                "facts": facts, "minimax_score": 1.0}
        alt  = {"path": [[5, 2], [4, 1]], "type": "simple",
                "facts": {"opponent_can_recapture": True, "captures_count": 0,
                          "net_gain": 0}, "minimax_score": 0.0}
        return _build_grounded_reasoning_seeds(move, [move, alt])

    def test_causal_seed_injected_when_frozen(self):
        seeds = self._seeds(_BUG7_FACTS_WITH_FROZEN)
        center_seeds = [s for s in seeds if "central control" in s.lower()
                        or "central board control" in s.lower()]
        assert center_seeds, "expected a center-control seed"
        assert any("restrict" in s.lower() or "3" in s for s in center_seeds), (
            "center seed must be causal (reference restriction count) "
            f"when frozen_enemy_pieces=3; got {center_seeds}"
        )

    def test_generic_seed_when_no_frozen(self):
        seeds = self._seeds(_BUG7_FACTS_NO_FROZEN)
        center_seeds = [s for s in seeds if "central" in s.lower()
                        or "center" in s.lower()]
        assert center_seeds, "expected a center seed"
        # No frozen count in the seed when frozen == 0
        assert not any("restrict" in s.lower() for s in center_seeds), (
            "center seed must NOT include restriction language when frozen=0"
        )


# ── BUG-8: Ally-support seed wording ──────────────────────────────────────────

_BUG8_FACTS = {
    "opponent_can_recapture": False,
    "captures_count": 0,
    "net_gain": 0,
    "leaves_piece_isolated": False,
}


class TestBug8AllySupportSeedWording:
    """BUG-8: the ally-support seed must say 'not left isolated', not
    'stays supported by adjacent allies'."""

    def _seeds(self) -> list[str]:
        move = {"path": [[5, 4], [4, 3]], "type": "simple",
                "facts": _BUG8_FACTS, "minimax_score": 0.5}
        alt  = {"path": [[5, 2], [4, 1]], "type": "simple",
                "facts": {"opponent_can_recapture": True, "captures_count": 0,
                          "net_gain": 0}, "minimax_score": 0.0}
        return _build_grounded_reasoning_seeds(move, [move, alt])

    def test_new_wording_present(self):
        seeds = self._seeds()
        assert any("not left isolated" in s.lower() for s in seeds), (
            f"expected 'not left isolated' seed; got {seeds}"
        )

    def test_old_wording_absent(self):
        seeds = self._seeds()
        assert not any("stays supported by adjacent allies" in s.lower() for s in seeds), (
            "old ally-support wording must not appear after BUG-8 fix"
        )

    def test_inversion_check_uses_new_seed_wording(self):
        """The inversion detection must fire when new-wording seed says
        'not left isolated' but reasoning says 'piece is isolated'."""
        seeds = ["The moved piece is not left isolated."]
        ws = _rt("the piece is isolated after this move", _BUG8_FACTS, seeds)
        inversion = [w for w in ws if "inversion detected" in w]
        assert inversion, "inversion must fire for 'piece is isolated' against new seed"


# ── BUG-9: Hollow strategic filler ────────────────────────────────────────────

class TestBug9HollowStrategicFiller:
    """BUG-9: 'tangible positional advantage' and 'strong positional edge' must
    be caught by ABSOLUTE_FORBIDDEN_VOCAB (both runtime and evaluator)."""

    _FACTS = {"opponent_can_recapture": False, "captures_count": 0, "net_gain": 0}

    def test_tangible_positional_advantage_in_forbidden_list(self):
        assert "tangible positional advantage" in ABSOLUTE_FORBIDDEN_VOCAB

    def test_strong_positional_edge_in_forbidden_list(self):
        assert "strong positional edge" in ABSOLUTE_FORBIDDEN_VOCAB

    def test_tangible_positional_advantage_runtime(self):
        ws = _rt("This gives us a tangible positional advantage.", self._FACTS)
        assert any("tangible positional advantage" in w for w in ws)

    def test_strong_positional_edge_runtime(self):
        ws = _rt("The move creates a strong positional edge.", self._FACTS)
        assert any("strong positional edge" in w for w in ws)

    def test_tangible_positional_advantage_evaluator(self):
        ws = _ev("This gives us a tangible positional advantage.", self._FACTS)
        assert any("tangible positional advantage" in w for w in ws)

    def test_runtime_evaluator_agree_hollow_filler(self):
        text = "This creates a strong positional edge for us."
        rt = _rt(text, self._FACTS)
        ev = _ev(text, self._FACTS)
        assert bool(rt) == bool(ev)

    def test_hollow_filler_prohibited_in_prompt(self):
        assert "tangible positional advantage" in RANKER_SEED_REASONING_SYSTEM
        assert "improved position" in RANKER_SEED_REASONING_SYSTEM
        assert "strong positional edge" in RANKER_SEED_REASONING_SYSTEM


# ── BUG-10: Coordinate reference in prompt ────────────────────────────────────

class TestBug10CoordinateReferenceInPrompt:
    """BUG-10: the user prompt built by _build_seed_reasoning_prompt must
    instruct the LLM to reference the move path coordinates."""

    _MOVE = {"path": [[5, 4], [4, 3]], "type": "simple",
             "facts": {"opponent_can_recapture": False, "captures_count": 0,
                       "net_gain": 0}}
    _SEEDS = ["The moved piece cannot be immediately recaptured."]

    def test_coordinate_hint_in_user_prompt(self):
        prompt = _build_seed_reasoning_prompt(self._MOVE, self._SEEDS)
        assert "REQUIRED" in prompt, "prompt must contain a REQUIRED coordinate instruction"
        assert str(self._MOVE["path"]) in prompt or "path" in prompt.lower()

    def test_coordinate_rule_in_system_prompt(self):
        assert "coordinate" in RANKER_SEED_REASONING_SYSTEM.lower() or \
               "path" in RANKER_SEED_REASONING_SYSTEM.lower(), (
            "system prompt must mention coordinate or path reference requirement"
        )
