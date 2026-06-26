"""
Phase-H regression tests for the two surgical fixes from the final_frozen_eval audit.

FIX 1 — Center seed gating (center_control guard)
    The seed "The destination is in the center of the board (column X)" must be
    emitted ONLY when center_control=True.  Previously it fired for any move to
    columns 2-5 regardless of center_control, injecting a false claim that also
    bypassed the verifier's seed-exempt check.

FIX 2 — Repair prompt vague-positional-language suppression
    Both _build_refinement_prompt and _build_targeted_refinement_prompt now
    contain an explicit instruction forbidding 'board control', 'piece
    coordination', 'board cohesion', 'connectivity' unless grounded in facts.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from checkers.agents.explainer_agent import (
    _build_grounded_reasoning_seeds,
    _build_refinement_prompt,
    _build_targeted_refinement_prompt,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CENTER_PHRASE = "The destination is in the center of the board"

def _make_facts(**overrides):
    """Minimal fact dict for seed generation tests."""
    base = {
        "move_type": "simple",
        "piece_type_moving": "regular",
        "captures_count": 0,
        "jump_count": 0,
        "is_multi_jump": False,
        "results_in_king": False,
        "near_promotion": False,
        "our_pieces_before": {"total": 12, "regular": 12, "kings": 0},
        "our_pieces_after":  {"total": 12, "regular": 12, "kings": 0},
        "opp_pieces_before": {"total": 12, "regular": 12, "kings": 0},
        "opp_pieces_after":  {"total": 12, "regular": 12, "kings": 0},
        "net_gain": 0,
        "material_advantage": 0,
        "center_control": False,
        "opponent_can_recapture": False,
        "moved_piece_is_threatened": False,
        "leaves_piece_isolated": False,
        "any_piece_isolated": False,
        "forced_opponent_jump_reply": False,
        "creates_immediate_threat": False,
        "opponent_mobility_before": 7,
        "opponent_mobility_after": 7,
        "our_mobility_before": 7,
        "our_mobility_after": 8,
        "mobility_reduction": 0,
        "frozen_enemy_pieces": 0,
        "restriction_score": 0,
        "minimax_score": -2.0,
        "symbolic_rank": 1,
        "weakens_king_row": False,
        "shot_sequence_available": False,
        "forces_exchange": False,
        "two_for_one_potential": False,
        "opponent_jump_count": 0,
        "opponent_near_promotion": False,
        "unsafe_simple_move": False,
        "our_pieces_threatened_before": 0,
        "our_pieces_threatened_after": 0,
        "king_activity_score": 0,
        "quiet_move_role": "STRUCTURAL",
        "blocks_opponent_landing": False,
        "can_be_recaptured": False,
        "improves_trade_conversion": False,
    }
    base.update(overrides)
    return base


def _make_candidate(facts, path=None):
    return {
        "path": path or [[5, 6], [4, 5]],
        "type": facts.get("move_type", "simple"),
        "score": facts.get("minimax_score", -2.0),
        "facts": facts,
    }


def _seeds_for(facts, path=None):
    cand = _make_candidate(facts, path)
    return _build_grounded_reasoning_seeds(cand, [cand])


# ---------------------------------------------------------------------------
# FIX 1: Center seed gating
# ---------------------------------------------------------------------------

class TestCenterSeedGating:

    # ── False cases (center_control=False) ──────────────────────────────────

    def test_no_center_seed_col2_center_control_false(self):
        """column 2 ∈ {2,3,4,5} but center_control=False → no center seed."""
        facts = _make_facts(center_control=False)
        seeds = _seeds_for(facts, path=[[5, 4], [4, 2]])
        center_seeds = [s for s in seeds if _CENTER_PHRASE in s]
        assert center_seeds == [], (
            f"Expected no center seed but got: {center_seeds}"
        )

    def test_no_center_seed_col3_center_control_false(self):
        """column 3, center_control=False → no center seed."""
        facts = _make_facts(center_control=False)
        seeds = _seeds_for(facts, path=[[5, 2], [4, 3]])
        center_seeds = [s for s in seeds if _CENTER_PHRASE in s]
        assert center_seeds == []

    def test_no_center_seed_col4_center_control_false(self):
        """column 4, center_control=False → no center seed."""
        facts = _make_facts(center_control=False)
        seeds = _seeds_for(facts, path=[[5, 6], [4, 4]])
        center_seeds = [s for s in seeds if _CENTER_PHRASE in s]
        assert center_seeds == []

    def test_no_center_seed_col5_center_control_false(self):
        """column 5, center_control=False → no center seed (the main audit case)."""
        facts = _make_facts(center_control=False)
        seeds = _seeds_for(facts, path=[[6, 5], [5, 4]])
        center_seeds = [s for s in seeds if _CENTER_PHRASE in s]
        assert center_seeds == []

    # ── True cases (center_control=True) ────────────────────────────────────

    def test_center_seed_emitted_col3_center_control_true(self):
        """center_control=True, column 3 → center seed with column number."""
        facts = _make_facts(center_control=True, frozen_enemy_pieces=0)
        seeds = _seeds_for(facts, path=[[5, 2], [4, 3]])
        center_seeds = [s for s in seeds if _CENTER_PHRASE in s]
        assert len(center_seeds) == 1
        assert "column 3" in center_seeds[0]

    def test_center_seed_emitted_col5_center_control_true(self):
        """center_control=True, column 5 → center seed emitted correctly."""
        facts = _make_facts(center_control=True, frozen_enemy_pieces=0)
        seeds = _seeds_for(facts, path=[[5, 6], [4, 5]])
        center_seeds = [s for s in seeds if _CENTER_PHRASE in s]
        assert len(center_seeds) == 1
        assert "column 5" in center_seeds[0]

    def test_center_seed_column_number_matches_destination(self):
        """Center seed column number exactly matches the destination column."""
        facts = _make_facts(center_control=True, frozen_enemy_pieces=0)
        seeds = _seeds_for(facts, path=[[5, 2], [4, 4]])
        center_seeds = [s for s in seeds if _CENTER_PHRASE in s]
        assert len(center_seeds) == 1
        assert "column 4" in center_seeds[0]

    # ── Edge / non-center columns unaffected ────────────────────────────────

    def test_no_center_seed_edge_column_0(self):
        """Edge column 0, center_control=False → no center seed."""
        facts = _make_facts(center_control=False)
        seeds = _seeds_for(facts, path=[[5, 1], [6, 0]])
        center_seeds = [s for s in seeds if _CENTER_PHRASE in s]
        assert center_seeds == []

    def test_no_center_seed_edge_column_7(self):
        """Edge column 7, center_control=False → no center seed."""
        facts = _make_facts(center_control=False)
        seeds = _seeds_for(facts, path=[[5, 6], [6, 7]])
        center_seeds = [s for s in seeds if _CENTER_PHRASE in s]
        assert center_seeds == []

    # ── Verifier bypass eliminated ───────────────────────────────────────────

    def test_verifier_catches_center_claim_when_no_center_seed(self):
        """Without the false center seed, a center claim is caught by verifier."""
        from checkers.evaluation.unified_verifier import verify_all
        from checkers.evaluation.unified_verifier import ClaimStatus

        para = (
            "The move advances a piece to (4, 5), landing in the center of the board "
            "and improving piece placement without capturing."
        )
        facts = _make_facts(center_control=False)
        # seeds contain no center-of-board phrase (as after the fix)
        seeds = ["The moved piece cannot be immediately recaptured."]

        results = verify_all(para, reasoning_seeds=seeds, facts=facts)
        center_results = [r for r in results if r.claim_type == "center_of_board_strategic"]
        assert len(center_results) == 1
        assert center_results[0].claim_status == ClaimStatus.CONTRADICTED

    def test_verifier_not_triggered_when_center_control_true(self):
        """center_control=True → no center_of_board_strategic contradiction."""
        from checkers.evaluation.unified_verifier import verify_all
        from checkers.evaluation.unified_verifier import ClaimStatus

        para = (
            "The move advances a piece to (4, 5), landing in the center of the board "
            "and improving piece placement."
        )
        facts_dict = dict(_make_facts(center_control=True))
        seeds = [_CENTER_PHRASE + " (column 5)."]

        results = verify_all(para, reasoning_seeds=seeds, facts=facts_dict)
        center_results = [r for r in results if r.claim_type == "center_of_board_strategic"]
        assert all(r.claim_status != ClaimStatus.CONTRADICTED for r in center_results)

    def test_false_center_seed_previously_bypassed_verifier(self):
        """Regression: when seeds contain the center phrase, verifier is bypassed."""
        from checkers.evaluation.unified_verifier import verify_all
        from checkers.evaluation.unified_verifier import ClaimStatus

        para = (
            "The move places the piece in the center of the board at column 3, "
            "improving its placement."
        )
        facts = _make_facts(center_control=False)
        # Simulate the OLD (broken) behavior: seed contains the center phrase
        bad_seed = "The destination is in the center of the board (column 3)."

        results = verify_all(para, reasoning_seeds=[bad_seed], facts=facts)
        center_results = [r for r in results if r.claim_type == "center_of_board_strategic"]
        # With the bad seed, verifier is bypassed — so contradiction is NOT raised
        assert all(r.claim_status != ClaimStatus.CONTRADICTED for r in center_results), (
            "This confirms that the old seed caused a verifier bypass. "
            "After FIX 1, this seed is no longer generated."
        )


# ---------------------------------------------------------------------------
# FIX 2: Repair prompt vague-positional language suppression
# ---------------------------------------------------------------------------

_VAGUE_VOCAB_INSTRUCTION = "Do not use vague positional descriptors"


class TestRepairPromptVagueVocabBan:

    def _make_minimal_move(self):
        facts = _make_facts()
        return {
            "path": [[5, 6], [4, 5]],
            "type": "simple",
            "score": -2.0,
            "facts": facts,
        }

    def test_full_repair_prompt_contains_vague_vocab_instruction(self):
        """_build_refinement_prompt includes the forbidden-vocab instruction."""
        move = self._make_minimal_move()
        prompt = _build_refinement_prompt(move, ["REASONING_CONTRADICTION: some error"])
        assert _VAGUE_VOCAB_INSTRUCTION in prompt, (
            "Full repair prompt missing forbidden-vocab instruction"
        )

    def test_full_repair_prompt_mentions_board_control(self):
        """The banned list in the full repair prompt names 'board control'."""
        move = self._make_minimal_move()
        prompt = _build_refinement_prompt(move, ["REASONING_CONTRADICTION: some error"])
        assert "board control" in prompt

    def test_full_repair_prompt_mentions_piece_coordination(self):
        """The banned list in the full repair prompt names 'piece coordination'."""
        move = self._make_minimal_move()
        prompt = _build_refinement_prompt(move, ["REASONING_CONTRADICTION: some error"])
        assert "coordination" in prompt

    def test_full_repair_prompt_mentions_board_cohesion(self):
        """The banned list in the full repair prompt names 'board cohesion'."""
        move = self._make_minimal_move()
        prompt = _build_refinement_prompt(move, ["REASONING_CONTRADICTION: some error"])
        assert "board cohesion" in prompt

    def test_full_repair_prompt_mentions_connectivity(self):
        """The banned list in the full repair prompt names 'connectivity'."""
        move = self._make_minimal_move()
        prompt = _build_refinement_prompt(move, ["REASONING_CONTRADICTION: some error"])
        assert "connectivity" in prompt

    def test_targeted_repair_prompt_contains_vague_vocab_instruction(self):
        """_build_targeted_refinement_prompt includes the forbidden-vocab instruction."""
        move = self._make_minimal_move()
        prompt = _build_targeted_refinement_prompt(
            move,
            bad_sentences=["The move improves board connectivity."],
            contradictions=["REASONING_CONTRADICTION: some error"],
        )
        assert _VAGUE_VOCAB_INSTRUCTION in prompt, (
            "Targeted repair prompt missing forbidden-vocab instruction"
        )

    def test_targeted_repair_prompt_mentions_board_control(self):
        """Targeted repair prompt's banned list names 'board control'."""
        move = self._make_minimal_move()
        prompt = _build_targeted_refinement_prompt(
            move,
            bad_sentences=["bad sentence"],
            contradictions=["REASONING_CONTRADICTION: some error"],
        )
        assert "board control" in prompt

    def test_targeted_repair_prompt_mentions_connectivity(self):
        """Targeted repair prompt's banned list names 'connectivity'."""
        move = self._make_minimal_move()
        prompt = _build_targeted_refinement_prompt(
            move,
            bad_sentences=["bad sentence"],
            contradictions=["REASONING_CONTRADICTION: some error"],
        )
        assert "connectivity" in prompt

    # ── Prompt structural integrity (no unintended changes) ─────────────────

    def test_full_repair_prompt_still_has_forced_move_rule(self):
        """FIX 2 did not accidentally remove the forced-move framing rule."""
        move = self._make_minimal_move()
        prompt = _build_refinement_prompt(move, ["REASONING_CONTRADICTION: forced"])
        assert "IF AND ONLY IF" in prompt
        assert "only legal move" in prompt

    def test_full_repair_prompt_still_has_seed_preservation_rule(self):
        """FIX 2 did not accidentally remove the seed-preservation instruction."""
        move = self._make_minimal_move()
        prompt = _build_refinement_prompt(move, ["REASONING_CONTRADICTION: some error"])
        assert "king-promotion" in prompt
        assert "mobility transitions" in prompt

    def test_targeted_repair_prompt_still_has_forced_move_rule(self):
        """FIX 2 did not accidentally remove the forced-move rule from targeted prompt."""
        move = self._make_minimal_move()
        prompt = _build_targeted_refinement_prompt(
            move,
            bad_sentences=["bad"],
            contradictions=["REASONING_CONTRADICTION: some error"],
        )
        assert "IF AND ONLY IF" in prompt
        assert "only legal move" in prompt

    def test_targeted_repair_prompt_still_has_seed_preservation_rule(self):
        """FIX 2 did not accidentally remove seed-preservation from targeted prompt."""
        move = self._make_minimal_move()
        prompt = _build_targeted_refinement_prompt(
            move,
            bad_sentences=["bad"],
            contradictions=["REASONING_CONTRADICTION: some error"],
        )
        assert "king-promotion" in prompt

    def test_reply_format_not_changed_full_prompt(self):
        """Full repair prompt still ends with the expected JSON reply format."""
        move = self._make_minimal_move()
        prompt = _build_refinement_prompt(move, ["REASONING_CONTRADICTION: x"])
        assert '{"reasoning":' in prompt

    def test_reply_format_not_changed_targeted_prompt(self):
        """Targeted repair prompt still ends with the expected JSON reply format."""
        move = self._make_minimal_move()
        prompt = _build_targeted_refinement_prompt(
            move, bad_sentences=["x"], contradictions=["REASONING_CONTRADICTION: x"]
        )
        assert '"replacements"' in prompt
