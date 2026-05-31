# checkers/tests/test_comparative_seeds.py
#
# Step 1 of the Comparative Reasoning v2 roadmap: focused unit tests for the
# grouped comparative seed builders in
# `checkers/agents/comparative_reasoner.py`. These tests cover ONLY the
# deterministic Python helpers; there is no runtime integration to test yet.
#
# Coverage matrix:
#   - deterministic grouping (same input -> same output)
#   - multi-theme membership (alt in multiple groups)
#   - empty-group dropping (theme tag absent from result dict)
#   - single-member fold-down (singular grammatical phrasing)
#   - tradeoff seed generation (each priority branch + None case)
#   - stable ordering (THEME_TAGS order + ascending index within group)

from __future__ import annotations

import copy
import pytest

from checkers.agents.comparative_reasoner import (
    THEME_TAGS,
    _cluster_alternatives_by_theme,
    build_comparative_group_seeds,
    build_comparative_tradeoff_seed,
)


# ── Test fixtures ───────────────────────────────────────────────────────────


def _move(path, **facts):
    """Build a minimal move dict with the given facts. Path is positional."""
    return {"path": path, "facts": dict(facts)}


# Quiet chosen move with all themes off. Most tests parameterise alts
# around this baseline.
CHOSEN = _move(
    [(5, 4), (4, 3)],
    opponent_can_recapture=False,
    captures_count=0,
    net_gain=0,
    creates_immediate_threat=False,
    shot_sequence_available=False,
    leaves_piece_isolated=False,
    weakens_king_row=False,
    results_in_king=False,
    near_promotion=False,
    our_pieces_threatened_after=0,
    opponent_mobility_before=8,
    opponent_mobility_after=8,
)


def _quiet_alt(path):
    """An alternative move with all facts in their 'neutral' state.
    Note: this triggers DEFENSIVE because opp_recap=False AND pta<=chosen_pta,
    which is the locked taxonomy behaviour."""
    return _move(
        path,
        opponent_can_recapture=False,
        captures_count=0,
        net_gain=0,
        creates_immediate_threat=False,
        shot_sequence_available=False,
        leaves_piece_isolated=False,
        weakens_king_row=False,
        results_in_king=False,
        near_promotion=False,
        our_pieces_threatened_after=0,
        opponent_mobility_before=8,
        opponent_mobility_after=8,
    )


def _aggressive_alt(path, *, opp_recap=True):
    return _move(
        path,
        creates_immediate_threat=True,
        captures_count=0,
        opponent_can_recapture=opp_recap,
        our_pieces_threatened_after=0,
        leaves_piece_isolated=False,
        weakens_king_row=False,
        results_in_king=False,
        near_promotion=False,
        opponent_mobility_before=8,
        opponent_mobility_after=8,
        shot_sequence_available=False,
    )


def _material_alt(path, *, caps=1):
    return _move(
        path,
        captures_count=caps,
        net_gain=caps,
        creates_immediate_threat=False,
        opponent_can_recapture=True,
        our_pieces_threatened_after=0,
        leaves_piece_isolated=False,
        weakens_king_row=False,
        results_in_king=False,
        near_promotion=False,
        opponent_mobility_before=8,
        opponent_mobility_after=8,
        shot_sequence_available=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Clustering
# ═══════════════════════════════════════════════════════════════════════════


class TestClustering:
    def test_empty_alternatives_yields_empty_dict(self):
        # Only the chosen move in the candidate list → no alternatives.
        groups = _cluster_alternatives_by_theme([CHOSEN], CHOSEN)
        assert groups == {}

    def test_aggressive_theme_fires_on_immediate_threat(self):
        alt = _aggressive_alt([(5, 2), (4, 3)])
        groups = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        assert "AGGRESSIVE" in groups
        assert groups["AGGRESSIVE"] == [(1, alt)]

    def test_aggressive_theme_fires_on_shot_sequence(self):
        alt = _move(
            [(5, 2), (4, 3)],
            creates_immediate_threat=False,
            shot_sequence_available=True,
            captures_count=0,
            opponent_can_recapture=True,
            our_pieces_threatened_after=0,
            leaves_piece_isolated=False,
            weakens_king_row=False,
            results_in_king=False,
            near_promotion=False,
            opponent_mobility_before=8,
            opponent_mobility_after=8,
        )
        groups = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        assert "AGGRESSIVE" in groups

    def test_material_theme(self):
        alt = _material_alt([(5, 2), (4, 3)], caps=2)
        groups = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        assert "MATERIAL" in groups
        assert groups["MATERIAL"][0][0] == 1

    def test_defensive_theme_with_quiet_alt(self):
        alt = _quiet_alt([(5, 2), (4, 3)])
        groups = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        # Quiet alt: opp_recap=False AND pta=0=chosen_pta → DEFENSIVE fires.
        assert "DEFENSIVE" in groups

    def test_structural_theme_when_isolation_differs(self):
        alt = _move(
            [(5, 2), (4, 3)],
            leaves_piece_isolated=True,  # chosen has False — differs
            captures_count=0,
            opponent_can_recapture=False,
            creates_immediate_threat=False,
            our_pieces_threatened_after=0,
            weakens_king_row=False,
            results_in_king=False,
            near_promotion=False,
            opponent_mobility_before=8,
            opponent_mobility_after=8,
            shot_sequence_available=False,
        )
        groups = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        assert "STRUCTURAL" in groups

    def test_structural_theme_when_king_row_differs(self):
        alt = _move(
            [(5, 2), (4, 3)],
            weakens_king_row=True,  # chosen has False — differs
            leaves_piece_isolated=False,
            captures_count=0,
            opponent_can_recapture=False,
            creates_immediate_threat=False,
            our_pieces_threatened_after=0,
            results_in_king=False,
            near_promotion=False,
            opponent_mobility_before=8,
            opponent_mobility_after=8,
            shot_sequence_available=False,
        )
        groups = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        assert "STRUCTURAL" in groups

    def test_promotion_theme_results_in_king(self):
        alt = _move(
            [(5, 2), (4, 3)],
            results_in_king=True,
            captures_count=0,
            opponent_can_recapture=False,
            creates_immediate_threat=False,
            our_pieces_threatened_after=0,
            leaves_piece_isolated=False,
            weakens_king_row=False,
            near_promotion=False,
            opponent_mobility_before=8,
            opponent_mobility_after=8,
            shot_sequence_available=False,
        )
        groups = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        assert "PROMOTION" in groups

    def test_promotion_theme_near_promotion(self):
        alt = _move(
            [(5, 2), (4, 3)],
            near_promotion=True,
            results_in_king=False,
            captures_count=0,
            opponent_can_recapture=False,
            creates_immediate_threat=False,
            our_pieces_threatened_after=0,
            leaves_piece_isolated=False,
            weakens_king_row=False,
            opponent_mobility_before=8,
            opponent_mobility_after=8,
            shot_sequence_available=False,
        )
        groups = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        assert "PROMOTION" in groups

    def test_mobility_theme(self):
        alt = _move(
            [(5, 2), (4, 3)],
            opponent_mobility_before=8,
            opponent_mobility_after=5,  # reduces opponent mobility
            captures_count=0,
            creates_immediate_threat=False,
            opponent_can_recapture=False,
            our_pieces_threatened_after=0,
            leaves_piece_isolated=False,
            weakens_king_row=False,
            results_in_king=False,
            near_promotion=False,
            shot_sequence_available=False,
        )
        groups = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        assert "MOBILITY" in groups

    def test_chosen_move_excluded_from_every_group(self):
        # Chosen has captures_count=0 so it cannot be MATERIAL.
        # But even if it could qualify, it must be excluded by path.
        groups = _cluster_alternatives_by_theme([CHOSEN], CHOSEN)
        for tag, members in groups.items():
            for idx, _ in members:
                assert idx != 0, f"chosen move appeared in group {tag}"

    def test_multi_theme_membership_aggressive_and_material(self):
        # Captures AND creates threat → should appear in both groups.
        alt = _move(
            [(5, 2), (4, 3)],
            captures_count=2,
            net_gain=2,
            creates_immediate_threat=True,
            opponent_can_recapture=True,
            our_pieces_threatened_after=0,
            leaves_piece_isolated=False,
            weakens_king_row=False,
            results_in_king=False,
            near_promotion=False,
            opponent_mobility_before=8,
            opponent_mobility_after=8,
            shot_sequence_available=False,
        )
        groups = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        assert "MATERIAL" in groups
        assert "AGGRESSIVE" in groups
        assert groups["MATERIAL"][0][0] == 1
        assert groups["AGGRESSIVE"][0][0] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 2. Determinism + stable ordering
# ═══════════════════════════════════════════════════════════════════════════


class TestDeterminismAndOrdering:
    def test_same_inputs_produce_same_output(self):
        alt = _aggressive_alt([(5, 2), (4, 3)])
        g1 = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        g2 = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        assert g1 == g2

    def test_members_sorted_ascending_by_index(self):
        a1 = _aggressive_alt([(5, 0), (4, 1)])
        a2 = _aggressive_alt([(5, 2), (4, 3)])
        a3 = _aggressive_alt([(5, 4), (4, 5)])
        groups = _cluster_alternatives_by_theme([CHOSEN, a1, a2, a3], CHOSEN)
        idxs = [i for i, _ in groups["AGGRESSIVE"]]
        assert idxs == [1, 2, 3]

    def test_dict_key_order_follows_THEME_TAGS(self):
        agg = _aggressive_alt([(5, 0), (4, 1)])
        mat = _material_alt([(5, 2), (4, 3)])
        groups = _cluster_alternatives_by_theme([CHOSEN, agg, mat], CHOSEN)
        keys = list(groups.keys())
        # In THEME_TAGS order, AGGRESSIVE precedes MATERIAL.
        assert keys.index("AGGRESSIVE") < keys.index("MATERIAL")
        # And every key is a valid theme.
        for k in keys:
            assert k in THEME_TAGS

    def test_never_mutates_inputs(self):
        before_chosen = copy.deepcopy(CHOSEN)
        alt = _aggressive_alt([(5, 2), (4, 3)])
        before_alt = copy.deepcopy(alt)
        _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        assert CHOSEN == before_chosen
        assert alt == before_alt


# ═══════════════════════════════════════════════════════════════════════════
# 3. Empty-group dropping
# ═══════════════════════════════════════════════════════════════════════════


class TestEmptyGroupDropping:
    def test_empty_groups_absent_from_result(self):
        # Single aggressive alt — only AGGRESSIVE (and possibly DEFENSIVE
        # via opp_recap default) should appear; MATERIAL, PROMOTION,
        # MOBILITY must NOT appear.
        alt = _aggressive_alt([(5, 2), (4, 3)], opp_recap=True)
        groups = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        for tag in ("MATERIAL", "PROMOTION", "MOBILITY"):
            assert tag not in groups, (
                f"empty group {tag} should have been dropped"
            )

    def test_empty_result_when_no_alts_match_any_theme(self):
        # Construct an alt that fails every theme:
        #   not aggressive, no captures, opp_can_recapture=True (no DEFENSIVE),
        #   isolation/king-row match chosen, no promotion, mobility unchanged.
        alt = _move(
            [(5, 2), (4, 3)],
            creates_immediate_threat=False,
            shot_sequence_available=False,
            captures_count=0,
            opponent_can_recapture=True,
            our_pieces_threatened_after=0,
            leaves_piece_isolated=False,
            weakens_king_row=False,
            results_in_king=False,
            near_promotion=False,
            opponent_mobility_before=8,
            opponent_mobility_after=8,
        )
        groups = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        assert groups == {}, f"expected empty result, got {groups}"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Group-seed builder (single-member fold-down + plural conjunctions)
# ═══════════════════════════════════════════════════════════════════════════


class TestGroupSeedBuilder:
    def test_empty_groups_produce_no_seeds(self):
        assert build_comparative_group_seeds({}, CHOSEN["facts"]) == []

    def test_single_member_uses_singular_phrasing(self):
        alt = _aggressive_alt([(5, 2), (4, 3)])
        groups = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        seeds = build_comparative_group_seeds(groups, CHOSEN["facts"])
        agg = next(s for s in seeds if s.startswith("Aggressive"))
        assert "alternative [1]" in agg
        assert "alternatives" not in agg

    def test_multi_member_uses_plural_phrasing_and_natural_conjunction(self):
        a1 = _aggressive_alt([(5, 0), (4, 1)])
        a2 = _aggressive_alt([(5, 2), (4, 3)])
        a3 = _aggressive_alt([(5, 4), (4, 5)])
        groups = _cluster_alternatives_by_theme([CHOSEN, a1, a2, a3], CHOSEN)
        seeds = build_comparative_group_seeds(groups, CHOSEN["facts"])
        agg = next(s for s in seeds if s.startswith("Aggressive"))
        assert "alternatives" in agg
        # Natural conjunction with Oxford comma:  "[1], [2], and [3]"
        assert "[1], [2], and [3]" in agg

    def test_two_member_uses_simple_and(self):
        a1 = _aggressive_alt([(5, 0), (4, 1)])
        a2 = _aggressive_alt([(5, 2), (4, 3)])
        groups = _cluster_alternatives_by_theme([CHOSEN, a1, a2], CHOSEN)
        seeds = build_comparative_group_seeds(groups, CHOSEN["facts"])
        agg = next(s for s in seeds if s.startswith("Aggressive"))
        # Two-member form:  "[1] and [2]"  (no Oxford comma)
        assert "[1] and [2]" in agg

    def test_seeds_emitted_in_THEME_TAGS_order(self):
        agg = _aggressive_alt([(5, 0), (4, 1)])
        mob = _move(
            [(5, 2), (4, 3)],
            opponent_mobility_before=8,
            opponent_mobility_after=5,
            creates_immediate_threat=False,
            captures_count=0,
            opponent_can_recapture=False,
            our_pieces_threatened_after=0,
            leaves_piece_isolated=False,
            weakens_king_row=False,
            results_in_king=False,
            near_promotion=False,
            shot_sequence_available=False,
        )
        groups = _cluster_alternatives_by_theme([CHOSEN, agg, mob], CHOSEN)
        seeds = build_comparative_group_seeds(groups, CHOSEN["facts"])
        agg_pos = next(i for i, s in enumerate(seeds) if s.startswith("Aggressive"))
        mob_pos = next(
            i for i, s in enumerate(seeds) if s.startswith("Mobility-restricting")
        )
        # AGGRESSIVE comes before MOBILITY in THEME_TAGS.
        assert agg_pos < mob_pos

    def test_aggressive_seed_includes_chosen_safety_contrast(self):
        # All aggressive alts allow recapture; chosen is safe.
        alt = _aggressive_alt([(5, 2), (4, 3)], opp_recap=True)
        groups = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        seeds = build_comparative_group_seeds(groups, CHOSEN["facts"])
        agg = next(s for s in seeds if s.startswith("Aggressive"))
        assert "allows recapture" in agg
        assert "chosen move forfeits initiative for safety" in agg

    def test_material_seed_includes_chosen_contrast(self):
        mat = _material_alt([(5, 2), (4, 3)])
        groups = _cluster_alternatives_by_theme([CHOSEN, mat], CHOSEN)
        seeds = build_comparative_group_seeds(groups, CHOSEN["facts"])
        mat_seed = next(s for s in seeds if s.startswith("Material"))
        assert "chosen move does not capture" in mat_seed

    def test_no_robotic_alternative_n_theme_pattern(self):
        # Step 1's prose contract: the seeds use natural-language framing
        # rather than the "Alternative [N] [THEME]:" robotic pattern that
        # the earlier comparative work used.
        alt = _aggressive_alt([(5, 2), (4, 3)])
        groups = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        seeds = build_comparative_group_seeds(groups, CHOSEN["facts"])
        for s in seeds:
            assert "[AGGRESSIVE]" not in s
            assert "[MATERIAL]" not in s
            assert "[DEFENSIVE]" not in s
            assert "[STRUCTURAL]" not in s
            assert "[PROMOTION]" not in s
            assert "[MOBILITY]" not in s


# ═══════════════════════════════════════════════════════════════════════════
# 5. Tradeoff seed
# ═══════════════════════════════════════════════════════════════════════════


class TestTradeoffSeed:
    def test_returns_none_when_no_groups(self):
        assert build_comparative_tradeoff_seed(CHOSEN["facts"], {}) is None

    def test_priority_1_aggressive_when_chosen_is_safe(self):
        # CHOSEN: not aggressive + opp_recap=False. Aggressive alts exist.
        alt = _aggressive_alt([(5, 2), (4, 3)], opp_recap=True)
        groups = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        seed = build_comparative_tradeoff_seed(CHOSEN["facts"], groups)
        assert seed is not None
        assert "aggressive options" in seed
        assert "recapture safety" in seed

    def test_priority_2_material_when_chosen_does_not_capture(self):
        # No aggressive alts. Material alts exist. Chosen does not capture.
        # We need to avoid priority 1 firing first — to do that, the
        # chosen-safe-defensive route is masked when there's no aggressive
        # group, so priority 2 wins.
        mat = _material_alt([(5, 2), (4, 3)])
        groups = _cluster_alternatives_by_theme([CHOSEN, mat], CHOSEN)
        seed = build_comparative_tradeoff_seed(CHOSEN["facts"], groups)
        assert seed is not None
        assert "material captures" in seed
        assert "positional advantage" in seed

    def test_priority_4_chosen_captures_with_recapture_risk(self):
        # Chosen: captures + recapture risk.
        # Defensive alts: opp_recap=False + pta<=chosen_pta.
        chosen_capturing = _move(
            [(5, 4), (4, 3)],
            captures_count=1,
            net_gain=1,
            opponent_can_recapture=True,
            creates_immediate_threat=False,
            shot_sequence_available=False,
            leaves_piece_isolated=False,
            weakens_king_row=False,
            results_in_king=False,
            near_promotion=False,
            our_pieces_threatened_after=1,
            opponent_mobility_before=8,
            opponent_mobility_after=8,
        )
        defensive_alt = _quiet_alt([(5, 2), (4, 3)])
        groups = _cluster_alternatives_by_theme(
            [chosen_capturing, defensive_alt], chosen_capturing,
        )
        seed = build_comparative_tradeoff_seed(
            chosen_capturing["facts"], groups,
        )
        assert seed is not None
        assert "accepts exposure" in seed
        assert "defensive options" in seed

    def test_uses_compact_bracket_format(self):
        # The roadmap example shows [1,3,5] without spaces — the tradeoff
        # seed uses _format_index_list_compact.
        a1 = _aggressive_alt([(5, 0), (4, 1)])
        a2 = _aggressive_alt([(5, 2), (4, 3)])
        groups = _cluster_alternatives_by_theme([CHOSEN, a1, a2], CHOSEN)
        seed = build_comparative_tradeoff_seed(CHOSEN["facts"], groups)
        assert seed is not None
        assert "[1,2]" in seed
        # NOT the natural form
        assert "[1] and [2]" not in seed

    def test_priority_cascade_first_match_wins(self):
        # Both aggressive AND material alts exist; chosen is safe and
        # doesn't capture. Priority 1 (aggressive vs safety) wins over
        # priority 2 (material).
        agg = _aggressive_alt([(5, 0), (4, 1)], opp_recap=True)
        mat = _material_alt([(5, 2), (4, 3)])
        groups = _cluster_alternatives_by_theme([CHOSEN, agg, mat], CHOSEN)
        seed = build_comparative_tradeoff_seed(CHOSEN["facts"], groups)
        assert seed is not None
        # Priority 1 phrasing
        assert "recapture safety" in seed
        assert "positional advantage" not in seed

    def test_never_mutates_inputs(self):
        before_facts = copy.deepcopy(CHOSEN["facts"])
        alt = _aggressive_alt([(5, 2), (4, 3)], opp_recap=True)
        groups = _cluster_alternatives_by_theme([CHOSEN, alt], CHOSEN)
        before_groups = copy.deepcopy(groups)
        build_comparative_tradeoff_seed(CHOSEN["facts"], groups)
        assert CHOSEN["facts"] == before_facts
        assert groups == before_groups
