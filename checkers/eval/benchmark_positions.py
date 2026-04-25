"""
checkers/eval/benchmark_positions.py
─────────────────────────────────────
Curated benchmark positions for repeatable move-quality evaluation.

Board encoding
--------------
  board is a list[list[int]] (8×8, row-major).
  Piece constants are imported from checkers.engine.board:
    EMPTY=0, RED=1, BLACK=2, RED_KING=3, BLACK_KING=4

Adding a new position
----------------------
Append a dict to BENCHMARK_POSITIONS following the schema below.
Required keys: position_id, category, board, side_to_move, tags.
Optional: expected_best_path, explanation, known_failure.

Schema
------
{
  "position_id":       str       # unique slug, e.g. "pos_t41_promo_tie"
  "category":          str       # opening|midgame_tactical|midgame_positional|endgame|known_failure
  "board":             list[list[int]]  # 8×8
  "side_to_move":      int       # RED (1) or BLACK (2)
  "tags":              list[str] # e.g. ["promotion","forced_capture"]
  "expected_best_path": list[list[int]] | None  # [[r,c],[r,c]] or None
  "explanation":       str       # human explanation of the correct move
  "known_failure":     bool      # True = engine was previously wrong here
}
"""

from __future__ import annotations

from checkers.engine.board import EMPTY as E, RED as R, BLACK as B, RED_KING as RK, BLACK_KING as BK

# ── helper ────────────────────────────────────────────────────────────────────

def _b(rows: list[list[int]]) -> list[list[int]]:
    """Identity – just makes the board literals readable."""
    assert len(rows) == 8 and all(len(r) == 8 for r in rows)
    return rows


# ── Positions ─────────────────────────────────────────────────────────────────

BENCHMARK_POSITIONS: list[dict] = [

    # ── 1. Opening ────────────────────────────────────────────────────────────
    {
        "position_id": "pos_opening_standard",
        "category": "opening",
        "board": _b([
            [E, B, E, B, E, B, E, B],
            [B, E, B, E, B, E, B, E],
            [E, B, E, B, E, B, E, B],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [R, E, R, E, R, E, R, E],
            [E, R, E, R, E, R, E, R],
            [R, E, R, E, R, E, R, E],
        ]),
        "side_to_move": R,
        "tags": ["opening"],
        "expected_best_path": None,
        "explanation": (
            "Standard opening position. No clearly forced best move exists at depth 6; "
            "benchmark records current engine choice as baseline."
        ),
        "known_failure": False,
    },

    # ── 2. Mandatory capture (single jump) ────────────────────────────────────
    {
        "position_id": "pos_mandatory_capture",
        "category": "midgame_tactical",
        "board": _b([
            [E, B, E, B, E, B, E, B],
            [B, E, B, E, E, E, B, E],  # E at (1,4)
            [E, B, E, B, E, E, E, E],  # B at (2,3)
            [E, E, R, E, E, E, E, E],  # RED at (3,2) must jump BLACK at (2,3)
            [E, E, E, E, E, E, E, E],
            [R, E, R, E, R, E, R, E],
            [E, R, E, R, E, R, E, R],
            [R, E, R, E, R, E, R, E],
        ]),
        "side_to_move": R,
        "tags": ["forced_capture", "mandatory_capture"],
        "expected_best_path": [[3, 2], [1, 4]],
        "explanation": (
            "RED has a piece at (3,2). BLACK is at (2,3) and (1,4) is empty. "
            "The engine must capture. Baseline: verify search always picks the capture when available."
        ),
        "known_failure": False,
    },

    # ── 3. Multi-jump (double capture) ────────────────────────────────────────
    {
        "position_id": "pos_multi_jump",
        "category": "midgame_tactical",
        "board": _b([
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, B, E, E, E, E],  # BLACK at (2,3)
            [E, E, E, E, E, E, E, E],
            [E, E, E, B, E, E, E, E],  # BLACK at (4,3)
            [E, E, R, E, E, E, E, E],  # RED at (5,2) — can double-jump (5,2)→(3,4)→(1,2)
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
        ]),
        "side_to_move": R,
        "tags": ["forced_capture", "double_jump", "multi_jump"],
        "expected_best_path": [[5, 2], [3, 4], [1, 2]],
        "explanation": (
            "RED at (5,2) can execute a double-jump: (5,2)→(3,4)→(1,2), capturing BLACK at "
            "(4,3) and (2,3). This is the only legal move and must be fully generated."
        ),
        "known_failure": False,
    },

    # ── 4. Simple promotion ───────────────────────────────────────────────────
    {
        "position_id": "pos_simple_promotion",
        "category": "midgame_positional",
        "board": _b([
            [E, E, E, E, E, E, E, E],  # row 0 — RED promotes here
            [E, E, E, E, E, R, E, E],  # RED at (1,5) — one step from promotion
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, B, E, E],  # BLACK at (6,5)
            [E, E, E, E, E, E, E, E],
        ]),
        "side_to_move": R,
        "tags": ["promotion"],
        "expected_best_path": [[1, 5], [0, 4]],
        "explanation": (
            "RED at (1,5) has two promotion options: (0,4) or (0,6). Both are empty. "
            "Either is correct. Expected: promotion happens (any row-0 destination)."
        ),
        "known_failure": False,
    },

    # ── 5. Endgame King activity ──────────────────────────────────────────────
    {
        "position_id": "pos_endgame_king_activity",
        "category": "endgame",
        "board": _b([
            [E, E, E, E, E, E, E, RK],  # RED King at (0,7)
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, BK, E, E],  # BLACK King at (6,5) — actively positioned
            [E, E, E, E, E, E, E, E],
        ]),
        "side_to_move": R,
        "tags": ["endgame", "king_activity", "endgame_conversion"],
        "expected_best_path": None,
        "explanation": (
            "Pure endgame King vs King. No material advantage. Baseline: record engine "
            "King activation score and chosen direction."
        ),
        "known_failure": False,
    },

    # ── 6. Known failure: T37 promotion safety-filter ─────────────────────────
    # Before fix: safety filter removed (1,6)→(0,7) because (0,5) could theoretically
    # reach (1,6). Post-fix: promotion survives filter unconditionally.
    {
        "position_id": "pos_t37_promo_safety_filter",
        "category": "known_failure",
        "board": _b([
            [E, B, E, B, E, B, E, E],  # BLACK at (0,1),(0,3),(0,5)
            [E, E, E, E, E, E, R, E],  # RED at (1,6) — one step from promotion
            [E, R, E, E, E, E, E, B],  # RED at (2,1), BLACK at (2,7)
            [E, E, E, E, E, E, E, E],
            [E, R, E, E, E, E, E, B],  # RED at (4,1), BLACK at (4,7)
            [R, E, R, E, R, E, E, E],  # RED at (5,0),(5,2),(5,4)
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, R, E, BK, E], # RED at (7,4), BLACK_KING at (7,6)
        ]),
        "side_to_move": R,
        "tags": ["promotion", "safety_filter", "known_failure"],
        "expected_best_path": [[1, 6], [0, 7]],
        "explanation": (
            "RED at (1,6) should promote to (0,7). Before the safety-filter fix this move "
            "was filtered out because (0,5) could theoretically reach (1,6). Post-fix "
            "the promotion must be present in the candidate set and score highest."
        ),
        "known_failure": True,
    },

    # ── 7. Known failure: T41 promotion tie-break ─────────────────────────────
    # Three moves tied at +96: (2,1)→(1,0), (1,6)→(0,7), (5,4)→(4,3).
    # Engine previously chose (2,1)→(1,0). Post tie-break fix: promotion preferred.
    {
        "position_id": "pos_t41_promo_tiebreak",
        "category": "known_failure",
        "board": _b([
            [E, B, E, B, E, B, E, E],  # BLACK at (0,1),(0,3),(0,5)
            [E, E, E, E, E, E, R, E],  # RED at (1,6) — can promote
            [E, R, E, E, E, E, E, B],  # RED at (2,1), BLACK at (2,7)
            [E, E, E, E, E, E, E, E],
            [E, R, E, E, E, E, E, E],  # RED at (4,1)
            [R, E, R, E, R, E, E, E],  # RED at (5,0),(5,2),(5,4)
            [E, E, E, E, E, B, E, E],  # BLACK at (6,5)
            [E, E, E, E, E, E, E, E],
        ]),
        "side_to_move": R,
        "tags": ["promotion", "ranker_tie", "tie_break", "known_failure"],
        "expected_best_path": [[1, 6], [0, 7]],
        "explanation": (
            "Three moves tie at +96 by depth-6 minimax. The promotion (1,6)→(0,7) must be "
            "selected by the tie-break rule (results_in_king=True preferred over near-tie). "
            "This verifies the promotion tie-break fix is active."
        ),
        "known_failure": True,
    },

    # ── 8. Known failure: T43 caged-corner promotion (structural) ─────────────
    # Promotion to (0,7) is forced best despite creating a caged King.
    # All non-promotion alternatives score -222 to -236 at depth 6.
    {
        "position_id": "pos_t43_caged_corner",
        "category": "known_failure",
        "board": _b([
            [E, B, E, B, E, B, E, E],  # BLACK at (0,1),(0,3),(0,5)
            [R, E, E, E, E, E, R, E],  # RED at (1,0),(1,6) — (1,6) can promote
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, B, E],  # BLACK at (3,6)
            [E, R, E, E, E, E, E, E],  # RED at (4,1)
            [R, E, R, E, R, E, E, E],  # RED at (5,0),(5,2),(5,4)
            [E, E, E, E, E, B, E, E],  # BLACK at (6,5)
            [E, E, E, E, E, E, E, E],
        ]),
        "side_to_move": R,
        "tags": ["promotion", "caged_corner", "horizon_effect", "known_failure"],
        "expected_best_path": [[1, 6], [0, 7]],
        "explanation": (
            "RED at (1,6) must promote to (0,7) — the only free row-0 square. "
            "All other moves score -222 to -236. The resulting corner King is caged "
            "(BLACK at (0,5) permanently guards exit (1,6)). This is a confirmed "
            "horizon effect: no evaluation fix is safe. Expected: promotion is chosen. "
            "Benchmark records the caged-corner score as a structural baseline."
        ),
        "known_failure": True,
    },

    # ── 9. T49 forced double-jump threat (defensive play) ─────────────────────
    # The previous expected_best_path [[4,3],[3,4]] was WRONG:
    #   - (4,3) is EMPTY on this board — no RED piece there.
    #   - The expected path was copy-pasted from the trace original board, not
    #     this reconstructed benchmark position.
    #
    # Verified at D6 with use_tt=False:
    #   Three moves tie at -8: (3,2)→(2,3), (5,2)→(4,1), (5,2)→(4,3).
    #   King activation (0,7)→(1,6) scores -324 (BLACK captures immediately).
    #   Near-promotion pushes (3,4)→(2,5) score -111.
    #   The engine is correct: quiet defensive moves score best.
    #   This is NOT an engine failure.
    {
        "position_id": "pos_t49_forced_double_jump",
        "category": "midgame_tactical",
        "board": _b([
            [E, B, E, B, E, B, E, RK],  # BLACK at (0,1),(0,3),(0,5), RED King at (0,7)
            [R, E, E, E, E, E, E, E],   # RED at (1,0)
            [E, E, E, E, E, E, E, E],
            [E, E, R, E, R, E, E, E],   # RED at (3,2),(3,4)
            [E, E, E, E, E, E, E, E],
            [R, E, R, E, B, E, E, E],   # RED at (5,0),(5,2), BLACK at (5,4)
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, BK, E, E, E],  # BLACK King at (7,4)
        ]),
        "side_to_move": R,
        "tags": ["double_jump_threat", "vulnerability", "defensive_move"],
        # D6 three-way tie at -8: (3,2)→(2,3), (5,2)→(4,1), (5,2)→(4,3).
        # Any of these is correct. Record the D6 argmax as canonical.
        "expected_best_path": [[3, 2], [2, 3]],
        "explanation": (
            "T49 position. BLACK at (5,4) threatens sequences, BLACK King at (7,4) is active. "
            "King activation (0,7)→(1,6) immediately loses the King to BLACK's jump (D6=-324). "
            "Near-promotion (3,4)→(2,5) scores -111 (enters BLACK King range). "
            "Three quiet defensive moves tie at D6=-8: (3,2)→(2,3), (5,2)→(4,1), (5,2)→(4,3). "
            "The engine correctly selects a defensive move. "
            "Previous expected_best_path [[4,3],[3,4]] was wrong — (4,3) is empty on this board. "
            "This is NOT an engine failure; it is a benchmark label correction."
        ),
        # At D4, moves diverge (29 vs 14) so a different defensive move wins.
        # The expected label targets D6 where the three-way tie converges.
        "known_failure": True,
    },

    # ── STRESS-TEST POSITIONS (Phase 5 evaluation audit) ─────────────────────
    # These test strategic edge cases not covered by the original 9 positions.
    # Expected behaviors are based on search verification at D4 and D6.

    {
        "position_id": "stress_winning_endgame_conversion",
        "category": "endgame",
        "board": _b([
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E,BK, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E,RK, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E,RK, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
        ]),
        "side_to_move": R,
        "tags": ["endgame", "conversion", "king_activity"],
        "expected_best_path": None,
        "explanation": (
            "2K vs 1K. RK(5,2) finds forced win at D4+. RK(3,4)→(2,5) "
            "approaches BK directly and scores very low (-66 at D6). "
            "Verifies search prefers coordinated conversion over reckless approach."
        ),
        "known_failure": False,
    },

    {
        "position_id": "stress_promote_vs_retreat",
        "category": "endgame",
        "board": _b([
            [E, E, E, E, E, B, E, E],
            [E, E, R, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E,RK, E, E, E, E],
            [E, E, E, E, R, E, E, E],
            [E, B, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
        ]),
        "side_to_move": R,
        "tags": ["promotion", "simplification", "endgame"],
        "expected_best_path": [[1, 2], [0, 3]],
        "explanation": (
            "RED up +175 (king). R(1,2) should promote to center (0,3). "
            "Verified at D4 and D6: promotion ties with king approach but never "
            "loses to king retreat. Tests that simplification bonus does not "
            "override active promotion."
        ),
        "known_failure": False,
    },

    {
        "position_id": "stress_good_simplification_advance",
        "category": "endgame",
        "board": _b([
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E,BK, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E,RK, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E,RK, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [B, E, E, E, E, E, E, E],
        ]),
        "side_to_move": R,
        "tags": ["endgame", "simplification", "king_activity"],
        "expected_best_path": None,
        "explanation": (
            "2K vs 1K+1man. RK(5,2) finds forced win through capturing B(7,0). "
            "Verifies simplification when genuinely ahead leads to conversion, "
            "not passive waiting."
        ),
        "known_failure": False,
    },

    {
        "position_id": "stress_active_vs_passive_king",
        "category": "endgame",
        "board": _b([
            [E,BK, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E,RK, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
        ]),
        "side_to_move": R,
        "tags": ["endgame", "king_activity", "balance"],
        "expected_best_path": None,
        "explanation": (
            "1K vs 1K, equal material. Static eval = +74 (center king advantage) "
            "but D6 search = -10 for all moves. King activity terms inflate the "
            "positional advantage beyond what search confirms. The 74-pt static "
            "advantage evaporates because both kings equalize mobility within a "
            "few plies. Stress-tests king activity term magnitude."
        ),
        "known_failure": False,
    },

    {
        "position_id": "stress_trapped_caged_king",
        "category": "endgame",
        "board": _b([
            [E, E, E, E, E, E, E,RK],
            [E, E, E, E, E, E, B, E],
            [E, E, E, E, E, B, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, R, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
        ]),
        "side_to_move": R,
        "tags": ["endgame", "trapped_king", "vulnerability"],
        "expected_best_path": [[5, 2], [4, 1]],
        "explanation": (
            "RK(0,7) is caged by B(1,6). R(5,2) must advance. "
            "king_chase_pressure awards +24 to the caged king even though it "
            "cannot move — Manhattan distance doesn't account for blocked paths. "
            "Search correctly prefers advancing R(5,2) over any other option."
        ),
        "known_failure": False,
    },

    {
        "position_id": "stress_promotion_path_quality",
        "category": "midgame_positional",
        "board": _b([
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, R, E, B, E],
            [E, E, E, E, E, E, E, B],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [R, E, E, E, E, E, E, E],
            [E, B, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
        ]),
        "side_to_move": R,
        "tags": ["promotion", "king_quality", "horizon"],
        "expected_best_path": [[1, 4], [0, 3]],
        "explanation": (
            "R(1,4) can promote to (0,3) center king or (0,5) near B(1,6) cage. "
            "D4: (0,3)=-199 vs (0,5)=-256. D6: (0,3)=-220 vs (0,5)=-258. "
            "Search correctly prefers center promotion. 38-pt gap at D6 shows "
            "the engine detects cage risk even without an explicit promotion-quality term."
        ),
        "known_failure": False,
    },

    {
        "position_id": "stress_centrality_vs_tactics",
        "category": "opening",
        "board": _b([
            [E, E, E, E, E, E, E, E],
            [B, E, B, E, B, E, B, E],
            [E, B, E, B, E, B, E, E],
            [E, E, E, E, E, E, B, E],
            [E, E, E, E, E, E, E, E],
            [R, E, R, E, R, E, R, E],
            [E, R, E, R, E, R, E, R],
            [E, E, E, E, E, E, E, E],
        ]),
        "side_to_move": R,
        "tags": ["opening", "column_centrality", "tactics"],
        "expected_best_path": None,
        "explanation": (
            "8R vs 8B. R(5,4)→(4,5) improves column centrality but B(3,6) can "
            "jump it. D4: (4,5)=-489 (last). D6: (4,5)=-467 (last). "
            "Verifies tactics completely dominate column centrality. "
            "Column centrality is -18 in static eval — negligible vs the tactical loss."
        ),
        "known_failure": False,
    },

    {
        "position_id": "stress_double_jump_escape",
        "category": "midgame_tactical",
        "board": _b([
            [E, E, E,BK, E, E, E, E],
            [E, E, R, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, R, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [R, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, B],
            [E, E, E, E, R, E, E, E],
        ]),
        "side_to_move": R,
        "tags": ["tactics", "double_jump", "vulnerability"],
        "expected_best_path": [[1, 2], [0, 1]],
        "explanation": (
            "BK(0,3) threatens double-jump through R(1,2) and R(3,2) next turn. "
            "R(1,2)→(0,1) escapes AND promotes: 135 at D6. Non-escape moves "
            "score -538 to -540. Tests that search sees the double-jump threat "
            "and prioritizes escape."
        ),
        "known_failure": False,
    },

    {
        "position_id": "stress_quiet_trap_reversal",
        "category": "midgame_tactical",
        "board": _b([
            [E, B, E, B, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, B, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, R, E, E, E],
            [E, E, E, E, E, E, E, E],
            [R, E, R, E, E, E, E, E],
        ]),
        "side_to_move": R,
        "tags": ["tactics", "trap", "quiet_move"],
        "expected_best_path": [[5, 4], [4, 3]],
        "explanation": (
            "R(5,4)→(4,5) looks like a trap (B(2,5) must go to jumpable squares) "
            "but at D6 scores -120 because after BLACK delays with other pieces, "
            "R(4,5) advancing to (3,4)/(3,6) gets counter-jumped by B(2,5). "
            "R(5,4)→(4,3) scores 0: R can retreat to (3,2) safely. "
            "Tests search depth seeing through apparent traps."
        ),
        "known_failure": False,
    },

    {
        "position_id": "stress_endgame_king_race",
        "category": "endgame",
        "board": _b([
            [E, E, E, E, E, E, E, E],
            [E, E, R, E, E, E, E, E],
            [E, E, E, E, E, E, E,BK],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [RK, E, E, E, E, E, E, E],
            [E, E, E, E, E, B, E, E],
            [E, E, E, E, E, E, E, E],
        ]),
        "side_to_move": R,
        "tags": ["endgame", "promotion", "king_race"],
        "expected_best_path": [[1, 2], [0, 3]],
        "explanation": (
            "Both sides have 1 man near promotion + 1 king. R(1,2) should promote "
            "to center (0,3). D4 and D6 both score promotion and king-advance "
            "equally at 0 — the position is drawn with best play from both sides. "
            "R(1,2)→(0,1) edge promotion scores slightly worse (-4 to -8)."
        ),
        "known_failure": False,
    },

    # ── Trace-derived position: T7 proposal miss ──────────────────────────────
    # From paired trace experiment (proposal-active vs all-legal, seed=standard).
    # Board state: TURN 7, before RED moves.
    # Symbolic best: (5,2)→(4,1) score=+35.0 (basis[1], presentation[3]).
    # The LLM correctly selected pres[3] → basis[1] in its raw output [0,1,3,5,7].
    # _apply_safety_net kept pres[3] in the merged list, BUT the final trimming
    # (via _role_pin + critical-injection priority) displaced it, leaving the
    # ranker with 5 moves none of which was basis[1].
    # This is the root cause of the 37-point divergence at T7.
    {
        "position_id": "trace_t7_proposal_miss",
        "category": "midgame_tactical",
        "board": [
            [E,B,E,B,E,B,E,B],
            [B,E,E,E,E,E,B,E],
            [E,B,E,B,E,B,E,B],
            [E,E,E,E,B,E,E,E],
            [E,E,E,E,E,E,E,E],
            [R,E,R,E,R,E,R,E],
            [E,R,E,R,E,R,E,E],
            [R,E,R,E,R,E,R,E],
        ],
        "side_to_move": R,
        "tags": ["tactical", "proposal_miss", "safety_net_trim", "opening"],
        "expected_best_path": [[5, 2], [4, 1]],
        "explanation": (
            "RED has 8 legal moves. Minimax (depth 6) scores (5,2)→(4,1) at +35.0, "
            "far ahead of the next-best at -1.0. The LLM correctly selects this move "
            "in presentation space (pres[3] → basis[1]), but _apply_safety_net "
            "trimming displaces it from the final 5-move menu due to critical-injection "
            "priority reordering. The ranker never sees the +35.0 move. "
            "The paired all-legal run correctly chose (5,2)→(4,1) and produced a "
            "divergent winning trajectory."
        ),
        "known_failure": True,
    },
]
