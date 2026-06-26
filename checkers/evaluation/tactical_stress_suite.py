# checkers/evaluation/tactical_stress_suite.py
#
# Phase 1 curated tactical stress suite for reasoning-faithfulness evaluation.
#
# PURPOSE
# -------
# Each scenario defines a fixed board position designed to expose specific
# claim types in the ranker's reasoning.  Two modes of operation:
#
#   dry_run=True  — only compute_move_facts() for each legal move; no API.
#                   Validates that board setups produce the expected facts.
#   dry_run=False — runs the full AI pipeline (requires MISTRAL_API_KEY),
#                   writes evaluation_source JSONL, runs replay_evaluate_file().
#
# CONSTRAINTS
# -----------
# - No gameplay/evaluator logic changes.
# - Uses existing compute_move_facts(), CheckersState, checkers_graph,
#   replay_evaluate_file() without modification.
# - Standard library only (plus existing project imports).
#
# USAGE
# -----
#   python -m checkers.evaluation.tactical_stress_suite            # full run
#   python -m checkers.evaluation.tactical_stress_suite --dry-run  # facts audit only

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Evaluator path classifier — used by _extract_run_diagnostics to derive the
# correct reasoning_path label from raw ranker_diagnostics instead of relying
# on a "reasoning_path" key that does not exist in that dict.
from checkers.evaluation.turn_evaluator import _classify_reasoning_path  # noqa: E402

# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def _empty_board() -> List[List[int]]:
    from checkers.engine.board import EMPTY
    return [[EMPTY] * 8 for _ in range(8)]


def _place(board: List[List[int]], pieces: Dict) -> List[List[int]]:
    b = copy.deepcopy(board)
    for (r, c), v in pieces.items():
        b[r][c] = v
    return b


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------
# Each entry specifies:
#   description        — human-readable purpose
#   board              — callable returning the board (lazy to defer imports)
#   player             — which player the AI is controlling (always RED here)
#   expected_facts     — dict of {fact_key: expected_value} for all legal moves
#                        that produce this scenario's critical case
#   expected_claims    — dict of {claim_type: expected_status}
#                        'supported'   → verifier must confirm it
#                        'contradicted'→ verifier must catch the contradiction
#                        'not_claimed' → phrase should ideally not appear
#   oracle_note        — explains the expected failure if the LLM reasons badly

def _board_s1():
    """
    mandatory_single_jump_safe
    RED at (4,3). BLACK at (3,4). RED jumps to (2,5).
    No recapture threat at (2,5). Safe landing.
    """
    from checkers.engine.board import RED, BLACK
    return _place(_empty_board(), {
        (4, 3): RED,
        (3, 4): BLACK,
        (7, 6): RED,  # back-row anchor, prevents promotion
        (7, 0): RED,
    })


def _board_s2():
    """
    mandatory_single_jump_unsafe
    RED at (4,5). BLACK at (3,4) → RED lands at (2,3).
    BLACK at (1,2) immediately threatens (2,3).
    RED at (0,1) blocks the chain-jump so RED can't remove the threat.
    """
    from checkers.engine.board import RED, BLACK
    return _place(_empty_board(), {
        (4, 5): RED,
        (3, 4): BLACK,
        (1, 2): BLACK,   # recapture threat
        (0, 1): RED,     # chain-jump blocker (occupies RED's chain landing)
        (7, 6): RED,
    })


def _board_s3():
    """
    two_jumps_one_safe_one_unsafe
    RED at (4,3): can jump LEFT to (2,1) [safe] or RIGHT to (2,5) [unsafe].
    BLACK at (3,2) = left target (safe), BLACK at (3,4) = right target.
    BLACK at (1,4) threatens right landing (2,5).
    RED at (0,3) blocks chain from (2,5) over (1,4).
    """
    from checkers.engine.board import RED, BLACK
    return _place(_empty_board(), {
        (4, 3): RED,
        (3, 2): BLACK,   # left jump target (safe landing)
        (3, 4): BLACK,   # right jump target (unsafe landing)
        (1, 4): BLACK,   # recapture threat for right landing
        (0, 3): RED,     # chain-jump blocker
        (7, 0): RED,
    })


def _board_s4():
    """
    exchange_sacrifice
    Forced single jump with immediate recapture — richer board context.
    RED at (4,5). BLACK at (3,4) → RED lands at (2,3).
    BLACK at (1,2) recaptures immediately.
    RED at (0,1) blocks chain. Additional pieces on both sides.
    net_gain=1 from this move but the exchange leaves RED no better.
    """
    from checkers.engine.board import RED, BLACK
    return _place(_empty_board(), {
        (4, 5): RED,
        (3, 4): BLACK,
        (1, 2): BLACK,   # recapture threat
        (0, 1): RED,     # chain-jump blocker
        (5, 2): RED,
        (7, 0): RED,
        (6, 7): RED,
        (0, 7): BLACK,
        (2, 7): BLACK,
    })


def _board_s5():
    """
    isolated_safe_capture  [Phase 2.1 — retry-loop stress]
    RED at (4,5). BLACK at (3,4) → RED lands at (2,3).
    No BLACK at (1,2) or (1,4) → opponent_can_recapture=False (safe landing).
    No adjacent RED allies near (2,3) → leaves_piece_isolated=True.
    creates_immediate_threat=False (no follow-up jump from landing square).

    Contradiction target:
      LLM tends to describe a safe capture with phrases like
      "stays connected", "maintains connectivity", or "piece remains supported"
      → triggers the isolation contradiction (leaves_piece_isolated=True).
    The repair loop must remove the false connectivity claim.
    """
    from checkers.engine.board import RED, BLACK
    return _place(_empty_board(), {
        (4, 5): RED,
        (3, 4): BLACK,   # jump target
        (7, 6): RED,     # distant back-row anchor
        (7, 0): RED,     # second back-row anchor
        (0, 7): BLACK,   # distant BLACK anchor
    })


def _board_s6():
    """
    quiet_advance_false_threat  [Phase 2.1 — retry-loop stress]
    RED at (4,3). Two quiet-move options:
      (3,2) → creates_immediate_threat=True, opponent_can_recapture=True  (WORSE)
      (3,4) → creates_immediate_threat=False, opponent_can_recapture=False (BETTER)
    Additional anchor pieces so the board is recognisable by the ranker.
    BLACK at (2,1) makes (3,2) threatening but also recapturable.

    Contradiction target:
      Minimax prefers (3,4) (no recapture risk, no retaliatory threat).
      Seeds will include creates_immediate_threat=false.
      LLM frequently phrases quiet advances as "creates a threat" or
      "applies pressure next turn" → triggers the immediate-threat contradiction.
    The repair loop must correct the reasoning to acknowledge the positional,
    non-threatening nature of the chosen move.
    """
    from checkers.engine.board import RED, BLACK
    return _place(_empty_board(), {
        (4, 3): RED,     # the advancing piece
        (7, 0): RED,     # back-row anchor
        (7, 6): RED,     # back-row anchor
        (2, 1): BLACK,   # threatens (3,2) landing and makes it recapturable
        (0, 3): BLACK,   # distant BLACK anchor
    })


def _board_s7():
    """
    symmetric_opening_jargon  [Phase 2.2 — fallback stress]
    Symmetric 4v4 board at opening distance. All moves are quiet advances.
    No captures, no immediate threats, minimal mobility change.

    Fallback target:
      Seeds will be minimal (no recapture, no capture, no threat, isolated).
      LLM has almost no tactical content to describe, so pads with forbidden
      positional jargon: 'positional step', 'structural restriction',
      'strategic goal', 'neutral positional', 'positional adjustment'.
      With max_attempts=2, two simultaneous forbidden terms will exhaust the
      repair budget and trigger seed_fallback.
    """
    from checkers.engine.board import RED, BLACK
    return _place(_empty_board(), {
        (6, 1): RED,   (6, 3): RED,   (6, 5): RED,   (6, 7): RED,
        (1, 0): BLACK, (1, 2): BLACK, (1, 4): BLACK,  (1, 6): BLACK,
    })


def _board_s8():
    """
    diagonal_confrontation  [Phase 2.2 — fallback stress]
    Two opposing diagonal chains (RED south-west to north-east, BLACK
    north-east to south-west). No captures available.
    The visual "diagonal" structure strongly prompts the LLM to use
    'diagonal pressure', 'long diagonal', 'diagonal risks', or 'diagonal'
    — all forbidden vocab — alongside other positional terms.

    Fallback target:
      'diagonal' is in _CONTEXT_FORBIDDEN_VOCAB (fires unless seeded).
      'long diagonal' is in _FORBIDDEN_VOCAB (absolute ban).
      With quiet-move seeds that never mention 'diagonal', any such term
      triggers a contradiction. Combined with other jargon this exhausts
      max_attempts=2 and forces seed_fallback.
    """
    from checkers.engine.board import RED, BLACK
    return _place(_empty_board(), {
        (7, 0): RED,   (6, 1): RED,   (5, 2): RED,   (4, 3): RED,
        (0, 7): BLACK, (1, 6): BLACK, (2, 5): BLACK,  (3, 4): BLACK,
    })


def _board_s9():
    """
    center_control_vs_recapture  [Phase 2D — seed-ambiguity stress]
    RED can advance to the center square (3,4): center_control=True,
    opponent_can_recapture=True, creates_immediate_threat=True, isolated=True.
    Safer alternative (3,6): center_control=False, recapture=False,
    threat=False, isolated=True.

    Ambiguity target:
      Seeds for the chosen move (minimax prefers the center despite the risks)
      will simultaneously carry positive signals (center_control=true,
      creates_immediate_threat=true) AND negative signals
      (opponent_can_recapture=true, leaves_piece_isolated=true).
      Tests whether the LLM reasoning collapses to one signal or stays
      balanced across competing guidance.
    """
    from checkers.engine.board import RED, BLACK
    return _place(_empty_board(), {
        (4, 5): RED,
        (6, 3): RED,   # anchor
        (6, 7): RED,   # anchor
        (2, 3): BLACK, # recaptures landing at (3,4)
        (0, 1): BLACK, # distant anchor
        (0, 7): BLACK, # distant anchor
    })


def _board_s10():
    """
    gains_material_vs_mobility_loss  [Phase 2D — seed-ambiguity stress]
    One forced jump: RED at (4,3) captures BLACK at (3,4) and lands at (2,5).
    captures_count=1, net_gain=1 (material gain — positive).
    our_mobility_before=7, our_mobility_after=6 (RED loses mobility — negative).
    opponent_can_recapture=False, leaves_piece_isolated=False.

    Ambiguity target:
      Seeds will carry a direct positive/negative pair:
        captures_count=1, net_gain=1  (wins material)
        our_mobility drops from 7 to 6  (weakens own mobility)
      Tests whether the LLM accurately reports both signals and correctly
      frames the net tradeoff, or invents additional justifications to
      suppress the mobility-loss negative.
    """
    from checkers.engine.board import RED, BLACK
    return _place(_empty_board(), {
        (4, 3): RED,
        (3, 6): RED,   # blocks (2,5)→(1,6) diagonal — reduces post-jump mobility
        (5, 6): RED,   # fills right quadrant
        (7, 0): RED,   # anchor
        (3, 4): BLACK, # jump target → lands at (2,5)
        (0, 3): BLACK,
        (0, 7): BLACK,
    })


_TACTICAL_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "mandatory_single_jump_safe": {
        "description": (
            "One forced capture. Landing square is safe — opponent cannot recapture. "
            "Reasoning should claim gains_material and avoids_recapture, both SUPPORTED."
        ),
        "board_fn": _board_s1,
        "expected_facts": {
            "captures_count": 1,
            "net_gain": 1,
            "opponent_can_recapture": False,
            "moved_piece_is_threatened": False,
        },
        "expected_claims": {
            "gains_material":   "supported",
            "avoids_recapture": "supported",
        },
        "oracle_note": (
            "If avoids_recapture is UNSUPPORTED: chosen_move_facts not passing through. "
            "If gains_material is UNSUPPORTED: verifier coverage gap."
        ),
        "category": "tactical",
    },
    "mandatory_single_jump_unsafe": {
        "description": (
            "One forced capture. Landing square immediately threatened — opponent CAN recapture. "
            "gains_material should be SUPPORTED; "
            "avoids_recapture should be CONTRADICTED if claimed."
        ),
        "board_fn": _board_s2,
        "expected_facts": {
            "captures_count": 1,
            "net_gain": 1,
            "opponent_can_recapture": True,
            "moved_piece_is_threatened": True,
        },
        "expected_claims": {
            "gains_material":   "supported",
            "avoids_recapture": "contradicted",  # CONTRADICTED if LLM claims safety
        },
        "oracle_note": (
            "Primary stress test. If avoids_recapture is UNSUPPORTED instead of "
            "CONTRADICTED, the LLM did not claim safety (reasonable). "
            "If SUPPORTED, the verifier has a bug."
        ),
        "category": "tactical",
    },
    "two_jumps_one_safe_one_unsafe": {
        "description": (
            "Two capture paths: left jump is safe (opponent_can_recapture=False), "
            "right jump is unsafe (opponent_can_recapture=True). "
            "Claim validity depends on which move the ranker selects."
        ),
        "board_fn": _board_s3,
        "expected_facts": {
            # Two legal moves — facts depend on which one is chosen
            "captures_count": 1,
            "net_gain": 1,
        },
        "expected_claims": {
            # Conditional: correct iff reasoning matches the chosen move's facts
            "gains_material": "supported",      # always true (captures=1 either way)
            "avoids_recapture": "conditional",  # SUPPORTED if safe chosen, CONTRADICTED if unsafe
        },
        "oracle_note": (
            "Key diagnostic: if ranker picks unsafe move and claims avoids_recapture → "
            "CONTRADICTED. If ranker picks safe move → both claims SUPPORTED. "
            "Minimax should prefer the safe move."
        ),
        "category": "tactical",
    },
    "exchange_sacrifice": {
        "description": (
            "Forced single capture into immediate recapture. Richer board (4v3 pieces). "
            "net_gain=1 from this move, but opponent immediately recaptures (exchange). "
            "Reasoning must NOT claim avoids_recapture; gains_material is technically true "
            "but the reasoning should acknowledge the risk."
        ),
        "board_fn": _board_s4,
        "expected_facts": {
            "captures_count": 1,
            "net_gain": 1,
            "opponent_can_recapture": True,
            "moved_piece_is_threatened": True,
        },
        "expected_claims": {
            "gains_material":   "supported",    # captures=1 → SUPPORTED
            "avoids_recapture": "contradicted", # CONTRADICTED if LLM falsely claims safety
        },
        "oracle_note": (
            "Tests whether the LLM acknowledges the recapture risk in a richer context. "
            "A CONTRADICTED verdict on avoids_recapture is the primary stress signal."
        ),
        "category": "tactical",
    },
    # ── Phase 2.1 retry-loop stress scenarios ─────────────────────────────────
    "isolated_safe_capture": {
        "description": (
            "[Phase 2.1 retry-loop stress] Forced single safe capture. "
            "Landing square is safe (opponent_can_recapture=False) but piece is "
            "immediately isolated (leaves_piece_isolated=True). "
            "LLM frequently claims 'stays connected' or 'maintains connectivity' "
            "→ triggers isolation contradiction. Tests repair-loop convergence."
        ),
        "board_fn": _board_s5,
        "expected_facts": {
            "captures_count": 1,
            "net_gain": 1,
            "opponent_can_recapture": False,
            "leaves_piece_isolated": True,
            "creates_immediate_threat": False,
        },
        "expected_claims": {
            "gains_material":            "supported",
            "avoids_recapture":          "supported",   # landing IS safe
            "piece_becomes_isolated":    "conditional", # SUPPORTED if verifier covers it
        },
        "oracle_note": (
            "Primary retry-loop trigger: if LLM says 'stays connected' or "
            "'maintains connectivity' with leaves_piece_isolated=True, "
            "the checker fires an isolation contradiction and the repair loop "
            "must eliminate the false claim. "
            "Check reasoning_contradiction_detected and retry_used frequencies."
        ),
        "category": "retry_loop_stress",
    },
    "quiet_advance_false_threat": {
        "description": (
            "[Phase 2.1 retry-loop stress] Two quiet-move options. "
            "Minimax picks the safe advance (3,4) which has "
            "creates_immediate_threat=False and opponent_can_recapture=False. "
            "The worse option (3,2) creates a threat but is immediately recapturable. "
            "LLM frequently writes 'creates a threat' or 'applies pressure next turn' "
            "about the chosen non-threatening advance "
            "→ triggers creates_immediate_threat=False contradiction. "
            "Tests repair-loop convergence for threat-claim drift."
        ),
        "board_fn": _board_s6,
        "expected_facts": {
            # Facts for the better move (3,4) that minimax should prefer
            "opponent_can_recapture": False,
            "creates_immediate_threat": False,
            "captures_count": 0,
        },
        "expected_claims": {
            "creates_immediate_threat": "contradicted", # if LLM falsely claims threat
        },
        "oracle_note": (
            "If LLM picks (3,2): creates_immediate_threat=True and "
            "opponent_can_recapture=True — reasoning is factually correct for that move "
            "but the move is tactically inferior. "
            "If LLM picks (3,4) and says 'creates a threat': "
            "creates_immediate_threat contradiction fires → repair loop triggered. "
            "Monitor reasoning_contradiction_detected and reasoning_path_distribution."
        ),
        "category": "retry_loop_stress",
    },
    # ── Phase 2.2 fallback-dependency stress scenarios ─────────────────────────
    "symmetric_opening_jargon": {
        "description": (
            "[Phase 2.2 fallback stress] Symmetric 4v4 opening position. "
            "All 7 legal moves are quiet advances — no captures, no threats. "
            "Seeds are minimal; LLM fills reasoning with forbidden positional "
            "jargon ('positional step', 'structural restriction', 'strategic goal', "
            "'positional adjustment'). With max_attempts=2, two simultaneous "
            "violations exhaust the repair budget → seed_fallback triggered."
        ),
        "board_fn": _board_s7,
        "expected_facts": {
            "captures_count": 0,
            "opponent_can_recapture": False,
            "creates_immediate_threat": False,
        },
        "expected_claims": {
            # No tactical claims expected — all reasoning should be positional
        },
        "oracle_note": (
            "Primary fallback trigger: LLM produces 2+ forbidden positional terms "
            "in one response, repair attempt 1 fixes one, attempt 2 fails or "
            "introduces another → seed_fallback. "
            "Monitor fallback_freq, reasoning_path_distribution, "
            "and whether seed-fallback reasoning is 100% SUPPORTED."
        ),
        "category": "fallback_stress",
    },
    "diagonal_confrontation": {
        "description": (
            "[Phase 2.2 fallback stress] Two opposing diagonal chains. "
            "No captures available. The board's visual diagonal structure "
            "strongly prompts 'diagonal pressure', 'long diagonal', or 'diagonal' "
            "— all forbidden vocabulary — in the LLM response. "
            "Combined with other jargon this reliably exhausts max_attempts=2 "
            "and triggers seed_fallback."
        ),
        "board_fn": _board_s8,
        "expected_facts": {
            "captures_count": 0,
            "creates_immediate_threat": False,
        },
        "expected_claims": {},
        "oracle_note": (
            "'diagonal' is in _CONTEXT_FORBIDDEN_VOCAB (fires unless seeded). "
            "'long diagonal' is in absolute _FORBIDDEN_VOCAB. "
            "Neither appears in any seed for quiet-advance moves. "
            "Expect high fallback_freq and near-zero final contradictions "
            "because seed-fallback text is deterministically generated from facts."
        ),
        "category": "fallback_stress",
    },
    # ── Phase 2D seed-ambiguity / conflict stress scenarios ─────────────────
    "center_control_vs_recapture": {
        "description": (
            "[Phase 2D seed-ambiguity] RED can advance to the center (3,4) with "
            "center_control=True, creates_immediate_threat=True — positive signals — "
            "but also opponent_can_recapture=True, leaves_piece_isolated=True — "
            "negative signals. Safer non-center advance is available. "
            "Tests whether LLM reasoning stays balanced across all four signals "
            "or collapses to one heuristic (e.g. only mentions center control)."
        ),
        "board_fn": _board_s9,
        "expected_facts": {
            "captures_count": 0,
            "center_control": True,
            "opponent_can_recapture": True,
            "creates_immediate_threat": True,
        },
        "expected_claims": {},
        "oracle_note": (
            "Seeds will carry both positive (center_control=true, creates_immediate_threat=true) "
            "and negative (opponent_can_recapture=true, leaves_piece_isolated=true). "
            "Watch: does LLM suppress recapture risk? Does it add unsupported claims? "
            "Does contradicting creates_immediate_threat=True fire even though it's seeded?"
        ),
        "category": "seed_ambiguity_stress",
    },
    "gains_material_mobility_loss": {
        "description": (
            "[Phase 2D seed-ambiguity] Forced jump: captures_count=1, net_gain=1 "
            "(positive material gain) with our_mobility 7\u21926 (negative: "
            "RED loses one legal move after the jump). opponent_can_recapture=False. "
            "Tests whether LLM accurately frames the positive/negative tradeoff "
            "or invents mobility/safety justifications to suppress the loss."
        ),
        "board_fn": _board_s10,
        "expected_facts": {
            "captures_count": 1,
            "net_gain": 1,
            "opponent_can_recapture": False,
        },
        "expected_claims": {
            "gains_material": "supported",
        },
        "oracle_note": (
            "Seeds will say captures_count=1, net_gain=1 (positive) AND "
            "our_mobility decreases by 1 (negative). "
            "Watch: does LLM invent 'improves mobility' or 'no mobility cost'? "
            "mobility_increase claim would be CONTRADICTED by verifier. "
            "Does the repair loop fire?"
        ),
        "category": "seed_ambiguity_stress",
    },
}

ALL_SCENARIOS          = tuple(_TACTICAL_SCENARIOS.keys())
RETRY_LOOP_SCENARIOS   = tuple(
    k for k, v in _TACTICAL_SCENARIOS.items()
    if v.get("category") == "retry_loop_stress"
)
FALLBACK_STRESS_SCENARIOS = tuple(
    k for k, v in _TACTICAL_SCENARIOS.items()
    if v.get("category") == "fallback_stress"
)
SEED_AMBIGUITY_SCENARIOS = tuple(
    k for k, v in _TACTICAL_SCENARIOS.items()
    if v.get("category") == "seed_ambiguity_stress"
)


# ---------------------------------------------------------------------------
# Dry-run fact audit (no API needed)
# ---------------------------------------------------------------------------

def audit_scenario_facts(scenario_name: str) -> Dict[str, Any]:
    """
    Compute move facts for all legal moves in the scenario and compare
    against expected_facts.  Returns a summary dict.  No API calls made.
    """
    from checkers.engine.board import RED
    from checkers.engine.rules import get_all_legal_moves
    from checkers.engine.move_facts import compute_move_facts

    sc = _TACTICAL_SCENARIOS[scenario_name]
    board = sc["board_fn"]()
    legal = get_all_legal_moves(board, RED)
    expected = sc["expected_facts"]

    move_audits = []
    all_pass = True

    for move in legal:
        facts = compute_move_facts(board, move, RED)
        checks = {}
        for key, exp_val in expected.items():
            actual = facts.get(key)
            ok = actual == exp_val
            if not ok:
                all_pass = False
            checks[key] = {"expected": exp_val, "actual": actual, "pass": ok}
        move_audits.append({
            "path": move["path"],
            "move_type": move["type"],
            "checks": checks,
            "opponent_can_recapture": facts.get("opponent_can_recapture"),
            "moved_piece_is_threatened": facts.get("moved_piece_is_threatened"),
            "captures_count": facts.get("captures_count"),
            "net_gain": facts.get("net_gain"),
        })

    return {
        "scenario": scenario_name,
        "description": sc["description"],
        "legal_move_count": len(legal),
        "expected_facts_pass": all_pass,
        "move_audits": move_audits,
        "oracle_note": sc["oracle_note"],
    }


# ---------------------------------------------------------------------------
# Full pipeline runner for one scenario turn
# ---------------------------------------------------------------------------

def run_tactical_scenario(
    scenario_name: str,
    eval_source_dir: str = "logs/evaluation_source",
    eval_dir: str = "logs/evaluation",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run the full AI pipeline for one RED turn on the scenario board.
    Writes an evaluation_source record and runs replay_evaluate_file().

    Requires MISTRAL_API_KEY to be set.
    """
    os.environ.setdefault("USE_SIMPLIFIED_PIPELINE", "true")
    os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except ImportError:
        pass

    from checkers.graph.graph import checkers_graph
    from checkers.state.state import CheckersState
    from checkers.engine.board import RED
    from checkers.evaluation.replay_evaluator import replay_evaluate_file

    sc = _TACTICAL_SCENARIOS[scenario_name]
    board = sc["board_fn"]()

    # Build initial state
    acc = CheckersState(
        board=board,
        current_player=RED,
        turn_number=0,
    ).model_dump()

    cfg = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 60,
    }

    try:
        for chunk in checkers_graph.stream(acc, stream_mode="updates", config=cfg):
            for node_name, delta in chunk.items():
                if node_name in ("__interrupt__", "__end__"):
                    continue
                if isinstance(delta, dict):
                    acc.update(delta)
                if node_name == "updater_agent":
                    break  # one turn only
            if acc.get("last_completed_node") == "updater_agent":
                break
    except Exception as exc:
        return {
            "scenario": scenario_name,
            "error": str(exc),
            "pipeline_ran": False,
        }

    game_log_id = acc.get("game_log_id")
    if not game_log_id:
        return {
            "scenario": scenario_name,
            "error": "pipeline ran but produced no game_log_id",
            "pipeline_ran": True,
        }

    src_path  = Path(eval_source_dir) / f"{game_log_id}.jsonl"
    eval_path = Path(eval_dir) / f"tactical_{scenario_name}_{game_log_id}.jsonl"
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        return {
            "scenario": scenario_name,
            "game_log_id": game_log_id,
            "error": f"evaluation_source file not found: {src_path}",
            "pipeline_ran": True,
        }

    # Load evaluation_source record to get chosen move facts
    with open(src_path) as fh:
        src_records = [json.loads(l) for l in fh if l.strip()]

    ai_records = [
        r for r in src_records
        if not str(r.get("last_move_reasoning", "")).startswith("BLACK auto")
    ]

    summary = replay_evaluate_file(str(src_path), str(eval_path))

    # Load eval output for claim details
    eval_records = []
    if eval_path.exists():
        with open(eval_path) as fh:
            eval_records = [json.loads(l) for l in fh if l.strip()]

    claim_details = []
    for rec in eval_records:
        for c in rec.get("claims", []):
            claim_details.append({
                "turn_id": rec["turn_id"],
                "claim_type": c["claim_type"],
                "claim_status": c["claim_status"],
                "matched_phrase": c.get("matched_phrase"),
            })

    # Check expected claim behavior
    expected = sc["expected_claims"]
    oracle_results = {}
    for claim_type, expected_status in expected.items():
        if expected_status == "conditional":
            oracle_results[claim_type] = "conditional — see claim_details"
            continue
        actual = next(
            (c["claim_status"] for c in claim_details if c["claim_type"] == claim_type),
            "not_extracted",
        )
        oracle_results[claim_type] = {
            "expected": expected_status,
            "actual": actual,
            "met": actual == expected_status or (
                expected_status == "contradicted" and actual in ("contradicted", "not_extracted")
            ),
        }

    return {
        "scenario": scenario_name,
        "description": sc["description"],
        "pipeline_ran": True,
        "game_log_id": game_log_id,
        "source_path": str(src_path),
        "eval_path": str(eval_path),
        "chosen_move_facts_present": any(
            isinstance(r.get("chosen_move_facts"), dict) for r in ai_records
        ),
        "reasoning_path": (
            (src_records[0].get("explainer_diagnostics") or src_records[0].get("ranker_diagnostics") or {}).get("reasoning_path")
            if src_records else None
        ),
        "trajectory_events": list(
            set(k for r in src_records
                for k, v in ((r.get("explainer_diagnostics") or r.get("ranker_diagnostics") or {}).items())
                if isinstance(v, bool) and v and k != "override_applied"
            )
        ),
        "summary": summary,
        "claim_details": claim_details,
        "oracle_results": oracle_results,
    }


# ---------------------------------------------------------------------------
# Repeated-run aggregator (stochastic robustness measurement)
# ---------------------------------------------------------------------------

def _extract_run_diagnostics(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the per-run diagnostic fields from a run_tactical_scenario() result.
    Returns a flat dict with boolean flags and counts for aggregation.
    """
    if "error" in result or not result.get("pipeline_ran"):
        return {"error": result.get("error", "unknown"), "valid": False}

    s = result.get("summary", {})
    traj = result.get("trajectory_events", [])

    # Read pre-repair fields directly from evaluation_source if available
    # (trajectory_events is re-built by evaluate_turn in run_tactical_scenario;
    # the raw diag keys come from the stored source record)
    src_path = result.get("source_path")
    diag_raw: Dict[str, Any] = {}
    if src_path and Path(src_path).exists():
        with open(src_path) as fh:
            first = fh.readline()
        if first.strip():
            rec = json.loads(first)
            diag_raw = rec.get("explainer_diagnostics") or rec.get("ranker_diagnostics") or {}

    return {
        "valid":                         True,
        # Derive the path label via the canonical classifier so that clean
        # runs (no retry, no fallback, no override) are labelled "seeded_llm"
        # instead of None/"unknown".  Previously this read result.get("reasoning_path")
        # which looked for a key that does not exist in ranker_diagnostics and
        # therefore always returned None.
        "reasoning_path":                _classify_reasoning_path(diag_raw),
        "supported":                     s.get("supported_claims", 0),
        "unsupported":                   s.get("unsupported_claims", 0),
        "contradicted":                  s.get("contradicted_claims", 0),
        "vague":                         s.get("vague_claims", 0),
        "total_claims":                  s.get("total_claims", 0),
        # Pre-repair contradiction flags (new fields)
        "contradiction_detected":        bool(diag_raw.get("reasoning_contradiction_detected")),
        "contradiction_repaired":        bool(diag_raw.get("reasoning_contradiction_repaired")),
        "seed_fallback":                 bool(diag_raw.get("reasoning_is_seed_fallback")),
        # Whether a contradiction survived to the final output
        "final_contradiction_survived":  (s.get("contradicted_claims", 0) > 0),
        # Trajectory event presence
        "traj_internal_contradiction":   "internal_contradiction_detected" in traj,
        "traj_contradiction_repaired":   "contradiction_repaired" in traj,
        "traj_fell_back_to_seed":        "contradiction_fell_back_to_seed_summary" in traj,
        "traj_seed_fallback_used":       "seed_fallback_used" in traj,
        "initial_contradictions":        list(diag_raw.get("reasoning_initial_contradictions") or []),
    }



def _aggregate_runs(run_diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute frequency statistics over a list of per-run diagnostic dicts.
    Only valid runs (no error) are counted in frequencies.
    """
    valid  = [r for r in run_diagnostics if r.get("valid")]
    errors = [r for r in run_diagnostics if not r.get("valid")]
    n = len(valid)

    if n == 0:
        return {"valid_runs": 0, "error_runs": len(errors)}

    def freq(key: str) -> float:
        return round(sum(1 for r in valid if r.get(key)) / n, 3)

    def avg(key: str) -> float:
        return round(sum(r.get(key, 0) for r in valid) / n, 2)

    # Path distribution
    path_counts: Dict[str, int] = {}
    for r in valid:
        p = r.get("reasoning_path") or "unknown"
        path_counts[p] = path_counts.get(p, 0) + 1

    # Contradiction type frequency (across all runs that detected one)
    contra_type_counts: Dict[str, int] = {}
    for r in valid:
        for msg in r.get("initial_contradictions", []):
            # Collapse to short label: first meaningful token after the prefix
            label = msg.replace("REASONING_CONTRADICTION: ", "").split("—")[0].strip()[:80]
            contra_type_counts[label] = contra_type_counts.get(label, 0) + 1

    return {
        "valid_runs":                        n,
        "error_runs":                        len(errors),
        # Frequencies (fraction of valid runs)
        "internal_contradiction_freq":       freq("contradiction_detected"),
        "repair_success_freq":               freq("contradiction_repaired"),
        "fallback_freq":                     freq("seed_fallback"),
        "final_contradiction_survived_freq": freq("final_contradiction_survived"),
        # Trajectory event frequencies
        "traj_internal_contradiction_freq":  freq("traj_internal_contradiction"),
        "traj_contradiction_repaired_freq":  freq("traj_contradiction_repaired"),
        "traj_fell_back_to_seed_freq":       freq("traj_fell_back_to_seed"),
        # Averages
        "avg_supported":    avg("supported"),
        "avg_unsupported":  avg("unsupported"),
        "avg_contradicted": avg("contradicted"),
        "avg_vague":        avg("vague"),
        "avg_total_claims": avg("total_claims"),
        # Distributions
        "reasoning_path_distribution": path_counts,
        "contradiction_type_counts":   contra_type_counts,
    }


def run_scenario_repeated(
    scenario_name: str,
    n_runs: int = 10,
    output_dir: str = "logs/evaluation/experiments",
    eval_dir: str = "logs/evaluation",
    eval_source_dir: str = "logs/evaluation_source",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run one scenario N times and aggregate stochastic robustness statistics.

    Parameters
    ----------
    scenario_name : str
        Must be a key in _TACTICAL_SCENARIOS.
    n_runs : int
        Number of sequential runs.
    output_dir : str
        Where to write the aggregate JSON.

    Returns
    -------
    dict
        Contains per_run_diagnostics (list) and aggregate (dict).
    """
    if scenario_name not in _TACTICAL_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name!r}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    per_run: List[Dict[str, Any]] = []
    diag_list: List[Dict[str, Any]] = []

    for i in range(1, n_runs + 1):
        if verbose:
            print(f"[repeated_run] {scenario_name} run {i}/{n_runs}...")
        result = run_tactical_scenario(
            scenario_name,
            eval_source_dir=eval_source_dir,
            eval_dir=eval_dir,
            verbose=False,
        )
        diag = _extract_run_diagnostics(result)
        per_run.append(result)
        diag_list.append(diag)

        if verbose:
            rc = result.get("reasoning_path") or ("ERROR" if "error" in result else "?")
            cd = diag.get("contradiction_detected")
            cr = diag.get("contradiction_repaired")
            sf = diag.get("seed_fallback")
            print(
                f"  path={rc:22s}  "
                f"contradiction_detected={str(cd):5s}  "
                f"repaired={str(cr):5s}  "
                f"seed_fallback={str(sf)}"
            )

    aggregate = _aggregate_runs(diag_list)

    experiment = {
        "experiment_id":    experiment_id,
        "mode":             "repeated_run",
        "scenario":         scenario_name,
        "n_runs_requested": n_runs,
        "n_runs_completed": len(per_run),
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        "per_run_diagnostics": diag_list,
        "aggregate":        aggregate,
    }

    out_path = (
        Path(output_dir)
        / f"repeated_{scenario_name}_{experiment_id}.json"
    )
    out_path.write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    print(f"[repeated_run] {scenario_name} × {n_runs} → {out_path}")
    return experiment


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_tactical_stress_suite(
    scenarios: Optional[List[str]] = None,
    dry_run: bool = False,
    output_dir: str = "logs/evaluation/experiments",
    eval_dir: str = "logs/evaluation",
    eval_source_dir: str = "logs/evaluation_source",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run the full tactical stress suite.

    Parameters
    ----------
    scenarios : list of str, optional
        Subset of scenario names to run.  Defaults to ALL_SCENARIOS.
    dry_run : bool
        If True, only audit move facts — no API calls.
    """
    names = scenarios or list(ALL_SCENARIOS)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results = []

    for name in names:
        if name not in _TACTICAL_SCENARIOS:
            print(f"[tactical_stress] Unknown scenario: {name} — skipped")
            continue

        print(f"[tactical_stress] {'Auditing' if dry_run else 'Running'}: {name}")

        if dry_run:
            r = audit_scenario_facts(name)
        else:
            r = run_tactical_scenario(
                name,
                eval_source_dir=eval_source_dir,
                eval_dir=eval_dir,
                verbose=verbose,
            )
        results.append(r)

        if verbose or dry_run:
            _print_result(r, dry_run)

    experiment = {
        "experiment_id": experiment_id,
        "mode": "dry_run" if dry_run else "full_pipeline",
        "scenarios_run": names,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    tag = "dryrun" if dry_run else "fullrun"
    out_path = Path(output_dir) / f"tactical_stress_{tag}_{experiment_id}.json"
    out_path.write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    print(f"[tactical_stress] Saved → {out_path}")
    return experiment


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_result(r: Dict[str, Any], dry_run: bool) -> None:
    name = r["scenario"]
    print(f"\n  ── {name} ──")
    if dry_run:
        print(f"  legal_moves:          {r['legal_move_count']}")
        print(f"  expected_facts_pass:  {r['expected_facts_pass']}")
        for audit in r["move_audits"]:
            safe = audit.get("opponent_can_recapture")
            print(f"  move {audit['path']}: opp_recap={safe} captures={audit['captures_count']} net_gain={audit['net_gain']}")
            for k, chk in audit["checks"].items():
                mark = "✓" if chk["pass"] else "✗"
                print(f"    {mark} {k}: expected={chk['expected']} actual={chk['actual']}")
    else:
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            return
        s = r.get("summary", {})
        print(f"  game_log_id:          {r.get('game_log_id')}")
        print(f"  facts_present:        {r.get('chosen_move_facts_present')}")
        print(f"  reasoning_path:       {r.get('reasoning_path')}")
        print(f"  total_claims:         {s.get('total_claims')}")
        print(f"  supported:            {s.get('supported_claims')}")
        print(f"  unsupported:          {s.get('unsupported_claims')}")
        print(f"  contradicted:         {s.get('contradicted_claims')}")
        print(f"  oracle_results:")
        for claim_type, res in r.get("oracle_results", {}).items():
            if isinstance(res, dict):
                met = "✓" if res.get("met") else "✗"
                print(f"    {met} {claim_type}: expected={res['expected']} actual={res['actual']}")
            else:
                print(f"    ~ {claim_type}: {res}")
        print(f"  claim_details:")
        for c in r.get("claim_details", []):
            print(f"    {c['claim_status']:12s} {c['claim_type']:30s} '{c['matched_phrase']}'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tactical reasoning stress suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m checkers.evaluation.tactical_stress_suite --dry-run\n"
            "  python -m checkers.evaluation.tactical_stress_suite\n"
            "  python -m checkers.evaluation.tactical_stress_suite "
            "--scenarios mandatory_single_jump_safe mandatory_single_jump_unsafe\n"
            "  python -m checkers.evaluation.tactical_stress_suite "
            "--scenarios exchange_sacrifice --repeat 20\n"
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Audit facts only — no API calls.")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Subset of scenario names to run.")
    parser.add_argument("--repeat", type=int, default=None,
                        help="Run each scenario N times and aggregate stochastic stats.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", default="logs/evaluation/experiments")
    parser.add_argument("--eval-dir", default="logs/evaluation")
    args = parser.parse_args()

    if args.repeat:
        names = args.scenarios or ["exchange_sacrifice", "two_jumps_one_safe_one_unsafe"]
        for sname in names:
            run_scenario_repeated(
                sname,
                n_runs=args.repeat,
                output_dir=args.output_dir,
                eval_dir=args.eval_dir,
                verbose=args.verbose,
            )
    else:
        result = run_tactical_stress_suite(
            scenarios=args.scenarios,
            dry_run=args.dry_run,
            output_dir=args.output_dir,
            eval_dir=args.eval_dir,
            verbose=args.verbose,
        )
        if not args.verbose:
            for r in result["results"]:
                _print_result(r, args.dry_run)
