# checkers/agents/update_agent.py
#
# Simplified-pipeline composite end-of-turn node.
#
# Execution sequence
# ──────────────────
#   Phase A  Apply the chosen move   (delegates to state_manager)
#   Phase B  Check end conditions    (delegates to win_condition)
#   Phase C  Log the completed turn  (delegates to logger_node)
#
# Terminal case
# ─────────────
# When the current player has no legal moves (stuck = loss), Phase A is
# skipped (defensive guard on state.chosen_move) and win_condition is
# invoked on the unmodified board. It correctly identifies the loser
# because current_player has not yet been switched. chosen_move is owned
# entirely by deterministic_proposal_node — ranker_agent never assigns it.
#
# Player perspective after Phase A
# ─────────────────────────────────
# state_manager switches current_player before returning, so
# post_move_state.current_player == the player who moves NEXT turn.
# win_condition already accounts for this (it computes player_who_just_moved
# as the opposite of current_player).

from __future__ import annotations

from checkers.state.state import CheckersState
from checkers.nodes.state_manager import state_manager
from checkers.nodes.win_condition import win_condition
from checkers.nodes.logger_node import logger_node


def update_agent(state: CheckersState) -> dict:
    """
    Composite end-of-turn node for the simplified pipeline.

    Executes state_manager → win_condition → logger_node in a single
    controlled sequence. Returns a merged dict of every state field changed
    across all phases so LangGraph can apply them atomically.

    Evaluation-field lifecycle
    ──────────────────────────
    chosen_move_facts is captured by ranker_agent (a read-only mirror of
    the proposal-chosen move's facts) and CLEARED by state_manager. To
    let logger_node (Phase C) export it for the evaluation-source JSONL,
    we snapshot it here before Phase A runs, then restore it only in a
    temporary log-only state copy passed to logger_node. The final merged
    dict still receives None, preventing any leakage into the next turn.
    """

    # ── Evaluation-field snapshot (before Phase A clears it) ───────────────
    _eval_chosen_facts = state.chosen_move_facts
    _eval_chosen_move_score = state.chosen_move_score
    _eval_proposal_diagnostics = state.proposal_diagnostics

    # ── Phase A: Apply chosen move ────────────────────────────────────────────
    if state.chosen_move is not None:
        sm_result = state_manager(state)
        post_move_state = state.model_copy(update=sm_result)
    else:
        sm_result = {}
        post_move_state = state

    # ── Phase B: End-condition checks ─────────────────────────────────────────
    wc_result = win_condition(post_move_state)
    post_wc_state = post_move_state.model_copy(update=wc_result)

    # ── Phase C: Logging ──────────────────────────────────────────────────────
    _log_state = post_wc_state.model_copy(
        update={
            "chosen_move_facts": _eval_chosen_facts,
            "chosen_move_score": _eval_chosen_move_score,
            "proposal_diagnostics": _eval_proposal_diagnostics,
        }
    )
    log_result = logger_node(_log_state)

    # ── Merge and stamp ───────────────────────────────────────────────────────
    merged = {**sm_result, **wc_result, **log_result}
    merged["last_completed_node"] = "update_agent"
    return merged
