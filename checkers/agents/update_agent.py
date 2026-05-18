# checkers/agents/update_agent.py
#
# Simplified-pipeline composite end-of-turn node.
# Only active when USE_SIMPLIFIED_PIPELINE=true.
# Old nodes are NOT modified — they are called directly here.
#
# Execution sequence
# ──────────────────
#   Phase A  Apply the chosen move         (delegates to state_manager)
#   Phase B  Check end conditions          (delegates to win_condition)
#   Phase C  Log the completed turn        (delegates to logger_node)
#   Phase D  Prepare next-turn context     (delegates to inter_turn_memory)
#            Skipped when game_over is True.
#
# Terminal case
# ─────────────
# ranker_agent sets chosen_move=None when legal_moves is empty (current
# player is stuck = loss). Phase A is skipped; win_condition receives
# the unmodified board and determines the winner correctly because
# current_player is still the stuck player at that point.
#
# Player perspective after Phase A
# ─────────────────────────────────
# state_manager switches current_player before returning, so
# post_move_state.current_player == the player who moves NEXT turn.
# win_condition already accounts for this (it computes player_who_just_moved
# as the opposite of current_player). inter_turn_memory therefore computes
# strategic_context from the next player's perspective — exactly correct.

from __future__ import annotations

from checkers.state.state import CheckersState
from checkers.nodes.state_manager import state_manager
from checkers.nodes.win_condition import win_condition
from checkers.nodes.logger_node import logger_node
from checkers.nodes.inter_turn_memory import inter_turn_memory


def update_agent(state: CheckersState) -> dict:
    """
    Composite end-of-turn node for the simplified pipeline.

    Executes state_manager → win_condition → logger_node → inter_turn_memory
    in a single controlled sequence. Returns a merged dict of every state
    field changed across all phases so LangGraph can apply them atomically.

    Old nodes remain registered in the graph for the old pipeline and are
    unchanged. This node only calls them — it does not replicate their logic.

    Evaluation-field lifecycle
    ──────────────────────────
    chosen_move_facts is set by ranker_agent and CLEARED by state_manager.
    To let logger_node (Phase C) export it for the evaluation-source JSONL,
    we snapshot it here before Phase A runs, then restore it only in a
    temporary log-only state copy passed to logger_node.  inter_turn_memory
    (Phase D) and the final merged dict still receive None, preventing any
    leakage into the next turn.  No decision logic is touched.
    """

    # ── Evaluation-field snapshot (before Phase A clears it) ───────────────
    # state_manager returns chosen_move_facts: None to clear the field for
    # the next turn. Snapshot here so logger_node can export the current
    # turn's facts into the evaluation-source JSONL without touching any
    # decision state.
    _eval_chosen_facts = state.chosen_move_facts

    # ── Phase A: Apply chosen move ────────────────────────────────────────────
    # Terminal guard: skip move application when ranker_agent received an
    # empty candidate list (chosen_move=None). The board stays as-is.
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
    # Build a log-only state copy with chosen_move_facts restored from the
    # pre-Phase-A snapshot so logger_node can write it to evaluation_source/.
    # This copy is NEVER merged back; inter_turn_memory still receives
    # post_wc_state (chosen_move_facts=None), preventing next-turn leakage.
    _log_state = post_wc_state.model_copy(
        update={"chosen_move_facts": _eval_chosen_facts}
    )
    log_result = logger_node(_log_state)

    # ── Phase D: Next-turn strategic context ──────────────────────────────────
    # Skipped when the game is over (no next turn to prepare for).
    # post_wc_state.current_player is already the next player to move, so
    # inter_turn_memory computes priorities from that player's perspective.
    itm_result: dict = {}
    if not post_wc_state.game_over:
        itm_result = inter_turn_memory(post_wc_state)

    # ── Merge and stamp ───────────────────────────────────────────────────────
    # Merge order: sm → wc → log → itm. Later dicts win on key conflicts.
    # Stamp last_completed_node so _update_agent_routing fires on this state.
    merged = {**sm_result, **wc_result, **log_result, **itm_result}
    merged["last_completed_node"] = "update_agent"
    return merged
