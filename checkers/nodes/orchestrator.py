# nodes/orchestrator.py
#
# The orchestrator is the central hub of the hub-and-spoke graph.
# Every node in the graph returns to the orchestrator after completing.
# The orchestrator itself does nothing — it is a pure passthrough.
#
# All routing decisions live in orchestrator_routing inside graph.py
# which reads state.last_completed_node and decides what runs next.
#
# This node must never:
#   - Modify any state fields
#   - Make routing decisions
#   - Call any engine functions
#   - Call any LLM
#
# It returns an empty dict because LangGraph merges return dicts
# into state — returning nothing means nothing changes.

from checkers.state.state import CheckersState


def orchestrator(state: CheckersState) -> dict:
    return {}
