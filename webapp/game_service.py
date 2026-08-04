#!/usr/bin/env python3
"""
game_service.py — Web-friendly wrapper around the simplified checkers pipeline.

This mirrors the turn logic in run_simplified_trace.py (the old terminal
runner) but returns structured, JSON-serializable data instead of printing
to stdout, so a browser frontend (webapp/static/index.html) can render the
same information — board, legal moves, scorer/proposal/reasoning trace,
move history — that used to only appear in the terminal.

Nothing about the underlying pipeline (graph, agents, engine) is changed.
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_SIMPLIFIED_PIPELINE", "true")
os.environ.setdefault("CHECKERS_LOGGER_PRINT", "false")

from dotenv import load_dotenv  # type: ignore
load_dotenv()

import uuid
from typing import Any

from checkers.graph.graph import checkers_graph
from checkers.state.state import CheckersState
from checkers.agents.updater_agent import updater_agent as _update_agent_fn
from checkers.engine.board import RED, BLACK, create_initial_board
from checkers.engine.move_facts import count_pieces
from checkers.engine.rules import get_all_legal_moves


# ── helpers ──────────────────────────────────────────────────────────────────

def _clean(obj: Any) -> Any:
    """Recursively convert tuples to lists so the result is JSON-serializable."""
    if isinstance(obj, tuple):
        return [_clean(x) for x in obj]
    if isinstance(obj, list):
        return [_clean(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    return obj


def _fmt_move_label(m: dict) -> str:
    path = m.get("path") or []
    cap = m.get("captured") or []
    if len(path) >= 2:
        a, b = path[0], path[-1]
        label = f"{m.get('type', '?')} [{a[0]},{a[1]}] \u2192 [{b[0]},{b[1]}]"
    else:
        label = str(m)
    if cap:
        label += f"  (captures {len(cap)})"
    return label


# ── session ──────────────────────────────────────────────────────────────────

class GameSession:
    """Holds one running game's state and exposes turn-by-turn actions.

    A single instance is shared by the Flask app (webapp/app.py) — this is a
    local, single-user dev tool, so an in-memory global session is enough.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.acc: dict[str, Any] = CheckersState(
            board=create_initial_board(),
            current_player=RED,
            turn_number=0,
        ).model_dump()

    # ── read-only views ───────────────────────────────────────────────────

    def board_view(self) -> dict[str, Any]:
        acc = self.acc
        rc = count_pieces(acc["board"], RED)
        bc = count_pieces(acc["board"], BLACK)
        return _clean({
            "board": acc["board"],
            "current_player": acc.get("current_player"),
            "turn_number": acc.get("turn_number"),
            "game_over": acc.get("game_over"),
            "winner": acc.get("winner"),
            "draw": acc.get("draw"),
            "red_pieces": rc,
            "black_pieces": bc,
            "move_history": [
                {
                    "player": r.get("player"),
                    "move_label": _fmt_move_label(r.get("move") or {}),
                    "promotion": bool(r.get("promotion")),
                }
                for r in (acc.get("move_history") or [])
            ],
            "game_log_id": acc.get("game_log_id"),
        })

    def legal_moves_view(self) -> list[dict[str, Any]]:
        player = self.acc["current_player"]
        moves = get_all_legal_moves(self.acc["board"], player)
        out = []
        for i, m in enumerate(moves):
            out.append(_clean({
                "index": i,
                "type": m.get("type"),
                "path": m.get("path"),
                "captured": m.get("captured", []),
                "label": _fmt_move_label(m),
            }))
        return out

    # ── actions ────────────────────────────────────────────────────────────

    def play_red_ply(self) -> dict[str, Any]:
        """Runs one RED (AI) ply through the LangGraph pipeline and returns a
        structured trace: scorer output, proposal, reasoning, applied move."""
        acc = self.acc
        trace: dict[str, Any] = {
            "scorer": None, "proposal": None, "explainer": None,
            "applied": None, "error": None,
        }

        acc["last_completed_node"] = None
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}

        try:
            for chunk in checkers_graph.stream(
                acc, stream_mode="updates",
                interrupt_after=["updater_agent"], config=cfg,
            ):
                for node_name, delta in chunk.items():
                    if node_name in ("__interrupt__", "__end__") or not isinstance(delta, dict):
                        continue
                    acc.update(delta)

                    if node_name == "scorer_node":
                        lm = acc.get("legal_moves") or []
                        trace["scorer"] = _clean({
                            "moves": [
                                {
                                    "label": _fmt_move_label(m),
                                    "score": (m.get("facts") or {}).get("minimax_score"),
                                    "rank": (m.get("facts") or {}).get("symbolic_rank"),
                                } for m in lm
                            ],
                            "best_score": acc.get("symbolic_best_score"),
                            "gap": acc.get("symbolic_gap"),
                        })

                    elif node_name == "proposer_agent":
                        cm = acc.get("chosen_move")
                        p_diag = acc.get("proposal_diagnostics") or {}
                        unchosen = acc.get("unchosen_moves") or []
                        trace["proposal"] = _clean({
                            "chosen_label": _fmt_move_label(cm) if cm else None,
                            "score": acc.get("chosen_move_score"),
                            "gap_to_2nd": p_diag.get("gap"),
                            "method": p_diag.get("selection_method"),
                            "n_legal": p_diag.get("n_legal"),
                            "alternatives": [_fmt_move_label(m) for m in unchosen[:3]],
                        })

                    elif node_name == "explainer_agent":
                        diag = acc.get("explainer_diagnostics") or {}
                        trace["explainer"] = _clean({
                            "reasoning": (acc.get("last_move_reasoning") or "").strip(),
                            "final_choice_source": diag.get("final_choice_source", "unknown"),
                            "seeds": len(diag.get("reasoning_seeds") or []),
                            "contradictions": len(diag.get("reasoning_initial_contradictions") or []),
                            "seed_fallback": diag.get("reasoning_is_seed_fallback", False),
                        })

                    elif node_name == "updater_agent":
                        mh = acc.get("move_history") or []
                        if mh:
                            last = mh[-1]
                            trace["applied"] = _clean({
                                "player": last.get("player"),
                                "move_label": _fmt_move_label(last.get("move") or {}),
                                "promotion": bool(last.get("promotion")),
                            })
        except Exception as e:
            trace["error"] = str(e)

        return trace

    def play_black_move(self, index: int) -> dict[str, Any]:
        """Applies a human (BLACK) move chosen by index into the current
        legal-move list, directly via updater_agent (bypassing the AI graph,
        same approach as run_simplified_trace.py's _run_black_ply)."""
        acc = self.acc
        legal = get_all_legal_moves(acc["board"], BLACK)
        if not (0 <= index < len(legal)):
            return {"error": f"Invalid move index {index}; valid range 0-{len(legal) - 1}"}

        move = legal[index]
        acc["chosen_move"] = move
        acc["last_move_reasoning"] = "BLACK human move"

        valid_fields = set(CheckersState.model_fields.keys())
        state = CheckersState(**{k: v for k, v in acc.items() if k in valid_fields})
        result = _update_agent_fn(state)
        acc.update(result)

        ok = result.get("last_completed_node") == "updater_agent"
        return _clean({"ok": ok, "move_label": _fmt_move_label(move)})
