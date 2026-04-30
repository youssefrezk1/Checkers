"""
endgame_eval_breakdown_audit.py

Read-only evaluator breakdown audit for known-failure endgame positions.

For each legal move in turns T41,T43,T45,T49,T51,T55,T59,T61,T63:
  - Apply the move to get the child board.
  - Call evaluate_board_breakdown on the child.
  - Record all eval terms plus move-geometry annotations.
  - Identify dominant positive terms, suspicious features, and likely missing
    evaluation signals.

Input:  logs/known_failure_positions_20260425_144451.json
Output: logs/endgame_eval_breakdown_audit_20260425_144451.json

No search is performed. Runtime < 10 s.
Does NOT modify any engine, evaluator, ranker, minimax, rules, or proposal code.
"""

from __future__ import annotations

import json
from pathlib import Path

from checkers.engine.board import RED, BLACK, RED_KING, BLACK_KING, BOARD_SIZE
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.engine.evaluation import (
    evaluate_board_breakdown,
    CAGED_KING_PENALTY,
    _king_positions,
    _is_king_caged,
)

IN_PATH  = "logs/known_failure_positions_20260425_144451.json"
OUT_PATH = "logs/endgame_eval_breakdown_audit_20260425_144451.json"
TARGET_TURNS = [41, 43, 45, 49, 51, 55, 59, 61, 63]

_CORNERS = {(0, 0), (0, 7), (7, 0), (7, 7)}
_EDGES   = {(r, c) for r in range(8) for c in range(8)
            if r in (0, 7) or c in (0, 7)} - _CORNERS


# ── geometry helpers ──────────────────────────────────────────────────────────

def _count_safe_exits(board: list[list[int]], kr: int, kc: int, player: int) -> int:
    """
    Count diagonal destinations for the king at (kr,kc) that are NOT
    immediately recapturable — i.e. exits that _is_king_caged would classify
    as safe.  Returns 0 if no legal exits at all (frozen king).
    """
    opp = BLACK if player == RED else RED
    opp_man  = opp
    opp_king = BLACK_KING if player == RED else RED_KING
    safe = 0
    for dr, dc in ((-1,-1),(-1,1),(1,-1),(1,1)):
        r2, c2 = kr+dr, kc+dc
        if not (0 <= r2 < BOARD_SIZE and 0 <= c2 < BOARD_SIZE):
            continue
        if board[r2][c2] != 0:
            continue
        # Check: can opponent immediately recapture from (r2,c2)?
        dest_safe = True
        for adr, adc in ((-1,-1),(-1,1),(1,-1),(1,1)):
            ar, ac = r2+adr, c2+adc
            lr, lc = r2-adr, c2-adc
            if not (0 <= ar < BOARD_SIZE and 0 <= ac < BOARD_SIZE): continue
            if not (0 <= lr < BOARD_SIZE and 0 <= lc < BOARD_SIZE): continue
            opp_piece = board[ar][ac]
            if opp_piece not in (opp_man, opp_king): continue
            if lr == kr and lc == kc:
                land_empty = True
            elif lr == r2 and lc == c2:
                land_empty = False
            else:
                land_empty = board[lr][lc] == 0
            if not land_empty: continue
            if opp_piece == opp_man:
                jdir = lr - ar
                if opp == RED and jdir >= 0: continue
                if opp == BLACK and jdir <= 0: continue
            dest_safe = False
            break
        if dest_safe:
            safe += 1
    return safe


def _delayed_cage_risk(board: list[list[int]], kr: int, kc: int, player: int) -> bool:
    """
    True when a king at (kr,kc) is NOT currently caged but is at a corner/edge
    and an opponent piece sits within 2 diagonal steps — meaning one opponent
    move could arm a cage trap.
    """
    if (kr, kc) not in _CORNERS and (kr, kc) not in _EDGES:
        return False
    if _is_king_caged(board, kr, kc, player):
        return False           # already caged, not merely at risk
    opp = BLACK if player == RED else RED
    opp_man  = opp
    opp_king = BLACK_KING if player == RED else RED_KING
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] not in (opp_man, opp_king):
                continue
            if abs(r - kr) <= 2 and abs(c - kc) <= 2:
                return True
    return False


def _square_class(r: int, c: int) -> str:
    if (r, c) in _CORNERS: return "corner"
    if (r, c) in _EDGES:   return "edge"
    return "center"


def _count_material(board, player):
    man  = RED if player == RED else BLACK
    king = RED_KING if player == RED else BLACK_KING
    men  = sum(1 for r in range(8) for c in range(8) if board[r][c] == man)
    kings= sum(1 for r in range(8) for c in range(8) if board[r][c] == king)
    return men, kings


def _dominant_terms(bd: dict, n: int = 3) -> list[str]:
    skip = {"total", "terminal"}
    items = [(k, v) for k, v in bd.items() if k not in skip and v != 0.0]
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    return [f"{k}={v:+.1f}" for k, v in items[:n]]


def _suspicious(bd: dict, board_after: list, player: int) -> list[str]:
    flags = []
    for kr, kc in _king_positions(board_after, player):
        sq = _square_class(kr, kc)
        exits = _count_safe_exits(board_after, kr, kc, player)
        if sq == "corner" and bd.get("caged_king", 0) == 0.0:
            flags.append(f"corner_king_at({kr},{kc})_caged_king=0")
        if sq in ("corner","edge") and exits <= 1 and bd.get("king_centralization",0) > -8:
            flags.append(f"low_exit_king({kr},{kc})_exits={exits}_centralization_not_penalised")
        if _delayed_cage_risk(board_after, kr, kc, player):
            flags.append(f"delayed_cage_risk_at({kr},{kc})")
    if bd.get("king_mobility", 0) == 0.0 and _king_positions(board_after, player):
        flags.append("king_mobility=0_but_kings_exist")
    return flags


def _likely_missing(suspicious: list[str], bd: dict) -> list[str]:
    missing = []
    if any("corner_king" in s and "caged_king=0" in s for s in suspicious):
        missing.append("static_delayed_cage_penalty(corner+attacker_nearby)")
    if any("low_exit_king" in s for s in suspicious):
        missing.append("king_exit_count_penalty(exits<=1)")
    if any("delayed_cage_risk" in s for s in suspicious):
        missing.append("opponent_proximity_to_corner_king_penalty")
    if bd.get("king_centralization", 0) > -10 and any("corner" in s for s in suspicious):
        missing.append("corner_penalty_stronger_than_centralization_distance")
    if not missing:
        missing.append("none_identified")
    return missing


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    data = json.loads(Path(IN_PATH).read_text())
    pos_by_turn = {p["turn"]: p for p in data["positions"]}

    header = (
        f"{'Turn':>4} | {'D6 best':>20} | {'Why evaluator likes it':>30} | "
        f"{'Suspicious':>35} | {'Likely missing':>40}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    audit_positions = []

    for turn in TARGET_TURNS:
        pos = pos_by_turn.get(turn)
        if pos is None:
            print(f"{turn:>4} | (not in benchmark)")
            continue

        board_before = pos["board"]
        d6_best_path = pos["d6_best_move"]
        tags         = pos.get("tags", [])

        # D6 scores indexed by path-key
        d6_scores = {
            tuple(tuple(sq) for sq in e["path"]): e["score"]
            for e in pos["d6_score_table"]
        }

        legal = get_all_legal_moves(board_before, RED)
        move_audits = []

        # Track the best-move breakdown for the summary line
        best_bd   = None
        best_susp = []
        best_miss = []

        for mv in legal:
            board_after = apply_move(board_before, mv)
            bd = evaluate_board_breakdown(board_after, RED, RED)

            path    = mv["path"]
            path_key= tuple(tuple(sq) for sq in path)
            d6_sc   = d6_scores.get(path_key)

            # Geometry
            dest = path[-1]
            promotes = (RED == RED and dest[0] == 0)
            dest_class = _square_class(*dest)

            men_after, kings_after = _count_material(board_after, RED)
            king_legal_moves = [
                m for m in get_all_legal_moves(board_after, RED)
                if board_after[m["path"][0][0]][m["path"][0][1]] == RED_KING
            ]

            king_details = []
            for kr, kc in _king_positions(board_after, RED):
                exits    = _count_safe_exits(board_after, kr, kc, RED)
                dcaged   = _delayed_cage_risk(board_after, kr, kc, RED)
                is_caged = _is_king_caged(board_after, kr, kc, RED)
                king_details.append({
                    "pos": [kr, kc],
                    "square_class": _square_class(kr, kc),
                    "safe_exits": exits,
                    "is_caged": is_caged,
                    "delayed_cage_risk": dcaged,
                })

            susp = _suspicious(bd, board_after, RED)
            miss = _likely_missing(susp, bd)
            dom  = _dominant_terms(bd, 4)

            is_best    = (path == d6_best_path or
                          path_key == tuple(tuple(sq) for sq in (d6_best_path or [])))
            is_chosen  = pos["old_chosen_move"]["path"] == path or \
                         tuple(tuple(sq) for sq in pos["old_chosen_move"]["path"]) == path_key

            entry = {
                "path": path,
                "is_d6_best": is_best,
                "is_old_chosen": is_chosen,
                "d6_score": d6_sc,
                "static_eval_total": round(bd.get("total", 0), 2),
                "promotes": promotes,
                "dest_square_class": dest_class,
                "red_kings_after": kings_after,
                "red_men_after":   men_after,
                "red_king_legal_moves_after": len(king_legal_moves),
                "king_details": king_details,
                "eval_breakdown": {k: round(v, 2) for k, v in bd.items()},
                "dominant_positive_terms": dom,
                "suspicious_features": susp,
                "likely_missing_signals": miss,
            }
            move_audits.append(entry)

            if is_best:
                best_bd   = bd
                best_susp = susp
                best_miss = miss
                best_dom  = dom

        audit_positions.append({
            "turn":        turn,
            "tags":        tags,
            "d6_best_path": d6_best_path,
            "old_chosen_path": pos["old_chosen_move"]["path"],
            "moves": move_audits,
        })

        # Compact summary
        d6_s    = "→".join(f"({r},{c})" for r,c in d6_best_path) if d6_best_path else "?"
        why     = ", ".join(best_dom[:2]) if best_bd else "?"
        susp_s  = "; ".join(best_susp[:2]) if best_susp else "-"
        miss_s  = "; ".join(best_miss[:2]) if best_miss else "-"
        print(f"{turn:>4} | {d6_s:>20} | {why:>30} | {susp_s:>35} | {miss_s:>40}")

    # ── Write output ──────────────────────────────────────────────────────────
    out = {
        "meta": {
            "input":  IN_PATH,
            "turns":  TARGET_TURNS,
            "description": (
                "Static evaluator breakdown audit for known-failure endgame positions. "
                "No search performed. evaluate_board_breakdown called on each child board. "
                "caged_king evaluator term (weight=75) is active."
            ),
        },
        "positions": audit_positions,
    }
    Path(OUT_PATH).write_text(json.dumps(out, indent=2))
    print(sep)
    print(f"\nWrote → {OUT_PATH}")


if __name__ == "__main__":
    main()
