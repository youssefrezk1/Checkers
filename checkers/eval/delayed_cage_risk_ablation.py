"""
delayed_cage_risk_ablation.py

Read-only ablation for a proposed delayed_cage_risk evaluator term.

Simulation method:
  For each legal move at each target turn, the D6 score stored in the benchmark
  JSON is adjusted post-hoc:
    adjusted_score = d6_score - penalty * delayed_cage_risk_count(child_board, player)

  delayed_cage_risk_count fires when ALL of:
    1. total_pieces(child_board) <= 14
    2. player has a king at a corner or edge square on the child board
    3. that king is NOT currently caged (_is_king_caged == False)
    4. that king has <= 1 safe legal exit
    5. at least one single opponent move would arm the cage
       (i.e., after that opponent move, _is_king_caged becomes True)
    6. caged_king does NOT already fire for that king (guaranteed by condition 3)

Input:   logs/known_failure_positions_20260425_144451.json
Output:  logs/delayed_cage_risk_ablation_20260425_144451.json

No search performed. Runtime < 5 s.
Does NOT modify any engine, evaluator, ranker, minimax, rules, or proposal code.
"""

from __future__ import annotations

import json
from pathlib import Path

from checkers.engine.board import RED, BLACK, RED_KING, BLACK_KING, BOARD_SIZE
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.engine.evaluation import _king_positions, _is_king_caged

IN_PATH  = "logs/known_failure_positions_20260425_144451.json"
OUT_PATH = "logs/delayed_cage_risk_ablation_20260425_144451.json"

TARGET_TURNS   = [41, 43, 45, 49, 51, 55, 59, 61, 63]
PENALTY_VALUES = [15, 25, 35, 50, 75]

_CORNERS = frozenset([(0,0),(0,7),(7,0),(7,7)])
_EDGES   = frozenset((r,c) for r in range(8) for c in range(8)
                     if (r in (0,7) or c in (0,7)) and (r,c) not in _CORNERS)


# ── helpers ───────────────────────────────────────────────────────────────────

def _total_pieces(board) -> int:
    return sum(1 for r in range(8) for c in range(8) if board[r][c] != 0)


def _count_safe_exits(board, kr, kc, player) -> int:
    opp      = BLACK if player == RED else RED
    opp_man  = opp
    opp_king = BLACK_KING if player == RED else RED_KING
    safe = 0
    for dr, dc in ((-1,-1),(-1,1),(1,-1),(1,1)):
        r2, c2 = kr+dr, kc+dc
        if not (0 <= r2 < BOARD_SIZE and 0 <= c2 < BOARD_SIZE): continue
        if board[r2][c2] != 0: continue
        dest_safe = True
        for adr, adc in ((-1,-1),(-1,1),(1,-1),(1,1)):
            ar, ac = r2+adr, c2+adc
            lr, lc = r2-adr, c2-adc
            if not (0 <= ar < BOARD_SIZE and 0 <= ac < BOARD_SIZE): continue
            if not (0 <= lr < BOARD_SIZE and 0 <= lc < BOARD_SIZE): continue
            opp_piece = board[ar][ac]
            if opp_piece not in (opp_man, opp_king): continue
            if lr == kr and lc == kc:  land_empty = True
            elif lr == r2 and lc == c2: land_empty = False
            else:                       land_empty = board[lr][lc] == 0
            if not land_empty: continue
            if opp_piece == opp_man:
                jdir = lr - ar
                if opp == RED  and jdir >= 0: continue
                if opp == BLACK and jdir <= 0: continue
            dest_safe = False
            break
        if dest_safe:
            safe += 1
    return safe


def _arming_move_exists(board, kr, kc, player) -> bool:
    """True if any single opponent move results in _is_king_caged(board', kr, kc, player)."""
    opp = BLACK if player == RED else RED
    for opp_mv in get_all_legal_moves(board, opp):
        after = apply_move(board, opp_mv)
        if _is_king_caged(after, kr, kc, player):
            return True
    return False


def _delayed_cage_risk_count(board, player) -> int:
    """
    Count kings of *player* that satisfy all 5 delayed-cage-risk conditions.
    Returns 0 when caged_king already applies (condition 3 excludes already-caged kings).
    """
    if _total_pieces(board) > 14:
        return 0
    count = 0
    for kr, kc in _king_positions(board, player):
        # condition 2: corner or edge
        if (kr, kc) not in _CORNERS and (kr, kc) not in _EDGES:
            continue
        # condition 3: not already caged
        if _is_king_caged(board, kr, kc, player):
            continue
        # condition 4: <= 1 safe exit
        if _count_safe_exits(board, kr, kc, player) > 1:
            continue
        # condition 5: arming move exists
        if _arming_move_exists(board, kr, kc, player):
            count += 1
    return count


def _path_key(path) -> tuple:
    return tuple(tuple(sq) for sq in path)


def _fmt(path) -> str:
    if not path: return "?"
    return "→".join(f"({r},{c})" for r,c in path)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    data = json.loads(Path(IN_PATH).read_text())
    pos_by_turn = {p["turn"]: p for p in data["positions"]}

    header = (
        f"{'Turn':>4} | {'Penalty':>7} | {'Old best':>18} | "
        f"{'Adj best':>18} | {'Changed?':>8} | Reason"
    )
    sep = "-" * 90
    print(header)
    print(sep)

    all_results = []

    for turn in TARGET_TURNS:
        pos = pos_by_turn.get(turn)
        if pos is None:
            continue

        board_before  = pos["board"]
        old_chosen_p  = pos["old_chosen_move"]["path"]
        d6_best_orig  = pos["d6_best_move"]
        tags          = pos.get("tags", [])

        # Map path_key → D6 score from benchmark
        d6_by_pk = {_path_key(e["path"]): e["score"] for e in pos["d6_score_table"]}

        legal = get_all_legal_moves(board_before, RED)

        # Pre-compute child-board delayed_cage_risk counts (penalty-independent)
        child_dcr: dict[tuple, int] = {}
        child_caged: dict[tuple, int] = {}  # for double-pen check
        for mv in legal:
            pk = _path_key(mv["path"])
            child = apply_move(board_before, mv)
            child_dcr[pk]   = _delayed_cage_risk_count(child, RED)
            # count already-caged kings (for double-pen sanity check)
            child_caged[pk] = sum(
                1 for kr, kc in _king_positions(child, RED)
                if _is_king_caged(child, kr, kc, RED)
            )

        turn_results = []

        for penalty in PENALTY_VALUES:
            # Adjusted scores
            adj: dict[tuple, float] = {}
            for mv in legal:
                pk   = _path_key(mv["path"])
                base = d6_by_pk.get(pk, float("-inf"))
                adj[pk] = base - penalty * child_dcr[pk]

            best_pk  = max(adj, key=lambda pk: adj[pk])
            best_adj = adj[best_pk]
            best_path = next(mv["path"] for mv in legal if _path_key(mv["path"]) == best_pk)

            orig_pk   = _path_key(d6_best_orig) if d6_best_orig else None
            changed   = best_pk != orig_pk

            old_chosen_pk  = _path_key(old_chosen_p)
            old_chosen_adj = adj.get(old_chosen_pk)

            # Double-pen check: any child with BOTH dcr>0 and caged>0?
            double_pen_risk = any(
                child_dcr[pk] > 0 and child_caged[pk] > 0 for pk in adj
            )

            reason_parts = []
            if changed:
                orig_adj = adj.get(orig_pk, float("-inf"))
                reason_parts.append(
                    f"gap_to_old_best={best_adj - orig_adj:+.1f}"
                )
            if double_pen_risk:
                reason_parts.append("DOUBLE_PEN_RISK")
            if not reason_parts:
                reason_parts.append("no_change")

            turn_results.append({
                "penalty":           penalty,
                "orig_d6_best_path": d6_best_orig,
                "adj_best_path":     best_path,
                "adj_best_score":    round(best_adj, 2),
                "changed":           changed,
                "double_pen_risk":   double_pen_risk,
                "reason":            "; ".join(reason_parts),
                "score_table": sorted(
                    [
                        {
                            "path":        mv["path"],
                            "d6_score":    d6_by_pk.get(_path_key(mv["path"])),
                            "dcr_count":   child_dcr[_path_key(mv["path"])],
                            "caged_count": child_caged[_path_key(mv["path"])],
                            "adj_score":   round(adj[_path_key(mv["path"])], 2),
                            "is_adj_best": _path_key(mv["path"]) == best_pk,
                            "is_old_chosen": _path_key(mv["path"]) == _path_key(old_chosen_p),
                        }
                        for mv in legal
                    ],
                    key=lambda x: x["adj_score"], reverse=True,
                ),
            })

            # Print first and last penalty per turn only (compact)
            if penalty in (PENALTY_VALUES[0], PENALTY_VALUES[-1]) or changed:
                chg_s = "YES ←" if changed else "no"
                orig_s = _fmt(d6_best_orig)
                adj_s  = _fmt(best_path)
                rsn    = "; ".join(reason_parts)
                print(f"{turn:>4} | {penalty:>7} | {orig_s:>18} | {adj_s:>18} | {chg_s:>8} | {rsn}")

        all_results.append({
            "turn":          turn,
            "tags":          tags,
            "old_chosen":    old_chosen_p,
            "d6_best_orig":  d6_best_orig,
            "penalties":     turn_results,
        })

    # ── Write output ──────────────────────────────────────────────────────────
    out = {
        "meta": {
            "input":          IN_PATH,
            "penalty_values": PENALTY_VALUES,
            "target_turns":   TARGET_TURNS,
            "description": (
                "Post-hoc ablation of delayed_cage_risk penalty. "
                "Adjusted score = D6_score - penalty * delayed_cage_risk_count(child_board). "
                "delayed_cage_risk_count fires only when king is corner/edge, NOT already caged, "
                "<=1 safe exit, and an arming opponent move exists. "
                "No search, no engine modification."
            ),
        },
        "positions": all_results,
    }
    Path(OUT_PATH).write_text(json.dumps(out, indent=2))
    print(sep)
    print(f"\nWrote → {OUT_PATH}")

    # ── Final answer section ──────────────────────────────────────────────────
    print()
    print("═" * 70)
    print("FINAL ANSWERS")
    print("═" * 70)

    # Q1: smallest penalty that changes T41
    pos41 = next((r for r in all_results if r["turn"] == 41), None)
    if pos41:
        for pr in pos41["penalties"]:
            if pr["changed"]:
                print(f"1. Smallest penalty changing T41: {pr['penalty']} pt")
                print(f"   Old best: {_fmt(pr['orig_d6_best_path'])}  →  New best: {_fmt(pr['adj_best_path'])}")
                break
        else:
            print("1. No tested penalty changes T41 away from promotion.")

    # Q2: T43 sanity
    pos43 = next((r for r in all_results if r["turn"] == 43), None)
    if pos43:
        changes43 = [(pr["penalty"], pr["adj_best_path"]) for pr in pos43["penalties"] if pr["changed"]]
        if changes43:
            print(f"2. T43 best changes at penalties: {changes43}")
        else:
            print("2. T43 best move unchanged across all penalties (sane).")

    # Q3: double-penalization at T45+
    double_pen_turns = [
        r["turn"] for r in all_results
        if r["turn"] >= 45 and any(pr["double_pen_risk"] for pr in r["penalties"])
    ]
    if double_pen_turns:
        print(f"3. Double-penalization risk at turns: {double_pen_turns}")
    else:
        print("3. No double-penalization at T45+ (caged_king and delayed_cage_risk are mutually exclusive).")

    # Q4: T61/T63 harm
    harm_turns = []
    for t in [61, 63]:
        pr_t = next((r for r in all_results if r["turn"] == t), None)
        if pr_t and any(pr["changed"] for pr in pr_t["penalties"]):
            harm_turns.append(t)
    if harm_turns:
        print(f"4. T61/T63 best move changes at some penalty: {harm_turns} — check score_table.")
    else:
        print("4. T61/T63 best move unchanged across all penalties (no regression).")

    # Q5-Q7: summary
    print()
    print("5. Recommended predicate:")
    print("   king at corner/edge, NOT currently caged, <=1 safe exit,")
    print("   AND any single opponent move would arm caged_king predicate.")
    print("   Phase gate: total_pieces <= 14.")
    print()
    print("6. Recommended penalty: see smallest-changing value above (or 25 if T41 unchanged).")
    print()

    # Check if any penalty changes T41
    t41_min = None
    if pos41:
        for pr in pos41["penalties"]:
            if pr["changed"]:
                t41_min = pr["penalty"]
                break
    if t41_min is not None:
        safe = t41_min <= 50 and not double_pen_turns
        print(f"7. Implementation {'SAFE' if safe else 'NEEDS REVIEW'}: "
              f"T41 changes at pen={t41_min}, no double-pen={not bool(double_pen_turns)}.")
    else:
        print("7. T41 does not change even at pen=75. Root-level post-hoc simulation "
              "under-estimates the in-search effect; may still be worth implementing "
              "inside the evaluator for full propagation through the search tree.")


if __name__ == "__main__":
    main()
