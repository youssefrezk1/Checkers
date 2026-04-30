#!/usr/bin/env python3
"""
diag_pv.py — Diagnose D8 principal variation anomaly for two specific root moves.

Reconstructs the PV from the transposition table (no re-search needed) and
compares leaf evaluations, forced-capture presence, and TT influence.

Classification:
  NORMAL_DEPTH_PARITY        — typical odd/even oscillation, gap < 5 pt
  EVALUATOR_LEAF_WEAKNESS    — D8 leaf evaluation is misleading; D10 resolves it
  TACTICAL_EXTENSION_SUSPECT — D8 leaf still has forced captures (ext may be shallow)
  TT_OR_SEARCH_BUG_SUSPECT   — TT disabled changes A/B ranking at D8
  UNCLEAR                    — none of the above conclusive

Usage:
    python3 diag_pv.py \\
        --trace logs/game_20260427_091537_540232.jsonl \\
        --turn 35 \\
        --move-a "2,5,1,6" \\
        --move-b "6,5,5,4"

    # Extra depths or skip TT comparison:
    python3 diag_pv.py --trace ... --turn 35 --move-a "2,5,1,6" --move-b "6,5,5,4" \\
        --depths 8,10 --no-tt-compare
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import checkers.search.minimax_core as _mc

from checkers.engine.board import (
    BLACK, BLACK_KING, BOARD_SIZE, EMPTY, RED, RED_KING,
    create_initial_board,
)
from checkers.engine.evaluation import evaluate_board_breakdown
from checkers.engine.rules import apply_move, get_all_legal_moves
from audit_strength import (
    load_trace, reconstruct, fmt_move, normalize_path,
    _counts, _player_label,
)

_OPP = {RED: BLACK, BLACK: RED}


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_path(s: str) -> list[list[int]]:
    """'2,5,1,6' → [[2,5],[1,6]]"""
    nums = [int(x.strip()) for x in s.split(",")]
    assert len(nums) % 2 == 0, f"Odd number of coords in '{s}'"
    return [[nums[i], nums[i + 1]] for i in range(0, len(nums), 2)]


def _find_legal(path: list[list[int]], legal: list[dict]) -> dict | None:
    needle = normalize_path(path)
    for mv in legal:
        if normalize_path(mv["path"]) == needle:
            return mv
    return None


def _score_of(scored: list[tuple], path) -> float | None:
    needle = normalize_path(path)
    for mv, sc in scored:
        if normalize_path(mv["path"]) == needle:
            return sc
    return None


def _fmt(v) -> str:
    return f"{v:.2f}" if v is not None else "—"


def _all_captures(board: list[list[int]], player: int) -> bool:
    legal = get_all_legal_moves(board, player)
    return bool(legal) and all(m["type"] == "jump" for m in legal)


def _board_line(board: list[list[int]]) -> str:
    c = _counts(board)
    return (f"RED {c['red_men']}m+{c['red_kings']}K  "
            f"BLACK {c['black_men']}m+{c['black_kings']}K  total={c['total']}")


def _eval_top(board: list[list[int]], root_player: int, n: int = 10) -> str:
    bd = evaluate_board_breakdown(board, root_player, root_player)
    items = sorted([(k, v) for k, v in bd.items() if v != 0.0],
                   key=lambda x: -abs(x[1]))
    lines = [f"    {'total':<35} {sum(bd.values()):>10.2f}"]
    for k, v in items[:n]:
        lines.append(f"    {k:<35} {v:>10.2f}")
    if len(items) > n:
        lines.append(f"    … ({len(items)-n} more non-zero)")
    return "\n".join(lines)


# ── PV reconstruction from TT ────────────────────────────────────────────────

def _pv_from_tt(
    board_after_root: list[list[int]],
    first_player: int,
    root_player: int,
    depth: int,           # full search depth (e.g. 8)
    max_plies: int = 12,
) -> tuple[list[dict], list[list[int]]]:
    """
    Follow best_move pointers in _TT starting from the position after the root
    move was applied.  Does NOT re-search — reads TT entries written during the
    prior search_root_all_scores call.

    Returns (move_sequence, leaf_board).
    move_sequence: list of dicts with keys move, player, bound, value, ply_from_root.
    """
    pv: list[dict] = []
    board = [row[:] for row in board_after_root]
    player = first_player
    ply = 1                        # root consumed ply 0
    remaining = depth - 1          # plies left for opponent and us

    for _ in range(min(remaining, max_plies)):
        found = None
        # Try extension_depth 0, 1, 2 in case the PV goes through an extension ply
        for ext in range(_mc.MAX_TACTICAL_EXTENSION_PLIES + 1):
            key = _mc._tt_key(board, player, root_player, ext, True, ply)
            entry = _mc._TT.get(key)
            if entry is not None and entry.best_move is not None:
                # Prefer EXACT, then LOWER (has a valid best move), skip UPPER
                if entry.bound_type in (_mc.TTBoundType.EXACT, _mc.TTBoundType.LOWER):
                    found = (entry, ext)
                    break
        if found is None:
            break

        entry, ext_used = found
        mv = entry.best_move
        pv.append({
            "move": mv,
            "player": player,
            "bound": entry.bound_type.value,
            "value": entry.value,
            "ply_from_root": ply,
            "ext": ext_used,
        })
        board = apply_move(board, mv)
        player = _OPP[player]
        ply += 1

    return pv, board


# ── per-depth analysis ───────────────────────────────────────────────────────

def _analyze_depth(
    board: list[list[int]],
    player: int,
    move_a: dict,
    move_b: dict,
    depth: int,
    use_tt: bool,
) -> dict[str, Any]:
    """
    Run a fresh search at `depth`, extract A/B scores, reconstruct PV from TT
    (only when use_tt=True since TT is not populated otherwise).
    Returns a result dict.
    """
    _mc.clear_transposition_table()
    _, _, scored, stats = _mc.search_root_all_scores(
        board=board,
        current_player=player,
        depth=depth,
        use_tt=use_tt,
        use_tactical_extension=True,
        use_phase7a=True,
    )

    sc_a = _score_of(scored, move_a["path"])
    sc_b = _score_of(scored, move_b["path"])

    pv_a = pv_b = []
    leaf_a = leaf_b = board
    if use_tt:
        board_a = apply_move(board, move_a)
        board_b = apply_move(board, move_b)
        pv_a, leaf_a = _pv_from_tt(board_a, _OPP[player], player, depth)
        pv_b, leaf_b = _pv_from_tt(board_b, _OPP[player], player, depth)

    return {
        "depth": depth,
        "use_tt": use_tt,
        "scored": scored,
        "sc_a": sc_a,
        "sc_b": sc_b,
        "b_wins": (sc_b is not None and sc_a is not None and sc_b > sc_a),
        "gap_ab": (sc_b - sc_a if sc_a is not None and sc_b is not None else None),
        "pv_a": pv_a,
        "pv_b": pv_b,
        "leaf_a": leaf_a,
        "leaf_b": leaf_b,
        "leaf_a_captures": _all_captures(leaf_a, player if len(pv_a) % 2 == 0 else _OPP[player]),
        "leaf_b_captures": _all_captures(leaf_b, player if len(pv_b) % 2 == 0 else _OPP[player]),
        "nodes": stats.nodes,
        "tt_hits": stats.tt_hits,
    }


# ── classification ────────────────────────────────────────────────────────────

def classify(results: dict[str, dict]) -> tuple[str, str]:
    d_list = sorted(results.keys())
    if len(d_list) < 2:
        return ("UNCLEAR", "Need at least two depths to classify")

    # Identify the "anomaly depth" (D8) and "consensus depth" (D10)
    # A = preferred by consensus; B = preferred only at anomaly depth
    r_anomaly = results[d_list[0]]    # e.g. D8
    r_consensus = results[d_list[1]]  # e.g. D10

    sc_a8  = r_anomaly["sc_a"]
    sc_b8  = r_anomaly["sc_b"]
    sc_a10 = r_consensus["sc_a"]
    sc_b10 = r_consensus["sc_b"]

    if any(v is None for v in [sc_a8, sc_b8, sc_a10, sc_b10]):
        return ("UNCLEAR", "Could not extract scores for A or B from search output")

    gap_d8  = sc_b8 - sc_a8    # positive if B wins at D8
    gap_d10 = sc_a10 - sc_b10  # positive if A wins at D10

    # TT effect: compare D8-tt vs D8-nott if available
    r_nott = results.get(f"{d_list[0]}_nott")
    tt_effect_a = tt_effect_b = None
    if r_nott:
        if r_nott["sc_a"] is not None and sc_a8 is not None:
            tt_effect_a = abs(sc_a8 - r_nott["sc_a"])
        if r_nott["sc_b"] is not None and sc_b8 is not None:
            tt_effect_b = abs(sc_b8 - r_nott["sc_b"])
        # TT changes ranking at anomaly depth?
        nott_b_wins = r_nott["b_wins"]
        if nott_b_wins != r_anomaly["b_wins"]:
            return (
                "TT_OR_SEARCH_BUG_SUSPECT",
                f"Ranking of A vs B REVERSES when TT is disabled at D{r_anomaly['depth']}. "
                f"TT-on: B wins by {gap_d8:.1f}pt; TT-off: "
                f"{'B wins by' if nott_b_wins else 'A wins by'} "
                f"{abs(r_nott['gap_ab']):.1f}pt. "
                "TT entries from sibling subtree searches are contaminating the score.",
            )
        if (tt_effect_a is not None and tt_effect_a > 10.0):
            return (
                "TT_OR_SEARCH_BUG_SUSPECT",
                f"Move A score changes by {tt_effect_a:.1f}pt when TT disabled at "
                f"D{r_anomaly['depth']} — substantial TT influence on A's evaluation.",
            )

    # Check if D8 anomaly exists at all
    if gap_d8 <= 0:
        return (
            "NORMAL_DEPTH_PARITY",
            f"A already beats B at D{r_anomaly['depth']} (A={sc_a8:.1f}, B={sc_b8:.1f}). "
            "No anomaly to diagnose.",
        )

    # Tactical extension check: D8 leaf for A still has forced captures?
    if r_anomaly["leaf_a_captures"]:
        return (
            "TACTICAL_EXTENSION_SUSPECT",
            f"Move A's D{r_anomaly['depth']} leaf still has forced captures for the "
            "side to move. MAX_TACTICAL_EXTENSION_PLIES=2 was reached before the "
            "capture sequence resolved. D10 sees two extra plies that resolve it, "
            "flipping the evaluation. Consider raising MAX_TACTICAL_EXTENSION_PLIES.",
        )

    # Pure horizon effect: large score gap between D8 and D10 for A
    if sc_a10 is not None and abs(sc_a10 - sc_a8) > 10.0:
        return (
            "EVALUATOR_LEAF_WEAKNESS",
            f"Move A scores {sc_a8:.1f} at D{r_anomaly['depth']} vs {sc_a10:.1f} "
            f"at D{r_consensus['depth']} — a {abs(sc_a10-sc_a8):.1f}pt swing. "
            "The static evaluator at the D8 leaf misvalues the position; "
            "extra plies resolve it. Classic horizon effect, not a search bug.",
        )

    # Small gap: normal odd/even parity
    if abs(gap_d8) < 5.0:
        return (
            "NORMAL_DEPTH_PARITY",
            f"D{r_anomaly['depth']} gap A-B = {gap_d8:.1f}pt (<5 pt). "
            "Typical odd/even depth parity oscillation. "
            "D10 reverses it by {gap_d10:.1f}pt — within expected noise.",
        )

    return (
        "UNCLEAR",
        f"D{r_anomaly['depth']}: B beats A by {gap_d8:.1f}pt; "
        f"D{r_consensus['depth']}: A beats B by {gap_d10:.1f}pt. "
        "No tactical-extension leaf captures, no TT flip, no large score swing. "
        "Likely positional: the extra 2 plies resolve a mid-depth tactical sequence "
        "that the static evaluator cannot see at D8.",
    )


# ── report ────────────────────────────────────────────────────────────────────

def _print_pv(pv: list[dict], root_player: int, label: str) -> None:
    if not pv:
        print(f"    (PV empty — position not in TT or no best_move recorded)")
        return
    for step in pv:
        mv   = step["move"]
        p    = step["player"]
        side = "our " if p == root_player else "opp "
        tag  = f"[{_player_label(p)}/{side}ply={step['ply_from_root']} "
        tag += f"ext={step['ext']} bound={step['bound']}]"
        print(f"    {tag}  {fmt_move(mv)}  val={step['value']:.2f}")


def _print_depth_block(r: dict, move_a: dict, move_b: dict, root_player: int) -> None:
    depth = r["depth"]
    tt_s  = "TT=on" if r["use_tt"] else "TT=off"
    print(f"\n{'─'*68}")
    print(f"  D{depth} {tt_s}   nodes={r['nodes']:,}  tt_hits={r['tt_hits']:,}")
    print(f"  {'Move':<6}  {'Score':>9}  {'Δ(B-A)':>9}  {'Best?':>6}")
    gap = r["gap_ab"]
    gap_s = f"{gap:+.2f}" if gap is not None else "—"
    print(f"  {'A':<6}  {_fmt(r['sc_a']):>9}  {gap_s:>9}  {'NO' if r['b_wins'] else 'YES':>6}")
    print(f"  {'B':<6}  {_fmt(r['sc_b']):>9}  {'':>9}  {'YES' if r['b_wins'] else 'NO':>6}")

    # Top-5 context
    print(f"\n  Top-5 moves at D{depth} {tt_s}:")
    for rank, (mv, sc) in enumerate(r["scored"][:5], 1):
        p = normalize_path(mv["path"])
        tag = ""
        if p == normalize_path(move_a["path"]):
            tag = "  [A]"
        elif p == normalize_path(move_b["path"]):
            tag = "  [B]"
        print(f"    {rank}. {fmt_move(mv):<28}  {sc:.2f}{tag}")

    if not r["use_tt"]:
        return   # no PV or leaf analysis without TT

    # PV for A
    print(f"\n  PV for A at D{depth} (from TT, {len(r['pv_a'])} plies recovered):")
    _print_pv(r["pv_a"], root_player, "A")
    leaf_a = r["leaf_a"]
    print(f"  Leaf A: {_board_line(leaf_a)}")
    print(f"  Leaf A has forced captures: {r['leaf_a_captures']}")
    print(f"  Leaf A evaluator (root={_player_label(root_player)}):")
    print(_eval_top(leaf_a, root_player))

    # PV for B
    print(f"\n  PV for B at D{depth} (from TT, {len(r['pv_b'])} plies recovered):")
    _print_pv(r["pv_b"], root_player, "B")
    leaf_b = r["leaf_b"]
    print(f"  Leaf B: {_board_line(leaf_b)}")
    print(f"  Leaf B has forced captures: {r['leaf_b_captures']}")
    print(f"  Leaf B evaluator (root={_player_label(root_player)}):")
    print(_eval_top(leaf_b, root_player))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="PV diagnostic: compare two root moves across depths.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--trace", required=True)
    ap.add_argument("--turn", type=int, required=True)
    ap.add_argument("--move-a", required=True,
                    help="Move A path as 'r1,c1,r2,c2[,r3,c3...]'")
    ap.add_argument("--move-b", required=True,
                    help="Move B path as 'r1,c1,r2,c2[,r3,c3...]'")
    ap.add_argument("--depths", default="8,10",
                    help="Comma-separated depths to compare (default: 8,10)")
    ap.add_argument("--no-tt-compare", action="store_true",
                    help="Skip the TT-disabled re-run at the first depth")
    args = ap.parse_args()

    depths = [int(d.strip()) for d in args.depths.split(",")]
    path_a = _parse_path(args.move_a)
    path_b = _parse_path(args.move_b)

    records = load_trace(args.trace)
    board, player = reconstruct(records, args.turn)
    legal = get_all_legal_moves(board, player)

    move_a = _find_legal(path_a, legal)
    move_b = _find_legal(path_b, legal)
    if not move_a:
        sys.exit(f"[ERROR] Move A {path_a} not found in legal moves for Turn {args.turn}")
    if not move_b:
        sys.exit(f"[ERROR] Move B {path_b} not found in legal moves for Turn {args.turn}")

    print(f"{'='*68}")
    print(f"PV DIAGNOSTIC  Turn {args.turn}  {_player_label(player)} to move")
    print(f"  Board: {_board_line(board)}")
    print(f"  Legal: {len(legal)} moves")
    print(f"  Move A: {fmt_move(move_a)}")
    print(f"  Move B: {fmt_move(move_b)}")
    print(f"  Depths: {depths}  TT-compare: {not args.no_tt_compare}")

    all_results: dict[str, dict] = {}

    for i, depth in enumerate(depths):
        key = str(depth)
        print(f"\n  Scoring D{depth} (TT on, fresh) …", flush=True)
        r = _analyze_depth(board, player, move_a, move_b, depth, use_tt=True)
        all_results[key] = r
        _print_depth_block(r, move_a, move_b, player)

        # TT-disabled run only for the FIRST (anomaly) depth
        if i == 0 and not args.no_tt_compare:
            print(f"\n  Scoring D{depth} (TT off) …", flush=True)
            r_nott = _analyze_depth(board, player, move_a, move_b, depth, use_tt=False)
            all_results[f"{key}_nott"] = r_nott
            _print_depth_block(r_nott, move_a, move_b, player)

    # Leaf eval comparison: A at D_anomaly vs D_consensus
    if len(depths) >= 2:
        k0, k1 = str(depths[0]), str(depths[1])
        r0, r1 = all_results.get(k0), all_results.get(k1)
        if r0 and r1:
            print(f"\n{'─'*68}")
            print("  LEAF EVAL COMPARISON FOR A")
            print(f"  {'Term':<35}  {'D'+str(depths[0]):>10}  {'D'+str(depths[1]):>10}  {'Δ':>8}")
            print("  " + "-" * 66)
            bd0 = evaluate_board_breakdown(r0["leaf_a"], player, player)
            bd1 = evaluate_board_breakdown(r1["leaf_a"], player, player)
            all_keys = sorted(set(bd0) | set(bd1))
            for k in all_keys:
                v0, v1 = bd0.get(k, 0.0), bd1.get(k, 0.0)
                if v0 == 0.0 and v1 == 0.0:
                    continue
                delta = v1 - v0
                print(f"  {k:<35}  {v0:>10.2f}  {v1:>10.2f}  {delta:>+8.2f}")
            t0 = sum(bd0.values())
            t1 = sum(bd1.values())
            print(f"  {'TOTAL':<35}  {t0:>10.2f}  {t1:>10.2f}  {t1-t0:>+8.2f}")

    # Classification
    print(f"\n{'='*68}")
    print("CLASSIFICATION")
    label, reason = classify(all_results)
    print(f"  {label}")
    print(f"  {reason}")
    print()


if __name__ == "__main__":
    main()
