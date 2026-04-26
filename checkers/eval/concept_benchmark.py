"""
checkers/eval/concept_benchmark.py

Paper-based concept benchmark: 15 positions covering tactical shots,
corner blocks, king endgame conversion, and forcing exchanges.

Source: concept_benchmark_plan.md (from checkers paper concepts)

Rules:
- No engine modifications.
- search_root_all_scores only (no score_move_with_minimax).
- D6 run on every position.
- D8 run automatically when D6 top-gap <= 30 OR D6 fails a hard assertion.
- Output: logs/concept_benchmark_results.json + compact terminal table.

Pass criteria by category:
- TACTICAL (P01–P05, P15): expected multi-jump/promotion path must be D6 rank-1.
- CAGE/PROMO (P06–P08):    report rank of expected move; note if non-rank-1.
- KING_ENDGAME (P09–P12):  D6 best score >= WIN_THRESHOLD (9000).
- STRATEGY (P13–P14):      informational — report ranks and mobility restriction.
  P13: engine should avoid a move that reduces to 1K vs 1K (equal).

Usage:
    venv/bin/python3 checkers/eval/concept_benchmark.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from checkers.engine.board import RED, BLACK, RED_KING, BLACK_KING
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.search.minimax_core import (
    search_root_all_scores,
    clear_transposition_table,
    get_d6_top_gap,
)

OUT_PATH = Path("logs/concept_benchmark_results.json")
D6_DEPTH = 6
D8_DEPTH = 8
WIN_THRESHOLD = 9_000.0
D8_GAP_TRIGGER = 30.0   # run D8 if D6 top-gap <= this

# ── board helpers ─────────────────────────────────────────────────────────────

_SYM = {"r": RED, "b": BLACK, "R": RED_KING, "B": BLACK_KING}


def _mk(pieces: list[tuple]) -> list[list[int]]:
    board = [[0] * 8 for _ in range(8)]
    for sym, r, c in pieces:
        board[r][c] = _SYM[sym]
    return board


def _pk(path) -> tuple:
    return tuple(tuple(sq) for sq in path)


def _path_str(path) -> str:
    if path is None:
        return "—"
    return "→".join(f"({r},{c})" for r, c in path)


def _count_pieces(board) -> tuple[int, int]:
    red_n = sum(1 for r in range(8) for c in range(8) if board[r][c] in (RED, RED_KING))
    blk_n = sum(1 for r in range(8) for c in range(8) if board[r][c] in (BLACK, BLACK_KING))
    return red_n, blk_n


# ── search wrapper ─────────────────────────────────────────────────────────────

def _search(board, player, depth) -> tuple[list, list, float, int]:
    """Run search_root_all_scores; return (best_path, scored_sorted, elapsed_s, nodes)."""
    legal = get_all_legal_moves(board, player)
    clear_transposition_table()
    t0 = time.perf_counter()
    best_mv, best_sc, scored, stats = search_root_all_scores(
        board=board, current_player=player, depth=depth,
        legal_moves=legal, use_tt=True, use_tactical_extension=True, use_phase7a=True,
    )
    elapsed = round(time.perf_counter() - t0, 2)
    scored.sort(key=lambda x: -x[1])
    nodes = getattr(stats, "nodes", None)
    best_path = best_mv["path"] if best_mv else None
    return best_path, scored, elapsed, nodes


def _rank_of(scored, target_path) -> int | None:
    """Return 1-indexed rank of target_path in scored list, or None."""
    tpk = _pk(target_path)
    for i, (mv, _) in enumerate(scored):
        if _pk(mv["path"]) == tpk:
            return i + 1
    return None


def _mobility_after(board, player, move) -> int:
    """Count legal moves for player after applying move."""
    b2 = apply_move(board, move)
    return len(get_all_legal_moves(b2, BLACK if player == RED else RED))


# ── position registry ──────────────────────────────────────────────────────────

def _define_positions() -> list[dict]:
    """
    Return all 15 benchmark positions as structured dicts.

    Fields:
        id, concept, category, pieces, player,
        expected_path (optional), expected_note,
        pass_rule ('rank1'|'win'|'info'|'rank_of'),
        win_threshold (float, for 'win' rule),
    """
    return [

        # ── Priority 1: Tactical shots ────────────────────────────────────────

        dict(
            id="P01",
            concept="Two-for-One Shot — double-jump wins over single jump",
            category="TACTICAL",
            pieces=[("r",5,0),("r",3,4),("b",4,1),("b",4,3),("b",2,3)],
            player=RED,
            expected_path=[[5,0],[3,2],[1,4]],
            expected_note="double jump (5,0)→(3,2)→(1,4) must be rank-1",
            pass_rule="rank1",
        ),

        dict(
            id="P02",
            concept="Three-for-One Shot — multi-jump chain preferred over single jump",
            category="TACTICAL",
            pieces=[("r",5,2),("r",5,6),
                    ("b",4,1),("b",4,3),("b",4,5),("b",2,3),("b",2,5)],
            player=RED,
            expected_path=None,
            expected_note="any double-jump (len(captured)>=2) must be rank-1; single jump (3,0) is inferior",
            pass_rule="multi_jump_rank1",
        ),

        dict(
            id="P03",
            concept="King Forced Capture Loop — king sweeps all 4 BLACK men",
            category="TACTICAL",
            pieces=[("R",5,4),("b",4,3),("b",4,5),("b",2,3),("b",2,5)],
            player=RED,
            expected_path=None,
            expected_note="only legal moves are 4-capture loops; both are equivalent; no selection failure possible",
            pass_rule="info",
        ),

        dict(
            id="P04",
            concept="Promotion via Forced Jump",
            category="TACTICAL",
            pieces=[("r",2,3),("r",4,1),("b",3,2),("b",1,4)],
            player=RED,
            expected_path=[[2,3],[0,5]],
            expected_note="(2,3)→(0,5): only legal jump, promotes RED to king",
            pass_rule="rank1",
        ),

        dict(
            id="P05",
            concept="Two-for-One via Compulsory Recapture (confirmed D6 pass)",
            category="TACTICAL",
            pieces=[("r",5,0),("r",3,4),("b",4,1),("b",4,3),("b",2,3)],
            player=RED,
            expected_path=[[5,0],[3,2],[1,4]],
            expected_note="identical to P01 — second confirmation",
            pass_rule="rank1",
        ),

        # ── Priority 2: Corner block positions ───────────────────────────────

        dict(
            id="P06",
            concept="Corner Cage — RED closes BLACK king into (0,7) corner",
            category="CAGE",
            pieces=[("B",0,7),("r",3,6),("r",5,4)],
            player=RED,
            expected_path=None,
            expected_note="moves ending near (0,7) must score WIN=10000; "
                          "moves into center score ~-97",
            pass_rule="win",
            win_threshold=WIN_THRESHOLD,
        ),

        dict(
            id="P07",
            concept="Corner Trap Terminal — RED wins immediately at (7,0)",
            category="CAGE",
            pieces=[("R",4,1),("r",6,1),("B",7,0)],
            player=RED,
            expected_path=None,
            expected_note="best score must be WIN (>=9000); any move to (5,2) achieves it",
            pass_rule="win",
            win_threshold=WIN_THRESHOLD,
        ),

        dict(
            id="P08",
            concept="Promotion Square Quality — non-corner preferred over (0,7) corner",
            category="CAGE",
            pieces=[("r",1,6),("r",1,4),("r",3,2),("b",2,7),("b",2,3)],
            player=RED,
            expected_path=None,
            expected_note="promotion to (0,7) must NOT be rank-1; "
                          "any non-corner promotion or (3,2) advance preferred",
            pass_rule="no_corner_promo_rank1",
        ),

        # ── Priority 3: 2K vs 1K conversion ─────────────────────────────────

        dict(
            id="P09",
            concept="2K vs 1K — open field opposition",
            category="KING_ENDGAME",
            pieces=[("R",3,2),("R",3,4),("B",1,4)],
            player=RED,
            expected_path=None,
            expected_note="all RED moves must score WIN (>=9000)",
            pass_rule="win",
            win_threshold=WIN_THRESHOLD,
        ),

        dict(
            id="P10",
            concept="2K vs 1K — approach to double-corner refuge",
            category="KING_ENDGAME",
            pieces=[("R",3,2),("R",5,4),("B",0,1)],
            player=RED,
            expected_path=None,
            expected_note="best score should be WIN (>=9000) or near-WIN (>=0); "
                          "failure = score <= -100 indicating horizon draw",
            pass_rule="near_win",
            win_threshold=0.0,
        ),

        dict(
            id="P11",
            concept="2K vs 1K — bridge formation forcing corner",
            category="KING_ENDGAME",
            pieces=[("R",2,1),("R",4,3),("B",1,4)],
            player=RED,
            expected_path=None,
            expected_note="best score should be WIN (>=9000); "
                          "engine must see the 2K vs 1K forced win",
            pass_rule="win",
            win_threshold=WIN_THRESHOLD,
        ),

        # ── Priority 4: 3K vs 2K ─────────────────────────────────────────────

        dict(
            id="P12",
            concept="3K vs 2K — forcing exchange to 2K vs 1K",
            category="KING_ENDGAME",
            pieces=[("R",1,2),("R",3,4),("R",5,2),("B",2,1),("B",4,1)],
            player=RED,
            expected_path=None,
            expected_note="best score must be WIN (>=9000); engine plans exchange to 2K vs 1K",
            pass_rule="win",
            win_threshold=WIN_THRESHOLD,
        ),

        dict(
            id="P13",
            concept="3K vs 2K — avoid losing exchange (material preservation)",
            category="STRATEGY",
            pieces=[("R",1,4),("R",3,4),("R",5,4),("B",2,3),("B",4,3)],
            player=RED,
            expected_path=None,
            expected_note="informational; rank-1 should prefer moves scoring >= second option; "
                          "avoid reducing to 1K vs 1K draw",
            pass_rule="info",
        ),

        dict(
            id="P14",
            concept="Forcing Opponent Kings to the Side (2K=2K positional)",
            category="STRATEGY",
            pieces=[("R",3,4),("R",5,2),("B",2,7),("B",4,7)],
            player=RED,
            expected_path=None,
            expected_note="informational; best move restricts BLACK king mobility; "
                          "report mobility after each option",
            pass_rule="info",
        ),

        dict(
            id="P15",
            concept="Reduction of Forces — correct tactical exchange choice",
            category="TACTICAL",
            pieces=[("r",5,2),("r",5,6),("r",3,4),
                    ("b",4,3),("b",4,5),("b",2,3),("b",2,5)],
            player=RED,
            expected_path=None,
            expected_note="both double-jumps score equally at D6 (+165); "
                          "engine must prefer a jump over a non-jump",
            pass_rule="multi_jump_rank1",
        ),
    ]


# ── evaluation logic ──────────────────────────────────────────────────────────

def _evaluate(pos: dict, d6_scored: list, board) -> tuple[str, str]:
    """
    Return (verdict, diagnosis) for a position given the D6 scored list.
    verdict: 'PASS' | 'FAIL' | 'INFO' | 'WARN'
    """
    rule = pos["pass_rule"]
    expected = pos.get("expected_path")

    if rule == "rank1":
        if expected is None:
            return "WARN", "expected_path not set"
        rank = _rank_of(d6_scored, expected)
        if rank == 1:
            return "PASS", f"expected path is rank-1 (score {d6_scored[0][1]:.1f})"
        else:
            sc = next((s for mv,s in d6_scored if _pk(mv["path"])==_pk(expected)), None)
            return "FAIL", f"expected path is rank-{rank} (score {sc}); rank-1={_path_str(d6_scored[0][0]['path'])} ({d6_scored[0][1]:.1f})"

    elif rule == "multi_jump_rank1":
        # Best move must be a multi-piece capture (len(captured)>=2)
        best_mv, best_sc = d6_scored[0]
        n_captured = len(best_mv.get("captured", []))
        if n_captured >= 2:
            return "PASS", f"rank-1 is multi-jump ({n_captured} captures, score {best_sc:.1f})"
        else:
            # Check if any multi-jump exists
            multi = [(mv,sc) for mv,sc in d6_scored if len(mv.get("captured",[]))>=2]
            if multi:
                return "FAIL", (
                    f"rank-1 is single/no-jump ({n_captured} cap, {best_sc:.1f}); "
                    f"best multi-jump is rank-{_rank_of(d6_scored,multi[0][0]['path'])} "
                    f"score={multi[0][1]:.1f}"
                )
            return "INFO", "no multi-jump available; all moves are single captures or simples"

    elif rule == "win":
        best_sc = d6_scored[0][1]
        thr = pos.get("win_threshold", WIN_THRESHOLD)
        if best_sc >= thr:
            return "PASS", f"best score {best_sc:.1f} >= threshold {thr:.0f}"
        return "FAIL", f"best score {best_sc:.1f} < WIN threshold {thr:.0f}"

    elif rule == "near_win":
        best_sc = d6_scored[0][1]
        thr = pos.get("win_threshold", 0.0)
        if best_sc >= thr:
            return "PASS", f"best score {best_sc:.1f} >= threshold {thr:.0f}"
        return "WARN", f"best score {best_sc:.1f} < near-win threshold {thr:.0f} (possible draw/horizon)"

    elif rule == "no_corner_promo_rank1":
        # Best move must NOT promote to corner squares (0,7) or (0,0)
        # (0,0) is not a dark square so only check (0,7)
        best_path = d6_scored[0][0]["path"]
        corner_pk = _pk([[1,6],[0,7]])
        if _pk(best_path) == corner_pk:
            rank2 = d6_scored[1][0]["path"] if len(d6_scored) > 1 else None
            return "FAIL", (
                f"rank-1 promotes to corner (0,7); gap={d6_scored[0][1]-d6_scored[1][1]:.1f}; "
                f"rank-2={_path_str(rank2)}"
            )
        # Check if corner promo is in top-3 at all
        corner_rank = _rank_of(d6_scored, [[1,6],[0,7]])
        return "PASS", (
            f"rank-1={_path_str(best_path)} (not corner); "
            f"corner (0,7) promo is rank-{corner_rank}"
        )

    elif rule == "info":
        best_sc = d6_scored[0][1]
        best_path = d6_scored[0][0]["path"]
        return "INFO", f"rank-1={_path_str(best_path)} score={best_sc:.1f}"

    return "WARN", f"unknown rule: {rule}"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    positions = _define_positions()
    results   = []

    BAR = "─" * 110
    print(f"\nCheckers Concept Benchmark  D6={D6_DEPTH}  D8={D8_DEPTH}  D8_trigger_gap≤{D8_GAP_TRIGGER}")
    print(BAR)
    print(f"{'ID':>4}  {'Cat':>12}  {'Verdict':>6}  {'D6 best':>24}  {'D6 sc':>8}  {'D6 gap':>7}  {'D8?':>4}  {'D8 best':>24}  {'D8 sc':>8}")
    print(BAR)

    for pos in positions:
        board  = _mk(pos["pieces"])
        player = pos["player"]
        red_n, blk_n = _count_pieces(board)
        n_legal = len(get_all_legal_moves(board, player))

        # ── D6 ───────────────────────────────────────────────────────────────
        d6_best, d6_scored, d6_elapsed, d6_nodes = _search(board, player, D6_DEPTH)
        d6_top_gap = get_d6_top_gap([(None, sc) for _, sc in d6_scored])
        d6_best_sc = d6_scored[0][1] if d6_scored else float("-inf")

        verdict, diagnosis = _evaluate(pos, d6_scored, board)

        # ── D8 (conditional) ─────────────────────────────────────────────────
        run_d8 = (
            (d6_top_gap <= D8_GAP_TRIGGER and d6_top_gap > 0)
            or verdict == "FAIL"
        )
        d8_best = d8_best_sc = d8_elapsed = d8_nodes = None
        if run_d8 and n_legal >= 2:
            d8_best, d8_scored, d8_elapsed, d8_nodes = _search(board, player, D8_DEPTH)
            d8_best_sc = d8_scored[0][1] if d8_scored else None
            # Re-evaluate with D8 if D6 failed
            if verdict == "FAIL":
                d8_verdict, d8_diagnosis = _evaluate(pos, d8_scored, board)
                if d8_verdict == "PASS":
                    diagnosis = f"[D6 FAIL → D8 PASS] {d8_diagnosis}"
                    verdict = "WARN"  # horizon effect confirmed

        # ── table row ────────────────────────────────────────────────────────
        d6_best_str = _path_str(d6_best)[:24]
        d8_str      = _path_str(d8_best)[:24] if d8_best else "—"
        d6_gap_str  = f"{d6_top_gap:+.1f}" if d6_top_gap != float("inf") else "  inf"
        d8_sc_str   = f"{d8_best_sc:+.1f}" if d8_best_sc is not None else "  n/a"
        d8_marker   = "D8" if run_d8 else " —"

        print(
            f"{pos['id']:>4}  {pos['category']:>12}  {verdict:>6}  {d6_best_str:>24}  "
            f"{d6_best_sc:>+8.1f}  {d6_gap_str:>7}  {d8_marker:>4}  {d8_str:>24}  {d8_sc_str:>8}"
        )

        if verdict in ("FAIL", "WARN"):
            print(f"       ↳ {pos['id']} [{pos['concept']}]")
            print(f"         Diagnosis: {diagnosis}")
            print(f"         Expected : {pos['expected_note']}")
            if run_d8 and d8_best:
                print(f"         D8 best  : {_path_str(d8_best)}  score={d8_best_sc:.1f}")

        # ── mobility detail for P14 ───────────────────────────────────────────
        if pos["id"] == "P14":
            print(f"       Mobility analysis for P14 (BLACK moves after each RED option):")
            for mv, sc in d6_scored[:5]:
                blk_mob = _mobility_after(board, player, mv)
                print(f"         {_path_str(mv['path']):30}  score={sc:+8.1f}  BLACK_mob_after={blk_mob}")

        # ── P13 extra: report whether rank-1 leads to 1K vs 1K ───────────────
        if pos["id"] == "P13":
            print(f"       P13 king exchange analysis:")
            for mv, sc in d6_scored[:4]:
                b2 = apply_move(board, mv)
                r2, b2_cnt = _count_pieces(b2)
                note = "→ 1K vs 1K (draw risk)" if r2 == 1 and b2_cnt == 1 else ""
                print(f"         {_path_str(mv['path']):30}  score={sc:+8.1f}  red={r2} blk={b2_cnt}  {note}")

        # ── accumulate result ─────────────────────────────────────────────────
        results.append({
            "id":            pos["id"],
            "concept":       pos["concept"],
            "category":      pos["category"],
            "verdict":       verdict,
            "diagnosis":     diagnosis,
            "expected_note": pos["expected_note"],
            "n_legal":       n_legal,
            "d6_best_path":  d6_best,
            "d6_best_score": round(d6_best_sc, 2),
            "d6_top_gap":    round(d6_top_gap, 2) if d6_top_gap != float("inf") else None,
            "d6_elapsed_s":  d6_elapsed,
            "d6_nodes":      d6_nodes,
            "d8_run":        run_d8,
            "d8_best_path":  d8_best,
            "d8_best_score": round(d8_best_sc, 2) if d8_best_sc is not None else None,
            "d8_elapsed_s":  d8_elapsed,
            "d8_nodes":      d8_nodes,
            "d6_top3":       [
                {"path": mv["path"], "score": round(float(sc), 2)}
                for mv, sc in d6_scored[:3]
            ],
        })

    print(BAR)

    # ── Summary ───────────────────────────────────────────────────────────────
    verdict_counts = {}
    for r in results:
        verdict_counts[r["verdict"]] = verdict_counts.get(r["verdict"], 0) + 1

    print()
    print("SUMMARY")
    for v, n in sorted(verdict_counts.items()):
        print(f"  {v:>6}: {n}")
    fails  = [r for r in results if r["verdict"] == "FAIL"]
    warns  = [r for r in results if r["verdict"] == "WARN"]
    if fails:
        print(f"\n  FAILED positions: {[r['id'] for r in fails]}")
        for r in fails:
            print(f"    {r['id']} — {r['concept']}")
            print(f"      {r['diagnosis']}")
    if warns:
        print(f"\n  WARN positions (D6 fail → D8 pass = horizon effect):")
        for r in warns:
            print(f"    {r['id']} — {r['concept']}")
            print(f"      {r['diagnosis']}")
    if not fails and not warns:
        print("\n  All positions passed or informational only.")
        print("  Depth-6 is sufficient for all tested concepts.")

    # ── Write JSON ────────────────────────────────────────────────────────────
    out = {
        "meta": {
            "d6_depth":       D6_DEPTH,
            "d8_depth":       D8_DEPTH,
            "d8_gap_trigger": D8_GAP_TRIGGER,
            "win_threshold":  WIN_THRESHOLD,
            "scorer":         "search_root_all_scores",
        },
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(results)} results → {OUT_PATH}")


if __name__ == "__main__":
    main()
