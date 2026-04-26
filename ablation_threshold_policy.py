"""
ablation_threshold_policy.py

Read-only ablation: verifies the integrated trace report claims for the
game log game_20260425_233613_685721.jsonl.

Claims verified:
  1. T51 and T67 are corner-promotion commitment points.
  2. T67 was missed because SELECTIVE_D8_GAP_THRESHOLD=30 was too low.
  3. Raising threshold to 50 would trigger D8 at T67.
  4. Score gap near corner promotion may not be a pure "corner square premium".
  5. T107 shows a real 3K vs 2K conversion tie / evaluator weakness.

Policies tested per suspicious turn:
  A. current  threshold=30  (exact ties skipped)
  B. threshold=50           (exact ties skipped)
  C. threshold=60           (exact ties skipped)
  D. promotion-tie: trigger if top-gap==0 AND chosen move is a promotion

Suspicious turns: T51, T67, T83, T107
  - T51:  RED corner promotion at (1,6)->(0,7), D6 exact tie
  - T67:  RED corner promotion at (1,6)->(0,7), D6 gap=35
  - T83:  exact tie, non-promotion (T51-like but NOT promotion)
  - T107: 6-way tie in 3K vs 2K position

Uses search_root_all_scores only. Does NOT modify any engine code.

Output: logs/ablation_threshold_policy.json  +  compact terminal table.
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

from checkers.engine.board import RED, BLACK, RED_KING, BLACK_KING
from checkers.engine.rules import get_all_legal_moves, apply_move
from checkers.search.minimax_core import search_root_all_scores, clear_transposition_table

# ── Config ────────────────────────────────────────────────────────────────────

GAME_LOG  = Path("logs/game_20260425_233613_685721.jsonl")
OUT_PATH  = Path("logs/ablation_threshold_policy.json")

# Suspicious turns from trace (RED turns only)
TARGET_TURNS = [51, 67, 83, 107]

D6_DEPTH = 6
D8_DEPTH = 8

PIECE_THRESHOLD = 14  # unchanged across all policies

# Policies: (label, gap_threshold, include_exact_ties, promotion_tie_only)
POLICIES = [
    ("A_thr30",   30.0, False, False),
    ("B_thr50",   50.0, False, False),
    ("C_thr60",   60.0, False, False),
    ("D_promtie", 30.0, False, True),   # exact-tie only when a promotion is involved
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _pk(path) -> tuple:
    return tuple(tuple(sq) for sq in path)


def _path_str(path) -> str:
    if path is None:
        return "—"
    return "→".join(f"({r},{c})" for r, c in path)


def _score_for(scored: list, path) -> float | None:
    key = _pk(path)
    for mv, sc in scored:
        if _pk(mv["path"]) == key:
            return float(sc)
    return None


def _top_gap(scored: list) -> float:
    if len(scored) < 2:
        return float("inf")
    s0 = float(scored[0][1])
    s1 = float(scored[1][1])
    return round(s0 - s1, 4)


def _total_pieces(board) -> int:
    return sum(1 for r in range(8) for c in range(8) if board[r][c] != 0)


def _is_promotion_move(board, move) -> bool:
    """True if applying this move promotes a RED man to king."""
    path = move["path"]
    if not path:
        return False
    src_r, src_c = path[0]
    dst_r, dst_c = path[-1]
    piece = board[src_r][src_c]
    return piece == RED and dst_r == 0


def _run_search(board, depth: int, move_list: list | None = None) -> tuple:
    """
    Returns (best_path, scored_list, elapsed_s, nodes).
    scored_list: [(move_dict, score), ...] sorted descending.
    """
    legal = move_list if move_list is not None else get_all_legal_moves(board, RED)
    clear_transposition_table()
    t0 = time.perf_counter()
    best_mv, best_sc, scored, stats = search_root_all_scores(
        board=board,
        current_player=RED,
        depth=depth,
        legal_moves=legal,
        use_tt=True,
        use_tactical_extension=True,
        use_phase7a=True,
    )
    elapsed = round(time.perf_counter() - t0, 2)
    nodes = getattr(stats, "nodes", None)
    best_path = best_mv["path"] if best_mv else None
    return best_path, scored, elapsed, nodes


# ── Board reconstruction ──────────────────────────────────────────────────────

def _make_start() -> list:
    b = [[0] * 8 for _ in range(8)]
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r][c] = RED
    for r in range(0, 3):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r][c] = BLACK
    return b


def _rebuild(records: list) -> dict:
    """boards[t] = board AFTER turn t = board BEFORE turn t+1."""
    board = _make_start()
    boards = {0: [row[:] for row in board]}
    for rec in records:
        t = rec["turn"]
        move = {"type": rec["move_type"], "path": rec["path"],
                "captured": rec.get("captured", [])}
        board = apply_move(board, move)
        boards[t] = [row[:] for row in board]
    return boards


# ── Policy trigger logic ──────────────────────────────────────────────────────

def _would_trigger(
    total_pcs: int,
    d6_gap: float,
    has_promotion: bool,
    gap_threshold: float,
    include_exact_ties: bool,
    promotion_tie_only: bool,
) -> tuple[bool, str]:
    """
    Returns (would_trigger, reason_string).
    """
    if total_pcs > PIECE_THRESHOLD:
        return False, f"pieces={total_pcs}>{PIECE_THRESHOLD}"

    if promotion_tie_only:
        # Policy D: trigger ONLY if exact tie AND there's a promotion move
        if d6_gap == 0.0 and has_promotion:
            return True, "exact_tie+promotion"
        if d6_gap == 0.0 and not has_promotion:
            return False, "exact_tie_no_promotion"
        # Non-tie: fall through to regular gap logic with base threshold=30
        if d6_gap > 30.0:
            return False, f"gap={d6_gap}>30.0"
        return True, f"gap={d6_gap}<=30.0"

    # Policies A/B/C
    if d6_gap == 0.0 and not include_exact_ties:
        return False, "exact_tie_skipped"
    if d6_gap > gap_threshold:
        return False, f"gap={d6_gap}>{gap_threshold}"
    return True, f"gap={d6_gap}<={gap_threshold}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not GAME_LOG.exists():
        raise FileNotFoundError(f"Game log not found: {GAME_LOG}")

    records = [json.loads(l) for l in GAME_LOG.read_text().splitlines() if l.strip()]
    rec_by_turn = {r["turn"]: r for r in records}
    boards = _rebuild(records)

    BAR = "─" * 110
    print()
    print("ABLATION: D8 THRESHOLD & PROMOTION TIE POLICY")
    print(f"  Game log  : {GAME_LOG.name}")
    print(f"  Turns     : {TARGET_TURNS}")
    print(f"  D8 depth  : {D8_DEPTH}")
    print(BAR)

    all_results = []

    for turn in TARGET_TURNS:
        rec = rec_by_turn.get(turn)
        if rec is None:
            print(f"  T{turn:>3}  ← not found in log (BLACK turn or missing); skipping")
            continue

        player_who_moved = rec.get("player_who_moved", None)
        if player_who_moved != 1:  # 1=RED in this log
            print(f"  T{turn:>3}  ← BLACK turn (player={player_who_moved}); skipping")
            continue

        board_before = boards[turn - 1]
        chosen_path  = rec["path"]
        total_pcs    = _total_pieces(board_before)
        legal_all    = get_all_legal_moves(board_before, RED)

        # Check if any legal move is a promotion
        has_promotion_legal = any(_is_promotion_move(board_before, m) for m in legal_all)
        chosen_is_promotion = _is_promotion_move(board_before, {"path": chosen_path})

        print()
        print(f"  ── T{turn} | pieces={total_pcs} | chosen={_path_str(chosen_path)} "
              f"| chosen_is_promo={chosen_is_promotion} | any_legal_promo={has_promotion_legal}")

        # ── D6 ─────────────────────────────────────────────────────────────────
        print(f"     Running D6...")
        d6_best, d6_scored, d6_elapsed, _ = _run_search(board_before, D6_DEPTH)
        d6_best_score   = float(d6_scored[0][1]) if d6_scored else float("-inf")
        d6_chosen_score = _score_for(d6_scored, chosen_path)
        d6_gap          = _top_gap(d6_scored)
        d6_tie_count    = sum(1 for _, sc in d6_scored if float(sc) == d6_best_score)

        print(f"     D6 best={_path_str(d6_best)} score={d6_best_score:.1f} "
              f"gap={d6_gap:.1f} tie_count={d6_tie_count} elapsed={d6_elapsed}s")

        # D6 full table (top-5)
        d6_top5 = [(mv["path"], float(sc)) for mv, sc in d6_scored[:5]]

        # ── D8 (run once, reuse for all policies) ─────────────────────────────
        print(f"     Running D8...")
        d8_best, d8_scored, d8_elapsed, d8_nodes = _run_search(board_before, D8_DEPTH)
        d8_best_score   = float(d8_scored[0][1]) if d8_scored else float("-inf")
        d8_chosen_score = _score_for(d8_scored, chosen_path)
        d8_gap          = _top_gap(d8_scored)
        d8_tie_count    = sum(1 for _, sc in d8_scored if float(sc) == d8_best_score)
        top1_changed    = (d8_best is not None and d6_best is not None and
                           _pk(d8_best) != _pk(d6_best))

        print(f"     D8 best={_path_str(d8_best)} score={d8_best_score:.1f} "
              f"gap={d8_gap:.1f} tie_count={d8_tie_count} elapsed={d8_elapsed}s nodes={d8_nodes}")
        print(f"     top1_changed={top1_changed} "
              f"d8_avoids_corner={'yes' if (top1_changed and chosen_is_promotion) else 'n/a'}")

        # ── Policy triggers ────────────────────────────────────────────────────
        policy_results = {}
        print(f"     Policy triggers (pieces={total_pcs}, d6_gap={d6_gap:.1f}):")
        for pol_label, gap_thr, incl_ties, promo_tie in POLICIES:
            triggered, reason = _would_trigger(
                total_pcs, d6_gap, has_promotion_legal,
                gap_thr, incl_ties, promo_tie
            )
            marker = "✓ TRIGGER" if triggered else "✗ skip   "
            print(f"       {pol_label:<12} {marker}  ({reason})")
            policy_results[pol_label] = {"triggered": triggered, "reason": reason}

        # Store result
        all_results.append({
            "turn":               turn,
            "chosen_path":        chosen_path,
            "chosen_is_promo":    chosen_is_promotion,
            "any_legal_promo":    has_promotion_legal,
            "total_pieces":       total_pcs,
            "d6_best_path":       d6_best,
            "d6_best_score":      round(d6_best_score, 2),
            "d6_chosen_score":    round(d6_chosen_score, 2) if d6_chosen_score is not None else None,
            "d6_gap":             round(d6_gap, 2),
            "d6_tie_count":       d6_tie_count,
            "d8_best_path":       d8_best,
            "d8_best_score":      round(d8_best_score, 2),
            "d8_chosen_score":    round(d8_chosen_score, 2) if d8_chosen_score is not None else None,
            "d8_gap":             round(d8_gap, 2),
            "d8_tie_count":       d8_tie_count,
            "d8_elapsed_s":       d8_elapsed,
            "d8_nodes":           d8_nodes,
            "top1_changed":       top1_changed,
            "d6_top5":            [(p, s) for p, s in d6_top5],
            "d8_top5":            [(mv["path"], float(sc)) for mv, sc in d8_scored[:5]],
            "policy_triggers":    policy_results,
        })

    # ── Compact summary table ──────────────────────────────────────────────────
    print()
    print(BAR)
    print("COMPACT TABLE")
    print(BAR)
    hdr = (
        f"{'Turn':>4}  {'pcs':>3}  {'D6gap':>6}  {'chosen':<22}  {'D6best':<22}  "
        f"{'D8best':<22}  {'thr30':>5}  {'thr50':>5}  {'thr60':>5}  {'promtie':>7}  "
        f"{'t1chg':>5}  {'d8ties':>6}  diagnosis"
    )
    print(hdr)
    print("─" * len(hdr))

    for r in all_results:
        t30 = "YES" if r["policy_triggers"].get("A_thr30", {}).get("triggered") else " no"
        t50 = "YES" if r["policy_triggers"].get("B_thr50", {}).get("triggered") else " no"
        t60 = "YES" if r["policy_triggers"].get("C_thr60", {}).get("triggered") else " no"
        tpt = "YES" if r["policy_triggers"].get("D_promtie", {}).get("triggered") else " no"
        t1  = "YES" if r["top1_changed"] else " no"

        # Diagnosis
        if r["d6_gap"] == 0.0 and r["chosen_is_promo"]:
            diag = "EXACT_TIE+CORNER_PROMO → H1+H2"
        elif r["d6_gap"] == 0.0:
            diag = "EXACT_TIE non-promo"
        elif r["chosen_is_promo"] and r["d6_gap"] <= 50.0:
            diag = "CORNER_PROMO gap<=50 (D8 missed at thr30)"
        elif r["d8_tie_count"] >= 4:
            diag = f"3K_CONV_DEGENERACY tie_count={r['d8_tie_count']}"
        else:
            diag = "no_issue"

        chsn = _path_str(r["chosen_path"])[:20]
        d6b  = _path_str(r["d6_best_path"])[:20]
        d8b  = _path_str(r["d8_best_path"])[:20]

        print(
            f"T{r['turn']:>3}  {r['total_pieces']:>3}  {r['d6_gap']:>6.1f}  "
            f"{chsn:<22}  {d6b:<22}  {d8b:<22}  "
            f"{t30:>5}  {t50:>5}  {t60:>5}  {tpt:>7}  "
            f"{t1:>5}  {r['d8_tie_count']:>6}  {diag}"
        )

    print()

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("VERDICT")
    print("─" * 80)

    t51 = next((r for r in all_results if r["turn"] == 51), None)
    t67 = next((r for r in all_results if r["turn"] == 67), None)
    t83 = next((r for r in all_results if r["turn"] == 83), None)
    t107 = next((r for r in all_results if r["turn"] == 107), None)

    # Q1: Is threshold 50 enough?
    q1 = "YES" if (t67 and t67["policy_triggers"]["B_thr50"]["triggered"]) else "NO"
    print(f"Q1. Is threshold=50 enough to catch T67?  → {q1}")
    if t67:
        print(f"    T67 d6_gap={t67['d6_gap']:.1f}, thr50_triggers={t67['policy_triggers']['B_thr50']['triggered']}")
        print(f"    D8 avoids corner promo: top1_changed={t67['top1_changed']}, chosen_is_promo={t67['chosen_is_promo']}")

    # Q2: Is exact-tie promotion D8 needed (for T51)?
    q2_needed = t51 and not t51["policy_triggers"]["A_thr30"]["triggered"] and t51["chosen_is_promo"]
    q2 = "YES" if q2_needed else "NO (already handled or not promo)"
    print(f"\nQ2. Is promotion-tie D8 needed (for T51)?  → {q2}")
    if t51:
        print(f"    T51 d6_gap={t51['d6_gap']:.1f}, chosen_is_promo={t51['chosen_is_promo']}")
        print(f"    promtie_triggers={t51['policy_triggers']['D_promtie']['triggered']}")
        print(f"    D8 would avoid corner: top1_changed={t51['top1_changed']}")

    # Q3: Is corner penalty still justified after D8?
    # If D8 would have caught BOTH T51 and T67 and avoided corner, penalty may be redundant
    d8_catches_t51 = t51 and t51["top1_changed"]
    d8_catches_t67 = t67 and t67["top1_changed"] and t67["policy_triggers"]["B_thr50"]["triggered"]
    q3 = "NO (D8 sufficient)" if (d8_catches_t51 and d8_catches_t67) else "YES (D8 alone insufficient)"
    print(f"\nQ3. Is corner penalty still needed after D8 fix?  → {q3}")
    print(f"    D8 catches T51={d8_catches_t51}, catches T67={d8_catches_t67}")

    # Q4: Is 3K vs 2K a separate evaluator problem?
    q4 = "YES" if (t107 and t107["d8_tie_count"] >= 4) else "NO (D8 breaks tie)"
    print(f"\nQ4. Is 3K vs 2K a separate evaluator problem?  → {q4}")
    if t107:
        print(f"    T107 d8_tie_count={t107['d8_tie_count']}, d8_gap={t107['d8_gap']:.1f}")
        print(f"    Diagnosis: D8 at depth 8 {'cannot' if t107['d8_tie_count']>=4 else 'CAN'} break the tie")

    # Q5: Minimal safe next change
    print(f"\nQ5. Minimal safe next change:")
    if q1 == "YES" and q2 == "YES":
        print("    → Raise SELECTIVE_D8_GAP_THRESHOLD to 50  (env var only, no code change)")
        print("    → Enable SELECTIVE_D8_INCLUDE_EXACT_TIES for promotions only (Policy D)")
        print("    → Do NOT add corner penalty yet — D8 may be sufficient")
    elif q1 == "YES":
        print("    → Raise SELECTIVE_D8_GAP_THRESHOLD to 50  (env var only)")
    else:
        print("    → Threshold 50 insufficient; corner penalty or threshold 60 needed")

    # ── Write JSON ────────────────────────────────────────────────────────────
    out = {
        "meta": {
            "game_log":    str(GAME_LOG),
            "target_turns": TARGET_TURNS,
            "d6_depth":    D6_DEPTH,
            "d8_depth":    D8_DEPTH,
            "scorer":      "search_root_all_scores",
            "policies": [
                {"label": pol[0], "gap_threshold": pol[1],
                 "include_exact_ties": pol[2], "promotion_tie_only": pol[3]}
                for pol in POLICIES
            ],
        },
        "results": all_results,
    }
    # Serialise paths (list of lists) cleanly
    def _serial(o):
        if isinstance(o, (list, tuple)) and len(o) == 2 and isinstance(o[0], (int, float)):
            return list(o)
        return o

    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {len(all_results)} results → {OUT_PATH}")


if __name__ == "__main__":
    main()
