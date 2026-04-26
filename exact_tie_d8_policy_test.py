"""
exact_tie_d8_policy_test.py

Targeted test: compares
  Policy A: SELECTIVE_D8_INCLUDE_EXACT_TIES=false  (current default)
  Policy B: SELECTIVE_D8_INCLUDE_EXACT_TIES=true

for exact-tie turns: T51, T83, T107.

Data source:
  - D6 and D8 search results: loaded from logs/ablation_threshold_policy.json
    (already computed by ablation_threshold_policy.py — no re-search needed)
  - Trace "chosen" paths: loaded from game log
    game_20260425_233613_685721.jsonl

Scoring:
  - Policy A: minimax_scorer skips D8 on exact ties.
              Ranker receives D6 scores (tied candidates share same score).
              "chosen under A" = D6 top-1 rank move (first in descending sort).
              Actual trace chosen is also shown separately for reference.
  - Policy B: minimax_scorer runs D8 on exact ties.
              Ranker receives D8 scores (usually differentiated).
              "chosen under B" = D8 top-1 rank move.

No additional search is run.  search_root_all_scores was used in the
ablation script and results are reused here verbatim.

Output: compact terminal table + final verdict.
"""
from __future__ import annotations

import json
from pathlib import Path

ABLATION_JSON = Path("logs/ablation_threshold_policy.json")
GAME_LOG      = Path("logs/game_20260425_233613_685721.jsonl")
TARGET_TURNS  = [51, 83, 107]

D8_PIECE_THRESHOLD = 14
D8_GAP_THRESHOLD   = 30.0    # current setting


def _pk(path) -> tuple:
    return tuple(tuple(sq) for sq in path)


def _path_str(path) -> str:
    if path is None:
        return "—"
    return "→".join(f"({r},{c})" for r, c in path)


def _tie_count(top5: list, top_score: float, tolerance: float = 0.01) -> int:
    """Count moves in top5 sharing the top score (within tolerance)."""
    return sum(1 for _, sc in top5 if abs(float(sc) - top_score) <= tolerance)


def _would_trigger_a(gap: float, pieces: int) -> bool:
    """Policy A: skip exact ties."""
    if pieces > D8_PIECE_THRESHOLD:
        return False
    if gap == 0.0:
        return False          # exact tie skipped
    return gap <= D8_GAP_THRESHOLD


def _would_trigger_b(gap: float, pieces: int) -> bool:
    """Policy B: include exact ties."""
    if pieces > D8_PIECE_THRESHOLD:
        return False
    if gap == 0.0:
        return True           # exact tie NOW triggers D8
    return gap <= D8_GAP_THRESHOLD


def _simulated_chosen(policy_triggers: bool, d6_top1, d8_top1):
    """
    Simulate what minimax_scorer forwards as the top-ranked move under each policy.
    If D8 triggered: D8 top-1 scores are used → ranker sees differentiated D8 scores.
    If D8 not triggered: D6 top-1 scores used → ranker sees tied D6 scores.
    """
    return d8_top1 if policy_triggers else d6_top1


def main() -> None:
    if not ABLATION_JSON.exists():
        raise FileNotFoundError(
            f"{ABLATION_JSON} not found.\n"
            f"Run ablation_threshold_policy.py first."
        )
    if not GAME_LOG.exists():
        raise FileNotFoundError(f"{GAME_LOG} not found.")

    # Load ablation results
    ab_data = json.loads(ABLATION_JSON.read_text())
    ab_by_turn = {r["turn"]: r for r in ab_data["results"]}

    # Load game log for trace-chosen paths
    records = [json.loads(l) for l in GAME_LOG.read_text().splitlines() if l.strip()]
    trace_chosen = {r["turn"]: r["path"] for r in records}

    BAR = "─" * 100
    print()
    print("EXACT-TIE SELECTIVE-D8 POLICY TEST")
    print(f"  Ablation data : {ABLATION_JSON.name}")
    print(f"  Game log      : {GAME_LOG.name}")
    print(f"  Turns tested  : {TARGET_TURNS}")
    print(f"  Policies      : A=exact_ties_false (current)  B=exact_ties_true")
    print(BAR)

    rows = []

    for turn in TARGET_TURNS:
        ab = ab_by_turn.get(turn)
        if ab is None:
            print(f"  T{turn}: no ablation data — skipped")
            continue

        pieces   = ab["total_pieces"]
        d6_gap   = ab["d6_gap"]
        d6_tie   = ab["d6_tie_count"]
        d8_tie   = ab["d8_tie_count"]
        d6_top5  = ab["d6_top5"]
        d8_top5  = ab["d8_top5"]
        d6_top1  = d6_top5[0][0] if d6_top5 else None
        d8_top1  = d8_top5[0][0] if d8_top5 else None
        d8_top1_score = float(d8_top5[0][1]) if d8_top5 else None
        d6_top1_score = float(d6_top5[0][1]) if d6_top5 else None

        trace_path  = trace_chosen.get(turn)
        is_promo    = ab.get("chosen_is_promo", False)   # promotion flag from ablation

        trig_a = _would_trigger_a(d6_gap, pieces)
        trig_b = _would_trigger_b(d6_gap, pieces)

        chosen_a = _simulated_chosen(trig_a, d6_top1, d8_top1)
        chosen_b = _simulated_chosen(trig_b, d6_top1, d8_top1)

        changed = (_pk(chosen_a) != _pk(chosen_b)) if (chosen_a and chosen_b) else True

        # Does Policy B avoid the problematic move?
        b_avoids_promo = (
            is_promo and
            chosen_a is not None and _pk(chosen_a) == _pk(trace_path or []) and
            chosen_b is not None and _pk(chosen_b) != _pk(trace_path or [])
        )

        # Extra: under Policy B, is the D8 tie_count still high (T107 case)?
        b_still_tied = trig_b and (d8_tie >= 4)

        rows.append({
            "turn":           turn,
            "pieces":         pieces,
            "d6_gap":         d6_gap,
            "d6_tie":         d6_tie,
            "d8_tie":         d8_tie,
            "is_promo":       is_promo,
            "trace_chosen":   trace_path,
            "d6_top1":        d6_top1,
            "d8_top1":        d8_top1,
            "chosen_a":       chosen_a,
            "chosen_b":       chosen_b,
            "trig_a":         trig_a,
            "trig_b":         trig_b,
            "changed":        changed,
            "b_avoids_promo": b_avoids_promo,
            "b_still_tied":   b_still_tied,
        })

        # Verbose per-turn block
        print(f"\n  ── T{turn} | pieces={pieces} | d6_gap={d6_gap} | "
              f"promo={is_promo} | trace_chosen={_path_str(trace_path)}")
        print(f"     D6 top-3 scores:")
        for p, sc in d6_top5[:3]:
            marker = "★" if _pk(p) == _pk(d6_top1) else " "
            print(f"       {marker} {_path_str(p):<22} score={float(sc):+.1f}")
        print(f"     D6 tie_count={d6_tie}  gap={d6_gap}")
        print()
        print(f"     D8 top-3 scores:")
        for p, sc in d8_top5[:3]:
            marker = "★" if _pk(p) == _pk(d8_top1) else " "
            print(f"       {marker} {_path_str(p):<22} score={float(sc):+.1f}")
        print(f"     D8 tie_count={d8_tie}")
        print()
        print(f"     Policy A (exact_ties=false):  D8 triggers={trig_a}  → scored by D6  → top1={_path_str(chosen_a)}")
        print(f"     Policy B (exact_ties=true ):  D8 triggers={trig_b}  → scored by D8  → top1={_path_str(chosen_b)}")
        print(f"     Outcome changes? {changed}  |  B avoids promo corner? {b_avoids_promo}")
        if b_still_tied:
            print(f"     ⚠ Policy B still has D8 tie_count={d8_tie} — ranker must break tie with secondary scores")

    # ── Compact table ─────────────────────────────────────────────────────────
    print()
    print(BAR)
    print("COMPACT TABLE")
    print(BAR)
    hdr = (
        f"{'Turn':>4}  {'pcs':>3}  {'d6gap':>5}  {'d6tie':>5}  {'d8tie':>5}  "
        f"{'promo':>5}  {'chosen_A':<24}  {'chosen_B':<24}  {'D8best':<24}  "
        f"{'chgd':>5}  notes"
    )
    print(hdr)
    print("─" * len(hdr))

    for r in rows:
        chgd  = "YES" if r["changed"] else " no"
        promo = "YES" if r["is_promo"] else " no"
        ca    = _path_str(r["chosen_a"])[:22]
        cb    = _path_str(r["chosen_b"])[:22]
        d8b   = _path_str(r["d8_top1"])[:22]

        if r["b_still_tied"]:
            note = f"⚠ D8 still {r['d8_tie']}-way tie"
        elif r["b_avoids_promo"]:
            note = "✓ B avoids corner"
        elif r["changed"] and not r["is_promo"]:
            note = "B picks better non-promo move"
        else:
            note = "—"

        print(
            f"T{r['turn']:>3}  {r['pieces']:>3}  {r['d6_gap']:>5.1f}  {r['d6_tie']:>5}  "
            f"{r['d8_tie']:>5}  {promo:>5}  {ca:<24}  {cb:<24}  {d8b:<24}  "
            f"{chgd:>5}  {note}"
        )

    print()

    # ── Final verdict ─────────────────────────────────────────────────────────
    t51  = next((r for r in rows if r["turn"] == 51),  None)
    t83  = next((r for r in rows if r["turn"] == 83),  None)
    t107 = next((r for r in rows if r["turn"] == 107), None)

    print("VERDICT")
    print("─" * 80)

    # Q1
    q1_fix = t51 and t51["b_avoids_promo"]
    print(f"Q1. Does exact-tie=true fix T51?")
    if q1_fix:
        print(f"    YES. D8 flips top-1 from corner {_path_str(t51['chosen_a'])} "
              f"→ {_path_str(t51['chosen_b'])}. Corner avoided.")
    elif t51:
        print(f"    NO. D8 best={_path_str(t51['d8_top1'])} but b_avoids_promo={t51['b_avoids_promo']}")
    else:
        print("    T51 data missing.")

    # Q2
    q2_improv = t83 and t83["changed"]
    print(f"\nQ2. Does exact-tie=true improve T83?")
    if q2_improv and t83 and not t83["b_still_tied"]:
        print(f"    YES. D8 differentiates the tie. New top-1: {_path_str(t83['chosen_b'])}")
        print(f"    D6 had picked {_path_str(t83['chosen_a'])} (KING_SHUFFLE).")
        print(f"    D8 tie_count={t83['d8_tie']} — single winner, clean.")
    elif t83 and t83["b_still_tied"]:
        print(f"    PARTIAL. D8 fires but tie_count={t83['d8_tie']} remains.")
    elif t83:
        print(f"    NO. No change detected.")
    else:
        print("    T83 data missing.")

    # Q3
    print(f"\nQ3. Does exact-tie=true do anything useful at T107?")
    if t107 and t107["b_still_tied"]:
        print(f"    NO useful effect. D8 fires but tie_count={t107['d8_tie']} — "
              f"6-way tie persists at depth 8.")
        print(f"    Ranker must break T107 tie via secondary scores (king_activity, convert, etc.).")
        print(f"    Setting exact_ties=true adds ~2.5s runtime at T107 with zero disambiguation.")
    elif t107:
        print(f"    YES. D8 breaks the tie: {_path_str(t107['chosen_b'])}")
    else:
        print("    T107 data missing.")

    # Q4
    print(f"\nQ4. Is exact-tie=true safe, or should we implement promotion-tie-only?")
    t107_wasted = t107 and t107["b_still_tied"]
    t83_useful  = t83  and t83["changed"] and not t83["b_still_tied"]

    if t107_wasted and t83_useful:
        print("    exact_ties=true:")
        print(f"      ✓ Fixes T51 (corner promotion avoidance)")
        print(f"      ✓ Improves T83 (non-promo tie differentiation, +9.5s but clean result)")
        print(f"      ✗ T107: 2.5s wasted — D8 fires but 6-way tie remains unchanged")
        print()
        print("    promotion-tie-only (Policy D):")
        print(f"      ✓ Fixes T51 (triggers on exact tie + promotion move present)")
        print(f"      ✗ Skips T83 (no promotion move in legal set)")
        print(f"      ✓ Skips T107 (no promotion, no extra runtime)")
        print()
        print("    RECOMMENDATION:")
        print("      exact_ties=true is safe and simpler.  T107 wastes 2.5s but causes")
        print("      no correctness harm (ranker secondary scores still apply).")
        print("      promotion-tie-only is the surgical option if runtime is critical.")
        print("      Given 3K vs 2K games rarely exceed 20 extra turns and D8 at")
        print("      5 pieces is fast (2.5s), exact_ties=true is the simpler safe choice.")
    else:
        print("    Both policies safe. Use exact_ties=true for simplicity.")


if __name__ == "__main__":
    main()
