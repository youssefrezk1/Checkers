"""
checkers/eval/proposal_coverage_benchmark.py
─────────────────────────────────────────────
Measures whether the Proposal Agent includes the symbolic-best move in its
shortlist, without modifying any engine/ranker/proposal logic.

Usage
-----
    python3 -m checkers.eval.proposal_coverage_benchmark --depth 6
    python3 -m checkers.eval.proposal_coverage_benchmark --depth 4 --near-margin 20
    python3 -m checkers.eval.proposal_coverage_benchmark --position-id pos_t41_promo_tiebreak
    # aliases also work:
    python3 -m checkers.eval.proposal_coverage_benchmark --margin 20 --id pos_t41_promo_tiebreak

Output
------
    logs/proposal_coverage_depth6.json

Score-gap direction
-------------------
    score_gap = best_symbolic_score - best_proposed_score
    Positive gap  → proposal missed a better move (bad)
    Zero gap      → proposal included the symbolic-best move (good)
    Negative gap  → impossible (proposed can't beat symbolic best)

Index mapping invariant
-----------------------
    proposal_agent internally translates LLM presentation indices to
    expansion_basis indices via _translate_to_basis_indices().
    The returned proposed_moves JSON therefore contains *expansion_basis*
    indices, NOT moves_with_facts presentation indices.
    The expansion_basis is state.legal_moves (the enriched list).
    We map proposed indices against state.legal_moves — the identical list
    the proposal agent used — so no re-building occurs.

State completeness
------------------
    Fields actually read by proposal_agent / build_proposal_prompts:
      board, current_player, strategic_context (required)
      symbolic_scored_moves  (optional — skipped here; triggers override pool)
      symbolic_best_score    (used in log line only; safe default 0.0)
      symbolic_gap           (used in log line only; safe default 0.0)
      turn_number            (printed in prompt; safe default 1)
      feedback               (retry feedback; None = no retry feedback)
      legal_moves            (expansion_basis for index translation)
    All fields are supplied to match the real graph path exactly.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ── Load .env before any module reads GROQ_API_KEY ────────────────────────────
# Must happen before importing proposal_agent (which reads env at module level).
from dotenv import load_dotenv, find_dotenv

env_path = find_dotenv(usecwd=True)
load_dotenv(env_path, override=True)


class _CaptureStdout:
    """Context manager: tee stdout to a StringIO buffer AND real stdout."""
    def __init__(self, buf: io.StringIO) -> None:
        self._buf = buf
        self._real: Any = None

    def __enter__(self) -> "_CaptureStdout":
        self._real = sys.stdout
        sys.stdout = self  # type: ignore[assignment]
        return self

    def write(self, data: str) -> int:
        self._buf.write(data)
        self._real.write(data)
        return len(data)

    def flush(self) -> None:
        self._real.flush()

    def __exit__(self, *_: Any) -> None:
        sys.stdout = self._real

from checkers.engine.board import RED, BLACK
from checkers.engine.move_facts import compute_move_facts
from checkers.engine.rules import get_all_legal_moves
from checkers.search.minimax_core import (
    clear_transposition_table,
    search_root_all_scores,
)
from checkers.state.state import CheckersState
from checkers.oldfiles.proposal_agent import (
    GROQ_PROPOSAL_MODEL,
    build_proposal_prompts,
    proposal_agent,
)
from checkers.eval.benchmark_positions import BENCHMARK_POSITIONS


# ── helpers ───────────────────────────────────────────────────────────────────

def _opp(p: int) -> int:
    return BLACK if p == RED else RED


def _norm_path(path: list) -> list[list[int]]:
    return [list(x) for x in path]


def _paths_match(a: list, b: list) -> bool:
    return _norm_path(a) == _norm_path(b)


def _enrich_legal_moves(
    board: list[list[int]], player: int
) -> list[dict[str, Any]]:
    """
    Return all legal moves with compute_move_facts injected — identical to
    what the real graph node (minimax_scorer) produces before calling
    proposal_agent.
    """
    legal = get_all_legal_moves(board, player)
    enriched: list[dict] = []
    for m in legal:
        facts = compute_move_facts(board, m, player)
        enriched.append({**m, "facts": facts})
    return enriched


def _tag_miss_reason(
    best_sym_move: dict[str, Any],
    board: list[list[int]],
    player: int,
) -> list[str]:
    reasons: list[str] = []
    path = best_sym_move.get("path", [])
    mtype = best_sym_move.get("type", "simple")
    captured = best_sym_move.get("captured", [])
    if mtype == "jump" or captured:
        reasons.append("forced_capture")
        if len(captured) >= 2:
            reasons.append("double_jump")
    if path:
        end_r = path[-1][0]
        start_r, start_c = path[0]
        piece = board[start_r][start_c]
        from checkers.engine.board import RED as _R, BLACK as _B, RED_KING, BLACK_KING
        if (player == _R and piece in (_R,) and end_r == 0) or \
           (player == _B and piece in (_B,) and end_r == 7):
            reasons.append("promotion")
    if not reasons:
        reasons.append("quiet_or_positional")
    return reasons


# ── per-position runner ───────────────────────────────────────────────────────

def run_coverage_position(
    pos: dict[str, Any],
    depth: int,
    near_margin: float,
) -> dict[str, Any]:
    pid = pos["position_id"]
    board = pos["board"]
    player = pos["side_to_move"]
    tags = pos.get("tags", [])

    # ── 1. Symbolic scoring (ground truth) ────────────────────────────────
    t_sym = time.perf_counter()
    clear_transposition_table()
    best_sym_move, best_sym_score, all_scored, sym_stats = search_root_all_scores(
        board=board,
        current_player=player,
        depth=depth,
        use_tt=True,
        use_tactical_extension=True,
        use_phase7a=True,
    )
    sym_time = round(time.perf_counter() - t_sym, 3)

    if not all_scored:
        return {
            "position_id": pid, "tags": tags, "category": pos.get("category"),
            "error": "terminal_no_legal_moves",
            "best_move_covered": False, "near_best_covered": False,
        }

    score_table = [
        {"rank": i + 1, "path": _norm_path(m["path"]), "score": round(s, 2)}
        for i, (m, s) in enumerate(all_scored)
    ]
    near_best_threshold = float(best_sym_score) - near_margin

    # ── 2. Build expansion_basis (state.legal_moves) ───────────────────────
    # This is identical to what the real graph path supplies.
    # proposal_agent uses this as the expansion_basis for index translation.
    enriched_legal = _enrich_legal_moves(board, player)

    # Inject minimax_score into facts so the proposal agent's internal sort
    # and safety-net work with real scores (same as real graph path).
    path_to_score: dict[str, float] = {
        str(_norm_path(m["path"])): float(s) for m, s in all_scored
    }
    for em in enriched_legal:
        key = str(_norm_path(em["path"]))
        em["facts"]["minimax_score"] = path_to_score.get(key, float("-inf"))

    # ── 3. Build CheckersState (all fields proposal_agent reads) ──────────
    # Fields read by build_proposal_prompts / proposal_agent:
    #   board, current_player               — always required
    #   legal_moves                         — expansion_basis for index map
    #   symbolic_scored_moves               — we leave None (no override pool)
    #   symbolic_best_score / symbolic_gap  — log lines only; default 0.0
    #   turn_number                         — printed in prompt; default 1
    #   feedback                            — retry feedback; None = clean run
    #   strategic_context                   — full context dict (required)
    state = CheckersState(
        board=board,
        current_player=player,
        legal_moves=enriched_legal,
        symbolic_scored_moves=[],     # empty = do NOT inject override pool
        symbolic_best_move=best_sym_move,
        symbolic_best_score=float(best_sym_score),
        symbolic_gap=0.0,
        turn_number=0,                # matches state default
        feedback=None,
        strategic_context={
            "game_phase": "MIDGAME",
            "score_state": "EQUAL",
            "strategic_priorities": ["PROMOTE", "CONTROL_CENTER", "INCREASE_MOBILITY"],
            "active_patterns": [],
            "winning_score": 0,
            "trends": {"material": 0, "center": 0},
        },
    )

    # ── 4. Verify index mapping (pre-call snapshot) ────────────────────────
    # build_proposal_prompts is deterministic for the same state.
    # We call it once now to snapshot moves_with_facts presentation order
    # and path_to_basis_idx for debugging.
    _, _, n_presentation, mwf_snapshot, ptb_snapshot, _mm_pin = build_proposal_prompts(state)

    index_debug: list[dict] = []
    for presentation_idx, (mv, _facts) in enumerate(mwf_snapshot):
        path_key = tuple(tuple(sq) for sq in mv["path"])
        basis_idx = ptb_snapshot.get(path_key)
        index_debug.append({
            "presentation_idx": presentation_idx,
            "basis_idx": basis_idx,
            "path": _norm_path(mv["path"]),
        })

    # ── 5. Run proposal agent ──────────────────────────────────────────────
    proposal_error: str | None = None
    fallback_used = False
    proposed_paths: list[list] = []
    proposed_basis_indices: list[int] = []
    proposal_time = 0.0
    raw_proposed_str = ""

    # Capture stdout from proposal_agent so we can detect the FALLBACK print
    # line reliably regardless of API key state or error type.
    captured_stdout = io.StringIO()
    try:
        tp0 = time.perf_counter()
        with _CaptureStdout(captured_stdout):
            result = proposal_agent(state)
        proposal_time = round(time.perf_counter() - tp0, 3)

        agent_stdout = captured_stdout.getvalue()
        # proposal_agent prints "[proposal_agent] FALLBACK selected indices:"
        # on every fallback path (API error, quota, network, parse failure).
        fallback_used = "FALLBACK selected indices" in agent_stdout

        raw_proposed_str = result.get("proposed_moves", "")

        # ── 6. Index mapping: parse basis indices from final JSON ──────────
        # proposal_agent already ran _translate_to_basis_indices internally.
        # The JSON contains *basis* (expansion_basis = state.legal_moves) indices.
        if isinstance(raw_proposed_str, str):
            try:
                parsed_prop = json.loads(raw_proposed_str)
                proposed_basis_indices = parsed_prop.get("selected_indices", [])
            except (json.JSONDecodeError, TypeError):
                proposal_error = f"json_parse_error: {raw_proposed_str[:120]}"
                proposed_basis_indices = []
        else:
            proposal_error = f"unexpected_type: {type(raw_proposed_str).__name__}"

        # Map basis indices to paths using state.legal_moves (the exact expansion_basis)
        for idx in proposed_basis_indices:
            if isinstance(idx, int) and 0 <= idx < len(enriched_legal):
                proposed_paths.append(_norm_path(enriched_legal[idx]["path"]))

        if not proposed_paths and not proposal_error:
            proposal_error = "no_paths_extracted"

    except Exception as exc:
        proposal_error = f"{type(exc).__name__}: {str(exc)[:200]}"
        fallback_used = True
        traceback.print_exc()

    # ── 7. Coverage metrics ────────────────────────────────────────────────
    best_sym_path = _norm_path(best_sym_move["path"]) if best_sym_move else []
    legal_count = len(all_scored)
    proposal_count = len(proposed_paths)

    # best_move_covered: is the symbolic-best path in any proposed path?
    best_move_covered = any(_paths_match(pp, best_sym_path) for pp in proposed_paths)

    # Score of each proposed path in the symbolic table
    proposed_scores: list[float] = []
    proposed_ranks: list[int] = []
    for pp in proposed_paths:
        for rank, (m, s) in enumerate(all_scored, start=1):
            if _paths_match(_norm_path(m["path"]), pp):
                proposed_scores.append(float(s))
                proposed_ranks.append(rank)
                break

    best_proposed_score = max(proposed_scores) if proposed_scores else None
    near_best_covered = (
        best_proposed_score is not None
        and best_proposed_score >= near_best_threshold
    )

    # score_gap = best_symbolic_score - best_proposed_score
    # Positive  → proposal missed a better move (bad)
    # Zero      → proposal included the best move (good)
    score_gap: float | None = (
        round(float(best_sym_score) - best_proposed_score, 2)
        if best_proposed_score is not None else None
    )

    # Rank of the best-scoring proposed move in the symbolic table (1 = best)
    rank_of_best_proposed: int | None = (
        min(proposed_ranks) if proposed_ranks else None
    )

    miss_reasons: list[str] = []
    if not best_move_covered and best_sym_move is not None:
        miss_reasons = _tag_miss_reason(best_sym_move, board, player)

    return {
        "position_id": pid,
        "category": pos.get("category"),
        "tags": tags,
        "side_to_move": "RED" if player == RED else "BLACK",
        "depth": depth,
        "legal_move_count": legal_count,
        "proposal_count": proposal_count,
        # ── LLM / fallback metadata ──
        "proposal_backend": GROQ_PROPOSAL_MODEL,
        "fallback_used": fallback_used,
        "proposal_error": proposal_error,
        # ── Ground truth ──
        "best_sym_path": best_sym_path,
        "best_sym_score": round(float(best_sym_score), 2),
        "near_best_margin": near_margin,
        "near_best_threshold": round(near_best_threshold, 2),
        "score_table_top5": score_table[:5],
        # ── Proposal results ──
        "proposed_basis_indices": proposed_basis_indices,
        "proposed_paths": proposed_paths,
        "proposed_scores": [round(s, 2) for s in proposed_scores],
        "best_proposed_score": (
            round(best_proposed_score, 2) if best_proposed_score is not None else None
        ),
        # ── Coverage ──
        # score_gap > 0 → missed a better move; == 0 → covered
        "score_gap": score_gap,
        "best_move_covered": best_move_covered,
        "near_best_covered": near_best_covered,
        "rank_of_best_proposed": rank_of_best_proposed,
        "miss_reasons": miss_reasons,
        # ── Index mapping snapshot (correctness audit) ──
        "index_mapping_snapshot": index_debug,
        # ── Timing ──
        "sym_nodes": sym_stats.nodes,
        "sym_time_s": sym_time,
        "proposal_time_s": proposal_time,
        "explanation": pos.get("explanation", ""),
        "known_failure": pos.get("known_failure", False),
    }


# ── aggregate summary ─────────────────────────────────────────────────────────

def _summarize(results: list[dict[str, Any]], near_margin: float) -> dict[str, Any]:
    valid = [r for r in results if "error" not in r]
    total = len(valid)
    if total == 0:
        return {"total": 0}

    # Exclude fallback positions from LLM coverage counts
    llm_valid = [r for r in valid if not r.get("fallback_used") and not r.get("proposal_error")]
    errored = sum(1 for r in valid if r.get("proposal_error"))
    fallbacks = sum(1 for r in valid if r.get("fallback_used"))

    best_covered_all = sum(1 for r in valid if r.get("best_move_covered"))
    near_covered_all = sum(1 for r in valid if r.get("near_best_covered"))
    best_covered_llm = sum(1 for r in llm_valid if r.get("best_move_covered"))
    near_covered_llm = sum(1 for r in llm_valid if r.get("near_best_covered"))

    gaps = [r["score_gap"] for r in valid if r.get("score_gap") is not None]
    avg_gap = round(sum(gaps) / len(gaps), 2) if gaps else None

    misses = [r for r in valid if not r.get("best_move_covered") and not r.get("proposal_error")]
    misses.sort(key=lambda r: r.get("score_gap") or 0, reverse=True)
    worst_misses = [
        {
            "position_id": r["position_id"],
            "score_gap": r.get("score_gap"),
            "best_sym_path": r.get("best_sym_path"),
            "proposed_paths": r.get("proposed_paths"),
            "miss_reasons": r.get("miss_reasons"),
            "fallback_used": r.get("fallback_used"),
            "tags": r.get("tags"),
        }
        for r in misses[:10]
    ]

    tag_stats: dict[str, dict] = {}
    for r in valid:
        for tag in r.get("tags", []):
            if tag not in tag_stats:
                tag_stats[tag] = {"total": 0, "best_covered": 0, "near_best_covered": 0, "fallbacks": 0}
            tag_stats[tag]["total"] += 1
            if r.get("best_move_covered"):
                tag_stats[tag]["best_covered"] += 1
            if r.get("near_best_covered"):
                tag_stats[tag]["near_best_covered"] += 1
            if r.get("fallback_used"):
                tag_stats[tag]["fallbacks"] += 1
    for tag in tag_stats:
        t = tag_stats[tag]
        t["best_pct"] = round(100 * t["best_covered"] / t["total"], 1)
        t["near_best_pct"] = round(100 * t["near_best_covered"] / t["total"], 1)

    best_pct_all = round(100 * best_covered_all / total, 1) if total else 0
    near_pct_all = round(100 * near_covered_all / total, 1) if total else 0
    best_pct_llm = round(100 * best_covered_llm / len(llm_valid), 1) if llm_valid else None
    near_pct_llm = round(100 * near_covered_llm / len(llm_valid), 1) if llm_valid else None

    pct_ref = best_pct_llm if best_pct_llm is not None else best_pct_all
    if pct_ref >= 90:
        recommendation = f"PASS — LLM best-move coverage {pct_ref}% ≥ 90%."
    elif (near_pct_llm or near_pct_all) >= 90:
        recommendation = (
            f"MARGINAL — near-best {near_pct_llm or near_pct_all}% acceptable but "
            f"best-move {pct_ref}% below 90%."
        )
    else:
        recommendation = (
            f"FAIL — best-move {pct_ref}%, near-best {near_pct_llm or near_pct_all}%. "
            f"Proposal needs improvement."
        )

    return {
        "total_positions": total,
        "errored_positions": errored,
        "fallback_positions": fallbacks,
        "llm_positions": len(llm_valid),
        "best_move_coverage_all_pct": best_pct_all,
        "near_best_coverage_all_pct": near_pct_all,
        "best_move_coverage_llm_pct": best_pct_llm,
        "near_best_coverage_llm_pct": near_pct_llm,
        "avg_score_gap": avg_gap,
        "worst_misses": worst_misses,
        "misses_by_tag": tag_stats,
        "recommendation": recommendation,
        "near_margin_used": near_margin,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Proposal coverage benchmark — does proposal include symbolic-best?"
    )
    p.add_argument("--depth", type=int, default=6)

    # Primary args (documented interface)
    p.add_argument("--near-margin", type=float, default=15.0, dest="near_margin",
                   help="Score margin for near-best coverage (default 15)")
    p.add_argument("--position-id", type=str, default=None, dest="position_id",
                   help="Run a single position by id")
    p.add_argument("--ids", type=str, nargs="+", default=None,
                   help="Run specific positions by id (space-separated, preserves declaration order)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of positions to run (applied after --ids filter)")
    # Aliases for backward compatibility
    p.add_argument("--margin", type=float, default=None,
                   help="Alias for --near-margin")
    p.add_argument("--id", type=str, default=None,
                   help="Alias for --position-id")

    p.add_argument("--out", type=str, default=None,
                   help="Output JSON path (default: logs/proposal_coverage_depth<D>.json)")
    args = p.parse_args(argv)

    # Resolve aliases
    near_margin = args.margin if args.margin is not None else args.near_margin
    position_id = args.id or args.position_id
    out_default = f"logs/proposal_coverage_depth{args.depth}.json"
    out_path = Path(args.out or out_default)

    positions = list(BENCHMARK_POSITIONS)
    # --ids: filter to named set (preserves BENCHMARK_POSITIONS declaration order)
    if args.ids:
        id_set = set(args.ids)
        positions = [pos for pos in positions if pos["position_id"] in id_set]
        missing = id_set - {pos["position_id"] for pos in positions}
        if missing:
            print(f"ERROR: unknown position ids: {sorted(missing)}")
            return 1
    elif position_id:
        # Single --position-id / --id
        positions = [pos for pos in positions if pos["position_id"] == position_id]
        if not positions:
            print(f"ERROR: no position with id={position_id!r}")
            return 1
    # --limit: cap after any id-filter
    if args.limit is not None:
        positions = positions[: args.limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    import hashlib as _hashlib

    api_key = os.environ.get("GROQ_API_KEY", "")
    key_loaded  = bool(api_key)
    key_prefix  = api_key[:4] if api_key else "N/A"
    key_len     = len(api_key)
    key_hash12  = _hashlib.sha256(api_key.encode()).hexdigest()[:12] if api_key else "N/A"

    print("=" * 70)
    print(f"PROPOSAL COVERAGE BENCHMARK  depth={args.depth}  margin={near_margin}")
    print(f"Positions    : {len(positions)}")
    print(f"LLM model    : {GROQ_PROPOSAL_MODEL}")
    print(f"── Key diagnostics ──────────────────────────────────────────────────")
    print(f"  env_path          : {env_path!r}")
    print(f"  GROQ_API_KEY set  : {key_loaded}")
    print(f"  key prefix (4c)   : {key_prefix}")
    print(f"  key length        : {key_len}")
    print(f"  key sha256[:12]   : {key_hash12}")
    if key_loaded and key_len < 40:
        print(f"  ⚠ key looks short — may be truncated in .env")
    if not key_loaded:
        print(f"  ⚠ No key found — all positions will FALLBACK")
    print(f"── ──────────────────────────────────────────────────────────────────")
    print(f"Output       : {out_path}")
    print("=" * 70)
    print(f"  {'✓=best ✗=miss ~=near-best':30}  {'gap':>8}  {'rank':>6}  props  id")
    print("  " + "-" * 68)


    results: list[dict] = []
    for pos in positions:
        r = run_coverage_position(pos, depth=args.depth, near_margin=near_margin)
        results.append(r)

        best_sym = "✓" if r.get("best_move_covered") else "✗"
        near_sym = "~" if r.get("near_best_covered") else " "
        gap_str  = f"gap={r['score_gap']:+.1f}" if r.get("score_gap") is not None else "gap= N/A"
        rank_str = f"r={r.get('rank_of_best_proposed', '?')}"
        props    = f"{r.get('proposal_count', 0)}/{r.get('legal_move_count', 0)}"
        fb_tag   = " [FALLBACK]" if r.get("fallback_used") else ""
        err_tag  = f" [ERR]" if r.get("proposal_error") and not r.get("fallback_used") else ""
        print(
            f"  {best_sym}{near_sym}  {r['position_id']:<42} "
            f"{gap_str:<12} {rank_str:<7} {props:>6}{fb_tag}{err_tag}"
        )

    summary = _summarize(results, near_margin=near_margin)

    print()
    print("── SUMMARY ──────────────────────────────────────────────────────────")
    print(f"  Total positions         : {summary['total_positions']}")
    print(f"  Fallback (no LLM)       : {summary['fallback_positions']}")
    print(f"  Errored                 : {summary['errored_positions']}")
    print(f"  LLM positions           : {summary['llm_positions']}")
    print(f"  Best-move cov (all)     : {summary['best_move_coverage_all_pct']}%")
    print(f"  Near-best cov (all)     : {summary['near_best_coverage_all_pct']}%")
    if summary["best_move_coverage_llm_pct"] is not None:
        print(f"  Best-move cov (LLM)     : {summary['best_move_coverage_llm_pct']}%")
        print(f"  Near-best cov (LLM)     : {summary['near_best_coverage_llm_pct']}%")
    print(f"  Avg score gap           : {summary['avg_score_gap']}")
    print(f"  → {summary['recommendation']}")
    if summary.get("worst_misses"):
        print()
        print("  Worst misses (LLM):")
        for m in summary["worst_misses"]:
            fb = " [FB]" if m.get("fallback_used") else ""
            print(f"    {m['position_id']:<42} gap={m['score_gap']}  {m['miss_reasons']}{fb}")
    if summary.get("misses_by_tag"):
        print()
        print("  Coverage by tag:")
        for tag, ts in sorted(summary["misses_by_tag"].items()):
            print(
                f"    {tag:<28} best={ts['best_pct']:>5}%  "
                f"near={ts['near_best_pct']:>5}%  "
                f"({ts['best_covered']}/{ts['total']})  fb={ts['fallbacks']}"
            )

    report = {
        "meta": {
            "depth": args.depth,
            "near_margin": near_margin,
            "position_count": len(positions),
            "llm_model": GROQ_PROPOSAL_MODEL,
            "api_key_set": key_loaded,
            "key_prefix": key_prefix,
            "key_length": key_len,
            "key_sha256_12": key_hash12,
            "env_path": env_path,
            "output": str(out_path),
        },
        "summary": summary,
        "results": results,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
