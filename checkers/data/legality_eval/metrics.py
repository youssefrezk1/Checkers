"""
checkers/data/legality_eval/metrics.py
=======================================
Aggregate per-scenario results using result_type as the primary discriminator.

Denominator hierarchy
---------------------
  n_total        all records (including api_failure)
  n_api_success  records where api_success=True
  n_evaluated    records where result_type in ("legal", "illegal")

Rates and their denominators
-----------------------------
  api_success_rate          n_api_success / n_total
  api_failure_rate          n_api_failure / n_total
  parse_success_rate        n_parse_success / n_api_success
  invalid_format_rate       (n_api_success - n_parse_success) / n_api_success
  legal_move_rate           n_legal / n_evaluated         ← NEVER includes api_failure
  illegal_move_rate         n_illegal / n_evaluated       ← NEVER includes api_failure
  wrong_direction_rate      count / n_evaluated
  mandatory_capture_viol_rate count / n_evaluated
  multi_jump_incomplete_rate  count / n_evaluated
  illegal_type_counts       raw counts, evaluated records only
  category_accuracy         legal_move_rate per category, evaluated records only
  difficulty_accuracy       legal_move_rate per difficulty, evaluated records only
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

# Lazy import to avoid circular deps — salvage.py imports nothing from metrics
def _get_aggregate_salvage():
    from checkers.data.legality_eval.salvage import aggregate_salvage
    return aggregate_salvage


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den > 0 else 0.0


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate a list of per-scenario result dicts into pilot-level metrics.

    Required fields per record
    --------------------------
      result_type              "api_failure" | "parse_failure" | "legal" | "illegal"
      api_success              bool
      parse_success            bool
      legal                    bool
      rate_limit_retry_count   int
      wrong_direction          str|None
      mandatory_violation      bool
      multi_jump_incomplete    bool
      illegal_move_type        str
      category                 str
      difficulty               str
    """
    n = len(results)
    if n == 0:
        return {}

    # ── partition by result_type ──────────────────────────────────────────────
    api_failures   = [r for r in results if r["result_type"] == "api_failure"]
    parse_failures = [r for r in results if r["result_type"] == "parse_failure"]
    evaluated      = [r for r in results if r["result_type"] in ("legal", "illegal")]
    legal_recs     = [r for r in evaluated if r["result_type"] == "legal"]
    illegal_recs   = [r for r in evaluated if r["result_type"] == "illegal"]

    n_api_failure  = len(api_failures)
    n_api_success  = n - n_api_failure        # api_success = NOT api_failure
    n_parse_fail   = len(parse_failures)
    n_parse_ok     = n_api_success - n_parse_fail   # = len(evaluated)
    n_evaluated    = len(evaluated)
    n_legal        = len(legal_recs)
    n_illegal      = len(illegal_recs)

    # ── API totals ────────────────────────────────────────────────────────────
    total_rate_limit_retries = sum(
        r.get("rate_limit_retry_count", 0) for r in results
    )

    # ── sub-rates among evaluated only ────────────────────────────────────────
    wrong_dir_count = sum(
        1 for r in evaluated if r.get("wrong_direction") is not None
    )
    mand_viol_count = sum(1 for r in evaluated if r.get("mandatory_violation"))
    multi_inc_count = sum(1 for r in evaluated if r.get("multi_jump_incomplete"))

    # ── illegal type breakdown (evaluated only) ───────────────────────────────
    illegal_type_counts = Counter(
        r["illegal_move_type"] for r in illegal_recs if r.get("illegal_move_type")
    )

    # ── category accuracy (legal_move_rate per category, evaluated only) ──────
    cat_results: dict[str, list[bool]] = defaultdict(list)
    for r in evaluated:
        cat_results[r.get("category", "unknown")].append(r["result_type"] == "legal")
    category_accuracy = {
        cat: _rate(sum(vals), len(vals))
        for cat, vals in cat_results.items()
    }

    # ── difficulty accuracy ───────────────────────────────────────────────────
    diff_results: dict[str, list[bool]] = defaultdict(list)
    for r in evaluated:
        diff_results[r.get("difficulty", "unknown")].append(r["result_type"] == "legal")
    difficulty_accuracy = {
        diff: _rate(sum(vals), len(vals))
        for diff, vals in diff_results.items()
    }

    # ── side-level analysis (evaluated records only) ─────────────────────────
    # Denominators match the global rule: api_failure excluded.
    sides = ["RED", "BLACK"]
    side_evaluated: dict[str, list] = {s: [] for s in sides}
    for r in evaluated:
        s = r.get("side_to_move", "UNKNOWN")
        if s in side_evaluated:
            side_evaluated[s].append(r)

    side_metrics: dict[str, dict] = {}
    for s, recs in side_evaluated.items():
        n_ev   = len(recs)
        n_leg  = sum(1 for r in recs if r["result_type"] == "legal")
        n_ill  = sum(1 for r in recs if r["result_type"] == "illegal")
        w_dir  = sum(1 for r in recs if r.get("wrong_direction") is not None)
        m_viol = sum(1 for r in recs if r.get("mandatory_violation"))
        m_inc  = sum(1 for r in recs if r.get("multi_jump_incomplete"))
        side_metrics[s] = {
            "n_evaluated":               n_ev,
            "n_legal":                   n_leg,
            "n_illegal":                 n_ill,
            "legal_move_rate":           _rate(n_leg,   n_ev),
            "illegal_move_rate":         _rate(n_ill,   n_ev),
            "wrong_direction_rate":      _rate(w_dir,   n_ev),
            "mandatory_capture_viol_rate": _rate(m_viol, n_ev),
            "multi_jump_incomplete_rate":  _rate(m_inc,  n_ev),
            "illegal_type_counts":       dict(Counter(
                r["illegal_move_type"]
                for r in recs
                if r["result_type"] == "illegal" and r.get("illegal_move_type")
            )),
        }

    # ── category × side legal_move_rate (evaluated only) ─────────────────────
    cat_side_results: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in evaluated:
        cat = r.get("category", "unknown")
        s   = r.get("side_to_move", "UNKNOWN")
        cat_side_results[cat][s].append(r["result_type"] == "legal")
    category_side_accuracy: dict[str, dict[str, float]] = {
        cat: {
            s: _rate(sum(vals), len(vals))
            for s, vals in side_data.items()
        }
        for cat, side_data in cat_side_results.items()
    }

    # ── salvage analysis (parse_failure records only) ─────────────────────────
    salvage_metrics = _get_aggregate_salvage()(results, n)

    return {
        # counts
        "n_scenarios":               n,
        "n_api_success":             n_api_success,
        "n_api_failure":             n_api_failure,
        "n_parse_success":           n_parse_ok,
        "n_evaluated":               n_evaluated,
        "n_legal":                   n_legal,
        "n_illegal":                 n_illegal,
        # API-level rates  (denominator: n_total)
        "api_success_rate":          _rate(n_api_success, n),
        "api_failure_rate":          _rate(n_api_failure, n),
        "total_api_failures":        n_api_failure,
        "total_rate_limit_retries":  total_rate_limit_retries,
        # parse-level rates  (denominator: n_api_success)
        "parse_success_rate":        _rate(n_parse_ok,  n_api_success),
        "invalid_format_rate":       _rate(n_parse_fail, n_api_success),
        # eval-level rates   (denominator: n_evaluated — EXCLUDES api_failure)
        "legal_move_rate":           _rate(n_legal,         n_evaluated),
        "illegal_move_rate":         _rate(n_illegal,        n_evaluated),
        "wrong_direction_rate":      _rate(wrong_dir_count,  n_evaluated),
        "mandatory_capture_viol_rate": _rate(mand_viol_count, n_evaluated),
        "multi_jump_incomplete_rate":  _rate(multi_inc_count,  n_evaluated),
        # breakdowns
        "illegal_type_counts":         dict(illegal_type_counts),
        "category_accuracy":           category_accuracy,
        "difficulty_accuracy":         difficulty_accuracy,
        # side-level analysis (new)
        "side_metrics":               side_metrics,
        "category_side_accuracy":     category_side_accuracy,
        # salvage analysis (secondary — does NOT affect legal_move_rate)
        "salvage":                     salvage_metrics,
    }


def format_report(
    pilot_name: str,
    baseline_metrics: dict[str, dict[str, Any]],
    n_scenarios: int,
    run_label: str = "UNKNOWN",
) -> str:
    bl    = list(baseline_metrics.keys())
    col_w = 20

    def _row(label: str, key: str) -> str:
        r = f"  {label:<38}"
        for b in bl:
            val = baseline_metrics[b].get(key, "n/a")
            r  += f"  {str(val):<{col_w}}"
        return r

    hdr = f"  {'Metric':<38}" + "".join(f"  {b:<{col_w}}" for b in bl)

    lines = [
        "=" * 72,
        f"LEGALITY-STRESS PILOT REPORT — {pilot_name}",
        f"Scenarios per baseline: {n_scenarios}",
        f"Run label: {run_label}",
        "=" * 72,
        "",
        "--- API Health ---",
        hdr,
    ]
    for key, label in [
        ("api_success_rate",    "API success rate"),
        ("api_failure_rate",    "API failure rate"),
        ("total_api_failures",  "Total API failures"),
        ("total_rate_limit_retries", "Total rate-limit retries"),
    ]:
        lines.append(_row(label, key))

    lines += ["", "--- Parse Quality (among API successes) ---", hdr]
    for key, label in [
        ("parse_success_rate",  "Parse success rate"),
        ("invalid_format_rate", "Invalid format rate"),
    ]:
        lines.append(_row(label, key))

    lines += ["", "--- Legality (among successfully parsed responses) ---", hdr]
    for key, label in [
        ("legal_move_rate",             "Legal move rate"),
        ("illegal_move_rate",           "Illegal move rate"),
        ("wrong_direction_rate",        "Wrong direction rate"),
        ("mandatory_capture_viol_rate", "Mandatory capture violation"),
        ("multi_jump_incomplete_rate",  "Multi-jump incomplete"),
    ]:
        lines.append(_row(label, key))

    # Category accuracy
    all_cats = sorted({
        c for m in baseline_metrics.values()
        for c in m.get("category_accuracy", {})
    })
    if all_cats:
        lines += ["", "--- Category Accuracy (legal_move_rate, evaluated only) ---", hdr]
        for cat in all_cats:
            row = f"  {cat:<38}"
            for b in bl:
                val = baseline_metrics[b].get("category_accuracy", {}).get(cat, "n/a")
                row += f"  {str(val):<{col_w}}"
            lines.append(row)

    # Difficulty accuracy
    all_diffs = sorted({
        d for m in baseline_metrics.values()
        for d in m.get("difficulty_accuracy", {})
    })
    if all_diffs:
        lines += ["", "--- Difficulty Accuracy (legal_move_rate, evaluated only) ---", hdr]
        for d in all_diffs:
            row = f"  {d:<38}"
            for b in bl:
                val = baseline_metrics[b].get("difficulty_accuracy", {}).get(d, "n/a")
                row += f"  {str(val):<{col_w}}"
            lines.append(row)

    # Illegal type breakdown
    all_types = sorted({
        t for m in baseline_metrics.values()
        for t in m.get("illegal_type_counts", {})
    })
    if all_types:
        lines += ["", "--- Illegal Move Type Breakdown (counts, evaluated only) ---", hdr]
        for t in all_types:
            row = f"  {t:<38}"
            for b in bl:
                val = baseline_metrics[b].get("illegal_type_counts", {}).get(t, 0)
                row += f"  {str(val):<{col_w}}"
            lines.append(row)

    # ── Side-to-move analysis ─────────────────────────────────────────────────
    sides = ["RED", "BLACK"]
    # Collect all side keys that actually appear across all baselines
    all_sides = sorted({
        s
        for m in baseline_metrics.values()
        for s in m.get("side_metrics", {})
    })
    if all_sides:
        lines += ["", "--- Side-to-Move Analysis (evaluated records only) ---"]
        for s in all_sides:
            lines.append(f"  [{s}]")
            side_hdr = f"    {'Metric':<36}" + "".join(f"  {b:<{col_w}}" for b in bl)
            lines.append(side_hdr)
            for key, label in [
                ("n_evaluated",               "  evaluated responses"),
                ("n_legal",                   "  legal responses"),
                ("n_illegal",                 "  illegal responses"),
                ("legal_move_rate",           "  legal_move_rate"),
                ("illegal_move_rate",         "  illegal_move_rate"),
                ("wrong_direction_rate",      "  wrong_direction_rate"),
                ("mandatory_capture_viol_rate", "  mandatory_capture_viol_rate"),
                ("multi_jump_incomplete_rate",  "  multi_jump_incomplete_rate"),
            ]:
                row = f"    {label:<36}"
                for b in bl:
                    val = baseline_metrics[b].get("side_metrics", {}).get(s, {}).get(key, "n/a")
                    row += f"  {str(val):<{col_w}}"
                lines.append(row)
            # Per-side illegal type counts
            all_itypes = sorted({
                t
                for m in baseline_metrics.values()
                for t in m.get("side_metrics", {}).get(s, {}).get("illegal_type_counts", {})
            })
            if all_itypes:
                lines.append(f"    {'  illegal type counts':<36}" +
                             "".join(f"  {'':<{col_w}}" for _ in bl))
                for t in all_itypes:
                    row = f"    {'    ' + t:<36}"
                    for b in bl:
                        val = (baseline_metrics[b]
                               .get("side_metrics", {}).get(s, {})
                               .get("illegal_type_counts", {}).get(t, 0))
                        row += f"  {str(val):<{col_w}}"
                    lines.append(row)

    # ── Category × side legal_move_rate ──────────────────────────────────────
    all_cats_cs = sorted({
        c
        for m in baseline_metrics.values()
        for c in m.get("category_side_accuracy", {})
    })
    if all_cats_cs and all_sides:
        for b in bl:
            lines += [
                "",
                f"--- Category x Side Legal Move Rate [{b}] ---",
                f"  {'Category':<32}  " + "  ".join(f"{s:<10}" for s in all_sides),
            ]
            for cat in all_cats_cs:
                row = f"  {cat:<32}  "
                for s in all_sides:
                    val = (baseline_metrics[b]
                           .get("category_side_accuracy", {}).get(cat, {}).get(s, "n/a"))
                    row += f"{str(val):<10}  "
                lines.append(row)

    # ── Raw Output Salvage Analysis ───────────────────────────────────────────
    has_salvage = any(
        bool(baseline_metrics[b].get("salvage"))
        for b in bl
    )
    if has_salvage:
        lines += ["", "--- Raw Output Salvage Analysis (parse_failure records only) ---"]
        lines += [
            "  NOTE: result_type and legal_move_rate are unchanged.",
            "  Salvage checks whether a usable move existed in the failed output.",
            hdr,
        ]
        for key, label in [
            ("parse_failure_count",           "Parse failure count"),
            ("salvage_success_count",          "Salvage succeeded (move recovered)"),
            ("salvage_legal_count",            "Salvaged move was legal"),
            ("salvage_illegal_count",          "Salvaged move was illegal"),
            ("no_usable_output_count",         "No usable output at all"),
            ("adjusted_e2e_legal_if_salvaged", "Adjusted end-to-end legal rate"),
            ("adjusted_e2e_unusable",          "Adjusted unusable/illegal rate"),
        ]:
            row = f"  {label:<38}"
            for b in bl:
                val = baseline_metrics[b].get("salvage", {}).get(key, "n/a")
                row += f"  {str(val):<{col_w}}"
            lines.append(row)

    lines += ["", "=" * 72]
    return "\n".join(lines)
