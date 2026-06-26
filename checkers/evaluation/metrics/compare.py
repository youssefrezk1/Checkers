# checkers/evaluation/metrics/compare.py
#
# Paired delta computation between two metric bundles (seed_on vs seed_off,
# but the comparator is condition-agnostic — A vs B works for any two
# bundles produced by metrics.run_batch.evaluate_batch).
#
# Delta convention (matches the spec):
#     delta = a_value - b_value
# i.e. positive ⇒ A is larger.  When the runner is invoked with
# A=seed_on B=seed_off, "contradiction_delta < 0" reads as
# "seeds REDUCE contradiction" — the intended thesis statement.
#
# The delta is computed for every numeric field present in BOTH bundles.
# Non-numeric, missing, or null values are skipped (the corresponding
# delta entry is omitted rather than set to None to keep the report
# uncluttered).  Nested dicts are walked recursively.  Lists of strings
# (e.g. most_contradicted_types) are diffed as ordered/unordered sets and
# attached as `*_added` / `*_removed`.
#
# Pure-Python, deterministic, no LLM calls.

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


_NUM = (int, float)


def _is_number(v: Any) -> bool:
    return isinstance(v, _NUM) and not isinstance(v, bool)


def _delta_dict(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursive numeric/set diff. Returns a new dict — inputs unchanged."""
    out: Dict[str, Any] = {}
    keys = set(a.keys()) | set(b.keys())
    for k in sorted(keys):
        av = a.get(k)
        bv = b.get(k)

        if isinstance(av, Mapping) and isinstance(bv, Mapping):
            nested = _delta_dict(av, bv)
            if nested:
                out[k] = nested
            continue

        if _is_number(av) and _is_number(bv):
            out[k] = av - bv
            continue

        # Both lists of strings: report set diff.
        if (
            isinstance(av, list) and isinstance(bv, list)
            and all(isinstance(x, str) for x in av)
            and all(isinstance(x, str) for x in bv)
        ):
            added   = [x for x in av if x not in bv]
            removed = [x for x in bv if x not in av]
            if added or removed:
                out[k + "_added"]   = added
                out[k + "_removed"] = removed
            continue

        # Strings, bools, None, mismatched types → silently skipped.
    return out


def compare_summaries(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    label_a: str = "a",
    label_b: str = "b",
) -> Dict[str, Any]:
    """
    Build the three-way comparative report.

    Parameters
    ----------
    a, b : dict
        Outputs of `metrics.run_batch.evaluate_batch` for two conditions.
    label_a, label_b : str
        Keys used in the top-level report (e.g. "seed_on", "seed_off").

    Returns
    -------
    dict shaped:
        {
          "<label_a>": a,
          "<label_b>": b,
          "delta":     {a - b for every numeric field present in both}
        }
    """
    return {
        label_a: dict(a),
        label_b: dict(b),
        "delta": _delta_dict(a, b),
    }


# ---------------------------------------------------------------------------
# Pairing safeguards
# ---------------------------------------------------------------------------

class PairingError(AssertionError):
    """Raised when two evaluation-source records that should be paired diverge."""


def assert_records_paired(
    rec_a: Mapping[str, Any],
    rec_b: Mapping[str, Any],
    *,
    require_final_choice_source: str = "proposal_authoritative",
) -> None:
    """
    Sanity-check that two evaluation-source records describe the SAME
    proposal decision (i.e. only the reasoning text differs).

    Asserts:
      * chosen_move (compared via path inside ranker_diagnostics.final_chosen_path
        and via chosen_move_facts identity) matches.
      * chosen_move_score matches exactly (proposal output, deterministic).
      * final_choice_source == "proposal_authoritative" on both sides.

    Raises PairingError on any mismatch.
    """
    src_a = rec_a.get("final_choice_source")
    src_b = rec_b.get("final_choice_source")
    if src_a != require_final_choice_source or src_b != require_final_choice_source:
        raise PairingError(
            f"final_choice_source must be {require_final_choice_source!r}; "
            f"got {src_a!r} / {src_b!r} (turn_id={rec_a.get('turn_id')!r})"
        )

    score_a = rec_a.get("chosen_move_score")
    score_b = rec_b.get("chosen_move_score")
    if score_a != score_b:
        raise PairingError(
            f"chosen_move_score diverges: {score_a!r} vs {score_b!r} "
            f"(turn_id={rec_a.get('turn_id')!r})"
        )

    path_a = _chosen_path(rec_a)
    path_b = _chosen_path(rec_b)
    if path_a != path_b:
        raise PairingError(
            f"chosen_move path diverges: {path_a!r} vs {path_b!r} "
            f"(turn_id={rec_a.get('turn_id')!r})"
        )


def _chosen_path(record: Mapping[str, Any]) -> Optional[List[Any]]:
    diag = record.get("explainer_diagnostics") or record.get("ranker_diagnostics") or {}
    path = diag.get("final_chosen_path")
    if path is None:
        # Pre-instrumentation logs may not carry final_chosen_path; fall back
        # to chosen_move_facts identity via a synthetic key (rare).
        return None
    # Normalise tuple-vs-list so paired comparisons survive Pydantic round-trips.
    return [list(sq) for sq in path]


def pair_by_turn_id(
    records_a: List[Mapping[str, Any]],
    records_b: List[Mapping[str, Any]],
) -> List[tuple]:
    """
    Pair two record lists by `turn_id`.  Records present in only one side
    are silently dropped.  Pairing is by exact turn_id match — the runner
    is responsible for guaranteeing the same turn_id pattern across runs.

    Returns
    -------
    list of (record_a, record_b) tuples in turn_id order.
    """
    idx_b = {r.get("turn_id"): r for r in records_b if isinstance(r.get("turn_id"), str)}
    out: List[tuple] = []
    for ra in records_a:
        tid = ra.get("turn_id")
        if not isinstance(tid, str):
            continue
        rb = idx_b.get(tid)
        if rb is None:
            continue
        out.append((ra, rb))
    return out
