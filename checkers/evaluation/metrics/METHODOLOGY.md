# Evaluation-metrics methodology notes

This file documents the **scientifically required usage** of the metrics in
this package.  Code-level docstrings explain *what* each module computes;
this file explains *how its outputs may and may not be cited* in the
thesis or any external report.

The rules below were derived from the post-fix forensic audit
(`audit + post-Python-fallback removal`) and the semantic-layer audit on
`logs/semantic_smoke/`.  Any future contributor — human or AI — who edits
the metric pipeline is expected to read this file first.

## 1. Single source of truth for contradictions

* The runtime refinement loop and every metric module MUST consult
  `checkers.evaluation.unified_verifier.verify_all`.
* Going through `extract_claims + verify_claims` directly bypasses the
  numeric (E.3), schema-leak (E.4), forbidden-vocab, mobility-reduction,
  safe-reply, and absence-claim layers and silently undercounts
  contradictions by ~8 percentage points.
* The CI test `checkers/tests/test_unified_verifier_invariant.py` enforces
  this — runtime ↔ evaluator ↔ metric-layer must all return the same
  contradiction set for every real record.

## 2. Pooled metric numbers are NEVER publishable

`run_batch.evaluate_batch` returns a single flat dict.  If you invoke it
on a directory that contains **both** `seed_on/` and `seed_off/` records
(e.g. by pointing it at `logs/<run>/evaluation_source/`), every aggregate
in the resulting `report.json` is a pooled mixture across conditions.

Pooled means are misleading whenever the two conditions have different
distributions — which is exactly the experimental regime the project
studies.  Concretely, on the semantic-smoke batch:

| Metric | Pooled (uninformative) | seed_on | seed_off |
|---|---:|---:|---:|
| `semantic.bertscore_f1.mean` | 0.523 | 0.817 | 0.436 |
| `factuality.post_repair_contradiction_rate_micro` | mixed | ≈ 0.005 | ≈ 0.205 |

**Rule:** any number quoted in the thesis or any external report MUST come
from comparative mode:

```bash
python -m checkers.evaluation.metrics.run_batch \
    --compare logs/<run>/evaluation_source/seed_on \
              logs/<run>/evaluation_source/seed_off \
    --out logs/<run>/report.json
```

(or, equivalently, from `run_ablation.py`, which always invokes
`compare_summaries` internally.)  The resulting `report.json` has the
shape `{seed_on: {...}, seed_off: {...}, delta: {...}}`.

The pooled-form `evaluate_batch` output is acceptable only for
operational debugging (e.g. spot-checking a single game's logs) and must
not appear in any external artefact.

## 3. Semantic metrics MUST be paired with factuality

`semantic.bertscore_f1` (and, when present, `semantic.bleurt`) measure
*how much surface text was preserved* between the pre-refinement and
post-refinement reasoning.  Read in isolation they cannot distinguish
three very different repair behaviours:

| BERTScore F1 | Contradiction resolved? | Reading |
|---|---|---|
| high (≥ 0.85) | yes | clean targeted repair |
| high (≥ 0.85) | no  | refinement barely touched the text — still contradicts |
| low  (< 0.70) | yes | invasive but successful rewrite |
| low  (< 0.70) | no  | refinement ran wild and still failed |

**Rule:** every published semantic figure must be accompanied by the
matching factuality figure for the same set of turns
(`factuality.post_repair_contradiction_rate_micro`,
`factuality.repair_effectiveness`, or the per-turn `contradiction_resolved`
flag from `pre_post_repair.PrePostRepairTurn`).

The 2×2 above is the minimum interpretive framework.  A bar chart that
shows only BERTScore deltas is misleading by construction.

## 4. Semantic gating coupling (filter only, never score)

The semantic module reads `reasoning_refinement_retry_count` and
`reasoning_contradiction_detected` from `ranker_diagnostics` to decide
*which turns to score*.  The score values themselves consult only the
two text strings.  This is documented filter-coupling, not score-coupling,
and is the correct behaviour (the alternative — gate on `pre != post`
alone — would count any whitespace difference as a refinement event).
Cite it in the methodology section of any external write-up.

## 5. BLEURT is a secondary, out-of-domain metric

BLEURT-20 was trained on WMT translations and news text, not on chess
explanations.  Its scores on this domain are noisier and less
interpretable than BERTScore's.  BLEURT is included for cross-metric
sanity-checking only; the headline preservation number must come from
BERTScore.  If BLEURT and BERTScore deltas disagree on direction,
disclose both and prefer BERTScore as the headline figure.

BLEURT is an optional dependency.  The pipeline runs without it; the
report's `semantic.bleurt` block degrades to a structured `error` shape
when the library is uninstalled.

## 6. Move-selection invariants

`run_ablation` already asserts that paired seed_on / seed_off turns share
`chosen_move`, `chosen_move_score`, and
`final_choice_source == "proposal_authoritative"`.  Any divergence aborts
the ablation.  Do not weaken these checks.

## 7. What NOT to compute

* Setup B — comparing seed_on paragraph text vs seed_off paragraph text
  directly with a similarity metric — was rejected during the planning
  audit because it measures *vocabulary divergence*, not quality.  If
  anyone re-adds it, frame the output as descriptive ("seeds change ~30 %
  of the surface vocabulary"), never as a quality claim.
* Setup C — comparing reasoning text against the seed list — is circular
  with the factuality verifier (which itself checks seed coverage) and is
  forbidden.
