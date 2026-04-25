# Proposal Coverage Baseline

**Date:** 2026-04-25
**Model:** `llama-3.1-8b-instant`
**Depth:** 6
**Near Margin:** 15.0
**Source JSON:** `logs/proposal_coverage_full19_depth6_real_llm.json`

## Summary Metrics (Current - After correcting pos_mandatory_capture)
- **Total positions evaluated:** 19
- **LLM proposals used:** 19 (0 fallbacks)
- **Best-move coverage:** 89.5% (17/19)
- **Near-best coverage:** 100.0% (19/19)
- **Average score gap:** 0.0

## Historical Metrics (Before correcting pos_mandatory_capture)
- **Best-move coverage:** 94.7% (18/19)
- **Near-best coverage:** 94.7% (18/19)
- **Average score gap:** 1.16

## Notes
- The proposal benchmark strictly measures **shortlist coverage** (whether the LLM includes the symbolic-best move in its output).
- It does **not** prove final gameplay strength. 
- The ranker, override logic, and minimax engines still determine the final move selection downstream.
- The 1 miss originally logged (`pos_mandatory_capture`) was due to a flawed benchmark configuration (the board lacked a true capture). After correcting the board to feature a true forced capture, the LLM successfully included it.
- **Current Status:** In the updated run, the LLM included a mathematically optimal move (score gap = 0.0) in **100%** of the 19 adversarial positions. Proposal coverage is strong on the current benchmark, but further real-game trace testing is needed before claiming general robustness.
