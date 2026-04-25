# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bachelor thesis project: an LLM-augmented American Checkers (8x8) engine. A LangGraph state machine orchestrates a pipeline where symbolic minimax search proposes and scores moves, LLM agents (Groq for proposals, Mistral for ranking) filter and select the final move, and symbolic nodes enforce legality at every step. The thesis evaluates how well LLMs can make strategic decisions when supported by symbolic move scoring.

## Commands

### Run a full game (RED = LangGraph pipeline, BLACK = human or minimax opponent)
```
venv/bin/python3 run_full_trace.py
venv/bin/python3 run_full_trace.py --mode=auto   # both sides automated
```

### Run tests
```
venv/bin/python3 -m pytest checkers/tests/ -v                          # all tests
venv/bin/python3 -m pytest checkers/tests/test_minimax_core.py -v      # single file
venv/bin/python3 -m pytest checkers/tests/test_minimax_core.py::test_name -v  # single test
venv/bin/python3 -m pytest test_proposal_edge_cases.py -v              # top-level test files
```

### Batch evaluation (ranker quality metrics)
```
venv/bin/python3 evaluate_ranker_batch.py --games 1 --max-plies 40 --seed 42 --out logs/quick_eval.json
venv/bin/python3 evaluate_ranker_batch.py --games 5 --max-plies 120 --seed 42 --out logs/final_eval.json
```

### Ablation evaluation (minimax-only vs full pipeline)
```
venv/bin/python3 -m checkers.eval.run_ablation_eval --games 3 --depth 3 --out logs/ablation.json
```

## Architecture

### LangGraph Hub-and-Spoke Pipeline

All nodes return to the **orchestrator** (pure passthrough); routing logic lives in `orchestrator_routing` inside `checkers/graph/graph.py`. State is a Pydantic model (`CheckersState` in `checkers/state/state.py`); nodes return dicts of changed fields that LangGraph merges.

**Per-turn flow** (in order):
1. **inter_turn_memory** — computes strategic context (game phase, score state, patterns, priorities) from board facts and a 5-turn sliding window
2. **symbolic_decision** — scores ALL legal moves via depth-3 negamax, sorts best-first, stores in `symbolic_scored_moves`. Always proceeds to proposal (never bypasses LLM)
3. **proposal_agent** — calls Groq (llama-3.1-8b) to shortlist 3-5 candidate indices from the sorted move list. Applies symbolic safety-net injection and role-pinning post-LLM
4. **format_checker** — parses LLM JSON output, expands indices to engine move dicts, enforces count rules. Retries back to proposal on failure
5. **validator** — checks proposed moves against engine legal moves, enriches with `move_facts`, deduplicates via Zobrist hashing. Retries back to proposal if all illegal
6. **minimax_scorer** — attaches minimax scores to shortlisted candidates (reuses symbolic_decision cache when available)
7. **ranker_agent** — calls Mistral (mistral-small) to pick the final move from scored candidates with strategic reasoning. Has its own safety filter and retry logic
8. **state_manager** — applies chosen move via `rules.apply_move`, switches player, clears per-turn fields, appends move history
9. **win_condition** — checks for win (no pieces/no moves) and draw by 3-fold repetition (Zobrist)
10. **logger_node** — writes JSONL game log, prints turn summary

### Engine Layer (`checkers/engine/`)
- `board.py` — piece constants (EMPTY=0, RED=1, BLACK=2, RED_KING=3, BLACK_KING=4), 8x8 board. RED starts rows 5-7, moves up; BLACK starts rows 0-2, moves down
- `rules.py` — legal move generation with mandatory capture rule, `apply_move` (immutable — returns new board)
- `evaluation.py` — multi-term symbolic evaluator (~18 terms: material, mobility, center, promotion, vulnerability, king features, confinement, column centrality, etc.)
- `minimax.py` — compatibility wrapper; `score_move_with_minimax` is the per-move scoring API
- `move_facts.py` — per-move tactical annotations (captures, safety, promotion, counterplay, etc.)
- `zobrist.py` — position hashing for transposition table and repetition detection

### Search (`checkers/search/minimax_core.py`)
Alpha-beta negamax with transposition table, iterative deepening, tactical extensions (captures extended up to 2 extra plies), and leaf tension penalty.

### Key Environment Variables
- `GROQ_API_KEY` — proposal agent LLM
- `MISTRAL_API_KEY` — ranker agent LLM
- `MINIMAX_DEPTH` (default 3) — shared search depth for symbolic_decision, minimax_scorer, and minimax wrapper
- `MINIMAX_ENABLED` (default true) — set false for ablation
- `RANKER_BACKEND` — fixed to "mistral"
- `DEBUG_ALL_LEGAL_TO_RANKER` — bypasses proposal narrowing, sends all legal moves to ranker

### Move Representation
Moves are dicts with `type` ("simple"/"jump"), `path` (list of [row,col] squares), and `captured` (list of [row,col] captured pieces). The engine's `get_all_legal_moves` is always ground truth — LLMs propose indices into this list, never raw coordinates.

### Thesis Instrumentation
State tracks `format_error_count`, `ranker_retry_count`, `ranker_fallback_count`, `llm_invoked`, `llm_agreed_with_symbolic_best` for thesis evaluation metrics. `evaluate_ranker_batch.py` computes filtered/full-legal mismatch rates.