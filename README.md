# An Agentic System for Reasoning in Rule-Based Games II

**Bachelor Project — Neuro-Symbolic Multi-Agent Checkers**

This repository contains my Bachelor Project, investigating how Large Language Models behave in rule-governed environments — where they tend to propose illegal actions, lose consistency across turns, and generate explanations that don't match the actual game state. It asks whether a neuro-symbolic, multi-agent architecture can make decision-making legal by construction while keeping the accompanying natural-language reasoning grounded and verifiable.

American Checkers is used as the evaluation domain because it is fully rule-specified: legality and state transitions can be checked exactly, and generated explanations can be automatically verified against that same symbolic ground truth. By design, move legality and selection are handled entirely deterministically (minimax search over a hand-written rules engine), making illegal moves structurally impossible rather than merely unlikely; an LLM is used only to **explain** the chosen move, and its explanations are grounded in the same symbolic facts and pass through an automated verification-and-repair pipeline before being shown.

## Architecture

The pipeline is a four-node [LangGraph](https://github.com/langchain-ai/langgraph) state machine:

```
scorer_agent → proposer_agent → explainer_agent → updater_agent
     ↑                                                   │
     └─────────────── (next turn) ──────────────────────┘
```

| Node | Role |
|------|------|
| `scorer_agent` | Scores all legal moves with negamax + transposition table; writes ranked move list to state |
| `proposer_agent` | Deterministically selects the best move by minimax rank — the **sole move-selection authority** |
| `explainer_agent` | Calls Mistral to generate a natural-language explanation of the chosen move; never re-selects |
| `updater_agent` | Applies the move, checks win conditions, logs the turn |

The LLM only explains. The engine decides.

## Key Design Ideas

- **Proposal-authoritative**: `proposer_agent` writes `chosen_move`; `explainer_agent` reads it but never modifies it.
- **Unified verifier (E.1)**: `explainer_agent` calls `unified_verifier.contradiction_strings()` at runtime, ensuring runtime and evaluation see identical contradiction detection.
- **Comparative reasoning**: `explainer_agent` calls `comparative_reasoner.generate_comparative_reasoning()` to produce a paragraph comparing the chosen move to alternatives. `comparative_reasoner` imports only `llm_provider` — zero dependency on the decision pipeline.
- **Provider isolation**: all Mistral HTTP calls go through `llm_provider.py`. No other module makes HTTP requests directly.

## Evaluation & Research Methodology

The system is evaluated across four research questions: baseline LLM reliability at move generation, whether multi-agent responsibility separation eliminates illegal moves, whether symbolic grounding improves explanation consistency, and whether verification/repair reduces contradictions without introducing new errors. The evaluation includes a baseline LLM move-generation study, a move-selection benchmark against the KingsRow engine, and a controlled ablation comparing grounded ("seeded") vs. ungrounded LLM explanations, cross-checked against a manual audit. Full methodology, statistical validation, and results are reported in the thesis; the evaluation and ablation code lives in `checkers/evaluation/` and `checkers/baseline_eval/`.

## Repository Structure

```
checkers/
├── agents/
│   ├── scorer_agent.py          # Minimax scoring logic
│   ├── proposer_agent.py        # Deterministic best-move selection
│   ├── explainer_agent.py       # LLM explanation + runtime verification
│   ├── comparative_reasoner.py  # Comparative reasoning paragraph
│   ├── updater_agent.py         # End-of-turn composite node
│   └── llm_provider.py          # Mistral HTTP client
├── engine/
│   ├── board.py                 # Board representation
│   ├── rules.py                 # Legal move generation
│   ├── evaluation.py            # Board evaluation function
│   ├── move_facts.py            # Move feature computation
│   ├── minimax.py               # Minimax driver
│   ├── zobrist.py               # Zobrist hashing for repetition detection
│   ├── win_condition.py         # Terminal state detection
│   ├── kingsrow_interface.py    # Optional KingsRow engine bridge (ctypes)
│   └── fen_utils.py             # PDN/FEN conversion utilities
├── search/
│   ├── minimax_core.py          # Negamax + transposition table
│   └── selective_d8.py          # Selective depth-8 extension
├── nodes/
│   ├── scorer_node.py           # LangGraph node wrapping scorer_agent
│   ├── proposer_node.py         # LangGraph node wrapping proposer_agent
│   ├── state_manager.py         # Move application + history tracking
│   ├── state_manager_verify.py  # Legality verification for applied moves
│   ├── win_condition.py         # Win/draw detection node
│   └── logger_node.py           # JSONL turn logger
├── graph/
│   └── graph.py                 # LangGraph StateGraph definition
├── state/
│   └── state.py                 # CheckersState Pydantic model
├── ontology/
│   ├── forbidden_vocab.py       # Vocabulary the LLM must never use
│   └── semantic_ontology.py     # Semantic phrase constraints
├── evaluation/
│   ├── unified_verifier.py      # Runtime + evaluation verifier
│   ├── claim_extractor.py       # Factual claim extraction
│   ├── claim_verifier.py        # Claim-level verification
│   ├── claim_taxonomy.py        # Claim type taxonomy
│   ├── reasoning_taxonomy.py    # Reasoning path classification
│   ├── turn_evaluator.py        # Per-turn evaluation
│   ├── replay_evaluator.py      # Replay-based evaluation
│   ├── experiment_runner.py     # Batch experiment runner
│   ├── tactical_stress_suite.py # Tactical position stress tests
│   ├── run_manual_eval_pipeline.py
│   ├── run_ablation.py
│   ├── run_claim_recall_audit.py
│   ├── run_seed_ack_audit.py
│   ├── metrics/                 # Evaluation metric modules
│   └── ...
├── baseline_eval/
│   ├── reasoning_checker.py     # Baseline reasoning quality checks
│   ├── run_baseline_human_trace.py
│   └── run_baseline_scenario_suite.py
└── tests/                       # Automated unit and integration test suite

webapp/                          # Flask web UI (primary interface, see main.py)
├── app.py                       # Flask routes
├── game_service.py              # Game-session orchestration
└── static/index.html            # Single-page frontend
```

## Entry Points

| Script | Purpose |
|--------|---------|
| `main.py` | **Primary entry point.** Launches the Flask web UI (board, move history, AI reasoning trace) at `http://127.0.0.1:5050` |
| `run_simplified_trace.py` | Legacy terminal game: AI (RED) vs human (BLACK) |
| `run_simplified_trace_reasoning.py` | Same terminal game with deep reasoning diagnostics |
| `run_presentation_trace.py` | Presentation-mode with colored HTML export |
| `run_kingsrow_benchmark_trace.py` | Benchmark AI moves against KingsRow engine |
| `benchmark_evaluator.py` | Analyse benchmark JSONL results |

The web UI (`main.py`) supersedes the original terminal runner as the primary way to play; the terminal scripts remain for diagnostics, presentations, and benchmarking.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add MISTRAL_API_KEY
```

## Running

```bash
# Web UI (primary): play as BLACK against the AI in the browser
python main.py

# Legacy terminal runner
python run_simplified_trace.py

# With reasoning diagnostics
python run_simplified_trace_reasoning.py --show-claims

# Quiet mode (summary only)
python run_simplified_trace.py --quiet
```

## Testing

```bash
pytest checkers/tests/
```

## Deployment

This is a later addition, separate from the bachelor research contribution above; it packages the same application as a deployable service.

**Primary path:** Docker container → GitHub Actions CI/CD → Amazon ECR → Amazon EKS (Kubernetes). See `Dockerfile`, `.github/workflows/deploy.yml`, `k8s/`.

**Independent alternative path:** Docker container → Ansible → Amazon EC2. See `ansible/`.

The two paths are independent alternatives rather than stages of one pipeline — either can be used on its own to run the web UI outside a local machine.

## Environment Variables

The table below lists the variables you are likely to change. See `.env.example` for the complete annotated list including advanced engine-tuning and evaluation options.

| Variable | Default | Purpose |
|----------|---------|---------|
| `MISTRAL_API_KEY` | — | **Required.** Mistral API key |
| `MISTRAL_COMPARATIVE_API_KEY` | `MISTRAL_API_KEY` | Separate key for the comparative reasoning stage |
| `MISTRAL_EXPLAINER_MODEL` | `mistral-small-latest` | Model for explainer and comparative reasoner |
| `EXPLAINER_TEMPERATURE` | `0.2` | Sampling temperature |
| `EXPLAINER_SEEDS_DISABLED` | `0` | Set to `1` to disable symbolic reasoning-seed grounding (the seed-off condition used in the seed-on/seed-off ablation) |
| `EXPLAINER_COMPARATIVE_STAGE_ENABLED` | `1` | Set to `0` to skip the comparative reasoning paragraph |
| `MINIMAX_DEPTH` | `6` | Negamax search depth |
| `MINIMAX_ENABLED` | `true` | Set to `false` to disable minimax (debugging only) |
| `SELECTIVE_D8_ENABLED` | `false` | Enable selective depth-8 extension for tactical positions |
| `CHECKERS_LOGGER_PRINT` | `true` | Set to `false` to suppress per-turn logger stdout |
| `CHECKERS_LOG_DIR` | `logs` | Directory for JSONL game logs |
| `BASELINE_MISTRAL_API_KEY` | `MISTRAL_API_KEY` | API key for baseline evaluation scripts |
| `KINGSROW_DLL_PATH` | — | Path to KingsRow `.so`/`.dll` (benchmark only) |
