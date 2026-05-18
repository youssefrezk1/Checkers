#!/usr/bin/env python3
"""
preview_legality_prompts.py
============================
Prints N sample prompts exactly as the LLM would see them for legality testing.

- Board state, side to move, rules, and output format are shown.
- hidden_legal_moves are NEVER included in the prompt.
- After the prompt block, the hidden_legal_moves are printed SEPARATELY
  (clearly labelled "GROUND TRUTH — NOT IN PROMPT") for human verification.

Usage (from project root, venv active):
    python preview_legality_prompts.py [N] [--category CATEGORY] [--seed SEED]

    N            Number of scenarios to preview (default: 10)
    --category   Filter by specific category (optional)
    --seed       Random seed for sampling (default: 42)

Does NOT create or run any LLM baseline.
"""

import os
import sys
import json
import random
import argparse
import textwrap

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EVAL_SUBSET = os.path.join(
    PROJECT_ROOT, "checkers", "data", "legality_stress", "eval_subset_balanced.jsonl"
)

# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------

PIECE_SYMBOLS = {0: ".", 1: "r", 2: "b", 3: "R", 4: "B"}
PIECE_NAMES = {
    0: "empty",
    1: "red man",
    2: "black man",
    3: "red king",
    4: "black king",
}


def render_board(board: list) -> str:
    """
    Render the 8×8 board as a plain-text grid suitable for an LLM prompt.
    Legend printed below the board.
    """
    lines = [
        "    col: 0  1  2  3  4  5  6  7",
        "         " + "-" * 24,
    ]
    for r, row in enumerate(board):
        cells = "  ".join(PIECE_SYMBOLS[c] for c in row)
        lines.append(f"  row {r} | {cells}")

    lines += [
        "",
        "  Legend:",
        "    r = red man     (moves UP   — row decreases)",
        "    R = red king    (moves in all 4 diagonal directions)",
        "    b = black man   (moves DOWN — row increases)",
        "    B = black king  (moves in all 4 diagonal directions)",
        "    . = empty square",
        "  Only dark squares (row+col is odd) are playable.",
    ]
    return "\n".join(lines)


def move_to_human(mv: dict) -> str:
    """Format an engine move dict as a human-readable string."""
    path = " → ".join(f"(row {r}, col {c})" for r, c in mv["path"])
    if mv["captured"]:
        caps = ", ".join(f"(row {r}, col {c})" for r, c in mv["captured"])
        return f"[JUMP] {path}   [captures: {caps}]"
    return f"[SIMPLE] {path}"


# ---------------------------------------------------------------------------
# Prompt builder  (what the LLM sees — NO hidden_legal_moves)
# ---------------------------------------------------------------------------

RULES_TEXT = """\
CHECKERS RULES (English draughts):
  1. Pieces move diagonally on dark squares only.
  2. Men (non-kings) move in ONE forward direction only:
       - Red men move UP   (to lower row numbers).
       - Black men move DOWN (to higher row numbers).
  3. Kings move in all FOUR diagonal directions.
  4. MANDATORY CAPTURE: if any capture (jump) is available for any of your
     pieces, you MUST capture — simple moves are forbidden that turn.
  5. MULTI-JUMP: after a capture, if the same piece can capture again
     from its new position, it MUST continue jumping in the same turn.
     The sequence ends only when no further capture is possible.
  6. A piece cannot move to or land on an occupied square.
  7. You may only move YOUR OWN pieces.
  8. PROMOTION: when a man reaches the opponent's back rank
       (row 0 for red, row 7 for black) it is immediately crowned king.
     The turn ends at the promotion square even mid-jump sequence."""


def build_prompt(sc: dict) -> str:
    """
    Build the exact prompt the LLM would receive.
    hidden_legal_moves are NOT included anywhere in this string.
    """
    side = sc["side_to_move"]      # "RED" or "BLACK"
    category = sc["category"]
    rule_hint = sc["expected_rule"]
    difficulty = sc["difficulty"]

    board_str = render_board(sc["board"])

    prompt = f"""\
╔══════════════════════════════════════════════════════════════╗
║          CHECKERS LEGALITY TASK — {difficulty.upper():<10}              ║
╚══════════════════════════════════════════════════════════════╝

Scenario ID : {sc['scenario_id']}
Category    : {category}
Side to move: {side}

{RULES_TEXT}

CURRENT BOARD STATE:
{board_str}

KEY RULE FOR THIS SCENARIO:
  {textwrap.fill(rule_hint, width=72, subsequent_indent='  ')}

YOUR TASK:
  List ALL and ONLY the legal moves available to {side} in the current
  position. Apply the rules above exactly.

OUTPUT FORMAT (one move per line):
  For a simple move : SIMPLE (row R1, col C1) -> (row R2, col C2)
  For a jump        : JUMP   (row R1, col C1) -> (row R2, col C2)  captures (row Rc, col Cc)
  For a multi-jump  : JUMP   (row R1, col C1) -> (row R2, col C2) -> (row R3, col C3)  captures (row Ra, col Ca), (row Rb, col Cb)

  List every legal move. If captures are mandatory, list only captures.
  Do NOT list any illegal moves. Do NOT guess or invent moves.

BEGIN YOUR ANSWER:"""

    return prompt


# ---------------------------------------------------------------------------
# Ground-truth printer (separate block — clearly labelled)
# ---------------------------------------------------------------------------

def print_ground_truth(sc: dict) -> None:
    legal = sc["hidden_legal_moves"]
    print(f"  ┌── GROUND TRUTH — NOT IN PROMPT ({len(legal)} legal moves) ──")
    for mv in legal:
        print(f"  │  {move_to_human(mv)}")
    print(f"  └─────────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Preview legality prompts without exposing ground truth.")
    parser.add_argument("n", nargs="?", type=int, default=10,
                        help="Number of scenarios to preview (default: 10)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter by category name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--input", type=str, default=EVAL_SUBSET,
                        help="Path to JSONL file (default: eval_subset_balanced.jsonl)")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        print("       Run build_eval_subset.py first.")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        scenarios = [json.loads(l) for l in f if l.strip()]

    # Filter by category if requested
    if args.category:
        scenarios = [s for s in scenarios if s["category"] == args.category]
        if not scenarios:
            print(f"No scenarios found for category: {args.category!r}")
            sys.exit(1)

    # Sample N
    rng = random.Random(args.seed)
    sample = rng.sample(scenarios, min(args.n, len(scenarios)))

    print(f"\n{'='*66}")
    print(f"  LEGALITY PROMPT PREVIEW  ({len(sample)} scenarios)")
    if args.category:
        print(f"  Category filter: {args.category}")
    print(f"  Source: {os.path.basename(input_path)}")
    print(f"  NOTE: hidden_legal_moves are printed AFTER each prompt")
    print(f"        under 'GROUND TRUTH — NOT IN PROMPT'")
    print(f"{'='*66}\n")

    for i, sc in enumerate(sample, 1):
        print(f"\n{'─'*66}")
        print(f"  SCENARIO {i} of {len(sample)}")
        print(f"{'─'*66}\n")

        # Prompt (no hidden_legal_moves)
        prompt = build_prompt(sc)
        print(prompt)

        print()
        # Ground truth (clearly separated)
        print_ground_truth(sc)
        print()


if __name__ == "__main__":
    main()
