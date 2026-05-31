#!/usr/bin/env python3
"""
Terminal-based annotation workflow for the human-validation study.

A lightweight alternative to the HTML packet workflow.  One paragraph at a
time, in the terminal; auto-saved after every answer; resumable.

INPUTS
------
  logs/evaluation/seed_ablation/human_validation/annotation_sample.csv
      (the blinded sample produced by scripts/build_human_validation_sample.py)

OUTPUTS  (auto-saved after every annotation; atomic writes)
------------------------------------------------------------
  logs/evaluation/seed_ablation/human_validation/responses/
      annotator_<X>/
          calibration_sheet.csv
          main_sheet.csv

The output CSVs match the schema used by Step 2 packet sheets:
  paragraph_id, Q1_factual_error, Q2_coherence, Q3_error_description, Q4_confidence

ORDERING
--------
Presentation order is computed deterministically with the SAME seeds as Step 2,
so a given annotator sees paragraphs in the same order whether they use the
terminal workflow or the HTML packet workflow.

  calibration : seed = SEED + 50           (fixed; shared across annotators)
  main        : seed = SEED + 100 + idx    (per-annotator)

where SEED = 20260527 and idx = ord('A') - ord('A') = 0, etc.  Annotator names
that are not single uppercase letters are hashed deterministically to an idx.

BLINDING
--------
The script reads ONLY annotation_sample.csv.  It never reads keyfile.csv or
any other file containing condition / verifier / game_id / turn_idx info.
The output sheet contains only paragraph_id + Q1-Q4 answers.

USAGE
-----
  # Calibration first
  python3 scripts/run_terminal_annotation.py --annotator A --mode calibration

  # Then main
  python3 scripts/run_terminal_annotation.py --annotator A --mode main

  # Resume after Ctrl-C / quit
  (re-run the same command; the script auto-resumes from the first
   incomplete paragraph)

Special commands available at every prompt:
  ?  or  help   show on-screen help
  q  or  quit   save and exit cleanly
  back          re-do the previous paragraph (clears its answers)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

# -- Configuration -----------------------------------------------------------

SEED                    = 20260527
CALIBRATION_SEED_OFFSET = 50
MAIN_SEED_OFFSET_BASE   = 100

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
HV_ROOT        = PROJECT_ROOT / 'logs' / 'evaluation' / 'seed_ablation' / 'human_validation'
SAMPLE_CSV     = HV_ROOT / 'annotation_sample.csv'
RESPONSES_ROOT = HV_ROOT / 'responses'

ANNOTATION_FIELDS = [
    'paragraph_id',
    'Q1_factual_error',          # 'YES' / 'NO'
    'Q2_coherence',              # 'YES' / 'NO'
    'Q3_error_description',      # free text (blank when Q1=NO)
    'Q4_confidence',             # '1'..'5'
]

# Strict blinding: any column from annotation_sample.csv whose name appears
# in this set would be a bug; the script enforces that none of these leak
# into the output sheet schema.
FORBIDDEN_OUTPUT_FIELDS = {
    'board_ascii', 'chosen_move_text', 'facts_table_json',
    'reasoning_text', 'calibration_flag', 'stratum',
    'condition', 'verifier_verdict', 'verifier_contradictions',
    'game_id', 'turn_idx',
}


# -- Helpers -----------------------------------------------------------------

def get_annotator_seed_offset(annotator: str) -> int:
    """Map an annotator name to a stable seed offset.

    Single uppercase letters map to A=0, B=1, ... (matches Step 2 exactly).
    Anything else is hashed deterministically.
    """
    if len(annotator) == 1 and annotator.isalpha() and annotator.isupper():
        return ord(annotator) - ord('A')
    h = hashlib.sha256(annotator.encode('utf-8')).hexdigest()
    return int(h[:8], 16) % 1000


def clear_screen() -> None:
    if os.environ.get('TERM', '') == 'dumb':
        print()
        return
    sys.stdout.write('\033[2J\033[H')
    sys.stdout.flush()


def atomic_write_csv(path: Path, rows: List[Dict]) -> None:
    """Write rows to CSV atomically (write-temp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=ANNOTATION_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in ANNOTATION_FIELDS})
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def is_complete(row: Dict) -> bool:
    q1 = row.get('Q1_factual_error', '')
    q2 = row.get('Q2_coherence', '')
    q3 = row.get('Q3_error_description', '')
    q4 = row.get('Q4_confidence', '')
    if q1 not in ('YES', 'NO'):
        return False
    if q2 not in ('YES', 'NO'):
        return False
    if q4 not in ('1', '2', '3', '4', '5'):
        return False
    if q1 == 'YES' and not (q3 or '').strip():
        return False
    return True


# -- Sample loading + ordering ----------------------------------------------

def load_sample() -> List[Dict]:
    if not SAMPLE_CSV.is_file():
        raise FileNotFoundError(
            f'annotation_sample.csv not found at {SAMPLE_CSV}.  '
            f'Run scripts/build_human_validation_sample.py first.'
        )
    with open(SAMPLE_CSV) as f:
        rows = list(csv.DictReader(f))
    required = {'paragraph_id', 'board_ascii', 'chosen_move_text',
                'facts_table_json', 'reasoning_text', 'calibration_flag'}
    missing = required - set(rows[0].keys())
    if missing:
        raise RuntimeError(f'annotation_sample.csv missing columns: {missing}')
    return rows


def compute_order(sample: List[Dict], mode: str, annotator: str) -> List[Dict]:
    """Compute the per-annotator presentation order matching Step 2."""
    subset = [r for r in sample if r['calibration_flag'] == mode]
    if not subset:
        raise RuntimeError(f'no paragraphs found with calibration_flag={mode!r}')
    sorted_subset = sorted(subset, key=lambda r: r['paragraph_id'])
    if mode == 'calibration':
        rng = random.Random(SEED + CALIBRATION_SEED_OFFSET)
    elif mode == 'main':
        idx = get_annotator_seed_offset(annotator)
        rng = random.Random(SEED + MAIN_SEED_OFFSET_BASE + idx)
    else:
        raise ValueError(f'unknown mode: {mode!r}')
    ordered = sorted_subset[:]
    rng.shuffle(ordered)
    return ordered


# -- Display -----------------------------------------------------------------

HRULE = '-' * 72


def display_paragraph(row: Dict, position: int, total: int) -> None:
    pid = row['paragraph_id']
    print()
    print(HRULE)
    print(f'  Item {position} of {total}        Paragraph ID: {pid}')
    print(HRULE)
    print()
    print('BOARD BEFORE THE MOVE')
    print(row['board_ascii'])
    print()
    print('CHOSEN MOVE')
    print(f'  {row["chosen_move_text"]}')
    print()
    print('SYMBOLIC FACTS')
    try:
        facts = json.loads(row['facts_table_json'])
    except Exception:
        facts = {}
    for k, v in facts.items():
        print(f'  {k:<30} {v}')
    print()
    print('REASONING PARAGRAPH TO EVALUATE')
    print()
    for line in row['reasoning_text'].split('\n'):
        print(f'  {line}')
    print()
    print(HRULE)
    print()


HELP_TEXT = """\
HELP
====
Q1  Factual error in this paragraph?
    YES  if the paragraph contains at least one statement that contradicts
         the symbolic facts above (wrong mobility number, claims
         center_control when destination_column is 0/1/6/7, claims "only
         legal move" when forced_move_for_us is False, etc.).
    NO   if every factual sentence in the paragraph is consistent with the
         facts table.  Vagueness, omission, and stylistic issues are NOT
         factual errors.

Q2  Is the paragraph coherent and on-topic?
    YES  if it reads as a sensible explanation of the chosen move (it can
         still contain factual errors and you would answer YES here).
    NO   if it is incoherent, off-topic, or describes a different move.

Q3  Error description (required only when Q1=YES).
    Quote or paraphrase each offending phrase and state which fact-table
    entry it contradicts.  One short line per error is enough.

Q4  Confidence in your Q1 answer.
    1 = guessing  |  2 = low  |  3 = moderate  |  4 = high  |  5 = certain

Commands available at any prompt:
  ?  or  help    show this help
  q  or  quit    save and exit
  back           re-do the PREVIOUS paragraph (clears its answers)
"""


# -- Prompts (with special-command handling) --------------------------------

class QuitRequested(Exception):
    pass


class BackRequested(Exception):
    pass


def _handle_special(s: str) -> bool:
    cmd = s.strip().lower()
    if cmd in ('?', 'help'):
        print(HELP_TEXT)
        return True
    if cmd in ('q', 'quit', 'exit'):
        raise QuitRequested()
    if cmd == 'back':
        raise BackRequested()
    return False


def ask_yes_no(prompt: str) -> str:
    while True:
        s = input(f'  {prompt} [Y/N]: ').strip()
        if _handle_special(s):
            continue
        c = s.lower()
        if c in ('y', 'yes'):
            return 'YES'
        if c in ('n', 'no'):
            return 'NO'
        print('    please answer Y or N')


def ask_freetext(prompt: str, required: bool) -> str:
    while True:
        s = input(f'  {prompt}\n  > ').rstrip('\n')
        if _handle_special(s):
            continue
        if not s.strip():
            if required:
                print('    required when Q1=YES; please enter at least one character')
                continue
            return ''
        return s


def ask_confidence(prompt: str) -> str:
    while True:
        s = input(f'  {prompt} [1-5]: ').strip()
        if _handle_special(s):
            continue
        if s in ('1', '2', '3', '4', '5'):
            return s
        print('    please enter a single integer between 1 and 5')


# -- Sheet state -------------------------------------------------------------

def init_or_load_sheet(sheet_path: Path,
                       expected_ids: List[str]) -> List[Dict]:
    """If sheet exists, validate ID/order; else create blank in given order."""
    if sheet_path.exists():
        with open(sheet_path) as f:
            existing = list(csv.DictReader(f))
        existing_ids = [r['paragraph_id'] for r in existing]
        if existing_ids != expected_ids:
            raise RuntimeError(
                f'existing sheet ID order does not match the expected order '
                f'for this annotator/mode.  This usually means the sample '
                f'changed since you started, or you used a different '
                f'--annotator name.\n'
                f'  sheet: {sheet_path}\n'
                f'To start over for this annotator+mode:\n'
                f'  rm {sheet_path}'
            )
        # ensure column set matches schema exactly
        if list(existing[0].keys()) != ANNOTATION_FIELDS:
            raise RuntimeError(
                f'existing sheet columns {list(existing[0].keys())} do not '
                f'match expected {ANNOTATION_FIELDS}.  Delete and restart.'
            )
        return existing

    rows = [{
        'paragraph_id': pid,
        'Q1_factual_error': '',
        'Q2_coherence': '',
        'Q3_error_description': '',
        'Q4_confidence': '',
    } for pid in expected_ids]
    atomic_write_csv(sheet_path, rows)
    return rows


# -- Startup banner + soft pre-checks ---------------------------------------

def print_intro(annotator: str, mode: str, sheet_path: Path,
                n_done: int, n_total: int) -> None:
    print()
    print('=' * 72)
    print('   Terminal Annotation Workflow')
    print('   Human Validation of the Verifier')
    print('=' * 72)
    print()
    print(f'   Annotator   : {annotator}')
    print(f'   Mode        : {mode}')
    print(f'   Output sheet: {sheet_path}')
    print(f'   Progress    : {n_done} / {n_total} completed')
    print()
    print('   Commands at any prompt:')
    print('     ?  or  help     show on-screen help')
    print('     q  or  quit     save and exit')
    print('     back            re-do the previous paragraph')
    print()
    print('-' * 72)


def maybe_warn_about_calibration(args) -> None:
    """If running --mode main while calibration isn't completed for this
    annotator, print a soft warning (does not block)."""
    if args.mode != 'main':
        return
    cal_path = RESPONSES_ROOT / f'annotator_{args.annotator}' / 'calibration_sheet.csv'
    if not cal_path.exists():
        print(f'WARN: no calibration sheet found for annotator {args.annotator}.')
        print(f'      It is strongly recommended to complete calibration first:')
        print(f'        python3 {Path(__file__).name} --annotator {args.annotator} --mode calibration')
        print()
        return
    with open(cal_path) as f:
        cal_rows = list(csv.DictReader(f))
    done = sum(1 for r in cal_rows if is_complete(r))
    if done < len(cal_rows):
        print(f'WARN: calibration is only {done}/{len(cal_rows)} complete for '
              f'annotator {args.annotator}.')
        print(f'      Consider finishing calibration before main.')
        print()


# -- Main pipeline -----------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Terminal annotation workflow for the human-validation study.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Example:\n'
            '  python3 scripts/run_terminal_annotation.py --annotator A --mode calibration\n'
            '  python3 scripts/run_terminal_annotation.py --annotator A --mode main'
        ),
    )
    parser.add_argument('--annotator', required=True,
                        help='Annotator id (typically a single uppercase letter, e.g. A)')
    parser.add_argument('--mode', required=True,
                        choices=['calibration', 'main'],
                        help='Which subset to annotate')
    parser.add_argument('--no-clear', action='store_true',
                        help='Do not clear the screen between paragraphs')
    args = parser.parse_args(argv)

    # Load + validate sample
    try:
        sample = load_sample()
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2

    # Schema check: required input columns must not be missing or wrong type
    if FORBIDDEN_OUTPUT_FIELDS & set(ANNOTATION_FIELDS):
        print('INTERNAL ERROR: ANNOTATION_FIELDS overlaps blinding-forbidden set',
              file=sys.stderr)
        return 4

    # Compute order
    try:
        ordered = compute_order(sample, args.mode, args.annotator)
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2
    expected_ids = [r['paragraph_id'] for r in ordered]
    n_total = len(ordered)

    # Soft pre-checks
    maybe_warn_about_calibration(args)

    # Init or load sheet
    sheet_path = RESPONSES_ROOT / f'annotator_{args.annotator}' / f'{args.mode}_sheet.csv'
    try:
        sheet_rows = init_or_load_sheet(sheet_path, expected_ids)
    except RuntimeError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 3

    n_done = sum(1 for r in sheet_rows if is_complete(r))

    print_intro(args.annotator, args.mode, sheet_path, n_done, n_total)

    if n_done >= n_total:
        print()
        print(f'All {n_total} paragraphs are already completed for '
              f'annotator {args.annotator} ({args.mode}).')
        print(f'Output: {sheet_path}')
        return 0

    try:
        input('Press Enter to begin (or Ctrl-C to exit)...')
    except (KeyboardInterrupt, EOFError):
        print()
        return 0

    # Find first incomplete
    i = 0
    while i < n_total and is_complete(sheet_rows[i]):
        i += 1

    while i < n_total:
        sample_row = ordered[i]
        sheet_row  = sheet_rows[i]
        if sheet_row['paragraph_id'] != sample_row['paragraph_id']:
            print('INTERNAL ERROR: sheet/order paragraph_id misalignment',
                  file=sys.stderr)
            return 5

        if not args.no_clear:
            clear_screen()
        display_paragraph(sample_row, i + 1, n_total)

        try:
            q1 = ask_yes_no('Q1. Factual error in this paragraph?')
            q2 = ask_yes_no('Q2. Is this paragraph coherent and on-topic?')
            if q1 == 'YES':
                q3 = ask_freetext(
                    'Q3. List each factual error you found '
                    '(required when Q1=YES):',
                    required=True,
                )
            else:
                q3 = ''
            q4 = ask_confidence('Q4. Confidence in your Q1 answer')

            sheet_row['Q1_factual_error']     = q1
            sheet_row['Q2_coherence']         = q2
            sheet_row['Q3_error_description'] = q3
            sheet_row['Q4_confidence']        = q4
            atomic_write_csv(sheet_path, sheet_rows)
            print(f'  saved.  ({i + 1}/{n_total})')
            i += 1

        except BackRequested:
            if i == 0:
                print('  already at the first paragraph; cannot go back.')
                try:
                    input('  press Enter to continue...')
                except (KeyboardInterrupt, EOFError):
                    pass
                continue
            i -= 1
            prev = sheet_rows[i]
            prev['Q1_factual_error']     = ''
            prev['Q2_coherence']         = ''
            prev['Q3_error_description'] = ''
            prev['Q4_confidence']        = ''
            atomic_write_csv(sheet_path, sheet_rows)
            print(f'  cleared previous answer (paragraph {prev["paragraph_id"]});'
                  f' will re-prompt.')
            try:
                input('  press Enter to continue...')
            except (KeyboardInterrupt, EOFError):
                pass

        except QuitRequested:
            n_done = sum(1 for r in sheet_rows if is_complete(r))
            print()
            print(f'saved.  exit.  progress: {n_done} / {n_total}')
            print(f'output: {sheet_path}')
            return 0

        except (KeyboardInterrupt, EOFError):
            n_done = sum(1 for r in sheet_rows if is_complete(r))
            print()
            print(f'interrupted.  saved.  progress: {n_done} / {n_total}')
            print(f'output: {sheet_path}')
            return 0

    # Completion summary
    if not args.no_clear:
        clear_screen()
    n_done = sum(1 for r in sheet_rows if is_complete(r))
    print()
    print('=' * 72)
    print(f'  All {n_total} paragraphs completed for annotator '
          f'{args.annotator} ({args.mode}).')
    print(f'  Output : {sheet_path}')
    print('=' * 72)
    print()

    # Per-question summary (descriptive only; never reveals condition/verdict)
    counts_q1 = {'YES': 0, 'NO': 0}
    counts_q2 = {'YES': 0, 'NO': 0}
    counts_q4 = {str(k): 0 for k in range(1, 6)}
    for r in sheet_rows:
        counts_q1[r['Q1_factual_error']] += 1
        counts_q2[r['Q2_coherence']]     += 1
        counts_q4[r['Q4_confidence']]    += 1
    print(f'  Q1 factual_error : YES={counts_q1["YES"]}  NO={counts_q1["NO"]}')
    print(f'  Q2 coherent      : YES={counts_q2["YES"]}  NO={counts_q2["NO"]}')
    print(f'  Q4 confidence    : 1={counts_q4["1"]}  2={counts_q4["2"]}  '
          f'3={counts_q4["3"]}  4={counts_q4["4"]}  5={counts_q4["5"]}')
    print()
    if n_done != n_total:
        print(f'WARN: {n_total - n_done} row(s) appear incomplete.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
