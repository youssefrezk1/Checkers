#!/usr/bin/env python3
"""
Step 2 of the human-validation study.

Render annotator-ready packets from the blinded sample produced by
scripts/build_human_validation_sample.py.  Stdlib-only.

Inputs (read only):
  logs/evaluation/seed_ablation/human_validation/annotation_sample.csv
  logs/evaluation/seed_ablation/human_validation/keyfile.csv      (calibration key only)

Outputs (overwritten on each run):
  logs/evaluation/seed_ablation/human_validation/packets/
    README.txt
    instructions.html
    instructions.md
    calibration_answer_key.csv                 # OWNER ONLY -- never give to annotators
    annotator_A/
      presentation_order.json
      calibration_packet.html                  # 16 paragraphs, fixed shared order
      calibration_sheet.csv                    # Q1-Q4 form, paragraph_id pre-filled
      main_packet.html                         # 96 paragraphs, per-annotator random order
      main_sheet.csv
    annotator_B/
      (same set of files, different random order on the main packet)

CLI:
  python3 scripts/render_annotation_packets.py
      [--annotators A B [C ...]]   default: A B
      [--out DIR]                  default: logs/evaluation/seed_ablation/human_validation/packets

Blinding guarantees:
  - Packet HTML and annotation sheets contain ONLY: paragraph_id, board, chosen move,
    facts table, reasoning text, and the in-packet "ID: XXXX" header.
  - The columns 'condition', 'verifier_verdict', 'game_id', 'turn_idx', and
    'stratum' from the keyfile are NEVER written into annotator-facing files.
  - The 'stratum' column from annotation_sample.csv is also withheld
    (it leaks one bit of pair joint-state and is unnecessary for the annotator).
  - The calibration_answer_key.csv is written at the packets root, not inside
    any annotator subdirectory, to keep it physically separate.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# -- Configuration -----------------------------------------------------------

SEED                 = 20260527
CALIBRATION_SEED_OFFSET = 50      # fixed-shared calibration order
MAIN_SEED_OFFSET_BASE   = 100     # per-annotator main order: SEED + 100 + idx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HV_ROOT      = PROJECT_ROOT / 'logs' / 'evaluation' / 'seed_ablation' / 'human_validation'
SAMPLE_CSV   = HV_ROOT / 'annotation_sample.csv'
KEYFILE_CSV  = HV_ROOT / 'keyfile.csv'

# Fields/values that must NEVER appear in annotator-facing files.
# Note: the English word "condition" is intentionally NOT included -- it
# occurs in LLM reasoning prose ("under these conditions") and is benign.
# The actual leakage risks are the condition values (seed_on/seed_off) and
# the diagnostic-column identifiers.
FORBIDDEN_TOKENS = (
    'seed_on', 'seed_off',
    'verifier_verdict', 'verifier_contradictions',
    'reasoning_seeds', 'reasoning_final_contradictions',
    'reasoning_initial_contradictions',
    'ranker_diagnostics',
)

ANNOTATION_FIELDS = [
    'paragraph_id',
    'Q1_factual_error',          # YES / NO
    'Q2_coherence',              # YES / NO
    'Q3_error_description',      # free text
    'Q4_confidence',             # 1-5
]

# -- Loaders -----------------------------------------------------------------

def load_sample() -> List[Dict]:
    with open(SAMPLE_CSV) as f:
        return list(csv.DictReader(f))


def load_keyfile() -> List[Dict]:
    with open(KEYFILE_CSV) as f:
        return list(csv.DictReader(f))


# -- HTML rendering ----------------------------------------------------------

CSS = """
  @page { size: A4; margin: 1.5cm; }
  @media print {
    body { margin: 0; }
    .paragraph { page-break-after: always; }
    .header-bar { background: #eee !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    .reasoning { background: #fffbe6 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    .board, .move { background: #f4f4f4 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif;
         max-width: 760px; margin: 1.5em auto; color: #1c1c1c; }
  .cover { padding: 2em 0; border-bottom: 2px solid #444; margin-bottom: 1em; }
  .cover h1 { font-size: 20pt; margin: 0; }
  .cover .subtitle { color: #555; font-size: 11pt; }
  .paragraph { padding: 1.5em 0; }
  .paragraph:not(:first-of-type) { border-top: 1.5px dashed #bbb; }
  .header-bar { background: #eee; padding: 0.4em 0.9em; border-radius: 4px;
                font-size: 11pt; display: flex; justify-content: space-between; }
  h2 { font-size: 12pt; margin: 1em 0 0.5em 0; color: #333;
       border-left: 3px solid #888; padding-left: 0.5em; }
  pre.board, p.move { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
                      background: #f4f4f4; padding: 0.7em 1em; border-radius: 4px;
                      margin: 0.3em 0; line-height: 1.4; font-size: 11pt; }
  table.facts { border-collapse: collapse; font-size: 10.5pt; margin: 0.3em 0; }
  table.facts td { padding: 2px 18px 2px 0; vertical-align: top; }
  table.facts td.k { font-family: monospace; color: #555; }
  table.facts td.v { font-family: monospace; color: #1c1c1c; }
  .reasoning { background: #fffbe6; padding: 0.9em 1.1em; border-radius: 4px;
               border-left: 3px solid #d4a017; line-height: 1.55; font-size: 11pt; }
  .record-note { color: #b00020; font-weight: 600; margin-top: 0.7em; font-size: 11pt; }
  .ann-id { font-family: monospace; font-weight: bold; }
""".strip()


HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
"""

HTML_FOOT = "</body></html>\n"


def render_paragraph_html(row: Dict, position: int, total: int) -> str:
    pid = html.escape(row['paragraph_id'])
    board = html.escape(row['board_ascii'])
    chosen = html.escape(row['chosen_move_text'])
    reasoning = html.escape(row['reasoning_text'])

    try:
        facts = json.loads(row['facts_table_json'])
    except Exception:
        facts = {}

    rows_html = []
    for k, v in facts.items():
        rows_html.append(
            f'<tr><td class="k">{html.escape(str(k))}</td>'
            f'<td class="v">{html.escape(str(v))}</td></tr>'
        )
    facts_table = '<table class="facts"><tbody>' + ''.join(rows_html) + '</tbody></table>'

    return f"""<div class="paragraph">
  <div class="header-bar">
    <span>Item <strong>{position}</strong> of <strong>{total}</strong></span>
    <span>Paragraph ID: <span class="ann-id">{pid}</span></span>
  </div>
  <h2>Board before the move</h2>
  <pre class="board">{board}</pre>
  <h2>Chosen move</h2>
  <p class="move">{chosen}</p>
  <h2>Symbolic facts</h2>
  {facts_table}
  <h2>Reasoning paragraph to evaluate</h2>
  <div class="reasoning">{reasoning}</div>
  <p class="record-note">Record your answers on the annotation sheet using
     Paragraph ID <span class="ann-id">{pid}</span>.</p>
</div>"""


def render_packet_html(rows: List[Dict], packet_title: str, subtitle: str) -> str:
    title = html.escape(packet_title)
    sub = html.escape(subtitle)
    total = len(rows)
    cover = f"""<div class="cover">
  <h1>{title}</h1>
  <div class="subtitle">{sub}</div>
  <div class="subtitle" style="margin-top:0.4em">Total items: {total}.
       Record your answers on the accompanying annotation sheet, matching by
       Paragraph ID.  Do not write on this document.</div>
</div>"""
    body_parts = [
        HTML_HEAD.format(title=title, css=CSS),
        cover,
    ]
    for i, row in enumerate(rows, start=1):
        body_parts.append(render_paragraph_html(row, i, total))
    body_parts.append(HTML_FOOT)
    return '\n'.join(body_parts)


# -- Annotation-sheet rendering ----------------------------------------------

def write_annotation_sheet(path: Path, ordered_rows: List[Dict]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ANNOTATION_FIELDS)
        w.writeheader()
        for r in ordered_rows:
            w.writerow({
                'paragraph_id'        : r['paragraph_id'],
                'Q1_factual_error'    : '',
                'Q2_coherence'        : '',
                'Q3_error_description': '',
                'Q4_confidence'       : '',
            })


# -- Calibration answer key (OWNER ONLY) -------------------------------------

def write_calibration_answer_key(
    path: Path,
    calibration_rows_from_sample: List[Dict],
    keyfile_index: Dict[str, Dict],
) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            'paragraph_id',
            'expected_verdict',
            'expected_Q1_factual_error',
            'verifier_contradictions',
        ])
        w.writeheader()
        for r in calibration_rows_from_sample:
            pid = r['paragraph_id']
            kf = keyfile_index[pid]
            verdict = kf['verifier_verdict']  # 'clean' or 'flagged'
            # The expected Q1 label (against which calibration is scored)
            # equals the verifier's verdict:
            #   verifier flagged -> expected Q1 = YES (factual error present)
            #   verifier clean   -> expected Q1 = NO
            expected_q1 = 'YES' if verdict == 'flagged' else 'NO'
            w.writerow({
                'paragraph_id'             : pid,
                'expected_verdict'         : verdict,
                'expected_Q1_factual_error': expected_q1,
                'verifier_contradictions'  : kf['verifier_contradictions'],
            })


# -- Instructions document ---------------------------------------------------

INSTRUCTIONS_MD = """\
# Annotation Instructions

## 1.  What you are evaluating

You will read short paragraphs that describe single checkers moves.  Each
paragraph was written by a software system and your job is to decide whether
the paragraph contains any **factual errors** about the move or the board.

You will be shown, for every paragraph:

  * The 8 by 8 board *before* the move (ASCII diagram).
  * The chosen move in coordinate notation, e.g. `(5, 4) -> (4, 3)`.
  * A table of symbolic facts the system used (mobility counts, capture
    count, recapturability, etc.).
  * The reasoning paragraph itself.

The fact table is the ground truth.  Every claim in the reasoning paragraph
that talks about the board or the move must be consistent with this table.

## 2.  Reading the board

  * Rows are numbered `0` (top) through `7` (bottom).
  * Columns are numbered `0` (left) through `7` (right).
  * `r` = RED regular piece.   `R` = RED king.
  * `b` = BLACK regular piece. `B` = BLACK king.
  * `.` = empty square.
  * RED is the player making the move in every paragraph you see.  RED moves
    *upward* on the board (toward row 0).  BLACK moves downward.
  * Forward moves are diagonal by one square.  Captures are diagonal jumps
    over an opponent piece into the empty square beyond.

## 3.  Reading the facts table

Each row in the facts table is a board-state predicate.  Examples:

  * `captures_count: 1` -> the move captures one piece.
  * `our_mobility_before: 9`, `our_mobility_after: 10` -> RED had 9 legal
    moves before, will have 10 after.
  * `opponent_can_recapture: True` -> the opponent has a legal jump on their
    next turn that captures the piece just moved.
  * `results_in_king: True` -> the piece reaches RED's king row (row 0) on
    this move and becomes a king.
  * `center_control: True` -> the destination square satisfies the central-
    control test (destination column is 2, 3, 4, or 5 AND the move occupies
    a key central position; the engine sets this boolean).
  * `forced_move_for_us: True` -> this was RED's only legal move; RED had
    no choice.

If you are unsure what a particular predicate means, just check whether the
paragraph asserts a claim consistent with the value shown.

## 4.  The four annotation questions

For every paragraph, fill four columns on the annotation sheet using the
paragraph's `paragraph_id`.

### Q1.  Does this paragraph contain at least one statement that is factually inconsistent with the symbolic facts above or with the chosen move?

A **factual inconsistency** is a sentence in the paragraph that asserts
something the facts table contradicts.  Examples:

  * The paragraph says "our mobility increases from 7 to 9" but the facts
    table shows `our_mobility_after: 8`.
  * The paragraph claims "the destination is in the center of the board"
    when `destination_column` is `0`, `1`, `6`, or `7`.
  * The paragraph says "the moved piece cannot be recaptured" when
    `opponent_can_recapture: True`.
  * The paragraph says "this move promotes the piece to king" when
    `results_in_king: False`.
  * The paragraph says "this is the only legal move available" when
    `forced_move_for_us: False`.
  * The paragraph claims "the move captures one piece" when
    `captures_count: 0`.
  * The paragraph says "this creates an immediate threat" when
    `creates_immediate_threat: False`.

Answer **YES** if at least one such sentence exists.
Answer **NO** if every factual sentence is consistent with the facts table.

Do **NOT** mark a paragraph as containing a factual error solely because:

  * it is vague, repetitive, or awkward;
  * it omits a fact (omission is not the same as contradiction);
  * it uses informal language;
  * it offers an interpretation you personally disagree with but cannot
    refute from the facts table.

### Q2.  Is this paragraph coherent and on-topic as an explanation of the chosen move?

Answer **YES** if the paragraph reads as a sensible explanation of why this
move was selected, even if it contains factual errors.
Answer **NO** only if the paragraph is incoherent, off-topic, nonsensical,
or describes a different move than the one specified.

### Q3.  If YES to Q1, list each factual error you identified.

Quote or paraphrase the offending phrase, and state briefly which fact-table
entry it contradicts.  One short line per error is enough.  Skip this if
you answered NO to Q1.

### Q4.  How confident are you in your answer to Q1?

| value | meaning                |
|-------|------------------------|
| 1     | essentially guessing   |
| 2     | low                    |
| 3     | moderate               |
| 4     | high                   |
| 5     | certain                |

## 5.  Procedure

1.  Complete the **calibration packet** first (16 items).  Once you finish,
    the study coordinator will check your calibration answers against a
    held-out key.  If your agreement is at least 12 of 16 you proceed; if
    not, the coordinator will re-explain the questions and you will redo
    calibration.
2.  Complete the **main packet** (96 items).  Take breaks; you may split
    across at most two sessions.  Approximate pace: 2-3 minutes per item.
3.  Submit your completed annotation sheets back to the coordinator.

## 6.  What this study is for

You are helping validate an automated symbolic verifier that judges whether
machine-generated reasoning is factually consistent with board state.  Your
labels will be compared to the verifier's labels to estimate the verifier's
agreement with human judgement.  Please answer based on what you actually
see in the paragraph and the facts table, not on what you think the study
is hoping to find.
"""

def md_to_html(md: str) -> str:
    """Minimal markdown -> HTML, supporting only the subset used above:
    headings, paragraphs, fenced inline code, lists, simple tables.
    No external dependencies."""
    lines = md.split('\n')
    out_lines: List[str] = []
    in_ul = False
    in_table = False
    table_rows: List[str] = []
    para_buf: List[str] = []

    def close_para():
        if para_buf:
            text = ' '.join(s.strip() for s in para_buf).strip()
            if text:
                out_lines.append(f'<p>{_inline(text)}</p>')
            para_buf.clear()

    def close_ul():
        nonlocal in_ul
        if in_ul:
            out_lines.append('</ul>')
            in_ul = False

    def _inline(s: str) -> str:
        # backticks -> <code>
        import re as _re
        s = html.escape(s)
        s = _re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
        s = _re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
        return s

    def close_table():
        nonlocal in_table, table_rows
        if in_table and table_rows:
            out_lines.append('<table class="md-table"><thead>')
            header_cells = [c.strip() for c in table_rows[0].strip('|').split('|')]
            out_lines.append('<tr>' + ''.join(f'<th>{_inline(c)}</th>' for c in header_cells) + '</tr>')
            out_lines.append('</thead><tbody>')
            for r in table_rows[2:]:
                cells = [c.strip() for c in r.strip('|').split('|')]
                out_lines.append('<tr>' + ''.join(f'<td>{_inline(c)}</td>' for c in cells) + '</tr>')
            out_lines.append('</tbody></table>')
        in_table = False
        table_rows = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith('# '):
            close_para(); close_ul(); close_table()
            out_lines.append(f'<h1>{_inline(line[2:].strip())}</h1>')
        elif line.startswith('## '):
            close_para(); close_ul(); close_table()
            out_lines.append(f'<h2>{_inline(line[3:].strip())}</h2>')
        elif line.startswith('### '):
            close_para(); close_ul(); close_table()
            out_lines.append(f'<h3>{_inline(line[4:].strip())}</h3>')
        elif line.lstrip().startswith('* '):
            close_para(); close_table()
            if not in_ul:
                out_lines.append('<ul>'); in_ul = True
            out_lines.append(f'<li>{_inline(line.lstrip()[2:])}</li>')
        elif line.lstrip().startswith('|') and line.rstrip().endswith('|'):
            close_para(); close_ul()
            in_table = True
            table_rows.append(line.strip())
        elif not line.strip():
            close_para(); close_ul(); close_table()
        else:
            if in_table:
                close_table()
            close_ul()
            para_buf.append(line)
    close_para(); close_ul(); close_table()

    body = '\n'.join(out_lines)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Annotation Instructions</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 780px; margin: 2em auto; color: #1c1c1c; line-height: 1.5; }}
  h1 {{ font-size: 20pt; }}
  h2 {{ font-size: 14pt; margin-top: 1.5em; border-left: 3px solid #888; padding-left: 0.5em; }}
  h3 {{ font-size: 12pt; margin-top: 1.2em; }}
  code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-family: ui-monospace, monospace; font-size: 10.5pt; }}
  table.md-table {{ border-collapse: collapse; margin: 0.5em 0; }}
  table.md-table th, table.md-table td {{ border: 1px solid #ccc; padding: 4px 10px; }}
  table.md-table th {{ background: #f4f4f4; }}
  ul {{ margin: 0.4em 0; }}
  li {{ margin: 0.2em 0; }}
</style></head>
<body>
{body}
</body></html>
"""


# -- README ------------------------------------------------------------------

README_TEXT = """\
HUMAN VALIDATION OF THE VERIFIER - ANNOTATION PACKETS
=====================================================

Generated by scripts/render_annotation_packets.py.

CONTENTS
--------
  README.txt                       this file
  instructions.html                annotator instructions (open in any browser)
  instructions.md                  markdown source for instructions
  calibration_answer_key.csv       OWNER-ONLY - do NOT share with annotators
  annotator_<X>/                   one subdirectory per annotator
    calibration_packet.html          16 items, fixed shared order
    calibration_sheet.csv            empty form (Q1-Q4) with paragraph_ids
    main_packet.html                 96 items, per-annotator random order
    main_sheet.csv                   empty form (Q1-Q4) with paragraph_ids
    presentation_order.json          records the exact order used

DISTRIBUTING TO ANNOTATORS
--------------------------
Send each annotator their own subdirectory (e.g. annotator_A/) PLUS the file
instructions.html.

Do NOT send: calibration_answer_key.csv, keyfile.csv, the other annotators'
subdirectories, or any file with the prefix 'annotation_sample' or 'keyfile'.

CONVERTING HTML PACKETS TO PDF
------------------------------
The HTML packets are styled for print.  To produce a PDF:

  1.  Open calibration_packet.html (or main_packet.html) in a modern browser
      (Safari, Chrome, Firefox, Edge).
  2.  Use File > Print (Cmd-P / Ctrl-P).
  3.  Set destination to "Save as PDF".
  4.  Ensure "Background graphics" / "Print backgrounds" is enabled so the
      coloured panels survive in the PDF.
  5.  Save.

Each item starts on a new printed page via CSS @page rules.

WORKFLOW
--------
  1.  Annotator opens instructions.html and reads the procedure.
  2.  Annotator works through calibration_packet.html, writing answers in
      calibration_sheet.csv (one row per paragraph, matched by paragraph_id).
  3.  Coordinator scores the calibration sheet against calibration_answer_key.csv:
        agreement >= 12 of 16 -> proceed to main
        agreement <  12 of 16 -> re-explain questions, redo calibration
  4.  Annotator works through main_packet.html, filling main_sheet.csv.
  5.  Annotator returns calibration_sheet.csv and main_sheet.csv to coordinator.
"""


# -- Validation --------------------------------------------------------------

def validate_packets(out_root: Path, annotators: List[str],
                     n_calibration: int, n_main: int) -> None:
    """Refuse to declare success if any blinding check fails."""
    def fail(msg: str):
        raise RuntimeError(f'PACKET VALIDATION FAILED: {msg}')

    # (a) per-annotator file existence and counts
    for ann in annotators:
        adir = out_root / f'annotator_{ann}'
        for name in ('calibration_packet.html', 'calibration_sheet.csv',
                     'main_packet.html', 'main_sheet.csv',
                     'presentation_order.json'):
            if not (adir / name).is_file():
                fail(f'missing {adir / name}')

        # sheet row counts
        with open(adir / 'calibration_sheet.csv') as f:
            cal_rows = list(csv.DictReader(f))
        if len(cal_rows) != n_calibration:
            fail(f'{ann} calibration_sheet rows {len(cal_rows)} != {n_calibration}')
        with open(adir / 'main_sheet.csv') as f:
            main_rows = list(csv.DictReader(f))
        if len(main_rows) != n_main:
            fail(f'{ann} main_sheet rows {len(main_rows)} != {n_main}')

        # column set on sheets
        if list(cal_rows[0].keys()) != ANNOTATION_FIELDS:
            fail(f'{ann} calibration_sheet columns wrong: {list(cal_rows[0].keys())}')
        if list(main_rows[0].keys()) != ANNOTATION_FIELDS:
            fail(f'{ann} main_sheet columns wrong: {list(main_rows[0].keys())}')

        # leakage scan over packets and sheets
        for fn in ('calibration_packet.html', 'main_packet.html',
                   'calibration_sheet.csv', 'main_sheet.csv'):
            text = (adir / fn).read_text(encoding='utf-8')
            hits = [t for t in FORBIDDEN_TOKENS if t in text]
            if hits:
                fail(f'leakage tokens in {ann}/{fn}: {hits}')

    # (b) presentation orders for main packet are DIFFERENT per annotator
    orders = {}
    for ann in annotators:
        with open(out_root / f'annotator_{ann}' / 'presentation_order.json') as f:
            orders[ann] = json.load(f)['main']
    if len(annotators) >= 2:
        first = orders[annotators[0]]
        for ann in annotators[1:]:
            if orders[ann] == first:
                fail(f'annotator {ann} main order identical to {annotators[0]}')

    # (c) calibration order identical across annotators (shared fixed order)
    cal_orders = {}
    for ann in annotators:
        with open(out_root / f'annotator_{ann}' / 'presentation_order.json') as f:
            cal_orders[ann] = json.load(f)['calibration']
    if len(annotators) >= 2:
        first = cal_orders[annotators[0]]
        for ann in annotators[1:]:
            if cal_orders[ann] != first:
                fail(f'annotator {ann} calibration order != {annotators[0]}')

    # (d) calibration answer key contains the expected number of rows and
    #     lives at the packets root (not inside any annotator subdir)
    key_path = out_root / 'calibration_answer_key.csv'
    if not key_path.is_file():
        fail('calibration_answer_key.csv missing')
    with open(key_path) as f:
        key_rows = list(csv.DictReader(f))
    if len(key_rows) != n_calibration:
        fail(f'calibration_answer_key rows {len(key_rows)} != {n_calibration}')

    # (e) the answer key MUST NOT exist inside any annotator subdir
    for ann in annotators:
        if (out_root / f'annotator_{ann}' / 'calibration_answer_key.csv').exists():
            fail(f'answer key leaked into annotator_{ann} subdirectory')


# -- Main --------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotators', nargs='+', default=['A', 'B'])
    parser.add_argument('--out', default=str(HV_ROOT / 'packets'))
    args = parser.parse_args(argv)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    annotators: List[str] = args.annotators

    print(f'[1/7] Loading sample from {SAMPLE_CSV}')
    sample = load_sample()
    keyfile = load_keyfile()
    if len(sample) != len(keyfile):
        raise RuntimeError('annotation_sample.csv and keyfile.csv row counts disagree')

    keyfile_index = {r['paragraph_id']: r for r in keyfile}

    # Partition
    main_rows = [r for r in sample if r['calibration_flag'] == 'main']
    cal_rows  = [r for r in sample if r['calibration_flag'] == 'calibration']
    n_main = len(main_rows)
    n_cal  = len(cal_rows)
    print(f'      main paragraphs: {n_main}   calibration: {n_cal}   total: {len(sample)}')

    # (1) Calibration: fixed shared shuffled order, sorted-first for reproducibility
    cal_rng = random.Random(SEED + CALIBRATION_SEED_OFFSET)
    cal_sorted = sorted(cal_rows, key=lambda r: r['paragraph_id'])
    cal_order  = cal_sorted[:]
    cal_rng.shuffle(cal_order)
    cal_order_ids = [r['paragraph_id'] for r in cal_order]

    # (2) Calibration answer key (OWNER ONLY)
    print(f'[2/7] Writing calibration answer key (owner-only)')
    write_calibration_answer_key(
        out_root / 'calibration_answer_key.csv',
        # Use sorted (not shuffled) for the answer key so the owner can scan
        cal_sorted,
        keyfile_index,
    )

    # (3) Instructions (markdown + html)
    print(f'[3/7] Writing instructions')
    (out_root / 'instructions.md').write_text(INSTRUCTIONS_MD, encoding='utf-8')
    (out_root / 'instructions.html').write_text(md_to_html(INSTRUCTIONS_MD), encoding='utf-8')

    # (4) Per-annotator packets
    print(f'[4/7] Rendering packets for annotators: {annotators}')
    for idx, ann in enumerate(annotators):
        ann_dir = out_root / f'annotator_{ann}'
        ann_dir.mkdir(parents=True, exist_ok=True)

        # Main: per-annotator random order
        main_rng = random.Random(SEED + MAIN_SEED_OFFSET_BASE + idx)
        main_sorted = sorted(main_rows, key=lambda r: r['paragraph_id'])
        main_order = main_sorted[:]
        main_rng.shuffle(main_order)
        main_order_ids = [r['paragraph_id'] for r in main_order]

        # presentation_order.json (record both orders for reproducibility)
        with open(ann_dir / 'presentation_order.json', 'w', encoding='utf-8') as f:
            json.dump({
                'annotator': ann,
                'seed_calibration': SEED + CALIBRATION_SEED_OFFSET,
                'seed_main'       : SEED + MAIN_SEED_OFFSET_BASE + idx,
                'calibration'     : cal_order_ids,
                'main'            : main_order_ids,
                'generated_utc'   : datetime.now(timezone.utc).isoformat(timespec='seconds'),
            }, f, indent=2)

        # Calibration packet HTML
        cal_packet = render_packet_html(
            cal_order,
            packet_title=f'Calibration packet  (annotator {ann})',
            subtitle=f'{n_cal} items.  Read instructions.html first.',
        )
        (ann_dir / 'calibration_packet.html').write_text(cal_packet, encoding='utf-8')

        # Main packet HTML
        main_packet = render_packet_html(
            main_order,
            packet_title=f'Main packet  (annotator {ann})',
            subtitle=f'{n_main} items.  Complete calibration first.',
        )
        (ann_dir / 'main_packet.html').write_text(main_packet, encoding='utf-8')

        # Annotation sheets (CSV, pre-populated paragraph_id in presentation order)
        write_annotation_sheet(ann_dir / 'calibration_sheet.csv', cal_order)
        write_annotation_sheet(ann_dir / 'main_sheet.csv', main_order)

        print(f'      annotator_{ann}: cal_packet + main_packet + 2 sheets written')

    # (5) README
    print(f'[5/7] Writing README')
    (out_root / 'README.txt').write_text(README_TEXT, encoding='utf-8')

    # (6) Validation
    print(f'[6/7] Validation checks')
    validate_packets(out_root, annotators, n_cal, n_main)
    print(f'      OK')

    # (7) Summary
    print(f'[7/7] Done.  Packets at: {out_root}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
