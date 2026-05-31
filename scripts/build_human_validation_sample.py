#!/usr/bin/env python3
"""
Step 1 of the human-validation study.

Build a stratified, paired, blinded annotation sample from the existing
seed-ablation evaluation_source artifacts.  No reruns, no LLM calls.

Inputs (read only):
  logs/evaluation/seed_ablation/ablation_20260527_112813_*_gNNN.jsonl
      (top-level per-game move logs -- used for board reconstruction)
  logs/evaluation/seed_ablation/evaluation_source/seed_on/*.jsonl
  logs/evaluation/seed_ablation/evaluation_source/seed_off/*.jsonl

Outputs (overwritten on each run):
  logs/evaluation/seed_ablation/human_validation/annotation_sample.csv
  logs/evaluation/seed_ablation/human_validation/keyfile.csv

Design:
  - Strata at the GAME-TURN PAIR level by joint verifier verdict:
        CC  seed_on clean, seed_off clean
        CF  seed_on clean, seed_off flagged
        FC  seed_on flagged, seed_off clean
        FF  seed_on flagged, seed_off flagged
    Verdict = (len(reasoning_final_contradictions) >= 1).

  - 14 pairs sampled per stratum (deterministic random.seed(20260527)):
        first 12  -> main study
        last  2   -> calibration set

  - 48 main pairs (96 main paragraphs) + 8 cal pairs (16 cal paragraphs)
    = 112 paragraphs total.
    24 main paragraphs per (condition x verifier_verdict) cell.

  - Each paragraph carries a blinded 4-char alphanumeric ID.  The keyfile
    holds the full mapping; the annotation file contains no leakage.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import string
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# -- Sampling configuration (independent of I/O paths) -----------------------

SEED                  = 20260527
# Target sample sizes per stratum.  These are upper bounds — when a stratum
# has fewer pairs available, the sample size automatically downscales to the
# stratum's actual population.  Large evaluations behave exactly as before;
# small evaluations (smoke tests, discovery runs) auto-fit without crashing.
TARGET_PAIRS_PER_STRATUM = 14
TARGET_MAIN_PER_STRATUM  = 12   # remainder = calibration when full target met
# ABLATION_PREFIX and N_GAMES are auto-detected from the evaluation tree at
# runtime (see _discover_run_layout).  No specific historical run-id is
# hard-coded; any per-game files matching ablation_*_gNNN.jsonl are accepted.
GAME_FILE_RE          = re.compile(r'^(ablation_.+)_g(\d{3})\.jsonl$')

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_ROOT = PROJECT_ROOT / 'logs' / 'evaluation' / 'seed_ablation'

PARAGRAPH_ID_LEN = 4
PARAGRAPH_ID_ALPHABET = string.ascii_uppercase + string.digits  # 36 chars


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Build a stratified, paired, blinded annotation sample from '
            'an existing seed-ablation evaluation_source tree.'
        ),
    )
    p.add_argument(
        '--input', type=Path, default=DEFAULT_EVAL_ROOT,
        help=(
            'Evaluation root directory. Must contain '
            'evaluation_source/seed_on and evaluation_source/seed_off '
            'subdirectories, plus the per-game ablation_*.jsonl files at '
            'its top level. '
            f'Default: {DEFAULT_EVAL_ROOT}'
        ),
    )
    p.add_argument(
        '--output', type=Path, default=None,
        help=(
            'Human-validation output directory. Default: '
            '<input>/human_validation. The annotation_sample.csv and '
            'keyfile.csv files are written here (created if missing).'
        ),
    )
    return p.parse_args(argv)


def _discover_run_layout(game_dir: Path, seed_on_dir: Path, seed_off_dir: Path
                         ) -> Tuple[str, int]:
    """Detect (ablation_prefix, n_games) by scanning game_dir for files
    matching ablation_*_gNNN.jsonl.  The prefix is whatever precedes the
    `_gNNN.jsonl` suffix; n_games is the count of contiguous game indices
    starting from g000.  Same prefix must exist in seed_on_dir and
    seed_off_dir for each game index; otherwise we fail with a clear
    error.  No specific historical run-id is assumed.
    """
    candidates: Dict[str, List[int]] = defaultdict(list)
    for p in game_dir.iterdir():
        if not p.is_file():
            continue
        m = GAME_FILE_RE.match(p.name)
        if not m:
            continue
        prefix, idx_str = m.group(1), m.group(2)
        candidates[prefix].append(int(idx_str))

    if not candidates:
        raise SystemExit(
            f'no per-game files matching ablation_*_gNNN.jsonl found under '
            f'--input: {game_dir}'
        )
    # If multiple runs are present in the same directory, prefer the prefix
    # with the most game files; tie-break alphabetically for determinism.
    prefix = sorted(
        candidates.keys(),
        key=lambda k: (-len(candidates[k]), k),
    )[0]
    indices = sorted(candidates[prefix])

    # Require a contiguous block starting at 000.
    contiguous = 0
    for i, idx in enumerate(indices):
        if idx != i:
            break
        contiguous += 1
    if contiguous == 0:
        raise SystemExit(
            f'detected ablation prefix {prefix!r} under --input but no '
            f'g000.jsonl file found; cannot determine game count.'
        )

    # Cross-check seed_on / seed_off contain matching per-game files.
    for i in range(contiguous):
        on_fn  = seed_on_dir  / f'{prefix}_g{i:03d}.jsonl'
        off_fn = seed_off_dir / f'{prefix}_g{i:03d}.jsonl'
        if not on_fn.is_file():
            raise SystemExit(
                f'missing seed_on per-game file for game {i:03d}: {on_fn}'
            )
        if not off_fn.is_file():
            raise SystemExit(
                f'missing seed_off per-game file for game {i:03d}: {off_fn}'
            )

    return prefix, contiguous


def resolve_paths(args: argparse.Namespace) -> Dict[str, object]:
    eval_root    = args.input.resolve()
    seed_on_dir  = eval_root / 'evaluation_source' / 'seed_on'
    seed_off_dir = eval_root / 'evaluation_source' / 'seed_off'
    game_dir     = eval_root
    out_dir      = (args.output if args.output is not None
                    else eval_root / 'human_validation').resolve()

    # Validate inputs exist before doing any work.
    if not eval_root.exists():
        raise SystemExit(f'--input path does not exist: {eval_root}')
    if not eval_root.is_dir():
        raise SystemExit(f'--input must be a directory: {eval_root}')
    if not seed_on_dir.is_dir():
        raise SystemExit(
            f'missing evaluation_source/seed_on under --input: {seed_on_dir}'
        )
    if not seed_off_dir.is_dir():
        raise SystemExit(
            f'missing evaluation_source/seed_off under --input: {seed_off_dir}'
        )

    # Auto-detect the run prefix and game count from the actual file layout.
    ablation_prefix, n_games = _discover_run_layout(
        game_dir, seed_on_dir, seed_off_dir,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        'eval_root'      : eval_root,
        'seed_on_dir'    : seed_on_dir,
        'seed_off_dir'   : seed_off_dir,
        'game_dir'       : game_dir,
        'out_dir'        : out_dir,
        'ablation_prefix': ablation_prefix,
        'n_games'        : n_games,
    }


def initial_board() -> List[List[int]]:
    b = [[0] * 8 for _ in range(8)]
    for r in range(3):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r][c] = 3
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r][c] = 1
    return b


def apply_move(board, path, captured, promotion, player) -> None:
    src_r, src_c = path[0]
    dst_r, dst_c = path[-1]
    piece = board[src_r][src_c]
    if piece == 0:
        raise RuntimeError(
            f'replay failed: source square ({src_r},{src_c}) is empty '
            f'for player={player}, path={path}'
        )
    board[src_r][src_c] = 0
    for cr, cc in captured:
        board[cr][cc] = 0
    is_king = piece in (2, 4) or promotion
    if player == 1:
        board[dst_r][dst_c] = 2 if is_king else 1
    else:
        board[dst_r][dst_c] = 4 if is_king else 3


def replay_to_turn(game_moves, target_turn):
    b = initial_board()
    for m in game_moves:
        if m['turn'] >= target_turn:
            break
        apply_move(b, m['path'], m['captured'], m['promotion'], m['player'])
    return b


def render_board_ascii(board) -> str:
    glyph = {0: '.', 1: 'r', 2: 'R', 3: 'b', 4: 'B'}
    lines = ['  0 1 2 3 4 5 6 7']
    for r in range(8):
        row = ' '.join(glyph[board[r][c]] for c in range(8))
        lines.append(f'{r} {row}')
    return '\n'.join(lines)


def load_eval_source(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_game_moves(game_idx, game_dir: Path, ablation_prefix: str):
    fn = game_dir / f'{ablation_prefix}_g{game_idx:03d}.jsonl'
    moves = []
    with open(fn) as f:
        for line in f:
            r = json.loads(line)
            moves.append({
                'turn': r['turn'], 'player': r['player_who_moved'],
                'path': r['path'], 'captured': r['captured'],
                'promotion': r['promotion'],
            })
    return moves


def verifier_flagged(rec) -> bool:
    final = (rec.get('ranker_diagnostics') or {}).get(
        'reasoning_final_contradictions'
    ) or []
    return len(final) > 0


def extract_chosen_reasoning(rec) -> str:
    diag = rec.get('ranker_diagnostics') or {}
    text = diag.get('chosen_reasoning')
    if isinstance(text, str) and text.strip():
        return text.strip()
    text = rec.get('last_move_reasoning')
    return text.strip() if isinstance(text, str) else ''


def extract_facts_table(rec, n_legal):
    f = rec.get('chosen_move_facts') or {}
    path = (rec.get('ranker_diagnostics') or {}).get('final_chosen_path') or []
    dest_col = path[-1][1] if path else None
    forced = (n_legal == 1) if isinstance(n_legal, int) else None
    return {
        'captures_count'             : f.get('captures_count'),
        'net_gain'                   : f.get('net_gain'),
        'our_mobility_before'        : f.get('our_mobility_before'),
        'our_mobility_after'         : f.get('our_mobility_after'),
        'opponent_mobility_before'   : f.get('opponent_mobility_before'),
        'opponent_mobility_after'    : f.get('opponent_mobility_after'),
        'opponent_can_recapture'     : f.get('opponent_can_recapture'),
        'our_pieces_threatened_after': f.get('our_pieces_threatened_after'),
        'leaves_piece_isolated'      : f.get('leaves_piece_isolated'),
        'creates_immediate_threat'   : f.get('creates_immediate_threat'),
        'forced_opponent_jump_reply' : f.get('forced_opponent_jump_reply'),
        'results_in_king'            : f.get('results_in_king'),
        'near_promotion'             : f.get('near_promotion'),
        'center_control'             : f.get('center_control'),
        'destination_column'         : dest_col,
        'forced_move_for_us'         : forced,
        'minimax_score'              : f.get('minimax_score'),
    }


def format_chosen_move_text(rec) -> str:
    diag = rec.get('ranker_diagnostics') or {}
    path = diag.get('final_chosen_path') or []
    cm = (rec.get('chosen_move_facts') or {})
    captured_count = cm.get('captures_count') or 0
    move_type = cm.get('move_type') or 'simple'
    if not path:
        return 'RED move: (unknown path)'
    if len(path) == 2:
        s, e = path[0], path[-1]
        notation = f'({s[0]}, {s[1]}) -> ({e[0]}, {e[1]})'
    else:
        notation = ' -> '.join(f'({p[0]}, {p[1]})' for p in path)
    return (
        f'RED {move_type} move: {notation}.  '
        f'Captures: {captured_count} piece(s).'
    )


def build_pair_index(seed_on_dir: Path, seed_off_dir: Path,
                     ablation_prefix: str, n_games: int):
    pairs = []
    for game_idx in range(n_games):
        on_fn  = seed_on_dir  / f'{ablation_prefix}_g{game_idx:03d}.jsonl'
        off_fn = seed_off_dir / f'{ablation_prefix}_g{game_idx:03d}.jsonl'
        on_recs  = load_eval_source(on_fn)
        off_recs = load_eval_source(off_fn)
        if len(on_recs) != len(off_recs):
            raise RuntimeError(
                f'game g{game_idx:03d}: length mismatch '
                f'on={len(on_recs)} off={len(off_recs)}'
            )
        for turn_idx, (a, b) in enumerate(zip(on_recs, off_recs)):
            if a.get('turn_id') != b.get('turn_id'):
                raise RuntimeError(
                    f'turn_id mismatch at g{game_idx:03d} idx {turn_idx}: '
                    f'on={a.get("turn_id")} off={b.get("turn_id")}'
                )
            pairs.append({
                'game_idx': game_idx, 'turn_idx': turn_idx,
                'turn_id': a['turn_id'], 'on': a, 'off': b,
                'on_flag': verifier_flagged(a),
                'off_flag': verifier_flagged(b),
            })
    return pairs


def stratum_of(p) -> str:
    return ('F' if p['on_flag'] else 'C') + ('F' if p['off_flag'] else 'C')


def gen_paragraph_id(rng, used):
    while True:
        pid = ''.join(rng.choice(PARAGRAPH_ID_ALPHABET) for _ in range(PARAGRAPH_ID_LEN))
        if pid not in used:
            used.add(pid)
            return pid


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    paths = resolve_paths(args)
    seed_on_dir     = paths['seed_on_dir']
    seed_off_dir    = paths['seed_off_dir']
    game_dir        = paths['game_dir']
    out_dir         = paths['out_dir']
    ablation_prefix = paths['ablation_prefix']
    n_games         = paths['n_games']
    print(f'      detected run prefix: {ablation_prefix} ({n_games} games)')

    rng = random.Random(SEED)

    print(f'[1/6] Loading paired index from {seed_on_dir} / {seed_off_dir}')
    pairs = build_pair_index(seed_on_dir, seed_off_dir,
                             ablation_prefix, n_games)
    print(f'      total paired turns: {len(pairs)}')

    strata = defaultdict(list)
    for p in pairs:
        strata[stratum_of(p)].append(p)

    # Per-stratum sample plan: clamp the target to the available population,
    # never raise on underfilled strata.  Empty strata emit a warning and are
    # skipped — the rest of the pipeline continues with whatever data exists.
    print('[2/6] Stratum populations (target = '
          f'{TARGET_PAIRS_PER_STRATUM} pairs, '
          f'{TARGET_MAIN_PER_STRATUM} main + '
          f'{TARGET_PAIRS_PER_STRATUM - TARGET_MAIN_PER_STRATUM} calibration):')
    plan = {}  # stratum → {'population','target','effective','n_main','n_cal'}
    for s in ('CC', 'CF', 'FC', 'FF'):
        pop = len(strata[s])
        effective = min(TARGET_PAIRS_PER_STRATUM, pop)
        # Preserve original main/calibration split when population is full.
        # When downscaled, fill main first, then calibration with what remains.
        n_main = min(TARGET_MAIN_PER_STRATUM, effective)
        n_cal  = effective - n_main
        plan[s] = {
            'population': pop, 'target': TARGET_PAIRS_PER_STRATUM,
            'effective' : effective, 'n_main': n_main, 'n_cal': n_cal,
        }
        print(f'      {s}: population={pop}  target={TARGET_PAIRS_PER_STRATUM}  '
              f'effective={effective}  (main={n_main}, calibration={n_cal})')
        if pop == 0:
            print(f'      WARNING: stratum {s} is empty — skipping.')

    sampled = {}
    for s in ('CC', 'CF', 'FC', 'FF'):
        if plan[s]['effective'] == 0:
            sampled[s] = []
            continue
        bucket = sorted(strata[s], key=lambda p: (p['game_idx'], p['turn_idx']))
        sampled[s] = rng.sample(bucket, plan[s]['effective'])

    total_sampled = sum(len(v) for v in sampled.values())
    if total_sampled == 0:
        raise SystemExit(
            'no pairs available in any stratum — nothing to sample. '
            'Check that evaluation_source/seed_on and seed_off contain '
            'paired per-turn records for the detected run prefix.'
        )
    nonempty_strata = sum(1 for s in plan if plan[s]['effective'] > 0)
    print(f'[3/6] Sampled {total_sampled} pair(s) across '
          f'{nonempty_strata}/{len(plan)} non-empty stratum(s) '
          f'(seed={SEED}); per-stratum splits shown above.')

    print('[4/6] Reconstructing boards by replaying move sequences...')
    needed_targets = defaultdict(list)
    for s_pairs in sampled.values():
        for p in s_pairs:
            try:
                turn_no = int(p['turn_id'].rsplit('_t', 1)[-1])
            except Exception as e:
                raise RuntimeError(f'cannot parse turn_id {p["turn_id"]}: {e}')
            needed_targets[p['game_idx']].append(turn_no)

    board_cache = {}
    for g, targets in needed_targets.items():
        moves = load_game_moves(g, game_dir, ablation_prefix)
        for t in set(targets):
            board_cache[(g, t)] = replay_to_turn(moves, t)
    print(f'      reconstructed {len(board_cache)} boards across '
          f'{len(needed_targets)} games')

    print('[5/6] Building blinded annotation rows + keyfile rows...')
    used_ids = set()
    annotation_rows = []
    keyfile_rows = []

    id_rng = random.Random(SEED + 1)

    for s in ('CC', 'CF', 'FC', 'FF'):
        for pair_idx_in_stratum, p in enumerate(sampled[s]):
            # Calibration cutoff scales with the actual sample size: the first
            # `n_main` pairs are study items, anything beyond is calibration.
            # Matches original behaviour exactly when effective == target.
            is_calibration = pair_idx_in_stratum >= plan[s]['n_main']
            calibration_flag = 'calibration' if is_calibration else 'main'

            turn_no = int(p['turn_id'].rsplit('_t', 1)[-1])
            board = board_cache[(p['game_idx'], turn_no)]
            board_ascii = render_board_ascii(board)

            n_legal = (
                (p['on'].get('proposal_diagnostics') or {}).get('n_legal')
            )

            for cond_label, rec in (('seed_on', p['on']),
                                    ('seed_off', p['off'])):
                pid = gen_paragraph_id(id_rng, used_ids)

                reasoning = extract_chosen_reasoning(rec)
                if not reasoning:
                    raise RuntimeError(
                        f'empty reasoning text at pair {p["turn_id"]} '
                        f'condition={cond_label}'
                    )

                chosen_move_text = format_chosen_move_text(rec)
                facts = extract_facts_table(rec, n_legal)

                verdict_label = 'flagged' if verifier_flagged(rec) else 'clean'
                final_contras = (
                    (rec.get('ranker_diagnostics') or {})
                    .get('reasoning_final_contradictions') or []
                )

                annotation_rows.append({
                    'paragraph_id': pid,
                    'board_ascii': board_ascii,
                    'chosen_move_text': chosen_move_text,
                    'facts_table_json': json.dumps(facts, ensure_ascii=False,
                                                   sort_keys=False),
                    'reasoning_text': reasoning,
                    'calibration_flag': calibration_flag,
                    'stratum': s,
                })
                keyfile_rows.append({
                    'paragraph_id': pid,
                    'game_id': f'g{p["game_idx"]:03d}',
                    'turn_idx': p['turn_idx'],
                    'condition': cond_label,
                    'verifier_verdict': verdict_label,
                    'verifier_contradictions': json.dumps(final_contras,
                                                          ensure_ascii=False),
                    'calibration_flag': calibration_flag,
                    'stratum': s,
                })

    print('[6/6] Validation checks (data-driven invariants)...')

    def fail(msg):
        raise RuntimeError(f'VALIDATION FAILED: {msg}')

    # Per-stratum: actual main/cal output rows must match the plan computed
    # from the sample population (×2 because each pair produces two
    # condition rows: seed_on + seed_off).
    for s in ('CC', 'CF', 'FC', 'FF'):
        n_main = sum(1 for r in keyfile_rows
                     if r['stratum'] == s and r['calibration_flag'] == 'main')
        n_cal  = sum(1 for r in keyfile_rows
                     if r['stratum'] == s and r['calibration_flag'] == 'calibration')
        expected_main = plan[s]['n_main'] * 2
        expected_cal  = plan[s]['n_cal']  * 2
        if n_main != expected_main:
            fail(f'stratum {s} main count {n_main} != {expected_main}')
        if n_cal != expected_cal:
            fail(f'stratum {s} cal count {n_cal} != {expected_cal}')

    # Main-cell breakdown: by construction each main pair contributes one row
    # under its stratum's joint (on, off) verdict to BOTH condition columns,
    # so the seed_on/seed_off cell totals must each equal the sum of n_main
    # over the matching stratum-verdict combinations.
    cell_counts = defaultdict(int)
    for r in keyfile_rows:
        if r['calibration_flag'] == 'main':
            cell_counts[(r['condition'], r['verifier_verdict'])] += 1
    stratum_to_verdicts = {
        'CC': ('clean',   'clean'),
        'CF': ('clean',   'flagged'),
        'FC': ('flagged', 'clean'),
        'FF': ('flagged', 'flagged'),
    }
    expected_by_cell: dict = defaultdict(int)
    for s, (on_v, off_v) in stratum_to_verdicts.items():
        expected_by_cell[('seed_on',  on_v)]  += plan[s]['n_main']
        expected_by_cell[('seed_off', off_v)] += plan[s]['n_main']
    for cell, expected in expected_by_cell.items():
        if cell_counts[cell] != expected:
            fail(f'main cell {cell}: {cell_counts[cell]} != {expected}')

    # Every sampled pair must appear exactly twice (one row per condition).
    pair_seen = defaultdict(int)
    for r in keyfile_rows:
        pair_seen[(r['game_id'], r['turn_idx'])] += 1
    bad_pairs = [k for k, v in pair_seen.items() if v != 2]
    if bad_pairs:
        fail(f'{len(bad_pairs)} pair(s) appear != 2 times')

    # Totals derived from the plan, not from a hardcoded constant.
    n_main_total = sum(1 for r in keyfile_rows if r['calibration_flag'] == 'main')
    n_cal_total  = sum(1 for r in keyfile_rows if r['calibration_flag'] == 'calibration')
    n_total      = len(keyfile_rows)
    expected_main_total = sum(plan[s]['n_main'] for s in plan) * 2
    expected_cal_total  = sum(plan[s]['n_cal']  for s in plan) * 2
    expected_total      = expected_main_total + expected_cal_total
    if n_main_total != expected_main_total:
        fail(f'main total {n_main_total} != {expected_main_total}')
    if n_cal_total != expected_cal_total:
        fail(f'cal total {n_cal_total} != {expected_cal_total}')
    if n_total != expected_total:
        fail(f'total {n_total} != {expected_total}')

    ids = [r['paragraph_id'] for r in annotation_rows]
    if len(set(ids)) != len(ids):
        fail('duplicate paragraph_ids detected')
    if any(len(i) != PARAGRAPH_ID_LEN for i in ids):
        fail('paragraph_id length mismatch')

    empties = [r['paragraph_id'] for r in annotation_rows if not r['reasoning_text']]
    if empties:
        fail(f'{len(empties)} paragraph(s) with empty reasoning text')

    leaked = {'condition', 'verifier_verdict', 'verifier_contradictions',
              'game_id', 'turn_idx', 'turn_id', 'seeds', 'reasoning_seeds'}
    if annotation_rows and leaked & set(annotation_rows[0].keys()):
        fail(f'annotation file leaks columns: '
             f'{leaked & set(annotation_rows[0].keys())}')

    print(f'      OK: {expected_total} paragraphs '
          f'({expected_main_total} main + {expected_cal_total} calibration); '
          f'per-stratum cell expectations satisfied.')

    ann_path = out_dir / 'annotation_sample.csv'
    key_path = out_dir / 'keyfile.csv'

    shuffle_rng = random.Random(SEED + 2)
    shuffle_rng.shuffle(annotation_rows)

    with open(ann_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            'paragraph_id', 'board_ascii', 'chosen_move_text',
            'facts_table_json', 'reasoning_text',
            'calibration_flag', 'stratum',
        ])
        w.writeheader()
        for row in annotation_rows:
            w.writerow(row)

    with open(key_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            'paragraph_id', 'game_id', 'turn_idx',
            'condition', 'verifier_verdict', 'verifier_contradictions',
            'calibration_flag', 'stratum',
        ])
        w.writeheader()
        for row in keyfile_rows:
            w.writerow(row)

    print(f'\nwrote {ann_path}')
    print(f'wrote {key_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
