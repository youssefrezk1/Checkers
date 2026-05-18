# checkers/data/pdn_importer/pdn_parser.py
"""
PDN file parser.

Handles three PDN file types found in this project:

  gem.pdn               — pure problem positions (Setup/FEN only, no move text)
  inferno.pdn           — full game records (move-text only, no FEN)
  Tricks traps and shots.pdn — mixed: some entries have FEN, some are game records

For each entry this module produces one or more "raw positions" — snapshots of
the board at a specific ply — with the following structure:

    {
        "source_file": str,          # basename of the PDN file
        "game_index":  int,          # index of the game/problem in the file
        "ply_index":   int,          # 0 = the position BEFORE any moves are made
                                     #     (i.e. the FEN or the game start)
                                     #     N = after N half-moves have been applied
        "board":       list[list[int]],   # 8×8 engine board
        "side_to_move": int,              # BLACK (2) or RED (1)
        "event":       str,          # [Event] tag value
        "has_fen":     bool,         # True = position came from a FEN tag
        "move_text":   str,          # raw PDN move text (empty for FEN-only entries)
    }

Design notes
- FEN-based entries produce exactly one position (ply_index = 0).
- Game records produce one position per ply (ply_index 0..N-1), capped at
  MAX_PLIES_PER_GAME to keep the dataset tractable.
- Positions where the current player has 0 legal moves (game over) are skipped.
- Parse errors are logged and skipped rather than crashing.
"""

import re
import os
import copy
import logging
from typing import Iterator

from checkers.engine.board import RED, BLACK, create_initial_board
from checkers.engine.rules import get_all_legal_moves, apply_move, _moves_match
from checkers.data.pdn_importer.fen_utils import (
    parse_fen, pdn_move_to_engine, board_to_serializable, side_to_str
)

log = logging.getLogger(__name__)

MAX_PLIES_PER_GAME = 80   # safety cap — no real game needs more than this


# ---------------------------------------------------------------------------
# Low-level PDN tag / move-text extraction
# ---------------------------------------------------------------------------

# Match a single [Tag "value"] header
_TAG_RE = re.compile(r'\[(\w+)\s+"([^"]*)"\]')

# Match a PDN move token: "11-15", "14x23", "9x14x23x16", "1/2-1/2", "0-1", "1-0"
_MOVE_TOKEN_RE = re.compile(
    r'\b(\d{1,2}(?:[x\-]\d{1,2})+)\b'   # real moves
)

# Result strings to ignore as move tokens
_RESULT_STRS = {'1-0', '0-1', '1/2-1/2'}


def _iter_games(pdn_text: str) -> Iterator[dict]:
    """
    Split a PDN file into individual game/problem blocks and yield tag dicts.

    Each yielded dict has:
        tags      : dict[str, str]
        move_text : str   (everything that isn't a tag line)
    """
    # Split on blank lines that precede a new [Event ...] tag
    # We collect blocks separated by double newlines followed by '['
    blocks = re.split(r'\n(?=\[Event\s+")', pdn_text.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        tags = {}
        move_parts = []

        for line in block.splitlines():
            tag_match = _TAG_RE.match(line.strip())
            if tag_match:
                tags[tag_match.group(1)] = tag_match.group(2)
            else:
                # Strip inline comments {…} before storing move text
                cleaned = re.sub(r'\{[^}]*\}', ' ', line)
                move_parts.append(cleaned)

        yield {
            "tags": tags,
            "move_text": ' '.join(move_parts).strip(),
        }


def _extract_move_tokens(move_text: str) -> list:
    """
    Extract ordered list of raw PDN move strings from a move-text block.
    Removes move numbers (e.g. '1.', '2.'), result strings, and comments.
    """
    # Remove move numbers like "1." or "12."
    cleaned = re.sub(r'\b\d+\.\s*', ' ', move_text)
    # Remove result strings
    for result in _RESULT_STRS:
        cleaned = cleaned.replace(result, ' ')

    tokens = _MOVE_TOKEN_RE.findall(cleaned)
    # Filter out anything that is purely a result
    return [t for t in tokens if t not in _RESULT_STRS]


# ---------------------------------------------------------------------------
# Position replay helpers
# ---------------------------------------------------------------------------

def _find_legal_move(board, side, engine_move: dict):
    """
    Find the engine's canonical version of engine_move in get_all_legal_moves.
    Returns the matched legal move dict, or None if no match found.
    """
    legal = get_all_legal_moves(board, side)
    for lm in legal:
        if _moves_match(engine_move, lm):
            return lm
    return None


def _apply_pdn_move(board, side, move_str: str):
    """
    Parse move_str, validate it against legal moves, apply it.
    Returns (new_board, new_side) or (None, None) on failure.
    """
    engine_move = pdn_move_to_engine(move_str)
    if engine_move is None:
        return None, None

    legal_match = _find_legal_move(board, side, engine_move)
    if legal_match is None:
        # PDN move doesn't match any legal move — log and skip
        log.debug("Move %r not found in legal moves for side %s", move_str, side_to_str(side))
        return None, None

    new_board = apply_move(board, legal_match)
    new_side = BLACK if side == RED else RED
    return new_board, new_side


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdn_file(filepath: str) -> list:
    """
    Parse a PDN file and return a flat list of raw position dicts.

    Each position dict:
        source_file, game_index, ply_index, board, side_to_move, event, has_fen

    Positions where side_to_move has zero legal moves are excluded.
    """
    source_file = os.path.basename(filepath)
    positions = []

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            pdn_text = f.read()
    except OSError as e:
        log.error("Cannot read %s: %s", filepath, e)
        return []

    for game_idx, game in enumerate(_iter_games(pdn_text)):
        tags = game["tags"]
        move_text = game["move_text"]
        event = tags.get("Event", f"game_{game_idx}")
        has_fen = "FEN" in tags and tags.get("Setup", "0") == "1"

        # ---- FEN-based problem position ----
        if has_fen:
            try:
                board, side = parse_fen(tags["FEN"])
            except Exception as e:
                log.warning("Bad FEN in %s game %d: %s", source_file, game_idx, e)
                continue

            legal = get_all_legal_moves(board, side)
            if not legal:
                continue    # terminal position — skip

            positions.append({
                "source_file":  source_file,
                "game_index":   game_idx,
                "ply_index":    0,
                "board":        board_to_serializable(board),
                "side_to_move": side_to_str(side),
                "event":        event,
                "has_fen":      True,
            })

            # Also replay any moves that follow the FEN (TTS problems have lines)
            if move_text:
                positions.extend(
                    _replay_moves(
                        board, side, move_text,
                        source_file, game_idx, event,
                        start_ply=1, has_fen=True
                    )
                )

        # ---- Full game record (no FEN) ----
        elif move_text:
            board = create_initial_board()
            side  = BLACK   # Standard: Black moves first in new games

            # Initial position
            legal = get_all_legal_moves(board, side)
            if legal:
                positions.append({
                    "source_file":  source_file,
                    "game_index":   game_idx,
                    "ply_index":    0,
                    "board":        board_to_serializable(board),
                    "side_to_move": side_to_str(side),
                    "event":        event,
                    "has_fen":      False,
                })

            positions.extend(
                _replay_moves(
                    board, side, move_text,
                    source_file, game_idx, event,
                    start_ply=1, has_fen=False
                )
            )

    return positions


def _replay_moves(board, side, move_text: str,
                  source_file: str, game_idx: int, event: str,
                  start_ply: int, has_fen: bool) -> list:
    """
    Replay PDN move tokens from move_text against the engine.
    Returns list of position dicts for each successfully applied ply.
    """
    positions = []
    tokens = _extract_move_tokens(move_text)
    current_board = copy.deepcopy(board)
    current_side  = side
    ply = start_ply

    for token in tokens:
        if ply >= MAX_PLIES_PER_GAME:
            break

        new_board, new_side = _apply_pdn_move(current_board, current_side, token)
        if new_board is None:
            # Invalid move — stop replaying this game
            break

        current_board = new_board
        current_side  = new_side

        legal = get_all_legal_moves(current_board, current_side)
        if legal:
            positions.append({
                "source_file":  source_file,
                "game_index":   game_idx,
                "ply_index":    ply,
                "board":        board_to_serializable(current_board),
                "side_to_move": side_to_str(current_side),
                "event":        event,
                "has_fen":      has_fen,
            })
        ply += 1

    return positions
