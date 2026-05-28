import ctypes
import os
import re
import sys
import time
from typing import Optional, Dict, Any

from checkers.engine.board import RED, BLACK, RED_KING, BLACK_KING, EMPTY
from checkers.data.pdn_importer.fen_utils import square_to_rowcol

# CheckerBoard Piece Values
CB_FREE = 0
CB_WHITE = 1
CB_BLACK = 2
CB_MAN = 4
CB_KING = 8


class coor(ctypes.Structure):
    _fields_ = [('x', ctypes.c_int), ('y', ctypes.c_int)]


class CBmove(ctypes.Structure):
    _fields_ = [
        ('jumps',    ctypes.c_int),
        ('newpiece', ctypes.c_int),
        ('oldpiece', ctypes.c_int),
        ('from_sq',  coor),
        ('to_sq',    coor),
        ('path',     coor * 12),
        ('delpath',  coor * 12),
        ('delpiece', ctypes.c_int * 12),
    ]


BoardType = (ctypes.c_int * 8) * 8


class KingsRowEngine:
    def __init__(self, dll_path: str):
        self.dll_path = os.path.abspath(dll_path)
        engine_dir = os.path.dirname(self.dll_path)

        old_cwd = os.getcwd()
        try:
            os.chdir(engine_dir)
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(engine_dir)
                cb_root = os.path.abspath(os.path.join(engine_dir, '..'))
                if os.path.exists(cb_root):
                    os.add_dll_directory(cb_root)

            self.dll = ctypes.WinDLL(self.dll_path)
            self.dll.getmove.argtypes = [
                BoardType, ctypes.c_int, ctypes.c_double, ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(CBmove),
            ]
            self.dll.getmove.restype = ctypes.c_int
        finally:
            os.chdir(old_cwd)

    # ------------------------------------------------------------------ board helpers

    def _rotate_board(self, engine_board: list) -> list:
        """Rotate 180° and swap colors so RED becomes CB_BLACK."""
        flipped = [[EMPTY] * 8 for _ in range(8)]
        for r in range(8):
            for c in range(8):
                piece = engine_board[r][c]
                if piece == EMPTY:
                    continue
                if piece == RED:          flipped[7 - r][7 - c] = BLACK
                elif piece == BLACK:      flipped[7 - r][7 - c] = RED
                elif piece == RED_KING:   flipped[7 - r][7 - c] = BLACK_KING
                elif piece == BLACK_KING: flipped[7 - r][7 - c] = RED_KING
        return flipped

    def _convert_board(self, engine_board: list) -> BoardType:
        """
        Convert to CheckerBoard column-major layout.
        CB stores cb_board[cb_c][cb_r] where cb_c = 7-c, cb_r = r.
        Dark squares satisfy (cb_c + cb_r) % 2 == 0, matching CB's convention.
        """
        cb_board = BoardType()
        for cb_c in range(8):
            for cb_r in range(8):
                cb_board[cb_c][cb_r] = CB_FREE

        for r in range(8):
            for c in range(8):
                piece = engine_board[r][c]
                if piece == EMPTY:
                    continue
                if piece == RED:          cb_val = CB_WHITE | CB_MAN
                elif piece == BLACK:      cb_val = CB_BLACK | CB_MAN
                elif piece == RED_KING:   cb_val = CB_WHITE | CB_KING
                elif piece == BLACK_KING: cb_val = CB_BLACK | CB_KING
                else:                     cb_val = CB_FREE

                cb_board[7 - c][r] = cb_val   # cb_c = 7-c, cb_r = r

        return cb_board

    # ------------------------------------------------------------------ coordinate helpers

    def _cbmove_to_engine_path(self, cbmove, rotated: bool) -> Optional[list]:
        """
        Convert a populated CBmove struct to an engine (row, col) path.

        CB stores pieces at cb_board[cb_c][cb_r] with cb_c=7-c, cb_r=r.
        Therefore:  cbmove.from_sq.x = cb_c  →  engine col = 7 - x
                    cbmove.from_sq.y = cb_r  →  engine row = y
        Coordinates are on the (possibly rotated) eval_board.
        When rotated=True, apply (7-r, 7-c) to map back to the original board.
        """
        def _sq(x: int, y: int) -> list:
            r, c = y, 7 - x
            if rotated:
                r, c = 7 - r, 7 - c
            return [r, c]

        # Both squares being zero means the DLL returned no move.
        if (cbmove.from_sq.x == 0 and cbmove.from_sq.y == 0 and
                cbmove.to_sq.x == 0 and cbmove.to_sq.y == 0):
            return None

        from_sq = _sq(cbmove.from_sq.x, cbmove.from_sq.y)
        to_sq   = _sq(cbmove.to_sq.x,   cbmove.to_sq.y)
        jumps   = cbmove.jumps

        if jumps == 0:
            return [from_sq, to_sq]

        # Multi-jump: cbmove.path[0..jumps-2] = intermediate landing squares.
        path = [from_sq]
        for i in range(jumps - 1):
            path.append(_sq(cbmove.path[i].x, cbmove.path[i].y))
        path.append(to_sq)
        return path

    def _pv_to_engine_path(self, pv_str: str, eval_board: list,
                           rotated: bool) -> Optional[list]:
        """
        Convert the first move of a KR PV string to an engine path.

        KR always uses STANDARD PDN square numbering (1-32, same as fen_utils).
        Coordinates are on the eval_board; apply (7-r, 7-c) when rotated=True.

        Validates that the 'from' square has a non-empty piece on eval_board
        so we discard stale PV lines caused by transposition-table hash holes.
        """
        if not pv_str:
            return None

        first_move = pv_str.split(' ')[0]
        is_jump    = 'x' in first_move.lower()
        normalized = first_move.lower().replace('x', '-')
        parts      = normalized.split('-')
        try:
            squares = [int(s.strip()) for s in parts if s.strip()]
        except ValueError:
            return None
        if len(squares) < 2:
            return None

        # Build path using standard PDN square→(row, col)
        try:
            path_on_eval = [list(square_to_rowcol(sq)) for sq in squares]
        except Exception:
            return None

        # Validate: from-square must contain a piece on the eval_board
        fr, fc = path_on_eval[0]
        if not (0 <= fr <= 7 and 0 <= fc <= 7):
            return None
        if eval_board[fr][fc] == EMPTY:
            return None

        # Unrotate back to original board coordinates when needed
        if rotated:
            path_on_eval = [[7 - sq[0], 7 - sq[1]] for sq in path_on_eval]

        return path_on_eval

    # ------------------------------------------------------------------ main search

    def get_best_move(self, board: list, current_player: int,
                      time_budget: float = 1.0,
                      target_depth: int = 6) -> Dict[str, Any]:
        """
        Ask KingsRow for the best move from *board* for *current_player*.

        Search mode
        -----------
        KingsRow uses the *sign* of the maxtime parameter to switch modes:
            maxtime > 0   → time-based search, bounded by maxtime seconds
            maxtime < 0   → fixed-depth search to |maxtime| plies
        (This was verified empirically against Kingsrow(x64) 1.19e: at the
        opening position, maxtime=1.0 reached depth ~26 in ~5.6s, while
        maxtime=-6 finished in ~15ms.)

        When target_depth > 0 we send maxtime = -target_depth so the search
        stops strictly at the requested ply count.  Otherwise we fall back to
        the time-based mode using time_budget.

        Other strategy
        --------------
        1.  If current_player is RED, rotate the board 180° and swap colours so
            the side-to-move is always CB_BLACK (KR's native perspective).
        2.  KR returns res=3 to yield mid-search; the loop polls until res!=3.
            A 3× time_budget Python safety cap prevents a runaway in the rare
            case the DLL keeps yielding past its own budget.
        3.  On every loop iteration, check whether the DLL has populated the
            cbmove struct.  Use the FIRST non-zero cbmove as the authoritative
            move source (raw board coordinates — no PDN translation needed).
        4.  If cbmove is never populated, fall back to the PV text string and
            convert using standard PDN square numbers, validating the from-square
            against the eval_board to skip stale hash-table lines.
        """
        rotated    = (current_player == RED)
        eval_board = self._rotate_board(board) if rotated else board
        cb_board   = self._convert_board(eval_board)

        output   = ctypes.create_string_buffer(1024)
        playnow  = ctypes.c_int(0)
        cbmove   = CBmove()
        cb_color = CB_BLACK

        # KR's mode selector and CB result semantics:
        #   maxtime > 0 → time-based search; res=3 (DRAW/UNKNOWN) is returned
        #                 on every yield until maxtime elapses, so we loop and
        #                 KR's TT lets each subsequent call CONTINUE the search.
        #   maxtime < 0 → fixed-depth search to |maxtime| plies; one call
        #                 already completes the requested depth, and a follow-up
        #                 call would RESTART the same depth-N search (infinite
        #                 loop on drawn positions where res stays 3). So in
        #                 fixed-depth mode we must NOT loop.
        if target_depth > 0:
            kr_maxtime    = float(-target_depth)
            safety_cap    = max(5.0, time_budget)   # only a hard runaway guard
            single_call   = True
        else:
            kr_maxtime    = float(time_budget)
            safety_cap    = time_budget * 3
            single_call   = False

        old_cwd = os.getcwd()
        os.chdir(os.path.dirname(self.dll_path))

        last_val   = 0.0
        last_pv    = ""
        last_depth = 0
        best_cbmove: Optional[CBmove] = None   # first non-zero cbmove seen
        best_pv     = ""                        # last PV whose from-sq is valid
        t_start    = time.perf_counter()

        try:
            res = 3
            while res == 3:
                if time.perf_counter() - t_start > safety_cap:
                    break

                res = self.dll.getmove(
                    cb_board,
                    ctypes.c_int(cb_color),
                    ctypes.c_double(kr_maxtime),
                    output,
                    ctypes.byref(playnow),
                    0, 0,                     # info=0; moreinfo unused by KR
                    ctypes.byref(cbmove),
                )
                out_str = output.value.decode('utf-8', errors='ignore')

                # ── print raw cbmove on every call so we can verify the struct ──
                print(
                    f"  [cbmove-raw] res={res}  "
                    f"from=({cbmove.from_sq.x},{cbmove.from_sq.y})  "
                    f"to=({cbmove.to_sq.x},{cbmove.to_sq.y})  "
                    f"jumps={cbmove.jumps}",
                    file=sys.stderr, flush=True,
                )

                # Store the first cbmove that has non-zero coordinates
                if (best_cbmove is None and
                        not (cbmove.from_sq.x == 0 and cbmove.from_sq.y == 0 and
                             cbmove.to_sq.x == 0 and cbmove.to_sq.y == 0)):
                    # Deep-copy the struct (ctypes objects are mutable)
                    best_cbmove = CBmove()
                    best_cbmove.jumps    = cbmove.jumps
                    best_cbmove.newpiece = cbmove.newpiece
                    best_cbmove.oldpiece = cbmove.oldpiece
                    best_cbmove.from_sq.x = cbmove.from_sq.x
                    best_cbmove.from_sq.y = cbmove.from_sq.y
                    best_cbmove.to_sq.x   = cbmove.to_sq.x
                    best_cbmove.to_sq.y   = cbmove.to_sq.y
                    for i in range(12):
                        best_cbmove.path[i].x    = cbmove.path[i].x
                        best_cbmove.path[i].y    = cbmove.path[i].y
                        best_cbmove.delpath[i].x = cbmove.delpath[i].x
                        best_cbmove.delpath[i].y = cbmove.delpath[i].y

                if 'Illegal position' in out_str or 'No moves' in out_str:
                    break

                val_match   = re.search(r'value=([-\d.]+)', out_str)
                pv_match    = re.search(r'pv\s+([0-9x-]+)', out_str)
                depth_match = re.search(r'depth\s+(\d+)', out_str)

                if val_match:
                    v = float(val_match.group(1))
                    if v != 3999:
                        last_val = v
                if pv_match:
                    last_pv = pv_match.group(1)
                    # Track the last PV whose from-sq is valid on the eval_board
                    candidate = self._pv_to_engine_path(last_pv, eval_board, rotated=False)
                    if candidate is not None:
                        best_pv = last_pv
                if depth_match:
                    last_depth = int(depth_match.group(1))

                # Fixed-depth mode: one call completes the depth-N search.
                # Calling getmove again would just re-run the same depth-N
                # search and never make progress (e.g. when res stays 3 for
                # a drawn line), so we stop after the first iteration.
                if single_call:
                    break

        finally:
            os.chdir(old_cwd)

        if rotated:
            last_val = -last_val

        # ── path extraction ─────────────────────────────────────────────────────
        engine_move = None

        # 1. CBmove struct (preferred — raw board coordinates, no PDN translation)
        if best_cbmove is not None:
            cbmove_path = self._cbmove_to_engine_path(best_cbmove, rotated)
            if cbmove_path is not None:
                print(
                    f"  [cbmove-convert] rotated={rotated}  "
                    f"from=({best_cbmove.from_sq.x},{best_cbmove.from_sq.y})→{cbmove_path[0]}  "
                    f"to=({best_cbmove.to_sq.x},{best_cbmove.to_sq.y})→{cbmove_path[-1]}",
                    file=sys.stderr, flush=True,
                )
                engine_move = {
                    "type":     "jump" if best_cbmove.jumps > 0 else "simple",
                    "path":     cbmove_path,
                    "captured": [],
                }

        # 2. PV text fallback — use last validated PV (from-sq confirmed on eval_board)
        if engine_move is None:
            pv_to_use = best_pv or last_pv
            if pv_to_use:
                path = self._pv_to_engine_path(pv_to_use, eval_board, rotated)
                if path is not None:
                    first_move = pv_to_use.split(' ')[0]
                    print(
                        f"  [pv-fallback] pv={pv_to_use!r}  "
                        f"first_move={first_move!r}  "
                        f"path={path}",
                        file=sys.stderr, flush=True,
                    )
                    engine_move = {
                        "type":     "jump" if 'x' in first_move.lower() else "simple",
                        "path":     path,
                        "captured": [],
                    }

        return {
            "score":       last_val,
            "depth":       last_depth,
            "pdn_move":    last_pv.split(' ')[0] if last_pv else "",
            "engine_move": engine_move,
        }
