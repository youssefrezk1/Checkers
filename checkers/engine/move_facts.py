# engine/move_facts.py

from checkers.engine.board import (
    RED, BLACK, RED_KING, BLACK_KING,
    BOARD_SIZE, in_bounds, is_king, is_own_piece, is_opponent_piece
)
from checkers.engine.rules import apply_move, get_all_legal_moves, get_all_moves_unfiltered


def count_pieces(board, player):
    """
    Counts how many pieces (regular + kings) the player has on the board.
    Returns a dict with total, regular, and king counts.
    """
    regular_count = 0
    king_count = 0

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]

            if player == RED:
                if piece == RED:
                    regular_count += 1
                elif piece == RED_KING:
                    king_count += 1

            elif player == BLACK:
                if piece == BLACK:
                    regular_count += 1
                elif piece == BLACK_KING:
                    king_count += 1

    total = regular_count + king_count

    return {
        "total": total,
        "regular": regular_count,
        "kings": king_count
    }


def results_in_promotion(board, move):
    """
    Checks if applying this move promotes a regular piece to a king.
    Compares the board before and after the move to detect a new king.
    """
    piece_at_start = board[move["path"][0][0]][move["path"][0][1]]

    if piece_at_start != RED and piece_at_start != BLACK:
        return False

    board_after = apply_move(board, move)

    final_row = move["path"][-1][0]
    final_col = move["path"][-1][1]

    piece_at_end = board_after[final_row][final_col]

    if piece_at_end == RED_KING or piece_at_end == BLACK_KING:
        return True

    return False


def _get_threatened_squares(board, opponent):
    """
    Returns the set of (row, col) squares that the opponent can capture
    on their next move from the given board position.

    This is used for board-wide threat detection — NOT just for the
    piece that just moved. Any of our pieces sitting on a square in
    this set is in immediate danger of being eaten next turn.
    """
    threatened: set[tuple[int, int]] = set()
    opp_moves = get_all_legal_moves(board, opponent)
    for m in opp_moves:
        if m["type"] == "jump":
            for cap in m["captured"]:
                threatened.add((cap[0], cap[1]))
    return threatened

def _max_opponent_jump_captures(board_after, opponent):
    """
    Returns the maximum number of pieces the opponent can capture
    in a single immediate jump reply after our move.
    """
    opp_moves = get_all_legal_moves(board_after, opponent)
    jump_moves = [m for m in opp_moves if m.get("type") == "jump"]
    if not jump_moves:
        return 0
    return max(len(m.get("captured", [])) for m in jump_moves)


def _forced_opponent_jump_reply(board_after, opponent):
    """
    Returns True if all legal opponent replies after our move are jumps.
    This means the opponent has a tactically forced capture reply.
    """
    opp_moves = get_all_legal_moves(board_after, opponent)
    if not opp_moves:
        return False
    return all(m.get("type") == "jump" for m in opp_moves)


def _creates_immediate_threat(board_after, current_player):
    """
    Returns True if after our move, we would have at least one jump available
    on our next turn from this resulting position.
    This is only a proxy for pressure creation, not a full search.
    """
    our_moves_after = get_all_legal_moves(board_after, current_player)
    return any(m.get("type") == "jump" for m in our_moves_after)

def _shot_sequence_available(board_after, current_player):
    """
    True if after our move, we already have at least one jump available
    from the resulting board. This is a tactical pressure signal.
    """
    our_moves_after = get_all_legal_moves(board_after, current_player)
    return any(m.get("type") == "jump" for m in our_moves_after)


def _forces_exchange_profile(board_after, current_player, opponent,
                             opp_moves_unfiltered=None):
    """
    Heuristic forcing-exchange detector.

    Returns:
        (forces_exchange: bool, forces_exchange_count: int)

    Uses the unfiltered move list so that mandatory capture alone does not
    make positions with many simple moves look "forced."
    """
    opp_moves = (opp_moves_unfiltered if opp_moves_unfiltered is not None
                 else get_all_moves_unfiltered(board_after, opponent))
    if not opp_moves:
        return False, 0

    jump_replies = [m for m in opp_moves if m.get("type") == "jump"]
    if not jump_replies:
        return False, 0

    forces_exchange = (len(jump_replies) == len(opp_moves)) or (len(opp_moves) <= 2)
    return forces_exchange, len(jump_replies)


def _two_for_one_profile(board_after, current_player):
    """
    Heuristic two-for-one detector.

    Returns:
        (two_for_one_potential: bool, two_for_one_score: int)

    Positive if from the resulting board we already have a jump sequence
    next turn that captures 2 or more pieces.
    """
    our_moves_after = get_all_legal_moves(board_after, current_player)
    jump_moves = [m for m in our_moves_after if m.get("type") == "jump"]

    best_capture_len = 0
    for jm in jump_moves:
        best_capture_len = max(best_capture_len, len(jm.get("captured", [])))

    two_for_one_potential = best_capture_len >= 2
    two_for_one_score = best_capture_len if best_capture_len >= 2 else 0
    return two_for_one_potential, two_for_one_score


def _restriction_profile(board_after, current_player, opponent,
                         opp_moves_unfiltered=None):
    """
    Structural restriction detector.

    Returns:
        (restriction_score: int, frozen_enemy_pieces: int)

    Counts how many opponent pieces have no move (simple OR jump) starting
    from their current square.  Uses the unfiltered move list so that pieces
    temporarily blocked only by mandatory capture are NOT counted as frozen.
    """
    opp_moves = (opp_moves_unfiltered if opp_moves_unfiltered is not None
                 else get_all_moves_unfiltered(board_after, opponent))

    movable_starts = set()
    for m in opp_moves:
        start = m["path"][0]
        movable_starts.add((start[0], start[1]))

    frozen_enemy_pieces = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = board_after[r][c]
            if is_opponent_piece(piece, current_player) and (r, c) not in movable_starts:
                frozen_enemy_pieces += 1

    restriction_score = frozen_enemy_pieces
    return restriction_score, frozen_enemy_pieces


def _king_squares(board, player):
    """
    Returns list of (row, col) positions of player's kings on the board.
    """
    king_piece = RED_KING if player == RED else BLACK_KING
    return [
        (r, c)
        for r in range(BOARD_SIZE)
        for c in range(BOARD_SIZE)
        if board[r][c] == king_piece
    ]


def _king_escape_squares(board, player):
    """
    Count total legal destination squares available to player's kings.
    Uses get_all_legal_moves filtered to king-starting moves.
    Lower = more restricted.
    """
    moves = get_all_legal_moves(board, player)
    king_piece = RED_KING if player == RED else BLACK_KING
    destinations: set[tuple[int, int]] = set()
    for m in moves:
        sr, sc = m["path"][0]
        if board[sr][sc] == king_piece:
            last = m["path"][-1]
            destinations.add((last[0], last[1]))
    return len(destinations)


def _corner_trap_pressure(board, opponent):
    """
    Heuristic: reward positions where opponent kings are near edges/corners
    with few escape diagonals. Returns int 0..N.

    A king on a corner has 1 escape diagonal.
    A king on an edge (non-corner) has 2.
    A king in the interior has up to 4.
    We reward confinement — the fewer diagonals, the higher the pressure.
    """
    king_piece = RED_KING if opponent == RED else BLACK_KING
    pressure = 0
    dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != king_piece:
                continue
            free_diags = sum(
                1 for dr, dc in dirs
                if in_bounds(r + dr, c + dc)
            )
            # Corner: free_diags==2 → pressure 3
            # Edge non-corner: free_diags==3 → pressure 1
            # Interior: free_diags==4 → pressure 0
            if free_diags == 2:
                pressure += 3
            elif free_diags == 3:
                pressure += 1
    return pressure


def _king_coordination_score(board, player):
    """
    Reward our kings being close enough to cooperate.
    Uses average pairwise Chebyshev distance — lower distance = better coordination.
    Returns int: 0 if 0 or 1 kings, else a score where closer = higher.
    """
    positions = _king_squares(board, player)
    if len(positions) < 2:
        return 0
    total_dist = 0
    pairs = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            r1, c1 = positions[i]
            r2, c2 = positions[j]
            dist = max(abs(r1 - r2), abs(c1 - c2))  # Chebyshev
            total_dist += dist
            pairs += 1
    avg_dist = total_dist / pairs
    # Score: 7 = max board distance, 0 = same square (impossible after move)
    # We reward avg_dist <= 4 (close enough to cooperate)
    if avg_dist <= 2:
        return 3
    elif avg_dist <= 4:
        return 1
    return 0


def _king_distance_pressure(board_before, board_after, move, current_player, opponent, is_safe):
    """
    For king moves only: reward reducing Chebyshev distance to nearest
    vulnerable opponent piece (king or man).
    Only fires when the move is safe (is_safe=True).
    Returns int 0..3.
    """
    if not is_safe:
        return 0

    start_row, start_col = move["path"][0]
    final_row, final_col = move["path"][-1]

    # Check if the moving piece is a king on the board_before
    moving_piece = board_before[start_row][start_col]
    if not is_king(moving_piece):
        return 0

    # Prefer opponent kings as targets; fall back to all opponent pieces
    targets = [
        (r, c)
        for r in range(BOARD_SIZE)
        for c in range(BOARD_SIZE)
        if is_king(board_after[r][c]) and is_opponent_piece(board_after[r][c], current_player)
    ]
    if not targets:
        targets = [
            (r, c)
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if is_opponent_piece(board_after[r][c], current_player)
        ]

    if not targets:
        return 0

    def chebyshev(r1, c1, r2, c2):
        return max(abs(r1 - r2), abs(c1 - c2))

    dist_before = min(chebyshev(start_row, start_col, tr, tc) for tr, tc in targets)
    dist_after  = min(chebyshev(final_row, final_col, tr, tc) for tr, tc in targets)

    improvement = dist_before - dist_after
    if improvement >= 3:
        return 3
    elif improvement >= 2:
        return 2
    elif improvement >= 1:
        return 1
    return 0


def _weakens_king_row(board, move, current_player):
    """
    Returns True if this move vacates a king-row square without
    compensation (capture, promotion, or blocking a threat).

    King row for RED = row 7. King row for BLACK = row 0.
    Moving a piece off the king row weakens back-rank structure.
    Only penalized in opening-like positions (many pieces, no kings yet).
    """
    start_row = move["path"][0][0]
    captures  = len(move["captured"])
    promotion = False

    # Check for promotion
    piece_at_start = board[move["path"][0][0]][move["path"][0][1]]
    if piece_at_start in (RED, BLACK):
        final_row = move["path"][-1][0]
        if current_player == RED and final_row == 0:
            promotion = True
        elif current_player == BLACK and final_row == 7:
            promotion = True

    king_row = 7 if current_player == RED else 0
    if start_row != king_row:
        return False

    # If it's a capture or promotion, it's compensated — not a weakness
    if captures > 0 or promotion:
        return False

    # Count how many of our pieces remain on king row after move
    remaining_king_row = 0
    for c in range(BOARD_SIZE):
        piece = board[king_row][c]
        if is_own_piece(piece, current_player):
            if (king_row, c) != (move["path"][0][0], move["path"][0][1]):
                remaining_king_row += 1

    # Penalize if we're breaking the last or second-to-last king-row piece
    # without compensation
    if remaining_king_row <= 2:
        return True

    return False


def _opens_long_diagonal_risk(board, move, current_player):
    """
    Returns True if this move vacates a square on a critical long diagonal,
    opening a lane that the opponent can exploit.

    Long diagonals in checkers: the two main diagonal spines of the board.
    For RED: the key long diagonal runs through (5,0),(4,1),(3,2),(2,3),(1,4),(0,5)
             and                                 (5,2),(4,3),(3,4),(2,5),(1,6),(0,7)
    For BLACK: mirror image.

    We flag the move if:
    - it vacates a square on one of these diagonals
    - the resulting square is now accessible to opponent jump lines
    - no capture compensates
    """
    start_row, start_col = move["path"][0]
    captures = len(move["captured"])

    if captures > 0:
        return False  # tactical compensation

    # Long diagonal squares — both main diagonal spines
    # Diagonal A: top-left to bottom-right family through col offsets
    # Diagonal B: other family
    # Practically: flag squares that form a diagonal from edge to edge
    # with length >= 6 (only the two longest diagonals)

    def _diagonal_family(r, c):
        """Return length of the diagonal through (r,c) in both directions."""
        # Down-right diagonal: r-c = constant
        len_dr = 0
        rr, cc = r, c
        while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
            len_dr += 1
            rr += 1
            cc += 1
        rr, cc = r - 1, c - 1
        while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
            len_dr += 1
            rr -= 1
            cc -= 1

        # Down-left diagonal: r+c = constant
        len_dl = 0
        rr, cc = r, c
        while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
            len_dl += 1
            rr += 1
            cc -= 1
        rr, cc = r - 1, c + 1
        while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
            len_dl += 1
            rr -= 1
            cc += 1

        return max(len_dr, len_dl)

    # Only flag if vacating a square on a long diagonal (length >= 6)
    if _diagonal_family(start_row, start_col) < 6:
        return False

    # Check if the vacated square is now accessible to opponent jump approach
    # Heuristic: does the opponent have a piece within 2 diagonal steps
    # of the vacated square in the direction of the long diagonal?
    opponent = BLACK if current_player == RED else RED
    dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dr, dc in dirs:
        r1, c1 = start_row + dr, start_col + dc
        r2, c2 = start_row + 2 * dr, start_col + 2 * dc
        if (
            in_bounds(r1, c1) and in_bounds(r2, c2)
            and is_opponent_piece(board[r1][c1], current_player)
        ):
            # Opponent is one diagonal step away from vacated square —
            # they now have an open approach lane
            return True

    return False


def _creates_forced_capture_risk(board_after, current_player, opponent):
    """
    Heuristic opening trap risk:
    after our move, opponent gets multiple jump options or multiple threatened
    targets, creating a likely forced-capture tactical structure.
    """
    opp_moves_after = get_all_legal_moves(board_after, opponent)
    opp_jumps = [m for m in opp_moves_after if m["type"] == "jump"]

    if len(opp_jumps) == 0:
        return False

    targeted_our_pieces: set[tuple[int, int]] = set()
    for jm in opp_jumps:
        for cap in jm["captured"]:
            targeted_our_pieces.add((cap[0], cap[1]))

    # Only treat it as real trap-risk if the pressure is clearly branching
    return len(targeted_our_pieces) >= 2 or len(opp_jumps) >= 2



def _edge_confinement_score(board, opponent):
    """
    Measures how confined opponent kings are to the board edge.
    Edge squares have fewer diagonal escape options.
    Returns int: higher = more confined.
    """
    king_piece = RED_KING if opponent == RED else BLACK_KING
    score = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != king_piece:
                continue
            # Edge = row 0, row 7, col 0, col 7
            on_edge = (r == 0 or r == 7 or c == 0 or c == 7)
            # Double corner zones: (0,0),(0,2),(1,1),(2,0) and mirrors
            in_double_corner_zone = (
                (r <= 2 and c <= 2) or
                (r <= 2 and c >= 5) or
                (r >= 5 and c <= 2) or
                (r >= 5 and c >= 5)
            )
            if in_double_corner_zone:
                score += 3
            elif on_edge:
                score += 2
    return score


def _exchange_pressure_score(board, move, current_player, opponent,
                              material_advantage, our_kings, opp_kings,
                              is_safe, board_after):
    """
    Rewards moves that increase king-for-king exchange likelihood
    when we are ahead. Heuristic only.

    Fires when:
    - we are ahead in material or king count
    - move is safe
    - after the move, we have a jump available that captures a king
    - or opponent has very few replies

    Returns int 0..4.
    """
    if not is_safe:
        return 0
    if material_advantage <= 0 and our_kings <= opp_kings:
        return 0

    score = 0
    # Can we capture a king next turn?
    our_jumps = [m for m in get_all_legal_moves(board_after, current_player)
                 if m["type"] == "jump"]
    for jm in our_jumps:
        for cap in jm["captured"]:
            if is_king(board_after[cap[0]][cap[1]]):
                score += 2
                break

    # Opponent has very few replies → pressure is working
    opp_replies = get_all_legal_moves(board_after, opponent)
    if len(opp_replies) <= 2:
        score += 2
    elif len(opp_replies) <= 4:
        score += 1

    return min(score, 4)


def _bridge_potential_score(board_after, current_player, final_row, final_col):
    """
    Heuristic bridge/support shape detector.

    A useful bridge shape: our king lands on a square where it:
    - supports another of our pieces diagonally (2-step away)
    - and the intermediate square is empty or controlled by us

    This approximates the endgame principle of diagonal support
    that controls escape squares and supports bottling.

    Returns int 0..3.
    """
    score = 0
    dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dr, dc in dirs:
        r1, c1 = final_row + dr, final_col + dc
        r2, c2 = final_row + 2*dr, final_col + 2*dc
        if not (in_bounds(r1, c1) and in_bounds(r2, c2)):
            continue
        # 2-step diagonal has our piece at far end
        if not is_own_piece(board_after[r2][c2], current_player):
            continue
        # Intermediate square is empty (open diagonal lane) or ours
        intermediate = board_after[r1][c1]
        if intermediate == 0 or is_own_piece(intermediate, current_player):
            score += 1

    return min(score, 3)

def _opponent_safe_reply_count(board_after, current_player, opponent,
                               opp_moves_unfiltered=None):
    """
    Counts how many opponent replies after our move are genuinely comfortable.

    A reply is considered safe only if:
    - the moved opponent piece is not immediately threatened by us
    - and the opponent does not leave any immediate forced jump reply for us
    - and the opponent does not leave multiple of their own pieces threatened

    Uses the unfiltered move list so that simple-move escapes are counted
    even when mandatory capture forces the opponent to jump first.
    """
    opp_moves = (opp_moves_unfiltered if opp_moves_unfiltered is not None
                 else get_all_moves_unfiltered(board_after, opponent))
    safe_count = 0

    for opp_move in opp_moves:
        opp_board_after = apply_move(board_after, opp_move)

        threatened_by_us = _get_threatened_squares(opp_board_after, current_player)
        final_row, final_col = opp_move["path"][-1]

        moved_piece_threatened = (final_row, final_col) in threatened_by_us

        opp_pieces_threatened_after_reply = sum(
            1
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if is_own_piece(opp_board_after[r][c], opponent) and (r, c) in threatened_by_us
        )

        our_replies = get_all_legal_moves(opp_board_after, current_player)
        our_jump_replies = [m for m in our_replies if m.get("type") == "jump"]
        forced_reply_for_us = len(our_replies) > 0 and len(our_jump_replies) == len(our_replies)

        if (
            not moved_piece_threatened
            and not forced_reply_for_us
            and opp_pieces_threatened_after_reply <= 1
        ):
            safe_count += 1

    return safe_count

def _anti_shuffle_penalty(board_before, board_after, move, current_player,
                          king_escape_reduction, edge_conf_delta,
                          exchange_pressure, king_dist_pressure,
                          restriction_score):
    """
    Penalizes king moves that make no measurable progress.

    A king move is a shuffle if:
    - it is a king moving
    - it does NOT improve any of: escape reduction, edge confinement,
      exchange pressure, distance pressure, restriction
    - the destination square is on the same diagonal as the source
      (oscillation between two squares)

    Returns int: 0 (no penalty) or -2(shuffle detected).
    """
    start_row, start_col = move["path"][0]
    final_row, final_col = move["path"][-1]
    moving_piece = board_before[start_row][start_col]

    if not is_king(moving_piece):
        return 0

    # Check if any pressure signal improved
    any_progress = (
        king_escape_reduction > 0 or
        edge_conf_delta > 0 or
        exchange_pressure > 0 or
        king_dist_pressure > 0 or
        restriction_score > 0
    )
    if any_progress:
        return 0

    # Same diagonal family = classic oscillation
    # Two squares are on the same diagonal if |dr| == |dc|
    dr = abs(final_row - start_row)
    dc = abs(final_col - start_col)
    on_same_diagonal = (dr == dc)

    if on_same_diagonal:
        return -2  # no-progress same-diagonal king move
    # No improvement, different diagonal — no penalty (acceptable quiet move)
    return 0


def _double_corner_smokeout_score(board_after, opponent, edge_conf_after,
                                   king_escape_after, restriction_score,
                                   bridge_score):
    """
    Heuristic: how much does this position pressure the opponent king
    out of a double-corner defensive zone?

    Double corners: squares (0,0),(0,2),(1,1),(2,0) and their mirrors.
    If an opponent king is in this zone, combine:
    - edge confinement (already in double corner = worse)
    - escape reduction (fewer exits)
    - bridge/support coverage
    - restriction (frozen opponent pieces)

    Returns int 0..6.
    """
    king_piece = RED_KING if opponent == RED else BLACK_KING

    opp_in_double_corner = False
    double_corner_squares = {
        (0, 0), (0, 2), (1, 1), (2, 0),
        (0, 5), (0, 7), (1, 6), (2, 7),
        (5, 0), (6, 1), (7, 0), (7, 2),
        (5, 7), (6, 6), (7, 5), (7, 7),
    }
    for r, c in double_corner_squares:
        if board_after[r][c] == king_piece:
            opp_in_double_corner = True
            break

    if not opp_in_double_corner:
        return 0

    score = 0
    if king_escape_after <= 2:
        score += 2
    elif king_escape_after <= 4:
        score += 1
    if edge_conf_after >= 3:
        score += 1
    if restriction_score > 0:
        score += 1
    if bridge_score >= 2:
        score += 1

    return min(score, 6)


def _simplification_value(material_advantage, our_kings, opp_kings,
                           total_pieces, is_safe, restriction_score,
                           king_coordination, exchange_pressure,
                           opponent_mobility_after):
    """
    Rewards positions that move toward a simpler, cleaner winning balance.

    Fires when:
    - we are ahead (material or king count)
    - position is endgame-like (few pieces)
    - the move is safe
    - we are restricting opponent and coordinating our kings

    Returns int 0..5.
    """
    if not is_safe:
        return 0
    if material_advantage <= 0 and our_kings <= opp_kings:
        return 0

    score = 0

    # Restriction = opponent options narrowing
    score += min(restriction_score, 2)

    # King coordination = we can cooperate to convert
    score += min(king_coordination, 2)

    # Exchange pressure helps when ahead
    score += min(exchange_pressure, 2)

    # Opponent mobility low = simplification working
    if opponent_mobility_after <= 3:
        score += 2
    elif opponent_mobility_after <= 5:
        score += 1

    # Cap at 5
    return min(score, 5)


def compute_move_facts(board, move, current_player):
    """
    Computes structured facts about a move before it is executed.
    These facts are passed to the LLM agents so they can reason
    about the quality and consequences of each candidate move.

    All facts are from the perspective of the CURRENT PLAYER.

    ── BASIC MOVE INFO ──────────────────────────────────────────────
        move_type           : "simple" or "jump"
        piece_type_moving   : "regular" or "king"
        path_length         : number of squares visited including start

    ── CAPTURE DETAILS ──────────────────────────────────────────────
        captures_count      : total opponent pieces captured this turn
        jump_count          : same as captures_count (explicit alias)
        is_multi_jump       : True if captures_count > 1
        kings_captured      : how many captured pieces were kings
        regulars_captured   : how many captured pieces were regulars

    ── PROMOTION ────────────────────────────────────────────────────
        results_in_king     : True if this move promotes one of our pieces
        near_promotion      : True if our piece lands one step from promotion
                              without promoting on this move

    ── PIECE COUNTS ─────────────────────────────────────────────────
        our_pieces_before   : our counts before move (total, regular, kings)
        our_pieces_after    : our counts after move
        opp_pieces_before   : opponent counts before move
        opp_pieces_after    : opponent counts after move
        net_gain            : opponent pieces removed this turn (0 for simple)
        material_advantage  : our_after.total - opp_after.total

    ── STRATEGIC CONTEXT ────────────────────────────────────────────
        center_control      : True if our piece lands in rows 3-4, cols 2-5

        opponent_can_recapture : True if the opponent can immediately capture
                              ANY of our pieces on their next turn after this
                              move is made. This is board-wide — it checks the
                              moved piece AND all existing pieces that remain
                              on the board. A move that leaves an existing
                              piece hanging is just as dangerous as one where
                              the moved piece itself is immediately taken.

                              IMPORTANT CHANGE FROM ORIGINAL: the original
                              only checked if the just-moved piece could be
                              recaptured. This caused the ranker to rate moves
                              like (6,3)→(5,2) as "safe" even when an existing
                              piece at (4,5) was about to be eaten by BLACK
                              at (3,6). Now we check ALL our pieces on the
                              board after the move.

        recapturable_piece_is_king : True when opponent_can_recapture=True and
                              the piece at risk is a king. This lets the ranker
                              weight the danger proportionally — losing a king
                              is far more costly than losing a regular piece.

        leaves_piece_isolated : True if the MOVED piece ends up with no
                              friendly diagonal neighbors after the move

        any_piece_isolated  : True if ANY of our pieces on the board after
                              the move has no friendly diagonal neighbors.
                              A move that creates isolated pieces elsewhere
                              on the board (not just the moved piece) weakens
                              our overall formation even if the moved piece
                              itself is well-connected.

        blocks_opponent_landing : True if our piece lands on a square that
                              the opponent would have used as a landing square
                              for a jump. This means we are actively denying
                              the opponent a capture they could have made.
                              A move that blocks an opponent landing square is
                              strategically valuable even if it does nothing
                              else — it removes a threat without us having to
                              capture the threatening piece directly.

        opponent_near_promotion : True if after our move at least one opponent
                              regular piece sits one step from their promotion row

        opponent_jump_count : how many jump moves the opponent has after our move.
                              Zero is ideal. High values mean we opened dangerous
                              lines for the opponent.

        our_pieces_threatened_before : how many of our pieces were threatened
                              (capturable by opponent) BEFORE this move was made.
                              Combined with opponent_can_recapture this tells you
                              whether the move improved, maintained, or worsened
                              our safety situation.

        our_pieces_threatened_after : how many of our pieces are threatened
                              (capturable by opponent) AFTER this move is made.
                              This is the key board-wide safety score for this move.
                              Lower is better. A move that reduces threats from 2
                              to 0 is excellent even if it gains no material.
                              The ranker should strongly prefer moves with a lower
                              our_pieces_threatened_after value.
    
        moved_piece_is_threatened : True if the moved piece's final square itself
                              is among the opponent's immediate capture targets
                              after this move. This distinguishes "some piece is
                              hanging" from "this move walked directly into danger."

        max_opponent_jump_captures : maximum number of our pieces the opponent
                              can capture in a single immediate reply after this
                              move. Higher = more severe tactical punishment.

        forced_opponent_jump_reply : True if all legal opponent replies after
                              this move are jumps. This means we have allowed a
                              tactically forced capture sequence.
    
    
    """
    

    if current_player == RED:
        opponent = BLACK
    else:
        opponent = RED
    
    opponent_moves_before = get_all_legal_moves(board, opponent)
    opponent_moves_before_unfiltered = get_all_moves_unfiltered(board, opponent)
    opponent_mobility_before = len(opponent_moves_before_unfiltered)
    board_after = apply_move(board, move)

    our_before = count_pieces(board, current_player)
    our_after = count_pieces(board_after, current_player)
    opp_before = count_pieces(board, opponent)
    opp_after = count_pieces(board_after, opponent)

    captures_count = len(move["captured"])
    jump_count = captures_count
    is_multi_jump = captures_count > 1
    promotion = results_in_promotion(board, move)
    path_length = len(move["path"])
    net_gain = opp_before["total"] - opp_after["total"]
    material_advantage = our_after["total"] - opp_after["total"]

    # Piece type making this move
    start_row = move["path"][0][0]
    start_col = move["path"][0][1]
    moving_piece = board[start_row][start_col]
    piece_type_moving = "king" if is_king(moving_piece) else "regular"

    # Types of captured pieces
    kings_captured = 0
    regulars_captured = 0
    for cap_pos in move["captured"]:
        cap_piece = board[cap_pos[0]][cap_pos[1]]
        if is_king(cap_piece):
            kings_captured += 1
        else:
            regulars_captured += 1

    # Final position of our moved piece
    final_row = move["path"][-1][0]
    final_col = move["path"][-1][1]

    # Center control
    in_center = (3 <= final_row <= 4 and 2 <= final_col <= 5)





   # Near promotion — only meaningful for regular pieces, never for kings
    if piece_type_moving == "king":
        near_promotion = False
    elif current_player == RED:
        near_promotion = (final_row == 1 and not promotion)
    else:
        near_promotion = (final_row == 6 and not promotion)
    # ── Board-wide threat analysis (KEY FIX) ─────────────────────────────────
    #
    # BEFORE this move: which of our squares can the opponent already jump to?
    threatened_before: set[tuple[int, int]] = _get_threatened_squares(board, opponent)
    our_pieces_threatened_before = sum(
        1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
        if is_own_piece(board[r][c], current_player) and (r, c) in threatened_before
    )

    # AFTER this move: which of our squares can the opponent jump to?
    threatened_after: set[tuple[int, int]] = _get_threatened_squares(board_after, opponent)
    our_pieces_threatened_after = sum(
        1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
        if is_own_piece(board_after[r][c], current_player) and (r, c) in threatened_after
    )

    # opponent_can_recapture: True if ANY of our pieces is threatened after the move.
    # This replaces the original which only checked the moved piece's final square.
    opponent_can_recapture = our_pieces_threatened_after > 0

    # New richer danger facts
    moved_piece_is_threatened = (final_row, final_col) in threatened_after

    # Is any threatened piece a king? (Proportional danger signal for the ranker)
    recapturable_piece_is_king = False
    if opponent_can_recapture:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = board_after[r][c]
                if is_own_piece(piece, current_player) and (r, c) in threatened_after:
                    if is_king(piece):
                        recapturable_piece_is_king = True
                        break

    max_opponent_jump_captures = _max_opponent_jump_captures(board_after, opponent)
    forced_opponent_jump_reply = _forced_opponent_jump_reply(board_after, opponent)

    


    # ── blocks_opponent_landing ───────────────────────────────────────────────
    # Does our piece land on a square the opponent would have used as a
    # landing square for a capture on THIS turn (before we moved)?
    # We compute opponent landing squares from the ORIGINAL board, then check
    # if our final position matches any of them.
    opponent_landing_squares: set[tuple[int, int]] = set()
    for m in opponent_moves_before:
        if m["type"] == "jump":
            # Landing square is the last square in the path
            last = m["path"][-1]
            opponent_landing_squares.add((last[0], last[1]))

    blocks_opponent_landing = (final_row, final_col) in opponent_landing_squares

    # ── Isolation checks ──────────────────────────────────────────────────────
    adjacent_directions = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]

    # Isolation of the moved piece specifically
    has_friendly_neighbor = False
    for dr, dc in adjacent_directions:
        nr, nc = final_row + dr, final_col + dc
        if in_bounds(nr, nc) and is_own_piece(board_after[nr][nc], current_player):
            has_friendly_neighbor = True
            break
    leaves_piece_isolated = not has_friendly_neighbor

    # Board-wide isolation: does any of our pieces have no friendly neighbor?
    any_piece_isolated = False
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if not is_own_piece(board_after[r][c], current_player):
                continue
            isolated = True
            for dr, dc in adjacent_directions:
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc) and is_own_piece(board_after[nr][nc], current_player):
                    isolated = False
                    break
            if isolated:
                any_piece_isolated = True
                break


    # ── Opponent near promotion ───────────────────────────────────────────────
    opponent_near_promotion = False
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board_after[row][col]
            if is_opponent_piece(piece, current_player) and not is_king(piece):
                if current_player == RED and row == 6:
                    opponent_near_promotion = True
                elif current_player == BLACK and row == 1:
                    opponent_near_promotion = True
    
    opponent_moves_after = get_all_legal_moves(board_after, opponent)
    opponent_moves_after_unfiltered = get_all_moves_unfiltered(board_after, opponent)
    opponent_mobility_after = len(opponent_moves_after_unfiltered)
    mobility_reduction = opponent_mobility_before - opponent_mobility_after

    forces_exchange, forces_exchange_count = _forces_exchange_profile(
        board_after, current_player, opponent,
        opp_moves_unfiltered=opponent_moves_after_unfiltered,
    )
    two_for_one_potential, two_for_one_score = _two_for_one_profile(
        board_after, current_player
    )
    restriction_score, frozen_enemy_pieces = _restriction_profile(
        board_after, current_player, opponent,
        opp_moves_unfiltered=opponent_moves_after_unfiltered,
    )
    opponent_safe_reply_count = _opponent_safe_reply_count(
        board_after, current_player, opponent,
        opp_moves_unfiltered=opponent_moves_after_unfiltered,
    )
    creates_immediate_threat = _creates_immediate_threat(board_after, current_player)
    shot_sequence_available = _shot_sequence_available(board_after, current_player)
    creates_real_trap = (
        opponent_safe_reply_count <= 2
        and mobility_reduction >= 1
        and (
            creates_immediate_threat
            or shot_sequence_available
            or forces_exchange
            or blocks_opponent_landing
        )
    )
    opponent_jump_count = sum(
        1 for m in opponent_moves_after if m["type"] == "jump"
    )


# ── Opening structure features ────────────────────────────────────────────
    # Only meaningful in opening-like positions: many pieces, no kings yet.
    total_pieces_before = our_before["total"] + opp_before["total"]
    no_kings_yet = (our_before["kings"] + opp_before["kings"]) == 0
    is_opening_like = total_pieces_before >= 18 and no_kings_yet

    weakens_king_row = (
        _weakens_king_row(board, move, current_player)
        if is_opening_like else False
    )
    opens_long_diagonal_risk = (
        _opens_long_diagonal_risk(board, move, current_player)
        if is_opening_like else False
    )
    creates_forced_capture_risk = (
    _creates_forced_capture_risk(board_after, current_player, opponent)
    if is_opening_like else False
)

    # ── King/endgame pressure features ───────────────────────────────────────
    is_safe_move = not opponent_can_recapture

    king_escape_squares_before = _king_escape_squares(board, opponent)
    king_escape_squares_after  = _king_escape_squares(board_after, opponent)
    king_escape_reduction      = king_escape_squares_before - king_escape_squares_after

    corner_trap_pressure_before = _corner_trap_pressure(board, opponent)
    corner_trap_pressure_after  = _corner_trap_pressure(board_after, opponent)
    corner_trap_pressure        = corner_trap_pressure_after - corner_trap_pressure_before
    king_coordination      = _king_coordination_score(board_after, current_player)
    king_dist_pressure     = _king_distance_pressure(
        board, board_after, move, current_player, opponent, is_safe_move
    )

    # Endgame-like signal: few total pieces or multiple kings in play
    total_pieces = our_after["total"] + opp_after["total"]
    is_endgame_like = (total_pieces <= 10) or (our_after["kings"] + opp_after["kings"] >= 3)
    endgame_weight = 2 if is_endgame_like else 1

    # ── Endgame conversion features ───────────────────────────────────────────
    edge_conf_before = _edge_confinement_score(board, opponent)
    edge_conf_after  = _edge_confinement_score(board_after, opponent)
    edge_conf_delta  = edge_conf_after - edge_conf_before

    exchange_pressure = _exchange_pressure_score(
        board, move, current_player, opponent,
        material_advantage, our_after["kings"], opp_after["kings"],
        is_safe_move, board_after
    )

    bridge_score = _bridge_potential_score(
        board_after, current_player, final_row, final_col
    ) if piece_type_moving == "king" else 0

    anti_shuffle = _anti_shuffle_penalty(
        board, board_after, move, current_player,
        king_escape_reduction, edge_conf_delta,
        exchange_pressure, king_dist_pressure,
        restriction_score
    )

    dc_smokeout = _double_corner_smokeout_score(
        board_after, opponent, edge_conf_after,
        king_escape_squares_after, restriction_score, bridge_score
    )

    simplification = _simplification_value(
        material_advantage, our_after["kings"], opp_after["kings"],
        total_pieces, is_safe_move, restriction_score,
        king_coordination, exchange_pressure, opponent_mobility_after
    )
    # Trade / conversion helper:
    # Good when ahead and the move safely reduces opponent options.
    improves_trade_conversion = (
        material_advantage > 0
        and mobility_reduction > 0
        and our_pieces_threatened_after <= our_pieces_threatened_before
        and not opponent_can_recapture
    )

    # A compact scalar the ranker can use in winning positions.
    # Higher is better for converting an advantage.
    winning_conversion_score = 0
    if mobility_reduction > 0:
        winning_conversion_score += mobility_reduction * 2
    if not opponent_can_recapture:
        winning_conversion_score += 2
    if blocks_opponent_landing:
        winning_conversion_score += 2
    if creates_immediate_threat:
        winning_conversion_score += 5
    if shot_sequence_available:
        winning_conversion_score += 3
    if forces_exchange:
        winning_conversion_score += 3
    winning_conversion_score += min(forces_exchange_count, 2)
    if two_for_one_potential:
        winning_conversion_score += 4
    winning_conversion_score += min(two_for_one_score, 3)
    winning_conversion_score += min(restriction_score, 3)
    winning_conversion_score += min(frozen_enemy_pieces, 2)
    if in_center:
        winning_conversion_score += 1
    if promotion:
        winning_conversion_score += 3
    if leaves_piece_isolated:
        winning_conversion_score -= 1
    # Opening structure penalties
    if weakens_king_row:
        winning_conversion_score -= 3
    if opens_long_diagonal_risk:
        winning_conversion_score -= 2
    if creates_forced_capture_risk:
        winning_conversion_score -= 3
    # King endgame pressure + targeted conversion signals.
    # Guard: only apply when the king move is safe.
    # An unsafe king "approach" (walking into recapture danger) must NOT
    # receive conversion bonuses — it is a liability, not a conversion.
    if piece_type_moving == "king":
        _king_move_safe = (
            not opponent_can_recapture
            and our_pieces_threatened_after <= our_pieces_threatened_before
        )
        if _king_move_safe:
            winning_conversion_score += min(king_escape_reduction, 3) * endgame_weight
            winning_conversion_score += min(corner_trap_pressure, 2) * endgame_weight
            winning_conversion_score += king_dist_pressure * endgame_weight
            # Endgame conversion (targeted — not everything)
            winning_conversion_score += exchange_pressure * endgame_weight
            winning_conversion_score += min(simplification, 3) * endgame_weight
            winning_conversion_score += min(dc_smokeout, 3) * endgame_weight
            winning_conversion_score += min(edge_conf_delta, 2)  # light, no endgame_weight
    else:
        # Regular pieces: simplification still matters when ahead
        winning_conversion_score += min(simplification, 2)


                        
    # Symbolic safety flag — True when a simple move leaves any piece threatened.
    # This is a stronger signal than prompt wording alone.
    unsafe_simple_move = (
        move["type"] == "simple"
        and our_pieces_threatened_after > 0
    )



    # Counterplay score — rewards safe active moves when behind.
    # Used by CREATE_THREATS and SEEK_COUNTERPLAY priorities.
    # Does NOT override safety — only distinguishes "safe passive" from "safe active".
    # Intentionally excludes opponent_can_recapture bonus to keep this score
    # focused on pressure and activity, not safety (which is handled elsewhere).
    counterplay_score = 0
    if creates_immediate_threat:
        counterplay_score += 5
    if shot_sequence_available:
        counterplay_score += 4
    if two_for_one_potential:
        counterplay_score += 5
    counterplay_score += min(two_for_one_score, 3)
    if forces_exchange:
        counterplay_score += 3
    counterplay_score += min(forces_exchange_count, 2)
    if mobility_reduction > 0:
        counterplay_score += 3
    if blocks_opponent_landing:
        counterplay_score += 2
    counterplay_score += min(restriction_score, 2)
    if frozen_enemy_pieces > 0:
        counterplay_score += min(frozen_enemy_pieces, 2)
    if in_center:
        counterplay_score += 1
    if unsafe_simple_move:
        counterplay_score -= 2
    if leaves_piece_isolated:
        counterplay_score -= 1
    # Opening structure penalties
    if weakens_king_row:
        counterplay_score -= 3
    if opens_long_diagonal_risk:
        counterplay_score -= 2
    if creates_forced_capture_risk:
        counterplay_score -= 3
    # King endgame pressure (existing signals only — counterplay stays focused)
    if piece_type_moving == "king":
        counterplay_score += min(king_escape_reduction, 3) * endgame_weight
        counterplay_score += min(corner_trap_pressure, 2) * endgame_weight
        counterplay_score += king_dist_pressure * endgame_weight
        # Light endgame additions only
        counterplay_score += min(edge_conf_delta, 2)
        counterplay_score += min(dc_smokeout, 2)

    # King activity score — reward kings for meaningful activity,
    # not just moving "forward" or closer geometrically.
    # Used as a tiebreak for ACTIVATE_KINGS / endgame king play.
    king_activity_score = 0
    if piece_type_moving == "king":
        if creates_immediate_threat:
            king_activity_score += 2
        if mobility_reduction >= 2:
            king_activity_score += 2
        # center only rewarded if it also creates real pressure
        if in_center and (creates_immediate_threat or mobility_reduction >= 2):
            king_activity_score += 1
        if blocks_opponent_landing:
            king_activity_score += 2
        if leaves_piece_isolated:
            king_activity_score -= 1
        # New king/endgame pressure signals
        king_activity_score += min(king_escape_reduction, 3) * endgame_weight
        king_activity_score += min(corner_trap_pressure, 3) * endgame_weight
        king_activity_score += king_coordination
        king_activity_score += king_dist_pressure * endgame_weight
        # Endgame conversion signals (lighter integration)
        king_activity_score += min(edge_conf_delta, 3) * endgame_weight
        king_activity_score += min(dc_smokeout, 3) * endgame_weight
        king_activity_score += min(bridge_score, 1)        # weak proxy — capped at 1
        king_activity_score += anti_shuffle                # -2 or 0 only    # Quiet move role — categorical label for shortlist diversity in quiet positions.
    # Evaluated in priority order: first matching category wins.
    # Used by proposal to ensure the shortlist covers distinct strategic ideas
    # rather than returning 5 nearly identical passive moves.
    # For jump moves this is set to "TACTICAL" — the label is most useful in quiet positions.
    if move["type"] == "jump":
        quiet_move_role = "TACTICAL"
    elif two_for_one_potential or shot_sequence_available or forces_exchange:
        quiet_move_role = "TACTICAL_PRESSURE"
    elif promotion or near_promotion:
        quiet_move_role = "PROMOTION_PUSH"
    elif (
        piece_type_moving == "king"
        and is_endgame_like
        and (dc_smokeout >= 2 or (edge_conf_delta >= 2 and exchange_pressure >= 2))
    ):
        quiet_move_role = "KING_ENDGAME_CONVERSION"
    elif (
        piece_type_moving == "king"
        and (
            in_center
            or mobility_reduction > 0
            or creates_immediate_threat
            or winning_conversion_score > 0
            or restriction_score > 0
        )
        and anti_shuffle == 0
    ):
        quiet_move_role = "KING_ACTIVATION"
    elif piece_type_moving == "king" and anti_shuffle < 0:
        quiet_move_role = "KING_SHUFFLE"
    elif restriction_score > 0 or frozen_enemy_pieces > 0:
        quiet_move_role = "STRUCTURAL_RESTRICTION"
    elif counterplay_score >= 3:
        quiet_move_role = "COUNTERPLAY"
    elif winning_conversion_score >= 3:
        quiet_move_role = "CONVERSION"
    elif (
        our_pieces_threatened_after < our_pieces_threatened_before
        or blocks_opponent_landing
    ):
        quiet_move_role = "DEFENSIVE_STABILIZATION"
    elif mobility_reduction > 0 or creates_immediate_threat:
        quiet_move_role = "MOBILITY_IMPROVEMENT"
    else:
        quiet_move_role = "QUIET_DEFAULT"

    facts = {
        # Basic move info
        "move_type": move["type"],
        "piece_type_moving": piece_type_moving,
        "path_length": path_length,

        # Capture details
        "captures_count": captures_count,
        "jump_count": jump_count,
        "is_multi_jump": is_multi_jump,
        "kings_captured": kings_captured,
        "regulars_captured": regulars_captured,

        # Promotion
        "results_in_king": promotion,
        "near_promotion": near_promotion,

        # Piece counts
        "our_pieces_before": our_before,
        "our_pieces_after": our_after,
        "opp_pieces_before": opp_before,
        "opp_pieces_after": opp_after,
        "net_gain": net_gain,
        "material_advantage": material_advantage,

        # Strategic context
        "center_control": in_center,

        # FIXED: board-wide recapture check (was: only checked moved piece)
        "opponent_can_recapture": opponent_can_recapture,
        "moved_piece_is_threatened": moved_piece_is_threatened,
        "recapturable_piece_is_king": recapturable_piece_is_king,
        "max_opponent_jump_captures": max_opponent_jump_captures,
        "forced_opponent_jump_reply": forced_opponent_jump_reply,

        # Isolation
        "leaves_piece_isolated": leaves_piece_isolated,
        "any_piece_isolated": any_piece_isolated,

        # NEW: does our move block an opponent landing square?
        "blocks_opponent_landing": blocks_opponent_landing,

        # Opponent threat metrics
        "opponent_near_promotion": opponent_near_promotion,
        "opponent_jump_count": opponent_jump_count,
        "opponent_mobility_before": opponent_mobility_before,
        "opponent_mobility_after": opponent_mobility_after,
        "mobility_reduction": mobility_reduction,

        "creates_immediate_threat": creates_immediate_threat,
        "shot_sequence_available": shot_sequence_available,
        "forces_exchange": forces_exchange,
        "forces_exchange_count": forces_exchange_count,
        "two_for_one_potential": two_for_one_potential,
        "two_for_one_score": two_for_one_score,
        "restriction_score": restriction_score,
        "frozen_enemy_pieces": frozen_enemy_pieces,
        "improves_trade_conversion": improves_trade_conversion,
        "winning_conversion_score": winning_conversion_score,
        
        # NEW: board-wide safety scores (lower = safer)
        "our_pieces_threatened_before": our_pieces_threatened_before,
        "our_pieces_threatened_after": our_pieces_threatened_after,

        # Symbolic safety flag
        "unsafe_simple_move": unsafe_simple_move,

        # Counterplay quality score (used when behind — higher = more active and safe)
        "counterplay_score": counterplay_score,
        "king_activity_score": king_activity_score,
        "quiet_move_role": quiet_move_role,

        # King/endgame pressure features
        "king_escape_squares_after": king_escape_squares_after,
        "king_escape_reduction": king_escape_reduction,
        "corner_trap_pressure": corner_trap_pressure,
        "king_coordination": king_coordination,
        "king_distance_pressure": king_dist_pressure,

        # Opening structure features
        "weakens_king_row": weakens_king_row,
        "opens_long_diagonal_risk": opens_long_diagonal_risk,
        "creates_forced_capture_risk": creates_forced_capture_risk,

        # Endgame conversion features
        "edge_confinement_delta": edge_conf_delta,
        "exchange_pressure_when_ahead": exchange_pressure,
        "bridge_potential": bridge_score,
        "anti_shuffle_penalty": anti_shuffle,
        "double_corner_smokeout_pressure": dc_smokeout,
        "simplification_value": simplification,

        "opponent_safe_reply_count": opponent_safe_reply_count,
        "creates_real_trap": creates_real_trap,
        }

    return facts
