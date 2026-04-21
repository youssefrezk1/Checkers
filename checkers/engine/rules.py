# engine/rules.py

from checkers.engine.board import (
    EMPTY, RED, BLACK, RED_KING, BLACK_KING,
    BOARD_SIZE, in_bounds, is_own_piece,
    is_opponent_piece, is_king, get_piece
)

def get_move_directions(piece):
    """
    Returns the diagonal directions a piece can move toward.
    RED moves up so row decreases → directions are (-1, -1) and (-1, +1)
    BLACK moves down so row increases → directions are (+1, -1) and (+1, +1)
    Kings can move in all 4 diagonal directions.
    """
    if piece == RED:
        directions = [(-1, -1), (-1, +1)]

    elif piece == BLACK:
        directions = [(+1, -1), (+1, +1)]

    elif piece == RED_KING or piece == BLACK_KING:
        directions = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]

    else:
        directions = []

    return directions


def get_simple_moves(board, row, col):
    """
    Returns all simple (non-capture) diagonal moves for the piece at (row, col).
    A simple move means moving one square diagonally to an empty square.
    Each move is represented as a tuple: (from_row, from_col, to_row, to_col)
    """
    piece = board[row][col]
    directions = get_move_directions(piece)
    simple_moves = []

    for direction in directions:
        row_step = direction[0]
        col_step = direction[1]

        target_row = row + row_step
        target_col = col + col_step

        if in_bounds(target_row, target_col):
            target_square = board[target_row][target_col]
            if target_square == EMPTY:
                move = (row, col, target_row, target_col)
                simple_moves.append(move)

    return simple_moves


def get_single_jumps(board, row, col, current_player):
    """
    Returns all single jumps available for the piece at (row, col).
    A jump means leaping over an opponent piece and landing on the empty square beyond.
    Each jump is represented as:
    (from_row, from_col, to_row, to_col, captured_row, captured_col)
    """
    piece = board[row][col]
    directions = get_move_directions(piece)
    jumps = []

    for direction in directions:
        row_step = direction[0]
        col_step = direction[1]

        # The square we are jumping over (must have an opponent piece)
        middle_row = row + row_step
        middle_col = col + col_step

        # The square we land on (must be empty)
        landing_row = row + 2 * row_step
        landing_col = col + 2 * col_step

        if in_bounds(landing_row, landing_col):
            middle_piece = get_piece(board, middle_row, middle_col)
            landing_square = board[landing_row][landing_col]

            middle_has_opponent = is_opponent_piece(middle_piece, current_player)
            landing_is_empty = landing_square == EMPTY

            if middle_has_opponent and landing_is_empty:
                jump = (row, col, landing_row, landing_col, middle_row, middle_col)
                jumps.append(jump)

    return jumps


def get_all_jump_sequences(board, row, col, current_player, path_so_far, captured_so_far):
    """
    Recursively finds all possible multi-jump sequences from (row, col).
    
    - path_so_far: list of (row, col) positions visited during this sequence
    - captured_so_far: list of (row, col) positions of pieces captured so far
    
    Returns a list of complete jump sequences.
    Each sequence is a dict with:
        "path": list of (row, col) squares visited including start
        "captured": list of (row, col) squares of captured pieces
    """
    available_jumps = get_single_jumps(board, row, col, current_player)

    # Filter out jumps that would capture a piece already captured this turn
    available_jumps = [
        jump for jump in available_jumps
        if (jump[4], jump[5]) not in captured_so_far
    ]

    # Base case: no more jumps available, this sequence is complete
    if len(available_jumps) == 0:
        sequence = {
            "path": path_so_far,
            "captured": captured_so_far
        }
        return [sequence]

    all_sequences = []

    for jump in available_jumps:
        landing_row = jump[2]
        landing_col = jump[3]
        captured_row = jump[4]
        captured_col = jump[5]

        # Temporarily apply this jump on a copy of the board
        temp_board = apply_jump_on_board(board, row, col, landing_row, landing_col, captured_row, captured_col)

        # Check if piece gets promoted — if so it cannot continue jumping
        piece_after_jump = temp_board[landing_row][landing_col]
        piece_before_jump = board[row][col]

        piece_was_promoted = (
            piece_before_jump == RED and piece_after_jump == RED_KING or
            piece_before_jump == BLACK and piece_after_jump == BLACK_KING
        )

        if piece_was_promoted:
            # Stop the sequence here — crowning ends the turn
            sequence = {
                "path": path_so_far + [(landing_row, landing_col)],
                "captured": captured_so_far + [(captured_row, captured_col)]
            }
            all_sequences.append(sequence)
        else:
            # Continue jumping recursively from landing square
            new_path = path_so_far + [(landing_row, landing_col)]
            new_captured = captured_so_far + [(captured_row, captured_col)]

            deeper_sequences = get_all_jump_sequences(
                temp_board, landing_row, landing_col,
                current_player, new_path, new_captured
            )
            all_sequences.extend(deeper_sequences)

    return all_sequences


def apply_jump_on_board(board, from_row, from_col, to_row, to_col, cap_row, cap_col):
    """
    Returns a new board with the jump applied.
    Moves the piece, removes the captured piece, handles promotion.
    Does NOT modify the original board.
    """
    # Deep copy the board
    new_board = [row[:] for row in board]

    piece = new_board[from_row][from_col]

    # Move piece to landing square
    new_board[to_row][to_col] = piece
    new_board[from_row][from_col] = EMPTY

    # Remove captured piece
    new_board[cap_row][cap_col] = EMPTY

    # Handle promotion
    if piece == RED and to_row == 0:
        new_board[to_row][to_col] = RED_KING
    elif piece == BLACK and to_row == BOARD_SIZE - 1:
        new_board[to_row][to_col] = BLACK_KING

    return new_board



def get_all_legal_moves(board, current_player):
    """
    Returns all legal moves for current_player.
    
    Mandatory capture rule: if ANY jump exists for ANY piece,
    only jumps are returned — simple moves are not allowed.
    
    Each move is a dict:
        Simple move:  {"type": "simple", "path": [(r1,c1),(r2,c2)], "captured": []}
        Jump move:    {"type": "jump",   "path": [(r1,c1),...],      "captured": [(cr,cc),...]}
    """
    all_simple_moves = []
    all_jump_sequences = []

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]

            if is_own_piece(piece, current_player):
                # Collect simple moves
                simple = get_simple_moves(board, row, col)
                for move in simple:
                    all_simple_moves.append({
                        "type": "simple",
                        "path": [(move[0], move[1]), (move[2], move[3])],
                        "captured": []
                    })

                # Collect jump sequences
                sequences = get_all_jump_sequences(
                    board, row, col, current_player,
                    path_so_far=[(row, col)],
                    captured_so_far=[]
                )
                for sequence in sequences:
                    if len(sequence["captured"]) > 0:
                        all_jump_sequences.append({
                            "type": "jump",
                            "path": sequence["path"],
                            "captured": sequence["captured"]
                        })

    # Mandatory capture — if jumps exist, only return jumps
    if len(all_jump_sequences) > 0:
        return all_jump_sequences

    return all_simple_moves




def apply_move(board, move):
    """
    Applies a complete move (simple or jump) to the board.
    Returns the new board after the move — does NOT modify original.
    
    move is a dict with "type", "path", and "captured".
    """
    new_board = [row[:] for row in board]

    from_row = move["path"][0][0]
    from_col = move["path"][0][1]
    to_row = move["path"][-1][0]
    to_col = move["path"][-1][1]

    piece = new_board[from_row][from_col]

    # Remove all captured pieces
    for captured_pos in move["captured"]:
        cap_row = captured_pos[0]
        cap_col = captured_pos[1]
        new_board[cap_row][cap_col] = EMPTY

    # Move the piece to final destination
    new_board[to_row][to_col] = piece
    new_board[from_row][from_col] = EMPTY

    # Handle promotion at final position
    if piece == RED and to_row == 0:
        new_board[to_row][to_col] = RED_KING
    elif piece == BLACK and to_row == BOARD_SIZE - 1:
        new_board[to_row][to_col] = BLACK_KING

    return new_board