# engine/zobrist.py

import random

from checkers.engine.board import (
    EMPTY, RED, BLACK, RED_KING, BLACK_KING,
    BOARD_SIZE
)


# All piece types that can sit on a square
PIECE_TYPES = [RED, BLACK, RED_KING, BLACK_KING]


def generate_zobrist_table():
    """
    Generates a table of random 64-bit integers.
    One unique random number for every (piece_type, row, col) combination.
    This table must stay the same for the entire game.
    """
    table = {}

    for piece in PIECE_TYPES:
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                # Generate a random 64-bit integer for this combination
                random_number = random.getrandbits(64)
                table[(piece, row, col)] = random_number

    return table


# Generate the table once when this file is imported
# Every other file that imports from here gets the same table
ZOBRIST_TABLE = generate_zobrist_table()


def compute_hash(board):
    """
    Computes a Zobrist hash for the entire board position.
    Loops through every square — if a piece is there, XOR its
    random number into the running hash value.

    Returns a single integer representing this board position.
    """
    hash_value = 0

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]

            if piece != EMPTY:
                random_number = ZOBRIST_TABLE[(piece, row, col)]
                hash_value = hash_value ^ random_number

    return hash_value


def update_hash(current_hash, piece, from_row, from_col, to_row, to_col):
    """
    Updates an existing hash when a piece moves from one square to another.
    Instead of recomputing the entire board hash from scratch,
    we just XOR out the old position and XOR in the new position.

    This is much faster than calling compute_hash() every turn.
    """
    # XOR out the piece from its old position
    old_value = ZOBRIST_TABLE[(piece, from_row, from_col)]
    new_hash = current_hash ^ old_value

    # XOR in the piece at its new position
    new_value = ZOBRIST_TABLE[(piece, to_row, to_col)]
    new_hash = new_hash ^ new_value

    return new_hash


def check_repetition(position_history, current_hash, repeat_limit=3):
    """
    Checks if the current board position has been seen repeat_limit times.
    position_history is a list of all hashes seen so far this game.

    Returns True if the position has repeated enough times for a draw.
    """
    count = 0

    for past_hash in position_history:
        if past_hash == current_hash:
            count += 1

    if count >= repeat_limit:
        return True

    return False