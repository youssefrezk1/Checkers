from checkers.engine.evaluation import evaluate_board_breakdown
from checkers.engine.board import RED, BLACK, RED_KING, BLACK_KING

board_t43 = [
    [0, BLACK, 0, BLACK, 0, BLACK, 0, BLACK],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, RED, 0, RED, 0, 0, 0, 0],
    [0, 0, 0, 0, BLACK, 0, RED, 0],
    [0, RED, 0, 0, 0, 0, 0, BLACK],
    [0, 0, RED, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, RED, 0, BLACK_KING, 0, 0, 0]
]

breakdown = evaluate_board_breakdown(board_t43, RED, RED)
print("Turn 43 Breakdown for RED:", breakdown)
