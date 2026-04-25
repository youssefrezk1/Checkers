"""
Replay a single ablation game and log every move for analysis.
Deterministic: same seed/game_index as run_ablation_eval produces identical games.
"""
from __future__ import annotations
import json, random, sys
from checkers.engine.board import BLACK, RED, RED_KING, BLACK_KING, EMPTY, create_initial_board, is_own_piece, BOARD_SIZE
from checkers.engine.evaluation import evaluate_board, evaluate_board_breakdown, MAN_VALUE, KING_VALUE
from checkers.engine.rules import apply_move, get_all_legal_moves
from checkers.engine.win_condition import check_win_condition
from checkers.engine.zobrist import compute_hash
from checkers.search.minimax_core import (
    SearchStats, clear_transposition_table, search_root_iterative, search_root
)

def _opponent(p): return BLACK if p == RED else RED
def _player_name(p): return "RED" if p == RED else "BLACK"

def _board_ascii(board):
    syms = {EMPTY: ".", RED: "r", BLACK: "b", RED_KING: "R", BLACK_KING: "B"}
    lines = ["  0 1 2 3 4 5 6 7"]
    for r in range(8):
        row_str = f"{r} " + " ".join(syms[board[r][c]] for c in range(8))
        lines.append(row_str)
    return "\n".join(lines)

def _material(board):
    red = sum(MAN_VALUE if board[r][c]==RED else KING_VALUE if board[r][c]==RED_KING else 0
              for r in range(8) for c in range(8))
    blk = sum(MAN_VALUE if board[r][c]==BLACK else KING_VALUE if board[r][c]==BLACK_KING else 0
              for r in range(8) for c in range(8))
    return red, blk

def _piece_counts(board):
    rr = sum(1 for r in range(8) for c in range(8) if board[r][c] in (RED, RED_KING))
    bb = sum(1 for r in range(8) for c in range(8) if board[r][c] in (BLACK, BLACK_KING))
    return rr, bb

def _path_str(move):
    path = move.get("path", [])
    return " → ".join(f"({r},{c})" for r,c in path)

def _get_all_candidate_scores(board, player, depth, use_7a, use_7b):
    """Score ALL legal moves for comparison."""
    legal = get_all_legal_moves(board, player)
    results = []
    for move in legal:
        clear_transposition_table()
        child = apply_move(board, move)
        from checkers.search.minimax_core import negamax, SearchStats as SS, order_moves
        stats = SS()
        score = negamax(
            board=child, depth=max(0, depth-1),
            current_player=_opponent(player), root_player=player,
            alpha=float("-inf"), beta=float("inf"),
            stats=stats, use_tt=True,
            extension_depth=0,
            use_tactical_extension=use_7b,
            use_phase7a=use_7a,
        )
        results.append({
            "path": _path_str(move),
            "type": move["type"],
            "captured": len(move.get("captured", [])),
            "score": round(float(score), 1),
            "nodes": stats.nodes,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def play_and_trace(config_name, use_7a, use_7b, depth=3, seed=42, game_index=1, max_turns=120):
    board = create_initial_board()
    current = RED if (game_index % 2 == 0) else BLACK
    seen = {}
    moves_log = []

    print(f"\n{'='*80}")
    print(f"CONFIG: {config_name}  (7A={'ON' if use_7a else 'OFF'}, 7B={'ON' if use_7b else 'OFF'})")
    print(f"Starting player: {_player_name(current)}")
    print(f"{'='*80}\n")

    for ply in range(1, max_turns + 1):
        h = compute_hash(board)
        seen[h] = seen.get(h, 0) + 1
        if seen[h] >= 3:
            print(f"--- REPETITION DRAW at ply {ply} ---")
            break

        legal = get_all_legal_moves(board, current)
        if not legal:
            winner = _opponent(current)
            print(f"--- {_player_name(current)} has no moves. {_player_name(winner)} wins at ply {ply} ---")
            break

        clear_transposition_table()
        best_move, best_score, stats = search_root_iterative(
            board=board, current_player=current, target_depth=depth,
            use_tt=True, use_tactical_extension=use_7b, use_phase7a=use_7a,
        )

        if best_move is None:
            print(f"--- No best move returned at ply {ply} ---")
            break

        # Get all candidate scores for comparison (top 5)
        all_candidates = _get_all_candidate_scores(board, current, depth, use_7a, use_7b)

        red_mat, blk_mat = _material(board)
        rp, bp = _piece_counts(board)

        entry = {
            "ply": ply,
            "player": _player_name(current),
            "move": _path_str(best_move),
            "type": best_move["type"],
            "captured": len(best_move.get("captured", [])),
            "score": round(float(best_score), 1),
            "nodes": stats.nodes,
            "legal_count": len(legal),
            "red_pieces": rp,
            "black_pieces": bp,
            "material": f"R={red_mat} B={blk_mat} (diff={red_mat-blk_mat:+d})",
        }
        moves_log.append(entry)

        # Print move
        marker = ""
        if best_move["type"] == "jump":
            marker = f" [CAPTURE x{len(best_move.get('captured',[]))}]"
        
        score_display = f"{best_score:+.1f}"
        
        # Show top candidates if more than 1 legal move
        cand_str = ""
        if len(all_candidates) > 1:
            top3 = all_candidates[:min(3, len(all_candidates))]
            cand_parts = [f"{c['path']}={c['score']:+.1f}" for c in top3]
            cand_str = f"  candidates: [{', '.join(cand_parts)}]"

        print(f"Ply {ply:3d} | {_player_name(current):5s} | {_path_str(best_move):30s}{marker:20s} | "
              f"score={score_display:>8s} | nodes={stats.nodes:5d} | "
              f"pieces R={rp} B={bp} | mat_diff={red_mat-blk_mat:+d}")
        if cand_str:
            print(f"        {cand_str}")

        board = apply_move(board, best_move)

        wc = check_win_condition(board, current)
        if wc.get("game_over"):
            winner = wc.get("winner")
            print(f"\n--- {_player_name(winner)} wins! Reason: {wc.get('reason')} ---")
            break

        current = _opponent(current)

    print(f"\nFinal board:")
    print(_board_ascii(board))
    red_mat, blk_mat = _material(board)
    print(f"Final material: RED={red_mat} BLACK={blk_mat} (diff={red_mat-blk_mat:+d})")
    return moves_log


if __name__ == "__main__":
    # Run the "full" config trace (primary interest)
    config = sys.argv[1] if len(sys.argv) > 1 else "full"
    
    configs = {
        "baseline":     (False, False),
        "phase7a_only": (True, False),
        "phase7b_only": (False, True),
        "full":         (True, True),
    }
    
    if config == "all":
        for name, (a, b) in configs.items():
            play_and_trace(name, a, b)
    else:
        a, b = configs[config]
        play_and_trace(config, a, b)
