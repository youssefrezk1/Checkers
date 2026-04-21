from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from checkers.engine.board import BLACK, BLACK_KING, EMPTY, RED, RED_KING, create_initial_board, is_own_piece
from checkers.engine.evaluation import MAN_VALUE, KING_VALUE, evaluate_board
from checkers.engine.rules import apply_move, get_all_legal_moves
from checkers.engine.win_condition import check_win_condition
from checkers.engine.zobrist import compute_hash
from checkers.search.minimax_core import SearchStats, clear_transposition_table, search_root_iterative


def _opponent(player: int) -> int:
    return BLACK if player == RED else RED


def _count_material_units(board: list[list[int]]) -> tuple[int, int]:
    red = 0
    black = 0
    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if p == RED:
                red += MAN_VALUE
            elif p == RED_KING:
                red += KING_VALUE
            elif p == BLACK:
                black += MAN_VALUE
            elif p == BLACK_KING:
                black += KING_VALUE
    return red, black


def _detect_promotion(board_before: list[list[int]], move: dict[str, Any], player: int) -> bool:
    path = move.get("path", [])
    if len(path) < 2:
        return False
    sr, sc = path[0]
    er, _ = path[-1]
    piece = board_before[sr][sc]
    if player == RED and piece == RED and er == 0:
        return True
    if player == BLACK and piece == BLACK and er == 7:
        return True
    return False


def _capture_count(move: dict[str, Any]) -> int:
    return len(move.get("captured", []) or [])


@dataclass(frozen=True)
class AblationConfig:
    name: str
    use_phase7a: bool
    use_tactical_extension: bool


@dataclass
class GameMetrics:
    config_name: str
    game_index: int
    winner: int | None
    draw: bool
    turn_count: int
    repetition_or_loop_detected: bool
    final_material_difference: int
    capture_count: int
    promotion_count: int
    average_nodes_per_move: float
    average_search_score_per_move: float
    average_depth_searched: float
    sharp_drop_count: int | None = None


def _choose_move(
    board: list[list[int]],
    player: int,
    depth: int,
    cfg: AblationConfig,
) -> tuple[dict[str, Any] | None, float, SearchStats]:
    clear_transposition_table()
    return search_root_iterative(
        board=board,
        current_player=player,
        target_depth=depth,
        use_tt=True,
        use_tactical_extension=cfg.use_tactical_extension,
        use_phase7a=cfg.use_phase7a,
    )


def play_game(
    cfg: AblationConfig,
    game_index: int,
    depth: int,
    seed: int,
    max_turns: int,
    track_sharp_drops: bool,
    sharp_drop_threshold: float,
) -> GameMetrics:
    rng = random.Random(seed + game_index)
    board = create_initial_board()

    current = RED if (game_index % 2 == 0) else BLACK
    move_scores: list[float] = []
    nodes: list[int] = []
    captures = 0
    promotions = 0

    sharp_drops = 0
    seen: dict[int, int] = {}
    repetition_or_loop = False

    for ply in range(1, max_turns + 1):
        h = compute_hash(board)
        seen[h] = seen.get(h, 0) + 1
        if seen[h] >= 3:
            repetition_or_loop = True
            break

        legal = get_all_legal_moves(board, current)
        if not legal:
            # No moves: opponent wins by blockade.
            winner = _opponent(current)
            red_mat, black_mat = _count_material_units(board)
            return GameMetrics(
                config_name=cfg.name,
                game_index=game_index,
                winner=winner,
                draw=False,
                turn_count=ply - 1,
                repetition_or_loop_detected=repetition_or_loop,
                final_material_difference=red_mat - black_mat,
                capture_count=captures,
                promotion_count=promotions,
                average_nodes_per_move=float(sum(nodes) / max(1, len(nodes))),
                average_search_score_per_move=float(sum(move_scores) / max(1, len(move_scores))),
                average_depth_searched=float(depth),
                sharp_drop_count=sharp_drops if track_sharp_drops else None,
            )

        best_move, best_score, stats = _choose_move(board, current, depth, cfg)
        if best_move is None:
            # Should not happen when legal moves exist; treat as draw-safe stop.
            repetition_or_loop = True
            break

        nodes.append(stats.nodes)
        move_scores.append(best_score)
        captures += _capture_count(best_move)
        if _detect_promotion(board, best_move, current):
            promotions += 1

        board_after = apply_move(board, best_move)

        if track_sharp_drops:
            # Score after our move (from our perspective), before opponent reply.
            pre = float(evaluate_board(board_after, _opponent(current), current, use_phase7a=cfg.use_phase7a))
            opp_move, _, _ = _choose_move(board_after, _opponent(current), depth, cfg)
            if opp_move is not None:
                board_after_reply = apply_move(board_after, opp_move)
                post = float(evaluate_board(board_after_reply, current, current, use_phase7a=cfg.use_phase7a))
                if (pre - post) > sharp_drop_threshold:
                    sharp_drops += 1

        board = board_after
        wc = check_win_condition(board, current)
        if wc.get("game_over"):
            winner = wc.get("winner")
            red_mat, black_mat = _count_material_units(board)
            return GameMetrics(
                config_name=cfg.name,
                game_index=game_index,
                winner=winner,
                draw=False,
                turn_count=ply,
                repetition_or_loop_detected=repetition_or_loop,
                final_material_difference=red_mat - black_mat,
                capture_count=captures,
                promotion_count=promotions,
                average_nodes_per_move=float(sum(nodes) / max(1, len(nodes))),
                average_search_score_per_move=float(sum(move_scores) / max(1, len(move_scores))),
                average_depth_searched=float(depth),
                sharp_drop_count=sharp_drops if track_sharp_drops else None,
            )

        current = _opponent(current)

    # Draw by repetition/turn cap.
    red_mat, black_mat = _count_material_units(board)
    return GameMetrics(
        config_name=cfg.name,
        game_index=game_index,
        winner=None,
        draw=True,
        turn_count=min(max_turns, ply if "ply" in locals() else 0),
        repetition_or_loop_detected=True,
        final_material_difference=red_mat - black_mat,
        capture_count=captures,
        promotion_count=promotions,
        average_nodes_per_move=float(sum(nodes) / max(1, len(nodes))),
        average_search_score_per_move=float(sum(move_scores) / max(1, len(move_scores))),
        average_depth_searched=float(depth),
        sharp_drop_count=sharp_drops if track_sharp_drops else None,
    )


def summarize(games: list[GameMetrics], track_sharp_drops: bool) -> dict[str, Any]:
    by_cfg: dict[str, list[GameMetrics]] = {}
    for g in games:
        by_cfg.setdefault(g.config_name, []).append(g)

    summary: dict[str, Any] = {}
    for cfg, rows in by_cfg.items():
        wins_red = sum(1 for r in rows if r.winner == RED)
        wins_black = sum(1 for r in rows if r.winner == BLACK)
        draws = sum(1 for r in rows if r.draw)
        loops = sum(1 for r in rows if r.repetition_or_loop_detected)
        avg_turns = sum(r.turn_count for r in rows) / max(1, len(rows))
        avg_caps = sum(r.capture_count for r in rows) / max(1, len(rows))
        avg_promos = sum(r.promotion_count for r in rows) / max(1, len(rows))
        avg_nodes = sum(r.average_nodes_per_move for r in rows) / max(1, len(rows))
        avg_score = sum(r.average_search_score_per_move for r in rows) / max(1, len(rows))
        out: dict[str, Any] = {
            "games": len(rows),
            "wins_red": wins_red,
            "wins_black": wins_black,
            "draws": draws,
            "loop_or_turncap_games": loops,
            "avg_turns": avg_turns,
            "avg_captures": avg_caps,
            "avg_promotions": avg_promos,
            "avg_nodes_per_move": avg_nodes,
            "avg_search_score_per_move": avg_score,
        }
        if track_sharp_drops:
            sharp_total = sum(int(r.sharp_drop_count or 0) for r in rows)
            out["sharp_drop_total"] = sharp_total
            out["sharp_drop_avg_per_game"] = sharp_total / max(1, len(rows))
        summary[cfg] = out
    return summary


def run_ablation(
    games: int,
    depth: int,
    seed: int,
    max_turns: int,
    track_sharp_drops: bool,
    sharp_drop_threshold: float,
) -> dict[str, Any]:
    configs = [
        AblationConfig("baseline", use_phase7a=False, use_tactical_extension=False),
        AblationConfig("phase7a_only", use_phase7a=True, use_tactical_extension=False),
        AblationConfig("phase7b_only", use_phase7a=False, use_tactical_extension=True),
        AblationConfig("full", use_phase7a=True, use_tactical_extension=True),
    ]

    all_games: list[GameMetrics] = []
    started = time.perf_counter()
    for cfg in configs:
        for gi in range(1, games + 1):
            all_games.append(
                play_game(
                    cfg=cfg,
                    game_index=gi,
                    depth=depth,
                    seed=seed,
                    max_turns=max_turns,
                    track_sharp_drops=track_sharp_drops,
                    sharp_drop_threshold=sharp_drop_threshold,
                )
            )
    elapsed_s = time.perf_counter() - started

    return {
        "meta": {
            "games_per_config": games,
            "depth": depth,
            "seed": seed,
            "max_turns": max_turns,
            "track_sharp_drops": track_sharp_drops,
            "sharp_drop_threshold": sharp_drop_threshold,
            "elapsed_s": elapsed_s,
        },
        "per_game": [asdict(g) for g in all_games],
        "summary": summarize(all_games, track_sharp_drops=track_sharp_drops),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run ablation self-play evaluation for 7A/7B configs.")
    p.add_argument("--games", type=int, default=2, help="Games per config")
    p.add_argument("--depth", type=int, default=2, help="Search depth (iterative deepening target)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--max-turns", type=int, default=120, help="Max plies per game (turn cap)")
    p.add_argument("--out", type=str, default="logs/ablation_eval.json", help="Output JSON path")
    p.add_argument("--track-sharp-drops", action="store_true", help="Track tactical sharp drops (slower)")
    p.add_argument("--sharp-drop-threshold", type=float, default=120.0, help="Sharp drop threshold (~1 man)")
    args = p.parse_args(argv)

    result = run_ablation(
        games=args.games,
        depth=args.depth,
        seed=args.seed,
        max_turns=args.max_turns,
        track_sharp_drops=args.track_sharp_drops,
        sharp_drop_threshold=args.sharp_drop_threshold,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Ablation summary (per config):")
    for cfg, s in result["summary"].items():
        line = (
            f"- {cfg}: games={s['games']} red_wins={s['wins_red']} black_wins={s['wins_black']} "
            f"draws={s['draws']} avg_turns={s['avg_turns']:.1f} avg_nodes/move={s['avg_nodes_per_move']:.1f}"
        )
        if args.track_sharp_drops:
            line += f" sharp_drop_avg/game={s['sharp_drop_avg_per_game']:.2f}"
        print(line)

    print(f"Wrote `{out_path.as_posix()}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

