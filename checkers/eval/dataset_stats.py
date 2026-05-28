import json
import os
import sys
from pathlib import Path
from checkers.engine.board import RED, BLACK
from checkers.engine.rules import get_all_legal_moves
from checkers.agents.scorer_agent import score_all_legal_moves
from checkers.agents.deterministic_proposal import select_best_move

def _norm_path(path):
    return [[int(sq[0]), int(sq[1])] for sq in (path or [])]

def analyze_dataset():
    here = Path(__file__).parent
    dataset_path = here.parent / "data" / "legality_stress" / "scenarios.jsonl"
    annotations_path = here.parent / "data" / "legality_stress" / "scenarios_bestmove_annotations.json"
    
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return
        
    print(f"Loading dataset: {dataset_path}")
    positions = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                positions.append(json.loads(line))
                
    print(f"Total positions in dataset: {len(positions)}")
    
    annotations = {}
    if annotations_path.exists():
        print(f"Loading annotations: {annotations_path}")
        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                annotations[entry["scenario_id"]] = entry
    else:
        print("Annotations not found!")
        
    # Analysis metrics
    branching_factors = []
    tactical_count = 0  # has capture
    quiet_count = 0
    difficulty_counts = {}
    only_one_legal_move = 0
    only_one_jump_move = 0
    
    # Overlap analysis
    kr_available_count = 0
    kr_engine_match = 0
    total_valid_for_overlap = 0
    
    for entry in positions:
        sid = entry.get("scenario_id")
        board = entry.get("board")
        side_str = entry.get("side_to_move", "RED")
        side = RED if side_str.upper() == "RED" else BLACK
        diff = entry.get("difficulty", "unknown")
        
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        legal = get_all_legal_moves(board, side)
        bf = len(legal)
        branching_factors.append(bf)
        
        if bf == 1:
            only_one_legal_move += 1
            
        has_jumps = any(m["type"] == "jump" for m in legal)
        if has_jumps:
            tactical_count += 1
            if bf == 1:
                only_one_jump_move += 1
        else:
            quiet_count += 1
            
        ann = annotations.get(sid, {})
        kr_path = ann.get("kr_path")
        engine_path = ann.get("engine_best_path")
        
        if kr_path is not None and engine_path is not None:
            kr_available_count += 1
            total_valid_for_overlap += 1
            if _norm_path(kr_path) == _norm_path(engine_path):
                kr_engine_match += 1
                
    # Print basic stats
    print("\n--- DATASET STATS ---")
    print(f"Total positions: {len(positions)}")
    print(f"Difficulties: {difficulty_counts}")
    print(f"Tactical (has capture): {tactical_count} ({tactical_count/len(positions)*100:.1f}%)")
    print(f"Quiet (no capture): {quiet_count} ({quiet_count/len(positions)*100:.1f}%)")
    
    # BF distribution
    bf_sorted = sorted(branching_factors)
    mean_bf = sum(bf_sorted) / len(bf_sorted)
    median_bf = bf_sorted[len(bf_sorted)//2]
    min_bf = bf_sorted[0]
    max_bf = bf_sorted[-1]
    
    bf_distribution = {}
    for bf in bf_sorted:
        bf_distribution[bf] = bf_distribution.get(bf, 0) + 1
        
    print(f"Branching factor: Min={min_bf}, Max={max_bf}, Mean={mean_bf:.2f}, Median={median_bf}")
    print(f"BF Distribution (bf: count): {bf_distribution}")
    print(f"Positions with ONLY 1 legal move: {only_one_legal_move} ({only_one_legal_move/len(positions)*100:.1f}%)")
    print(f"Tactical positions with ONLY 1 legal jump (mandatory capture): {only_one_jump_move} ({only_one_jump_move/tactical_count*100:.1f}% of tactical)")
    
    # Overlap stats
    print(f"\n--- ENGINE vs KINGSROW OVERLAP ---")
    print(f"Positions with both KR and Engine best moves: {kr_available_count}")
    if kr_available_count > 0:
        print(f"Overlap agreement: {kr_engine_match} / {kr_available_count} ({kr_engine_match/kr_available_count*100:.1f}%)")

if __name__ == "__main__":
    analyze_dataset()
