import json
from pathlib import Path
from checkers.engine.board import RED, BLACK
from checkers.engine.rules import get_all_legal_moves

def run_simulation():
    here = Path(__file__).parent
    dataset_path = here.parent / "data" / "legality_stress" / "scenarios.jsonl"
    
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return
        
    positions = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                positions.append(json.loads(line))
                
    # Subsets
    subsets = {
        "All": positions,
        "Easy": [p for p in positions if p.get("difficulty") == "easy"],
        "Medium": [p for p in positions if p.get("difficulty") == "medium"],
        "Hard": [p for p in positions if p.get("difficulty") == "hard"],
        "Medium_Hard": [p for p in positions if p.get("difficulty") in ["medium", "hard"]]
    }
    
    for name, subset in subsets.items():
        print(f"\n--- Subset: {name} (N={len(subset)}) ---")
        if not subset:
            continue
            
        bfs = []
        tactical_count = 0
        bf_le_5_count = 0
        
        for entry in subset:
            board = entry.get("board")
            side_str = entry.get("side_to_move", "RED")
            side = RED if side_str.upper() == "RED" else BLACK
            legal = get_all_legal_moves(board, side)
            bf = len(legal)
            bfs.append(bf)
            
            if bf <= 5:
                bf_le_5_count += 1
                
            if any(m["type"] == "jump" for m in legal):
                tactical_count += 1
                
        print(f"  Average branching factor: {sum(bfs)/len(bfs):.2f}")
        print(f"  Tactical fraction: {tactical_count/len(subset)*100:.1f}%")
        print(f"  BF <= 5 fraction: {bf_le_5_count/len(subset)*100:.1f}%")
        
        for K in [1, 2, 3, 4, 5]:
            total_expected_cov = 0.0
            for entry in subset:
                board = entry.get("board")
                side_str = entry.get("side_to_move", "RED")
                side = RED if side_str.upper() == "RED" else BLACK
                legal = get_all_legal_moves(board, side)
                N = len(legal)
                prob = min(1.0, K / N) if N > 0 else 0.0
                total_expected_cov += prob
            avg_expected_cov = total_expected_cov / len(subset)
            print(f"  For K = {K} proposals: Expected Random Coverage = {avg_expected_cov*100:.1f}%")

if __name__ == "__main__":
    run_simulation()
