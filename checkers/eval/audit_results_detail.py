import json
from pathlib import Path

def is_jump(path):
    if not path or len(path) < 2:
        return False
    return abs(path[0][0] - path[1][0]) >= 2

def analyze_details(file_path: Path):
    print(f"\n========================================\nAnalyzing: {file_path.name}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: {e}")
        return

    results = data.get("results", [])
    if not results and isinstance(data, list):
        results = data
        
    if not results:
        print("No results found.")
        return

    total = len(results)
    
    # 1. BF distribution in results
    bfs = [r.get("legal_move_count", 0) for r in results]
    avg_bf = sum(bfs) / total if total else 0
    
    # 2. Quiet vs Tactical composition in results
    # Use best_sym_path to check if it's a jump
    tactical_positions = []
    quiet_positions = []
    for r in results:
        best_path = r.get("best_sym_path")
        # fallback
        if best_path is None:
            best_path = r.get("engine_best_move")
        if best_path is None:
            best_path = r.get("kingsrow_best_move")
            
        if best_path and is_jump(best_path):
            tactical_positions.append(r)
        else:
            quiet_positions.append(r)
    
    # 3. High BF (BF >= 6) vs Low BF
    high_bf_positions = [r for r in results if r.get("legal_move_count", 0) >= 6]
    low_bf_positions = [r for r in results if r.get("legal_move_count", 0) < 6]
    
    # Helper to check coverage
    def get_coverage(subset, label):
        if not subset:
            print(f"  {label}: No positions in this subset.")
            return
        
        eng_cov = 0
        kr_cov = 0
        eng_valid = 0
        kr_valid = 0
        
        for r in subset:
            contains_engine = r.get("contains_engine_best")
            contains_kr = r.get("contains_kingsrow_best")
            
            # fallback
            if contains_engine is None:
                contains_engine = r.get("best_move_covered")
            if contains_kr is None:
                contains_kr = r.get("near_best_covered")
                
            if contains_engine is not None:
                eng_valid += 1
                if contains_engine is True:
                    eng_cov += 1
            if contains_kr is not None:
                kr_valid += 1
                if contains_kr is True:
                    kr_cov += 1
                    
        print(f"  {label} (N={len(subset)}):")
        if eng_valid > 0:
            print(f"    Engine coverage  : {eng_cov}/{eng_valid} ({eng_cov/eng_valid*100:.1f}%)")
        if kr_valid > 0:
            print(f"    KingsRow coverage: {kr_cov}/{kr_valid} ({kr_cov/kr_valid*100:.1f}%)")

    # 4. Coverage on subsets
    get_coverage(results, "Overall")
    get_coverage(tactical_positions, "Tactical")
    get_coverage(quiet_positions, "Quiet")
    get_coverage(high_bf_positions, "High BF (>=6)")
    get_coverage(low_bf_positions, "Low BF (<6)")
    
    # 5. Perfect vs Imperfect Proposals
    perfect_pos = []
    imperfect_pos = []
    for r in results:
        classification = r.get("classification", r.get("proposal_classification", {}).get("classification", "unknown"))
        if classification == "perfect":
            perfect_pos.append(r)
        else:
            imperfect_pos.append(r)
            
    get_coverage(perfect_pos, "Perfect Proposals")
    get_coverage(imperfect_pos, "Imperfect Proposals")
    
    # 6. Proposal sets details (Accidental coverage in weak/illegal sets)
    # Check if proposals contained illegal moves
    has_illegal_moves = []
    no_illegal_moves = []
    for r in results:
        prop_class = r.get("proposal_classification", {})
        illegal_count = prop_class.get("illegal_proposed", 0)
        
        # fallback for old formats
        if not prop_class:
            tax = r.get("failure_taxonomy", {})
            illegal_count = tax.get("illegal_geometry_moves", 0) + tax.get("out_of_bounds_coordinates", 0)
            
        if illegal_count > 0:
            has_illegal_moves.append(r)
        else:
            no_illegal_moves.append(r)
            
    get_coverage(has_illegal_moves, "Proposals with Illegal Moves")
    get_coverage(no_illegal_moves, "Proposals without Illegal Moves")
    
    # 7. Incomplete vs Complete (proposed fewer moves than legal moves)
    incomplete_pos = []
    complete_pos = []
    for r in results:
        legal_count = r.get("legal_move_count", 0)
        
        proposed_paths = r.get("proposed_paths")
        if proposed_paths is not None:
            proposed_count = len(proposed_paths)
        else:
            prop_class = r.get("proposal_classification", {})
            proposed_count = prop_class.get("proposed_count", 0)
            
        if proposed_count < legal_count:
            incomplete_pos.append(r)
        else:
            complete_pos.append(r)
            
    get_coverage(incomplete_pos, "Proposals with count < legal moves")
    get_coverage(complete_pos, "Proposals with count >= legal moves")

def main():
    logs_dir = Path(__file__).parent.parent.parent / "logs"
    for f in logs_dir.glob("*.json"):
        if f.name in ["proposal_coverage_clean_postprocess.json", "proposal_coverage_smoke_depth6_real_llm.json", "proposal_coverage_full19_depth6_real_llm.json"]:
            analyze_details(f)

if __name__ == "__main__":
    main()
