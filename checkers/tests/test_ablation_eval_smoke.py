from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from checkers.eval.run_ablation_eval import run_ablation


def test_run_ablation_smoke_output_structure() -> None:
    result = run_ablation(
        games=1,
        depth=1,
        seed=1,
        max_turns=8,
        track_sharp_drops=False,
        sharp_drop_threshold=120.0,
    )
    assert "meta" in result
    assert "per_game" in result
    assert "summary" in result
    assert len(result["per_game"]) == 4  # one game per config
    assert set(result["summary"].keys()) == {"baseline", "phase7a_only", "phase7b_only", "full"}

    with TemporaryDirectory() as td:
        p = Path(td) / "out.json"
        p.write_text(json.dumps(result), encoding="utf-8")
        loaded = json.loads(p.read_text(encoding="utf-8"))
        assert "summary" in loaded

