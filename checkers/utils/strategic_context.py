# Shared formatting for strategic_context (memory → LLM prompts).
# Kept separate from proposal_agent / ranker_agent to avoid cross-agent imports.

from __future__ import annotations

import json
from typing import Any, Optional


def format_strategic_context(ctx: Optional[dict[str, Any]]) -> str:
    if not ctx:
        return "(no strategic context yet)"
    try:
        return json.dumps(ctx, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(ctx)
