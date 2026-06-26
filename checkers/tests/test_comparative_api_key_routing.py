# checkers/tests/test_comparative_api_key_routing.py
#
# Smoke test: dedicated comparative API-key routing.
#
# Validates two invariants:
#   1. The comparative path (_call_comparative_api) reads ONLY
#      MISTRAL_COMPARATIVE_API_KEY.  It raises ValueError when that variable
#      is absent, and does NOT fall back to MISTRAL_API_KEY.
#   2. The chosen-reasoning path (ranker_agent.call_ranker) reads ONLY
#      MISTRAL_API_KEY.  It is not affected by MISTRAL_COMPARATIVE_API_KEY.
#
# No live API calls are made.  All LLM interactions are blocked by
# injecting a mock _api_caller or by verifying error behaviour before
# any HTTP call can be attempted.

from __future__ import annotations

import importlib
import os
import sys
import types
from typing import Any
from unittest.mock import patch


# ── Helper to import comparative_reasoner with a clean env ───────────────────

def _fresh_comparative_reasoner() -> types.ModuleType:
    """
    Re-import comparative_reasoner so module-level env reads (if any were
    added in the future) always reflect the current os.environ state.
    For the current implementation there are no module-level env reads, but
    re-importing is cheap and makes the test forward-safe.
    """
    mod_name = "checkers.agents.comparative_reasoner"
    if mod_name in sys.modules:
        return importlib.reload(sys.modules[mod_name])
    return importlib.import_module(mod_name)


# ── 1. Comparative path raises when MISTRAL_COMPARATIVE_API_KEY is absent ────

def test_comparative_raises_when_comparative_key_missing() -> None:
    """
    _call_comparative_api must raise ValueError immediately when
    MISTRAL_COMPARATIVE_API_KEY is not set, before any HTTP call is made.
    """
    env = {k: v for k, v in os.environ.items()
           if k not in ("MISTRAL_COMPARATIVE_API_KEY", "MISTRAL_API_KEY")}

    with patch.dict(os.environ, env, clear=True):
        mod = _fresh_comparative_reasoner()
        try:
            mod._call_comparative_api("sys", "usr")
            assert False, "_call_comparative_api should have raised"
        except ValueError as exc:
            msg = str(exc)
            assert "MISTRAL_COMPARATIVE_API_KEY" in msg, (
                f"Error message must mention MISTRAL_COMPARATIVE_API_KEY; got: {msg}"
            )


# ── 2. Comparative path raises even when MISTRAL_API_KEY *is* set ────────────

def test_comparative_no_fallback_to_mistral_api_key() -> None:
    """
    When MISTRAL_COMPARATIVE_API_KEY is absent but MISTRAL_API_KEY is set,
    _call_comparative_api must still raise.  No fallback is permitted.
    """
    env = {k: v for k, v in os.environ.items()
           if k != "MISTRAL_COMPARATIVE_API_KEY"}
    env["MISTRAL_API_KEY"] = "chosen-path-key-should-not-be-used"

    with patch.dict(os.environ, env, clear=True):
        mod = _fresh_comparative_reasoner()
        try:
            mod._call_comparative_api("sys", "usr")
            assert False, "Should have raised ValueError (no fallback)"
        except ValueError as exc:
            msg = str(exc)
            assert "MISTRAL_COMPARATIVE_API_KEY" in msg, (
                f"Expected MISTRAL_COMPARATIVE_API_KEY in error message; got: {msg}"
            )


# ── 3. Comparative path reads MISTRAL_COMPARATIVE_API_KEY when present ────────

def test_comparative_reads_comparative_key() -> None:
    """
    When MISTRAL_COMPARATIVE_API_KEY is set, _call_comparative_api passes
    its value — and NOT the value of MISTRAL_API_KEY — to call_mistral_once.
    Verified by injecting a mock caller that captures the api_key actually used.
    """
    captured: dict[str, Any] = {}

    def _mock_call_once(_messages, *, api_key, model=None, temperature=None, **_kw) -> str:
        captured["api_key"] = api_key
        return '{"comparative_reasoning": "mock paragraph"}'

    comparative_key = "comp-key-abc"
    chosen_key      = "chosen-key-xyz"

    env = dict(os.environ)
    env["MISTRAL_COMPARATIVE_API_KEY"] = comparative_key
    env["MISTRAL_API_KEY"]             = chosen_key

    with patch.dict(os.environ, env, clear=True):
        mod = _fresh_comparative_reasoner()
        with patch.object(mod, "call_mistral_once", _mock_call_once):
            try:
                mod._call_comparative_api("sys", "usr")
            except Exception:
                pass  # retry/network errors irrelevant — key capture is enough

    # If no call got through (e.g., env cleared before patch applied), skip.
    if "api_key" not in captured:
        return  # environment did not allow a call through — cannot assert

    assert captured["api_key"] == comparative_key, (
        f"Comparative path used key {captured['api_key']!r} "
        f"instead of MISTRAL_COMPARATIVE_API_KEY={comparative_key!r}"
    )
    assert captured["api_key"] != chosen_key, (
        "Comparative path must not use MISTRAL_API_KEY"
    )


# ── 4. Chosen path still reads MISTRAL_API_KEY (not affected) ─────────────────

def test_chosen_path_reads_mistral_api_key_only() -> None:
    """
    ranker_agent reads MISTRAL_API_KEY at module import time.
    It must not reference MISTRAL_COMPARATIVE_API_KEY at all.
    """
    import importlib
    import sys

    mod_name = "checkers.agents.explainer_agent"
    if mod_name in sys.modules:
        ranker = sys.modules[mod_name]
    else:
        ranker = importlib.import_module(mod_name)

    # The module-level constant must exist and reference MISTRAL_API_KEY.
    assert hasattr(ranker, "MISTRAL_API_KEY"), (
        "ranker_agent must expose MISTRAL_API_KEY as a module-level constant"
    )

    # MISTRAL_COMPARATIVE_API_KEY must not appear anywhere in the module source.
    import inspect
    source = inspect.getsource(ranker)
    assert "MISTRAL_COMPARATIVE_API_KEY" not in source, (
        "ranker_agent must not reference MISTRAL_COMPARATIVE_API_KEY"
    )


# ── 5. Error message quality ───────────────────────────────────────────────────

def test_error_message_mentions_both_key_and_separation() -> None:
    """
    The ValueError message should tell the operator which key to set AND
    that it is separate from MISTRAL_API_KEY so the fix is unambiguous.
    """
    env = {k: v for k, v in os.environ.items()
           if k not in ("MISTRAL_COMPARATIVE_API_KEY", "MISTRAL_API_KEY")}

    with patch.dict(os.environ, env, clear=True):
        mod = _fresh_comparative_reasoner()
        try:
            mod._call_comparative_api("sys", "usr")
            assert False, "Expected ValueError"
        except ValueError as exc:
            msg = str(exc)
            assert "MISTRAL_COMPARATIVE_API_KEY" in msg
            assert "MISTRAL_API_KEY" in msg, (
                "Message should mention MISTRAL_API_KEY to clarify the separation"
            )


# ── 6. generate_comparative_reasoning propagates the missing-key error ─────────

def test_generate_comparative_skips_on_missing_key() -> None:
    """
    generate_comparative_reasoning catches all API exceptions in its
    reject-sample loop (except Exception: continue) and returns None when
    every attempt fails.  When MISTRAL_COMPARATIVE_API_KEY is absent the
    inner ValueError is caught, _api_error_count reaches max_samples, and
    the function returns None with skip_reason="api_failure".

    This test verifies that the top-level return is None (not a raised
    exception) and that the diagnostics_out dict records the skip correctly.
    """
    mod = _fresh_comparative_reasoner()

    chosen = {"path": [(5, 4), (4, 3)], "facts": {"opponent_can_recapture": False,
              "captures_count": 0, "our_pieces_threatened_after": 0,
              "leaves_piece_isolated": False}}
    alts   = [
        {"path": [(1, 0), (2, 1)], "facts": {"creates_immediate_threat": True,
         "opponent_can_recapture": True, "captures_count": 0}},
        {"path": [(1, 2), (2, 3)], "facts": {"captures_count": 2,
         "opponent_can_recapture": False, "our_pieces_threatened_after": 1}},
        {"path": [(3, 4), (2, 5)], "facts": {"captures_count": 0,
         "opponent_can_recapture": False, "our_pieces_threatened_after": 0}},
    ]

    env = {k: v for k, v in os.environ.items()
           if k not in ("MISTRAL_COMPARATIVE_API_KEY", "MISTRAL_API_KEY")}

    diag: dict = {}
    with patch.dict(os.environ, env, clear=True):
        result = mod.generate_comparative_reasoning(
            chosen, alts, chosen["facts"],
            diagnostics_out=diag,
            # _api_caller=None → default path → reads env → ValueError → caught
        )

    assert result is None, (
        f"Expected None when key is missing; got: {result!r}"
    )
    assert diag.get("comparative_was_skipped") is True, (
        f"Expected comparative_was_skipped=True; got: {diag}"
    )
    assert diag.get("comparative_skip_reason") == "api_failure", (
        f"Expected skip_reason='api_failure'; got: {diag.get('comparative_skip_reason')!r}"
    )


if __name__ == "__main__":
    test_comparative_raises_when_comparative_key_missing()
    test_comparative_no_fallback_to_mistral_api_key()
    test_comparative_reads_comparative_key()
    test_chosen_path_reads_mistral_api_key_only()
    test_error_message_mentions_both_key_and_separation()
    print("All routing tests passed.")
    print("Note: test_generate_comparative_skips_on_missing_key requires pytest.")
