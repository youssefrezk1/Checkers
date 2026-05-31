# checkers/agents/llm_provider.py
#
# Step 8 — Provider abstraction layer (provider split + resilience).
#
# This module is the ONLY place in the project that constructs a Mistral HTTP
# request.  Both the chosen-reasoning path (ranker_agent) and the comparative-
# reasoning path (comparative_reasoner) call call_mistral_once from here.
# Neither path imports from the other — this module is the split boundary.
#
# Design rules:
#   - Single attempt per call. Zero retry. Callers own the retry/backoff policy.
#   - ProviderHTTPError carries the HTTP status code so callers can handle
#     429 (rate-limit) separately from 4xx/5xx errors.
#   - No imports from ranker_agent, comparative_reasoner, or any evaluator.
#   - Deterministic: same inputs → same HTTP request. No randomness.
#   - Timeout, model, temperature, and max_tokens are all caller-supplied.
#     No module-level defaults: config is explicit at every call site.

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


# ---------------------------------------------------------------------------
# Provider exception
# ---------------------------------------------------------------------------

class ProviderHTTPError(Exception):
    """
    HTTP-level error returned by the provider API.

    Attributes
    ----------
    code : int   HTTP status code (e.g. 429, 500).
    body : str   First 300 chars of the response body.
    """

    def __init__(self, code: int, body: str) -> None:
        super().__init__(f"HTTP {code}: {body}")
        self.code = code
        self.body = body


# ---------------------------------------------------------------------------
# Single-attempt call
# ---------------------------------------------------------------------------

_MISTRAL_BASE_URL = "https://api.mistral.ai/v1/chat/completions"


def call_mistral_once(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int = 512,
    timeout: float = 60.0,
    _base_url: str = _MISTRAL_BASE_URL,
) -> str:
    """
    Single-attempt Mistral chat-completion call with JSON response format.

    Parameters
    ----------
    messages      List of {"role": ..., "content": ...} dicts.
    api_key       Mistral API key (caller reads from env).
    model         Model ID string.
    temperature   Sampling temperature.
    max_tokens    Max tokens in the completion (default 512).
    timeout       HTTP socket timeout in seconds (default 60.0).
    _base_url     Override for testing; default is Mistral production URL.

    Returns
    -------
    str   The raw string content of choices[0].message.content.

    Raises
    ------
    ProviderHTTPError   HTTP-level error from the API (includes .code).
    ValueError          Unexpected response structure or non-string content.
    OSError / other     Network-level failure (urlopen raises).

    Notes
    -----
    No retry. No fallback. No 429 sleep. Callers are fully responsible for
    retry scheduling and rate-limit handling.
    """
    payload: dict[str, Any] = {
        "model":            model,
        "temperature":      temperature,
        "max_tokens":       max_tokens,
        "response_format":  {"type": "json_object"},
        "messages":         messages,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _base_url,
        data=body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept":        "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        raise ProviderHTTPError(e.code, body_text[:300]) from e

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(
            f"Unexpected Mistral response structure: {str(data)[:300]}"
        ) from e

    if not isinstance(content, str):
        raise ValueError(f"Mistral content is not a string: {type(content)}")

    return content
