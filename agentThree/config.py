"""Module-level configuration constants for the agentThree package."""

from __future__ import annotations

import os
from pathlib import Path

# ---- HTTP / model -----------------------------------------------------

OLLAMA_URL: str = "http://localhost:11434/api/chat"

#DEFAULT_MODEL: str = "minimax-m3:cloud"
#DEFAULT_MODEL: str = "kimi-k2.7-code:cloud"
DEFAULT_MODEL: str = "glm-5.2:cloud"
#DEFAULT_MODEL: str = "deepseek-v4-pro:cloud"
#DEFAULT_MODEL: str = "deepseek-v4-flash:cloud"
#DEFAULT_MODEL: str = "minimax-m2.7:cloud"
#DEFAULT_MODEL: str = "gemma4:31b-cloud"

#: Models known to be available on this Ollama instance, drawn from the
#: commented ``DEFAULT_MODEL`` candidates above.  Used by the ``/model``
#: slash command for argument completion and as a quick reference list.
#: The list is advisory only - any model name may be passed to ``/model``
#: at runtime, even one not listed here.
KNOWN_MODELS: list[str] = [
    "minimax-m3:cloud",
    "kimi-k2.7-code:cloud",
    "glm-5.2:cloud",
    "deepseek-v4-pro:cloud",
    "deepseek-v4-flash:cloud",
    "minimax-m2.7:cloud",
    "gemma4:31b-cloud",
]


def available_models() -> list[str]:
    """Return the list of known model names.

    Tries to refresh the list from the live Ollama server first (best
    effort); falls back to :data:`KNOWN_MODELS` from ``config.py`` when the
    server is unreachable or ``requests`` is not available.  The result is
    a plain ``list[str]`` suitable for tab-completion in the ``/model``
    slash command.
    """
    # Best effort: ask the Ollama server what models it actually has.
    try:
        import requests  # local import; optional dependency at import time
        resp = requests.get(
            "http://localhost:11434/api/tags",
            timeout=3,
        )
        resp.raise_for_status()
        tags = resp.json()
        live = [
            m.get("name", "")
            for m in tags.get("models", [])
            if m.get("name")
        ]
        # Merge live models with the curated KNOWN_MODELS list so that
        # cloud-only entries (which may not appear in /api/tags) are still
        # offered as completion candidates.
        merged: list[str] = []
        seen: set[str] = set()
        for m in live + KNOWN_MODELS:
            if m and m not in seen:
                seen.add(m)
                merged.append(m)
        return sorted(merged)
    except Exception:
        # Ollama not reachable or requests missing: fall back to the
        # static list compiled from config.py comments.
        return list(KNOWN_MODELS)


#: If True, the agent requests NDJSON streaming from Ollama and prints
#: tokens as they arrive.  Can be toggled at runtime via /stream.
STREAM: bool = True

# ---- Logging ---------------------------------------------------------

LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

#: Main log: lifecycle events, tool-call summaries, one-line HTTP summaries.
LOG_FILE: str = "agentThree/agent.log"

#: Full request/response bodies with the Ollama endpoint (DEBUG).
HTTP_LOG_FILE: str = "agentThree/agent.http.log"

#: Full JSON payloads sent to the LLM, one block per request (INFO).
LLM_PAYLOAD_LOG_FILE: str = "agentThree/agent.llm_payload.log"

# ---- Filesystem / workspace safety -----------------------------------

#: Root directory inside which all file tools are allowed to operate.
WORKSPACE_ROOT: Path = Path(os.getcwd()).resolve()

#: Extensions that mutating file tools may create/modify. Must be a
#: tuple (or any iterable accepted by ``str.endswith``) - a bare ``str``
#: of several extensions would silently let all of them through.
ALLOWED_EXTENSION: tuple[str, ...] = (".py", ".md", ".txt", ".json", ".scad", ".html")

# ---- Diff tool -------------------------------------------------------

#: Path to a kdiff3-compatible binary. Set at startup by
#: ``agentThree.diff_tool.configure_diff_tool``; ``None`` means ask
#: the user / fall back to the text diff.
DIFF_TOOL_PATH: str = "C:/Program Files/KDiff3/bin/kdiff3.exe"