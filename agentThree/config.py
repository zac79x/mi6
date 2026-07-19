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
ALLOWED_EXTENSION: tuple[str, ...] = (".py", ".md", ".txt", ".json", ".scad")

# ---- Diff tool -------------------------------------------------------

#: Path to a kdiff3-compatible binary. Set at startup by
#: ``agentThree.diff_tool.configure_diff_tool``; ``None`` means ask
#: the user / fall back to the text diff.
DIFF_TOOL_PATH: str = "C:/Program Files/KDiff3/bin/kdiff3.exe"