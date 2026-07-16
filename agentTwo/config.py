"""Module-level configuration constants for the agentTwo package.

Centralising every magic string, file path, and tunable default in one
place keeps the rest of the package free of inline literals and makes
the agent easier to configure for different deployments.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---- HTTP / model -----------------------------------------------------

#: Default endpoint of the local Ollama server.
OLLAMA_URL: str = "http://localhost:11434/api/chat"

#: Default model identifier (any tool-capable Ollama model works).
DEFAULT_MODEL: str = "minimax-m3:cloud"
#DEFAULT_MODEL: str = "kimi-k2.7-code:cloud"
#DEFAULT_MODEL: str = "glm-5.2:cloud"
#DEFAULT_MODEL: str = "deepseek-v4-pro:cloud"
#DEFAULT_MODEL: str = "deepseek-v4-flash:cloud"
#DEFAULT_MODEL: str = "minimax-m2.7:cloud"
#DEFAULT_MODEL: str = "gemma4:31b-cloud"

# ---- Logging ---------------------------------------------------------

#: Format string used by the agent's log handlers.
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

#: File the file handler appends to.  Holds the agent's normal log
#: lines: lifecycle events, tool-call summaries, compaction stats,
#: one-line summaries of each HTTP exchange with the LLM, etc.  The
#: full request/response bodies go to the two dedicated loggers
#: (``ollama_agent.http`` and ``ollama_agent.llm_payload``) so the
#: main log stays small enough to ``tail -f`` comfortably even on
#: long sessions.
LOG_FILE: str = "agentTwo/agent.log"

#: File for the full request/response bodies exchanged with the
#: Ollama HTTP endpoint.  Written by the ``ollama_agent.http``
#: sub-logger at DEBUG level.  Kept separate from :data:`LOG_FILE`
#: because each exchange can be hundreds of KB of pretty-printed
#: JSON.
HTTP_LOG_FILE: str = "agentTwo/agent.http.log"

#: File for the full JSON payloads sent to the LLM (one banner
#: block per request).  Written by the ``ollama_agent.llm_payload``
#: sub-logger at INFO level.  Useful for replaying or diffing
#: prompts between runs without re-running the session.
LLM_PAYLOAD_LOG_FILE: str = "agentTwo/agent.llm_payload.log"

# ---- Filesystem / workspace safety -----------------------------------

#: Root directory inside which all file tools are allowed to operate.
WORKSPACE_ROOT: Path = Path(os.getcwd()).resolve()

#: The file extensions the agent's *mutating* tools (``create_file`` and
#: ``update_file``) are permitted to create or modify.  This is a tuple
#: of extensions; :func:`agentTwo.safety.validate_path` accepts any path
#: whose suffix is one of these (via ``str.endswith``).  Read-only tools
#: and ``compile_python_file`` are not bound by this list - the latter
#: still only accepts ``.py`` because it actually compiles Python.
#:
#: Note: this must stay a *tuple* (or any iterable of strings accepted
#: by ``str.endswith``); annotating it as a single ``str`` while listing
#: several extensions would silently let every listed extension through
#: while the type annotation claimed otherwise.
ALLOWED_EXTENSION: tuple[str, ...] = (".py", ".md", ".txt", ".json", ".scad")

# ---- Diff tool -------------------------------------------------------

#: Path to a kdiff3-compatible diff/merge binary. Populated at startup
#: by :func:`agentTwo.diff_tool.configure_diff_tool`. ``None`` means "ask
#: the user at startup" / "use the text-diff fallback".
DIFF_TOOL_PATH: str = "C:/Program Files/KDiff3/bin/kdiff3.exe"