"""Logger configuration and small string helpers used across the package.

Importing this module configures the package-wide loggers exactly once.
The helpers :func:`truncate` and :func:`safe_json` are intentionally
lightweight: they exist so the rest of the code can produce readable
log lines without worrying about size or serialisability.

Log layout (as of the "split the noisy stuff out of agent.log" change):

* ``agent/agent.log`` (the :data:`agent.config.LOG_FILE`) - the main
  log.  Receives the root ``agent`` logger's records: lifecycle
  events, tool-call summaries, compaction stats, errors, and a
  one-line summary of every HTTP exchange with the LLM.  Suitable
  for ``tail -f`` even on long sessions.

* ``agent/agent.http.log`` (the :data:`agent.config.HTTP_LOG_FILE`) -
  receives the ``ollama_agent.http`` sub-logger's records: the full
  request and response bodies for every Ollama call.  Each call can
  be hundreds of KB of pretty-printed JSON, so this is kept off the
  main log.

* ``agent/agent.llm_payload.log`` (the
  :data:`agent.config.LLM_PAYLOAD_LOG_FILE`) - receives the
  ``ollama_agent.llm_payload`` sub-logger's records: the full JSON
  payload sent to the LLM, banner-delimited.  Useful for replaying
  or diffing prompts between runs.

The sub-loggers have ``propagate=False`` so their records do not
also land in the main log; the root ``agent`` logger's own
DEBUG-level lines (e.g. compaction stats) still go to ``agent.log``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.config import (
    HTTP_LOG_FILE,
    LLM_PAYLOAD_LOG_FILE,
    LOG_FILE,
    LOG_FORMAT,
)

# ---- Logger wiring ---------------------------------------------------

# Root logger for the agent. Captures everything; per-handler levels
# control what is actually persisted/displayed.
logger = logging.getLogger("agent")
logger.setLevel(logging.DEBUG)

# Reset any existing handlers (useful when re-importing in notebooks/tests)
logger.handlers.clear()
logger.propagate = False

try:
    _file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(_file_handler)
except OSError as exc:
    logger.warning("Could not open log file %s: %s", LOG_FILE, exc)


def _add_sublogger_file_handler(
    sub_name: str,
    log_path: str,
    level: int = logging.DEBUG,
) -> None:
    """Wire a sub-logger to its own file and stop it from propagating
    to the root ``agent`` logger.

    The default behaviour of :mod:`logging` is for sub-loggers to
    forward their records up to the root logger (and therefore to
    every handler attached there).  That is exactly what we do *not*
    want for the HTTP and LLM-payload sub-loggers: their records are
    the largest in the system, and the whole point of giving them
    their own files is to keep them out of :data:`LOG_FILE`.

    Idempotent: clears any pre-existing handlers on the sub-logger
    before attaching the new one, so re-importing the module (e.g.
    in tests / notebooks) does not stack duplicate handlers.
    """
    sub = logging.getLogger(sub_name)
    sub.setLevel(level)
    sub.handlers.clear()
    sub.propagate = False
    try:
        h = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        h.setLevel(level)
        h.setFormatter(logging.Formatter(LOG_FORMAT))
        sub.addHandler(h)
    except OSError as exc:
        # Don't crash agent startup if the dedicated log file is
        # unwritable - just log a warning and let the sub-logger run
        # with no handlers (its records are dropped on the floor).
        logger.warning("Could not open log file %s: %s", log_path, exc)


# Dedicated logger for HTTP traffic.  Gets its own file so the
# full request/response bodies do not bloat the main log.
http_logger = logging.getLogger("ollama_agent.http")
_add_sublogger_file_handler("ollama_agent.http", HTTP_LOG_FILE)

# Dedicated logger for the JSON payload sent to the LLM.  Gets its
# own file so the pretty-printed JSON of every prompt does not
# bloat the main log.
llm_payload_logger = logging.getLogger("ollama_agent.llm_payload")
_add_sublogger_file_handler("ollama_agent.llm_payload", LLM_PAYLOAD_LOG_FILE)


# ---- Helpers ---------------------------------------------------------

def truncate(text: str, limit: int = 200) -> str:
    """Shorten long strings for compact log lines."""
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + (
        f"... [truncated, total {len(text)} chars]"
    )


def safe_json(obj: Any) -> str:
    """Pretty-print an object as JSON, falling back to repr on failure."""
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(obj)
