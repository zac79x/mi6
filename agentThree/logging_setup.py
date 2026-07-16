"""Logger configuration and small string helpers."""

from __future__ import annotations

import json
import logging
from typing import Any

from agentThree.config import HTTP_LOG_FILE, LLM_PAYLOAD_LOG_FILE, LOG_FILE, LOG_FORMAT

logger = logging.getLogger("agentThree")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
logger.propagate = False

try:
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(fh)
except OSError as exc:
    logger.warning("Could not open log file %s: %s", LOG_FILE, exc)


def _add_sublogger(sub_name: str, log_path: str, level: int = logging.DEBUG) -> None:
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
        logger.warning("Could not open log file %s: %s", log_path, exc)


http_logger = logging.getLogger("ollama_agent.http")
_add_sublogger("ollama_agent.http", HTTP_LOG_FILE)

llm_payload_logger = logging.getLogger("ollama_agent.llm_payload")
_add_sublogger("ollama_agent.llm_payload", LLM_PAYLOAD_LOG_FILE)


def truncate(text: str, limit: int = 200) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"... [truncated, total {len(text)} chars]"


def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(obj)
