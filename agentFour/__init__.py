"""Ollama tool-calling agent - package.

Re-exports the public API: ``agentFour``, ``main``, ``tool``,
``DEFAULT_MODEL``, ``OLLAMA_URL``.
"""

from agentFour import config
from agentFour import logging_setup
from agentFour import tools_misc
from agentFour import tools_filesystem
from agentFour import tools_web
from agentFour import diff_tool
from agentFour import a2a_tools  # noqa: F401  - registers A2A tools

from agentFour.agent import agentFour
from agentFour.tools_registry import tool
from agentFour.config import DEFAULT_MODEL, OLLAMA_URL, LOG_FILE, WORKSPACE_ROOT
from agentFour.repl import main

__all__ = [
    "agentFour",
    "DEFAULT_MODEL",
    "LOG_FILE",
    "OLLAMA_URL",
    "WORKSPACE_ROOT",
    "main",
    "tool",
]