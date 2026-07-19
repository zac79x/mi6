"""Ollama tool-calling agent - package.

Re-exports the public API: ``agentThree``, ``main``, ``tool``,
``DEFAULT_MODEL``, ``OLLAMA_URL``.
"""

from agentThree import config
from agentThree import logging_setup
from agentThree import tools_misc
from agentThree import tools_filesystem
from agentThree import tools_web
from agentThree import diff_tool
from agentThree import a2a_tools  # noqa: F401  - registers A2A tools

from agentThree.agent import agentThree
from agentThree.tools_registry import tool
from agentThree.config import DEFAULT_MODEL, OLLAMA_URL, LOG_FILE, WORKSPACE_ROOT
from agentThree.repl import main

__all__ = [
    "agentThree",
    "DEFAULT_MODEL",
    "LOG_FILE",
    "OLLAMA_URL",
    "WORKSPACE_ROOT",
    "main",
    "tool",
]