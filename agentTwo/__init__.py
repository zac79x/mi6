"""Ollama tool-calling agent - package.

This is the modularised form of the original ``agentNew.py`` script.
It groups the agent's responsibilities into focused submodules:

    config            - module-level constants
    logging_setup     - logger configuration and small string helpers
    tools_registry    - ``Tool`` dataclass and ``@tool`` decorator
    safety            - workspace path validation
    diff_tool         - kdiff3 configuration and diff-approval workflow
    tools_misc        - ``calculate``, ``get_current_time``, ``word_count``
    tools_filesystem  - file/directory tools
    agent             - the :class:`agentTwo` class itself
    repl              - the interactive ``main()`` entry point

The public API (``agentTwo``, ``main``, ``tool``, ``DEFAULT_MODEL``,
``OLLAMA_URL``) is re-exported here so callers can keep doing
``from agent import agentTwo``.
"""

# Importing submodules has the side-effect of registering all built-in tools
# with the global ``@tool`` registry, which is required before constructing
# an :class:`agentTwo` instance.
from agent import config               # noqa: F401  (sets up constants)
from agent import logging_setup        # noqa: F401  (configures the loggers)
from agent import tools_misc           # noqa: F401  (registers small tools)
from agent import tools_filesystem     # noqa: F401  (registers file tools)
from agent import diff_tool            # noqa: F401  (registers /kdiff handler)

from agent.agent import agentTwo
from agent.tools_registry import tool
from agent.config import (
    DEFAULT_MODEL,
    OLLAMA_URL,
    LOG_FILE,
    WORKSPACE_ROOT,
)
from agent.repl import main

__all__ = [
    "agentTwo",
    "DEFAULT_MODEL",
    "LOG_FILE",
    "OLLAMA_URL",
    "WORKSPACE_ROOT",
    "main",
    "tool",
]