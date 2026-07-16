"""Ollama tool-calling agent - package.

This is the modularised form of the original ``agentNew.py`` script.
It groups the agent's responsibilities into focused submodules:

    config            - module-level constants
    logging_setup     - logger configuration and small string helpers
    tools_registry    - ``Tool`` dataclass and ``@tool`` decorator
    safety            - workspace path validation
    diff_tool         - diff-then-approve workflow (uses hard-coded
                        ``DIFF_TOOL_PATH`` from config)
    tools_misc        - ``calculate``, ``get_current_time``, ``word_count``
    tools_filesystem  - file/directory tools
    agent             - the :class:`agentTwo` class itself
    repl              - the interactive ``main()`` entry point

The public API (``agentTwo``, ``main``, ``tool``, ``DEFAULT_MODEL``,
``OLLAMA_URL``) is re-exported here so callers can keep doing
``from agentTwo import agentTwo``.
"""

# Importing submodules has the side-effect of registering all built-in tools
# with the global ``@tool`` registry, which is required before constructing
# an :class:`agentTwo` instance.
from agentTwo import config               # noqa: F401  (sets up constants)
from agentTwo import logging_setup        # noqa: F401  (configures the loggers)
from agentTwo import tools_misc           # noqa: F401  (registers small tools)
from agentTwo import tools_filesystem     # noqa: F401  (registers file tools)
from agentTwo import diff_tool            # noqa: F401  (diff-then-approve workflow)

from agentTwo.agent import agentTwo
from agentTwo.tools_registry import tool
from agentTwo.config import (
    DEFAULT_MODEL,
    OLLAMA_URL,
    LOG_FILE,
    WORKSPACE_ROOT,
)
from agentTwo.repl import main

__all__ = [
    "agentTwo",
    "DEFAULT_MODEL",
    "LOG_FILE",
    "OLLAMA_URL",
    "WORKSPACE_ROOT",
    "main",
    "tool",
]