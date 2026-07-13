"""Tool registry: the ``Tool`` dataclass and the ``@tool`` decorator.

The :func:`tool` decorator inspects the wrapped function's signature and
docstring to build a minimal JSON Schema describing its arguments, and
registers it in the package-wide :data:`_TOOL_REGISTRY`. The
:class:`Agent` picks up registered tools automatically the first time
it is constructed.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from agent.logging_setup import logger

# Python type -> JSON Schema type
_TYPE_MAP: dict[type, str] = {
    int:   "integer",
    float: "number",
    str:   "string",
    bool:  "boolean",
    list:  "array",
    dict:  "object",
}


@dataclass
class Tool:
    """A callable exposed to the LLM."""

    name: str
    description: str
    func: Callable
    parameters: dict

    def to_ollama_schema(self) -> dict:
        """Convert to the schema format expected by Ollama's /api/chat."""
        return {
            "type": "function",
            "function": {
                "name":        self.name,
                "description": self.description,
                "parameters":  self.parameters,
            },
        }


# Global registry. Populated by ``@tool`` at import time.
_TOOL_REGISTRY: dict[str, Tool] = {}


def _first_paragraph(doc: str | None) -> str:
    """Use the function's docstring (up to the first blank line) as description."""
    if not doc:
        return ""
    lines = [ln.strip() for ln in doc.strip().splitlines() if ln.strip()]
    return " ".join(lines)


def _schema_from_signature(func: Callable) -> dict:
    """Build a minimal JSON schema from the function's signature & type hints."""
    sig = inspect.signature(func)
    properties: dict[str, dict] = {}
    required:   list[str]       = []

    for name, param in sig.parameters.items():
        ann = param.annotation
        json_type = (
            _TYPE_MAP.get(ann, "string")
            if ann is not inspect.Parameter.empty
            else "string"
        )
        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def tool(
    _func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict | None = None,
) -> Callable:
    """Decorator that registers a function as a callable tool for the agent.

    Usage:
        @tool
        def add(a: int, b: int) -> int:
            \"\"\"Add two integers.\"\"\"
            return a + b

        @tool(description="Read a file", parameters={...})
        def read_file(path: str) -> str: ...
    """

    def wrap(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or _first_paragraph(func.__doc__)
        tool_params = parameters or _schema_from_signature(func)
        _TOOL_REGISTRY[tool_name] = Tool(tool_name, tool_desc, func, tool_params)
        logger.debug(
            "Registered tool: %s | description=%r | params=%s",
            tool_name, tool_desc, tool_params,
        )
        return func

    # Support both @tool and @tool(...)
    if _func is not None and callable(_func):
        return wrap(_func)
    return wrap
