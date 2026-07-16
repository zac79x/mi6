"""Tool registry: Tool dataclass and @tool decorator."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from agentThree.logging_setup import logger

_TYPE_MAP: dict[type, str] = {
    int: "integer", float: "number", str: "string",
    bool: "boolean", list: "array", dict: "object",
}


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters: dict

    def to_ollama_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_TOOL_REGISTRY: dict[str, Tool] = {}


def _first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    lines = [ln.strip() for ln in doc.strip().splitlines() if ln.strip()]
    return " ".join(lines)


def _schema_from_signature(func: Callable) -> dict:
    sig = inspect.signature(func)
    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        ann = param.annotation
        json_type = _TYPE_MAP.get(ann, "string") if ann is not inspect.Parameter.empty else "string"
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
    def wrap(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or _first_paragraph(func.__doc__)
        tool_params = parameters or _schema_from_signature(func)
        _TOOL_REGISTRY[tool_name] = Tool(tool_name, tool_desc, func, tool_params)
        logger.debug("Registered tool: %s", tool_name)
        return func

    if _func is not None and callable(_func):
        return wrap(_func)
    return wrap
