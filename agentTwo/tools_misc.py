"""Miscellaneous built-in tools (``calculate``, ``get_current_time``,
``word_count``, ``recall_cached_result``). These are loaded by
:mod:`agentTwo` at import time so that the global tool registry is
populated before the :class:`agentTwo` is constructed.

The ``recall_cached_result`` tool is a small exception: it is
registered here (so the model sees a stable schema) but its actual
implementation lives on :class:`agentTwo.agent.agentTwo` because the
underlying cache is a per-agent attribute. The agent intercepts
calls to it in ``_execute_tool_call`` before they reach the global
registry.
"""

from __future__ import annotations

import ast
import operator
import time
from typing import Any

from agentTwo.tools_registry import tool

_SAFE_OPS: dict[type, Any] = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> float:
    """Evaluate a numeric expression that may only use safe operators."""
    tree = ast.parse(expr, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise ValueError("only numeric literals are allowed")
        if isinstance(node, ast.BinOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"operator {type(node.op).__name__} not allowed")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"unary operator {type(node.op).__name__} not allowed")
            return op(_eval(node.operand))
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    return _eval(tree)


@tool
def calculate(expression: str) -> str:
    """Evaluate a safe arithmetic expression like ``2 * (3 + 4)``."""
    try:
        return str(_safe_eval(expression))
    except Exception as exc:                       # noqa: BLE001
        return f"Error: {exc}"


@tool
def get_current_time() -> str:
    """Return the current local date and time."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


@tool
def recall_cached_result(ref: str) -> str:
    """Retrieve the full content of a tool result that was compacted
    on the wire (a stub of the form ``[cached:<call_key> -> N chars;
    ref=<ref>]``). Always prefer this over re-issuing the original
    read - the bytes are already cached, so recalling is one cheap
    call instead of another large result on the wire and in the
    history."""
    # This implementation is a placeholder. The agent's
    # ``_execute_tool_call`` intercepts calls to ``recall_cached_result``
    # before they reach the registry, so the model never actually sees
    # this return value. We still need a real function here because
    # ``@tool`` validates the signature and registers the schema.
    return ""