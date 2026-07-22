"""Miscellaneous built-in tools: calculate, get_current_time, word_count, recall_cached_result."""

from __future__ import annotations

import ast
import operator
import time
from typing import Any

from agentThree.tools_registry import tool

_SAFE_OPS: dict[type, Any] = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> float:
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


@tool(description="Evaluate a safe arithmetic expression like ``2 * (3 + 4)``.")
def calculate(expression: str) -> str:
    try:
        return str(_safe_eval(expression))
    except Exception as exc:
        return f"Error: {exc}"


@tool(description="Return the current local date and time.")
def get_current_time() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@tool(description="Count the words in a piece of text.")
def word_count(text: str) -> int:
    return len(text.split())


@tool(description="Retrieve the full content of a tool result that was compacted on the wire.")
def recall_cached_result(ref: str) -> str:
    return ""
