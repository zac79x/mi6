"""Smoke-test import: load the package and construct an agentTwo.

Used after a non-trivial refactor to catch:
- syntax errors that py_compile misses (e.g. an import-time side
  effect that NameErrors),
- registry/circular-import issues,
- dataclass changes that broke an old ``Tool`` constructor site.

Run from the project root:

    python _smoketest.py

Exits 0 on success, non-zero on failure. Prints a small report so
the failure is diagnosable from the log alone.
"""

from __future__ import annotations

import sys
import traceback


def main() -> int:
    print("step 1: import agentTwo package ...")
    try:
        import agentTwo
    except Exception:
        traceback.print_exc()
        return 1
    print("  ok")

    print("step 2: import agentTwo class directly ...")
    try:
        from agentTwo.agent import agentTwo
    except Exception:
        traceback.print_exc()
        return 2
    print("  ok")

    print("step 3: construct agentTwo (no Ollama call) ...")
    try:
        a = agentTwo()
    except Exception:
        traceback.print_exc()
        return 3
    print("  ok; tools registered:", [t.name for t in a.tools])

    print("step 4: confirm new attributes exist ...")
    for attr in ("_event_handlers", "_stats", "_undo_stack", "_redo_stack"):
        if not hasattr(a, attr):
            print(f"  FAIL: agentTwo has no attribute {attr!r}")
            return 4
    print("  ok")

    print("step 5: register a handler and fire an event ...")
    seen: list[tuple[str, dict]] = []
    a.on_event(lambda kind, payload: seen.append((kind, payload)))
    a._emit("test", {"x": 1})
    if seen != [("test", {"x": 1})]:
        print(f"  FAIL: handler did not see event; saw {seen!r}")
        return 5
    print("  ok")

    print("step 6: token_stats() returns a dict ...")
    stats = a.token_stats()
    if not isinstance(stats, dict):
        print(f"  FAIL: token_stats returned {type(stats).__name__}")
        return 6
    if "prompt_tokens" not in stats:
        print(f"  FAIL: stats missing 'prompt_tokens'; got {list(stats)}")
        return 6
    print("  ok; stats keys:", sorted(stats.keys()))

    print("step 7: undo/redo round-trip ...")
    a.messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    n = a.undo_last_turn()
    if n != 2:
        print(f"  FAIL: undo removed {n} messages, expected 2")
        return 7
    if len(a.messages) != 2:
        print(f"  FAIL: after undo, messages len is {len(a.messages)}, expected 2")
        return 7
    n2 = a.redo_last_turn()
    if n2 != 2:
        print(f"  FAIL: redo added {n2} messages, expected 2")
        return 7
    if len(a.messages) != 4:
        print(f"  FAIL: after redo, messages len is {len(a.messages)}, expected 4")
        return 7
    print("  ok")

    print("step 8: Tool dataclass has permission field ...")
    from agentTwo.tools_registry import Tool
    if not hasattr(Tool, "permission"):
        print("  FAIL: Tool has no 'permission' field")
        return 8
    sample = Tool(name="x", description="d", func=lambda: None, parameters={})
    if sample.permission != "allow":
        print(f"  FAIL: default permission is {sample.permission!r}, expected 'allow'")
        return 8
    print("  ok")

    print("step 9: filesystem tools declare permission='ask' ...")
    from agentTwo.tools_filesystem import create_file, update_file  # noqa: F401
    from agentTwo.tools_registry import _TOOL_REGISTRY
    if _TOOL_REGISTRY["create_file"].permission != "ask":
        print(f"  FAIL: create_file permission is {_TOOL_REGISTRY['create_file'].permission!r}")
        return 9
    if _TOOL_REGISTRY["update_file"].permission != "ask":
        print(f"  FAIL: update_file permission is {_TOOL_REGISTRY['update_file'].permission!r}")
        return 9
    print("  ok")

    print()
    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())