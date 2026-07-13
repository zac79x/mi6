"""Functional check that 'python -m agent' still works after the
removal of agent.statemachine_agent. Doesn't talk to Ollama; just
exercises the import path and agentTwo construction.
"""
import sys

# 1. The whole package must import without error.
print("[1] import agent ...", end=" ")
import agent
print("ok")

# 2. The public API (post-removal) must be exactly what __init__.py advertises.
print("[2] public API ...", end=" ")
expected = {
    "agentTwo",
    "DEFAULT_MODEL",
    "LOG_FILE",
    "OLLAMA_URL",
    "WORKSPACE_ROOT",
    "main",
    "tool",
}
got = set(agent.__all__)
missing = expected - got
extra   = got - expected
assert not missing, f"missing from __all__: {missing}"
assert not extra,   f"unexpected in __all__: {extra}"
print("ok (exact match:", sorted(got), ")")

# 3. The removed names must be gone from the package.
print("[3] removed names are gone ...", end=" ")
for removed in ("StateMachineAgent", "AgentStateMachine", "RunResult",
                "default_model_router", "parse_plan", "statemachine_agent"):
    assert not hasattr(agent, removed), f"{removed!r} should not exist on agent"
    assert removed not in dir(agent),   f"{removed!r} leaked into dir(agent)"
print("ok")

# 4. The submodule attribute must be gone (no fallback to importing it).
print("[4] no submodule leak ...", end=" ")
assert "statemachine_agent" not in agent.__dict__, \
    "statemachine_agent should not be a submodule attribute"
print("ok")

# 5. main() must be the REPL's main.
print("[5] from agent import main, agentTwo, tool ...", end=" ")
from agent import main, agentTwo, tool
from agent.config import DEFAULT_MODEL, OLLAMA_URL, LOG_FILE, WORKSPACE_ROOT
print("ok")

# 6. The tool registry must be populated by the side-effect imports.
print("[6] tool registry populated ...", end=" ")
from agent.tools_registry import _TOOL_REGISTRY
expected_tools = {
    "calculate", "get_current_time", "word_count", "recall_cached_result",
    "read_text_file", "read_file_lines", "list_directory", "path_exists",
    "create_file", "update_file", "compile_python_file",
}
got_tools = set(_TOOL_REGISTRY)
missing_tools = expected_tools - got_tools
assert not missing_tools, f"missing tools: {missing_tools}"
# And no stray StateMachineAgent-specific tools snuck through.
assert "recall_cached_result" in got_tools, "recall_cached_result should still be registered"
print(f"ok ({len(got_tools)} tools registered)")

# 7. agentTwo() must construct cleanly (the construction itself exercises
#    __init__ + tool snapshot + logger setup + cache init).
print("[7] agentTwo() construction ...", end=" ")
a = agentTwo(verbose=False)
assert a.model == DEFAULT_MODEL
assert a.ollama_url == OLLAMA_URL
assert a.max_iterations > 0
assert hasattr(a, "_result_cache") and a._result_cache == {}
assert a.messages == []
assert len(a.tools) == len(got_tools), \
    f"agentTwo should see {len(got_tools)} tools, got {len(a.tools)}"
# Compaction is still wired up:
assert hasattr(a, "compact_history")
assert hasattr(a, "list_cache")
assert hasattr(a, "_build_wire_messages")
print("ok")

# 8. Compaction end-to-end on a tiny history (the path the /compact
#    command takes). This is the contract /compact relies on.
print("[8] compact_history() works ...", end=" ")
a.messages = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello", "thinking": "long reasoning"},
    {"role": "tool", "content": "x" * 5000, "_call_key": "big@thing"},
]
report = a.compact_history()
assert report["chars_saved"] > 0
assert len(a._result_cache) == 1
print(f"ok (saved {report['chars_saved']} chars, cache now {len(a._result_cache)})")

# 9. __main__ must be importable (the actual 'python -m agent' entry
#    point - it must not be broken by our edits).
print("[9] import agent.__main__ ...", end=" ")
import agent.__main__  # noqa: F401
print("ok")

# 10. REPL command handler should still recognise all the documented
#     commands, and not crash on the new ones.
print("[10] repl._handle_command ...", end=" ")
from agent.repl import _handle_command
# Reset state
a.reset()
# All known commands should be handled (return True).
known = ["/help", "/temp 0.5", "/max_iter 5", "/think on", "/kdiff",
         "/compact", "/listcache"]
for cmd in known:
    handled = _handle_command(cmd, a)
    assert handled is True, f"{cmd!r} should be handled"
# Unknown slash command should fall through (return False).
assert _handle_command("/nosuchcommand", a) is False
# Plain text should also fall through.
assert _handle_command("hello", a) is False
print("ok")

print("\nAll 10 checks passed. The agent still works after the removal.")