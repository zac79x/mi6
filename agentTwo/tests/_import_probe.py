"""Smoke test: import the package and confirm the public surface works.

Run:  python _import_probe.py
"""
import sys

# Probe 1: importing the package must not raise.
from agentTwo import agentTwo, main, tool, DEFAULT_MODEL, OLLAMA_URL, LOG_FILE, WORKSPACE_ROOT
print("import OK")
print(f"  agentTwo        = {agentTwo}")
print(f"  main           = {main}")
print(f"  tool           = {tool}")
print(f"  DEFAULT_MODEL  = {DEFAULT_MODEL}")
print(f"  OLLAMA_URL     = {OLLAMA_URL}")
print(f"  LOG_FILE       = {LOG_FILE}")
print(f"  WORKSPACE_ROOT = {WORKSPACE_ROOT}")

# Probe 2: the agent module must expose the Ollama-specific bits we expect
# (this is the contract that the previous turn's bugfix relies on).
from agentTwo.agent import (
    agentTwo as _A,
    RECALL_CACHED_RESULT,
    CACHE_THRESHOLD_CHARS,
    DEDUP_WINDOW,
    CACHE_REF_HEX_LEN,
)
assert RECALL_CACHED_RESULT == "recall_cached_result"
assert isinstance(CACHE_THRESHOLD_CHARS, int) and CACHE_THRESHOLD_CHARS > 0
assert isinstance(DEDUP_WINDOW, int) and DEDUP_WINDOW > 0
assert isinstance(CACHE_REF_HEX_LEN, int) and CACHE_REF_HEX_LEN > 0
print(f"  RECALL_CACHED_RESULT  = {RECALL_CACHED_RESULT}")
print(f"  CACHE_THRESHOLD_CHARS = {CACHE_THRESHOLD_CHARS}")
print(f"  DEDUP_WINDOW          = {DEDUP_WINDOW}")
print(f"  CACHE_REF_HEX_LEN     = {CACHE_REF_HEX_LEN}")

# Probe 3: the agent must have the recall_cached_result and wire-building bits.
for name in ("_build_wire_messages", "_recall_cached_result",
             "_make_cache_ref", "_execute_tool_call",
             "chat", "compact_history", "list_cache", "reset"):
    assert hasattr(_A, name), f"agentTwo missing attribute: {name}"
print("  agentTwo has all expected methods")

# Probe 4: tools must be registered (importing the package triggers
# tools_misc and tools_filesystem to register their @tool functions).
from agentTwo.tools_registry import _TOOL_REGISTRY
expected = {
    "calculate", "get_current_time", "word_count",
    "read_text_file", "read_file_lines", "list_directory", "path_exists",
    "create_file", "update_file", "compile_python_file",
    "recall_cached_result",
}
missing = expected - _TOOL_REGISTRY.keys()
assert not missing, f"missing tools: {missing}"
print(f"  Registered tools: {sorted(_TOOL_REGISTRY)}")

# Probe 5: constructing an agentTwo must succeed (it builds the tool map
# and an empty history).
a = agentTwo(model=DEFAULT_MODEL, verbose=False)
assert a.messages == []
assert a._result_cache == {}
assert set(a.tool_map) == expected
print(f"  agentTwo(model={DEFAULT_MODEL!r}) constructed OK")
print(f"  tool_map has {len(a.tool_map)} tools")

# Probe 6: the recall_cached_result bug fix from the previous turn is
# still in place - i.e. the wire form, not the raw content, drives dedup.
# Simulate: read a large file (cached as stub), recall it, build the
# wire list, and assert the recalled message is NOT collapsed to
# "[same as previous result; not repeated]".
import json
import os

probe_path = os.path.abspath("agentTwo/agent.py")
with open(probe_path, "r", encoding="utf-8") as fh:
    big = fh.read()
assert len(big) > CACHE_THRESHOLD_CHARS, "probe needs a large file"
print(f"  probe file: {probe_path} ({len(big)} chars)")

a.messages.append({"role": "user", "content": "show me agent.py"})
a.messages.append({
    "role": "assistant", "content": "",
    "tool_calls": [{"function": {"name": "read_text_file", "arguments": {"path": probe_path}}}],
})
a.messages.append({"role": "tool", "content": big, "_call_key": f"read_text_file@{probe_path}"})

print(f"  a.messages has {len(a.messages)} entries before wire-build")
print(f"  a._result_cache is empty before wire-build: {a._result_cache == {}}")

# First wire-build: populates the cache with a ref for the read.
wire1 = a._build_wire_messages()
print(f"  after wire-build: cache has {len(a._result_cache)} entries")
print(f"  cache keys: {list(a._result_cache.keys())}")
print(f"  wire1 has {len(wire1)} entries")

# Be defensive: if the cache has more than 1 entry, dump full diagnostics
# so the next person to debug this has real numbers, not a generic
# unpacking error.
assert len(a._result_cache) >= 1, "expected at least 1 cache entry"
cache_keys = list(a._result_cache.keys())
for i, k in enumerate(cache_keys):
    v = a._result_cache[k]
    print(f"    [{i}] ref={k}  chars={len(v)}  matches_big={v == big}")

# Use the first key (deterministic). The exact ref doesn't matter for
# the regression test - what matters is that the recall gets through
# Pass 2.
ref = cache_keys[0]
recalled = a._recall_cached_result({"ref": ref})
assert recalled == big, f"recall should return the cached bytes; got {len(recalled)} chars"

a.messages.append({
    "role": "assistant", "content": "",
    "tool_calls": [{"function": {"name": "recall_cached_result", "arguments": {"ref": ref}}}],
})
a.messages.append({"role": "tool", "content": recalled, "_call_key": f"recall_cached_result@{ref}"})

# Second wire-build: the recalled message must NOT be collapsed.
wire2 = a._build_wire_messages()
tool_msgs = [m for m in wire2 if m.get("role") == "tool"]
print(f"  after second wire-build: {len(tool_msgs)} tool msgs on the wire, "
      f"cache has {len(a._result_cache)} entries")
for i, m in enumerate(tool_msgs):
    print(f"    tool[{i}]: {m['content'][:80]!r}")

assert len(tool_msgs) == 2, f"expected 2 tool messages on the wire, got {len(tool_msgs)}"
assert "[cached:" in tool_msgs[0]["content"]
assert "[same as previous" not in tool_msgs[1]["content"], (
    "BUG REGRESSED: recall_cached_result was wrongly deduped on the wire"
)
assert "[cached:" in tool_msgs[1]["content"], (
    "expected the recalled message to be a cache stub on the wire"
)
print("  Pass 1 / Pass 2 interaction: recall is NOT wrongly deduped. OK")

# Probe 7: consecutive small duplicates still dedup (Pass 2 contract).
print()
print("Probe 7: consecutive small duplicates still dedup")
b = agentTwo(model=DEFAULT_MODEL, verbose=False)
b.messages.append({"role": "user", "content": "hi"})
for i in range(2):
    b.messages.append({
        "role": "tool",
        "content": "small result: 42",
        "_call_key": f"foo{i}",
    })
wire_b = b._build_wire_messages()
tool_b = [m for m in wire_b if m.get("role") == "tool"]
assert tool_b[0]["content"] == "small result: 42", "first occurrence must be preserved"
assert tool_b[1]["content"] == "[same as previous result; not repeated]", (
    f"second occurrence should be deduped; got: {tool_b[1]['content']!r}"
)
print("  OK")

# Probe 8: repeat recall of the same ref is deduped on the wire.
print()
print("Probe 8: repeated recall of the same ref is deduped on the wire")
c = agentTwo(model=DEFAULT_MODEL, verbose=False)
c.messages.append({"role": "user", "content": "x"})
c.messages.append({
    "role": "tool", "content": big, "_call_key": f"read_text_file@{probe_path}",
})
c._build_wire_messages()
ref_c = next(iter(c._result_cache.keys()))
recalled_c = c._recall_cached_result({"ref": ref_c})
# First recall: append and wire-build (gets a fresh cache stub, NOT
# the same ref as the original read, because the recall produces the
# same bytes but the cache key for the *recalled message* is the hash
# of those bytes - which equals the original ref). Either way, the
# two wire stubs carry the same ref because they hash to the same
# content, and Pass 2 must dedup them.
c.messages.append({"role": "tool", "content": recalled_c, "_call_key": f"recall_cached_result@{ref_c}"})
c.messages.append({"role": "tool", "content": recalled_c, "_call_key": f"recall_cached_result@{ref_c}"})
wire_c = c._build_wire_messages()
tool_c = [m for m in wire_c if m.get("role") == "tool"]
print(f"  wire has {len(tool_c)} tool msgs")
for i, m in enumerate(tool_c):
    print(f"    [{i}]: {m['content'][:80]!r}")
# First is the original read (cached), second/third are recalls of the
# same content - their wire form is the same stub as the first one,
# so Pass 2 dedups them.
assert tool_c[0]["content"].startswith("[cached:")
# Consecutive dedup applies to the second & third (same recall twice
# in a row -> second is deduped).
assert tool_c[-1]["content"] == "[same as previous result; not repeated]", (
    f"repeated recall should be deduped; got: {tool_c[-1]['content']!r}"
)
print("  OK")

print("\nAll probes passed.")