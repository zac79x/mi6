"""Diagnostic: dump the state of the agent after the first wire-build."""
from agentTwo import agentTwo, DEFAULT_MODEL, CACHE_THRESHOLD_CHARS

a = agentTwo(model=DEFAULT_MODEL, verbose=False)
print(f"a._result_cache at start: {list(a._result_cache.items())[:3]}")
print(f"a.messages at start: {a.messages!r}")

probe_path = r"C:\docs\projects\python\agent3\agent\agent.py"
with open(probe_path, "r", encoding="utf-8") as fh:
    big = fh.read()
print(f"big file length: {len(big)} (threshold={CACHE_THRESHOLD_CHARS})")

a.messages.append({"role": "user", "content": "show me agent.py"})
a.messages.append({
    "role": "assistant", "content": "",
    "tool_calls": [{"function": {"name": "read_text_file", "arguments": {"path": probe_path}}}],
})
a.messages.append({"role": "tool", "content": big, "_call_key": f"read_text_file@{probe_path}"})

print(f"\na.messages before wire build ({len(a.messages)} entries):")
for i, m in enumerate(a.messages):
    role = m.get("role")
    content_preview = repr(m.get("content", ""))[:60]
    print(f"  [{i}] role={role!r}  content={content_preview}  _call_key={m.get('_call_key')!r}")
    if m.get("tool_calls"):
        print(f"        tool_calls={m['tool_calls']}")

wire1 = a._build_wire_messages()
print(f"\nwire1 has {len(wire1)} entries:")
for i, m in enumerate(wire1):
    role = m.get("role")
    content_preview = repr(m.get("content", ""))[:80]
    print(f"  [{i}] role={role!r}  content={content_preview}")

print(f"\nlen(a._result_cache) = {len(a._result_cache)}")
print(f"a._result_cache keys = {list(a._result_cache.keys())}")
for k, v in a._result_cache.items():
    print(f"  {k} -> {len(v)} chars (matches big? {v == big})")