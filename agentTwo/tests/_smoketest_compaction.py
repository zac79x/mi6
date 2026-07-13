"""Smoke test for /compact, /listcache, and the cache_threshold_chars ctor arg.

Doesn't talk to Ollama at all - just drives the agentTwo's compaction code
directly, then exercises the new REPL command handlers (which only need
an agentTwo and a string).
"""
import json
import sys

from agent.agent import agentTwo, CACHE_THRESHOLD_CHARS
from agent.repl import _handle_command


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def make_history():
    """Build a fake conversation that exercises every compaction pass.

    Two big read_text_file results (Pass 1 -> cache), one repeated
    error message (Pass 2 -> dedup), one assistant turn with thinking
    (Pass 3 -> strip), and one assistant turn with empty content +
    tool_calls (Pass 4 -> drop content).
    """
    big1 = "# file one\n" + ("x" * 3000)
    big2 = "# file two\n" + ("y" * 3000)
    err  = "Error: file './nope' does not exist"

    return [
        {"role": "system", "content": "you are a test"},  # skipped, rebuilt per-call
        {"role": "user",   "content": "read two files and summarise"},
        # turn 1: read big1
        {"role": "assistant", "content": "", "thinking": "long reasoning...",
         "tool_calls": [{"id": "c1", "function": {"name": "read_text_file",
                                                  "arguments": {"path": "a.py"}}}]},
        {"role": "tool", "content": big1, "_call_key": "read_text_file@a.py"},
        # turn 2: read big2
        {"role": "assistant", "content": "", "thinking": "more reasoning...",
         "tool_calls": [{"id": "c2", "function": {"name": "read_text_file",
                                                  "arguments": {"path": "b.py"}}}]},
        {"role": "tool", "content": big2, "_call_key": "read_text_file@b.py"},
        # turn 3: read missing file, then try again, then succeed -> Pass 2 dedup
        {"role": "assistant", "content": "", "thinking": "retry",
         "tool_calls": [{"id": "c3", "function": {"name": "read_text_file",
                                                  "arguments": {"path": "x"}}}]},
        {"role": "tool", "content": err, "_call_key": "read_text_file@x"},
        {"role": "assistant", "content": "", "thinking": "retry 2",
         "tool_calls": [{"id": "c4", "function": {"name": "read_text_file",
                                                  "arguments": {"path": "x"}}}]},
        {"role": "tool", "content": err, "_call_key": "read_text_file@x"},
        {"role": "assistant", "content": "", "thinking": "retry 3",
         "tool_calls": [{"id": "c5", "function": {"name": "read_text_file",
                                                  "arguments": {"path": "x"}}}]},
        {"role": "tool", "content": err, "_call_key": "read_text_file@x"},
        # final answer with no thinking
        {"role": "assistant", "content": "Done."},
    ]


def test_compact():
    section("Test 1: /compact shrinks the stored history and populates the cache")
    agent = agentTwo(verbose=False)  # default cache threshold = 2_000
    # Skip the system message in our fake history (rebuild on each call)
    agent.messages = make_history()[1:]

    before_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in agent.messages)
    print(f"Before: {len(agent.messages)} messages, {before_chars} chars, "
          f"cache has {len(agent._result_cache)} entries")

    report = agent.compact_history()

    after_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in agent.messages)
    print(f"After : {len(agent.messages)} messages, {after_chars} chars, "
          f"cache has {len(agent._result_cache)} entries")
    print(f"Report: {report}")

    # The two big files (3000+ chars) should have been cached; the
    # three identical error messages should have collapsed to one
    # full copy + two dedup markers; the three 'thinking' blocks
    # should be gone.
    cached_now = len(agent._result_cache)
    tool_msgs  = [m for m in agent.messages if m["role"] == "tool"]
    dedup_markers = [m for m in tool_msgs
                     if m["content"].startswith("[same as previous result")]
    cached_stubs   = [m for m in tool_msgs
                     if m["content"].startswith("[cached:")]
    full_results   = [m for m in tool_msgs if not (
        m["content"].startswith("[cached:") or
        m["content"].startswith("[same as previous result"))]

    assert cached_now == 2, f"expected 2 cached entries, got {cached_now}"
    assert len(cached_stubs) == 2, f"expected 2 cache stubs, got {len(cached_stubs)}"
    # 3 identical error messages: 1 full + 2 dedup markers
    assert len(full_results) == 1, f"expected 1 full result, got {len(full_results)}"
    assert len(dedup_markers) == 2, f"expected 2 dedup markers, got {len(dedup_markers)}"
    # No 'thinking' fields left
    assert all("thinking" not in m for m in agent.messages if m["role"] == "assistant")
    # The system_prompt from the agentTwo is rebuilt each call, so we
    # can't assert on it here. Just check report sanity:
    assert report["chars_saved"] > 0, "expected chars_saved > 0"
    print("OK: /compact behaved as expected.")


def test_compact_threshold_arg():
    section("Test 2: cache_threshold_chars constructor arg actually takes effect")
    # Threshold of 100 chars: even the 3000-char results should be cached,
    # plus a 50-char result too.
    agent = agentTwo(verbose=False, cache_threshold_chars=100)
    agent.messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "ok", "tool_calls": []},
        {"role": "tool", "content": "tiny", "_call_key": "t@tiny"},
        {"role": "tool", "content": "x" * 50, "_call_key": "t@50"},
        {"role": "tool", "content": "y" * 3000, "_call_key": "t@big"},
    ]
    wire = agent._build_wire_messages()
    tool_wire = [m for m in wire if m["role"] == "tool"]
    # 50 chars > 100? no. 3000 > 100? yes.
    assert tool_wire[0]["content"] == "tiny", f"expected passthrough, got {tool_wire[0]}"
    assert tool_wire[1]["content"] == "x" * 50, "50-char result should be untouched"
    assert tool_wire[2]["content"].startswith("[cached:"), \
        f"3000-char result should be cached, got {tool_wire[2]}"
    assert len(agent._result_cache) == 1
    print(f"OK: threshold respected (cache size = {len(agent._result_cache)}).")


def test_listcache():
    section("Test 3: /listcache prints refs and sizes, not the content")
    agent = agentTwo(verbose=False)
    agent._result_cache = {
        "abcd1234": "secret" * 200,        # 1200 chars
        "deadbeef": "another " * 500,      # 4500 chars
    }
    entries = agent.list_cache()
    assert len(entries) == 2
    # Sorted by ref
    assert entries[0]["ref"] == "abcd1234"
    assert entries[1]["ref"] == "deadbeef"
    # Previews are short, single-line, and don't contain "secret" repeated 200x
    for e in entries:
        assert e["chars"] == len(agent._result_cache[e["ref"]])
        assert "secret" not in e["preview"] or e["chars"] <= 120, \
            f"preview should be truncated, got {e['preview']!r}"
    print(f"OK: list_cache returned {len(entries)} entries with sane previews.")
    for e in entries:
        print(f"  {e['ref']}  {e['chars']:>8} chars  | {e['preview']}")


def test_handle_command():
    section("Test 4: _handle_command recognises /compact and /listcache")
    agent = agentTwo(verbose=False)
    agent.messages = [{"role": "user", "content": "hi"}]

    # /listcache on an empty cache
    handled = _handle_command("/listcache", agent)
    assert handled is True, "/listcache should be handled (return True)"

    # /compact on a non-empty history
    agent._result_cache["x"] = "x" * 5000
    agent.messages.append({
        "role": "tool", "content": "x" * 5000, "_call_key": "big@thing",
    })
    handled = _handle_command("/compact", agent)
    assert handled is True, "/compact should be handled (return True)"

    # /compact on an empty history
    agent.messages = []
    handled = _handle_command("/compact", agent)
    assert handled is True, "/compact on empty should still be handled"

    # Unknown command falls through
    handled = _handle_command("/nosuchcommand", agent)
    assert handled is False, "unknown /command should fall through to the LLM"

    print("OK: all four /command cases routed correctly.")


if __name__ == "__main__":
    test_compact()
    test_compact_threshold_arg()
    test_listcache()
    test_handle_command()
    print("\nAll smoke tests passed.")