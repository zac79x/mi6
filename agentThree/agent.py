"""Core agent module: the agentThree class that talks to an OpenAI-compatible chat-completions endpoint with tool calling."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import requests

from agentThree.config import DEFAULT_MODEL, OLLAMA_URL
from agentThree.approval import request_continue_approval
from agentThree.logging_setup import llm_payload_logger, logger, safe_json, truncate
from agentThree.tools_registry import Tool, _TOOL_REGISTRY

try:
    from agentThree.cli_ui import c as _colour, colours_enabled as _colours_enabled, Spinner
except ImportError:
    def _colour(text, *styles):
        return text
    def _colours_enabled():
        return False
    class Spinner:
        def __init__(self, text=""): pass
        def start(self): pass
        def stop(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass


CACHE_THRESHOLD_CHARS = 2_000
DEDUP_WINDOW = 5
CACHE_REF_HEX_LEN = 8
RECALL_CACHED_RESULT = "recall_cached_result"
DEFAULT_SESSION_FILE = ".agent_session_state.json"


class agentThree:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        ollama_url: str = OLLAMA_URL,
        max_iterations: int = 20,
        verbose: bool = True,
        temperature: float | None = None,
        show_thinking: bool = True,
        cache_threshold_chars: int = CACHE_THRESHOLD_CHARS,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.temperature = temperature
        self.show_thinking = show_thinking

        if cache_threshold_chars < 0:
            raise ValueError(f"cache_threshold_chars must be >= 0, got {cache_threshold_chars}")
        self.cache_threshold_chars = cache_threshold_chars

        self.system_prompt = system_prompt or (
            "You are a helpful assistant with access to tools. "
            "When a tool would help answer the user, call it. "
            "If a tool returns an error, explain it to the user. "
            "When you have the final answer, reply normally without calling any tool. "
            "Use `create_file` to create new files and `update_file` to modify "
            "existing files (update_file will show you a diff before applying)."
            "If you need to modify your agent scripts then read readme.md before"
            "reading the python scripts code, so that you know which files to modify,"
            "based on the user's request."
            "\n\nReading files: prefer reading the whole file in a single call "
            "rather than paging through it. For source files and most project "
            "files, use `read_text_file(path)` (default `max_chars=40000` is "
            "enough for typical files). Only fall back to `read_file_lines` "
            "with a `start_line`/`num_lines` slice when (a) `read_text_file` "
            "truncated the file and you need a specific later section, or "
            "(b) the file is genuinely too large to fit in one read. "
            "`read_file_lines` has a default `num_lines` of 2000, which "
            "covers an entire typical source file in a single call - "
            "there is no need to paginate in 25-, 100- or even 500-line "
            "chunks. Each paginated call costs a full LLM round-trip, "
            "so always prefer one large read over several small ones. "
            "If you do need to slice (the file is genuinely > 2000 lines), "
            "read at least 1000 lines per call and re-anchor `start_line` "
            "to the last `Lines a..b` header you received."
            "\n\nConversation-history compaction: large tool results in the "
            "history are replaced by short stubs of the form "
            "`[cached:<call_key> -> N chars; ref=<ref>]`. If you need the "
            "full content back, call the `recall_cached_result` tool with "
            "the `ref` from the stub. Identical consecutive tool results "
            "are also collapsed to `[same as previous result; not repeated]` "
            "- the first occurrence still has the full content. The result "
            "of a `recall_cached_result` call is *always* sent to the model "
            "verbatim on the next wire call - even when it is large enough "
            "to be cached in isolation - so the model can actually read the "
            "bytes it asked for."
        )

        self.tools = tools or list(_TOOL_REGISTRY.values())
        self.tool_map: dict[str, Tool] = {t.name: t for t in self.tools}
        self.messages: list[dict[str, Any]] = []
        self._result_cache: dict[str, str] = {}

        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.llm_call_count = 0
        self.state: str = "idle"
        self._session_start_time = time.time()

        logger.info(
            "Agent initialised | model=%s | url=%s | max_iterations=%d | temperature=%s | show_thinking=%s | cache_threshold_chars=%d | tools=%s",
            self.model, self.ollama_url, self.max_iterations, self.temperature, self.show_thinking, self.cache_threshold_chars, [t.name for t in self.tools],
        )

    @staticmethod
    def _format_tokens(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        if n >= 1_000:
            return f"{n / 1000:.1f}k"
        return str(n)

    def render_status_bar(self) -> str:
        turns = sum(1 for msg in self.messages if msg.get("role") == "user")
        temp_str = "default" if self.temperature is None else f"{self.temperature}"
        think_str = "on" if self.show_thinking else "off"
        return (
            f" LLM: {self.model} | state: {self.state} | temp: {temp_str} | think: {think_str} "
            f"| tokens: {self._format_tokens(self.session_prompt_tokens)} in / {self._format_tokens(self.session_completion_tokens)} out "
            f"| calls: {self.llm_call_count} | turns: {turns} "
        )

    def print_status_bar(self) -> None:
        bar = self.render_status_bar()
        print(_colour(bar, "dim", "gray") if _colours_enabled() else bar)

    @staticmethod
    def _make_cache_ref(result: str) -> str:
        return hashlib.sha1(result.encode("utf-8", errors="replace")).hexdigest()[:CACHE_REF_HEX_LEN]

    @staticmethod
    def _is_recall_tool_message(m: dict[str, Any]) -> bool:
        key = m.get("_call_key") or ""
        return key == RECALL_CACHED_RESULT or key.startswith(RECALL_CACHED_RESULT + "@")

    def _build_wire_messages(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        recent_tool_results: list[str] = []
        thinking_stripped = results_cached = results_deduped = recalls_preserved = 0

        for msg in self.messages:
            m = dict(msg)
            role = m.get("role")

            if role == "assistant":
                if m.pop("thinking", None):
                    thinking_stripped += 1
                if not m.get("content") and m.get("tool_calls"):
                    m.pop("content", None)

            elif role == "tool":
                content = m.get("content", "") or ""
                is_recall = self._is_recall_tool_message(m)

                if is_recall:
                    wire_content = content
                    if len(content) > self.cache_threshold_chars:
                        recalls_preserved += 1
                elif len(content) > self.cache_threshold_chars:
                    ref = self._make_cache_ref(content)
                    self._result_cache[ref] = content
                    call_key = m.get("_call_key") or "tool"
                    wire_content = f"[cached:{call_key} -> {len(content)} chars; ref={ref}]"
                else:
                    wire_content = content

                if wire_content and wire_content in recent_tool_results[-DEDUP_WINDOW:]:
                    m["content"] = "[same as previous result; not repeated]"
                    results_deduped += 1
                else:
                    if wire_content:
                        recent_tool_results.append(wire_content)
                    if not is_recall and len(content) > self.cache_threshold_chars:
                        m["content"] = wire_content
                        results_cached += 1

                m.pop("_call_key", None)
                if not m.get("content"):
                    m["content"] = "(empty result)"

            out.append(m)

        if thinking_stripped or results_cached or results_deduped or recalls_preserved:
            logger.debug(
                "Wire-message compaction: stripped 'thinking' from %d, cached %d large tool result(s), shipped %d recall(s) verbatim, deduped %d consecutive duplicate(s). Cache size: %d entry/ies.",
                thinking_stripped, results_cached, recalls_preserved, results_deduped, len(self._result_cache),
            )
        return out

    def _call_ollama(self) -> dict:
        wire_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *self._build_wire_messages(),
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": wire_messages,
            "stream": False,
        }
        if self.temperature is not None:
            payload["options"] = {"temperature": float(self.temperature)}
        if self.tools:
            payload["tools"] = [t.to_ollama_schema() for t in self.tools]

        payload_json = safe_json(payload)
        banner_line = "=" * 72
        llm_payload_logger.info("%s\nLLM REQUEST  ->  %s\n%s\n%s\n%s", banner_line, self.ollama_url, banner_line, payload_json, banner_line)
        logger.debug("HTTP request to %s\n%s", self.ollama_url, payload_json)

        t0 = time.time()
        try:
            resp = requests.post(self.ollama_url, json=payload, timeout=500)
            elapsed = time.time() - t0
            try:
                body_for_log = safe_json(resp.json()) if resp.ok else truncate(resp.text, 500)
            except Exception:
                body_for_log = truncate(resp.text, 500)
            logger.debug("HTTP response status=%d, elapsed=%.2fs | body\n%s", resp.status_code, elapsed, body_for_log)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout as exc:
            logger.error("Timeout calling Ollama after %.2fs: %s", time.time() - t0, exc)
            self.state = "error"
            raise
        except requests.exceptions.HTTPError as exc:
            body = exc.response.text if exc.response is not None else ""
            logger.error("HTTP error from Ollama: %s | body=%s", exc, truncate(body, 500))
            self.state = "error"
            raise
        except requests.exceptions.RequestException as exc:
            logger.error("RequestException calling Ollama: %s", exc)
            self.state = "error"
            raise
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode JSON response from Ollama: %s", exc)
            self.state = "error"
            raise

        logger.debug("Received response from Ollama in %.2fs | body\n%s", elapsed, safe_json(data))

        try:
            self.llm_call_count += 1
            prompt_toks = data.get("prompt_eval_count")
            if isinstance(prompt_toks, int):
                self.session_prompt_tokens += prompt_toks
            completion_toks = data.get("eval_count")
            if isinstance(completion_toks, int):
                self.session_completion_tokens += completion_toks
            total_toks = data.get("total_count")
            if isinstance(total_toks, int):
                self.session_total_tokens += total_toks
            else:
                p = prompt_toks if isinstance(prompt_toks, int) else 0
                c_tok = completion_toks if isinstance(completion_toks, int) else 0
                self.session_total_tokens += p + c_tok
        except Exception:
            logger.debug("Could not parse token usage from Ollama response", exc_info=True)

        return data

    def _recall_cached_result(self, raw_args: dict) -> str:
        ref = (raw_args or {}).get("ref", "")
        if not isinstance(ref, str) or not ref:
            return "Error: 'ref' is required and must be a non-empty string"
        if ref not in self._result_cache:
            available = ", ".join(sorted(self._result_cache)) or "(none)"
            return f"Error: no cached result with ref={ref!r}. Available refs: {available}"
        return self._result_cache[ref]

    def _execute_tool_call(self, tool_call: dict) -> str:
        fn = tool_call.get("function", {}) or {}
        name = fn.get("name", "")
        raw_args = fn.get("arguments", {}) or {}

        logger.debug("Tool call requested: %s | raw_args=%s", name, safe_json(raw_args))

        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                logger.error("Tool '%s' received malformed arguments JSON: %s | raw=%s", name, exc, truncate(raw_args))
                return f"Error: tool '{name}' received malformed arguments JSON"

        self.state = f"calling tool: {name}"
        self.print_status_bar()

        if name == RECALL_CACHED_RESULT:
            return self._recall_cached_result(raw_args)

        tool = self.tool_map.get(name)
        if tool is None:
            logger.error("Unknown tool requested: %s | available=%s", name, list(self.tool_map.keys()))
            return f"Error: unknown tool '{name}'"

        try:
            result = tool.func(**raw_args)
            result_str = str(result)
            logger.debug("Tool '%s' executed | args=%s | result=%s", name, safe_json(raw_args), truncate(result_str, 5000))
            return result_str
        except TypeError as exc:
            logger.error("Bad arguments for tool '%s': %s | args=%s", name, exc, safe_json(raw_args))
            return f"Error: bad arguments for '{name}': {exc}"
        except Exception as exc:
            logger.exception("Unhandled exception while executing tool '%s'", name)
            return f"Error executing '{name}': {exc}"

    @staticmethod
    def _call_key_for(name: str, raw_args: Any) -> str:
        if name == RECALL_CACHED_RESULT:
            return RECALL_CACHED_RESULT
        if not isinstance(raw_args, dict):
            return name
        for v in raw_args.values():
            if isinstance(v, str) and v:
                subject = v if len(v) <= 80 else v[:77] + "..."
                return f"{name}@{subject}"
        return name

    def chat(self, user_message: str) -> str:
        logger.debug("User message:\n%s", user_message)
        self.messages.append({"role": "user", "content": user_message})

        try:
            while True:
                for i in range(self.max_iterations):
                    logger.info("--- Agent iteration %d/%d ---", i + 1, self.max_iterations)
                    self.state = "thinking"
                    self.print_status_bar()
                    try:
                        with Spinner("thinking...", elapsed=True):
                            data = self._call_ollama()
                    except Exception as exc:
                        logger.exception("Iteration %d failed during Ollama call", i + 1)
                        self.state = "error"
                        raise

                    message = data.get("message", {}) or {}
                    logger.debug("Assistant message object:\n%s", safe_json(message))
                    self.messages.append(message)

                    thinking = message.get("thinking")
                    if self.show_thinking and thinking and self.verbose:
                        tag = _colour("[think]", "bold", "bright_blue")
                        print(f"  {tag} {truncate(str(thinking), 400)}")

                    tool_calls = message.get("tool_calls") or []
                    if not tool_calls:
                        final = (message.get("content") or "").strip()
                        logger.debug("Final assistant answer:\n%s", final)
                        self.state = "done"
                        return final

                    logger.debug("Model requested %d tool call(s): %s", len(tool_calls), safe_json(tool_calls))
                    for tc in tool_calls:
                        fn = tc.get("function", {}) or {}
                        fn_name = fn.get("name", "?")
                        fn_args = fn.get("arguments", {}) or {}
                        if isinstance(fn_args, str):
                            try:
                                fn_args = json.loads(fn_args)
                            except json.JSONDecodeError:
                                fn_args = {}
                        result = self._execute_tool_call(tc)
                        self.messages.append({
                            "role": "tool",
                            "content": result,
                            "_call_key": self._call_key_for(fn_name, fn_args),
                        })
                        if self.verbose:
                            preview = result if len(result) <= 120 else result[:120] + "..."
                            tag = _colour("[tool]", "bold", "bright_magenta")
                            name_col = _colour(fn_name, "magenta")
                            print(f"  {tag} {name_col}(...) -> {preview}")

                logger.warning("Reached max_iterations=%d without a final answer", self.max_iterations)
                self.state = "idle"
                if not request_continue_approval(self.max_iterations):
                    return "Sorry - I could not reach a final answer within the iteration limit."
        except KeyboardInterrupt:
            logger.info("Chat interrupted by user (Ctrl-C) at state=%r", self.state)
            self.state = "idle"
            print()
            if self.verbose:
                print(_colour("[interrupted by user - task stopped]", "bold", "bright_yellow") if _colours_enabled() else "[interrupted by user - task stopped]")
            return "[interrupted by user - task stopped]"

    def compact_history(self) -> dict[str, int]:
        before_count = len(self.messages)
        before_chars = sum(len(json.dumps(m, ensure_ascii=False, default=str)) for m in self.messages)
        before_cache = len(self._result_cache)

        self.messages = self._build_wire_messages()

        after_count = len(self.messages)
        after_chars = sum(len(json.dumps(m, ensure_ascii=False, default=str)) for m in self.messages)
        after_cache = len(self._result_cache)

        report = {
            "messages_before": before_count, "messages_after": after_count,
            "chars_before": before_chars, "chars_after": after_chars,
            "chars_saved": before_chars - after_chars,
            "cache_before": before_cache, "cache_after": after_cache,
        }
        logger.info("Manual history compaction: messages %d -> %d, chars %d -> %d (saved %d), cache %d -> %d entry/ies.",
                     before_count, after_count, before_chars, after_chars, before_chars - after_chars, before_cache, after_cache)
        return report

    def list_cache(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for ref, content in self._result_cache.items():
            preview = content if len(content) <= 120 else content[:117] + "..."
            preview = preview.replace("\n", "\\n").replace("\r", "\\r")
            out.append({"ref": ref, "chars": len(content), "preview": preview})
        out.sort(key=lambda d: d["ref"])
        return out

    def reset(self) -> None:
        cache_size = len(self._result_cache)
        logger.info("Conversation reset (cleared %d messages, %d cached result(s))", len(self.messages), cache_size)
        self.messages = []
        self._result_cache = {}
        self.state = "idle"

    def _settings_dict(self) -> dict[str, Any]:
        return {
            "model": self.model, "ollama_url": self.ollama_url,
            "max_iterations": self.max_iterations, "verbose": self.verbose,
            "temperature": self.temperature, "show_thinking": self.show_thinking,
            "cache_threshold_chars": self.cache_threshold_chars, "system_prompt": self.system_prompt,
        }

    def _apply_settings(self, settings: dict[str, Any]) -> None:
        for key in ("model", "ollama_url", "max_iterations", "verbose", "temperature", "show_thinking", "cache_threshold_chars", "system_prompt"):
            if key in settings:
                setattr(self, key, settings[key])

    def save_session(self, path: str | None = None) -> str:
        target = os.path.abspath(path or DEFAULT_SESSION_FILE)
        state = {"history": self.messages, "result_cache": self._result_cache, "settings": self._settings_dict()}
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2)
        logger.info("Session saved to %s (%d messages, %d cache entries).", target, len(self.messages), len(self._result_cache))
        return target

    def restore_session(self, path: str | None = None) -> dict[str, Any]:
        source = os.path.abspath(path or DEFAULT_SESSION_FILE)
        with open(source, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        self.messages = state.get("history", [])
        self._result_cache = state.get("result_cache", {})
        self._apply_settings(state.get("settings", {}))
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.llm_call_count = 0
        self.state = "idle"
        self._session_start_time = time.time()
        report = {"path": source, "messages": len(self.messages), "cache_entries": len(self._result_cache)}
        logger.info("Session restored from %s (%d messages, %d cache entries).", source, len(self.messages), len(self._result_cache))
        return report
