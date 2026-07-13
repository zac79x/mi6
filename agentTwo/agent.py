"""Core agent module: the :class:`agentTwo` that talks to an OpenAI-compatible
chat-completions endpoint with tool calling, plus the compaction and
result-cache machinery that keeps long conversations affordable.

The agent is intentionally transport-agnostic with respect to the
tool implementations: every callable registered via :func:`agent.tool`
is reachable by name through :attr:`agentTwo.tool_map`.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

from agent.config import DEFAULT_MODEL, OLLAMA_URL
from agent.approval import request_continue_approval
from agent.logging_setup import (
    llm_payload_logger,
    logger,
    safe_json,
    truncate,
)
from agent.tools_registry import Tool, _TOOL_REGISTRY

# Human-friendly CLI helpers: ANSI colours (graceful degradation).  The
# import is soft: if `cli_ui` is somehow missing the agent still runs with
# plain (uncoloured) output.
try:
    from agent.cli_ui import c as _colour, colours_enabled as _colours_enabled, Spinner
except ImportError:  # pragma: no cover - cli_ui should always be present
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


# Threshold (in characters) above which a tool result is moved into
# the per-agent ``_result_cache`` and replaced on the wire by a small
# reference stub. 2 KB is roughly the size of a 500-line script's worth
# of "import" block, which is well below the threshold where
# compaction starts paying for itself in token-savings, and well above
# the threshold where the stub text is bigger than the original.
# This is the *default*; each agentTwo can override it via the
# ``cache_threshold_chars`` constructor argument.
CACHE_THRESHOLD_CHARS = 2_000

# How many of the most recent tool-result contents to keep in a
# short-window buffer for Pass 2 (dedup of consecutive duplicates).
# Anything older than this is assumed not to be a duplicate of the
# current call and is preserved verbatim.
DEDUP_WINDOW = 5

# Length of the short hex prefix used to identify a cached result.
# 8 hex chars = 32 bits of entropy, which is plenty to be unique
# within a single conversation (typical sessions see < 100 cache
# entries).
CACHE_REF_HEX_LEN = 8

# Name of the special tool the model can call to fetch a cached
# result back. Registered globally via ``@tool`` in
# :mod:`agent.tools_misc`, but its implementation lives here so that
# the lookup is per-agent (the cache is a per-agent attribute).
RECALL_CACHED_RESULT = "recall_cached_result"

# Default file name used by ``/save`` and ``/restore`` when the user
# does not supply an explicit path. Resolved relative to the current
# working directory.
DEFAULT_SESSION_FILE = ".agent_session_state.json"


class agentTwo:
    """A simple conversational agent that can call registered tools via Ollama."""

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
        self.model          = model
        self.ollama_url     = ollama_url
        self.max_iterations = max_iterations
        self.verbose        = verbose
        self.temperature    = temperature  # None -> let Ollama use its default
        self.show_thinking  = show_thinking

        # Per-instance override for the Pass-1 cache threshold.
        # Stored on the instance so the constructor argument actually
        # takes effect in ``_build_wire_messages``, and so that
        # ``compact_history`` uses the same threshold the user
        # configured.
        if cache_threshold_chars < 0:
            raise ValueError(
                f"cache_threshold_chars must be >= 0, got {cache_threshold_chars}"
            )
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

        if tools is None:
            tools = list(_TOOL_REGISTRY.values())
        self.tools:    list[Tool]            = tools
        self.tool_map: dict[str, Tool]       = {t.name: t for t in tools}

        self.messages: list[dict[str, Any]]  = []

        # Per-agent store of large tool results that have been
        # compacted on the wire. The full content lives here; the
        # conversation history and the on-disk log only ever see a
        # short reference stub. The model can pull a result back via
        # the ``recall_cached_result`` tool (see ``_execute_tool_call``).
        self._result_cache: dict[str, str] = {}

        # ------------------------------------------------------------------
        # Session statistics & status-bar state
        # ------------------------------------------------------------------
        # Cumulative token usage accumulated across every LLM call during
        # this session.  Ollama reports per-response token counts under the
        # ``prompt_eval_count`` (input) and ``eval_count`` (output) keys;
        # we sum them up here so the status bar can show session totals.
        self.session_prompt_tokens     = 0
        self.session_completion_tokens = 0
        self.session_total_tokens      = 0
        self.llm_call_count            = 0
        # Human-readable label of the current agent state, shown in the
        # status bar.  Updated by chat() as work progresses:
        #   "idle"            - waiting for user input
        #   "thinking"        - a request to the LLM is in flight
        #   "calling tool: X" - executing tool X
        #   "done"            - produced a final answer, back to idle
        #   "error"           - the last call failed
        self.state: str = "idle"
        # Wall-clock start of the session, used for the elapsed-time field.
        self._session_start_time = time.time()

        logger.info(
            "Agent initialised | model=%s | url=%s | max_iterations=%d | "
            "temperature=%s | show_thinking=%s | cache_threshold_chars=%d | "
            "tools=%s",
            self.model, self.ollama_url, self.max_iterations,
            self.temperature, self.show_thinking,
            self.cache_threshold_chars,
            [t.name for t in self.tools],
        )
        logger.debug("System prompt: %s", self.system_prompt)

    # ------------------------- Status bar --------------------------------

    @staticmethod
    def _format_tokens(n: int) -> str:
        """Compact human-readable formatting of a token count."""
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        if n >= 1_000:
            return f"{n / 1000:.1f}k"
        return str(n)

    def render_status_bar(self) -> str:
        """Build the single-line status bar string.

        Displays: the LLM model in use, the current agent state, the
        current temperature setting, whether chain-of-thought display is
        on or off, the cumulative token usage for the session (prompt /
        completion), the number of LLM calls, and the number of
        conversation turns.

        This is intended to be used as a ``prompt_toolkit`` bottom
        toolbar callback (see ColouredPrompt(bottom_toolbar=...)), so it
        is cheap and side-effect free.
        """
        turns = sum(1 for msg in self.messages if msg.get("role") == "user")
        temp_str = "default" if self.temperature is None else f"{self.temperature}"
        think_str = "on" if self.show_thinking else "off"

        return (
            f" LLM: {self.model} "
            f"| state: {self.state} "
            f"| temp: {temp_str} "
            f"| think: {think_str} "
            f"| tokens: {self._format_tokens(self.session_prompt_tokens)} in / "
            f"{self._format_tokens(self.session_completion_tokens)} out "
            f"| calls: {self.llm_call_count} "
            f"| turns: {turns} "
        )

    def print_status_bar(self) -> None:
        """Print the status bar as a single dim line to the terminal.

        This is used during the agent's response phase, when the live
        prompt_toolkit bottom toolbar is no longer visible (because the
        PromptSession UI only exists while waiting for input).  Printing
        the bar here keeps the information on screen while the agent is
        thinking / calling tools.
        """
        bar = self.render_status_bar()
        if _colours_enabled():
            print(_colour(bar, "dim", "gray"))
        else:
            print(bar)

    # ------------------------- Compaction ---------------------------------
    #
    # The conversation stored in ``self.messages`` is the full, unedited
    # history: every ``thinking`` block, every byte of every tool result,
    # every assistant turn. That's what we log, and what ``reset()``
    # clears.
    #
    # The list we *send* to Ollama (``_build_wire_messages``) is a
    # compacted copy of that history. The passes implemented here are
    # intentionally simple, lossless, and easy to reason about:
    #
    #   Pass 1 - for tool results above ``self.cache_threshold_chars``,
    #            move the full content into ``self._result_cache``
    #            and replace the wire copy with a small stub:
    #            ``[cached:<call_key> -> N chars; ref=<hex>]``.
    #            The model can still get the bytes back by calling the
    #            ``recall_cached_result`` tool with the ``ref``. Nothing
    #            is deleted; the cache survives for the lifetime of
    #            the agent (or until ``reset()``).
    #
    #            EXCEPTION: a tool message whose originating call is
    #            ``recall_cached_result`` is *never* re-stubbed by
    #            Pass 1, regardless of its size. The model
    #            intentionally asked for those bytes back; shipping a
    #            stub in response would just hide them behind another
    #            reference and force the model into a recall loop.
    #            Pass 2 (wire-form dedup) still collapses an exact
    #            repeat of the same recall, so the cost is bounded.
    #
    #   Pass 2 - if a new tool result is byte-identical to one of the
    #            last ``DEDUP_WINDOW`` tool results we sent, replace it
    #            with the marker
    #            ``[same as previous result; not repeated]``.
    #            The first occurrence is always preserved, so the model
    #            never loses the information - only the waste.
    #
    #   Pass 3 - strip the chain-of-thought (``thinking``) from
    #            assistant messages before re-feeding it to the model.
    #            The model has already produced and acted on its own
    #            reasoning once; re-shipping it just biases the next
    #            turn toward re-asserting the same plan. The original
    #            is preserved in ``self.messages`` (and therefore in
    #            ``agent.log``) until ``compact_history`` runs.
    #
    #   Pass 4 - normalise the wire format: drop an empty ``content``
    #            from assistant messages that carry ``tool_calls``
    #            (some Ollama builds prefer the key to be absent
    #            rather than empty), and replace an empty ``tool``
    #            result with the placeholder ``"(empty result)"`` so
    #            the message is never sent as an empty string.
    #
    #   Important interaction between Pass 1 and Pass 2:
    #
    #   The dedup window must store the *wire form* of each tool
    #   result, not the raw ``content`` field of ``self.messages``.
    #   If we stored the raw content, then a ``recall_cached_result``
    #   call that returns the same bytes as the original read would
    #   be flagged as a duplicate and collapsed on the very next
    #   iteration - the model would never actually see the bytes it
    #   just recalled. By storing the wire form (which, for a normal
    #   cached read, is the stub carrying the cache ``ref``) instead,
    #   the recalled result is recognised as a different message on
    #   the wire: it carries the raw content, which is almost
    #   certainly different from any stub in the window. The recall
    #   only gets collapsed if the model issues the *exact* same
    #   recall again - which is the rare case we do want to dedup.
    #   See the comment in ``_build_wire_messages`` for the precise
    #   spot where this matters.

    @staticmethod
    def _make_cache_ref(result: str) -> str:
        """Return a short, stable hex identifier for a tool result.

        ``id()`` would be faster but is not stable across processes
        or after a ``reset()``; a content hash is. We don't need
        cryptographic strength here - we only need enough bits to
        avoid collisions within one conversation.
        """
        import hashlib
        return hashlib.sha1(result.encode("utf-8", errors="replace")).hexdigest()[:CACHE_REF_HEX_LEN]

    @staticmethod
    def _is_recall_tool_message(m: dict[str, Any]) -> bool:
        """Return True if the given tool message originated from a
        ``recall_cached_result`` call.

        Detected via the cosmetic ``_call_key`` we set when the tool
        call was dispatched (see ``_call_key_for``); that field is
        stripped from the wire but is still on the message in
        ``self.messages`` while we are deciding what to do with it.
        The ``recall_cached_result`` tool always builds a call key of
        the form ``recall_cached_result@<ref>`` (it has no
        "subject" string to pick, only the ``ref`` argument).
        """
        key = m.get("_call_key") or ""
        return key == RECALL_CACHED_RESULT or key.startswith(RECALL_CACHED_RESULT + "@")

    def _build_wire_messages(self) -> list[dict[str, Any]]:
        """Return the compacted copy of ``self.messages`` that goes on the wire.

        The returned list is a fresh list of shallow dict copies: nothing
        in ``self.messages`` is mutated, so the on-disk log and any
        in-memory introspection keep seeing the full, uncompacted
        history (until ``compact_history`` is called explicitly).
        """
        out: list[dict[str, Any]] = []
        # Pass 2 short window. Holds the *wire* form of each recent
        # tool result (i.e. the stub when Pass 1 fires, or the raw
        # content otherwise) - never the uncompacted ``self.messages``
        # content for a large result. Storing the stub means a
        # ``recall_cached_result`` for the same bytes is recognised as
        # a *different* wire message and is not wrongly deduped.
        recent_tool_results: list[str] = []
        thinking_stripped = 0                # stats for the DEBUG line
        results_cached   = 0
        results_deduped  = 0
        recalls_preserved = 0                # recalls shipped raw despite size

        for msg in self.messages:
            m = dict(msg)  # shallow copy; never mutate the stored message
            role = m.get("role")

            if role == "assistant":
                # Pass 3: drop chain-of-thought from the wire.
                if m.pop("thinking", None):
                    thinking_stripped += 1
                # Pass 4: if there's no spoken content but there are
                # tool calls, omit ``content`` entirely from the wire.
                if not m.get("content") and m.get("tool_calls"):
                    m.pop("content", None)

            elif role == "tool":
                content = m.get("content", "") or ""

                # Decide the *wire form* of this tool result first:
                # if the raw content is larger than the cache
                # threshold, the wire form is the cache stub; otherwise
                # it is the raw content itself. The dedup window then
                # sees the wire form - never the raw bytes of a
                # compacted result. This is the fix for the
                # ``recall_cached_result``-gets-deduped bug: when the
                # model recalls a cached result, the recalled tool
                # message's wire form is a stub with a *different*
                # ``ref`` (the hash of the recalled bytes), so it
                # is not a duplicate of the original read's stub.
                #
                # EXCEPTION: the result of a ``recall_cached_result``
                # call is *always* shipped verbatim, even when its
                # length exceeds ``cache_threshold_chars``. The model
                # explicitly asked for those bytes; replacing them
                # with another stub would hide the content the model
                # is trying to read and trap it in a recall loop.
                # We still let Pass 2 see this raw content for dedup
                # purposes, so an exact repeat of the same recall
                # still collapses to the "not repeated" marker.
                is_recall = self._is_recall_tool_message(m)
                if is_recall:
                    wire_content = content
                    if len(content) > self.cache_threshold_chars:
                        recalls_preserved += 1
                elif len(content) > self.cache_threshold_chars:
                    ref = self._make_cache_ref(content)
                    self._result_cache[ref] = content
                    call_key = m.get("_call_key") or "tool"
                    wire_content = (
                        f"[cached:{call_key} -> {len(content)} chars; "
                        f"ref={ref}]"
                    )
                else:
                    wire_content = content

                # Pass 2: dedup of consecutive identical wire forms.
                # We compare against the wire form we just computed
                # (which, for a large non-recall result, is the stub;
                # for a small one or a recall, is the content
                # itself). Two consecutive small reads of the same
                # file still collapse; an exact repeat of the same
                # recall also collapses; a recall of a
                # previously-cached result that has not been recalled
                # in the last DEDUP_WINDOW turns does not.
                if wire_content and wire_content in recent_tool_results[-DEDUP_WINDOW:]:
                    m["content"] = "[same as previous result; not repeated]"
                    results_deduped += 1
                else:
                    if wire_content:
                        recent_tool_results.append(wire_content)
                    # Only install the cache stub for non-recall
                    # messages. Recalls always keep their raw content
                    # on the wire (see the EXCEPTION above).
                    if (not is_recall
                            and len(content) > self.cache_threshold_chars):
                        m["content"] = wire_content  # install the stub
                        results_cached += 1

                # Strip the helper field from the wire - it was only
                # useful for naming the cache stub and detecting
                # recalls.
                m.pop("_call_key", None)

                # Pass 4: never send an empty tool result. The model
                # would have nothing to act on, and some Ollama builds
                # reject empty ``tool`` content outright.
                if not m.get("content"):
                    m["content"] = "(empty result)"

            out.append(m)

        if thinking_stripped or results_cached or results_deduped or recalls_preserved:
            logger.debug(
                "Wire-message compaction: stripped 'thinking' from %d, "
                "cached %d large tool result(s), shipped %d recall(s) "
                "verbatim to avoid stub-of-a-stub, deduped %d "
                "consecutive duplicate tool result(s). Cache size: %d "
                "entry/ies. (history in self.messages unchanged.)",
                thinking_stripped, results_cached, recalls_preserved,
                results_deduped, len(self._result_cache),
            )
        return out

    # ------------------------- HTTP ---------------------------------------

    def _call_ollama(self) -> dict:
        """POST the current conversation to Ollama and return the parsed JSON."""
        # The system message is built fresh on every call (it can carry
        # the user's runtime overrides) and is *not* compacted, so it
        # never had a ``thinking`` field to strip anyway. The rest of
        # the history goes through the compaction helper.
        wire_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *self._build_wire_messages(),
        ]
        payload: dict[str, Any] = {
            "model":    self.model,
            "messages": wire_messages,
            "stream":   False,
        }
        # Only include temperature if the user has explicitly set one.
        if self.temperature is not None:
            payload["options"] = {"temperature": float(self.temperature)}
        if self.tools:
            payload["tools"] = [t.to_ollama_schema() for t in self.tools]

        # ---------------------------------------------------------------
        # Log the full JSON message being sent to the LLM.
        # The dedicated `llm_payload_logger` writes to `agent.log` (via
        # the root `agent` file handler configured above) and uses a
        # clearly delimited banner so the payload is easy to find/grep.
        # ---------------------------------------------------------------
        payload_json = safe_json(payload)
        banner = "=" * 72
        llm_payload_logger.info(
            "%s\nLLM REQUEST  ->  %s\n%s\n%s\n%s",
            banner, self.ollama_url, banner, payload_json, banner,
        )
        logger.debug("HTTP request to %s\n%s", self.ollama_url, payload_json)

        t0 = time.time()
        try:
            resp = requests.post(self.ollama_url, json=payload, timeout=500)
            elapsed = time.time() - t0
            try:
                body_for_log = safe_json(resp.json()) if resp.ok else truncate(resp.text, 500)
            except Exception:                       # noqa: BLE001
                body_for_log = truncate(resp.text, 500)
            logger.debug("HTTP response status=%d, elapsed=%.2fs | body\n%s",
                         resp.status_code, elapsed, body_for_log)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout as exc:
            logger.error("Timeout calling Ollama after %.2fs: %s", time.time() - t0, exc)
            self.state = "error"
            raise
        except requests.exceptions.HTTPError as exc:
            body = ""
            try:
                body = exc.response.text if exc.response is not None else ""
            except Exception:                       # noqa: BLE001
                pass
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

        logger.debug("Received response from Ollama in %.2fs | body\n%s",
                     elapsed, safe_json(data))

        # ------------------------------------------------------------------
        # Accumulate token usage reported by Ollama for the status bar.
        # Ollama exposes per-response counts at the top level of the JSON:
        #   prompt_eval_count  - input/prompt tokens
        #   eval_count         - generated/completion tokens
        # Some endpoints (e.g. cloud proxies) may not report them; we
        # guard every access so the agent keeps working regardless.
        # ------------------------------------------------------------------
        try:
            self.llm_call_count += 1
            prompt_toks = data.get("prompt_eval_count")
            if isinstance(prompt_toks, int):
                self.session_prompt_tokens += prompt_toks
            completion_toks = data.get("eval_count")
            if isinstance(completion_toks, int):
                self.session_completion_tokens += completion_toks
            # total_count is not always present; fall back to the sum.
            total_toks = data.get("total_count")
            if isinstance(total_toks, int):
                self.session_total_tokens += total_toks
            else:
                p = prompt_toks if isinstance(prompt_toks, int) else 0
                c_tok = completion_toks if isinstance(completion_toks, int) else 0
                self.session_total_tokens += p + c_tok
        except Exception:  # noqa: BLE001 - never let stats break the chat
            logger.debug("Could not parse token usage from Ollama response", exc_info=True)

        return data

    # ------------------------- Tools --------------------------------------

    def _recall_cached_result(self, raw_args: dict) -> str:
        """Implement the ``recall_cached_result`` tool against this agent's cache.

        Lives on the agent (rather than as a free function) because the
        cache is a per-agent attribute. The tool is registered globally
        by :mod:`agent.tools_misc` so the model can see it in the
        schema; here we just intercept the call and look the ref up in
        ``self._result_cache``.
        """
        ref = (raw_args or {}).get("ref", "")
        if not isinstance(ref, str) or not ref:
            return "Error: 'ref' is required and must be a non-empty string"
        if ref not in self._result_cache:
            available = ", ".join(sorted(self._result_cache)) or "(none)"
            return (f"Error: no cached result with ref={ref!r}. "
                    f"Available refs: {available}")
        return self._result_cache[ref]

    def _execute_tool_call(self, tool_call: dict) -> str:
        """Dispatch a single tool call from the model and return its stringified result."""
        fn       = tool_call.get("function", {}) or {}
        name     = fn.get("name", "")
        raw_args = fn.get("arguments", {}) or {}

        logger.debug("Tool call requested: %s | raw_args=%s", name, safe_json(raw_args))

        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                logger.error("Tool '%s' received malformed arguments JSON: %s | raw=%s",
                             name, exc, truncate(raw_args))
                return f"Error: tool '{name}' received malformed arguments JSON"

        # Update the status bar to show which tool is being executed.
        # This covers both normal registry tools and the special
        # ``recall_cached_result`` tool intercepted below.
        self.state = f"calling tool: {name}"
        self.print_status_bar()

        # The cache-recall tool is special: it doesn't live in the
        # global tool registry the same way; its implementation needs
        # access to this agent's ``_result_cache``. Intercept it before
        # the registry lookup.
        if name == RECALL_CACHED_RESULT:
            return self._recall_cached_result(raw_args)

        tool = self.tool_map.get(name)
        if tool is None:
            logger.error("Unknown tool requested: %s | available=%s",
                         name, list(self.tool_map.keys()))
            return f"Error: unknown tool '{name}'"

        try:
            result = tool.func(**raw_args)
            result_str = str(result)
            logger.debug("Tool '%s' executed | args=%s | result=%s",
                         name, safe_json(raw_args), truncate(result_str, 5000))
            return result_str
        except TypeError as exc:
            logger.error("Bad arguments for tool '%s': %s | args=%s",
                         name, exc, safe_json(raw_args))
            return f"Error: bad arguments for '{name}': {exc}"
        except Exception as exc:                                 # noqa: BLE001
            logger.exception("Unhandled exception while executing tool '%s'", name)
            return f"Error executing '{name}': {exc}"

    # ------------------------- Conversation -------------------------------

    @staticmethod
    def _call_key_for(name: str, raw_args: Any) -> str:
        """Build a short, human-readable identifier for a tool call.

        Used as the ``_call_key`` field on outgoing tool messages, so
        that the cache stub in Pass 1 can show e.g.
        ``[cached:read_text_file@agentNew.py -> 21820 chars; ref=...]``
        instead of an anonymous ``[cached:tool -> ...]``. Purely
        cosmetic - the ref itself is what uniquely identifies the
        cached payload.

        ``recall_cached_result`` is special: it has no "subject"
        string to pick (its only argument is the ref of the cached
        payload), so the call key is the bare tool name. That gives
        ``_is_recall_tool_message`` a stable signal to look for when
        deciding whether to skip Pass 1's stub-replacement for this
        tool message.
        """
        if name == RECALL_CACHED_RESULT:
            return RECALL_CACHED_RESULT
        if not isinstance(raw_args, dict):
            return name
        # Pick the first string-looking argument as the "subject".
        for v in raw_args.values():
            if isinstance(v, str) and v:
                # Keep it short; file paths are usually fine but a
                # giant blob would defeat the point.
                subject = v if len(v) <= 80 else v[:77] + "..."
                return f"{name}@{subject}"
        return name

    def chat(self, user_message: str) -> str:
        """Send a user message and return the agent's final answer
        (running tools as needed).

        A ``KeyboardInterrupt`` (Ctrl-C) raised while the agent is busy -
        either waiting for an LLM response or executing a tool - is
        caught here so that the *current task* is abandoned gracefully
        and control returns to the interactive REPL.  The conversation
        history accumulated so far is preserved, so the user can simply
        type a new request (or a corrective one) and the agent will
        continue from where it left off.  Ctrl-C therefore never exits
        the session; it only interrupts the in-flight call sequence.
        """
        logger.debug("User message:\n%s", user_message)
        self.messages.append({"role": "user", "content": user_message})

        # ------------------------------------------------------------------
        # The entire call sequence (the ``while``/``for`` loop below) is
        # wrapped in a ``try`` so that a Ctrl-C (KeyboardInterrupt, a
        # BaseException that is *not* an Exception) interrupts whatever
        # step is currently in flight - an HTTP request to the LLM or the
        # execution of a tool - and is handled cleanly instead of
        # tearing down the whole program.
        # ------------------------------------------------------------------
        try:
            while True:
                # We loop here so that, if the user agrees to keep going when
                # the iteration limit is hit, we can resume the agent's work
                # for another `max_iterations` rounds. The inner `for` loop
                # only ever consumes at most `max_iterations` iterations per
                # pass.
                for i in range(self.max_iterations):
                    logger.info("--- Agent iteration %d/%d ---", i + 1, self.max_iterations)
                    self.state = "thinking"
                    self.print_status_bar()
                    try:
                        with Spinner("thinking...", elapsed=True):
                            data = self._call_ollama()
                    except Exception as exc:                       # noqa: BLE001
                        logger.exception("Iteration %d failed during Ollama call", i + 1)
                        self.state = "error"
                        raise

                    message = data.get("message", {}) or {}
                    logger.debug("Assistant message object:\n%s", safe_json(message))
                    self.messages.append(message)

                    # Display the model's chain-of-thought if the user asked for it
                    # and the model actually produced any. Ollama exposes this under
                    # the `thinking` key of the assistant message.
                    thinking = message.get("thinking")
                    if self.show_thinking and thinking:
                        if self.verbose:
                            # Coloured chain-of-thought echo.
                            tag = _colour("[think]", "bold", "bright_blue")
                            print(f"  {tag} {truncate(str(thinking), 400)}")

                    tool_calls = message.get("tool_calls") or []
                    if not tool_calls:
                        final = (message.get("content") or "").strip()
                        logger.debug("Final assistant answer:\n%s", final)
                        self.state = "done"
                        return final

                    logger.debug("Model requested %d tool call(s): %s",
                                 len(tool_calls), safe_json(tool_calls))
                    for tc in tool_calls:
                        fn      = tc.get("function", {}) or {}
                        fn_name = fn.get("name", "?")
                        fn_args = fn.get("arguments", {}) or {}
                        if isinstance(fn_args, str):
                            try:
                                fn_args = json.loads(fn_args)
                            except json.JSONDecodeError:
                                fn_args = {}
                        result = self._execute_tool_call(tc)
                        self.messages.append({
                            "role":      "tool",
                            "content":   result,
                            # Cosmetic helper for Pass 1's cache stub AND
                            # for ``_is_recall_tool_message`` to detect
                            # recall responses (so Pass 1 does not turn
                            # them back into stubs). Stripped from the
                            # wire by ``_build_wire_messages``.
                            "_call_key": self._call_key_for(fn_name, fn_args),
                        })
                        if self.verbose:
                            preview = result if len(result) <= 120 else result[:120] + "..."
                            # Coloured tool-call echo.
                            tag = _colour("[tool]", "bold", "bright_magenta")
                            name_col = _colour(fn_name, "magenta")
                            print(f"  {tag} {name_col}(...) -> {preview}")

                # Exhausted `self.max_iterations` rounds without a final answer.
                # Ask the user whether they want the agent to keep going.
                logger.warning("Reached max_iterations=%d without a final answer",
                               self.max_iterations)
                self.state = "idle"
                if not request_continue_approval(self.max_iterations):
                    return ("Sorry - I could not reach a final answer within "
                            "the iteration limit.")
        except KeyboardInterrupt:
            # ----------------------------------------------------------------
            # The user pressed Ctrl-C to interrupt the agent's call
            # sequence (an in-flight LLM request or a running tool).  We
            # stop the current task gracefully, leaving the conversation
            # history intact, and hand control back to the main REPL loop
            # so the user can issue a new request.  The agent stays alive
            # - only this task is cancelled.
            # ----------------------------------------------------------------
            logger.info("Chat interrupted by user (Ctrl-C) at state=%r", self.state)
            # Reset to a clean idle state so the status bar / next prompt
            # do not show a stale "thinking" / "calling tool" label.
            self.state = "idle"
            # Emit a newline after the "^C" the terminal echoes so the
            # interruption message starts on a fresh line, then warn the
            # user with a coloured marker when colours are available.
            print()
            if self.verbose:
                if _colours_enabled():
                    print(_colour("[interrupted by user - task stopped]",
                                  "bold", "bright_yellow"))
                else:
                    print("[interrupted by user - task stopped]")
            return "[interrupted by user - task stopped]"

    def compact_history(self) -> dict[str, int]:
        """Compact ``self.messages`` in place using the same passes as
        the wire-compaction helper.

        After this call, ``self.messages`` holds the *compacted* view
        (cache stubs instead of large results, dedup markers instead
        of consecutive duplicates, no ``thinking`` blocks, normalised
        empty content). The full original bytes are still reachable
        via ``self._result_cache`` and the ``recall_cached_result``
        tool - so this is lossless with respect to what the LLM can
        see, but frees the in-process memory that the uncompacted
        history was using.

        Returns a small report of what changed, so the REPL can print
        a one-line summary.
        """
        before_count    = len(self.messages)
        before_chars    = sum(
            len(json.dumps(m, ensure_ascii=False, default=str))
            for m in self.messages
        )
        before_cache    = len(self._result_cache)

        compacted = self._build_wire_messages()

        # ``_build_wire_messages`` is documented to return a fresh
        # list of shallow dict copies, so assigning it back is safe
        # and doesn't share any references with the old list.
        self.messages = compacted

        after_count = len(self.messages)
        after_chars = sum(
            len(json.dumps(m, ensure_ascii=False, default=str))
            for m in self.messages
        )
        after_cache = len(self._result_cache)

        report = {
            "messages_before":  before_count,
            "messages_after":   after_count,
            "chars_before":     before_chars,
            "chars_after":      after_chars,
            "chars_saved":      before_chars - after_chars,
            "cache_before":     before_cache,
            "cache_after":      after_cache,
        }
        logger.info(
            "Manual history compaction: messages %d -> %d, "
            "chars %d -> %d (saved %d), cache %d -> %d entry/ies.",
            before_count, after_count,
            before_chars, after_chars, before_chars - after_chars,
            before_cache, after_cache,
        )
        return report

    def list_cache(self) -> list[dict[str, Any]]:
        """Return a debug-friendly view of the per-agent result cache.

        Each entry is a dict with ``ref``, ``chars`` and a short
        ``preview`` of the cached content. The model never sees this
        method - it's only meant for ``/listcache`` in the REPL, so
        a human can see what refs the agent has accumulated (e.g.
        when the model gets stuck in a recall loop).
        """
        out: list[dict[str, Any]] = []
        for ref, content in self._result_cache.items():
            preview = content if len(content) <= 120 else content[:117] + "..."
            # Collapse newlines so the preview stays one line in the REPL.
            preview = preview.replace("\n", "\\n").replace("\r", "\\r")
            out.append({"ref": ref, "chars": len(content), "preview": preview})
        # Stable order makes the output diff-friendly.
        out.sort(key=lambda d: d["ref"])
        return out

    def reset(self) -> None:
        """Clear the conversation history and the per-agent result cache."""
        cache_size = len(self._result_cache)
        logger.info(
            "Conversation reset (cleared %d messages, %d cached result(s))",
            len(self.messages), cache_size,
        )
        self.messages = []
        self._result_cache = {}
        self.state = "idle"

    # ------------------------- Session save / restore ---------------------

    def _settings_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of all user-tunable settings.

        These are the attributes that the REPL commands (``/temp``,
        ``/max_iter``, ``/think``, ...) can change at runtime, plus the
        model and system prompt. Everything needed to reconstruct the
        agent's *configuration* (as opposed to its *conversation*) is
        here, so ``/restore`` can put the agent back into the exact
        state it was in when ``/save`` was called.
        """
        return {
            "model":                self.model,
            "ollama_url":           self.ollama_url,
            "max_iterations":       self.max_iterations,
            "verbose":              self.verbose,
            "temperature":          self.temperature,
            "show_thinking":        self.show_thinking,
            "cache_threshold_chars": self.cache_threshold_chars,
            "system_prompt":        self.system_prompt,
        }

    def _apply_settings(self, settings: dict[str, Any]) -> None:
        """Apply a settings snapshot previously produced by ``_settings_dict``.

        Missing keys are silently skipped, so sessions saved by older
        versions of the agent (with fewer settings) can still be
        restored without error.
        """
        if "model" in settings:
            self.model = settings["model"]
        if "ollama_url" in settings:
            self.ollama_url = settings["ollama_url"]
        if "max_iterations" in settings:
            self.max_iterations = settings["max_iterations"]
        if "verbose" in settings:
            self.verbose = settings["verbose"]
        if "temperature" in settings:
            self.temperature = settings["temperature"]
        if "show_thinking" in settings:
            self.show_thinking = settings["show_thinking"]
        if "cache_threshold_chars" in settings:
            self.cache_threshold_chars = settings["cache_threshold_chars"]
        if "system_prompt" in settings:
            self.system_prompt = settings["system_prompt"]

    def save_session(self, path: str | None = None) -> str:
        """Persist the current session to *path* and return the absolute path.

        The saved file contains:
        - the full conversation ``history`` (``self.messages``),
        - the per-agent ``result_cache`` (large tool results that were
          compacted on the wire, so the model can still recall them
          after a restore),
        - all runtime ``settings`` that the user may have changed via
          REPL commands.

        If *path* is ``None`` the default file name
        ``DEFAULT_SESSION_FILE`` is used, resolved relative to the
        current working directory.
        """
        target = os.path.abspath(path or DEFAULT_SESSION_FILE)
        state = {
            "history":      self.messages,
            "result_cache": self._result_cache,
            "settings":     self._settings_dict(),
        }
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2)
        logger.info(
            "Session saved to %s (%d messages, %d cache entries).",
            target, len(self.messages), len(self._result_cache),
        )
        return target

    def restore_session(self, path: str | None = None) -> dict[str, Any]:
        """Restore a session previously saved by :meth:`save_session`.

        Replaces the current conversation history, result cache, and
        all runtime settings with the values from the file. Returns a
        small report dict with the number of messages and cache
        entries loaded, so the REPL can print a one-line summary.

        If *path* is ``None`` the default file name
        ``DEFAULT_SESSION_FILE`` is used, resolved relative to the
        current working directory.

        Raises ``FileNotFoundError`` if the file does not exist, and
        ``json.JSONDecodeError`` if it is not valid JSON. Any other
        I/O or format problem is raised to the caller as well, so the
        REPL can show the user a useful error message.
        """
        source = os.path.abspath(path or DEFAULT_SESSION_FILE)
        with open(source, "r", encoding="utf-8") as fh:
            state = json.load(fh)

        self.messages = state.get("history", [])
        self._result_cache = state.get("result_cache", {})
        self._apply_settings(state.get("settings", {}))

        # Reset the per-session statistics: the restored history belongs
        # to a different (previous) session, so the accumulated token
        # counts and LLM call count from before the restore no longer
        # apply. The status bar starts fresh from the restored state.
        self.session_prompt_tokens     = 0
        self.session_completion_tokens = 0
        self.session_total_tokens      = 0
        self.llm_call_count            = 0
        self.state                     = "idle"
        self._session_start_time       = time.time()

        report = {
            "path":          source,
            "messages":      len(self.messages),
            "cache_entries": len(self._result_cache),
        }
        logger.info(
            "Session restored from %s (%d messages, %d cache entries).",
            source, len(self.messages), len(self._result_cache),
        )
        return report