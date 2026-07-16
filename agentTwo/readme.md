# agentTwo

A small Python agent framework that uses an **Ollama** chat API with tool
calling. It runs an interactive REPL, shows a live status bar (model, state,
token usage, call counts), and can compact long conversations by caching
large tool results and deduplicating consecutive duplicates.

The agent class is **`agentTwo`**, defined in `agentTwo/agent.py` and
re-exported from the top-level package.

## Requirements

- Python 3.10+ (the code uses `from __future__ import annotations` and PEP 604
  `X | Y` type syntax).
- [`requests`](https://pypi.org/project/requests/) for HTTP calls to Ollama.
- [`prompt_toolkit`](https://pypi.org/project/prompt-toolkit/) (optional) for
  the coloured prompt with tab completion and a live bottom status bar. If it
  is not installed the agent falls back to a plain but still colour-aware
  `input()` prompt.
- A running **Ollama** server with at least one tool-capable model pulled.

## Setup

### 1. Install Python dependencies

```bash
pip install requests prompt_toolkit
```

`prompt_toolkit` is optional but recommended for tab completion and the
status bar.

### 2. Start an Ollama server

The agent talks to a local Ollama instance over HTTP. Start the server in a
separate terminal:

```bash
ollama serve
```

By default the agent expects Ollama at `http://localhost:11434/api/chat`
(this is `OLLAMA_URL` in `agentTwo/config.py`). The REPL also probes
`http://localhost:11434/api/tags` at startup to confirm the server is
reachable, and exits with a clear message if it is not.

### 3. Choose / pull a model

The default model is configured by `DEFAULT_MODEL` in `agentTwo/config.py`. Any
tool-capable Ollama model works. Pull one if you have not already, e.g.:

```bash
ollama pull glm-5.2:cloud
```

To use a different model, either edit `DEFAULT_MODEL` in `agentTwo/config.py`
before starting, or change it at runtime with the `/model`-equivalent by
editing the source (the REPL does not currently expose a live `/model`
command — model and server URL are read from `config.py` at startup).

### 4. (Optional) Configure a diff tool for `update_file`

`update_file` shows a diff before applying changes. It looks for a
kdiff3-compatible binary. You can set its path via the `/kdiff` command inside
the REPL (which writes `KDIFF3_PATH=...` to a local `.env` file), or let the
agent prompt you for it at startup. If no diff tool is configured, a
text-based diff fallback is used.

## Running the agent

From the repository root:

```bash
python -m agentTwo
```

Or run the module directly:

```bash
python -m agentTwo.agent
```

The agent will prompt you for input. Type `/help` (or `?`) to see the
available commands.

## REPL commands

| Command | Description |
| --- | --- |
| `/help` or `/?` | Show the list of commands. |
| `quit` / `exit` / `/quit` / `/exit` | Leave the REPL. |
| `reset` / `/reset` | Clear the current conversation history (settings are kept). |
| **Ctrl-C** | Interrupt the agent while it is working (thinking or calling a tool) and return to the prompt. The session is **not** ended — only the current task is stopped, so you can immediately type a new request. |
| `/temp [value]` | Set the sampling temperature, e.g. `/temp 0.7`. Blank resets to the model default. Must be between 0.0 and 2.0. |
| `/max_iter <n>` | Set the maximum number of tool-calling iterations per turn, e.g. `/max_iter 10`. |
| `/think on\|off` | Enable or disable display of the model's chain-of-thought (`thinking`) blocks. |
| `/kdiff [path]` | Set the kdiff3 binary path (writes `KDIFF3_PATH=...` to `.env`). With no argument, prompts interactively. |
| `/compact` | Compact the in-memory conversation history in place. |
| `/listcache` | List the cached tool-result refs and their sizes (does not print the content). |
| `/save [path]` | Save the current session to disk. Defaults to `.agent_session_state.json`. |
| `/restore [path]` | Restore a previously saved session. Defaults to `.agent_session_state.json`. |

The prompt supports **TAB completion** for the slash commands and their
arguments (e.g. `/think on|off`, `/kdiff <file>`, `/save <file>`). Output is
colourised when the terminal supports it; set `NO_COLOR=1` to disable colours.

Anything that does not start with `/` (and is not one of the bare words
above) is sent to the model as a user message.

### Status bar

While you are at the prompt a single-line **status bar** is shown (as a
`prompt_toolkit` bottom toolbar, or printed by the agent during a response).
It displays: the LLM model in use, the current agent state (`idle`,
`thinking`, `calling tool: X`, `done`, `error`), the temperature setting,
whether chain-of-thought display is on/off, the cumulative token usage
(prompt / completion) for the session, the number of LLM calls, and the
number of conversation turns.

### Saving and restoring sessions

Use `/save` to persist the current state of your session:

```
> /save
Session saved to /path/to/.agent_session_state.json
```

You can also provide a custom path:

```
> /save my_session.json
Session saved to /path/to/my_session.json
```

Later, or in a new REPL instance, restore the session with `/restore`:

```
> /restore
Session restored from .../.agent_session_state.json (12 messages, 2 cache entries).
```

`/restore` also accepts a custom path:

```
> /restore my_session.json
```

The saved session includes:

- the full conversation `history`,
- the per-agent `result_cache` (large tool results that were compacted on
  the wire, so the model can still recall them after a restore),
- all runtime settings (`model`, `ollama_url`, `max_iterations`,
  `system_prompt`, `verbose`, `temperature`, `show_thinking`,
  `cache_threshold_chars`).

This makes it possible to continue a long-running task across separate REPL
sessions without losing context or settings that you changed with commands.

## Conversation compaction

The agent keeps a full message history in memory. Before each call to the LLM,
`agentTwo._build_wire_messages` produces a compacted copy of that history to
send on the wire. The compaction is lossless with respect to what the model can
see:

- **Pass 1** - tool results above `cache_threshold_chars` are moved into the
  per-agent `_result_cache` and replaced on the wire by a short reference stub
  of the form `[cached:<call_key> -> N chars; ref=<ref>]`. The model can fetch
  the full bytes back with the `recall_cached_result` tool. (The result of a
  `recall_cached_result` call is *never* re-stubbed, so the model always sees
  the bytes it just asked for.)
- **Pass 2** - a tool result whose wire form is byte-identical to one of the
  last `DEDUP_WINDOW` results is replaced by
  `[same as previous result; not repeated]`. The first occurrence is always
  preserved.
- **Pass 3** - chain-of-thought (`thinking`) blocks are stripped from assistant
  messages before re-feeding them to the model. The originals are preserved in
  `self.messages` (and in `agent.log`) until `/compact` is run.
- **Pass 4** - empty `content` is normalised (dropped from assistant messages
  that carry `tool_calls`, and replaced with `(empty result)` for empty tool
  messages).

`/compact` runs the same passes on the stored history itself, freeing
in-process memory. The full bytes stay reachable via the result cache and the
`recall_cached_result` tool, so nothing the model can see is lost.

## Extending the agent with tools

New tools are registered with the global `@tool` decorator from
`agentTwo.tools_registry`. The built-in tools live in two modules that are
imported (and therefore auto-registered) by the package `__init__.py`:

- `agentTwo/tools_misc.py` - `calculate`, `get_current_time`, `word_count`,
  `recall_cached_result` (and a fallback `c`, `colours_enabled`, `Spinner`
  shim is provided by `cli_ui`).
- `agentTwo/tools_filesystem.py` - filesystem tools (`read_text_file`,
  `read_file_lines`, `list_directory`, `path_exists`, `create_file`,
  `update_file`, `compile_python_file`).

To add your own tool, define a function decorated with `@tool` in any module
that is imported at startup (or import it from `agentTwo/__init__.py`). See
`tools_registry.py` for the exact decorator usage and JSON schema helpers.

## Project layout

```
agentTwo/
  __init__.py         - Package init; imports submodules (which registers the
                        built-in tools), and re-exports agentTwo, main, tool,
                        DEFAULT_MODEL, OLLAMA_URL, LOG_FILE, WORKSPACE_ROOT.
  __main__.py         - `python -m agentTwo` entry point.
  agent.py            - The agentTwo class: core loop, Ollama HTTP calls,
                        compaction, result cache, session save/restore,
                        status bar.
  approval.py         - "Keep going?" prompt for the iteration limit.
  cli_ui.py           - ANSI colours, prompt_toolkit prompt, tab completion,
                        status-bar toolbar, Spinner.
  config.py           - Module-level configuration constants (OLLAMA_URL,
                        DEFAULT_MODEL, log file paths, WORKSPACE_ROOT,
                        ALLOWED_EXTENSION, diff-tool path).
  diff_tool.py        - kdiff3 configuration and diff-approval workflow.
  logging_setup.py    - Centralized logging configuration (agent.log,
                        agent.http.log, agent.llm_payload.log).
  readme.md           - This file.
  repl.py             - REPL loop, slash-command handling, help text, main().
  safety.py           - Path safety / workspace validation checks.
  tools_filesystem.py - Filesystem tools (read, create, update, list, ...).
  tools_misc.py       - Small utility tools (calculate, word_count,
                        get_current_time, recall_cached_result).
  tools_registry.py   - Tool registry, Tool dataclass and @tool decorator.
  agent.log           - Main runtime log (lifecycle events, tool-call
                        summaries, compaction stats).
  agent.http.log      - Full Ollama HTTP request/response bodies (DEBUG).
  agent.llm_payload.log - Full JSON payloads sent to the LLM (INFO), one
                        banner block per request.
```