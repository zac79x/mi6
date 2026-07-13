# agentNew.py — Ollama Tool-Calling Agent

A conversational AI agent that connects to a local [Ollama](https://ollama.com/) server and uses a function-calling (tool-use) model to answer user requests. The agent can read/write files, run calculations, list directories, compile Python scripts, and more — all through registered tools that the LLM invokes autonomously.

## What It Does

- **Interactive REPL** — a human-friendly prompt with tab completion, coloured output, and a live status bar showing the active model, token usage, and conversation state.
- **Tool-use loop** — the LLM decides when to call a tool; the agent executes it, feeds the result back, and repeats until a final answer is produced.
- **Safe file editing** — `create_file` writes new `.py` files without approval; `update_file` shows a visual diff (via kdiff3 or a text fallback) and asks for confirmation before applying changes.
- **Session persistence** — all conversation history, token counts, and settings (temperature, max iterations, thinking toggle) are kept for the duration of the session.
- **Graceful interruption** — Ctrl‑C stops the current task without ending the session, so you can correct or re-prompt the agent.

## Requirements

- **Python 3.10+**
- **Ollama** running locally on `http://localhost:11434` with a tool-capable model pulled (e.g. `minimax-m3:cloud`, `qwen3.5:2b`, `gemma4:e4b`).
- Python packages:
  - `requests` (required)
  - `prompt_toolkit` (optional — provides tab completion and the bottom toolbar; the agent works without it)
- A diff/merge tool (optional) — e.g. [KDiff3](https://kdiff3.sourceforge.net/). If none is configured, `update_file` falls back to a unified text diff printed to the terminal.

## How to Run

```bash
# 1. Make sure Ollama is running
ollama serve

# 2. Pull a tool-capable model (if not already present)
ollama pull minimax-m3:cloud

# 3. Run the agent
python agentNew.py
```

On startup the agent will:
1. Print a banner and help text.
2. Ask for the path to a diff tool (or press Enter to use the text-diff fallback).
3. Verify that Ollama is reachable.
4. Present an interactive prompt (`You> `).

### Interactive Commands

| Command | Description |
|---|---|
| `quit` / `exit` | End the session |
| `reset` | Clear the conversation history |
| `Ctrl-C` | Interrupt the current task (agent stays alive) |
| `/?` / `/help` | Show help |
| `/temp <n>` | Set sampling temperature (blank = default) |
| `/max_iter <n>` | Set max tool-calling iterations |
| `/think on\|off` | Enable/disable chain-of-thought display |
| `/kdiff [<path>]` | Set the kdiff3 binary path (persisted to `.env`) |

### Configuration

- **Model** — change `DEFAULT_MODEL` at the top of `agentNew.py`.
- **Diff tool** — set `KDIFF3_PATH` in a `.env` file, or use the `/kdiff` command at runtime.
- **Logging** — detailed logs are written to `agent.log` in the current directory.
