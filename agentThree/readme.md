# agentThree

A small Python agent framework that uses an **Ollama** chat API with tool calling.

## Requirements

- Python 3.10+
- `requests`
- `prompt_toolkit` (optional, for tab completion and status bar)
- A running **Ollama** server with a tool-capable model

## Setup

```bash
pip install requests prompt_toolkit
ollama serve
```

## Running

```bash
python -m agentThree
```

## REPL commands

| Command | Description |
| --- | --- |
| `/help` or `/?` | Show help |
| `/quit` or `/exit` | Leave the REPL |
| `/reset` | Clear conversation history |
| **Ctrl-C** | Interrupt the agent (session continues) |
| `/temp [value]` | Set sampling temperature |
| `/max_iter <n>` | Set max tool-calling iterations |
| `/think on\|off` | Enable/disable chain-of-thought display |
| `/compact` | Compact conversation history in place |
| `/listcache` | List cached tool-result refs |
| `/save [path]` | Save session to disk |
| `/restore [path]` | Restore a saved session |

## Project layout

```
agentThree/
  __init__.py         - Package init; imports submodules and re-exports public API
  __main__.py         - `python -m agentThree` entry point
  agent.py            - The agentThree class: core loop, Ollama HTTP, compaction, cache
  approval.py         - "Keep going?" prompt for iteration limit
  cli_ui.py           - ANSI colours, prompt_toolkit prompt, tab completion, Spinner
  config.py           - Configuration constants
  diff_tool.py        - kdiff3 configuration and diff-approval workflow
  logging_setup.py    - Centralized logging configuration
  readme.md           - This file
  repl.py             - REPL loop, slash-command handling, main()
  safety.py           - Path safety / workspace validation
  tools_filesystem.py - Filesystem tools (read, create, update, list, compile)
  tools_misc.py       - Utility tools (calculate, word_count, get_current_time, recall_cached_result)
  tools_registry.py   - Tool dataclass and @tool decorator
```
