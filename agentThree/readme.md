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

## Streaming

By default the agent requests **NDJSON streaming** from Ollama and prints the
model's reply token-by-token as it arrives (including the `[think]` chain-of-
thought block, when `/think on`).  The full prompt/response bodies are still
logged to `agentThree/agent.http.log` and `agentThree/agent.llm_payload.log`
after the stream is assembled, so nothing is lost.

If your model or terminal misbehaves with live tokens (e.g. the spinner is
distracting, or you want to log the response in one piece), disable streaming
with:

```
/stream off
```

The `STREAM` config flag in `config.py` controls the default on startup.

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
| `/stream on\|off` | Enable/disable live token streaming from the LLM |
| `/compact` | Compact conversation history in place |
| `/listcache` | List cached tool-result refs |
| `/save [path]` | Save session to disk |
| `/restore [path]` | Restore a saved session |

## Agent2Agent (A2A) Protocol Support

agentThree implements the **Agent2Agent (A2A) Protocol v1.0** — an open
standard by Google/the Linux Foundation for inter-agent communication.
The implementation follows the recommended integration approach:

- **Role C (Client + Server combined):** agentThree can act as both an
  A2A Server (accepting tasks from remote agents) and an A2A Client
  (delegating to remote agents).
- **Binding i (Dual JSON-RPC + REST):** the server exposes both the
  JSON-RPC 2.0 binding (`POST /rpc`) and the HTTP+JSON/REST binding
  (Section 11 endpoints) on a single HTTP listener.

### Architecture (three layers)

| Layer | Module | Description |
| --- | --- | --- |
| 1 — Data Model | `a2a_models.py` | Protocol-agnostic builders for Task, Message, Part, Artifact, AgentCard, streaming events, and error objects. Pure stdlib. |
| 2 — Operations | (within server/client) | Send Message, Send Streaming Message, Get Task, List Tasks, Cancel Task, Subscribe, Get Extended Agent Card. |
| 3 — Protocol Bindings | `a2a_server.py`, `a2a_client.py` | JSON-RPC 2.0 over HTTP + REST endpoints with SSE streaming. |

### A2A Server

```python
from agentThree.agent import agentThree
from agentThree.a2a_server import create_a2a_server

agent = agentThree(model="your-model")
server = create_a2a_server(agent, host="0.0.0.0", port=8080,
                            agent_name="MyAgent",
                            agent_description="Does cool stuff")
server.serve_forever()
```

Endpoints exposed:

| Method | Path | Binding | Operation |
| --- | --- | --- | --- |
| GET | `/.well-known/agent-card.json` | both | Agent Card discovery |
| POST | `/rpc` | JSON-RPC | All JSON-RPC methods |
| POST | `/message:send` | REST | Send Message (blocking) |
| POST | `/message:stream` | REST | Send Streaming Message (SSE) |
| GET | `/tasks/{id}` | REST | Get Task |
| GET | `/tasks` | REST | List Tasks |
| POST | `/tasks/{id}:cancel` | REST | Cancel Task |
| POST | `/tasks/{id}:subscribe` | REST | Subscribe to Task (SSE) |
| GET | `/extendedAgentCard` | REST | Get Extended Agent Card |

### A2A Client

```python
from agentThree.a2a_client import A2AClient

client = A2AClient("http://localhost:8080")

# Blocking
answer = client.ask("What is the weather today?")

# Streaming (prints live)
answer = client.ask("Write a report on climate change", stream=True)

# Low-level
result = client.send_message("Hello")
task = client.get_task("task-uuid")
tasks = client.list_tasks(status="TASK_STATE_WORKING")
client.cancel_task("task-uuid")

# JSON-RPC variants (for JSON-RPC-only servers)
result = client.send_message_rpc("Hello")
```

### What's implemented

- ✅ Agent Card with well-known URI discovery
- ✅ SendMessage (blocking) — REST + JSON-RPC
- ✅ SendStreamingMessage — REST SSE
- ✅ GetTask / ListTasks / CancelTask — REST + JSON-RPC
- ✅ SubscribeToTask — REST SSE
- ✅ GetExtendedAgentCard
- ✅ A2A version negotiation (`A2A-Version` header)
- ✅ A2A-specific error codes (JSON-RPC -32001..-32009, HTTP status mapping)
- ✅ Multi-turn context via `contextId` / `taskId`
- ✅ In-memory thread-safe task store
- ⏳ Push notifications (stubbed — returns `PushNotificationNotSupportedError`)
- ⏳ gRPC binding (not implemented; architecture supports adding it)

## Project layout

```
agentThree/
  __init__.py         - Package init; imports submodules and re-exports public API
  __main__.py         - `python -m agentThree` entry point
  a2a_client.py       - A2A Client (discover + talk to remote agents)
  a2a_models.py       - A2A protocol data-model builders (Layer 1)
  a2a_server.py       - A2A Server (dual JSON-RPC + REST binding)
  agent.py            - The agentThree class: core loop, Ollama HTTP, streaming, compaction, cache
  approval.py         - "Keep going?" prompt for iteration limit
  cli_ui.py           - ANSI colours, prompt_toolkit prompt, tab completion, Spinner
  config.py           - Configuration constants (model, URL, stream flag, etc.)
  diff_tool.py        - kdiff3 configuration and diff-approval workflow
  logging_setup.py    - Centralized logging configuration
  readme.md           - This file
  repl.py             - REPL loop, slash-command handling, main()
  safety.py           - Path safety / workspace validation
  tools_filesystem.py - Filesystem tools (read, create, update, list, compile)
  tools_misc.py       - Utility tools (calculate, word_count, get_current_time, recall_cached_result)
  tools_registry.py   - Tool dataclass and @tool decorator
  tools_web.py        - Web tools (web_search, fetch_url, web_search_fetch)
```