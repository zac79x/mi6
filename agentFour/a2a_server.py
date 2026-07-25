"""A2A Server: exposes agentFour as an A2A-compliant HTTP+JSON endpoint.

Implements **both** the JSON-RPC 2.0 binding (``POST /rpc``) and the
HTTP+JSON/REST binding (Section 11 of the A2A v1.0 specification) on a
single HTTP server.  This is binding choice **i** (dual JSON-RPC + REST).

The server wraps an ``agentFour`` instance.  Each ``SendMessage`` creates
a task; the agent runs synchronously (blocking mode by default) and the
final answer is returned as an artifact on the completed task.

Benchmark metrics (token consumption, LLM calls, turns, tool calls) are
stored in the task's ``metadata`` field so external orchestrators can
collect them via the A2A protocol without importing agentFour.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse, parse_qs

from agentFour.a2a_models import (
    A2A_PROTOCOL_VERSION,
    ERROR_HTTP_STATUS,
    TERMINAL_STATES,
    ROLE_AGENT,
    a2a_error,
    agent_card,
    artifact,
    artifact_text,
    extract_text,
    message as make_message,
    now_iso,
    task as make_task,
    text_part,
)
from agentFour.logging_setup import logger

# ---------------------------------------------------------------------------
# JSON-RPC error codes  (Section 5.4 / 9.5)
# ---------------------------------------------------------------------------

JSONRPC_ERROR_CODES: dict[str, int] = {
    "TaskNotFoundError": -32001,
    "TaskNotCancelableError": -32002,
    "PushNotificationNotSupportedError": -32003,
    "UnsupportedOperationError": -32004,
    "ContentTypeNotSupportedError": -32005,
    "InvalidAgentResponseError": -32006,
    "ExtendedAgentCardNotConfiguredError": -32007,
    "ExtensionSupportRequiredError": -32008,
    "VersionNotSupportedError": -32009,
}

# Task-state constants (mirror a2a_models for brevity)
_TS_SUBMITTED = "TASK_STATE_SUBMITTED"
_TS_WORKING = "TASK_STATE_WORKING"
_TS_COMPLETED = "TASK_STATE_COMPLETED"
_TS_FAILED = "TASK_STATE_FAILED"
_TS_CANCELED = "TASK_STATE_CANCELED"


# ---------------------------------------------------------------------------
# Task store (in-memory, thread-safe)
# ---------------------------------------------------------------------------


def _collect_agent_metrics(agent) -> dict[str, Any]:
    """Extract benchmark metrics from an agentFour instance after chat()."""
    turns = sum(1 for m in agent.messages if m.get("role") == "user")
    tool_calls = sum(1 for m in agent.messages if m.get("role") == "tool")
    return {
        "promptTokens": agent.session_prompt_tokens,
        "completionTokens": agent.session_completion_tokens,
        "totalTokens": agent.session_total_tokens,
        "llmCalls": agent.llm_call_count,
        "turns": turns,
        "toolCalls": tool_calls,
    }


def _messages_to_history(agent) -> list[dict[str, Any]]:
    """Convert the agent's Ollama-format messages to A2A message history."""
    history: list[dict[str, Any]] = []
    for m in agent.messages:
        role = m.get("role", "")
        if role == "user":
            history.append(make_message(role="ROLE_USER", parts=[text_part(m.get("content", ""))]))
        elif role == "assistant":
            content = m.get("content", "")
            if content:
                history.append(make_message(role="ROLE_AGENT", parts=[text_part(content)]))
        elif role == "tool":
            # Tool results are part of the agent's internal dialogue;
            # include them as agent messages with the tool result text.
            history.append(make_message(
                role="ROLE_AGENT",
                parts=[text_part(f"[tool result] {m.get('content', '')}")],
            ))
    return history


class _TaskStore:
    """Thread-safe in-memory store for A2A tasks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._contexts: dict[str, list[str]] = {}  # contextId -> [taskIds]

    def create(
        self,
        agent,
        message_parts: list[dict[str, Any]],
        context_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a task, run the agent synchronously, and store the result."""
        task_id = str(uuid.uuid4())
        ctx_id = context_id or str(uuid.uuid4())
        now = now_iso()

        task = make_task(
            id=task_id,
            context_id=ctx_id,
            state=_TS_SUBMITTED,
            timestamp=now,
            history=[make_message(role="ROLE_USER", parts=message_parts)],
        )

        with self._lock:
            self._tasks[task_id] = task
            self._contexts.setdefault(ctx_id, []).append(task_id)

        # Extract text from parts for the agent
        user_text = extract_text(message_parts)

        # Run the agent synchronously (blocking mode)
        try:
            task["status"]["state"] = _TS_WORKING
            task["status"]["timestamp"] = now_iso()

            answer = agent.chat(user_text)

            # Collect benchmark metrics from the agent instance
            task["metadata"] = _collect_agent_metrics(agent)

            # Update history with the full conversation
            task["history"] = _messages_to_history(agent)

            art = artifact(parts=[text_part(answer)], name="response")
            task["artifacts"] = [art]
            task["status"]["state"] = _TS_COMPLETED
            task["status"]["timestamp"] = now_iso()
            task["status"]["message"] = make_message(
                role=ROLE_AGENT,
                parts=[text_part(answer)],
            )
        except Exception as exc:
            logger.exception("A2A task %s failed: %s", task_id, exc)
            task["status"]["state"] = _TS_FAILED
            task["status"]["timestamp"] = now_iso()
            task["status"]["message"] = make_message(
                role=ROLE_AGENT,
                parts=[text_part(f"Error: {exc}")],
            )
            # Still try to collect partial metrics
            try:
                task["metadata"] = _collect_agent_metrics(agent)
            except Exception:
                task["metadata"] = {}

        with self._lock:
            self._tasks[task_id] = task

        return task

    def get(self, task_id: str, history_length: int | None = None) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task = json.loads(json.dumps(task))  # deep copy
        if history_length is not None and history_length >= 0:
            if history_length == 0:
                task.pop("history", None)
            else:
                task["history"] = task.get("history", [])[-history_length:]
        return task

    def cancel(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            state = task["status"]["state"]
            if state in TERMINAL_STATES:
                return task  # already terminal
            task["status"]["state"] = _TS_CANCELED
            task["status"]["timestamp"] = now_iso()
            return task

    def list(
        self,
        context_id: str | None = None,
        status: str | None = None,
        page_size: int = 50,
        page_token: str = "",
        include_artifacts: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            all_tasks = list(self._tasks.values())

        # Filter
        if context_id:
            all_tasks = [t for t in all_tasks if t.get("contextId") == context_id]
        if status:
            all_tasks = [t for t in all_tasks if t["status"]["state"] == status]

        # Sort by status timestamp descending
        all_tasks.sort(
            key=lambda t: t["status"].get("timestamp", ""),
            reverse=True,
        )

        total = len(all_tasks)
        # Simple offset-based pagination using page_token as integer
        offset = 0
        if page_token:
            try:
                offset = int(page_token)
            except ValueError:
                offset = 0
        page_size = max(1, min(page_size, 100))
        page = all_tasks[offset : offset + page_size]
        next_offset = offset + len(page)
        next_token = str(next_offset) if next_offset < total else ""

        # Deep-copy page so callers can mutate
        page = [json.loads(json.dumps(t)) for t in page]

        # Optionally strip artifacts
        if not include_artifacts:
            for t in page:
                t.pop("artifacts", None)

        return {
            "tasks": page,
            "nextPageToken": next_token,
            "pageSize": page_size,
            "totalSize": total,
        }


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------


def _jsonrpc_error(req_id: Any, code: int, message: str, data: list | None = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _jsonrpc_result(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------


class A2ARequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for A2A JSON-RPC and REST endpoints."""

    # Set by the server factory
    agent = None
    task_store: _TaskStore = None
    agent_card: dict[str, Any] = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _check_version(self) -> str | None:
        """Validate A2A-Version header. Returns the bad version or None."""
        version = self.headers.get("A2A-Version", "")
        if not version:
            version = "0.3"  # default per spec
        if version in ("1.0", "0.3", "1.0.0"):
            return None
        return version

    def _send_json(self, status_code: int, body: dict[str, Any], content_type: str | None = None) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", content_type or "application/a2a+json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("A2A-Version", A2A_PROTOCOL_VERSION)
        self.end_headers()
        self.wfile.write(payload)

    def _send_error_rest(self, http_status: int, reason: str, message: str, metadata: dict | None = None) -> None:
        status, body = a2a_error(reason, message, http_status, extra_details=metadata and [metadata] or None)
        self._send_json(status, body)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    # ------------------------------------------------------------------ #
    # REST endpoints  (Section 11)
    # ------------------------------------------------------------------ #

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        # Agent Card (well-known)
        if path == "/.well-known/agent-card.json":
            self._send_json(200, self.agent_card)
            return

        # Get extended agent card
        if path == "/extendedAgentCard":
            # No auth layer yet — return the same card
            self._send_json(200, self.agent_card)
            return

        # Get task: GET /tasks/{id}
        if path.startswith("/tasks/") and ":" not in path:
            task_id = path.split("/")[2]
            qs = parse_qs(parsed.query)
            hl = qs.get("historyLength", [None])[0]
            history_length = int(hl) if hl is not None else None
            task = self.task_store.get(task_id, history_length)
            if task is None:
                self._send_error_rest(404, "TASK_NOT_FOUND", f"Task {task_id} not found")
                return
            self._send_json(200, task)
            return

        # List tasks: GET /tasks
        if path == "/tasks":
            qs = parse_qs(parsed.query)
            result = self.task_store.list(
                context_id=qs.get("contextId", [None])[0],
                status=qs.get("status", [None])[0],
                page_size=int(qs.get("pageSize", ["50"])[0]),
                page_token=qs.get("pageToken", [""])[0],
                include_artifacts=qs.get("includeArtifacts", ["false"])[0].lower() == "true",
            )
            self._send_json(200, result)
            return

        self._send_error_rest(404, "TASK_NOT_FOUND", f"Unknown endpoint: {path}")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        # Version check
        bad_version = self._check_version()
        if bad_version is not None:
            self._send_error_rest(
                400, "VERSION_NOT_SUPPORTED",
                f"A2A version {bad_version} not supported. Supported: 1.0",
            )
            return

        # REST: Send message: POST /message:send
        if path == "/message:send":
            self._handle_send_message_rest()
            return

        # REST: Send streaming message: POST /message:stream
        if path == "/message:stream":
            self._handle_send_message_stream_rest()
            return

        # REST: Cancel task: POST /tasks/{id}:cancel
        if path.startswith("/tasks/") and path.endswith(":cancel"):
            task_id = path.split("/")[2].removesuffix(":cancel")
            task = self.task_store.cancel(task_id)
            if task is None:
                self._send_error_rest(404, "TASK_NOT_FOUND", f"Task {task_id} not found")
                return
            self._send_json(200, task)
            return

        # REST: Subscribe to task: POST /tasks/{id}:subscribe
        if path.startswith("/tasks/") and path.endswith(":subscribe"):
            task_id = path.split("/")[2].removesuffix(":subscribe")
            task = self.task_store.get(task_id)
            if task is None:
                self._send_error_rest(404, "TASK_NOT_FOUND", f"Task {task_id} not found")
                return
            # SSE: send current state then close
            self._send_sse([{"task": task}])
            return

        # JSON-RPC endpoint: POST /rpc (or root)
        if path in ("/rpc", "/"):
            self._handle_jsonrpc()
            return

        # Push notification config (stub — not fully supported)
        if "/pushNotificationConfigs" in path:
            self._send_error_rest(400, "PUSH_NOTIFICATION_NOT_SUPPORTED",
                                  "Push notifications are not supported by this agent")
            return

        self._send_error_rest(404, "TASK_NOT_FOUND", f"Unknown endpoint: {path}")

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if "/pushNotificationConfigs" in path:
            self._send_error_rest(400, "PUSH_NOTIFICATION_NOT_SUPPORTED",
                                  "Push notifications are not supported by this agent")
            return
        self._send_error_rest(404, "TASK_NOT_FOUND", f"Unknown endpoint: {path}")

    # ------------------------------------------------------------------ #
    # JSON-RPC handler  (Section 9)
    # ------------------------------------------------------------------ #

    def _handle_jsonrpc(self) -> None:
        try:
            body = self._read_body()
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(200, _jsonrpc_error(None, -32700, "Invalid JSON payload"))
            return

        # Batch?
        if isinstance(body, list):
            results = [self._dispatch_jsonrpc(req) for req in body]
            results = [r for r in results if r is not None]
            self._send_json(200, results if results else {})
            return

        result = self._dispatch_jsonrpc(body)
        if result is not None:
            self._send_json(200, result)
        else:
            self._send_json(200, {})

    def _dispatch_jsonrpc(self, req: dict[str, Any]) -> dict[str, Any] | None:
        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {}) or {}

        if not method:
            return _jsonrpc_error(req_id, -32600, "Invalid Request")

        try:
            if method == "SendMessage":
                return self._rpc_send_message(req_id, params)
            elif method == "GetTask":
                return self._rpc_get_task(req_id, params)
            elif method == "ListTasks":
                return self._rpc_list_tasks(req_id, params)
            elif method == "CancelTask":
                return self._rpc_cancel_task(req_id, params)
            elif method == "GetExtendedAgentCard":
                return _jsonrpc_result(req_id, self.agent_card)
            elif method in (
                "CreateTaskPushNotificationConfig",
                "GetTaskPushNotificationConfig",
                "ListTaskPushNotificationConfigs",
                "DeleteTaskPushNotificationConfig",
            ):
                return _jsonrpc_error(
                    req_id, JSONRPC_ERROR_CODES["PushNotificationNotSupportedError"],
                    "Push notifications not supported",
                )
            elif method == "SendStreamingMessage":
                # JSON-RPC streaming needs SSE; fall back to regular result
                return self._rpc_send_message(req_id, params)
            elif method == "SubscribeToTask":
                task = self.task_store.get(params.get("id", ""))
                if task is None:
                    return _jsonrpc_error(
                        req_id, JSONRPC_ERROR_CODES["TaskNotFoundError"],
                        "Task not found",
                    )
                return _jsonrpc_result(req_id, {"task": task})
            else:
                return _jsonrpc_error(req_id, -32601, f"Method not found: {method}")
        except Exception as exc:
            logger.exception("JSON-RPC dispatch error for method %s", method)
            return _jsonrpc_error(req_id, -32603, f"Internal error: {exc}")

    # ---- JSON-RPC method implementations ----------------------------------

    def _rpc_send_message(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        msg = params.get("message", {})
        parts = msg.get("parts", [])
        if not parts:
            return _jsonrpc_error(
                req_id, JSONRPC_ERROR_CODES["ContentTypeNotSupportedError"],
                "Message must contain at least one part",
            )
        context_id = msg.get("contextId")
        task = self.task_store.create(self.agent, parts, context_id)
        return _jsonrpc_result(req_id, {"task": task})

    def _rpc_get_task(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        task_id = params.get("id", "")
        hl = params.get("historyLength")
        task = self.task_store.get(task_id, hl)
        if task is None:
            return _jsonrpc_error(
                req_id, JSONRPC_ERROR_CODES["TaskNotFoundError"],
                f"Task {task_id} not found",
            )
        return _jsonrpc_result(req_id, task)

    def _rpc_list_tasks(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        result = self.task_store.list(
            context_id=params.get("contextId"),
            status=params.get("status"),
            page_size=int(params.get("pageSize", 50)),
            page_token=params.get("pageToken", ""),
            include_artifacts=params.get("includeArtifacts", False),
        )
        return _jsonrpc_result(req_id, result)

    def _rpc_cancel_task(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        task_id = params.get("id", "")
        task = self.task_store.cancel(task_id)
        if task is None:
            return _jsonrpc_error(
                req_id, JSONRPC_ERROR_CODES["TaskNotFoundError"],
                f"Task {task_id} not found",
            )
        return _jsonrpc_result(req_id, task)

    # ------------------------------------------------------------------ #
    # REST send-message helpers
    # ------------------------------------------------------------------ #

    def _handle_send_message_rest(self) -> None:
        try:
            body = self._read_body()
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_error_rest(400, "CONTENT_TYPE_NOT_SUPPORTED", "Invalid JSON")
            return

        msg = body.get("message", {})
        parts = msg.get("parts", [])
        if not parts:
            self._send_error_rest(400, "CONTENT_TYPE_NOT_SUPPORTED",
                                  "Message must contain at least one part")
            return
        context_id = msg.get("contextId")
        task = self.task_store.create(self.agent, parts, context_id)
        self._send_json(200, {"task": task})

    def _handle_send_message_stream_rest(self) -> None:
        """SSE streaming: send the task as events then close."""
        try:
            body = self._read_body()
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_error_rest(400, "CONTENT_TYPE_NOT_SUPPORTED", "Invalid JSON")
            return

        msg = body.get("message", {})
        parts = msg.get("parts", [])
        if not parts:
            self._send_error_rest(400, "CONTENT_TYPE_NOT_SUPPORTED",
                                  "Message must contain at least one part")
            return
        context_id = msg.get("contextId")
        task = self.task_store.create(self.agent, parts, context_id)
        self._send_sse([
            {"task": task},
            {"statusUpdate": {
                "taskId": task["id"],
                "contextId": task["contextId"],
                "status": task["status"],
            }},
        ])

    def _send_sse(self, events: list[dict[str, Any]]) -> None:
        """Send Server-Sent Events."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("A2A-Version", A2A_PROTOCOL_VERSION)
        self.end_headers()
        for event in events:
            data = json.dumps(event, ensure_ascii=False)
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("A2A HTTP: " + format, *args)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def build_card(
    name: str,
    description: str,
    base_url: str,
    skills: list[dict] | None = None,
) -> dict[str, Any]:
    """Build the Agent Card for this server.

    Advertises two protocol bindings so clients can discover the correct
    endpoint URLs:

    * **HTTP+JSON** (REST) at ``{base_url}`` — used for ``/message:send``,
      ``/tasks/{id}``, etc.
    * **JSONRPC** at ``{base_url}/rpc`` — used for JSON-RPC 2.0 calls.
    """
    return agent_card(
        name=name,
        description=description,
        url=base_url,  # REST base URL (not /rpc!)
        skills=skills or [],
        capabilities={
            "streaming": True,
            "pushNotifications": False,
            "extendedAgentCard": True,
        },
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        # Add a second interface for JSON-RPC so the client can discover
        # the correct /rpc endpoint for JSON-RPC calls.
        extra_interfaces=[
            {
                "url": f"{base_url}/rpc",
                "protocolBinding": "JSONRPC",
                "protocolVersion": A2A_PROTOCOL_VERSION,
            },
        ],
    )


def create_a2a_server(
    agent,
    host: str = "localhost",
    port: int = 8080,
    agent_name: str = "agentFour",
    agent_description: str = "An Ollama-powered tool-calling agent exposed via A2A",
    base_url: str | None = None,
    skills: list[dict] | None = None,
) -> ThreadingHTTPServer:
    """Create an A2A HTTP server wrapping an ``agentFour`` instance.

    Parameters
    ----------
    agent : agentFour
        The agent instance to serve.
    host, port : str, int
        Bind address.
    agent_name, agent_description : str
        Used in the Agent Card.
    base_url : str | None
        External URL for the Agent Card.  Defaults to ``http://{host}:{port}``.
    skills : list[dict] | None
        Optional AgentSkills for the card.

    Returns
    -------
    ThreadingHTTPServer
        Call ``.serve_forever()`` to start, ``.shutdown()`` to stop.
    """
    if base_url is None:
        base_url = f"http://{host}:{port}"

    card = build_card(agent_name, agent_description, base_url, skills)

    task_store = _TaskStore()

    class _Handler(A2ARequestHandler):
        pass

    _Handler.agent = agent
    _Handler.task_store = task_store
    _Handler.agent_card = card

    server = ThreadingHTTPServer((host, port), _Handler)
    logger.info(
        "A2A server created on %s:%d | agent=%s | card=%s/.well-known/agent-card.json",
        host, port, agent_name, base_url,
    )
    return server


def serve_in_thread(agent, **kwargs) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """ Convenience: create the server, start it in a daemon thread, return both. """
    server = create_a2a_server(agent, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("A2A server thread started")
    return server, thread
