"""A2A Client: discover and talk to remote A2A agents (standalone for orchestrator).

This is a self-contained copy of agentThree/a2a_client.py with imports
adjusted so the orchestrator package has zero runtime dependency on
agentThree.  It uses only ``requests`` and the standard library.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Iterator

import requests

from orchestrator.a2a_models import (
    A2A_PROTOCOL_VERSION,
    ROLE_USER,
    TERMINAL_STATES,
    text_part,
    message,
    extract_text,
    artifact_text,
)

logger = logging.getLogger("orchestrator.a2a_client")

DEFAULT_TIMEOUT = 120  # seconds
SSE_READ_TIMEOUT = 300  # longer for streaming


class A2AClient:
    """Client for a single remote A2A agent endpoint."""

    def __init__(self, base_url: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.agent_card: dict[str, Any] | None = None
        self.rpc_url: str | None = None
        self.rest_url: str | None = None
        self._discover()

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def _discover(self) -> None:
        card_url = f"{self.base_url}/.well-known/agent-card.json"
        logger.debug("A2A client fetching agent card from %s", card_url)
        resp = requests.get(card_url, timeout=self.timeout)
        resp.raise_for_status()
        self.agent_card = resp.json()
        logger.info(
            "A2A agent card retrieved: name=%s, version=%s",
            self.agent_card.get("name"),
            self.agent_card.get("version"),
        )

        interfaces = self.agent_card.get("supportedInterfaces", [])
        if not interfaces:
            self.rpc_url = f"{self.base_url}/rpc"
            self.rest_url = self.base_url
            return

        rest_if = None
        rpc_if = None
        for iface in interfaces:
            binding = iface.get("protocolBinding", "").upper()
            if "HTTP" in binding and rest_if is None:
                rest_if = iface
            if "JSONRPC" in binding and rpc_if is None:
                rpc_if = iface

        if rest_if:
            self.rest_url = rest_if["url"]
        elif interfaces:
            self.rest_url = interfaces[0]["url"]
        else:
            self.rest_url = self.base_url

        if rpc_if:
            self.rpc_url = rpc_if["url"]
        else:
            self.rpc_url = f"{self.rest_url.rstrip('/')}/rpc"

    # ------------------------------------------------------------------ #
    # Low-level helpers
    # ------------------------------------------------------------------ #

    def _headers(self, extra: dict | None = None) -> dict[str, str]:
        h = {
            "Content-Type": "application/a2a+json",
            "Accept": "application/a2a+json",
            "A2A-Version": A2A_PROTOCOL_VERSION,
        }
        if extra:
            h.update(extra)
        return h

    # ------------------------------------------------------------------ #
    # Send Message (blocking, returns a Task or Message)
    # ------------------------------------------------------------------ #

    def send_message(
        self,
        text: str,
        context_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        parts = [text_part(text)]
        msg = message(
            role=ROLE_USER,
            parts=parts,
            task_id=task_id,
            context_id=context_id,
        )
        body = {"message": msg}

        url = f"{self.rest_url.rstrip('/')}/message:send"
        logger.debug("A2A send_message -> %s", url)
        resp = requests.post(
            url,
            json=body,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    # Send Streaming Message (SSE)
    # ------------------------------------------------------------------ #

    def send_streaming_message(
        self,
        text: str,
        context_id: str | None = None,
        task_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        parts = [text_part(text)]
        msg = message(
            role=ROLE_USER,
            parts=parts,
            task_id=task_id,
            context_id=context_id,
        )
        body = {"message": msg}

        url = f"{self.rest_url.rstrip('/')}/message:stream"
        logger.debug("A2A send_streaming_message -> %s", url)
        resp = requests.post(
            url,
            json=body,
            headers=self._headers(),
            stream=True,
            timeout=(self.timeout, SSE_READ_TIMEOUT),
        )
        resp.raise_for_status()

        data_lines: list[str] = []

        for raw_line in resp.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.strip()
            if not line:
                if data_lines:
                    data_str = "\n".join(data_lines)
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning("A2A SSE: could not parse: %s", data_str[:200])
                    data_lines = []
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

        if data_lines:
            data_str = "\n".join(data_lines)
            try:
                yield json.loads(data_str)
            except json.JSONDecodeError:
                logger.warning("A2A SSE: could not parse final: %s", data_str[:200])
        resp.close()

    # ------------------------------------------------------------------ #
    # Task operations (REST)
    # ------------------------------------------------------------------ #

    def get_task(self, task_id: str, history_length: int | None = None) -> dict[str, Any]:
        url = f"{self.rest_url.rstrip('/')}/tasks/{task_id}"
        params = {}
        if history_length is not None:
            params["historyLength"] = history_length
        resp = requests.get(url, params=params, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def list_tasks(
        self,
        context_id: str | None = None,
        status: str | None = None,
        page_size: int = 50,
        page_token: str = "",
        include_artifacts: bool = False,
    ) -> dict[str, Any]:
        url = f"{self.rest_url.rstrip('/')}/tasks"
        params: dict[str, Any] = {"pageSize": page_size}
        if context_id:
            params["contextId"] = context_id
        if status:
            params["status"] = status
        if page_token:
            params["pageToken"] = page_token
        if include_artifacts:
            params["includeArtifacts"] = "true"
        resp = requests.get(url, params=params, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        url = f"{self.rest_url.rstrip('/')}/tasks/{task_id}:cancel"
        resp = requests.post(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    # High-level convenience
    # ------------------------------------------------------------------ #

    def ask(self, text: str, stream: bool = False) -> str:
        if stream:
            return self._ask_streaming(text)
        else:
            result = self.send_message(text)
            return self._result_to_text(result)

    def _ask_streaming(self, text: str) -> str:
        final_text = ""
        for event in self.send_streaming_message(text):
            if "task" in event:
                task = event["task"]
                state = task.get("status", {}).get("state", "")
                if state in TERMINAL_STATES:
                    final_text = artifact_text(task.get("artifacts")) or self._status_text(task)
            elif "message" in event:
                msg = event["message"]
                final_text = extract_text(msg.get("parts"))
                print(final_text, end="", flush=True)
            elif "statusUpdate" in event:
                su = event["statusUpdate"]
                state = su.get("status", {}).get("state", "")
                msg = su.get("status", {}).get("message")
                if msg:
                    txt = extract_text(msg.get("parts"))
                    if txt:
                        print(txt, end="", flush=True)
                if state in TERMINAL_STATES:
                    logger.debug("A2A stream terminal state: %s", state)
            elif "artifactUpdate" in event:
                au = event["artifactUpdate"]
                art = au.get("artifact", {})
                txt = extract_text(art.get("parts"))
                if txt:
                    print(txt, end="", flush=True)
                    final_text = txt
        print()
        return final_text

    @staticmethod
    def _result_to_text(result: dict[str, Any]) -> str:
        if "task" in result:
            task = result["task"]
            art_text = artifact_text(task.get("artifacts"))
            if art_text:
                return art_text
            return A2AClient._status_text(task)
        if "message" in result:
            return extract_text(result["message"].get("parts"))
        return json.dumps(result, indent=2)

    @staticmethod
    def _status_text(task: dict[str, Any]) -> str:
        status = task.get("status", {})
        msg = status.get("message")
        if msg:
            return extract_text(msg.get("parts"))
        return f"[Task state: {status.get('state', 'unknown')}]"

    # ------------------------------------------------------------------ #
    # JSON-RPC methods
    # ------------------------------------------------------------------ #

    def _jsonrpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        body = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params,
        }
        resp = requests.post(
            self.rpc_url,
            json=body,
            headers=self._headers({"Content-Type": "application/json"}),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise A2AClientError(
                code=data["error"].get("code"),
                message=data["error"].get("message", "Unknown error"),
            )
        return data.get("result", {})

    def send_message_rpc(
        self, text: str, context_id: str | None = None, task_id: str | None = None,
    ) -> dict[str, Any]:
        parts = [text_part(text)]
        msg = message(role=ROLE_USER, parts=parts, task_id=task_id, context_id=context_id)
        return self._jsonrpc("SendMessage", {"message": msg})

    def get_task_rpc(self, task_id: str, history_length: int | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"id": task_id}
        if history_length is not None:
            params["historyLength"] = history_length
        return self._jsonrpc("GetTask", params)

    def list_tasks_rpc(self, **kwargs: Any) -> dict[str, Any]:
        return self._jsonrpc("ListTasks", kwargs)

    def cancel_task_rpc(self, task_id: str) -> dict[str, Any]:
        return self._jsonrpc("CancelTask", {"id": task_id})


class A2AClientError(Exception):
    """Error returned by the A2A server."""

    def __init__(self, code: int | None = None, message: str = "") -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}" if code else message)