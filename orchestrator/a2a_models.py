"""A2A protocol data-model builders (Layer 1).

Pure-Python helpers that construct the JSON objects defined by the
Agent2Agent (A2A) Protocol v1.0 specification.  All objects use
``camelCase`` field names and ``SCREAMING_SNAKE_CASE`` enum values,
matching the ProtoJSON serialization rules in Section 5.5 of the spec.

This module has *no* dependencies beyond the standard library so it can
be imported by both the server and client bindings without side effects.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

# --------------------------------------------------------------------- #
# Version
# --------------------------------------------------------------------- #

A2A_PROTOCOL_VERSION: str = "1.0"

# --------------------------------------------------------------------- #
# Task states  (Section 4.1.3)
# --------------------------------------------------------------------- #

TASK_STATE_SUBMITTED = "TASK_STATE_SUBMITTED"
TASK_STATE_WORKING = "TASK_STATE_WORKING"
TASK_STATE_COMPLETED = "TASK_STATE_COMPLETED"
TASK_STATE_FAILED = "TASK_STATE_FAILED"
TASK_STATE_CANCELED = "TASK_STATE_CANCELED"
TASK_STATE_REJECTED = "TASK_STATE_REJECTED"
TASK_STATE_INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
TASK_STATE_AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"

TERMINAL_STATES = frozenset({
    TASK_STATE_COMPLETED, TASK_STATE_FAILED,
    TASK_STATE_CANCELED, TASK_STATE_REJECTED,
})
CANCELABLE_STATES = frozenset({
    TASK_STATE_SUBMITTED, TASK_STATE_WORKING,
    TASK_STATE_INPUT_REQUIRED, TASK_STATE_AUTH_REQUIRED,
})
ALL_TASK_STATES = (
    TASK_STATE_SUBMITTED, TASK_STATE_WORKING, TASK_STATE_COMPLETED,
    TASK_STATE_FAILED, TASK_STATE_CANCELED, TASK_STATE_REJECTED,
    TASK_STATE_INPUT_REQUIRED, TASK_STATE_AUTH_REQUIRED,
)

# --------------------------------------------------------------------- #
# Roles  (Section 4.1.5)
# --------------------------------------------------------------------- #

ROLE_USER = "ROLE_USER"
ROLE_AGENT = "ROLE_AGENT"

# --------------------------------------------------------------------- #
# Error -> HTTP status mapping  (Section 5.4)
# --------------------------------------------------------------------- #

ERROR_HTTP_STATUS: dict[str, int] = {
    "TASK_NOT_FOUND": 404,
    "TASK_NOT_CANCELABLE": 400,
    "PUSH_NOTIFICATION_NOT_SUPPORTED": 400,
    "UNSUPPORTED_OPERATION": 400,
    "CONTENT_TYPE_NOT_SUPPORTED": 400,
    "INVALID_AGENT_RESPONSE": 500,
    "EXTENDED_AGENT_CARD_NOT_CONFIGURED": 400,
    "EXTENSION_SUPPORT_REQUIRED": 400,
    "VERSION_NOT_SUPPORTED": 400,
}

ERROR_MESSAGES: dict[str, str] = {
    "TASK_NOT_FOUND": "The specified task ID does not exist or is not accessible.",
    "TASK_NOT_CANCELABLE": "The task is not in a cancelable state.",
    "PUSH_NOTIFICATION_NOT_SUPPORTED": "Push notifications are not supported by this agent.",
    "UNSUPPORTED_OPERATION": "The requested operation is not supported.",
    "CONTENT_TYPE_NOT_SUPPORTED": "A media type in the request is not supported.",
    "INVALID_AGENT_RESPONSE": "The agent returned an invalid response.",
    "EXTENDED_AGENT_CARD_NOT_CONFIGURED": "No extended agent card is configured.",
    "EXTENSION_SUPPORT_REQUIRED": "A required extension was not declared by the client.",
    "VERSION_NOT_SUPPORTED": "The requested A2A protocol version is not supported.",
}

_HTTP_STATUS_NAMES: dict[int, str] = {
    400: "INVALID_ARGUMENT",
    404: "NOT_FOUND",
    500: "INTERNAL",
}

# --------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------- #

def now_iso() -> str:
    """Current UTC time as an ISO-8601 string with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_id(prefix: str = "") -> str:
    """A fresh UUID, optionally prefixed for readability."""
    return prefix + str(uuid.uuid4()) if prefix else str(uuid.uuid4())


# --------------------------------------------------------------------- #
# Part builders  (Section 4.1.6)
# --------------------------------------------------------------------- #

def text_part(text: str) -> dict:
    """A ``TextPart``: ``{"text": "..."}``."""
    return {"text": text}


def data_part(data, media_type: str = "application/json") -> dict:
    """A ``DataPart``: ``{"data": {...}, "mediaType": "..."}``."""
    return {"data": data, "mediaType": media_type}


def file_part_raw(b64: str, filename: str | None = None,
                  media_type: str | None = None) -> dict:
    """A ``FilePart`` carrying inline base64 bytes."""
    p: dict = {"raw": b64}
    if filename:
        p["filename"] = filename
    if media_type:
        p["mediaType"] = media_type
    return p


def file_part_url(url: str, filename: str | None = None,
                  media_type: str | None = None) -> dict:
    """A ``FilePart`` referencing a remote URL."""
    p: dict = {"url": url}
    if filename:
        p["filename"] = filename
    if media_type:
        p["mediaType"] = media_type
    return p


# --------------------------------------------------------------------- #
# Message  (Section 4.1.4)
# --------------------------------------------------------------------- #

def message(
    role: str,
    parts: list[dict],
    message_id: str | None = None,
    task_id: str | None = None,
    context_id: str | None = None,
    reference_task_ids: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    m: dict = {
        "role": role,
        "parts": parts,
        "messageId": message_id or new_id("msg-"),
    }
    if task_id:
        m["taskId"] = task_id
    if context_id:
        m["contextId"] = context_id
    if reference_task_ids:
        m["referenceTaskIds"] = reference_task_ids
    if metadata:
        m["metadata"] = metadata
    return m


# --------------------------------------------------------------------- #
# Artifact  (Section 4.1.7)
# --------------------------------------------------------------------- #

def artifact(
    parts: list[dict],
    artifact_id: str | None = None,
    name: str | None = None,
    metadata: dict | None = None,
) -> dict:
    a: dict = {
        "artifactId": artifact_id or new_id("art-"),
        "parts": parts,
    }
    if name:
        a["name"] = name
    if metadata:
        a["metadata"] = metadata
    return a


# --------------------------------------------------------------------- #
# Task + TaskStatus  (Section 4.1.1 / 4.1.2)
# --------------------------------------------------------------------- #

def task(
    id: str,
    context_id: str,
    state: str,
    message: dict | None = None,
    artifacts: list[dict] | None = None,
    history: list[dict] | None = None,
    timestamp: str | None = None,
) -> dict:
    t: dict = {
        "id": id,
        "contextId": context_id,
        "status": {"state": state, "timestamp": timestamp or now_iso()},
    }
    if message is not None:
        t["status"]["message"] = message
    if artifacts is not None:
        t["artifacts"] = artifacts
    if history is not None:
        t["history"] = history
    return t


# --------------------------------------------------------------------- #
# Streaming events  (Section 4.2)
# --------------------------------------------------------------------- #

def status_update(
    task_id: str,
    state: str,
    context_id: str | None = None,
    message: dict | None = None,
    final: bool | None = None,
    timestamp: str | None = None,
) -> dict:
    su: dict = {
        "taskId": task_id,
        "status": {"state": state, "timestamp": timestamp or now_iso()},
    }
    if context_id:
        su["contextId"] = context_id
    if message is not None:
        su["status"]["message"] = message
    if final is not None:
        su["final"] = final
    return su


def artifact_update(
    task_id: str,
    artifact_obj: dict,
    context_id: str | None = None,
    append: bool | None = None,
) -> dict:
    au: dict = {"taskId": task_id, "artifact": artifact_obj}
    if context_id:
        au["contextId"] = context_id
    if append is not None:
        au["append"] = append
    return au


def stream_response(
    *,
    task: dict | None = None,
    message: dict | None = None,
    status_update: dict | None = None,
    artifact_update: dict | None = None,
) -> dict:
    """Build a ``StreamResponse`` wrapper containing exactly one payload."""
    if task is not None:
        return {"task": task}
    if message is not None:
        return {"message": message}
    if status_update is not None:
        return {"statusUpdate": status_update}
    if artifact_update is not None:
        return {"artifactUpdate": artifact_update}
    return {}


# --------------------------------------------------------------------- #
# Push-notification config  (Section 4.3.1)
# --------------------------------------------------------------------- #

def push_notification_config(
    url: str,
    config_id: str | None = None,
    authentication: dict | None = None,
    task_id: str | None = None,
) -> dict:
    c: dict = {"url": url}
    if config_id:
        c["id"] = config_id
    if authentication:
        c["authentication"] = authentication
    if task_id:
        c["taskId"] = task_id
    return c


# --------------------------------------------------------------------- #
# Agent-card objects  (Section 4.4)
# --------------------------------------------------------------------- #

def agent_skill(
    skill_id: str,
    name: str,
    description: str,
    tags: list[str] | None = None,
    examples: list[str] | None = None,
    input_modes: list[str] | None = None,
    output_modes: list[str] | None = None,
) -> dict:
    s: dict = {"id": skill_id, "name": name, "description": description}
    if tags:
        s["tags"] = tags
    if examples:
        s["examples"] = examples
    s["inputModes"] = input_modes or ["text/plain"]
    s["outputModes"] = output_modes or ["text/plain"]
    return s


def agent_card(
    name: str,
    description: str,
    url: str,
    skills: list[dict],
    capabilities: dict | None = None,
    provider: dict | None = None,
    version: str = "1.0.0",
    default_input_modes: list[str] | None = None,
    default_output_modes: list[str] | None = None,
    security_schemes: dict | None = None,
    security: list | None = None,
    documentation_url: str | None = None,
) -> dict:
    card: dict = {
        "name": name,
        "description": description,
        "version": version,
        "supportedInterfaces": [
            {
                "url": url,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": A2A_PROTOCOL_VERSION,
            }
        ],
        "capabilities": capabilities or {
            "streaming": True,
            "pushNotifications": True,
            "extendedAgentCard": True,
        },
        "defaultInputModes": default_input_modes or ["text/plain"],
        "defaultOutputModes": default_output_modes or ["text/plain"],
        "skills": skills,
    }
    if provider:
        card["provider"] = provider
    if security_schemes:
        card["securitySchemes"] = security_schemes
    if security:
        card["security"] = security
    if documentation_url:
        card["documentationUrl"] = documentation_url
    return card


# --------------------------------------------------------------------- #
# Error response builder  (Section 11.6)
# --------------------------------------------------------------------- #

def a2a_error(
    reason: str,
    message: str | None = None,
    http_status: int | None = None,
    extra_details: list[dict] | None = None,
) -> tuple[int, dict]:
    """Return ``(http_status, body)`` for an A2A-specific error."""
    status = http_status or ERROR_HTTP_STATUS.get(reason, 500)
    details: list[dict] = [
        {
            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
            "reason": reason,
            "domain": "a2a-protocol.org",
        }
    ]
    if extra_details:
        details.extend(extra_details)
    body = {
        "error": {
            "code": status,
            "status": _HTTP_STATUS_NAMES.get(status, "INTERNAL"),
            "message": message or ERROR_MESSAGES.get(reason, reason),
            "details": details,
        }
    }
    return status, body


# --------------------------------------------------------------------- #
# Helpers used by both client and server
# --------------------------------------------------------------------- #

def extract_text(parts: list | None) -> str:
    """Concatenate all textual content from a list of Parts."""
    if not parts:
        return ""
    chunks: list[str] = []
    for p in parts:
        if isinstance(p, dict):
            if p.get("text"):
                chunks.append(p["text"])
            elif "data" in p:
                try:
                    chunks.append(json.dumps(p["data"], ensure_ascii=False))
                except (TypeError, ValueError):
                    chunks.append(str(p["data"]))
    return "\n".join(chunks)


def artifact_text(artifacts: list[dict] | None) -> str:
    """Concatenate text from all artifacts (for quick display)."""
    if not artifacts:
        return ""
    return "\n\n".join(extract_text(a.get("parts")) for a in artifacts if isinstance(a, dict))