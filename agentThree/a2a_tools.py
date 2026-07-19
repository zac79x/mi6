"""A2A tools: let the agent call remote A2A agents as a client.

Registers tools that allow agentThree to discover and interact with
other A2A-compliant agents during a chat session.  This implements
**Role = C** (client) at the tool layer.

Tools provided
--------------
* ``a2a_discover``     - Fetch and display an agent's Agent Card.
* ``a2a_send_message`` - Send a message to a remote agent (blocking),
                         return the final text answer.
* ``a2a_get_task``      - Retrieve the status/artifacts of a task.
* ``a2a_list_tasks``    - List tasks on a remote agent.

The ``A2AClient`` instance is cached per base URL so repeated calls
to the same agent reuse the discovered Agent Card.
"""

from __future__ import annotations

from typing import Any

from agentThree.logging_setup import logger
from agentThree.tools_registry import tool

# Cache: base_url -> A2AClient
_client_cache: dict[str, Any] = {}


def _get_client(base_url: str) -> Any:
    """Return a cached ``A2AClient`` for *base_url*, creating one on first use."""
    from agentThree.a2a_client import A2AClient

    base_url = base_url.rstrip("/")
    if base_url not in _client_cache:
        _client_cache[base_url] = A2AClient(base_url)
        logger.info("A2A client created for %s", base_url)
    return _client_cache[base_url]


# --------------------------------------------------------------------- #
# Tool parameter schemas
# --------------------------------------------------------------------- #

_DISCOVER_PARAMS: dict = {
    "type": "object",
    "properties": {
        "base_url": {
            "type": "string",
            "description": "Root URL of the remote A2A agent, e.g. http://localhost:8080",
        },
    },
    "required": ["base_url"],
}

_SEND_MESSAGE_PARAMS: dict = {
    "type": "object",
    "properties": {
        "base_url": {
            "type": "string",
            "description": "Root URL of the remote A2A agent, e.g. http://localhost:8080",
        },
        "text": {
            "type": "string",
            "description": "The message text to send to the remote agent.",
        },
        "context_id": {
            "type": "string",
            "description": "Optional context ID for multi-turn conversations.",
        },
    },
    "required": ["base_url", "text"],
}

_GET_TASK_PARAMS: dict = {
    "type": "object",
    "properties": {
        "base_url": {
            "type": "string",
            "description": "Root URL of the remote A2A agent.",
        },
        "task_id": {
            "type": "string",
            "description": "The task ID to retrieve.",
        },
        "history_length": {
            "type": "integer",
            "description": "Maximum number of history messages to return.",
        },
    },
    "required": ["base_url", "task_id"],
}

_LIST_TASKS_PARAMS: dict = {
    "type": "object",
    "properties": {
        "base_url": {
            "type": "string",
            "description": "Root URL of the remote A2A agent.",
        },
        "context_id": {
            "type": "string",
            "description": "Filter tasks by context ID.",
        },
        "status": {
            "type": "string",
            "description": "Filter by task state, e.g. TASK_STATE_WORKING.",
        },
        "page_size": {
            "type": "integer",
            "description": "Maximum tasks per page (1-100). Default 50.",
        },
    },
    "required": ["base_url"],
}


# --------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------- #


@tool(
    name="a2a_discover",
    description=(
        "Discover a remote A2A agent: fetch its Agent Card (name, "
        "description, skills, capabilities) from /.well-known/agent-card.json."
    ),
    parameters=_DISCOVER_PARAMS,
)
def a2a_discover(base_url: str) -> str:
    import json

    try:
        client = _get_client(base_url)
    except Exception as exc:
        logger.warning("a2a_discover failed for %s: %s", base_url, exc)
        return f"Error discovering agent at {base_url}: {exc}"

    card = client.agent_card
    if not card:
        return f"No agent card found at {base_url}"

    lines = [
        f"Agent Card for {base_url}:",
        f"  Name: {card.get('name', '?')}",
        f"  Description: {card.get('description', '?')}",
        f"  Version: {card.get('version', '?')}",
    ]
    caps = card.get("capabilities", {})
    lines.append(
        f"  Capabilities: streaming={caps.get('streaming')}, "
        f"pushNotifications={caps.get('pushNotifications')}, "
        f"extendedAgentCard={caps.get('extendedAgentCard')}"
    )
    skills = card.get("skills", [])
    if skills:
        lines.append(f"  Skills ({len(skills)}):")
        for s in skills:
            lines.append(f"    - {s.get('name', '?')}: {s.get('description', '')}")
    interfaces = card.get("supportedInterfaces", [])
    if interfaces:
        lines.append("  Interfaces:")
        for iface in interfaces:
            lines.append(
                f"    - {iface.get('protocolBinding', '?')} v{iface.get('protocolVersion', '?')} @ {iface.get('url', '?')}"
            )
    return "\n".join(lines)


@tool(
    name="a2a_send_message",
    description=(
        "Send a message to a remote A2A agent and return its final text answer. "
        "The remote agent processes the message (possibly calling its own tools) "
        "and the completed task's artifact text is returned."
    ),
    parameters=_SEND_MESSAGE_PARAMS,
)
def a2a_send_message(base_url: str, text: str, context_id: str | None = None) -> str:
    try:
        client = _get_client(base_url)
        result = client.send_message(text, context_id=context_id)
    except Exception as exc:
        logger.warning("a2a_send_message failed for %s: %s", base_url, exc)
        return f"Error sending message to {base_url}: {exc}"

    from agentThree.a2a_client import A2AClient

    return A2AClient._result_to_text(result)


@tool(
    name="a2a_get_task",
    description="Retrieve the current status, artifacts, and optionally history of a task on a remote A2A agent.",
    parameters=_GET_TASK_PARAMS,
)
def a2a_get_task(base_url: str, task_id: str, history_length: int | None = None) -> str:
    import json

    try:
        client = _get_client(base_url)
        task = client.get_task(task_id, history_length=history_length)
    except Exception as exc:
        logger.warning("a2a_get_task failed for %s/%s: %s", base_url, task_id, exc)
        return f"Error getting task {task_id} from {base_url}: {exc}"

    return json.dumps(task, indent=2, ensure_ascii=False)


@tool(
    name="a2a_list_tasks",
    description="List tasks on a remote A2A agent with optional filtering by context ID and status.",
    parameters=_LIST_TASKS_PARAMS,
)
def a2a_list_tasks(
    base_url: str,
    context_id: str | None = None,
    status: str | None = None,
    page_size: int = 50,
) -> str:
    import json

    try:
        client = _get_client(base_url)
        result = client.list_tasks(
            context_id=context_id, status=status, page_size=page_size,
        )
    except Exception as exc:
        logger.warning("a2a_list_tasks failed for %s: %s", base_url, exc)
        return f"Error listing tasks from {base_url}: {exc}"

    return json.dumps(result, indent=2, ensure_ascii=False)