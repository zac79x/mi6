"""REPL entry point: main() with slash-command handling and tab completion."""

from __future__ import annotations

import os

import requests

from agentThree.agent import agentThree, DEFAULT_SESSION_FILE
from agentThree.cli_ui import (
    AgentCompleter, ColouredPrompt, c, colours_enabled, info as ui_info,
    success as ui_success, warn as ui_warn, error as ui_error, dim as ui_dim, banner as ui_banner,
)
from agentThree.config import DEFAULT_MODEL, DIFF_TOOL_PATH, LOG_FILE, OLLAMA_URL
from agentThree.logging_setup import logger
from agentThree.tools_registry import _TOOL_REGISTRY

HELP_TEXT = """\
Commands:
  /quit  or  /exit   end the session
  /reset             clear the conversation
  Ctrl-C            interrupt the agent while it is working and return to the prompt
  /?  or  /help     show this help message
  /temp <value>     set sampling temperature, e.g. /temp 0.7  (blank = default)
  /max_iter <n>     set max tool-calling iterations, e.g. /max_iter 10
  /think on|off     enable/disable display of the model's chain-of-thought
  /compact          compact the in-memory conversation history in place
  /listcache        list the cached tool-result refs and their sizes
  /save [path]      save the current session to disk (default: .agent_session_state.json)
  /restore [path]   restore a previously saved session (default: .agent_session_state.json)
"""


def _print_help() -> None:
    if colours_enabled():
        lines = HELP_TEXT.splitlines(keepends=True)
        if lines:
            print(c(lines[0], "bold", "bright_blue"), end="")
            for ln in lines[1:]:
                print(ln, end="")
        else:
            print(HELP_TEXT, end="" if HELP_TEXT.endswith("\n") else "\n")
    else:
        print(HELP_TEXT, end="" if HELP_TEXT.endswith("\n") else "\n")


def _local_files() -> list[str]:
    try:
        return sorted(f for f in os.listdir(".") if os.path.isfile(f) and (f.endswith(".py") or f.endswith(".json")))
    except OSError:
        return []


def _handle_command(cmd: str, agent: agentThree) -> bool:
    parts = cmd.split()
    head = parts[0].lower()

    if head in ("/?", "/help", "?"):
        _print_help()
        logger.info("User requested help via %r", cmd)
        return True

    if head == "/temp":
        if len(parts) == 1:
            agent.temperature = None
            ui_dim("[temperature reset to model default]")
            logger.info("User reset temperature to default")
        else:
            try:
                value = float(parts[1])
            except ValueError:
                ui_error(f"{parts[1]!r} is not a valid number. Usage: /temp <value between 0 and 2>")
                return True
            if not 0.0 <= value <= 2.0:
                ui_error(f"temperature must be between 0.0 and 2.0, got {value}")
                return True
            agent.temperature = value
            ui_success(f"[temperature set to {value}]")
            logger.info("User set temperature to %s", value)
        print()
        return True

    if head == "/max_iter":
        if len(parts) != 2:
            ui_warn("Usage: /max_iter <positive integer>")
            print()
            return True
        try:
            value = int(parts[1])
        except ValueError:
            ui_error(f"{parts[1]!r} is not a valid integer. Usage: /max_iter <positive integer>")
            print()
            return True
        if value < 1:
            ui_error(f"max_iterations must be >= 1, got {value}")
            print()
            return True
        agent.max_iterations = value
        ui_success(f"[max_iterations set to {value}]")
        logger.info("User set max_iterations to %d", value)
        print()
        return True

    if head == "/think":
        if len(parts) != 2 or parts[1].lower() not in {"on", "off", "true", "false", "1", "0", "yes", "no"}:
            ui_warn("Usage: /think on | /think off")
            print()
            return True
        new_state = parts[1].lower() in {"on", "true", "1", "yes"}
        agent.show_thinking = new_state
        ui_success(f"[thinking display {'enabled' if new_state else 'disabled'}]")
        logger.info("User %s thinking display", "enabled" if new_state else "disabled")
        print()
        return True

    if head == "/compact":
        if len(parts) != 1:
            ui_warn("Usage: /compact")
            print()
            return True
        if not agent.messages:
            ui_warn("[conversation is empty - nothing to compact]")
            logger.info("User ran /compact on an empty history")
            print()
            return True
        report = agent.compact_history()
        ui_success(f"[compacted: messages {report['messages_before']} -> {report['messages_after']}, chars {report['chars_before']} -> {report['chars_after']} (saved {report['chars_saved']}), cache {report['cache_before']} -> {report['cache_after']} entry/ies]")
        print()
        return True

    if head == "/listcache":
        if len(parts) != 1:
            ui_warn("Usage: /listcache")
            print()
            return True
        entries = agent.list_cache()
        if not entries:
            ui_dim("[result cache is empty]")
            logger.info("User ran /listcache on an empty cache")
            print()
            return True
        total_chars = sum(e["chars"] for e in entries)
        ui_info(f"[result cache: {len(entries)} entry/ies, {total_chars} total chars]")
        for e in entries:
            print(f"  {e['ref']}  {e['chars']:>8} chars  | {e['preview']}")
        print()
        logger.info("User ran /listcache: %d entries, %d total chars", len(entries), total_chars)
        return True

    if head == "/save":
        if len(parts) > 2:
            ui_warn("Usage: /save [path]")
            print()
            return True
        path = parts[1] if len(parts) == 2 else None
        try:
            saved_to = agent.save_session(path)
        except OSError as exc:
            ui_error(f"Error saving session: {exc}")
            logger.error("Failed to save session: %s", exc)
            print()
            return True
        ui_success(f"Session saved to {saved_to}")
        logger.info("User saved session to %s", saved_to)
        print()
        return True

    if head == "/restore":
        if len(parts) > 2:
            ui_warn("Usage: /restore [path]")
            print()
            return True
        path = parts[1] if len(parts) == 2 else None
        try:
            report = agent.restore_session(path)
        except FileNotFoundError as exc:
            ui_error(f"session file not found: {exc.filename} (use /save first, or provide a valid path)")
            logger.error("Session file not found for /restore: %s", exc)
            print()
            return True
        except (ValueError, OSError) as exc:
            ui_error(f"Error restoring session: {exc}")
            logger.error("Failed to restore session: %s", exc)
            print()
            return True
        ui_success(f"Session restored from {report['path']} ({report['messages']} messages, {report['cache_entries']} cache entries).")
        logger.info("User restored session from %s (%d messages, %d cache entries)", report["path"], report["messages"], report["cache_entries"])
        print()
        return True

    return False


def main() -> None:
    ui_banner("Ollama tool-calling agent")
    print(c(f"Model  : {DEFAULT_MODEL}", "cyan"))
    print(c(f"Server : {OLLAMA_URL}", "cyan"))
    print(c(f"Tools  : {[t.name for t in _TOOL_REGISTRY.values()]}", "green"))
    print(c(f"Logfile: {os.path.abspath(LOG_FILE)}", "gray"))
    print()
    _print_help()

    logger.info("Agent started | model=%s | url=%s | diff_tool=%s | tools=%s", DEFAULT_MODEL, OLLAMA_URL, DIFF_TOOL_PATH, [t.name for t in _TOOL_REGISTRY.values()])

    try:
        logger.debug("Probing Ollama at http://localhost:11434/api/tags")
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        logger.info("Ollama server reachable (status=%d)", r.status_code)
    except requests.exceptions.RequestException as exc:
        logger.critical("Could not reach Ollama at %s: %s", OLLAMA_URL, exc)
        raise SystemExit(f"Could not reach Ollama at {OLLAMA_URL}: {exc}\nIs 'ollama serve' running?")

    agent = agentThree(model=DEFAULT_MODEL)
    completer = AgentCompleter(extra_files=_local_files())
    prompt = ColouredPrompt(completer=completer, bottom_toolbar=agent.render_status_bar)

    while True:
        user_input = prompt.read()
        if user_input is None:
            logger.info("User ended the session (EOF or KeyboardInterrupt)")
            print()
            break
        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "/quit", "/exit"}:
            logger.info("User typed '%s' - exiting", user_input)
            break
        if user_input.lower() in {"reset", "/reset"}:
            agent.reset()
            ui_success("[conversation reset]")
            print()
            continue
        if user_input.lower() in {"help", "?"}:
            _print_help()
            logger.info("User requested help via %r", user_input)
            continue
        if user_input.startswith("/") and _handle_command(user_input, agent):
            continue

        print(c("Agent> ", "bold", "bright_yellow"), end="", flush=True)
        try:
            answer = agent.chat(user_input)
        except KeyboardInterrupt:
            logger.info("Chat interrupted by user (Ctrl-C) in main loop")
            agent.state = "idle"
            print()
            answer = "[interrupted by user - task stopped]"
        except requests.exceptions.RequestException as exc:
            logger.error("Connection error during chat: %s", exc)
            answer = f"[connection error: {exc}]"
        except Exception as exc:
            logger.exception("Unhandled exception during chat")
            answer = f"[error: {exc}]"

        agent.state = "idle"
        print(answer)
        agent.print_status_bar()
        print()
        logger.info("Printed agent answer to console (%d chars)", len(answer))

    logger.info("Agent session ended")
