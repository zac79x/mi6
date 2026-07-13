"""This module owns the ``main()`` entry point: it prints the startup
banner, configures the diff tool, instantiates the :class:`agentTwo`, and
runs the read-eval-print loop with its ``/`` commands.

The REPL uses the human-friendly CLI helpers from :mod:`agent.cli_ui`:
ANSI colours (graceful degradation on non-TTY terminals), semantic
printers and a prompt_toolkit-backed prompt with tab completion for
the slash commands.  When ``prompt_toolkit`` is not installed the
agent falls back to a plain but still colour-aware ``input()`` prompt.
"""

from __future__ import annotations

import os

import requests

from agent.agent import agentTwo, DEFAULT_SESSION_FILE
from agent.cli_ui import (
    AgentCompleter,
    ColouredPrompt,
    c,
    colours_enabled,
    enable_colours,
    info as ui_info,
    success as ui_success,
    warn as ui_warn,
    error as ui_error,
    dim as ui_dim,
    banner as ui_banner,
)
from agent.config import (
    DEFAULT_MODEL,
    DIFF_TOOL_PATH,
    LOG_FILE,
    OLLAMA_URL,
)
from agent.diff_tool import (
    configure_diff_tool,
    set_kdiff3_path_interactively,
)
from agent.logging_setup import logger
from agent.tools_registry import _TOOL_REGISTRY

# --------------------------------------------------------------------------- #
# cli_ui fallbacks
# --------------------------------------------------------------------------- #
# Soft import already guarantees the functions exist, but keep explicit
# fallbacks anyway so a broken/missing cli_ui never crashes the REPL.
if AgentCompleter is None:           # pragma: no cover
    class AgentCompleter:            # type: ignore[no-redef]
        def __init__(self, *a, **k):
            pass
if ColouredPrompt is None:           # pragma: no cover
    class ColouredPrompt:            # type: ignore[no-redef]
        def __init__(self, *a, **k):
            pass
        def read(self):
            try:
                return input("You> ")
            except (EOFError, KeyboardInterrupt):
                return None

# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

# Help text shown at startup and by the /help (/?) command.
# Kept in one place so the banner and the help command never drift apart.
HELP_TEXT = """\
Commands:
  /quit  or  /exit   end the session
  /reset             clear the conversation
  Ctrl-C            interrupt the agent while it is working (thinking or
                    calling a tool) and return to the prompt.  The session
                    is NOT ended - only the current task is stopped, so
                    you can immediately type a new request.
  /?  or  /help     show this help message
  /temp <value>     set sampling temperature, e.g. /temp 0.7  (blank = default)
  /max_iter <n>     set max tool-calling iterations, e.g. /max_iter 10
  /think on|off     enable/disable display of the model's chain-of-thought
  /kdiff [<path>]   set the kdiff3 binary path in .env (KDIFF3_PATH=...).
                    With no argument, prompts for the path interactively.
                    Updates the running session immediately.
  /compact          compact the in-memory conversation history in place
                    (cache large tool results, dedup consecutive duplicates,
                    strip 'thinking'). Full bytes stay reachable via the
                    result cache and the recall_cached_result tool.
  /listcache        list the cached tool-result refs and their sizes
                    (does not print the content).
  /save [path]      save the current session (history, result cache and all
                    runtime settings) to disk.  Defaults to
                    .agent_session_state.json in the current directory.
  /restore [path]   restore a previously saved session, replacing the
                    current history, cache and settings.  Defaults to
                    .agent_session_state.json in the current directory.

The prompt supports TAB completion for the slash commands above and their
arguments (e.g. /think on|off, /kdiff <file>, /save <file>).  Output is
colourised when the terminal supports it; set NO_COLOR=1 to disable colours.
"""


def _print_help() -> None:
    """Print the interactive-command help text (single source of truth)."""
    # Colourise the section header line when colours are available.
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


# ---------------------------------------------------------------------------
# Local-file helper for tab completion
# ---------------------------------------------------------------------------

def _local_files() -> list[str]:
    """Return a sorted list of files in the current directory.

    Used to feed tab completion for the path-taking slash commands
    (``/kdiff`` wants ``*.py``, ``/save`` / ``/restore`` want ``*.json``).
    The completer itself filters by extension based on the command being
    completed, so we can hand it a single combined list here.
    """
    try:
        return sorted(
            f for f in os.listdir(".")
            if os.path.isfile(f) and (f.endswith(".py") or f.endswith(".json"))
        )
    except OSError:
        return []


# ---------------------------------------------------------------------------
# /command dispatch
# ---------------------------------------------------------------------------

def _handle_command(cmd: str, agent: agentTwo) -> bool:
    """Handle one of the ``/`` interactive commands.

    Returns True if the input was a recognised command (and the main
    loop should NOT forward it to the LLM), False otherwise.
    """
    parts = cmd.split()
    head = parts[0].lower()

    if head in ("/?", "/help", "?"):
        _print_help()
        logger.info("User requested help via %r", cmd)
        return True

    if head == "/temp":
        if len(parts) == 1:
            # Blank -> reset to model default
            agent.temperature = None
            ui_dim("[temperature reset to model default]")
            logger.info("User reset temperature to default")
        else:
            try:
                value = float(parts[1])
            except ValueError:
                ui_error(f"{parts[1]!r} is not a valid number. "
                         f"Usage: /temp <value between 0 and 2>")
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
            ui_error(f"{parts[1]!r} is not a valid integer. "
                     f"Usage: /max_iter <positive integer>")
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
        if (len(parts) != 2 or
                parts[1].lower() not in {"on", "off", "true", "false", "1", "0", "yes", "no"}):
            ui_warn("Usage: /think on | /think off")
            print()
            return True
        new_state = parts[1].lower() in {"on", "true", "1", "yes"}
        agent.show_thinking = new_state
        ui_success(f"[thinking display {'enabled' if new_state else 'disabled'}]")
        logger.info("User %s thinking display", "enabled" if new_state else "disabled")
        print()
        return True

    if head == "/kdiff":
        return set_kdiff3_path_interactively(parts, env_path=".env")

    if head == "/compact":
        # Run the same passes the wire-compaction helper uses, but on
        # the stored history itself. Frees in-process memory and makes
        # future log dumps of ``self.messages`` much smaller.
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
        ui_success(
            f"[compacted: messages {report['messages_before']} -> "
            f"{report['messages_after']}, "
            f"chars {report['chars_before']} -> {report['chars_after']} "
            f"(saved {report['chars_saved']}), "
            f"cache {report['cache_before']} -> {report['cache_after']} "
            f"entry/ies]"
        )
        print()
        return True

    if head == "/listcache":
        # Debug-friendly view of the per-agent result cache. Does NOT
        # print the cached content (the LLM has the recall tool for
        # that) - just refs and sizes, so a human can see at a glance
        # what the agent has accumulated.
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
        # Compute the total for a one-line summary at the bottom.
        total_chars = sum(e["chars"] for e in entries)
        ui_info(f"[result cache: {len(entries)} entry/ies, {total_chars} total chars]")
        for e in entries:
            print(f"  {e['ref']}  {e['chars']:>8} chars  | {e['preview']}")
        print()
        logger.info(
            "User ran /listcache: %d entries, %d total chars",
            len(entries), total_chars,
        )
        return True

    if head == "/save":
        # Persist the full session: the conversation history, the
        # per-agent result cache (so compacted tool results stay
        # recallable after a restore), and all runtime settings that
        # the user may have changed via /temp, /max_iter, /think, etc.
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
        # Replace the current history, result cache, and settings with
        # the contents of a previously /save'd file. This lets the user
        # continue a long-running task across separate REPL sessions
        # without losing context or configuration tweaks.
        if len(parts) > 2:
            ui_warn("Usage: /restore [path]")
            print()
            return True
        path = parts[1] if len(parts) == 2 else None
        try:
            report = agent.restore_session(path)
        except FileNotFoundError as exc:
            ui_error(f"session file not found: {exc.filename}"
                     f" (use /save first, or provide a valid path)")
            logger.error("Session file not found for /restore: %s", exc)
            print()
            return True
        except (ValueError, OSError) as exc:
            ui_error(f"Error restoring session: {exc}")
            logger.error("Failed to restore session: %s", exc)
            print()
            return True
        ui_success(
            f"Session restored from {report['path']} "
            f"({report['messages']} messages, {report['cache_entries']} "
            f"cache entries)."
        )
        logger.info(
            "User restored session from %s (%d messages, %d cache entries)",
            report["path"], report["messages"], report["cache_entries"],
        )
        print()
        return True

    # Not a recognised command - let it fall through to the LLM
    return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the interactive agent REPL."""
    # Coloured startup banner.
    ui_banner("Ollama tool-calling agent")
    print(c(f"Model  : {DEFAULT_MODEL}", "cyan"))
    print(c(f"Server : {OLLAMA_URL}", "cyan"))
    print(c(f"Tools  : {[t.name for t in _TOOL_REGISTRY.values()]}", "green"))
    print(c(f"Logfile: {os.path.abspath(LOG_FILE)}", "gray"))
    print()
    _print_help()

    # Ask the user where the diff tool is located (used by update_file).
    configure_diff_tool()

    logger.info("Agent started | model=%s | url=%s | diff_tool=%s | tools=%s",
                DEFAULT_MODEL, OLLAMA_URL, DIFF_TOOL_PATH,
                [t.name for t in _TOOL_REGISTRY.values()])

    # Make sure the server is reachable
    try:
        logger.debug("Probing Ollama at http://localhost:11434/api/tags")
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        logger.info("Ollama server reachable (status=%d)", r.status_code)
    except requests.exceptions.RequestException as exc:
        logger.critical("Could not reach Ollama at %s: %s", OLLAMA_URL, exc)
        raise SystemExit(f"Could not reach Ollama at {OLLAMA_URL}: {exc}\n"
                         f"Is 'ollama serve' running?")

    agent = agentTwo(model=DEFAULT_MODEL)

    # ---- Human-friendly prompt with tab completion -----------------------
    completer = AgentCompleter(extra_files=_local_files())
    # Hook the agent's status-bar builder up as the live ``prompt_toolkit``
    # bottom toolbar, so the bar is a persistent on-screen element while
    # the user is at the input prompt (model / state / temperature /
    # token usage / call & turn counts).  ``render_status_bar`` is a
    # cheap, side-effect-free callable, which is exactly what the toolbar
    # callback contract expects.  The plain ``input()`` fallback in
    # ``ColouredPrompt`` has no toolbar equivalent and silently ignores
    # the argument.
    prompt = ColouredPrompt(completer=completer,
                            bottom_toolbar=agent.render_status_bar)

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
        # Also accept bare 'help' / '?' (mirrors 'quit'/'exit'/'reset')
        if user_input.lower() in {"help", "?"}:
            _print_help()
            logger.info("User requested help via %r", user_input)
            continue
        # Interactive session commands (/temp, /max_iter, /think, /help, /?)
        if user_input.startswith("/") and _handle_command(user_input, agent):
            continue

        # Coloured "Agent> " prefix instead of the old inline print().
        print(c("Agent> ", "bold", "bright_yellow"), end="", flush=True)
        try:
            answer = agent.chat(user_input)
        except KeyboardInterrupt:
            # Defensive fallback: normally agentTwo.chat() already absorbs a
            # Ctrl-C raised during its call sequence and returns a message.
            # If a Ctrl-C ever slips through here (e.g. raised before the
            # chat loop's try block is entered), we still handle it so the
            # session never crashes - we just drop back to the prompt.
            logger.info("Chat interrupted by user (Ctrl-C) in main loop")
            agent.state = "idle"
            print()
            answer = "[interrupted by user - task stopped]"
        except requests.exceptions.RequestException as exc:
            logger.error("Connection error during chat: %s", exc)
            answer = f"[connection error: {exc}]"
        except Exception as exc:                                  # noqa: BLE001
            logger.exception("Unhandled exception during chat")
            answer = f"[error: {exc}]"
        # The agent finished its turn; go back to idle for the next prompt.
        agent.state = "idle"
        # Print a final status bar line after the answer so the latest
        # token totals / call counts remain visible above the next prompt.
        print(answer)
        agent.print_status_bar()
        print()
        logger.info("Printed agent answer to console (%d chars)", len(answer))

    logger.info("Agent session ended")