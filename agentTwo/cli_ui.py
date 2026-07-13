"""Human-friendly interactive CLI helpers for the ``agent`` package.

This is the package-side companion of the standalone ``cli_ui.py`` that
ships next to ``agentNew.py``.  It provides:

  * ANSI colour helpers (``c()``) with automatic, graceful degradation on
    non-TTY / dumb terminals, and respect for the ``NO_COLOR`` convention.
  * Semantic printers (``info`` / ``success`` / ``warn`` / ``error`` /
    ``dim`` / ``banner``) built on top of ``c()``.
  * A coloured, tab-completing input prompt backed by ``prompt_toolkit``
    when available, falling back to a simple ``input()`` wrapper otherwise.
  * :class:`AgentCompleter` - context-aware tab completion for the
    package's full set of slash commands (``/help``, ``/temp``,
    ``/max_iter``, ``/think``, ``/kdiff``, ``/compact``, ``/listcache``,
    ``/save``, ``/restore``) and their arguments.

Everything is dependency-free except the *optional* ``prompt_toolkit``
import.  When that library is missing the agent keeps working with a
plain but still colour-aware prompt.
"""

from __future__ import annotations

import os
import sys
import threading
import time

# --------------------------------------------------------------------------- #
# Optional prompt_toolkit import
# --------------------------------------------------------------------------- #
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import (
        Completer,
        Completion,
    )
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.styles import Style
    _HAS_PTK = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_PTK = False


# --------------------------------------------------------------------------- #
# Colour support detection
# --------------------------------------------------------------------------- #
def _colours_supported() -> bool:
    """Return True if the terminal most likely supports ANSI colours."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if sys.platform.startswith("win"):
        # Modern Windows terminals support ANSI; very old ones may not.
        return True
    if not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "")
    if term in ("", "dumb", "unknown"):
        return False
    return True


_COLOURS_ON = _colours_supported()


# --------------------------------------------------------------------------- #
# ANSI codes
# --------------------------------------------------------------------------- #
_ANSI = {
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "dim":     "\033[2m",
    "italic":  "\033[3m",
    "underline": "\033[4m",
    "red":     "\033[31m",
    "green":   "\033[32m",
    "yellow":  "\033[33m",
    "blue":    "\033[34m",
    "magenta": "\033[35m",
    "cyan":    "\033[36m",
    "gray":    "\033[90m",
    "bright_red":     "\033[91m",
    "bright_green":   "\033[92m",
    "bright_yellow":  "\033[93m",
    "bright_blue":     "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan":    "\033[96m",
}


def c(text: str, *styles: str) -> str:
    """Wrap *text* in the given ANSI *styles*.

    Examples::

        c("hello", "bold", "green")
        c("Error!", "red")

    If colour output is disabled (non-TTY, ``NO_COLOR`` set, dumb TERM),
    the text is returned unchanged.
    """
    if not _COLOURS_ON or not styles:
        return text
    prefix = "".join(_ANSI.get(s, "") for s in styles)
    if not prefix:
        return text
    return f"{prefix}{text}{_ANSI['reset']}"


def enable_colours(force: bool | None = None) -> None:
    """Turn colour support on or off globally.

    With *force=None* the value is auto-detected again from the environment.
    """
    global _COLOURS_ON
    if force is None:
        _COLOURS_ON = _colours_supported()
    else:
        _COLOURS_ON = bool(force)


def colours_enabled() -> bool:
    """Return whether coloured output is currently active."""
    return _COLOURS_ON


# --------------------------------------------------------------------------- #
# Semantic printers (thin wrappers around c())
# --------------------------------------------------------------------------- #
def info(msg: str) -> None:
    print(c(msg, "cyan"))


def success(msg: str) -> None:
    print(c(msg, "green"))


def warn(msg: str) -> None:
    print(c(msg, "yellow"))


def error(msg: str) -> None:
    print(c(msg, "bright_red"))


def dim(msg: str) -> None:
    print(c(msg, "gray", "dim"))


# --------------------------------------------------------------------------- #
# Animated spinner
# --------------------------------------------------------------------------- #

# Braille-based spinner frames -- compact, smooth, and works in any terminal.
_SPINNER_FRAMES = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c",
                   "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]
_SPINNER_INTERVAL = 0.08  # seconds between frame advances

CR = "\r"
EL = "\033[K"


class Spinner:
    """Animated braille spinner that runs in a background thread.

    Usage as a context manager::

        with Spinner("thinking...", elapsed=True):
            result = do_slow_work()
        # spinner is stopped and cleared automatically

    Or manually::

        sp = Spinner("loading")
        sp.start()
        ...
        sp.stop()

    When *elapsed* is True, a live timer ``(Ns)`` is appended to the
    text so the user can see how long the operation has been running.
    """

    def __init__(self, text: str = "", elapsed: bool = False) -> None:
        self._text = text
        self._elapsed = elapsed
        self._thread: threading.Thread | None = None
        self._running = False
        self._frame_idx = 0
        self._start_time = 0.0

    def _spin(self) -> None:
        """Background thread: write a frame, sleep, advance, repeat."""
        while self._running:
            frame = _SPINNER_FRAMES[self._frame_idx % len(_SPINNER_FRAMES)]
            self._frame_idx += 1
            label = self._text
            if self._elapsed:
                elapsed_s = time.time() - self._start_time
                label = f"{self._text} ({elapsed_s:.1f}s)"
            line = CR + "  " + c(frame, "cyan") + " " + label
            sys.stderr.write(line + EL)
            sys.stderr.flush()
            time.sleep(_SPINNER_INTERVAL)

    def start(self) -> None:
        """Begin animating the spinner on stderr."""
        if self._running:
            return
        self._running = True
        self._frame_idx = 0
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the spinner and clear the line."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        sys.stderr.write(CR + EL)
        sys.stderr.flush()

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


def banner(msg: str, char: str = "=") -> None:
    print(c(char * 60, "bright_blue"))
    print(c(msg, "bold", "bright_blue"))
    print(c(char * 60, "bright_blue"))


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
# Consistent, colour-aware prompt string used by both backends.
PROMPT_TEXT = "You> "


class AgentCompleter(Completer if _HAS_PTK else object):  # type: ignore[misc]
    """Context-aware tab completer for the agent REPL.

    Completion candidates depend on what the user has typed so far:

    * Leading ``/``  -> the agent's slash commands plus argument completion
      for the commands that take one (``/think``, ``/kdiff``, ``/save``,
      ``/restore``).
    * Otherwise      -> nothing (free-form natural-language input is
      intentionally left alone so it doesn't fight the user).
    """

    # (command, short description) - shown as the meta line in the
    # completion menu.  Kept in sync with the commands handled in
    # ``agent.repl._handle_command`` and the main loop in ``main()``.
    SLASH_COMMANDS = [
        ("/help",      "show this help message"),
        ("/?",         "show this help message"),
        ("/quit",      "end the session"),
        ("/exit",      "end the session"),
        ("/reset",     "clear the conversation history and result cache"),
        ("/temp",      "set sampling temperature"),
        ("/max_iter",  "set max tool-calling iterations"),
        ("/think",     "enable/disable chain-of-thought display"),
        ("/kdiff",     "set the kdiff3 binary path"),
        ("/compact",   "compact the conversation history in place"),
        ("/listcache", "list cached tool-result refs and sizes"),
        ("/save",      "save the current session to disk"),
        ("/restore",   "restore a previously saved session"),
    ]

    def __init__(self, extra_files: list[str] | None = None) -> None:
        # Local files used for path-argument completion (/kdiff wants .py,
        # /save & /restore want .json).
        self.extra_files = extra_files or []

    def get_completions(self, document, complete_event):  # noqa: D401
        text = document.text_before_cursor
        stripped = text.lstrip()
        if not stripped.startswith("/"):
            return
        parts = stripped.split()
        # Completing the first token (the command name).
        if len(parts) == 1 and not stripped.endswith(" "):
            word = parts[0]
            for cmd, desc in self.SLASH_COMMANDS:
                if cmd.startswith(word):
                    yield Completion(
                        cmd,
                        start_position=-len(word),
                        display=cmd,
                        display_meta=desc,
                        style="bg:ansicyan fg:ansiblack",
                    )
            return
        # Completing an argument.
        head = parts[0].lower()
        arg = parts[-1] if len(parts) > 1 else ""
        candidates: list[str] = []
        if head == "/kdiff":
            candidates = [f for f in self.extra_files if f.endswith(".py")]
        elif head in ("/save", "/restore"):
            candidates = [f for f in self.extra_files if f.endswith(".json")]
        elif head == "/think":
            candidates = ["on", "off", "yes", "no", "true", "false", "1", "0"]
        for cand in candidates:
            if cand.startswith(arg):
                yield Completion(
                    cand,
                    start_position=-len(arg),
                    display=cand,
                    style="bg:ansigreen fg:ansiblack",
                )


class ColouredPrompt:
    """High-level prompt wrapper with a prompt_toolkit backend and an
    ``input()`` fallback.

    Usage::

        prompt = ColouredPrompt(completer=AgentCompleter([...]))
        while True:
            line = prompt.read()
            if line is None:
                break

    ``bottom_toolbar`` optionally takes a zero-argument callable (or
    plain string) that ``prompt_toolkit`` renders as a live,
    persistent status line underneath the input area while the prompt
    is waiting for input.  This is how the :class:`~agent.agent.Agent`
    status bar (``Agent.render_status_bar``) is displayed as a real
    on-screen UI element rather than as transient printed lines.  The
    toolbar is only shown by the ``prompt_toolkit`` backend; the plain
    ``input()`` fallback has no equivalent and silently ignores it.
    """

    def __init__(self, completer=None, prompt_text: str = PROMPT_TEXT,
                 bottom_toolbar=None) -> None:
        self.prompt_text = prompt_text
        if _HAS_PTK:
            style = Style.from_dict({
                "prompt": "bold ansibrightgreen",
                # Bottom toolbar (status bar): dim gray text on a black
                # background so it reads as a quiet info line under the
                # input area, matching the dim style used by
                # ``Agent.print_status_bar`` during the response phase.
                "bottom-toolbar": "bg:ansiblack fg:ansigray",
            })
            formatted_prompt = FormattedText([("class:prompt", prompt_text)])
            self._session = PromptSession(
                message=formatted_prompt,
                style=style,
                completer=completer,
                complete_while_typing=True,
                bottom_toolbar=bottom_toolbar,
            )
        else:
            self._session = None

    def read(self) -> str | None:
        """Read one line.

        Returns the input (not stripped), or ``None`` on EOF / Ctrl-C.
        """
        try:
            if self._session is not None:
                text = self._session.prompt()
            else:
                # Coloured, plain-input fallback.
                disp = (c(self.prompt_text, "bold", "green")
                        if colours_enabled() else self.prompt_text)
                text = input(disp)
        except (EOFError, KeyboardInterrupt):
            return None
        return text