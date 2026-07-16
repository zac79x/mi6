"""Human-friendly interactive CLI helpers."""

from __future__ import annotations

import os
import sys
import threading
import time

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.styles import Style
    _HAS_PTK = True
except ImportError:
    _HAS_PTK = False


def _colours_supported() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if sys.platform.startswith("win"):
        return True
    if not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "")
    return term not in ("", "dumb", "unknown")


_COLOURS_ON = _colours_supported()

_ANSI = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m", "italic": "\033[3m",
    "underline": "\033[4m", "red": "\033[31m", "green": "\033[32m",
    "yellow": "\033[33m", "blue": "\033[34m", "magenta": "\033[35m",
    "cyan": "\033[36m", "gray": "\033[90m", "bright_red": "\033[91m",
    "bright_green": "\033[92m", "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m", "bright_magenta": "\033[95m", "bright_cyan": "\033[96m",
}


def c(text: str, *styles: str) -> str:
    if not _COLOURS_ON or not styles:
        return text
    prefix = "".join(_ANSI.get(s, "") for s in styles)
    return f"{prefix}{text}{_ANSI['reset']}" if prefix else text


def enable_colours(force: bool | None = None) -> None:
    global _COLOURS_ON
    _COLOURS_ON = _colours_supported() if force is None else bool(force)


def colours_enabled() -> bool:
    return _COLOURS_ON


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


_SPINNER_FRAMES = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]
_SPINNER_INTERVAL = 0.08
CR = "\r"
EL = "\033[K"


class Spinner:
    def __init__(self, text: str = "", elapsed: bool = False) -> None:
        self._text = text
        self._elapsed = elapsed
        self._thread: threading.Thread | None = None
        self._running = False
        self._frame_idx = 0
        self._start_time = 0.0

    def _spin(self) -> None:
        while self._running:
            frame = _SPINNER_FRAMES[self._frame_idx % len(_SPINNER_FRAMES)]
            self._frame_idx += 1
            label = self._text
            if self._elapsed:
                label = f"{self._text} ({time.time() - self._start_time:.1f}s)"
            sys.stderr.write(CR + "  " + c(frame, "cyan") + " " + label + EL)
            sys.stderr.flush()
            time.sleep(_SPINNER_INTERVAL)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._frame_idx = 0
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
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


PROMPT_TEXT = "You> "


class AgentCompleter(Completer if _HAS_PTK else object):
    SLASH_COMMANDS = [
        ("/help", "show this help message"),
        ("/?", "show this help message"),
        ("/quit", "end the session"),
        ("/exit", "end the session"),
        ("/reset", "clear the conversation history and result cache"),
        ("/temp", "set sampling temperature"),
        ("/max_iter", "set max tool-calling iterations"),
        ("/think", "enable/disable chain-of-thought display"),
        ("/kdiff", "set the kdiff3 binary path"),
        ("/compact", "compact the conversation history in place"),
        ("/listcache", "list cached tool-result refs and sizes"),
        ("/save", "save the current session to disk"),
        ("/restore", "restore a previously saved session"),
    ]

    def __init__(self, extra_files: list[str] | None = None) -> None:
        self.extra_files = extra_files or []

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        stripped = text.lstrip()
        if not stripped.startswith("/"):
            return
        parts = stripped.split()
        if len(parts) == 1 and not stripped.endswith(" "):
            word = parts[0]
            for cmd, desc in self.SLASH_COMMANDS:
                if cmd.startswith(word):
                    yield Completion(cmd, start_position=-len(word), display=cmd, display_meta=desc, style="bg:ansicyan fg:ansiblack")
            return
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
                yield Completion(cand, start_position=-len(arg), display=cand, style="bg:ansigreen fg:ansiblack")


class ColouredPrompt:
    def __init__(self, completer=None, prompt_text: str = PROMPT_TEXT, bottom_toolbar=None) -> None:
        self.prompt_text = prompt_text
        if _HAS_PTK:
            style = Style.from_dict({
                "prompt": "bold ansibrightgreen",
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
        try:
            if self._session is not None:
                return self._session.prompt()
            disp = c(self.prompt_text, "bold", "green") if colours_enabled() else self.prompt_text
            return input(disp)
        except (EOFError, KeyboardInterrupt):
            return None
