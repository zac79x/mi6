"""
Created on Thu Jul  9 22:47:43 2026

@author: HP
"""

from __future__ import annotations

import ast
import difflib
import inspect
import json
import logging
import operator
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable
from pathlib import Path
import py_compile

try:
    import requests
except ImportError:
    raise SystemExit("Please install the 'requests' library:  pip install requests")

# Human-friendly CLI helpers: ANSI colours (graceful degradation) and an
# optional prompt_toolkit-backed prompt with tab completion.  The import is
# soft: if `cli_ui` is somehow missing the agent still runs with plain I/O.
try:
    from cli_ui import (
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
except ImportError:  # pragma: no cover - cli_ui should always be present
    # Minimal fallbacks so the agent keeps working without cli_ui.
    def c(text, *styles):
        return text
    def colours_enabled():
        return False
    def enable_colours(force=None):
        pass
    def ui_info(msg):
        print(msg)
    def ui_success(msg):
        print(msg)
    def ui_warn(msg):
        print(msg)
    def ui_error(msg):
        print(msg)
    def ui_dim(msg):
        print(msg)
    def ui_banner(msg, char="="):
        print(char * 60)
        print(msg)
        print(char * 60)
    class AgentCompleter:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            pass
    class ColouredPrompt:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            pass
        def read(self):
            try:
                return input("You> ")
            except (EOFError, KeyboardInterrupt):
                return None


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_FILE   = "agent.log"

# Root logger for the agent
logger = logging.getLogger("agent")
logger.setLevel(logging.DEBUG)  # capture everything; handlers can filter

# Reset any existing handlers (useful when re-importing in notebooks/tests)
logger.handlers.clear()
logger.propagate = False


# File handler (DEBUG and above - full detail)
try:
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
except OSError as e:
    logger.warning("Could not open log file %s: %s", LOG_FILE, e)

# A sub-logger for HTTP traffic (DEBUG only)
http_logger = logging.getLogger("ollama_agent.http")
http_logger.setLevel(logging.DEBUG)

# Dedicated logger for the JSON payload sent to the LLM (so it can be
# filtered/tail-ed easily in agent.log)
llm_payload_logger = logging.getLogger("ollama_agent.llm_payload")
llm_payload_logger.setLevel(logging.DEBUG)


def _truncate(text: str, limit: int = 200) -> str:
    """Shorten long strings for compact log lines."""
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"... [truncated, total {len(text)} chars]"


def _safe_json(obj: Any) -> str:
    """Pretty-print an object as JSON, falling back to repr on failure."""
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(obj)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL  = "http://localhost:11434/api/chat"
#DEFAULT_MODEL = "gemma4:e4b"   # any tool-capable Ollama model works
#DEFAULT_MODEL = "qwen3.5:2b"
DEFAULT_MODEL = "minimax-m3:cloud"


# ---------------------------------------------------------------------------
# Diff tool configuration
# ---------------------------------------------------------------------------
# Path to a diff/merge tool used by the `update_file` tool to show the user
# the proposed changes BEFORE they are applied to a file. Examples:
#   * "kdiff3"                    (if it is on PATH)
#   * "kdiff"                     (older alias for kdiff3)
#   * "C:/Program Files/KDiff3/kdiff3.exe"
# This value is populated at startup by `configure_diff_tool()` and is
# then consulted by `_run_diff_tool()`.
DIFF_TOOL_PATH: str | None = None


# ===========================================================================
# Tool registry
# ===========================================================================

# Python type -> JSON Schema type
_TYPE_MAP: dict[type, str] = {
    int:   "integer",
    float: "number",
    str:   "string",
    bool:  "boolean",
    list:  "array",
    dict:  "object",
}


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters: dict

    def to_ollama_schema(self) -> dict:
        """Convert to the schema format expected by Ollama's /api/chat."""
        return {
            "type": "function",
            "function": {
                "name":        self.name,
                "description": self.description,
                "parameters":  self.parameters,
            },
        }


_TOOL_REGISTRY: dict[str, Tool] = {}


def _first_paragraph(doc: str | None) -> str:
    """Use the function's docstring (up to the first blank line) as description."""
    if not doc:
        return ""
    lines = [ln.strip() for ln in doc.strip().splitlines() if ln.strip()]
    return " ".join(lines)


def _schema_from_signature(func: Callable) -> dict:
    """Build a minimal JSON schema from the function's signature & type hints."""
    sig = inspect.signature(func)
    properties: dict[str, dict] = {}
    required:   list[str]       = []

    for name, param in sig.parameters.items():
        ann = param.annotation
        json_type = _TYPE_MAP.get(ann, "string") if ann is not inspect.Parameter.empty else "string"
        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def tool(
    _func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict | None = None,
) -> Callable:
    """Decorator that registers a function as a callable tool for the agent.

    Usage:
        @tool
        def add(a: int, b: int) -> int:
            \"\"\"Add two integers.\"\"\"
            return a + b

        @tool(description="Read a file", parameters={...})
        def read_file(path: str) -> str: ...
    """
    def wrap(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or _first_paragraph(func.__doc__)
        tool_params = parameters or _schema_from_signature(func)
        _TOOL_REGISTRY[tool_name] = Tool(tool_name, tool_desc, func, tool_params)
        logger.debug("Registered tool: %s | description=%r | params=%s",
                     tool_name, tool_desc, tool_params)
        return func

    # Support both @tool and @tool(...)
    if _func is not None and callable(_func):
        return wrap(_func)
    return wrap


# ===========================================================================
# Example tools
# ===========================================================================

# A safe calculator that only accepts numeric expressions
_SAFE_OPS: dict[type, Any] = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise ValueError("only numeric literals are allowed")
        if isinstance(node, ast.BinOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"operator {type(node.op).__name__} not allowed")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"unary operator {type(node.op).__name__} not allowed")
            return op(_eval(node.operand))
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    return _eval(tree)


@tool
def calculate(expression: str) -> str:
    """Evaluate a safe arithmetic expression like '2 * (3 + 4)'."""
    try:
        return str(_safe_eval(expression))
    except Exception as e:                       # noqa: BLE001
        return f"Error: {e}"


@tool
def get_current_time() -> str:
    """Return the current local date and time."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


@tool
def read_text_file(path: str, max_chars: int = 40000) -> str:
    """Read a UTF-8 text file. Truncates very long files to max_chars."""
    if not os.path.exists(path):
        return f"Error: file '{path}' does not exist"
    try:
        try:
            max_chars = int(max_chars) if max_chars is not None else None
        except (TypeError, ValueError):
            max_chars = None
        with open(path, "r", encoding="utf-8") as fh:
            data = fh.read(max_chars + 1)
        if len(data) > max_chars:
            data = data[:max_chars] + f"\n... [truncated, file is longer than {max_chars} chars]"
        return data
    except Exception as e:                       # noqa: BLE001
        return f"Error reading file: {e}"

@tool
def read_file_lines(path: str, start_line: int = 0, num_lines: int = 100) -> str:
    """Read a slice of lines from a text file. Useful for inspecting large files without loading the whole
thing."""
    import os
    if not os.path.exists(path):
        return f"Error: file '{path}' does not exist"
    if not os.path.isfile(path):
        return f"Error: path '{path}' is not a regular file"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            try:
                start_line = int(start_line) if start_line is not None else None
            except (TypeError, ValueError):
                start_line = None
            try:
                num_lines = int(num_lines) if num_lines is not None else None
            except (TypeError, ValueError):
                num_lines = None
            for _ in range(max(0, start_line)):
                next(fh, None)
            chunk = []
            for _ in range(max(0, num_lines)):
                line = fh.readline()
                if not line:
                    break
                chunk.append(line)
        if not chunk:
            return f"No content read from '{path}' (start_line={start_line}, num_lines={num_lines})."
        header = f"Lines {start_line}..{start_line + len(chunk) - 1} of '{path}':\n"
        return header + "".join(chunk)
    except UnicodeDecodeError:
        return f"Error: '{path}' is not a valid UTF-8 text file"
    except PermissionError:
        return f"Error: permission denied to read '{path}'"
    except OSError as e:
        return f"Error reading file: {e}"

@tool
def list_directory(path: str = ".", show_hidden: bool = False) -> str:
    """List files and directories in the given path. Returns a formatted listing with sizes and types."""
    import os

    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist."

    if not os.path.isdir(path):
        return f"Error: Path '{path}' is not a directory."

    try:
        entries = os.listdir(path)
        if not show_hidden:
            entries = [e for e in entries if not e.startswith(".")]

        if not entries:
            return f"Directory '{path}' is empty."

        entries.sort(key=str.lower)

        lines = [f"Contents of '{os.path.abspath(path)}':"]
        for entry in entries:
            full_path = os.path.join(path, entry)
            try:
                if os.path.isdir(full_path):
                    lines.append(f"  [DIR]  {entry}/")
                elif os.path.isfile(full_path):
                    size = os.path.getsize(full_path)
                    lines.append(f"  [FILE] {entry}  ({size:,} bytes)")
                elif os.path.islink(full_path):
                    lines.append(f"  [LINK] {entry}")
                else:
                    lines.append(f"  [???]  {entry}")
            except OSError:
                lines.append(f"  [???]  {entry}  (unreadable)")

        return "\n".join(lines)

    except PermissionError:
        return f"Error: Permission denied to read '{path}'."
    except OSError as e:
        return f"Error listing directory '{path}': {e}"


@tool
def path_exists(path: str) -> str:
    """Check whether a path exists and report what kind of filesystem object it is (file, directory, or
neither)."""
    import os
    if not os.path.exists(path):
        return f"Path '{path}' does not exist."
    if os.path.isdir(path):
        return f"Path '{path}' exists and is a directory."
    if os.path.isfile(path):
        size = os.path.getsize(path)
        return f"Path '{path}' exists and is a file ({size:,} bytes)."
    return f"Path '{path}' exists but is neither a regular file nor a directory."

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

WORKSPACE_ROOT = Path(os.getcwd()).resolve()
ALLOWED_EXTENSION = ".py"


# --------------------------------------------------------------------------- #
# Safety helpers
# --------------------------------------------------------------------------- #

def _validate_path(path: str) -> Path:
    """Resolve `path` and confirm it stays inside WORKSPACE_ROOT.

    Raises:
        ValueError: if the path is outside the workspace, is not a `.py`
                    file, or points to an existing non-file (e.g. a directory).
    """
    if not path.endswith(ALLOWED_EXTENSION):
        raise ValueError(
            f"Refused: only {ALLOWED_EXTENSION!r} files are allowed, got {path!r}."
        )

    candidate = (WORKSPACE_ROOT / path).resolve()
    try:
        candidate.relative_to(WORKSPACE_ROOT)
    except ValueError:
        raise ValueError(
            f"Refused: {path!r} resolves outside the workspace {WORKSPACE_ROOT}."
        )

    if candidate.exists() and not candidate.is_file():
        raise ValueError(f"Refused: {candidate} exists and is not a regular file.")

    return candidate


def _content_looks_partial(original: str, content: str) -> str | None:
    """Heuristically detect whether `content` is a full file or just the
    changed hunks.

    Returns a human-readable explanation when `content` looks partial
    (only the diff), otherwise None.  Conservative: only flags content
    that is BOTH much smaller than the original AND missing most of the
    original's non-blank lines, so legitimate edits and most full
    rewrites are not blocked.
    """
    if not original.strip() or not content.strip():
        return None
    old_lines = [ln for ln in original.splitlines() if ln.strip()]
    new_lines = [ln for ln in content.splitlines() if ln.strip()]
    if len(old_lines) < 20 or not new_lines:
        return None
    # Half 1: the new content is much smaller than the original.
    if len(new_lines) >= len(old_lines) * 0.5:
        return None
    # Half 2: most of the original's lines are gone. A real full-file
    # edit keeps the unchanged lines; a "diff hunks only" payload drops them.
    old_set = set(old_lines)
    preserved = len(old_set & set(new_lines)) / len(old_set)
    if preserved >= 0.5:
        return None
    return (
        f"The proposed content has {len(new_lines)} non-blank lines versus "
        f"{len(old_lines)} in the original and only {preserved:.0%} of the "
        "original's lines are still present. `update_file` requires the "
        "COMPLETE file contents, not just the changed lines. Please re-read "
        "the file with `read_text_file` and resend the entire file with your "
        "changes applied."
    )


# --------------------------------------------------------------------------- #
# Diff tool configuration
# --------------------------------------------------------------------------- #

def _read_kdiff3_path_from_env(env_path: str | Path = ".env") -> str | None:
    """Read the Kdiff3 path from a `.env` file, if present.

    Looks for an entry like:
        KDIFF3_PATH=C:/Program Files/KDiff3/kdiff3.exe
    Blank lines and lines starting with '#' (comments) are ignored.
    Surrounding single or double quotes around the value are stripped.

    Args:
        env_path: Path to the `.env` file. Defaults to ".env" in the
            current working directory.

    Returns:
        The path string if a non-commented, non-empty KDIFF3_PATH entry
        is found, otherwise None. Returns None silently if the file does
        not exist or cannot be read.
    """
    try:
        env_file = Path(env_path)
        if not env_file.is_file():
            return None
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip().upper() == "KDIFF3_PATH":
                v = value.strip().strip('"').strip("'")
                if v:
                    logger.debug("Found KDIFF3_PATH in %s: %s", env_path, v)
                    return v
                return None
    except OSError as e:
        logger.debug("Could not read .env file %s: %s", env_path, e)
    return None


def _format_env_value(value: str) -> str:
    """Format a value for safe inclusion in a `.env` file.

    Double-quotes the value when it contains whitespace, '#', or quote
    characters, escaping any embedded double-quotes. Values that do not
    need quoting are written bare (this matches the existing entries in
    the project's `.env`, e.g. ``KDIFF3_PATH=C:/Program Files/KDiff3/...``
    which is the only one that actually needs quoting and is currently
    stored unquoted - we keep the same style for consistency).
    """
    v = value.strip()
    if v == "":
        return ""
    needs_quoting = any(ch.isspace() for ch in v) or any(ch in v for ch in '"#')
    if not needs_quoting:
        return v
    escaped = v.replace('"', '\\"')
    return f'"{escaped}"'


def _write_kdiff3_path_to_env(
    value: str,
    env_path: str | Path = ".env",
    *,
    overwrite: bool = True,
) -> dict:
    """Write the Kdiff3 path to a `.env` file.

    Behaviour:
        * If the file does not exist, it is created with the single
          ``KDIFF3_PATH=<value>`` line.
        * If the file exists and contains a ``KDIFF3_PATH=...`` line
          (case-insensitive, ignoring comments), that line is replaced
          in place while every other line is preserved exactly as is
          (comments and blank lines included).
        * If the file exists and contains no ``KDIFF3_PATH`` entry, the
          new line is appended at the end.
        * The value is written using ``_format_env_value`` so that
          paths containing spaces are safely quoted.

    Args:
        value:     The new kdiff3 path to persist.
        env_path:  Path to the `.env` file (default: ``.env`` in cwd).
        overwrite: If True (default) and a previous ``KDIFF3_PATH``
                   entry exists, it is replaced. If False and a previous
                   entry exists, the call is a no-op and a `reason` is
                   returned explaining what happened.

    Returns:
        A dict with keys:
            ok     (bool)  - True on success.
            action (str)   - "created" | "updated" | "appended" | "kept" |
                             "refused".
            path   (str)   - The env_path used, as a string.
            reason (str|None) - Human-readable explanation on failure
                                (None on success).
    """
    if value is None or str(value).strip() == "":
        return {
            "ok": False,
            "action": "refused",
            "path": str(env_path),
            "reason": "Refused to write an empty KDIFF3_PATH value.",
        }

    new_value = str(value).strip()
    env_file = Path(env_path)
    formatted = f"KDIFF3_PATH={_format_env_value(new_value)}"

    # Make sure the parent directory exists (no-op for ".").
    try:
        if env_file.parent and str(env_file.parent) not in ("", "."):
            env_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return {
            "ok": False,
            "action": "refused",
            "path": str(env_path),
            "reason": f"Could not create parent directory: {e}",
        }

    if not env_file.is_file():
        try:
            env_file.write_text(formatted + "\n", encoding="utf-8")
            logger.info("Created .env with KDIFF3_PATH=%s", new_value)
            return {"ok": True, "action": "created", "path": str(env_path), "reason": None}
        except OSError as e:
            return {
                "ok": False,
                "action": "refused",
                "path": str(env_path),
                "reason": f"Could not create {env_path}: {e}",
            }

    # Read existing content and look for a KDIFF3_PATH line.
    try:
        original_text = env_file.read_text(encoding="utf-8")
    except OSError as e:
        return {
            "ok": False,
            "action": "refused",
            "path": str(env_path),
            "reason": f"Could not read {env_path}: {e}",
        }

    original_lines = original_text.splitlines()
    had_trailing_newline = original_text.endswith("\n")

    replaced = False
    new_lines: list[str] = []
    for raw in original_lines:
        stripped = raw.lstrip()
        is_kdiff_line = False
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, _, _ = stripped.partition("=")
            if key.strip().upper() == "KDIFF3_PATH":
                is_kdiff_line = True
        if is_kdiff_line:
            if not overwrite:
                logger.info("KDIFF3_PATH already present in %s; not overwriting",
                            env_path)
                return {
                    "ok": True,
                    "action": "kept",
                    "path": str(env_path),
                    "reason": "KDIFF3_PATH already present (overwrite=False).",
                }
            new_lines.append(formatted)
            replaced = True
        else:
            new_lines.append(raw)

    if not replaced:
        new_lines.append(formatted)

    new_text = "\n".join(new_lines)
    if had_trailing_newline or new_lines:
        new_text += "\n"

    try:
        env_file.write_text(new_text, encoding="utf-8")
    except OSError as e:
        return {
            "ok": False,
            "action": "refused",
            "path": str(env_path),
            "reason": f"Could not write {env_path}: {e}",
        }

    action = "updated" if replaced else "appended"
    logger.info("Wrote KDIFF3_PATH to %s (action=%s, value=%s)",
                env_path, action, new_value)
    return {"ok": True, "action": action, "path": str(env_path), "reason": None}


def _set_kdiff3_path_interactively(parts: list[str], env_path: str | Path = ".env") -> bool:
    """Handle the ``/kdiff`` interactive command.

    With no argument, the user is prompted for a path. With one argument,
    that argument is used as the new path directly.

    The new path is:
        * persisted to ``env_path`` (``KDIFF3_PATH=...``),
        * loaded back via :func:`_read_kdiff3_path_from_env` to confirm
          it round-trips correctly,
        * stored in the module-level :data:`DIFF_TOOL_PATH` so the next
          :func:`update_file` call uses it immediately.

    Returns True if the command was handled (the main loop should NOT
    forward it to the LLM).
    """
    global DIFF_TOOL_PATH

    if len(parts) == 1:
        try:
            answer = input("Path to kdiff3 binary: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNo interactive stdin available - /kdiff cancelled.")
            return True
        if not answer:
            print("No path entered - /kdiff cancelled.\n")
            return True
        new_path = answer
    elif len(parts) == 2:
        new_path = parts[1].strip()
        if not new_path:
            print("Error: empty path. Usage: /kdiff [<path>]\n")
            return True
    else:
        print("Usage: /kdiff [<path to kdiff3 binary>]\n")
        return True

    print(f"Persisting KDIFF3_PATH={new_path!r} to {env_path} ...")
    result = _write_kdiff3_path_to_env(new_path, env_path=env_path)

    if not result.get("ok"):
        print(f"Error: {result.get('reason', 'unknown error')}\n")
        logger.error("/kdiff failed: %s", result)
        return True

    # Confirm it round-trips through the reader.
    verified = _read_kdiff3_path_from_env(env_path)
    if verified != new_path:
        print("Warning: wrote KDIFF3_PATH but could not read it back exactly. "
              f"Read: {verified!r}, Wrote: {new_path!r}\n")
        logger.warning("KDIFF3_PATH round-trip mismatch: wrote=%r read=%r",
                       new_path, verified)
    else:
        print(f"Verified KDIFF3_PATH in {env_path}: {verified}")

    # Activate the new value for the rest of this session.
    DIFF_TOOL_PATH = new_path
    print(f"[diff tool set to {DIFF_TOOL_PATH}]\n")
    logger.info("User set KDIFF3_PATH via /kdiff: %s (file action=%s)",
                DIFF_TOOL_PATH, result.get("action"))
    return True


def configure_diff_tool() -> str:
    """Ask the user where the diff tool is located and store it in DIFF_TOOL_PATH.

    Called once at startup (in `main()`). The chosen path is used by the
    `update_file` tool to show the user the proposed changes before they
    are applied to a file.

    The default value is read from the `.env` file (key `KDIFF3_PATH`).
    If that entry is present and non-empty, it is used as the default.
    Else it ia asked from user.
    
    Returns:
        The configured diff-tool path (also stored in module-level
        DIFF_TOOL_PATH so that `_run_diff_tool` can find it).
    """
    global DIFF_TOOL_PATH
    env_default = _read_kdiff3_path_from_env()
    print("=" * 60)
    print("DIFF TOOL CONFIGURATION")
    print("-" * 60)
    print("The 'update_file' tool uses an external diff/merge program to show")
    print("you the proposed changes to a file BEFORE they are written.")
    if env_default:
        print(f"Detected KDIFF3_PATH in .env: {env_default}")
        DIFF_TOOL_PATH = env_default
    else:
        print("No KDIFF3_PATH found in .env.")
        prompt = "Path to diff tool: "
        try:
            answer = input(prompt).strip()
            DIFF_TOOL_PATH = answer
        except (EOFError, KeyboardInterrupt):
            print("\nNo interactive stdin available.")
            DIFF_TOOL_PATH = "kdiff3"   
    print("=" * 60)
    logger.info("Diff tool configured: %s", DIFF_TOOL_PATH)
    return DIFF_TOOL_PATH


def _print_text_diff(original: str, modified: str) -> bool:
    """Show a unified text diff between two files using `difflib`.

    Used as a fallback when no diff tool is configured or when the
    configured tool cannot be found.

    Returns True on success, False on failure.
    """
    try:
        with open(original, "r", encoding="utf-8") as fh:
            old_lines = fh.readlines()
        with open(modified, "r", encoding="utf-8") as fh:
            new_lines = fh.readlines()
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=original,
            tofile=modified,
            n=3,
        )
        for line in diff:
            print(line, end="")
        return True
    except Exception as e:                       # noqa: BLE001
        print(f"Error producing text diff: {e}")
        return False


def _run_diff_tool(original: str, modified: str) -> bool:
    """Run the configured diff tool to compare `original` and `modified`.

    If no diff tool is configured, falls back to a unified text diff printed
    to the terminal (using the standard `difflib` module) via
    `_print_text_diff`.

    Args:
        original: Path to the existing file.
        modified: Path to the temporary file containing the proposed content.

    Returns:
        True  - the diff was shown successfully (the user has been able to
                see it; whether they approve is asked separately).
        False - the diff tool could not be run at all.
    """
    if not DIFF_TOOL_PATH:
        return _print_text_diff(original, modified)

    try:
        # `kdiff3 original modified` opens a GUI; the process returns when
        # the user closes the window. We pass the existing file first so the
        # left-hand pane shows the old content and the right-hand pane shows
        # the proposed new content.
        result = subprocess.run([DIFF_TOOL_PATH, original, modified])
        logger.info("Diff tool '%s' exited with code %d",
                    DIFF_TOOL_PATH, result.returncode)
        return True
    except FileNotFoundError:
        print(f"Error: diff tool not found at {DIFF_TOOL_PATH!r}.")
        print("Falling back to text diff.")
        return _print_text_diff(original, modified)
    except Exception as e:                       # noqa: BLE001
        print(f"Error running diff tool {DIFF_TOOL_PATH!r}: {e}")
        return _print_text_diff(original, modified)


def _request_update_approval(path: str) -> bool:
    """Ask the user in the terminal whether to apply the proposed changes.

    Called after the diff tool has been shown. Returns True only if the
    user explicitly answers 'y' or 'yes'. Anything else (including EOF
    or KeyboardInterrupt) is treated as "no".
    """
    try:
        answer = input(f"Apply changes to {path}? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nNo interactive stdin available - refusing by default.")
        return False
    return answer in {"y", "yes"}


def _request_continue_approval(max_iterations: int) -> bool:
    """Ask the user whether to keep going after max_iterations was reached.

    Called from `Agent.chat` when the inner iteration loop completes
    without a final answer. Returns True only if the user explicitly
    answers 'y' or 'yes'. Anything else (including EOF or
    KeyboardInterrupt) is treated as "no" so the chat call returns
    with the standard "iteration limit" message.

    Args:
        max_iterations: The current value of `Agent.max_iterations`,
            used purely for display in the prompt.

    Returns:
        True if the agent should run another batch of iterations,
        False otherwise.
    """
    try:
        answer = input(
            f"Reached max_iterations={max_iterations} without a final answer. "
            f"Continue for another {max_iterations} iterations? [y/N]: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nNo interactive stdin available - stopping by default.")
        return False
    return answer in {"y", "yes"}


# --------------------------------------------------------------------------- #
# create_file tool  -  no user approval required
# --------------------------------------------------------------------------- #
@tool
def create_file(path: str, content: str) -> dict:
    """Create a NEW Python file. Does NOT require user approval.

    Refuses to overwrite an existing file - use `update_file` for that.

    Args:
        path:    Path relative to the workspace, e.g. "my_module.py".
        content: Full contents to write to the file.

    Returns:
        A dict with keys: ok (bool), action ("create"|"refused"),
        path (str), reason (str|None).
    """
    try:
        target = _validate_path(path)
    except ValueError as exc:
        return {"ok": False, "action": "refused", "path": path, "reason": str(exc)}

    if target.exists():
        return {
            "ok": False,
            "action": "refused",
            "path": path,
            "reason": "File already exists; use update_file to modify it.",
        }

    try:
        target.write_text(content, encoding="utf-8")
        logger.info("Created file: %s (%d bytes)", path, len(content))
        return {
            "ok": True,
            "action": "create",
            "path": path,
            "reason": None,
        }
    except OSError as e:
        return {"ok": False, "action": "refused", "path": path, "reason": str(e)}


# --------------------------------------------------------------------------- #
# update_file tool  -  diff-then-approve workflow using the diff tool
# --------------------------------------------------------------------------- #
@tool
def update_file(path: str, content: str) -> dict:
    """Update an EXISTING Python file after showing the user a diff.

    Workflow:
        1. Validate that `path` is a .py file inside the workspace and
           that the file already exists.
        2. Guard against partial content: heuristically detect whether
           `content` is just the changed hunks rather than the full
           file and, if so, refuse before the diff tool is ever launched.
           The refusal message is returned to the model so it can
           re-read the file and resend the complete contents.
        3. Save a backup of the original file's content in memory.
        4. Write the proposed `content` to a temporary file next to the
           original. The temporary file always contains the complete script,
           not just the changes.
        5. Launch the configured diff tool (see `configure_diff_tool()`)
           to compare the existing file with the temporary file so the
           user can review the changes visually.
        6. After the diff tool returns, restore the original file from the
           backup.  This is necessary because some diff/merge tools (e.g.
           kdiff3 in merge mode) may write their merge result directly to
           the original file when the user saves inside the tool, which
           would otherwise leave the file containing only the changed
           hunks instead of the full content.
        7. Ask the user in the terminal to confirm the change.
        8. If approved, overwrite the original file with the new content.
           The temporary file is always cleaned up.

    Args:
        path:    Path relative to the workspace, e.g. "agentNew.py".
        content: Full new contents for the file.

    Returns:
        A dict with keys: ok (bool), action ("update"|"refused"),
        path (str), reason (str|None).
    """
    try:
        target = _validate_path(path)
    except ValueError as exc:
        return {"ok": False, "action": "refused", "path": path, "reason": str(exc)}

    if not target.exists():
        return {
            "ok": False,
            "action": "refused",
            "path": path,
            "reason": "File does not exist; use create_file to create a new file.",
        }

    # Guard against the model sending only the changed hunks instead of the
    # full file.  Without this, the diff tool shows a misleading "new" side
    # and, on approval, the original code would be replaced by the partial
    # content - losing most of the file.
    original_content_for_check = target.read_text(encoding="utf-8")
    partial_warning = _content_looks_partial(original_content_for_check, content)
    if partial_warning is not None:
        logger.warning("update_file refused partial content for %s: %s",
                       path, partial_warning)
        print("=" * 60)
        print(f"REFUSED PARTIAL UPDATE: {path}")
        print(partial_warning)
        print("=" * 60)
        return {
            "ok": False,
            "action": "refused",
            "path": path,
            "reason": partial_warning,
        }

    # Write the proposed content to a temp file next to the original so the
    # diff tool can render proper filenames in its window.
    original_path = str(target)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".py.tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_fh:
            tmp_fh.write(content)

        # Back up the original content before launching the diff tool.
        # Some diff/merge tools (e.g. kdiff3 in merge mode) may write their
        # merge result directly to the original file when the user saves
        # inside the tool.  Without this backup we would lose the original
        # content and the file would end up containing only the changed
        # hunks.  We restore from this backup after the diff tool returns.
        original_content = target.read_text(encoding="utf-8")

        print("=" * 60)
        print(f"PROPOSED UPDATE: {path}")
        print(f"Temp file:       {tmp_path}")
        print("Launching diff tool for review - close it when you are done...")
        print("=" * 60)

        if not _run_diff_tool(original_path, tmp_path):
            # Restore the original in case the diff tool modified it.
            target.write_text(original_content, encoding="utf-8")
            return {
                "ok": False,
                "action": "refused",
                "path": path,
                "reason": "Diff tool failed; not applying changes.",
            }

        # Always restore the original content after the diff tool returns.
        # The diff tool may have written its merge result to the original
        # file; we undo that here so the file is in a known-good state
        # before we ask for approval.
        target.write_text(original_content, encoding="utf-8")

        if not _request_update_approval(path):
            return {
                "ok": False,
                "action": "refused",
                "path": path,
                "reason": "User did not approve the change.",
            }

        target.write_text(content, encoding="utf-8")
        logger.info("Updated file: %s (%d bytes)", path, len(content))
        return {
            "ok": True,
            "action": "update",
            "path": path,
            "reason": None,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            # Best-effort cleanup; ignore failures (e.g. on Windows the
            # diff tool may briefly hold the file).
            pass


@tool
def compile_python_file(path: str) -> str:
    """Compile a Python script to check for syntax errors without executing it.

    Uses py_compile to parse the file and report any syntax errors.
    The path must be a .py file inside the workspace.

    Note: this may create a __pycache__ directory containing .pyc bytecode
    files next to the source file (standard CPython caching behaviour).
    """
    try:
        target = _validate_path(path)
    except ValueError as exc:
        return f"Error: {exc}"

    if not target.exists():
        return f"Error: file '{path}' does not exist"

    try:
        # doraise=True makes py_compile raise PyCompileError on syntax issues
        # quiet=1     suppresses the "Compiling ..." message on stdout
        py_compile.compile(str(target), doraise=True, quiet=1)
        return f"OK: '{path}' compiled successfully (no syntax errors)."
    except py_compile.PyCompileError as e:
        return f"Syntax error in '{path}':\n{e.msg}"
    except Exception as e:                       # noqa: BLE001
        return f"Error compiling '{path}': {e}"
# ================================================================================================================================================
# Agent
# ================================================================================================================================================

class Agent:
    """A simple conversational agent that can call registered tools via Ollama."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        ollama_url: str = OLLAMA_URL,
        max_iterations: int = 20,
        verbose: bool = True,
        temperature: float | None = None,
        show_thinking: bool = True,
    ) -> None:
        self.model         = model
        self.ollama_url    = ollama_url
        self.max_iterations = max_iterations
        self.verbose       = verbose
        self.temperature   = temperature  # None -> let Ollama use its default
        self.show_thinking = show_thinking

        self.system_prompt = system_prompt or (
            "You are a helpful assistant with access to tools. "
            "When a tool would help answer the user, call it. "
            "If a tool returns an error, explain it to the user. "
            "When you have the final answer, reply normally without calling any tool. "
            "Use `create_file` to create new files and `update_file` to modify "
            "existing files (update_file will show you a diff before applying). "
            "When calling `update_file`, always pass the COMPLETE new contents "
            "of the file in the `content` argument - never just the changed lines "
            "or a diff/patch. If you are unsure of the current contents, read "
            "the file first with `read_text_file` and then resend the whole file "
            "with your edits applied."
        )

        if tools is None:
            tools = list(_TOOL_REGISTRY.values())
        self.tools:    list[Tool]            = tools
        self.tool_map: dict[str, Tool]       = {t.name: t for t in tools}
        self.messages: list[dict[str, Any]]  = []

        # ------------------------------------------------------------------
        # Session statistics & status-bar state
        # ------------------------------------------------------------------
        # Cumulative token usage accumulated across every LLM call during
        # this session.  Ollama reports per-response token counts under the
        # ``prompt_eval_count`` (input) and ``eval_count`` (output) keys;
        # we sum them up here so the status bar can show session totals.
        self.session_prompt_tokens     = 0
        self.session_completion_tokens = 0
        self.session_total_tokens      = 0
        self.llm_call_count            = 0
        # Human-readable label of the current agent state, shown in the
        # status bar.  Updated by chat() as work progresses:
        #   "idle"            - waiting for user input
        #   "thinking"        - a request to the LLM is in flight
        #   "calling tool: X" - executing tool X
        #   "done"            - produced a final answer, back to idle
        #   "error"           - the last call failed
        self.state: str = "idle"
        # Wall-clock start of the session, used for the elapsed-time field.
        self._session_start_time = time.time()

        logger.info("Agent initialised | model=%s | url=%s | max_iterations=%d | temperature=%s | show_thinking=%s | tools=%s",
                    self.model, self.ollama_url, self.max_iterations,
                    self.temperature, self.show_thinking,
                    [t.name for t in self.tools])
        logger.debug("System prompt: %s", self.system_prompt)

    # ------------------------- Status bar --------------------------------

    @staticmethod
    def _format_tokens(n: int) -> str:
        """Compact human-readable formatting of a token count."""
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        if n >= 1_000:
            return f"{n / 1000:.1f}k"
        return str(n)

    def render_status_bar(self) -> str:
        """Build the single-line status bar string.

        Displays: the LLM model in use, the current agent state, the
        current temperature setting, whether chain-of-thought display is
        on or off, the cumulative token usage for the session (prompt /
        completion), the number of LLM calls, and the number of
        conversation turns.

        This is intended to be used as a ``prompt_toolkit`` bottom
        toolbar callback (see ColouredPrompt(bottom_toolbar=...)), so it
        is cheap and side-effect free.
        """
        turns = sum(1 for msg in self.messages if msg.get("role") == "user")
        temp_str = "default" if self.temperature is None else f"{self.temperature}"
        think_str = "on" if self.show_thinking else "off"

        return (
            f" LLM: {self.model} "
            f"| state: {self.state} "
            f"| temp: {temp_str} "
            f"| think: {think_str} "
            f"| tokens: {self._format_tokens(self.session_prompt_tokens)} in / "
            f"{self._format_tokens(self.session_completion_tokens)} out "
            f"| calls: {self.llm_call_count} "
            f"| turns: {turns} "
        )

    def print_status_bar(self) -> None:
        """Print the status bar as a single dim line to the terminal.

        This is used during the agent's response phase, when the live
        prompt_toolkit bottom toolbar is no longer visible (because the
        PromptSession UI only exists while waiting for input).  Printing
        the bar here keeps the information on screen while the agent is
        thinking / calling tools.
        """
        bar = self.render_status_bar()
        if colours_enabled():
            print(c(bar, "dim", "gray"))
        else:
            print(bar)

    # ------------------------- HTTP ---------------------------------------

    def _call_ollama(self) -> dict:
        payload: dict[str, Any] = {
            "model":    self.model,
            "messages": [{"role": "system", "content": self.system_prompt}, *self.messages],
            "stream":   False,
        }
        # Only include temperature if the user has explicitly set one.
        if self.temperature is not None:
            payload["options"] = {"temperature": float(self.temperature)}

        if self.tools:
            payload["tools"] = [t.to_ollama_schema() for t in self.tools]

        # ---------------------------------------------------------------
        # Log the full JSON message being sent to the LLM.
        # The dedicated `llm_payload_logger` writes to `agent.log` (via
        # the root `agent` file handler configured above) and uses a
        # clearly delimited banner so the payload is easy to find/grep.
        # ---------------------------------------------------------------
        payload_json = _safe_json(payload)
        banner = "=" * 72
        llm_payload_logger.info(
            "%s\nLLM REQUEST  ->  %s\n%s\n%s\n%s",
            banner, self.ollama_url, banner, payload_json, banner,
        )
        logger.debug("HTTP request to %s\n%s", self.ollama_url, payload_json)

        t0 = time.time()
        try:
            resp = requests.post(self.ollama_url, json=payload, timeout=500)
            elapsed = time.time() - t0
            try:
                body_for_log = _safe_json(resp.json()) if resp.ok else _truncate(resp.text, 500)
            except Exception:                       # noqa: BLE001
                body_for_log = _truncate(resp.text, 500)

            logger.debug("HTTP response status=%d, elapsed=%.2fs | body\n%s",
                         resp.status_code, elapsed, body_for_log)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout as e:
            logger.error("Timeout calling Ollama after %.2fs: %s", time.time() - t0, e)
            self.state = "error"
            raise
        except requests.exceptions.HTTPError as e:
            # Try to capture the response body for debugging
            body = ""
            try:
                body = e.response.text if e.response is not None else ""
            except Exception:                       # noqa: BLE001
                pass
            logger.error("HTTP error from Ollama: %s | body=%s", e, _truncate(body, 500))
            self.state = "error"
            raise
        except requests.exceptions.RequestException as e:
            logger.error("RequestException calling Ollama: %s", e)
            self.state = "error"
            raise
        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON response from Ollama: %s", e)
            self.state = "error"
            raise

        logger.debug("Received response from Ollama in %.2fs | body\n%s",
                     elapsed, _safe_json(data))

        # ------------------------------------------------------------------
        # Accumulate token usage reported by Ollama for the status bar.
        # Ollama exposes per-response counts at the top level of the JSON:
        #   prompt_eval_count  - input/prompt tokens
        #   eval_count         - generated/completion tokens
        # Some endpoints (e.g. cloud proxies) may not report them; we
        # guard every access so the agent keeps working regardless.
        # ------------------------------------------------------------------
        try:
            self.llm_call_count += 1
            prompt_toks = data.get("prompt_eval_count")
            if isinstance(prompt_toks, int):
                self.session_prompt_tokens += prompt_toks
            completion_toks = data.get("eval_count")
            if isinstance(completion_toks, int):
                self.session_completion_tokens += completion_toks
            # total_count is not always present; fall back to the sum.
            total_toks = data.get("total_count")
            if isinstance(total_toks, int):
                self.session_total_tokens += total_toks
            else:
                p = prompt_toks if isinstance(prompt_toks, int) else 0
                c = completion_toks if isinstance(completion_toks, int) else 0
                self.session_total_tokens += p + c
        except Exception:  # noqa: BLE001 - never let stats break the chat
            logger.debug("Could not parse token usage from Ollama response", exc_info=True)

        return data

    # ------------------------- Tools --------------------------------------

    def _execute_tool_call(self, tool_call: dict) -> str:
        fn       = tool_call.get("function", {}) or {}
        name     = fn.get("name", "")
        raw_args = fn.get("arguments", {}) or {}

        logger.debug("Tool call requested: %s | raw_args=%s", name, _safe_json(raw_args))

        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                logger.error("Tool '%s' received malformed arguments JSON: %s | raw=%s",
                             name, e, _truncate(raw_args))
                return f"Error: tool '{name}' received malformed arguments JSON"

        tool = self.tool_map.get(name)
        if tool is None:
            logger.error("Unknown tool requested: %s | available=%s",
                         name, list(self.tool_map.keys()))
            return f"Error: unknown tool '{name}'"

        self.state = f"calling tool: {name}"
        self.print_status_bar()
        try:
            result = tool.func(**raw_args)
            result_str = str(result)
            logger.debug("Tool '%s' executed | args=%s | result=%s",
                         name, _safe_json(raw_args), _truncate(result_str, 5000))
            return result_str
        except TypeError as e:
            logger.error("Bad arguments for tool '%s': %s | args=%s", name, e, _safe_json(raw_args))
            return f"Error: bad arguments for '{name}': {e}"
        except Exception as e:                                 # noqa: BLE001
            logger.exception("Unhandled exception while executing tool '%s'", name)
            return f"Error executing '{name}': {e}"

    # ------------------------- Conversation -------------------------------

    def chat(self, user_message: str) -> str:
        """Send a user message and return the agent's final answer
            (running tools as needed).

        A ``KeyboardInterrupt`` (Ctrl-C) raised while the agent is busy -
        either waiting for an LLM response or executing a tool - is
        caught here so that the *current task* is abandoned gracefully
        and control returns to the interactive REPL.  The conversation
        history accumulated so far is preserved, so the user can simply
        type a new request (or a corrective one) and the agent will
        continue from where it left off.  Ctrl-C therefore never exits
        the session; it only interrupts the in-flight call sequence.
        """
        logger.debug("User message:\n%s", user_message)
        self.messages.append({"role": "user", "content": user_message})

        # ------------------------------------------------------------------
        # The entire call sequence (the ``while``/``for`` loop below) is
        # wrapped in a ``try`` so that a Ctrl-C (KeyboardInterrupt, a
        # BaseException that is *not* an Exception) interrupts whatever
        # step is currently in flight - an HTTP request to the LLM or the
        # execution of a tool - and is handled cleanly instead of
        # tearing down the whole program.
        # ------------------------------------------------------------------
        try:
            while True:
                # We loop here so that, if the user agrees to keep going when
                # the iteration limit is hit, we can resume the agent's work
                # for another `max_iterations` rounds. The inner `for` loop
                # only ever consumes at most `max_iterations` iterations per
                # pass.
                for i in range(self.max_iterations):
                    logger.info("--- Agent iteration %d/%d ---", i + 1, self.max_iterations)
                    self.state = "thinking"
                    self.print_status_bar()
                    try:
                        data    = self._call_ollama()
                    except Exception as e:
                        logger.exception("Iteration %d failed during Ollama call", i + 1)
                        self.state = "error"
                        raise

                    message = data.get("message", {}) or {}
                    logger.debug("Assistant message object:\n%s", _safe_json(message))
                    self.messages.append(message)

                    # Display the model's chain-of-thought if the user asked for it
                    # and the model actually produced any. Ollama exposes this under
                    # the `thinking` key of the assistant message.
                    thinking = message.get("thinking")
                    if self.show_thinking and thinking:
                        if self.verbose:
                            print(f"  [think] {_truncate(str(thinking), 400)}")

                    tool_calls = message.get("tool_calls") or []
                    if not tool_calls:
                        final = (message.get("content") or "").strip()
                        logger.debug("Final assistant answer:\n%s", final)
                        self.state = "done"
                        return final

                    logger.debug("Model requested %d tool call(s): %s",
                                 len(tool_calls), _safe_json(tool_calls))
                    for tc in tool_calls:
                        result = self._execute_tool_call(tc)
                        fn_name = (tc.get("function") or {}).get("name", "?")
                        self.messages.append({"role": "tool", "content": result})
                        if self.verbose:
                            preview = result if len(result) <= 120 else result[:120] + "..."
                            # Coloured tool-call echo.
                            tag = c("[tool]", "bold", "bright_magenta")
                            name_col = c(fn_name, "magenta")
                            print(f"  {tag} {name_col}(...) -> {preview}")

                # Exhausted `self.max_iterations` rounds without a final answer.
                # Ask the user whether they want the agent to keep going.
                logger.warning("Reached max_iterations=%d without a final answer",
                               self.max_iterations)
                self.state = "idle"
                if not _request_continue_approval(self.max_iterations):
                    return ("Sorry \u2014 I could not reach a final answer within "
                            "the iteration limit.")
        except KeyboardInterrupt:
            # ----------------------------------------------------------------
            # The user pressed Ctrl-C to interrupt the agent's call
            # sequence (an in-flight LLM request or a running tool).  We
            # stop the current task gracefully, leaving the conversation
            # history intact, and hand control back to the main REPL loop
            # so the user can issue a new request.  The agent stays alive
            # - only this task is cancelled.
            # ----------------------------------------------------------------
            logger.info("Chat interrupted by user (Ctrl-C) at state=%r", self.state)
            # Reset to a clean idle state so the status bar / next prompt
            # do not show a stale "thinking" / "calling tool" label.
            self.state = "idle"
            # Emit a newline after the "^C" the terminal echoes so the
            # interruption message starts on a fresh line, then warn the
            # user with a coloured marker when colours are available.
            print()
            if self.verbose:
                if colours_enabled():
                    print(c("[interrupted by user - task stopped]",
                            "bold", "bright_yellow"))
                else:
                    print("[interrupted by user - task stopped]")
            return "[interrupted by user - task stopped]"

    def reset(self) -> None:
        logger.info("Conversation reset (cleared %d messages)",
len(self.messages))
        self.messages = []
        self.state = "idle"


# ================================================================================================================================================
# Interactive demo
# ================================================================================================================================================

# Help text shown at startup and by the /help (/?) command.
# Kept in one place so the banner and the help command never drift apart.
HELP_TEXT = """\
Commands:
  quit / exit      end the session
  reset            clear the conversation
  Ctrl-C           interrupt the agent while it is working (thinking or
                   calling a tool) and return to the prompt.  The session
                   is NOT ended - only the current task is stopped, so you
                   can immediately type a new request.
  /?  or  /help    show this help message
  /temp <value>    set sampling temperature, e.g. /temp 0.7  (blank = default)
  /max_iter <n>    set max tool-calling iterations, e.g. /max_iter 10
  /think on|off    enable/disable display of the model's chain-of-thought
  /kdiff [<path>]  set the kdiff3 binary path in .env (KDIFF3_PATH=...).
                   With no argument, prompts for the path interactively.
                   Updates the running session immediately.

The prompt supports TAB completion for the slash commands above and their
arguments (e.g. /think on|off).  Output is colourised when the terminal
supports it; set NO_COLOR=1 to disable colours.  A live status bar at the
bottom of the terminal shows the active LLM, current state, temperature,
chain-of-thought toggle, token usage, LLM call count and conversation turns.
"""


def _print_help() -> None:
    """Print the interactive-command help text (single source of truth)."""
    # Colourise the section header line if colours are available.
    if colours_enabled():
        lines = HELP_TEXT.splitlines(keepends=True)
        if lines:
            # Header "Commands:" in bold.
            print(c(lines[0], "bold", "bright_blue"), end="")
            for ln in lines[1:]:
                # Highlight the leading command token.
                stripped = ln.lstrip()
                if stripped and not stripped.startswith(" ") and ":" not in stripped[:1]:
                    pass
                print(ln, end="")
        else:
            print(HELP_TEXT, end="")
    else:
        print(HELP_TEXT, end="" if HELP_TEXT.endswith("\n") else "\n")


def _local_py_files() -> list[str]:
    """Return a sorted list of ``*.py`` files in the current directory.

    Used to feed tab completion for the ``/kdiff`` command's file argument.
    """
    try:
        return sorted(
            f for f in os.listdir(".")
            if f.endswith(".py") and os.path.isfile(f)
        )
    except OSError:
        return []


def main() -> None:
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
    except requests.exceptions.RequestException as e:
        logger.critical("Could not reach Ollama at %s: %s", OLLAMA_URL, e)
        raise SystemExit(f"Could not reach Ollama at {OLLAMA_URL}: {e}\n"
                         f"Is 'ollama serve' running?")

    agent = Agent(model=DEFAULT_MODEL)

    def _handle_command(cmd: str) -> bool:
        """Handle one of the '/' interactive commands.

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
                print("[temperature reset to model default]\n")
                logger.info("User reset temperature to default")
            else:
                try:
                    value = float(parts[1])
                except ValueError:
                    print(f"Error: {parts[1]!r} is not a valid number. "
                          f"Usage: /temp <value between 0 and 2>\n")
                    return True
                if not 0.0 <= value <= 2.0:
                    print(f"Error: temperature must be between 0.0 and 2.0, got {value}\n")
                    return True
                agent.temperature = value
                print(f"[temperature set to {value}]\n")
                logger.info("User set temperature to %s", value)
            return True

        if head == "/max_iter":
            if len(parts) != 2:
                print("Usage: /max_iter <positive integer>\n")
                return True
            try:
                value = int(parts[1])
            except ValueError:
                print(f"Error: {parts[1]!r} is not a valid integer. "
                      f"Usage: /max_iter <positive integer>\n")
                return True
            if value < 1:
                print(f"Error: max_iterations must be >= 1, got {value}\n")
                return True
            agent.max_iterations = value
            print(f"[max_iterations set to {value}]\n")
            logger.info("User set max_iterations to %d", value)
            return True

        if head == "/think":
            if (len(parts) != 2 or
                    parts[1].lower() not in {"on", "off", "true", "false", "1", "0", "yes", "no"}):
                print("Usage: /think on | /think off\n")
                return True
            new_state = parts[1].lower() in {"on", "true", "1", "yes"}
            agent.show_thinking = new_state
            print(f"[thinking display {'enabled' if new_state else 'disabled'}]\n")
            logger.info("User %s thinking display", "enabled" if new_state else "disabled")
            return True

        if head == "/kdiff":
            return _set_kdiff3_path_interactively(parts, env_path=".env")

        # Not a recognised command - let it fall through to the LLM
        return False

    # ---- Human-friendly prompt with tab completion -----------------------
    # The prompt also shows a live status bar pinned at the bottom of the
    # terminal (via prompt_toolkit's bottom_toolbar).  agent.render_status_bar
    # is called every refresh tick so the current agent state stays current
    # while the user types.
    completer = AgentCompleter(extra_files=_local_py_files())
    prompt = ColouredPrompt(
        completer=completer,
        bottom_toolbar=agent.render_status_bar,
    )

    while True:
        user_input = prompt.read()
        if user_input is None:
            logger.info("User ended the session (EOF or KeyboardInterrupt)")
            print()
            break
        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            logger.info("User typed '%s' - exiting", user_input)
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("[conversation reset]\n")
            continue
        # Also accept bare 'help' / '?' (mirrors 'quit'/'exit'/'reset')
        if user_input.lower() in {"help", "?"}:
            _print_help()
            logger.info("User requested help via %r", user_input)
            continue
        # Interactive session commands (/temp, /max_iter, /think, /help, /?)
        if user_input.startswith("/") and _handle_command(user_input):
            continue

        # Coloured "Agent> " prefix instead of the old inline print().
        print(c("Agent> ", "bold", "bright_yellow"), end="", flush=True)
        try:
            answer = agent.chat(user_input)
        except KeyboardInterrupt:
            # Defensive fallback: normally Agent.chat() already absorbs a
            # Ctrl-C raised during its call sequence and returns a message.
            # If a Ctrl-C ever slips through here (e.g. raised before the
            # chat loop's try block is entered), we still handle it so the
            # session never crashes - we just drop back to the prompt.
            logger.info("Chat interrupted by user (Ctrl-C) in main loop")
            agent.state = "idle"
            print()
            answer = "[interrupted by user - task stopped]"
        except requests.exceptions.RequestException as e:
            logger.error("Connection error during chat: %s", e)
            answer = f"[connection error: {e}]"
        except Exception as e:                                  #noqa: BLE001
            logger.exception("Unhandled exception during chat")
            answer = f"[error: {e}]"
        # The agent finished its turn; go back to idle for the next prompt.
        agent.state = "idle"
        # Print a final status bar line after the answer so the latest
        # token totals / call counts remain visible above the next prompt.
        print(answer)
        agent.print_status_bar()
        print()
        logger.info("Printed agent answer to console (%d chars)", len(answer))

    logger.info("Agent session ended")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        logger.critical("Fatal error in main()", exc_info=True)
        raise