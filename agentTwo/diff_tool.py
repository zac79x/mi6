"""External diff tool configuration and diff-then-approve workflow.

This module owns:

* reading/writing the diff-tool path in the project's ``.env``
  (accepts both ``DIFF_TOOL_PATH`` and the legacy ``KDIFF3_PATH`` key),
* configuring the diff tool at startup,
* showing the user a diff (either with an external GUI tool or a
  built-in text fallback),
* prompting the user for approval before any change is persisted,
* the ``/kdiff`` interactive command.

``DIFF_TOOL_PATH`` is stored in :mod:`agent.config` and is read/written
through this module at runtime.
"""

from __future__ import annotations

import difflib
import shutil
import subprocess
from pathlib import Path

from agent.cli_ui import c as _colour, colours_enabled as _colours_enabled
from agent.config import DIFF_TOOL_PATH
from agent.logging_setup import logger

# Env keys accepted when reading/writing the diff-tool path from ``.env``.
# ``DIFF_TOOL_PATH`` is the preferred, generic key; ``KDIFF3_PATH`` is the
# legacy key kept for backward compatibility.  When *writing*, we always
# use ``DIFF_TOOL_PATH``; when *reading*, we try both and the first hit
# wins (preferring the generic key).
_ENV_KEYS = ("DIFF_TOOL_PATH", "KDIFF3_PATH")

# Diff-tool binaries looked for during auto-detection (``shutil.which``).
# Ordered roughly by popularity on the target platforms.
_AUTO_DETECT_CANDIDATES = (
    "kdiff3",
    "meld",
    "bcomp",       # Beyond Compare (CLI launcher)
    "opendiff",    # macOS FileMerge wrapper
    "winmergeu",   # WinMerge
    "diff",        # plain text diff (not GUI, but always works)
)

# ---------------------------------------------------------------------------
# .env helpers
# ---------------------------------------------------------------------------

def read_kdiff3_path_from_env(env_path: str | Path = ".env") -> str | None:
    """Read the diff-tool path from a ``.env`` file, if present.

    Looks for either ``DIFF_TOOL_PATH`` (preferred) or the legacy
    ``KDIFF3_PATH`` key.  Blank lines and lines starting with ``#``
    (comments) are ignored.  Surrounding single or double quotes around
    the value are stripped.  Returns ``None`` silently if the file does
    not exist or cannot be read.
    """
    try:
        env_file = Path(env_path)
        if not env_file.is_file():
            return None
        # Collect all matching entries; prefer the generic key.
        found: dict[str, str] = {}
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key_norm = key.strip().upper()
            if key_norm in _ENV_KEYS:
                v = value.strip().strip('"').strip("'")
                if v:
                    found[key_norm] = v
        for key in _ENV_KEYS:
            if key in found:
                logger.debug("Found %s in %s: %s", key, env_path, found[key])
                return found[key]
    except OSError as exc:
        logger.debug("Could not read .env file %s: %s", env_path, exc)
    return None


def _format_env_value(value: str) -> str:
    """Format a value for safe inclusion in a ``.env`` file.

    Double-quotes the value when it contains whitespace, ``#``, or
    quote characters, escaping any embedded double-quotes. Bare values
    are written unquoted for consistency with the project's existing
    ``.env`` style.
    """
    v = value.strip()
    if v == "":
        return ""
    needs_quoting = any(ch.isspace() for ch in v) or any(ch in v for ch in '"#')
    if not needs_quoting:
        return v
    escaped = v.replace('"', '\\"')
    return f'"{escaped}"'


def write_kdiff3_path_to_env(
    value: str,
    env_path: str | Path = ".env",
    *,
    overwrite: bool = True,
) -> dict:
    """Write the diff-tool path to a ``.env`` file.

    Always writes under the ``DIFF_TOOL_PATH`` key (the preferred,
    generic name).  When replacing, a legacy ``KDIFF3_PATH`` line is
    also updated so old sessions don't leave a stale value behind.

    Behaviour:
        * If the file does not exist, it is created with the single
          ``DIFF_TOOL_PATH=<value>`` line.
        * If the file contains a ``DIFF_TOOL_PATH=...`` (or legacy
          ``KDIFF3_PATH=...``) line, that line is replaced in place
          while every other line is preserved exactly as is
          (comments and blank lines included).
        * Otherwise the new line is appended.
        * The value is written using :func:`_format_env_value` so that
          paths containing spaces are safely quoted.

    Returns:
        A dict with ``ok`` (bool), ``action`` (str), ``path`` (str) and
        ``reason`` (str | None) keys.
    """
    if value is None or str(value).strip() == "":
        return {
            "ok": False, "action": "refused", "path": str(env_path),
            "reason": "Refused to write an empty diff-tool path value.",
        }

    new_value = str(value).strip()
    env_file = Path(env_path)
    formatted = f"DIFF_TOOL_PATH={_format_env_value(new_value)}"

    try:
        if env_file.parent and str(env_file.parent) not in ("", "."):
            env_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {
            "ok": False, "action": "refused", "path": str(env_path),
            "reason": f"Could not create parent directory: {exc}",
        }

    if not env_file.is_file():
        try:
            env_file.write_text(formatted + "\n", encoding="utf-8")
            logger.info("Created .env with DIFF_TOOL_PATH=%s", new_value)
            return {"ok": True, "action": "created", "path": str(env_path), "reason": None}
        except OSError as exc:
            return {
                "ok": False, "action": "refused", "path": str(env_path),
                "reason": f"Could not create {env_path}: {exc}",
            }

    try:
        original_text = env_file.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "ok": False, "action": "refused", "path": str(env_path),
            "reason": f"Could not read {env_path}: {exc}",
        }

    original_lines = original_text.splitlines()
    had_trailing_newline = original_text.endswith("\n")

    replaced = False
    new_lines: list[str] = []
    for raw in original_lines:
        stripped = raw.lstrip()
        is_diff_line = False
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, _, _ = stripped.partition("=")
            if key.strip().upper() in _ENV_KEYS:
                is_diff_line = True
        if is_diff_line:
            if not overwrite:
                logger.info("Diff-tool path already present in %s; not overwriting", env_path)
                return {
                    "ok": True, "action": "kept", "path": str(env_path),
                    "reason": "Diff-tool path already present (overwrite=False).",
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
    except OSError as exc:
        return {
            "ok": False, "action": "refused", "path": str(env_path),
            "reason": f"Could not write {env_path}: {exc}",
        }

    action = "updated" if replaced else "appended"
    logger.info("Wrote DIFF_TOOL_PATH to %s (action=%s, value=%s)",
                env_path, action, new_value)
    return {"ok": True, "action": action, "path": str(env_path), "reason": None}


# ---------------------------------------------------------------------------
# Interactive /kdiff command
# ---------------------------------------------------------------------------

def set_kdiff3_path_interactively(
    parts: list[str], env_path: str | Path = ".env",
) -> bool:
    """Handle the ``/kdiff`` interactive command.

    With no argument, the user is prompted for a path. With one
    argument, that argument is used as the new path directly. The new
    path is persisted to ``env_path``, round-tripped through the reader
    to confirm it, and stored in :data:`agent.config.DIFF_TOOL_PATH`.

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

    print(f"Persisting DIFF_TOOL_PATH={new_path!r} to {env_path} ...")
    result = write_kdiff3_path_to_env(new_path, env_path=env_path)

    if not result.get("ok"):
        print(f"Error: {result.get('reason', 'unknown error')}\n")
        logger.error("/kdiff failed: %s", result)
        return True

    verified = read_kdiff3_path_from_env(env_path)
    if verified != new_path:
        print("Warning: wrote DIFF_TOOL_PATH but could not read it back exactly. "
              f"Read: {verified!r}, Wrote: {new_path!r}\n")
        logger.warning("DIFF_TOOL_PATH round-trip mismatch: wrote=%r read=%r",
                       new_path, verified)
    else:
        print(f"Verified DIFF_TOOL_PATH in {env_path}: {verified}")

    DIFF_TOOL_PATH = new_path
    print(f"[diff tool set to {DIFF_TOOL_PATH}]\n")
    logger.info("User set DIFF_TOOL_PATH via /kdiff: %s (file action=%s)",
                DIFF_TOOL_PATH, result.get("action"))
    return True


# ---------------------------------------------------------------------------
# Startup configuration
# ---------------------------------------------------------------------------

def _auto_detect_diff_tool() -> str | None:
    """Search PATH for a known diff/merge binary.

    Returns the first match found by :func:`shutil.which`, or ``None``
    if none of the candidates in :data:`_AUTO_DETECT_CANDIDATES` are
    on the PATH.
    """
    for candidate in _AUTO_DETECT_CANDIDATES:
        path = shutil.which(candidate)
        if path:
            logger.info("Auto-detected diff tool: %s -> %s", candidate, path)
            return path
    return None


def configure_diff_tool() -> str | None:
    """Ask the user where the diff tool is located.

    Called once at startup.  Resolution order:

    1. ``DIFF_TOOL_PATH`` (or legacy ``KDIFF3_PATH``) in ``.env``.
    2. Auto-detection via :func:`shutil.which` of common binaries
       (kdiff3, meld, bcomp, opendiff, winmergeu, diff).
    3. Interactive prompt (if the user is at a terminal).
    4. ``None`` (use the built-in text-diff fallback).

    The chosen path is stored in :data:`agent.config.DIFF_TOOL_PATH`.

    Returns:
        The configured diff-tool path, or ``None`` to use the
        built-in text-diff fallback.
    """
    global DIFF_TOOL_PATH

    env_default = read_kdiff3_path_from_env()
    print("=" * 60)
    print("DIFF TOOL CONFIGURATION")
    print("-" * 60)
    print("The 'update_file' tool uses an external diff/merge program to show")
    print("you the proposed changes to a file BEFORE they are written.")
    if env_default:
        print(f"Detected diff-tool path in .env: {env_default}")
        DIFF_TOOL_PATH = env_default
    else:
        auto = _auto_detect_diff_tool()
        if auto:
            print(f"Auto-detected diff tool on PATH: {auto}")
            DIFF_TOOL_PATH = auto
        else:
            print("No diff-tool path found in .env or on PATH.")
            try:
                answer = input("Path to diff tool (press Enter for text-diff fallback): ").strip()
                DIFF_TOOL_PATH = answer or None
            except (EOFError, KeyboardInterrupt):
                print("\nNo interactive stdin available - using built-in text diff.")
                DIFF_TOOL_PATH = None
    print("=" * 60)
    logger.info("Diff tool configured: %s", DIFF_TOOL_PATH)
    return DIFF_TOOL_PATH


# ---------------------------------------------------------------------------
# Diff rendering & approval
# ---------------------------------------------------------------------------

def _diff_stats(old_lines: list[str], new_lines: list[str]) -> dict[str, int]:
    """Compute a quick added/removed/changed line summary.

    Uses :class:`difflib.SequenceMatcher` on the line lists to get
    opcodes, then maps each opcode to a category.  A replaced block
    counts as both removed and added (we don't try to guess how many
    lines "changed" vs. were wholesale replaced; the sum is the
    important number for a quick triage signal).
    """
    sm = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    added = removed = changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            removed += i2 - i1
            added += j2 - j1
            changed += min(i2 - i1, j2 - j1)
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "insert":
            added += j2 - j1
    return {"added": added, "removed": removed, "changed": changed}


def print_text_diff(original: str, modified: str) -> bool:
    """Show a unified text diff between two files using :mod:`difflib`.

    Used as a fallback when no diff tool is configured or when the
    configured tool cannot be found.  The diff is colourised when the
    terminal supports it (added lines green, removed lines red, hunk
    markers cyan).  A one-line statistics summary is printed after the
    diff.  Returns True on success.
    """
    try:
        with open(original, "r", encoding="utf-8") as fh:
            old_lines = fh.readlines()
        with open(modified, "r", encoding="utf-8") as fh:
            new_lines = fh.readlines()
        diff = difflib.unified_diff(
            old_lines, new_lines, fromfile=original, tofile=modified, n=3,
        )
        use_colour = _colours_enabled()
        for line in diff:
            if use_colour:
                if line.startswith("+") and not line.startswith("+++"):
                    print(_colour(line, "green"), end="")
                elif line.startswith("-") and not line.startswith("---"):
                    print(_colour(line, "red"), end="")
                elif line.startswith("@@"):
                    print(_colour(line, "cyan"), end="")
                else:
                    print(line, end="")
            else:
                print(line, end="")
        # Diff statistics
        stats = _diff_stats(old_lines, new_lines)
        if stats["added"] or stats["removed"]:
            summary = (f"  {stats['added']} added, {stats['removed']} removed"
                       f", {stats['changed']} changed")
            if use_colour:
                print(_colour(summary, "dim", "gray"))
            else:
                print(summary)
        return True
    except Exception as exc:                       # noqa: BLE001
        print(f"Error producing text diff: {exc}")
        return False


def run_diff_tool(original: str, modified: str) -> bool:
    """Run the configured diff tool to compare ``original`` and ``modified``.

    Falls back to :func:`print_text_diff` if no tool is configured or
    the configured tool cannot be found.

    Returns:
        True  - the diff was shown successfully.
        False - the diff tool could not be run at all.
    """
    if not DIFF_TOOL_PATH:
        return print_text_diff(original, modified)

    try:
        result = subprocess.run([DIFF_TOOL_PATH, original, modified], timeout=300)
        logger.info("Diff tool '%s' exited with code %d",
                    DIFF_TOOL_PATH, result.returncode)
        # Exit code 0 = files identical, 1 = differences found (normal
        # for a diff tool). Any other code signals a real failure;
        # fall back to the text diff so the user is never left without
        # a reviewable change.
        if result.returncode not in (0, 1):
            print(f"Warning: diff tool exited with code {result.returncode}.")
            print("Falling back to text diff.")
            return print_text_diff(original, modified)
        return True
    except FileNotFoundError:
        print(f"Error: diff tool not found at {DIFF_TOOL_PATH!r}.")
        print("Falling back to text diff.")
        return print_text_diff(original, modified)
    except subprocess.TimeoutExpired:
        print(f"Error: diff tool '{DIFF_TOOL_PATH}' timed out after 300s.")
        print("Falling back to text diff.")
        return print_text_diff(original, modified)
    except Exception as exc:                       # noqa: BLE001
        print(f"Error running diff tool {DIFF_TOOL_PATH!r}: {exc}")
        return print_text_diff(original, modified)


def request_update_approval(path: str) -> tuple[bool, str | None]:
    """Ask the user in the terminal whether to apply the proposed changes.

    If the user rejects, follow up with a free-form clarification
    prompt so the agent (and the model on its next turn) can act on
    the feedback. The user can skip clarification by pressing Enter,
    in which case ``None`` is returned alongside the rejection.

    Returns:
        A ``(approved, clarification)`` tuple:

        * ``(True, None)``         - user approved the change.
        * ``(False, None)``        - user rejected, no clarification
                                     provided (or stdin unavailable).
        * ``(False, "<text>")``    - user rejected and provided
                                     free-form clarification. The
                                     text is included verbatim so the
                                     model can act on it on the next
                                     turn.
    """
    try:
        answer = input(f"Apply changes to {path}? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nNo interactive stdin available - refusing by default.")
        return (False, None)
    if answer in {"y", "yes"}:
        return (True, None)

    # Rejected - ask the user for free-form clarification so the
    # agent has something concrete to act on. The user can press
    # Enter to skip.
    try:
        clarification = input(
            "Changes rejected. What would you like changed? "
            "(press Enter to skip): "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNo interactive stdin available - skipping clarification.")
        return (False, None)
    if not clarification:
        return (False, None)
    return (False, clarification)



