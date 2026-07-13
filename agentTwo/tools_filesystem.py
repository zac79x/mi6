"""Filesystem-oriented tools.

Two of them (``create_file`` and ``update_file``) mutate the workspace
and therefore go through :func:`agent.safety.validate_path`. The rest
are read-only inspection helpers.

The ``update_file`` tool uses the diff-then-approve workflow defined in
:mod:`agent.diff_tool`.
"""

from __future__ import annotations

import os
import py_compile
import tempfile
from typing import Union

from agent.diff_tool import run_diff_tool, request_update_approval
from agent.logging_setup import logger
from agent.safety import validate_path
from agent.tools_registry import tool


# ---------------------------------------------------------------------------
# Read-only tools
# ---------------------------------------------------------------------------

@tool
def read_text_file(path: str, max_chars: int = 40000) -> str:
    """Read a UTF-8 text file. Truncates very long files to ``max_chars``."""
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
    except Exception as exc:                       # noqa: BLE001
        return f"Error reading file: {exc}"


@tool
def read_file_lines(path: str, start_line: int = 0, num_lines: int = 2000) -> str:
    """Read a slice of lines from a text file. Useful for inspecting large files without loading the whole thing.

    The default ``num_lines`` is 2000, which covers the entire source
    file in a single call for essentially every script in this project
    (and most Python projects). Prefer one large call over multiple
    small paginated calls: each tool call is an extra LLM round-trip,
    so fewer, larger reads materially reduce agent iterations.

    Pass ``num_lines`` explicitly only when (a) you need a later
    section of a file that genuinely has more than 2000 lines, or
    (b) you already know the slice you want and want to skip the
    first few lines via ``start_line``.
    """
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
    except OSError as exc:
        return f"Error reading file: {exc}"


@tool
def list_directory(path: str = ".", show_hidden: bool = False) -> str:
    """List files and directories in the given path. Returns a formatted listing with sizes and types."""
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
    except OSError as exc:
        return f"Error listing directory '{path}': {exc}"


@tool
def path_exists(path: str) -> str:
    """Check whether a path exists and report what kind of filesystem object it is (file, directory, or neither)."""
    if not os.path.exists(path):
        return f"Path '{path}' does not exist."
    if os.path.isdir(path):
        return f"Path '{path}' exists and is a directory."
    if os.path.isfile(path):
        size = os.path.getsize(path)
        return f"Path '{path}' exists and is a file ({size:,} bytes)."
    return f"Path '{path}' exists but is neither a regular file nor a directory."


# ---------------------------------------------------------------------------
# Mutating tools (workspace-validated)
# ---------------------------------------------------------------------------

@tool
def create_file(path: str, content: str) -> dict:
    """Create a NEW file. Does NOT require user approval.

    Only workspace paths whose suffix is one of the allowed extensions
    (see :data:`agent.config.ALLOWED_EXTENSION`) may be created. Refuses
    to overwrite an existing file - use :func:`update_file` for that.
    """
    try:
        target = validate_path(path)
    except ValueError as exc:
        return {"ok": False, "action": "refused", "path": path, "reason": str(exc)}

    if target.exists():
        return {
            "ok": False, "action": "refused", "path": path,
            "reason": "File already exists; use update_file to modify it.",
        }

    try:
        target.write_text(content, encoding="utf-8")
        logger.info("Created file: %s (%d bytes)", path, len(content))
        return {"ok": True, "action": "create", "path": path, "reason": None}
    except OSError as exc:
        return {"ok": False, "action": "refused", "path": path, "reason": str(exc)}


@tool
def update_file(path: str, content: str) -> dict:
    """Update an EXISTING file after showing the user a diff.

    Only workspace paths whose suffix is one of the allowed extensions
    (see :data:`agent.config.ALLOWED_EXTENSION`) may be modified.

    Workflow:
        1. Validate that ``path`` has an allowed extension, stays
           inside the workspace, and that the file already exists.
        2. Write the proposed ``content`` to a temporary file next to
           the original.
        3. Launch the configured diff tool to compare the existing file
           with the temporary file.
        4. Ask the user in the terminal to confirm the change. If the
           user rejects, follow up with a free-form clarification
           prompt so the model can act on the feedback on the next
           turn. The clarification (if any) is included in the
           returned ``reason`` so the model can read it.
        5. If approved, overwrite the original file with the new content.
           The temporary file is always cleaned up.
    """
    try:
        target = validate_path(path)
    except ValueError as exc:
        return {"ok": False, "action": "refused", "path": path, "reason": str(exc)}

    if not target.exists():
        return {
            "ok": False, "action": "refused", "path": path,
            "reason": "File does not exist; use create_file to create a new file.",
        }

    # No-change short-circuit: if the proposed content is byte-identical
    # to the existing file, skip the diff display and approval prompt
    # entirely.  This saves a tool round-trip and avoids confusing the
    # user with an empty diff.
    try:
        existing = target.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "ok": False, "action": "refused", "path": path,
            "reason": f"Could not read existing file for comparison: {exc}",
        }
    if existing == content:
        logger.info("update_file: no changes (content identical) for %s", path)
        return {"ok": True, "action": "noop", "path": path,
                "reason": "No changes - content is identical."}

    original_path = str(target)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=target.suffix + ".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_fh:
            tmp_fh.write(content)

        print("=" * 60)
        print(f"PROPOSED UPDATE: {path}")
        print(f"Temp file:       {tmp_path}")
        print("Launching diff tool for review - close it when you are done...")
        print("=" * 60)

        if not run_diff_tool(original_path, tmp_path):
            return {
                "ok": False, "action": "refused", "path": path,
                "reason": "Diff tool failed; not applying changes.",
            }

        approved, clarification = request_update_approval(path)
        if not approved:
            if clarification:
                reason = (
                    "User did not approve the change. "
                    f"User clarification: {clarification}"
                )
            else:
                reason = "User did not approve the change."
            return {
                "ok": False, "action": "refused", "path": path,
                "reason": reason,
            }

        target.write_text(content, encoding="utf-8")
        logger.info("Updated file: %s (%d bytes)", path, len(content))
        return {"ok": True, "action": "update", "path": path, "reason": None}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Multi-file compile tool
# ---------------------------------------------------------------------------

# JSON Schema override for the LLM-facing tool description. The registry
# would otherwise map the ``list`` annotation to a bare ``"array"`` with
# no item type; this is clearer for the model and explicitly documents
# that each entry is a workspace-relative ``.py`` path string.
_COMPILE_PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "One or more workspace-relative ``.py`` paths. Pass a "
                "single string to compile one file, or a list of "
                "strings to compile several at once."
            ),
        }
    },
    "required": ["paths"],
}


@tool(
    name="compile_python_file",
    description=(
        "Compile one or more Python scripts to check for syntax errors "
        "without executing them."
    ),
    parameters=_COMPILE_PARAMETERS,
)
def compile_python_file(paths: Union[str, list]) -> str:
    """Compile one or more Python scripts to check for syntax errors without executing them.

    ``paths`` may be a single string (one file) or a list of strings
    (a batch of files). The tool iterates over the supplied paths,
    runs :func:`py_compile.compile` on each, and returns a single
    combined report so a batch can be checked in one call.

    For each file the report marks the outcome as one of:

    * ``[OK]``   - compiled without errors
    * ``[FAIL]`` - had a syntax error (the ``PyCompileError`` message is
      included on the same line)
    * ``[SKIP]`` - the path was rejected by workspace validation
      (e.g. not a ``.py`` file, outside the workspace, or points to a
      non-regular file) or the file does not exist
    * ``[ERR]``  - an unexpected error happened while compiling
      (e.g. ``PermissionError``)

    Each file is processed independently: a failure on one file does
    not stop the others, and the report always includes one line per
    input path in the order they were given.

    When called with a single string, the format is still the
    multi-line report (with a one-line header), so callers can rely on
    a consistent shape.

    Example::

        compile_python_file("agent/foo.py")
        compile_python_file(["agent/foo.py", "agent/bar.py"])

    Note: this may create a ``__pycache__`` directory containing
    ``.pyc`` bytecode files next to each source file (standard CPython
    caching).
    """
    # Normalise input: accept either a single path or a list of paths.
    if isinstance(paths, str):
        path_list = [paths]
    else:
        try:
            path_list = list(paths)
        except TypeError:
            return f"Error: 'paths' must be a string or a list of strings, got {type(paths).__name__}."

    if not path_list:
        return "Error: 'paths' must contain at least one file path."

    results: list[str] = []
    ok_count = 0
    fail_count = 0

    for path in path_list:
        # Validate the path against the workspace. Per-file validation
        # lets the report cover the whole batch even if one entry is bad.
        try:
            target = validate_path(path)
        except ValueError as exc:
            results.append(f"  [SKIP] {path}: {exc}")
            continue

        if not target.exists():
            results.append(f"  [SKIP] {path}: file does not exist")
            continue

        try:
            py_compile.compile(str(target), doraise=True, quiet=1)
        except py_compile.PyCompileError as exc:
            results.append(f"  [FAIL] {path}: {exc.msg}")
            fail_count += 1
            logger.warning("Compile failed for %s: %s", path, exc.msg)
            continue
        except Exception as exc:                   # noqa: BLE001
            results.append(f"  [ERR]  {path}: {exc}")
            fail_count += 1
            logger.error("Unexpected error compiling %s: %s", path, exc)
            continue

        results.append(f"  [OK]   {path}")
        ok_count += 1
        logger.debug("Compiled OK: %s", path)

    total = len(path_list)
    header = f"Compile results ({total} file{'s' if total != 1 else ''}): {ok_count} OK, {fail_count} failed"
    logger.info("compile_python_file: %d/%d OK, %d failed", ok_count, total, fail_count)
    return header + "\n" + "\n".join(results)