"""Filesystem-oriented tools."""

from __future__ import annotations

import os
import py_compile
import tempfile
from typing import Union

from agentThree.diff_tool import run_diff_tool, request_update_approval
from agentThree.logging_setup import logger
from agentThree.safety import validate_path
from agentThree.tools_registry import tool


@tool
def read_text_file(path: str, max_chars: int = 40000) -> str:
    """Read a UTF-8 text file. Truncates very long files to ``max_chars``."""
    if not os.path.exists(path):
        return f"Error: file '{path}' does not exist"
    try:
        max_chars = int(max_chars) if max_chars is not None else 40000
        with open(path, "r", encoding="utf-8") as fh:
            data = fh.read(max_chars + 1)
        if len(data) > max_chars:
            data = data[:max_chars] + f"\n... [truncated, file is longer than {max_chars} chars]"
        return data
    except Exception as exc:
        return f"Error reading file: {exc}"


@tool
def read_file_lines(path: str, start_line: int = 0, num_lines: int = 2000) -> str:
    """Read a slice of lines from a text file."""
    if not os.path.exists(path):
        return f"Error: file '{path}' does not exist"
    if not os.path.isfile(path):
        return f"Error: path '{path}' is not a regular file"
    try:
        start_line = max(0, int(start_line) if start_line is not None else 0)
        num_lines = max(0, int(num_lines) if num_lines is not None else 2000)
        with open(path, "r", encoding="utf-8") as fh:
            for _ in range(start_line):
                next(fh, None)
            chunk = [fh.readline() for _ in range(num_lines)]
            chunk = [l for l in chunk if l]
        if not chunk:
            return f"No content read from '{path}' (start_line={start_line}, num_lines={num_lines})."
        return f"Lines {start_line}..{start_line + len(chunk) - 1} of '{path}':\n" + "".join(chunk)
    except UnicodeDecodeError:
        return f"Error: '{path}' is not a valid UTF-8 text file"
    except PermissionError:
        return f"Error: permission denied to read '{path}'"
    except OSError as exc:
        return f"Error reading file: {exc}"


@tool
def list_directory(path: str = ".", show_hidden: bool = False) -> str:
    """List files and directories in the given path."""
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist."
    if not os.path.isdir(path):
        return f"Error: Path '{path}' is not a directory."
    try:
        entries = [e for e in os.listdir(path) if show_hidden or not e.startswith(".")]
        if not entries:
            return f"Directory '{path}' is empty."
        entries.sort(key=str.lower)
        lines = [f"Contents of '{os.path.abspath(path)}':"]
        for entry in entries:
            full = os.path.join(path, entry)
            try:
                if os.path.isdir(full):
                    lines.append(f"  [DIR]  {entry}/")
                elif os.path.isfile(full):
                    lines.append(f"  [FILE] {entry}  ({os.path.getsize(full):,} bytes)")
                elif os.path.islink(full):
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
    """Check whether a path exists and report what kind of filesystem object it is."""
    if not os.path.exists(path):
        return f"Path '{path}' does not exist."
    if os.path.isdir(path):
        return f"Path '{path}' exists and is a directory."
    if os.path.isfile(path):
        return f"Path '{path}' exists and is a file ({os.path.getsize(path):,} bytes)."
    return f"Path '{path}' exists but is neither a regular file nor a directory."


@tool
def create_file(path: str, content: str) -> dict:
    """Create a NEW file. Does NOT require user approval."""
    try:
        target = validate_path(path)
    except ValueError as exc:
        return {"ok": False, "action": "refused", "path": path, "reason": str(exc)}
    if target.exists():
        return {"ok": False, "action": "refused", "path": path, "reason": "File already exists; use update_file to modify it."}
    try:
        target.write_text(content, encoding="utf-8")
        logger.info("Created file: %s (%d bytes)", path, len(content))
        return {"ok": True, "action": "create", "path": path, "reason": None}
    except OSError as exc:
        return {"ok": False, "action": "refused", "path": path, "reason": str(exc)}


@tool
def update_file(path: str, content: str) -> dict:
    """Update an EXISTING file after showing the user a diff."""
    try:
        target = validate_path(path)
    except ValueError as exc:
        return {"ok": False, "action": "refused", "path": path, "reason": str(exc)}
    if not target.exists():
        return {"ok": False, "action": "refused", "path": path, "reason": "File does not exist; use create_file to create a new file."}
    try:
        existing = target.read_text(encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "action": "refused", "path": path, "reason": f"Could not read existing file: {exc}"}
    if existing == content:
        logger.info("update_file: no changes for %s", path)
        return {"ok": True, "action": "noop", "path": path, "reason": "No changes - content is identical."}

    original_path = str(target)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=f".{target.name}.", suffix=target.suffix + ".tmp", dir=str(target.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_fh:
            tmp_fh.write(content)
        print("=" * 60)
        print(f"PROPOSED UPDATE: {path}")
        print(f"Temp file:       {tmp_path}")
        print("Launching diff tool for review - close it when you are done...")
        print("=" * 60)
        if not run_diff_tool(original_path, tmp_path):
            return {"ok": False, "action": "refused", "path": path, "reason": "Diff tool failed; not applying changes."}
        approved, clarification = request_update_approval(path)
        if not approved:
            reason = f"User did not approve the change." + (f" User clarification: {clarification}" if clarification else "")
            return {"ok": False, "action": "refused", "path": path, "reason": reason}
        target.write_text(content, encoding="utf-8")
        logger.info("Updated file: %s (%d bytes)", path, len(content))
        return {"ok": True, "action": "update", "path": path, "reason": None}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


_COMPILE_PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "One or more workspace-relative ``.py`` paths.",
        }
    },
    "required": ["paths"],
}


@tool(name="compile_python_file", description="Compile one or more Python scripts to check for syntax errors without executing them.", parameters=_COMPILE_PARAMETERS)
def compile_python_file(paths: Union[str, list]) -> str:
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
    ok_count = fail_count = 0
    for path in path_list:
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
            continue
        except Exception as exc:
            results.append(f"  [ERR]  {path}: {exc}")
            fail_count += 1
            continue
        results.append(f"  [OK]   {path}")
        ok_count += 1

    total = len(path_list)
    header = f"Compile results ({total} file{'s' if total != 1 else ''}): {ok_count} OK, {fail_count} failed"
    return header + "\n" + "\n".join(results)