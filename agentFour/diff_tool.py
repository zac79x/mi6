"""External diff tool configuration and diff-then-approve workflow."""

from __future__ import annotations

import difflib
import subprocess

from agentFour.cli_ui import c as _colour, colours_enabled as _colours_enabled
from agentFour.config import DIFF_TOOL_PATH
from agentFour.logging_setup import logger


def _diff_stats(old_lines: list[str], new_lines: list[str]) -> dict[str, int]:
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
    try:
        with open(original, encoding="utf-8") as fh:
            old_lines = fh.readlines()
        with open(modified, encoding="utf-8") as fh:
            new_lines = fh.readlines()
        diff = difflib.unified_diff(old_lines, new_lines, fromfile=original, tofile=modified, n=3)
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
        stats = _diff_stats(old_lines, new_lines)
        if stats["added"] or stats["removed"]:
            summary = f"  {stats['added']} added, {stats['removed']} removed, {stats['changed']} changed"
            print(_colour(summary, "dim", "gray") if use_colour else summary)
        return True
    except Exception as exc:
        print(f"Error producing text diff: {exc}")
        return False


def run_diff_tool(original: str, modified: str) -> bool:
    if not DIFF_TOOL_PATH:
        return print_text_diff(original, modified)
    try:
        result = subprocess.run([DIFF_TOOL_PATH, original, modified], timeout=300)
        logger.info("Diff tool '%s' exited with code %d", DIFF_TOOL_PATH, result.returncode)
        if result.returncode not in (0, 1):
            print(f"Warning: diff tool exited with code {result.returncode}. Falling back to text diff.")
            return print_text_diff(original, modified)
        return True
    except FileNotFoundError:
        print(f"Error: diff tool not found at {DIFF_TOOL_PATH!r}. Falling back to text diff.")
        return print_text_diff(original, modified)
    except subprocess.TimeoutExpired:
        print(f"Error: diff tool '{DIFF_TOOL_PATH}' timed out after 300s. Falling back to text diff.")
        return print_text_diff(original, modified)
    except Exception as exc:
        print(f"Error running diff tool {DIFF_TOOL_PATH!r}: {exc}")
        return print_text_diff(original, modified)


def request_update_approval(path: str) -> tuple[bool, str | None]:
    try:
        answer = input(f"Apply changes to {path}? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nNo interactive stdin available - refusing by default.")
        return (False, None)
    if answer in {"y", "yes"}:
        return (True, None)
    try:
        clarification = input("Changes rejected. What would you like changed? (press Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNo interactive stdin available - skipping clarification.")
        return (False, None)
    return (False, clarification or None)
