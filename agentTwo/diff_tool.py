"""External diff tool configuration and diff-then-approve workflow.

This module owns:

* showing the user a diff (either with an external GUI tool or a
  built-in text fallback),
* prompting the user for approval before any change is persisted.

The diff-tool path is a hard-coded constant in :mod:`agentTwo.config`
(``DIFF_TOOL_PATH``).
"""

from __future__ import annotations

import difflib
import subprocess

from agentTwo.cli_ui import c as _colour, colours_enabled as _colours_enabled
from agentTwo.config import DIFF_TOOL_PATH
from agentTwo.logging_setup import logger


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
