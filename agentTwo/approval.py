"""Interactive approval prompts that are not specific to the diff tool.

Currently owns:

* :func:`request_continue_approval` - asking the user whether to keep
  the agent running after it hits ``max_iterations`` without a final
  answer.

This module exists so that :mod:`agentTwo.diff_tool` stays focused on its
name (diff rendering, external-tool configuration, diff-then-approve
workflow) rather than acting as a catch-all for every interactive
prompt.
"""

from __future__ import annotations


def request_continue_approval(max_iterations: int) -> bool:
    """Ask the user whether to keep going after max_iterations was reached."""
    try:
        answer = input(
            f"Reached max_iterations={max_iterations} without a final answer. "
            f"Continue for another {max_iterations} iterations? [y/N]: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nNo interactive stdin available - stopping by default.")
        return False
    return answer in {"y", "yes"}