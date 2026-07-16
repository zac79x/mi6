"""Interactive approval prompts."""

from __future__ import annotations


def request_continue_approval(max_iterations: int) -> bool:
    try:
        answer = input(
            f"Reached max_iterations={max_iterations} without a final answer. "
            f"Continue for another {max_iterations} iterations? [y/N]: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nNo interactive stdin available - stopping by default.")
        return False
    return answer in {"y", "yes"}
