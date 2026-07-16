"""Workspace path validation.

All file-mutating tools must route their target through
:func:`validate_path` to make sure they only touch files inside the
project workspace. Keeping that check in one place makes the rule
impossible to bypass accidentally.
"""

from __future__ import annotations

from pathlib import Path

from agentTwo.config import ALLOWED_EXTENSION, WORKSPACE_ROOT


def validate_path(path: str) -> Path:
    """Resolve ``path`` and confirm it stays inside ``WORKSPACE_ROOT``.

    Raises:
        ValueError: if the path is outside the workspace, does not end
            with one of the allowed extensions (see
            :data:`agentTwo.config.ALLOWED_EXTENSION`), or points to an
            existing non-file (e.g. a directory).
    """
    if not path.endswith(ALLOWED_EXTENSION):
        # ``ALLOWED_EXTENSION`` is a tuple of suffixes; render it as a
        # human-readable list in the error message rather than the raw
        # tuple repr.  Guard against a single-string value too, so the
        # message stays sensible if the constant is ever narrowed back
        # to one extension.
        if isinstance(ALLOWED_EXTENSION, tuple):
            allowed = ", ".join(ALLOWED_EXTENSION)
        else:
            allowed = str(ALLOWED_EXTENSION)
        raise ValueError(
            f"Refused: only {allowed} files are allowed, "
            f"got {path!r}."
        )

    candidate = (WORKSPACE_ROOT / path).resolve()

    try:
        candidate.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Refused: {path!r} resolves outside the workspace "
            f"{WORKSPACE_ROOT}."
        ) from exc

    if candidate.exists() and not candidate.is_file():
        raise ValueError(
            f"Refused: {candidate} exists and is not a regular file."
        )

    return candidate