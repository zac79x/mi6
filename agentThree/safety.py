"""Workspace path validation for file- and directory-mutating tools."""

from __future__ import annotations

from pathlib import Path

from agentThree.config import ALLOWED_EXTENSION, WORKSPACE_ROOT


def validate_path(path: str) -> Path:
    if not path.endswith(ALLOWED_EXTENSION):
        allowed = ", ".join(ALLOWED_EXTENSION) if isinstance(ALLOWED_EXTENSION, tuple) else str(ALLOWED_EXTENSION)
        raise ValueError(f"Refused: only {allowed} files are allowed, got {path!r}.")

    candidate = (WORKSPACE_ROOT / path).resolve()
    try:
        candidate.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError(f"Refused: {path!r} resolves outside the workspace {WORKSPACE_ROOT}.") from exc

    if candidate.exists() and not candidate.is_file():
        raise ValueError(f"Refused: {candidate} exists and is not a regular file.")

    return candidate


def validate_directory_path(path: str) -> Path:
    """Validate that ``path`` stays inside the workspace and is (or will
    be) a directory.  No extension check is performed.

    Raises ``ValueError`` if the path escapes the workspace or already
    exists as a non-directory.
    """
    candidate = (WORKSPACE_ROOT / path).resolve()
    try:
        candidate.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError(f"Refused: {path!r} resolves outside the workspace {WORKSPACE_ROOT}.") from exc

    if candidate.exists() and not candidate.is_dir():
        raise ValueError(f"Refused: {candidate} exists and is not a directory.")

    return candidate
