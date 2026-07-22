"""Improve mode: benchmark → cross-improve → re-benchmark → apply or log.

This module implements the "improve" mode for the orchestrator.  The flow:

1. Run a baseline benchmark (Agent A vs Agent B).
2. Ask Agent A to propose code improvements to Agent B's source files.
3. Ask Agent B to propose code improvements to Agent A's source files.
4. Apply Agent A's proposed changes for B, and Agent B's proposed changes
   for A, to temporary copies.
5. Re-run the benchmark with the improved agents.
6. Compare: if **both** improved agents score better than their originals,
   apply the changes to the real source files (A's changes to B first,
   then B's changes to A in the next loop iteration).
7. If no improvement, do not apply any changes.  Log the proposals and
   benchmark results to an improvement log file.

The "score" for comparison is a composite: lower total tokens, higher
success rate, lower total time.  A simple weighted sum is used.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from agentThree.agent import agentThree
from agentThree.config import OLLAMA_URL
from agentThree.logging_setup import logger

from orchestrator.orchestrator import (
    BenchmarkReport,
    BenchmarkTask,
    Orchestrator,
    AgentMetrics,
    TaskResult,
)

try:
    from agentThree.cli_ui import c as _c, colours_enabled as _colours
except ImportError:
    def _c(text: str, *a: str) -> str:
        return text
    def _colours() -> bool:
        return False


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

#: Root of the mi6 project (parent of the orchestrator package).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: The agentThree package directory.
_AGENT_THREE_DIR = _PROJECT_ROOT / "agentThree"

#: Directory for improvement logs.
_LOG_DIR = _PROJECT_ROOT / "orchestrator" / "improve_logs"

#: The source files of agentThree that agents are allowed to improve.
#: We focus on the core logic files, not logs or __pycache__.
_IMPROVABLE_FILES = [
    "agent.py",
    "config.py",
    "tools_filesystem.py",
    "tools_misc.py",
    "tools_registry.py",
    "tools_web.py",
    "a2a_client.py",
    "a2a_models.py",
    "a2a_server.py",
    "a2a_tools.py",
    "cli_ui.py",
    "diff_tool.py",
    "logging_setup.py",
    "repl.py",
    "safety.py",
    "approval.py",
    "_check_path.py",
]

#: Files that are excluded from improvement (logs, caches, __init__, __main__).
_EXCLUDED_FILES = {
    "__init__.py",
    "__main__.py",
    "agent.log",
    "agent.http.log",
    "agent.llm_payload.log",
    "readme.md",
}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _score(summary: dict[str, Any]) -> float:
    """Compute a composite score for a benchmark summary.

    Lower is better.  Weights:
      - Total tokens: 1.0 per token
      - Total time: 100 per second
      - Failed tasks: 10000 per failure
    """
    total_tokens = summary.get("total_tokens", 0)
    total_time = summary.get("total_time_seconds", 0.0)
    tasks_run = summary.get("tasks_run", 1)
    tasks_succeeded = summary.get("tasks_succeeded", tasks_run)
    failures = tasks_run - tasks_succeeded

    return float(total_tokens) + 100.0 * float(total_time) + 10000.0 * float(failures)


def _is_improvement(old_score: float, new_score: float) -> bool:
    """Return True if new_score is strictly better (lower) than old_score."""
    return new_score < old_score


# --------------------------------------------------------------------------- #
# Backup / restore
# --------------------------------------------------------------------------- #

def _backup_agent_files(dest_dir: Path) -> dict[str, Path]:
    """Copy all improvable agentThree source files to dest_dir.

    Returns a mapping of relative filename -> backup path.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    backups: dict[str, Path] = {}
    for fname in _IMPROVABLE_FILES:
        src = _AGENT_THREE_DIR / fname
        if src.exists() and src.is_file():
            dst = dest_dir / fname
            shutil.copy2(src, dst)
            backups[fname] = dst
    return backups


def _restore_agent_files(backups: dict[str, Path]) -> None:
    """Restore agentThree source files from backups."""
    for fname, src in backups.items():
        dst = _AGENT_THREE_DIR / fname
        shutil.copy2(src, dst)


def _apply_changes(changes: dict[str, str]) -> None:
    """Apply proposed file changes directly to the agentThree source.

    ``changes`` maps relative filename -> new file content.
    """
    for fname, content in changes.items():
        dst = _AGENT_THREE_DIR / fname
        dst.write_text(content, encoding="utf-8")
        logger.info("Applied improved version of %s", fname)


# --------------------------------------------------------------------------- #
# Improvement prompts
# --------------------------------------------------------------------------- #

_IMPROVE_SYSTEM_PROMPT = (
    "You are a code optimization expert. You have access to file tools to read "
    "and modify source code. Your task is to improve the code of an AI agent "
    "system to make it more efficient — reduce token consumption, reduce LLM "
    "round-trips, and improve response speed — while preserving all existing "
    "functionality.\n\n"
    "Rules:\n"
    "1. Read the readme.md first to understand the project structure.\n"
    "2. Read the source files you want to improve.\n"
    "3. Make targeted, minimal changes. Do not rewrite entire files unless necessary.\n"
    "4. Preserve all public APIs and tool signatures.\n"
    "5. Focus on: reducing unnecessary LLM calls, smarter caching, more efficient "
    "prompt construction, reducing redundant tool calls, faster response paths.\n"
    "6. Use update_file to apply each change. Your changes will be tested.\n"
    "7. Do NOT modify log files, __pycache__, or __init__.py / __main__.py.\n"
)


def _build_improve_prompt(target_label: str, baseline_report: BenchmarkReport) -> str:
    """Build the prompt asking an agent to improve the other agent's code.

    ``target_label`` is "A" or "B" — the agent whose code should be improved.
    """
    # Summarise the baseline benchmark for context
    summary = baseline_report.summary_a if target_label == "A" else baseline_report.summary_b
    model = baseline_report.model_a if target_label == "A" else baseline_report.model_b

    lines = [
        f"Please improve the source code of the agentThree package to make Agent {target_label} ({model}) more efficient.",
        "",
        "Here are the current benchmark results for this agent:",
        json.dumps(summary, indent=2),
        "",
        "The agentThree source code is in the 'agentThree/' directory. Start by reading",
        "'agentThree/readme.md' to understand the project structure, then read the source",
        "files and propose improvements.",
        "",
        "Focus on reducing token consumption and LLM calls while preserving functionality.",
        "Apply your changes using the update_file tool. After making changes, briefly",
        "summarise what you changed and why.",
        "",
        f"The files you may modify are: {', '.join(_IMPROVABLE_FILES)}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Extract changes from agent conversation
# --------------------------------------------------------------------------- #

def _extract_applied_changes(agent: agentThree) -> dict[str, str]:
    """Inspect the agent's tool-call history to find which files were modified.

    Returns a mapping of relative filename -> new content (read back from disk
    after the agent applied its changes via update_file / create_file).
    """
    changed_files: set[str] = set()
    for msg in agent.messages:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        # The update_file / create_file tools return a dict that gets
        # str()'d by _execute_tool_call.  That may produce either valid JSON
        # (double quotes) or a Python repr (single quotes).  Try JSON first,
        # fall back to ast.literal_eval for Python dict reprs.
        try:
            result = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            try:
                import ast
                result = ast.literal_eval(content)
            except (ValueError, SyntaxError, TypeError):
                continue
        if isinstance(result, dict) and result.get("ok"):
            path = result.get("path", "")
            if path:
                fname = os.path.basename(path)
                if fname in _IMPROVABLE_FILES:
                    changed_files.add(fname)

    # Read back the current content of changed files
    changes: dict[str, str] = {}
    for fname in changed_files:
        fpath = _AGENT_THREE_DIR / fname
        if fpath.exists():
            changes[fname] = fpath.read_text(encoding="utf-8")
    return changes


# --------------------------------------------------------------------------- #
# Improvement log
# --------------------------------------------------------------------------- #

def _write_improvement_log(
    loop: int,
    baseline: BenchmarkReport,
    post: BenchmarkReport,
    changes_a: dict[str, str],
    changes_b: dict[str, str],
    answer_a: str,
    answer_b: str,
    applied: bool,
    reason: str,
) -> str:
    """Write an improvement log entry as JSON. Returns the log file path."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"improve_{timestamp}_loop{loop}.json"
    log_path = _LOG_DIR / filename

    entry = {
        "timestamp": timestamp,
        "loop": loop,
        "applied": applied,
        "reason": reason,
        "baseline": {
            "model_a": baseline.model_a,
            "model_b": baseline.model_b,
            "summary_a": baseline.summary_a,
            "summary_b": baseline.summary_b,
        },
        "post_improvement": {
            "model_a": post.model_a,
            "model_b": post.model_b,
            "summary_a": post.summary_a,
            "summary_b": post.summary_b,
        },
        "scores": {
            "baseline_a": _score(baseline.summary_a),
            "baseline_b": _score(baseline.summary_b),
            "post_a": _score(post.summary_a),
            "post_b": _score(post.summary_b),
        },
        "proposed_changes_a_to_b": {
            fname: len(content) for fname, content in changes_a.items()
        },
        "proposed_changes_b_to_a": {
            fname: len(content) for fname, content in changes_b.items()
        },
        "answer_a": answer_a[:2000],
        "answer_b": answer_b[:2000],
    }

    log_path.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Improvement log written to %s", log_path)
    return str(log_path)


# --------------------------------------------------------------------------- #
# Improve mode runner
# --------------------------------------------------------------------------- #

class ImproveMode:
    """Run the improve-mode loop.

    Parameters mirror those of ``Orchestrator``.
    """

    def __init__(
        self,
        model_a: str,
        model_b: str,
        tasks: list[BenchmarkTask],
        ollama_url: str = OLLAMA_URL,
        temperature: float | None = None,
        max_iterations: int = 20,
        stream: bool = False,
        verbose: bool = False,
        loops: int = 1,
    ) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self.tasks = tasks
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.stream = stream
        self.verbose = verbose
        self.loops = max(1, loops)

    def _make_orchestrator(self) -> Orchestrator:
        return Orchestrator(
            model_a=self.model_a,
            model_b=self.model_b,
            tasks=self.tasks,
            ollama_url=self.ollama_url,
            temperature=self.temperature,
            max_iterations=self.max_iterations,
            stream=self.stream,
            verbose=self.verbose,
        )

    def _make_improver_agent(self, model: str) -> agentThree:
        """Create an agentThree instance configured for code improvement."""
        return agentThree(
            model=model,
            ollama_url=self.ollama_url,
            max_iterations=self.max_iterations,
            temperature=self.temperature,
            verbose=self.verbose,
            show_thinking=False,
            stream=self.stream,
            system_prompt=_IMPROVE_SYSTEM_PROMPT,
        )

    def run(self) -> None:
        """Execute the improve-mode flow for the configured number of loops."""
        coloured = _colours()

        def hdr(text: str) -> str:
            return _c(text, "bold", "bright_blue") if coloured else text

        print(f"\n{'#'*70}")
        print(hdr("  IMPROVE MODE"))
        print(f"{'#'*70}")
        print(f"  Agent A: {_c(self.model_a, 'cyan') if coloured else self.model_a}")
        print(f"  Agent B: {_c(self.model_b, 'magenta') if coloured else self.model_b}")
        print(f"  Loops:   {self.loops}")
        print(f"  Tasks:   {len(self.tasks)}")
        print()

        for loop in range(1, self.loops + 1):
            print(f"\n{'='*70}")
            print(hdr(f"  LOOP {loop}/{self.loops}"))
            print(f"{'='*70}")

            # --- Step 1: Baseline benchmark ---
            print(f"\n  [{_c('1', 'bold', 'green') if coloured else '1'}] Running baseline benchmark...")
            orch = self._make_orchestrator()
            baseline = orch.run()
            Orchestrator.print_report(baseline)

            score_a_baseline = _score(baseline.summary_a)
            score_b_baseline = _score(baseline.summary_b)

            print(f"\n  Baseline scores: A={score_a_baseline:.1f}  B={score_b_baseline:.1f}")

            # --- Step 2: Agent A improves Agent B ---
            print(f"\n  [{_c('2', 'bold', 'green') if coloured else '2'}] Agent A is improving Agent B's code...")
            backup_b = _backup_agent_files(Path(tempfile.mkdtemp(prefix="improve_b_")))

            improver_a = self._make_improver_agent(self.model_a)
            prompt_a = _build_improve_prompt("B", baseline)
            try:
                answer_a = improver_a.chat(prompt_a)
            except KeyboardInterrupt:
                print("\n  Interrupted during Agent A improvement.")
                _restore_agent_files(backup_b)
                shutil.rmtree(backup_b.get(next(iter(backup_b), ""), Path()).parent, ignore_errors=True)
                return

            changes_a = _extract_applied_changes(improver_a)
            print(f"\n  Agent A proposed changes to {len(changes_a)} file(s): {', '.join(changes_a.keys()) or 'none'}")
            print(f"\n  Agent A summary:\n  {answer_a[:500]}")

            if not changes_a:
                print("\n  Agent A made no code changes. Skipping Agent B improvement and re-benchmark.")
                _write_improvement_log(
                    loop, baseline, baseline, {}, {}, answer_a, "", False, "Agent A proposed no changes."
                )
                continue

            # --- Step 3: Agent B improves Agent A ---
            # Note: at this point Agent A's changes to B's code are on disk.
            # We need to back up A's code before Agent B modifies it.
            print(f"\n  [{_c('3', 'bold', 'green') if coloured else '3'}] Agent B is improving Agent A's code...")
            backup_a = _backup_agent_files(Path(tempfile.mkdtemp(prefix="improve_a_")))

            improver_b = self._make_improver_agent(self.model_b)
            prompt_b = _build_improve_prompt("A", baseline)
            try:
                answer_b = improver_b.chat(prompt_b)
            except KeyboardInterrupt:
                print("\n  Interrupted during Agent B improvement.")
                _restore_agent_files(backup_a)
                _restore_agent_files(backup_b)
                return

            changes_b = _extract_applied_changes(improver_b)
            print(f"\n  Agent B proposed changes to {len(changes_b)} file(s): {', '.join(changes_b.keys()) or 'none'}")
            print(f"\n  Agent B summary:\n  {answer_b[:500]}")

            if not changes_b:
                print("\n  Agent B made no code changes. Reverting Agent A's changes and logging.")
                _restore_agent_files(backup_b)
                _restore_agent_files(backup_a)
                _write_improvement_log(
                    loop, baseline, baseline, changes_a, {}, answer_a, answer_b,
                    False, "Agent B proposed no changes."
                )
                continue

            # --- Step 4: Re-benchmark with improved code ---
            print(f"\n  [{_c('4', 'bold', 'green') if coloured else '4'}] Running post-improvement benchmark...")
            orch2 = self._make_orchestrator()
            post = orch2.run()
            Orchestrator.print_report(post)

            score_a_post = _score(post.summary_a)
            score_b_post = _score(post.summary_b)

            print(f"\n  Post-improvement scores: A={score_a_post:.1f}  B={score_b_post:.1f}")
            print(f"  Baseline scores:         A={score_a_baseline:.1f}  B={score_b_baseline:.1f}")

            # --- Step 5: Decide ---
            improved_a = _is_improvement(score_a_baseline, score_a_post)
            improved_b = _is_improvement(score_b_baseline, score_b_post)

            if improved_a and improved_b:
                print(f"\n  {_c('✓ BOTH agents improved!', 'bold', 'green') if coloured else '✓ BOTH agents improved!'}")
                print(f"  A: {score_a_baseline:.1f} → {score_a_post:.1f}  B: {score_b_baseline:.1f} → {score_b_post:.1f}")
                print(f"\n  Keeping changes. Agent A's improvements to B are already applied.")
                print(f"  Agent B's improvements to A are already applied.")
                _write_improvement_log(
                    loop, baseline, post, changes_a, changes_b,
                    answer_a, answer_b, True,
                    f"Both improved: A {score_a_baseline:.1f}→{score_a_post:.1f}, B {score_b_baseline:.1f}→{score_b_post:.1f}"
                )
                # Clean up backups — changes are kept
                for backup in (backup_a, backup_b):
                    d = next(iter(backup.values()), None)
                    if d:
                        shutil.rmtree(d.parent, ignore_errors=True)
            else:
                print(f"\n  {_c('✗ No improvement (or only one side improved).', 'red') if coloured else '✗ No improvement (or only one side improved).'}")
                if not improved_a:
                    print(f"  Agent A: {score_a_baseline:.1f} → {score_a_post:.1f} (no improvement)")
                if not improved_b:
                    print(f"  Agent B: {score_b_baseline:.1f} → {score_b_post:.1f} (no improvement)")
                print(f"\n  Reverting all changes to original state.")
                _restore_agent_files(backup_a)
                _restore_agent_files(backup_b)
                log_path = _write_improvement_log(
                    loop, baseline, post, changes_a, changes_b,
                    answer_a, answer_b, False,
                    f"No improvement: A {'improved' if improved_a else 'did not improve'}, "
                    f"B {'improved' if improved_b else 'did not improve'}"
                )
                print(f"\n  Improvement log saved to: {_c(log_path, 'green') if coloured else log_path}")
                # Clean up backups
                for backup in (backup_a, backup_b):
                    d = next(iter(backup.values()), None)
                    if d:
                        shutil.rmtree(d.parent, ignore_errors=True)

        print(f"\n{'#'*70}")
        print(hdr("  IMPROVE MODE COMPLETE"))
        print(f"{'#'*70}\n")