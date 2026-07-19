"""Orchestrator: benchmark two agentThree instances and compare performance.

The orchestrator is a **standalone** Python program that instantiates two
``agentThree`` instances (optionally with different models), runs a
configurable set of benchmark prompts against each one, and collects
performance metrics:

* **Token consumption** – prompt tokens, completion tokens, total tokens
* **LLM calls** – number of HTTP round-trips to Ollama
* **Turns** – number of user turns (one per ``chat()`` call)
* **Tool calls** – number of tool invocations across all turns
* **Wall-clock time** – total elapsed seconds per task and across all tasks

By default the orchestrator **asks the user interactively** for a task prompt
to run on both agents.  Use ``--default-tasks`` to run the built-in benchmark
suite instead, or pass one or more ``--task`` flags to specify tasks on the
command line.

Usage as a script::

    # Interactive: prompts the user for a task
    python -m orchestrator --model-a glm-5.2:cloud --model-b deepseek-v4-pro:cloud

    # Built-in suite
    python -m orchestrator --default-tasks --model-a glm-5.2:cloud --model-b deepseek-v4-pro:cloud

    # Custom tasks from the command line
    python -m orchestrator --model-a glm-5.2:cloud --model-b deepseek-v4-pro:cloud \
        --task "What is 2+2?" --task "Write a haiku"

    # Improve mode: benchmark, cross-improve, re-benchmark, apply or log
    python -m orchestrator --mode improve --default-tasks --loops 3

Or programmatically::

    from orchestrator.orchestrator import Orchestrator, BenchmarkTask

    tasks = [BenchmarkTask("What is 2+2?", "math")]
    orch = Orchestrator(model_a="glm-5.2:cloud", model_b="deepseek-v4-pro:cloud", tasks=tasks)
    report = orch.run()
    Orchestrator.print_report(report)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from agentThree.agent import agentThree
from agentThree.config import DEFAULT_MODEL, OLLAMA_URL
from agentThree.logging_setup import logger

try:
    from agentThree.cli_ui import c as _c, colours_enabled as _colours
except ImportError:
    def _c(text: str, *a: str) -> str:
        return text
    def _colours() -> bool:
        return False


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class BenchmarkTask:
    """A single prompt to benchmark."""
    prompt: str
    category: str = "general"
    label: str = ""
    max_iterations: int = 20

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.prompt[:60] + ("..." if len(self.prompt) > 60 else "")


@dataclass
class AgentMetrics:
    """Snapshot of agent performance for one task."""
    label: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    turns: int = 0
    tool_calls: int = 0
    elapsed_seconds: float = 0.0
    answer: str = ""
    error: str | None = None
    success: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskResult:
    """Result of running one task on both agents."""
    task: BenchmarkTask
    agent_a: AgentMetrics = field(default_factory=AgentMetrics)
    agent_b: AgentMetrics = field(default_factory=AgentMetrics)


@dataclass
class BenchmarkReport:
    """Full benchmark report."""
    model_a: str
    model_b: str
    task_count: int
    results: list[TaskResult] = field(default_factory=list)
    # Aggregate summaries
    summary_a: dict[str, Any] = field(default_factory=dict)
    summary_b: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "task_count": self.task_count,
            "results": [
                {
                    "task": asdict(r.task),
                    "agent_a": r.agent_a.as_dict(),
                    "agent_b": r.agent_b.as_dict(),
                }
                for r in self.results
            ],
            "summary_a": self.summary_a,
            "summary_b": self.summary_b,
        }


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

class Orchestrator:
    """Create two agentThree instances and benchmark them against tasks."""

    def __init__(
        self,
        model_a: str = DEFAULT_MODEL,
        model_b: str = DEFAULT_MODEL,
        tasks: list[BenchmarkTask] | None = None,
        ollama_url: str = OLLAMA_URL,
        temperature: float | None = None,
        max_iterations: int = 20,
        system_prompt: str | None = None,
        stream: bool = False,
        verbose: bool = False,
    ) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self.tasks = tasks or self._default_tasks()
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        self.stream = stream
        self.verbose = verbose

        # Lazy-create agents so __init__ doesn't fail if Ollama is down
        self._agent_a: agentThree | None = None
        self._agent_b: agentThree | None = None

    # ---- default benchmark suite ----------------------------------------

    @staticmethod
    def _default_tasks() -> list[BenchmarkTask]:
        return [
            BenchmarkTask(
                prompt="What is 17 multiplied by 23? Show your reasoning briefly.",
                category="math",
                label="basic_math",
            ),
            BenchmarkTask(
                prompt="List the first 10 Fibonacci numbers.",
                category="reasoning",
                label="fibonacci",
            ),
            BenchmarkTask(
                prompt="Write a haiku about the ocean.",
                category="creative",
                label="haiku",
            ),
            BenchmarkTask(
                prompt="Explain the difference between TCP and UDP in three sentences.",
                category="knowledge",
                label="tcp_udp",
            ),
            BenchmarkTask(
                prompt="Count the words in this sentence: 'The quick brown fox jumps over the lazy dog.'",
                category="tool_use",
                label="word_count_tool",
            ),
            BenchmarkTask(
                prompt="What time is it right now? Use the appropriate tool.",
                category="tool_use",
                label="time_tool",
            ),
        ]

    # ---- agent management -----------------------------------------------

    def _ensure_agents(self) -> None:
        if self._agent_a is None:
            self._agent_a = agentThree(
                model=self.model_a,
                ollama_url=self.ollama_url,
                max_iterations=self.max_iterations,
                temperature=self.temperature,
                verbose=self.verbose,
                show_thinking=False,
                stream=self.stream,
                system_prompt=self.system_prompt,
            )
        if self._agent_b is None:
            self._agent_b = agentThree(
                model=self.model_b,
                ollama_url=self.ollama_url,
                max_iterations=self.max_iterations,
                temperature=self.temperature,
                verbose=self.verbose,
                show_thinking=False,
                stream=self.stream,
                system_prompt=self.system_prompt,
            )

    @staticmethod
    def _count_tool_calls(messages: list[dict[str, Any]]) -> int:
        """Count tool-result messages (role == 'tool') in the history."""
        return sum(1 for m in messages if m.get("role") == "tool")

    @staticmethod
    def _count_turns(messages: list[dict[str, Any]]) -> int:
        """Count user turns (role == 'user') in the history."""
        return sum(1 for m in messages if m.get("role") == "user")

    def _run_task_on_agent(self, agent: agentThree, task: BenchmarkTask, model_name: str) -> AgentMetrics:
        """Run a single task on one agent and collect metrics."""
        agent.reset()
        agent.max_iterations = task.max_iterations or self.max_iterations

        metrics = AgentMetrics(label=task.label, model=model_name)

        t0 = time.time()
        try:
            answer = agent.chat(task.prompt)
            elapsed = time.time() - t0

            metrics.prompt_tokens = agent.session_prompt_tokens
            metrics.completion_tokens = agent.session_completion_tokens
            metrics.total_tokens = agent.session_total_tokens
            metrics.llm_calls = agent.llm_call_count
            metrics.turns = self._count_turns(agent.messages)
            metrics.tool_calls = self._count_tool_calls(agent.messages)
            metrics.elapsed_seconds = round(elapsed, 3)
            metrics.answer = answer[:500]  # truncate for the report
            metrics.success = not answer.startswith("[interrupted")
        except KeyboardInterrupt:
            elapsed = time.time() - t0
            metrics.elapsed_seconds = round(elapsed, 3)
            metrics.error = "interrupted"
            metrics.success = False
            raise
        except Exception as exc:
            elapsed = time.time() - t0
            metrics.elapsed_seconds = round(elapsed, 3)
            metrics.error = str(exc)
            metrics.success = False
            logger.exception("Task %r failed on model %s: %s", task.label, model_name, exc)

        return metrics

    # ---- main run -------------------------------------------------------

    def run(self) -> BenchmarkReport:
        """Run all tasks on both agents and return a BenchmarkReport."""
        self._ensure_agents()
        report = BenchmarkReport(
            model_a=self.model_a,
            model_b=self.model_b,
            task_count=len(self.tasks),
        )

        total = len(self.tasks)
        for idx, task in enumerate(self.tasks):
            print(f"\n{'='*70}")
            print(f"Task {idx+1}/{total}: {task.label} [{task.category}]")
            print(f"  Prompt: {task.prompt[:80]}{'...' if len(task.prompt) > 80 else ''}")
            print(f"{'='*70}")

            # Run on Agent A
            print(f"\n  [A] {self.model_a} ...", end=" ", flush=True)
            metrics_a = self._run_task_on_agent(self._agent_a, task, self.model_a)
            status_a = "OK" if metrics_a.success else "FAIL"
            print(f"done ({metrics_a.elapsed_seconds:.1f}s, {metrics_a.llm_calls} calls, "
                  f"{metrics_a.total_tokens} tokens) [{status_a}]")

            # Run on Agent B
            print(f"\n  [B] {self.model_b} ...", end=" ", flush=True)
            metrics_b = self._run_task_on_agent(self._agent_b, task, self.model_b)
            status_b = "OK" if metrics_b.success else "FAIL"
            print(f"done ({metrics_b.elapsed_seconds:.1f}s, {metrics_b.llm_calls} calls, "
                  f"{metrics_b.total_tokens} tokens) [{status_b}]")

            result = TaskResult(task=task, agent_a=metrics_a, agent_b=metrics_b)
            report.results.append(result)

        # Compute aggregates
        report.summary_a = self._compute_summary(report.results, "agent_a")
        report.summary_b = self._compute_summary(report.results, "agent_b")

        return report

    @staticmethod
    def _compute_summary(results: list[TaskResult], attr: str) -> dict[str, Any]:
        """Compute aggregate statistics for one agent across all tasks."""
        metrics_list = [getattr(r, attr) for r in results]
        n = len(metrics_list)
        if n == 0:
            return {}

        total_prompt = sum(m.prompt_tokens for m in metrics_list)
        total_completion = sum(m.completion_tokens for m in metrics_list)
        total_tokens = sum(m.total_tokens for m in metrics_list)
        total_calls = sum(m.llm_calls for m in metrics_list)
        total_turns = sum(m.turns for m in metrics_list)
        total_tools = sum(m.tool_calls for m in metrics_list)
        total_time = sum(m.elapsed_seconds for m in metrics_list)
        successes = sum(1 for m in metrics_list if m.success)

        return {
            "tasks_run": n,
            "tasks_succeeded": successes,
            "success_rate": round(successes / n * 100, 1) if n else 0,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "avg_tokens_per_task": round(total_tokens / n) if n else 0,
            "total_llm_calls": total_calls,
            "avg_calls_per_task": round(total_calls / n, 1) if n else 0,
            "total_turns": total_turns,
            "avg_turns_per_task": round(total_turns / n, 1) if n else 0,
            "total_tool_calls": total_tools,
            "avg_tool_calls_per_task": round(total_tools / n, 1) if n else 0,
            "total_time_seconds": round(total_time, 3),
            "avg_time_per_task": round(total_time / n, 3) if n else 0,
        }

    # ---- reporting ------------------------------------------------------

    @staticmethod
    def print_report(report: BenchmarkReport) -> None:
        """Print a human-readable comparison report to stdout."""
        coloured = _colours()

        def hdr(text: str) -> str:
            return _c(text, "bold", "bright_blue") if coloured else text

        def label_a(text: str) -> str:
            return _c(text, "cyan") if coloured else text

        def label_b(text: str) -> str:
            return _c(text, "magenta") if coloured else text

        def win(text: str) -> str:
            return _c(text, "bold", "green") if coloured else text

        def lose(text: str) -> str:
            return _c(text, "red") if coloured else text

        print(f"\n{'#'*70}")
        print(hdr("  BENCHMARK REPORT"))
        print(f"{'#'*70}")
        print(f"  Agent A: {label_a(report.model_a)}")
        print(f"  Agent B: {label_b(report.model_b)}")
        print(f"  Tasks:   {report.task_count}")
        print()

        # ---- Per-task table ----
        print(hdr("  Per-task results:"))
        print(f"  {'Task':<20} {'A tokens':>10} {'B tokens':>10}  {'A calls':>8} {'B calls':>8}  {'A time':>8} {'B time':>8}  {'A tools':>7} {'B tools':>7}  Winner")
        print(f"  {'-'*20} {'-'*10} {'-'*10}  {'-'*8} {'-'*8}  {'-'*8} {'-'*8}  {'-'*7} {'-'*7}  {'-'*10}")

        for r in report.results:
            a, b = r.agent_a, r.agent_b
            # Determine winner by total tokens (lower is better)
            if a.success and b.success:
                if a.total_tokens < b.total_tokens:
                    winner = label_a("A")
                elif b.total_tokens < a.total_tokens:
                    winner = label_b("B")
                else:
                    winner = "tie"
            elif a.success and not b.success:
                winner = label_a("A")
            elif b.success and not a.success:
                winner = label_b("B")
            else:
                winner = lose("both failed")

            a_toks = str(a.total_tokens) if a.success else "FAIL"
            b_toks = str(b.total_tokens) if b.success else "FAIL"
            a_calls = str(a.llm_calls) if a.success else "-"
            b_calls = str(b.llm_calls) if b.success else "-"
            a_time = f"{a.elapsed_seconds:.2f}s" if a.success else "-"
            b_time = f"{b.elapsed_seconds:.2f}s" if b.success else "-"
            a_tools = str(a.tool_calls) if a.success else "-"
            b_tools = str(b.tool_calls) if b.success else "-"

            print(f"  {a.label:<20} {a_toks:>10} {b_toks:>10}  {a_calls:>8} {b_calls:>8}  {a_time:>8} {b_time:>8}  {a_tools:>7} {b_tools:>7}  {winner}")

        # ---- Summary comparison ----
        sa, sb = report.summary_a, report.summary_b
        print(f"\n{'='*70}")
        print(hdr("  SUMMARY"))
        print(f"{'='*70}")
        print(f"  {'Metric':<28} {'Agent A':>14} {'Agent B':>14}  {'Better':>10}")
        print(f"  {'-'*28} {'-'*14} {'-'*14}  {'-'*10}")

        rows = [
            ("Tasks succeeded", sa.get("tasks_succeeded", 0), sb.get("tasks_succeeded", 0), "higher"),
            ("Success rate (%)", sa.get("success_rate", 0), sb.get("success_rate", 0), "higher"),
            ("Total prompt tokens", sa.get("total_prompt_tokens", 0), sb.get("total_prompt_tokens", 0), "lower"),
            ("Total completion tokens", sa.get("total_completion_tokens", 0), sb.get("total_completion_tokens", 0), "lower"),
            ("Total tokens", sa.get("total_tokens", 0), sb.get("total_tokens", 0), "lower"),
            ("Avg tokens/task", sa.get("avg_tokens_per_task", 0), sb.get("avg_tokens_per_task", 0), "lower"),
            ("Total LLM calls", sa.get("total_llm_calls", 0), sb.get("total_llm_calls", 0), "lower"),
            ("Avg calls/task", sa.get("avg_calls_per_task", 0), sb.get("avg_calls_per_task", 0), "lower"),
            ("Total turns", sa.get("total_turns", 0), sb.get("total_turns", 0), "lower"),
            ("Avg turns/task", sa.get("avg_turns_per_task", 0), sb.get("avg_turns_per_task", 0), "lower"),
            ("Total tool calls", sa.get("total_tool_calls", 0), sb.get("total_tool_calls", 0), "lower"),
            ("Avg tool calls/task", sa.get("avg_tool_calls_per_task", 0), sb.get("avg_tool_calls_per_task", 0), "lower"),
            ("Total time (s)", sa.get("total_time_seconds", 0), sb.get("total_time_seconds", 0), "lower"),
            ("Avg time/task (s)", sa.get("avg_time_per_task", 0), sb.get("avg_time_per_task", 0), "lower"),
        ]

        for name, va, vb, direction in rows:
            va_s = str(va)
            vb_s = str(vb)
            if direction == "lower":
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    if va < vb:
                        better = label_a("A")
                        va_s = win(va_s)
                    elif vb < va:
                        better = label_b("B")
                        vb_s = win(vb_s)
                    else:
                        better = "tie"
                else:
                    better = "-"
            else:  # higher
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    if va > vb:
                        better = label_a("A")
                        va_s = win(va_s)
                    elif vb > va:
                        better = label_b("B")
                        vb_s = win(vb_s)
                    else:
                        better = "tie"
                else:
                    better = "-"

            print(f"  {name:<28} {va_s:>14} {vb_s:>14}  {better:>10}")

        print()

    @staticmethod
    def save_report(report: BenchmarkReport, path: str) -> str:
        """Save the report as JSON."""
        abspath = os.path.abspath(path)
        with open(abspath, "w", encoding="utf-8") as fh:
            json.dump(report.as_dict(), fh, ensure_ascii=False, indent=2)
        logger.info("Benchmark report saved to %s", abspath)
        return abspath


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="orchestrator",
        description="Benchmark two agentThree instances and compare performance.",
    )
    p.add_argument("--model-a", default=DEFAULT_MODEL,
                   help=f"Model for Agent A (default: {DEFAULT_MODEL})")
    p.add_argument("--model-b", default=DEFAULT_MODEL,
                   help=f"Model for Agent B (default: {DEFAULT_MODEL})")
    p.add_argument("--ollama-url", default=OLLAMA_URL,
                   help=f"Ollama API URL (default: {OLLAMA_URL})")
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature for both agents (default: model default)")
    p.add_argument("--max-iter", type=int, default=20,
                   help="Max tool-calling iterations per task (default: 20)")
    p.add_argument("--stream", action="store_true", default=False,
                   help="Enable LLM streaming (default: off for clean benchmarks)")
    p.add_argument("--verbose", action="store_true", default=False,
                   help="Verbose agent output during benchmark")
    p.add_argument("--save", default="",
                   help="Save JSON report to this path")
    p.add_argument("--task", action="append", default=[],
                   help="Add a custom task prompt (can be repeated). "
                        "When given, skips the interactive prompt.")
    p.add_argument("--task-category", default="custom",
                   help="Category for custom tasks (default: custom)")
    p.add_argument("--default-tasks", action="store_true", default=False,
                   help="Run the built-in benchmark suite instead of prompting "
                        "interactively for a task.")
    p.add_argument("--mode", choices=["benchmark", "improve"], default="benchmark",
                   help="Run mode: 'benchmark' (default) runs a standard benchmark. "
                        "'improve' runs a benchmark, then each agent improves the "
                        "other's code, then re-benchmarks to decide if changes are kept.")
    p.add_argument("--loops", type=int, default=1,
                   help="Number of improve-mode loops (default: 1). "
                        "Only used in improve mode.")
    p.add_argument("--interactive", action="store_true", default=False,
                   help="Force the interactive prompt for a task (this is the "
                        "default when --task and --default-tasks are not given).")
    return p


def _print_banner(args: argparse.Namespace, task_count: int) -> None:
    """Print the orchestrator banner with model info."""
    print(f"{'='*70}")
    header = _c("  ORCHESTRATOR - agentThree Benchmark", "bold", "bright_blue") if _colours() else "  ORCHESTRATOR - agentThree Benchmark"
    print(header)
    print(f"{'='*70}")
    print(f"  Agent A: {_c(args.model_a, 'cyan') if _colours() else args.model_a}")
    print(f"  Agent B: {_c(args.model_b, 'magenta') if _colours() else args.model_b}")
    print(f"  Tasks:   {task_count}")
    print(f"  URL:     {args.ollama_url}")
    print()


def _prompt_for_models(args: argparse.Namespace) -> argparse.Namespace:
    """Interactively ask the user which models to use for Agent A and Agent B.

    Called only when neither --model-a nor --model-b was passed on the CLI.
    Offers the list of known models with sensible defaults.
    """
    from agentThree.config import available_models

    models = available_models()
    # Suggested defaults for the benchmark
    default_a = "glm-5.2:cloud"
    default_b = "deepseek-v4-pro:cloud"

    # Make sure the suggested defaults are in the list even if the
    # Ollama server didn't report them (cloud models may not appear).
    for m in (default_a, default_b):
        if m not in models:
            models.append(m)
    models = sorted(models)

    print(f"{'='*70}")
    header = _c("  ORCHESTRATOR - Model Selection", "bold", "bright_blue") if _colours() else "  ORCHESTRATOR - Model Selection"
    print(header)
    print(f"{'='*70}")
    print()
    print("  Available models:")
    for i, m in enumerate(models, 1):
        marker = ""
        if m == default_a:
            marker = _c("  <- Agent A default", "dim") if _colours() else "  <- Agent A default"
        elif m == default_b:
            marker = _c("  <- Agent B default", "dim") if _colours() else "  <- Agent B default"
        print(f"    {i:>2}. {m}{marker}")
    print()

    # Agent A
    while True:
        hint = _c(f" [{default_a}]", "dim") if _colours() else f" [{default_a}]"
        raw = input(f"  Agent A model{hint}: ").strip()
        if not raw:
            args.model_a = default_a
            break
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            args.model_a = models[int(raw) - 1]
            break
        if raw in models:
            args.model_a = raw
            break
        print(f"  Unknown model '{raw}'. Type a name from the list or a number 1-{len(models)}.")

    # Agent B
    while True:
        hint = _c(f" [{default_b}]", "dim") if _colours() else f" [{default_b}]"
        raw = input(f"  Agent B model{hint}: ").strip()
        if not raw:
            args.model_b = default_b
            break
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            args.model_b = models[int(raw) - 1]
            break
        if raw in models:
            args.model_b = raw
            break
        print(f"  Unknown model '{raw}'. Type a name from the list or a number 1-{len(models)}.")

    print()
    print(f"  Agent A: {_c(args.model_a, 'cyan') if _colours() else args.model_a}")
    print(f"  Agent B: {_c(args.model_b, 'magenta') if _colours() else args.model_b}")
    print()
    return args


def _prompt_for_tasks(args: argparse.Namespace) -> list[BenchmarkTask]:
    """Interactively ask the user for one or more task prompts.

    The user can enter multiple tasks one at a time.  Typing 'default' runs
    the built-in suite.  An empty line finishes input (or quits if no task
    was entered yet).
    """
    print(f"{'='*70}")
    header = _c("  ORCHESTRATOR - agentThree Benchmark", "bold", "bright_blue") if _colours() else "  ORCHESTRATOR - agentThree Benchmark"
    print(header)
    print(f"{'='*70}")
    print(f"  Agent A: {_c(args.model_a, 'cyan') if _colours() else args.model_a}")
    print(f"  Agent B: {_c(args.model_b, 'magenta') if _colours() else args.model_b}")
    print(f"  URL:     {args.ollama_url}")
    print()
    print("  Enter task prompts for the agents to run.")
    print("  Type 'default' to use the built-in benchmark suite.")
    print("  Press Enter on an empty line to finish (or to quit if no task entered yet).")
    print()

    tasks: list[BenchmarkTask] = []
    while True:
        try:
            prompt_label = f"  Task {len(tasks)+1}" if tasks else "  Task"
            hint = _c(" (or 'default', or Enter to finish)", "dim") if _colours() else " (or 'default', or Enter to finish)"
            raw = input(f"{prompt_label}{hint}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            break
        if raw.lower() == "default":
            tasks = Orchestrator._default_tasks()
            print(f"  -> Using built-in benchmark suite ({len(tasks)} tasks).")
            break

        category = args.task_category if args.task_category != "custom" else "interactive"
        task = BenchmarkTask(prompt=raw, category=category)
        tasks.append(task)
        print(f"  -> Added task: {task.label}")

    if not tasks:
        print("  No task provided. Exiting.")
        sys.exit(0)

    return tasks


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    # Interactive model selection when neither --model-a nor --model-b
    # was passed on the command line.
    model_a_passed = any(a in ("--model-a", "-ma") for a in sys.argv)
    model_b_passed = any(a in ("--model-b", "-mb") for a in sys.argv)
    if not model_a_passed and not model_b_passed:
        args = _prompt_for_models(args)

    # Build task list -- three modes:
    #   1. --task flags given         -> use those tasks directly
    #   2. --default-tasks flag       -> built-in suite
    #   3. neither (default)          -> interactive prompt
    if args.task:
        tasks = [BenchmarkTask(prompt=p, category=args.task_category) for p in args.task]
        _print_banner(args, len(tasks))
    elif args.default_tasks:
        tasks = Orchestrator._default_tasks()
        _print_banner(args, len(tasks))
    else:
        tasks = _prompt_for_tasks(args)

    # Dispatch based on mode
    if args.mode == "improve":
        from orchestrator.improve_mode import ImproveMode
        improve = ImproveMode(
            model_a=args.model_a,
            model_b=args.model_b,
            tasks=tasks,
            ollama_url=args.ollama_url,
            temperature=args.temperature,
            max_iterations=args.max_iter,
            stream=args.stream,
            verbose=args.verbose,
            loops=args.loops,
        )
        try:
            improve.run()
        except KeyboardInterrupt:
            print("\n\nImprove mode interrupted by user.")
            sys.exit(1)
    else:
        orch = Orchestrator(
            model_a=args.model_a,
            model_b=args.model_b,
            tasks=tasks,
            ollama_url=args.ollama_url,
            temperature=args.temperature,
            max_iterations=args.max_iter,
            stream=args.stream,
            verbose=args.verbose,
        )

        try:
            report = orch.run()
        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user.")
            sys.exit(1)

        Orchestrator.print_report(report)

        if args.save:
            saved_to = Orchestrator.save_report(report, args.save)
            print(f"\nReport saved to: {_c(saved_to, 'green') if _colours() else saved_to}")


if __name__ == "__main__":
    main()