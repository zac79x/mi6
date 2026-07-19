"""Orchestrator package: benchmark two agentThree instances and compare performance.

This is a standalone Python program that creates two ``agentThree`` instances
(potentially with different models), runs a configurable set of benchmark
prompts against each, and collects performance metrics:

* Token consumption (prompt / completion / total)
* LLM calls (HTTP round-trips to the model endpoint)
* Turns (user→agent turns in the conversation)
* Tool calls (tool invocations across all turns)
* Wall-clock time (total elapsed seconds per task and across all tasks)

Usage as a script::

    python -m orchestrator --model-a glm-5.2:cloud --model-b deepseek-v4-pro:cloud

Or programmatically::

    from orchestrator.orchestrator import Orchestrator, BenchmarkTask

    tasks = [BenchmarkTask("What is 2+2?", "math")]
    orch = Orchestrator(model_a="glm-5.2:cloud", model_b="kimi-k2.7-code:cloud", tasks=tasks)
    report = orch.run()
    orch.print_report(report)
"""

from orchestrator.orchestrator import Orchestrator, BenchmarkTask, BenchmarkReport, AgentMetrics, TaskResult

__all__ = [
    "Orchestrator",
    "BenchmarkTask",
    "BenchmarkReport",
    "AgentMetrics",
    "TaskResult",
]