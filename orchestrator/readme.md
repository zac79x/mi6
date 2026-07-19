# orchestrator

A standalone benchmarking orchestrator that instantiates two `agentThree`
instances (optionally with different models), runs a configurable set of
benchmark prompts against each one, and compares their performance.

It supports two modes:

- **`benchmark`** (default): run benchmarks and print/save a report.
- **`improve`**: run a benchmark, then each agent improves the other's source
  code, then re-benchmark. If both agents improved, keep the changes; otherwise
  revert and log the proposals.

## What it measures

| Metric | Description |
| --- | --- |
| **Token consumption** | Prompt tokens, completion tokens, total tokens |
| **LLM calls** | Number of HTTP round-trips to Ollama |
| **Turns** | Number of user turns (one per `chat()` call) |
| **Tool calls** | Number of tool invocations across all turns |
| **Wall-clock time** | Total elapsed seconds per task and across all tasks |
| **Success rate** | Percentage of tasks that completed successfully |

## Requirements

- Python 3.10+
- `requests`
- A running **Ollama** server with a tool-capable model
- The `agentThree` package on the Python path (same project)

## Quick start

```bash
# Interactive: the orchestrator prompts you for a task to run on both agents
python -m orchestrator --model-a glm-5.2:cloud --model-b deepseek-v4-pro:cloud

# Run the built-in benchmark suite
python -m orchestrator --default-tasks --model-a glm-5.2:cloud --model-b deepseek-v4-pro:cloud

# Specify tasks on the command line and save a JSON report
python -m orchestrator --model-a glm-5.2:cloud --model-b kimi-k2.7-code:cloud \
    --task "What is the capital of France?" \
    --task "Write a Python function to reverse a list" \
    --save benchmark_report.json
```

## Task input modes

The orchestrator supports three ways to specify benchmark tasks, tried in
this order:

1. **`--task "prompt"` flags** – one or more tasks specified directly on the
   command line.  When given, the interactive prompt is skipped.

2. **`--default-tasks` flag** – runs the built-in benchmark suite (6 tasks
   covering math, reasoning, creative writing, knowledge, and tool use).

3. **Interactive prompt (default)** – if neither `--task` nor `--default-tasks`
   is given, the orchestrator interactively asks the user to enter task
   prompts.  You can enter multiple tasks one at a time:
   - Type a task prompt and press Enter to add it.
   - Type `default` to switch to the built-in suite.
   - Press Enter on an empty line to finish (or quit if no task entered yet).

## CLI options

| Option | Default | Description |
| --- | --- | --- |
| `--mode benchmark\|improve` | `benchmark` | Run mode: benchmark only, or improve (benchmark → cross-improve → re-benchmark) |
| `--loops <n>` | `1` | Number of improve-mode loops (only used in improve mode) |
| `--model-a` | `DEFAULT_MODEL` from config | Model for Agent A |
| `--model-b` | `DEFAULT_MODEL` from config | Model for Agent B |
| `--ollama-url` | `OLLAMA_URL` from config | Ollama API URL |
| `--temperature` | _(model default)_ | Sampling temperature for both agents |
| `--max-iter` | `20` | Max tool-calling iterations per task |
| `--stream` | off | Enable LLM streaming (off by default for clean benchmarks) |
| `--verbose` | off | Verbose agent output during benchmark |
| `--save <path>` | _(none)_ | Save JSON report to file |
| `--task "prompt"` | _(none)_ | Add a custom task (repeatable; skips interactive prompt) |
| `--task-category <cat>` | `custom` | Category for custom/interactive tasks |
| `--default-tasks` | off | Run the built-in benchmark suite instead of prompting |
| `--interactive` | off | Force the interactive prompt (default when no --task/--default-tasks) |

## Programmatic usage

```python
from orchestrator.orchestrator import Orchestrator, BenchmarkTask

tasks = [
    BenchmarkTask("What is 2+2?", category="math", label="add"),
    BenchmarkTask("Write a poem", category="creative", label="poem"),
]

orch = Orchestrator(
    model_a="glm-5.2:cloud",
    model_b="deepseek-v4-pro:cloud",
    tasks=tasks,
)

report = orch.run()
Orchestrator.print_report(report)
Orchestrator.save_report(report, "benchmark.json")
```

## How it works

1. **Creates two `agentThree` instances** – each with its own model
   configuration.  Agents are created lazily (on first `run()` call) so
   that `__init__` doesn't fail if Ollama is temporarily down.

2. **Runs each benchmark task on both agents** – sequentially, first on
   Agent A then on Agent B.  Before each task the agent is `reset()` so
   conversation history doesn't carry over.

3. **Collects metrics from the agent instance** after each `chat()` call:
   - `agent.session_prompt_tokens`
   - `agent.session_completion_tokens`
   - `agent.session_total_tokens`
   - `agent.llm_call_count`
   - Turn count (user messages in `agent.messages`)
   - Tool-call count (tool-result messages in `agent.messages`)
   - Wall-clock elapsed time

4. **Computes aggregate summaries** across all tasks for each agent.

5. **Prints a comparison report** showing per-task results and a summary
   table with winners highlighted (lower is better for tokens/calls/time,
   higher is better for success rate).

## Default benchmark suite

| Label | Category | Prompt |
| --- | --- | --- |
| `basic_math` | math | What is 17 multiplied by 23? Show your reasoning briefly. |
| `fibonacci` | reasoning | List the first 10 Fibonacci numbers. |
| `haiku` | creative | Write a haiku about the ocean. |
| `tcp_udp` | knowledge | Explain the difference between TCP and UDP in three sentences. |
| `word_count_tool` | tool_use | Count the words in this sentence: 'The quick brown fox jumps over the lazy dog.' |
| `time_tool` | tool_use | What time is it right now? Use the appropriate tool. |

## Improve mode

Improve mode (`--mode improve`) runs a multi-step loop:

1. **Baseline benchmark** – runs the standard benchmark on both agents.
2. **Agent A improves Agent B** – Agent A is given the benchmark results and
   asked to read and modify Agent B's source code (the `agentThree/` package)
   to make it more efficient. Changes are applied directly to disk.
3. **Agent B improves Agent A** – Agent B does the same for Agent A's code.
4. **Post-improvement benchmark** – re-runs the benchmark with the improved
   code.
5. **Decision**:
   - If **both** agents improved (lower composite score: tokens + time +
     failures), the changes are **kept**.
   - If either or both did not improve, all changes are **reverted** to the
     original state, and the proposals + benchmark results are written to
     an improvement log file in `orchestrator/improve_logs/`.
6. The loop repeats for `--loops N` iterations (default: 1).

### Composite score

Lower is better:
```
score = total_tokens + 100 * total_time_seconds + 10000 * failed_tasks
```

### Improvement logs

When no improvement is achieved, a JSON log is written to
`orchestrator/improve_logs/improve_YYYYMMDD_HHMMSS_loopN.json` containing:
- Baseline and post-improvement benchmark summaries
- Scores for both agents
- Which files were proposed for change
- Agent answer summaries
- The reason changes were not applied

### Usage

```bash
# Single improve loop with default tasks
python -m orchestrator --mode improve --default-tasks \
    --model-a glm-5.2:cloud --model-b deepseek-v4-pro:cloud

# 3 improve loops with custom tasks
python -m orchestrator --mode improve --loops 3 \
    --model-a glm-5.2:cloud --model-b deepseek-v4-pro:cloud \
    --task "What is 2+2?" --task "Write a haiku"
```

## Project layout

```
orchestrator/
  __init__.py       - Package init
  __main__.py       - Entry point (python -m orchestrator)
  orchestrator.py   - Main orchestrator: benchmark logic, reporting, CLI
  improve_mode.py   - Improve mode: benchmark → cross-improve → re-benchmark
  readme.md         - This file
  improve_logs/     - JSON logs from improve mode (auto-created)
```