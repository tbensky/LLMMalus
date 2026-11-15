"""The conversation loop, and where each run's files go.

One session: send the task, let the model call tools until it calls finish
(or runs out of calls), then check its law against fresh readings.

    Runs/<model>/001/transcript.jsonl   every message and tool call
                     measurements.csv   every reading it took
                     notebook.md        what it wrote down (and .jsonl)
                     summary.json       outcome, its law, the check
    Runs/<model>/002/...
    Runs/<model>/runs_summary.json      how many runs found the law
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .apparatus import Apparatus
from .evaluate import score_formula
from .llm import ChatClient, LLMError, ToolCall, extract_text_tool_calls
from .notebook import Notebook
from .tools import LabState, ToolResult, Toolbox, default_tools

NUDGE = ("Please carry on by using your tools. When you are satisfied with "
         "your findings, call finish.")
TRIM_NOTE = ("(Earlier parts of this conversation have been removed to save "
             "space. Your notebook still has everything you wrote down.)")


@dataclass
class LabConfig:
    max_calls: int = 150          # model calls before the session is stopped
    keep_last: int = 0            # keep only the last N exchanges in context (0 = all)
    max_idle: int = 5             # replies in a row with no tool call before stopping
    holdout_points: int = 40      # fresh readings used to check the final law
    holdout_samples: int = 3
    threshold: float = 0.05       # normalized RMS error that counts as discovered
    verbose: bool = True


def trim_history(messages: list[dict[str, Any]], keep_last: int) -> list[dict[str, Any]]:
    """System prompt + task + the last `keep_last` exchanges.

    An exchange starts at an assistant message and includes the tool results
    and nudges after it, so a tool result is never separated from the call
    it answers (OpenAI-style APIs reject that).
    """
    if keep_last <= 0 or len(messages) <= 2:
        return messages
    head, rest = messages[:2], messages[2:]
    starts = [i for i, m in enumerate(rest) if m.get("role") == "assistant"]
    if len(starts) <= keep_last:
        return messages
    cut = starts[len(starts) - keep_last]
    return head + [{"role": "user", "content": TRIM_NOTE}] + rest[cut:]


class _Transcript:
    def __init__(self, folder: Path | None):
        self.path = folder / "transcript.jsonl" if folder else None

    def write(self, **record: Any) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _short(text: str, limit: int = 400) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + " ..."


def run_session(client: ChatClient, apparatus: Apparatus, *, system_prompt: str,
                task_prompt: str, folder: Path | None = None,
                config: LabConfig | None = None, holdout_seed: int | None = None,
                extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one experiment session and return (and save) its summary."""
    config = config or LabConfig()
    say: Callable[[str], None] = print if config.verbose else (lambda _msg: None)
    notebook = Notebook(folder)
    state = LabState(apparatus=apparatus, notebook=notebook, folder=folder)
    toolbox = Toolbox(default_tools(apparatus))
    transcript = _Transcript(folder)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]
    transcript.write(call=0, role="system", content=system_prompt)
    transcript.write(call=0, role="user", content=task_prompt)
    say(f"[task] {_short(task_prompt, 300)}")

    started = time.monotonic()
    status, error = "call_limit", None
    finish_data: dict[str, Any] = {}
    tool_counts: Counter[str] = Counter()
    usage = Counter()
    idle = 0
    text_fallbacks = 0
    calls_made = 0

    for call_number in range(1, config.max_calls + 1):
        state.call = call_number
        calls_made = call_number
        try:
            reply = client.chat(trim_history(messages, config.keep_last), toolbox.schemas())
        except LLMError as exc:
            status, error = "error", str(exc)
            say(f"[error] {exc}")
            break
        usage.update(reply.usage)
        calls = list(reply.tool_calls)
        from_text = False
        if not calls and reply.text:
            calls = extract_text_tool_calls(reply.text, set(toolbox.tools))
            from_text = bool(calls)
            text_fallbacks += int(from_text)
        messages.append(reply.assistant_message)
        transcript.write(call=call_number, role="assistant", content=reply.text,
                         tool_calls=[asdict(c) for c in calls], from_text=from_text,
                         usage=reply.usage)
        if reply.text.strip():
            say(f"[{call_number}] model: {_short(reply.text)}")

        if not calls:
            idle += 1
            if idle >= config.max_idle:
                status = "no_tool_calls"
                say(f"[stopped] {idle} replies in a row without a tool call")
                break
            messages.append({"role": "user", "content": NUDGE})
            transcript.write(call=call_number, role="user", content=NUDGE)
            continue
        idle = 0

        text_results = []
        finished = False
        for tool_call in calls:
            if tool_call.arguments is None:
                result = ToolResult(f"The arguments for {tool_call.name} could not be read "
                                    f"({tool_call.parse_error}). Send them as a JSON object.")
            else:
                result = toolbox.run(state, tool_call.name, tool_call.arguments)
            tool_counts[tool_call.name] += 1
            transcript.write(call=call_number, role="tool", name=tool_call.name,
                             arguments=tool_call.arguments, content=result.text,
                             data=result.data)
            say(f"[{call_number}] {tool_call.name}({json.dumps(tool_call.arguments)}) -> "
                f"{_short(result.text, 200)}")
            if from_text:
                text_results.append(f"Result of {tool_call.name}: {result.text}")
            else:
                messages.append(client.tool_result_message(tool_call, result.text))
            if result.finished:
                finished = True
                finish_data = result.data
                break
        if text_results:
            messages.append({"role": "user", "content": "\n".join(text_results)})
        if finished:
            status = "finished"
            break

    evaluation: dict[str, Any] | None = None
    if status == "finished":
        say("[check] testing the law against fresh readings...")
        try:
            evaluation = score_formula(
                finish_data.get("formula", ""), apparatus,
                points=config.holdout_points, samples=config.holdout_samples,
                seed=holdout_seed, threshold=config.threshold,
            )
        except Exception as exc:  # noqa: BLE001 - a failed check must not lose the run
            evaluation = {"error": f"{type(exc).__name__}: {exc}", "discovered": False}

    summary: dict[str, Any] = {
        "status": status,
        "discovered": bool(evaluation and evaluation.get("discovered")),
        "conclusion": finish_data.get("conclusion"),
        "formula": finish_data.get("formula"),
        "nrmse": evaluation.get("nrmse") if evaluation else None,
        "r2": evaluation.get("r2") if evaluation else None,
        "model_calls": calls_made,
        "tool_counts": dict(tool_counts),
        "readings": state.readings,
        "knob_moves": state.knob_moves,
        "notebook_entries": len(notebook.entries),
        "text_tool_calls": text_fallbacks,
        "usage": dict(usage),
        "elapsed_s": round(time.monotonic() - started, 1),
        "error": error,
        **(extra or {}),
        "evaluation": evaluation,
    }
    if folder is not None:
        (folder / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=_jsonable) + "\n",
            encoding="utf-8")
    verdict = "found the law" if summary["discovered"] else "did not find the law"
    if status != "finished":
        verdict = f"stopped ({status})"
    detail = f", nrmse={summary['nrmse']:.3f}" if isinstance(summary["nrmse"], float) else ""
    say(f"[result] {verdict}{detail}; formula: {summary['formula']!r}")
    return summary


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return str(value)


# --------------------------------------------------------------------------
# Run folders: Runs/<model>/001, 002, ... and Runs/<model>/runs_summary.json
# --------------------------------------------------------------------------
RUN_DIR_RE = re.compile(r"^\d{3,}$")
# A session that never got a verdict through no fault of the model.
NO_VERDICT = {"error", "interrupted", "missing"}


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"


def numbered_run_dirs(model_dir: Path) -> list[Path]:
    if not model_dir.is_dir():
        return []
    return sorted((p for p in model_dir.iterdir() if p.is_dir() and RUN_DIR_RE.match(p.name)),
                  key=lambda p: int(p.name))


def next_run_dir(model_dir: Path) -> tuple[int, Path]:
    """Create the next numbered folder; numbering continues across batches."""
    model_dir.mkdir(parents=True, exist_ok=True)
    number = max((int(p.name) for p in numbered_run_dirs(model_dir)), default=0) + 1
    while True:
        run_dir = model_dir / f"{number:03d}"
        try:
            run_dir.mkdir()
            return number, run_dir
        except FileExistsError:
            number += 1


def write_runs_summary(model_dir: Path, model: str) -> dict[str, Any]:
    """Rebuild runs_summary.json from every numbered run folder."""
    rows = []
    for run_dir in numbered_run_dirs(model_dir):
        path = run_dir / "summary.json"
        try:
            s = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            s = {"status": "missing"}
        rows.append({
            "run": run_dir.name,
            "status": s.get("status"),
            "discovered": bool(s.get("discovered")),
            "formula": s.get("formula"),
            "nrmse": s.get("nrmse"),
            "model_calls": s.get("model_calls"),
            "readings": s.get("readings"),
            "notebook_entries": s.get("notebook_entries"),
            "seed": s.get("seed"),
            "error": s.get("error"),
        })
    judged = [r for r in rows if r["status"] not in NO_VERDICT]
    found = [r for r in rows if r["discovered"]]
    calls = [r["model_calls"] for r in found if isinstance(r["model_calls"], int)]
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[str(r["status"])] = statuses.get(str(r["status"]), 0) + 1
    summary = {
        "model": model,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": len(rows),
        # A run that hit the call limit or stopped calling tools is a miss;
        # one lost to an endpoint error or Ctrl-C has no verdict.
        "judged": len(judged),
        "discovered": len(found),
        "discovery_rate": round(len(found) / len(judged), 3) if judged else None,
        "statuses": dict(sorted(statuses.items())),
        "median_calls_when_discovered": statistics.median(calls) if calls else None,
        "run_list": rows,
    }
    (model_dir / "runs_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
