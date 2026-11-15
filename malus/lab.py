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
from .evaluate import collect_holdout, score_formula
from .llm import ChatClient, LLMError, ToolCall, extract_text_tool_calls
from .notebook import Notebook
from .tools import LabState, ToolResult, Toolbox, default_tools

NUDGE = ("Please carry on by using your tools. When you are satisfied with "
         "your findings, call finish.")
# Sent in reply to the model's FIRST finish: it now says it understands the
# knobs, so ask for the end-of-lab write-up. Overridden by
# prompts/fit_request.txt when the command line finds one.
DEFAULT_FIT_REQUEST = (
    "Before you finish, please complete the experiment the way a student does at the end "
    "of a lab:\n"
    "1. Take a systematic set of data: sweep a knob in even steps across its range while "
    "holding the other knobs fixed.\n"
    "2. Fit your law to that data by hand. Choose values for your constants, use plot to "
    "graph the data with your curve, look at how well the curve goes through the points, "
    "then adjust the constants yourself and plot again, until you are satisfied with the "
    "fit. Do not use any automatic fitting procedure; choose the numbers yourself.\n"
    "3. Record your final constants in your notebook, then call finish again with your "
    "fitted law."
)
FIT_REMINDER = ("Before finishing, make at least one plot of systematic data with your "
                "hand-fitted curve, using plot, as described above.")
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
    # The first finish means "I understand the knobs": answer it with
    # fit_request (take systematic data, fit by hand, plot) and accept a
    # later finish once at least one new plot exists. After
    # max_finish_refusals finishes without a plot, accept anyway.
    fit_stage: bool = True
    fit_request: str = DEFAULT_FIT_REQUEST
    max_finish_refusals: int = 3
    # What the model reads about finish (None: tools.DEFAULT_FINISH_DESCRIPTION,
    # which already asks for the systematic data and the hand fit). A model
    # that plots a curve before its first finish has done the fit unprompted,
    # and that finish is accepted.
    finish_description: str | None = None


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


def result_formula(data: dict[str, Any]) -> str:
    return str(data.get("formula") or "")


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
    toolbox = Toolbox(default_tools(apparatus, config.finish_description))
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
    fit_state = "not_reached" if config.fit_stage else "off"
    initial: dict[str, Any] = {}      # the law at the model's first finish
    plots_at_request = readings_at_request = 0
    finish_refusals = 0

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
            calls = extract_text_tool_calls(reply.text, toolbox.schemas())
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
            if result.finished and config.fit_stage:
                if fit_state == "not_reached":
                    initial = {**result.data, "call": call_number}
                    if any(p.get("formula") for p in state.plots):
                        # finish's own description asks for the hand fit;
                        # this model did it before finishing, unasked.
                        fit_state = "unprompted"
                        say(f"[fit] first finish at call {call_number} came after "
                            f"{len(state.plots)} plot(s): the hand fit was done unprompted")
                    else:
                        fit_state = "requested"
                        plots_at_request, readings_at_request = len(state.plots), state.readings
                        result = ToolResult(config.fit_request, data={"fit_request": True})
                        say(f"[fit] first finish at call {call_number}: "
                            f"{result_formula(initial)!r}; asking for systematic data and a hand fit")
                elif fit_state == "requested" and len(state.plots) == plots_at_request:
                    finish_refusals += 1
                    if finish_refusals < config.max_finish_refusals:
                        result = ToolResult(FIT_REMINDER, data={"fit_reminder": finish_refusals})
                    else:
                        fit_state = "skipped"
                elif fit_state == "requested":
                    fit_state = "done"
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

    # Score the final law and, with the fit stage, the law at the first
    # finish too, against the same fresh readings.
    evaluation: dict[str, Any] | None = None
    initial_evaluation: dict[str, Any] | None = None
    if status == "finished" or initial:
        say("[check] testing the law against fresh readings...")
        try:
            holdout = collect_holdout(apparatus, points=config.holdout_points,
                                      samples=config.holdout_samples, seed=holdout_seed)
            if status == "finished":
                evaluation = score_formula(finish_data.get("formula", ""), apparatus,
                                           threshold=config.threshold, holdout=holdout)
            if initial:
                initial_evaluation = score_formula(initial.get("formula", ""), apparatus,
                                                   threshold=config.threshold, holdout=holdout)
                initial_evaluation.pop("points", None)
        except Exception as exc:  # noqa: BLE001 - a failed check must not lose the run
            evaluation = {"error": f"{type(exc).__name__}: {exc}", "discovered": False}

    summary: dict[str, Any] = {
        "status": status,
        "discovered": bool(evaluation and evaluation.get("discovered")),
        "conclusion": finish_data.get("conclusion"),
        "formula": finish_data.get("formula"),
        "nrmse": evaluation.get("nrmse") if evaluation else None,
        "r2": evaluation.get("r2") if evaluation else None,
        # The fit stage: off, not_reached (never called finish), unprompted
        # (plotted a curve before its first finish, as finish asks), requested
        # (asked for the fit, never finished), done (fitted after being asked),
        # or skipped (finished again without plotting max_finish_refusals times).
        "fit_stage": fit_state,
        "initial_formula": initial.get("formula"),
        "initial_conclusion": initial.get("conclusion"),
        "initial_call": initial.get("call"),
        "initial_discovered": bool(initial_evaluation and initial_evaluation.get("discovered")),
        "initial_nrmse": initial_evaluation.get("nrmse") if initial_evaluation else None,
        "plots": len(state.plots),
        "plot_files": [p["file"] for p in state.plots if p.get("file")],
        "readings_after_fit_request": (state.readings - readings_at_request
                                       if fit_state in {"requested", "done", "skipped"} else None),
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
        "initial_evaluation": initial_evaluation,
    }
    if folder is not None:
        (folder / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=_jsonable) + "\n",
            encoding="utf-8")
    verdict = "found the law" if summary["discovered"] else "did not find the law"
    if status != "finished":
        verdict = f"stopped ({status})"
    detail = f", nrmse={summary['nrmse']:.3f}" if isinstance(summary["nrmse"], float) else ""
    if initial:
        first = "right" if summary["initial_discovered"] else "wrong"
        first_detail = (f", nrmse={summary['initial_nrmse']:.3f}"
                        if isinstance(summary["initial_nrmse"], float) else "")
        say(f"[result] law at first finish (call {initial['call']}): {first}{first_detail}; "
            f"{summary['initial_formula']!r}")
    say(f"[result] {verdict}{detail}; formula: {summary['formula']!r}; "
        f"fit stage: {fit_state}, {len(state.plots)} plot(s)")
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
            "initial_discovered": bool(s.get("initial_discovered")),
            "initial_formula": s.get("initial_formula"),
            "fit_stage": s.get("fit_stage"),
            "plots": s.get("plots"),
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
        # The law the model gave the first time it called finish, before the
        # hand fit; and how many runs completed the fit stage.
        "discovered_at_first_finish": sum(r["initial_discovered"] for r in rows),
        "fit_stage_done": sum(r["fit_stage"] in {"done", "unprompted"} for r in rows),
        "fit_done_unprompted": sum(r["fit_stage"] == "unprompted" for r in rows),
        "statuses": dict(sorted(statuses.items())),
        "median_calls_when_discovered": statistics.median(calls) if calls else None,
        "run_list": rows,
    }
    (model_dir / "runs_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
