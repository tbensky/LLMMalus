"""The tools the model can call.

Each tool is one small class with a name, a description, a JSON-schema for
its arguments and a run() method. They reach the bench only through the
Apparatus interface, so they work unchanged on the simulation and on real
hardware. To add a tool, write a class like the ones below and add it to
default_tools().

The descriptions are deliberately plain: knobs and a voltage, no light, no
polarizers. They are what the model reads.
"""

from __future__ import annotations

import csv
import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .apparatus import Apparatus, ApparatusError
from .notebook import Notebook

@dataclass
class ToolResult:
    text: str                      # what the model is shown
    finished: bool = False         # the model announced it is done
    data: dict[str, Any] = field(default_factory=dict)  # for the logs, not the model


@dataclass
class LabState:
    """Shared by all tools during one session."""

    apparatus: Apparatus
    notebook: Notebook
    folder: Path | None = None
    call: int = 0                  # current model call, set by the loop
    readings: int = 0              # read_voltage calls (one detector reading each)
    knob_moves: int = 0
    # Every read_voltage result, so the data check can trace plotted points.
    # "mean" is the single reading (the data check also reads older runs,
    # where it was an average).
    measurements: list[dict[str, Any]] = field(default_factory=list)
    plots: list[dict[str, Any]] = field(default_factory=list)
    plot_checks: list[dict[str, Any]] = field(default_factory=list)  # audit of each plot
    # What each plot was audited from, so finish can audit it again with the
    # model's law: its points, its curve at each point, readings taken by then.
    plot_inputs: list[dict[str, Any]] = field(default_factory=list)
    _csv_started: bool = False

    def log_measurement(self, settings: dict[str, float], value: float) -> None:
        self.measurements.append({"call": self.call, "settings": dict(settings),
                                  "mean": value})
        if self.folder is None:
            return
        knob_names = list(settings)
        with (self.folder / "measurements.csv").open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not self._csv_started:
                writer.writerow(["call", "time", *knob_names, "voltage"])
                self._csv_started = True
            writer.writerow([self.call, round(time.time(), 3),
                             *[settings[k] for k in knob_names], value])


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        """OpenAI-style function schema (Ollama accepts the same shape)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    def run(self, state: LabState, args: dict[str, Any]) -> ToolResult:
        ...


def _fmt(value: float) -> str:
    return f"{value:g}"


def _settings_text(settings: dict[str, float]) -> str:
    return ", ".join(f"{k} = {_fmt(v)}" for k, v in settings.items())


class SetKnob(Tool):
    """Turn one knob: its name and the setting to turn it to."""

    name = "set_knob"

    def __init__(self, apparatus: Apparatus):
        self.names = [k.name for k in apparatus.knobs]
        knob_lines = " ".join(f"{k.name}: {k.description}" for k in apparatus.knobs)
        self.description = (
            "Turn one knob to a new setting. The other knobs stay where they are. "
            f"{knob_lines}"
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "knob": {"type": "string", "enum": self.names,
                         "description": f"Which knob to turn: {', '.join(self.names)}."},
                "value": {"type": "number",
                          "description": "The setting to turn it to."},
            },
            "required": ["knob", "value"],
        }

    def run(self, state: LabState, args: dict[str, Any]) -> ToolResult:
        knob = str(args.get("knob") or "").strip().upper()
        if knob not in self.names:
            return ToolResult(f"Name the knob to turn: one of {', '.join(self.names)}.")
        if args.get("value") is None:
            return ToolResult(f"Give the setting to turn knob {knob} to.")
        try:
            state.apparatus.set_knob(knob, args["value"])
        except ApparatusError as exc:
            settings = state.apparatus.get_knobs()
            return ToolResult(str(exc),
                              data={"settings": settings, "problems": [str(exc)]})
        state.knob_moves += 1
        settings = state.apparatus.get_knobs()
        # No positions in the reply: keeping track of them is the model's job.
        return ToolResult(f"Ok. Knob {knob} has been set.",
                          data={"settings": settings, "problems": []})


class ReadVoltage(Tool):
    """One reading of the voltage at the current settings."""

    name = "read_voltage"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, apparatus: Apparatus):
        self.description = (f"Read the output {apparatus.signal_name} at the current knob "
                            "settings.")

    def run(self, state: LabState, args: dict[str, Any]) -> ToolResult:
        app = state.apparatus
        settings = app.get_knobs()
        try:
            value = app.read_signal()
        except ApparatusError as exc:
            return ToolResult(f"The reading failed: {exc}")
        state.readings += 1
        state.log_measurement(settings, value)
        # Just the reading: keeping track of the knob settings is the model's job.
        return ToolResult(f"{value:.4f} {app.signal_units}",
                          data={"settings": settings, "value": value, "mean": value})


class WriteDown(Tool):
    name = "write_down"
    description = (
        "Write something in your lab notebook: data, observations, ideas, plans, "
        "anything you want to keep. Only you read it."
    )
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "What to write down."}},
        "required": ["text"],
    }

    def run(self, state: LabState, args: dict[str, Any]) -> ToolResult:
        text = str(args.get("text") or "").strip()
        if not text:
            return ToolResult("Nothing was written: the text was empty.")
        entry = state.notebook.write(text, call=state.call)
        return ToolResult(f"Written as notebook entry {entry.number}.",
                          data={"entry": entry.number})


class ReadNotebook(Tool):
    name = "read_notebook"
    description = "Read back everything you have written in your lab notebook."
    parameters = {"type": "object", "properties": {}}

    def run(self, state: LabState, args: dict[str, Any]) -> ToolResult:
        return ToolResult(state.notebook.read(),
                          data={"entries": len(state.notebook.entries)})


class Plot(Tool):
    """Graph (x, y) points the model supplies, with its own curve.

    Like a student typing data into a spreadsheet: the model decides what
    goes on each axis and sends the numbers itself, and may add a curve
    written in terms of x with constants it chose. It gets back only
    "Plot N saved.": no curve values, differences or RMS. A text model cannot
    look at the PNG, so judging whether its curve goes through its points
    means working that out itself, from its own data and its own curve.

    Because the model types the numbers, a plot could show values it never
    measured, a real reading moved to the wrong settings, or idealized values
    that sit on its curve. Each plot is therefore checked by audit.audit_points:
    every point is clean, rounded, moved or untraced, and a plot with a curve
    is also tested for idealized data. The counts and the reading each point
    came from are kept in plots.jsonl and the summary. plot itself says
    nothing about it; the finish gate in lab.py does, if the model tries to
    finish on a plot that fails.
    """

    name = "plot"
    MAX_POINTS = 500
    CURVE_POINTS = 241

    def __init__(self, apparatus: Apparatus):
        self.units = apparatus.signal_units
        self.knobs = apparatus.knobs
        self.description = (
            "Make an X/Y graph from data you supply, and optionally draw your own curve "
            "through it, the way you would make a plot in a spreadsheet. You choose what goes "
            "on each axis and give the points yourself. The graph is saved with your lab work."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "description": ("The data to plot, as a list of [x, y] pairs, for example "
                                    "[[1, 2.3], [2, 4.1]]."),
                    "items": {"type": "array", "items": {"type": "number"},
                              "minItems": 2, "maxItems": 2},
                },
                "formula": {
                    "type": "string",
                    "description": ("Optional. Your curve: a Python expression in x with your "
                                    "chosen constants filled in as numbers, for example "
                                    "\"2*x + 1\". Functions from Python's math module may be "
                                    "used by name."),
                },
                "x_label": {"type": "string", "description": "Optional label for the x axis."},
                "y_label": {"type": "string", "description": "Optional label for the y axis."},
                "title": {"type": "string", "description": "Optional title for the graph."},
            },
            "required": ["points"],
        }

    @staticmethod
    def _read_points(raw: Any) -> tuple[list[tuple[float, float]] | None, str | None]:
        """Accept [[x, y], ...] and, leniently, [{"x": .., "y": ..}, ...]."""
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return None, "points must be a list of [x, y] pairs."
        if not isinstance(raw, list) or not raw:
            return None, "points must be a non-empty list of [x, y] pairs."
        points = []
        for item in raw:
            if isinstance(item, dict) and "x" in item and "y" in item:
                item = [item["x"], item["y"]]
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                return None, f"Each point must be an [x, y] pair; {item!r} is not."
            try:
                x, y = float(item[0]), float(item[1])
            except (TypeError, ValueError):
                return None, f"Each point must be two numbers; {item!r} is not."
            if not (math.isfinite(x) and math.isfinite(y)):
                return None, f"Each point must be two finite numbers; {item!r} is not."
            points.append((x, y))
        return points, None

    def run(self, state: LabState, args: dict[str, Any]) -> ToolResult:
        from .audit import audit_points
        from .evaluate import FormulaError, compile_formula
        from .plotting import render

        points, problem = self._read_points(args.get("points"))
        if problem:
            return ToolResult(problem)
        if len(points) > self.MAX_POINTS:
            return ToolResult(f"That is {len(points)} points; plot at most {self.MAX_POINTS}.")
        points.sort()

        formula = str(args.get("formula") or "").strip()
        law = None
        if formula:
            try:
                law = compile_formula(formula, ("x",))
            except FormulaError as exc:
                return ToolResult(f"Your formula could not be used: {exc} "
                                  "The curve must be written in terms of x.")
        rows, curve = [], None
        try:
            for x, y in points:
                rows.append((x, y, law(x) if law is not None else None))
            lo, hi = points[0][0], points[-1][0]
            if law is not None and hi > lo:
                step = (hi - lo) / (self.CURVE_POINTS - 1)
                curve = [(lo + i * step, law(lo + i * step)) for i in range(self.CURVE_POINTS)]
        except (ArithmeticError, ValueError, TypeError) as exc:
            return ToolResult(f"Your formula failed to evaluate: {exc}")

        check = audit_points(points, state.measurements, self.knobs,
                             curve=[c for _, _, c in rows] if law is not None else None)

        number = len(state.plots) + 1
        x_label = str(args.get("x_label") or "").strip() or "x"
        y_label = str(args.get("y_label") or "").strip() or "y"
        title = str(args.get("title") or "").strip() or f"{y_label} vs {x_label}"
        saved = None
        if state.folder is not None:
            folder = state.folder / "plots"
            folder.mkdir(exist_ok=True)
            saved = render(folder / f"plot_{number:02d}", x_label=x_label, y_label=y_label,
                           title=title, points=points, curve=curve, curve_label=formula)
        residuals = [y - c for _, y, c in rows if c is not None]
        rms = math.sqrt(sum(r * r for r in residuals) / len(residuals)) if residuals else None
        record = {"number": number, "call": state.call, "formula": formula, "title": title,
                  "x_label": x_label, "y_label": y_label, "points": len(points),
                  "rms": rms, "file": saved.name if saved else None,
                  # the data check (audit.py); the model hears about it only if
                  # it tries to finish on a plot that fails it
                  "axis": check["axis"], "y_quantity": check["y_quantity"],
                  "clean": check["clean"], "moved": check["moved"],
                  "rounded": check["rounded"], "untraced": check["untraced"],
                  "too_good": check["too_good"], "rms_readings": check["rms_readings"]}
        state.plots.append(record)
        state.plot_checks.append(check)
        state.plot_inputs.append({"points": points,
                                  "curve": [c for _, _, c in rows] if law is not None else None,
                                  "readings": len(state.measurements)})
        if state.folder is not None:
            with (state.folder / "plots" / "plots.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({**record, "data": rows, "check": check["points"]},
                                   ensure_ascii=False) + "\n")

        # Nothing computed for the model: no curve values, no differences, no
        # RMS. Comparing its curve with its data is its own work. (The record
        # keeps the RMS and the data check for the summary.)
        return ToolResult(f"Plot {number} saved.", data=record)


# What the model reads about finish. The command line replaces it with
# prompts/finish_tool.txt when that file exists.
DEFAULT_FINISH_DESCRIPTION = (
    "Announce that you are done, once you are satisfied with your findings. Before you "
    "finish, complete the experiment the way a student does at the end of a lab: take a "
    "set of data that you feel is appropriate to make an X/Y plot that will concretely "
    "illustrate and prove your conclusions, fit your law to that data by hand using plot "
    "(choose values for "
    "your constants, graph the data with your curve, adjust the constants yourself and plot "
    "again until you are satisfied; no automatic fitting), and record your final constants "
    "in your notebook. When that is done, calling finish ends the experiment."
)


class Finish(Tool):
    name = "finish"

    def __init__(self, description: str | None = None):
        self.description = description or DEFAULT_FINISH_DESCRIPTION
    parameters = {
        "type": "object",
        "properties": {
            "conclusion": {
                "type": "string",
                "description": "Your findings: the law you discovered and how you know.",
            },
            "formula": {
                "type": "string",
                "description": (
                    "Your law as a single Python expression for the voltage in terms of "
                    "the knob settings A, B and C, exactly as you set them (B and C in "
                    "degrees), with your fitted constants filled in as numbers. Functions "
                    "from Python's math module may be used by name."
                ),
            },
        },
        "required": ["conclusion", "formula"],
    }

    def run(self, state: LabState, args: dict[str, Any]) -> ToolResult:
        conclusion = str(args.get("conclusion") or "").strip()
        formula = str(args.get("formula") or "").strip()
        return ToolResult(
            "Thank you. The experiment is over.",
            finished=True,
            data={"conclusion": conclusion, "formula": formula},
        )


def default_tools(apparatus: Apparatus, finish_description: str | None = None) -> list[Tool]:
    return [SetKnob(apparatus), ReadVoltage(apparatus), WriteDown(),
            ReadNotebook(), Plot(apparatus), Finish(finish_description)]


class Toolbox:
    """Looks tools up by name and runs them, turning every failure into text."""

    def __init__(self, tools: list[Tool]):
        self.tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    def run(self, state: LabState, name: str, args: dict[str, Any] | None) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(f"There is no tool called {name!r}. "
                              f"The tools are: {', '.join(self.tools)}.")
        if not isinstance(args, dict):
            return ToolResult(f"The arguments for {name} could not be read. "
                              "Send them as a JSON object.")
        try:
            return tool.run(state, args)
        except Exception as exc:  # noqa: BLE001 - shown to the model, logged
            return ToolResult(f"{name} failed: {exc}", data={"exception": repr(exc)})


def parse_arguments(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Tool arguments arrive as a dict (Ollama) or a JSON string (OpenAI)."""
    if raw is None or raw == "":
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        return None, f"could not parse arguments: {exc}"
    if not isinstance(value, dict):
        return None, "arguments were not a JSON object"
    return value, None
