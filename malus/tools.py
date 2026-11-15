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
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .apparatus import Apparatus, ApparatusError
from .notebook import Notebook

MAX_SAMPLES = 20


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
    readings: int = 0              # detector readings taken (each sample counts)
    knob_moves: int = 0
    # Every read_voltage result, so plot can draw only data actually taken.
    measurements: list[dict[str, Any]] = field(default_factory=list)
    plots: list[dict[str, Any]] = field(default_factory=list)
    _csv_started: bool = False

    def log_measurement(self, settings: dict[str, float], values: list[float]) -> None:
        self.measurements.append({
            "call": self.call, "settings": dict(settings),
            "mean": statistics.fmean(values), "samples": len(values),
        })
        if self.folder is None:
            return
        path = self.folder / "measurements.csv"
        knob_names = list(settings)
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not self._csv_started:
                writer.writerow(["call", "time", *knob_names, "samples", "mean", "stdev"])
                self._csv_started = True
            mean = statistics.fmean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else ""
            writer.writerow([self.call, round(time.time(), 3),
                             *[settings[k] for k in knob_names],
                             len(values), mean, stdev])


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


class SetKnobs(Tool):
    name = "set_knobs"

    def __init__(self, apparatus: Apparatus):
        knob_lines = " ".join(f"{k.name}: {k.description}" for k in apparatus.knobs)
        self.description = (
            "Turn one or more knobs to new settings. Knobs you leave out stay "
            f"where they are. {knob_lines}"
        )
        self.parameters = {
            "type": "object",
            "properties": {
                k.name: {"type": "number", "description": k.description}
                for k in apparatus.knobs
            },
        }

    def run(self, state: LabState, args: dict[str, Any]) -> ToolResult:
        requested = {k: v for k, v in args.items() if v is not None}
        if not requested:
            names = ", ".join(k.name for k in state.apparatus.knobs)
            return ToolResult(f"No knob was given. Name at least one of {names}.")
        problems = []
        for name, value in requested.items():
            try:
                state.apparatus.set_knob(str(name), value)
                state.knob_moves += 1
            except ApparatusError as exc:
                problems.append(str(exc))
        settings = state.apparatus.get_knobs()
        text = f"Knobs now: {_settings_text(settings)}."
        if problems:
            text = " ".join(problems) + " " + text
        return ToolResult(text, data={"settings": settings, "problems": problems})


class ReadVoltage(Tool):
    name = "read_voltage"

    def __init__(self, apparatus: Apparatus):
        self.description = (
            f"Read the output {apparatus.signal_name} at the current knob settings. "
            f"Optionally average several readings with 'samples' (1 to {MAX_SAMPLES})."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "samples": {
                    "type": "integer",
                    "description": f"Number of readings to average, 1 to {MAX_SAMPLES} (default 1).",
                }
            },
        }

    def run(self, state: LabState, args: dict[str, Any]) -> ToolResult:
        try:
            samples = int(args.get("samples") or 1)
        except (TypeError, ValueError):
            samples = 1
        samples = max(1, min(MAX_SAMPLES, samples))
        app = state.apparatus
        settings = app.get_knobs()
        try:
            values = [app.read_signal() for _ in range(samples)]
        except ApparatusError as exc:
            return ToolResult(f"The reading failed: {exc}")
        state.readings += samples
        state.log_measurement(settings, values)
        mean = statistics.fmean(values)
        units = app.signal_units
        if samples == 1:
            text = f"At {_settings_text(settings)}: {mean:.4f} {units}"
        else:
            spread = statistics.stdev(values)
            text = (f"At {_settings_text(settings)}: {mean:.4f} {units} "
                    f"(mean of {samples} readings, standard deviation {spread:.4f} {units})")
        return ToolResult(text, data={"settings": settings, "values": values, "mean": mean})


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
    written in terms of x with constants it chose. It gets back a table of
    each point next to its curve's value: a text model cannot look at the
    PNG, so the table is how it "sees" whether the curve goes through the
    points.

    Because the model now types the numbers, a plot could show values it
    never measured. Each plot is therefore audited silently (the model is
    not told): every y value is compared with the readings the lab actually
    took, and the count that match is kept in plots.jsonl and the summary.
    A y the model derived itself (V/A, a normalized value...) will not
    match a raw reading; that is a choice to look at, not an error.
    """

    name = "plot"
    MAX_POINTS = 500
    MAX_ROWS = 50
    CURVE_POINTS = 241
    TRACE_TOLERANCE = 0.001        # how close a y must be to a recorded reading

    def __init__(self, apparatus: Apparatus):
        self.units = apparatus.signal_units
        self.description = (
            "Make an X/Y graph from data you supply, and optionally draw your own curve "
            "through it, the way you would make a plot in a spreadsheet. You choose what goes "
            "on each axis and give the points yourself. The graph is saved with your lab work, "
            "and you get back a table of each point next to your curve's value at that x."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "description": ("The data to plot, as a list of [x, y] pairs, for example "
                                    "[[0, 4.99], [15, 4.66], [30, 3.75]]."),
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

        recorded = [m["mean"] for m in state.measurements]
        traced = sum(1 for _, y in points
                     if any(abs(y - r) <= self.TRACE_TOLERANCE for r in recorded))

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
                  "traced_points": traced, "rms": rms, "file": saved.name if saved else None}
        state.plots.append(record)
        if state.folder is not None:
            with (state.folder / "plots" / "plots.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({**record, "data": rows}, ensure_ascii=False) + "\n")

        # Show 0 instead of floating-point dust such as -4.4e-16, judged
        # against the size of the data itself.
        scale = max(abs(y) for _, y in points) or 1.0

        def num(value: float, signed: bool = False) -> str:
            if abs(value) < scale * 1e-9:
                value = 0.0
            return f"{value:+.4g}" if signed else f"{value:.4g}"

        lines = [f"Plot {number}: {len(points)} points, {y_label} against {x_label}"
                 + (", with your curve." if law else ", data only.")]
        if law is not None:
            for x, y, c in rows[: self.MAX_ROWS]:
                lines.append(f"  x = {x:g}: y = {num(y)}, your curve {num(c)}, "
                             f"difference {num(y - c, signed=True)}")
            if len(rows) > self.MAX_ROWS:
                lines.append(f"  ... and {len(rows) - self.MAX_ROWS} more points")
            lines.append("Root-mean-square difference between your curve and the data: "
                         f"{num(rms)}")
        if len(points) < 3:
            lines.append("Only a few points; a convincing plot usually needs more.")
        return ToolResult("\n".join(lines), data=record)


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
    return [SetKnobs(apparatus), ReadVoltage(apparatus), WriteDown(),
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
