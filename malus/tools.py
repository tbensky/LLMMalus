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
    """Graph measured voltage against one knob, with the model's curve.

    The points come from the lab's own record of read_voltage results,
    never from numbers the model types in, so a graph can only show data
    that was really taken. The model chooses the data set (which knob on
    the x-axis, where the others were held) and the curve (its law with
    constants it picked itself). It gets back a table of each point next to
    its curve's value: a text model cannot look at the PNG, so the table is
    how it "sees" whether the curve goes through the points.
    """

    name = "plot"
    MATCH_TOLERANCE = 1e-6
    MAX_ROWS = 50
    CURVE_POINTS = 241

    def __init__(self, apparatus: Apparatus):
        self.knob_names = [k.name for k in apparatus.knobs]
        self.signal = f"{apparatus.signal_name} ({apparatus.signal_units})"
        self.units = apparatus.signal_units
        self.description = (
            f"Make a graph of the {apparatus.signal_name}s you have measured against one knob, "
            "with the other knobs held at fixed settings, and optionally draw your own curve "
            "through them. Only readings you have actually taken are plotted. The graph is "
            "saved with your lab work, and you get back a table of each data point next to "
            "your curve's value at that point."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "x": {"type": "string", "enum": self.knob_names,
                      "description": "The knob for the horizontal axis."},
                "fixed": {
                    "type": "object",
                    "description": ("Where every other knob was held for this data set, e.g. "
                                    "{\"A\": 100, \"C\": 0}. Only readings taken at exactly "
                                    "these settings are plotted."),
                    "properties": {k: {"type": "number"} for k in self.knob_names},
                },
                "formula": {
                    "type": "string",
                    "description": ("Optional. Your curve: a Python expression for the "
                                    f"{apparatus.signal_name} in terms of "
                                    f"{', '.join(self.knob_names)}, with your chosen constants "
                                    "filled in as numbers (the same form as for finish)."),
                },
                "title": {"type": "string", "description": "Optional title for the graph."},
            },
            "required": ["x", "fixed"],
        }

    def run(self, state: LabState, args: dict[str, Any]) -> ToolResult:
        from .evaluate import FormulaError, compile_formula
        from .plotting import render

        x = str(args.get("x") or "")
        if x not in self.knob_names:
            return ToolResult(f"x must be one of {', '.join(self.knob_names)}.")
        fixed_raw = args.get("fixed") or {}
        if not isinstance(fixed_raw, dict):
            return ToolResult("fixed must be an object such as {\"A\": 100, \"C\": 0}.")
        try:
            fixed = {str(k): float(v) for k, v in fixed_raw.items() if str(k) != x}
        except (TypeError, ValueError):
            return ToolResult("Every value in fixed must be a number.")
        missing = [k for k in self.knob_names if k != x and k not in fixed]
        if missing:
            return ToolResult(f"Say where the other knobs were held: fixed is missing "
                              f"{', '.join(missing)}.")

        points = sorted(
            (m["settings"][x], m["mean"]) for m in state.measurements
            if all(abs(m["settings"][k] - v) <= self.MATCH_TOLERANCE for k, v in fixed.items())
        )
        held = ", ".join(f"{k} = {v:g}" for k, v in fixed.items())
        if not points:
            return ToolResult(f"You have no readings with {held}. Take some with "
                              "set_knobs and read_voltage first.")

        formula = str(args.get("formula") or "").strip()
        law = None
        if formula:
            try:
                law = compile_formula(formula, tuple(self.knob_names))
            except FormulaError as exc:
                return ToolResult(f"Your formula could not be used: {exc}")

        def curve_at(xv: float) -> float:
            return law(*[xv if k == x else fixed[k] for k in self.knob_names])

        curve = None
        rows = []
        try:
            if law is not None:
                lo, hi = points[0][0], points[-1][0]
                if hi > lo:
                    step = (hi - lo) / (self.CURVE_POINTS - 1)
                    curve = [(lo + i * step, curve_at(lo + i * step))
                             for i in range(self.CURVE_POINTS)]
            for xv, measured in points:
                rows.append((xv, measured, curve_at(xv) if law is not None else None))
        except (ArithmeticError, ValueError, TypeError) as exc:
            return ToolResult(f"Your formula failed to evaluate: {exc}")

        number = len(state.plots) + 1
        title = str(args.get("title") or "").strip() or f"{self.signal} vs knob {x}, {held}"
        saved = None
        if state.folder is not None:
            folder = state.folder / "plots"
            folder.mkdir(exist_ok=True)
            saved = render(folder / f"plot_{number:02d}", x_label=f"knob {x}",
                           y_label=self.signal, title=title, points=points, curve=curve,
                           curve_label=formula)
        residuals = [m - c for _, m, c in rows if c is not None]
        rms = (sum(r * r for r in residuals) / len(residuals)) ** 0.5 if residuals else None
        record = {"number": number, "call": state.call, "x": x, "fixed": fixed,
                  "formula": formula, "title": title, "points": len(points),
                  "rms": rms, "file": saved.name if saved else None}
        state.plots.append(record)
        if state.folder is not None:
            with (state.folder / "plots" / "plots.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({**record, "data": rows}, ensure_ascii=False) + "\n")

        lines = [f"Plot {number}: {len(points)} readings of the {self.signal} against knob {x} "
                 f"with {held}" + (", with your curve." if law else ", data only.")]
        for xv, measured, predicted in rows[: self.MAX_ROWS]:
            if predicted is None:
                lines.append(f"  {x} = {xv:g}: measured {measured:.4f} {self.units}")
            else:
                lines.append(f"  {x} = {xv:g}: measured {measured:.4f}, your curve "
                             f"{predicted:.4f}, difference {measured - predicted:+.4f} {self.units}")
        if len(rows) > self.MAX_ROWS:
            lines.append(f"  ... and {len(rows) - self.MAX_ROWS} more readings")
        if rms is not None:
            lines.append(f"Root-mean-square difference between your curve and the data: "
                         f"{rms:.4f} {self.units}")
        if len(points) < 3:
            lines.append("Only a few readings match these settings; a sweep needs more.")
        return ToolResult("\n".join(lines), data=record)


class Finish(Tool):
    name = "finish"
    description = (
        "Announce that you are done, once you are satisfied with your findings. "
        "This ends the experiment."
    )
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


def default_tools(apparatus: Apparatus) -> list[Tool]:
    return [SetKnobs(apparatus), ReadVoltage(apparatus), WriteDown(),
            ReadNotebook(), Plot(apparatus), Finish()]


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
