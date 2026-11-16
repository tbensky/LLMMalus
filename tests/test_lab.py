"""Offline tests: the bench, the formula check, and full sessions against a
fake OpenAI-style and a fake Ollama server (no real model, no network)."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from malus.apparatus import ApparatusError, SimulatedPolarizerApparatus, SimulationConfig  # noqa: E402
from malus.evaluate import FormulaError, compile_formula, score_formula  # noqa: E402
from malus.lab import LabConfig, run_session, trim_history  # noqa: E402
from malus.llm import make_client  # noqa: E402

CORRECT = "V = 5*(A/100)*cos(radians(C - B))^2"


def quiet_bench(**kw) -> SimulatedPolarizerApparatus:
    return SimulatedPolarizerApparatus(SimulationConfig(noise_volts=0.0, **kw), seed=1)


class ApparatusTests(unittest.TestCase):
    def reading(self, bench, **settings):
        for name, value in settings.items():
            bench.set_knob(name, value)
        return bench.read_signal()

    def test_malus_law(self):
        bench = quiet_bench()
        self.assertAlmostEqual(self.reading(bench, A=100, B=0, C=0), 5.0)
        self.assertAlmostEqual(self.reading(bench, A=100, B=0, C=90), 0.0)
        self.assertAlmostEqual(self.reading(bench, A=100, B=0, C=45), 2.5)
        self.assertAlmostEqual(self.reading(bench, A=50, B=10, C=70), 5 * 0.5 * 0.25)
        self.assertAlmostEqual(self.reading(bench, A=100, B=400, C=40), 5.0)  # B wraps

    def test_the_less_ideal_bench(self):
        bench = quiet_bench(dark_volts=0.1, extinction=0.02, angle_offset_deg=30)
        self.assertAlmostEqual(self.reading(bench, A=0, B=0, C=0), 0.1)
        self.assertAlmostEqual(self.reading(bench, A=100, B=30, C=0), 5.1)   # aligned
        self.assertAlmostEqual(self.reading(bench, A=100, B=0, C=60), 0.1 + 5 * 0.02)

    def test_knob_limits_and_bad_names(self):
        bench = quiet_bench()
        with self.assertRaises(ApparatusError):
            bench.set_knob("A", 101)
        with self.assertRaises(ApparatusError):
            bench.set_knob("D", 1)
        self.assertEqual(bench.set_knob("C", -725.5), -725.5)

    def test_noise_is_reproducible_by_seed(self):
        a = SimulatedPolarizerApparatus(SimulationConfig(), seed=7)
        b = SimulatedPolarizerApparatus(SimulationConfig(), seed=7)
        for bench in (a, b):
            bench.set_knob("A", 80)
        self.assertEqual([a.read_signal() for _ in range(5)],
                         [b.read_signal() for _ in range(5)])


class FormulaTests(unittest.TestCase):
    def test_a_correct_law_passes_and_wrong_forms_fail(self):
        self.assertTrue(score_formula(CORRECT, quiet_bench(), seed=3)["discovered"])
        for wrong in ("5*(A/100)*cos(radians(C - B))",        # cos, not cos^2
                      "5*(A/100)*cos(radians(C))**2",          # ignores B
                      "5*(A/100)"):                             # no angle at all
            self.assertFalse(score_formula(wrong, quiet_bench(), seed=3)["discovered"], wrong)

    def test_common_ways_of_writing_it(self):
        for text in (CORRECT, "5*A/100*math.cos(math.radians(C-B))**2",
                     "f(A,B,C) = 0.05*A*cos(pi*(C-B)/180)**2", "`5*A/100*cos(radians(C-B))**2`."):
            self.assertAlmostEqual(compile_formula(text)(100, 0, 0), 5.0, msg=text)

    def test_unsafe_or_unknown_input_is_rejected(self):
        for bad in ("__import__('os').system('true')", "().__class__", "open('x')",
                    "(lambda: 1)()", "A if B else C", "D * 2", "", "cos(x=1)"):
            with self.assertRaises(FormulaError, msg=bad):
                compile_formula(bad)

    def test_a_runaway_power_errors_instead_of_hanging(self):
        started = time.monotonic()
        result = score_formula("10**10**10 + A", quiet_bench(), points=3)
        self.assertLess(time.monotonic() - started, 2)
        self.assertFalse(result["discovered"])
        self.assertIn("error", result)


class TrimTests(unittest.TestCase):
    def test_tool_results_stay_with_their_call(self):
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
        for n in range(6):
            messages.append({"role": "assistant", "content": None,
                             "tool_calls": [{"id": f"c{n}"}]})
            messages.append({"role": "tool", "tool_call_id": f"c{n}", "content": "r"})
        trimmed = trim_history(messages, keep_last=2)
        self.assertEqual(trimmed[:2], messages[:2])
        self.assertEqual(trimmed[3]["role"], "assistant")
        self.assertEqual([m.get("tool_call_id") for m in trimmed if m["role"] == "tool"],
                         ["c4", "c5"])
        self.assertIs(trim_history(messages, 0), messages)


# ---------------------------------------------------------------------------
# A fake model server: answers each request from a script, and checks that
# the history it is sent is well formed.
# ---------------------------------------------------------------------------
class FakeModelServer:
    def __init__(self, style: str, script: list):
        self.style, self.script, self.requests, self.problems = style, list(script), [], []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(payload)
                outer.check(payload)
                step = outer.script.pop(0) if outer.script else ("text", "I am done.")
                body = json.dumps(outer.respond(step)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        path = "/v1/chat/completions" if style == "openai" else "/api/chat"
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}{path}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def check(self, payload):
        names = {t["function"]["name"] for t in payload.get("tools", [])}
        if names != {"set_knob", "read_voltage", "write_down", "read_notebook", "plot",
                     "finish"}:
            self.problems.append(f"tools offered: {sorted(names)}")
        messages = payload["messages"]
        open_ids: list = []
        for m in messages:
            if m["role"] == "assistant":
                open_ids = [c.get("id") for c in m.get("tool_calls") or []]
            elif m["role"] == "tool":
                if self.style == "openai" and m.get("tool_call_id") not in open_ids:
                    self.problems.append(f"orphan tool result {m.get('tool_call_id')}")
                if self.style == "ollama" and not m.get("tool_name"):
                    self.problems.append("ollama tool result without tool_name")

    def respond(self, step):
        kind, value = step
        if kind == "text":
            message = {"role": "assistant", "content": value}
        else:   # ("tools", [(name, args), ...])
            calls = []
            for n, (name, args) in enumerate(value):
                function = {"name": name,
                            "arguments": json.dumps(args) if self.style == "openai" else args}
                call = {"function": function}
                if self.style == "openai":
                    call.update({"id": f"call_{len(self.requests)}_{n}", "type": "function"})
                calls.append(call)
            message = {"role": "assistant", "content": "", "tool_calls": calls}
        if self.style == "openai":
            return {"choices": [{"message": message}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
        return {"message": message, "prompt_eval_count": 10, "eval_count": 2}


def turn(**settings):
    """set_knob calls, one per knob, as a model would make them."""
    return [("set_knob", {"knob": k, "value": v}) for k, v in settings.items()]


EXPERIMENT = [
    ("tools", turn(A=100, B=0, C=0)),
    ("tools", [("read_voltage", {})]),
    ("tools", turn(C=90) + [("read_voltage", {})]),
    ("tools", [("write_down", {"text": "Full at C=B, zero 90 degrees apart."})]),
    ("tools", [("read_notebook", {})]),
    ("tools", [("finish", {"conclusion": "V goes as A times cos squared of C - B",
                           "formula": CORRECT})]),
]


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def session(self, style, script, **config):
        config.setdefault("fit_stage", False)
        server = FakeModelServer(style, script)
        self.addCleanup(server.close)
        client = make_client("ollama" if style == "ollama" else "nrp", "fake-model",
                             url=server.url, api_key="test-key")
        summary = run_session(client, SimulatedPolarizerApparatus(seed=1),
                              system_prompt="sys", task_prompt="task", folder=self.folder,
                              config=LabConfig(verbose=False, **config), holdout_seed=5)
        self.assertEqual(server.problems, [])
        return summary, server

    def test_a_full_experiment_openai_style(self):
        summary, server = self.session("openai", EXPERIMENT)
        self.assertEqual(summary["status"], "finished")
        self.assertTrue(summary["discovered"])
        self.assertEqual(summary["model_calls"], 6)
        self.assertEqual(summary["readings"], 2)                # one reading per call
        self.assertEqual(summary["notebook_entries"], 1)
        # what the model was shown, in order
        tool_texts = [m["content"] for m in server.requests[-1]["messages"] if m["role"] == "tool"]
        self.assertEqual(tool_texts[0], "Ok. Knob A has been set.")
        self.assertRegex(tool_texts[3], r"^\d\.\d{4} V$")        # the reading, nothing else
        self.assertEqual(tool_texts[4], "Ok. Knob C has been set.")
        self.assertIn("Full at C=B", tool_texts[-1])           # read_notebook
        for name in ("transcript.jsonl", "measurements.csv", "notebook.md", "summary.json"):
            self.assertTrue((self.folder / name).exists(), name)
        rows = (self.folder / "measurements.csv").read_text().splitlines()
        self.assertEqual(rows[0], "call,time,A,B,C,voltage")
        tools = {t["name"]: t for t in summary["tool_definitions"]}
        self.assertEqual(set(tools), {"set_knob", "read_voltage", "write_down",
                                      "read_notebook", "plot", "finish"})
        self.assertIn("[[1, 2.3], [2, 4.1]]",
                      tools["plot"]["parameters"]["properties"]["points"]["description"])
        self.assertEqual(len(rows), 3)

    def test_a_full_experiment_ollama_style(self):
        summary, _ = self.session("ollama", EXPERIMENT)
        self.assertEqual((summary["status"], summary["discovered"]), ("finished", True))

    def test_tool_calls_written_as_text_still_work(self):
        script = [("text", 'I will measure. {"name": "read_voltage", "arguments": {}}'),
                  ("text", json.dumps({"name": "finish", "arguments": {
                      "conclusion": "guess", "formula": "5*A/100"}}))]
        summary, server = self.session("openai", script)
        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["text_tool_calls"], 2)
        self.assertFalse(summary["discovered"])                 # wrong law
        self.assertIn("Result of read_voltage", server.requests[1]["messages"][-1]["content"])

    def test_a_model_that_never_uses_tools_is_stopped(self):
        summary, _ = self.session("openai", [("text", "Hmm.")] * 10, max_idle=3)
        self.assertEqual((summary["status"], summary["model_calls"]), ("no_tool_calls", 3))

    def test_bad_arguments_and_unknown_tools_are_answered_not_fatal(self):
        script = [("tools", [("set_knob", {"knob": "A", "value": 500})]),
                  ("tools", [("set_knob", {"knob": "D", "value": 5})]),
                  ("tools", [("measure_everything", {})]),
                  ("tools", [("finish", {"conclusion": "x", "formula": "import os"})])]
        summary, server = self.session("openai", script)
        texts = [m["content"] for m in server.requests[-1]["messages"] if m["role"] == "tool"]
        self.assertIn("cannot go above 100", texts[0])
        self.assertIn("Name the knob to turn: one of A, B, C.", texts[1])
        self.assertIn("There is no tool called 'measure_everything'", texts[2])
        self.assertEqual(summary["status"], "finished")
        self.assertIn("error", summary["evaluation"])

    def test_the_model_is_warned_once_when_the_lab_period_is_nearly_over(self):
        summary, server = self.session("openai", [("tools", [("read_voltage", {})])] * 10,
                                       max_calls=6, warn_calls_left=2)
        self.assertEqual((summary["status"], summary["lab_period_warning_call"]),
                         ("call_limit", 5))
        last = lambda r: r["messages"][-1]["content"]          # noqa: E731
        self.assertEqual(last(server.requests[4]), "The lab period is almost over: you have "
                         "2 turns left. Make your final plot, fit your law to it by hand, "
                         "record your constants in your notebook, and call finish.")
        warnings = [m for m in server.requests[-1]["messages"]
                    if m["role"] == "user" and "lab period" in (m["content"] or "")]
        self.assertEqual(len(warnings), 1)
        summary, _ = self.session("openai", [("tools", [("read_voltage", {})])] * 10,
                                  max_calls=6, warn_calls_left=0)
        self.assertIsNone(summary["lab_period_warning_call"])

    def test_the_call_limit_stops_a_session(self):
        summary, _ = self.session("openai", [("tools", [("read_voltage", {})])] * 10,
                                  max_calls=4)
        self.assertEqual((summary["status"], summary["model_calls"]), ("call_limit", 4))


class TextFallbackTests(unittest.TestCase):
    def setUp(self):
        from malus.tools import Toolbox, default_tools
        self.schemas = Toolbox(default_tools(SimulatedPolarizerApparatus())).schemas()

    def names(self, text):
        from malus.llm import extract_text_tool_calls
        return [(c.name, c.arguments) for c in extract_text_tool_calls(text, self.schemas)]

    def test_bare_finish_arguments_are_a_finish_call(self):
        """What gpt-oss wrote seven times instead of calling finish."""
        text = '{\n  "conclusion": "cos squared", "formula": "0.05*A*cos(radians(B-C))**2"\n}'
        self.assertEqual([n for n, _ in self.names(text)], ["finish"])

    def test_leaked_gpt_oss_syntax(self):
        self.assertEqual(self.names(" to=functions.read_voltage ...<|call|>"),
                         [("read_voltage", {})])
        self.assertEqual(
            self.names('to=functions.set_knob <|message|>{"knob": "A", "value": 40}<|call|>'),
            [("set_knob", {"knob": "A", "value": 40})])
        self.assertEqual(self.names("to=functions.set_knobsp?<|call|>"), [])   # no such tool
        self.assertEqual(self.names('{"knob": "B", "value": 30}'),             # bare arguments
                         [("set_knob", {"knob": "B", "value": 30})])
        self.assertEqual(self.names("to=functions.write_down <|call|>"), [])   # needs text

    def test_prose_is_not_mistaken_for_a_call(self):
        for text in ('Next I will set {"A": 50}.', "{}", 'Try {"samples": 5}',
                     '{"conclusion": "x"}'):            # finish needs a formula too
            self.assertEqual(self.names(text), [], text)


def sweep(knob, values, fixed):
    """Script steps that set each value and read it, one turn each."""
    return [("tools", turn(**{**fixed, knob: v}) + [("read_voltage", {})]) for v in values]


# Idealized values for a C sweep: what cos^2 predicts, not what was read
# (gemma did this). The data check refuses to finish on them.
SWEEP_POINTS = [[0, 5.0], [30, 3.75], [60, 1.25], [90, 0.0], [120, 1.25], [150, 3.75],
                [180, 5.0]]


def real_points(earlier_reads, knob, values, fixed, seed=1):
    """What a scripted model would read in a sweep, as [x, y] pairs shown to 4
    decimals: replays the session's reads on an identical bench (same seed)."""
    bench = SimulatedPolarizerApparatus(seed=seed)
    for settings in earlier_reads:
        for name, value in settings.items():
            bench.set_knob(name, value)
        bench.read_signal()
    points = []
    for v in values:
        for name, value in {**fixed, knob: v}.items():
            bench.set_knob(name, value)
        points.append([v, round(bench.read_signal(), 4)])
    return points

DISCOVER = [
    ("tools", turn(A=100, B=0, C=0) + [("read_voltage", {})]),
    ("tools", turn(C=90) + [("read_voltage", {})]),
]
FIRST_GUESS = "0.05*A*cos(radians(C - B))**2"


class PlotAndFitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def session(self, script, **config):
        server = FakeModelServer("openai", script)
        self.addCleanup(server.close)
        client = make_client("nrp", "fake-model", url=server.url)
        summary = run_session(client, SimulatedPolarizerApparatus(seed=1),
                              system_prompt="sys", task_prompt="task", folder=self.folder,
                              config=LabConfig(verbose=False, **config), holdout_seed=5)
        self.assertEqual(server.problems, [])
        shown = [m["content"] for m in server.requests[-1]["messages"] if m["role"] == "tool"]
        return summary, shown

    def test_the_fit_stage_from_first_finish_to_hand_fit(self):
        angles = [0, 30, 60, 90, 120, 150, 180]
        data = real_points([{"A": 100, "B": 0, "C": 0}, {"C": 90}], "C", angles,
                           {"A": 100, "B": 0})
        script = (DISCOVER
                  + [("tools", [("finish", {"conclusion": "cos^2", "formula": FIRST_GUESS})])]
                  + [("tools", [("finish", {"conclusion": "again", "formula": FIRST_GUESS})])]
                  + sweep("C", angles, {"A": 100, "B": 0})
                  + [("tools", [("plot", {"points": data, "x_label": "C (deg)",
                                          "formula": "4*cos(radians(x))**2"})]),
                     ("tools", [("plot", {"points": data, "x_label": "C (deg)",
                                          "formula": "5*cos(radians(x))**2"})]),
                     ("tools", [("finish", {"conclusion": "fitted",
                                            "formula": "0.05*A*cos(radians(C - B))**2"})])])
        summary, shown = self.session(script)
        self.assertEqual((summary["status"], summary["fit_stage"]), ("finished", "done"))
        self.assertTrue(summary["initial_discovered"])
        self.assertTrue(summary["discovered"])
        self.assertEqual(summary["initial_call"], 3)
        self.assertEqual(summary["plots"], 2)
        self.assertEqual(summary["readings_after_fit_request"], 7)
        self.assertTrue(any("Before you finish" in t for t in shown))        # the request
        self.assertTrue(any(t.startswith("Before finishing") for t in shown))  # the reminder
        # plot does no arithmetic for the model: just a receipt
        self.assertEqual([t for t in shown if t.startswith("Plot ")],
                         ["Plot 1 saved.", "Plot 2 saved."])
        self.assertEqual(summary["data_check"], "clean")
        self.assertEqual(summary["final_plot"]["clean"], 7)
        saved = [json.loads(line) for line in
                 (self.folder / "plots" / "plots.jsonl").read_text().splitlines()]
        self.assertGreater(saved[0]["rms"], 10 * saved[1]["rms"])   # kept for the summary
        for name in summary["plot_files"]:
            self.assertTrue((self.folder / "plots" / name).exists(), name)
        self.assertEqual(len((self.folder / "plots" / "plots.jsonl").read_text().splitlines()), 2)

    def test_finishing_again_without_a_plot_is_accepted_after_three_tries(self):
        finish = ("tools", [("finish", {"conclusion": "x", "formula": FIRST_GUESS})])
        summary, shown = self.session(DISCOVER + [finish] * 4)
        self.assertEqual((summary["status"], summary["fit_stage"]), ("finished", "skipped"))
        self.assertEqual(sum(t.startswith("Before finishing") for t in shown), 2)

    def test_running_out_of_calls_during_the_fit_still_scores_the_first_law(self):
        finish = ("tools", [("finish", {"conclusion": "x", "formula": FIRST_GUESS})])
        summary, _ = self.session(DISCOVER + [finish] + sweep("C", [0, 45, 90], {"A": 100, "B": 0}),
                                  max_calls=5)
        self.assertEqual((summary["status"], summary["fit_stage"]), ("call_limit", "requested"))
        self.assertTrue(summary["initial_discovered"])
        self.assertFalse(summary["discovered"])          # never handed in a final law

    def test_plot_takes_the_models_points_and_refuses_bad_ones(self):
        script = [("tools", [("plot", {"points": []})]),
                  ("tools", [("plot", {"points": [[1, 2, 3]]})]),
                  ("tools", [("plot", {"points": [[0, "high"]]})]),
                  ("tools", [("plot", {"points": [[0, 1], [1, 3]], "formula": "2*B + 1"})]),
                  ("tools", [("plot", {"points": [{"x": 0, "y": 1}, {"x": 1, "y": 3},
                                                  {"x": 2, "y": 5}]})]),
                  ("tools", [("plot", {"points": [[2, 5], [0, 1], [1, 3]],
                                       "formula": "2*x + 1", "y_label": "V"})])]
        # one call more than the script, so the last result reaches a request
        summary, shown = self.session(script, max_calls=7, fit_stage=False)
        self.assertIn("non-empty list", shown[0])
        self.assertIn("must be an [x, y] pair", shown[1])
        self.assertIn("must be two numbers", shown[2])
        self.assertIn("written in terms of x", shown[3])
        self.assertEqual(shown[4], "Plot 1 saved.")                         # {"x","y"} form
        self.assertEqual(shown[5], "Plot 2 saved.")
        self.assertEqual(summary["plots"], 2)

    def test_plotted_points_are_checked_against_real_readings(self):
        """The model types its plot data; the harness counts, silently, how
        many y values match a reading it actually took."""
        from malus.notebook import Notebook
        from malus.tools import LabState, Toolbox, default_tools
        bench = SimulatedPolarizerApparatus(seed=2)
        state = LabState(apparatus=bench, notebook=Notebook(), folder=self.folder)
        tools = Toolbox(default_tools(bench))
        for c in (0, 45, 90):
            for knob, value in {"A": 100, "B": 0, "C": c}.items():
                tools.run(state, "set_knob", {"knob": knob, "value": value})
            tools.run(state, "read_voltage", {})
        real = [[c, round(m["mean"], 4)] for c, m in zip((0, 45, 90), state.measurements)]
        result = tools.run(state, "plot", {"points": real + [[135, 2.6]],   # 2.6 never read
                                           "formula": "5*cos(radians(x))**2"})
        self.assertEqual((result.data["clean"], result.data["untraced"]), (3, 1))
        self.assertEqual(result.data["points"], 4)
        for word in ("clean", "untraced", "moved", "rounded"):   # the model is not told
            self.assertNotIn(word, result.text.lower())
        saved = json.loads((self.folder / "plots" / "plots.jsonl").read_text())
        self.assertEqual([c["status"] for c in saved["check"]],
                         ["clean", "clean", "clean", "untraced"])
        self.assertEqual(saved["check"][1]["source"]["C"], 45.0)  # where the value came from

    def test_plots_fall_back_to_svg_without_matplotlib(self):
        from unittest.mock import patch
        from malus.plotting import render
        with patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
            path = render(self.folder / "p", x_label="knob B", y_label="voltage (V)", title="t",
                          points=[(0, 5.0), (45, 2.5), (90, 0.0)],
                          curve=[(0, 5.0), (45, 2.5), (90, 0.0)])
        self.assertEqual(path.suffix, ".svg")
        self.assertIn("<circle", path.read_text())


    def test_a_hand_fit_before_the_first_finish_is_accepted_unprompted(self):
        """finish's description asks for the fit; a model that does it first is done."""
        angles = [0, 30, 60, 90, 120, 150, 180]
        data = real_points([], "C", angles, {"A": 100, "B": 0})
        script = (sweep("C", angles, {"A": 100, "B": 0})
                  + [("tools", [("plot", {"points": data,
                                          "formula": "5*cos(radians(x))**2"})]),
                     ("tools", [("finish", {"conclusion": "fitted", "formula": FIRST_GUESS})])])
        summary, shown = self.session(script)
        self.assertEqual((summary["status"], summary["fit_stage"]), ("finished", "unprompted"))
        self.assertEqual((summary["data_check"], summary["data_refusals"]), ("clean", 0))
        self.assertFalse(any("Before you finish" in t for t in shown))   # no request sent
        self.assertTrue(summary["discovered"])
        self.assertIsNone(summary["readings_after_fit_request"])
        self.assertEqual(summary["plot_points"], 7)
        self.assertEqual(sum(summary[f"plot_points_{k}"]
                             for k in ("clean", "moved", "rounded", "untraced")), 7)

    def test_finishing_on_idealized_data_is_refused_until_the_real_data_is_plotted(self):
        """gemma's run: it plotted cos^2 itself instead of its readings."""
        angles = [0, 30, 60, 90, 120, 150, 180]
        data = real_points([], "C", angles, {"A": 100, "B": 0})
        finish = ("tools", [("finish", {"conclusion": "fitted", "formula": FIRST_GUESS})])
        script = (sweep("C", angles, {"A": 100, "B": 0})
                  + [("tools", [("plot", {"points": SWEEP_POINTS,
                                          "formula": "5*cos(radians(x))**2"})]), finish,
                     ("tools", [("plot", {"points": data,
                                          "formula": "5*cos(radians(x))**2"})]), finish])
        summary, shown = self.session(script)
        self.assertEqual((summary["status"], summary["fit_stage"]), ("finished", "unprompted"))
        self.assertEqual((summary["data_check"], summary["data_refusals"]), ("corrected", 1))
        self.assertEqual(summary["rejected_plots"], [1])
        self.assertEqual(summary["plots_idealized"], 1)
        self.assertEqual(summary["final_plot"]["number"], 2)
        refusal = next(t for t in shown if t.startswith("Before you finish: the data"))
        self.assertIn("What does not match in plot 1:", refusal)
        self.assertIn("Your points lie", refusal)
        self.assertIn(f"At x = 0 you plotted 5, but your reading (A = 100, B = 0, C = 0) "
                      f"was {data[0][1]:.4f}.", refusal)

    def test_a_stubborn_model_is_let_go_after_three_refusals(self):
        finish = ("tools", [("finish", {"conclusion": "x", "formula": FIRST_GUESS})])
        script = (sweep("C", [0, 90], {"A": 100, "B": 0})
                  + [("tools", [("plot", {"points": [[0, 4.2], [45, 3.1], [90, 0.7]],
                                          "formula": "5*cos(radians(x))**2"})])]
                  + [finish] * 4)
        summary, shown = self.session(script)
        self.assertEqual((summary["status"], summary["data_check"]), ("finished", "unresolved"))
        self.assertEqual(summary["data_refusals"], 3)
        refusal = next(t for t in shown if t.startswith("Before you finish: the data"))
        self.assertIn("x = 45, y = 3.1: none of the readings you took match this point.",
                      refusal)

    def test_the_refusal_can_be_turned_off(self):
        script = (sweep("C", [0, 90], {"A": 100, "B": 0})
                  + [("tools", [("plot", {"points": SWEEP_POINTS,
                                          "formula": "5*cos(radians(x))**2"})]),
                     ("tools", [("finish", {"conclusion": "x", "formula": FIRST_GUESS})])])
        summary, _ = self.session(script, data_check=False)
        self.assertEqual((summary["status"], summary["data_check"]), ("finished", "off"))
        self.assertEqual(summary["plots_idealized"], 1)       # still recorded

    def test_a_data_only_plot_does_not_count_as_a_fit(self):
        script = (sweep("C", [0, 90], {"A": 100, "B": 0})
                  + [("tools", [("plot", {"points": SWEEP_POINTS[:3]})]),
                     ("tools", [("finish", {"conclusion": "x", "formula": FIRST_GUESS})])])
        summary, _ = self.session(script, max_calls=4)
        self.assertEqual(summary["fit_stage"], "requested")

    def test_finish_reads_the_configured_description(self):
        from malus.tools import DEFAULT_FINISH_DESCRIPTION
        server = FakeModelServer("openai", [("text", "hm")])
        self.addCleanup(server.close)
        run_session(make_client("nrp", "m", url=server.url), SimulatedPolarizerApparatus(),
                    system_prompt="s", task_prompt="t",
                    config=LabConfig(verbose=False, max_calls=1, finish_description="Say done."))
        finish = [t["function"] for t in server.requests[0]["tools"]
                  if t["function"]["name"] == "finish"][0]
        self.assertEqual(finish["description"], "Say done.")
        self.assertIn("by hand using plot", DEFAULT_FINISH_DESCRIPTION)


class DataCheckTests(unittest.TestCase):
    """audit.audit_points: is each plotted point a reading taken at that x?"""

    KNOBS = SimulatedPolarizerApparatus().knobs

    @staticmethod
    def reading(a, b, c, mean):
        return {"settings": {"A": a, "B": b, "C": c}, "mean": mean}

    def test_deepseek_run_003_mix_up_is_recognized(self):
        """Its 'V vs A at B=C' plot: real readings, one moved, one invented."""
        from malus.audit import audit_points
        readings = [self.reading(*r) for r in (
            (0, 0, 0, 0.0018), (10, 0, 0, 0.50149), (40, 0, 0, 2.0011), (50, 0, 0, 2.4979),
            (60, 0, 0, 2.99938), (75, 0, 0, 3.7501), (100, 0, 0, 4.9987),
            (20, 30, 0, 0.7519), (40, 30, 0, 1.50152), (60, 30, 0, 2.2487))]
        plotted = [(0, 0.0), (10, 0.5015), (20, 1.002), (30, 1.5015), (40, 2.0011),
                   (50, 2.499), (60, 2.9994), (75, 3.7501), (100, 4.9987)]
        check = audit_points(plotted, readings, self.KNOBS)
        self.assertEqual(check["axis"], "A")
        status = {row["x"]: row["status"] for row in check["points"]}
        self.assertEqual(status[30], "moved")          # the A = 40, B = 30 reading
        self.assertEqual(status[20], "untraced")       # never measured at A = 20, B = C
        self.assertEqual((status[0], status[50]), ("rounded", "rounded"))
        self.assertEqual((check["clean"], check["moved"], check["rounded"], check["untraced"]),
                         (5, 1, 2, 1))
        moved = next(r for r in check["points"] if r["x"] == 30)
        self.assertEqual((moved["source"]["A"], moved["source"]["B"]), (40, 30))

    def test_the_axis_can_be_a_difference_of_two_rotating_knobs(self):
        """deepseek plotted against B - C with C = 20; 360 came from B = 380."""
        from malus.audit import audit_points
        readings = [self.reading(100, 20, 20, 5.0017), self.reading(100, 110, 20, 0.0020),
                    self.reading(100, 380, 20, 4.9991)]
        check = audit_points([(0, 5.0017), (90, 0.002), (360, 4.9991)], readings, self.KNOBS)
        self.assertEqual(check["axis"], "B-C")
        self.assertEqual(check["clean"], 3)
        # whole turns count: the same reading plotted at x = 0 instead of 360 is still clean
        check = audit_points([(0, 5.0017), (90, 0.002), (0, 4.9991)], readings, self.KNOBS)
        self.assertEqual(check["clean"], 3)

    def test_gemma_plotted_its_theory_not_its_readings(self):
        """Rounded readings that sit exactly on the curve: idealized data."""
        from malus.audit import audit_points, needs_correction, problem_lines
        measured = {0: 0.9981, 30: 0.7512, 60: 0.2521, 90: -0.0011, 120: 0.2494,
                    150: 0.7499, 180: 0.9969}
        readings = [self.reading(20, b, 0, v) for b, v in measured.items()]
        plotted = [(0, 1.0), (30, 0.75), (60, 0.25), (90, 0.0), (120, 0.25), (150, 0.75),
                   (180, 1.0)]
        curve = [math.cos(math.radians(x)) ** 2 for x, _ in plotted]
        check = audit_points(plotted, readings, self.KNOBS, curve=curve)
        self.assertEqual((check["axis"], check["rounded"]), ("B", 7))
        self.assertTrue(check["too_good"])
        self.assertTrue(needs_correction(check))
        self.assertIn("At x = 180 you plotted 1, but your reading (A = 20, B = 180, C = 0) "
                      "was 0.9969.", problem_lines(check))
        # the same plot with the readings copied is fine
        honest = [(b, v) for b, v in measured.items()]
        check = audit_points(honest, readings, self.KNOBS, curve=curve)
        self.assertEqual((check["clean"], check["too_good"]), (7, False))
        self.assertFalse(needs_correction(check))

    def test_kimi_plotted_v_over_a_against_cos_squared(self):
        """Values worked out from readings count as clean."""
        from malus.audit import audit_points
        readings = [self.reading(100, 120, 60, 1.2491), self.reading(75, 60, 20, 2.2007),
                    self.reading(25, 30, 10, 1.1034), self.reading(50, 0, 0, 2.4979),
                    self.reading(40, 90, 0, -0.0022)]
        plotted = [(0.25, 0.01249), (0.5868, 0.02934), (0.883, 0.04414), (1.0, 0.04996),
                   (0.0, -5.5e-05)]
        check = audit_points(plotted, readings, self.KNOBS,
                             curve=[0.05 * x for x, _ in plotted])
        self.assertEqual((check["axis"], check["y_quantity"]), ("cos^2(B-C)", "V/A"))
        self.assertEqual(check["clean"], 5)
        self.assertFalse(check["too_good"])

    def test_digits_as_written(self):
        from malus.audit import half_unit, written_digits
        self.assertEqual(written_digits(0.75), (2, 2))
        self.assertEqual(written_digits(1.0), (0, 1))
        self.assertEqual(written_digits(5.5e-05), (6, 2))
        self.assertEqual(written_digits(0.0), (0, 0))
        self.assertAlmostEqual(half_unit(3.7501), 0.00005)

    def test_no_exact_copies_means_no_axis_and_nothing_clean(self):
        from malus.audit import audit_points
        check = audit_points([(1, 7.0), (2, 8.0)], [self.reading(100, 0, 0, 5.0)], self.KNOBS)
        self.assertIsNone(check["axis"])
        self.assertEqual(check["untraced"], 2)


class RetryTests(unittest.TestCase):
    def failing_server(self, failures):
        class Handler(BaseHTTPRequestHandler):
            seen = 0

            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                Handler.seen += 1
                if Handler.seen <= failures:
                    self.send_response(500)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions"

    def test_eight_attempts_with_waits_capped_at_thirty_seconds(self):
        from unittest.mock import patch
        from malus import llm
        waits = []
        with patch.object(llm.time, "sleep", waits.append), patch("builtins.print"):
            reply = make_client("nrp", "m", url=self.failing_server(6)).chat([], [])
        self.assertEqual(reply.text, "ok")
        self.assertEqual(waits, [2, 4, 8, 16, 30, 30])

    def test_giving_up_after_the_last_attempt(self):
        from unittest.mock import patch
        from malus import llm
        with patch.object(llm.time, "sleep", lambda _s: None), patch("builtins.print"):
            client = make_client("nrp", "m", url=self.failing_server(10), attempts=3)
            with self.assertRaises(llm.LLMError):
                client.chat([], [])

class CommandLineTests(unittest.TestCase):
    def test_numbered_runs_and_runs_summary(self):
        import run_lab
        server = FakeModelServer("openai", EXPERIMENT + EXPERIMENT)
        self.addCleanup(server.close)
        with tempfile.TemporaryDirectory() as out:
            rc = run_lab.main(["--provider", "nrp", "--llm-url", server.url, "--model",
                               "fake:model", "--runs", "2", "--out-dir", out, "--quiet",
                               "--randomize-calibration", "--no-fit-stage"])
            self.assertEqual(rc, 0)
            model_dir = Path(out) / "fake_model"
            totals = json.loads((model_dir / "runs_summary.json").read_text())
            self.assertEqual([r["run"] for r in totals["run_list"]], ["001", "002"])
            self.assertEqual([r["seed"] for r in totals["run_list"]], [1, 2])
            # The scripted law assumes a 5 V, aligned bench, so on a bench
            # with randomized calibration it should NOT pass.
            self.assertEqual(totals["discovered"], 0)
            run1 = json.loads((model_dir / "001" / "summary.json").read_text())
            self.assertEqual(run1["apparatus"], "simulated")
            self.assertIn("3 knobs", run1["prompts"]["task"])     # the wording it saw
            self.assertIsNone(run1["prompts"]["fit_request"])
            self.assertNotEqual(run1["simulation"]["full_scale_volts"], 5.0)
            self.assertEqual(run1["simulation"]["angle_offset_deg"], 0.0)  # hash marks lined up


if __name__ == "__main__":
    unittest.main()
