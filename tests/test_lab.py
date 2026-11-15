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
        if names != {"set_knobs", "read_voltage", "write_down", "read_notebook", "finish"}:
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


EXPERIMENT = [
    ("tools", [("set_knobs", {"A": 100, "B": 0, "C": 0})]),
    ("tools", [("read_voltage", {"samples": 3})]),
    ("tools", [("set_knobs", {"C": 90}), ("read_voltage", {})]),
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
        self.assertEqual(summary["readings"], 4)                # 3 samples + 1
        self.assertEqual(summary["notebook_entries"], 1)
        # what the model was shown, in order
        tool_texts = [m["content"] for m in server.requests[-1]["messages"] if m["role"] == "tool"]
        self.assertIn("Knobs now: A = 100, B = 0, C = 0.", tool_texts[0])
        self.assertIn("mean of 3 readings", tool_texts[1])
        self.assertIn("Full at C=B", tool_texts[-1])           # read_notebook
        for name in ("transcript.jsonl", "measurements.csv", "notebook.md", "summary.json"):
            self.assertTrue((self.folder / name).exists(), name)
        rows = (self.folder / "measurements.csv").read_text().splitlines()
        self.assertEqual(rows[0], "call,time,A,B,C,samples,mean,stdev")
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
        script = [("tools", [("set_knobs", {"A": 500})]),
                  ("tools", [("measure_everything", {})]),
                  ("tools", [("finish", {"conclusion": "x", "formula": "import os"})])]
        summary, server = self.session("openai", script)
        texts = [m["content"] for m in server.requests[-1]["messages"] if m["role"] == "tool"]
        self.assertIn("cannot go above 100", texts[0])
        self.assertIn("There is no tool called 'measure_everything'", texts[1])
        self.assertEqual(summary["status"], "finished")
        self.assertIn("error", summary["evaluation"])

    def test_the_call_limit_stops_a_session(self):
        summary, _ = self.session("openai", [("tools", [("read_voltage", {})])] * 10,
                                  max_calls=4)
        self.assertEqual((summary["status"], summary["model_calls"]), ("call_limit", 4))


class CommandLineTests(unittest.TestCase):
    def test_numbered_runs_and_runs_summary(self):
        import run_lab
        server = FakeModelServer("openai", EXPERIMENT + EXPERIMENT)
        self.addCleanup(server.close)
        with tempfile.TemporaryDirectory() as out:
            rc = run_lab.main(["--provider", "nrp", "--llm-url", server.url, "--model",
                               "fake:model", "--runs", "2", "--out-dir", out, "--quiet",
                               "--randomize-calibration"])
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
            self.assertNotEqual(run1["simulation"]["full_scale_volts"], 5.0)


if __name__ == "__main__":
    unittest.main()
