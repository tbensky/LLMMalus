# Malus: can an LLM discover a law of physics from knobs and a voltmeter?

The bench is a bulb with an intensity controller, two rotatable polarizers and
a photodetector, so the signal follows Malus's law:

    V = V0 * (A/100) * cos^2(C - B)

The model is told none of that. It gets this task and nothing else:

> You have 3 knobs. One goes from 0 to 100, the other are continuously
> rotateable. Depending on your knob settings, you'll get a single output
> voltage, which is your signal. There is a law of physics here, and I want you
> to discover what mathematical form it has. You have a lab notebook you can use
> to record things you find. Use "write_down(...)" to record whatever you wish.
> To read it back at any time, use "read_notebook. You are to work by yourself,
> with no internet searches and you cannot ask questions. Just figure it all out
> for yourself.

(The exact text is in `prompts/task.txt`.)

It works the bench through tools, keeps a lab notebook, and says when it is done.
Its law is then checked against fresh readings at knob settings it never chose.

## Quick start

Standard library only (Python 3.10+), nothing to install.

```bash
# A hosted model on NRP, using the same LLM_* settings as GASolver's client
export LLM_API_KEY=...                       # your NRP key
python3 run_lab.py --provider nrp --model deepseek-v4-flash

# A local Ollama model, 10 independent runs
python3 run_lab.py --provider ollama --model qwen3:8b --runs 10
```

`--provider` can be `nrp`, `openai` or `deepseek` (any OpenAI-compatible
`/v1/chat/completions` endpoint; give `--llm-url` for others) or `ollama` (native
`/api/chat` on `localhost:11434`). `LLM_PROVIDER`, `LLM_URL`, `LLM_MODEL` and
`LLM_API_KEY` set the defaults. The model must support tool calling. Calls a model
writes as text instead are accepted too and counted in `text_tool_calls`: a call
written out as JSON, just a tool's arguments (gpt-oss handed in `finish`'s
`{"conclusion": ..., "formula": ...}` this way), or gpt-oss's own
`to=functions.read_voltage` syntax leaking through the host.

When the host errors (HTTP 429 or 5xx, or a timeout) a model call is retried up to
8 times, waiting 2, 4, 8, 16 and then 30 seconds (`--retries`). After the last
attempt the run ends with status `error`, which `runs_summary.json` leaves out.

## The tools

| Tool | What it does |
| --- | --- |
| `set_knobs` | Turn one or more of A, B, C. A is limited to 0..100; B and C take any angle in degrees. |
| `read_voltage` | Read the voltage at the current settings, optionally averaging up to 20 readings. |
| `write_down` | Add an entry to the lab notebook. |
| `read_notebook` | Read back every notebook entry. |
| `plot` | Graph a list of (x, y) points it supplies itself, like making a plot in a spreadsheet, optionally with its own curve written in terms of x. Saves a PNG and answers with each point next to the curve's value. |
| `finish` | Announce it is done, with a written conclusion and the law as a Python expression in A, B and C. Its description also asks for systematic data and a hand fit with `plot` before finishing (`prompts/finish_tool.txt`). |

The tool descriptions say "knob" and "voltage", never light or polarizers. To see
exactly what the model reads, look at the top of `malus/tools.py` and the files in
`prompts/` (edit those to change the wording).

## The finish line: systematic data and a hand fit

`finish`'s own description (`prompts/finish_tool.txt`) tells the model what is
expected before it finishes: take a set of data it judges right for an X/Y plot
that illustrates and proves its conclusions, fit the law to it
**by hand** (choose the constants, plot, look, adjust, plot again; no automatic
fitting) and record the constants. Its law at the first `finish` is recorded
(`initial_formula`) either way. What happens next depends on whether it did the fit:

- **It already plotted a curve:** the finish is accepted (`fit_stage: unprompted`).
- **It hasn't:** the harness treats the finish as "I understand the knobs" and,
  instead of ending, answers with `prompts/fit_request.txt`, which repeats the
  request; it then has to call `finish` again. A plot of data alone, without a
  curve, does not count as a fit.

- **The model never runs code.** `plot` is drawn by the harness. The model
  sends a list of `[x, y]` points it took from its own work, axis labels, and
  optionally a curve: a Python expression in `x` with its hand-picked constants,
  e.g. `3.7*cos(radians(x + 25))**2`. It decides what goes on each axis, so it can
  plot voltage against a knob, against B − C, or anything it derives.
- **It gets a table back.** A text-only model cannot look at the PNG, so `plot`
  answers with each point next to the curve's value and the RMS difference.
  That table is its view of the graph when it adjusts its constants.
- **Its data is checked, silently** (`malus/audit.py`). Because the model types
  the numbers, a plot could show values it never measured, or a real reading
  moved to the wrong settings. The model never says what its x axis is, so the
  check infers it: the knob, or difference of two knobs (whole turns allowed for
  B and C), that the most exactly-copied points line up with. Each point is then
  `clean` (an exact copy, to the 4 decimals it was shown, of a reading taken at
  that x), `moved` (an exact copy of a reading taken at other settings),
  `rounded` (within 0.002 of a reading at that x) or `untraced` (no reading at
  that x comes close). The counts are in each summary (`plot_points_clean` and
  so on, out of `plot_points`), and `plots/plots.jsonl` keeps, for every point,
  its status and the reading it came from. The model is not told. A y it derived
  itself, such as V/A, shows as untraced; that is worth a look, not
  automatically an error. Checked on real runs: qwen3's 13 points are all clean;
  deepseek's "V vs A" plot (run 003) has 5 clean, 1 moved (its A = 40, B = 30
  reading plotted as A = 30), 2 rounded and 1 untraced (A = 20, never measured
  there).
- **When a second finish is accepted:** once at least one new plot exists
  since the request. After three finishes without a plot it is accepted anyway
  (`fit_stage: skipped`), so a stubborn model cannot loop forever.
- **Both laws are scored:** the first law and the final hand-fitted law are
  scored against the same fresh readings, so a run shows whether it understood
  the knobs and whether its fit held up.

`--no-fit-stage` turns the gate off (the first `finish` ends the run; `finish`'s
description still asks for the fit unless you change `prompts/finish_tool.txt`),
and `--fit-request-file` / `--finish-tool-file` point at different wording.

## What a run produces

```text
Runs/<model>/001/transcript.jsonl   every message, tool call and result
                 measurements.csv   every reading it took, with the knob settings
                 notebook.md        what it wrote down (also notebook.jsonl)
                 plots/plot_01.png  each graph it made (plots.jsonl has the data and curve)
                 summary.json       outcome, its first and final law, both checks,
                                    and the exact prompts it saw
Runs/<model>/002/...
Runs/<model>/runs_summary.json      how many runs found the law
```

Numbering continues across batches, so a second `--runs 50` adds 051 to 100.

**Finding the law** is judged from the formula passed to `finish`. It is
evaluated with a whitelist parser (numbers, A, B, C, arithmetic and `math`
functions, nothing else), then compared with 40 fresh readings at random
settings. The score is the RMS error divided by the range of the readings; 5% or
less counts as found (`--threshold`). A correct law with fitted constants passes
easily; cos instead of cos^2, or a law that ignores a knob, fails.

Run statuses: `finished` (its final finish was accepted), `call_limit` (hit
`--max-calls`, default 250), `no_tool_calls` (five replies in a row without using
a tool), `error` (the endpoint kept failing) and `interrupted` (Ctrl-C).
`fit_stage` in each summary is `unprompted` (fitted before its first finish),
`done` (fitted after being asked), `skipped`, `requested` (asked for the fit but
never finished again), `not_reached` (never called finish) or `off`.

`runs_summary.json` gives `discovered` (the final law passed) out of `judged`
runs, where judged leaves out `error` and `interrupted`, plus
`discovered_at_first_finish`, `fit_stage_done` (unprompted or after being asked)
and `fit_done_unprompted`.

## Making the bench harder

The simulation defaults to a clean bench (5 V full scale, 5 mV noise, perfect
polarizers). Options:

| Option | Effect |
| --- | --- |
| `--noise 0.02` | reading noise, volts |
| `--dark 0.05` | detector offset with no light |
| `--extinction 0.01` | light leaking through crossed polarizers |
| `--angle-offset 37` | knob B's zero mark is 37 degrees off knob C's |
| `--bulb-exponent 2.5` | the intensity controller is not linear |
| `--saturation 4.5` | the detector clips |
| `--randomize-calibration` | random full scale, dark offset and angle offset each run, so no constant can be memorized |
| `--keep-last 10` | only the last 10 exchanges stay in context; older findings survive only in the notebook |

Run NNN uses `--seed` + NNN − 1 for its noise and calibration, so runs repeat exactly.

## Connecting the real bench

Everything that knows about the physics is in `malus/apparatus.py`. The tools,
the loop and the check only use the `Apparatus` interface:

```python
class Apparatus:
    knobs                         # KnobSpec for A, B, C (what the model is told)
    def set_knob(name, value)     # move a knob, block until settled, return actual position
    def get_knobs()               # where every knob is now
    def read_signal()             # one detector reading, volts
    def close()                   # bulb off, park motors
```

`HardwareApparatus` is a stub with those methods: fill them in for the bulb
controller, the two rotation stages and the ADC, then run with
`--apparatus hardware`. Raise `ApparatusError` with a plain message for anything
the bench cannot do; the model sees it as the tool result. The final check takes
its 40 fresh readings through the same interface, so it works on the real bench
unchanged (it just takes as long as 40 real measurements).

## Layout

```text
run_lab.py            command line: runs, folders, summaries
malus/apparatus.py    Apparatus interface, simulated bench, hardware stub
malus/tools.py        the six tools
malus/plotting.py     drawing a plot (matplotlib PNG, or SVG without matplotlib)
malus/notebook.py     the lab notebook
malus/llm.py          Ollama and OpenAI-compatible tool calling
malus/lab.py          the conversation loop and run folders
malus/evaluate.py     safe formula parsing and the final check
prompts/              system prompt, task, finish's description, the fit request
tests/test_lab.py     offline tests against fake OpenAI and Ollama servers
```

Run the tests with `python3 -m unittest tests.test_lab`.
