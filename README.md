# Malus: can an LLM discover a law of physics from knobs and a voltmeter?

The bench is a bulb with an intensity controller, two rotatable polarizers and
a photodetector, so the signal follows Malus's law:

    V = V0 * L(A) * cos^2(C - B),    L(A) = 1 / (1 + exp(-0.17 * (A - 40)))

L(A) is the bulb's brightness, modeled on the real bench's bulb: dark until
A is about 20, half brightness at 40, saturated by about 65 (0.001 at A = 0,
0.03 at 20, 0.97 at 60, 1.00 at 100). Runs before this bulb was added used a
linear one, L(A) = A/100; `--bulb power` brings it back, and each run's
`summary.json` records which bulb it had under `simulation`.

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
DeepSeek's own call markup can leak too, folding a second call into the
first one's arguments: run 005 sent `write_down` with
`{"text": "...<｜DSML｜ invoke name=\"set_knob\">", "knob": "B", "value": 0}`,
so the notebook got the markup and the knob never moved. Such a call is split
back into the two calls the model made (the note, cut where the markup
starts, and the knob move) and counted in `leaked_tool_calls`.

When the host errors (HTTP 429 or 5xx, or a timeout) a model call is retried up to
8 times, waiting 2, 4, 8, 16 and then 30 seconds (`--retries`). After the last
attempt the run ends with status `error`, which `runs_summary.json` leaves out.

## The tools

| Tool | What it does |
| --- | --- |
| `set_knob` | Turn one knob: `knob` (A, B or C) and `value`. A is limited to 0..100; B and C take any angle in degrees. Replies only "Ok. Knob B has been set.", with no positions: the model keeps track of its settings itself. |
| `read_voltage` | Read the voltage at the current settings: one reading (5 mV of noise), no averaging, and only the number (e.g. `0.9487 V`): keeping track of the knob settings is up to the model. To average, it has to call it again. |
| `write_down` | Add an entry to the lab notebook. |
| `read_notebook` | Read back every notebook entry. |
| `plot` | Graph a list of (x, y) points it supplies itself, like making a plot in a spreadsheet, optionally with its own curve written in terms of x. Saves a PNG and answers only "Plot N saved."; it computes nothing for the model. |
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
- **It gets nothing back but a receipt.** `plot` answers only "Plot 2 saved.":
  no curve values, no differences, no RMS. A text-only model cannot look at the
  PNG, so to judge its fit it has to work out its curve at its own points and
  compare. (The harness still records each plot's RMS for the summary.) The
  example in `plot`'s description is a neutral `[[1, 2.3], [2, 4.1]]`; until
  this change it was 5·cos² at 0, 15 and 30 degrees, which every run before it
  saw.
- **Its data is checked** (`malus/audit.py`). Because the model types the
  numbers, a plot could show values it never measured, a real reading moved to
  the wrong settings, or idealized values that sit on its curve. The model never
  says what is on its axes, so the check infers them, trying every pairing of
  an x (a knob, a difference of two knobs with whole turns allowed for B and C,
  or cos^2 or cos of an angle or angle difference) with a y (the voltage, V/A or
  V/(A/100)) and taking the one that accounts for the most points. Each point
  is then `clean` (its reading at that x, copied exactly to the 4 decimals shown,
  or worked out from it to at least 3 significant figures), `rounded` (its
  reading at that x with fewer digits, e.g. 0.75 for 0.7512), `moved` (an exact
  copy of a reading taken at other settings) or `untraced` (no reading at that
  x comes close). A plot with a curve is also flagged as idealized when its
  points lie on the curve far more closely (4x or more, in RMS) than the
  readings they stand for, or exactly on it.
  At `finish` the model's law is one more possible x: what the law predicts
  at each reading's settings. That recognizes a "measured against predicted"
  plot (kimi 002 plotted each reading against its law's prediction, with the
  line y = x, and was wrongly refused three times before this was added). A
  prediction worked out wrong is still refused, with "your law does not give
  0.589 there" and no correct value.
- **Finishing on bad data is refused.** At `finish`, the last plot with a curve
  is judged. If it has moved or untraced points, or is idealized, the finish is
  refused with `prompts/data_check.txt` followed by the specific points (e.g.
  "At x = 180 you plotted 1, but your reading (A = 20, B = 180, C = 0) was
  0.9969."). Rounded points are fine. After three refusals the finish is
  accepted anyway (`data_check: unresolved`). `--no-data-check` keeps the check
  and its counts but never refuses.
- **Checked on the runs so far:** qwen3 (13 points) and deepseek 003's B − C
  plot (24) are all clean; kimi's V/A against cos^2(B − C) is recognized (12
  clean, 2 rounded); deepseek 003's "V vs A" plot would be refused (its A = 40,
  B = 30 reading plotted at A = 30, and a point at A = 20 it never measured);
  gemma's plot would be refused as idealized (7 rounded points lying exactly on
  cos^2, RMS 1e-16 against 0.0017 for its readings).
- **When a second finish is accepted:** once at least one new plot exists
  since the request. After three finishes without a plot it is accepted anyway
  (`fit_stage: skipped`), so a stubborn model cannot loop forever.
- **Both laws are scored:** the first law and the final hand-fitted law are
  scored against the same fresh readings, so a run shows whether it understood
  the knobs and whether its fit held up.

`--no-fit-stage` turns the gate off (the first `finish` ends the run; `finish`'s
description still asks for the fit unless you change `prompts/finish_tool.txt`),
and `--fit-request-file` / `--finish-tool-file` / `--data-check-file` point at
different wording.

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

**The lab period.** A run gets 400 model calls (`--max-calls`). When 40 are
left (`--warn-calls-left`, 0 for never) the model is told once, like a class
near the end of a lab: "The lab period is almost over: you have 40 turns left.
Make your final plot, fit your law to it by hand, record your constants in your
notebook, and call finish." (`prompts/lab_period.txt`; `{left}` is filled in).
`lab_period_warning_call` in the summary says when it was told. minimax ran out
of calls at the old limit of 250 without ever finishing, with the right law in
its notebook since call 34.

Run statuses: `finished` (its final finish was accepted), `call_limit` (hit
`--max-calls`, default 400), `no_tool_calls` (five replies in a row without using
a tool), `error` (the endpoint kept failing) and `interrupted` (Ctrl-C).
`fit_stage` in each summary is `unprompted` (fitted before its first finish),
`done` (fitted after being asked), `skipped`, `requested` (asked for the fit but
never finished again), `not_reached` (never called finish) or `off`.
`data_check` is `clean` (its fit plot passed the first time), `corrected`
(passed after being refused), `unresolved` (accepted after three refusals),
`rejected` (refused and never finished again), `not_checked` (never finished on
a plot with a curve) or `off`; `data_refusals`, `rejected_plots`,
`plots_idealized` and `final_plot` (the plot it was judged on, with its
counts) go with it.

`runs_summary.json` gives `discovered` (the final law passed) out of `judged`
runs, where judged leaves out `error` and `interrupted`, plus
`discovered_at_first_finish`, `fit_stage_done` (unprompted or after being asked),
`fit_done_unprompted`, and `data_clean_first_time`, `data_corrected` and
`data_unresolved`.

## Making the bench harder

The simulation defaults to a clean bench apart from the bulb (5 V full scale,
5 mV noise, perfect polarizers, the logistic bulb above). Options:

| Option | Effect |
| --- | --- |
| `--noise 0.02` | reading noise, volts |
| `--dark 0.05` | detector offset with no light |
| `--extinction 0.01` | light leaking through crossed polarizers |
| `--angle-offset 37` | knob B's zero mark is 37 degrees off knob C's |
| `--bulb-midpoint 50 --bulb-steepness 0.1` | a different logistic bulb: half brightness at A = 50, a gentler rise |
| `--bulb power` | brightness (A/100)**`--bulb-exponent`; exponent 1 (the default) is the linear bulb of earlier runs |
| `--saturation 4.5` | the detector clips |
| `--randomize-calibration` | random full scale (2 to 9 V) and dark offset (0 to 0.05 V) each run, so no constant can be memorized. The angle offset is not randomized: real polarizers have a hash mark on the transmission axis that students line up as the zero. Add `--angle-offset` for a misaligned bench |
| `--keep-last 10` | only the last 10 exchanges stay in context; older findings survive only in the notebook |

Run NNN uses `--seed` + NNN − 1 for its noise and calibration, so runs repeat exactly.

## Running on the cluster

Open-weight models can run on CSU TIDE through Ollama on a GPU, with models
cached on a claim so each is downloaded once:

```bash
./malusctl build && ./malusctl volumes      # once
./malusctl start qwen3:8b 10                # a batch of 10 runs
./malusctl status
./malusctl fetch qwen3:8b                   # -> Runs/k8s/qwen3_8b/
```

See `k8s/README.md`.

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
malus/audit.py        the data check on plotted points
prompts/              system prompt, task, finish's description, the fit request,
                      the data-check refusal, the lab-period warning
tests/test_lab.py     offline tests against fake OpenAI and Ollama servers
malusctl, k8s/        running batches on CSU TIDE (k8s/README.md)
```

Run the tests with `python3 -m unittest tests.test_lab`.
