# Malus: can an LLM discover a law of physics from knobs and a voltmeter?

The bench is a bulb with an intensity controller, two rotatable polarizers and
a photodetector, so the signal follows Malus's law:

    V = V0 * (A/100) * cos^2(C - B)

The model is told none of that. It gets this task and nothing else:

> You have 3 knobs. One goes from 0 to 100; the other two rotate continuously.
> Depending on your knob settings, you'll get a single output voltage, which is
> your signal. There is a law of physics here, and I want you to discover what
> mathematical form it has.

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
`LLM_API_KEY` set the defaults. The model must support tool calling; if it writes
a tool call as JSON in plain text instead, that is accepted too and counted in
`text_tool_calls`.

## The tools

| Tool | What it does |
| --- | --- |
| `set_knobs` | Turn one or more of A, B, C. A is limited to 0..100; B and C take any angle in degrees. |
| `read_voltage` | Read the voltage at the current settings, optionally averaging up to 20 readings. |
| `write_down` | Add an entry to the lab notebook. |
| `read_notebook` | Read back every notebook entry. |
| `finish` | Announce it is done, with a written conclusion and the law as a Python expression in A, B and C. |

The tool descriptions say "knob" and "voltage", never light or polarizers. To see
exactly what the model reads, look at the top of `malus/tools.py` and at
`prompts/system.txt` and `prompts/task.txt` (edit those to change the wording).

## What a run produces

```text
Runs/<model>/001/transcript.jsonl   every message, tool call and result
                 measurements.csv   every reading it took, with the knob settings
                 notebook.md        what it wrote down (also notebook.jsonl)
                 summary.json       outcome, its conclusion and law, the check
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

Run statuses: `finished` (it called finish), `call_limit` (hit `--max-calls`,
default 150), `no_tool_calls` (five replies in a row without using a tool),
`error` (the endpoint kept failing) and `interrupted` (Ctrl-C).
`runs_summary.json` gives `discovered` out of `judged` runs, where judged leaves
out `error` and `interrupted`.

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
malus/tools.py        the five tools
malus/notebook.py     the lab notebook
malus/llm.py          Ollama and OpenAI-compatible tool calling
malus/lab.py          the conversation loop and run folders
malus/evaluate.py     safe formula parsing and the final check
prompts/              system prompt and task
tests/test_lab.py     offline tests against fake OpenAI and Ollama servers
```

Run the tests with `python3 -m unittest tests.test_lab`.
