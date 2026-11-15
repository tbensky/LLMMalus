#!/usr/bin/env python3
"""Let an LLM loose on the knob-and-voltage experiment and see if it finds the law.

Examples:
    # a hosted model on NRP (same LLM_* settings as GASolver's client)
    export LLM_API_KEY=...            # your NRP key
    python3 run_lab.py --provider nrp --model deepseek-v4-flash

    # a local Ollama model, 10 independent runs
    python3 run_lab.py --provider ollama --model qwen3:8b --runs 10

    # a less ideal bench: noisy, leaky polarizers, unaligned zero marks
    python3 run_lab.py --noise 0.02 --extinction 0.01 --angle-offset 37

Results go in Runs/<model>/<NNN>/, with Runs/<model>/runs_summary.json
counting how many runs found the law.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from malus.apparatus import HardwareApparatus, SimulatedPolarizerApparatus, SimulationConfig  # noqa: E402
from malus.lab import (DEFAULT_FIT_REQUEST, LabConfig, next_run_dir, run_session,  # noqa: E402
                       safe_model_name, write_runs_summary)
from malus.llm import make_client  # noqa: E402
from malus.tools import DEFAULT_FINISH_DESCRIPTION  # noqa: E402

MAX_CONSECUTIVE_ERRORS = 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    llm = p.add_argument_group("model")
    llm.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "nrp"),
                     choices=("nrp", "openai", "deepseek", "ollama"),
                     help="nrp/openai/deepseek: OpenAI-compatible host; ollama: local /api/chat")
    llm.add_argument("--llm-url", default=os.environ.get("LLM_URL"),
                     help="Endpoint URL (default depends on --provider)")
    llm.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-v4-flash"))
    llm.add_argument("--llm-api-key", default=os.environ.get("LLM_API_KEY", ""),
                     help="API key for a hosted model (default: $LLM_API_KEY)")
    llm.add_argument("--temperature", type=float, default=None,
                     help="Sampling temperature (default: the host's own)")
    llm.add_argument("--num-ctx", type=int, default=16384, help="Ollama context window")
    llm.add_argument("--think", choices=("on", "off"), default=None,
                     help="Ollama only: turn a thinking model's thinking on or off")
    llm.add_argument("--timeout", type=float, default=300.0, help="Seconds per model call")
    llm.add_argument("--retries", type=int, default=8,
                     help="Attempts per model call when the host errors (waits 2, 4, 8, 16, "
                          "then 30 s); after the last one the run ends with status error")

    lab = p.add_argument_group("session")
    lab.add_argument("--runs", type=int, default=1, help="Independent runs to do in a row")
    lab.add_argument("--max-calls", type=int, default=250,
                     help="Model calls per run (the fit stage needs room after the first finish)")
    lab.add_argument("--keep-last", type=int, default=0,
                     help="Keep only the last N exchanges in context (0 = all); "
                          "anything older survives only in the notebook")
    lab.add_argument("--threshold", type=float, default=0.05,
                     help="Normalized RMS error on fresh readings that counts as finding the law")
    lab.add_argument("--holdout-points", type=int, default=40,
                     help="Fresh readings used to check the final law")
    lab.add_argument("--system-prompt-file", default=str(HERE / "prompts" / "system.txt"))
    lab.add_argument("--task-prompt-file", default=str(HERE / "prompts" / "task.txt"))
    lab.add_argument("--fit-request-file", default=str(HERE / "prompts" / "fit_request.txt"),
                     help="Sent in reply to the first finish: take data and fit by hand")
    lab.add_argument("--finish-tool-file", default=str(HERE / "prompts" / "finish_tool.txt"),
                     help="What the model reads about the finish tool")
    lab.add_argument("--no-fit-stage", action="store_true",
                     help="Accept the first finish; no systematic data, hand fit or plots asked for")
    lab.add_argument("--out-dir", default=str(HERE / "Runs"))
    lab.add_argument("--quiet", action="store_true", help="Only print one line per run")

    bench = p.add_argument_group("apparatus (simulation)")
    bench.add_argument("--apparatus", choices=("sim", "hardware"), default="sim")
    bench.add_argument("--seed", type=int, default=1,
                       help="Base seed; run NNN uses seed + NNN - 1 for noise and calibration")
    bench.add_argument("--full-scale", type=float, default=5.0, help="Volts at A=100, aligned")
    bench.add_argument("--dark", type=float, default=0.0, help="Detector offset, volts")
    bench.add_argument("--noise", type=float, default=0.005, help="Reading noise, volts (std dev)")
    bench.add_argument("--extinction", type=float, default=0.0,
                       help="Fraction leaking through crossed polarizers")
    bench.add_argument("--angle-offset", type=float, default=0.0,
                       help="Hidden offset between the B and C zero marks, degrees")
    bench.add_argument("--bulb-exponent", type=float, default=1.0,
                       help="Light ~ (A/100)**exponent; 1 = linear intensity controller")
    bench.add_argument("--saturation", type=float, default=None, help="Detector clips here, volts")
    bench.add_argument("--randomize-calibration", action="store_true",
                       help="Random full scale, dark offset and angle offset per run")
    return p


def make_apparatus(args: argparse.Namespace, seed: int):
    if args.apparatus == "hardware":
        return HardwareApparatus(), {"apparatus": "hardware"}
    fixed = {"noise_volts": args.noise, "extinction": args.extinction,
             "bulb_exponent": args.bulb_exponent, "saturation_volts": args.saturation}
    if args.randomize_calibration:
        config = SimulationConfig.randomized(random.Random(seed), **fixed)
    else:
        config = SimulationConfig(full_scale_volts=args.full_scale, dark_volts=args.dark,
                                  angle_offset_deg=args.angle_offset, **fixed)
    return (SimulatedPolarizerApparatus(config, seed=seed),
            {"apparatus": "simulated", "simulation": asdict(config)})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.runs < 1:
        print("--runs must be 1 or more", file=sys.stderr)
        return 2
    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
    task_prompt = Path(args.task_prompt_file).read_text(encoding="utf-8").strip()
    think = None if args.think is None else args.think == "on"
    client = make_client(args.provider, args.model, url=args.llm_url,
                         api_key=args.llm_api_key or None, temperature=args.temperature,
                         num_ctx=args.num_ctx, think=think, timeout=args.timeout,
                         attempts=args.retries)
    fit_request_path = Path(args.fit_request_file)
    fit_request = (fit_request_path.read_text(encoding="utf-8").strip()
                   if fit_request_path.exists() else DEFAULT_FIT_REQUEST)
    finish_path = Path(args.finish_tool_file)
    finish_description = (finish_path.read_text(encoding="utf-8").strip()
                          if finish_path.exists() else DEFAULT_FINISH_DESCRIPTION)
    config = LabConfig(max_calls=args.max_calls, keep_last=args.keep_last,
                       holdout_points=args.holdout_points, threshold=args.threshold,
                       verbose=not args.quiet, fit_stage=not args.no_fit_stage,
                       fit_request=fit_request, finish_description=finish_description)
    model_dir = Path(args.out_dir).resolve() / safe_model_name(args.model)
    consecutive_errors = 0

    for index in range(args.runs):
        number, run_dir = next_run_dir(model_dir)
        seed = args.seed + number - 1
        print(f"\n[run {index + 1}/{args.runs}] {run_dir} seed={seed}")
        try:
            apparatus, bench_info = make_apparatus(args, seed)
        except NotImplementedError as exc:
            print(f"[error] {exc}")
            run_dir.rmdir()
            return 2
        extra = {"model": args.model, "provider": args.provider, "run": f"{number:03d}",
                 "seed": seed, **bench_info,
                 "settings": {"max_calls": args.max_calls, "keep_last": args.keep_last,
                              "threshold": args.threshold, "temperature": args.temperature,
                              "fit_stage": not args.no_fit_stage},
                 # which wording this run saw, so runs on different prompts
                 # can always be told apart
                 "prompts": {"system": system_prompt, "task": task_prompt,
                             "fit_request": None if args.no_fit_stage else fit_request,
                             "finish_tool": finish_description}}
        try:
            summary = run_session(client, apparatus, system_prompt=system_prompt,
                                  task_prompt=task_prompt, folder=run_dir, config=config,
                                  holdout_seed=seed + 1_000_000, extra=extra)
        except KeyboardInterrupt:
            (run_dir / "summary.json").write_text(
                json.dumps({"status": "interrupted", **extra}, indent=2) + "\n", encoding="utf-8")
            write_runs_summary(model_dir, args.model)
            print(f"\n[interrupted] summary so far: {model_dir / 'runs_summary.json'}")
            return 130
        finally:
            apparatus.close()
        totals = write_runs_summary(model_dir, args.model)
        print(f"[run {number:03d}] {summary['status']}, "
              f"{'found the law' if summary['discovered'] else 'law not found'} | "
              f"{args.model}: {totals['discovered']} of {totals['judged']} runs found it")
        if summary["status"] == "error":
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print("[stopped] three runs in a row ended in an error; check the model endpoint")
                return 1
        else:
            consecutive_errors = 0
    print(f"[runs summary] {model_dir / 'runs_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
