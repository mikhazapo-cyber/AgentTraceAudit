from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

from .config import Config, load_config
from .eval import eval_console, load_labels, score_results, write_eval
from .llm import client_from_env, load_env_files
from .normalize import apply_overlays, load_dataset, load_trace, read_prompt_value
from .pipeline import analyze_trace
from .report import (
    ascii_safe,
    run_console_summary,
    trace_console_lines,
    trace_report,
    write_run_artifacts,
)
from .schemas import CanonicalTrace, TraceResult


_USAGE = """\
traceaudit
traceaudit mytrace.json
traceaudit mytrace.json --task @task.txt --instructions @policy.md --api-key sk-or-...
traceaudit data/dev --labels data/labels/dev --yes
"""

_NO_KEY = """\
error: no API key

This audit uses two high-power models, then a judge.
Set a key in one of these ways:

  1. .env file next to the repo:   OPENROUTER_API_KEY=sk-or-...
  2. macOS / Linux:                export OPENROUTER_API_KEY=sk-or-...
  3. Windows PowerShell:           $env:OPENROUTER_API_KEY = "sk-or-..."
  4. flag:                         --api-key sk-or-...
"""


def _flush_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        try:
            stream.flush()
        except Exception:
            pass


def say(*parts: str) -> None:
    print(ascii_safe(" ".join(parts)), flush=True)


def fail(msg: str, *hints: str) -> int:
    print(ascii_safe(f"error: {msg}"), file=sys.stderr, flush=True)
    for hint in hints:
        print(ascii_safe(f"  {hint}"), file=sys.stderr, flush=True)
    return 2


def _prompt_value(value: str) -> str:
    return read_prompt_value(value).lstrip("\ufeff") if value else ""


def _client(args, cfg: Config):
    if getattr(args, "api_key", ""):
        os.environ["OPENROUTER_API_KEY"] = args.api_key
    return client_from_env(cfg)


def _load_dataset_or_explain(path: str):
    target = Path(path)
    if not target.exists():
        return None, fail(
            f"no such file or directory: {target}",
            "Pass one trace file, or a folder of traces.",
            "Example: traceaudit examples/contradict.json",
        )
    if target.is_file() and target.stat().st_size == 0:
        return None, fail(f"{target} is empty")
    try:
        dataset = load_dataset(path)
    except FileNotFoundError:
        return None, fail(
            f"{target} does not look like an agent trace",
            "Need messages, tool calls, or canonical steps -- JSON or JSONL.",
        )
    except json.JSONDecodeError as exc:
        return None, fail(f"{target} is not valid JSON: {exc}")
    except (OSError, UnicodeDecodeError) as exc:
        return None, fail(f"could not read {target}: {exc}")
    return dataset, None


def load_finished(out_dir: str | Path) -> dict[str, TraceResult]:
    path = Path(out_dir) / "findings.jsonl"
    done: dict[str, TraceResult] = {}
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = TraceResult.model_validate_json(line)
        except Exception:
            continue
        if rec.status == "ok":
            done[rec.trace_id] = rec
    return done


def _confirm_spend(n: int, cfg: Config, yes: bool) -> bool:
    if n <= 1 or yes:
        return True
    typical = n * cfg.target_mean_usd
    say(
        f"About to audit {n} traces. Typical spend is about ${typical:.2f} "
        f"(${cfg.target_mean_usd:.2f} mean, ${cfg.cost_cap_usd:.2f} cap each, "
        f"session cap ${cfg.max_spend_usd:.2f})."
    )
    if not sys.stdin.isatty():
        say("Pass --yes to run a multi-trace audit without a prompt.")
        return False
    try:
        answer = input("Continue? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _analyze_one(
    tid: str, meta: dict, cfg: Config, client, task: str, instructions: str
):
    try:
        trace: CanonicalTrace = load_trace(
            tid, meta.get("source") or "", meta["files"], cfg.max_step_chars
        )
    except Exception as exc:
        return TraceResult(
            trace_id=tid, status="error", error=f"could not load trace: {exc}"
        ), None
    apply_overlays(trace, task=task, instructions=instructions)
    if not trace.steps:
        gaps = "; ".join(trace.capture_gaps) or "no steps recovered"
        return (
            TraceResult(
                trace_id=tid, status="error", error=f"no steps could be read ({gaps})"
            ),
            trace,
        )

    def on_stage(stage: str) -> None:
        if stage == "auditors":
            say(f"           {len(trace.steps)} steps  auditors A + B (parallel)...")
        elif stage == "judge":
            say("           judge...")

    try:
        result = analyze_trace(
            trace, cfg, client, cfg_hash=cfg.fingerprint(), on_stage=on_stage
        )
    except Exception as exc:
        result = TraceResult(
            trace_id=tid, status="error", error=f"{type(exc).__name__}: {exc}"
        )
    return result, trace


def cmd_run(args) -> int:
    path = getattr(args, "trace", "") or ""
    if not path:
        print(_USAGE)
        return 2
    load_env_files()
    cfg = load_config(args.config)
    client = _client(args, cfg)
    if not getattr(client, "available", False):
        print(ascii_safe(_NO_KEY), file=sys.stderr, flush=True)
        return 2

    dataset, err = _load_dataset_or_explain(path)
    if err is not None:
        return err

    labels = None
    if getattr(args, "labels", ""):
        try:
            labels = load_labels(args.labels)
        except (OSError, ValueError) as exc:
            return fail(f"could not read labels at {args.labels}: {exc}")

    if not _confirm_spend(len(dataset), cfg, bool(getattr(args, "yes", False))):
        return 1

    task = _prompt_value(args.task)
    instructions = _prompt_value(args.instructions)

    say(f"traceaudit: auditing {len(dataset)} trace(s) from {path}")
    say(
        f"  models: A={cfg.agent_a.models[0]}  B={cfg.agent_b.models[0]}  "
        f"judge={cfg.judge.models[0]}"
    )
    say(
        f"  target=${cfg.target_mean_usd:.2f}/trace  cap=${cfg.cost_cap_usd:.2f}/trace  "
        f"(pack holds the mean; no output token cap)"
    )
    if labels:
        say(
            f"  labels: {args.labels}  ({len(labels)} file(s); development, not held-out)"
        )
    if task or instructions:
        used = [n for n, v in (("task", task), ("instructions", instructions)) if v]
        say(f"  using the {' and '.join(used)} you supplied")
    else:
        say(
            "  reading task and instructions from the trace (pass --task / --instructions to override)"
        )
    say("")

    results: list[TraceResult] = []
    traces: dict[str, CanonicalTrace] = {}
    session_spend = 0.0
    print_report = len(dataset) == 1
    finished = {} if getattr(args, "fresh", False) else load_finished(args.out)
    if finished:
        say(
            f"  resume: {len(finished)} finished trace(s) in {args.out} (pass --fresh to redo)"
        )
    for n, (tid, meta) in enumerate(dataset.items(), 1):
        if session_spend >= cfg.max_spend_usd:
            say(
                f"  stopping: session spend ${session_spend:.2f} hit ${cfg.max_spend_usd:.2f}"
            )
            break
        if tid in finished:
            result = finished[tid]
            session_spend += result.cost_usd
            results.append(result)
            say(f"  [{n}/{len(dataset)}] {tid}  resume")
            for line in trace_console_lines(result, n, len(dataset), session_spend):
                print(line, flush=True)
            continue
        say(f"  [{n}/{len(dataset)}] {tid}  loading...")
        result, trace = _analyze_one(tid, meta, cfg, client, task, instructions)
        session_spend += result.cost_usd
        if trace is not None:
            traces[tid] = trace
        results.append(result)
        write_run_artifacts(args.out, results, traces=traces)
        for line in trace_console_lines(result, n, len(dataset), session_spend):
            print(line, flush=True)
        if print_report:
            print(ascii_safe(trace_report(result, trace)), flush=True)
        err = (result.error or "").lower() + " " + " ".join(result.degraded).lower()
        if result.status in {"error", "model_error"} and (
            "payment required" in err or " 402 " in err or "402 payment" in err
        ):
            say("stopping: API returned payment required. Add credits and re-run.")
            break

    out = write_run_artifacts(args.out, results, traces=traces)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    say("")
    print(run_console_summary(summary, out), flush=True)
    if labels:
        report = score_results(results, labels)
        write_eval(out, report)
        say("")
        print(ascii_safe(eval_console(report)), flush=True)
        say(f"  wrote {out / 'eval.md'}")
    return 0


def _ask(label: str) -> str:
    try:
        return input(label).strip()
    except EOFError:
        return ""


def cmd_interactive() -> int:
    say("traceaudit")
    say(
        "Two high-power models read the trace; a judge keeps confirmed findings and assigns families."
    )
    say("")
    try:
        key = getpass.getpass("OpenRouter API key (blank uses env): ").strip()
    except Exception:
        key = _ask("OpenRouter API key (blank uses env): ")
    path = _ask("Trace file or folder: ")
    instructions = _ask("Instructions (text, @file, or Enter to use the trace): ")
    task = _ask("Task (text, @file, or Enter to use the trace): ")
    if not path:
        return fail("no trace path")
    argv = [path]
    if key:
        argv += ["--api-key", key]
    if instructions:
        argv += ["--instructions", instructions]
    if task:
        argv += ["--task", task]
    argv.append("--yes")
    return cmd_run(build_parser().parse_args(argv))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="traceaudit",
        description=(
            "Audit AI agent traces. Two high-power models independently derive "
            "checks and locate problems; a judge keeps proven findings and assigns families."
        ),
        usage="traceaudit [TRACE] [options]",
        epilog=_USAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "trace",
        nargs="?",
        default="",
        help="Trace file, folder, or dataset with index.json",
    )
    p.add_argument("--task", default="", help="Task the agent was given, or @path")
    p.add_argument("--instructions", default="", help="Rules / system prompt, or @path")
    p.add_argument(
        "--api-key", default="", help="OpenRouter (or OpenAI-compatible) API key"
    )
    p.add_argument("--out", default="traceaudit-out", help="Output directory")
    p.add_argument("--config", help="Optional config JSON")
    p.add_argument(
        "--labels",
        default="",
        help="Folder of gold JSON (same family + overlapping steps). Development scores are not held-out.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the multi-trace spend confirmation",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore finished traces already in --out and audit them again",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    _flush_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        if sys.stdin.isatty():
            return cmd_interactive()
        print(_USAGE, flush=True)
        return 2
    args = build_parser().parse_args(argv)
    if not args.trace:
        print(_USAGE, flush=True)
        return 2
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
