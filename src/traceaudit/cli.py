from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import adapter_profiles
from .config import Config, load_config
from .eval.metrics import evaluate, write_eval, write_eval_md
from .llm import NullClient, client_from_env, load_env_files, semantic_mode
from .normalize import load_dataset, load_trace, verify_dataset
from .normalize.input import apply_overlays, detect_source, looks_like_trace, read_prompt_value
from .pipeline import analyze_trace
from .report import (
    ascii_safe,
    dataset_hash,
    run_console_summary,
    trace_console_lines,
    trace_report,
    write_run_artifacts,
)
from .schemas import CanonicalTrace, TraceResult


def _pkg_root() -> Path:
    return Path(__file__).resolve().parents[2]


def say(*parts: str) -> None:
    """Print ASCII. A PowerShell console is cp1252 and trace quotes are not."""
    print(ascii_safe(" ".join(parts)))


def fail(msg: str, *hints: str) -> int:
    print(ascii_safe(f"error: {msg}"), file=sys.stderr)
    for hint in hints:
        print(ascii_safe(f"  {hint}"), file=sys.stderr)
    return 2


_NO_KEY_NOTICE = [
    "",
    "NOTE: no API key found, so this run uses the mechanical lane only.",
    "  It still proves schema violations, undeclared tools, malformed calls and",
    "  identical repeats, but it cannot read your instructions, so most",
    "  instruction violations and inefficiencies will be missed.",
    "  To run the full audit, pick one:",
    '    1. put  OPENROUTER_API_KEY=sk-...  in a .env file in this directory',
    '    2. $env:OPENROUTER_API_KEY = "sk-..."     (PowerShell)',
    "    3. pass --api-key sk-...",
    "  Pass --deterministic to choose the mechanical lane on purpose and hide this.",
    "",
]


def _format_names() -> list[str]:
    return [p.name for p in adapter_profiles.all_profiles()]


def _add_common(p: argparse.ArgumentParser, positional: bool = False) -> None:
    if positional:
        p.add_argument(
            "trace",
            nargs="?",
            help="Trace file or directory (same as --dataset, just shorter)",
        )
    p.add_argument("--dataset", help="Trace file or directory")
    p.add_argument("--out", default="traceaudit-out", help="Output directory")
    p.add_argument("--config", help="Config JSON path")
    p.add_argument("--labels", help="Directory of gold label JSON files")
    _add_prompt_flags(p)
    p.add_argument(
        "--source",
        default="",
        help="Force an adapter instead of auto-detecting. One of: " + ", ".join(_format_names()),
    )
    p.add_argument("--traces", default="", help="Only these trace ids, comma-separated")
    p.add_argument("--labelled", action="store_true", help="Only traces that have labels")
    _add_model_flags(p)


def _add_prompt_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--task",
        default="",
        help="The task the agent was given, if the trace does not carry it. "
        "Literal text, or @path/to/task.txt to read a file",
    )
    p.add_argument(
        "--instructions",
        default="",
        help="The rules or system prompt the agent was under, if the trace does not "
        "carry them. Literal text, or @path/to/policy.md to read a file",
    )


def _add_model_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mock", action="store_true", help="Stub the models; every stage returns empty JSON")
    p.add_argument("--deterministic", action="store_true", help="Mechanical lane only, no model calls")
    p.add_argument("--api-key", default="", help="API key; defaults to OPENROUTER_API_KEY or OPENAI_API_KEY")
    p.add_argument("--model", default="", help="Use one model for every stage")
    p.add_argument(
        "--show-report",
        action="store_true",
        help="Print each trace's full report to the terminal as well as writing it",
    )


def _client(args, cfg: Config):
    if getattr(args, "api_key", ""):
        os.environ["OPENROUTER_API_KEY"] = args.api_key
    if getattr(args, "deterministic", False):
        return NullClient()
    return client_from_env(cfg, mock=getattr(args, "mock", False))


def _cfg(args) -> Config:
    load_env_files()
    path = args.config
    if not path:
        bundled = _pkg_root() / "configs" / "default.json"
        path = str(bundled) if bundled.is_file() else None
    cfg = load_config(path)
    if getattr(args, "model", ""):
        for stage in cfg.llm_stages():
            stage.models = [args.model]
    return cfg


def _select(dataset: dict, args, labels: set[str] | None) -> dict:
    out = dataset
    if getattr(args, "labelled", False) and labels is not None:
        out = {k: v for k, v in out.items() if k in labels}
    if getattr(args, "traces", ""):
        want = {t.strip() for t in args.traces.split(",") if t.strip()}
        out = {k: v for k, v in out.items() if k in want}
    return out


def _dataset_path(args) -> str:
    """--dataset or the bare positional; they mean the same thing."""
    return getattr(args, "dataset", "") or getattr(args, "trace", "") or ""


def _prompt_value(value: str) -> str:
    """Read --task / --instructions, tolerating a byte-order mark.

    PowerShell's Out-File and Notepad both write UTF-8 with a BOM, so a
    hand-made task.txt otherwise starts with a stray character that ends up
    quoted back at the user inside the report.
    """
    return read_prompt_value(value).lstrip("\ufeff") if value else ""


def _load_dataset_or_explain(path: str, cfg: Config, source: str):
    """Return (dataset, None) or (None, exit_code) with an actionable message."""
    target = Path(path)
    if not target.exists():
        return None, fail(
            f"no such file or directory: {target}",
            "Pass the path to one trace file, or to a folder of them.",
            "Example: traceaudit run examples/openai_chat.json",
        )
    if target.is_file() and target.stat().st_size == 0:
        return None, fail(
            f"{target} is empty (0 bytes)",
            "A trace file needs to hold the agent's messages or spans.",
        )
    try:
        dataset = load_dataset(path, max_step_chars=cfg.max_step_chars, source_override=source)
    except FileNotFoundError:
        if target.is_file():
            return None, fail(
                f"{target} parses as JSON but does not look like an agent trace",
                "traceaudit needs the agent's messages, tool calls or spans -- not a",
                "config or a result blob. If the format is one it knows, name it:",
                "  traceaudit run <path> --source openai-chat",
                "See the full list with:  traceaudit formats",
            )
        return None, fail(
            f"could not recognise any trace in {target}",
            "traceaudit reads JSON or JSONL. Check the file parses, then either",
            "name the format explicitly with --source, or let the generic reader try:",
            "  traceaudit run <path> --source canonical",
            "Known formats: " + ", ".join(_format_names()),
        )
    except (OSError, UnicodeDecodeError) as exc:
        return None, fail(
            f"could not read {target}: {exc}",
            "The file must be UTF-8 encoded JSON or JSONL.",
        )
    except json.JSONDecodeError as exc:
        return None, fail(
            f"{target} is not valid JSON: {exc}",
            "Fix the JSON, or point at the raw harness export you started from.",
        )
    if not dataset:
        return None, fail(f"no traces selected from {target}")
    if target.is_file() and not source and not detect_source(target) and not looks_like_trace(target):
        return None, fail(
            f"{target} parses as JSON but does not look like an agent trace",
            "traceaudit needs the agent's messages, tool calls or spans -- not a",
            "config or a result blob. If the format is one it knows, name it:",
            "  traceaudit run <path> --source openai-chat",
            "See the full list with:  traceaudit formats",
        )
    return dataset, None


def _analyze_one(tid: str, meta: dict, cfg: Config, client, task: str, instructions: str):
    """Run one trace. Returns (result, trace|None)."""
    try:
        trace: CanonicalTrace = load_trace(tid, meta["source"], meta["files"], cfg.max_step_chars)
    except Exception as exc:
        return TraceResult(trace_id=tid, status="error", error=f"could not load trace: {exc}"), None
    apply_overlays(trace, task=task, instructions=instructions)
    if not trace.steps:
        gaps = "; ".join(trace.capture_gaps) or "the adapter recovered no steps"
        return (
            TraceResult(
                trace_id=tid,
                status="error",
                error=(
                    f"no steps could be read from this trace ({gaps}). "
                    f"Detected format was '{meta.get('source') or 'unknown'}'; "
                    "override it with --source if that is wrong."
                ),
            ),
            trace,
        )
    try:
        result = analyze_trace(trace, cfg, client, cfg_hash=cfg.fingerprint())
    except Exception as exc:
        result = TraceResult(trace_id=tid, status="error", error=f"{type(exc).__name__}: {exc}")
    return result, trace


def cmd_run(args) -> int:
    path = _dataset_path(args)
    if not path:
        return fail(
            "no trace given",
            "Pass a trace file or folder, either way round:",
            "  traceaudit run examples/openai_chat.json",
            "  traceaudit run --dataset data/dev --out results/dev",
        )
    cfg = _cfg(args)
    client = _client(args, cfg)
    dataset, err = _load_dataset_or_explain(path, cfg, args.source)
    if err is not None:
        return err

    label_ids: set[str] = set()
    if args.labels:
        from .eval.matching import load_labels

        label_ids = set(load_labels(args.labels))
    dataset = _select(dataset, args, label_ids)
    if not dataset:
        return fail(
            "every trace was filtered out",
            "Check --traces and --labelled against the ids actually in the dataset.",
        )

    task = _prompt_value(args.task)
    instructions = _prompt_value(args.instructions)
    mode = semantic_mode(client)

    say(f"traceaudit: auditing {len(dataset)} trace(s) from {path}")
    formats = sorted({m.get("source") or "unknown" for m in dataset.values()})
    detected = "forced with --source" if args.source else "auto-detected"
    say(f"  format: {', '.join(formats)} ({detected}; override with --source)")
    say(
        f"  mode={mode}  target=${getattr(cfg, 'target_mean_usd', 1.5):.2f}/trace  "
        f"cap=${cfg.cost_cap_usd:.2f}/trace  session cap=${cfg.max_spend_usd:.2f}"
    )
    supplied = [n for n, v in (("task", task), ("instructions", instructions)) if v]
    if supplied:
        say(f"  using the {' and '.join(supplied)} you supplied, overriding anything in the trace")
    else:
        say("  reading the task and instructions from the trace itself "
            "(pass --task / --instructions to supply your own)")
    if mode == "deterministic" and not args.deterministic:
        for line in _NO_KEY_NOTICE:
            say(line)
    say("")

    results: list[TraceResult] = []
    traces: dict[str, CanonicalTrace] = {}
    session_spend = 0.0
    for n, (tid, meta) in enumerate(dataset.items(), 1):
        if session_spend >= cfg.max_spend_usd:
            say(f"  stopping: session spend ${session_spend:.2f} hit the ${cfg.max_spend_usd:.2f} cap")
            break
        result, trace = _analyze_one(tid, meta, cfg, client, task, instructions)
        session_spend += result.cost_usd
        if trace is not None:
            traces[tid] = trace
        results.append(result)
        for line in trace_console_lines(result, n, len(dataset), session_spend):
            print(line)
        if args.show_report:
            print(ascii_safe(trace_report(result, trace)))

    out = write_run_artifacts(
        args.out, results, cfg=cfg, dataset_hash=dataset_hash(path), traces=traces
    )

    overall = None
    if args.labels:
        report = evaluate(out / "findings.jsonl", args.labels, semantic_mode=mode)
        write_eval(out / "eval.json", report)
        write_eval_md(out / "eval.md", report)
        overall = report["overall"]["strict"]
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    say("")
    print(run_console_summary(summary, out, overall, wrote_eval=bool(args.labels)))
    return 0


def cmd_explain(args) -> int:
    """Audit one trace and print the whole report. `run` for a single file."""
    if not args.trace:
        return fail(
            "no trace given",
            "Usage: traceaudit explain <trace.json> [--task @task.txt] [--instructions @rules.md]",
        )
    args.dataset = args.trace
    args.show_report = True
    return cmd_run(args)


def cmd_eval(args) -> int:
    if not args.results or not args.labels:
        return fail("--results and --labels are both required")
    if not Path(args.results).exists():
        return fail(f"no such results file: {args.results}", "Point at findings.jsonl from a run.")
    report = evaluate(args.results, args.labels, semantic_mode=args.semantic_mode or "unknown")
    dest = Path(args.out or Path(args.results).parent)
    dest.mkdir(parents=True, exist_ok=True)
    write_eval(dest / "eval.json", report)
    write_eval_md(dest / "eval.md", report)
    o = report["overall"]["strict"]
    clean = report["false_positives_on_clean_traces"]
    say(f"scored {report['n_evaluated']} of {report['n_labelled']} labelled traces "
        f"({report.get('split', 'development')})")
    say(f"  strict   P {o['precision']:.3f}  R {o['recall']:.3f}  F1 {o['f1']:.3f}  "
        f"TP {o['tp']} FP {o['fp']} FN {o['fn']}")
    say(f"  clean-trace false positives: {clean['n_fp']} of {clean['n_clean_traces']} clean traces")
    say(f"  abstentions (never an FP): "
        f"{report['insufficient_evidence_tracked_separately']['count']}")
    say(f"  cost mean/p95 ${report['cost_usd']['mean']} / ${report['cost_usd']['p95']}   "
        f"latency mean/p95 {report['latency_s']['mean']}s / {report['latency_s']['p95']}s")
    say(f"wrote {dest / 'eval.md'} and {dest / 'eval.json'}")
    return 0


def cmd_demo(args) -> int:
    root = _pkg_root()
    args.dataset = args.dataset or args.trace or str(root / "data" / "dev")
    args.labels = args.labels or str(root / "data" / "labels" / "dev")
    args.out = args.out or "traceaudit-out/demo"
    return cmd_run(args)


def cmd_normalize(args) -> int:
    path = _dataset_path(args)
    if not path:
        return fail("no trace given", "Usage: traceaudit normalize <trace.json>")
    cfg = _cfg(args)
    dataset, err = _load_dataset_or_explain(path, cfg, args.source)
    if err is not None:
        return err
    dataset = _select(dataset, args, None)
    for tid, meta in dataset.items():
        trace = load_trace(tid, meta["source"], meta["files"], cfg.max_step_chars)
        print(json.dumps(trace.model_dump(), indent=2, default=str)[: args.limit])
    return 0


def cmd_formats(args) -> int:
    say("Trace formats traceaudit reads. The format is detected automatically;")
    say("pass --source <name> to force one.")
    say("")
    for prof in adapter_profiles.all_profiles():
        say(f"  {prof.name:<24}{prof.description}")
    say("")
    say("  canonical               traceaudit's own normalized JSON (also the generic fallback)")
    return 0


def cmd_doctor(args) -> int:
    root = _pkg_root()
    say(f"package root: {root}")
    say(f"python: {sys.version.split()[0]}")
    cfg = _cfg(args)
    client = _client(args, cfg)
    mode = semantic_mode(client)
    say(f"mode: {mode}")
    if mode == "deterministic":
        say("  no API key found: mechanical lane only.")
        say('  set OPENROUTER_API_KEY in .env, or pass --api-key, for the full audit.')
    elif mode == "live":
        say("  API key found. A full audit will make model calls and cost money:")
        say(f"  target ${getattr(cfg, 'target_mean_usd', 1.5):.2f}/trace, "
            f"hard cap ${cfg.cost_cap_usd:.2f}/trace, session cap ${cfg.max_spend_usd:.2f}.")
    dev = root / "data" / "dev"
    if (dev / "index.json").is_file():
        say(f"dev dataset: {verify_dataset(dev)}")
    sample = root / "examples" / "openai_chat.json"
    if sample.is_file():
        say("")
        say("try one trace:")
        say(f"  traceaudit explain {sample}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="traceaudit",
        description="Find errors and inefficiencies in AI agent traces",
        epilog=(
            "audit one trace:  traceaudit explain mytrace.json\n"
            "with your own task and rules:\n"
            "  traceaudit explain mytrace.json --task @task.txt --instructions @rules.md\n"
            "see docs/usage.md for the full guide."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Audit traces, and score them if --labels is given")
    _add_common(run, positional=True)
    run.set_defaults(func=cmd_run)

    explain = sub.add_parser(
        "explain",
        help="Audit ONE trace file and print the whole report to the terminal",
    )
    explain.add_argument("trace", nargs="?", help="Path to your trace file")
    explain.add_argument("--out", default="traceaudit-out/explain", help="Output directory")
    explain.add_argument("--config", help="Config JSON path")
    explain.add_argument(
        "--source",
        default="",
        help="Force an adapter instead of auto-detecting. One of: " + ", ".join(_format_names()),
    )
    _add_prompt_flags(explain)
    _add_model_flags(explain)
    explain.set_defaults(func=cmd_explain, dataset="", labels="", labelled=False, traces="")

    demo = sub.add_parser("demo", help="Audit the shipped 17-trace development set")
    _add_common(demo, positional=True)
    demo.set_defaults(func=cmd_demo)

    ev = sub.add_parser("eval", help="Score an existing run against gold labels")
    ev.add_argument("--results", required=True, help="findings.jsonl from a run")
    ev.add_argument("--labels", required=True, help="Directory of gold label JSON files")
    ev.add_argument("--out", default="", help="Where to write eval.md and eval.json")
    ev.add_argument("--semantic-mode", default="", help="Mode to record in the report")
    ev.set_defaults(func=cmd_eval)

    norm = sub.add_parser("normalize", help="Print the canonical JSON for a trace")
    _add_common(norm, positional=True)
    norm.add_argument("--limit", type=int, default=8000, help="Characters to print per trace")
    norm.set_defaults(func=cmd_normalize)

    fmt = sub.add_parser("formats", help="List the trace formats traceaudit can read")
    fmt.set_defaults(func=cmd_formats)

    doc = sub.add_parser("doctor", help="Check the install, the dataset, and the API key")
    _add_common(doc, positional=True)
    doc.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
