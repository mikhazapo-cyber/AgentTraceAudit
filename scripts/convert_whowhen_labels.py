#!/usr/bin/env python3
"""Convert Who&When Pro into a traceaudit dataset + gold labels.

    python scripts/fetch_public_corpus.py --corpus whowhen
    python scripts/convert_whowhen_labels.py --src data/raw/whowhen --limit 500
    traceaudit run --dataset data/raw/whowhen-dataset --labels data/labels/whowhen \\
                  --labelled --config configs/default.json --out results/whowhen

Why bother when TRAIL exists: TRAIL has no leaf for `ignored_feedback` and only
reaches `unsupported_success` through a family-ambiguous type. Who&When Pro's
`inadequate verification`, `premature termination` and `over-reliance on other
agents` modes cover both, and it is CC-BY-4.0 rather than gated.

**Recall and localisation only.** Every trace has exactly one injected decisive
error and nothing else is annotated, so a correct finding elsewhere in the trace
scores as a false positive. The emitted labels carry
`gold_completeness: decisive_error_only`, which makes the evaluator print that
warning above any P/R/F1. Take precision from TRAIL or the dev set.

Trajectories are written as OpenAI-chat-shaped JSON (the `openai-chat` adapter
reads them), since Who&When Pro ships framework-specific turn lists rather than
spans. Per-framework quirks across its 15 frameworks are reported, not guessed:
run with `--inspect` first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

MAPPING_PATH = REPO_ROOT / "data" / "labels" / "mappings" / "whowhen_to_traceveri.json"

TRAJECTORY_KEYS = ("trajectory", "history", "turns", "messages", "rollout")
GROUND_TRUTH_KEYS = ("ground_truth", "label", "gold", "answer")
TRACE_ID_KEYS = ("id", "trace_id", "uuid")
TASK_KEYS = ("task", "query", "question", "problem")
MODE_KEYS = ("mode", "error_mode", "failure_mode", "type")
STEP_KEYS = ("step", "decisive_step", "error_step", "step_index")
AGENT_KEYS = ("agent", "responsible_agent", "who")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _first(row, keys):
    for key in keys:
        if isinstance(row, dict) and row.get(key) not in (None, "", [], {}):
            return row[key]
    return None


def _as_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def load_mapping(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    buckets = spec["buckets"]
    return {
        "version": str(spec.get("mapping_version") or "?"),
        "gold_completeness": str(spec.get("gold_completeness") or "complete"),
        "direct": {_norm(k): v["family"] for k, v in buckets["direct"]["entries"].items()},
        "localization_only": {_norm(k): k for k in buckets["localization_only"]["entries"]},
        "excluded": {_norm(k): v.get("reason", "out_of_scope") for k, v in buckets["excluded"]["entries"].items()},
    }


def iter_rows(src: Path):
    parquet = sorted(src.rglob("*.parquet"))
    if parquet:
        try:
            from datasets import load_dataset
        except ImportError:
            print("parquet files found but `datasets` is not installed:", file=sys.stderr)
            print("  python -m pip install datasets", file=sys.stderr)
            raise SystemExit(2) from None
        ds = load_dataset("parquet", data_files=[str(p) for p in parquet])
        for split in ds:
            yield from ds[split]
        return
    for path in sorted(src.rglob("*.json")) + sorted(src.rglob("*.jsonl")):
        if path.name in ("index.json", "taxonomy.yaml"):
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    yield json.loads(line)
        else:
            doc = json.loads(text)
            if isinstance(doc, list):
                yield from (d for d in doc if isinstance(d, dict))
            elif isinstance(doc, dict):
                yield doc


def _turn_to_message(turn) -> dict | None:
    """One Who&When Pro turn -> one OpenAI-chat message."""
    if isinstance(turn, str):
        return {"role": "assistant", "content": turn}
    if not isinstance(turn, dict):
        return None
    role = str(turn.get("role") or turn.get("name") or turn.get("agent") or "assistant")
    role = role if role in ("user", "assistant", "system", "tool") else "assistant"
    content = turn.get("content")
    if content is None:
        content = turn.get("message") or turn.get("text") or turn.get("output") or ""
    message: dict = {"role": role, "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)}
    calls = turn.get("tool_calls") or turn.get("tool_call") or turn.get("function_call")
    if calls:
        calls = calls if isinstance(calls, list) else [calls]
        normalized = []
        for i, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            fn = call.get("function") if isinstance(call.get("function"), dict) else call
            args = fn.get("arguments", fn.get("args", {}))
            normalized.append(
                {
                    "id": str(call.get("id") or f"wwp-{i}"),
                    "type": "function",
                    "function": {"name": str(fn.get("name") or ""), "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False, default=str)},
                }
            )
        if normalized:
            message["tool_calls"] = normalized
    if turn.get("agent") or turn.get("name"):
        message["name"] = str(turn.get("agent") or turn.get("name"))
    return message


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dataset-out", default=str(REPO_ROOT / "data" / "raw" / "whowhen-dataset"))
    parser.add_argument("--labels-out", default=str(REPO_ROOT / "data" / "labels" / "whowhen"))
    parser.add_argument("--mapping", default=str(MAPPING_PATH))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--modality", default="text", help="keep only this modality ('' for all)")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        print(f"source dir not found: {src}", file=sys.stderr)
        return 2
    mapping = load_mapping(Path(args.mapping))
    rows = iter_rows(src)

    if args.inspect:
        for i, row in enumerate(rows):
            if i >= 3:
                break
            print(f"--- row {i} ---")
            print(f"  row keys:          {sorted(row) if isinstance(row, dict) else type(row)}")
            print(f"  trajectory key:    {next((k for k in TRAJECTORY_KEYS if k in row), None)!r}")
            gt = _as_json(_first(row, GROUND_TRUTH_KEYS))
            print(f"  ground truth:      {gt if not isinstance(gt, dict) else sorted(gt)}")
            traj = _as_json(_first(row, TRAJECTORY_KEYS))
            if isinstance(traj, list) and traj:
                print(f"  first turn keys:   {sorted(traj[0]) if isinstance(traj[0], dict) else type(traj[0])}")
        return 0

    dataset_out, labels_out = (Path(args.dataset_out), Path(args.labels_out))
    (dataset_out / "traces").mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    totals = Counter()
    unmapped: Counter = Counter()
    frameworks: Counter = Counter()
    skipped = Counter()

    for n, row in enumerate(rows):
        if args.limit and len(entries) >= args.limit:
            break
        if args.modality and str(row.get("modality") or "text") != args.modality:
            skipped["modality"] += 1
            continue
        trajectory = _as_json(_first(row, TRAJECTORY_KEYS))
        if not isinstance(trajectory, list) or not trajectory:
            skipped["no_trajectory"] += 1
            continue
        messages = [m for m in (_turn_to_message(t) for t in trajectory) if m]
        if not messages:
            skipped["no_messages"] += 1
            continue

        task = _as_json(_first(row, TASK_KEYS)) or {}
        query = task.get("query") if isinstance(task, dict) else task
        if query and not any(m["role"] == "user" for m in messages):
            messages.insert(0, {"role": "user", "content": str(query)})

        trace_id = str(_first(row, TRACE_ID_KEYS) or f"wwp-{n:05d}")
        framework = str(row.get("framework") or "unknown")
        frameworks[framework] += 1
        payload = {"model": f"whowhen-pro/{framework}", "messages": messages}
        rel = f"traces/{trace_id}.json"
        blob = json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")
        (dataset_out / rel).write_bytes(blob)
        entries.append(
            {
                "trace_id": trace_id,
                "source": "openai-chat",
                "format": "native_json",
                "files": [{"path": rel, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}],
                "source_repository": "Leoxx/whowhen_pro",
                "source_framework": framework,
                "source_benchmark_task": str(row.get("benchmark") or ""),
            }
        )

        gt = _as_json(_first(row, GROUND_TRUTH_KEYS)) or {}
        gt = gt if isinstance(gt, dict) else {}
        mode = str(_first(gt, MODE_KEYS) or "")
        raw_step = _first(gt, STEP_KEYS)
        # Who&When Pro indexes turns; canonical step indices are assigned after
        # normalization, so a turn index is a best-effort anchor. Relaxed
        # matching (centre distance <= 3) absorbs the usual off-by-a-few.
        step = int(raw_step) if isinstance(raw_step, (int, float)) or (isinstance(raw_step, str) and raw_step.isdigit()) else None
        steps = [step] if step is not None else []
        agent = str(_first(gt, AGENT_KEYS) or "")
        description = f"Injected decisive error ({mode}) at turn {step}" + (f", agent '{agent}'" if agent else "")

        key = _norm(mode)
        findings, excluded, loc_only = [], [], []
        totals["annotations"] += 1
        if key in mapping["direct"]:
            totals["scored"] += 1
            findings.append({"family": mapping["direct"][key], "steps": steps, "description": description, "severity": "major", "notes": f"Who&When Pro mode: {mode}"})
        elif key in mapping["localization_only"]:
            totals["localization_only"] += 1
            loc_only.append({"source_type": mode, "reason": "family_ambiguous", "steps": steps, "description": description})
        elif key in mapping["excluded"]:
            totals["excluded"] += 1
            excluded.append({"source_type": mode, "reason": mapping["excluded"][key], "steps": steps, "description": description})
        else:
            unmapped[mode] += 1
            excluded.append({"source_type": mode, "reason": "unmapped_type", "steps": steps, "description": description})

        label = {
            "trace_id": trace_id,
            "clean": not findings,
            "findings": findings,
            "excluded_findings": excluded,
            "localization_only": loc_only,
            "source_benchmark": "Who&When Pro",
            "mapping_version": mapping["version"],
            "gold_completeness": mapping["gold_completeness"],
            "annotator": "Who&When Pro warm-start injection; labels exact by construction (arXiv:2607.09996)",
            "review": "synthetic-exact",
            "notes": f"framework={framework}. One injected decisive error per trace: recall/localisation only, precision not interpretable.",
        }
        (labels_out / f"{trace_id}.json").write_text(json.dumps(label, indent=2, ensure_ascii=False), encoding="utf-8")

    if not entries:
        print("no traces converted — run with --inspect to see the schema that was found.", file=sys.stderr)
        return 1

    index = {
        "format_version": 2,
        "dataset_version": "whowhen-pro-v1",
        "source_dataset_version": "Leoxx/whowhen_pro",
        "annotation_status": "labelled",
        "trace_count": len(entries),
        "source_counts": dict(Counter(e["source"] for e in entries)),
        "licence": "CC-BY-4.0 — cite Liu et al. 2026, arXiv:2607.09996",
        "note": "Converted from Who&When Pro. One injected decisive error per trace: recall and localisation only.",
        "traces": entries,
    }
    (dataset_out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(f"dataset: {dataset_out}  ({len(entries)} traces)")
    print(f"labels:  {labels_out}")
    print(f"frameworks: {dict(frameworks.most_common())}")
    print(f"gold annotations: {totals['annotations']}")
    print(f"  scored (direct family):     {totals['scored']}")
    print(f"  localization only:          {totals['localization_only']}")
    print(f"  excluded (out of taxonomy): {totals['excluded']}")
    print("reminder: precision is not interpretable on this corpus (one injected error per trace).")
    if unmapped:
        print("warning: modes absent from the mapping file (treated as excluded/unmapped_type):")
        for mode, count in unmapped.most_common():
            print(f"  {mode}: {count}")
        print(f"  add them to {args.mapping} and re-run.")
    if skipped:
        print(f"skipped rows: {dict(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
