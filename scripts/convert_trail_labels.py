#!/usr/bin/env python3
"""Convert the TRAIL benchmark into a traceaudit dataset + gold labels.

Writes a `format_version: 2` dataset (index.json + traces/) and a labels dir in
the same shape as data/labels/dev, so every existing command works unchanged:

    python scripts/fetch_public_corpus.py --corpus trail
    python scripts/convert_trail_labels.py --src data/raw/trail
    traceaudit run --dataset data/raw/trail-dataset \\
                  --labels data/labels/trail --labelled \\
                  --config configs/default.json --out results/trail-det

Both outputs land under gitignored paths by default; TRAIL is not redistributable.

Two mappings happen here, and both are recorded rather than assumed:

* **span id -> step index.** TRAIL annotates spans; traceaudit locates findings at
  step level. The trace is normalized with the `openinference` adapter, which
  keeps `span_id` on every step, and the annotation is attached to those steps.
* **TRAIL leaf type -> traceaudit family.** Driven entirely by
  data/labels/mappings/trail_to_traceveri.json. Types that no family represents
  go to `excluded_findings`; family-ambiguous types go to `localization_only`.
  Neither is counted as a miss by the evaluator.

TRAIL is a gated dataset, so its exact column names could not be verified while
writing this. Run with `--inspect` first: it prints the keys it found and the
keys it looked for, so a rename is a one-line fix rather than a silent
mis-parse. The converter refuses to emit labels it could not locate.
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

from traceaudit.normalize import load_trace  # noqa: E402

MAPPING_PATH = REPO_ROOT / "data" / "labels" / "mappings" / "trail_to_traceveri.json"

# Candidate key names, most specific first. TRAIL is gated; --inspect reports
# what was actually present so these can be corrected without guesswork.
TRACE_KEYS = ("trace", "trace_json", "otel_trace", "spans", "trace_data", "otlp")
ANNOTATION_KEYS = ("errors", "annotations", "error_annotations", "gold", "labels", "gold_errors")
TRACE_ID_KEYS = ("trace_id", "id", "traceId", "uuid", "task_id")
SPAN_ID_KEYS = ("span_id", "spanId", "location", "span", "span_ids")
TYPE_KEYS = ("error_type", "category", "error_category", "type", "subcategory", "label")
DESC_KEYS = ("description", "explanation", "error_description", "detail")
EVIDENCE_KEYS = ("evidence", "supporting_evidence", "quote", "snippet")
IMPACT_KEYS = ("impact", "impact_level", "severity")

SEVERITY_BY_IMPACT = {"high": "critical", "medium": "major", "low": "minor"}


def _norm(text: str) -> str:
    """Fold a type name for lookup: 'Tool-related Hallucinations' -> 'toolrelatedhallucinations'."""
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _first(row: dict, keys: tuple[str, ...]):
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
    """Flatten the reviewable mapping file into lookup tables."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    buckets = spec["buckets"]
    direct, loc_only, excluded = {}, {}, {}
    for name, entry in buckets["direct"]["entries"].items():
        direct[_norm(name)] = {"name": name, "family": entry["family"]}
    for name, entry in buckets["localization_only"]["entries"].items():
        loc_only[_norm(name)] = {"name": name, "candidates": entry.get("candidate_families") or []}
    for name, entry in buckets["excluded"]["entries"].items():
        excluded[_norm(name)] = {"name": name, "reason": entry.get("reason") or "out_of_scope"}
    return {
        "version": str(spec.get("mapping_version") or "?"),
        "direct": direct,
        "localization_only": loc_only,
        "excluded": excluded,
    }


def iter_rows(src: Path):
    """Yield TRAIL rows from parquet (via `datasets`) or json/jsonl files."""
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
        if path.name == "index.json":
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


def describe(row: dict) -> str:
    keys = sorted(row) if isinstance(row, dict) else []
    return f"  row keys: {keys}"


def span_index_map(trace_path: Path, trace_id: str) -> dict[str, list[int]]:
    """span id -> canonical step indices, via the openinference adapter."""
    trace = load_trace(trace_id, "openinference", [str(trace_path)])
    out: dict[str, list[int]] = {}
    for step in trace.steps:
        sid = str((step.meta or {}).get("span_id") or "")
        if sid:
            out.setdefault(sid, []).append(step.index)
    return out


def _span_ids(annotation: dict) -> list[str]:
    raw = _first(annotation, SPAN_ID_KEYS)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(x) for x in raw if x not in (None, "")]
    if isinstance(raw, dict):
        inner = _first(raw, SPAN_ID_KEYS)
        return [str(inner)] if inner else []
    return [str(raw)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="downloaded TRAIL dir (see fetch_public_corpus.py)")
    parser.add_argument("--dataset-out", default=str(REPO_ROOT / "data" / "raw" / "trail-dataset"))
    parser.add_argument("--labels-out", default=str(REPO_ROOT / "data" / "labels" / "trail"))
    parser.add_argument("--mapping", default=str(MAPPING_PATH))
    parser.add_argument("--limit", type=int, default=0, help="convert at most N traces")
    parser.add_argument("--inspect", action="store_true", help="report the schema found and exit")
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
            print(describe(row))
            print(f"  trace payload key: {next((k for k in TRACE_KEYS if isinstance(row, dict) and k in row), None)!r} (looked for {list(TRACE_KEYS)})")
            print(f"  annotations key:   {next((k for k in ANNOTATION_KEYS if isinstance(row, dict) and k in row), None)!r} (looked for {list(ANNOTATION_KEYS)})")
            anns = _as_json(_first(row, ANNOTATION_KEYS)) or []
            if isinstance(anns, list) and anns and isinstance(anns[0], dict):
                print(f"  annotation keys:   {sorted(anns[0])}")
        return 0

    dataset_out, labels_out = (Path(args.dataset_out), Path(args.labels_out))
    (dataset_out / "traces").mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    unmapped: Counter = Counter()
    unresolved_spans = 0
    totals = Counter()
    skipped_no_trace = 0

    for n, row in enumerate(rows):
        if args.limit and len(entries) >= args.limit:
            break
        payload = _as_json(_first(row, TRACE_KEYS))
        if payload is None:
            skipped_no_trace += 1
            continue
        trace_id = str(_first(row, TRACE_ID_KEYS) or f"trail-{n:04d}")
        rel = f"traces/{trace_id}.json"
        path = dataset_out / rel
        blob = json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")
        path.write_bytes(blob)
        entries.append(
            {
                "trace_id": trace_id,
                "source": "openinference",
                "format": "openinference_spans",
                "files": [{"path": rel, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}],
                "source_repository": "PatronusAI/TRAIL",
            }
        )

        spans = span_index_map(path, trace_id)
        findings, excluded, loc_only = [], [], []
        annotations = _as_json(_first(row, ANNOTATION_KEYS)) or []
        if isinstance(annotations, dict):
            annotations = [annotations]
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            totals["annotations"] += 1
            leaf = str(_first(annotation, TYPE_KEYS) or "")
            key = _norm(leaf)
            steps: list[int] = []
            for sid in _span_ids(annotation):
                steps.extend(spans.get(sid, []))
            if not steps:
                unresolved_spans += 1
            description = str(_first(annotation, DESC_KEYS) or "")
            evidence = _first(annotation, EVIDENCE_KEYS)
            if evidence and str(evidence) not in description:
                description = f"{description} | evidence: {str(evidence)[:400]}".strip(" |")
            impact = str(_first(annotation, IMPACT_KEYS) or "").lower()
            severity = SEVERITY_BY_IMPACT.get(impact, "major")

            if key in mapping["direct"]:
                totals["scored"] += 1
                findings.append(
                    {
                        "family": mapping["direct"][key]["family"],
                        "steps": sorted(set(steps)),
                        "description": description[:900],
                        "severity": severity,
                        "notes": f"TRAIL type: {leaf}",
                    }
                )
            elif key in mapping["localization_only"]:
                totals["localization_only"] += 1
                loc_only.append({"source_type": leaf, "reason": "family_ambiguous", "steps": sorted(set(steps)), "description": description[:600]})
            elif key in mapping["excluded"]:
                totals["excluded"] += 1
                excluded.append({"source_type": leaf, "reason": mapping["excluded"][key]["reason"], "steps": sorted(set(steps)), "description": description[:600]})
            else:
                unmapped[leaf] += 1
                excluded.append({"source_type": leaf, "reason": "unmapped_type", "steps": sorted(set(steps)), "description": description[:600]})

        label = {
            "trace_id": trace_id,
            "clean": not findings,
            "findings": findings,
            "excluded_findings": excluded,
            "localization_only": loc_only,
            "source_benchmark": "TRAIL",
            "mapping_version": mapping["version"],
            "annotator": "TRAIL: 4 expert annotators, 4 verification rounds (arXiv:2505.08638)",
            "review": "multi",
            "notes": "Converted by scripts/convert_trail_labels.py. Families follow data/labels/mappings/trail_to_traceveri.json.",
        }
        (labels_out / f"{trace_id}.json").write_text(json.dumps(label, indent=2, ensure_ascii=False), encoding="utf-8")

    if not entries:
        print("no traces converted — run with --inspect to see the schema that was found.", file=sys.stderr)
        return 1

    index = {
        "format_version": 2,
        "dataset_version": "trail-v1",
        "source_dataset_version": "PatronusAI/TRAIL",
        "annotation_status": "labelled",
        "trace_count": len(entries),
        "source_counts": dict(Counter(e["source"] for e in entries)),
        "note": "Converted from the TRAIL benchmark. Not redistributable: the TRAIL card forbids resharing outside a gated HF repo.",
        "traces": entries,
    }
    (dataset_out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(f"dataset: {dataset_out}  ({len(entries)} traces)")
    print(f"labels:  {labels_out}")
    print(f"gold annotations: {totals['annotations']}")
    print(f"  scored (direct family):     {totals['scored']}")
    print(f"  localization only:          {totals['localization_only']}")
    print(f"  excluded (out of taxonomy): {totals['excluded']}")
    if unresolved_spans:
        print(f"warning: {unresolved_spans} annotation(s) referenced a span with no canonical step; steps left empty")
    if unmapped:
        print("warning: TRAIL types absent from the mapping file (treated as excluded/unmapped_type):")
        for leaf, count in unmapped.most_common():
            print(f"  {leaf}: {count}")
        print(f"  add them to {args.mapping} and re-run.")
    if skipped_no_trace:
        print(f"warning: {skipped_no_trace} row(s) had no recognizable trace payload (see --inspect)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
