"""Pretty-print contract: gold steps still load after 2-space JSON."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from traceaudit.eval import load_labels
from traceaudit.normalize import load_dataset, load_trace


def _pretty_text(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _assert_pretty_json(path: Path) -> None:
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n"), path
    data = json.loads(raw)
    assert raw.replace("\r\n", "\n") == _pretty_text(data), path


def _gold_pairs(
    dataset_dir: Path, labels_dir: Path
) -> dict[str, list[tuple[str, tuple[int, ...]]]]:
    dataset = load_dataset(dataset_dir)
    labels = load_labels(labels_dir)
    assert set(labels) <= set(dataset)
    out: dict[str, list[tuple[str, tuple[int, ...]]]] = {}
    for tid, label in labels.items():
        trace = load_trace(tid, dataset[tid].get("source") or "", dataset[tid]["files"])
        n = len(trace.steps)
        assert n >= 1, tid
        pairs = []
        for finding in label.findings:
            assert finding.steps, tid
            assert all(0 <= i < n for i in finding.steps), (tid, finding.steps, n)
            pairs.append((finding.family, tuple(finding.steps)))
        out[tid] = pairs
    return out


def test_dev_and_checked_json_is_pretty(repo_root: Path):
    roots = [
        repo_root / "data" / "dev",
        repo_root / "data" / "checked",
        repo_root / "data" / "labels",
        repo_root / "data" / "precision",
        repo_root / "examples",
    ]
    files: list[Path] = []
    for root in roots:
        files.extend(p for p in root.rglob("*.json") if p.is_file())
    assert files
    for path in files:
        _assert_pretty_json(path)


def test_gold_steps_survive_pretty_load(repo_root: Path):
    dev = _gold_pairs(repo_root / "data" / "dev", repo_root / "data" / "labels" / "dev")
    checked = _gold_pairs(
        repo_root / "data" / "checked", repo_root / "data" / "labels" / "checked"
    )
    assert len(dev) == 17
    assert len(checked) >= 90
    dirty = sum(1 for pairs in checked.values() if pairs)
    assert dirty >= 85


def test_indexes_match_files_and_labels(repo_root: Path):
    checked_index = json.loads(
        (repo_root / "data" / "checked" / "index.json").read_text(encoding="utf-8")
    )
    precision = json.loads(
        (repo_root / "data" / "precision" / "index.json").read_text(encoding="utf-8")
    )
    labels = load_labels(repo_root / "data" / "labels" / "checked")
    traces_dir = repo_root / "data" / "checked" / "traces"
    disk_ids = {p.stem for p in traces_dir.glob("*.json")}
    index_ids = {row["trace_id"] for row in checked_index["traces"]}
    precision_ids = {row["trace_id"] for row in precision["traces"]}
    assert index_ids == disk_ids == set(labels) == precision_ids
    assert checked_index["trace_count"] == len(index_ids)
    assert precision["trace_count"] == len(precision_ids)
    for row in checked_index["traces"]:
        for item in row["files"]:
            path = repo_root / "data" / "checked" / item["path"]
            blob = path.read_bytes()
            assert item["bytes"] == len(blob), path
            assert item["sha256"] == hashlib.sha256(blob).hexdigest(), path
