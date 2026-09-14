"""Guards against drift between the shipped dataset, its labels, and the docs."""

import collections
import hashlib
import json
import re
from pathlib import Path

from traceaudit.normalize import _ADAPTER_MODULE_MAP, verify_dataset

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dev"
LABELS = ROOT / "data" / "labels" / "dev"


def _index():
    return json.loads((DATASET / "index.json").read_text(encoding="utf-8"))


def test_index_counts_match_listed_traces():
    idx = _index()
    traces = idx["traces"]
    assert idx["trace_count"] == len(traces)
    histogram = collections.Counter(t["source"] for t in traces)
    assert idx["source_counts"] == dict(histogram)
    assert sum(idx["source_counts"].values()) == idx["trace_count"]


def test_index_file_hashes_are_current():
    for entry in _index()["traces"]:
        for f in entry["files"]:
            path = DATASET / f["path"]
            assert path.exists(), f"{f['path']} listed in index.json but missing"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == f["sha256"], f"{f['path']} content differs from index.json"
            assert path.stat().st_size == f["bytes"]


def test_labels_and_index_cover_the_same_traces():
    index_ids = {t["trace_id"] for t in _index()["traces"]}
    label_ids = {p.stem for p in LABELS.glob("*.json")}
    assert index_ids == label_ids


def test_gold_label_composition_is_what_the_docs_claim():
    positive, clean, findings = 0, 0, 0
    for path in sorted(LABELS.glob("*.json")):
        label = json.loads(path.read_text(encoding="utf-8"))
        n = len(label.get("findings") or [])
        findings += n
        if n:
            positive += 1
            assert label["clean"] is False, f"{path.name} has findings but clean=True"
        else:
            clean += 1
            assert label["clean"] is True, f"{path.name} has no findings but clean=False"
    assert (positive, clean, findings) == (5, 12, 7)


def test_verify_dataset_passes_on_the_shipped_set():
    v = verify_dataset(DATASET)
    assert v["n_traces"] == v["declared_traces"] == 17
    assert v["missing"] == 0
    assert v["mismatched"] == 0


def test_verify_dataset_notices_tampering(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "t.json").write_text("changed", encoding="utf-8")
    index = {
        "trace_count": 2,
        "traces": [
            {"trace_id": "t", "files": [{"path": "traces/t.json", "sha256": "0" * 64}]},
            {"trace_id": "gone", "files": [{"path": "traces/gone.json", "sha256": "1" * 64}]},
        ],
    }
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    v = verify_dataset(tmp_path)
    assert v["mismatched"] == 1
    assert v["missing"] == 1


def test_report_quotes_the_real_adapter_count():
    formats = len(_ADAPTER_MODULE_MAP)
    modules = len(set(_ADAPTER_MODULE_MAP.values()))
    text = (ROOT / "docs" / "report.md").read_text(encoding="utf-8")
    match = re.search(r"(\d+) harness formats \((\d+) adapter modules", text)
    assert match, "docs/report.md must state the harness format/module count"
    assert (int(match.group(1)), int(match.group(2))) == (formats, modules)
