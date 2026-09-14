"""Development-set floor: deterministic scan, no API.

These numbers are a recall floor, not the live derive → agents → judge result.
"""

from pathlib import Path

from traceaudit.config import Config
from traceaudit.eval.metrics import evaluate
from traceaudit.llm import NullClient
from traceaudit.normalize import load_dataset, load_trace
from traceaudit.pipeline import analyze_trace
from traceaudit.report import write_run_artifacts

ROOT = Path(__file__).resolve().parents[1]


def test_dev_deterministic_precision_floor(tmp_path: Path):
    dataset = load_dataset(ROOT / "data" / "dev")
    results = []
    for tid, meta in dataset.items():
        trace = load_trace(tid, meta["source"], meta["files"], 6000)
        results.append(analyze_trace(trace, Config(), NullClient()))
    out = write_run_artifacts(tmp_path / "run", results)
    report = evaluate(out / "findings.jsonl", ROOT / "data" / "labels" / "dev", semantic_mode="deterministic")
    strict = report["overall"]["strict"]
    clean = report["false_positives_on_clean_traces"]
    # Precision first: a clean-trace FP means the structural layer is too eager.
    assert clean["n_fp"] == 0, clean["detail"]
    assert strict["precision"] == 1.0
    # Mechanical gold (schema, unknown tool, identical repeats, formation errors, phantom failure).
    # The two remaining misses are instruction-precondition golds (live LLM path).
    assert strict["tp"] >= 5
    assert strict["recall"] >= 0.71
    assert strict["f1"] >= 0.83
    per = report["per_trace"]
    assert per["external-041"]["strict"]["tp"] >= 1
    assert per["external-042"]["strict"]["tp"] >= 1
    assert per["tavii-022"]["strict"]["tp"] >= 1


def test_dev_traces_normalize_nonempty():
    dataset = load_dataset(ROOT / "data" / "dev")
    assert len(dataset) == 17
    for tid, meta in dataset.items():
        trace = load_trace(tid, meta["source"], meta["files"], 4000)
        assert trace.steps, tid
        assert trace.trace_id == tid
