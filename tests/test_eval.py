from pathlib import Path

from traceaudit.config import Config
from traceaudit.eval.metrics import evaluate
from traceaudit.llm import NullClient
from traceaudit.normalize import load_dataset, load_trace
from traceaudit.pipeline import analyze_trace
from traceaudit.report import write_run_artifacts

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_deterministic_eval(tmp_path: Path):
    dataset = load_dataset(ROOT / "data" / "synthetic")
    results = [
        analyze_trace(load_trace(tid, meta["source"], meta["files"], 4000), Config(), NullClient())
        for tid, meta in dataset.items()
    ]
    out = write_run_artifacts(tmp_path, results)
    report = evaluate(out / "findings.jsonl", ROOT / "data" / "synthetic" / "labels", semantic_mode="deterministic")
    assert report["false_positives_on_clean_traces"]["n_fp"] == 0
    assert report["overall"]["strict"]["tp"] >= 3
    assert report["insufficient_evidence_tracked_separately"]["count"] >= 0
