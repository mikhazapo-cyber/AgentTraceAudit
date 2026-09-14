"""TRAIL conversion: span ids become step indices, taxonomy gaps stay visible.

TRAIL is gated, so this exercises the converter against a TRAIL-shaped row built
from the OpenInference fixture. What it pins down is the logic that would
otherwise silently produce a meaningless F1: which annotations are scored, which
are held out, and whether span references resolve to real steps.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

FIX = Path(__file__).parent / "fixtures"

from traceaudit.eval.matching import load_labels  # noqa: E402
from traceaudit.eval.metrics import _coverage, _coverage_markdown, evaluate  # noqa: E402


def _load_script(name):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


convert = _load_script("convert_trail_labels")

MAPPING = REPO_ROOT / "data" / "labels" / "mappings" / "trail_to_traceveri.json"


# --- the mapping file itself ------------------------------------------------


def test_mapping_buckets_are_disjoint_and_complete():
    spec = json.loads(MAPPING.read_text(encoding="utf-8"))
    buckets = spec["buckets"]
    names = [set(buckets[b]["entries"]) for b in ("direct", "localization_only", "excluded")]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not (a & b), f"a TRAIL type appears in two buckets: {a & b}"
    summary = spec["impact_summary"]
    assert [len(n) for n in names] == [
        summary["leaf_types_direct"],
        summary["leaf_types_localization_only"],
        summary["leaf_types_excluded"],
    ]
    assert sum(len(n) for n in names) == summary["leaf_types_total"]


def test_every_direct_mapping_names_a_real_family():
    from traceaudit.schemas import FAMILIES

    spec = json.loads(MAPPING.read_text(encoding="utf-8"))
    for name, entry in spec["buckets"]["direct"]["entries"].items():
        assert entry["family"] in FAMILIES, f"{name} maps to unknown family {entry['family']!r}"
        assert entry["rationale"], f"{name} has no rationale"


def test_ambiguous_and_excluded_types_claim_no_family():
    spec = json.loads(MAPPING.read_text(encoding="utf-8"))
    for entry in spec["buckets"]["localization_only"]["entries"].values():
        assert entry["family"] is None
        assert len(entry["candidate_families"]) >= 1
    for entry in spec["buckets"]["excluded"]["entries"].values():
        assert entry["reason"] in {"environment_fault", "no_traceveri_family"}


def test_loader_folds_type_names_for_lookup():
    mapping = convert.load_mapping(MAPPING)
    assert mapping["direct"][convert._norm("Instruction Non-compliance")]["family"] == "instruction_violation"
    assert mapping["direct"][convert._norm("resource abuse")]["family"] == "redundant_action"
    assert convert._norm("Tool-related Hallucinations") == "toolrelatedhallucinations"


# --- end-to-end conversion -------------------------------------------------


def _trail_row():
    """A TRAIL-shaped row: an OpenInference trace plus span-keyed annotations."""
    return {
        "trace_id": "trail-demo-1",
        "trace": json.loads((FIX / "openinference_otlp.json").read_text(encoding="utf-8")),
        "errors": [
            {
                "span_id": "aaaa0005",
                "error_category": "Instruction Non-compliance",
                "description": "Final answer invents a third moon and cites no tool.",
                "evidence": "Mars has three moons",
                "impact": "High",
            },
            {
                "span_id": "aaaa0004",
                "error_category": "Tool Selection Errors",
                "description": "Used the calculator for a factual lookup.",
                "impact": "Medium",
            },
            {
                "span_id": "aaaa0004",
                "error_category": "Formatting Errors",
                "description": "Argument was prose, not an expression.",
                "impact": "Low",
            },
            {
                "span_id": "aaaa0003",
                "error_category": "Rate Limiting",
                "description": "Search backend returned 429 once.",
                "impact": "Low",
            },
            {
                "span_id": "aaaa0002",
                "error_category": "Telepathy Errors",
                "description": "A type the mapping file has never heard of.",
                "impact": "Low",
            },
        ],
    }


@pytest.fixture
def converted(tmp_path, monkeypatch, capsys):
    src = tmp_path / "src"
    src.mkdir()
    (src / "rows.jsonl").write_text(json.dumps(_trail_row()) + "\n", encoding="utf-8")
    dataset_out, labels_out = (tmp_path / "ds", tmp_path / "labels")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_trail_labels",
            "--src", str(src),
            "--dataset-out", str(dataset_out),
            "--labels-out", str(labels_out),
        ],
    )
    assert convert.main() == 0
    return dataset_out, labels_out, capsys.readouterr().out


def test_dataset_is_written_in_the_standard_shape(converted):
    dataset_out, _, _ = converted
    index = json.loads((dataset_out / "index.json").read_text(encoding="utf-8"))
    assert index["format_version"] == 2
    assert index["trace_count"] == 1
    assert index["source_counts"] == {"openinference": 1}
    entry = index["traces"][0]
    assert entry["source"] == "openinference"
    blob = (dataset_out / entry["files"][0]["path"]).read_bytes()
    import hashlib

    assert hashlib.sha256(blob).hexdigest() == entry["files"][0]["sha256"]
    assert len(blob) == entry["files"][0]["bytes"]


def test_converted_dataset_loads_through_the_adapter(converted):
    dataset_out, _, _ = converted
    from traceaudit.normalize import load_dataset, load_trace

    entries = load_dataset(dataset_out)
    tid, info = next(iter(entries.items()))
    trace = load_trace(tid, info["source"], info["files"])
    assert trace.source_format == "openinference"
    assert any(s.kind == "tool_call" for s in trace.steps)


def test_span_references_resolve_to_real_step_indices(converted):
    _, labels_out, _ = converted
    label = load_labels(labels_out)["trail-demo-1"]
    scored = {f.family: f for f in label.findings}
    assert set(scored) == {"instruction_violation", "incorrect_tool_use"}
    for finding in label.findings:
        assert finding.steps, f"{finding.family} resolved to no steps"
        assert all(isinstance(s, int) for s in finding.steps)


def test_impact_becomes_severity(converted):
    _, labels_out, _ = converted
    label = load_labels(labels_out)["trail-demo-1"]
    by_family = {f.family: f for f in label.findings}
    assert by_family["instruction_violation"].severity == "critical"  # High
    assert by_family["incorrect_tool_use"].severity == "major"  # Medium


def test_evidence_is_folded_into_the_description(converted):
    _, labels_out, _ = converted
    label = load_labels(labels_out)["trail-demo-1"]
    text = next(f.description for f in label.findings if f.family == "instruction_violation")
    assert "evidence: Mars has three moons" in text


def test_ambiguous_and_out_of_scope_types_are_held_out_not_counted(converted):
    _, labels_out, _ = converted
    label = load_labels(labels_out)["trail-demo-1"]
    assert [e.source_type for e in label.localization_only] == ["Formatting Errors"]
    reasons = {e.source_type: e.reason for e in label.excluded_findings}
    assert reasons["Rate Limiting"] == "environment_fault"
    # An unrecognised type must be parked, never guessed into a family.
    assert reasons["Telepathy Errors"] == "unmapped_type"


def test_unmapped_types_are_reported_loudly(converted):
    _, _, output = converted
    assert "Telepathy Errors" in output
    assert "absent from the mapping file" in output


def test_label_records_its_provenance(converted):
    _, labels_out, _ = converted
    label = load_labels(labels_out)["trail-demo-1"]
    assert label.source_benchmark == "TRAIL"
    assert label.mapping_version == "1"
    assert "4 expert annotators" in label.annotator
    assert label.review == "multi"


# --- coverage reporting ----------------------------------------------------


def test_coverage_counts_the_held_out_annotations(converted):
    _, labels_out, _ = converted
    cov = _coverage(load_labels(labels_out))
    assert cov["gold_findings_total"] == 5
    assert cov["scored"] == 2
    assert cov["localization_only"] == 1
    assert cov["excluded"] == 2
    assert cov["scored_fraction"] == pytest.approx(0.4)
    assert cov["excluded_by_reason"] == {"environment_fault": 1, "unmapped_type": 1}
    assert cov["source_benchmarks"] == ["TRAIL"]


def test_eval_markdown_leads_with_scored_coverage(converted):
    dataset_out, labels_out, _ = converted
    cov = _coverage(load_labels(labels_out))
    md = "\n".join(_coverage_markdown(cov))
    assert "**2 of 5 gold findings** (40%)" in md
    assert "Do not quote P/R/F1 without this line" in md
    assert "environment_fault=1" in md
    assert "gold source: TRAIL · mapping v1" in md


def test_coverage_block_is_absent_for_fully_scorable_labels():
    # The shipped dev labels have nothing held out; no warning should appear.
    cov = _coverage(load_labels(REPO_ROOT / "data" / "labels" / "dev"))
    assert cov["excluded"] == 0
    assert cov["localization_only"] == 0
    assert _coverage_markdown(cov) == []


def test_live5_eval_if_recorded():
    live = REPO_ROOT / "results" / "live-5" / "findings.jsonl"
    if not live.is_file():
        pytest.skip("results/live-5 not recorded")
    ids = {"external-009", "external-017", "external-041", "external-042", "tavii-022"}
    rep = evaluate(live, REPO_ROOT / "data" / "labels" / "dev", only_ids=ids)
    assert rep["scored_coverage"]["scored"] == 7
    assert rep["overall"]["strict"]["f1"] == pytest.approx(1.0)
