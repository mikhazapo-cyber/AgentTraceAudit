"""Who&When Pro conversion: covers the families TRAIL cannot, without lying about precision.

Every trace holds exactly one injected error and nothing else is annotated, so a
correct finding elsewhere scores as a false positive. The converted labels must
carry that fact, and the evaluator must print it above any metric.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from traceaudit.eval.matching import load_labels  # noqa: E402
from traceaudit.eval.metrics import _coverage, _coverage_markdown  # noqa: E402
from traceaudit.schemas import FAMILIES  # noqa: E402


def _load_script(name):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


convert = _load_script("convert_whowhen_labels")
MAPPING = REPO_ROOT / "data" / "labels" / "mappings" / "whowhen_to_traceveri.json"


# --- the mapping file ------------------------------------------------------


def test_mapping_buckets_are_disjoint_and_counted():
    spec = json.loads(MAPPING.read_text(encoding="utf-8"))
    buckets = spec["buckets"]
    names = [set(buckets[b]["entries"]) for b in ("direct", "localization_only", "excluded")]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not (a & b)
    summary = spec["impact_summary"]
    assert [len(n) for n in names] == [
        summary["leaf_types_direct"],
        summary["leaf_types_localization_only"],
        summary["leaf_types_excluded"],
    ]
    assert sum(len(n) for n in names) == summary["leaf_types_total"]


def test_direct_mappings_name_real_families_with_rationale():
    spec = json.loads(MAPPING.read_text(encoding="utf-8"))
    for name, entry in spec["buckets"]["direct"]["entries"].items():
        assert entry["family"] in FAMILIES, f"{name} -> {entry['family']!r}"
        assert entry["rationale"]


def test_it_covers_the_families_trail_cannot():
    """The reason for adding this corpus at all."""
    spec = json.loads(MAPPING.read_text(encoding="utf-8"))
    families = {e["family"] for e in spec["buckets"]["direct"]["entries"].values()}
    assert "ignored_feedback" in families
    assert "unsupported_success" in families

    trail = json.loads((REPO_ROOT / "data" / "labels" / "mappings" / "trail_to_traceveri.json").read_text(encoding="utf-8"))
    trail_families = {e["family"] for e in trail["buckets"]["direct"]["entries"].values()}
    assert "ignored_feedback" not in trail_families
    assert "unsupported_success" not in trail_families


def test_mapping_declares_the_gold_set_incomplete():
    spec = json.loads(MAPPING.read_text(encoding="utf-8"))
    assert spec["gold_completeness"] == "decisive_error_only"
    assert "not interpretable" in spec["scoring_warning"].lower()


# --- end-to-end conversion -------------------------------------------------


def _rows():
    return [
        {
            "id": "wwp-verify",
            "framework": "smolagents",
            "modality": "text",
            "benchmark": "gaia",
            "task": {"query": "Confirm the invoice was paid."},
            "trajectory": [
                {"role": "assistant", "content": "Checking the ledger.", "tool_calls": [{"id": "t1", "function": {"name": "get_invoice", "arguments": {"id": 7}}}]},
                {"role": "tool", "content": "{\"status\": \"pending\"}"},
                {"role": "assistant", "content": "The invoice has been paid."},
            ],
            "ground_truth": {"mode": "inadequate verification", "agent": "planner", "step": 2},
        },
        {
            "id": "wwp-loop",
            "framework": "MetaGPT",
            "modality": "text",
            "task": {"query": "Find the population."},
            "trajectory": [
                {"role": "assistant", "content": "search", "tool_calls": [{"id": "a", "function": {"name": "search", "arguments": {"q": "pop"}}}]},
                {"role": "tool", "content": "545000"},
                {"role": "assistant", "content": "search", "tool_calls": [{"id": "b", "function": {"name": "search", "arguments": {"q": "pop"}}}]},
            ],
            "ground_truth": {"mode": "repetitive looping", "step": 2},
        },
        {
            "id": "wwp-halluc",
            "framework": "debate",
            "modality": "text",
            "trajectory": [{"role": "assistant", "content": "Mars has three moons."}],
            "ground_truth": {"mode": "hallucination", "step": 0},
        },
        {
            "id": "wwp-visual",
            "framework": "PixelCraft",
            "modality": "text",
            "trajectory": [{"role": "assistant", "content": "The sign reads STOP."}],
            "ground_truth": {"mode": "visual misidentification", "step": 0},
        },
        {
            "id": "wwp-novel",
            "framework": "CoAct",
            "modality": "text",
            "trajectory": [{"role": "assistant", "content": "..."}],
            "ground_truth": {"mode": "Quantum Indecision", "step": 0},
        },
        {
            "id": "wwp-video",
            "framework": "EfficientVideoAgent",
            "modality": "video",
            "trajectory": [{"role": "assistant", "content": "frame 12"}],
            "ground_truth": {"mode": "spatial grounding", "step": 0},
        },
    ]


@pytest.fixture
def converted(tmp_path, monkeypatch, capsys):
    src = tmp_path / "src"
    src.mkdir()
    (src / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in _rows()), encoding="utf-8")
    dataset_out, labels_out = (tmp_path / "ds", tmp_path / "labels")
    monkeypatch.setattr(
        sys,
        "argv",
        ["convert_whowhen_labels", "--src", str(src), "--dataset-out", str(dataset_out), "--labels-out", str(labels_out)],
    )
    assert convert.main() == 0
    return dataset_out, labels_out, capsys.readouterr().out


def test_only_the_requested_modality_is_kept(converted):
    dataset_out, _, output = converted
    index = json.loads((dataset_out / "index.json").read_text(encoding="utf-8"))
    assert index["trace_count"] == 5  # the video row is dropped
    assert "'modality': 1" in output


def test_trajectories_become_loadable_openai_chat_traces(converted):
    dataset_out, _, _ = converted
    from traceaudit.normalize import load_dataset, load_trace

    entries = load_dataset(dataset_out)
    assert all(info["source"] == "openai-chat" for info in entries.values())
    trace = load_trace("wwp-verify", "openai-chat", entries["wwp-verify"]["files"])
    assert trace.task == "Confirm the invoice was paid."
    call = next(s for s in trace.steps if s.kind == "tool_call")
    assert call.name == "get_invoice"
    assert call.arguments == {"id": 7}
    assert any(s.kind == "tool_result" for s in trace.steps)


def test_modes_reach_the_families_the_dev_set_lacks(converted):
    _, labels_out, _ = converted
    labels = load_labels(labels_out)
    assert [f.family for f in labels["wwp-verify"].findings] == ["unsupported_success"]
    assert [f.family for f in labels["wwp-loop"].findings] == ["redundant_action"]
    assert labels["wwp-verify"].findings[0].steps == [2]


def test_ambiguous_and_out_of_scope_modes_are_held_out(converted):
    _, labels_out, _ = converted
    labels = load_labels(labels_out)
    assert labels["wwp-halluc"].findings == []
    assert [e.source_type for e in labels["wwp-halluc"].localization_only] == ["hallucination"]
    assert labels["wwp-visual"].excluded_findings[0].reason == "out_of_modality"
    assert labels["wwp-novel"].excluded_findings[0].reason == "unmapped_type"


def test_unmapped_modes_are_reported(converted):
    _, _, output = converted
    assert "Quantum Indecision" in output
    assert "absent from the mapping file" in output


def test_labels_record_that_precision_is_unreadable(converted):
    _, labels_out, output = converted
    label = load_labels(labels_out)["wwp-verify"]
    assert label.gold_completeness == "decisive_error_only"
    assert label.source_benchmark == "Who&When Pro"
    assert "framework=smolagents" in label.notes
    assert "precision is not interpretable" in output


def test_coverage_flags_precision_as_uninterpretable(converted):
    _, labels_out, _ = converted
    cov = _coverage(load_labels(labels_out))
    assert cov["precision_interpretable"] is False
    assert cov["gold_completeness"] == ["decisive_error_only"]
    assert cov["scored"] == 2
    assert cov["localization_only"] == 1
    assert cov["excluded"] == 2
    md = "\n".join(_coverage_markdown(cov))
    assert "Precision is not interpretable on this gold set" in md
    assert "counts as a false positive" in md


def test_complete_gold_sets_are_not_warned_about():
    cov = _coverage(load_labels(REPO_ROOT / "data" / "labels" / "dev"))
    assert cov["precision_interpretable"] is True
    assert cov["gold_completeness"] == ["complete"]
    assert _coverage_markdown(cov) == []
