import json
from pathlib import Path

from traceaudit.eval import load_labels, match_findings
from traceaudit.normalize import load_dataset, load_trace
from traceaudit.schemas import Finding


def test_dev_dataset_still_seventeen(repo_root: Path):
    dataset = load_dataset(repo_root / "data" / "dev")
    labels = load_labels(repo_root / "data" / "labels" / "dev")
    assert len(dataset) == 17
    assert len(labels) == 17


def test_checked_dataset_recovers_steps(repo_root: Path):
    root = repo_root / "data" / "checked"
    dataset = load_dataset(root)
    assert len(dataset) >= 90
    for tid, meta in dataset.items():
        trace = load_trace(tid, meta.get("source") or "", meta["files"])
        assert trace.steps, tid


def test_checked_gold_steps_exist(repo_root: Path):
    dataset = load_dataset(repo_root / "data" / "checked")
    labels = load_labels(repo_root / "data" / "labels" / "checked")
    assert set(dataset) == set(labels)
    dirty = 0
    for tid, label in labels.items():
        trace = load_trace(tid, dataset[tid].get("source") or "", dataset[tid]["files"])
        n = len(trace.steps)
        assert n >= 1, tid
        if label.clean:
            assert label.findings == []
            continue
        dirty += 1
        assert label.findings, tid
        for finding in label.findings:
            assert finding.family != "insufficient_evidence", tid
            assert finding.steps, tid
            assert all(0 <= i < n for i in finding.steps), (tid, finding.steps, n)
    assert dirty >= 85
    assert sum(1 for lab in labels.values() if lab.clean) >= 4


def test_checked_probes_and_eval_match(repo_root: Path):
    dataset = load_dataset(repo_root / "data" / "checked")
    labels = load_labels(repo_root / "data" / "labels" / "checked")
    probes = {
        "arx-tau-002": ("instruction_violation", [2], "list_all_product_types"),
        "const-contradict": ("evidence_contradiction", [3], "28"),
        "const-auth-skip": ("instruction_violation", [1], "refund_order"),
    }
    for tid in (
        "const-clean-retry",
        "const-clean-search",
        "const-clean-verify",
        "const-clean-skill",
    ):
        assert labels[tid].clean
        assert labels[tid].findings == []
    for tid, (family, steps, needle) in probes.items():
        trace = load_trace(tid, dataset[tid]["source"], dataset[tid]["files"])
        blob = " ".join(trace.steps[i].text() for i in steps)
        assert needle.lower() in blob.lower(), (tid, blob[:200])
        pred = Finding(
            finding_id="t",
            trace_id=tid,
            family=family,  # type: ignore[arg-type]
            description="d",
            steps=steps,
            status="confirmed",
        )
        matches, fps, _misses = match_findings([pred], labels[tid].findings)
        assert matches, tid
        assert fps == []


def test_precision_index_lists_checked_sample(repo_root: Path):
    index = json.loads(
        (repo_root / "data" / "precision" / "index.json").read_text(encoding="utf-8")
    )
    labels = load_labels(repo_root / "data" / "labels" / "checked")
    ids = [row["trace_id"] for row in index["traces"]]
    assert index["trace_count"] == len(ids) >= 90
    assert set(ids) <= set(labels)
    assert index["n_injected"] >= 20
    assert index["n_human"] >= 60
    assert index["n_clean"] >= 4
    assert any(row["clean"] for row in index["traces"])
    assert all(row.get("dataset") == "data/checked" for row in index["traces"])


def test_whowhen_gold_names_the_labelled_agent(repo_root: Path):
    dataset = load_dataset(repo_root / "data" / "checked")
    labels = load_labels(repo_root / "data" / "labels" / "checked")
    n = 0
    for tid, label in labels.items():
        if not tid.startswith("ww-"):
            continue
        n += 1
        agent = ""
        if "mistake_agent=" in label.notes:
            agent = label.notes.split("mistake_agent=", 1)[1].split(";", 1)[0].strip()
        key = agent.lower().split()[0]
        assert key, tid
        trace = load_trace(tid, dataset[tid].get("source") or "", dataset[tid]["files"])
        named = False
        for i in label.findings[0].steps:
            if 0 <= i < len(trace.steps) and key in (trace.steps[i].name or "").lower():
                named = True
                break
        assert named, (
            tid,
            agent,
            label.findings[0].steps,
            [
                (i, trace.steps[i].name, trace.steps[i].role)
                for i in label.findings[0].steps
                if 0 <= i < len(trace.steps)
            ],
        )
    assert n >= 20


def test_aegis_rows_are_marked_injected(repo_root: Path):
    labels = load_labels(repo_root / "data" / "labels" / "checked")
    injected = [lab for tid, lab in labels.items() if tid.startswith("aegis-")]
    assert injected
    for lab in injected:
        assert lab.review == "injected"
        assert lab.findings
        assert not lab.clean


def test_agentrx_map_is_honest_on_source_category(repo_root: Path):
    import importlib.util

    path = repo_root / "scripts" / "import_checked_traces.py"
    spec = importlib.util.spec_from_file_location("import_checked_traces", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.map_agentrx_family("Instruction Adherence Failure") == (
        "instruction_violation",
        "broke_explicit_rule",
    )
    assert mod.map_agentrx_family("Plan Adherence Failure") == (
        "instruction_violation",
        "role_or_spec_deviation",
    )
    assert mod.map_agentrx_family("Intent Plan Misalignment") == (
        "other",
        "other_error",
    )
    assert mod.map_agentrx_family("Underspecified User Intent") == (
        "other",
        "other_error",
    )
    assert mod.map_agentrx_family("Intent Not Supported") == ("other", "other_error")


def test_agentrx_checked_labels_follow_source(repo_root: Path):
    labels = load_labels(repo_root / "data" / "labels" / "checked")
    n = 0
    for tid, lab in labels.items():
        if not tid.startswith("arx-"):
            continue
        n += 1
        for finding in lab.findings:
            cat = finding.notes or ""
            if any(
                key in cat
                for key in (
                    "Intent Plan",
                    "Underspecified",
                    "Intent Not Supported",
                )
            ):
                assert finding.family == "other", (tid, cat, finding.family)
            if "Instruction Adherence" in cat or (
                "Plan Adherence" in cat and "Intent Plan" not in cat
            ):
                assert finding.family == "instruction_violation", (
                    tid,
                    cat,
                    finding.family,
                )
    assert n >= 20
