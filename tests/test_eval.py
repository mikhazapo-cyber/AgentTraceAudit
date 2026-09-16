from traceaudit.eval import (
    Label,
    LabelFinding,
    match_findings,
    mappable_gold_findings,
    score_results,
)
from traceaudit.schemas import Finding, TraceResult


def _f(family: str, steps: list[int], status: str = "confirmed") -> Finding:
    return Finding(
        finding_id="x",
        trace_id="t",
        family=family,  # type: ignore[arg-type]
        description="d",
        steps=steps,
        status=status,  # type: ignore[arg-type]
    )


def test_strict_match_family_and_overlap():
    matches, fps, misses = match_findings(
        [_f("incorrect_tool_use", [4, 5])],
        [LabelFinding(family="incorrect_tool_use", steps=[5])],
    )
    assert len(matches) == 1
    assert fps == []
    assert misses == []


def test_abstention_is_not_fp():
    matches, fps, misses = match_findings(
        [_f("incorrect_tool_use", [4], "insufficient_evidence")],
        [LabelFinding(family="incorrect_tool_use", steps=[4])],
    )
    assert matches == []
    assert fps == []
    assert len(misses) == 1


def test_score_results_marks_development():
    result = TraceResult(
        trace_id="t",
        findings=[_f("instruction_violation", [1])],
    )
    report = score_results(
        [result],
        {
            "t": Label(
                trace_id="t",
                findings=[LabelFinding(family="instruction_violation", steps=[1])],
            )
        },
    )
    assert report["split"] == "development"
    assert report["overall"]["tp"] == 1
    assert report["overall"]["f1"] == 1.0
    assert report["mappable"]["held_out"] is False
    assert report["mappable"]["tp"] == 1
    assert report["per_trace"][0]["mappable_tp"] == 1
    assert report["per_trace"][0]["n_mappable_gold"] == 1


def test_eval_still_matches_without_gold_class():
    pred = _f("incorrect_tool_use", [4, 5])
    pred.error_class = "bad_arguments"
    pred.confidence = 91
    matches, fps, misses = match_findings(
        [pred],
        [LabelFinding(family="incorrect_tool_use", steps=[5])],
    )
    assert len(matches) == 1
    assert fps == []
    assert misses == []


def test_score_results_records_class_without_scoring_it():
    finding = _f("instruction_violation", [1])
    finding.error_class = "broke_explicit_rule"
    finding.confidence = 88
    result = TraceResult(trace_id="t", findings=[finding])
    report = score_results(
        [result],
        {
            "t": Label(
                trace_id="t",
                findings=[LabelFinding(family="instruction_violation", steps=[1])],
            )
        },
    )
    assert report["overall"]["tp"] == 1
    row = report["per_trace"][0]
    assert row["predicted_classes"][0]["error_class"] == "broke_explicit_rule"
    assert row["predicted_classes"][0]["confidence"] == 88


def test_load_labels_keeps_meta_and_ignores_unknown(tmp_path):
    from traceaudit.eval import load_labels

    path = tmp_path / "t.json"
    path.write_text(
        """
        {
          "trace_id": "t",
          "clean": false,
          "findings": [{"family": "other", "steps": [1], "description": "x"}],
          "meta": {"corpus": "whowhen"},
          "source_benchmark": "Who&When",
          "gold_completeness": "who_when_only"
        }
        """,
        encoding="utf-8",
    )
    labels = load_labels(path)
    assert labels["t"].meta["corpus"] == "whowhen"
    assert labels["t"].findings[0].family == "other"


def test_payment_abort_is_not_scored_as_fn():
    result = TraceResult(
        trace_id="t",
        status="error",
        error="provider payment required; add credits and retry",
    )
    report = score_results(
        [result],
        {
            "t": Label(
                trace_id="t",
                findings=[LabelFinding(family="instruction_violation", steps=[1])],
            )
        },
    )
    assert report["overall"]["fn"] == 0
    assert report["mappable"]["fn"] == 0
    assert report["per_trace"][0]["unscored"] is True


def test_mappable_subset_excludes_other_and_injected():
    pred_ok = _f("instruction_violation", [1])
    pred_other = _f("other", [2])
    pred_other.error_class = "other_error"
    result = TraceResult(trace_id="t", findings=[pred_ok, pred_other])
    label = Label(
        trace_id="t",
        review="human",
        findings=[
            LabelFinding(family="instruction_violation", steps=[1]),
            LabelFinding(family="other", steps=[2], description="who/when only"),
        ],
    )
    report = score_results([result], {"t": label})
    assert report["overall"]["tp"] == 2
    assert report["overall"]["fn"] == 0
    assert report["mappable"]["tp"] == 1
    assert report["mappable"]["fn"] == 0
    assert report["mappable"]["fp"] == 0
    assert report["mappable"]["n_gold_findings"] == 1
    assert report["mappable"]["held_out"] is False
    assert "not `other`" in report["mappable"]["note"]
    assert mappable_gold_findings(label)[0].family == "instruction_violation"

    injected = Label(
        trace_id="inj",
        review="injected",
        findings=[LabelFinding(family="instruction_violation", steps=[1])],
    )
    inj_report = score_results(
        [TraceResult(trace_id="inj", findings=[_f("instruction_violation", [1])])],
        {"inj": injected},
    )
    assert inj_report["overall"]["tp"] == 1
    assert inj_report["mappable"]["n_gold_findings"] == 0
    assert inj_report["mappable"]["tp"] == 0
    assert inj_report["per_trace"][0]["n_mappable_gold"] == 0


def test_mappable_miss_does_not_change_full_set():
    result = TraceResult(trace_id="t", findings=[])
    report = score_results(
        [result],
        {
            "t": Label(
                trace_id="t",
                findings=[
                    LabelFinding(family="incorrect_tool_use", steps=[4]),
                    LabelFinding(family="other", steps=[8]),
                ],
            )
        },
    )
    assert report["overall"]["fn"] == 2
    assert report["mappable"]["fn"] == 1
    assert report["mappable"]["n_gold_findings"] == 1
