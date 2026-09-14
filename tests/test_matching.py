from traceaudit.eval.matching import LabelFinding, match_findings
from traceaudit.schemas import Finding


def _f(family, steps, status="confirmed"):
    return Finding(
        finding_id="x",
        trace_id="t",
        family=family,
        description="d",
        steps=steps,
        status=status,
    )


def test_strict_overlap_and_family():
    pred = [_f("incorrect_tool_use", [4, 5])]
    gold = [LabelFinding(family="incorrect_tool_use", steps=[4, 5], description="g")]
    matches, fps, misses = match_findings(pred, gold, "strict")
    assert len(matches) == 1 and not fps and not misses


def test_insufficient_is_not_scored():
    pred = [_f("incorrect_tool_use", [4], status="insufficient_evidence")]
    gold = [LabelFinding(family="incorrect_tool_use", steps=[4], description="g")]
    matches, fps, misses = match_findings(pred, gold, "strict")
    assert not matches and not fps and len(misses) == 1


def test_family_mismatch_is_fp_and_fn():
    pred = [_f("redundant_action", [4, 5])]
    gold = [LabelFinding(family="incorrect_tool_use", steps=[4, 5], description="g")]
    matches, fps, misses = match_findings(pred, gold, "strict")
    assert not matches and fps and misses


def test_relaxed_center_slack():
    pred = [_f("instruction_violation", [10])]
    gold = [LabelFinding(family="instruction_violation", steps=[12, 13], description="g")]
    matches, fps, misses = match_findings(pred, gold, "relaxed", center_slack=3)
    assert len(matches) == 1
