"""Guards against re-introducing detectors that only work on the dev traces.

Every trace here is written with wording the labelled set does not contain. If a
detector needs a phrase it learned from `data/dev`, these fail.
"""

import json

from traceaudit.config import Config
from traceaudit.eval.metrics import evaluate
from traceaudit.llm import NullClient
from traceaudit.pipeline import analyze_trace
from traceaudit.report import write_run_artifacts
from traceaudit.schemas import (
    CandidateFinding,
    CanonicalStep,
    CanonicalTrace,
    Evidence,
    ToolSpec,
)
from traceaudit.structural import (
    analyze,
    names_unsupplied_argument,
    objective_findings,
)
from traceaudit.verify import cited_action_only_failed, quick_check


def _repeat_after_success(narration: str) -> CanonicalTrace:
    """A success, something said, then the identical call again."""
    return CanonicalTrace(
        trace_id="repeat",
        source_format="canonical",
        task="Convert 90 degrees.",
        instructions="Use the converter.",
        tools=[ToolSpec(name="to_radians", declared=True, parameters={"type": "object"})],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="90 degrees please"),
            CanonicalStep(index=1, kind="tool_call", name="to_radians", arguments={"deg": 90}),
            CanonicalStep(index=2, kind="tool_result", name="to_radians", content="1.5707963"),
            CanonicalStep(index=3, kind="message", role="assistant", content=narration),
            CanonicalStep(index=4, kind="tool_call", name="to_radians", arguments={"deg": 90}),
            CanonicalStep(index=5, kind="tool_result", name="to_radians", content="1.5707963"),
        ],
    )


def test_phantom_failure_needs_no_apology_phrasing():
    """Detection is structural, so novel wording still fires."""
    trace = _repeat_after_success("Hmm, the converter seems to have produced nothing usable.")
    kinds = {f.kind for f in analyze(trace).flags}
    assert "phantom_failure" in kinds


def test_phantom_failure_fires_on_wording_never_seen_in_dev():
    trace = _repeat_after_success("Recalibrating: the previous invocation yielded no usable value.")
    kinds = {f.kind for f in analyze(trace).flags}
    assert "phantom_failure" in kinds


def test_no_phantom_failure_without_a_repeat():
    """An apology on its own is not a contradiction."""
    trace = _repeat_after_success("My apologies, let me correct the usage.")
    trace.steps = trace.steps[:4]
    kinds = {f.kind for f in analyze(trace).flags}
    assert "phantom_failure" not in kinds


def test_empty_required_argument_is_caught_by_the_schema_not_a_name_list():
    """Field naming is irrelevant; only the schema's `required` matters."""
    trace = CanonicalTrace(
        trace_id="empty-arg",
        source_format="canonical",
        instructions="Fill in the report.",
        tools=[
            ToolSpec(
                name="submit_report",
                declared=True,
                parameters={
                    "type": "object",
                    "properties": {"zorblatt": {"type": "string"}},
                    "required": ["zorblatt"],
                },
            )
        ],
        steps=[
            CanonicalStep(index=0, kind="tool_call", name="submit_report", arguments={"zorblatt": ""}),
            CanonicalStep(index=1, kind="tool_result", name="submit_report", content='{"ok": true}'),
        ],
    )
    flags = analyze(trace)
    schema_flags = [f for f in flags.flags if f.kind == "schema_violation"]
    assert schema_flags
    assert "empty" in schema_flags[0].detail
    assert any(c.family == "incorrect_tool_use" for c in objective_findings(trace, flags))


def _call(args: dict) -> CanonicalStep:
    return CanonicalStep(index=0, kind="tool_call", name="t", arguments=args)


def test_runtime_naming_an_unsupplied_argument_is_structural():
    assert names_unsupplied_argument(_call({"action": "react"}), '{"error": "messageId required"}')
    assert names_unsupplied_argument(_call({"a": 1}), "missing field user_name")


def test_runtime_echoing_our_own_argument_is_a_resource_error():
    assert not names_unsupplied_argument(
        _call({"order_id": "ORD-99"}), '{"error": "order_id ORD-99 unavailable"}'
    )
    assert not names_unsupplied_argument(
        _call({"path": "/srv/data-set/file.txt"}),
        "ENOENT: no such file or directory /srv/data-set/file.txt",
    )


def test_prose_free_error_still_needs_no_vocabulary():
    """A runtime that only returns a field name is handled without prose cues."""
    trace = CanonicalTrace(
        trace_id="terse",
        source_format="canonical",
        instructions="Send it.",
        tools=[ToolSpec(name="send", declared=True)],
        steps=[
            CanonicalStep(index=0, kind="tool_call", name="send", arguments={"body": "hi"}),
            CanonicalStep(
                index=1,
                kind="tool_result",
                name="send",
                content='{"error": "recipientId"}',
                is_error=True,
            ),
        ],
    )
    kinds = {f.kind for f in analyze(trace).flags}
    assert "call_formation_error" in kinds


# --- precision holes that used to let LLM findings through ----------------


def _weather_trace() -> CanonicalTrace:
    return CanonicalTrace(
        trace_id="wx",
        source_format="canonical",
        task="Temperature in Oslo?",
        instructions="Answer only from tool results.",
        tools=[ToolSpec(name="get_weather", declared=True, parameters={"type": "object"})],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="Oslo temp?"),
            CanonicalStep(index=1, kind="tool_call", name="get_weather", arguments={"city": "Oslo"}),
            CanonicalStep(index=2, kind="tool_result", name="get_weather", content='{"temp_c": 12}'),
            CanonicalStep(index=3, kind="message", role="assistant", content="It is 28C in Oslo."),
        ],
    )


def test_incorrect_tool_use_needs_proof_not_just_a_quote():
    """A verified quote alone used to be enough to confirm this family."""
    cand = CandidateFinding(
        family="incorrect_tool_use",
        description="the weather call was somehow wrong",
        steps=[1],
        evidence=[Evidence(step=1, quote='"city": "Oslo"')],
        source="llm",
    )
    verdict, reason = quick_check(_weather_trace(), cand)
    assert verdict == "insufficient_evidence"
    assert "schema" in reason or "undeclared" in reason or "formation" in reason


def test_contradiction_needs_both_sides_quoted():
    """Citing only the assistant's claim does not prove it contradicts anything."""
    cand = CandidateFinding(
        family="evidence_contradiction",
        description="claimed 28C",
        steps=[3],
        evidence=[Evidence(step=3, quote="It is 28C in Oslo.")],
        source="llm",
    )
    verdict, reason = quick_check(_weather_trace(), cand)
    assert verdict == "insufficient_evidence"
    assert "tool-result" in reason

    both = CandidateFinding(
        family="evidence_contradiction",
        description="claimed 28C but the tool said 12",
        steps=[2, 3],
        evidence=[
            Evidence(step=2, quote='"temp_c": 12'),
            Evidence(step=3, quote="It is 28C in Oslo."),
        ],
        source="llm",
    )
    assert quick_check(_weather_trace(), both)[0] == "confirmed"


def test_instruction_rule_must_actually_appear_in_the_instructions():
    cand = CandidateFinding(
        family="instruction_violation",
        description="broke a rule the operator never wrote",
        steps=[1],
        evidence=[Evidence(step=1, quote='"city": "Oslo"')],
        source="llm",
        meta={"rule_ref": "You must always call get_weather twice before answering."},
    )
    verdict, reason = quick_check(_weather_trace(), cand)
    assert verdict == "insufficient_evidence"
    assert "not found in task/instructions" in reason


def test_citing_only_the_error_result_still_counts_as_a_failed_attempt():
    trace = CanonicalTrace(
        trace_id="failed",
        source_format="canonical",
        instructions="Read the file at the exact path.",
        tools=[ToolSpec(name="read", declared=True)],
        steps=[
            CanonicalStep(index=0, kind="tool_call", name="read", arguments={"path": "/nope"}),
            CanonicalStep(index=1, kind="tool_result", name="read", content="ENOENT", is_error=True),
        ],
    )
    cand = CandidateFinding(family="instruction_violation", description="x", steps=[1], source="llm")
    assert cited_action_only_failed(trace, cand)


def test_duplicate_trace_rows_are_not_counted_twice(tmp_path):
    """A re-run appended to the same results file must not double the FP count."""
    trace = CanonicalTrace(
        trace_id="dupe",
        source_format="canonical",
        instructions="Send the message.",
        tools=[
            ToolSpec(
                name="send",
                declared=True,
                parameters={
                    "type": "object",
                    "properties": {"to": {"type": "string"}},
                    "required": ["to"],
                },
            )
        ],
        steps=[
            CanonicalStep(index=0, kind="tool_call", name="send", arguments={}),
            CanonicalStep(index=1, kind="tool_result", name="send", content='{"ok": true}'),
        ],
    )
    result = analyze_trace(trace, Config(), NullClient())
    assert [f for f in result.findings if f.status == "confirmed"], "need a finding to double"

    out = write_run_artifacts(tmp_path / "run", [result, result])
    assert len([ln for ln in (out / "findings.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]) == 2

    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "dupe.json").write_text(
        json.dumps({"trace_id": "dupe", "clean": True, "findings": []}), encoding="utf-8"
    )
    report = evaluate(out / "findings.jsonl", labels, semantic_mode="deterministic")
    assert report["n_evaluated"] == 1
    # One underlying mistake, counted once, even though the row appears twice.
    assert report["overall"]["strict"]["fp"] == 1
