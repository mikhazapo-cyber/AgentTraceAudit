from traceaudit.quotes import quote_supported
from traceaudit.retrieve import retrieve_step_indices
from traceaudit.schemas import (
    CandidateFinding,
    CanonicalStep,
    CanonicalTrace,
    DerivedCheck,
    Evidence,
    ToolSpec,
)
from traceaudit.verify import inspect, quick_check


def _trace():
    return CanonicalTrace(
        trace_id="t",
        source_format="canonical",
        task="Refund ORD-1",
        instructions="Before any refund you must authenticate the user.",
        tools=[ToolSpec(name="refund_order", parameters={"type": "object", "required": ["order_id"]})],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="Refund ORD-1"),
            CanonicalStep(index=1, kind="tool_call", name="refund_order", arguments={"order_id": "ORD-1"}),
            CanonicalStep(index=2, kind="tool_result", name="refund_order", content='{"refunded": true}'),
            CanonicalStep(index=3, kind="message", role="assistant", content="Refund submitted for ORD-1."),
        ],
    )


def test_empty_quotes_fail_inspect():
    cand = CandidateFinding(
        family="instruction_violation",
        description="skipped auth",
        steps=[1],
        evidence=[],
        source="llm",
        meta={"rule_ref": "Before any refund you must authenticate the user."},
    )
    ver = inspect(_trace(), cand)
    assert not ver.quotes_ok
    assert not ver.passed


def test_invented_quote_fails():
    ev = Evidence(step=3, quote="this text is not present in the assistant message")
    assert not quote_supported(_trace(), ev)


def test_instruction_gate_requires_excerpt():
    cand = CandidateFinding(
        family="instruction_violation",
        description="skipped auth",
        steps=[1],
        evidence=[Evidence(step=1, quote="refund_order")],
        source="llm",
        meta={"rule_ref": "a rule that does not appear anywhere"},
    )
    ver = inspect(_trace(), cand)
    assert not ver.family_gate


def test_quick_check_requires_quote_in_step():
    cand = CandidateFinding(
        family="other",
        description="invented",
        steps=[3],
        evidence=[Evidence(step=3, quote="this text is not present in the assistant message")],
        source="llm",
    )
    verdict, reason = quick_check(_trace(), cand)
    assert verdict == "insufficient_evidence"
    assert "quote" in reason


def test_quick_check_keeps_quoted_claim():
    cand = CandidateFinding(
        family="unsupported_success",
        description="claimed the refund was submitted with no tool result showing it",
        steps=[3],
        evidence=[Evidence(step=3, quote="Refund submitted for ORD-1.")],
        source="llm",
    )
    verdict, _reason = quick_check(_trace(), cand)
    assert verdict == "confirmed"


def test_quick_check_will_not_confirm_the_other_catch_all():
    cand = CandidateFinding(
        family="other",
        description="something off about this trace",
        steps=[3],
        evidence=[Evidence(step=3, quote="Refund submitted for ORD-1.")],
        source="llm",
    )
    verdict, reason = quick_check(_trace(), cand)
    assert verdict == "insufficient_evidence"
    assert "other" in reason


def test_retrieve_prefers_named_tools():
    trace = _trace()
    checks = [
        DerivedCheck(
            check_id="C1",
            family="instruction_violation",
            description="refund requires authentication",
            condition="refund_order called before authenticate",
            source_refs=["Before any refund you must authenticate the user."],
        )
    ]
    idxs = retrieve_step_indices(trace, checks)
    assert 1 in idxs
