from traceaudit.pack import (
    call_outline,
    distill_instructions,
    pack_char_budget,
    pack_judge,
    pack_trace,
)
from traceaudit.schemas import (
    CanonicalStep,
    CanonicalTrace,
    CandidateFinding,
    DerivedCheck,
    Evidence,
    ToolSpec,
)


def _trace(n_extra: int = 0) -> CanonicalTrace:
    steps = [
        CanonicalStep(index=0, kind="message", role="user", content="Oslo?"),
        CanonicalStep(
            index=1, kind="tool_call", name="get_weather", arguments={"city": "Oslo"}
        ),
        CanonicalStep(
            index=2, kind="tool_result", name="get_weather", content='{"temp_c": 12}'
        ),
        CanonicalStep(index=3, kind="message", role="assistant", content="12C"),
    ]
    for i in range(n_extra):
        steps.append(
            CanonicalStep(
                index=4 + i,
                kind="message",
                role="assistant",
                content=("x" * 5000),
            )
        )
    return CanonicalTrace(
        trace_id="wx",
        source_format="canonical",
        task="Temperature?",
        instructions="Use the tool.",
        tools=[ToolSpec(name="get_weather")],
        steps=steps,
    )


def test_pack_includes_task_tools_and_steps():
    blob = pack_trace(_trace(), max_chars=20000, instruction_chars=4000)
    assert "Temperature?" in blob
    assert "get_weather" in blob
    assert "[step 2]" in blob
    assert "TOOL-CALL OUTLINE" in blob
    assert "step 1 get_weather" in call_outline(_trace())


def test_pack_clips_when_over_budget():
    blob = pack_trace(_trace(n_extra=20), max_chars=8000, instruction_chars=500)
    assert len(blob) <= 9000
    assert "[step 0]" in blob
    assert "get_weather" in blob


def test_pack_budget_leaves_room_for_uncapped_output():
    chars = pack_char_budget(1.50, 0.0, copies=2)
    assert 20_000 <= chars <= 280_000
    after = pack_char_budget(1.50, 1.10, copies=1)
    assert after < chars or after <= 280_000
    assert after >= 8_000


def test_distill_keeps_requirements():
    blob = (
        "Welcome to the playbook.\n"
        "You must authenticate before refunds.\n"
        "You have to locate the user id via email.\n"
        "A long style paragraph about tone and greetings.\n"
        "Never call refund_order without a user_id.\n"
    ) * 40
    out = distill_instructions(blob, 500)
    assert "must authenticate" in out.lower()
    assert "have to locate" in out.lower()
    assert "never call refund_order" in out.lower()
    assert len(out) <= 520


def test_judge_pack_numbers_proposals():
    proposals = [
        CandidateFinding(
            family="evidence_contradiction",
            description="wrong temp",
            steps=[2, 3],
            evidence=[Evidence(step=2, quote="temp_c")],
            auditor="agent_a",
        )
    ]
    checks = [
        DerivedCheck(
            check_id="C1",
            family="evidence_contradiction",
            description="match the tool",
            source="agent_a",
        )
    ]
    blob = pack_judge(
        _trace(), checks, proposals, max_chars=20000, instruction_chars=4000
    )
    assert "NUMBERED PROPOSALS" in blob
    assert "wrong temp" in blob
    assert "NEIGHBORHOOD" in blob
    assert "error_class" in blob
    assert "contradicts_tool_output" in blob
    assert "confidence" in blob
    tight = pack_judge(
        _trace(), checks, proposals, max_chars=1200, instruction_chars=200
    )
    assert len(tight) <= 1220


def test_judge_pack_keeps_outline_on_tight_budget():
    proposals = [
        CandidateFinding(
            family="incorrect_tool_use",
            description="schema miss",
            steps=[1],
            evidence=[Evidence(step=1, quote="get_weather")],
            auditor="agent_a",
        )
    ]
    checks = [
        DerivedCheck(
            check_id="T1",
            family="incorrect_tool_use",
            description="declared tools only",
            source="agent_a",
        )
    ]
    blob = pack_judge(
        _trace(n_extra=30),
        checks,
        proposals,
        max_chars=2500,
        instruction_chars=200,
    )
    assert "TOOL-CALL OUTLINE" in blob
    assert "get_weather" in blob
    assert "step 1 get_weather" in blob
