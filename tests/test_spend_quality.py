"""Spend-on-quality: fuller evidence packs and a budgeted critique pass."""

from traceaudit.config import Config
from traceaudit.critique import should_critique
from traceaudit.evidence import evidence_index
from traceaudit.llm import ScriptedClient, usage_tokens
from traceaudit.pipeline import analyze_trace
from traceaudit.retrieve import neighborhood_indices
from traceaudit.schemas import CanonicalStep, CanonicalTrace, DerivedCheck, ToolSpec


def _long_result_trace() -> CanonicalTrace:
    payload = "TEMPERATURE_C=12.5 " + ("alpha " * 400)
    return CanonicalTrace(
        trace_id="long",
        source_format="canonical",
        task="Report the temperature.",
        instructions="Answer only from tool results. Do not invent numbers.",
        tools=[
            ToolSpec(
                name="get_weather",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            )
        ],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="Oslo?"),
            CanonicalStep(index=1, kind="tool_call", name="get_weather", arguments={"city": "Oslo"}),
            CanonicalStep(index=2, kind="tool_result", name="get_weather", content=payload),
            CanonicalStep(index=3, kind="message", role="assistant", content="It is 12.5C."),
        ],
    )


def test_evidence_index_uses_budget_instead_of_tiny_clips():
    trace = _long_result_trace()
    packed = evidence_index(trace, 160000)
    assert "TEMPERATURE_C=12.5" in packed
    # The old first tier clipped content at 1400 chars. A 160k budget must keep more.
    assert packed.count("alpha") > 200
    assert "[...index truncated...]" not in packed


def test_neighborhood_includes_paired_result():
    trace = _long_result_trace()
    around = neighborhood_indices(trace, [1], radius=1)
    assert 0 in around
    assert 1 in around
    assert 2 in around


def test_usage_tokens_folds_reasoning_when_omitted_from_completion():
    inp, out = usage_tokens(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 800},
        }
    )
    assert inp == 100
    assert out == 800


def test_usage_tokens_does_not_double_count_included_reasoning():
    inp, out = usage_tokens(
        {
            "prompt_tokens": 100,
            "completion_tokens": 820,
            "completion_tokens_details": {"reasoning_tokens": 800},
        }
    )
    assert inp == 100
    assert out == 820


def test_critique_skipped_once_spend_hits_target():
    cfg = Config()
    assert not should_critique(
        cfg,
        spend=1.55,
        checks=[DerivedCheck(check_id="C1", family="other", description="x", condition="y")],
        findings=[],
        proposals=[],
    )


def test_critique_runs_under_target_and_gates_new_findings():
    """Critique can raise a missed contradiction; unverified quotes still abstain."""

    def handler(stage, _messages):
        if stage == "derive":
            return {
                "checks": [
                    {
                        "check_id": "C1",
                        "family": "evidence_contradiction",
                        "description": "weather claim must match tool",
                        "condition": "stated temperature differs",
                        "justified_when": "the tool was not called",
                        "source_refs": ["Answer only from tool results"],
                    }
                ]
            }
        if stage in {"agent_a", "agent_b"}:
            return {"findings": []}
        if stage == "critique":
            return {
                "findings": [
                    {
                        "check_id": "C1",
                        "decision": "violated",
                        "family": "evidence_contradiction",
                        "steps": [2, 3],
                        "description": "Said 28C but tool returned 12.5",
                        "explanation": "The 12.5 result was already visible",
                        "evidence": [
                            {"step": 2, "quote": "TEMPERATURE_C=12.5"},
                            {"step": 3, "quote": "It is 28C."},
                        ],
                        "rule_ref": "Answer only from tool results",
                    }
                ]
            }
        return {"findings": []}

    result = analyze_trace(_long_result_trace(), Config(), ScriptedClient(handler))
    assert any(c.stage == "critique" for c in result.calls)
    # Quote "It is 28C." is not in the assistant step, so the gate must abstain.
    assert not [
        f
        for f in result.findings
        if f.status == "confirmed" and f.family == "evidence_contradiction" and f.source == "llm"
    ]
    assert any(f.status == "insufficient_evidence" for f in result.findings)


def test_critique_confirms_value_level_contradiction_when_quotes_hold():
    trace = _long_result_trace()
    trace.steps[3].content = "It is 28C in Oslo."

    def handler(stage, _messages):
        if stage == "derive":
            return {
                "checks": [
                    {
                        "check_id": "C1",
                        "family": "evidence_contradiction",
                        "description": "weather claim must match tool",
                        "condition": "stated temperature differs",
                        "source_refs": ["Answer only from tool results"],
                    }
                ]
            }
        if stage in {"agent_a", "agent_b"}:
            return {"findings": []}
        if stage == "critique":
            return {
                "findings": [
                    {
                        "check_id": "C1",
                        "decision": "violated",
                        "family": "evidence_contradiction",
                        "steps": [2, 3],
                        "description": "Said 28C but tool returned 12.5",
                        "explanation": "The 12.5 result was already visible",
                        "evidence": [
                            {"step": 2, "quote": "TEMPERATURE_C=12.5"},
                            {"step": 3, "quote": "It is 28C in Oslo."},
                        ],
                        "rule_ref": "Answer only from tool results",
                    }
                ]
            }
        return {"findings": []}

    result = analyze_trace(trace, Config(), ScriptedClient(handler))
    assert [
        f
        for f in result.findings
        if f.status == "confirmed" and f.family == "evidence_contradiction"
    ]


def test_critique_disabled_makes_no_call():
    seen: list[str] = []

    def handler(stage, _messages):
        seen.append(stage)
        if stage == "derive":
            return {"checks": [{"check_id": "C1", "family": "other", "description": "x", "condition": "y"}]}
        return {"findings": []}

    cfg = Config(critique_enabled=False)
    analyze_trace(_long_result_trace(), cfg, ScriptedClient(handler))
    assert "critique" not in seen
    assert "derive" in seen
