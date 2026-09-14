import threading
from concurrent.futures import ThreadPoolExecutor

from traceaudit.config import Config
from traceaudit.llm import ScriptedClient
from traceaudit.pipeline import analyze_trace
from traceaudit.schemas import CanonicalStep, CanonicalTrace, ToolSpec


def _weather_trace():
    return CanonicalTrace(
        trace_id="wx",
        source_format="canonical",
        task="Temperature in Oslo?",
        instructions="Answer only from tool results. Do not invent numbers.",
        tools=[ToolSpec(name="get_weather", parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="Oslo temp?"),
            CanonicalStep(index=1, kind="tool_call", name="get_weather", arguments={"city": "Oslo"}),
            CanonicalStep(index=2, kind="tool_result", name="get_weather", content='{"temp_c": 12}'),
            CanonicalStep(index=3, kind="message", role="assistant", content="It is 28C in Oslo."),
        ],
    )


def _contradiction_finding():
    return {
        "check_id": "C1",
        "decision": "violated",
        "family": "evidence_contradiction",
        "steps": [2, 3],
        "description": "Said 28C but tool returned 12",
        "explanation": "The 12C result was already visible",
        "evidence": [{"step": 2, "quote": "temp_c\": 12"}, {"step": 3, "quote": "It is 28C in Oslo."}],
        "rule_ref": "Answer only from tool results",
    }


def test_parallel_agents_then_judge_confirms_contradiction():
    seen: list[str] = []

    def handler(stage, _messages):
        seen.append(stage)
        if stage == "derive":
            return {
                "checks": [
                    {
                        "check_id": "C1",
                        "family": "evidence_contradiction",
                        "description": "Final weather claim must match get_weather",
                        "condition": "stated temperature differs from the tool result",
                        "justified_when": "the tool was not called",
                        "source_refs": ["Answer only from tool results"],
                    }
                ]
            }
        if stage in {"agent_a", "agent_b"}:
            return {"findings": [_contradiction_finding()]}
        if stage == "judge":
            return {
                "verdicts": [
                    {"index": 0, "verdict": "confirmed", "reason": "quote-backed contradiction"},
                    {"index": 1, "verdict": "confirmed", "reason": "quote-backed contradiction"},
                ]
            }
        return {"verdicts": []}

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    conf = [f for f in result.findings if f.status == "confirmed" and f.family == "evidence_contradiction"]
    assert conf
    assert result.checks
    assert conf[0].evidence
    assert "agent_a" in seen and "agent_b" in seen
    assert "judge" in seen
    assert result.status == "ok"


def test_unverified_quotes_become_insufficient():
    def handler(stage, _messages):
        if stage == "derive":
            return {"checks": [{"check_id": "C1", "family": "other", "description": "x", "condition": "y"}]}
        if stage in {"agent_a", "agent_b"}:
            return {
                "findings": [
                    {
                        "check_id": "C1",
                        "decision": "violated",
                        "family": "other",
                        "steps": [3],
                        "description": "something invented",
                        "evidence": [{"step": 3, "quote": "this quote is not in the step at all"}],
                    }
                ]
            }
        return {
            "verdicts": [
                {"index": 0, "verdict": "confirmed", "reason": "should be demoted"},
                {"index": 1, "verdict": "confirmed", "reason": "should be demoted"},
            ]
        }

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    assert not [f for f in result.findings if f.status == "confirmed" and f.source == "llm"]
    assert any(f.status == "insufficient_evidence" for f in result.findings)


def test_additional_without_decision_is_dropped():
    def handler(stage, _messages):
        if stage == "derive":
            return {"checks": [{"check_id": "C1", "family": "other", "description": "x", "condition": "y"}]}
        if stage in {"agent_a", "agent_b"}:
            return {
                "findings": [
                    {
                        "family": "other",
                        "steps": [3],
                        "description": "sneaky extra",
                        "evidence": [{"step": 3, "quote": "It is 28C in Oslo."}],
                    }
                ]
            }
        return {"verdicts": [{"index": 0, "verdict": "confirmed", "reason": "should not run"}]}

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    assert not [f for f in result.findings if f.status == "confirmed" and "sneaky" in f.description]


def test_quick_check_demotes_characterisation_success():
    def handler(stage, _messages):
        if stage == "derive":
            return {"checks": [{"check_id": "C1", "family": "unsupported_success", "description": "x", "condition": "y"}]}
        if stage in {"agent_a", "agent_b"}:
            return {
                "findings": [
                    {
                        "check_id": "C1",
                        "decision": "violated",
                        "family": "unsupported_success",
                        "steps": [3],
                        "description": "This is a comprehensive strategy overview",
                        "explanation": "characterisation only",
                        "evidence": [{"step": 3, "quote": "It is 28C in Oslo."}],
                    }
                ]
            }
        return {
            "verdicts": [
                {"index": 0, "verdict": "confirmed", "reason": "should be demoted"},
                {"index": 1, "verdict": "confirmed", "reason": "should be demoted"},
            ]
        }

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    assert not [f for f in result.findings if f.status == "confirmed" and f.family == "unsupported_success"]


def test_quick_check_demotes_inefficiency_without_alternative():
    def handler(stage, _messages):
        if stage == "derive":
            return {"checks": [{"check_id": "C1", "family": "redundant_action", "description": "x", "condition": "y"}]}
        if stage in {"agent_a", "agent_b"}:
            return {
                "findings": [
                    {
                        "check_id": "C1",
                        "decision": "violated",
                        "family": "redundant_action",
                        "steps": [1],
                        "description": "called get_weather twice",
                        "explanation": "repeat",
                        "evidence": [{"step": 1, "quote": "\"city\": \"Oslo\""}],
                        "alternative": "",
                    }
                ]
            }
        return {
            "verdicts": [
                {"index": 0, "verdict": "confirmed", "reason": "should be demoted"},
                {"index": 1, "verdict": "confirmed", "reason": "should be demoted"},
            ]
        }

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    assert not [f for f in result.findings if f.status == "confirmed" and f.family == "redundant_action"]
    assert any(f.status == "insufficient_evidence" for f in result.findings)


def _gift_trace():
    return CanonicalTrace(
        trace_id="gc",
        source_format="canonical",
        task="Swap pending items and pay the difference with a gift card.",
        instructions=(
            "If the user provides a gift card, it must have enough balance to cover the price difference. "
            "Call get_user_details before charging."
        ),
        tools=[
            ToolSpec(
                name="get_user_details",
                declared=True,
                parameters={"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]},
            ),
            ToolSpec(
                name="modify_pending_order_items",
                declared=True,
                parameters={"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
            ),
        ],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="Use gift card GC-9 for the difference."),
            CanonicalStep(index=1, kind="tool_call", name="modify_pending_order_items", arguments={"order_id": "A1", "payment": "GC-9"}),
            CanonicalStep(index=2, kind="tool_result", name="modify_pending_order_items", content='{"status":"ok"}'),
        ],
    )


def test_scripted_gift_card_instruction_violation():
    def handler(stage, _messages):
        if stage == "derive":
            return {"checks": []}
        if stage in {"agent_a", "agent_b"}:
            return {
                "findings": [
                    {
                        "check_id": "OBL-1",
                        "decision": "violated",
                        "family": "instruction_violation",
                        "steps": [1, 2],
                        "description": "Gift card used without verifying balance",
                        "explanation": "get_user_details was never called",
                        "evidence": [
                            {"step": 1, "quote": "\"order_id\": \"A1\""},
                            {"step": 0, "quote": "Use gift card GC-9 for the difference."},
                        ],
                        "rule_ref": "If the user provides a gift card, it must have enough balance to cover the price difference.",
                    }
                ]
            }
        return {
            "verdicts": [
                {"index": 0, "verdict": "confirmed", "reason": "required balance check skipped"},
                {"index": 1, "verdict": "confirmed", "reason": "required balance check skipped"},
            ]
        }

    result = analyze_trace(_gift_trace(), Config(), ScriptedClient(handler))
    conf = [f for f in result.findings if f.status == "confirmed" and f.family == "instruction_violation"]
    assert conf


def test_unknown_tool_kept_when_judge_misses():
    trace = CanonicalTrace(
        trace_id="unk",
        source_format="canonical",
        tools=[
            ToolSpec(
                name="search",
                declared=True,
                parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
            )
        ],
        steps=[
            CanonicalStep(index=0, kind="tool_call", name="explode", arguments={"q": "x"}),
            CanonicalStep(index=1, kind="tool_result", name="explode", content='{"ok": true}'),
        ],
    )

    def handler(stage, _messages):
        if stage == "derive":
            return {"checks": []}
        if stage in {"agent_a", "agent_b"}:
            return {"findings": []}
        return {"verdicts": []}

    result = analyze_trace(trace, Config(), ScriptedClient(handler))
    assert result.status == "ok"
    assert any(f.status == "confirmed" and f.family == "incorrect_tool_use" for f in result.findings)


def test_llm_path_does_not_skip_agents_when_key_present():
    stages: list[str] = []

    def handler(stage, _messages):
        stages.append(stage)
        if stage == "derive":
            return {"checks": []}
        if stage in {"agent_a", "agent_b"}:
            return {"findings": []}
        return {"verdicts": []}

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    assert result.status == "ok"
    assert "derive" in stages
    assert "agent_a" in stages
    assert "agent_b" in stages


def test_agents_run_in_parallel():
    """A and B must overlap in time (ThreadPoolExecutor), not run sequentially."""
    barrier = threading.Barrier(2, timeout=3)
    overlapped: list[str] = []
    stages: list[str] = []

    def handler(stage, _messages):
        stages.append(stage)
        if stage == "derive":
            return {"checks": [{"check_id": "C1", "family": "other", "description": "x", "condition": "y"}]}
        if stage in {"agent_a", "agent_b"}:
            barrier.wait()
            overlapped.append(stage)
            return {"findings": []}
        return {"verdicts": []}

    client = ScriptedClient(handler)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pool.submit(lambda: client.complete([], models=["m"], temperature=0, max_tokens=10, stage="probe_a"))
        pool.submit(lambda: client.complete([], models=["m"], temperature=0, max_tokens=10, stage="probe_b"))

    analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    assert "agent_a" in stages and "agent_b" in stages
    assert set(overlapped) == {"agent_a", "agent_b"}


def test_judge_still_runs_when_spend_exceeds_target_mean():
    """$1.50 is the quality target, not an abort. A ~$1.70 trace must still be judged."""
    seen: list[str] = []

    def handler(stage, _messages):
        seen.append(stage)
        if stage == "derive":
            return {
                "checks": [
                    {
                        "check_id": "C1",
                        "family": "evidence_contradiction",
                        "description": "weather claim must match tool",
                        "condition": "stated temperature differs",
                    }
                ]
            }
        if stage in {"agent_a", "agent_b"}:
            return {"findings": [_contradiction_finding()]}
        if stage == "judge":
            return {
                "verdicts": [
                    {"index": 0, "verdict": "confirmed", "reason": "quote-backed"},
                    {"index": 1, "verdict": "confirmed", "reason": "quote-backed"},
                ]
            }
        return {"verdicts": []}

    cfg = Config()
    assert cfg.target_mean_usd == 1.5
    assert cfg.cost_cap_usd == 3.0
    client = ScriptedClient(
        handler,
        costs={"derive": 0.21, "agent_a": 0.70, "agent_b": 0.79, "judge": 0.31},
    )
    result = analyze_trace(_weather_trace(), cfg, client)
    assert "judge" in seen
    assert result.status == "ok"
    assert result.cost_usd > cfg.target_mean_usd
    assert result.cost_usd < cfg.cost_cap_usd
    assert [f for f in result.findings if f.status == "confirmed" and f.family == "evidence_contradiction"]


def test_judge_skipped_only_after_hard_cap():
    seen: list[str] = []

    def handler(stage, _messages):
        seen.append(stage)
        if stage == "derive":
            return {"checks": [{"check_id": "C1", "family": "other", "description": "x", "condition": "y"}]}
        if stage in {"agent_a", "agent_b"}:
            return {"findings": [_contradiction_finding()]}
        return {
            "verdicts": [
                {"index": 0, "verdict": "confirmed", "reason": "should not run"},
                {"index": 1, "verdict": "confirmed", "reason": "should not run"},
            ]
        }

    client = ScriptedClient(handler, costs={"derive": 1.0, "agent_a": 1.1, "agent_b": 1.1})
    result = analyze_trace(_weather_trace(), Config(), client)
    assert "judge" not in seen
    assert "agent_a" in seen and "agent_b" in seen
    assert not [f for f in result.findings if f.status == "confirmed" and f.source == "llm"]
    assert any(f.status == "insufficient_evidence" for f in result.findings)


def test_both_agents_receive_the_full_check_list():
    """Agents are complementary briefs, not obligation/residual lanes."""
    blobs: dict[str, str] = {}

    def handler(stage, messages):
        user = messages[-1]["content"] if messages else ""
        if stage == "derive":
            return {
                "checks": [
                    {
                        "check_id": "C-instr",
                        "family": "instruction_violation",
                        "description": "honor the no-invent rule",
                        "condition": "invented a number",
                        "justified_when": "the tool was not called",
                        "source_refs": ["Do not invent numbers"],
                    },
                    {
                        "check_id": "C-redun",
                        "family": "redundant_action",
                        "description": "do not repeat a successful call",
                        "condition": "identical successful repeat",
                        "justified_when": "retry after a transient error",
                    },
                ]
            }
        if stage in {"agent_a", "agent_b"}:
            blobs[stage] = user
            return {"findings": []}
        return {"verdicts": []}

    analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    assert set(blobs) == {"agent_a", "agent_b"}
    for blob in blobs.values():
        assert "C-instr" in blob
        assert "C-redun" in blob


def test_schema_loss_does_not_auto_confirm(tmp_path=None):
    from traceaudit.llm import NullClient
    from traceaudit.structural import analyze, objective_findings

    trace = CanonicalTrace(
        trace_id="loss",
        source_format="canonical",
        tools=[
            ToolSpec(
                name="dispatch",
                declared=True,
                parameters={"type": "object", "properties": {"strategyRef": {"type": "string"}}, "required": ["strategyRef"]},
            )
        ],
        steps=[
            CanonicalStep(index=0, kind="tool_call", name="dispatch", arguments={"strategyRef": ["general_strategy"]}),
            CanonicalStep(index=1, kind="tool_result", name="dispatch", content='{"ok": true}'),
        ],
        meta={"loss_events": [{"reason": "tool schema not captured", "detail": "parameters missing"}]},
    )
    flags = analyze(trace)
    assert not [f for f in flags.flags if f.kind == "schema_violation"]
    cands = objective_findings(trace, flags)
    result = analyze_trace(trace, Config(), NullClient())
    assert not [f for f in result.findings if f.status == "confirmed" and f.family == "incorrect_tool_use"]
    assert cands == [] or all(c.family != "incorrect_tool_use" or "schema" not in (c.check_id or "") for c in cands)
