"""Spec-content, ingest, and quality-lever regressions.

None of these detectors are tuned on the labelled development traces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traceaudit.adapter_profiles import detect
from traceaudit.config import Config
from traceaudit.llm import NullClient, ScriptedClient
from traceaudit.normalize import load_dataset, load_trace
from traceaudit.normalize.input import detect_source, looks_like_trace, read_prompt_value
from traceaudit.normalize.tavii import _parse_call_arguments
from traceaudit.obligations import extend_from_derived
from traceaudit.pipeline import analyze_trace
from traceaudit.quotes import supporting_instruction, unique_evidence
from traceaudit.schemas import (
    CanonicalStep,
    CanonicalTrace,
    DerivedCheck,
    Evidence,
    Finding,
    ToolSpec,
)
from traceaudit.structural import analyze, format_steps, is_replay

ROOT = Path(__file__).resolve().parents[1]


def test_openai_chat_example_is_not_langchain():
    path = ROOT / "examples" / "openai_chat.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    ranked = detect(doc, str(path), top_k=3)
    assert ranked[0][0] == "openai-chat"
    assert detect_source(path) == "openai-chat"


def test_read_prompt_value_strips_utf8_bom(tmp_path):
    path = tmp_path / "policy.txt"
    path.write_bytes(b"\xef\xbb\xbfNever skip auth.")
    assert read_prompt_value(f"@{path}") == "Never skip auth."


def test_single_non_trace_file_is_rejected(tmp_path):
    path = tmp_path / "notes.json"
    path.write_text(json.dumps({"todo": "buy milk"}), encoding="utf-8")
    assert not looks_like_trace(path)
    with pytest.raises(FileNotFoundError, match="no recognizable traces"):
        load_dataset(path)


def test_canonical_file_looks_like_a_trace(tmp_path):
    path = tmp_path / "canon.json"
    path.write_text(
        json.dumps(
            {
                "trace_id": "c",
                "steps": [{"kind": "message", "role": "user", "content": "hi"}],
            }
        ),
        encoding="utf-8",
    )
    assert looks_like_trace(path)
    assert detect_source(path) == "canonical"


def test_supporting_instruction_quotes_the_tool_sentence():
    trace = CanonicalTrace(
        trace_id="t",
        source_format="canonical",
        task="Refund the order.",
        instructions="Always call check_balance before refund_order. Be polite.",
        tools=[ToolSpec(name="refund_order", declared=True)],
        steps=[],
    )
    excerpt = supporting_instruction(trace, "refund_order")
    assert "check_balance before refund_order" in excerpt
    assert excerpt in trace.instructions


def test_format_steps_is_not_a_python_list():
    assert format_steps([11, 11, 11, 11]) == "11"
    assert format_steps([4, 5, 6]) == "4-6"
    assert format_steps([4, 9]) == "4, 9"


def test_unique_evidence_drops_duplicate_quotes():
    items = [
        Evidence(step=2, quote='{"ok": false}'),
        Evidence(step=2, quote='{"ok": false}'),
        Evidence(step=1, quote="x"),
    ]
    out = unique_evidence(items)
    assert len(out) == 2
    assert [e.step for e in out] == [2, 1]


def test_mechanical_finding_carries_available_info_and_source():
    trace = CanonicalTrace(
        trace_id="unk",
        source_format="canonical",
        task="Search the docs.",
        instructions="Use search. Never call explode.",
        tools=[
            ToolSpec(
                name="search",
                declared=True,
                parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
            )
        ],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="find the spec"),
            CanonicalStep(index=1, kind="tool_call", name="explode", arguments={"q": "x"}),
            CanonicalStep(index=2, kind="tool_result", name="explode", content='{"ok": true}'),
        ],
    )
    result = analyze_trace(trace, Config(), NullClient())
    conf = [f for f in result.findings if f.status == "confirmed"]
    assert conf
    finding = conf[0]
    assert finding.available_info
    assert "find the spec" in finding.available_info
    assert finding.evidence
    assert finding.evidence[0].source == "tool_call"
    assert "explode" in finding.rule_ref or "Never call explode" in finding.rule_ref


def test_python_repr_arguments_do_not_become_raw():
    parsed = _parse_call_arguments("{'path': '/tmp/x', 'ok': True}")
    assert parsed == {"path": "/tmp/x", "ok": True}


def test_tavii_replays_earlier_turns_from_prompt_messages(tmp_path):
    rows = [
        {"seq": 1, "kind": "run.start", "data": {"input": {"prompt": "Continue the thread"}}},
        {
            "seq": 2,
            "kind": "state",
            "data": {
                "type": "openclaw.llm_input",
                "value": {
                    "system_prompt": "Be brief.",
                    "tools": [
                        {
                            "name": "read",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                            },
                        }
                    ],
                },
            },
        },
        {
            "seq": 3,
            "kind": "state",
            "data": {
                "type": "openclaw.prompt_messages",
                "value": [
                    {"role": "user", "content": [{"type": "text", "text": "Read /tmp/a"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "toolCall",
                                "id": "old-1",
                                "name": "read",
                                "arguments": "{'path': '/tmp/a'}",
                            }
                        ],
                    },
                    {
                        "role": "toolResult",
                        "toolCallId": "old-1",
                        "toolName": "read",
                        "content": [{"type": "text", "text": "alpha"}],
                    },
                    {"role": "user", "content": [{"type": "text", "text": "Now the other file"}]},
                ],
            },
        },
        {
            "seq": 4,
            "kind": "tool.call",
            "data": {"name": "read", "native_call_id": "new-1", "input": {"path": "/tmp/b"}},
        },
        {
            "seq": 5,
            "kind": "tool.result",
            "data": {
                "name": "read",
                "native_call_id": "new-1",
                "status": "succeeded",
                "output": "beta",
            },
        },
    ]
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    trace = load_trace("replay", "tavii", [str(path)], 2000)
    roles = [(s.kind, s.role, s.name, (s.meta or {}).get("provenance")) for s in trace.steps]
    assert any(s.kind == "message" and s.role == "user" and "Read /tmp/a" in s.content for s in trace.steps)
    assert any(s.kind == "tool_call" and s.call_id == "old-1" and is_replay(s) for s in trace.steps)
    assert any(s.kind == "tool_call" and s.call_id == "new-1" and not is_replay(s) for s in trace.steps)
    assert [s.call_id for s in trace.steps if s.kind == "tool_call"] == ["old-1", "new-1"]
    # Replay must not create a mechanical unknown-tool / schema finding.
    flags = analyze(trace)
    assert not [f for f in flags.flags if f.kind in {"unknown_tool", "schema_violation"}]
    _ = roles


def test_extend_from_derived_adds_quoted_instruction():
    trace = CanonicalTrace(
        trace_id="t",
        source_format="canonical",
        instructions="Only use notify after the user confirms.",
        steps=[],
    )
    checks = [
        DerivedCheck(
            check_id="C1",
            family="instruction_violation",
            description="honor the confirm gate",
            condition="notify without confirm",
            source_refs=["Only use notify after the user confirms."],
            lane="obligation",
        )
    ]
    extra = extend_from_derived(trace, [], checks)
    assert extra
    assert extra[0].rule.startswith("Only use notify")


def test_falsify_disproof_becomes_an_abstention():
    trace = CanonicalTrace(
        trace_id="wx",
        source_format="canonical",
        task="Temperature in Oslo?",
        instructions="Answer only from tool results. Do not invent numbers.",
        tools=[
            ToolSpec(
                name="get_weather",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            )
        ],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="Oslo temp?"),
            CanonicalStep(index=1, kind="tool_call", name="get_weather", arguments={"city": "Oslo"}),
            CanonicalStep(index=2, kind="tool_result", name="get_weather", content='{"temp_c": 12}'),
            CanonicalStep(index=3, kind="message", role="assistant", content="It is 28C in Oslo."),
        ],
    )
    finding = {
        "check_id": "C1",
        "decision": "violated",
        "family": "evidence_contradiction",
        "steps": [2, 3],
        "description": "Said 28C but tool returned 12",
        "explanation": "The 12C result was already visible",
        "evidence": [
            {"step": 2, "quote": "temp_c\": 12", "source": "tool_result"},
            {"step": 3, "quote": "It is 28C in Oslo.", "source": "agent_message"},
        ],
        "rule_ref": "Answer only from tool results",
    }

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
            return {"findings": [finding]}
        if stage == "judge":
            return {
                "verdicts": [
                    {"index": 0, "verdict": "confirmed", "reason": "quote-backed"},
                    {"index": 1, "verdict": "confirmed", "reason": "quote-backed"},
                ]
            }
        if stage == "falsify":
            return {
                "verdicts": [
                    {"index": 0, "verdict": "disproved", "reason": "the 28C claim is a joke the user asked for"},
                    {"index": 1, "verdict": "disproved", "reason": "the 28C claim is a joke the user asked for"},
                ]
            }
        return {"verdicts": []}

    result = analyze_trace(trace, Config(), ScriptedClient(handler))
    assert not [f for f in result.findings if f.status == "confirmed" and f.family == "evidence_contradiction"]
    assert any(f.status == "insufficient_evidence" and f.meta.get("falsified") for f in result.findings)


def test_classify_marks_formation_and_mechanical_confirms():
    trace = CanonicalTrace(
        trace_id="form",
        source_format="canonical",
        tools=[
            ToolSpec(
                name="lookup",
                declared=True,
                parameters={
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            )
        ],
        steps=[
            CanonicalStep(index=0, kind="tool_call", name="lookup", arguments={"id": "x"}),
            CanonicalStep(
                index=1,
                kind="tool_result",
                name="lookup",
                content="llamada mal formada: parametro desconocido",
                is_error=True,
            ),
        ],
    )

    def handler(stage, _messages):
        if stage == "classify_errors":
            return {"classifications": [{"step": 1, "kind": "formation"}]}
        if stage == "derive":
            return {"checks": []}
        if stage in {"agent_a", "agent_b"}:
            return {"findings": []}
        return {"verdicts": []}

    result = analyze_trace(trace, Config(), ScriptedClient(handler))
    assert any(
        f.status == "confirmed" and f.family == "incorrect_tool_use" for f in result.findings
    )


def test_two_mechanical_findings_are_not_merged_by_shared_instruction():
    """A shared instruction sentence must not collapse two proven events."""
    from traceaudit.pipeline import _dedupe_same_rule

    shared = "Use get_weather. Quote the returned temperature exactly."
    a = Finding(
        finding_id="A",
        trace_id="t",
        family="incorrect_tool_use",
        description="bad call",
        steps=[1, 2],
        status="confirmed",
        source="deterministic",
        rule_ref=shared,
    )
    b = Finding(
        finding_id="B",
        trace_id="t",
        family="redundant_action",
        description="repeat",
        steps=[2, 4],
        status="confirmed",
        source="deterministic",
        rule_ref=shared,
    )
    kept = _dedupe_same_rule([a, b])
    assert len([f for f in kept if f.status == "confirmed"]) == 2
