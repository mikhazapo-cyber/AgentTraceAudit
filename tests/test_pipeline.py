from traceaudit.config import Config
from traceaudit.llm import PaymentRequired, ScriptedClient
from traceaudit.pipeline import analyze_trace, local_proof, parse_findings
from traceaudit.schemas import (
    CanonicalStep,
    CanonicalTrace,
    CandidateFinding,
    Evidence,
    Finding,
    ToolSpec,
    clamp_confidence,
    normalize_class_and_family,
)


def _weather_trace():
    return CanonicalTrace(
        trace_id="wx",
        source_format="canonical",
        task="Temperature in Oslo?",
        instructions="Answer only from tool results. Do not invent numbers.",
        tools=[
            ToolSpec(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            )
        ],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="Oslo temp?"),
            CanonicalStep(
                index=1,
                kind="tool_call",
                name="get_weather",
                arguments={"city": "Oslo"},
            ),
            CanonicalStep(
                index=2,
                kind="tool_result",
                name="get_weather",
                content='{"temp_c": 12}',
            ),
            CanonicalStep(
                index=3, kind="message", role="assistant", content="It is 28C in Oslo."
            ),
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
        "evidence": [
            {"step": 2, "quote": 'temp_c": 12'},
            {"step": 3, "quote": "It is 28C in Oslo."},
        ],
        "rule_ref": "Answer only from tool results",
    }


def test_two_auditors_then_judge_confirms():
    seen: list[str] = []

    def handler(stage, _messages):
        seen.append(stage)
        if stage in {"agent_a", "agent_b"}:
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
                ],
                "findings": [_contradiction_finding()],
            }
        if stage == "judge":
            return {
                "verdicts": [
                    {
                        "index": 0,
                        "verdict": "confirmed",
                        "family": "evidence_contradiction",
                        "reason": "quote-backed",
                    },
                    {
                        "index": 1,
                        "verdict": "confirmed",
                        "family": "evidence_contradiction",
                        "reason": "quote-backed",
                    },
                ]
            }
        raise AssertionError(stage)

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    conf = [
        f
        for f in result.findings
        if f.status == "confirmed" and f.family == "evidence_contradiction"
    ]
    assert len(conf) == 1
    assert result.checks
    assert conf[0].evidence
    assert "agent_a" in seen and "agent_b" in seen and "judge" in seen
    assert result.status == "ok"


def test_judge_assigns_family():
    def handler(stage, _messages):
        if stage in {"agent_a", "agent_b"}:
            item = dict(_contradiction_finding())
            item["family"] = "other"
            return {"checks": [], "findings": [item]}
        return {
            "verdicts": [
                {
                    "index": 0,
                    "verdict": "confirmed",
                    "family": "evidence_contradiction",
                    "reason": "refiled",
                },
                {"index": 1, "verdict": "rejected", "reason": "duplicate"},
            ]
        }

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    conf = [f for f in result.findings if f.status == "confirmed"]
    assert conf[0].family == "evidence_contradiction"
    assert conf[0].error_class == "contradicts_tool_output"


def test_unverified_quotes_become_insufficient():
    def handler(stage, _messages):
        if stage in {"agent_a", "agent_b"}:
            return {
                "checks": [],
                "findings": [
                    {
                        "check_id": "X",
                        "decision": "violated",
                        "family": "evidence_contradiction",
                        "steps": [3],
                        "description": "Invented a number",
                        "evidence": [
                            {"step": 3, "quote": "this quote is not in the step"}
                        ],
                    }
                ],
            }
        return {
            "verdicts": [
                {"index": 0, "verdict": "confirmed", "reason": "ok"},
                {"index": 1, "verdict": "confirmed", "reason": "ok"},
            ]
        }

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    assert all(f.status == "insufficient_evidence" for f in result.findings)
    assert not [f for f in result.findings if f.status == "confirmed"]


def test_empty_proposals_skip_judge():
    seen: list[str] = []

    def handler(stage, _messages):
        seen.append(stage)
        return {"checks": [], "findings": []}

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    assert result.status == "ok"
    assert result.findings == []
    assert "judge" not in seen


def test_inefficiency_needs_alternative():
    cand = CandidateFinding(
        family="redundant_action",
        description="called twice",
        steps=[1],
        evidence=[Evidence(step=1, quote="get_weather")],
        alternative="",
    )
    ok, reason = local_proof(_weather_trace(), cand)
    assert not ok
    assert "alternative" in reason


def test_peer_agreement_confirms_judge_abstention():
    seen = []

    def handler(stage, _messages):
        seen.append(stage)
        if stage in {"agent_a", "agent_b"}:
            return {"checks": [], "findings": [_contradiction_finding()]}
        return {
            "verdicts": [
                {"index": 0, "verdict": "insufficient_evidence", "reason": ""},
                {"index": 1, "verdict": "insufficient_evidence", "reason": ""},
            ]
        }

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    conf = [f for f in result.findings if f.status == "confirmed"]
    assert len(conf) == 1
    assert conf[0].family == "evidence_contradiction"
    assert "judge" in seen


def test_explicit_reject_is_not_overridden_by_peers():
    def handler(stage, _messages):
        if stage in {"agent_a", "agent_b"}:
            return {"checks": [], "findings": [_contradiction_finding()]}
        return {
            "verdicts": [
                {"index": 0, "verdict": "rejected", "reason": "justified"},
                {"index": 1, "verdict": "rejected", "reason": "justified"},
            ]
        }

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    assert [f for f in result.findings if f.status == "confirmed"] == []
    assert result.rejected


def test_lone_abstention_stays_insufficient():
    def handler(stage, _messages):
        if stage == "agent_a":
            return {"checks": [], "findings": [_contradiction_finding()]}
        if stage == "agent_b":
            return {"checks": [], "findings": []}
        return {
            "verdicts": [
                {"index": 0, "verdict": "insufficient_evidence", "reason": "not proven"}
            ]
        }

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    assert [f for f in result.findings if f.status == "confirmed"] == []
    assert any(f.status == "insufficient_evidence" for f in result.findings)


def test_peer_agree_confirms_judge_abstention():
    def handler(stage, _messages):
        if stage in {"agent_a", "agent_b"}:
            return {"checks": [], "findings": [_contradiction_finding()]}
        return {
            "verdicts": [
                {"index": 0, "verdict": "insufficient_evidence", "reason": ""},
                {"index": 1, "verdict": "insufficient_evidence", "reason": ""},
            ]
        }

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    conf = [f for f in result.findings if f.status == "confirmed"]
    assert len(conf) == 1
    assert conf[0].family == "evidence_contradiction"


def test_justified_retry_abstention_is_not_peer_promoted():
    def handler(stage, _messages):
        if stage in {"agent_a", "agent_b"}:
            return {"checks": [], "findings": [_contradiction_finding()]}
        return {
            "verdicts": [
                {
                    "index": 0,
                    "verdict": "insufficient_evidence",
                    "reason": "justified retry after the transient 429",
                },
                {
                    "index": 1,
                    "verdict": "insufficient_evidence",
                    "reason": "justified retry after the transient 429",
                },
            ]
        }

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    assert [f for f in result.findings if f.status == "confirmed"] == []
    assert any(f.status == "insufficient_evidence" for f in result.findings)


def test_explicit_reject_not_overridden_by_peers():
    def handler(stage, _messages):
        if stage in {"agent_a", "agent_b"}:
            return {"checks": [], "findings": [_contradiction_finding()]}
        return {
            "verdicts": [
                {"index": 0, "verdict": "rejected", "reason": "justified"},
                {
                    "index": 1,
                    "verdict": "insufficient_evidence",
                    "reason": "not proven",
                },
            ]
        }

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    assert [f for f in result.findings if f.status == "confirmed"] == []
    assert result.rejected


def test_on_stage_emits_auditors_and_judge():
    seen: list[str] = []

    def handler(stage, _messages):
        if stage in {"agent_a", "agent_b"}:
            return {"checks": [], "findings": [_contradiction_finding()]}
        return {
            "verdicts": [
                {
                    "index": 0,
                    "verdict": "confirmed",
                    "family": "evidence_contradiction",
                    "reason": "ok",
                },
                {"index": 1, "verdict": "rejected", "reason": "dup"},
            ]
        }

    analyze_trace(
        _weather_trace(),
        Config(),
        ScriptedClient(handler),
        on_stage=seen.append,
    )
    assert seen[:2] == ["auditors", "judge"]
    assert seen[-1] == "done"


def test_both_auditors_payment_required_is_error():
    def handler(_stage, _messages):
        raise PaymentRequired("402 Payment Required")

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    assert result.status == "error"
    assert "payment required" in result.error.lower()


def test_one_auditor_payment_still_ok_if_other_ran():
    def handler(stage, _messages):
        if stage == "agent_b":
            raise PaymentRequired("402 Payment Required")
        return {"checks": [], "findings": []}

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    assert result.status == "ok"
    assert result.findings == []


def test_judge_reject_is_not_a_finding():
    def handler(stage, _messages):
        if stage in {"agent_a", "agent_b"}:
            return {"checks": [], "findings": [_contradiction_finding()]}
        return {
            "verdicts": [
                {"index": 0, "verdict": "rejected", "reason": "justified"},
                {"index": 1, "verdict": "rejected", "reason": "justified"},
            ]
        }

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    assert [f for f in result.findings if f.status == "confirmed"] == []
    assert result.rejected


def test_omitted_class_and_confidence_defaults():
    def handler(stage, _messages):
        if stage in {"agent_a", "agent_b"}:
            return {"checks": [], "findings": [_contradiction_finding()]}
        return {
            "verdicts": [
                {
                    "index": 0,
                    "verdict": "confirmed",
                    "family": "evidence_contradiction",
                    "reason": "ok",
                },
                {"index": 1, "verdict": "rejected", "reason": "dup"},
            ]
        }

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    conf = [f for f in result.findings if f.status == "confirmed"]
    assert len(conf) == 1
    assert conf[0].error_class == "contradicts_tool_output"
    assert conf[0].confidence == 80


def test_low_confidence_confirm_becomes_abstention():
    def handler(stage, _messages):
        if stage in {"agent_a", "agent_b"}:
            item = dict(_contradiction_finding())
            item["error_class"] = "contradicts_tool_output"
            item["confidence"] = 90
            return {"checks": [], "findings": [item]}
        return {
            "verdicts": [
                {
                    "index": 0,
                    "verdict": "confirmed",
                    "family": "evidence_contradiction",
                    "error_class": "contradicts_tool_output",
                    "confidence": 60,
                    "reason": "thin",
                },
                {"index": 1, "verdict": "rejected", "reason": "dup"},
            ]
        }

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    assert [f for f in result.findings if f.status == "confirmed"] == []
    abst = [f for f in result.findings if f.status == "insufficient_evidence"]
    assert abst
    assert abst[0].confidence == 60
    assert "below 70" in abst[0].adjudication_reason
    assert result.rejected


def test_min_confidence_config_threshold():
    def handler(stage, _messages):
        if stage == "agent_a":
            item = dict(_contradiction_finding())
            item["error_class"] = "contradicts_tool_output"
            item["confidence"] = 88
            return {"checks": [], "findings": [item]}
        if stage == "agent_b":
            return {"checks": [], "findings": []}
        return {
            "verdicts": [
                {
                    "index": 0,
                    "verdict": "confirmed",
                    "family": "evidence_contradiction",
                    "error_class": "contradicts_tool_output",
                    "confidence": 80,
                    "reason": "ok",
                }
            ]
        }

    result = analyze_trace(
        _weather_trace(), Config(min_confidence=90), ScriptedClient(handler)
    )
    assert [f for f in result.findings if f.status == "confirmed"] == []
    assert any(f.status == "insufficient_evidence" for f in result.findings)


def test_unknown_error_class_falls_back_to_other():
    def handler(stage, _messages):
        if stage == "agent_a":
            item = dict(_contradiction_finding())
            item["error_class"] = "jailbreak_attempt"
            item["confidence"] = 95
            return {"checks": [], "findings": [item]}
        if stage == "agent_b":
            return {"checks": [], "findings": []}
        return {
            "verdicts": [
                {
                    "index": 0,
                    "verdict": "confirmed",
                    "family": "evidence_contradiction",
                    "error_class": "jailbreak_attempt",
                    "confidence": 95,
                    "reason": "ok",
                }
            ]
        }

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    conf = [f for f in result.findings if f.status == "confirmed"]
    assert len(conf) == 1
    assert conf[0].family == "evidence_contradiction"
    assert conf[0].error_class == "contradicts_tool_output"


def test_merge_keeps_higher_confidence_class():
    def _tool_item(error_class: str, confidence: int, description: str) -> dict:
        return {
            "check_id": "T1",
            "decision": "violated",
            "family": "incorrect_tool_use",
            "error_class": error_class,
            "confidence": confidence,
            "steps": [1],
            "description": description,
            "explanation": "schema governs city",
            "evidence": [{"step": 1, "quote": "get_weather"}],
            "rule_ref": "get_weather",
        }

    def handler(stage, _messages):
        if stage == "agent_a":
            return {
                "checks": [],
                "findings": [_tool_item("bad_arguments", 92, "wrong city value")],
            }
        if stage == "agent_b":
            return {
                "checks": [],
                "findings": [_tool_item("malformed_call", 71, "schema-invalid city")],
            }
        return {
            "verdicts": [
                {
                    "index": 0,
                    "verdict": "confirmed",
                    "family": "incorrect_tool_use",
                    "error_class": "bad_arguments",
                    "confidence": 92,
                    "reason": "ok",
                },
                {
                    "index": 1,
                    "verdict": "confirmed",
                    "family": "incorrect_tool_use",
                    "error_class": "malformed_call",
                    "confidence": 71,
                    "reason": "ok",
                },
            ]
        }

    result = analyze_trace(_weather_trace(), Config(), ScriptedClient(handler))
    conf = [f for f in result.findings if f.status == "confirmed"]
    assert len(conf) == 1
    assert conf[0].error_class == "bad_arguments"
    assert conf[0].confidence == 92
    assert conf[0].family == "incorrect_tool_use"


def test_parse_findings_auditor_defaults_and_drop():
    parsed = parse_findings(
        {
            "findings": [
                {
                    "family": "incorrect_tool_use",
                    "description": "bad id",
                    "steps": [1],
                    "evidence": [{"step": 1, "quote": "x"}],
                },
                {"description": "no parent or class", "steps": [1]},
            ]
        },
        "agent_a",
    )
    assert len(parsed) == 1
    assert parsed[0].error_class == "bad_arguments"
    assert parsed[0].confidence == 70


def test_finding_models_sanitize_unknown_class():
    finding = Finding(
        finding_id="x",
        trace_id="t",
        family="instruction_violation",
        error_class="jailbreak_attempt",
        description="d",
    )
    assert finding.family == "instruction_violation"
    assert finding.error_class == "broke_explicit_rule"
    cand = CandidateFinding(
        family="evidence_contradiction",
        error_class="not_a_class",
        description="d",
    )
    assert cand.family == "evidence_contradiction"
    assert cand.error_class == "contradicts_tool_output"


def test_normalize_unknown_and_known_class():
    assert normalize_class_and_family("bad_arguments", "other") == (
        "bad_arguments",
        "incorrect_tool_use",
    )
    assert normalize_class_and_family("not_a_class", "instruction_violation") == (
        "broke_explicit_rule",
        "instruction_violation",
    )
    assert normalize_class_and_family("not_a_class", "") == (
        "other_error",
        "other",
    )
    assert normalize_class_and_family("", "redundant_action") == (
        "duplicate_successful_call",
        "redundant_action",
    )
    assert normalize_class_and_family("", "not_a_family") is None
    assert clamp_confidence(None, 70) == 70
    assert clamp_confidence(0.86, 70) == 86
    assert clamp_confidence(140, 70) == 100
    assert clamp_confidence(-3, 70) == 0
