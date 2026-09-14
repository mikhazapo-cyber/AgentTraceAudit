"""Staged-lane regressions.

Each case here is a dev-set gold finding the live run missed, or a false
positive the live run confirmed. They are shaped from the trace structure, not
copied from the labelled traces, so they stay honest about generalisation.
"""

from traceaudit.config import Config
from traceaudit.derive import split_lanes
from traceaudit.llm import ScriptedClient
from traceaudit.obligations import call_outline, compile_obligations
from traceaudit.pipeline import analyze_trace
from traceaudit.schemas import (
    CandidateFinding,
    CanonicalStep,
    CanonicalTrace,
    DerivedCheck,
    Evidence,
    ToolSpec,
)
from traceaudit.verify import quick_check, repeated_same_call_after_error


def _restricted_tool_trace():
    """A gated tool used when the gate is closed (external-009 shape)."""
    return CanonicalTrace(
        trace_id="gated",
        source_format="canonical",
        task="what famous people does this comic do impressions of",
        instructions=(
            "get_relations(variable) -> list of relations. "
            "get_attributes(variable) -> list of attributes. "
            "Please only use it if the question seeks for a superlative accumulation "
            "(i.e., argmax or argmin)."
        ),
        tools=[
            ToolSpec(name="get_relations", declared=True, parameters={"type": "object"}),
            ToolSpec(name="get_attributes", declared=True, parameters={"type": "object"}),
        ],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="what impressions does the comic do"),
            CanonicalStep(index=1, kind="tool_call", name="get_relations", arguments={"variable": "#0"}),
            CanonicalStep(index=2, kind="tool_result", name="get_relations", content="impersonated_by, notable_for"),
            CanonicalStep(index=3, kind="tool_call", name="get_attributes", arguments={"variable": "#0"}),
            CanonicalStep(index=4, kind="tool_result", name="get_attributes", content="height, birth_date"),
        ],
    )


def _prerequisite_trace():
    """A required balance check skipped, and the later action still succeeds (external-017 shape)."""
    return CanonicalTrace(
        trace_id="prereq",
        source_format="canonical",
        task="Swap the item and put the difference on my gift card.",
        instructions=(
            "The user must provide a payment method to pay the price difference. "
            "If the user provides a gift card, it must have enough balance to cover "
            "the price difference. "
            "Call get_user_details to inspect balances."
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
            CanonicalStep(
                index=0,
                kind="message",
                role="user",
                content="Use my gift card GC-9 to cover the price difference please.",
            ),
            CanonicalStep(
                index=1,
                kind="tool_call",
                name="modify_pending_order_items",
                arguments={"order_id": "W-1", "payment_method_id": "gift_card_9"},
            ),
            CanonicalStep(index=2, kind="tool_result", name="modify_pending_order_items", content='{"status": "ok"}'),
            CanonicalStep(index=3, kind="message", role="assistant", content="Done, the difference went on the gift card."),
        ],
    )


def _did_not_retry_trace():
    """An error the agent never retried (tavii-022 shape). Mechanical owns this."""
    return CanonicalTrace(
        trace_id="noretry",
        source_format="canonical",
        task="React to the message.",
        instructions="Use the message tool to react.",
        tools=[
            ToolSpec(
                name="message",
                declared=True,
                parameters={
                    "type": "object",
                    "properties": {"action": {"type": "string"}, "messageId": {"type": "string"}},
                    "required": ["action", "messageId"],
                },
            )
        ],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="Please react with eyes."),
            CanonicalStep(index=1, kind="tool_call", name="message", arguments={"action": "react", "emoji": "eyes"}),
            CanonicalStep(
                index=2,
                kind="tool_result",
                name="message",
                content='{"status": "error", "tool": "message", "error": "messageId required"}',
                is_error=True,
            ),
            CanonicalStep(index=3, kind="message", role="assistant", content="Moving on to the shortlist."),
        ],
    )


# --- obligation compiler -------------------------------------------------


def test_restricted_tool_binds_through_pronoun():
    obligations = compile_obligations(_restricted_tool_trace())
    restricted = [o for o in obligations if o.kind == "restricted_tool"]
    assert restricted, [(o.kind, o.rule) for o in obligations]
    ob = restricted[0]
    assert ob.bound_tool == "get_attributes"
    assert "superlative" in ob.rule.lower()
    assert 3 in ob.relevant_steps


def test_prerequisite_fires_from_trace_terms():
    obligations = compile_obligations(_prerequisite_trace())
    prereq = [o for o in obligations if o.kind == "prerequisite"]
    assert prereq, [(o.kind, o.rule) for o in obligations]
    assert any("balance" in o.rule.lower() for o in prereq)
    assert all(o.fired for o in obligations)


def test_obligations_do_not_fire_when_situation_never_arose():
    trace = _prerequisite_trace()
    trace.instructions += " Never issue a refund to a closed account."
    rules = [o.rule for o in compile_obligations(trace)]
    assert not any("closed account" in r for r in rules)


def test_call_outline_lists_every_call_for_absence_proofs():
    outline = call_outline(_prerequisite_trace())
    assert "modify_pending_order_items" in outline
    assert "get_user_details" not in outline


def test_split_lanes_separates_instruction_rules_from_the_rest():
    checks = [
        DerivedCheck(check_id="O1", family="instruction_violation", description="d", condition="c", lane="obligation"),
        DerivedCheck(check_id="R1", family="redundant_action", description="d", condition="c", lane="residual"),
        DerivedCheck(check_id="S1", family="incorrect_tool_use", description="d", condition="c", lane="mechanical"),
    ]
    obligation, residual = split_lanes(checks)
    assert [c.check_id for c in obligation] == ["O1"]
    assert [c.check_id for c in residual] == ["R1"]


# --- recall: the two instruction golds ----------------------------------


def _lane_handler(finding, checks=None):
    def handler(stage, _messages):
        if stage == "derive":
            return {"checks": checks or []}
        if stage == "agent_a":
            return {"findings": [finding]}
        if stage == "agent_b":
            return {"findings": []}
        return {"verdicts": [{"index": 0, "verdict": "confirmed", "reason": "rule applies"}]}

    return handler


def test_restricted_tool_violation_is_confirmed():
    finding = {
        "check_id": "OBL-1",
        "decision": "violated",
        "family": "instruction_violation",
        "steps": [3],
        "description": "get_attributes used although the question seeks no superlative",
        "explanation": "The task asks which people are impersonated, not an argmax or argmin",
        "evidence": [{"step": 3, "quote": '{"variable": "#0"}'}],
        "rule_ref": "Please only use it if the question seeks for a superlative accumulation",
    }
    result = analyze_trace(_restricted_tool_trace(), Config(), ScriptedClient(_lane_handler(finding)))
    conf = [f for f in result.findings if f.status == "confirmed" and f.family == "instruction_violation"]
    assert conf, [(f.family, f.status, f.verification.reason) for f in result.findings]
    assert conf[0].verification.family_gate


def test_skipped_prerequisite_is_confirmed_even_though_the_action_succeeded():
    finding = {
        "check_id": "OBL-1",
        "decision": "violated",
        "family": "instruction_violation",
        "steps": [0, 1],
        "description": "Gift card applied without verifying its balance",
        "explanation": "get_user_details never ran, so the balance was never checked",
        "evidence": [{"step": 0, "quote": "Use my gift card GC-9 to cover the price difference"}],
        "rule_ref": "If the user provides a gift card, it must have enough balance to cover the price difference.",
    }
    result = analyze_trace(_prerequisite_trace(), Config(), ScriptedClient(_lane_handler(finding)))
    conf = [f for f in result.findings if f.status == "confirmed" and f.family == "instruction_violation"]
    assert conf, [(f.family, f.status, f.verification.reason) for f in result.findings]
    assert "balance" in conf[0].rule_ref.lower()


# --- precision: the two false-positive patterns -------------------------


def test_not_retrying_an_error_is_not_ignored_feedback():
    trace = _did_not_retry_trace()
    finding = {
        "decision": "violated",
        "family": "ignored_feedback",
        "steps": [1, 2],
        "description": "The agent did not correct the missing messageId after the call failed",
        "explanation": "It moved on instead of retrying",
        "evidence": [{"step": 2, "quote": '"error": "messageId required"'}],
        "alternative": "Retry the message call including the required messageId.",
    }
    result = analyze_trace(trace, Config(), ScriptedClient(_lane_handler(finding)))
    assert not [f for f in result.findings if f.status == "confirmed" and f.family == "ignored_feedback"]
    # The real error at the same steps is still reported, under the right family.
    assert [f for f in result.findings if f.status == "confirmed" and f.family == "incorrect_tool_use"]


def test_narrating_failure_after_success_is_not_ignored_feedback():
    trace = CanonicalTrace(
        trace_id="phantom",
        source_format="canonical",
        task="Convert 90 degrees to radians.",
        instructions="Use the calculator tools.",
        tools=[ToolSpec(name="degrees_to_radians", declared=True, parameters={"type": "object"})],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="90 degrees in radians?"),
            CanonicalStep(index=1, kind="tool_call", name="degrees_to_radians", arguments={"degrees": 90}),
            CanonicalStep(index=2, kind="tool_result", name="degrees_to_radians", content="1.5707963267948966"),
            CanonicalStep(
                index=3,
                kind="message",
                role="assistant",
                content="My apologies. Let me correct the usage by passing the correct parameter names.",
            ),
        ],
    )
    finding = {
        "decision": "violated",
        "family": "ignored_feedback",
        "steps": [2, 3],
        "description": "The assistant implies the call failed although it succeeded",
        "explanation": "It ignored the successful result",
        "evidence": [{"step": 3, "quote": "Let me correct the usage by passing the correct parameter names."}],
        "alternative": "Accept the successful result from step 2.",
    }
    result = analyze_trace(trace, Config(), ScriptedClient(_lane_handler(finding)))
    assert not [f for f in result.findings if f.status == "confirmed" and f.family == "ignored_feedback"]


def test_same_args_retry_after_error_still_counts_as_ignored_feedback():
    trace = CanonicalTrace(
        trace_id="retry",
        source_format="canonical",
        instructions="Use the search tool.",
        tools=[ToolSpec(name="search", declared=True, parameters={"type": "object"})],
        steps=[
            CanonicalStep(index=0, kind="tool_call", name="search", arguments={"q": ""}),
            CanonicalStep(
                index=1,
                kind="tool_result",
                name="search",
                content="invalid request: q must not be empty",
                is_error=True,
            ),
            CanonicalStep(index=2, kind="tool_call", name="search", arguments={"q": ""}),
            CanonicalStep(
                index=3,
                kind="tool_result",
                name="search",
                content="invalid request: q must not be empty",
                is_error=True,
            ),
        ],
    )
    cand = CandidateFinding(
        family="ignored_feedback",
        description="repeated the identical failing call",
        steps=[0, 1, 2],
        evidence=[Evidence(step=1, quote="q must not be empty")],
        alternative="Change q before retrying.",
        source="llm",
    )
    assert repeated_same_call_after_error(trace, cand)


def test_different_arguments_are_search_not_redundancy():
    trace = CanonicalTrace(
        trace_id="search",
        source_format="canonical",
        instructions="Use the smallest meaningful verification step.",
        tools=[ToolSpec(name="read_saved_search", declared=True, parameters={"type": "object"})],
        steps=[
            CanonicalStep(index=0, kind="tool_call", name="read_saved_search", arguments={"id": "s1"}),
            CanonicalStep(index=1, kind="tool_result", name="read_saved_search", content='{"title": "ops"}'),
            CanonicalStep(
                index=2,
                kind="tool_call",
                name="read_saved_search",
                arguments={"id": "s1", "includeArtifacts": False},
            ),
            CanonicalStep(index=3, kind="tool_result", name="read_saved_search", content='{"title": "ops"}'),
        ],
    )
    cand = CandidateFinding(
        family="redundant_action",
        description="reread the saved search after the listing already had the metadata",
        steps=[0, 2],
        evidence=[Evidence(step=2, quote='"includeArtifacts": false')],
        alternative="Reuse the metadata from step 1.",
        source="llm",
    )
    verdict, reason = quick_check(trace, cand)
    assert verdict == "rejected"
    assert "identical arguments" in reason


def test_skill_file_process_rule_cannot_be_confirmed():
    """Reading which SKILL.md to open is process guidance, not a domain action."""
    trace = CanonicalTrace(
        trace_id="skill",
        source_format="canonical",
        instructions="Constraints: never read more than one skill up front; only read after selecting.",
        tools=[ToolSpec(name="read", declared=True), ToolSpec(name="dispatch_search", declared=True)],
        steps=[
            CanonicalStep(index=0, kind="tool_call", name="read", arguments={"path": "/skills/a/SKILL.md"}),
            CanonicalStep(index=1, kind="tool_result", name="read", content="skill a"),
            CanonicalStep(index=2, kind="tool_call", name="read", arguments={"path": "/skills/b/SKILL.md"}),
            CanonicalStep(index=3, kind="tool_result", name="read", content="skill b"),
        ],
    )
    cand = CandidateFinding(
        family="instruction_violation",
        description="read two skills up front",
        condition="never read more than one skill up front",
        steps=[0, 2],
        evidence=[Evidence(step=0, quote='"/skills/a/SKILL.md"'), Evidence(step=2, quote='"/skills/b/SKILL.md"')],
        meta={"rule_ref": "Constraints: never read more than one skill up front; only read after selecting."},
        source="llm",
    )
    assert quick_check(trace, cand)[0] == "insufficient_evidence"


def test_prose_skill_route_is_process_guidance():
    trace = CanonicalTrace(
        trace_id="route3",
        source_format="canonical",
        instructions="For role-focused sourcing, use `dispatch-search`, then `search-dispatch`, before choosing search tools.",
        tools=[ToolSpec(name="read", declared=True), ToolSpec(name="dispatch_search", declared=True)],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="find similar profiles"),
            CanonicalStep(index=1, kind="tool_call", name="dispatch_search", arguments={"q": "x"}),
            CanonicalStep(index=2, kind="tool_result", name="dispatch_search", content='{"ok": true}'),
        ],
    )
    cand = CandidateFinding(
        family="instruction_violation",
        description="skipped the skill route",
        condition="should have read dispatch-search first",
        steps=[0, 1],
        evidence=[Evidence(step=0, quote="find similar profiles"), Evidence(step=1, quote='"q": "x"')],
        meta={
            "rule_ref": "For role-focused, judgment-heavy sourcing, use `dispatch-search`, then `search-dispatch`, before choosing search tools."
        },
        source="llm",
    )
    assert quick_check(trace, cand)[0] == "insufficient_evidence"


def test_backticked_skill_slug_is_still_process_guidance():
    """Skill names in backticks are not declared tools."""
    trace = CanonicalTrace(
        trace_id="route2",
        source_format="canonical",
        instructions="| role-focused sourcing | `dispatch-search`, then `search-dispatch` |",
        tools=[ToolSpec(name="read", declared=True), ToolSpec(name="dispatch_search", declared=True)],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="find similar profiles"),
            CanonicalStep(index=1, kind="tool_call", name="dispatch_search", arguments={"q": "x"}),
            CanonicalStep(index=2, kind="tool_result", name="dispatch_search", content='{"ok": true}'),
        ],
    )
    cand = CandidateFinding(
        family="instruction_violation",
        description="skipped the routing table",
        condition="should have read dispatch-search first",
        steps=[0, 1],
        evidence=[Evidence(step=0, quote="find similar profiles"), Evidence(step=1, quote='"q": "x"')],
        meta={
            "rule_ref": "| role-focused sourcing | `dispatch-search`, then `search-dispatch`, before choosing search tools |"
        },
        source="llm",
    )
    assert quick_check(trace, cand)[0] == "insufficient_evidence"


def test_category_relabel_is_not_a_value_contradiction():
    trace = CanonicalTrace(
        trace_id="alumni",
        source_format="canonical",
        task="Find people who have worked here in the past.",
        instructions="List alumni only.",
        tools=[ToolSpec(name="search_people", declared=True)],
        steps=[
            CanonicalStep(index=0, kind="tool_call", name="search_people", arguments={"q": "x"}),
            CanonicalStep(
                index=1,
                kind="tool_result",
                name="search_people",
                content='{"displayName": "Ada", "title": "CMO", "company": "acme"}',
            ),
            CanonicalStep(
                index=2,
                kind="message",
                role="assistant",
                content="Found past Acme profiles. Ada is an alumna.",
            ),
        ],
    )
    cand = CandidateFinding(
        family="evidence_contradiction",
        description="called a current employee an alumna",
        condition="company field shows current employer",
        steps=[1, 2],
        evidence=[
            Evidence(step=1, quote='"title": "CMO", "company": "acme"'),
            Evidence(step=2, quote="Found past Acme profiles. Ada is an alumna."),
        ],
        source="llm",
    )
    assert quick_check(trace, cand)[0] == "insufficient_evidence"


def test_numeric_mismatch_is_still_a_contradiction():
    trace = CanonicalTrace(
        trace_id="wx",
        source_format="canonical",
        instructions="Answer only from tool results.",
        tools=[ToolSpec(name="get_weather", declared=True)],
        steps=[
            CanonicalStep(index=0, kind="tool_call", name="get_weather", arguments={"city": "Oslo"}),
            CanonicalStep(index=1, kind="tool_result", name="get_weather", content='{"temp_c": 12}'),
            CanonicalStep(index=2, kind="message", role="assistant", content="It is 28C in Oslo."),
        ],
    )
    cand = CandidateFinding(
        family="evidence_contradiction",
        description="Said 28C but tool returned 12",
        condition="stated temperature differs",
        steps=[1, 2],
        evidence=[
            Evidence(step=1, quote="temp_c\": 12"),
            Evidence(step=2, quote="It is 28C in Oslo."),
        ],
        source="llm",
    )
    assert quick_check(trace, cand)[0] == "confirmed"


def test_markdown_routing_table_is_process_guidance():
    trace = CanonicalTrace(
        trace_id="route",
        source_format="canonical",
        instructions="| role-focused sourcing | dispatch-search, then search-dispatch, before choosing search tools |",
        tools=[ToolSpec(name="read", declared=True), ToolSpec(name="dispatch_search", declared=True)],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="find similar profiles"),
            CanonicalStep(index=1, kind="tool_call", name="dispatch_search", arguments={"q": "x"}),
            CanonicalStep(index=2, kind="tool_result", name="dispatch_search", content='{"ok": true}'),
        ],
    )
    cand = CandidateFinding(
        family="instruction_violation",
        description="skipped the routing table",
        condition="should have read dispatch-search first",
        steps=[0, 1],
        evidence=[Evidence(step=0, quote="find similar profiles"), Evidence(step=1, quote='"q": "x"')],
        meta={
            "rule_ref": "| role-focused, judgment-heavy sourcing | dispatch-search, then search-dispatch, before choosing search tools |"
        },
        source="llm",
    )
    assert quick_check(trace, cand)[0] == "insufficient_evidence"


def test_presentation_rule_cannot_be_confirmed():
    """A rule that is genuinely in the instructions, but only governs wording."""
    trace = _prerequisite_trace()
    rule = "Always end with a next step and a question for the user."
    trace.instructions += " " + rule
    cand = CandidateFinding(
        family="instruction_violation",
        description="the final message ends without a next step or a calibration question",
        steps=[3],
        evidence=[Evidence(step=3, quote="Done, the difference went on the gift card.")],
        source="llm",
        meta={"rule_ref": rule},
    )
    verdict, reason = quick_check(trace, cand)
    assert verdict == "insufficient_evidence"
    assert "wording" in reason


def test_required_utterance_script_is_still_wording():
    """A scripted closing line is wording even when the policy calls it required.

    Shaped like a context-update handoff, not copied from a labelled trace.
    """
    trace = _prerequisite_trace()
    rule = (
        "After saving the note, explicitly state that no follow-up job has started, "
        "then end with one direct question. This required handoff is not optional."
    )
    trace.instructions += " " + rule
    cand = CandidateFinding(
        family="instruction_violation",
        description="replied after a note save without the required closing question",
        steps=[2],
        evidence=[Evidence(step=2, quote='{"status": "ok"}')],
        source="llm",
        meta={"rule_ref": rule},
    )
    verdict, reason = quick_check(trace, cand)
    assert verdict == "insufficient_evidence"
    assert "wording" in reason


def test_silent_turn_protocol_is_not_a_domain_action():
    """When a harness may omit a reply is channel protocol, not a tool rule."""
    trace = CanonicalTrace(
        trace_id="quiet",
        source_format="canonical",
        task="Update the running note.",
        instructions=(
            "In user-initiated chat threads, return NO_REPLY only when a tool in "
            "this turn already sent a visible message or status card. A memory "
            "edit does not count."
        ),
        tools=[
            ToolSpec(name="edit", declared=True, parameters={"type": "object"}),
            ToolSpec(name="memory_search", declared=True, parameters={"type": "object"}),
        ],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="Save that the shortlist is stale."),
            CanonicalStep(index=1, kind="tool_call", name="memory_search", arguments={"q": "shortlist"}),
            CanonicalStep(index=2, kind="tool_result", name="memory_search", content="no hits"),
            CanonicalStep(index=3, kind="tool_call", name="edit", arguments={"path": "NOTE.md"}),
            CanonicalStep(index=4, kind="tool_result", name="edit", content="replaced 1 block"),
        ],
    )
    cand = CandidateFinding(
        family="instruction_violation",
        description="the turn ended after a note edit with no user-visible reply",
        steps=[0, 4],
        evidence=[
            Evidence(step=0, quote="Save that the shortlist is stale."),
            Evidence(step=4, quote="replaced 1 block"),
        ],
        source="llm",
        meta={
            "rule_ref": (
                "In user-initiated chat threads, return NO_REPLY only when a tool in "
                "this turn already sent a visible message or status card. A memory "
                "edit does not count."
            )
        },
    )
    verdict, reason = quick_check(trace, cand)
    assert verdict == "insufficient_evidence"
    assert "wording" in reason or "process" in reason


def test_clarification_playbook_before_a_named_tool_is_wording():
    """A propose/ask/then-call loop is conversation, even if it names a tool."""
    trace = CanonicalTrace(
        trace_id="brief",
        source_format="canonical",
        task="Find operators who later moved into consulting.",
        instructions=(
            "Default role-sourcing loop: 1. read the brief 2. list strategies "
            "3. propose a concise approach 4. get correction if the role is "
            "broad or missing a major axis 5. call `dispatch_search`"
        ),
        tools=[
            ToolSpec(name="list_search_strategies", declared=True, parameters={"type": "object"}),
            ToolSpec(name="dispatch_search", declared=True, parameters={"type": "object"}),
        ],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="Find operators who later moved into consulting."),
            CanonicalStep(index=1, kind="tool_call", name="list_search_strategies", arguments={}),
            CanonicalStep(index=2, kind="tool_result", name="list_search_strategies", content='{"ref":"general_strategy"}'),
            CanonicalStep(index=3, kind="tool_call", name="dispatch_search", arguments={"strategyRef": "general_strategy"}),
            CanonicalStep(
                index=4,
                kind="tool_result",
                name="dispatch_search",
                content='{"ok": true, "acknowledgementSent": true}',
            ),
        ],
    )
    cand = CandidateFinding(
        family="instruction_violation",
        description="dispatched a broad brief without proposing an approach first",
        steps=[3],
        evidence=[Evidence(step=3, quote='"strategyRef": "general_strategy"')],
        source="llm",
        meta={
            "rule_ref": (
                "Default role-sourcing loop: ... 3. propose a concise approach "
                "4. get correction if the role is broad or missing a major axis "
                "5. call `dispatch_search`"
            )
        },
    )
    verdict, reason = quick_check(trace, cand)
    assert verdict == "insufficient_evidence"
    assert "wording" in reason


def test_pronoun_tool_restriction_is_still_confirmable():
    """'Please only use it if ...' still constrains the bound tool."""
    trace = _restricted_tool_trace()
    cand = CandidateFinding(
        family="instruction_violation",
        description="used the gated tool when the question was not a superlative",
        steps=[3],
        evidence=[Evidence(step=3, quote='{"variable": "#0"}')],
        source="llm",
        meta={
            "rule_ref": "Please only use it if the question seeks for a superlative accumulation (i.e., argmax or argmin)."
        },
    )
    verdict, _reason = quick_check(trace, cand)
    assert verdict == "confirmed"


def test_action_constraining_rule_is_still_confirmable():
    trace = _prerequisite_trace()
    cand = CandidateFinding(
        family="instruction_violation",
        description="gift card applied without verifying the balance",
        steps=[0, 1],
        evidence=[Evidence(step=0, quote="Use my gift card GC-9 to cover the price difference")],
        source="llm",
        meta={"rule_ref": "If the user provides a gift card, it must have enough balance to cover the price difference."},
    )
    verdict, _reason = quick_check(trace, cand)
    assert verdict == "confirmed"


def test_instruction_violation_on_a_rejected_call_abstains():
    trace = CanonicalTrace(
        trace_id="probe",
        source_format="canonical",
        instructions="Read the skill with `read` at the exact location; never guess a path.",
        tools=[ToolSpec(name="read", declared=True, parameters={"type": "object"})],
        steps=[
            CanonicalStep(index=0, kind="tool_call", name="read", arguments={"path": "/guessed/SKILL.md"}),
            CanonicalStep(
                index=1,
                kind="tool_result",
                name="read",
                content="ENOENT: no such file or directory",
                is_error=True,
            ),
            CanonicalStep(index=2, kind="message", role="assistant", content="That path was wrong, moving on."),
        ],
    )
    cand = CandidateFinding(
        family="instruction_violation",
        description="the agent read a guessed SKILL.md path instead of the exact location",
        steps=[0, 1],
        evidence=[Evidence(step=0, quote='"path": "/guessed/SKILL.md"')],
        source="llm",
        meta={"rule_ref": "Read the skill with `read` at the exact location; never guess a path."},
    )
    verdict, reason = quick_check(trace, cand)
    assert verdict == "insufficient_evidence"
    assert "rejected by the environment" in reason


def test_one_rule_at_overlapping_steps_is_one_finding():
    trace = _prerequisite_trace()
    rule = "If the user provides a gift card, it must have enough balance to cover the price difference."

    def handler(stage, _messages):
        if stage == "derive":
            return {"checks": []}
        if stage == "agent_a":
            return {
                "findings": [
                    {
                        "decision": "violated",
                        "family": "instruction_violation",
                        "steps": [0, 1],
                        "description": "charged the difference to the gift card without checking the balance",
                        "evidence": [{"step": 0, "quote": "Use my gift card GC-9 to cover the price difference"}],
                        "rule_ref": rule,
                    }
                ]
            }
        if stage == "agent_b":
            return {
                "findings": [
                    {
                        "decision": "violated",
                        "family": "incorrect_tool_use",
                        "steps": [1],
                        "description": "charged a gift card without establishing sufficient balance",
                        "evidence": [{"step": 1, "quote": '"payment_method_id": "gift_card_9"'}],
                        "rule_ref": rule,
                    }
                ]
            }
        return {
            "verdicts": [
                {"index": 0, "verdict": "confirmed", "reason": "rule applies"},
                {"index": 1, "verdict": "confirmed", "reason": "rule applies"},
            ]
        }

    result = analyze_trace(trace, Config(), ScriptedClient(handler))
    conf = [f for f in result.findings if f.status == "confirmed"]
    assert len(conf) == 1, [(f.family, f.steps) for f in conf]
    assert conf[0].family == "instruction_violation"


def test_llm_proposal_overlapping_a_proven_event_is_dropped():
    trace = _did_not_retry_trace()
    recast = {
        "decision": "violated",
        "family": "other",
        "steps": [1],
        "description": "The react call was malformed",
        "explanation": "duplicate of the mechanical formation error",
        "evidence": [{"step": 1, "quote": '{"action": "react"'}],
    }
    result = analyze_trace(trace, Config(), ScriptedClient(_lane_handler(recast)))
    assert not [f for f in result.findings if f.family == "other"]
    assert any(c.meta.get("dropped_reason") for c in result.rejected)
