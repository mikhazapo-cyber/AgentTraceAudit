from __future__ import annotations

import re

from .obligations import _PREREQUISITE, _names_at_boundaries
from .quotes import excerpt_in_source, locatable, quote_supported, verify_candidate
from .schemas import CandidateFinding, CanonicalTrace, Evidence, Verification
from .structural import (
    _arg_hash,
    _is_call_formation_error,
    logical_tool_name,
    pair_calls_results,
    schema_unreliable,
    validate_against_schema,
)

_WRAPPERS = {"env_action", "bash", "finish"}
_COMPLETION = (
    "sent",
    "emailed",
    "created",
    "deleted",
    "paid",
    "refunded",
    "shipped",
    "submitted",
    "published",
    "approved",
    "refunded",
)
_MECHANICAL_PREFIX = "DET-"

_IGNORED_FEEDBACK_MISFILE = (
    "ignored_feedback needs the same tool repeated with the same arguments after an "
    "error, and the trace shows no such repeat. Not retrying, retrying differently, "
    "or narrating failure after a success belongs to another family."
)
_NO_ALTERNATIVE = "inefficiency needs a cheaper alternative that existed then"
_REDUNDANT_MISFILE = (
    "redundant_action needs a repeat with identical arguments; lookups with different "
    "arguments are search, not repeated work"
)
_UNGROUNDED_RULE = (
    "the cited rule governs wording, not an observable action, so the trace cannot "
    "settle it either way"
)
_PROCESS_GUIDANCE = (
    "the cited rule is process guidance about which document or skill to read, "
    "not an observable domain action the trace can settle"
)
_DOC_PROCESS = re.compile(
    r"(?i)\b(skill(?:s)?(?:\.md)?|playbook|runbook|readme)\b"
)
# A rule whose obligation is an utterance (what to say / ask / close with)
# is wording, even if it also says the line is "required".
_UTTERANCE_OBLIGATION = re.compile(
    r"(?i)\b("
    r"state that|say that|end with|close with|ask the user|ask whether|"
    r"ask (?:one |a )?(?:clarification|question)|"
    r"propose (?:a |an )?(?:approach|plan)|"
    r"get correction|"
    r"reply with|respond with|tell the user|phrasing|wording|"
    r"closing (?:phrase|question)|acknowledgement"
    r")\b"
)
# Restriction verbs that bind a tool (or "it"). Bare "only when" is too
# generic: channel-protocol rules use it without naming a tool.
_TOOL_RESTRICT = re.compile(
    r"(?i)("
    r"only use|use it only|please only|can only be used|should only be used|"
    r"may only be used|must only be used|only be called|"
    r"do not use|don't use|never use|never call|must not be used|"
    r"may not be used|is not allowed"
    r")"
)
_CHANNEL_PROTOCOL = re.compile(
    r"(?i)("
    r"\bno_reply\b|"
    r"status card|"
    r"visible (?:slack |user )?message|"
    r"user-visible|"
    r"silent turn"
    r")"
)
_ONLY_FAILED_CALLS = (
    "every cited call was rejected by the environment, so the attempt changed nothing; "
    "the failure itself is the reportable fact"
)
# Gate failures that mean "this belongs to a different family", not "evidence is thin".
# Those are rejections, because abstaining would imply the claim might still be provable.
_MISFILED_FAMILY = frozenset({_IGNORED_FEEDBACK_MISFILE, _REDUNDANT_MISFILE})


def _arg_hash_groups(trace: CanonicalTrace) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for step in trace.steps:
        if step.kind == "tool_call":
            groups.setdefault(_arg_hash(step), []).append(step.index)
    return [idxs for idxs in groups.values() if len(idxs) > 1]


def repeated_identical_call(trace: CanonicalTrace, cand: CandidateFinding) -> bool:
    """True when the cited steps touch a call issued twice with identical arguments.

    redundant_action means repeated work that added no information. Two lookups
    with different arguments are search, so the family needs an identical repeat.
    """
    cited = set(cand.steps)
    if not cited:
        return False
    pairing = pair_calls_results(trace)
    for indices in _arg_hash_groups(trace):
        touched = set(indices)
        for idx in indices:
            ri = pairing.get(idx)
            if ri is not None:
                touched.add(ri)
        if cited & touched:
            return True
    return False


def _domain_tools(trace: CanonicalTrace) -> list[str]:
    """Callable tools other than generic file/shell readers."""
    skip = {"read", "exec", "bash", "cat", "open"}
    return [n for n in (trace.declared_tool_names() | trace.tool_names()) if n and n not in skip]


_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
_BOOLISH = re.compile(r"\b(true|false|yes|no|null|none|ok|error|succeeded|failed)\b", re.I)


def _direct_value_conflict(result_quotes: list[str], claim_quotes: list[str]) -> bool:
    """True when the claim uses a number or bool the result does not.

    Category relabels (alumni vs current employee) are judgement, not a
    value-level contradiction. A stated 28C against a returned 12C is.
    """
    result = " ".join(result_quotes)
    claim = " ".join(claim_quotes)
    rnums = set(_NUMBER.findall(result))
    cnums = set(_NUMBER.findall(claim))
    # Both sides must carry numbers, and they must disagree. A count the agent
    # invented in the claim (87 profiles) is not a contradiction of a name field.
    if rnums and cnums and rnums != cnums:
        return True
    rbool = {m.group(0).lower() for m in _BOOLISH.finditer(result)}
    cbool = {m.group(0).lower() for m in _BOOLISH.finditer(claim)}
    if rbool and cbool and rbool != cbool:
        return True
    return False


def is_process_guidance(trace: CanonicalTrace, rule: str) -> bool:
    """True when a rule is about which document to read, not a domain action.

    Skill files, playbooks, and markdown routing tables are process. A rule that
    names a declared callable tool is not: that is an observable action.
    """
    text = (rule or "").strip()
    if len(text) < 12:
        return False
    if _names_at_boundaries(text, _domain_tools(trace)):
        return False
    quoted = re.findall(r"`([A-Za-z_][\w.-]*)`", text)
    declared = {n.lower() for n in _domain_tools(trace)}
    if any(q.lower() in declared for q in quoted):
        return False
    if text.startswith("|") and text.count("|") >= 2:
        return True
    # Quoted slugs that are not declared tools are skill/workflow names.
    if quoted and _PREREQUISITE.search(text):
        return True
    if _CHANNEL_PROTOCOL.search(text):
        return True
    return bool(_DOC_PROCESS.search(text))


def rule_constrains_action(trace: CanonicalTrace, rule: str) -> bool:
    """True when a quoted rule governs an observable action rather than wording.

    An instruction_violation has to be provable from the trace. Rules that name a
    tool, gate a tool, or order one step before another are provable. Rules about
    how to phrase, format, or close a message are not: whether the agent honored
    them is the annotator's judgement, not a fact in the trace.
    """
    text = (rule or "").strip()
    if len(text) < 12:
        return False
    names = [n for n in (trace.declared_tool_names() | trace.tool_names()) if n]
    # Conversation scripts (ask / propose / say X) are wording even when a
    # later step of the same paragraph names a tool.
    if _UTTERANCE_OBLIGATION.search(text):
        return False
    if _CHANNEL_PROTOCOL.search(text):
        return False
    if _names_at_boundaries(text, names):
        return True
    # "After saving, state that X then end with a question" is a script, not a
    # tool obligation. The word "required" in that script must not promote it.
    return bool(_PREREQUISITE.search(text) or _TOOL_RESTRICT.search(text))


def cited_action_only_failed(trace: CanonicalTrace, cand: CandidateFinding) -> bool:
    """True when every cited tool call was rejected by the environment.

    A call the environment refused changed nothing, and the mechanical layer
    already reports the failure itself. Calling the attempt an instruction
    violation on top of that double-reports one event.
    """
    pairing = pair_calls_results(trace)
    result_to_call = {ri: ci for ci, ri in pairing.items()}
    calls: set[int] = set()
    for i in cand.steps:
        if not (0 <= i < len(trace.steps)):
            continue
        step = trace.steps[i]
        if step.kind == "tool_call":
            calls.add(i)
        elif step.kind == "tool_result" and i in result_to_call:
            # Citing only the error result is the same claim as citing the call.
            calls.add(result_to_call[i])
    if not calls:
        return False
    failed = False
    for idx in sorted(calls):
        ri = pairing.get(idx)
        if ri is None or not (0 <= ri < len(trace.steps)):
            # Outcome unknown, so we cannot say the attempt changed nothing.
            return False
        result = trace.steps[ri]
        if result.is_error or _is_call_formation_error(result.content):
            failed = True
        else:
            return False
    return failed


def repeated_same_call_after_error(trace: CanonicalTrace, cand: CandidateFinding) -> bool:
    """True when some cited step belongs to a same-tool same-args retry after an error.

    This is the only shape ignored_feedback can take. Without it, the finding is a
    misfiled formation error (nothing was retried) or a misfiled contradiction (the
    earlier call actually succeeded).
    """
    pairing = pair_calls_results(trace)
    cited = set(cand.steps)
    groups: dict[str, list[int]] = {}
    for step in trace.steps:
        if step.kind == "tool_call":
            groups.setdefault(_arg_hash(step), []).append(step.index)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        for earlier, later in zip(indices, indices[1:]):
            ri = pairing.get(earlier)
            if ri is None or not (0 <= ri < len(trace.steps)):
                continue
            result = trace.steps[ri]
            if not (result.is_error or _is_call_formation_error(result.content)):
                continue
            if cited & {earlier, ri, later}:
                return True
    return False


def is_mechanical(cand: CandidateFinding) -> bool:
    return cand.source == "deterministic" or (cand.check_id or "").startswith(_MECHANICAL_PREFIX)


def _rule_excerpt(cand: CandidateFinding) -> str:
    return str(cand.meta.get("rule_ref") or cand.condition or "")


def _family_gate(trace: CanonicalTrace, cand: CandidateFinding) -> tuple[bool, str]:
    family = cand.family
    if family == "instruction_violation":
        excerpt = _rule_excerpt(cand)
        refs = cand.meta.get("source_refs") or []
        grounded = excerpt_in_source(excerpt, trace.task, trace.instructions) or (
            isinstance(refs, list)
            and any(excerpt_in_source(str(r), trace.task, trace.instructions) for r in refs)
        )
        if not grounded:
            return False, "instruction excerpt not found in task/instructions"
        rule = excerpt or " ".join(str(r) for r in refs if r)
        if not rule_constrains_action(trace, rule):
            return False, _UNGROUNDED_RULE
        if cited_action_only_failed(trace, cand):
            return False, _ONLY_FAILED_CALLS
        if is_process_guidance(trace, rule):
            return False, _PROCESS_GUIDANCE
        return True, "instruction excerpt grounded and action-constraining"

    if family == "incorrect_tool_use":
        if schema_unreliable(trace) and (cand.check_id or "").endswith("schema_violation"):
            return False, "schema capture is unreliable"
        declared = trace.declared_tool_names()
        pairing = pair_calls_results(trace)
        for idx in cand.steps:
            if not (0 <= idx < len(trace.steps)):
                continue
            step = trace.steps[idx]
            if step.kind != "tool_call":
                continue
            name = logical_tool_name(step) or step.name
            if declared and name not in declared and step.name not in declared and name not in _WRAPPERS:
                return True, "undeclared tool"
            schema = trace.schema_for(step.name) or trace.schema_for(name)
            if schema and not schema_unreliable(trace):
                problems = validate_against_schema(step.arguments or {}, schema)
                if problems:
                    return True, problems[0]
            ri = pairing.get(step.index)
            if ri is not None and 0 <= ri < len(trace.steps):
                result = trace.steps[ri]
                if _is_call_formation_error(result.content):
                    return True, "environment rejected the call as formed"
        if is_mechanical(cand) and (cand.check_id or "").startswith("DET-"):
            return True, "mechanical tool-use check"
        return False, "no schema, undeclared-tool, or formation error proven"

    if family == "evidence_contradiction":
        kinds = []
        result_quotes: list[str] = []
        claim_quotes: list[str] = []
        for ev in cand.evidence:
            if not (0 <= ev.step < len(trace.steps)):
                continue
            if not quote_supported(trace, ev):
                continue
            kind = trace.steps[ev.step].kind
            kinds.append(kind)
            if kind == "tool_result":
                result_quotes.append(ev.quote)
            elif kind in {"message", "tool_call"}:
                claim_quotes.append(ev.quote)
        if "tool_result" in kinds and "message" in kinds:
            if is_mechanical(cand) or _direct_value_conflict(result_quotes, claim_quotes):
                return True, "result and claim quotes"
            return False, "category label is not a value-level contradiction"
        if is_mechanical(cand) and "message" in kinds:
            # Phantom-failure findings cite the assistant claim; the paired result is in steps.
            if any(
                0 <= i < len(trace.steps) and trace.steps[i].kind == "tool_result" for i in cand.steps
            ):
                return True, "mechanical phantom failure"
        return False, "need a tool-result quote and a contradicting assistant quote"

    if family == "unsupported_success":
        text = f"{cand.description} {cand.condition}".lower()
        if any(s in text for s in _COMPLETION):
            return True, "factual completion claim"
        return False, "not a factual completed-action claim; a characterisation is not a finding"

    if family in {"redundant_action", "ignored_feedback"}:
        if not (cand.alternative or "").strip():
            return False, _NO_ALTERNATIVE
        if family == "ignored_feedback" and not repeated_same_call_after_error(trace, cand):
            return False, _IGNORED_FEEDBACK_MISFILE
        if family == "redundant_action" and not repeated_identical_call(trace, cand):
            return False, _REDUNDANT_MISFILE
        return True, "alternative named"

    if family == "other":
        return is_mechanical(cand), "other needs mechanical proof or a judge"

    return False, f"no gate for {family}"


def _backfill_quotes(trace: CanonicalTrace, cand: CandidateFinding) -> None:
    if cand.evidence and any(quote_supported(trace, e) for e in cand.evidence):
        return
    for idx in cand.steps:
        if not (0 <= idx < len(trace.steps)):
            continue
        text = (trace.steps[idx].text() or "").strip()
        if len(text) < 12:
            continue
        cand.evidence = [Evidence(step=idx, quote=text[:240])]
        return


def inspect(trace: CanonicalTrace, cand: CandidateFinding) -> Verification:
    if is_mechanical(cand):
        _backfill_quotes(trace, cand)
    verify_candidate(trace, cand)
    located = locatable(trace, cand)
    verified = int(cand.meta.get("quote_verified") or 0)
    total = int(cand.meta.get("quote_total") or 0)
    quotes_ok = total > 0 and verified > 0
    gate_ok, gate_reason = _family_gate(trace, cand) if located else (False, "steps not locatable")
    mechanical = is_mechanical(cand)
    reasons = []
    if not located:
        reasons.append("not locatable")
    if not quotes_ok:
        reasons.append("no verified quote")
    if not gate_ok:
        reasons.append(gate_reason)
    ok = located and quotes_ok and gate_ok
    return Verification(
        locatable=located,
        quotes_ok=quotes_ok,
        family_gate=gate_ok,
        mechanical=mechanical,
        adjudicator="skipped",
        reason="hard-verify passed" if ok else "; ".join(reasons) or gate_reason,
        quote_verified=verified,
        quote_total=total,
    )


_CAPTURE_GUESS = re.compile(
    r"\b(not captured|capture gap|recorder (?:lost|gap|dropped)|"
    r"missing from (?:the )?trace|was(?: not|n't) logged|unpaired (?:call|result)|"
    r"probably (?:never )?(?:called|logged)|guess(?:ing)? (?:that|the))\b",
    re.IGNORECASE,
)


def _looks_like_capture_guess(cand: CandidateFinding) -> bool:
    blob = f"{cand.description} {cand.condition} {cand.available_info}"
    return bool(_CAPTURE_GUESS.search(blob))


def quick_check(trace: CanonicalTrace, cand: CandidateFinding) -> tuple[str, str]:
    """Fast local gates after the judge. Not another model pass.

    Returns (verdict, reason) where verdict is confirmed, insufficient_evidence, or rejected.
    """
    if _looks_like_capture_guess(cand):
        return "insufficient_evidence", "capture-gap guess: abstain instead of inventing an event"
    if not locatable(trace, cand):
        return "insufficient_evidence", "steps not locatable"
    verify_candidate(trace, cand)
    verified = int(cand.meta.get("quote_verified") or 0)
    total = int(cand.meta.get("quote_total") or 0)
    if total == 0 or verified == 0:
        return "insufficient_evidence", "quote not in cited step"
    # One gate, both paths. The judge's verdict does not exempt a finding from the
    # proof its family requires, and the mechanical lane is held to the same bar.
    gate_ok, reason = _family_gate(trace, cand)
    if gate_ok:
        return "confirmed", "quick-check passed"
    if reason in _MISFILED_FAMILY:
        return "rejected", reason
    return "insufficient_evidence", reason
