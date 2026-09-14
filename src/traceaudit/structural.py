from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .quotes import unique_evidence
from .schemas import CandidateFinding, CanonicalStep, CanonicalTrace, Evidence, ToolSpec

PROMPT_REPLAY = "prompt_replay"


def is_replay(step: CanonicalStep) -> bool:
    """Earlier-turn steps recovered from a prompt snapshot, not the live capture."""
    return (step.meta or {}).get("provenance") == PROMPT_REPLAY


def format_steps(steps: list[int]) -> str:
    """'4', '4-6', or '4, 9' -- never a raw Python list."""
    ordered = sorted({int(s) for s in steps})
    if not ordered:
        return "none"
    if len(ordered) == 1:
        return str(ordered[0])
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]}-{ordered[-1]}"
    return ", ".join(str(i) for i in ordered)

_WRAPPED_CALL_RE = re.compile(r"^([A-Za-z_][\w.]*)\((.*)\)\s*$")

# Last-resort fallback only. `call_rejected_as_formed` tries the schema and the
# unsupplied-argument test first, since those are structural and language-neutral.
# Keep this list generic: never add phrasing copied from a trace being scored.
_REFUSAL_VOCABULARY = (
    "cannot be executed",
    "failed to parse",
    "unknown parameter",
    "malformed",
    "not a valid",
)
# snake_case, kebab-case, or camelCase - the shape of an argument name, not prose.
_IDENTIFIER_RE = re.compile(
    r"\b([a-z][a-z0-9]*(?:[_\-][a-z0-9]+)+|[a-z]+[A-Z][A-Za-z0-9]*)\b"
)
_REQUIRED_ERR_RE = re.compile(
    r"(?i)\b([\w.]+ required|missing required|(?:is|are) required|"
    r"required (?:argument|parameter|field|property))\b"
)
_RESOURCE_HINTS = (
    "not found",
    "no such",
    "does not exist",
    "enoent",
    "permission denied",
    "unauthorized",
    "forbidden",
    "404",
    "403",
    "401",
)
_JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def schema_unreliable(trace: CanonicalTrace) -> bool:
    """True when the recorder dropped tool schemas, so schema checks are not evidence."""
    for ev in trace.meta.get("loss_events") or []:
        blob = f"{ev.get('reason', '')} {ev.get('detail', '')}".lower()
        if any(tok in blob for tok in ("schema", "parameters", "tool spec", "input_schema")):
            return True
    return False


def logical_tool_name(step: CanonicalStep) -> str:
    """Tool the agent meant, not a harness wrapper such as env_action."""
    name = step.name or ""
    if name == "env_action" and isinstance(step.arguments, dict):
        action = str(step.arguments.get("action") or "").strip()
        match = _WRAPPED_CALL_RE.match(action)
        if match:
            return match.group(1)
    return name


@dataclass
class Flag:
    kind: str
    steps: list[int]
    detail: str
    severity_hint: str = "major"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "steps": self.steps,
            "detail": self.detail,
            "severity_hint": self.severity_hint,
        }


@dataclass
class HeuristicReport:
    flags: list[Flag] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[Flag]:
        return [f for f in self.flags if f.kind == kind]

    def to_dict(self) -> dict:
        return {"flags": [f.to_dict() for f in self.flags]}


def _arg_hash(step: CanonicalStep) -> str:
    try:
        canon = json.dumps(step.arguments or {}, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        canon = str(step.arguments)
    name = logical_tool_name(step) or step.name
    return hashlib.sha256(f"{name}\x00{canon}".encode()).hexdigest()[:16]


def pair_calls_results(trace: CanonicalTrace) -> dict[int, int]:
    pairing: dict[int, int] = {}
    results = [s for s in trace.steps if s.kind == "tool_result"]
    used: set[int] = set()
    by_id: dict[str, list[CanonicalStep]] = {}
    for r in results:
        if r.call_id:
            by_id.setdefault(r.call_id, []).append(r)
    for c in trace.steps:
        if c.kind == "tool_call" and c.call_id:
            bucket = by_id.get(c.call_id)
            if bucket:
                r = bucket.pop(0)
                if r.index not in used:
                    pairing[c.index] = r.index
                    used.add(r.index)
    for c in trace.steps:
        if c.kind != "tool_call" or c.index in pairing:
            continue
        for r in results:
            if r.index in used or r.index <= c.index:
                continue
            cname = logical_tool_name(c) or c.name
            rname = logical_tool_name(r) or r.name
            if cname and rname and cname != rname and c.name != r.name:
                continue
            pairing[c.index] = r.index
            used.add(r.index)
            break
    return pairing


def _result_text(trace: CanonicalTrace, call: CanonicalStep, pairing: dict[int, int]) -> str | None:
    ri = pairing.get(call.index)
    if ri is None or not (0 <= ri < len(trace.steps)):
        return None
    return trace.steps[ri].content


def _is_hollow(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _type_ok(value: Any, expected: str) -> bool:
    if expected not in _JSON_TYPE_MAP:
        return True
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return isinstance(value, _JSON_TYPE_MAP[expected])


def validate_against_schema(args: dict, schema: dict) -> list[str]:
    problems: list[str] = []
    if not isinstance(schema, dict) or not isinstance(args, dict):
        return problems
    if "_raw" in args and len(args) == 1:
        return ["arguments were not valid JSON"]
    for req in schema.get("required") or []:
        if req not in args:
            problems.append(f"missing required argument '{req}'")
        elif _is_hollow(args.get(req)):
            problems.append(f"required argument '{req}' is empty")
    props = schema.get("properties") or {}
    additional = schema.get("additionalProperties", True)
    if additional is False:
        allowed = set(props.keys())
        for k in args:
            if k not in allowed and k != "_raw":
                problems.append(f"unknown argument '{k}'")
    for name, value in args.items():
        if name == "_raw":
            continue
        spec = props.get(name)
        if not isinstance(spec, dict):
            continue
        expected = spec.get("type")
        if isinstance(expected, str) and not _type_ok(value, expected):
            problems.append(f"argument '{name}' should be {expected}, got {type(value).__name__}")
        enum = spec.get("enum")
        if isinstance(enum, list) and enum and value not in enum:
            problems.append(f"argument '{name}'={value!r} not in enum {enum}")
    return problems


def _is_call_formation_error(text: str) -> bool:
    """Prose fallback: did the runtime say it refused the call as written?

    Language-dependent by nature, so `call_rejected_as_formed` consults the
    structural tests first and only falls back here.
    """
    t = (text or "").strip()
    if not t:
        return False
    low = t.lower()
    if any(h in low for h in _RESOURCE_HINTS):
        return False
    if any(h in low for h in _REFUSAL_VOCABULARY):
        return True
    # Weak cues ("required", "invalid") only on short error-shaped replies.
    # Long success prose often contains "the required documents".
    if len(t) > 180:
        return False
    if _REQUIRED_ERR_RE.search(t):
        return True
    return any(h in low for h in ("invalid", "validation error", "unexpected argument", "missing argument"))


def _argument_surface(call: CanonicalStep) -> tuple[set[str], str]:
    args = call.arguments or {}
    keys = {str(k).lower() for k in args}
    try:
        values = json.dumps(args, ensure_ascii=False, default=str).lower()
    except (TypeError, ValueError):
        values = str(args).lower()
    return keys, values


def names_unsupplied_argument(call: CanonicalStep, text: str) -> bool:
    """True when the runtime names an argument the call never supplied.

    Structural and language-independent: argument names are code identifiers, so
    this works regardless of the prose around them. A runtime echoing a value we
    sent (a path, an id) is describing a missing resource, not a malformed call,
    so tokens already present in the arguments are ignored.
    """
    body = (text or "").strip()
    if not body:
        return False
    if any(h in body.lower() for h in _RESOURCE_HINTS):
        return False
    keys, values = _argument_surface(call)
    for token in set(_IDENTIFIER_RE.findall(body)):
        low = token.lower()
        if low in keys or low in values:
            continue
        return True
    return False


def call_rejected_as_formed(
    trace: CanonicalTrace, call: CanonicalStep, result: CanonicalStep
) -> tuple[bool, str]:
    """Did the environment refuse this call because of how it was written?

    Ordered structural first: a declared schema settles it without reading any
    prose, then the unsupplied-argument test, then the refusal vocabulary.
    """
    schema = trace.schema_for(call.name) or trace.schema_for(logical_tool_name(call))
    if schema and not schema_unreliable(trace):
        problems = validate_against_schema(call.arguments or {}, schema)
        if problems:
            return True, problems[0]
    if result.is_error and names_unsupplied_argument(call, result.content):
        return True, "the runtime named an argument the call did not supply"
    if (result.meta or {}).get("error_kind") == "formation":
        return True, "the runtime refused the call as written"
    if _is_call_formation_error(result.content):
        return True, "the runtime reported the call could not be executed as formed"
    return False, ""


def analyze(trace: CanonicalTrace) -> HeuristicReport:
    report = HeuristicReport()
    # Replay steps are earlier turns recovered from a prompt snapshot. The
    # bundle's tool list is for the captured turn, so schema / undeclared-tool
    # checks against them are unsound. Auditors still see the full trace.
    calls = [s for s in trace.steps if s.kind == "tool_call" and not is_replay(s)]
    pairing = pair_calls_results(trace)
    groups: dict[str, list[CanonicalStep]] = {}
    for c in calls:
        groups.setdefault(_arg_hash(c), []).append(c)
    for group in groups.values():
        if len(group) < 2:
            continue
        results = [_result_text(trace, c, pairing) for c in group]
        same_result = all(r is not None for r in results) and len(set(results)) == 1
        first_ok = False
        ri0 = pairing.get(group[0].index)
        if ri0 is not None and 0 <= ri0 < len(trace.steps):
            first_ok = not trace.steps[ri0].is_error
        detail = (
            f"tool '{logical_tool_name(group[0]) or group[0].name}' called "
            f"{len(group)} times with identical arguments at steps {format_steps([c.index for c in group])}"
        )
        if same_result:
            detail += "; every call returned an identical result"
        report.flags.append(
            Flag(
                kind="duplicate_call",
                steps=[c.index for c in group],
                detail=detail,
                severity_hint="major" if same_result and first_ok else "minor",
            )
        )
        if same_result and first_ok:
            report.flags.append(
                Flag(
                    kind="identical_success_repeat",
                    steps=[c.index for c in group],
                    detail=detail + " (first call already succeeded)",
                    severity_hint="major",
                )
            )
    declared = trace.declared_tool_names()
    specs = {
        t.name: t.parameters
        for t in trace.tools
        if t.declared and t.parameters and not schema_unreliable(trace)
    }
    if declared:
        for c in calls:
            if is_replay(c):
                continue
            logical = logical_tool_name(c)
            if c.name and c.name not in declared and logical not in declared:
                # env_action / bash wrappers are declared even when they wrap a named tool
                if c.name in {"env_action", "bash", "finish"}:
                    continue
                report.flags.append(
                    Flag(
                        kind="unknown_tool",
                        steps=[c.index],
                        detail=f"step {c.index} calls '{c.name}' which is not among the {len(declared)} declared tools",
                        severity_hint="critical",
                    )
                )
    for c in calls:
        if is_replay(c):
            continue
        schema = specs.get(c.name)
        if not schema:
            continue
        problems = validate_against_schema(c.arguments or {}, schema)
        if problems:
            report.flags.append(
                Flag(
                    kind="schema_violation",
                    steps=[c.index],
                    detail=f"step {c.index} call to '{c.name}': " + "; ".join(problems),
                    severity_hint="major",
                )
            )
    for ci, ri in pairing.items():
        if not (0 <= ri < len(trace.steps)):
            continue
        result = trace.steps[ri]
        call = trace.steps[ci]
        if is_replay(call) or is_replay(result):
            continue
        rejected, why = call_rejected_as_formed(trace, call, result)
        if not rejected:
            continue
        name = logical_tool_name(call)
        later_same = any(
            s.kind == "tool_call" and s.index > ci and logical_tool_name(s) == name
            for s in trace.steps
        )
        report.flags.append(
            Flag(
                kind="call_formation_error",
                steps=[ci, ri],
                detail=(
                    f"step {ci} call to '{name}' failed as formed ({why}): "
                    f"{result.content[:140]!r}"
                    + ("; agent later used the same tool again" if later_same else "")
                ),
                severity_hint="minor" if later_same else "major",
            )
        )
    for flag in _phantom_failures(trace, pairing):
        report.flags.append(flag)
    for flag in _ignored_same_args(trace, pairing):
        report.flags.append(flag)
    return report


def _ignored_same_args(trace: CanonicalTrace, pairing: dict[int, int]) -> list[Flag]:
    """Same tool+args after a call-formation / required-argument error."""
    flags: list[Flag] = []
    calls = [s for s in trace.steps if s.kind == "tool_call" and not is_replay(s)]
    by_key: dict[str, list[CanonicalStep]] = {}
    for c in calls:
        by_key.setdefault(_arg_hash(c), []).append(c)
    for group in by_key.values():
        if len(group) < 2:
            continue
        first = group[0]
        ri = pairing.get(first.index)
        if ri is None:
            continue
        result = trace.steps[ri]
        if not (result.is_error or _is_call_formation_error(result.content)):
            continue
        if not _is_call_formation_error(result.content):
            continue
        later = [c.index for c in group[1:]]
        flags.append(
            Flag(
                kind="ignored_feedback",
                steps=[first.index, ri] + later,
                detail=(
                    f"tool '{logical_tool_name(first) or first.name}' retried with identical "
                    f"arguments after a formation/validation error at step {ri}"
                ),
                severity_hint="major",
            )
        )
    return flags


def _phantom_failures(trace: CanonicalTrace, pairing: dict[int, int]) -> list[Flag]:
    """An agent that redoes work it already succeeded at, having said something in between.

    Structural, with no phrase list: a call succeeds, the agent speaks, and then
    the agent re-issues byte-identical arguments to the same tool. Re-running a
    call whose result you already hold only makes sense if you believe it failed,
    so the message in between contradicts the captured result. What the message
    actually says is left to the auditors; this only marks where to look.
    """
    flags: list[Flag] = []
    groups: dict[str, list[CanonicalStep]] = {}
    for step in trace.steps:
        if step.kind == "tool_call" and not is_replay(step):
            groups.setdefault(_arg_hash(step), []).append(step)

    for group in groups.values():
        if len(group) < 2:
            continue
        first = group[0]
        ri = pairing.get(first.index)
        if ri is None or not (0 <= ri < len(trace.steps)):
            continue
        result = trace.steps[ri]
        if result.is_error or not (result.content or "").strip():
            continue
        repeat = group[1]
        spoken = [
            s.index
            for s in trace.steps
            if s.kind == "message"
            and s.role == "assistant"
            and ri < s.index < repeat.index
            and (s.content or "").strip()
        ]
        if not spoken:
            continue
        name = logical_tool_name(first) or first.name
        flags.append(
            Flag(
                kind="phantom_failure",
                steps=[ri] + spoken,
                detail=(
                    f"step {ri} already returned a successful result for '{name}', the assistant "
                    f"then spoke at step(s) {format_steps(spoken)}, and step {repeat.index} re-issued the "
                    f"identical call - the intervening message treats a captured success as a failure"
                ),
                severity_hint="minor",
            )
        )
    return flags


def _quote(step: CanonicalStep, limit: int = 240) -> str:
    return (step.text().strip() or "")[:limit]


def objective_findings(trace: CanonicalTrace, flags: HeuristicReport) -> list[CandidateFinding]:
    by_step = {s.index: s for s in trace.steps}
    out: list[CandidateFinding] = []
    out.extend(_group_flags(trace, flags, by_step, "unknown_tool", "incorrect_tool_use"))
    out.extend(_group_flags(trace, flags, by_step, "schema_violation", "incorrect_tool_use"))
    out.extend(_call_formation_findings(trace, flags, by_step))
    out.extend(_redundant_from_repeats(trace, flags, by_step))
    out.extend(_phantom_findings(trace, flags, by_step))
    out.extend(_ignored_feedback_findings(trace, flags, by_step))
    return out


def _ignored_feedback_findings(
    trace: CanonicalTrace, flags: HeuristicReport, by_step: dict[int, CanonicalStep]
) -> list[CandidateFinding]:
    out = []
    for flag in flags.by_kind("ignored_feedback"):
        steps = sorted(set(flag.steps))
        call = next((by_step[i] for i in steps if i in by_step and by_step[i].kind == "tool_call"), None)
        name = logical_tool_name(call) if call else "?"
        quote = _quote(call) if call else flag.detail
        ev = []
        if quote and call:
            ev.append(Evidence(step=call.index, quote=quote, source="tool_call"))
        out.append(
            CandidateFinding(
                check_id="DET-ignored_feedback",
                family="ignored_feedback",
                description=flag.detail[:500],
                condition="identical retry after a validation/formation error adds no information and ignores the error",
                steps=steps,
                evidence=unique_evidence(ev),
                severity="major",
                confidence=0.9,
                source="deterministic",
                alternative=f"Change the arguments of '{name}' in response to the error instead of repeating the same call.",
            )
        )
    return out


def _group_flags(
    trace: CanonicalTrace,
    flags: HeuristicReport,
    by_step: dict[int, CanonicalStep],
    kind: str,
    family: str,
) -> list[CandidateFinding]:
    groups: dict[str, dict] = {}
    order: list[str] = []
    for flag in flags.by_kind(kind):
        call_idx = next((i for i in flag.steps if by_step.get(i) and by_step[i].kind == "tool_call"), flag.steps[0])
        step = by_step.get(call_idx)
        if step is None:
            continue
        name = logical_tool_name(step) or step.name or "?"
        if name not in groups:
            groups[name] = {"steps": [], "flag": flag, "step": step}
            order.append(name)
        groups[name]["steps"].extend(flag.steps)
    n_tools = len(trace.declared_tool_names()) or len(trace.tool_names())
    out = []
    for name in order:
        g = groups[name]
        steps = sorted(set(g["steps"]))
        if kind == "unknown_tool":
            desc = (
                f"tool '{name}' called {len([i for i in steps if by_step.get(i) and by_step[i].kind == 'tool_call'])} "
                f"time(s) but is not among the {n_tools} declared tools (steps {format_steps(steps)})"
            )
            condition = "called a tool that is not in the declared inventory"
        else:
            tail = g["flag"].detail.split(": ", 1)[-1]
            desc = f"call(s) to '{name}' violate its declared schema (steps {format_steps(steps)}): {tail}"
            condition = "declared tool schema / required arguments"
        quote = _quote(g["step"])
        ev = (
            [Evidence(step=g["step"].index, quote=quote, source="tool_call")] if quote else []
        )
        out.append(
            CandidateFinding(
                check_id=f"DET-{kind}",
                family=family,  # type: ignore[arg-type]
                description=desc[:500],
                condition=condition,
                steps=steps,
                evidence=unique_evidence(ev),
                severity="major" if g["flag"].severity_hint != "minor" else "minor",
                confidence=1.0,
                source="deterministic",
            )
        )
    return out


def _call_formation_findings(
    trace: CanonicalTrace, flags: HeuristicReport, by_step: dict[int, CanonicalStep]
) -> list[CandidateFinding]:
    out = []
    for flag in flags.by_kind("call_formation_error"):
        steps = sorted(set(flag.steps))
        call = next((by_step[i] for i in steps if i in by_step and by_step[i].kind == "tool_call"), None)
        name = logical_tool_name(call) if call else "?"
        quote = _quote(call) if call else flag.detail
        result = next((by_step[i] for i in steps if i in by_step and by_step[i].kind == "tool_result"), None)
        evidence = []
        if quote:
            evidence.append(
                Evidence(step=call.index if call else steps[0], quote=quote, source="tool_call")
            )
        if result:
            evidence.append(Evidence(step=result.index, quote=_quote(result), source="tool_result"))
        evidence = unique_evidence(evidence)
        out.append(
            CandidateFinding(
                check_id="DET-call_formation_error",
                family="incorrect_tool_use",
                description=flag.detail[:500],
                condition="tool result reports the call could not be executed as formed",
                steps=steps,
                evidence=evidence,
                severity="minor" if flag.severity_hint == "minor" else "major",
                confidence=1.0,
                source="deterministic",
                alternative=f"Form the '{name}' call with an argument the tool accepts (an entity or variable, not a type/category) before retrying.",
            )
        )
    return out


def _redundant_from_repeats(
    trace: CanonicalTrace, flags: HeuristicReport, by_step: dict[int, CanonicalStep]
) -> list[CandidateFinding]:
    by_tool: dict[str, list[int]] = {}
    first_by_tool: dict[str, int] = {}
    for flag in flags.by_kind("identical_success_repeat"):
        if len(flag.steps) < 2:
            continue
        first = by_step.get(flag.steps[0])
        name = (logical_tool_name(first) if first else "") or (first.name if first else "?")
        wasted = flag.steps[1:]
        by_tool.setdefault(name, []).extend(wasted)
        first_by_tool.setdefault(name, flag.steps[0])
    out = []
    for name, wasted in by_tool.items():
        steps = sorted(set(wasted))
        first = first_by_tool.get(name, steps[0])
        step = by_step.get(steps[0])
        quote = _quote(step) if step else ""
        out.append(
            CandidateFinding(
                check_id="DET-duplicate_call",
                family="redundant_action",
                description=(
                    f"tool '{name}' re-invoked with identical arguments and identical results "
                    f"at steps {format_steps(steps)}; the earlier successful results could have been reused"
                ),
                condition="identical calls with identical successful results add no information after the first",
                steps=steps,
                evidence=unique_evidence(
                    [Evidence(step=steps[0], quote=quote, source="tool_call")] if quote else []
                ),
                severity="major",
                confidence=1.0,
                source="deterministic",
                alternative=f"Reuse the result from step {first} instead of re-invoking {name}.",
            )
        )
    return out


def _phantom_findings(
    trace: CanonicalTrace, flags: HeuristicReport, by_step: dict[int, CanonicalStep]
) -> list[CandidateFinding]:
    steps: list[int] = []
    quotes: list[Evidence] = []
    for flag in flags.by_kind("phantom_failure"):
        steps.extend(flag.steps)
        for idx in flag.steps:
            st = by_step.get(idx)
            if st and st.kind in {"message", "tool_result"}:
                src = "tool_result" if st.kind == "tool_result" else "agent_message"
                quotes.append(Evidence(step=idx, quote=_quote(st), source=src))
    if not steps:
        return []
    steps = sorted(set(steps))
    quotes = unique_evidence(quotes)
    spoken = [e.step for e in quotes if by_step.get(e.step) and by_step[e.step].kind == "message"]
    return [
        CandidateFinding(
            check_id="DET-phantom_failure",
            family="evidence_contradiction",
            description=(
                f"The assistant spoke at step(s) {format_steps(spoken)} between a successful tool result and a "
                f"byte-identical re-issue of the same call, treating a captured success as a failure"
            ),
            condition="work already succeeded was redone after an intervening assistant message",
            steps=steps,
            evidence=quotes[:4],
            severity="minor",
            confidence=0.95,
            source="deterministic",
        )
    ]


def tool_call_violations(step: CanonicalStep, tools: list[ToolSpec]) -> list[str]:
    if step.kind != "tool_call":
        return []
    spec = next((t for t in tools if t.name == step.name), None)
    if spec is None:
        return [f"tool '{step.name}' is not declared"]
    return validate_against_schema(step.arguments or {}, spec.parameters or {})
