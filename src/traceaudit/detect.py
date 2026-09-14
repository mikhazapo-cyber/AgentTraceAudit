from __future__ import annotations

from .config import Config, StageConfig
from .evidence import evidence_index, render_tools
from .llm import CallError, LLMClient, LLMResponse
from .parsing import extract_json
from .retrieve import render_retrieved, retrieve_step_indices
from .schemas import FAMILIES, CallRecord, CandidateFinding, CanonicalTrace, DerivedCheck, Evidence
from .structural import HeuristicReport


def _record(stage: str, resp: LLMResponse) -> CallRecord:
    return CallRecord(
        stage=stage,
        model=resp.model,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        cost_usd=resp.cost_usd,
        latency_s=resp.latency_s,
        cached=resp.cached,
    )


def _as_finding(item: dict, default_family: str = "other") -> CandidateFinding | None:
    family = str(item.get("family") or default_family)
    if family not in FAMILIES:
        family = "other"
    steps = item.get("steps") or []
    if isinstance(steps, int):
        steps = [steps]
    steps = [int(s) for s in steps if isinstance(s, (int, float))]
    ev = []
    for e in item.get("evidence") or []:
        if isinstance(e, dict) and "step" in e:
            ev.append(
                Evidence(
                    step=int(e["step"]),
                    quote=str(e.get("quote") or "")[:400],
                    source=str(e.get("source") or ""),
                )
            )
    desc = str(item.get("description") or item.get("explanation") or "").strip()
    if not desc:
        return None
    return CandidateFinding(
        check_id=str(item.get("check_id") or ""),
        family=family,  # type: ignore[arg-type]
        description=desc[:500],
        condition=str(item.get("explanation") or item.get("rule_ref") or "")[:500],
        steps=steps,
        evidence=ev,
        severity=item.get("severity") if item.get("severity") in ("critical", "major", "minor") else "major",
        confidence=0.6,
        alternative=str(item.get("alternative") or ""),
        source="llm",
        available_info=str(item.get("available_info") or ""),
        meta={"rule_ref": str(item.get("rule_ref") or ""), "decision": item.get("decision")},
    )


def _iter_items(parsed: object) -> list[dict]:
    if isinstance(parsed, dict):
        items = parsed.get("findings") or parsed.get("verdicts") or parsed.get("additional") or []
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


def parse_agent_findings(parsed: object, checks: list[DerivedCheck]) -> list[CandidateFinding]:
    """Keep only explicit violated and insufficient_evidence rows."""
    by_id = {c.check_id: c for c in checks}
    out: list[CandidateFinding] = []
    for item in _iter_items(parsed):
        decision = str(item.get("decision") or "").lower()
        if decision not in {"violated", "insufficient_evidence"}:
            continue
        check = by_id.get(str(item.get("check_id") or ""))
        if check and not item.get("family"):
            item["family"] = check.family
        cand = _as_finding(item, default_family=check.family if check else "other")
        if cand is None:
            continue
        if check:
            cand.check_id = check.check_id
            if check.source_refs:
                cand.meta["source_refs"] = check.source_refs
            if not cand.meta.get("rule_ref") and check.source_refs:
                cand.meta["rule_ref"] = check.source_refs[0]
        cand.meta["decision"] = decision
        out.append(cand)
    return out


def _blob(checks: list[DerivedCheck]) -> list[dict]:
    return [
        {
            "check_id": c.check_id,
            "family": c.family,
            "description": c.description,
            "condition": c.condition,
            "justified_when": c.justified_when,
            "source_refs": c.source_refs,
        }
        for c in checks
    ]


def _agent_user(
    trace: CanonicalTrace,
    checks: list[DerivedCheck],
    flags: HeuristicReport,
    cfg: Config,
    extra_context: str = "",
    secondary_checks: list[DerivedCheck] | None = None,
) -> str:
    secondary = list(secondary_checks or [])
    retrieved = render_retrieved(
        trace,
        retrieve_step_indices(trace, checks + secondary, max_steps=64),
        min(120000, max(32000, cfg.max_trace_chars // 2)),
    )
    extra = f"{extra_context}\n\n" if extra_context.strip() else ""
    also = ""
    if secondary:
        also = (
            "ALSO DERIVED, OWNED BY THE OTHER AUDITOR (secondary — raise one of these "
            "only if the trace clearly violates it):\n"
            f"{_blob(secondary)}\n\n"
        )
    return (
        f"TASK:\n{(trace.task or '')[:8000]}\n\n"
        f"INSTRUCTIONS:\n{(trace.instructions or '')[: cfg.max_instruction_chars]}\n\n"
        f"TOOLS:\n{render_tools(trace.tools, 20000)}\n\n"
        f"CAPTURE GAPS:\n{trace.capture_gaps or ['none recorded']}\n\n"
        f"STRUCTURAL FLAGS (hints only — do not treat as confirmed findings):\n{flags.to_dict()}\n\n"
        f"{extra}"
        f"DERIVED AUDIT INSTRUCTIONS (yours to work through):\n{_blob(checks)}\n\n"
        f"{also}"
        f"EVIDENCE INDEX (what the agent saw, by step):\n"
        f"{evidence_index(trace, cfg.max_trace_chars)}\n\n"
        f"FULLER EXCERPTS (retrieved for steps that can prove the checks):\n{retrieved}\n"
    )


def run_agent(
    trace: CanonicalTrace,
    checks: list[DerivedCheck],
    flags: HeuristicReport,
    cfg: Config,
    client: LLMClient,
    *,
    stage: str,
    system: str,
    stage_cfg: StageConfig,
    extra_context: str = "",
    secondary_checks: list[DerivedCheck] | None = None,
) -> tuple[list[CandidateFinding], list[CallRecord]]:
    if not getattr(client, "available", True):
        return [], []
    try:
        resp = client.complete(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": _agent_user(
                        trace, checks, flags, cfg, extra_context, secondary_checks
                    ),
                },
            ],
            models=stage_cfg.models,
            temperature=stage_cfg.temperature,
            max_tokens=stage_cfg.max_output_tokens,
            stage=stage,
            response_schema={"type": "json_object"},
            reasoning_effort=stage_cfg.reasoning_effort,
        )
    except (CallError, Exception) as exc:
        return [], [
            CallRecord(stage=stage, model=stage_cfg.models[0] if stage_cfg.models else "", error=str(exc))
        ]
    calls = [_record(stage, resp)]
    try:
        parsed = extract_json(resp.content)
    except ValueError:
        return [], calls
    findings = parse_agent_findings(parsed, checks + list(secondary_checks or []))
    for cand in findings:
        cand.meta["agent"] = stage
    return findings, calls
