"""Second-pass review that spends leftover budget on unused checks and abstentions."""

from __future__ import annotations

from .config import Config
from .detect import parse_agent_findings
from .evidence import evidence_index, info_available_at, render_tools
from .llm import CallError, LLMClient, LLMResponse
from .parsing import extract_json
from .prompts import CRITIQUE_SYSTEM
from .retrieve import neighborhood_indices, render_retrieved, retrieve_step_indices
from .schemas import CallRecord, CandidateFinding, CanonicalTrace, DerivedCheck, Finding


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


def should_critique(
    cfg: Config,
    spend: float,
    checks: list[DerivedCheck],
    findings: list[Finding],
    proposals: list[CandidateFinding],
) -> bool:
    """Run only while the trace is still under the $1.50 quality target."""
    if not getattr(cfg, "critique_enabled", True):
        return False
    if cfg.cost_cap_usd > 0 and spend >= cfg.cost_cap_usd:
        return False
    if spend >= cfg.target_mean_usd:
        return False
    if checks or findings or proposals:
        return True
    return False


def _unused_checks(checks: list[DerivedCheck], findings: list[Finding]) -> list[DerivedCheck]:
    taken = {f.check_id for f in findings if f.status == "confirmed" and f.check_id}
    return [c for c in checks if c.check_id not in taken]


def _disagreements(proposals: list[CandidateFinding]) -> list[CandidateFinding]:
    by_key: dict[tuple, list[CandidateFinding]] = {}
    for cand in proposals:
        key = (cand.family, tuple(sorted(cand.steps)), cand.check_id)
        by_key.setdefault(key, []).append(cand)
    out = []
    for group in by_key.values():
        agents = {str(c.meta.get("agent") or "") for c in group}
        if len(agents) >= 2:
            out.extend(group)
    return out


def _overlap_confirmed(cand: CandidateFinding, confirmed: list[Finding]) -> bool:
    steps = set(cand.steps)
    if not steps:
        return False
    for f in confirmed:
        if f.family == cand.family and steps & set(f.steps):
            return True
    return False


def critique_review(
    trace: CanonicalTrace,
    checks: list[DerivedCheck],
    proposals: list[CandidateFinding],
    findings: list[Finding],
    cfg: Config,
    client: LLMClient,
    spend: float,
) -> tuple[list[CandidateFinding], list[CallRecord]]:
    """Review unused checks, abstentions, and disagreements. May return new candidates."""
    if not should_critique(cfg, spend, checks, findings, proposals):
        return [], []
    if not getattr(client, "available", True):
        return [], []

    unused = _unused_checks(checks, findings)
    abstentions = [f for f in findings if f.status == "insufficient_evidence"]
    disagreed = _disagreements(proposals)
    seeds: list[int] = []
    for item in list(abstentions) + list(disagreed):
        seeds.extend(item.steps)
    local = render_retrieved(
        trace,
        neighborhood_indices(
            trace,
            seeds + retrieve_step_indices(trace, unused, max_steps=48),
            radius=1,
        ),
        char_budget=min(80000, max(24000, cfg.max_trace_chars // 2)),
    )

    unused_blob = [
        {
            "check_id": c.check_id,
            "family": c.family,
            "description": c.description,
            "condition": c.condition,
            "justified_when": c.justified_when,
            "source_refs": c.source_refs,
        }
        for c in unused[:24]
    ]
    abs_blob = []
    for f in abstentions[:12]:
        abs_blob.append(
            {
                "finding_id": f.finding_id,
                "family": f.family,
                "description": f.description,
                "steps": f.steps,
                "reason": f.adjudication_reason or f.verification.reason,
                "available_info": info_available_at(trace, f.steps[0], 2000) if f.steps else "",
            }
        )
    disagree_blob = [
        {
            "agent": c.meta.get("agent", ""),
            "family": c.family,
            "check_id": c.check_id,
            "description": c.description,
            "steps": c.steps,
            "decision": c.meta.get("decision", ""),
        }
        for c in disagreed[:12]
    ]

    user = (
        f"TASK:\n{(trace.task or '')[:8000]}\n\n"
        f"INSTRUCTIONS:\n{(trace.instructions or '')[: cfg.max_instruction_chars]}\n\n"
        f"TOOLS:\n{render_tools(trace.tools, 20000)}\n\n"
        f"UNUSED DERIVED CHECKS (work through each):\n{unused_blob}\n\n"
        f"ABSTENTIONS FROM THE FIRST PASS:\n{abs_blob or ['none']}\n\n"
        f"AUDITOR DISAGREEMENTS:\n{disagree_blob or ['none']}\n\n"
        f"EVIDENCE INDEX:\n{evidence_index(trace, cfg.max_trace_chars)}\n\n"
        f"LOCAL EXCERPTS AROUND OPEN QUESTIONS:\n{local}\n"
    )
    stage_cfg = cfg.critique
    try:
        resp = client.complete(
            [
                {"role": "system", "content": CRITIQUE_SYSTEM},
                {"role": "user", "content": user},
            ],
            models=stage_cfg.models,
            temperature=stage_cfg.temperature,
            max_tokens=stage_cfg.max_output_tokens,
            stage="critique",
            response_schema={"type": "json_object"},
            reasoning_effort=stage_cfg.reasoning_effort,
        )
    except (CallError, Exception):
        return [], []

    calls = [_record("critique", resp)]
    try:
        parsed = extract_json(resp.content)
    except ValueError:
        return [], calls
    extra = parse_agent_findings(parsed, unused + [c for c in checks if c not in unused])
    confirmed = [f for f in findings if f.status == "confirmed"]
    kept = []
    for cand in extra:
        if _overlap_confirmed(cand, confirmed):
            continue
        cand.meta["agent"] = "critique"
        kept.append(cand)
    return kept, calls
