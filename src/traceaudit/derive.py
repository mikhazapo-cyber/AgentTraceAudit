from __future__ import annotations

from .config import Config
from .evidence import evidence_index, render_tools, step_outline
from .llm import CallError, LLMClient, LLMResponse
from .obligations import compile_obligations
from .parsing import extract_json
from .prompts import DERIVE_SYSTEM
from .schemas import FAMILIES, CallRecord, CanonicalTrace, DerivedCheck
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


def obligation_checks(trace: CanonicalTrace) -> list[DerivedCheck]:
    """Fallback checks read from THIS trace's own text. Not a bug catalogue."""
    checks: list[DerivedCheck] = [ob.to_check() for ob in compile_obligations(trace)]
    for i, t in enumerate(trace.tools):
        if not t.declared:
            continue
        req = (t.parameters or {}).get("required") or []
        if not req:
            continue
        checks.append(
            DerivedCheck(
                check_id=f"SCHEMA-{i}",
                family="incorrect_tool_use",
                description=f"'{t.name}' must receive required arguments: {', '.join(map(str, req))}",
                condition=f"a call to '{t.name}' omits any of {req} or supplies an empty/wrong-typed value",
                justified_when="the tool was never called, or the schema marks the field optional",
                scope=f"calls to {t.name}",
                severity="major",
                rationale="declared tool schema",
                needs_llm=False,
                source_refs=[t.name],
                lane="mechanical",
            )
        )
    return checks


def split_lanes(checks: list[DerivedCheck]) -> tuple[list[DerivedCheck], list[DerivedCheck]]:
    """Return (obligation lane, residual lane).

    Instruction rules go to the obligation lane so they are judged on their own
    evidence instead of competing for attention in one long mixed checklist.
    Schema checks belong to the mechanical lane and are dropped from both.
    """
    obligation = [c for c in checks if c.lane == "obligation"]
    residual = [c for c in checks if c.lane == "residual"]
    return obligation, residual


def _parse_checks(raw: object) -> list[DerivedCheck]:
    if isinstance(raw, dict):
        items = raw.get("checks") or raw.get("obligations") or []
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    out: list[DerivedCheck] = []
    seen: set[str] = set()
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        family = str(item.get("family") or "other")
        if family not in FAMILIES:
            family = "other"
        cid = str(item.get("check_id") or f"C{i}")
        if cid in seen:
            cid = f"{cid}-{i}"
        seen.add(cid)
        refs = item.get("source_refs") or []
        if isinstance(refs, str):
            refs = [refs]
        out.append(
            DerivedCheck(
                check_id=cid,
                family=family,  # type: ignore[arg-type]
                description=str(item.get("description") or item.get("condition") or cid)[:400],
                condition=str(item.get("condition") or "")[:500],
                justified_when=str(item.get("justified_when") or "")[:400],
                severity=item.get("severity") if item.get("severity") in ("critical", "major", "minor") else "major",
                rationale="derived from task, instructions, and tools",
                needs_llm=True,
                source_refs=[str(r)[:300] for r in refs][:4],
                lane="obligation" if family == "instruction_violation" else "residual",
            )
        )
    return out


def derive_checks(
    trace: CanonicalTrace,
    flags: HeuristicReport,
    cfg: Config,
    client: LLMClient,
) -> tuple[list[DerivedCheck], list[CallRecord]]:
    stubs = obligation_checks(trace)
    if not getattr(client, "available", True):
        return stubs, []
    user = (
        f"TASK:\n{(trace.task or '')[:8000]}\n\n"
        f"INSTRUCTIONS:\n{(trace.instructions or '')[: cfg.max_instruction_chars]}\n\n"
        f"TOOLS:\n{render_tools(trace.tools, 20000)}\n\n"
        f"STEP OUTLINE (compact map):\n{step_outline(trace, 12000)}\n\n"
        f"STEP EXCERPTS (mine testable obligations from what this run touched; "
        f"do not emit findings):\n{evidence_index(trace, cfg.max_trace_chars)}\n\n"
        f"STRUCTURAL FLAGS (hints about what an auditor might inspect, not findings):\n{flags.to_dict()}\n"
    )
    try:
        resp = client.complete(
            [
                {"role": "system", "content": DERIVE_SYSTEM},
                {"role": "user", "content": user},
            ],
            models=cfg.derive.models,
            temperature=cfg.derive.temperature,
            max_tokens=cfg.derive.max_output_tokens,
            stage="derive",
            response_schema={"type": "json_object"},
            reasoning_effort=cfg.derive.reasoning_effort,
        )
    except (CallError, Exception):
        return stubs, []
    calls = [_record("derive", resp)]
    try:
        parsed = extract_json(resp.content)
        derived = _parse_checks(parsed)
    except ValueError:
        return stubs, calls
    have = {c.description[:80].lower() for c in derived}
    for stub in stubs:
        key = stub.source_refs[0][:80].lower() if stub.source_refs else stub.description[:80].lower()
        if not any(key[:40] in h or h[:40] in key for h in have):
            derived.append(stub)
    return derived, calls
