"""Pack a trace so three uncapped calls stay near the $1.50 mean.

Output tokens are not limited. The mean is held by how much input we send:
requirement lines first, then as many steps as the leftover input budget fits.
"""

from __future__ import annotations

import json
import re

from .schemas import CanonicalStep, CanonicalTrace, CandidateFinding, DerivedCheck

# OpenRouter input list price used for pack sizing (Opus / Sol, Sept 2026).
_INPUT_USD_PER_MTOK = 5.0
_CHARS_PER_TOKEN = 4.0
_REQ = re.compile(
    r"\b(must|must not|never|required|do not|don't|shall not|only use|always|have to)\b",
    re.IGNORECASE,
)


def clip(text: str, limit: int) -> str:
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    keep = max(80, limit - 20)
    return text[:keep] + " [...]"


def distill_instructions(text: str, limit: int) -> str:
    """Keep explicit requirements. Drop playbook prose that is not a rule."""
    text = text or ""
    if not text.strip():
        return ""
    if len(text) <= limit:
        return text
    required = [
        ln.strip() for ln in text.splitlines() if _REQ.search(ln) and ln.strip()
    ]
    if len(required) >= 3:
        body = "\n".join(required)
        if len(body) <= limit:
            return body
        return clip(body, limit)
    return clip(text, limit)


def pack_char_budget(target_usd: float, spent_usd: float = 0.0, copies: int = 2) -> int:
    """Characters of input one pack may use so `copies` sends stay inside target.

    About 38% of the remaining dollars is left for uncapped output / reasoning.
    """
    remaining = max(0.25, float(target_usd) - float(spent_usd))
    input_usd = remaining * 0.62
    tokens = (input_usd * 1_000_000) / (_INPUT_USD_PER_MTOK * max(1, copies))
    return int(max(8_000, min(280_000, tokens * _CHARS_PER_TOKEN)))


def render_tools(trace: CanonicalTrace, limit: int = 12000) -> str:
    rows = []
    for tool in trace.tools:
        try:
            params = (
                json.dumps(tool.parameters, ensure_ascii=False)
                if tool.parameters
                else ""
            )
        except (TypeError, ValueError):
            params = str(tool.parameters)
        rows.append(f"- {tool.name}: {tool.description}\n  params: {params}")
    blob = "\n".join(rows) or "(no declared tools)"
    return clip(blob, limit)


def render_step(step: CanonicalStep, content_limit: int) -> str:
    head = f"[step {step.index}] {step.kind}"
    if step.role:
        head += f" role={step.role}"
    if step.name:
        head += f" {step.name}"
    if step.is_error:
        head += " ERROR"
    if step.call_id:
        head += f" id={step.call_id}"
    if step.arguments is not None:
        try:
            head += " " + json.dumps(step.arguments, ensure_ascii=False)
        except (TypeError, ValueError):
            head += " " + str(step.arguments)
    body = step.content or ""
    if content_limit and len(body) > content_limit:
        body = clip(body, content_limit)
    return head + ("\n" + body if body else "")


def call_outline(trace: CanonicalTrace) -> str:
    lines = []
    for step in trace.steps:
        if step.kind != "tool_call":
            continue
        try:
            args = (
                json.dumps(step.arguments, ensure_ascii=False)
                if step.arguments is not None
                else ""
            )
        except (TypeError, ValueError):
            args = str(step.arguments)
        lines.append(f"  step {step.index} {step.name} {args}".rstrip())
    return "\n".join(lines) or "  (none)"


def info_available_at(trace: CanonicalTrace, index: int, limit: int = 800) -> str:
    prior: list[str] = []
    for step in trace.steps:
        if step.index >= index:
            break
        if step.kind == "tool_result":
            prior.append(f"step {step.index} {step.name} -> {clip(step.content, 120)}")
        elif step.kind == "tool_call":
            prior.append(f"step {step.index} called {step.name}")
        elif step.kind == "message" and step.role == "user":
            prior.append(f"step {step.index} user: {clip(step.content, 80)}")
    return clip("; ".join(prior[-8:]), limit)


def pack_trace(trace: CanonicalTrace, max_chars: int, instruction_chars: int) -> str:
    """Full audit pack. Shrinks per-step content only when the whole pack would overflow."""
    instructions = distill_instructions(trace.instructions, instruction_chars)
    if not instructions.strip():
        instructions = "(none beyond the declared tools and the task text)"
    header = [
        f"TRACE {trace.trace_id}  format={trace.source_format}  steps={len(trace.steps)}",
        "",
        "TASK",
        clip(trace.task, 8000) or "(not provided)",
        "",
        "INSTRUCTIONS (explicit requirements; playbook prose omitted when long)",
        instructions,
        "",
        "DECLARED TOOLS",
        render_tools(trace),
        "",
        "TOOL-CALL OUTLINE (use this to see a required call that never happened)",
        call_outline(trace),
    ]
    if trace.capture_gaps:
        header += [
            "",
            "CAPTURE GAPS",
            "\n".join(f"- {g}" for g in trace.capture_gaps[:12]),
        ]
    prefix = "\n".join(header) + "\n\nSTEPS\n"
    reserved = len(prefix) + 80
    budget = max(4000, max_chars - reserved)
    n = max(1, len(trace.steps))
    per = max(200, budget // n)

    def body(limit: int) -> str:
        return "\n\n".join(render_step(s, limit) for s in trace.steps)

    packed = body(per)
    if len(prefix) + len(packed) > max_chars and per > 200:
        packed = body(max(120, per // 2))
    return prefix + packed


def render_neighborhood(
    trace: CanonicalTrace,
    steps: list[int],
    radius: int = 2,
    content_limit: int = 2000,
) -> str:
    wanted: set[int] = set()
    n = len(trace.steps)
    for raw in steps:
        if not (0 <= raw < n):
            continue
        for i in range(raw - radius, raw + radius + 1):
            if 0 <= i < n:
                wanted.add(i)
    if not wanted:
        return "(no locatable cited steps)"
    return "\n\n".join(
        render_step(trace.steps[i], content_limit) for i in sorted(wanted)
    )


def pack_judge(
    trace: CanonicalTrace,
    checks: list[DerivedCheck],
    proposals: list[CandidateFinding],
    max_chars: int,
    instruction_chars: int,
) -> str:
    check_blob = json.dumps(
        [
            {
                "check_id": c.check_id,
                "family": c.family,
                "description": c.description,
                "condition": c.condition,
                "justified_when": c.justified_when,
                "source_refs": c.source_refs[:3],
                "auditor": c.source,
            }
            for c in checks
        ],
        ensure_ascii=False,
        indent=2,
    )
    proposal_blob = json.dumps(
        [
            {
                "index": i,
                "auditor": p.auditor,
                "check_id": p.check_id,
                "family": p.family,
                "error_class": p.error_class,
                "confidence": p.confidence,
                "decision": p.meta.get("decision"),
                "steps": p.steps,
                "description": p.description,
                "explanation": p.condition,
                "evidence": [{"step": e.step, "quote": e.quote} for e in p.evidence],
                "alternative": p.alternative,
                "rule_ref": p.meta.get("rule_ref") or "",
                "available_info": p.available_info,
            }
            for i, p in enumerate(proposals)
        ],
        ensure_ascii=False,
        indent=2,
    )
    cited = [s for p in proposals for s in p.steps]
    instructions = distill_instructions(
        trace.instructions, min(instruction_chars, 24000)
    )
    outline_block = (
        "TOOL-CALL OUTLINE (use this to see a required call that never happened)\n"
        + call_outline(trace)
    )
    judge_rest = "\n\n".join(
        [
            f"TRACE {trace.trace_id}",
            "TASK\n" + clip(trace.task, 4000),
            "INSTRUCTIONS\n" + (instructions or "(none beyond declared tools)"),
            "DECLARED TOOLS\n" + render_tools(trace, 8000),
            "CHECKS THE AUDITORS DERIVED\n" + clip(check_blob, 20000),
            "NUMBERED PROPOSALS\n" + clip(proposal_blob, 40000),
            "NEIGHBORHOOD AROUND CITED STEPS\n"
            + render_neighborhood(trace, cited, radius=2, content_limit=2000),
        ]
    )
    full = pack_trace(trace, max_chars, instruction_chars)
    if len(full) + len(judge_rest) + 32 <= max_chars:
        return full + "\n\n--- JUDGE MATERIAL ---\n\n" + judge_rest
    # Leftover budget is tight: keep a reserved outline slice so schema/type
    # misses stay visible (the full pack already includes the outline).
    outline_cap = min(len(outline_block), max(240, min(8000, max_chars // 2)))
    outline_kept = clip(outline_block, outline_cap)
    rest_budget = max(400, max_chars - len(outline_kept) - 4)
    rest = (
        judge_rest if len(judge_rest) <= rest_budget else clip(judge_rest, rest_budget)
    )
    return outline_kept + "\n\n" + rest
