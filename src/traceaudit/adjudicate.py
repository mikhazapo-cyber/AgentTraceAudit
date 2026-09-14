from __future__ import annotations

from .config import Config
from .evidence import info_available_at
from .llm import CallError, LLMClient, LLMResponse
from .parsing import extract_json
from .prompts import FALSIFY_SYSTEM, JUDGE_SYSTEM
from .retrieve import neighborhood_indices, render_retrieved
from .schemas import CallRecord, CandidateFinding, CanonicalTrace, DerivedCheck


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


def _excerpt(trace: CanonicalTrace, cand: CandidateFinding, limit: int = 16, clip: int = 2400) -> str:
    seeds = list(cand.steps[:limit])
    around = neighborhood_indices(trace, seeds, radius=1)
    return render_retrieved(trace, around, char_budget=24000, arg_limit=2400, content_limit=clip)


def _parse_verdicts(parsed: object, n: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = [("insufficient_evidence", "unparsable judge output")] * n
    if isinstance(parsed, dict) and isinstance(parsed.get("verdicts"), list):
        items = parsed["verdicts"]
    elif isinstance(parsed, dict) and parsed.get("verdict"):
        items = [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        return out
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index", i))
        except (TypeError, ValueError):
            idx = i
        if not (0 <= idx < n):
            idx = i if i < n else 0
        verdict = str(item.get("verdict") or "").lower()
        reason = str(item.get("reason") or "")
        if verdict == "not_a_problem":
            verdict = "rejected"
        if verdict not in {"confirmed", "rejected", "insufficient_evidence"}:
            verdict = "insufficient_evidence"
            reason = reason or "invalid judge verdict"
        out[idx] = (verdict, reason)
    return out


def judge_findings(
    trace: CanonicalTrace,
    checks: list[DerivedCheck],
    proposals: list[CandidateFinding],
    cfg: Config,
    client: LLMClient,
    spend: float,
) -> tuple[list[tuple[str, str]], list[CallRecord]]:
    """Return (verdict, reason) per proposal plus call records."""
    if not proposals:
        return [], []
    if not cfg.judge_enabled or not getattr(client, "available", True):
        return [("insufficient_evidence", "judge skipped")] * len(proposals), []

    # Hard abort only. target_mean_usd is a target, not a stop: a $1.70 trace must
    # still reach the judge. Skip only once spend passes the cap.
    if cfg.cost_cap_usd > 0 and spend > cfg.cost_cap_usd:
        return [("insufficient_evidence", "per-trace cost cap already exceeded")] * len(proposals), []

    check_blob = [
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
    blocks = []
    for i, cand in enumerate(proposals):
        first = cand.steps[0] if cand.steps else 0
        agent = cand.meta.get("agent", "")
        decision = cand.meta.get("decision", "")
        blocks.append(
            f"CANDIDATE {i} (index={i}) agent={agent} decision={decision}\n"
            f"FAMILY: {cand.family}\n"
            f"CHECK_ID: {cand.check_id}\n"
            f"DESCRIPTION: {cand.description}\n"
            f"EXPLANATION: {cand.condition}\n"
            f"STEPS: {cand.steps}\n"
            f"ALTERNATIVE: {cand.alternative}\n"
            f"RULE: {cand.meta.get('rule_ref', '')}\n"
            f"AVAILABLE INFO AT FIRST CITED STEP:\n{info_available_at(trace, first, 4000)}\n"
            f"CITED STEPS AND NEIGHBOURS:\n{_excerpt(trace, cand)}\n"
        )
    local_seeds = [idx for cand in proposals for idx in cand.steps]
    local_pack = render_retrieved(
        trace,
        neighborhood_indices(trace, local_seeds, radius=1),
        char_budget=48000,
    )
    user = (
        f"DERIVED AUDIT INSTRUCTIONS:\n{check_blob}\n\n"
        f"CAPTURE GAPS:\n{trace.capture_gaps or ['none recorded']}\n\n"
        f"LOCAL EVIDENCE AROUND CITED STEPS:\n{local_pack}\n\n"
        f"Decide a verdict for each candidate. Use the candidate index shown.\n\n"
        + "\n---\n".join(blocks)
    )
    try:
        resp = client.complete(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            models=cfg.judge.models,
            temperature=cfg.judge.temperature,
            max_tokens=cfg.judge.max_output_tokens,
            stage="judge",
            response_schema={"type": "json_object"},
            reasoning_effort=cfg.judge.reasoning_effort,
        )
    except (CallError, Exception) as exc:
        return [("insufficient_evidence", f"judge error: {exc}")] * len(proposals), []

    try:
        parsed = extract_json(resp.content)
        verdicts = _parse_verdicts(parsed, len(proposals))
    except (ValueError, AttributeError):
        verdicts = [("insufficient_evidence", "unparsable judge output") for _ in proposals]
    return verdicts, [_record("judge", resp)]


def _parse_falsify(parsed: object, n: int) -> list[tuple[str, str]]:
    """stands by default so a mute or 'confirmed' reply cannot silently drop recall."""
    out: list[tuple[str, str]] = [("stands", "")] * n
    if isinstance(parsed, dict) and isinstance(parsed.get("verdicts"), list):
        items = parsed["verdicts"]
    elif isinstance(parsed, list):
        items = parsed
    else:
        return out
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index", i))
        except (TypeError, ValueError):
            idx = i
        if not (0 <= idx < n):
            continue
        verdict = str(item.get("verdict") or "").lower()
        reason = str(item.get("reason") or "")
        if verdict in {"disproved", "rejected", "not_a_problem"}:
            out[idx] = ("disproved", reason or "falsifier found a concrete disproof")
        elif verdict == "insufficient_evidence":
            out[idx] = ("insufficient_evidence", reason or "falsifier could not settle")
        else:
            out[idx] = ("stands", reason)
    return out


def falsify_candidates(
    trace: CanonicalTrace,
    proposals: list[CandidateFinding],
    cfg: Config,
    client: LLMClient,
    spend: float,
) -> tuple[list[tuple[str, str]], list[CallRecord]]:
    """Try to disprove surviving LLM findings. Returns (verdict, reason) per item.

    Verdict is stands | disproved | insufficient_evidence. Only an explicit
    disproof downgrades; a mute or 'confirmed' reply leaves the finding up.
    """
    if not proposals or not getattr(cfg, "falsify_enabled", True):
        return [("stands", "")] * len(proposals), []
    if not getattr(client, "available", True):
        return [("stands", "")] * len(proposals), []
    if cfg.cost_cap_usd > 0 and spend > cfg.cost_cap_usd:
        return [("stands", "per-trace cost cap already exceeded")] * len(proposals), []

    cap = max(0, int(getattr(cfg, "max_falsify_per_trace", 3) or 0))
    if cap == 0:
        return [("stands", "")] * len(proposals), []

    def _corroboration(i: int) -> int:
        a = proposals[i]
        sa = set(a.steps)
        return sum(
            1
            for b in proposals
            if b.family == a.family and sa and sa & set(b.steps)
        )

    # Prefer findings a single auditor raised. Still fill the cap if only
    # corroborated findings remain.
    order = sorted(range(len(proposals)), key=lambda i: (_corroboration(i), i))
    chosen = set(order[:cap])
    # Re-sort chosen back into original index order for the prompt.
    chosen_idx = [i for i in range(len(proposals)) if i in chosen]
    if not chosen_idx:
        return [("stands", "")] * len(proposals), []

    blocks = []
    for pos, i in enumerate(chosen_idx):
        cand = proposals[i]
        first = cand.steps[0] if cand.steps else 0
        blocks.append(
            f"CANDIDATE {pos} (index={pos}, original={i}) agent={cand.meta.get('agent', '')}\n"
            f"FAMILY: {cand.family}\n"
            f"DESCRIPTION: {cand.description}\n"
            f"EXPLANATION: {cand.condition}\n"
            f"RULE: {cand.meta.get('rule_ref', '')}\n"
            f"STEPS: {cand.steps}\n"
            f"ALTERNATIVE: {cand.alternative}\n"
            f"AVAILABLE INFO AT FIRST CITED STEP:\n{info_available_at(trace, first, 3200)}\n"
            f"CITED STEPS AND NEIGHBOURS:\n{_excerpt(trace, cand)}\n"
        )
    user = (
        f"TASK:\n{(trace.task or '')[:4000]}\n\n"
        f"INSTRUCTIONS:\n{(trace.instructions or '')[:8000]}\n\n"
        f"Try to disprove each candidate. Use the candidate index shown.\n\n"
        + "\n---\n".join(blocks)
    )
    try:
        resp = client.complete(
            [
                {"role": "system", "content": FALSIFY_SYSTEM},
                {"role": "user", "content": user},
            ],
            models=cfg.falsify.models,
            temperature=cfg.falsify.temperature,
            max_tokens=cfg.falsify.max_output_tokens,
            stage="falsify",
            response_schema={"type": "json_object"},
            reasoning_effort=cfg.falsify.reasoning_effort,
        )
    except (CallError, Exception):
        return [("stands", "")] * len(proposals), []

    try:
        parsed = extract_json(resp.content)
        subset = _parse_falsify(parsed, len(chosen_idx))
    except (ValueError, AttributeError):
        subset = [("stands", "")] * len(chosen_idx)
    out = [("stands", "")] * len(proposals)
    for local, orig in enumerate(chosen_idx):
        out[orig] = subset[local]
    return out, [_record("falsify", resp)]
