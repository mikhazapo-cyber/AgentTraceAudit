from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from .adjudicate import falsify_candidates, judge_findings
from .classify import classify_error_results
from .config import Config
from .critique import critique_review, should_critique
from .derive import derive_checks, split_lanes
from .detect import run_agent
from .evidence import info_available_at
from .llm import LLMClient
from .obligations import compile_obligations, extend_from_derived, obligation_context
from .prompts import AGENT_A_SYSTEM, AGENT_B_SYSTEM, OBLIGATION_SYSTEM
from .quotes import (
    annotate_evidence_sources,
    excerpt_in_source,
    locatable,
    supporting_instruction,
    unique_evidence,
)
from .schemas import CandidateFinding, Finding, TraceResult, Verification
from .structural import analyze as run_heuristics
from .structural import logical_tool_name, objective_findings
from .verify import inspect, quick_check


def _input_hash(trace) -> str:
    try:
        blob = trace.model_dump_json()
    except Exception:
        blob = json.dumps(trace, default=str, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _instruction_support(trace, cand: CandidateFinding) -> str:
    existing = str(cand.meta.get("rule_ref") or "").strip()
    if existing and excerpt_in_source(existing, trace.task, trace.instructions):
        return existing
    for idx in cand.steps:
        if not (0 <= idx < len(trace.steps)):
            continue
        step = trace.steps[idx]
        if step.kind != "tool_call":
            continue
        excerpt = supporting_instruction(trace, logical_tool_name(step) or step.name)
        if excerpt:
            return excerpt
    return existing


def _enrich_candidate(trace, cand: CandidateFinding) -> None:
    annotate_evidence_sources(trace, cand.evidence)
    cand.evidence = unique_evidence(cand.evidence)
    if not (cand.available_info or "").strip() and cand.steps:
        cand.available_info = info_available_at(trace, min(cand.steps))
    support = _instruction_support(trace, cand)
    if support:
        cand.meta["rule_ref"] = support


def _to_finding(
    trace,
    cand: CandidateFinding,
    status: str,
    reason: str,
    verification: Verification | None = None,
) -> Finding:
    _enrich_candidate(trace, cand)
    ver = verification or Verification()
    ver.adjudicator = status if status in {"confirmed", "insufficient_evidence"} else "rejected"
    if reason and (not ver.reason or ver.reason == "hard-verify passed"):
        ver.reason = reason
    return Finding(
        finding_id="",
        trace_id=trace.trace_id,
        family=cand.family,
        description=cand.description,
        explanation=cand.condition or reason,
        steps=cand.steps,
        evidence=cand.evidence,
        severity=cand.severity,
        status=status,  # type: ignore[arg-type]
        adjudication_reason=reason,
        confidence=cand.confidence,
        check_id=cand.check_id,
        source=cand.source,
        alternative=cand.alternative,
        rule_ref=str(cand.meta.get("rule_ref") or cand.condition),
        available_info=cand.available_info,
        meta=cand.meta,
        verification=ver,
    )


def _same_event(a: Finding | CandidateFinding, b: Finding | CandidateFinding) -> bool:
    sa, sb = set(a.steps), set(b.steps)
    if not (sa and sb and sa & sb):
        return False
    if a.family == b.family:
        return True
    if a.check_id and a.check_id == b.check_id:
        return True
    return False


def _rule_key(f: Finding) -> str:
    rule = (f.rule_ref or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", " ", rule).strip() if len(rule) >= 20 else ""


def _dedupe_same_rule(findings: list[Finding]) -> list[Finding]:
    """Collapse one rule violated at overlapping steps into one finding.

    Two auditors filing the same skipped policy check under instruction_violation
    and incorrect_tool_use describe one event; reporting both costs a false
    positive. The instruction family wins, because the shared evidence is a rule.
    """
    confirmed = [f for f in findings if f.status == "confirmed"]
    rest = [f for f in findings if f.status != "confirmed"]
    kept: list[Finding] = []
    for f in confirmed:
        key = _rule_key(f)
        partner = None
        if key:
            partner = next(
                (
                    k
                    for k in kept
                    if _rule_key(k) == key
                    and set(k.steps) & set(f.steps)
                    # Two mechanical findings are distinct proven events.
                    and not (k.source == "deterministic" and f.source == "deterministic")
                ),
                None,
            )
        if partner is None:
            kept.append(f)
            continue
        winner, loser = partner, f
        if f.family == "instruction_violation" and partner.family != "instruction_violation":
            winner, loser = f, partner
            kept[kept.index(partner)] = f
        winner.steps = sorted(set(winner.steps) | set(loser.steps))
        winner.meta["merged_duplicate_of_rule"] = loser.family
    return kept + rest


def _merge(findings: list[Finding]) -> list[Finding]:
    rank = {
        "incorrect_tool_use": 0,
        "evidence_contradiction": 1,
        "instruction_violation": 2,
        "ignored_feedback": 3,
        "redundant_action": 4,
        "unsupported_success": 5,
        "other": 6,
    }
    confirmed = [f for f in findings if f.status == "confirmed"]
    other = [f for f in findings if f.status != "confirmed"]
    kept: list[Finding] = []
    for f in sorted(confirmed, key=lambda x: (rank.get(x.family, 9), -(x.confidence or 0))):
        partner = next((k for k in kept if _same_event(f, k)), None)
        if partner is None:
            kept.append(f)
            continue
        partner.steps = sorted(set(partner.steps) | set(f.steps))
        if f.description not in partner.description:
            partner.description = (partner.description + "; " + f.description)[:700]
        if f.alternative and not partner.alternative:
            partner.alternative = f.alternative
    return kept + other


def _verification_from_quick(trace, cand: CandidateFinding, verdict: str, reason: str) -> Verification:
    """Record what verification actually found, not what the verdict implies.

    `family_gate` has to be the gate's own answer. Deriving it from the verdict
    made every confirmed finding claim a gate it may never have passed, which is
    the one field a reviewer would check first.
    """
    ver = inspect(trace, cand)
    ver.adjudicator = verdict if verdict in {"confirmed", "insufficient_evidence"} else "rejected"
    ver.reason = reason
    return ver


def _mechanical_lane(
    trace,
    det_cands: list[CandidateFinding],
    findings: list[Finding],
    rejected: list,
) -> None:
    """Confirm mechanical catches on hard verify alone, with no model veto.

    Runs before the agents so their proposals de-duplicate against proven events
    instead of being re-filed under a second family.
    """
    for cand in det_cands:
        if not locatable(trace, cand):
            rejected.append(cand)
            continue
        ver = inspect(trace, cand)
        if ver.passed and ver.mechanical:
            findings.append(_to_finding(trace, cand, "confirmed", "mechanical hard-verify", ver))
        else:
            rejected.append(cand)


def _overlaps_confirmed(cand: CandidateFinding, confirmed: list[Finding]) -> Finding | None:
    """Return a proven finding sharing a step with this candidate, in any family."""
    steps = set(cand.steps)
    if not steps:
        return None
    for f in confirmed:
        if steps & set(f.steps):
            return f
    return None


def analyze_trace(
    trace,
    cfg: Config,
    client: LLMClient,
    input_hash: str = "",
    cfg_hash: str = "",
    on_stage=None,
) -> TraceResult:
    stage = on_stage or (lambda label, detail="": None)
    t0 = time.monotonic()
    calls = []
    degraded: list[str] = []
    rejected: list[CandidateFinding] = []
    ih = input_hash or _input_hash(trace)

    spend = 0.0
    if getattr(client, "available", True) and cfg.classify_errors:
        stage("classify errors")
        ccalls = classify_error_results(trace, cfg, client)
        calls += ccalls
        spend += sum(c.cost_usd for c in ccalls)

    flags = run_heuristics(trace)
    det_cands = objective_findings(trace, flags)
    stage("structural", f"{len(flags.flags)} flag(s) · {len(det_cands)} objective finding(s)")

    findings: list[Finding] = []

    def finish(status, checks=None, pending=None, error=""):
        merged = _merge(_dedupe_same_rule(findings)) if cfg.merge_similar else findings
        merged = sorted(merged, key=lambda f: (f.steps[0] if f.steps else 10**9, f.family))
        for n, f in enumerate(merged, 1):
            f.finding_id = f"{trace.trace_id}-F{n:02d}"
        return TraceResult(
            trace_id=trace.trace_id,
            status=status,
            error=error,
            findings=merged,
            rejected=rejected,
            checks=checks or [],
            pending_checks=pending or [],
            calls=calls,
            degraded=degraded,
            input_hash=ih,
            cfg_hash=cfg_hash,
            latency_s=time.monotonic() - t0,
            meta={"flags": flags.to_dict()},
        )

    # Mechanical lane first: proven events, no model in the loop.
    _mechanical_lane(trace, det_cands, findings, rejected)
    mechanical_confirmed = [f for f in findings if f.status == "confirmed"]
    stage("mechanical", f"{len(mechanical_confirmed)} confirmed")

    if not getattr(client, "available", True):
        return finish("deterministic_only", pending=[c.check_id for c in det_cands])

    stage("derive instructions")
    checks, dcalls = derive_checks(trace, flags, cfg, client)
    calls += dcalls
    spend += sum(c.cost_usd for c in dcalls)
    obligation_lane_checks, residual_checks = split_lanes(checks)
    stage("derived", f"{len(obligation_lane_checks)} obligation · {len(residual_checks)} residual")

    # Hard abort only. target_mean_usd is a target, not a stop.
    if cfg.cost_cap_usd > 0 and spend >= cfg.cost_cap_usd:
        degraded.append("agents skipped: per-trace cost cap reached after derive")
        return finish("budget_exhausted", checks=checks)

    stage("parallel agents")
    obligations = extend_from_derived(
        trace, compile_obligations(trace), obligation_lane_checks
    )
    # Both lanes always run. With no quotable obligations, agent A falls back to
    # its generalist brief so the trace still gets a correctness auditor.
    focused = bool(obligations and obligation_lane_checks)
    lane_a_checks = obligation_lane_checks or residual_checks
    lane_b_checks = residual_checks or obligation_lane_checks
    a_secondary = [c for c in residual_checks if c not in lane_a_checks]
    b_secondary = [c for c in obligation_lane_checks if c not in lane_b_checks]

    def _agent_a():
        return run_agent(
            trace,
            lane_a_checks,
            flags,
            cfg,
            client,
            stage="agent_a",
            system=OBLIGATION_SYSTEM if focused else AGENT_A_SYSTEM,
            stage_cfg=cfg.agent_a,
            extra_context=obligation_context(trace, obligations) if focused else "",
            secondary_checks=a_secondary,
        )

    def _agent_b():
        return run_agent(
            trace,
            lane_b_checks,
            flags,
            cfg,
            client,
            stage="agent_b",
            system=AGENT_B_SYSTEM,
            stage_cfg=cfg.agent_b,
            secondary_checks=b_secondary,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(_agent_a)
        fut_b = pool.submit(_agent_b)
        try:
            found_a, calls_a = fut_a.result()
        except Exception as exc:
            found_a, calls_a = [], []
            degraded.append(f"agent_a failed: {exc}")
        try:
            found_b, calls_b = fut_b.result()
        except Exception as exc:
            found_b, calls_b = [], []
            degraded.append(f"agent_b failed: {exc}")

    calls += calls_a + calls_b
    spend += sum(c.cost_usd for c in calls_a + calls_b)
    stage("agents", f"{len(found_a)} from A · {len(found_b)} from B")

    proposals: list[CandidateFinding] = []
    for cand in list(found_a) + list(found_b):
        proven = _overlaps_confirmed(cand, mechanical_confirmed)
        if proven is None:
            proposals.append(cand)
            continue
        cand.meta["dropped_as_duplicate_of"] = proven.finding_id or proven.check_id
        cand.meta["dropped_reason"] = (
            f"steps overlap a proven {proven.family} finding at {proven.steps}"
        )
        rejected.append(cand)
    dropped = len(found_a) + len(found_b) - len(proposals)
    if dropped:
        stage("dedupe", f"{dropped} proposal(s) already proven mechanically")

    confirmed_llm: list[tuple[CandidateFinding, Finding]] = []

    if proposals:
        stage("judge")
        verdicts, jcalls = judge_findings(trace, checks, proposals, cfg, client, spend)
        calls += jcalls
        spend += sum(c.cost_usd for c in jcalls)

        stage("quick check")
        for cand, (verdict, reason) in zip(proposals, verdicts):
            if verdict == "rejected":
                rejected.append(cand)
                continue
            if verdict == "insufficient_evidence":
                ver = _verification_from_quick(trace, cand, verdict, reason)
                findings.append(_to_finding(trace, cand, "insufficient_evidence", reason, ver))
                continue
            qc_verdict, qc_reason = quick_check(trace, cand)
            ver = _verification_from_quick(trace, cand, qc_verdict, qc_reason)
            ver.adjudicator = "confirmed" if qc_verdict == "confirmed" else qc_verdict
            if qc_verdict == "confirmed":
                finding = _to_finding(trace, cand, "confirmed", reason or qc_reason, ver)
                findings.append(finding)
                if cand.source != "deterministic":
                    confirmed_llm.append((cand, finding))
            elif qc_verdict == "insufficient_evidence":
                findings.append(
                    _to_finding(trace, cand, "insufficient_evidence", qc_reason, ver)
                )
            else:
                rejected.append(cand)

    if should_critique(cfg, spend, checks, findings, proposals):
        stage("critique")
        extra, ccalls = critique_review(
            trace, checks, proposals, findings, cfg, client, spend
        )
        calls += ccalls
        spend += sum(c.cost_usd for c in ccalls)
        for cand in extra:
            proven = _overlaps_confirmed(cand, [f for f in findings if f.status == "confirmed"])
            if proven is not None:
                cand.meta["dropped_as_duplicate_of"] = proven.finding_id or proven.check_id
                rejected.append(cand)
                continue
            qc_verdict, qc_reason = quick_check(trace, cand)
            ver = _verification_from_quick(trace, cand, qc_verdict, qc_reason)
            ver.adjudicator = "confirmed" if qc_verdict == "confirmed" else qc_verdict
            if qc_verdict == "confirmed":
                finding = _to_finding(trace, cand, "confirmed", qc_reason, ver)
                findings.append(finding)
                confirmed_llm.append((cand, finding))
            elif qc_verdict == "insufficient_evidence":
                findings.append(
                    _to_finding(trace, cand, "insufficient_evidence", qc_reason, ver)
                )
            else:
                rejected.append(cand)

    if confirmed_llm and cfg.falsify_enabled:
        stage("falsify")
        fverdicts, fcalls = falsify_candidates(
            trace, [c for c, _ in confirmed_llm], cfg, client, spend
        )
        calls += fcalls
        spend += sum(c.cost_usd for c in fcalls)
        for (_cand, finding), (fverdict, freason) in zip(confirmed_llm, fverdicts):
            if fverdict != "disproved":
                continue
            finding.status = "insufficient_evidence"
            finding.adjudication_reason = freason or finding.adjudication_reason
            finding.verification.adjudicator = "insufficient_evidence"
            finding.verification.reason = freason or "falsified"
            finding.meta["falsified"] = True

    status = "ok"
    if cfg.cost_cap_usd > 0 and spend > cfg.cost_cap_usd:
        degraded.append(f"spent ${spend:.2f} above cap ${cfg.cost_cap_usd:.2f}")
    return finish(status, checks=checks)
