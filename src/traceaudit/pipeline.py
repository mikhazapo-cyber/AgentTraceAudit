"""Two high-power auditors in parallel, then one judge.

Each auditor derives task-specific checks and locates findings. The judge
confirms what to keep and assigns families. No mechanical lane, no extra
review passes.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

from .config import Config, StageConfig
from .llm import CallError, LLMClient, LLMResponse, PaymentRequired
from .pack import info_available_at, pack_char_budget, pack_judge, pack_trace
from .parsing import extract_json
from .prompts import ANALYST_A, ANALYST_B, JUDGE
from .schemas import (
    FAMILIES,
    FAMILY_DEFAULT_CLASS,
    CallRecord,
    CandidateFinding,
    DerivedCheck,
    Evidence,
    Finding,
    TraceResult,
    Verdict,
    clamp_confidence,
    normalize_class_and_family,
)


class ParsedVerdict(NamedTuple):
    verdict: Verdict
    reason: str
    family: str
    error_class: str
    confidence: int | None


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


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def quote_in_step(trace, step_idx: int, quote: str) -> bool:
    if not quote or not (0 <= step_idx < len(trace.steps)):
        return False
    hay = _norm(trace.steps[step_idx].text())
    needle = _norm(quote)
    if not needle:
        return False
    if needle in hay:
        return True
    return len(needle) >= 24 and needle[:80] in hay


def local_proof(trace, cand: CandidateFinding) -> tuple[bool, str]:
    if not cand.steps:
        return False, "no step cited"
    for idx in cand.steps:
        if not (0 <= idx < len(trace.steps)):
            return False, f"step {idx} is not in the trace"
    if not cand.evidence:
        return False, "no evidence quotes"
    if not any(quote_in_step(trace, e.step, e.quote) for e in cand.evidence):
        return False, "quoted evidence does not appear in the cited steps"
    if (
        cand.family in {"redundant_action", "ignored_feedback"}
        and not (cand.alternative or "").strip()
    ):
        return False, "inefficiency needs a cheaper alternative that existed then"
    return True, "steps and quotes check out"


def _family(raw: object, fallback: str = "other") -> str:
    value = str(raw or fallback)
    return value if value in FAMILIES else fallback


def parse_checks(parsed: object, auditor: str) -> list[DerivedCheck]:
    items = parsed.get("checks") if isinstance(parsed, dict) else []
    out: list[DerivedCheck] = []
    for i, item in enumerate(items or [], 1):
        if not isinstance(item, dict):
            continue
        desc = str(item.get("description") or item.get("condition") or "").strip()
        if not desc:
            continue
        refs = item.get("source_refs") or []
        if isinstance(refs, str):
            refs = [refs]
        out.append(
            DerivedCheck(
                check_id=str(item.get("check_id") or f"{auditor}-{i}"),
                family=_family(item.get("family")),  # type: ignore[arg-type]
                description=desc[:400],
                condition=str(item.get("condition") or "")[:400],
                justified_when=str(item.get("justified_when") or "")[:400],
                source_refs=[str(r)[:300] for r in refs if r][:6],
                source=auditor,
            )
        )
    return out


def parse_findings(parsed: object, auditor: str) -> list[CandidateFinding]:
    items = parsed.get("findings") if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        items = []
    out: list[CandidateFinding] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        decision = str(item.get("decision") or "violated").lower()
        if decision in {"not_violated", "ok", "rejected"}:
            continue
        if decision not in {"violated", "insufficient_evidence"}:
            decision = "violated"
        desc = str(item.get("description") or item.get("explanation") or "").strip()
        if not desc:
            continue
        mapped = normalize_class_and_family(item.get("error_class"), item.get("family"))
        if mapped is None:
            continue
        error_class, family = mapped
        steps = item.get("steps") or []
        if isinstance(steps, int):
            steps = [steps]
        steps = [int(s) for s in steps if isinstance(s, (int, float))]
        evidence = []
        for ev in item.get("evidence") or []:
            if isinstance(ev, dict) and "step" in ev:
                evidence.append(
                    Evidence(
                        step=int(ev["step"]), quote=str(ev.get("quote") or "")[:400]
                    )
                )
        omitted_conf = (
            item.get("confidence", None) is None or item.get("confidence") == ""
        )
        out.append(
            CandidateFinding(
                check_id=str(item.get("check_id") or ""),
                family=family,  # type: ignore[arg-type]
                error_class=error_class,
                description=desc[:500],
                condition=str(item.get("explanation") or item.get("rule_ref") or "")[
                    :700
                ],
                steps=steps,
                evidence=evidence,
                severity=item.get("severity")
                if item.get("severity") in ("critical", "major", "minor")
                else "major",
                confidence=clamp_confidence(item.get("confidence"), 70),
                alternative=str(item.get("alternative") or ""),
                available_info=str(item.get("available_info") or ""),
                auditor=auditor,
                meta={
                    "decision": decision,
                    "rule_ref": str(item.get("rule_ref") or ""),
                    "confidence_omitted": omitted_conf,
                },
            )
        )
    return out


def parse_verdicts(parsed: object, n: int) -> list[ParsedVerdict]:
    items = parsed.get("verdicts") if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        items = []
    by_index: dict[int, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        by_index[idx] = item
    out: list[ParsedVerdict] = []
    for i in range(n):
        item = by_index.get(i) or {}
        raw = str(item.get("verdict") or "insufficient_evidence").lower()
        if raw not in {"confirmed", "rejected", "insufficient_evidence"}:
            raw = "insufficient_evidence"
        if "confidence" in item and item.get("confidence") not in (None, ""):
            conf: int | None = clamp_confidence(item.get("confidence"), 80)
        else:
            conf = None
        out.append(
            ParsedVerdict(
                raw,  # type: ignore[arg-type]
                str(item.get("reason") or ""),
                _family(item.get("family"), ""),
                str(item.get("error_class") or "").strip(),
                conf,
            )
        )
    return out


def _resolve_class_family(
    cand: CandidateFinding,
    judge_family: str,
    judge_class: str,
) -> tuple[str, str]:
    if judge_class:
        mapped = normalize_class_and_family(judge_class, judge_family or cand.family)
        if mapped is not None:
            return mapped
    if judge_family:
        mapped = normalize_class_and_family("", judge_family)
        if mapped is not None:
            return mapped
    mapped = normalize_class_and_family(cand.error_class, cand.family)
    if mapped is None:
        family = cand.family if cand.family in FAMILIES else "other"
        return FAMILY_DEFAULT_CLASS.get(family, "other_error"), family
    return mapped


def _to_finding(
    trace,
    cand: CandidateFinding,
    status: Verdict,
    reason: str,
    family: str = "",
    error_class: str = "",
    confidence: int | None = None,
) -> Finding:
    if not (cand.available_info or "").strip() and cand.steps:
        cand.available_info = info_available_at(trace, min(cand.steps))
    cls, fam = _resolve_class_family(cand, family, error_class)
    if confidence is None:
        confidence = 80 if status == "confirmed" else cand.confidence
    return Finding(
        finding_id="",
        trace_id=trace.trace_id,
        family=fam,  # type: ignore[arg-type]
        error_class=cls,
        description=cand.description,
        explanation=cand.condition or reason,
        steps=cand.steps,
        evidence=cand.evidence,
        severity=cand.severity,
        status=status,
        adjudication_reason=reason,
        confidence=clamp_confidence(confidence, cand.confidence),
        check_id=cand.check_id,
        alternative=cand.alternative,
        rule_ref=str(cand.meta.get("rule_ref") or ""),
        available_info=cand.available_info,
        auditor=cand.auditor,
        meta=cand.meta,
    )


def apply_confidence_threshold(finding: Finding, min_confidence: int) -> Finding:
    if finding.status == "confirmed" and finding.confidence < min_confidence:
        finding.status = "insufficient_evidence"
        note = f"confidence {finding.confidence} below {min_confidence}"
        finding.adjudication_reason = (
            f"{finding.adjudication_reason} ({note})"
            if finding.adjudication_reason
            else note
        )
    return finding


# Judge named a concrete look-alike / disproof. Do not override that abstention.
_PEER_BLOCK_REASON = re.compile(
    r"(?i)("
    r"justified|"
    r"\bretr(?:y|ied)\b|"
    r"playbook|"
    r"already (?:in|appears?|present|quoted)|"
    r"appeared in the (?:question|user|evidence|trace)|"
    r"string already|"
    r"in the (?:user )?(?:question|prompt)\b|"
    r"not a (?:hard )?rule|"
    r"preferred (?:skill|order)|"
    r"extra verification|"
    r"search, not|"
    r"changed (?:the )?(?:request|arguments?|target)|"
    r"look[- ]alike|"
    r"does not apply|"
    r"not an? (?:error|inefficiency|violation|finding)|"
    r"\bdisproof\b|"
    r"\breject(?:ed|ion)?\b"
    r")"
)
_PEER_CONFIRM_MIN_CONF = 70


def _judge_reason_blocks_peer(reason: str) -> bool:
    return bool(_PEER_BLOCK_REASON.search(reason or ""))


def _judge_reason_thin(reason: str) -> bool:
    text = " ".join((reason or "").split())
    if _judge_reason_blocks_peer(text):
        return False
    return len(text) < 40


def _peer_supports(
    cand: CandidateFinding,
    peers: list[CandidateFinding],
    verdicts: list[ParsedVerdict],
    trace,
    cand_row: ParsedVerdict,
) -> bool:
    """Promote an abstention only when both auditors independently proved it.

    Same parent family, overlapping steps, both quotes pass local_proof,
    neither proposal was rejected, and the judge did not name a concrete
    justification or disproof. Empty/thin judge reasons may promote;
    a longer reason may promote only if both auditors are confident.
    """
    if cand.meta.get("decision") != "violated":
        return False
    if cand_row.verdict == "rejected":
        return False
    if _judge_reason_blocks_peer(cand_row.reason):
        return False
    mine = set(cand.steps)
    for peer, row in zip(peers, verdicts):
        if peer is cand or peer.auditor == cand.auditor:
            continue
        if row.verdict == "rejected":
            continue
        if peer.family != cand.family:
            continue
        if peer.meta.get("decision") != "violated":
            continue
        if not (mine & set(peer.steps)):
            continue
        if _judge_reason_blocks_peer(row.reason):
            continue
        ok, _ = local_proof(trace, peer)
        if not ok:
            continue
        if _judge_reason_thin(cand_row.reason) or (
            cand.confidence >= _PEER_CONFIRM_MIN_CONF
            and peer.confidence >= _PEER_CONFIRM_MIN_CONF
        ):
            return True
    return False


def _merge(findings: list[Finding]) -> list[Finding]:
    confirmed = [f for f in findings if f.status == "confirmed"]
    rest = [f for f in findings if f.status != "confirmed"]
    kept: list[Finding] = []
    for finding in confirmed:
        partner = next(
            (
                k
                for k in kept
                if k.family == finding.family and set(k.steps) & set(finding.steps)
            ),
            None,
        )
        if partner is None:
            kept.append(finding)
            continue
        winner, extra = (
            (finding, partner)
            if finding.confidence > partner.confidence
            else (partner, finding)
        )
        if winner is finding:
            kept[kept.index(partner)] = winner
        winner.steps = sorted(set(winner.steps) | set(extra.steps))
        if extra.description and extra.description not in winner.description:
            winner.description = (winner.description + "; " + extra.description)[:700]
        if extra.alternative and not winner.alternative:
            winner.alternative = extra.alternative
        if extra.evidence:
            seen = {(e.step, e.quote) for e in winner.evidence}
            winner.evidence = list(winner.evidence) + [
                e for e in extra.evidence if (e.step, e.quote) not in seen
            ]
    return kept + rest


def _complete(
    client: LLMClient,
    stage: str,
    system: str,
    user: str,
    stage_cfg: StageConfig,
) -> tuple[object, CallRecord]:
    resp = client.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        models=stage_cfg.models,
        temperature=stage_cfg.temperature,
        stage=stage,
        response_schema={"type": "json_object"},
        reasoning_effort=stage_cfg.reasoning_effort,
    )
    try:
        parsed = extract_json(resp.content)
    except ValueError:
        parsed = {}
    return parsed, _record(stage, resp)


def analyze_trace(
    trace,
    cfg: Config,
    client: LLMClient,
    cfg_hash: str = "",
    on_stage: Callable[[str], None] | None = None,
) -> TraceResult:
    t0 = time.monotonic()
    calls: list[CallRecord] = []
    degraded: list[str] = []
    rejected: list[CandidateFinding] = []

    def note(stage: str) -> None:
        if on_stage:
            on_stage(stage)

    def finish(
        status: str,
        findings: list[Finding] | None = None,
        checks: list[DerivedCheck] | None = None,
        error: str = "",
    ) -> TraceResult:
        merged = _merge(findings or []) if cfg.merge_similar else list(findings or [])
        merged = sorted(
            merged, key=lambda f: (f.steps[0] if f.steps else 10**9, f.family)
        )
        for n, finding in enumerate(merged, 1):
            finding.finding_id = f"{trace.trace_id}-F{n:02d}"
        return TraceResult(
            trace_id=trace.trace_id,
            status=status,  # type: ignore[arg-type]
            error=error,
            findings=merged,
            rejected=rejected,
            checks=checks or [],
            calls=calls,
            degraded=degraded,
            latency_s=time.monotonic() - t0,
            meta={"cfg_hash": cfg_hash, "n_steps": len(trace.steps)},
        )

    if not getattr(client, "available", True):
        return finish("error", error="no API key")

    auditor_chars = pack_char_budget(cfg.target_mean_usd, 0.0, copies=2)
    instr_chars = min(cfg.max_instruction_chars, max(4000, auditor_chars // 8))
    pack = pack_trace(trace, auditor_chars, instr_chars)
    note("auditors")

    def run_auditor(stage: str, system: str, stage_cfg: StageConfig):
        parsed, rec = _complete(client, stage, system, pack, stage_cfg)
        return parse_checks(parsed, stage), parse_findings(parsed, stage), rec

    checks_a: list[DerivedCheck] = []
    checks_b: list[DerivedCheck] = []
    found_a: list[CandidateFinding] = []
    found_b: list[CandidateFinding] = []
    auditor_ok = {"agent_a": False, "agent_b": False}
    paywall = False

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(run_auditor, "agent_a", ANALYST_A, cfg.agent_a)
        fut_b = pool.submit(run_auditor, "agent_b", ANALYST_B, cfg.agent_b)
        try:
            checks_a, found_a, rec_a = fut_a.result()
            calls.append(rec_a)
            auditor_ok["agent_a"] = True
        except PaymentRequired as exc:
            degraded.append(f"agent_a failed: {exc}")
            paywall = True
        except Exception as exc:
            degraded.append(f"agent_a failed: {exc}")
        try:
            checks_b, found_b, rec_b = fut_b.result()
            calls.append(rec_b)
            auditor_ok["agent_b"] = True
        except PaymentRequired as exc:
            degraded.append(f"agent_b failed: {exc}")
            paywall = True
        except Exception as exc:
            degraded.append(f"agent_b failed: {exc}")

    checks = checks_a + checks_b
    proposals = found_a + found_b
    spend = sum(c.cost_usd for c in calls)

    if not proposals:
        note("done")
        if paywall and not any(auditor_ok.values()):
            return finish(
                "error",
                checks=checks,
                error="provider payment required; add credits and retry",
            )
        return finish("ok", checks=checks)

    if cfg.cost_cap_usd > 0 and spend >= cfg.cost_cap_usd:
        degraded.append("judge skipped: per-trace cost cap reached after the auditors")
        note("done")
        return finish("budget_exhausted", checks=checks)

    note("judge")
    judge_chars = pack_char_budget(cfg.target_mean_usd, spend, copies=1)
    judge_instr = min(cfg.max_instruction_chars, max(3000, judge_chars // 8))
    judge_pack = pack_judge(trace, checks, proposals, judge_chars, judge_instr)
    try:
        parsed, rec_j = _complete(client, "judge", JUDGE, judge_pack, cfg.judge)
        calls.append(rec_j)
        verdicts = parse_verdicts(parsed, len(proposals))
    except (CallError, Exception) as exc:
        degraded.append(f"judge failed: {exc}")
        note("done")
        return finish(
            "model_error",
            findings=[
                _to_finding(
                    trace, c, "insufficient_evidence", "judge did not return a verdict"
                )
                for c in proposals
            ],
            checks=checks,
            error=str(exc),
        )

    findings: list[Finding] = []
    for cand, row in zip(proposals, verdicts):
        verdict, reason, family = row.verdict, row.reason, row.family
        if verdict == "rejected":
            rejected.append(cand)
            continue
        if verdict == "insufficient_evidence":
            ok, proof = local_proof(trace, cand)
            if (
                ok
                and cand.meta.get("decision") == "violated"
                and _peer_supports(cand, proposals, verdicts, trace, row)
            ):
                peer_conf = (
                    row.confidence if row.confidence is not None else cand.confidence
                )
                finding = _to_finding(
                    trace,
                    cand,
                    "confirmed",
                    reason or "both auditors located this and quotes check out",
                    family,
                    row.error_class,
                    peer_conf,
                )
            else:
                finding = _to_finding(
                    trace,
                    cand,
                    "insufficient_evidence",
                    reason or proof,
                    family,
                    row.error_class,
                    row.confidence,
                )
            findings.append(apply_confidence_threshold(finding, cfg.min_confidence))
            continue
        ok, proof = local_proof(trace, cand)
        if ok:
            finding = _to_finding(
                trace,
                cand,
                "confirmed",
                reason or proof,
                family,
                row.error_class,
                row.confidence,
            )
        else:
            finding = _to_finding(
                trace,
                cand,
                "insufficient_evidence",
                proof,
                family,
                row.error_class,
                row.confidence,
            )
        findings.append(apply_confidence_threshold(finding, cfg.min_confidence))

    note("done")
    return finish("ok", findings=findings, checks=checks)
