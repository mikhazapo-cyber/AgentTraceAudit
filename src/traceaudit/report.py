"""Terminal + file reports. ASCII-safe so a Windows cp1252 console does not crash."""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

from .schemas import (
    ERROR_CLASS_GLOSSARY,
    FAMILIES,
    FAMILY_GLOSSARY,
    CanonicalTrace,
    Finding,
    TraceResult,
)

CONFIRMED = "confirmed"
ABSTAINED = "insufficient_evidence"

_ASCII_SUBS = {
    "\u2014": "--",
    "\u2013": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u00b7": "-",
    "\u2192": "->",
    "\u00b0": " deg",
    "\u2265": ">=",
    "\u2264": "<=",
    "\u00a0": " ",
}


def ascii_safe(text: str) -> str:
    for src, dst in _ASCII_SUBS.items():
        text = text.replace(src, dst)
    return text.encode("ascii", "replace").decode("ascii")


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width) or [""]


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    idx = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
    return xs[idx]


def confirmed_findings(result: TraceResult) -> list[Finding]:
    return [f for f in result.findings if f.status == CONFIRMED]


def abstained_findings(result: TraceResult) -> list[Finding]:
    return [f for f in result.findings if f.status == ABSTAINED]


def family_counts(findings: list[Finding]) -> dict[str, int]:
    seen: dict[str, int] = {}
    for f in findings:
        seen[f.family] = seen.get(f.family, 0) + 1
    ordered = {fam: seen[fam] for fam in FAMILIES if fam in seen}
    ordered.update({k: v for k, v in seen.items() if k not in ordered})
    return ordered


def class_counts(findings: list[Finding]) -> dict[str, int]:
    seen: dict[str, int] = {}
    for f in findings:
        key = f.error_class or f.family
        seen[key] = seen.get(key, 0) + 1
    return seen


def finding_tag(f: Finding) -> str:
    cls = f.error_class or f.family
    return f"{f.family} / {cls}  {int(f.confidence)}%"


def step_span(steps: list[int]) -> str:
    if not steps:
        return "no step located"
    ordered = sorted({int(s) for s in steps})
    if len(ordered) == 1:
        return f"step {ordered[0]}"
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"steps {ordered[0]}-{ordered[-1]}"
    return "steps " + ", ".join(str(s) for s in ordered)


def _family_brief(counts: dict[str, int]) -> str:
    return ", ".join(f"{fam} {n}" for fam, n in counts.items())


def _findings_brief(findings: list[Finding]) -> str:
    return ", ".join(finding_tag(f) for f in findings)


def _quote_block(text: str, limit: int = 600) -> str:
    body = (text or "").strip()
    if not body:
        return "> (not recorded)"
    if len(body) > limit:
        body = body[:limit].rstrip() + " [...]"
    return "\n".join(f"> {line}" if line.strip() else ">" for line in body.splitlines())


def finding_markdown(f: Finding, ordinal: int = 0) -> str:
    label = f"{ordinal}. " if ordinal else ""
    cls = f.error_class or f.family
    gloss = ERROR_CLASS_GLOSSARY.get(cls) or FAMILY_GLOSSARY.get(f.family, "")
    lines = [
        f"### {label}`{f.family}` / `{cls}`  {int(f.confidence)}% at {step_span(f.steps)}",
        "",
        gloss,
        "",
        f"- **Class:** `{cls}`",
        f"- **Confidence:** {int(f.confidence)}%",
        f"- **Severity:** {f.severity}",
        f"- **Status:** {f.status.upper()}",
        f"- **Finding id:** `{f.finding_id}`",
        "",
        f"**What happened.** {f.description}",
    ]
    if f.explanation:
        lines += ["", f"**Why it is a problem.** {f.explanation}"]
    if f.rule_ref:
        lines += ["", "**Rule or tool excerpt.**", _quote_block(f.rule_ref)]
    if f.available_info:
        lines += ["", f"**What the agent could see then.** {f.available_info}"]
    if f.alternative:
        lines += ["", f"**Cheaper alternative available then.** {f.alternative}"]
    lines += ["", "**Supporting excerpts.**"]
    seen: set[tuple[int, str]] = set()
    for e in f.evidence:
        quote = (e.quote or "").strip()[:300]
        if (e.step, quote) in seen:
            continue
        seen.add((e.step, quote))
        lines.append(f"- step {e.step}: `{quote}`")
    if not seen:
        lines.append("- (none recorded)")
    if f.adjudication_reason:
        lines += ["", f"**Judge.** {f.adjudication_reason}"]
    lines.append("")
    return "\n".join(lines)


def trace_report(result: TraceResult, trace: CanonicalTrace | None = None) -> str:
    conf, abst = confirmed_findings(result), abstained_findings(result)
    body = [
        f"# Audit report: `{result.trace_id}`",
        "",
        "## Verdict",
        "",
    ]
    if result.status == "error":
        body += [f"The run failed: {result.error or 'no detail recorded'}", ""]
    elif conf:
        body += [f"**{len(conf)} problem(s) confirmed**: {_findings_brief(conf)}.", ""]
    else:
        body += [
            "**Nothing confirmed.** Actions were justified or not provable from this trace.",
            "",
        ]
    if abst:
        body.append(
            f"{len(abst)} item(s) were abstained for insufficient evidence; those are not accusations."
        )
    body += [
        "",
        f"| confirmed | abstained | status | cost | latency |",
        "|---:|---:|---|---:|---:|",
        f"| **{len(conf)}** | {len(abst)} | {result.status} | "
        f"${result.cost_usd:.4f} | {result.latency_s:.1f}s |",
        "",
    ]
    if result.degraded:
        body += ["> Degraded: " + "; ".join(result.degraded), ""]
    if trace is not None:
        body += ["## What the agent was asked to do", ""]
        body += ["**Task**", _quote_block(trace.task, 800), ""]
        if trace.instructions:
            body += ["**Instructions**", _quote_block(trace.instructions, 1200), ""]
        if trace.tools:
            names = ", ".join(f"`{t.name}`" for t in trace.tools[:25])
            body += [f"**Tools:** {names}", ""]
        body += [
            f"**Trace:** {len(trace.steps)} steps, format `{trace.source_format}`",
            "",
        ]

    if conf:
        body += ["## Confirmed findings", ""]
        body += [finding_markdown(f, i) for i, f in enumerate(conf, 1)]
    elif result.status != "error":
        body += ["## Confirmed findings", "", "None.", ""]

    if abst:
        body += [
            "## Abstained: insufficient evidence",
            "",
            "Not findings. Tracked separately and never scored as false positives.",
            "",
        ]
        body += [finding_markdown(f, i) for i, f in enumerate(abst, 1)]

    if result.checks:
        body += [
            "## Checks derived for this trace",
            "",
            "Compiled from this task, these instructions, and these tools.",
            "",
        ]
        for c in result.checks:
            src = f" Source: _{c.source_refs[0][:200]}_" if c.source_refs else ""
            body.append(f"- `{c.check_id}` [{c.family}] {c.description}{src}")
        body.append("")
    return "\n".join(body)


def trace_console_lines(
    result: TraceResult,
    index: int = 0,
    total: int = 0,
    session_spend: float | None = None,
) -> list[str]:
    conf, abst = confirmed_findings(result), abstained_findings(result)
    counter = f"[{index}/{total}]" if total else ""
    status = "ERROR" if result.status == "error" else result.status
    head = (
        f"  {counter:>7} {result.trace_id:<22.22} {status:<18.18} "
        f"${result.cost_usd:>7.3f} {result.latency_s:>7.1f}s"
    )
    if session_spend is not None:
        head += f"  session ${session_spend:.3f}"
    lines = [head]
    if result.status == "error":
        why = " ".join((result.error or "no detail recorded").split())
        lines += [f"           {chunk}" for chunk in _wrap(f"FAILED: {why}", 100)]
        return [ascii_safe(x) for x in lines]
    if conf:
        detail = f"{len(conf)} confirmed ({_findings_brief(conf)})"
    else:
        detail = "0 confirmed"
    detail += f" | {len(abst)} abstained"
    lines.append(f"           {detail}")
    if result.degraded:
        lines.append(f"           degraded: {'; '.join(result.degraded)[:140]}")
    return [ascii_safe(x) for x in lines]


def run_console_summary(summary: dict, out_dir: str | Path) -> str:
    width = 74
    rule = "-" * width
    cost, lat = summary["cost_usd"], summary["latency_s"]
    lines = [
        "=" * width,
        f"AUDIT SUMMARY   {summary['n_traces']} trace(s)",
        rule,
        f"CONFIRMED FINDINGS      {summary['n_confirmed_findings']}",
    ]
    for fam, n in (summary.get("confirmed_by_family") or {}).items():
        lines.append(f"    {fam:<24}{n}")
    if not summary.get("confirmed_by_family"):
        lines.append("    (none)")
    by_class = summary.get("confirmed_by_class") or {}
    if by_class:
        lines.append("CONFIRMED BY CLASS")
        for cls, n in by_class.items():
            lines.append(f"    {cls:<24}{n}")
    lines += [
        f"ABSTAINED               {summary['n_insufficient_evidence']}"
        "   insufficient evidence, never a false positive",
        rule,
        f"COST      mean ${cost['mean']:.4f}   p95 ${cost['p95']:.4f}   "
        f"total ${cost['total']:.4f}",
        f"LATENCY   mean {lat['mean']:.1f}s   p95 {lat['p95']:.1f}s   "
        f"total {lat['total']:.1f}s",
        rule,
        f"ARTIFACTS {Path(out_dir)}",
        "    findings.jsonl           one record per trace",
        "    findings.csv             one row per finding",
        "    reports/<id>.md          per-trace report",
        "    summary.json             run totals",
        "=" * width,
    ]
    return ascii_safe("\n".join(lines))


_CSV_FIELDS = [
    "trace_id",
    "finding_id",
    "family",
    "error_class",
    "confidence",
    "status",
    "severity",
    "steps",
    "description",
    "explanation",
    "rule_ref",
    "available_info",
    "alternative",
    "evidence_quotes",
    "auditor",
    "trace_cost_usd",
    "trace_latency_s",
]


def build_summary(results: list[TraceResult]) -> dict:
    costs = [r.cost_usd for r in results]
    lats = [r.latency_s for r in results if r.latency_s]
    all_conf = [f for r in results for f in confirmed_findings(r)]
    return {
        "n_traces": len(results),
        "n_confirmed_findings": len(all_conf),
        "confirmed_by_family": family_counts(all_conf),
        "confirmed_by_class": class_counts(all_conf),
        "n_insufficient_evidence": sum(len(abstained_findings(r)) for r in results),
        "cost_usd": {
            "mean": round(sum(costs) / len(costs), 4) if costs else 0.0,
            "p95": round(_percentile(costs, 0.95) or 0, 4),
            "total": round(sum(costs), 4),
        },
        "latency_s": {
            "mean": round(sum(lats) / len(lats), 2) if lats else 0.0,
            "p95": round(_percentile(lats, 0.95) or 0, 2),
            "total": round(sum(lats), 2),
        },
        "statuses": {
            s: sum(1 for r in results if r.status == s)
            for s in {r.status for r in results}
        },
        "per_trace": [
            {
                "trace_id": r.trace_id,
                "status": r.status,
                "n_confirmed": len(confirmed_findings(r)),
                "confirmed_by_family": family_counts(confirmed_findings(r)),
                "confirmed_by_class": class_counts(confirmed_findings(r)),
                "n_insufficient_evidence": len(abstained_findings(r)),
                "cost_usd": round(r.cost_usd, 4),
                "latency_s": round(r.latency_s, 2),
                "error": r.error,
            }
            for r in results
        ],
    }


def summary_to_md(summary: dict) -> str:
    cost, lat = summary["cost_usd"], summary["latency_s"]
    lines = [
        "# Run summary",
        "",
        f"{summary['n_traces']} trace(s). "
        f"**{summary['n_confirmed_findings']} confirmed**, "
        f"{summary['n_insufficient_evidence']} abstained.",
        "",
        "| metric | mean | p95 | total |",
        "|---|---:|---:|---:|",
        f"| cost per trace (USD) | {cost['mean']:.4f} | {cost['p95']:.4f} | {cost['total']:.4f} |",
        f"| latency per trace (s) | {lat['mean']:.1f} | {lat['p95']:.1f} | {lat.get('total', 0):.1f} |",
        "",
        "## Confirmed by family",
        "",
    ]
    fams = summary.get("confirmed_by_family") or {}
    lines += [f"- `{fam}`: {n}" for fam, n in fams.items()] or ["- none"]
    classes = summary.get("confirmed_by_class") or {}
    if classes:
        lines += ["", "## Confirmed by class", ""]
        lines += [f"- `{cls}`: {n}" for cls, n in classes.items()]
    lines += [
        "",
        "## Per trace",
        "",
        "| trace | status | confirmed | abstained | cost | latency |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for t in summary.get("per_trace", []):
        lines.append(
            f"| `{t['trace_id']}` | {t['status']} | {t['n_confirmed']} | "
            f"{t['n_insufficient_evidence']} | ${t['cost_usd']:.4f} | {t['latency_s']:.1f}s |"
        )
    return "\n".join(lines) + "\n"


def write_run_artifacts(
    out_dir: str | Path,
    results: list[TraceResult],
    traces: dict[str, CanonicalTrace] | None = None,
) -> Path:
    traces = traces or {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports = out / "reports"
    reports.mkdir(exist_ok=True)

    with (out / "findings.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(r.model_dump_json() + "\n")
    (out / "findings.json").write_text(
        json.dumps([r.model_dump() for r in results], indent=2, default=str),
        encoding="utf-8",
    )
    with (out / "findings.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in results:
            for f in r.findings:
                writer.writerow(
                    {
                        "trace_id": r.trace_id,
                        "finding_id": f.finding_id,
                        "family": f.family,
                        "error_class": f.error_class,
                        "confidence": int(f.confidence),
                        "status": f.status,
                        "severity": f.severity,
                        "steps": " ".join(map(str, f.steps)),
                        "description": f.description,
                        "explanation": f.explanation,
                        "rule_ref": f.rule_ref,
                        "available_info": f.available_info,
                        "alternative": f.alternative,
                        "evidence_quotes": " || ".join(
                            (e.quote or "").replace("\n", " ")[:300] for e in f.evidence
                        ),
                        "auditor": f.auditor,
                        "trace_cost_usd": round(r.cost_usd, 6),
                        "trace_latency_s": round(r.latency_s, 3),
                    }
                )
    for r in results:
        (reports / f"{r.trace_id}.md").write_text(
            trace_report(r, traces.get(r.trace_id)), encoding="utf-8"
        )
    summary = build_summary(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(summary_to_md(summary), encoding="utf-8")
    return out
