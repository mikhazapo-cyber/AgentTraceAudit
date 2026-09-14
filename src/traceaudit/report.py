"""Render a run: structured artifacts on disk, readable text in the terminal.

Terminal text is folded to 7-bit ASCII. A Windows PowerShell console is cp1252
by default, so an em dash in a heading or a degree sign inside a quoted trace
excerpt either prints as mojibake or raises UnicodeEncodeError mid-run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import textwrap
from pathlib import Path

from .schemas import FAMILIES, FAMILY_GLOSSARY, CanonicalStep, CanonicalTrace, Finding, TraceResult

CONFIRMED = "confirmed"
ABSTAINED = "insufficient_evidence"

ABSTAINED_HEADING = "Abstained: insufficient evidence (NOT findings)"
ABSTAINED_NOTE = (
    "These are not accusations. The auditor could not settle the question from "
    "this trace, so nothing is alleged against the agent. Abstentions are tracked "
    "separately and are never scored as false positives."
)

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

# A tool result or a final outcome is environment-supplied: quoting one is
# evidence the agent could not have authored. An assistant message is not.
_ENV_KINDS = {"tool_result", "outcome"}


def ascii_safe(text: str) -> str:
    """Fold text to ASCII so it survives a cp1252 console."""
    for src, dst in _ASCII_SUBS.items():
        text = text.replace(src, dst)
    return text.encode("ascii", "replace").decode("ascii")


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width) or [""]


def split_capture_gaps(gaps: list[str]) -> tuple[list[str], list[str]]:
    """Separate real data loss from adapter provenance notes.

    Adapters record which module read the file in the same list as genuine loss.
    Announcing 'nothing is confirmed on missing data' over a note that says
    which adapter ran makes a healthy trace look damaged.
    """
    loss, notes = [], []
    for gap in gaps or []:
        text = gap.strip()
        if "alias mapped" in text or " adapter: " in text:
            notes.append(text)
        else:
            loss.append(text)
    return loss, notes


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    idx = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
    return xs[idx]


def config_fingerprint(cfg) -> str:
    return cfg.fingerprint() if hasattr(cfg, "fingerprint") else ""


def load_results(path: Path) -> list[TraceResult]:
    path = Path(path)
    if path.is_dir():
        path = path / "findings.jsonl"
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json" and text.lstrip().startswith("{"):
        data = json.loads(text)
        items = data.get("results") or data.get("traces") or [data]
        return [TraceResult.model_validate(x) for x in items]
    out = []
    for line in text.splitlines():
        if line.strip():
            rec = json.loads(line)
            if "findings" in rec and "trace_id" in rec and "status" in rec:
                out.append(TraceResult.model_validate(rec))
            else:
                # findings-only jsonl: wrap
                out.append(
                    TraceResult(
                        trace_id=rec.get("trace_id", ""),
                        findings=[Finding.model_validate(rec)],
                    )
                )
    return out


# --------------------------------------------------------------------------
# confirmed vs abstained: one definition, used by every renderer
# --------------------------------------------------------------------------


def confirmed_findings(result: TraceResult) -> list[Finding]:
    return [f for f in result.findings if f.status == CONFIRMED]


def abstained_findings(result: TraceResult) -> list[Finding]:
    return [f for f in result.findings if f.status == ABSTAINED]


def family_counts(findings: list[Finding]) -> dict[str, int]:
    """Counts per family, in the canonical family order."""
    seen: dict[str, int] = {}
    for f in findings:
        seen[f.family] = seen.get(f.family, 0) + 1
    ordered = {fam: seen[fam] for fam in FAMILIES if fam in seen}
    ordered.update({k: v for k, v in seen.items() if k not in ordered})
    return ordered


def _family_brief(counts: dict[str, int]) -> str:
    return ", ".join(f"{fam} {n}" for fam, n in counts.items())


def step_span(steps: list[int]) -> str:
    """'step 4', 'steps 4-7', or 'steps 4, 9' -- never a bare python list."""
    if not steps:
        return "no step located"
    ordered = sorted({int(s) for s in steps})
    if len(ordered) == 1:
        return f"step {ordered[0]}"
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"steps {ordered[0]}-{ordered[-1]}"
    return "steps " + ", ".join(str(s) for s in ordered)


def _step_at(trace: CanonicalTrace | None, index: int) -> CanonicalStep | None:
    if trace is None:
        return None
    for st in trace.steps:
        if st.index == index:
            return st
    return None


def _step_label(trace: CanonicalTrace | None, index: int) -> str:
    st = _step_at(trace, index)
    if st is None:
        return f"step {index}"
    detail = st.kind if not st.name else f"{st.kind}: {st.name}"
    return f"step {index} ({detail})"


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


INSTRUCTION_QUOTE = "verbatim instruction quote"
MECHANICAL_RULE = "mechanical rule (local check, not a quote from your instructions)"
TOOL_RESULT_EXCERPT = "tool result excerpt"
AGENT_STEP_EXCERPT = "agent step excerpt"


def excerpt_sources(f: Finding, trace: CanonicalTrace | None = None) -> list[str]:
    """Which supporting excerpts this finding actually carries.

    The spec asks every finding to quote the instructions or a tool result. A
    `rule_ref` is only an instruction quote when it really appears in the task
    or the instructions; the mechanical lane cites its own rule name instead,
    which is grounded in the declared tool schema but is not a quote. Saying
    which is which is the difference between an audit and a claim.
    """
    sources: list[str] = []
    rule = (f.rule_ref or "").strip()
    if rule:
        haystack = _norm(f"{getattr(trace, 'instructions', '')} {getattr(trace, 'task', '')}")
        quoted = len(rule) >= 12 and haystack and _norm(rule) in haystack
        sources.append(INSTRUCTION_QUOTE if quoted else MECHANICAL_RULE)
    if any((e.source or "") == "instructions" for e in f.evidence):
        if INSTRUCTION_QUOTE not in sources:
            sources.insert(0, INSTRUCTION_QUOTE)
    cited = [st for st in (_step_at(trace, e.step) for e in f.evidence) if st is not None]
    if any((e.source or "") == "tool_result" for e in f.evidence) or any(
        st.kind in _ENV_KINDS or st.is_error for st in cited
    ):
        sources.append(TOOL_RESULT_EXCERPT)
    elif f.evidence:
        sources.append(AGENT_STEP_EXCERPT)
    return sources


def has_supporting_excerpt(f: Finding, trace: CanonicalTrace | None = None) -> bool:
    """True when the finding is anchored to something outside its own assertion."""
    return bool(excerpt_sources(f, trace))


def excerpt_coverage(
    results: list[TraceResult],
    traces: dict[str, CanonicalTrace] | None = None,
) -> dict:
    """How well confirmed findings are sourced, counted rather than asserted."""
    traces = traces or {}
    tally = {
        "n_confirmed": 0,
        "with_instruction_quote": 0,
        "with_tool_result_excerpt": 0,
        "with_mechanical_rule_only": 0,
        "with_nothing": 0,
        "note": (
            "Every confirmed finding should quote the instructions or a tool result. "
            "A mechanical rule is grounded in the declared tool schema, not in your "
            "instruction text."
        ),
    }
    for r in results:
        for f in confirmed_findings(r):
            sources = excerpt_sources(f, traces.get(r.trace_id))
            tally["n_confirmed"] += 1
            if INSTRUCTION_QUOTE in sources:
                tally["with_instruction_quote"] += 1
            if TOOL_RESULT_EXCERPT in sources:
                tally["with_tool_result_excerpt"] += 1
            if not sources:
                tally["with_nothing"] += 1
            elif sources == [MECHANICAL_RULE] or sources == [MECHANICAL_RULE, AGENT_STEP_EXCERPT]:
                tally["with_mechanical_rule_only"] += 1
    return tally


# --------------------------------------------------------------------------
# terminal rendering
# --------------------------------------------------------------------------


def trace_console_lines(
    result: TraceResult,
    index: int = 0,
    total: int = 0,
    session_spend: float | None = None,
) -> list[str]:
    """Two compact ASCII lines per trace: what it cost, and what was found."""
    conf, abst = confirmed_findings(result), abstained_findings(result)
    counter = f"[{index}/{total}]" if total else ""
    status = "ERROR" if result.status == "error" else result.status
    money = f"${result.cost_usd:.3f}"
    head = (
        f"  {counter:>7} {result.trace_id:<22.22} {status:<18.18} "
        f"{money:>8} {result.latency_s:>7.1f}s"
    )
    if session_spend is not None:
        head += f"  session ${session_spend:.3f}"
    lines = [head]

    if result.status == "error":
        why = " ".join((result.error or "no detail recorded").split())
        lines += [f"           {chunk}" for chunk in _wrap(f"FAILED: {why}", 100)]
        return [ascii_safe(x) for x in lines]

    if conf:
        detail = f"{len(conf)} confirmed ({_family_brief(family_counts(conf))})"
    else:
        detail = "0 confirmed - nothing provable from this trace"
    detail += f" | {len(abst)} abstained"
    lines.append(f"           {detail}")
    if result.degraded:
        lines.append(f"           degraded: {'; '.join(result.degraded)[:140]}")
    return [ascii_safe(x) for x in lines]


_ARTIFACT_GUIDE = [
    ("findings.jsonl", "one JSON record per trace, findings included"),
    ("findings.csv", "one row per finding, for a spreadsheet"),
    ("reports/<trace_id>.md", "readable per-trace report: verdict, then evidence"),
    ("summary.md", "this summary, as a file"),
    ("summary.json", "machine-readable run totals"),
]


def run_console_summary(
    summary: dict,
    out_dir: str | Path,
    eval_overall: dict | None = None,
    wrote_eval: bool = False,
) -> str:
    """The end-of-run block. The only thing many users will read."""
    width = 74
    rule = "-" * width
    cost, lat = summary["cost_usd"], summary["latency_s"]
    lines = [
        "=" * width,
        f"AUDIT SUMMARY   {summary['n_traces']} trace(s) analysed",
        rule,
        f"CONFIRMED FINDINGS      {summary['n_confirmed_findings']}",
    ]
    for fam, n in (summary.get("confirmed_by_family") or {}).items():
        lines.append(f"    {fam:<24}{n}")
    if not summary.get("confirmed_by_family"):
        lines.append("    (none)")
    lines += [
        f"ABSTAINED               {summary['n_insufficient_evidence']}"
        "   insufficient evidence, never scored as a false positive",
        rule,
        f"COST      mean ${cost['mean']:.4f}   p95 ${cost['p95']:.4f}   "
        f"total ${cost['total']:.4f}",
        f"LATENCY   mean {lat['mean']:.1f}s   p95 {lat['p95']:.1f}s   "
        f"total {lat['total']:.1f}s",
    ]
    statuses = summary.get("statuses") or {}
    if set(statuses) - {"ok", "deterministic_only"}:
        lines.append("STATUS    " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())))
    cov = summary.get("excerpt_coverage") or {}
    if cov.get("n_confirmed"):
        lines.append(
            f"EVIDENCE  {cov['with_instruction_quote']} quote your instructions, "
            f"{cov['with_tool_result_excerpt']} quote a tool result, "
            f"{cov['with_mechanical_rule_only']} cite a mechanical rule only"
        )
    if cov.get("with_nothing"):
        lines.append(f"WARNING   {cov['with_nothing']} confirmed finding(s) carry no excerpt at all")
    if eval_overall:
        lines += [
            rule,
            f"SCORED    P {eval_overall['precision']:.3f}  R {eval_overall['recall']:.3f}  "
            f"F1 {eval_overall['f1']:.3f}   TP {eval_overall['tp']} "
            f"FP {eval_overall['fp']} FN {eval_overall['fn']}",
        ]
    lines += [rule, f"ARTIFACTS {Path(out_dir)}"]
    guide = list(_ARTIFACT_GUIDE)
    if wrote_eval:
        guide.append(("eval.md / eval.json", "precision, recall, F1, clean-trace FPs, cost, latency"))
    for name, what in guide:
        lines.append(f"    {name:<24}{what}")
    lines.append("=" * width)
    return ascii_safe("\n".join(lines))


# --------------------------------------------------------------------------
# per-trace markdown
# --------------------------------------------------------------------------


def _quote_block(text: str, limit: int = 600) -> str:
    body = (text or "").strip()
    if not body:
        return "> (not recorded in this trace)"
    if len(body) > limit:
        body = body[:limit].rstrip() + " [...]"
    return "\n".join(f"> {line}" if line.strip() else ">" for line in body.splitlines())


def finding_markdown(f: Finding, trace: CanonicalTrace | None = None, ordinal: int = 0) -> str:
    """One finding: what, where, which rule, what evidence, what to do instead."""
    label = f"{ordinal}. " if ordinal else ""
    head = f"### {label}`{f.family}` at {step_span(f.steps)}"
    meta = [
        f"- **Severity:** {f.severity}",
        f"- **Status:** {'CONFIRMED' if f.status == CONFIRMED else f.status.upper()}",
        f"- **Finding id:** `{f.finding_id}` (detected by: {f.source})",
    ]
    body = [head, "", FAMILY_GLOSSARY.get(f.family, ""), "", *meta, "", f"**What happened.** {f.description}"]
    sources = excerpt_sources(f, trace)
    if f.explanation and _norm(f.explanation) != _norm(f.rule_ref):
        body += ["", f"**Why it is wrong.** {f.explanation}"]
    if f.rule_ref:
        label = (
            "Rule broken, quoted from your instructions"
            if INSTRUCTION_QUOTE in sources
            else "Rule applied (a local mechanical check, not a quote from your instructions)"
        )
        body += ["", f"**{label}.**", _quote_block(f.rule_ref)]
    if f.available_info:
        body += ["", f"**What the agent could see at that point.** {f.available_info}"]
    if f.alternative:
        body += ["", f"**Cheaper alternative available then.** {f.alternative}"]
    body += ["", "**Supporting excerpts.**"]
    seen: set[tuple[int, str]] = set()
    for e in f.evidence:
        quote = (e.quote or "").strip()[:300]
        if (e.step, quote) in seen:
            continue
        seen.add((e.step, quote))
        body.append(f"- {_step_label(trace, e.step)}: `{quote}`")
    if not seen:
        body.append("- (none recorded)")
    body.append("")
    body.append(f"**Excerpt sources.** {'; '.join(sources) if sources else 'NONE -- unsupported'}")
    ver = f.verification
    body += [
        "",
        f"**Verification.** steps locatable={ver.locatable}, "
        f"quotes verified {ver.quote_verified}/{ver.quote_total}, "
        f"family gate={ver.family_gate}, adjudicator={ver.adjudicator or 'n/a'}. {ver.reason}",
        "",
    ]
    return "\n".join(body)


def _verdict_paragraph(result: TraceResult, conf: list[Finding], abst: list[Finding]) -> list[str]:
    if result.status == "error":
        return [
            f"**This trace was not audited.** The run failed: {result.error or 'no detail recorded'}",
            "",
            "No conclusion should be drawn about the agent from this report.",
        ]
    if conf:
        head = (
            f"**{len(conf)} problem(s) confirmed**: {_family_brief(family_counts(conf))}."
        )
    else:
        head = "**Nothing confirmed.** Every action was either justified or not provable from this trace."
    tail = (
        f" {len(abst)} further concern(s) were **abstained** for insufficient evidence; "
        "those are not accusations."
        if abst
        else " No abstentions."
    )
    return [head + tail]


def trace_report(result: TraceResult, trace: CanonicalTrace | None = None) -> str:
    """Verdict first, then the ask, then each finding, then abstentions."""
    conf, abst = confirmed_findings(result), abstained_findings(result)
    body = [
        f"# Audit report: `{result.trace_id}`",
        "",
        "## Verdict",
        "",
        *_verdict_paragraph(result, conf, abst),
        "",
        "| confirmed | abstained (insufficient evidence) | status | cost | latency |",
        "|---:|---:|---|---:|---:|",
        f"| **{len(conf)}** | {len(abst)} | {result.status} | "
        f"${result.cost_usd:.4f} | {result.latency_s:.1f}s |",
        "",
    ]
    if result.degraded:
        body += ["> **Degraded run.** " + "; ".join(result.degraded), ""]
    loss, notes = split_capture_gaps(trace.capture_gaps if trace is not None else [])
    if loss:
        body += [
            "> **Capture gaps.** " + "; ".join(loss[:4]),
            "> Nothing is confirmed on missing data, so these are abstained rather than guessed.",
            "",
        ]
    if notes:
        body += ["_Adapter notes: " + "; ".join(notes[:3]) + "_", ""]

    if trace is not None:
        body += ["## What the agent was asked to do", ""]
        body += ["**Task**", _quote_block(trace.task, 800), ""]
        if trace.instructions:
            body += ["**Instructions**", _quote_block(trace.instructions, 1200), ""]
        if trace.tools:
            names = ", ".join(f"`{t.name}`" for t in trace.tools[:25])
            body += [f"**Tools available:** {names}", ""]
        body += [f"**Trace:** {len(trace.steps)} steps, source format `{trace.source_format}`", ""]

    if conf:
        body += ["## What went wrong", ""]
        body += [finding_markdown(f, trace, i) for i, f in enumerate(conf, 1)]
    elif result.status != "error":
        body += [
            "## What went wrong",
            "",
            "Nothing was confirmed. This is a clean verdict, not a silent failure: "
            "the checks below ran and none of them held.",
            "",
        ]

    if abst:
        body += [
            f"## {ABSTAINED_HEADING}",
            "",
            f"> {ABSTAINED_NOTE}",
            "",
        ]
        body += [finding_markdown(f, trace, i) for i, f in enumerate(abst, 1)]

    if result.checks:
        body += [
            "## Checks derived for this trace",
            "",
            "Compiled from this task, these instructions, and these tools -- not a fixed checklist.",
            "",
        ]
        for c in result.checks:
            src = f" Source: _{c.source_refs[0][:200]}_" if c.source_refs else ""
            body.append(f"- `{c.check_id}` [{c.family}, {c.lane} lane] {c.description}{src}")
        body.append("")
    return "\n".join(body)


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------

_CSV_FIELDS = [
    "trace_id",
    "finding_id",
    "family",
    "status",
    "is_confirmed",
    "severity",
    "steps",
    "description",
    "explanation",
    "rule_ref",
    "available_info",
    "alternative",
    "evidence_steps",
    "evidence_quotes",
    "excerpt_sources",
    "detected_by",
    "confidence",
    "trace_cost_usd",
    "trace_latency_s",
]


def _csv_row(r: TraceResult, f: Finding, trace: CanonicalTrace | None) -> dict:
    return {
        "trace_id": r.trace_id,
        "finding_id": f.finding_id,
        "family": f.family,
        "status": f.status,
        "is_confirmed": "yes" if f.status == CONFIRMED else "no",
        "severity": f.severity,
        "steps": " ".join(map(str, f.steps)),
        "description": f.description,
        "explanation": f.explanation,
        "rule_ref": f.rule_ref,
        "available_info": f.available_info,
        "alternative": f.alternative,
        "evidence_steps": " ".join(str(e.step) for e in f.evidence),
        "evidence_quotes": " || ".join((e.quote or "").replace("\n", " ")[:300] for e in f.evidence),
        "excerpt_sources": "; ".join(excerpt_sources(f, trace)),
        "detected_by": f.source,
        "confidence": f.confidence,
        "trace_cost_usd": round(r.cost_usd, 6),
        "trace_latency_s": round(r.latency_s, 3),
    }


def build_summary(
    results: list[TraceResult],
    traces: dict[str, CanonicalTrace] | None = None,
) -> dict:
    traces = traces or {}
    costs = [r.cost_usd for r in results]
    lats = [r.latency_s for r in results if r.latency_s]
    all_conf = [f for r in results for f in confirmed_findings(r)]
    n_ins = sum(len(abstained_findings(r)) for r in results)
    coverage = excerpt_coverage(results, traces)
    return {
        "n_traces": len(results),
        "n_confirmed_findings": len(all_conf),
        "confirmed_by_family": family_counts(all_conf),
        "n_insufficient_evidence": n_ins,
        "abstention_note": ABSTAINED_NOTE,
        "excerpt_coverage": coverage,
        "n_confirmed_without_excerpt": coverage["with_nothing"],
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
        "statuses": {s: sum(1 for r in results if r.status == s) for s in {r.status for r in results}},
        "per_trace": [
            {
                "trace_id": r.trace_id,
                "status": r.status,
                "n_confirmed": len(confirmed_findings(r)),
                "confirmed_by_family": family_counts(confirmed_findings(r)),
                "n_insufficient_evidence": len(abstained_findings(r)),
                "cost_usd": round(r.cost_usd, 4),
                "latency_s": round(r.latency_s, 2),
                "error": r.error,
            }
            for r in results
        ],
    }


def write_run_artifacts(
    out_dir: str | Path,
    results: list[TraceResult],
    cfg=None,
    dataset_hash: str = "",
    traces: dict[str, CanonicalTrace] | None = None,
) -> Path:
    traces = traces or {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports = out / "reports"
    reports.mkdir(exist_ok=True)

    jsonl = out / "findings.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
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
                writer.writerow(_csv_row(r, f, traces.get(r.trace_id)))

    for r in results:
        (reports / f"{r.trace_id}.md").write_text(
            trace_report(r, traces.get(r.trace_id)), encoding="utf-8"
        )

    summary = build_summary(results, traces)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(summary_to_md(summary), encoding="utf-8")

    manifest = {
        "cfg_hash": config_fingerprint(cfg) if cfg is not None else "",
        "dataset_hash": dataset_hash,
        "n_traces": len(results),
        "result_ids": [r.trace_id for r in results],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "report.md").write_text(
        summary_to_md(summary)
        + "\n\n"
        + "\n\n".join(trace_report(r, traces.get(r.trace_id)) for r in results[:12]),
        encoding="utf-8",
    )
    return out


def summary_to_md(summary: dict) -> str:
    cost, lat = summary["cost_usd"], summary["latency_s"]
    lines = [
        "# Run summary",
        "",
        f"{summary['n_traces']} trace(s). "
        f"**{summary['n_confirmed_findings']} confirmed finding(s)**, "
        f"{summary['n_insufficient_evidence']} abstained.",
        "",
        f"> Abstained = insufficient evidence. {ABSTAINED_NOTE}",
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
    cov = summary.get("excerpt_coverage") or {}
    if cov.get("n_confirmed"):
        lines += [
            "",
            "## How the confirmed findings are sourced",
            "",
            f"- quote your instructions verbatim: {cov['with_instruction_quote']}"
            f" of {cov['n_confirmed']}",
            f"- quote a tool result: {cov['with_tool_result_excerpt']} of {cov['n_confirmed']}",
            f"- cite a local mechanical rule only: {cov['with_mechanical_rule_only']}"
            " (grounded in the declared tool schema, not in your instruction text)",
            f"- cite nothing at all: {cov['with_nothing']}",
        ]
        if cov.get("with_nothing"):
            lines += [
                "",
                f"> **{cov['with_nothing']} confirmed finding(s) carry no excerpt.** "
                "Every confirmed finding is supposed to quote one.",
            ]
    lines += ["", "## Per trace", "", "| trace | status | confirmed | abstained | cost | latency |", "|---|---|---:|---:|---:|---:|"]
    for t in summary.get("per_trace", []):
        lines.append(
            f"| `{t['trace_id']}` | {t['status']} | {t['n_confirmed']} | "
            f"{t['n_insufficient_evidence']} | ${t['cost_usd']:.4f} | {t['latency_s']:.1f}s |"
        )
    return "\n".join(lines) + "\n"


def dataset_hash(dataset_dir: str | Path) -> str:
    root = Path(dataset_dir)
    h = hashlib.sha256()
    if root.is_file():
        h.update(root.read_bytes())
        return h.hexdigest()[:16]
    index = root / "index.json"
    if index.is_file():
        h.update(index.read_bytes())
        return h.hexdigest()[:16]
    for path in sorted(root.rglob("*")):
        if path.is_file():
            h.update(path.name.encode())
            h.update(str(path.stat().st_size).encode())
    return h.hexdigest()[:16]
