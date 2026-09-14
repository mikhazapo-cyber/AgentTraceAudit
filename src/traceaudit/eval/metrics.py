from __future__ import annotations

import json
from pathlib import Path

from ..report import _percentile, load_results
from ..schemas import FAMILIES, TraceResult
from .matching import load_labels, match_findings


def _prf(d: dict) -> dict:
    tp, fp, fn = d["tp"], d["fp"], d["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), **d}


_PRECISION_WARNING = (
    "> **Precision is not interpretable on this gold set.** It marks one injected "
    "error per trace, so a correct finding elsewhere counts as a false positive. "
    "Read recall and step localisation here; take precision from a fully "
    "annotated set."
)


def _coverage(labels: dict) -> dict:
    """Share of the gold set these metrics actually score.

    Converted corpora carry error types the seven families cannot represent.
    Those sit in `excluded_findings` or `localization_only` instead of counting
    as misses, so P/R/F1 is meaningless without this.
    """
    scored = sum(len(lab.findings) for lab in labels.values())
    excluded = [e for lab in labels.values() for e in lab.excluded_findings]
    loc_only = [e for lab in labels.values() for e in lab.localization_only]
    total = scored + len(excluded) + len(loc_only)
    by_reason: dict[str, int] = {}
    for entry in excluded:
        key = entry.reason or "unspecified"
        by_reason[key] = by_reason.get(key, 0) + 1
    by_type: dict[str, int] = {}
    for entry in excluded + loc_only:
        if entry.source_type:
            by_type[entry.source_type] = by_type.get(entry.source_type, 0) + 1
    completeness = sorted({lab.gold_completeness for lab in labels.values()})
    return {
        "gold_findings_total": total,
        "scored": scored,
        "excluded": len(excluded),
        "localization_only": len(loc_only),
        "scored_fraction": round(scored / total, 4) if total else None,
        "excluded_by_reason": by_reason,
        "out_of_scope_by_source_type": dict(sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))),
        "source_benchmarks": sorted({lab.source_benchmark for lab in labels.values() if lab.source_benchmark}),
        "mapping_versions": sorted({lab.mapping_version for lab in labels.values() if lab.mapping_version}),
        "gold_completeness": completeness,
        "precision_interpretable": all(c == "complete" for c in completeness),
        "splits": sorted({getattr(lab, "split", "dev") or "dev" for lab in labels.values()}),
    }


def _split_label(splits: list[str]) -> str:
    """Name the split so development numbers can never be read as held-out ones."""
    held = {"heldout", "held-out", "test"}
    seen = {str(s) for s in splits}
    if seen and seen <= held:
        return "held-out"
    if seen & held:
        return "mixed development + held-out"
    return "development"


def _coverage_markdown(cov: dict) -> list[str]:
    """State the scored share above any P/R/F1."""
    if not cov:
        return []
    if not cov.get("precision_interpretable", True):
        prefix = [_PRECISION_WARNING, ""]
    elif not (cov.get("excluded") or cov.get("localization_only")):
        return []
    else:
        prefix = []
    total, scored = (cov.get("gold_findings_total") or 0, cov.get("scored") or 0)
    pct = f" ({cov['scored_fraction'] * 100:.0f}%)" if cov.get("scored_fraction") is not None else ""
    lines = prefix + [
        f"> **Scored coverage.** **{scored} of {total} gold findings**{pct}. "
        f"Outside this taxonomy: {cov.get('excluded', 0)}. "
        f"Family-ambiguous, localisation only: {cov.get('localization_only', 0)}. "
        "Neither counts as a miss. Do not quote P/R/F1 without this line.",
        "",
    ]
    if cov.get("source_benchmarks"):
        detail = ", ".join(cov["source_benchmarks"])
        if cov.get("mapping_versions"):
            detail += f" · mapping v{', v'.join(cov['mapping_versions'])}"
        lines += [f"- gold source: {detail}", ""]
    if cov.get("excluded_by_reason"):
        lines += [
            "- excluded by reason: "
            + ", ".join(f"{k}={v}" for k, v in sorted(cov["excluded_by_reason"].items())),
            "",
        ]
    if cov.get("out_of_scope_by_source_type"):
        top = list(cov["out_of_scope_by_source_type"].items())[:8]
        lines += ["- out-of-scope source types: " + ", ".join(f"{k}={v}" for k, v in top), ""]
    return lines


def evaluate(
    results_path: str | Path,
    labels_dir: str | Path,
    modes: tuple[str, ...] = ("strict", "relaxed"),
    only_ids: set[str] | None = None,
    semantic_mode: str = "",
) -> dict:
    loaded = load_results(Path(results_path))
    results: dict[str, TraceResult] = {}
    for r in loaded:
        if r.trace_id not in results:
            results[r.trace_id] = r
            continue
        # A results file can hold the same trace twice (a re-run appended, a
        # duplicated line). Counting those findings twice inflates FP and makes
        # the run look worse than it was, so merge on identity.
        existing = results[r.trace_id]
        seen = {(f.finding_id, f.family, tuple(f.steps)) for f in existing.findings}
        for f in r.findings:
            key = (f.finding_id, f.family, tuple(f.steps))
            if key not in seen:
                seen.add(key)
                existing.findings.append(f)
    labels = load_labels(labels_dir)
    if only_ids is not None:
        labels = {k: v for k, v in labels.items() if k in only_ids}
    missing = sorted(set(labels) - set(results))
    per_trace = {}
    agg = {m: {"tp": 0, "fp": 0, "fn": 0} for m in modes}
    per_family = {f: {m: {"tp": 0, "fp": 0, "fn": 0} for m in modes} for f in FAMILIES}
    insufficient = []
    clean_fps = []
    costs, latencies = [], []
    n_missing = n_error = 0
    for tid, label in sorted(labels.items()):
        res = results.get(tid)
        if res is None:
            n_missing += 1
            for mode in modes:
                agg[mode]["fn"] += len(label.findings)
                for lf in label.findings:
                    per_family[lf.family][mode]["fn"] += 1
            per_trace[tid] = {
                "status": "missing",
                "n_confirmed": 0,
                "n_insufficient": 0,
                **{mode: {"tp": 0, "fp": 0, "fn": len(label.findings)} for mode in modes},
            }
            continue
        if res.status == "error":
            n_error += 1
        costs.append(res.cost_usd)
        latencies.append(res.latency_s)
        for f in res.findings:
            if f.status == "insufficient_evidence":
                insufficient.append(
                    {
                        "trace_id": tid,
                        "finding_id": f.finding_id,
                        "family": f.family,
                        "description": f.description[:200],
                    }
                )
        entry = {
            "status": res.status,
            "cost_usd": round(res.cost_usd, 4),
            "latency_s": round(res.latency_s, 2),
            "split": getattr(label, "split", "dev"),
            "n_confirmed": sum(1 for f in res.findings if f.status == "confirmed"),
            "n_insufficient": sum(1 for f in res.findings if f.status == "insufficient_evidence"),
        }
        for mode in modes:
            matches, fps, misses = match_findings(res.findings, label.findings, mode)
            agg[mode]["tp"] += len(matches)
            agg[mode]["fp"] += len(fps)
            agg[mode]["fn"] += len(misses)
            for p, _l in matches:
                per_family[p.family][mode]["tp"] += 1
            for p in fps:
                per_family[p.family][mode]["fp"] += 1
                if label.clean and mode == modes[0]:
                    clean_fps.append(
                        {
                            "trace_id": tid,
                            "finding_id": p.finding_id,
                            "family": p.family,
                            "description": p.description[:200],
                        }
                    )
            for m in misses:
                per_family[m.family][mode]["fn"] += 1
            entry[mode] = {
                "tp": len(matches),
                "fp": len(fps),
                "fn": len(misses),
                "fp_ids": [p.finding_id for p in fps],
                "fn_descriptions": [m.description[:160] for m in misses],
            }
        per_trace[tid] = entry

    n_clean = sum(1 for lab in labels.values() if lab.clean and lab.trace_id in results)
    coverage = _coverage(labels)
    return {
        "split": _split_label(coverage["splits"]),
        "semantic_mode": semantic_mode or "unknown",
        "scored_coverage": coverage,
        "n_labelled": len(labels),
        "n_evaluated": len([t for t in labels if t in results]),
        "n_missing": n_missing,
        "n_error": n_error,
        "missing_results_for_labels": missing,
        "overall": {m: _prf(agg[m]) for m in modes},
        "per_family": {
            f: {m: _prf(per_family[f][m]) for m in modes}
            for f in FAMILIES
            if any(per_family[f][m][k] for m in modes for k in ("tp", "fp", "fn"))
        },
        "false_positives_on_clean_traces": {
            "n_clean_traces": n_clean,
            "n_fp": len(clean_fps),
            "fp_per_clean_trace": round(len(clean_fps) / n_clean, 3) if n_clean else None,
            "detail": clean_fps,
        },
        "insufficient_evidence_tracked_separately": {
            "count": len(insufficient),
            "note": "Abstentions. Never counted as TP or FP.",
            "detail": insufficient,
        },
        "cost_usd": {
            "mean": round(sum(costs) / len(costs), 4) if costs else None,
            "p50": round(_percentile(costs, 0.5) or 0, 4),
            "p95": round(_percentile(costs, 0.95) or 0, 4),
            "max": round(max(costs), 4) if costs else None,
            "total": round(sum(costs), 4),
        },
        "latency_s": {
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "p50": round(_percentile(latencies, 0.5) or 0, 2),
            "p95": round(_percentile(latencies, 0.95) or 0, 2),
        },
        "per_trace": per_trace,
    }


def write_eval(out_path: str | Path, report: dict) -> None:
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def _num(value, places: int, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.{places}f}{suffix}"


def write_eval_md(out_path: str | Path, report: dict) -> None:
    o = report["overall"]["strict"]
    rel = report["overall"].get("relaxed", {})
    clean = report["false_positives_on_clean_traces"]
    cov = report["scored_coverage"]
    lines = [
        f"# Evaluation ({report.get('split', 'development')})",
        "",
        f"Mode: **{report.get('semantic_mode', 'unknown')}**. "
        f"`insufficient_evidence` is an abstention, never a false positive.",
        "",
        f"Gold: {cov['scored']} scored of {cov['gold_findings_total']} annotated "
        f"({cov['excluded']} excluded, {cov['localization_only']} localisation-only).",
        "",
    ]
    lines += _coverage_markdown(cov)
    lines += [
        "| Mode | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| strict | {o['precision']:.3f} | {o['recall']:.3f} | {o['f1']:.3f} | {o['tp']} | {o['fp']} | {o['fn']} |",
    ]
    if rel:
        lines.append(
            f"| relaxed | {rel['precision']:.3f} | {rel['recall']:.3f} | {rel['f1']:.3f} | "
            f"{rel['tp']} | {rel['fp']} | {rel['fn']} |"
        )
    lines += [
        "",
        f"False positives on clean traces: **{clean['n_fp']}** of {clean['n_clean_traces']}.",
        "",
        f"Cost mean/p95: ${_num(report['cost_usd']['mean'], 4)} / "
        f"${_num(report['cost_usd']['p95'], 4)}. "
        f"Latency mean/p95: {_num(report['latency_s']['mean'], 2, 's')} / "
        f"{_num(report['latency_s']['p95'], 2, 's')}.",
        "",
        f"Abstentions: {report['insufficient_evidence_tracked_separately']['count']}.",
        "",
        "## Per family (strict)",
        "",
    ]
    for fam, stats in report.get("per_family", {}).items():
        s = stats["strict"]
        lines.append(f"- `{fam}`: P {s['precision']:.2f} · R {s['recall']:.2f} · F1 {s['f1']:.2f} (tp {s['tp']} fp {s['fp']} fn {s['fn']})")
    lines += _per_trace_markdown(report.get("per_trace", {}))
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _per_trace_markdown(per_trace: dict) -> list[str]:
    """Per-trace cost and latency, so the aggregate is auditable, not just asserted."""
    if not per_trace:
        return []
    lines = [
        "",
        "## Per trace",
        "",
        "Abstentions are listed beside confirmed findings and are never scored either way.",
        "",
        "| trace | split | status | confirmed | abstained | TP | FP | FN | cost | latency |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tid, entry in sorted(per_trace.items()):
        strict = entry.get("strict") or {}
        lines.append(
            f"| `{tid}` | {entry.get('split', 'dev')} | {entry.get('status', '?')} "
            f"| {entry.get('n_confirmed', 0)} | {entry.get('n_insufficient', 0)} "
            f"| {strict.get('tp', 0)} | {strict.get('fp', 0)} | {strict.get('fn', 0)} "
            f"| ${_num(entry.get('cost_usd'), 4)} | {_num(entry.get('latency_s'), 1, 's')} |"
        )
    return lines
