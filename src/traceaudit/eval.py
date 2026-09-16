"""Score confirmed findings against human labels.

Strict match: same family and overlapping steps. Abstentions are never
false positives. Development scores are not held-out.
"""

from __future__ import annotations

import json
from pathlib import Path

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .report import confirmed_findings
from .schemas import FAMILIES, Family, Finding, TraceResult


class LabelFinding(BaseModel):
    model_config = ConfigDict(extra="ignore")
    family: Family
    steps: list[int] = Field(default_factory=list)
    description: str = ""
    severity: str = "major"
    notes: str = ""
    error_class: str = ""
    confidence: int | None = None


class Label(BaseModel):
    model_config = ConfigDict(extra="ignore")
    trace_id: str
    clean: bool = False
    findings: list[LabelFinding] = Field(default_factory=list)
    notes: str = ""
    annotator: str = ""
    review: str = ""
    split: str = "dev"
    meta: dict[str, Any] = Field(default_factory=dict)


def load_labels(labels_dir: str | Path) -> dict[str, Label]:
    labels_dir = Path(labels_dir)
    if labels_dir.is_file():
        paths = [labels_dir]
    else:
        paths = sorted(labels_dir.glob("*.json"))
    out: dict[str, Label] = {}
    for path in paths:
        label = Label.model_validate_json(path.read_text(encoding="utf-8"))
        for finding in label.findings:
            if finding.family not in FAMILIES:
                raise ValueError(f"{path.name}: unknown family {finding.family!r}")
        out[label.trace_id] = label
    if not out:
        raise FileNotFoundError(f"no label JSON at {labels_dir}")
    return out


def _overlap(pred: list[int], gold: list[int]) -> bool:
    return bool(set(pred) & set(gold))


def match_findings(
    predicted: list[Finding],
    labels: list[LabelFinding],
) -> tuple[list[tuple[Finding, LabelFinding]], list[Finding], list[LabelFinding]]:
    preds = [p for p in predicted if p.status == "confirmed"]
    adj: list[list[int]] = []
    for pred in preds:
        hits = [
            i
            for i, lab in enumerate(labels)
            if pred.family == lab.family and _overlap(pred.steps, lab.steps)
        ]
        adj.append(hits)
    match_of_label: dict[int, int] = {}

    def try_assign(pi: int, seen: set[int]) -> bool:
        for li in adj[pi]:
            if li in seen:
                continue
            seen.add(li)
            if li not in match_of_label or try_assign(match_of_label[li], seen):
                match_of_label[li] = pi
                return True
        return False

    for pi in sorted(range(len(preds)), key=lambda i: len(adj[i])):
        try_assign(pi, set())
    matched_preds = set(match_of_label.values())
    matches = [(preds[pi], labels[li]) for li, pi in sorted(match_of_label.items())]
    fps = [p for i, p in enumerate(preds) if i not in matched_preds]
    misses = [lab for li, lab in enumerate(labels) if li not in match_of_label]
    return matches, fps, misses


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def mappable_gold_findings(label: Label) -> list[LabelFinding]:
    """Gold findings on the closed families.

    Family ``other`` is excluded (no honest closed mapping). Injected review
    is excluded. This subset is not held-out.
    """
    if (label.review or "").lower() == "injected":
        return []
    return [f for f in label.findings if f.family != "other"]


def mappable_predictions(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.family != "other"]


def score_results(results: list[TraceResult], labels: dict[str, Label]) -> dict:
    by_id = {r.trace_id: r for r in results}
    tp = fp = fn = 0
    m_tp = m_fp = m_fn = 0
    m_gold_n = 0
    m_pred_n = 0
    m_traces = 0
    abstain = 0
    clean_fp = 0
    n_clean = 0
    per_trace: list[dict] = []
    missing: list[str] = []
    for tid, label in sorted(labels.items()):
        result = by_id.get(tid)
        aborted = bool(
            result is not None
            and result.status in {"error", "model_error"}
            and "payment required"
            in ((result.error or "") + " ".join(result.degraded)).lower()
        )
        if result is None or aborted:
            if result is None:
                missing.append(tid)
            per_trace.append(
                {
                    "trace_id": tid,
                    "status": "missing" if result is None else result.status,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "mappable_tp": 0,
                    "mappable_fp": 0,
                    "mappable_fn": 0,
                    "n_mappable_gold": 0,
                    "n_mappable_predicted": 0,
                    "n_confirmed": 0,
                    "n_abstained": 0,
                    "unscored": True,
                    "reason": "payment required" if aborted else "no result",
                }
            )
            continue
        matches, fps, misses = match_findings(result.findings, label.findings)
        if (label.review or "").lower() == "injected":
            map_gold = []
            map_pred = []
        else:
            map_gold = mappable_gold_findings(label)
            map_pred = mappable_predictions(result.findings)
        map_matches, map_fps, map_misses = match_findings(map_pred, map_gold)
        n_abs = sum(1 for f in result.findings if f.status == "insufficient_evidence")
        abstain += n_abs
        tp += len(matches)
        fp += len(fps)
        fn += len(misses)
        m_tp += len(map_matches)
        m_fp += len(map_fps)
        m_fn += len(map_misses)
        m_gold_n += len(map_gold)
        m_pred_n += sum(1 for p in map_pred if p.status == "confirmed")
        if map_gold:
            m_traces += 1
        if label.clean:
            n_clean += 1
            clean_fp += len(fps)
        confirmed = confirmed_findings(result)
        per_trace.append(
            {
                "trace_id": tid,
                "status": result.status,
                "tp": len(matches),
                "fp": len(fps),
                "fn": len(misses),
                "mappable_tp": len(map_matches),
                "mappable_fp": len(map_fps),
                "mappable_fn": len(map_misses),
                "n_mappable_gold": len(map_gold),
                "n_mappable_predicted": sum(
                    1 for p in map_pred if p.status == "confirmed"
                ),
                "n_confirmed": len(confirmed),
                "n_abstained": n_abs,
                "fp_ids": [p.finding_id for p in fps],
                "fn_descriptions": [m.description[:160] for m in misses],
                "predicted_classes": [
                    {
                        "family": p.family,
                        "error_class": p.error_class,
                        "confidence": int(p.confidence),
                        "steps": p.steps,
                    }
                    for p in confirmed
                ],
                "cost_usd": round(result.cost_usd, 4),
                "latency_s": round(result.latency_s, 2),
            }
        )
    overall = _prf(tp, fp, fn)
    mappable = {
        "note": (
            "Mappable subset: gold findings whose family is not `other`; "
            "review=injected excluded. Not held-out."
        ),
        "held_out": False,
        "n_gold_findings": m_gold_n,
        "n_predicted": m_pred_n,
        "n_traces_with_mappable_gold": m_traces,
        **_prf(m_tp, m_fp, m_fn),
    }
    return {
        "split": "development",
        "note": "Development labels. Do not quote as held-out.",
        "n_labelled": len(labels),
        "n_evaluated": len(labels) - len(missing),
        "missing": missing,
        "overall": overall,
        "mappable": mappable,
        "n_abstained": abstain,
        "clean_traces": {"n": n_clean, "false_positives": clean_fp},
        "per_trace": per_trace,
    }


def eval_console(report: dict) -> str:
    o = report["overall"]
    clean = report["clean_traces"]
    lines = [
        "LABELLED EVAL   development — not held-out",
        f"  strict  P {o['precision']:.3f}  R {o['recall']:.3f}  F1 {o['f1']:.3f}"
        f"   TP {o['tp']}  FP {o['fp']}  FN {o['fn']}",
    ]
    m = report.get("mappable") or {}
    if m:
        lines.append(
            f"  mappable P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}"
            f"   TP {m['tp']}  FP {m['fp']}  FN {m['fn']}"
            f"   (family≠other, not injected; not held-out; "
            f"{m.get('n_gold_findings', 0)} gold)"
        )
    lines += [
        f"  abstentions {report['n_abstained']} (never scored as FP)",
        f"  clean-trace FPs: {clean['false_positives']} of {clean['n']}",
    ]
    if report.get("missing"):
        lines.append("  not scored (no result): " + ", ".join(report["missing"]))
    for row in report.get("per_trace") or []:
        mark = f"  {row['trace_id']:<22} tp {row['tp']} fp {row['fp']} fn {row['fn']}"
        if row.get("unscored"):
            mark += f"  unscored ({row.get('reason') or row.get('status')})"
        lines.append(mark)
    return "\n".join(lines)


def write_eval(out_dir: str | Path, report: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    o = report["overall"]
    lines = [
        "# Labelled evaluation (development — not held-out)",
        "",
        report.get("note") or "",
        "",
        f"Strict match is the same family plus overlapping steps. "
        f"`error_class` and `confidence` are recorded but not scored. "
        f"`insufficient_evidence` is an abstention and is never a false positive.",
        "",
        f"| set | P | R | F1 | TP | FP | FN | abstentions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| full (not held-out) | {o['precision']:.3f} | {o['recall']:.3f} | {o['f1']:.3f} | "
        f"{o['tp']} | {o['fp']} | {o['fn']} | {report['n_abstained']} |",
    ]
    m = report.get("mappable") or {}
    if m:
        lines.append(
            f"| mappable (family≠other, not injected; not held-out) | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
            f"{m['tp']} | {m['fp']} | {m['fn']} | |"
        )
    lines += [
        "",
        f"False positives on clean traces: "
        f"{report['clean_traces']['false_positives']} of {report['clean_traces']['n']}.",
    ]
    if m:
        lines.append(
            f"Mappable gold findings scored: {m.get('n_gold_findings', 0)} "
            f"on {m.get('n_traces_with_mappable_gold', 0)} traces (not held-out)."
        )
    lines += [
        "",
        "## Per trace",
        "",
        "| trace | status | TP | FP | FN | mappable TP/FP/FN | confirmed | abstained | cost | latency |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in report.get("per_trace") or []:
        lines.append(
            f"| `{row['trace_id']}` | {row.get('status', '')} | {row['tp']} | {row['fp']} | "
            f"{row['fn']} | {row.get('mappable_tp', 0)}/{row.get('mappable_fp', 0)}/"
            f"{row.get('mappable_fn', 0)} | {row.get('n_confirmed', 0)} | "
            f"{row.get('n_abstained', 0)} | "
            f"${row.get('cost_usd', 0) or 0:.4f} | {row.get('latency_s', 0) or 0:.1f}s |"
        )
    (out / "eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
