from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ..schemas import FAMILIES, Family, Finding


class LabelFinding(BaseModel):
    family: Family
    steps: list[int] = Field(default_factory=list)
    description: str = ""
    severity: str = "major"
    notes: str = ""


class ExcludedFinding(BaseModel):
    source_type: str = ""
    reason: str = ""
    steps: list[int] = Field(default_factory=list)
    description: str = ""


class Label(BaseModel):
    trace_id: str
    clean: bool = False
    findings: list[LabelFinding] = Field(default_factory=list)
    excluded_findings: list[ExcludedFinding] = Field(default_factory=list)
    localization_only: list[ExcludedFinding] = Field(default_factory=list)
    source_benchmark: str = ""
    mapping_version: str = ""
    gold_completeness: str = "complete"
    annotator: str = ""
    review: str = ""
    notes: str = ""
    split: str = "dev"


def load_labels(labels_dir: str | Path) -> dict[str, Label]:
    labels_dir = Path(labels_dir)
    out: dict[str, Label] = {}
    for path in sorted(labels_dir.glob("*.json")):
        label = Label.model_validate_json(path.read_text(encoding="utf-8"))
        for f in label.findings:
            if f.family not in FAMILIES:
                raise ValueError(f"{path.name}: unknown family {f.family!r}")
        out[label.trace_id] = label
    return out


def _interval(steps: list[int]) -> tuple[int, int]:
    if not steps:
        return (-1, -1)
    return (min(steps), max(steps))


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    if a[0] < 0 or b[0] < 0:
        return 0
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def _center_dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    if a[0] < 0 or b[0] < 0:
        return float("inf")
    return abs((a[0] + a[1]) / 2 - (b[0] + b[1]) / 2)


def _eligible(p: Finding, lab: LabelFinding, mode: str, center_slack: int) -> int:
    if lab.family != p.family:
        return -1
    pi, loi = _interval(p.steps), _interval(lab.steps)
    ov = _overlap(pi, loi)
    if mode == "strict":
        return ov if ov > 0 else -1
    if ov > 0 or _center_dist(pi, loi) <= center_slack:
        return ov if ov > 0 else 0
    return -1


def match_findings(
    predicted: list[Finding],
    labels: list[LabelFinding],
    mode: str = "strict",
    center_slack: int = 3,
) -> tuple[list[tuple[Finding, LabelFinding]], list[Finding], list[LabelFinding]]:
    preds = [p for p in predicted if p.status == "confirmed"]
    adj: list[list[tuple[int, int]]] = []
    for p in preds:
        row = []
        for li, lab in enumerate(labels):
            ov = _eligible(p, lab, mode, center_slack)
            if ov >= 0:
                row.append((ov, li))
        row.sort(key=lambda t: -t[0])
        adj.append(row)
    match_of_label: dict[int, int] = {}

    def try_assign(pi: int, seen: set[int]) -> bool:
        for _ov, li in adj[pi]:
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
