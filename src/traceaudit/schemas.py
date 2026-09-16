from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

FAMILIES = (
    "instruction_violation",
    "incorrect_tool_use",
    "evidence_contradiction",
    "unsupported_success",
    "redundant_action",
    "ignored_feedback",
    "other",
)
Family = Literal[
    "instruction_violation",
    "incorrect_tool_use",
    "evidence_contradiction",
    "unsupported_success",
    "redundant_action",
    "ignored_feedback",
    "other",
]
Severity = Literal["critical", "major", "minor"]
StepKind = Literal["message", "tool_call", "tool_result", "outcome", "other"]
Verdict = Literal["confirmed", "rejected", "insufficient_evidence"]
TraceStatus = Literal["ok", "budget_exhausted", "model_error", "error"]

FAMILY_GLOSSARY: dict[str, str] = {
    "instruction_violation": "broke an explicit rule in the task or instructions",
    "incorrect_tool_use": "wrong tool, schema-invalid arguments, or a malformed call",
    "evidence_contradiction": "a statement that contradicts what a tool actually returned",
    "unsupported_success": "claimed a completed effect the trace does not evidence",
    "redundant_action": "repeated work that yielded no new information",
    "ignored_feedback": "kept the same failing approach after errors or feedback",
    "other": "a real problem outside the families above",
}

# Finer grain under the seven parent families. Eval still matches parent + steps.
ERROR_CLASSES = (
    "skipped_required_check",
    "broke_explicit_rule",
    "role_or_spec_deviation",
    "wrong_tool",
    "undeclared_tool",
    "malformed_call",
    "bad_arguments",
    "contradicts_tool_output",
    "invented_observation",
    "false_completion",
    "skipped_verification",
    "duplicate_successful_call",
    "ignored_tool_error",
    "ignored_user_or_peer",
    "other_error",
    "stale_or_lost_context",
)
ErrorClass = Literal[
    "skipped_required_check",
    "broke_explicit_rule",
    "role_or_spec_deviation",
    "wrong_tool",
    "undeclared_tool",
    "malformed_call",
    "bad_arguments",
    "contradicts_tool_output",
    "invented_observation",
    "false_completion",
    "skipped_verification",
    "duplicate_successful_call",
    "ignored_tool_error",
    "ignored_user_or_peer",
    "other_error",
    "stale_or_lost_context",
]

ERROR_CLASS_FAMILY: dict[str, Family] = {
    "skipped_required_check": "instruction_violation",
    "broke_explicit_rule": "instruction_violation",
    "role_or_spec_deviation": "instruction_violation",
    "wrong_tool": "incorrect_tool_use",
    "undeclared_tool": "incorrect_tool_use",
    "malformed_call": "incorrect_tool_use",
    "bad_arguments": "incorrect_tool_use",
    "contradicts_tool_output": "evidence_contradiction",
    "invented_observation": "evidence_contradiction",
    "false_completion": "unsupported_success",
    "skipped_verification": "unsupported_success",
    "duplicate_successful_call": "redundant_action",
    "ignored_tool_error": "ignored_feedback",
    "ignored_user_or_peer": "ignored_feedback",
    "other_error": "other",
    "stale_or_lost_context": "other",
}

FAMILY_DEFAULT_CLASS: dict[str, str] = {
    "instruction_violation": "broke_explicit_rule",
    "incorrect_tool_use": "bad_arguments",
    "evidence_contradiction": "contradicts_tool_output",
    "unsupported_success": "false_completion",
    "redundant_action": "duplicate_successful_call",
    "ignored_feedback": "ignored_tool_error",
    "other": "other_error",
}

ERROR_CLASS_GLOSSARY: dict[str, str] = {
    "skipped_required_check": "required auth/confirm/verify never run",
    "broke_explicit_rule": "violated a must/never/required line",
    "role_or_spec_deviation": "ignored role, task spec, or hard constraint",
    "wrong_tool": "used a tool that cannot do the job",
    "undeclared_tool": "called something not in the tool list",
    "malformed_call": "schema or arguments invalid",
    "bad_arguments": "well-formed call with wrong values",
    "contradicts_tool_output": "claim contradicts a tool observation",
    "invented_observation": "cites a result that is not in the trace",
    "false_completion": "said done or success without evidence",
    "skipped_verification": "no check that the claimed effect happened",
    "duplicate_successful_call": "identical successful call repeated",
    "ignored_tool_error": "same approach after a tool error",
    "ignored_user_or_peer": "ignored an explicit correction",
    "other_error": "real labelled problem that does not fit a closer class",
    "stale_or_lost_context": "acted on dropped or reset state",
}

ERROR_CLASS_PROMPT = (
    "error_class (exactly one; parent family in parentheses):\n"
    "- instruction_violation: skipped_required_check, broke_explicit_rule, role_or_spec_deviation\n"
    "- incorrect_tool_use: wrong_tool, undeclared_tool, malformed_call, bad_arguments\n"
    "- evidence_contradiction: contradicts_tool_output, invented_observation\n"
    "- unsupported_success: false_completion, skipped_verification\n"
    "- redundant_action: duplicate_successful_call\n"
    "- ignored_feedback: ignored_tool_error, ignored_user_or_peer\n"
    "- other: other_error, stale_or_lost_context"
)


def clamp_confidence(raw: object, default: int) -> int:
    if raw is None or raw == "":
        return int(default)
    try:
        n = float(raw)
    except (TypeError, ValueError):
        return int(default)
    if 0 < n < 1:
        n *= 100
    return int(round(max(0, min(100, n))))


def normalize_class_and_family(
    raw_class: object,
    raw_family: object,
) -> tuple[str, str] | None:
    """Map model output onto the closed class list.

    Known class wins and sets its parent. Omitted or unknown class uses
    the parent family's default so eval still matches that family.
    Unknown class with no parent becomes other_error / other.
    Returns None when there is no usable parent and no known class.
    """
    cls = str(raw_class or "").strip()
    fam = str(raw_family or "").strip()
    if fam not in FAMILIES:
        fam = ""
    if cls in ERROR_CLASS_FAMILY:
        return cls, ERROR_CLASS_FAMILY[cls]
    if fam:
        return FAMILY_DEFAULT_CLASS[fam], fam
    if not cls:
        return None
    return "other_error", "other"


class ToolSpec(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    declared: bool = True


class CanonicalStep(BaseModel):
    index: int
    kind: StepKind
    role: str = ""
    name: str = ""
    content: str = ""
    arguments: dict[str, Any] | None = None
    call_id: str = ""
    is_error: bool = False
    truncated: bool = False
    raw_len: int = 0
    source_ref: str = ""
    time: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)

    def text(self) -> str:
        parts = [self.name, self.content]
        if self.arguments is not None:
            import json

            try:
                parts.append(
                    json.dumps(self.arguments, ensure_ascii=False, sort_keys=True)
                )
            except (TypeError, ValueError):
                parts.append(str(self.arguments))
        return "\n".join(p for p in parts if p)


class CanonicalTrace(BaseModel):
    trace_id: str
    source_format: str
    task: str = ""
    instructions: str = ""
    tools: list[ToolSpec] = Field(default_factory=list)
    steps: list[CanonicalStep] = Field(default_factory=list)
    outcome: str = ""
    capture_gaps: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    def tool_names(self) -> set[str]:
        return {t.name for t in self.tools}


class DerivedCheck(BaseModel):
    check_id: str
    family: Family
    description: str
    condition: str = ""
    justified_when: str = ""
    severity: Severity = "major"
    source_refs: list[str] = Field(default_factory=list)
    source: str = ""


class Evidence(BaseModel):
    step: int
    quote: str
    source: str = ""


class CandidateFinding(BaseModel):
    check_id: str = ""
    family: Family
    error_class: str = ""
    description: str
    condition: str = ""
    steps: list[int] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    severity: Severity = "major"
    confidence: int = 70
    alternative: str = ""
    available_info: str = ""
    auditor: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, value: object) -> int:
        return clamp_confidence(value, 70)

    @model_validator(mode="after")
    def _default_class(self) -> CandidateFinding:
        mapped = normalize_class_and_family(self.error_class, self.family)
        if mapped is None:
            self.error_class = FAMILY_DEFAULT_CLASS.get(self.family, "other_error")
            return self
        cls, fam = mapped
        self.error_class = cls
        if fam in FAMILIES:
            self.family = fam  # type: ignore[assignment]
        return self


class Finding(BaseModel):
    finding_id: str
    trace_id: str
    family: Family
    error_class: str = ""
    description: str
    explanation: str = ""
    steps: list[int] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    severity: Severity = "major"
    status: Verdict = "confirmed"
    adjudication_reason: str = ""
    confidence: int = 80
    check_id: str = ""
    alternative: str = ""
    rule_ref: str = ""
    available_info: str = ""
    auditor: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, value: object) -> int:
        return clamp_confidence(value, 80)

    @model_validator(mode="after")
    def _default_class(self) -> Finding:
        mapped = normalize_class_and_family(self.error_class, self.family)
        if mapped is None:
            self.error_class = FAMILY_DEFAULT_CLASS.get(self.family, "other_error")
            return self
        cls, fam = mapped
        self.error_class = cls
        if fam in FAMILIES:
            self.family = fam  # type: ignore[assignment]
        return self


class CallRecord(BaseModel):
    stage: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cached: bool = False
    error: str = ""


class TraceResult(BaseModel):
    trace_id: str
    status: TraceStatus = "ok"
    error: str = ""
    findings: list[Finding] = Field(default_factory=list)
    rejected: list[CandidateFinding] = Field(default_factory=list)
    checks: list[DerivedCheck] = Field(default_factory=list)
    calls: list[CallRecord] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    latency_s: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)
