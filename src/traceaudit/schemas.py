from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

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
TraceStatus = Literal["ok", "deterministic_only", "budget_exhausted", "model_error", "error"]
FindingSource = Literal["deterministic", "llm", "seed"]

FAMILY_GLOSSARY: dict[str, str] = {
    "instruction_violation": "broke an explicit rule in the task or instructions",
    "incorrect_tool_use": "wrong or hallucinated tool, arguments that violate the schema, or a malformed call",
    "evidence_contradiction": "a statement that contradicts what a tool actually returned",
    "unsupported_success": "claimed a completed effect the trace does not evidence",
    "redundant_action": "repeated work that yielded no new information",
    "ignored_feedback": "kept the same failing approach after errors or feedback",
    "other": "a real problem outside the families above",
}


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
        parts = [self.content]
        if self.arguments is not None:
            import json

            try:
                parts.append(json.dumps(self.arguments, ensure_ascii=False, sort_keys=True))
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

    def declared_tool_names(self) -> set[str]:
        return {t.name for t in self.tools if t.declared}

    def schema_for(self, name: str) -> dict[str, Any] | None:
        for tool in self.tools:
            if tool.name == name and tool.declared and tool.parameters:
                return tool.parameters
        return None


class DerivedCheck(BaseModel):
    check_id: str
    family: Family
    description: str
    condition: str
    justified_when: str = ""
    scope: str = ""
    severity: Severity = "major"
    rationale: str = ""
    needs_llm: bool = True
    source_refs: list[str] = Field(default_factory=list)
    # Which lane owns this check: obligation, residual, or mechanical.
    lane: str = "residual"


class Evidence(BaseModel):
    step: int
    quote: str
    source: str = ""


class Verification(BaseModel):
    locatable: bool = False
    quotes_ok: bool = False
    family_gate: bool = False
    mechanical: bool = False
    adjudicator: str = ""
    reason: str = ""
    quote_verified: int = 0
    quote_total: int = 0

    @property
    def passed(self) -> bool:
        return self.locatable and self.quotes_ok and self.family_gate


class CandidateFinding(BaseModel):
    check_id: str = ""
    family: Family
    description: str
    condition: str = ""
    steps: list[int] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    severity: Severity = "major"
    confidence: float = 0.5
    alternative: str = ""
    source: FindingSource = "llm"
    available_info: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class Finding(BaseModel):
    finding_id: str
    trace_id: str
    family: Family
    description: str
    explanation: str = ""
    steps: list[int] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    severity: Severity = "major"
    status: Verdict = "confirmed"
    adjudication_reason: str = ""
    confidence: float = 0.5
    check_id: str = ""
    source: FindingSource = "llm"
    alternative: str = ""
    rule_ref: str = ""
    available_info: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)
    verification: Verification = Field(default_factory=Verification)


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
    pending_checks: list[str] = Field(default_factory=list)
    calls: list[CallRecord] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    coverage: dict[str, Any] = Field(default_factory=dict)
    input_hash: str = ""
    cfg_hash: str = ""
    latency_s: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def tokens(self) -> tuple[int, int]:
        return (
            sum(c.input_tokens for c in self.calls),
            sum(c.output_tokens for c in self.calls),
        )
