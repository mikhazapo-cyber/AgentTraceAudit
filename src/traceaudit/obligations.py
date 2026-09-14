"""Compile THIS task's instruction sentences into typed, testable obligations.

Input is the instruction text shipped with the trace. Output is a small set of
rules that can fire on this trace, each bound to the steps that could settle it.
Not a catalogue of known bugs.

Three shapes cover most agent policies:

- restricted_tool: a tool is allowed only under a stated condition
- prerequisite:    something must happen, or hold, before an action
- generic:         quotable, but not bindable to a shape
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .evidence import _clip, obligation_sentences
from .quotes import excerpt_in_source
from .schemas import CanonicalTrace, DerivedCheck
from .structural import logical_tool_name, pair_calls_results

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\n])\s+")
_PRONOUN = re.compile(r"\b(it|its|this|that|these|those|them|they)\b", re.IGNORECASE)

_RESTRICTED = re.compile(
    r"(only use|use it only|please only|can only be used|should only be used|"
    r"may only be used|must only be used|only be called|only if|only when|"
    r"only after|do not use|don't use|never use|never call|must not be used|"
    r"is not allowed|may not be used)",
    re.IGNORECASE,
)
_PREREQUISITE = re.compile(
    r"\b(before|prior to|beforehand|first|must have|it must|they must|"
    r"has to be done|have to|you must|must be (?:verified|authenticated|"
    r"confirmed|checked|obtained|provided)|must provide|must receive|"
    r"require[sd]?|only after|not before|after (?:authenticating|confirmation))\b",
    re.IGNORECASE,
)

_STOP = {
    "a", "an", "and", "any", "are", "as", "at", "be", "been", "but", "by", "can", "cannot",
    "do", "does", "done", "dont", "each", "else", "even", "every", "for", "from", "had",
    "has", "have", "how", "if", "in", "into", "is", "it", "its", "may", "must", "never",
    "not", "of", "on", "only", "or", "other", "our", "out", "over", "own", "per", "please",
    "same", "shall", "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "to", "up", "use", "used", "user",
    "using", "via", "was", "were", "what", "when", "which", "while", "who", "will", "with",
    "would", "you", "your",
}
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*")
_TRACE_BLOB_LIMIT = 400_000


@dataclass
class Obligation:
    obligation_id: str
    kind: str
    rule: str
    bound_tool: str = ""
    terms: list[str] = field(default_factory=list)
    relevant_steps: list[int] = field(default_factory=list)
    fired: bool = False
    fired_because: str = ""
    groundedness: int = 0

    def to_check(self) -> DerivedCheck:
        if self.kind == "restricted_tool":
            desc = (
                f"'{self.bound_tool}' is restricted by an explicit instruction; "
                f"decide whether the stated condition held when it was called."
            )
            condition = (
                f"a call to '{self.bound_tool}' does not satisfy the stated condition in: {self.rule}"
            )
            justified = (
                "the stated condition genuinely held for this task, or the tool was never called"
            )
        elif self.kind == "prerequisite":
            desc = "An explicit instruction requires something before or alongside an action."
            condition = f"the required step or state is missing when the action happened: {self.rule}"
            justified = (
                "the triggering situation never arose, or the required step is present earlier "
                "in the trace"
            )
        else:
            desc = "Verify the agent honored this instruction obligation."
            condition = f"an action or omission violates: {self.rule}"
            justified = (
                "the cited sentence does not apply to what the agent was doing, or the required "
                "information was not available"
            )
        return DerivedCheck(
            check_id=self.obligation_id,
            family="instruction_violation",
            description=desc[:400],
            condition=condition[:500],
            justified_when=justified[:400],
            scope=f"calls to {self.bound_tool}" if self.bound_tool else "whole trace",
            severity="major",
            rationale=f"compiled obligation ({self.kind})",
            needs_llm=True,
            source_refs=[self.rule[:300]],
            lane="obligation",
        )


def _tool_vocabulary(trace: CanonicalTrace) -> list[str]:
    names = {t.name for t in trace.tools if t.name and len(t.name) >= 3}
    for step in trace.steps:
        if step.kind == "tool_call":
            name = logical_tool_name(step) or step.name
            if name and len(name) >= 3:
                names.add(name)
    return sorted(names, key=len, reverse=True)


_IDENT_CHARS = r"A-Za-z0-9_.\-"


def _names_at_boundaries(text: str, vocabulary: list[str]) -> list[str]:
    """Match tool names on identifier boundaries.

    A substring test would bind 'read' inside 'threads' and hand the auditor a
    rule about a tool the sentence never mentions.
    """
    low = text.lower()
    hits = []
    for name in vocabulary:
        pattern = rf"(?<![{_IDENT_CHARS}]){re.escape(name.lower())}(?![{_IDENT_CHARS}])"
        if re.search(pattern, low):
            hits.append(name)
    return hits


def _tools_in(sentence: str, vocabulary: list[str]) -> list[str]:
    return _names_at_boundaries(sentence, vocabulary)


def _terms(sentence: str) -> list[str]:
    words = [w.lower() for w in _WORD.findall(sentence)]
    return [w for w in words if len(w) >= 4 and w not in _STOP]


def _bigrams(sentence: str) -> list[str]:
    words = [w.lower() for w in _WORD.findall(sentence)]
    out = []
    for a, b in zip(words, words[1:]):
        if a in _STOP or b in _STOP or len(a) < 3 or len(b) < 3:
            continue
        out.append(f"{a} {b}")
    return out


def _trace_blob(trace: CanonicalTrace) -> str:
    parts = []
    size = 0
    for step in trace.steps:
        text = step.text() or ""
        parts.append(text)
        size += len(text)
        if size > _TRACE_BLOB_LIMIT:
            break
    return "\n".join(parts).lower()


def _calls_to(trace: CanonicalTrace, tool: str) -> list[int]:
    out = []
    for step in trace.steps:
        if step.kind != "tool_call":
            continue
        name = logical_tool_name(step) or step.name
        if name == tool or step.name == tool:
            out.append(step.index)
    return out


def _normalized(rule: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", rule.lower()).strip()


def _identifier_like(sentence: str) -> bool:
    """A rule naming a concrete identifier is testable; prose about tone is not."""
    if "`" in sentence:
        return True
    for word in _WORD.findall(sentence):
        if len(word) < 4:
            continue
        if "_" in word or "." in word or "-" in word:
            return True
        if word[0].islower() and any(ch.isupper() for ch in word[1:]):
            return True
    return False


def _groundedness(
    ob: Obligation, declared: set[str] | list[str], called: set[str], matched_grams: int
) -> int:
    """How strongly a rule ties to something observable in the trace.

    Domain-agnostic on purpose: prefers rules testable against tools, arguments,
    and sequencing over rules about tone or style.
    """
    score = 0
    if ob.bound_tool and ob.bound_tool in called:
        score += 4
    elif ob.bound_tool and ob.bound_tool in declared:
        score += 2
    if ob.kind == "restricted_tool":
        score += 2
    elif ob.kind == "prerequisite":
        score += 1
    if _names_at_boundaries(ob.rule, [n for n in declared if n]):
        score += 2
    score += min(2, matched_grams)
    if _identifier_like(ob.rule):
        score += 1
    return score


def _steps_mentioning(trace: CanonicalTrace, needles: list[str], limit: int = 6) -> list[int]:
    if not needles:
        return []
    out: list[int] = []
    for step in trace.steps:
        hay = (step.text() or "").lower()
        if any(n in hay for n in needles):
            out.append(step.index)
        if len(out) >= limit:
            break
    return out


def compile_obligations(
    trace: CanonicalTrace,
    limit: int = 16,
    min_groundedness: int = 3,
) -> list[Obligation]:
    """Typed obligations that can fire on THIS trace, in instruction order.

    Long policies carry many sentences a trace cannot settle: tone, style,
    posture. Sending those to an auditor invites false positives on clean
    traces, so obligations are ranked by groundedness and the weak tail is
    dropped.
    """
    text = trace.instructions or ""
    if not text.strip():
        return []
    vocabulary = _tool_vocabulary(trace)
    # Keyed on the whole sentence. A 90-character prefix collides on policies
    # built from repeated boilerplate openings, which both drops real rules and
    # lets near-duplicates through.
    wanted = {_normalized(s) for s in obligation_sentences(text, limit=64)}
    blob = _trace_blob(trace) if trace.steps else ""
    has_steps = bool(trace.steps)
    declared = trace.declared_tool_names() or trace.tool_names()
    called = {
        (logical_tool_name(s) or s.name) for s in trace.steps if s.kind == "tool_call"
    }

    out: list[Obligation] = []
    seen: set[str] = set()
    last_tools: list[str] = []
    for raw in _SENTENCE_SPLIT.split(text):
        sentence = raw.strip().replace("\n", " ")
        if not sentence:
            continue
        here = _tools_in(sentence, vocabulary)
        norm = _normalized(sentence)
        if norm not in wanted:
            if here:
                last_tools = here
            continue

        if norm in seen:
            if here:
                last_tools = here
            continue
        seen.add(norm)

        bound = ""
        if len(here) == 1:
            bound = here[0]
        elif not here and len(last_tools) == 1 and _PRONOUN.search(sentence):
            bound = last_tools[0]
        if here:
            last_tools = here

        if _RESTRICTED.search(sentence) and bound:
            kind = "restricted_tool"
        elif _PREREQUISITE.search(sentence):
            kind = "prerequisite"
        else:
            kind = "generic"

        ob = Obligation(
            obligation_id="",
            kind=kind,
            rule=sentence[:400],
            bound_tool=bound,
            terms=_terms(sentence)[:12],
        )

        steps = _calls_to(trace, bound) if bound else []
        why = ""
        grams: list[str] = []
        if steps:
            why = f"'{bound}' called at {steps[:6]}"
        if has_steps and not steps:
            grams = [g for g in _bigrams(sentence) if g in blob][:4]
            if grams:
                steps = _steps_mentioning(trace, grams)
                why = f"instruction terms present in trace: {grams}"
            else:
                present = [t for t in ob.terms if t in blob]
                if len(present) >= 2:
                    steps = _steps_mentioning(trace, present[:4])
                    why = f"instruction terms present in trace: {present[:4]}"
        ob.relevant_steps = steps
        ob.fired = bool(steps) or not has_steps
        ob.fired_because = why or ("no trace steps to filter against" if not has_steps else "")
        ob.groundedness = _groundedness(ob, declared, called, len(grams))
        if ob.fired and (not has_steps or ob.groundedness >= min_groundedness):
            out.append(ob)

    order = {id(ob): i for i, ob in enumerate(out)}
    kept = sorted(out, key=lambda o: (-o.groundedness, order[id(o)]))[:limit]
    kept.sort(key=lambda o: order[id(o)])
    for i, ob in enumerate(kept, 1):
        ob.obligation_id = f"OBL-{i}"
    return kept


def extend_from_derived(
    trace: CanonicalTrace,
    obligations: list[Obligation],
    checks: list[DerivedCheck],
) -> list[Obligation]:
    """Fold LLM-extracted instruction excerpts into the obligation brief.

    The regex compiler is English-shaped. Derived checks already quote THIS
    trace's instructions, so they backstop the compiler without another call
    and without a phrase list from the labelled set.
    """
    seen = {_normalized(o.rule) for o in obligations}
    extra: list[Obligation] = []
    n = len(obligations)
    for check in checks:
        if check.family != "instruction_violation":
            continue
        for ref in check.source_refs:
            text = str(ref or "").strip()
            if len(text) < 16:
                continue
            if not excerpt_in_source(text, trace.task, trace.instructions):
                continue
            norm = _normalized(text)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            n += 1
            extra.append(
                Obligation(
                    obligation_id=f"OBL-{n}",
                    kind="generic",
                    rule=text[:400],
                    groundedness=3,
                    fired=True,
                    fired_because="quoted in derived audit instructions",
                )
            )
    return obligations + extra


def call_outline(trace: CanonicalTrace, limit: int = 12000) -> str:
    """Every tool call with its result status. Proves a required call is absent."""
    pairing = pair_calls_results(trace)
    rows = []
    for step in trace.steps:
        if step.kind != "tool_call":
            continue
        name = logical_tool_name(step) or step.name or "?"
        ri = pairing.get(step.index)
        tag = "no result captured"
        if ri is not None and 0 <= ri < len(trace.steps):
            tag = "ERROR" if trace.steps[ri].is_error else "ok"
        rows.append(f"#{step.index} {name} -> {tag}")
    if not rows:
        return "(no tool calls in this trace)"
    return _clip("\n".join(rows), limit)


def render_obligation_brief(
    trace: CanonicalTrace,
    obligations: list[Obligation],
    limit: int = 40000,
) -> str:
    if not obligations:
        return "(no instruction obligations compiled for this trace)"
    blocks = []
    for ob in obligations:
        lines = [
            f"[{ob.obligation_id}] kind={ob.kind}" + (f" tool={ob.bound_tool}" if ob.bound_tool else ""),
            f'RULE (verbatim): "{ob.rule}"',
        ]
        if ob.fired_because:
            lines.append(f"WHY IT IS IN PLAY: {ob.fired_because}")
        if ob.relevant_steps:
            lines.append("STEPS THAT COULD PROVE OR DISPROVE IT:")
            for idx in ob.relevant_steps[:8]:
                if 0 <= idx < len(trace.steps):
                    lines.append(f"  #{idx} {_clip(trace.steps[idx].text(), 2000)}")
        blocks.append("\n".join(lines))
    return _clip("\n\n".join(blocks), limit)


def obligation_context(trace: CanonicalTrace, obligations: list[Obligation]) -> str:
    return (
        "OBLIGATIONS COMPILED FROM THESE INSTRUCTIONS (test each one):\n"
        f"{render_obligation_brief(trace, obligations)}\n\n"
        "COMPLETE TOOL-CALL OUTLINE (every call in the trace, in order).\n"
        "Use this to decide whether a required call is genuinely absent:\n"
        f"{call_outline(trace)}\n"
    )
