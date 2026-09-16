#!/usr/bin/env python3
"""Fetch and convert checked public traces into data/checked + data/labels/checked.

Does not spend OpenRouter credits. Does not invent families: source types that
cannot be mapped honestly are skipped (or mapped to `other` when the source
marks a real error without a closer family). TRAIL is gated and not reshared;
this script converts it only if you already fetched it into data/raw/trail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from traceaudit.normalize import ingest_doc, load_trace  # noqa: E402

RAW = REPO / "data" / "raw"
CHECKED = REPO / "data" / "checked"
LABELS = REPO / "data" / "labels" / "checked"
PRECISION = REPO / "data" / "precision"
TRACES = CHECKED / "traces"
UNMAPPED = CHECKED / "unmapped"

MAX_TRACE_BYTES = 120_000
MAX_STEP_CHARS = 6000

# --- family maps (honest, recorded) -----------------------------------------

# (parent family, error_class). Instruction / Plan Adherence remap honestly.
# Intent-Plan Misalignment and Underspecified User Intent stay `other`.
AGENTRX_FAMILY: dict[str, tuple[str, str]] = {
    "instruction adherence failure": ("instruction_violation", "broke_explicit_rule"),
    "instruction/plan adherence failure": (
        "instruction_violation",
        "broke_explicit_rule",
    ),
    "plan adherence failure": ("instruction_violation", "role_or_spec_deviation"),
    "invalid invocation": ("incorrect_tool_use", "malformed_call"),
    "misinterpretation of tool output": (
        "evidence_contradiction",
        "contradicts_tool_output",
    ),
    "invention of new information": ("evidence_contradiction", "invented_observation"),
    "invention of information": ("evidence_contradiction", "invented_observation"),
    "intent plan misalignment": ("other", "other_error"),
    "intent-plan misalignment": ("other", "other_error"),
    "intent–plan misalignment": ("other", "other_error"),
    "underspecified user intent": ("other", "other_error"),
    "under-specified user intent": ("other", "other_error"),
    "intent not supported": ("other", "other_error"),
    "guardrails triggered": ("other", "other_error"),
    "system failure": ("other", "other_error"),
}

AEB_FAMILY = {
    "constraint_ignorance": "instruction_violation",
    "invalid_action": "incorrect_tool_use",
    "format_error": "incorrect_tool_use",
    "parameter_error": "incorrect_tool_use",
    "hallucination": "evidence_contradiction",
    "outcome_misinterpretation": "evidence_contradiction",
    "progress_misjudge": "unsupported_success",
    "inefficient_plan": "other",
    "plan_inefficient": "other",
    "impossible_action": "other",
    "misalignment": "other",
    "over_simplification": "other",
    "causal_misattribution": "other",
    "memory_retrieval_failure": "other",
    "step_limit": "other",
    "environment_error": "other",
    "tool_execution_error": "other",
    "llm_limit": "other",
}

AEGIS_FAMILY = {
    "FM-1.1": "instruction_violation",
    "FM-1.2": "instruction_violation",
    "FM-2.3": "instruction_violation",
    "FM-1.3": "redundant_action",
    "FM-2.1": "redundant_action",
    "FM-2.5": "ignored_feedback",
    "FM-3.1": "unsupported_success",
    "FM-3.2": "unsupported_success",
    "FM-3.3": "unsupported_success",
}

TRAIL_FAMILY = {
    "instruction non-compliance": "instruction_violation",
    "goal deviation": "instruction_violation",
    "tool selection errors": "incorrect_tool_use",
    "tool-related hallucinations": "incorrect_tool_use",
    "formatting errors": "incorrect_tool_use",
    "tool output misinterpretation": "evidence_contradiction",
    "resource abuse": "redundant_action",
}

TRAIL_OTHER = {
    "language-only hallucinations",
    "language only hallucinations",
    "text-only hallucinations",
    "incorrect problem identification",
    "poor information retrieval",
    "context handling failures",
    "task orchestration",
    "tool definition issues",
    "environment setup errors",
    "rate limiting",
    "authentication errors",
    "service errors",
    "resource not found",
    "resource exhaustion",
    "timeout issues",
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _as_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def map_agentrx_family(category: str) -> tuple[str, str] | None:
    key = _norm(category)
    if key in {"inconclusive", ""}:
        return None
    if key in AGENTRX_FAMILY:
        return AGENTRX_FAMILY[key]
    # Honest substring fallback: do not treat intent-plan as instruction.
    if "intent plan" in key or key.startswith("underspecified"):
        return ("other", "other_error")
    if "instruction adherence" in key:
        return ("instruction_violation", "broke_explicit_rule")
    if "plan adherence" in key and "intent" not in key:
        return ("instruction_violation", "role_or_spec_deviation")
    # source labelled a real error we do not have a closer family for
    return ("other", "other_error")


def map_aeb_family(ftype: str) -> str | None:
    key = str(ftype or "").strip().lower().replace(" ", "")
    if not key:
        return None
    return AEB_FAMILY.get(key, "other")


def map_aegis_family(code: str) -> str:
    return AEGIS_FAMILY.get(str(code or "").strip().upper(), "other")


def map_trail_family(leaf: str) -> str | None:
    key = _norm(leaf)
    if key in TRAIL_FAMILY:
        return TRAIL_FAMILY[key]
    if key in TRAIL_OTHER:
        return "other"
    return None


def history_to_messages(history: list) -> list[dict]:
    messages = []
    for turn in history or []:
        if isinstance(turn, str):
            messages.append({"role": "assistant", "content": turn})
            continue
        if not isinstance(turn, dict):
            continue
        raw = str(turn.get("role") or turn.get("from") or "assistant")
        low = raw.lower()
        if low in {"human", "user", "customer"}:
            role = "user"
        elif low in {"system", "developer"}:
            role = "system"
        elif low in {"tool", "function", "observation"}:
            role = "tool"
        else:
            role = "assistant"
        content = turn.get("content")
        if content is None:
            content = turn.get("value") or turn.get("text") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        msg: dict = {"role": role, "content": content}
        name = turn.get("name") or turn.get("agent") or ""
        if name:
            msg["name"] = str(name)
        elif role == "assistant" and raw not in {"assistant", "gpt", "ai"}:
            msg["name"] = raw
        messages.append(msg)
    return messages


def ingest_mapped(trace_id: str, messages: list[dict], extra: dict | None = None):
    extra = dict(extra or {})
    mapping: dict[int, list[int]] = {}
    acc: list[dict] = []
    prev = 0
    for i, msg in enumerate(messages):
        acc.append(msg)
        trace = ingest_doc(trace_id, {**extra, "messages": acc}, MAX_STEP_CHARS)
        mapping[i] = list(range(prev, len(trace.steps)))
        prev = len(trace.steps)
    final = ingest_doc(trace_id, {**extra, "messages": acc}, MAX_STEP_CHARS)
    return final, mapping


def _history_blob(turn: object) -> str:
    if isinstance(turn, str):
        return turn.lower()
    if isinstance(turn, dict):
        name = str(turn.get("name") or turn.get("role") or turn.get("agent") or "")
        body = turn.get("content") or turn.get("value") or turn.get("text") or ""
        if not isinstance(body, str):
            body = json.dumps(body, ensure_ascii=False, default=str)
        return f"{name}\n{body}".lower()
    return ""


def resolve_whowhen_index(src_step: int, history: list, agent: str) -> int | None:
    """Who&When `mistake_step` is a 0-based history index (not 1-based).

    Prefer the 0-based slot when it names the labelled agent; fall back to
    1-based only if that slot matches the agent and 0-based does not.
    """
    n = len(history)
    zero = src_step if 0 <= src_step < n else None
    one = src_step - 1 if src_step >= 1 and src_step <= n else None
    key = agent.lower().split()[0] if agent else ""

    def matches(i: int) -> bool:
        if not key:
            return False
        return key in _history_blob(history[i])

    if zero is not None and (not key or matches(zero)):
        return zero
    if one is not None and matches(one):
        return one
    if zero is not None:
        return zero
    return one


def tau_messages(traj: list) -> tuple[list[dict], list[int | None]]:
    messages, source_ids = [], []
    for item in traj or []:
        if not isinstance(item, dict):
            continue
        keep: dict = {}
        for key in ("role", "content", "tool_calls", "tool_call_id", "name"):
            if item.get(key) is not None:
                keep[key] = item[key]
        if not keep.get("role"):
            continue
        if "content" not in keep and not keep.get("tool_calls"):
            keep["content"] = ""
        messages.append(keep)
        src = item.get("index")
        source_ids.append(int(src) if isinstance(src, (int, float)) else None)
    return messages, source_ids


class Writer:
    def __init__(self) -> None:
        self.entries: list[dict] = []
        self.precision: list[dict] = []
        self.stats = Counter()
        self.skipped = Counter()

    def add(
        self,
        *,
        trace_id: str,
        payload: dict,
        findings: list[dict],
        clean: bool,
        notes: str,
        annotator: str,
        review: str,
        corpus: str,
        license_name: str,
        original_id: str,
        original_label: str,
        label_kind: str,
        source_format: str,
        source_repository: str,
        extra_label: dict | None = None,
    ) -> bool:
        TRACES.mkdir(parents=True, exist_ok=True)
        LABELS.mkdir(parents=True, exist_ok=True)
        rel = f"traces/{trace_id}.json"
        blob = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
        if len(blob) > MAX_TRACE_BYTES:
            self.skipped["too_large"] += 1
            return False
        path = CHECKED / rel
        path.write_bytes(blob)
        # re-ingest from disk so gold steps are those load_trace will see
        recovered = load_trace(trace_id, source_format, [str(path)], MAX_STEP_CHARS)
        if not recovered.steps:
            path.unlink(missing_ok=True)
            self.skipped["no_steps"] += 1
            return False
        n = len(recovered.steps)
        kept = []
        for finding in findings:
            steps = sorted(
                {int(s) for s in finding.get("steps") or [] if isinstance(s, int)}
            )
            steps = [s for s in steps if 0 <= s < n]
            if not steps:
                self.skipped["finding_step_missing"] += 1
                continue
            finding = dict(finding)
            finding["steps"] = steps
            kept.append(finding)
        if not clean and not kept:
            path.unlink(missing_ok=True)
            self.skipped["no_locatable_finding"] += 1
            return False
        label = {
            "trace_id": trace_id,
            "clean": bool(clean and not kept),
            "findings": kept,
            "notes": notes,
            "annotator": annotator,
            "review": review,
            "split": "checked",
            "meta": {
                "corpus": corpus,
                "license": license_name,
                "original_id": original_id,
                "original_label": original_label,
                "label_kind": label_kind,
                "source_repository": source_repository,
            },
        }
        if extra_label:
            label.update(extra_label)
        (LABELS / f"{trace_id}.json").write_text(
            json.dumps(label, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self.entries.append(
            {
                "trace_id": trace_id,
                "source": source_format,
                "source_id": original_id,
                "format": "native_json",
                "files": [{"path": rel, "bytes": len(blob), "sha256": _sha(blob)}],
                "source_repository": source_repository,
                "corpus": corpus,
                "label_kind": label_kind,
                "license": license_name,
            }
        )
        self.precision.append(
            {
                "trace_id": trace_id,
                "corpus": corpus,
                "label_kind": label_kind,
                "clean": bool(clean and not kept),
                "n_findings": len(kept),
                "families": sorted({f["family"] for f in kept}),
                "dataset": "data/checked",
                "labels": "data/labels/checked",
            }
        )
        self.stats["traces"] += 1
        self.stats[f"corpus:{corpus}"] += 1
        self.stats["clean" if clean and not kept else "dirty"] += 1
        self.stats[f"kind:{label_kind}"] += 1
        self.stats["findings"] += len(kept)
        for finding in kept:
            self.stats[f"family:{finding['family']}"] += 1
        return True


def convert_construction(writer: Writer) -> None:
    examples = REPO / "examples"
    specs = [
        (
            "const-clean-retry",
            "clean-retry.json",
            True,
            [],
            "Construction-verified justified retry. No gold finding.",
        ),
        (
            "const-clean-search",
            "clean-search.json",
            True,
            [],
            "Construction-verified sequential lookups with different arguments. Clean.",
        ),
        (
            "const-clean-verify",
            "clean-verify.json",
            True,
            [],
            "Construction-verified required extra verification after a transfer. Clean.",
        ),
        (
            "const-clean-skill",
            "clean-skill.json",
            True,
            [],
            "Construction-verified playbook preferred-order nit; allowed path completed. Clean.",
        ),
        (
            "const-contradict",
            "contradict.json",
            False,
            [
                {
                    "family": "evidence_contradiction",
                    "error_class": "contradicts_tool_output",
                    "steps": [3],
                    "description": "Assistant states 28°C and sunny after get_weather returned temp_c 12, cloudy.",
                    "severity": "major",
                    "notes": "construction: examples/contradict.json",
                }
            ],
            "Construction-verified contradiction against the tool result.",
        ),
        (
            "const-auth-skip",
            "auth-skip.json",
            False,
            [
                {
                    "family": "instruction_violation",
                    "error_class": "skipped_required_check",
                    "steps": [1],
                    "description": "refund_order called before find_user_id, violating the authenticate-first rule.",
                    "severity": "major",
                    "notes": "construction: examples/auth-skip.json",
                }
            ],
            "Construction-verified instruction skip.",
        ),
    ]
    for tid, name, clean, findings, notes in specs:
        src = examples / name
        if not src.is_file():
            writer.skipped["missing_example"] += 1
            continue
        payload = json.loads(src.read_text(encoding="utf-8"))
        payload.setdefault("meta", {})
        payload["meta"].update(
            {
                "corpus": "construction",
                "license": "MIT (this repo)",
                "original_id": name,
                "label_kind": "construction",
            }
        )
        writer.add(
            trace_id=tid,
            payload=payload,
            findings=findings,
            clean=clean,
            notes=notes,
            annotator="repo construction examples",
            review="construction",
            corpus="construction",
            license_name="MIT",
            original_id=name,
            original_label="construction",
            label_kind="construction",
            source_format="canonical",
            source_repository="examples/",
        )


def convert_agentrx_tau(writer: Writer) -> None:
    traj_path = RAW / "agentrx" / "tau_dataset_failed.json"
    gold_path = RAW / "agentrx" / "tau_ground_truth.json"
    if not traj_path.is_file() or not gold_path.is_file():
        print("skip AgentRx tau: raw files missing (run --fetch)")
        return
    traces = {
        int(t["task_id"]): t for t in json.loads(traj_path.read_text(encoding="utf-8"))
    }
    golds = json.loads(gold_path.read_text(encoding="utf-8"))
    for gold in golds:
        tid_src = int(gold["trajectory_id"])
        rec = traces.get(tid_src)
        if not rec:
            writer.skipped["agentrx_missing_traj"] += 1
            continue
        messages, source_ids = tau_messages(rec.get("traj") or [])
        if not messages:
            writer.skipped["agentrx_no_messages"] += 1
            continue
        info = rec.get("info") or {}
        task = ""
        if isinstance(info.get("task"), dict):
            task = str(info["task"].get("instruction") or "")
        instructions = next(
            (m.get("content") or "" for m in messages if m.get("role") == "system"), ""
        )
        extra = {"task": task, "instructions": instructions}
        recovered, mapping = ingest_mapped(f"arx-tau-{tid_src:03d}", messages, extra)
        src_to_msg = {sid: i for i, sid in enumerate(source_ids) if sid is not None}
        findings = []
        labels = []
        for fail in gold.get("failures") or []:
            if not isinstance(fail, dict):
                continue
            mapped = map_agentrx_family(str(fail.get("failure_category") or ""))
            if mapped is None:
                continue
            family, error_class = mapped
            step_no = fail.get("step_number")
            try:
                step_no = int(step_no)
            except (TypeError, ValueError):
                continue
            mi = src_to_msg.get(step_no)
            steps = mapping.get(mi, []) if mi is not None else []
            if not steps:
                continue
            desc = str(fail.get("step_reason") or fail.get("category_reason") or "")
            findings.append(
                {
                    "family": family,
                    "error_class": error_class,
                    "steps": steps,
                    "description": desc[:900],
                    "severity": "major",
                    "notes": f"AgentRx category: {fail.get('failure_category')}",
                }
            )
            labels.append(str(fail.get("failure_category") or ""))
        payload = {
            "task": task,
            "instructions": instructions,
            "messages": messages,
            "meta": {
                "corpus": "agentrx",
                "license": "CC-BY-4.0",
                "original_id": str(tid_src),
                "label_kind": "human",
                "domain": "tau_retail",
            },
        }
        writer.add(
            trace_id=f"arx-tau-{tid_src:03d}",
            payload=payload,
            findings=findings,
            clean=False,
            notes=str(gold.get("failure_summary") or ""),
            annotator="AgentRx human annotators (arXiv:2602.02475)",
            review="human",
            corpus="agentrx",
            license_name="CC-BY-4.0",
            original_id=str(tid_src),
            original_label="; ".join(labels),
            label_kind="human",
            source_format="openai-chat",
            source_repository="microsoft/AgentRx",
            extra_label={"source_benchmark": "AgentRx tau_retail"},
        )
        _ = recovered


def convert_whowhen(writer: Writer, limit: int = 12) -> None:
    root = RAW / "whowhen" / "Who&When"
    if not root.is_dir():
        print("skip Who&When: raw dir missing (run --fetch)")
        return
    rows: list[tuple[int, Path, str]] = []
    for kind, folder in (
        ("hc", root / "Hand-Crafted"),
        ("ag", root / "Algorithm-Generated"),
    ):
        if not folder.is_dir():
            continue
        for path in folder.glob("*.json"):
            rows.append((path.stat().st_size, path, kind))
    rows.sort()
    taken = {"hc": 0, "ag": 0}
    per = max(1, limit // 2)
    for size, path, kind in rows:
        if taken[kind] >= per:
            continue
        if size > MAX_TRACE_BYTES:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        history = doc.get("history") or []
        if not isinstance(history, list) or not history:
            continue
        raw_step = doc.get("mistake_step")
        try:
            src_step = int(raw_step)
        except (TypeError, ValueError):
            writer.skipped["ww_no_step"] += 1
            continue
        agent = str(doc.get("mistake_agent") or "")
        hist_i = resolve_whowhen_index(src_step, history, agent)
        if hist_i is None:
            writer.skipped["ww_step_oob"] += 1
            continue
        messages = history_to_messages(history)
        question = str(doc.get("question") or "")
        instructions = str(doc.get("system_prompt") or "")
        extra = {"task": question, "instructions": instructions}
        tid = f"ww-{kind}-{path.stem.zfill(3)}"
        recovered, mapping = ingest_mapped(tid, messages, extra)
        steps = mapping.get(hist_i, [])
        if not steps:
            writer.skipped["ww_step_unmapped"] += 1
            continue
        reason = str(doc.get("mistake_reason") or "")
        findings = [
            {
                "family": "other",
                "steps": steps,
                "description": f"{agent}: {reason}".strip(": "),
                "severity": "major",
                "notes": "Who&When has who/when/reason only; no source taxonomy, so family=other.",
            }
        ]
        payload = {
            "task": question,
            "instructions": instructions,
            "messages": messages,
            "meta": {
                "corpus": "whowhen",
                "license": "MIT",
                "original_id": str(doc.get("question_ID") or path.stem),
                "label_kind": "human",
                "split_kind": kind,
            },
        }
        if writer.add(
            trace_id=tid,
            payload=payload,
            findings=findings,
            clean=False,
            notes=f"mistake_agent={agent}; mistake_step={src_step}",
            annotator="Who&When human annotators (arXiv:2505.00212)",
            review="human",
            corpus="whowhen",
            license_name="MIT",
            original_id=str(doc.get("question_ID") or path.stem),
            original_label=reason[:160],
            label_kind="human",
            source_format="openai-chat",
            source_repository="Kevin355/Who_and_When",
        ):
            taken[kind] += 1
        _ = recovered
        if sum(taken.values()) >= limit:
            break


def convert_agentrx_magentic(writer: Writer) -> None:
    """Magentic-One labelled rows need the 44 benchmark traces, not GitHub samples.

    GitHub `trajectories/magentic-one` is a handful of category demos; two already
    exceed MAX_TRACE_BYTES and several are screenshot/OCR dumps. Skip rather than
    invent gold or vendor huge files.
    """
    gold_path = RAW / "agentrx" / "magentic_one_ground_truth.json"
    sample_dir = RAW / "agentrx" / "magentic-one"
    if gold_path.is_file():
        writer.skipped["agentrx_magentic_no_labelled_traj"] += 1
    if sample_dir.is_dir():
        writer.skipped["agentrx_magentic_samples_skipped"] += 1
    print(
        "skip AgentRx Magentic-One: labelled 44 traces are not in-tree as compact "
        "files; GitHub samples are not that gold set and several are too large."
    )


def snapshot_existing_corpus(prefix: str, review: str | None = None) -> list[dict]:
    """Keep already-vendored traces when rebuilding a corpus."""
    rows = []
    if not LABELS.is_dir() or not TRACES.is_dir():
        return rows
    for path in sorted(LABELS.glob(f"{prefix}*.json")):
        label = json.loads(path.read_text(encoding="utf-8"))
        if review is not None and (label.get("review") or "").lower() != review:
            continue
        tpath = TRACES / f"{label['trace_id']}.json"
        if not tpath.is_file():
            continue
        rows.append(
            {
                "payload": json.loads(tpath.read_text(encoding="utf-8")),
                "label": label,
            }
        )
    return rows


def snapshot_existing_aegis() -> list[dict]:
    return snapshot_existing_corpus("aegis-", review="injected")


def convert_preserved_aegis(writer: Writer, rows: list[dict]) -> set[str]:
    seen_orig: set[str] = set()
    for row in rows:
        label = row["label"]
        payload = row["payload"]
        meta = label.get("meta") or {}
        orig = str(meta.get("original_id") or "")
        if orig:
            seen_orig.add(orig)
        writer.add(
            trace_id=label["trace_id"],
            payload=payload,
            findings=list(label.get("findings") or []),
            clean=False,
            notes=label.get("notes")
            or "Injected AEGIS error (construction-verified). Not naturalistic gold.",
            annotator=label.get("annotator")
            or "AEGIS injection plan (arXiv:2509.14295)",
            review="injected",
            corpus="aegis",
            license_name="MIT",
            original_id=orig,
            original_label=str(meta.get("original_label") or ""),
            label_kind="injected",
            source_format="openai-chat",
            source_repository="Fancylalala/AEGIS",
            extra_label={
                "source_benchmark": "AEGIS",
                "gold_completeness": "injected_agents",
            },
        )
    return seen_orig


def aeb_step_message_indices(messages: list[dict], step_no: int) -> list[int]:
    """Map an AgentErrorBench decision-step number onto message indices.

    These logs are usually repeating user/assistant cycles (one ReAct turn
    per pair). Step N is the Nth pair, not raw message N. Fall back to a
    1-based message index when the log is not alternating.
    """
    n = len(messages)
    if n < 1 or step_no < 1:
        return []
    roles = [str(m.get("role") or "") for m in messages]
    pairs = n >= 2 and all(
        roles[i] == ("user" if i % 2 == 0 else "assistant") for i in range(n)
    )
    if pairs:
        user_i = 2 * (step_no - 1)
        asst_i = user_i + 1
        return [i for i in (user_i, asst_i) if 0 <= i < n]
    mi = step_no - 1
    return [mi] if 0 <= mi < n else []


def convert_preserved_generic(
    writer: Writer, rows: list[dict], *, corpus: str
) -> set[str]:
    seen: set[str] = set()
    for row in rows:
        label = row["label"]
        payload = row["payload"]
        meta = label.get("meta") or {}
        orig = str(meta.get("original_id") or "")
        tid = label["trace_id"]
        seen.add(tid)
        writer.add(
            trace_id=tid,
            payload=payload,
            findings=list(label.get("findings") or []),
            clean=bool(label.get("clean")),
            notes=label.get("notes") or "",
            annotator=label.get("annotator") or "",
            review=label.get("review") or "human",
            corpus=corpus,
            license_name=str(meta.get("license") or ""),
            original_id=orig,
            original_label=str(meta.get("original_label") or ""),
            label_kind=str(meta.get("label_kind") or "human"),
            source_format=str(payload.get("source_format") or "openai-chat"),
            source_repository=str(meta.get("source_repository") or ""),
        )
    return seen


def convert_agenterrorbench(
    writer: Writer,
    limit: int = 18,
    preserved: list[dict] | None = None,
) -> None:
    seen_tids = convert_preserved_generic(
        writer, preserved or [], corpus="agenterrorbench"
    )
    if writer.stats.get("corpus:agenterrorbench", 0) >= limit:
        return
    data_dir = RAW / "agenterrorbench" / "data"
    parquets = sorted(data_dir.glob("*.parquet"))
    if not parquets:
        print("skip AgentErrorBench: parquet missing (run --fetch)")
        return
    try:
        from datasets import load_dataset
    except ImportError:
        print("skip AgentErrorBench: `datasets` not installed")
        return
    files = {p.stem.split("-")[0]: str(p) for p in parquets}
    ds = load_dataset("parquet", data_files=files)
    taken: Counter = Counter()
    for split in ds:
        for row in ds[split]:
            if writer.stats["corpus:agenterrorbench"] >= limit:
                return
            traj = _as_json(row.get("full_trajectory"))
            if not isinstance(traj, dict):
                continue
            messages = [m for m in (traj.get("messages") or []) if isinstance(m, dict)]
            if not messages:
                continue
            blob_len = len(row.get("full_trajectory") or "")
            if blob_len > 90_000:
                continue
            anns = _as_json(row.get("step_annotations")) or []
            if isinstance(anns, dict):
                anns = [anns]
            candidates: list[tuple[int, str, str]] = []
            for ann in anns:
                if not isinstance(ann, dict):
                    continue
                try:
                    step_no = int(ann.get("step"))
                except (TypeError, ValueError):
                    continue
                for key, val in ann.items():
                    if key == "step" or not isinstance(val, dict):
                        continue
                    ftype = str(val.get("failure_type") or "").strip()
                    family = map_aeb_family(ftype)
                    if family is None:
                        continue
                    candidates.append(
                        (step_no, family, str(val.get("reasoning") or ftype))
                    )
            if not candidates:
                fts = [
                    str(t).strip()
                    for t in (row.get("failure_types") or [])
                    if str(t).strip()
                ]
                reasons = list(row.get("failure_reasonings") or [])
                try:
                    step_no = int(row.get("critical_failure_step"))
                except (TypeError, ValueError):
                    continue
                if fts:
                    family = map_aeb_family(fts[0])
                    if family:
                        candidates.append(
                            (step_no, family, str(reasons[0] if reasons else fts[0]))
                        )
            if not candidates:
                continue
            primary = candidates[0][1]
            # stratify a bit
            if taken[primary] >= 2 and primary not in {
                "instruction_violation",
                "incorrect_tool_use",
                "evidence_contradiction",
                "unsupported_success",
            }:
                if sum(taken.values()) >= 4:
                    continue
            extra = {
                "task": next(
                    (
                        m.get("content") or ""
                        for m in messages
                        if m.get("role") == "user"
                    ),
                    "",
                ),
            }
            tid_src = str(row.get("trajectory_id") or "")
            slug = re.sub(r"[^A-Za-z0-9]+", "-", tid_src).strip("-")[:40].lower()
            tid = f"aeb-{slug}"
            if tid in seen_tids:
                continue
            recovered, mapping = ingest_mapped(tid, messages, extra)
            findings = []
            labels = []
            for step_no, family, desc in candidates:
                msg_idxs = aeb_step_message_indices(messages, step_no)
                prefer = [
                    i for i in msg_idxs if messages[i].get("role") == "assistant"
                ] or msg_idxs
                steps: list[int] = []
                for mi in prefer:
                    steps.extend(mapping.get(mi, []))
                steps = sorted(set(steps))
                if not steps:
                    continue
                findings.append(
                    {
                        "family": family,
                        "steps": steps,
                        "description": desc[:900],
                        "severity": "major",
                        "notes": f"AgentErrorBench step {step_no}",
                    }
                )
                labels.append(family)
            if not findings:
                writer.skipped["aeb_unmapped_step"] += 1
                continue
            payload = {
                "task": extra["task"],
                "messages": messages,
                "meta": {
                    "corpus": "agenterrorbench",
                    "license": "unspecified-public-research",
                    "original_id": tid_src,
                    "label_kind": "human",
                    "task_type": str(row.get("task_type") or ""),
                },
            }
            if writer.add(
                trace_id=tid,
                payload=payload,
                findings=findings,
                clean=False,
                notes=f"task_type={row.get('task_type')}; module={row.get('critical_failure_module')}",
                annotator="AgentErrorBench annotators (arXiv:2509.25370)",
                review="human",
                corpus="agenterrorbench",
                license_name="unspecified (public HF mirror davide221/agenterrorbench)",
                original_id=tid_src,
                original_label="; ".join(labels),
                label_kind="human",
                source_format="openai-chat",
                source_repository="davide221/agenterrorbench",
            ):
                taken[primary] += 1
                seen_tids.add(tid)
            _ = recovered


def convert_aegis(
    writer: Writer,
    limit: int = 26,
    preserved: list[dict] | None = None,
) -> None:
    seen_orig = convert_preserved_aegis(writer, preserved or [])
    have = writer.stats.get("corpus:aegis", 0)
    if have >= limit:
        return
    selected_path = RAW / "aegis" / "selected.jsonl"
    rows: list[dict] = []
    if selected_path.is_file():
        for line in selected_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                orig = str(rec.get("id") or "")
                if orig and orig in seen_orig:
                    continue
                rows.append(rec)
    if have + len(rows) < limit:
        try:
            from datasets import load_dataset
        except ImportError:
            print("skip AEGIS: `datasets` not installed")
            return
        print("streaming AEGIS (injected; taking a small slice)...")
        ds = load_dataset("Fancylalala/AEGIS", split="train", streaming=True)
        buckets: dict[str, list] = {}
        for row in ds:
            inp = _as_json(row.get("input")) or {}
            gt = _as_json(row.get("ground_truth")) or {}
            out = _as_json(row.get("output")) or {}
            meta = _as_json(row.get("metadata")) or {}
            hist = inp.get("conversation_history") if isinstance(inp, dict) else None
            if not isinstance(hist, list) or len(hist) < 2:
                continue
            agents = []
            if isinstance(gt, dict):
                agents = list(gt.get("injected_agents") or [])
            if not agents and isinstance(out, dict):
                agents = list(out.get("faulty_agents") or [])
            agents = [a for a in agents if isinstance(a, dict) and a.get("error_type")]
            if len(agents) != 1:
                continue
            code = str(agents[0].get("error_type") or "")
            family = map_aegis_family(code)
            rec = {
                "id": row.get("id"),
                "metadata": meta,
                "input": inp,
                "ground_truth": gt,
                "output": out,
                "family": family,
                "code": code,
            }
            bucket = buckets.setdefault(family, [])
            orig = str(rec.get("id") or "")
            if orig and orig in seen_orig:
                continue
            if len(bucket) >= 5:
                continue
            bucket.append(rec)
            if sum(len(v) for v in buckets.values()) >= max(limit * 2, 12):
                break
        picked: list[dict] = []
        order = [
            "instruction_violation",
            "incorrect_tool_use",
            "evidence_contradiction",
            "unsupported_success",
            "redundant_action",
            "ignored_feedback",
            "other",
        ]
        while len(picked) < limit and any(buckets.values()):
            for fam in order:
                if buckets.get(fam):
                    picked.append(buckets[fam].pop(0))
                if len(picked) >= limit:
                    break
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in picked) + "\n",
            encoding="utf-8",
        )
        rows = picked
        print(f"wrote {selected_path} ({len(rows)} extra injected traces)")

    start = 0
    for path in LABELS.glob("aegis-*.json") if LABELS.is_dir() else []:
        m = re.match(r"aegis-(\d+)$", path.stem)
        if m:
            start = max(start, int(m.group(1)) + 1)
    for existing in preserved or []:
        m = re.match(
            r"aegis-(\d+)$", str(existing.get("label", {}).get("trace_id") or "")
        )
        if m:
            start = max(start, int(m.group(1)) + 1)
    remaining = max(0, limit - writer.stats.get("corpus:aegis", 0))
    for i, row in enumerate(rows[:remaining]):
        inp = row.get("input") or {}
        gt = row.get("ground_truth") or {}
        out = row.get("output") or {}
        meta = row.get("metadata") or {}
        hist = inp.get("conversation_history") or []
        query = str(inp.get("query") or "")
        messages = [{"role": "user", "content": query}] if query else []
        agent_to_msg: dict[str, list[int]] = {}
        for turn in hist:
            if not isinstance(turn, dict):
                continue
            name = str(turn.get("agent_name") or "")
            messages.append(
                {
                    "role": "assistant",
                    "name": name,
                    "content": str(turn.get("content") or ""),
                }
            )
            agent_to_msg.setdefault(name, []).append(len(messages) - 1)
        agents = list(gt.get("injected_agents") or []) or list(
            out.get("faulty_agents") or []
        )
        agents = [a for a in agents if isinstance(a, dict)]
        extra = {"task": query}
        tid = f"aegis-{start + i:02d}"
        recovered, mapping = ingest_mapped(tid, messages, extra)
        findings = []
        labels = []
        for agent in agents:
            family = map_aegis_family(str(agent.get("error_type") or ""))
            name = str(agent.get("agent_name") or "")
            msg_idxs = agent_to_msg.get(name) or []
            steps: list[int] = []
            for mi in msg_idxs:
                steps.extend(mapping.get(mi, []))
            steps = sorted(set(steps))
            if not steps:
                continue
            desc = str(agent.get("malicious_action_description") or "")
            if not desc:
                desc = f"Injected {agent.get('error_type')} via {agent.get('injection_strategy')} on agent {name}"
            findings.append(
                {
                    "family": family,
                    "steps": steps,
                    "description": desc[:900],
                    "severity": "major",
                    "notes": f"AEGIS injected {agent.get('error_type')} ({agent.get('injection_strategy')})",
                }
            )
            labels.append(str(agent.get("error_type") or ""))
        payload = {
            "task": query,
            "messages": messages,
            "meta": {
                "corpus": "aegis",
                "license": "MIT",
                "original_id": str(row.get("id") or ""),
                "label_kind": "injected",
                "framework": str(meta.get("framework") or ""),
                "injected": True,
            },
        }
        writer.add(
            trace_id=tid,
            payload=payload,
            findings=findings,
            clean=False,
            notes="Injected AEGIS error (construction-verified). Not naturalistic gold.",
            annotator="AEGIS injection plan (arXiv:2509.14295)",
            review="injected",
            corpus="aegis",
            license_name="MIT",
            original_id=str(row.get("id") or ""),
            original_label="; ".join(labels),
            label_kind="injected",
            source_format="openai-chat",
            source_repository="Fancylalala/AEGIS",
            extra_label={
                "source_benchmark": "AEGIS",
                "gold_completeness": "injected_agents",
            },
        )
        _ = recovered


def _trail_attr(span: dict, *keys: str):
    attrs = span.get("attributes") or span.get("span_attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}
    for key in keys:
        if attrs.get(key) not in (None, ""):
            return attrs[key]
        if span.get(key) not in (None, ""):
            return span[key]
    return None


def trail_spans_to_canonical(
    trace_id: str, payload: dict
) -> tuple[dict, dict[str, list[int]]]:
    """Flatten nested TRAIL/OpenInference spans into a canonical trace."""
    spans = payload.get("spans") or payload.get("trace") or []
    if isinstance(spans, dict):
        spans = spans.get("spans") or spans.get("child_spans") or [spans]
    steps: list[dict] = []
    span_map: dict[str, list[int]] = {}

    def walk(nodes, depth=0) -> None:
        if depth > 30:
            return
        if isinstance(nodes, dict):
            nodes = [nodes]
        if not isinstance(nodes, list):
            return
        for span in nodes:
            if not isinstance(span, dict):
                continue
            ctx = span.get("context") if isinstance(span.get("context"), dict) else {}
            sid = str(span.get("span_id") or ctx.get("span_id") or "")
            kind = str(
                _trail_attr(span, "openinference.span.kind", "span.kind") or ""
            ).upper()
            name = str(
                _trail_attr(span, "tool.name", "llm.model_name")
                or span.get("name")
                or ""
            )
            inp = _trail_attr(span, "input.value", "input")
            out = _trail_attr(span, "output.value", "output")
            inp_text = (
                inp
                if isinstance(inp, str)
                else json.dumps(inp, ensure_ascii=False, default=str)
                if inp
                else ""
            )
            out_text = (
                out
                if isinstance(out, str)
                else json.dumps(out, ensure_ascii=False, default=str)
                if out
                else ""
            )
            added: list[int] = []
            if kind in {"TOOL"}:
                args = _as_json(inp) if isinstance(inp, (str, dict)) else {}
                if not isinstance(args, dict):
                    args = {"_raw": inp_text}
                steps.append(
                    {
                        "index": len(steps),
                        "kind": "tool_call",
                        "role": "assistant",
                        "name": name,
                        "arguments": args,
                        "meta": {"span_id": sid},
                    }
                )
                added.append(len(steps) - 1)
                if out_text:
                    steps.append(
                        {
                            "index": len(steps),
                            "kind": "tool_result",
                            "role": "tool",
                            "name": name,
                            "content": out_text,
                            "meta": {"span_id": sid},
                        }
                    )
                    added.append(len(steps) - 1)
            elif inp_text or out_text or kind in {"LLM", "AGENT", "CHAIN"}:
                body = out_text or inp_text
                if body:
                    steps.append(
                        {
                            "index": len(steps),
                            "kind": "message",
                            "role": "assistant"
                            if kind in {"LLM", "AGENT"}
                            else "unknown",
                            "name": name,
                            "content": body,
                            "meta": {"span_id": sid},
                        }
                    )
                    added.append(len(steps) - 1)
            if sid:
                span_map.setdefault(sid, []).extend(added)
            kids = span.get("child_spans") or span.get("children") or []
            walk(kids, depth + 1)

    walk(spans)
    canonical = {
        "trace_id": trace_id,
        "source_format": "canonical",
        "task": str(payload.get("task") or payload.get("question") or ""),
        "instructions": str(payload.get("instructions") or payload.get("system") or ""),
        "steps": steps,
        "meta": {"corpus": "trail", "label_kind": "human", "not_redistributed": True},
    }
    return canonical, span_map


def convert_trail(writer: Writer, limit: int = 20) -> None:
    src = RAW / "trail"
    if not src.is_dir():
        print(
            "skip TRAIL: gated / not downloaded (expected). Converter is ready if you fetch into data/raw/trail."
        )
        return
    files = [
        p
        for p in src.rglob("*.json")
        if p.name != "index.json" and ".cache" not in p.parts
    ]
    if not files:
        print("skip TRAIL: no json under data/raw/trail")
        return
    n = 0
    for path in sorted(files):
        if n >= limit:
            break
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        anns = _as_json(
            doc.get("errors")
            or doc.get("annotations")
            or doc.get("error_annotations")
            or []
        )
        if isinstance(anns, dict):
            anns = [anns]
        payload = _as_json(doc.get("trace") or doc.get("spans") and doc or None) or doc
        tid = str(doc.get("trace_id") or path.stem)[:40]
        canonical, span_map = trail_spans_to_canonical(
            f"trail-{tid}", payload if isinstance(payload, dict) else doc
        )
        if not canonical["steps"]:
            continue
        findings = []
        labels = []
        for ann in anns or []:
            if not isinstance(ann, dict):
                continue
            leaf = str(
                ann.get("category") or ann.get("error_type") or ann.get("type") or ""
            )
            family = map_trail_family(leaf)
            if family is None:
                UNMAPPED.mkdir(parents=True, exist_ok=True)
                continue
            loc = ann.get("location") or ann.get("span_id") or ""
            steps = list(span_map.get(str(loc), []))
            if not steps:
                continue
            findings.append(
                {
                    "family": family,
                    "steps": steps,
                    "description": str(
                        ann.get("description") or ann.get("evidence") or leaf
                    )[:900],
                    "severity": {
                        "high": "critical",
                        "medium": "major",
                        "low": "minor",
                    }.get(str(ann.get("impact") or "").lower(), "major"),
                    "notes": f"TRAIL type: {leaf}",
                }
            )
            labels.append(leaf)
        if not findings:
            continue
        # TRAIL card forbids resharing outside a gated HF repo. Write only if --allow-trail.
        writer.add(
            trace_id=f"trail-{re.sub(r'[^A-Za-z0-9]+', '-', tid).strip('-')[:32]}",
            payload=canonical,
            findings=findings,
            clean=False,
            notes="TRAIL human debug labels. Do not reshare the raw corpus.",
            annotator="TRAIL expert annotators (arXiv:2505.08638)",
            review="human",
            corpus="trail",
            license_name="MIT (gated; no reshape/reshare)",
            original_id=tid,
            original_label="; ".join(labels),
            label_kind="human",
            source_format="canonical",
            source_repository="PatronusAI/TRAIL",
        )
        n += 1


def fetch(args: argparse.Namespace) -> None:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError
    except ImportError:
        print("pip install huggingface_hub datasets", file=sys.stderr)
        raise SystemExit(2)

    import urllib.request

    (RAW / "agentrx").mkdir(parents=True, exist_ok=True)
    gh = "https://raw.githubusercontent.com/microsoft/AgentRx/main"
    for dest, url in {
        RAW
        / "agentrx"
        / "tau_ground_truth.json": f"{gh}/data/ground_truth/tau_ground_truth.json",
        RAW
        / "agentrx"
        / "tau_dataset_failed.json": f"{gh}/data/tau_retail/tau_dataset_failed.json",
        RAW
        / "agentrx"
        / "magentic_one_ground_truth.json": f"{gh}/data/ground_truth/magentic_one_ground_truth.json",
    }.items():
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"have {dest}")
            continue
        print(f"fetch {url}")
        urllib.request.urlretrieve(url, dest)

    for repo_id, dest in (
        ("Kevin355/Who_and_When", RAW / "whowhen"),
        ("davide221/agenterrorbench", RAW / "agenterrorbench"),
    ):
        print(f"fetch {repo_id}")
        snapshot_download(repo_id, repo_type="dataset", local_dir=str(dest))

    print("fetch PatronusAI/TRAIL (may be gated)")
    try:
        snapshot_download(
            "PatronusAI/TRAIL", repo_type="dataset", local_dir=str(RAW / "trail")
        )
        print("TRAIL downloaded into gitignored data/raw/trail (do not vendor)")
    except GatedRepoError:
        print(
            "TRAIL gated — skipped. Accept terms + huggingface-cli login to fetch later."
        )
    except Exception as exc:
        print(f"TRAIL skipped: {type(exc).__name__}: {exc}")

    if args.fetch_aegis:
        print("AEGIS is streamed during convert (small injected slice only).")


def write_indexes(writer: Writer) -> None:
    CHECKED.mkdir(parents=True, exist_ok=True)
    PRECISION.mkdir(parents=True, exist_ok=True)
    index = {
        "format_version": 2,
        "dataset_version": "traceaudit-checked-v1",
        "annotation_status": "checked",
        "trace_count": len(writer.entries),
        "source_counts": dict(
            Counter(e.get("corpus") or e.get("source") for e in writer.entries)
        ),
        "note": (
            "Checked / construction-verified traces for tests and precision experiments. "
            "Not held-out. AEGIS rows are injected, not naturalistic gold. "
            "Existing data/dev is unchanged."
        ),
        "traces": writer.entries,
    }
    (CHECKED / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    precision = {
        "format_version": 2,
        "dataset_version": "traceaudit-precision-v1",
        "note": (
            "Offline precision-experiment sample. Pair traces with data/labels/checked. "
            "Do not quote as held-out. No live LLM spend path."
        ),
        "trace_count": len(writer.precision),
        "n_human": sum(1 for r in writer.precision if r["label_kind"] == "human"),
        "n_injected": sum(1 for r in writer.precision if r["label_kind"] == "injected"),
        "n_construction": sum(
            1 for r in writer.precision if r["label_kind"] == "construction"
        ),
        "n_clean": sum(1 for r in writer.precision if r["clean"]),
        "n_dirty": sum(1 for r in writer.precision if not r["clean"]),
        "traces": writer.precision,
    }
    (PRECISION / "index.json").write_text(
        json.dumps(precision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def verify_samples(writer: Writer) -> None:
    dataset = json.loads((CHECKED / "index.json").read_text(encoding="utf-8"))
    print("\n--- ingest verification (first of each corpus) ---")
    seen = set()
    for entry in dataset["traces"]:
        corpus = entry.get("corpus") or ""
        if corpus in seen:
            continue
        seen.add(corpus)
        tid = entry["trace_id"]
        files = [str(CHECKED / f["path"]) for f in entry["files"]]
        trace = load_trace(tid, entry.get("source") or "", files, MAX_STEP_CHARS)
        label = json.loads((LABELS / f"{tid}.json").read_text(encoding="utf-8"))
        print(
            f"{tid}  steps={len(trace.steps)}  clean={label['clean']}  findings={len(label['findings'])}"
        )
        for finding in label["findings"][:2]:
            steps = finding["steps"]
            blob = " | ".join(
                trace.steps[i].text()[:80].replace("\n", " ") for i in steps[:3]
            )
            print(f"  {finding['family']} {steps} :: {blob[:160]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch", action="store_true", help="download raw corpora into data/raw/"
    )
    parser.add_argument(
        "--fetch-aegis", action="store_true", help="allow AEGIS stream during convert"
    )
    parser.add_argument(
        "--convert", action="store_true", help="write data/checked from data/raw"
    )
    parser.add_argument(
        "--allow-trail",
        action="store_true",
        help="convert TRAIL if present in data/raw/trail",
    )
    parser.add_argument("--ww-limit", type=int, default=24)
    parser.add_argument("--aeb-limit", type=int, default=18)
    parser.add_argument("--aegis-limit", type=int, default=26)
    args = parser.parse_args()
    if not (args.fetch or args.convert):
        args.fetch = True
        args.convert = True
        args.fetch_aegis = True

    if args.fetch:
        fetch(args)
    if not args.convert:
        return 0

    preserved_aegis = snapshot_existing_aegis()
    preserved_aeb = snapshot_existing_corpus("aeb-")
    if CHECKED.exists():
        for child in (TRACES, LABELS):
            if child.exists():
                shutil.rmtree(child)
    TRACES.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)

    writer = Writer()
    convert_construction(writer)
    convert_agentrx_tau(writer)
    convert_agentrx_magentic(writer)
    convert_whowhen(writer, limit=args.ww_limit)
    convert_agenterrorbench(writer, limit=args.aeb_limit, preserved=preserved_aeb)
    convert_aegis(writer, limit=args.aegis_limit, preserved=preserved_aegis)
    if args.allow_trail:
        convert_trail(writer)
    else:
        print(
            "skip TRAIL vendoring (gated, no-reshare). Use --allow-trail only on a private checkout."
        )

    write_indexes(writer)
    verify_samples(writer)
    print("\nstats", dict(writer.stats))
    print("skipped", dict(writer.skipped))
    print(f"wrote {CHECKED} ({len(writer.entries)} traces)")
    print(f"wrote {LABELS}")
    print(f"wrote {PRECISION / 'index.json'}")
    return 0 if writer.entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
