from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from ..schemas import CanonicalTrace
from . import generic

_ADAPTER_MODULE_MAP = {'canonical': 'traceaudit.normalize.generic', 'tavii': 'traceaudit.normalize.tavii', 'agentinstruct': 'traceaudit.normalize.agentinstruct', 'apigen-mt': 'traceaudit.normalize.apigen_mt', 'nemotron-interactive': 'traceaudit.normalize.nemotron', 'nemotron-tool-calling': 'traceaudit.normalize.nemotron', 'toucan-mcp': 'traceaudit.normalize.toucan', 'openai-chat': 'traceaudit.normalize.openai_chat', 'openai-responses': 'traceaudit.normalize.openai_responses', 'anthropic-messages': 'traceaudit.normalize.anthropic_messages', 'google-gemini': 'traceaudit.normalize.google_gemini', 'langchain': 'traceaudit.normalize.langchain', 'autogen': 'traceaudit.normalize.autogen', 'metagpt': 'traceaudit.normalize.metagpt', 'bedrock-trajectory': 'traceaudit.normalize.bedrock_trajectory', 'mlflow-trace': 'traceaudit.normalize.mlflow_trace', 'otel-genai': 'traceaudit.normalize.otel_genai', 'openinference': 'traceaudit.normalize.openinference', 'mcp-events': 'traceaudit.normalize.mcp_events', 'litellm-telemetry': 'traceaudit.normalize.litellm_telemetry'}
FORMAT_ALIASES = {'semantic_trace_bundle': 'tavii', 'trail': 'openinference', 'phoenix': 'openinference', 'openinference-otlp': 'openinference', 'canonical': 'canonical'}

def _source_to_module(source: str) -> str | None:
    source = FORMAT_ALIASES.get(source, source)
    if source in _ADAPTER_MODULE_MAP:
        return _ADAPTER_MODULE_MAP[source]
    try:
        from .. import adapter_profiles
        prof = adapter_profiles.get(source)
        if prof:
            return prof.loader
    except Exception:
        pass
    return None

def _detect_unpaired_calls(trace: CanonicalTrace) -> None:
    call_indices: dict[str, int] = {}
    result_ids: set[str] = set()
    for step in trace.steps:
        if step.kind == 'tool_call' and step.call_id:
            call_indices.setdefault(step.call_id, step.index)
        elif step.kind == 'tool_result' and step.call_id:
            result_ids.add(step.call_id)
    unpaired = [idx for cid, idx in call_indices.items() if cid not in result_ids]
    if unpaired:
        trace.capture_gaps.append(f'unpaired tool_call at step(s) {unpaired}: no matching tool_result')

def load_trace(trace_id: str, source: str, files: list[str], max_step_chars: int=6000) -> CanonicalTrace:
    source = FORMAT_ALIASES.get(source, source)
    mod_name = _source_to_module(source)
    try:
        if mod_name is None:
            trace = generic.load(trace_id, source, files, max_step_chars)
        else:
            mod = importlib.import_module(mod_name)
            trace = mod.load(trace_id, source, files, max_step_chars)
    except Exception as exc:
        trace = generic.load(trace_id, source, files, max_step_chars)
        trace.capture_gaps.append(f"primary adapter for '{source}' failed ({type(exc).__name__}: {exc}); generic fallback used")
    _detect_unpaired_calls(trace)
    if not trace.steps and not trace.capture_gaps:
        names = [Path(f).name for f in files]
        trace.capture_gaps.append(
            f"adapter '{source}' recovered no steps from {names}; nothing in this capture was audited"
        )
    for i, step in enumerate(trace.steps):
        step.index = i
    return trace

def load_dataset(dataset_dir: str | Path, max_step_chars: int=6000, source_override: str='') -> dict[str, dict]:
    dataset_dir = Path(dataset_dir)
    if dataset_dir.is_file() or not (dataset_dir / 'index.json').is_file():
        from .input import load_loose_dataset
        return load_loose_dataset(dataset_dir, source_override=source_override)
    index_path = dataset_dir / 'index.json'
    with open(index_path, encoding='utf-8') as fh:
        index = json.load(fh)
    out = {}
    for entry in index.get('traces', []):
        files = [str(dataset_dir / f['path']) for f in entry.get('files', [])]
        src = source_override or entry.get('source', '')
        out[entry['trace_id']] = {'source': src, 'format': entry.get('format', ''), 'files': files}
    return out
def verify_dataset(dataset_dir: str | Path) -> dict:
    """Check a dataset's files against the sha256 digests recorded in index.json.

    Answers "is my data intact", which is otherwise only discoverable after a
    run fails partway through.
    """
    root = Path(dataset_dir)
    index = json.loads((root / 'index.json').read_text(encoding='utf-8'))
    traces = index.get('traces') or []
    checked = missing = mismatched = 0
    for entry in traces:
        for f in entry.get('files') or []:
            want = f.get('sha256')
            if not want:
                continue
            checked += 1
            path = root / f['path']
            if not path.is_file():
                missing += 1
            elif hashlib.sha256(path.read_bytes()).hexdigest() != want:
                mismatched += 1
    return {'n_traces': len(traces), 'declared_traces': index.get('trace_count'), 'files_checked': checked, 'missing': missing, 'mismatched': mismatched, 'sources': index.get('source_counts') or {}}
__all__ = ['load_trace', 'load_dataset', 'verify_dataset', 'CanonicalTrace']
