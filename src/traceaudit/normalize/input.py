"""Turn a raw file or folder into the same dataset dict `load_dataset` returns.

A packaged dir with `index.json` is unchanged. Everything else — one JSON file,
a folder of JSON/JSONL traces, Tavii bundles — is detected and listed so boot
does not require the user to write an index first.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import adapter_profiles

_TRACE_SUFFIXES = {'.json', '.jsonl'}
_PROMPT_SUFFIXES = {'.txt', '.md', '.prompt', '.text'}
_SKIP_NAMES = {'index.json', 'complete.json'}


def _read_probe(path: Path):
    """First JSON value in a file, used only for format detection."""
    text = path.read_text(encoding='utf-8')[:2_000_000]
    stripped = text.lstrip()
    if not stripped:
        return None
    if path.suffix.lower() == '.jsonl' or stripped[0] not in '{[':
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            return json.loads(line)
        return None
    return json.loads(text)


def detect_source(path: Path | str, override: str = '') -> str:
    if override:
        return override
    path = Path(path)
    try:
        doc = _read_probe(path)
    except (OSError, ValueError):
        return ''
    ranked = adapter_profiles.detect(doc, str(path), top_k=3)
    if ranked and ranked[0][1] > 0:
        return ranked[0][0]
    if _is_canonical(doc):
        return "canonical"
    if isinstance(doc, dict) and 'messages' in doc:
        return 'openai-chat'
    if isinstance(doc, list) and doc and isinstance(doc[0], dict) and ('from' in doc[0] or 'value' in doc[0]):
        return 'agentinstruct'
    return ''


def _is_canonical(doc) -> bool:
    if not isinstance(doc, dict):
        return False
    steps = doc.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    first = steps[0]
    return isinstance(first, dict) and "kind" in first


def looks_like_trace(path: Path | str) -> bool:
    path = Path(path)
    if not path.is_file() or path.suffix.lower() not in _TRACE_SUFFIXES:
        return False
    if path.name.lower() in _SKIP_NAMES:
        return False
    try:
        doc = _read_probe(path)
    except (OSError, ValueError):
        return False
    if _is_canonical(doc):
        return True
    ranked = adapter_profiles.detect(doc, str(path), top_k=1)
    if ranked and ranked[0][1] > 0:
        return True
    if isinstance(doc, dict) and any(k in doc for k in (
        'messages', 'conversations', 'spans', 'output', 'trace_records',
        'available_tools', 'modelInvocations', 'contents', 'chat_messages',
    )):
        return True
    if isinstance(doc, list) and doc and isinstance(doc[0], dict) and any(
        k in doc[0] for k in ('role', 'from', 'content', 'value', 'tool_calls')
    ):
        return True
    return False


def _tavii_bundle_files(folder: Path) -> list[Path] | None:
    trace = folder / 'trace.jsonl'
    manifest = folder / 'manifest.json'
    if not (trace.is_file() and manifest.is_file()):
        return None
    files = [manifest, trace]
    complete = folder / 'complete.json'
    if complete.is_file():
        files.append(complete)
    return files


def collect_loose_traces(path: Path | str) -> list[tuple[str, list[str], str]]:
    """Return `(trace_id, files, source)` for a file or a folder without index.json."""
    path = Path(path)
    items: list[tuple[str, list[str], str]] = []
    seen: set[str] = set()

    def add(tid: str, files: list[Path], source: str) -> None:
        base = tid
        n = 2
        while tid in seen:
            tid = f'{base}-{n}'
            n += 1
        seen.add(tid)
        items.append((tid, [str(f) for f in files], source))

    if path.is_file():
        if not looks_like_trace(path):
            return items
        source = detect_source(path)
        add(path.stem, [path], source or 'openai-chat')
        return items
    if not path.is_dir():
        return items
    for child in sorted(path.iterdir()):
        if child.is_file() and looks_like_trace(child):
            add(child.stem, [child], detect_source(child) or 'openai-chat')
            continue
        if child.is_dir():
            bundle = _tavii_bundle_files(child)
            if bundle:
                add(child.name, bundle, 'tavii')
    return items


def load_loose_dataset(path: Path | str, source_override: str = '') -> dict[str, dict]:
    out: dict[str, dict] = {}
    for tid, files, source in collect_loose_traces(path):
        src = source_override or source
        out[tid] = {'source': src, 'format': src, 'files': files}
    if not out:
        raise FileNotFoundError(f'no recognizable traces at {path}')
    return out


def apply_overlays(trace, task: str = '', instructions: str = ''):
    """Overwrite task / instructions captured from the trace when the user supplied them."""
    if task:
        trace.task = task
        trace.meta['task_overlay'] = True
    if instructions:
        trace.instructions = instructions
        trace.meta['instructions_overlay'] = True
    return trace


def read_prompt_value(value: str) -> str:
    """A literal string, or the contents of `@path` / a short `.txt` path."""
    if not value:
        return ''
    raw = value[1:] if value.startswith('@') else value
    path = Path(raw)
    explicit = value.startswith('@')
    looks_file = path.suffix.lower() in _PROMPT_SUFFIXES and len(raw) < 512
    if (explicit or looks_file) and path.is_file():
        return path.read_text(encoding='utf-8-sig')
    return value
