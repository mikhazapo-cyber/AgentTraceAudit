from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

Detector = Callable[[object, str], float]

@dataclass(frozen=True)
class AdapterProfile:
    name: str
    loader: str
    description: str
    detectors: tuple[Detector, ...] = field(default_factory=tuple)
    loss_fields: tuple[str, ...] = field(default_factory=tuple)
_REGISTRY: dict[str, AdapterProfile] = {}

def register(profile: AdapterProfile) -> AdapterProfile:
    _REGISTRY[profile.name] = profile
    return profile

def get(name: str) -> AdapterProfile | None:
    return _REGISTRY.get(name)

def all_profiles() -> list[AdapterProfile]:
    return sorted(_REGISTRY.values(), key=lambda p: p.name)

def detect(record_or_doc: object, file_path: str, top_k: int=3) -> list[tuple[str, float]]:
    scored: list[tuple[str, float, int]] = []
    for prof in _REGISTRY.values():
        if not prof.detectors:
            continue
        hits = 0
        for det in prof.detectors:
            try:
                if det(record_or_doc, file_path):
                    hits += 1
            except Exception:
                continue
        score = hits / len(prof.detectors)
        scored.append((prof.name, score, hits))
    # Ratio first, then raw hits so a two-detector profile that both fire
    # beats a one-detector profile that also scores 1.0. Alphabetical last.
    scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return [(name, score) for name, score, _hits in scored[:top_k]]

def _has(obj, *keys) -> bool:
    return isinstance(obj, dict) and all((k in obj for k in keys))

def _is_first_value(obj, key, predicate) -> bool:
    if not isinstance(obj, dict):
        return False
    val = obj.get(key)
    if not isinstance(val, list) or not val:
        return False
    return predicate(val[0])

def _by_top_level(doc, key) -> bool:
    return isinstance(doc, dict) and key in doc

def det_openai_chat_doc(doc, path: str) -> bool:
    if not isinstance(doc, dict):
        return False
    if 'messages' not in doc or 'model' not in doc:
        return False
    msgs = doc.get('messages') or []
    if not isinstance(msgs, list) or not msgs:
        return False
    return any((isinstance(m, dict) and (m.get('role') == 'assistant' or m.get('role') == 'tool') and (m.get('content') is not None) for m in msgs[:6]))

def det_openai_chat_record(rec, path: str) -> bool:
    return isinstance(rec, dict) and ('messages' in rec and 'model' in rec or (rec.get('object') == 'thread.run' and rec.get('choices') is not None))

def det_openai_responses_doc(doc, path: str) -> bool:
    if not isinstance(doc, dict):
        return False
    if not isinstance(doc.get('output'), list):
        return False
    om = doc['output']
    return any((isinstance(b, dict) and b.get('type') in ('tool_call', 'function_call', 'web_search_call', 'file_search_call', 'code_interpreter_call', 'mcp_call', 'reasoning') for b in om[:20]))

def det_openai_responses_record(rec, path: str) -> bool:
    return isinstance(rec, dict) and isinstance(rec.get('output'), list) and any((isinstance(b, dict) and b.get('type') in ('tool_call', 'function_call', 'web_search_call', 'reasoning') for b in rec['output'][:10])) if isinstance(rec.get('output'), list) else False

def det_anthropic_doc(doc, path: str) -> bool:
    if not isinstance(doc, dict):
        return False
    msgs = doc.get('messages') or []
    if not isinstance(msgs, list) or not msgs:
        return False
    for m in msgs[:8]:
        if not isinstance(m, dict) or m.get('role') not in ('user', 'assistant'):
            continue
        content = m.get('content')
        if isinstance(content, list):
            if any((isinstance(b, dict) and b.get('type') in ('tool_use', 'tool_result', 'thinking') for b in content[:6])):
                return True
    if isinstance(doc.get('model'), str) and doc['model'].lower().startswith('claude'):
        return True
    return False

def det_gemini_doc(doc, path: str) -> bool:
    if not isinstance(doc, dict):
        return False
    for key in ('contents', 'messages'):
        conts = doc.get(key)
        if isinstance(conts, list):
            for c in conts[:6]:
                parts = (c or {}).get('parts') if isinstance(c, dict) else None
                if isinstance(parts, list) and any((isinstance(p, dict) and ('functionCall' in p or 'functionResponse' in p or 'inlineData' in p) for p in parts[:6])):
                    return True
    return False

def det_langchain_doc(doc, path: str) -> bool:
    if isinstance(doc, dict):
        if {'type', 'id', 'kwargs'}.intersection(doc.keys()):
            return True
        msgs = doc.get('messages') or []
        type_keys = sum((1 for m in msgs[:6] if isinstance(m, dict) and isinstance(m.get('type'), str) and m['type'].endswith('Message')))
        tool_keys = any((isinstance(m.get('tool_calls'), list) for m in msgs[:6] if isinstance(m, dict)))
        return type_keys >= 2 or tool_keys
    return False

def det_autogen_doc(doc, path: str) -> bool:
    if isinstance(doc, dict):
        return 'chat_messages' in doc or ('config' in doc and isinstance(doc.get('config'), dict) and ('messages' in doc.get('config', {})))
    if isinstance(doc, list) and doc:
        return isinstance(doc[0], dict) and any((k in doc[0] for k in ('name', 'role', 'content', 'tool_calls', 'tool_call_id')))
    return False

def det_metagpt_doc(doc, path: str) -> bool:
    if not isinstance(doc, dict):
        return False
    if 'roles' in doc and isinstance(doc.get('roles'), list):
        return True
    if 'environment' in doc and isinstance(doc.get('environment'), dict):
        return True
    if 'actions' in doc and isinstance(doc.get('actions'), list):
        return True
    return False

def det_bedrock_doc(doc, path: str) -> bool:
    if not isinstance(doc, dict):
        return False
    model_inv = doc.get('modelInvocations') or doc.get('model_invocations')
    tool_inv = doc.get('toolInvocations') or doc.get('tool_invocations')
    return isinstance(model_inv, list) or isinstance(tool_inv, list)

def det_mlflow_doc(doc, path: str) -> bool:
    if not isinstance(doc, dict):
        return False
    if isinstance(doc.get('spans'), list):
        for s in doc['spans'][:6]:
            if not isinstance(s, dict):
                continue
            if s.get('span_type', '').upper() == 'LLM' or s.get('name', '').endswith('llm'):
                return True
            events = s.get('events') or []
            if any((isinstance(e, dict) and e.get('type') in ('chat_completion_request', 'chat_completion_response') for e in events[:6])):
                return True
    return False

def det_otel_doc(doc, path: str) -> bool:
    if isinstance(doc, dict):
        rs = doc.get('resourceSpans')
        ss = doc.get('scopeSpans')
        if isinstance(rs, list) and rs and isinstance(ss, list) and ss:
            import json as _json
            first = _json.dumps(rs[:1] + ss[:1], default=str)
            return 'gen_ai' in first
    return False

def det_openinference_doc(doc, path: str) -> bool:
    """OpenInference spans: OTLP-shaped, but attributes are openinference/llm/tool."""
    import json as _json
    if not isinstance(doc, (dict, list)):
        return False
    try:
        blob = _json.dumps(doc, default=str)[:200000]
    except (TypeError, ValueError):
        return False
    if 'openinference.span.kind' in blob or 'openinference.span_kind' in blob:
        return True
    return ('llm.input_messages' in blob or 'llm.output_messages' in blob) and 'gen_ai.' not in blob

def det_openinference_path(doc, path: str) -> bool:
    return bool(re.search('trail|phoenix|openinference', path or '', re.I))

def det_mcp_doc(doc, path: str) -> bool:
    if isinstance(doc, dict):
        if doc.get('jsonrpc') or doc.get('method', '').startswith(('tools/', 'prompts/', 'resources/', 'initialize')):
            return True
        if 'id' in doc and ('result' in doc or 'error' in doc):
            return True
    if isinstance(doc, list) and doc and isinstance(doc[0], dict):
        first = doc[0]
        return first.get('jsonrpc') or first.get('method', '').startswith('tools/')
    return False

def det_mcp_record(rec, path: str) -> bool:
    return isinstance(rec, dict) and (rec.get('jsonrpc') or rec.get('method', '').startswith('tools/'))

def det_litellm_doc(doc, path: str) -> bool:
    if not isinstance(doc, dict):
        return False
    if 'trace_id' in doc and isinstance(doc.get('trace_id'), str):
        spans = doc.get('spans')
        if isinstance(spans, list) and spans:
            return True
        if 'calls' in doc or 'litellm_params' in doc:
            return True
    return False

def det_tavii_doc(doc, path: str) -> bool:
    if not isinstance(doc, dict):
        return False
    if doc.get('format') == 'semantic_trace_bundle' or 'trace_records' in doc:
        return True
    return 'trace.jsonl' in path and isinstance(doc.get('trace_id'), str)

def det_agentinstruct_doc(doc, path: str) -> bool:
    if not isinstance(doc, list) or not doc:
        return False
    sample = doc[:12]
    roles = {str(m.get('from', '')).lower() for m in sample if isinstance(m, dict)}
    if not roles.intersection({'gpt', 'human'}):
        return False
    text = '\n'.join((str(m.get('value', '')) for m in sample if isinstance(m, dict)))
    return bool(re.search('\\b(Think:\\s*|Act:\\s*(bash|finish|answer)|ACTION:\\s*)', text, re.I))

def det_apigen_doc(doc, path: str) -> bool:
    return isinstance(doc, dict) and 'conversations' in doc and isinstance(doc.get('tools'), (list, str))

def det_nemotron_doc(doc, path: str) -> bool:
    if not isinstance(doc, dict):
        return False
    if 'reasoning' in doc and isinstance(doc.get('messages'), list):
        return True
    msgs = doc.get('messages') or []
    if not isinstance(msgs, list) or not msgs:
        return False
    return any((isinstance(m, dict) and m.get('reasoning_content') for m in msgs[:6]))

def det_toucan_doc(doc, path: str) -> bool:
    if not isinstance(doc, dict):
        return False
    if isinstance(doc.get('available_tools'), str) and isinstance(doc.get('messages'), str):
        return True
    return 'toucan' in path.lower() and 'messages' in doc

register(AdapterProfile(name='openai-chat', loader='traceaudit.normalize.openai_chat', description='OpenAI Chat Completions wire format (messages+tool_calls).', detectors=(det_openai_chat_doc, det_openai_chat_record), loss_fields=('tokens (estimated)', 'exact refusal reason')))
register(AdapterProfile(name='openai-responses', loader='traceaudit.normalize.openai_responses', description='OpenAI Responses API (output[].type=tool_call/web_search/etc.).', detectors=(det_openai_responses_doc, det_openai_responses_record), loss_fields=('streamed deltas collapsed to final block',)))
register(AdapterProfile(name='anthropic-messages', loader='traceaudit.normalize.anthropic_messages', description='Anthropic Messages API (content[].type=tool_use/tool_result).', detectors=(det_anthropic_doc,), loss_fields=('thinking tokens summarised',)))
register(AdapterProfile(name='google-gemini', loader='traceaudit.normalize.google_gemini', description='Google Gemini (contents[].parts[] with functionCall).', detectors=(det_gemini_doc,), loss_fields=('inlineData blobs omitted',)))
register(AdapterProfile(name='langchain', loader='traceaudit.normalize.langchain', description='LangChain / LangGraph serialized messages.', detectors=(det_langchain_doc,)))
register(AdapterProfile(name='autogen', loader='traceaudit.normalize.autogen', description='AutoGen v0.x / v0.2 chat-history trace.', detectors=(det_autogen_doc,)))
register(AdapterProfile(name='metagpt', loader='traceaudit.normalize.metagpt', description='MetaGPT software-company trace (roles/actions/environment).', detectors=(det_metagpt_doc,)))
register(AdapterProfile(name='bedrock-trajectory', loader='traceaudit.normalize.bedrock_trajectory', description='AWS Bedrock Agent Runtime trajectory (modelInvocations/toolInvocations).', detectors=(det_bedrock_doc,)))
register(AdapterProfile(name='mlflow-trace', loader='traceaudit.normalize.mlflow_trace', description='MLflow Trace span exporter (LLM spans with chat_completion events).', detectors=(det_mlflow_doc,)))
register(AdapterProfile(name='otel-genai', loader='traceaudit.normalize.otel_genai', description='OTel gen_ai.* telemetry stream.', detectors=(det_otel_doc,)))
register(AdapterProfile(name='openinference', loader='traceaudit.normalize.openinference', description='OpenInference spans (Arize Phoenix, TRAIL benchmark): openinference.span.kind + llm.*/tool.* attributes.', detectors=(det_openinference_doc, det_openinference_path), loss_fields=('span timing', 'token counts', 'repeated conversation prefixes collapsed')))
register(AdapterProfile(name='mcp-events', loader='traceaudit.normalize.mcp_events', description='MCP server JSON-RPC request/response capture.', detectors=(det_mcp_doc, det_mcp_record), loss_fields=('streaming notifications collapsed',)))
register(AdapterProfile(name='litellm-telemetry', loader='traceaudit.normalize.litellm_telemetry', description='LiteLLM telemetry spans (spans with litellm_params).', detectors=(det_litellm_doc,)))
register(AdapterProfile(name='tavii', loader='traceaudit.normalize.tavii', description='Tavii semantic_trace_bundle (seq-ordered JSONL).', detectors=(det_tavii_doc,), loss_fields=('intermediate LLM I/O collapsed', 'span timing')))
register(AdapterProfile(name='agentinstruct', loader='traceaudit.normalize.agentinstruct', description='AgentInstruct Think/Act and ALFWorld text protocols.', detectors=(det_agentinstruct_doc,), loss_fields=('message timing', 'turn boundaries')))
register(AdapterProfile(name='apigen-mt', loader='traceaudit.normalize.apigen_mt', description='APIGen-MT {system, tools, conversations} benchmark capture.', detectors=(det_apigen_doc,), loss_fields=('message timing', 'turn boundaries')))
register(AdapterProfile(name='nemotron-interactive', loader='traceaudit.normalize.nemotron', description='Nemotron interactive traces with reasoning_content.', detectors=(det_nemotron_doc,)))
register(AdapterProfile(name='nemotron-tool-calling', loader='traceaudit.normalize.nemotron', description='Nemotron tool-calling traces with reasoning_content.', detectors=(det_nemotron_doc,)))
register(AdapterProfile(name='toucan-mcp', loader='traceaudit.normalize.toucan', description='Toucan MCP trace (JSON-encoded messages + available_tools).', detectors=(det_toucan_doc,), loss_fields=('tool timing', 'nested call_id links')))
__all__ = ['AdapterProfile', 'Detector', 'register', 'get', 'all_profiles', 'detect']
