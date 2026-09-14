from traceaudit.derive import obligation_checks
from traceaudit.evidence import obligation_sentences
from traceaudit.schemas import CanonicalTrace, ToolSpec


def test_extracts_superlative_restriction():
    text = (
        "get_attributes(variable) -> list of attributes. "
        "Please only use it if the question seeks for a superlative accumulation (i.e., argmax or argmin)."
    )
    sents = obligation_sentences(text)
    assert any("superlative" in s.lower() for s in sents)


def test_extracts_gift_card_balance_rule():
    text = (
        "If the user provides a gift card, it must have enough balance to cover the price difference. "
        "Call get_user_details before charging."
    )
    sents = obligation_sentences(text)
    assert any("gift card" in s.lower() or "must have" in s.lower() for s in sents)


def test_schema_checks_from_tools():
    trace = CanonicalTrace(
        trace_id="t",
        source_format="canonical",
        tools=[
            ToolSpec(
                name="message",
                parameters={"type": "object", "properties": {"messageId": {"type": "string"}}, "required": ["messageId"]},
            )
        ],
    )
    checks = obligation_checks(trace)
    assert any(c.family == "incorrect_tool_use" and "messageId" in c.description for c in checks)
