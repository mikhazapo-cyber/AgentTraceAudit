from traceaudit.parsing import extract_json


def test_extract_plain_object():
    assert extract_json('{"findings": []}') == {"findings": []}


def test_extract_repairs_truncated_object():
    parsed = extract_json('{"findings": [{"family": "other", "description": "x"')
    assert parsed["findings"][0]["family"] == "other"
