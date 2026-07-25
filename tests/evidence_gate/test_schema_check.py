from qwen_agentworld.evidence_gate.schema_check import is_schema_valid, validate_schema

SCHEMA = {
    "type": "object",
    "properties": {"status": {"type": "string"}, "id": {"type": "integer"}},
    "required": ["status", "id"],
}


def test_valid_dict_passes():
    assert is_schema_valid({"status": "ok", "id": 1}, SCHEMA)


def test_missing_required_field_fails():
    ok, errors = validate_schema({"status": "ok"}, SCHEMA)
    assert not ok
    assert errors


def test_wrong_type_fails():
    ok, _ = validate_schema({"status": "ok", "id": "not-an-int"}, SCHEMA)
    assert not ok


def test_malformed_json_string_fails():
    ok, errors = validate_schema("{not json", SCHEMA)
    assert not ok
    assert "not valid JSON" in errors[0]


def test_valid_json_string_passes():
    assert is_schema_valid('{"status": "ok", "id": 1}', SCHEMA)


def test_no_schema_only_checks_json_validity():
    assert is_schema_valid({"anything": "goes"}, None)
    assert not is_schema_valid("{broken", None)
