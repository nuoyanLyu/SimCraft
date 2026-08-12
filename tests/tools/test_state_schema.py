"""Declarative per-family state schema (`tools.state_schema`).

The cases below are the defects the 240-task `mcp_notes` bank actually
contains, restated as assertions: a note object with no `tags` field, a
`notes` key modelled as a dict instead of a list, and a checker reading a
`note['id']` this domain never had.
"""

import pytest

from qwen_agentworld.tools.state_schema import (
    FAMILY_STATE_SCHEMAS,
    MCP_NOTES_SCHEMA,
    EntrySpec,
    ObjectSpec,
    StateSchema,
    get_schema,
    neutral_value,
)


def test_every_catalog_family_has_a_schema():
    from qwen_agentworld.tools.families import ALL_FAMILIES

    for family in ALL_FAMILIES:
        assert get_schema(family) is not None, f"{family} has no declared state schema"


def test_unknown_family_returns_none_rather_than_raising():
    """Every call site treats the schema as optional, so an unregistered
    family must degrade to the pre-schema behaviour instead of crashing."""
    assert get_schema("no_such_family") is None
    assert get_schema(None) is None


def test_neutral_values_are_typed_and_not_shared():
    assert neutral_value("list[string]") == []
    assert neutral_value("string") == ""
    assert neutral_value("integer") == 0
    first = neutral_value("list[string]")
    first.append("x")
    assert neutral_value("list[string]") == []  # not a shared mutable default


# --- validation: the bank's real defects --------------------------------- #


def test_missing_tags_field_is_a_violation():
    state = {"notes": [{"title": "A", "content": "a"}]}
    violations = MCP_NOTES_SCHEMA.validate_state(state)
    assert any("tags" in v for v in violations)


def test_notes_as_a_dict_is_a_violation():
    violations = MCP_NOTES_SCHEMA.validate_state({"notes": {"title": "A"}})
    assert any("must be a list" in v for v in violations)


def test_invented_field_is_a_violation():
    state = {"notes": [{"title": "A", "content": "a", "tags": [], "id": "n1"}]}
    violations = MCP_NOTES_SCHEMA.validate_state(state)
    assert any("'id'" in v for v in violations)


def test_wrong_field_type_is_a_violation():
    state = {"notes": [{"title": "A", "content": "a", "tags": "client"}]}
    violations = MCP_NOTES_SCHEMA.validate_state(state)
    assert any("list[string]" in v for v in violations)


def test_conforming_state_has_no_violations():
    state = {"notes": [{"title": "A", "content": "a", "tags": ["x"]}]}
    assert MCP_NOTES_SCHEMA.validate_state(state) == []


def test_action_log_can_be_ignored_by_the_caller():
    state = {"notes": [], "_action_log": []}
    assert MCP_NOTES_SCHEMA.validate_state(state) != []
    assert MCP_NOTES_SCHEMA.validate_state(state, ignore_keys=frozenset({"_action_log"})) == []


def test_violations_say_what_to_change():
    """They are fed back to the teacher verbatim as retry feedback."""
    violations = MCP_NOTES_SCHEMA.validate_state({"notes": [{"title": "A"}]})
    assert all(v[0].islower() for v in violations)
    assert any(v.startswith(("add ", "remove ")) for v in violations)


# --- repair -------------------------------------------------------------- #


def test_conform_state_adds_missing_fields_only():
    state = {"notes": [{"title": "A", "content": "keep me"}]}
    conformed = MCP_NOTES_SCHEMA.conform_state(state)
    assert conformed["notes"][0] == {"title": "A", "content": "keep me", "tags": []}


def test_conform_state_adds_a_missing_collection_as_empty():
    assert MCP_NOTES_SCHEMA.conform_state({}) == {"notes": []}


def test_conform_state_leaves_undeclared_keys_alone():
    """Dropping them would discard whatever the agent did; that is the
    checker's judgement to make, not the schema's."""
    conformed = MCP_NOTES_SCHEMA.conform_state({"notes": [], "_action_log": [{"tool": "x"}]})
    assert conformed["_action_log"] == [{"tool": "x"}]


# --- prompting ----------------------------------------------------------- #


def test_describe_names_every_field_and_shows_an_example():
    text = MCP_NOTES_SCHEMA.describe()
    for token in ("notes", "title", "content", "tags", "list[string]"):
        assert token in text
    assert "mcp_notes" in text


def test_example_state_conforms_to_its_own_schema():
    for schema in FAMILY_STATE_SCHEMAS.values():
        assert schema.validate_state(schema.as_example()) == [], schema.family


def test_allowed_keys_covers_top_level_and_fields():
    assert MCP_NOTES_SCHEMA.allowed_keys() == {"notes", "title", "content", "tags"}


# --- construction guards ------------------------------------------------- #


def test_unknown_type_token_is_rejected_at_declaration_time():
    with pytest.raises(ValueError):
        ObjectSpec(fields={"title": "str"})


def test_identity_must_be_a_declared_field():
    with pytest.raises(ValueError):
        ObjectSpec(fields={"title": "string"}, identity="id")


def test_list_entry_requires_an_item_spec():
    with pytest.raises(ValueError):
        EntrySpec(kind="list")


def test_scalar_entry_requires_a_type_token():
    with pytest.raises(ValueError):
        EntrySpec(kind="scalar")


def test_schema_family_keys_match_their_declarations():
    for name, schema in FAMILY_STATE_SCHEMAS.items():
        assert isinstance(schema, StateSchema)
        assert schema.family == name
