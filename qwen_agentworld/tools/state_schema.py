"""Declarative canonical-state schema per tool family.

Until now nothing declared what a family's state looks like. `families.py`
said so outright ("`initial_state` is not defined here"), and the only
artifact that came close, `FAMILY_STATE_HINTS`, was a prose sentence
referenced by a single test and never shown to any model. The shape was
therefore invented three separate times per task, by two different models:
the teacher invented it writing `initial_state`, invented it again writing
the checker, and the simulator invented it a third time predicting each
next state. Nothing made those three agree.

Measured consequences on the 240-task `mcp_notes` bank (2026-08-05):

  * 145 of 509 note objects (28%) have no `tags` field at all, while 364 do,
    and one task models `notes` as a dict instead of a list;
  * 7 checkers (2.9%) subscript a field the domain never had -- `note['id']`
    five times, an `arguments['note_title']` that no tool declares;
  * the simulator wrote `"tag": "client"` where the state uses
    `"tags": [...]`, on tasks whose initial state was `notes: []` -- with no
    sibling object to copy a shape from, `env.complete_fields` cannot repair
    that, and neither can anything else that only looks at the data.

All of those are the same defect: a schema that exists only in each model's
head. A checker subscripting a missing key does not evaluate to False, it
raises KeyError, which `judge_checker` scores as "task not passed" -- so a
task the agent solved correctly is recorded as one it keeps failing, and
that silently lowers every pass rate computed from it (the difficulty band,
the A/B arms, the validation utility).

This module makes the shape a first-class declaration, so that the same
object can be shown to the teacher while it writes the state, shown to the
teacher again while it writes the checker, shown to the simulator while it
predicts transitions, and used mechanically to audit all three. It is
deliberately a small closed vocabulary rather than JSON Schema: everything
downstream needs only "which top-level keys exist, which fields does an
object of this collection have, and what is each field's neutral value".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# Type tokens. Small on purpose: every consumer must be able to produce a
# neutral value for one, and the teacher must be able to read one unambiguously.
STRING = "string"
INTEGER = "integer"
NUMBER = "number"
BOOLEAN = "boolean"
STRING_LIST = "list[string]"
OBJECT = "object"

_NEUTRAL: dict[str, object] = {
    STRING: "",
    INTEGER: 0,
    NUMBER: 0,
    BOOLEAN: False,
    STRING_LIST: [],
    OBJECT: {},
}


def neutral_value(type_token: str) -> object:
    """The empty-but-well-typed value of a declared field.

    Used to complete an object the simulator emitted without a field, and to
    repair a state the teacher generated with one missing. A neutral value is
    never a guess about what the agent did -- it is the value that says "this
    field exists and is empty", which is what a checker needs in order to
    return False instead of raising.
    """
    value = _NEUTRAL[type_token]
    return list(value) if isinstance(value, list) else dict(value) if isinstance(value, dict) else value


def _matches(value: object, type_token: str) -> bool:
    if type_token == STRING:
        return isinstance(value, str)
    if type_token == INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if type_token == NUMBER:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_token == BOOLEAN:
        return isinstance(value, bool)
    if type_token == STRING_LIST:
        return isinstance(value, list) and all(isinstance(v, str) for v in value)
    if type_token == OBJECT:
        return isinstance(value, dict)
    raise KeyError(f"unknown type token: {type_token!r}")


@dataclass(frozen=True)
class ObjectSpec:
    """The fields every object in a collection has -- all of them, always.

    `identity` names the field that identifies the same object across states.
    `env.complete_fields` uses it to tell "the note that already existed, now
    missing a field the simulator did not re-emit" from "a note the agent
    just created": the former keeps its prior value, the latter gets a
    neutral one.
    """

    fields: dict[str, str]
    identity: str | None = None

    def __post_init__(self) -> None:
        for name, token in self.fields.items():
            if token not in _NEUTRAL:
                raise ValueError(f"field {name!r} has unknown type {token!r}")
        if self.identity is not None and self.identity not in self.fields:
            raise ValueError(f"identity {self.identity!r} is not one of the declared fields")


@dataclass(frozen=True)
class EntrySpec:
    """One top-level key of the canonical state.

    kind="list"   -- a list of homogeneous objects (`item` describes them)
    kind="object" -- a single object with declared fields (`item`)
    kind="scalar" -- a bare value of type `type_token`
    """

    kind: str
    item: ObjectSpec | None = None
    type_token: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind in ("list", "object") and self.item is None:
            raise ValueError(f"kind={self.kind!r} requires an `item` ObjectSpec")
        if self.kind == "scalar" and self.type_token not in _NEUTRAL:
            raise ValueError(f"kind='scalar' requires a known type_token, got {self.type_token!r}")
        if self.kind not in ("list", "object", "scalar"):
            raise ValueError(f"unknown entry kind: {self.kind!r}")


@dataclass(frozen=True)
class StateSchema:
    family: str
    entries: dict[str, EntrySpec] = field(default_factory=dict)

    # -- vocabulary ------------------------------------------------------- #

    def top_level_keys(self) -> set[str]:
        return set(self.entries)

    def field_names(self) -> set[str]:
        names: set[str] = set()
        for entry in self.entries.values():
            if entry.item is not None:
                names.update(entry.item.fields)
        return names

    def allowed_keys(self) -> set[str]:
        """Every key a checker may legitimately subscript on this domain's state."""
        return self.top_level_keys() | self.field_names()

    # -- prompting -------------------------------------------------------- #

    def as_example(self) -> dict:
        """A minimal conforming state: every declared key present, every
        collection empty. Shown to models as the authoritative shape.
        """
        example: dict = {}
        for key, entry in self.entries.items():
            if entry.kind == "list":
                example[key] = [{f: neutral_value(t) for f, t in entry.item.fields.items()}]
            elif entry.kind == "object":
                example[key] = {f: neutral_value(t) for f, t in entry.item.fields.items()}
            else:
                example[key] = neutral_value(entry.type_token)
        return example

    def describe(self) -> str:
        """The schema as prompt text: the rule, the field table, the example.

        Every model that touches the state sees this exact string, which is
        the whole point -- three independent inventions of the shape become
        one declaration read three times.
        """
        lines = [
            f"Canonical state schema for the '{self.family}' domain. "
            "The state is a JSON object with EXACTLY these top-level keys, and "
            "every object inside carries EXACTLY the fields listed for it -- "
            "including the ones a given step does not touch. Never rename a "
            "field, never add one, and never omit one because it is empty: an "
            "empty field is written as its empty value ([] for a list, \"\" for "
            "a string), not left out.",
            "",
        ]
        for key, entry in self.entries.items():
            if entry.kind == "scalar":
                lines.append(f"- {key}: {entry.type_token}{' -- ' + entry.description if entry.description else ''}")
                continue
            container = "list of objects" if entry.kind == "list" else "object"
            lines.append(f"- {key}: {container}{' -- ' + entry.description if entry.description else ''}")
            for fname, token in entry.item.fields.items():
                marker = "  (identity)" if fname == entry.item.identity else ""
                lines.append(f"    - {fname}: {token}{marker}")
        lines.append("")
        lines.append("A minimal conforming state looks like:")
        lines.append(json.dumps(self.as_example(), indent=2))
        return "\n".join(lines)

    # -- validation ------------------------------------------------------- #

    def validate_state(self, state: object, *, ignore_keys: frozenset[str] = frozenset()) -> list[str]:
        """Violations of this schema in `state`, phrased as instructions.

        The messages are fed straight back to the teacher as retry feedback,
        so each one says what to change rather than merely what is wrong.
        """
        if not isinstance(state, dict):
            return ["state must be a JSON object"]

        violations: list[str] = []
        for key in sorted(set(state) - self.top_level_keys() - set(ignore_keys)):
            violations.append(f"remove top-level key {key!r}: it is not part of this domain's state")
        for key in sorted(self.top_level_keys() - set(state)):
            violations.append(f"add the missing top-level key {key!r}")

        for key, entry in self.entries.items():
            if key not in state:
                continue
            value = state[key]
            if entry.kind == "list":
                if not isinstance(value, list):
                    violations.append(f"{key!r} must be a list of objects, not {type(value).__name__}")
                    continue
                for index, item in enumerate(value):
                    violations.extend(self._validate_object(item, entry.item, f"{key}[{index}]"))
            elif entry.kind == "object":
                if not isinstance(value, dict):
                    violations.append(f"{key!r} must be an object, not {type(value).__name__}")
                    continue
                violations.extend(self._validate_object(value, entry.item, key))
            elif not _matches(value, entry.type_token):
                violations.append(f"{key!r} must be a {entry.type_token}")
        return violations

    @staticmethod
    def _validate_object(item: object, spec: ObjectSpec, where: str) -> list[str]:
        if not isinstance(item, dict):
            return [f"{where} must be an object with fields {sorted(spec.fields)}"]
        violations = []
        for name in sorted(set(item) - set(spec.fields)):
            violations.append(f"remove field {name!r} from {where}: this domain's objects do not have it")
        for name in sorted(set(spec.fields) - set(item)):
            violations.append(
                f"add field {name!r} to {where} (type {spec.fields[name]}); "
                "every object carries every declared field, empty if unused"
            )
        for name, token in spec.fields.items():
            if name in item and not _matches(item[name], token):
                violations.append(f"{where}[{name!r}] must be a {token}")
        return violations

    # -- repair ----------------------------------------------------------- #

    def conform_state(self, state: object, *, keep_keys: frozenset[str] = frozenset()) -> object:
        """`state` with every declared-but-missing field added at its neutral
        value. Additive only: nothing declared-but-present is overwritten, and
        undeclared keys are left alone (dropping them here would silently
        discard whatever the agent did, which is the checker's call to make,
        not ours).
        """
        if not isinstance(state, dict):
            return state
        conformed = dict(state)
        for key, entry in self.entries.items():
            if key not in conformed:
                conformed[key] = [] if entry.kind == "list" else (
                    self._conform_object({}, entry.item) if entry.kind == "object"
                    else neutral_value(entry.type_token)
                )
                continue
            value = conformed[key]
            if entry.kind == "list" and isinstance(value, list):
                conformed[key] = [
                    self._conform_object(item, entry.item) if isinstance(item, dict) else item
                    for item in value
                ]
            elif entry.kind == "object" and isinstance(value, dict):
                conformed[key] = self._conform_object(value, entry.item)
        return conformed

    @staticmethod
    def _conform_object(item: dict, spec: ObjectSpec) -> dict:
        filled = dict(item)
        for name, token in spec.fields.items():
            if name not in filled:
                filled[name] = neutral_value(token)
        return filled

    def object_spec_for(self, key: str) -> ObjectSpec | None:
        entry = self.entries.get(key)
        return entry.item if entry is not None else None


# --------------------------------------------------------------------------- #
# The catalog. One schema per family in `families.ALL_FAMILIES`, plus the toy
# `mcp_notes` family that scripts/ defines and the entire 240-task bank uses.
#
# Shapes are chosen to be flat lists of homogeneous objects wherever possible:
# a checker can quantify over a list with `any(... for x in state['k'])`, which
# is the one construct the predicate allowlist supports well, whereas a
# name-keyed dict pushes it toward `.items()` and positional indexing.
# --------------------------------------------------------------------------- #

MCP_NOTES_SCHEMA = StateSchema(
    family="mcp_notes",
    entries={
        "notes": EntrySpec(
            kind="list",
            description="every note in the workspace",
            item=ObjectSpec(
                fields={"title": STRING, "content": STRING, "tags": STRING_LIST},
                identity="title",
            ),
        )
    },
)

MCP_API_SCHEMA = StateSchema(
    family="mcp_api",
    entries={
        "records": EntrySpec(
            kind="list",
            description="every record across all collections",
            item=ObjectSpec(
                fields={"record_id": STRING, "collection": STRING, "fields": OBJECT},
                identity="record_id",
            ),
        )
    },
)

TERMINAL_OPS_SCHEMA = StateSchema(
    family="terminal_ops",
    entries={
        "cwd": EntrySpec(kind="scalar", type_token=STRING, description="absolute current directory"),
        "entries": EntrySpec(
            kind="list",
            description="every file and directory in the filesystem",
            item=ObjectSpec(
                # `kind` is 'file' or 'directory'; a directory's text is "".
                fields={"path": STRING, "kind": STRING, "text": STRING},
                identity="path",
            ),
        ),
    },
)

WEB_RESEARCH_SCHEMA = StateSchema(
    family="web_research",
    entries={
        "last_results": EntrySpec(
            kind="list",
            description="ranked stubs from the most recent web_search",
            item=ObjectSpec(
                fields={"rank": INTEGER, "title": STRING, "url": STRING, "snippet": STRING},
                identity="rank",
            ),
        ),
        "notebook": EntrySpec(
            kind="list",
            description="notes saved with save_note",
            item=ObjectSpec(fields={"headline": STRING, "body": STRING}, identity="headline"),
        ),
    },
)

CODE_REPO_SCHEMA = StateSchema(
    family="code_repo",
    entries={
        "branch": EntrySpec(kind="scalar", type_token=STRING, description="currently checked-out branch"),
        "modules": EntrySpec(
            kind="list",
            description="every source module in the repository",
            item=ObjectSpec(
                fields={"module": STRING, "source": STRING, "modified": BOOLEAN},
                identity="module",
            ),
        ),
        "last_test_run": EntrySpec(
            kind="object",
            description="result of the most recent run_test_suite call",
            item=ObjectSpec(
                fields={"suite": STRING, "passed": INTEGER, "failed": INTEGER, "failing": STRING_LIST}
            ),
        ),
    },
)

FAMILY_STATE_SCHEMAS: dict[str, StateSchema] = {
    schema.family: schema
    for schema in (
        MCP_NOTES_SCHEMA,
        MCP_API_SCHEMA,
        TERMINAL_OPS_SCHEMA,
        WEB_RESEARCH_SCHEMA,
        CODE_REPO_SCHEMA,
    )
}


def get_schema(family: str | None) -> StateSchema | None:
    """The schema for `family`, or None if it has none declared.

    Returning None rather than raising keeps every call site optional: a
    caller with an unregistered family gets exactly the unvalidated,
    unprompted behaviour that existed before this module.
    """
    if family is None:
        return None
    return FAMILY_STATE_SCHEMAS.get(family)
