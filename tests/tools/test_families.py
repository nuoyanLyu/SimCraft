"""Tests for the multi-domain tool catalog (qwen_agentworld.tools.families)."""

from itertools import combinations

import pytest

from qwen_agentworld.tools.families import (
    ALL_FAMILIES,
    DEFAULT_HELDOUT_FAMILY,
    DEFAULT_TRAINING_FAMILIES,
    FAMILY_STATE_HINTS,
    build_registry,
    family_names,
    get_family,
    training_split,
)
from qwen_agentworld.tools.family_split import assert_family_isolation


def test_catalog_has_four_named_domains():
    assert set(family_names()) == {"mcp_api", "terminal_ops", "web_research", "code_repo"}
    assert set(FAMILY_STATE_HINTS) == set(ALL_FAMILIES)


def test_every_tool_is_tagged_with_its_family():
    for fam, tools in ALL_FAMILIES.items():
        assert tools, f"{fam} has no tools"
        assert all(t.family == fam for t in tools)


def test_tool_names_are_globally_unique():
    names = [t.name for tools in ALL_FAMILIES.values() for t in tools]
    assert len(names) == len(set(names))


def test_parameter_vocabularies_are_pairwise_disjoint():
    def keys(fam):
        ks = set()
        for t in ALL_FAMILIES[fam]:
            ks |= set(t.function.parameters.get("properties", {}).keys())
        return ks

    for a, b in combinations(family_names(), 2):
        assert not (keys(a) & keys(b)), f"param key overlap between {a} and {b}"


def test_any_train_eval_pair_passes_isolation_audit():
    for train_fam, eval_fam in combinations(family_names(), 2):
        report = assert_family_isolation(
            get_family(train_fam), get_family(eval_fam), train_fam, eval_fam
        )
        assert report.is_clean


def test_build_registry_all_families():
    reg = build_registry()
    assert reg.families() == set(family_names())
    total = sum(len(v) for v in ALL_FAMILIES.values())
    assert len(reg) == total


def test_build_registry_subset_scopes_wire_and_strips_internal_field():
    reg = build_registry(["mcp_api"])
    assert reg.families() == {"mcp_api"}
    wire = reg.to_wire(family="mcp_api")
    assert len(wire) == len(ALL_FAMILIES["mcp_api"])
    assert all("family" not in w for w in wire)


def test_default_training_split_is_clean_and_covers_the_rest():
    train, heldout = training_split()  # holds out web_research by default
    assert heldout is ALL_FAMILIES[DEFAULT_HELDOUT_FAMILY]
    train_families = {t.family for t in train}
    assert train_families == set(DEFAULT_TRAINING_FAMILIES)
    assert DEFAULT_HELDOUT_FAMILY not in train_families
    # train (multiple domains) vs held-out must stay isolated
    report = assert_family_isolation(train, heldout, "train_mix", DEFAULT_HELDOUT_FAMILY)
    assert report.is_clean


def test_training_split_rejects_unknown_family():
    with pytest.raises(KeyError):
        training_split("does_not_exist")
