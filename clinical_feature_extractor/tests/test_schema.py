"""Tests for the read-only NHSN hierarchical-schema loader.

These assert the workbook loads cleanly and exposes all 181 features (93 positive +
88 negated) 1:1 with the vocabulary — the invariant the Model-3 stage depends on.
They touch the real workbook and the real vocabulary (both read-only), never a model.
"""

from __future__ import annotations

import pytest

from clinical_feature_extractor import config, schema
from clinical_feature_extractor.extractor import NEGATED_FEATURE_PREFIX
from clinical_feature_extractor.schema import DETAILS, GROUPS, TIMING, SchemaError
from clinical_feature_extractor.vocabulary import load_vocabulary


@pytest.fixture(scope="module")
def vocab():
    return load_vocabulary()


@pytest.fixture(scope="module")
def feature_schema(vocab):
    return schema.load_schema(vocabulary=vocab)


def _positive_items(vocab) -> list[str]:
    return [i for i in vocab.items
            if not i.strip().lower().startswith(NEGATED_FEATURE_PREFIX)]


def _negated_items(vocab) -> list[str]:
    return [i for i in vocab.items
            if i.strip().lower().startswith(NEGATED_FEATURE_PREFIX)]


def test_workbook_is_present():
    assert config.SCHEMA_XLSX.exists(), (
        f"schema workbook missing at {config.SCHEMA_XLSX}")


def test_loads_all_181_features(feature_schema, vocab):
    """The schema now covers every vocabulary feature: 93 positive + 88 negated."""
    assert len(feature_schema) == 181
    assert len(feature_schema.features) == 181
    assert len(_positive_items(vocab)) == 93
    assert len(_negated_items(vocab)) == 88


def test_features_map_one_to_one_with_vocabulary(feature_schema, vocab):
    """Every vocabulary item (positive AND negated) has a schema entry, and vice versa."""
    assert len(vocab.items) == 181

    # Every feature looks up by its canonical spelling, exactly as the pipeline emits
    # it — including cosmetic mismatches the normalizer resolves (e.g. the
    # "pre-transfusion" vs "pretransfusion" hyphen-optional case).
    for item in vocab.items:
        assert item in feature_schema, f"feature missing from schema: {item!r}"

    # And every schema feature is a vocabulary item (no extras).
    vocab_keys = {i.strip().lower() for i in vocab.items}
    for name in feature_schema.features:
        assert name.strip().lower() in vocab_keys, (
            f"schema feature is not a vocabulary item: {name!r}")


def test_every_feature_has_both_groups_with_variables(feature_schema, vocab):
    for item in vocab.items:
        for group in GROUPS:
            variables = feature_schema.variables(item, group)
            assert variables, f"{item!r} has no {group} variables"


def test_variables_are_ordered_and_populated(feature_schema, vocab):
    """Each variable carries a non-empty id/definition/response type, sorted by order."""
    for item in vocab.items:
        for group in GROUPS:
            variables = feature_schema.variables(item, group)
            orders = [v.order for v in variables]
            assert orders == sorted(orders), f"{item!r}/{group} not order-sorted"
            for v in variables:
                assert v.variable_id, f"{item!r}/{group} has a blank variable id"
                assert v.definition, f"{v.variable_id!r} has a blank definition"
                assert v.response_type, f"{v.variable_id!r} has a blank response type"


def test_numeric_variables_have_a_units_menu(feature_schema):
    """A numeric variable must expose a units menu (quantity needs a unit)."""
    a_numeric = feature_schema.variables(
        "unexplained clinical illness consistent with infection", TIMING)
    numerics = [v for v in a_numeric if v.response_type == "numeric"]
    assert numerics, "expected at least one numeric timing variable in the fixture"
    for v in numerics:
        assert v.units, f"numeric variable {v.variable_id!r} has no units menu"


def test_allowed_values_are_split_into_a_menu(feature_schema):
    variables = feature_schema.variables(
        "unexplained clinical illness consistent with infection", TIMING)
    first = variables[0]
    assert first.variable_id == "feature_onset_or_observation_datetime"
    assert first.allowed_values == (
        "date/time value or verbatim temporal phrase only",
        "not_documented", "uncertain", "not_applicable")


def test_unknown_feature_returns_empty(feature_schema):
    assert feature_schema.variables("not a real feature", TIMING) == ()
    assert "not a real feature" not in feature_schema


def test_unknown_group_raises(feature_schema):
    with pytest.raises(SchemaError):
        feature_schema.variables(
            "unexplained clinical illness consistent with infection", "Nonsense")
