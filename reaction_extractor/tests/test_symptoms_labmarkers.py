"""Tests for the deterministic symptoms/lab-markers extractor (no Groq)."""

from __future__ import annotations

import json

from reaction_extractor import symptoms_labmarkers_extractor as sle


def _doc(reactions):
    return {"metadata": {"count": len(reactions)}, "reactions": reactions}


def _reaction(page, time_window, signs, labs):
    return {
        "reaction": f"R{page}",
        "definition": "...",
        "features": {
            "time_window": time_window,
            "signs_symptoms": signs,
            "lab_markers": labs,
        },
        "source_page": page,
    }


def test_deduplicates_overlapping_features():
    document = _doc([
        _reaction(11, "within 24 hours", ["Fever", "Chills"], ["Elevated LDH"]),
        _reaction(13, "within 24 hours", ["Fever", "Hypotension"], ["Elevated LDH"]),
        _reaction(14, None, ["Cough"], ["Elevated BNP"]),
    ])
    items = sle.extract_symptoms_labmarkers(document)["items"]

    assert items.count("Fever") == 1
    assert items.count("Elevated LDH") == 1
    assert sorted(items) == sorted(set(items))
    assert set(items) == {
        "Fever", "Chills", "Hypotension", "Cough",
        "Elevated LDH", "Elevated BNP",
    }


def test_time_window_values_never_included():
    document = _doc([
        _reaction(11, "within 24 hours", ["Fever"], ["LDH"]),
        _reaction(13, "6 to 72 hours after transfusion", ["Chills"], ["BNP"]),
    ])
    items = sle.extract_symptoms_labmarkers(document)["items"]
    assert "within 24 hours" not in items
    assert "6 to 72 hours after transfusion" not in items
    assert set(items) == {"Fever", "Chills", "LDH", "BNP"}


def test_signs_symptoms_null_is_skipped():
    document = _doc([_reaction(11, "within 4 hours", None, ["LDH"])])
    result = sle.extract_symptoms_labmarkers(document)
    assert result["items"] == ["LDH"]
    assert result["source_counts"]["signs_symptoms"] == 0


def test_lab_markers_null_is_skipped():
    document = _doc([_reaction(11, "within 4 hours", ["Fever"], None)])
    result = sle.extract_symptoms_labmarkers(document)
    assert result["items"] == ["Fever"]
    assert result["source_counts"]["lab_markers"] == 0


def test_all_signs_before_all_labs():
    document = _doc([
        _reaction(11, "TW1", ["S1"], ["L1"]),
        _reaction(13, "TW2", ["S2"], ["L2"]),
    ])
    items = sle.extract_symptoms_labmarkers(document)["items"]
    assert items == ["S1", "S2", "L1", "L2"]
    # Every signs_symptoms item precedes every lab_markers item.
    assert items.index("S2") < items.index("L1")


def test_dedup_is_case_insensitive_first_casing_wins():
    document = _doc([
        _reaction(11, None, ["Fever"], None),
        _reaction(13, None, ["fever"], None),
    ])
    items = sle.extract_symptoms_labmarkers(document)["items"]
    assert items == ["Fever"]  # first-seen casing preserved, "fever" dropped


def test_cross_field_value_appears_once_in_signs_section():
    # "Hypotension" is a sign in p11 and (contrived) a lab in p13; because signs
    # are collected first, it survives once, in the signs section.
    document = _doc([
        _reaction(11, None, ["Hypotension", "Fever"], ["LDH"]),
        _reaction(13, None, ["Chills"], ["Hypotension", "BNP"]),
    ])
    items = sle.extract_symptoms_labmarkers(document)["items"]
    assert items.count("Hypotension") == 1
    # It sits in the signs block, i.e. before the first pure lab marker "LDH".
    assert items.index("Hypotension") < items.index("LDH")
    assert items == ["Hypotension", "Fever", "Chills", "LDH", "BNP"]


def test_source_counts_are_pre_dedup():
    document = _doc([
        _reaction(11, "TW", ["Fever", "Chills"], ["LDH"]),
        _reaction(13, "TW", ["Fever"], ["LDH", "BNP"]),
    ])
    counts = sle.extract_symptoms_labmarkers(document)["source_counts"]
    assert counts == {"signs_symptoms": 3, "lab_markers": 3}


def test_build_writes_file_with_exact_lowercase_key_and_matching_count(tmp_path):
    src = tmp_path / "reaction_features.json"
    src.write_text(json.dumps(_doc([
        _reaction(11, "within 24 hours", ["Fever"], ["LDH"]),
    ])), encoding="utf-8")
    out = tmp_path / "symptoms_labmarkers.json"

    sle.build_symptoms_labmarkers(input_path=src, output_path=out)
    written = json.loads(out.read_text(encoding="utf-8"))

    # Exact key name — all lowercase; no capitalized variant.
    assert "symptoms_labmarkers" in written
    assert "Symptoms_labmarkers" not in written
    assert list(written.keys()) == ["metadata", "symptoms_labmarkers"]

    total = written["metadata"]["total_unique_items"]
    assert total == len(written["symptoms_labmarkers"])
    assert written["metadata"]["source_filename"] == "reaction_features.json"


def test_build_does_not_modify_source_file(tmp_path):
    src = tmp_path / "reaction_features.json"
    original = json.dumps(_doc([_reaction(11, "TW", ["Fever"], ["LDH"])]))
    src.write_text(original, encoding="utf-8")

    sle.build_symptoms_labmarkers(input_path=src, output_path=tmp_path / "s.json")

    assert src.read_text(encoding="utf-8") == original
