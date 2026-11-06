"""Tests for the deterministic extended-features generator (no Groq)."""

from __future__ import annotations

import json

from reaction_extractor import extended_features_generator as efg


def _unique_doc(features):
    return {"metadata": {"total_unique_features": len(features)},
            "Unique_features": features}


def test_three_features_produce_six_items():
    extended = efg.extend_features(["Fever", "Chills", "Back pain"])
    assert len(extended) == 6


def test_interleaving_original_then_negated_pair():
    extended = efg.extend_features(["Fever", "Chills", "Back pain"])
    assert extended == [
        "Fever", "Evidence for absence of fever",
        "Chills", "Evidence for absence of chills",
        "Back pain", "Evidence for absence of back pain",
    ]


def test_original_casing_preserved():
    extended = efg.extend_features(["Fever"])
    assert extended[0] == "Fever"  # not "fever"


def test_negated_version_fully_lowercased():
    extended = efg.extend_features(["Fever"])
    assert extended[1] == "Evidence for absence of fever"
    assert extended[1] != "Evidence for absence of Fever"


def test_prefix_is_exact_no_variations():
    assert efg.ABSENCE_PREFIX == "Evidence for absence of "
    neg = efg.negate("Chills")
    # Exactly one space between "of" and the feature; no leading/trailing drift.
    assert neg == "Evidence for absence of chills"
    assert neg.startswith("Evidence for absence of ")
    assert "  " not in neg


def test_special_characters_lowercased_correctly():
    neg = efg.negate("PaO2/FiO2 less than or equal to 300 mm Hg")
    assert neg == ("Evidence for absence of pao2/fio2 less than or equal to "
                   "300 mm hg")


def test_build_writes_file_with_exact_key_and_double_count(tmp_path):
    src = tmp_path / "unique_features.json"
    src.write_text(json.dumps(_unique_doc(["Fever", "Chills"])),
                   encoding="utf-8")
    out = tmp_path / "extended_features.json"

    efg.build_extended_features(input_path=src, output_path=out)
    written = json.loads(out.read_text(encoding="utf-8"))

    # Exact key name — capital E, capital F.
    assert "Extended_features" in written
    assert "extended_features" not in written
    assert list(written.keys()) == ["metadata", "Extended_features"]

    total = written["metadata"]["total_features"]
    assert total == len(written["Extended_features"]) == 4  # 2 × 2
    assert written["metadata"]["source_filename"] == "unique_features.json"


def test_total_features_is_exactly_twice_input(tmp_path):
    src = tmp_path / "unique_features.json"
    features = [f"f{i}" for i in range(37)]
    src.write_text(json.dumps(_unique_doc(features)), encoding="utf-8")

    report = efg.build_extended_features(
        input_path=src, output_path=tmp_path / "e.json")
    assert report["document"]["metadata"]["total_features"] == 74


def test_source_file_not_modified(tmp_path):
    src = tmp_path / "unique_features.json"
    original = json.dumps(_unique_doc(["Fever", "Chills"]))
    src.write_text(original, encoding="utf-8")

    efg.build_extended_features(input_path=src, output_path=tmp_path / "e.json")

    assert src.read_text(encoding="utf-8") == original
