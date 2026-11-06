"""Tests for the deterministic extended symptoms/lab-markers generator (no Groq)."""

from __future__ import annotations

import json

from reaction_extractor import extended_symptoms_labmarkers_generator as eslg


def _source_doc(items):
    return {"metadata": {"total_unique_items": len(items)},
            "symptoms_labmarkers": items}


def test_three_features_produce_six_items():
    extended = eslg.extend(["Fever", "Chills", "Back pain"])
    assert len(extended) == 6


def test_interleaving_original_then_negated_pair():
    extended = eslg.extend(["Fever", "Chills", "Back pain"])
    assert extended == [
        "Fever", "Evidence for absence of fever",
        "Chills", "Evidence for absence of chills",
        "Back pain", "Evidence for absence of back pain",
    ]


def test_original_casing_preserved():
    extended = eslg.extend(["Elevated LDH"])
    assert extended[0] == "Elevated LDH"  # not "elevated ldh"


def test_negated_version_fully_lowercased():
    extended = eslg.extend(["Elevated LDH"])
    assert extended[1] == "Evidence for absence of elevated ldh"
    assert extended[1] != "Evidence for absence of Elevated LDH"


def test_prefix_is_exact_no_variations():
    assert eslg.ABSENCE_PREFIX == "Evidence for absence of "
    neg = eslg.negate("Chills")
    assert neg == "Evidence for absence of chills"
    assert neg.startswith("Evidence for absence of ")
    # Guard against the classic "Evidence of absence of" typo and double spaces.
    assert not neg.startswith("Evidence of absence of ")
    assert "  " not in neg


def test_special_characters_lowercased_correctly():
    neg = eslg.negate("PaO2/FiO2 less than or equal to 300 mm Hg")
    assert neg == ("Evidence for absence of pao2/fio2 less than or equal to "
                   "300 mm hg")


def test_build_writes_file_with_exact_lowercase_key_and_double_count(tmp_path):
    src = tmp_path / "symptoms_labmarkers.json"
    src.write_text(json.dumps(_source_doc(["Fever", "Chills"])),
                   encoding="utf-8")
    out = tmp_path / "extended_symptoms_labmarkers.json"

    eslg.build_extended_symptoms_labmarkers(input_path=src, output_path=out)
    written = json.loads(out.read_text(encoding="utf-8"))

    # Exact key name — all lowercase.
    assert "extended_symptoms_labmarkers" in written
    assert "Extended_symptoms_labmarkers" not in written
    assert list(written.keys()) == ["metadata", "extended_symptoms_labmarkers"]

    total = written["metadata"]["total_features"]
    assert total == len(written["extended_symptoms_labmarkers"]) == 4  # 2 × 2
    assert written["metadata"]["source_filename"] == "symptoms_labmarkers.json"


def test_total_features_is_exactly_twice_input(tmp_path):
    src = tmp_path / "symptoms_labmarkers.json"
    items = [f"f{i}" for i in range(93)]
    src.write_text(json.dumps(_source_doc(items)), encoding="utf-8")

    report = eslg.build_extended_symptoms_labmarkers(
        input_path=src, output_path=tmp_path / "e.json")
    assert report["document"]["metadata"]["total_features"] == 186


def test_source_file_not_modified(tmp_path):
    src = tmp_path / "symptoms_labmarkers.json"
    original = json.dumps(_source_doc(["Fever", "Chills"]))
    src.write_text(original, encoding="utf-8")

    eslg.build_extended_symptoms_labmarkers(
        input_path=src, output_path=tmp_path / "e.json")

    assert src.read_text(encoding="utf-8") == original
