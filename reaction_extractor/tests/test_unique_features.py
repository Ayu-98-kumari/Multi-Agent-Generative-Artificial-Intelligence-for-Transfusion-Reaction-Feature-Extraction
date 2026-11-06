"""Tests for the deterministic unique-features extractor (no Groq)."""

from __future__ import annotations

import json

from reaction_extractor import unique_features_extractor as ufe


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
    result = ufe.extract_unique_features(document)
    unique = result["unique"]

    assert unique.count("Fever") == 1
    assert unique.count("within 24 hours") == 1
    assert unique.count("Elevated LDH") == 1
    # Every distinct value present exactly once.
    assert sorted(unique) == sorted(set(unique))
    assert set(unique) == {
        "within 24 hours", "Fever", "Chills", "Hypotension", "Cough",
        "Elevated LDH", "Elevated BNP",
    }


def test_time_window_null_is_skipped():
    document = _doc([_reaction(11, None, ["Fever"], ["LDH"])])
    unique = ufe.extract_unique_features(document)["unique"]
    assert "Fever" in unique and "LDH" in unique
    assert all(x not in ("null", "None", None) for x in unique)


def test_signs_symptoms_null_is_skipped():
    document = _doc([_reaction(11, "within 4 hours", None, ["LDH"])])
    result = ufe.extract_unique_features(document)
    assert result["unique"] == ["within 4 hours", "LDH"]
    assert result["source_counts"]["signs_symptoms"] == 0


def test_lab_markers_null_is_skipped():
    document = _doc([_reaction(11, "within 4 hours", ["Fever"], None)])
    result = ufe.extract_unique_features(document)
    assert result["unique"] == ["within 4 hours", "Fever"]
    assert result["source_counts"]["lab_markers"] == 0


def test_dedup_is_case_insensitive_first_casing_wins():
    document = _doc([
        _reaction(11, None, ["Fever"], None),
        _reaction(13, None, ["fever"], None),
    ])
    unique = ufe.extract_unique_features(document)["unique"]
    assert unique == ["Fever"]  # first-seen casing preserved, "fever" dropped


def test_ordering_time_window_then_signs_then_labs():
    document = _doc([
        _reaction(11, "TW1", ["S1"], ["L1"]),
        _reaction(13, "TW2", ["S2"], ["L2"]),
    ])
    unique = ufe.extract_unique_features(document)["unique"]
    assert unique == ["TW1", "TW2", "S1", "S2", "L1", "L2"]
    # time_window block entirely precedes signs block, which precedes labs.
    assert unique.index("TW2") < unique.index("S1")
    assert unique.index("S2") < unique.index("L1")


def test_source_counts_are_pre_dedup():
    document = _doc([
        _reaction(11, "TW", ["Fever", "Chills"], ["LDH"]),
        _reaction(13, "TW", ["Fever"], ["LDH", "BNP"]),
    ])
    counts = ufe.extract_unique_features(document)["source_counts"]
    assert counts == {"time_window": 2, "signs_symptoms": 3, "lab_markers": 3}


def test_build_writes_file_with_exact_key_and_matching_count(tmp_path):
    src = tmp_path / "reaction_features.json"
    src.write_text(json.dumps(_doc([
        _reaction(11, "within 24 hours", ["Fever"], ["LDH"]),
    ])), encoding="utf-8")
    out = tmp_path / "unique_features.json"

    report = ufe.build_unique_features(input_path=src, output_path=out)
    written = json.loads(out.read_text(encoding="utf-8"))

    # Exact key name — capital U, capital F; no lowercase variant.
    assert "Unique_features" in written
    assert "unique_features" not in written
    assert list(written.keys()) == ["metadata", "Unique_features"]

    total = written["metadata"]["total_unique_features"]
    assert total == len(written["Unique_features"])
    assert written["metadata"]["source_filename"] == "reaction_features.json"
    assert report["document"] is not None


def test_build_does_not_modify_source_file(tmp_path):
    src = tmp_path / "reaction_features.json"
    original = json.dumps(_doc([_reaction(11, "TW", ["Fever"], ["LDH"])]))
    src.write_text(original, encoding="utf-8")

    ufe.build_unique_features(input_path=src, output_path=tmp_path / "u.json")

    assert src.read_text(encoding="utf-8") == original
