"""Vocabulary tests — case-insensitive match, canonical casing, load validation."""

from __future__ import annotations

import json

import pytest

from clinical_feature_extractor import config
from clinical_feature_extractor.vocabulary import (
    Vocabulary,
    VocabularyError,
    load_vocabulary,
)


def test_case_insensitive_match_returns_canonical_casing():
    vocab = Vocabulary(["dyspnea", "Elevated LDH", "Evidence for absence of cough"])
    assert vocab.match("Dyspnea") == "dyspnea"
    assert vocab.match("DYSPNEA") == "dyspnea"
    assert vocab.match("elevated ldh") == "Elevated LDH"
    # Surrounding whitespace is ignored.
    assert vocab.match("  dyspnea  ") == "dyspnea"


def test_off_list_feature_is_not_matched():
    vocab = Vocabulary(["dyspnea", "cough"])
    assert vocab.match("chills") is None
    assert "chills" not in vocab
    assert "dyspnea" in vocab


def test_negated_form_matches_verbatim_entry():
    vocab = Vocabulary(["cough", "Evidence for absence of cough"])
    assert vocab.match("evidence for absence of cough") == "Evidence for absence of cough"


def test_first_seen_casing_wins_on_collision():
    vocab = Vocabulary(["Dyspnea", "dyspnea"])
    assert vocab.match("DYSPNEA") == "Dyspnea"  # first-seen spelling


def test_non_string_input_is_no_match():
    vocab = Vocabulary(["dyspnea"])
    assert vocab.match(None) is None  # type: ignore[arg-type]
    assert vocab.match(123) is None  # type: ignore[arg-type]


def test_load_vocabulary_from_file(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({config.VOCABULARY_KEY: ["dyspnea", "Cough"]}),
                    encoding="utf-8")
    vocab = load_vocabulary(path)
    assert len(vocab) == 2
    assert vocab.match("COUGH") == "Cough"


def test_load_vocabulary_missing_key_raises(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"wrong_key": ["dyspnea"]}), encoding="utf-8")
    with pytest.raises(VocabularyError):
        load_vocabulary(path)


def test_load_vocabulary_missing_file_raises(tmp_path):
    with pytest.raises(VocabularyError):
        load_vocabulary(tmp_path / "does_not_exist.json")


# --------------------------------------------------------------------------- #
# Tier 2: formatting-normalized fallback
# --------------------------------------------------------------------------- #
def test_normalized_match_rescues_unbalanced_parenthesis():
    """A model string with a closing paren matches a list entry missing it."""
    vocab = Vocabulary(["Fever (greater than or equal to 38°C from pre value"])
    got = vocab.match("Fever (greater than or equal to 38°C from pre value)")
    assert got == "Fever (greater than or equal to 38°C from pre value"  # list spelling


def test_normalized_match_ignores_extra_whitespace_and_edge_punctuation():
    vocab = Vocabulary(["Generalized flushing"])
    assert vocab.match("  Generalized   flushing.  ") == "Generalized flushing"


def test_hyphen_is_optional_in_normalized_match():
    """A hyphen is optional: hyphenated, spaced-hyphen, and joined forms all match
    the vocabulary spelling (in either direction)."""
    vocab = Vocabulary(["Decrease to 80% of pre-transfusion count"])
    for probe in ("Decrease to 80% of pre-transfusion count",   # exact
                  "Decrease to 80% of pre- transfusion count",  # spaced hyphen
                  "Decrease to 80% of pretransfusion count"):   # no hyphen
        assert vocab.match(probe) == "Decrease to 80% of pre-transfusion count"
    # And the reverse: a joined vocab spelling is found by a hyphenated query.
    joined = Vocabulary(["nonhemolytic reaction"])
    assert joined.match("non-hemolytic reaction") == "nonhemolytic reaction"


def test_hyphen_removal_still_keeps_distinct_numeric_ranges_apart():
    """Dropping hyphens must not merge different numeric ranges."""
    vocab = Vocabulary(["platelet drop 20-80%", "platelet drop 20-30%"])
    assert vocab.match("platelet drop 20-80%") == "platelet drop 20-80%"
    assert vocab.match("platelet drop 20-30%") == "platelet drop 20-30%"


def test_exact_match_still_wins_and_returns_that_spelling():
    vocab = Vocabulary(["Respiratory distress", "respiratory distress (acute)"])
    # Exact (case-insensitive) beats any normalized fallback.
    assert vocab.match("RESPIRATORY DISTRESS") == "Respiratory distress"


def test_normalized_does_not_merge_meaningful_value_differences():
    """Different thresholds must stay distinct — normalization preserves numbers."""
    vocab = Vocabulary(["Oxygen saturation less than 90%",
                        "Oxygen saturation less than 80%"])
    assert vocab.match("oxygen saturation less than 90%") == \
        "Oxygen saturation less than 90%"
    assert vocab.match("oxygen saturation less than 80%") == \
        "Oxygen saturation less than 80%"
    # A third, off-list threshold is not force-matched to either.
    assert vocab.match("oxygen saturation less than 70%") is None


def test_ambiguous_normalized_key_is_not_guessed():
    """If two entries normalize alike, a normalized-only query matches neither."""
    vocab = Vocabulary(["Rash (mild)", "Rash [mild]"])  # both normalize to 'rash mild'
    # Exact still works for each…
    assert vocab.match("rash (mild)") == "Rash (mild)"
    assert vocab.match("rash [mild]") == "Rash [mild]"
    # …but a differently-punctuated form that only hits the ambiguous key is dropped.
    assert vocab.match("rash mild") is None


def test_off_list_still_none_under_normalization():
    vocab = Vocabulary(["dyspnea", "cough"])
    assert vocab.match("splenomegaly (massive)") is None


def test_real_vocabulary_source_loads():
    """The actual read-only dedup JSON loads and is non-trivial."""
    vocab = load_vocabulary()
    assert len(vocab) > 0
    # Sanity: a known entry from the source list.
    assert vocab.match("back pain") == "Back pain"


def test_real_fever_entry_rescued_by_normalization():
    """The real malformed Fever entry (missing ')') matches the balanced model form."""
    vocab = load_vocabulary()
    model_form = ("Fever (greater than or equal to 38°C/100.4°F oral and a change "
                  "of at least 1°C/1.8°F from pre- transfusion value)")
    got = vocab.match(model_form)
    assert got is not None
    assert got.startswith("Fever (")
    assert not got.endswith(")")  # returns the list's (paren-less) spelling


# --------------------------------------------------------------------------- #
# Hyphen-adjacent whitespace (Rule 3) + the "output is verbatim canonical" crux
# --------------------------------------------------------------------------- #
def test_hyphen_space_artifact_matches_both_directions():
    # Vocab has the artifact 'pre- transfusion'; model sends clean 'pre-transfusion'.
    vocab = Vocabulary(["Change from pre- transfusion value"])
    assert vocab.match("Change from pre-transfusion value") == \
        "Change from pre- transfusion value"
    # And the reverse: vocab clean, model with the artifact.
    vocab2 = Vocabulary(["Change from pre-transfusion value"])
    assert vocab2.match("Change from pre- transfusion value") == \
        "Change from pre-transfusion value"


def test_crux_output_is_verbatim_canonical_not_model_string():
    """THE crux: hyphen + unbalanced-paren artifacts differ, yet the WRITTEN value
    is the exact vocabulary string (with the artifacts), never the model's clean one."""
    vocab = load_vocabulary()
    canonical = next(v for v in vocab.items
                     if v.startswith("Fever (") and "pre- transfusion" in v)
    model_clean = ("Fever (greater than or equal to 38°C/100.4°F oral and a change "
                   "of at least 1°C/1.8°F from pre-transfusion value)")  # no space, has )
    assert model_clean != canonical                 # they really do differ
    assert vocab.match(model_clean) == canonical     # matched, and output IS canonical


def test_normalization_is_general_not_hardcoded():
    """A never-before-seen artifact (multiple internal spaces) still matches — proof
    the rules are general, not keyed to any specific entry."""
    vocab = Vocabulary(["multi   space  term"])
    assert vocab.match("multi space term") == "multi   space  term"


def test_nfc_normalization_makes_equivalent_unicode_match():
    import unicodedata
    composed = "Malais" + "\u00e9"           # NFC: single-codepoint accented e
    decomposed = unicodedata.normalize("NFD", composed)  # e + combining accent
    assert composed != decomposed               # genuinely different byte forms
    vocab = Vocabulary([composed])
    assert vocab.match(decomposed) == composed  # compare equal, canonical returned
