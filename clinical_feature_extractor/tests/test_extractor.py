"""Extractor + pipeline tests with a FAKE Groq client — no network, no spend.

The fake mimics groq's response shape:
    resp.choices[0].message.content   -> the JSON string
    resp.choices[0].finish_reason     -> "stop" | "length"
A behavior list makes each .create() return the next canned response (or raise
the next exception), driving retry/truncation paths deterministically.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from clinical_feature_extractor import config, extractor as extractor_mod
from clinical_feature_extractor.extractor import (
    SCHEMA_KEY,
    ClinicalFeatureExtractor,
    ExtractionError,
    TruncationError,
    dedupe_absence_features,
    merge_schema_group,
    passthrough_features,
    positive_features,
    validate_features,
)
from clinical_feature_extractor.pipeline import (
    Pipeline,
    collect_data_rows,
    find_header_row,
)
from clinical_feature_extractor.schema import (
    DETAILS,
    TIMING,
    FeatureSchema,
    SchemaVariable,
)
from clinical_feature_extractor.vocabulary import Vocabulary


# --------------------------------------------------------------------------- #
# Fixtures / fakes
# --------------------------------------------------------------------------- #
def _resp(content: str, finish: str = "stop"):
    msg = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)])


def _features_json(features):
    return json.dumps({"features": features})


def _time_json(times=None):
    return json.dumps({"times": times or []})


def _time_resp(times=None):
    """A Stage-2 (Model 2) response."""
    return _resp(_time_json(times))


def _treatments_json(treatments=None):
    return json.dumps({"treatments": treatments or []})


def _treatments_resp(treatments=None):
    """A Stage-4 (Model 4) response."""
    return _resp(_treatments_json(treatments))


def _pmh_json(pmh=None):
    return json.dumps({"past_medical_history": pmh or []})


def _pmh_resp(pmh=None):
    """A Model 5 (past medical history) response."""
    return _resp(_pmh_json(pmh))


def _schema_json(variables=None):
    return json.dumps({"variables": variables or []})


def _schema_resp(variables=None):
    """A Stage-3 (Model 3, NHSN schema) group response."""
    return _resp(_schema_json(variables))


class _FakeCompletions:
    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = 0        # every create() call (all stages)
        self.pos = 0          # index into the positional behaviors list
        self.kwargs_log = []

    def create(self, **kwargs):
        self.kwargs_log.append(kwargs)
        self.calls += 1
        b = self.behaviors[min(self.pos, len(self.behaviors) - 1)]
        self.pos += 1
        if isinstance(b, Exception):
            raise b
        return b


class FakeClient:
    def __init__(self, behaviors):
        self.chat = SimpleNamespace(completions=_FakeCompletions(behaviors))


VOCAB = Vocabulary([
    "dyspnea",
    "cough",
    "fever",
    "Elevated LDH",
    "Evidence for absence of cough",
])


# --------------------------------------------------------------------------- #
# Fake NHSN schema (Model-3 input). Injected into the extractor so tests never
# touch the real workbook and the schema stage is deterministic. The DEFAULT is
# empty (no feature has a menu), so the schema stage makes NO model calls unless a
# test opts in with menus — keeping the many unrelated tests to features/time/
# treatments only.
# --------------------------------------------------------------------------- #
def _svar(variable_id, order, response_type, *, allowed=(), units=(),
          notes=None, dependencies=None, definition="a definition"):
    return SchemaVariable(variable_id, order, definition, response_type,
                          tuple(allowed), tuple(units), notes, dependencies)


def make_feature_schema(menus: dict) -> FeatureSchema:
    """Build a FeatureSchema from ``{feature_name: {TIMING: [SchemaVariable], ...}}``."""
    by_feature: dict = {}
    canonical: dict = {}
    for name, groups in menus.items():
        key = name.strip().lower()
        canonical[key] = name
        by_feature[key] = {g: tuple(vs) for g, vs in groups.items()}
    return FeatureSchema(by_feature, canonical)


EMPTY_SCHEMA = make_feature_schema({})


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(extractor_mod.time, "sleep", lambda *_: None)


@pytest.fixture
def validation_on(monkeypatch):
    """Force the Model 1 vocabulary post-check ON, regardless of the config default."""
    monkeypatch.setattr(config, "VALIDATE_FEATURES", True)


@pytest.fixture
def validation_off(monkeypatch):
    """Force the Model 1 vocabulary post-check OFF, regardless of the config default."""
    monkeypatch.setattr(config, "VALIDATE_FEATURES", False)


def make_extractor(behaviors, tmp_path, feature_schema=EMPTY_SCHEMA):
    return ClinicalFeatureExtractor(
        VOCAB, client=FakeClient(behaviors), cache_dir=tmp_path / "cache",
        feature_schema=feature_schema)


def make_workbook(path, texts, *, header_row=1):
    """Write a minimal Sheet1 workbook with the two headers and given texts."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = config.SHEET_NAME
    ws.cell(row=header_row, column=1).value = config.CLINICAL_TEXT_HEADER
    ws.cell(row=header_row, column=2).value = config.EXTRACTED_FEATURES_HEADER
    for i, text in enumerate(texts):
        ws.cell(row=header_row + 1 + i, column=1).value = text
    wb.save(path)
    return path


# --------------------------------------------------------------------------- #
# validate_features (pure, deterministic vocabulary post-check)
# --------------------------------------------------------------------------- #
def test_valid_features_kept():
    raw = [{"feature": "dyspnea", "evidence": "was short of breath"}]
    out = validate_features(raw, VOCAB)
    assert out["features"] == raw
    assert out["dropped"] == []


def test_casing_variant_normalized_to_canonical():
    raw = [{"feature": "Dyspnea", "evidence": "x"},
           {"feature": "elevated ldh", "evidence": "y"}]
    out = validate_features(raw, VOCAB)
    assert [f["feature"] for f in out["features"]] == ["dyspnea", "Elevated LDH"]
    assert out["dropped"] == []


def test_off_list_feature_dropped_and_recorded():
    raw = [{"feature": "dyspnea", "evidence": "a"},
           {"feature": "splenomegaly", "evidence": "b"}]
    out = validate_features(raw, VOCAB)
    assert [f["feature"] for f in out["features"]] == ["dyspnea"]
    assert len(out["dropped"]) == 1
    assert out["dropped"][0]["feature"] == "splenomegaly"


def test_negated_feature_present_in_vocab_kept():
    raw = [{"feature": "Evidence for absence of cough", "evidence": "no cough noted"}]
    out = validate_features(raw, VOCAB)
    assert out["features"][0]["feature"] == "Evidence for absence of cough"
    assert out["dropped"] == []


def test_malformed_item_without_feature_dropped():
    raw = [{"evidence": "orphan"}, "a bare string not in vocab", 42]
    out = validate_features(raw, VOCAB)
    assert out["features"] == []
    assert len(out["dropped"]) == 3


# -- regression: the ~50% drop bug (model returned bare strings) ------------- #
def test_bare_string_feature_matched_and_reconstructed():
    """A flat list of strings (the real model shape) is matched, not dropped."""
    raw = ["Dyspnea", "cough", "Elevated LDH"]
    out = validate_features(raw, VOCAB)
    assert [f["feature"] for f in out["features"]] == ["dyspnea", "cough", "Elevated LDH"]
    assert out["dropped"] == []


def test_bare_string_off_list_dropped_with_value_not_null():
    raw = ["splenomegaly"]
    out = validate_features(raw, VOCAB)
    assert out["features"] == []
    assert out["dropped"][0]["feature"] == "splenomegaly"  # value, not null


def test_evidence_field_preserved_on_kept_object():
    raw = [{"feature": "Dyspnea", "evidence": "was short of breath", "time": "2h"}]
    out = validate_features(raw, VOCAB)
    kept = out["features"][0]
    assert kept["feature"] == "dyspnea"      # normalized
    assert kept["evidence"] == "was short of breath"  # preserved
    assert kept["time"] == "2h"              # preserved


# --------------------------------------------------------------------------- #
# passthrough_features (pure, no-validation path)
# --------------------------------------------------------------------------- #
def test_passthrough_keeps_everything_and_preserves_spelling():
    raw = [{"feature": "Dyspnea", "evidence": "sob"},          # spelling untouched
           {"feature": "totally-off-list", "evidence": "x"},   # off-list kept
           "BareString"]                                       # -> object
    out = passthrough_features(raw)
    assert out["dropped"] == []
    assert out["features"][0] == {"feature": "Dyspnea", "evidence": "sob"}
    assert out["features"][1]["feature"] == "totally-off-list"
    assert out["features"][2] == {"feature": "BareString"}


def test_passthrough_skips_empty_strings_only():
    out = passthrough_features(["", "   ", "real"])
    assert out["features"] == [{"feature": "real"}]
    assert out["dropped"] == []


# --------------------------------------------------------------------------- #
# dedupe_absence_features (pure, deterministic redundancy trim)
# --------------------------------------------------------------------------- #
def test_dedupe_keeps_first_negated_per_shared_evidence():
    feats = [
        {"feature": "Evidence for absence of DIC", "evidence": "hemodynamically stable"},
        {"feature": "Evidence for absence of cough", "evidence": "hemodynamically stable"},
        {"feature": "Evidence for absence of oliguria", "evidence": "hemodynamically stable"},
    ]
    out = dedupe_absence_features(feats)
    assert [f["feature"] for f in out] == ["Evidence for absence of DIC"]  # first kept


def test_dedupe_never_touches_positives_sharing_evidence():
    feats = [
        {"feature": "acute pain", "evidence": "unwell"},
        {"feature": "tachycardia", "evidence": "unwell"},
        {"feature": "dyspnea", "evidence": "unwell"},
    ]
    assert len(dedupe_absence_features(feats)) == 3   # presence > absence: all kept


def test_dedupe_mixed_group_positives_kept_negated_collapsed():
    feats = [
        {"feature": "fever", "evidence": "X"},
        {"feature": "Evidence for absence of cough", "evidence": "X"},
        {"feature": "Evidence for absence of DIC", "evidence": "X"},
        {"feature": "tachycardia", "evidence": "X"},
    ]
    out = dedupe_absence_features(feats)
    assert [f["feature"] for f in out] == ["fever", "Evidence for absence of cough",
                                           "tachycardia"]


def test_dedupe_empty_evidence_negated_all_kept():
    feats = [
        {"feature": "Evidence for absence of cough", "evidence": ""},
        {"feature": "Evidence for absence of DIC"},        # missing evidence entirely
    ]
    assert len(dedupe_absence_features(feats)) == 2        # empties never collapsed


def test_dedupe_distinct_evidence_kept():
    feats = [
        {"feature": "Evidence for absence of cough", "evidence": "no cough"},
        {"feature": "Evidence for absence of DIC", "evidence": "no DIC"},
    ]
    assert len(dedupe_absence_features(feats)) == 2        # different evidence -> all kept


# --------------------------------------------------------------------------- #
# merge_time (pure, deterministic Stage-1 + Stage-2 merge)
# --------------------------------------------------------------------------- #
def test_merge_time_attaches_by_index():
    from clinical_feature_extractor.extractor import merge_time
    feats = [{"feature": "Back pain", "evidence": "back pain"},
             {"feature": "Fever", "evidence": "fever"}]
    td = {"times": [{"index": 2, "time": "within 24h of transfusion"}]}
    out = merge_time(feats, td)
    assert out[0]["time"] == "uncertain"                     # no index-1 time given
    assert out[1]["time"] == "within 24h of transfusion"     # index-2 attached


def test_merge_time_is_authoritative_model2_cannot_alter_features():
    """Even if Stage-2 echoes different feature/evidence text, Stage-1 wins."""
    from clinical_feature_extractor.extractor import merge_time
    feats = [{"feature": "Fever", "evidence": "high temp"}]
    # Model 2 tries to smuggle in a changed feature/evidence + an extra bogus index.
    td = {"times": [{"index": 1, "time": "2h after transfusion",
                     "feature": "HALLUCINATED", "evidence": "WRONG"},
                    {"index": 99, "time": "ignored"}]}
    out = merge_time(feats, td)
    assert len(out) == 1
    assert out[0]["feature"] == "Fever"        # unchanged
    assert out[0]["evidence"] == "high temp"   # unchanged
    assert out[0]["time"] == "2h after transfusion"
    assert "HALLUCINATED" not in json.dumps(out)


def test_merge_time_non_string_or_missing_index_falls_back_to_uncertain():
    from clinical_feature_extractor.extractor import merge_time
    feats = [{"feature": "Fever", "evidence": "e"}]
    td = {"times": [{"index": 1, "time": 123}]}  # bad time type
    out = merge_time(feats, td)
    assert out[0]["time"] == "uncertain"


# --------------------------------------------------------------------------- #
# ClinicalFeatureExtractor.extract (fake client)
# --------------------------------------------------------------------------- #
def test_extract_two_stage_attaches_time(tmp_path):
    ext = make_extractor(
        [_resp(_features_json([{"feature": "dyspnea", "evidence": "sob"}])),
         _time_resp(times=[{"index": 1, "time": "2h after the transfusion"}])], tmp_path)
    out = ext.extract("patient had shortness of breath", use_cache=False)
    assert out["features"][0]["feature"] == "dyspnea"          # Stage-1 feature
    assert out["features"][0]["evidence"] == "sob"             # preserved
    assert out["features"][0]["time"] == "2h after the transfusion"  # Stage-2 merged


def test_extract_uncertain_when_no_time_supplied(tmp_path):
    ext = make_extractor(
        [_resp(_features_json([{"feature": "cough", "evidence": "c"}])),
         _time_resp(times=[])], tmp_path)          # Stage-2 supplies no time
    out = ext.extract("text", use_cache=False)
    assert out["features"][0]["time"] == "uncertain"


def test_extract_off_list_dropped_when_validation_enabled(tmp_path, validation_on):
    ext = make_extractor(
        [_resp(_features_json([{"feature": "not-a-real-feature", "evidence": "x"}])),
         _time_resp()], tmp_path)
    out = ext.extract("text", use_cache=False)
    assert out["features"] == []
    assert out["dropped"][0]["feature"] == "not-a-real-feature"


def test_extract_keeps_off_list_when_validation_disabled(tmp_path, validation_off):
    """With validation off: every Model 1 feature is kept, none dropped."""
    ext = make_extractor(
        [_resp(_features_json([{"feature": "totally-off-list", "evidence": "x"}])),
         _time_resp(), _treatments_resp()], tmp_path)
    out = ext.extract("text", use_cache=False)
    assert [f["feature"] for f in out["features"]] == ["totally-off-list"]
    assert out["dropped"] == []


def test_extract_caches_all_stages_and_reuses(tmp_path):
    ext = make_extractor(
        [_resp(_features_json([{"feature": "cough", "evidence": "x"}])),
         _time_resp(), _treatments_resp()], tmp_path)
    ext.extract("same text", use_cache=True)
    ext.extract("same text", use_cache=True)  # all stages served from cache
    # 3 stages run (features, time, treatments); the schema stage makes no call
    # because the default (empty) schema has no menu for "cough".
    assert ext.client.chat.completions.calls == 3  # 3 first run, 0 second


def test_time_cache_is_prompt_aware(tmp_path):
    """Changing the time prompt re-runs ONLY the time stage; features/treatments stay cached."""
    ext = make_extractor(
        [_resp(_features_json([{"feature": "cough", "evidence": "x"}])),
         _time_resp(), _treatments_resp(), _time_resp()], tmp_path)
    ext.extract("same text", use_cache=True)          # 3 calls (features, time, treatments)
    ext._time_system = ext._time_system + "\nEXTRA GLOBAL-TIME GUIDANCE"
    ext.extract("same text", use_cache=True)          # only the time stage re-runs
    assert ext.client.chat.completions.calls == 4     # 3 + 1 (others from cache)


def test_extract_no_cache_bypasses(tmp_path):
    feat = _resp(_features_json([{"feature": "cough", "evidence": "x"}]))
    ext = make_extractor(
        [feat, _time_resp(), _treatments_resp(),
         feat, _time_resp(), _treatments_resp()], tmp_path)
    ext.extract("same text", use_cache=False)
    ext.extract("same text", use_cache=False)
    assert ext.client.chat.completions.calls == 6  # 3 stages x 2 runs


def test_malformed_json_raises_never_silent_empty(tmp_path):
    ext = make_extractor([_resp("this is not json")], tmp_path)
    with pytest.raises(ExtractionError):
        ext.extract("text", use_cache=False)


def test_missing_features_key_raises(tmp_path):
    ext = make_extractor([_resp(json.dumps({"foo": 1}))], tmp_path)
    with pytest.raises(ExtractionError):
        ext.extract("text", use_cache=False)


def test_time_stage_non_object_raises(tmp_path):
    """A malformed Stage-2 response is not silently accepted."""
    ext = make_extractor(
        [_resp(_features_json([{"feature": "cough", "evidence": "x"}])),
         _resp(json.dumps(["not", "an", "object"]))], tmp_path)
    with pytest.raises(ExtractionError):
        ext.extract("text", use_cache=False)


def test_truncation_raises(tmp_path, no_sleep):
    """A `length` finish is a hard error now -- no token-escalation retry."""
    payload = _features_json([{"feature": "cough", "evidence": "x"}])
    ext = make_extractor([_resp(payload, "length")], tmp_path)
    with pytest.raises(TruncationError):
        ext.extract("text", use_cache=False)


def test_features_call_uses_flat_completion_budget(tmp_path):
    """Every call uses the single flat MAX_COMPLETION_TOKENS budget."""
    ext = make_extractor(
        [_resp(_features_json([{"feature": "cough", "evidence": "x"}])),
         _time_resp()], tmp_path)
    ext.extract("text", use_cache=False)
    assert ext.client.chat.completions.kwargs_log[0]["max_completion_tokens"] == \
        config.MAX_COMPLETION_TOKENS


def test_two_stage_system_prompts_and_models(tmp_path):
    """Stage 1 uses features_system.md; Stage 2 uses time.md; models from config."""
    ext = make_extractor(
        [_resp(_features_json([{"feature": "cough", "evidence": "x"}])),
         _time_resp()], tmp_path)
    ext.extract("text", use_cache=False)
    log = ext.client.chat.completions.kwargs_log
    assert log[0]["messages"][0]["content"] == (
        config.PROMPTS_DIR / "features_system.md").read_text(encoding="utf-8")
    assert log[1]["messages"][0]["content"] == (
        config.PROMPTS_DIR / "time.md").read_text(encoding="utf-8")
    assert log[0]["model"] == config.FEATURES_MODEL
    assert log[1]["model"] == config.TIME_MODEL
    assert log[0]["reasoning_effort"] == config.REASONING_EFFORT


def test_stage1_prompt_forbids_time_stage2_requires_context(tmp_path):
    """Model 1 must not ask for time; Model 2 must require anchoring context."""
    ext = make_extractor(
        [_resp(_features_json([{"feature": "cough", "evidence": "x"}])),
         _time_resp()], tmp_path)
    ext.extract("text", use_cache=False)
    log = ext.client.chat.completions.kwargs_log
    stage1_user = log[0]["messages"][1]["content"]
    stage1_sys = log[0]["messages"][0]["content"]
    stage2_sys = log[1]["messages"][0]["content"]
    # Stage 1: features + evidence only, explicitly no time.
    assert '"evidence"' in stage1_user
    assert "Do NOT include any \"time\"" in stage1_user
    assert "Do NOT extract" in stage1_sys
    # Stage 2: contextual time requirement with the worked example.
    assert "anchoring context" in stage2_sys
    assert "after completion of the apheresis platelet transfusion" in stage2_sys


# --------------------------------------------------------------------------- #
# Model 4 (treatments) — Stage 4
# --------------------------------------------------------------------------- #
def test_positive_features_excludes_negated():
    merged = [
        {"feature": "Fever", "evidence": "temp 39", "time": "2h after transfusion"},
        {"feature": "Evidence for absence of cough", "evidence": "no cough",
         "time": "uncertain"},
    ]
    out = positive_features(merged)
    assert [f["feature"] for f in out] == ["Fever"]   # negated dropped


def test_all_positive_features_forwarded_to_model4(tmp_path, validation_off):
    """The relatedness gate is gone: EVERY positive feature is forwarded to Model 4
    (negated features are still withheld)."""
    ext = make_extractor(
        [_resp(_features_json([{"feature": "fever", "evidence": "a"},
                               {"feature": "dyspnea", "evidence": "b"},
                               {"feature": "Evidence for absence of cough",
                                "evidence": "c"}])),
         _time_resp(), _treatments_resp()], tmp_path)
    ext.extract("note", use_cache=False)
    stage4_sys = ext.client.chat.completions.kwargs_log[-1]["messages"][0]["content"]
    assert "fever" in stage4_sys and "dyspnea" in stage4_sys   # both positives forwarded
    assert "Evidence for absence of cough" not in stage4_sys   # negated withheld


def test_extract_returns_treatments_verbatim_no_validation(tmp_path):
    """Model 4 output is stored exactly as returned — never dropped or reshaped."""
    raw_treatments = [
        {"treatment": "Lasix", "response": "Improvement in O2 saturation",
         "timing": "2 hours after transfusion", "feature_list": True},
        # An item with an unexpected extra key must survive untouched (no validation).
        {"treatment": "Mystery drug", "extra_key": "kept", "feature_list": False},
    ]
    ext = make_extractor(
        [_resp(_features_json([{"feature": "dyspnea", "evidence": "sob"}])),
         _time_resp(),
         _treatments_resp(raw_treatments)], tmp_path)
    out = ext.extract("text", use_cache=False)
    assert out["treatments"] == raw_treatments


def test_treatments_prompt_gets_positive_features_evidence_time_and_model(tmp_path):
    ext = make_extractor(
        [_resp(_features_json([
            {"feature": "dyspnea", "evidence": "short of breath"},
            {"feature": "Evidence for absence of cough", "evidence": "no cough"}])),
         _time_resp(times=[{"index": 1, "time": "2h after transfusion"}]),
         _treatments_resp()], tmp_path)
    ext.extract("patient text here", use_cache=False)
    log = ext.client.chat.completions.kwargs_log
    # With the default (empty) schema no schema call happens, so the calls are:
    # features(0), time(1), treatments(2).
    stage4_sys = log[2]["messages"][0]["content"]
    stage4_user = log[2]["messages"][1]["content"]
    assert log[2]["model"] == config.TREATMENTS_MODEL
    # Positive feature passed WITH its evidence + time (the locators).
    assert "dyspnea" in stage4_sys
    assert "short of breath" in stage4_sys        # evidence
    assert "2h after transfusion" in stage4_sys   # time
    # Negated feature is NOT sent to Model 4.
    assert "Evidence for absence of cough" not in stage4_sys
    # The nested schema object is NOT part of Model 4's input (stage stays unchanged).
    assert '"schema"' not in stage4_sys
    # The marker was filled, not left literally in the prompt.
    assert "<Positive Features List>" not in stage4_sys
    # The clinical note rides in the user message.
    assert "patient text here" in stage4_user


def test_extract_dedupes_redundant_absences_and_reports_count(tmp_path, validation_off):
    """Redundant same-evidence absences are trimmed; positives kept; count reported."""
    raw = [
        {"feature": "fever", "evidence": "temp 39"},
        {"feature": "Evidence for absence of cough", "evidence": "stable"},
        {"feature": "Evidence for absence of DIC", "evidence": "stable"},
    ]
    ext = make_extractor(
        [_resp(_features_json(raw)), _time_resp(), _treatments_resp()], tmp_path)
    out = ext.extract("text", use_cache=False)
    names = [f["feature"] for f in out["features"] if isinstance(f, dict) and "feature" in f]
    assert names == ["fever", "Evidence for absence of cough"]   # 2nd absence dropped
    assert out["absences_deduped"] == 1
    assert [f["feature"] for f in out["absences_removed"]] == ["Evidence for absence of DIC"]


def test_extract_dedup_disabled_keeps_all(tmp_path, validation_off, monkeypatch):
    monkeypatch.setattr(config, "DEDUPE_ABSENCE_FEATURES", False)
    raw = [
        {"feature": "Evidence for absence of cough", "evidence": "stable"},
        {"feature": "Evidence for absence of DIC", "evidence": "stable"},
    ]
    ext = make_extractor(
        [_resp(_features_json(raw)), _time_resp(), _treatments_resp()], tmp_path)
    out = ext.extract("text", use_cache=False)
    names = [f["feature"] for f in out["features"] if isinstance(f, dict) and "feature" in f]
    assert len(names) == 2
    assert out["absences_deduped"] == 0


def test_treatments_stage_non_object_raises(tmp_path):
    """A malformed Stage-4 response is not silently accepted."""
    ext = make_extractor(
        [_resp(_features_json([{"feature": "cough", "evidence": "x"}])),
         _time_resp(),
         _resp(json.dumps(["not", "an", "object"]))], tmp_path)
    with pytest.raises(ExtractionError):
        ext.extract("text", use_cache=False)


# --------------------------------------------------------------------------- #
# Model 5 (past medical history) — INDEPENDENT stage
# --------------------------------------------------------------------------- #
def test_pmh_returns_list_verbatim(tmp_path):
    """extract_past_medical_history returns the model's raw list, unvalidated."""
    raw_pmh = [
        {"condition": "Mild asthma", "detail": "only with infections"},
        {"condition": "Aortic stenosis", "detail": "unavailable"},
    ]
    ext = make_extractor([_pmh_resp(raw_pmh)], tmp_path)
    out = ext.extract_past_medical_history("patient text here", use_cache=False)
    assert out == raw_pmh


def test_pmh_prompt_is_note_only_and_uses_pmh_model(tmp_path):
    """Model 5 gets the note in the user message, no vocabulary, and its own model id."""
    ext = make_extractor([_pmh_resp()], tmp_path)
    ext.extract_past_medical_history("history of osteoporosis and asthma", use_cache=False)
    call = ext.client.chat.completions.kwargs_log[-1]
    system = call["messages"][0]["content"]
    user = call["messages"][1]["content"]
    assert call["model"] == config.PMH_MODEL
    # The note rides in the user message.
    assert "history of osteoporosis and asthma" in user
    # No controlled vocabulary is injected (unlike Stage 1).
    assert "dyspnea" not in system and "Elevated LDH" not in system


def test_pmh_missing_key_yields_empty_list(tmp_path):
    """A JSON object without the past_medical_history key returns [] (never raises)."""
    ext = make_extractor([_resp(json.dumps({"something_else": 1}))], tmp_path)
    assert ext.extract_past_medical_history("t", use_cache=False) == []


def test_pmh_non_object_raises(tmp_path):
    """A malformed (non-object) response is not silently accepted."""
    ext = make_extractor([_resp(json.dumps(["not", "an", "object"]))], tmp_path)
    with pytest.raises(ExtractionError):
        ext.extract_past_medical_history("t", use_cache=False)


def test_pmh_is_cached(tmp_path):
    """Second call for the same note serves from cache (no extra model call)."""
    ext = make_extractor([_pmh_resp([{"condition": "Chronic pain", "detail": "unavailable"}])],
                         tmp_path)
    first = ext.extract_past_medical_history("same note", use_cache=True)
    second = ext.extract_past_medical_history("same note", use_cache=True)
    assert first == second
    assert ext.client.chat.completions.calls == 1  # one call, second from cache


def test_pmh_does_not_run_in_extract(tmp_path):
    """extract() (the four-stage pipeline) never invokes Model 5 — it stays independent."""
    ext = make_extractor(
        [_resp(_features_json([{"feature": "cough", "evidence": "x"}])),
         _time_resp(), _treatments_resp()], tmp_path)
    ext.extract("text", use_cache=False)
    systems = [c["messages"][0]["content"]
               for c in ext.client.chat.completions.kwargs_log]
    # The PMH system prompt is never among the calls extract() makes.
    assert not any("past medical history" in s.lower() for s in systems)


# --------------------------------------------------------------------------- #
# Model 3 (NHSN schema fill-in) — Stage 3
# --------------------------------------------------------------------------- #
# A small fake menu reused across the schema tests: one feature ("fever") with a
# categorical + a numeric Timing variable and a datetime Details variable.
_FEVER_SCHEMA = make_feature_schema({
    "fever": {
        TIMING: [
            _svar("elapsed_from_transfusion_start", 1, "numeric",
                  allowed=["documented", "not_documented"],
                  units=["minutes", "hours"],
                  notes="a timing note", dependencies="dep_a"),
            _svar("transfusion_within_30d", 2, "categorical",
                  allowed=["yes", "no", "uncertain"]),
        ],
        DETAILS: [
            _svar("onset_datetime", 1, "datetime_or_verbatim",
                  allowed=["documented", "not_documented"]),
        ],
    },
})


def test_merge_schema_group_keeps_only_menu_variables_and_reports_unknown():
    menu = _FEVER_SCHEMA.variables("fever", TIMING)
    group_data = {"variables": [
        {"variable_id": "transfusion_within_30d", "value": "yes"},
        {"variable_id": "hallucinated_variable", "value": "whatever"},  # off-menu
    ]}
    entries, unknown = merge_schema_group(menu, group_data)
    assert [e["variable_id"] for e in entries] == ["transfusion_within_30d"]
    assert entries[0]["value"] == "yes"
    assert unknown == ["hallucinated_variable"]              # dropped + reported


def test_merge_schema_group_numeric_value_is_a_real_number_with_unit():
    menu = _FEVER_SCHEMA.variables("fever", TIMING)
    group_data = {"variables": [
        {"variable_id": "elapsed_from_transfusion_start", "value": 6.25,
         "unit": "hours"},
    ]}
    entries, unknown = merge_schema_group(menu, group_data)
    assert unknown == []
    e = entries[0]
    assert e["value"] == 6.25 and isinstance(e["value"], float)  # number, not "6.25"
    assert e["unit"] == "hours"                              # unit alone, no amount
    assert e["notes"] == "a timing note"                     # from SCHEMA, not model
    assert e["dependencies"] == "dep_a"                      # from SCHEMA, not model


def test_merge_schema_group_numeric_string_is_coerced_to_number():
    menu = _FEVER_SCHEMA.variables("fever", TIMING)
    group_data = {"variables": [
        {"variable_id": "elapsed_from_transfusion_start", "value": "6.25",
         "unit": "hours"},
    ]}
    entries, _ = merge_schema_group(menu, group_data)
    assert entries[0]["value"] == 6.25 and isinstance(entries[0]["value"], float)


def test_merge_schema_group_numeric_float_noise_is_normalized_and_ints_stay_ints():
    menu = _FEVER_SCHEMA.variables("fever", TIMING)
    noisy, integral = merge_schema_group(menu, {"variables": [
        {"variable_id": "elapsed_from_transfusion_start", "value": 6.2500001},
    ]})[0], merge_schema_group(menu, {"variables": [
        {"variable_id": "elapsed_from_transfusion_start", "value": 59.0},
    ]})[0]
    assert noisy[0]["value"] == 6.25                          # float noise rounded off
    assert integral[0]["value"] == 59 and isinstance(integral[0]["value"], int)  # 59, not 59.0


def test_merge_schema_group_numeric_categorical_fallback_stays_text():
    menu = _FEVER_SCHEMA.variables("fever", TIMING)
    group_data = {"variables": [
        {"variable_id": "elapsed_from_transfusion_start", "value": "not_applicable"},
    ]}
    entries, _ = merge_schema_group(menu, group_data)
    assert entries[0]["value"] == "not_applicable"           # non-numeric answer kept as text


def test_merge_schema_group_categorical_never_gets_a_unit():
    menu = _FEVER_SCHEMA.variables("fever", TIMING)
    # Even if the model wrongly supplies a unit on a categorical variable, it is dropped.
    group_data = {"variables": [
        {"variable_id": "transfusion_within_30d", "value": "no", "unit": "days"},
    ]}
    entries, _ = merge_schema_group(menu, group_data)
    assert "unit" not in entries[0]


def test_merge_schema_group_preserves_menu_order_and_omits_unanswered():
    menu = _FEVER_SCHEMA.variables("fever", TIMING)
    # Model answers the 2nd menu variable first and skips nothing usable for #1.
    group_data = {"variables": [
        {"variable_id": "transfusion_within_30d", "value": "yes"},
        {"variable_id": "elapsed_from_transfusion_start", "value": "documented",
         "unit": "3 hours"},
    ]}
    entries, _ = merge_schema_group(menu, group_data)
    # Output follows MENU order (elapsed is order 1), not the model's order.
    assert [e["variable_id"] for e in entries] == [
        "elapsed_from_transfusion_start", "transfusion_within_30d"]


def test_merge_schema_group_omits_variable_with_blank_value():
    menu = _FEVER_SCHEMA.variables("fever", TIMING)
    group_data = {"variables": [
        {"variable_id": "transfusion_within_30d", "value": "   "},   # unusable
    ]}
    entries, unknown = merge_schema_group(menu, group_data)
    assert entries == [] and unknown == []


def test_extract_attaches_nested_schema_to_positive_features(tmp_path, validation_off):
    """Each positive feature gets a nested `schema` with its Timing/Details fills."""
    ext = make_extractor(
        [_resp(_features_json([{"feature": "fever", "evidence": "temp 39"}])),
         _time_resp(times=[{"index": 1, "time": "2h after transfusion"}]),
         _schema_resp([{"variable_id": "transfusion_within_30d", "value": "yes"}]),   # Timing
         _schema_resp([{"variable_id": "onset_datetime",
                        "value": "6th May, 2 days after the transfusion"}]),          # Details
         _treatments_resp()], tmp_path, feature_schema=_FEVER_SCHEMA)
    out = ext.extract("note", use_cache=False)
    feat = out["features"][0]
    assert feat["feature"] == "fever"
    assert SCHEMA_KEY in feat
    assert feat[SCHEMA_KEY]["timing"] == [
        {"variable_id": "transfusion_within_30d", "value": "yes"}]
    assert feat[SCHEMA_KEY]["details"] == [
        {"variable_id": "onset_datetime",
         "value": "6th May, 2 days after the transfusion"}]
    assert out["schema_dropped"] == []


def test_extract_negated_feature_without_menu_gets_empty_schema_and_no_call(
        tmp_path, validation_off):
    """A negated feature with no menu gets `schema: {}` and makes no Model-3 call."""
    ext = make_extractor(
        [_resp(_features_json([{"feature": "Evidence for absence of cough",
                                "evidence": "no cough"}])),
         _time_resp(), _treatments_resp()], tmp_path, feature_schema=_FEVER_SCHEMA)
    out = ext.extract("note", use_cache=False)
    neg = out["features"][0]
    assert neg["feature"] == "Evidence for absence of cough"
    assert neg[SCHEMA_KEY] == {}          # menu absent -> empty, but the key is present
    # No schema call was made (fever is the only menu; the absence isn't in it).
    assert ext.client.chat.completions.calls == 3   # features, time, treatments only


# A negated feature's menu: a full Timing group (two variables here) and the single
# generic Details variable every absence shares.
_NEGATED_SCHEMA = make_feature_schema({
    "Evidence for absence of cough": {
        TIMING: [
            _svar("absence_timing_type", 1, "categorical",
                  allowed=["point_observation", "interval", "not_documented"],
                  notes="a timing-type note"),
            _svar("absence_observation_datetime", 2, "datetime_or_verbatim",
                  allowed=["documented", "not_documented"]),
        ],
        DETAILS: [
            _svar("Details", 1, "text",
                  allowed=["documented", "not_documented", "uncertain",
                           "not_applicable"],
                  notes="the only detail note"),
        ],
    },
})


def test_extract_negated_feature_trims_timing_and_stamps_details(tmp_path, validation_off):
    """A negated feature: full Timing call trimmed to its first variable; Details
    stamped deterministically with no Details LLM call."""
    ext = make_extractor(
        [_resp(_features_json([{"feature": "Evidence for absence of cough",
                                "evidence": "no cough"}])),
         _time_resp(),
         # The Timing call returns BOTH variables; only the first (by menu order) is kept.
         _schema_resp([{"variable_id": "absence_observation_datetime",
                        "value": "on admission"},
                       {"variable_id": "absence_timing_type",
                        "value": "point_observation"}]),
         _treatments_resp()], tmp_path, feature_schema=_NEGATED_SCHEMA)
    out = ext.extract("note", use_cache=False)
    neg = out["features"][0]
    # Timing trimmed to the first supported variable (menu order -> absence_timing_type).
    assert neg[SCHEMA_KEY]["timing"] == [
        {"variable_id": "absence_timing_type", "value": "point_observation",
         "notes": "a timing-type note"}]
    # Details stamped deterministically with the feature's verbatim evidence
    # ("no cough"), carrying its schema note -- no Details LLM call.
    assert neg[SCHEMA_KEY]["details"] == [
        {"variable_id": "Details", "value": "no cough",
         "notes": "the only detail note"}]
    assert out["schema_dropped"] == []
    # Exactly one schema call (Timing); Details made NO call.
    # Calls: features(0), time(1), timing(2), treatments(3).
    assert ext.client.chat.completions.calls == 4


def test_extract_negated_feature_without_evidence_omits_details(tmp_path, validation_off):
    """A negated feature with no evidence phrase emits no Details entry (no placeholder)."""
    ext = make_extractor(
        [_resp(_features_json([{"feature": "Evidence for absence of cough",
                                "evidence": ""}])),          # blank evidence
         _time_resp(),
         _schema_resp([{"variable_id": "absence_timing_type",
                        "value": "point_observation"}]),
         _treatments_resp()], tmp_path, feature_schema=_NEGATED_SCHEMA)
    out = ext.extract("note", use_cache=False)
    neg = out["features"][0]
    assert neg[SCHEMA_KEY]["details"] == []                  # nothing stamped, no placeholder
    # Details still made NO LLM call: features(0), time(1), timing(2), treatments(3).
    assert ext.client.chat.completions.calls == 4


def test_extract_feature_without_menu_gets_empty_schema_and_no_call(tmp_path, validation_off):
    """A positive feature with no schema menu gets `schema: {}` and makes no Model-3 call."""
    ext = make_extractor(
        [_resp(_features_json([{"feature": "cough", "evidence": "c"}])),   # not in _FEVER_SCHEMA
         _time_resp(), _treatments_resp()], tmp_path, feature_schema=_FEVER_SCHEMA)
    out = ext.extract("note", use_cache=False)
    assert out["features"][0][SCHEMA_KEY] == {}
    assert ext.client.chat.completions.calls == 3   # features, time, treatments only


def test_extract_reports_schema_dropped_for_off_menu_variable(tmp_path, validation_off):
    """An off-menu variable_id the model returns is dropped and surfaced in schema_dropped."""
    ext = make_extractor(
        [_resp(_features_json([{"feature": "fever", "evidence": "temp 39"}])),
         _time_resp(),
         _schema_resp([{"variable_id": "made_up_var", "value": "x"}]),   # Timing: off-menu
         _schema_resp([]),                                               # Details: empty
         _treatments_resp()], tmp_path, feature_schema=_FEVER_SCHEMA)
    out = ext.extract("note", use_cache=False)
    assert out["features"][0][SCHEMA_KEY]["timing"] == []      # nothing kept
    assert len(out["schema_dropped"]) == 1
    rec = out["schema_dropped"][0]
    assert rec["feature"] == "fever"
    assert rec["group"] == TIMING
    assert rec["variable_id"] == "made_up_var"


def test_schema_stage_uses_its_prompts_and_models(tmp_path, validation_off):
    """The Timing/Details groups use their own prompts and model ids, and the user
    message carries the feature + its variable menu."""
    ext = make_extractor(
        [_resp(_features_json([{"feature": "fever", "evidence": "temp 39"}])),
         _time_resp(),
         _schema_resp(), _schema_resp(), _treatments_resp()],
        tmp_path, feature_schema=_FEVER_SCHEMA)
    ext.extract("note", use_cache=False)
    log = ext.client.chat.completions.kwargs_log
    # Calls: features(0), time(1), schema-timing(2), schema-details(3), treatments(4).
    timing_sys, timing_user = log[2]["messages"][0]["content"], log[2]["messages"][1]["content"]
    details_sys = log[3]["messages"][0]["content"]
    assert log[2]["model"] == config.TIMING_SCHEMA_MODEL
    assert log[3]["model"] == config.DETAILS_SCHEMA_MODEL
    assert timing_sys == (config.PROMPTS_DIR / "schema_timing.md").read_text(encoding="utf-8")
    assert details_sys == (config.PROMPTS_DIR / "schema_details.md").read_text(encoding="utf-8")
    # The Timing user message shows the feature and its menu (ids + allowed values + units).
    assert "fever" in timing_user
    assert "elapsed_from_transfusion_start" in timing_user
    assert "minutes | hours" in timing_user           # the numeric variable's unit menu


# --------------------------------------------------------------------------- #
# Header detection + scalable row collection
# --------------------------------------------------------------------------- #
def test_header_detection_is_not_hardcoded_to_row_1(tmp_path):
    """Header can sit below row 1 (blank rows above) and still anchor detection."""
    import openpyxl

    path = tmp_path / "shifted.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = config.SHEET_NAME
    ws.cell(row=3, column=1).value = config.CLINICAL_TEXT_HEADER
    ws.cell(row=3, column=2).value = config.EXTRACTED_FEATURES_HEADER
    ws.cell(row=4, column=1).value = "first text"
    ws.cell(row=5, column=1).value = "second text"
    wb.save(path)

    wb2 = openpyxl.load_workbook(path)
    ws2 = wb2[config.SHEET_NAME]
    header_row = find_header_row(ws2)
    assert header_row == 3
    rows = collect_data_rows(ws2, header_row)
    assert [r for r, _ in rows] == [4, 5]


# --------------------------------------------------------------------------- #
# Pipeline end-to-end (fake client) — scalability, write-back, logging, CSV
# --------------------------------------------------------------------------- #
def _pipeline_for(texts, behaviors, tmp_path, *, header_row=1):
    xlsx = make_workbook(tmp_path / "in.xlsx", texts, header_row=header_row)
    csv_out = tmp_path / "out" / "extracted.csv"
    ext = make_extractor(behaviors, tmp_path)
    pipe = Pipeline(vocabulary=VOCAB, extractor=ext)
    report = pipe.run(xlsx, csv_out, use_cache=False)
    return report, xlsx, csv_out


@pytest.mark.parametrize("n", [3, 25])
def test_scalability_processes_all_rows_no_hardcoded_count(tmp_path, n):
    texts = [f"clinical text number {i}" for i in range(n)]
    behaviors = [
        _resp(_features_json([{"feature": "cough", "evidence": f"e{i}"}]))
        for i in range(n)
    ]
    report, xlsx, csv_out = _pipeline_for(texts, behaviors, tmp_path)
    assert report["rows_processed"] == n
    assert report["total_features"] == n
    # Column B filled for every data row.
    import openpyxl
    ws = openpyxl.load_workbook(xlsx)[config.SHEET_NAME]
    for i in range(n):
        cell = ws.cell(row=2 + i, column=2).value
        assert json.loads(cell)["features"][0]["feature"] == "cough"
    # CSV has n data rows + 1 header.
    assert len(csv_out.read_text(encoding="utf-8").strip().splitlines()) == n + 1


def test_pipeline_logs_dropped_features_with_row_number(tmp_path, validation_on):
    texts = ["row two text", "row three text"]
    behaviors = [
        _resp(_features_json([{"feature": "dyspnea", "evidence": "a"},
                              {"feature": "off-list-x", "evidence": "b"}])),
        _resp(_features_json([{"feature": "cough", "evidence": "c"}])),
    ]
    report, xlsx, csv_out = _pipeline_for(texts, behaviors, tmp_path)
    assert report["dropped_count"] == 1
    log_path = report["dropped_log"]
    assert log_path.exists()
    assert log_path.name == "in.dropped.log"   # named after the input file (in.xlsx)
    entries = [json.loads(line) for line in
               log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) == 1
    assert entries[0]["feature"] == "off-list-x"
    assert entries[0]["row"] == 2  # first data row


def test_pipeline_writes_absences_log(tmp_path, validation_off):
    """Redundant absences removed by the dedup step are logged per input file."""
    texts = ["note with two same-evidence absences"]
    behaviors = [
        _resp(_features_json([
            {"feature": "fever", "evidence": "febrile"},
            {"feature": "Evidence for absence of cough", "evidence": "stable"},
            {"feature": "Evidence for absence of DIC", "evidence": "stable"},
        ])),
        _time_resp(), _treatments_resp(),
    ]
    report, xlsx, csv_out = _pipeline_for(texts, behaviors, tmp_path)
    assert report["absences_deduped"] == 1
    log = report["absences_log"]
    assert log.exists()
    assert log.name == "in.absences.log"           # named after the input (in.xlsx)
    entries = [json.loads(line) for line in
               log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) == 1
    assert entries[0]["row"] == 2
    assert entries[0]["feature"] == "Evidence for absence of DIC"
    assert entries[0]["evidence"] == "stable"


def test_pipeline_writes_schema_log_and_keeps_rows(tmp_path, validation_off):
    """An off-menu variable Model 3 returned is logged per input file; the run
    completes and the row (with its nested schema) is saved."""
    texts = ["a note with a fever"]
    behaviors = [
        _resp(_features_json([{"feature": "fever", "evidence": "f"}])),
        _time_resp(),
        _schema_resp([{"variable_id": "off_menu_var", "value": "x"}]),   # Timing: off-menu
        _schema_resp([{"variable_id": "onset_datetime",
                       "value": "6th May, day of transfusion"}]),        # Details: valid
        _treatments_resp(),
        _pmh_resp(),
    ]
    xlsx = make_workbook(tmp_path / "in.xlsx", texts)
    csv_out = tmp_path / "out" / "extracted.csv"
    ext = make_extractor(behaviors, tmp_path, feature_schema=_FEVER_SCHEMA)
    report = Pipeline(vocabulary=VOCAB, extractor=ext).run(xlsx, csv_out, use_cache=False)

    assert report["schema_dropped"] == 1
    log = report["schema_log"]
    assert log.exists()
    assert log.name == "in.schema.log"             # named after the input (in.xlsx)
    entries = [json.loads(line) for line in
               log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) == 1
    assert entries[0]["row"] == 2
    assert entries[0]["feature"] == "fever"
    assert entries[0]["variable_id"] == "off_menu_var"
    # The row still produced output: the feature (with its nested schema) is in Column B.
    import openpyxl
    ws = openpyxl.load_workbook(xlsx)[config.SHEET_NAME]
    feats = json.loads(ws.cell(row=2, column=2).value)["features"]
    assert feats[0]["feature"] == "fever"
    assert feats[0]["schema"]["details"][0]["variable_id"] == "onset_datetime"


def test_pipeline_flags_zero_feature_row(tmp_path):
    texts = ["has feature", "has none"]
    behaviors = [
        _resp(_features_json([{"feature": "cough", "evidence": "a"}])),
        _resp(_features_json([])),
    ]
    report, xlsx, csv_out = _pipeline_for(texts, behaviors, tmp_path)
    assert report["zero_feature_rows"] == [3]  # second data row is empty


def test_single_row_processes_only_that_row(tmp_path):
    texts = ["row2", "row3", "row4"]
    behaviors = [
        _resp(_features_json([{"feature": "cough", "evidence": "x"}])),
        _time_resp(),
        _treatments_resp(),
    ]
    xlsx = make_workbook(tmp_path / "in.xlsx", texts)
    ext = make_extractor(behaviors, tmp_path)
    pipe = Pipeline(vocabulary=VOCAB, extractor=ext)
    report = pipe.run(xlsx, tmp_path / "out.csv", use_cache=False, single_row=3)
    assert report["rows_processed"] == 1
    # one row, four model calls: features, time, treatments, PMH (Model 5). The schema
    # stage adds no call because the default (empty) schema has no menu for "cough".
    assert ext.client.chat.completions.calls == 4
    # Only row 3's Columns B and C were written.
    import openpyxl
    ws = openpyxl.load_workbook(xlsx)[config.SHEET_NAME]
    assert ws.cell(row=3, column=2).value is not None
    assert ws.cell(row=3, column=3).value is not None
    assert ws.cell(row=2, column=2).value is None
    assert ws.cell(row=4, column=2).value is None


def test_pipeline_writes_treatments_into_column_c(tmp_path):
    texts = ["only text"]
    behaviors = [
        _resp(_features_json([{"feature": "cough", "evidence": "coughing"}])),
        _time_resp(times=[{"index": 1, "time": "2h after transfusion"}]),
        _treatments_resp([{"treatment": "Diphenhydramine", "response": "Uncertain",
                           "timing": "Before transfusion", "feature_list": True}]),
    ]
    report, xlsx, csv_out = _pipeline_for(texts, behaviors, tmp_path)
    assert report["total_treatments"] == 1
    assert report["absences_deduped"] == 0   # positive-only row: nothing to dedup
    import openpyxl
    ws = openpyxl.load_workbook(xlsx)[config.SHEET_NAME]
    # Column C labelled and filled with the treatments JSON (separate from features).
    assert ws.cell(row=1, column=3).value == config.TREATMENTS_HEADER
    col_c = json.loads(ws.cell(row=2, column=3).value)
    assert col_c["treatments"][0]["treatment"] == "Diphenhydramine"
    # Column B (features) untouched by Model 3.
    col_b = json.loads(ws.cell(row=2, column=2).value)
    assert col_b["features"][0]["feature"] == "cough"
    # The exported CSV carries the third column.
    assert config.TREATMENTS_HEADER in csv_out.read_text(encoding="utf-8").splitlines()[0]


def test_single_row_out_of_range_raises(tmp_path):
    from clinical_feature_extractor.pipeline import PipelineError

    texts = ["row2", "row3"]
    xlsx = make_workbook(tmp_path / "in.xlsx", texts)
    ext = make_extractor([_resp(_features_json([]))], tmp_path)
    pipe = Pipeline(vocabulary=VOCAB, extractor=ext)
    with pytest.raises(PipelineError):
        pipe.run(xlsx, tmp_path / "out.csv", use_cache=False, single_row=99)


def test_vocabulary_source_not_modified_by_a_run(tmp_path):
    """A full run never writes to the read-only vocabulary JSON."""
    before = config.VOCABULARY_JSON.read_bytes()
    texts = ["only text"]
    behaviors = [_resp(_features_json([{"feature": "cough", "evidence": "x"}]))]
    _pipeline_for(texts, behaviors, tmp_path)
    assert config.VOCABULARY_JSON.read_bytes() == before
