"""Features-step tests with a FAKE Groq client — no network, no spend.

The fake mimics groq's response shape:
    resp.choices[0].message.content   -> the JSON string
    resp.choices[0].finish_reason     -> "stop" | "length"
A behavior list makes each .create() return the next canned response (or raise
the next exception), driving retry/truncation paths deterministically.

Column 3 is now a structured object with exactly three keys:
    {"time_window": str|null, "signs_symptoms": list|null, "lab_markers": list|null}
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from reaction_extractor import config, features
from reaction_extractor.features import (
    FeatureExtractor,
    TruncationError,
    _split_shared_prefix_list,
    _split_shared_suffix_pair,
    _split_symptom_item,
    normalize_feature_list,
    normalize_output_file,
    split_signs_symptoms,
    split_slash_terms,
)
from reaction_extractor.pipeline import Pipeline
from reaction_extractor.extractor import ReactionRecord


# --------------------------------------------------------------------------- #
# Fake client
# --------------------------------------------------------------------------- #
def _resp(content: str, finish: str = "stop"):
    msg = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)])


def _features_json(*, time_window=None, signs_symptoms=None, lab_markers=None):
    return json.dumps({"time_window": time_window,
                       "signs_symptoms": signs_symptoms,
                       "lab_markers": lab_markers})


class _FakeCompletions:
    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = 0
        self.kwargs_log = []

    def create(self, **kwargs):
        self.kwargs = kwargs
        self.kwargs_log.append(kwargs)
        b = self.behaviors[min(self.calls, len(self.behaviors) - 1)]
        self.calls += 1
        if isinstance(b, Exception):
            raise b
        return b


class FakeClient:
    def __init__(self, behaviors):
        self.chat = SimpleNamespace(completions=_FakeCompletions(behaviors))


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(features.time, "sleep", lambda *_: None)


@pytest.fixture
def tmp_cache(tmp_path):
    return tmp_path / "cache"


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_parse_features_normalizes_object():
    out = FeatureExtractor._parse_features(_features_json(
        time_window="within 24 hours",
        signs_symptoms=["fever", "hypotension"],
        lab_markers=["elevated LDH"]))
    assert out == {"time_window": "within 24 hours",
                   "signs_symptoms": ["fever", "hypotension"],
                   "lab_markers": ["elevated LDH"]}


def test_parse_features_filters_empty_list_items():
    out = FeatureExtractor._parse_features(_features_json(
        signs_symptoms=["fever", "", "  ", "chills"]))
    assert out["signs_symptoms"] == ["fever", "chills"]


def test_parse_features_empty_list_becomes_null():
    out = FeatureExtractor._parse_features(_features_json(lab_markers=[]))
    assert out["lab_markers"] is None


def test_parse_features_blank_time_window_becomes_null():
    out = FeatureExtractor._parse_features(_features_json(time_window="   "))
    assert out["time_window"] is None


def test_parse_features_strips_code_fence():
    body = "```json\n" + _features_json(time_window="within 12 hours") + "\n```"
    assert FeatureExtractor._parse_features(body)["time_window"] == "within 12 hours"


def test_parse_features_rejects_no_known_keys():
    with pytest.raises(features.FeatureError):
        FeatureExtractor._parse_features('{"nope": 1}')


# --------------------------------------------------------------------------- #
# "/" term-splitting (real cases from the actual output)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("item, expected", [
    ("Back/flank pain", ["Back pain", "Flank pain"]),   # trailing context shared
    ("Chills/rigors", ["Chills", "Rigors"]),            # standalone, capitalized
    ("Oliguria/anuria", ["Oliguria", "Anuria"]),
])
def test_split_slash_terms_splits_independent_terms(item, expected):
    assert split_slash_terms(item) == expected


@pytest.mark.parametrize("item", [
    "PaO2/FiO2 less than or equal to 300 mm Hg",        # medical ratio (digits)
    "Pain and/or oozing at IV site",                    # logical connector
    "Plasma discoloration c/w hemolysis",               # abbreviation
    "Fever (greater than or equal to 38°C/100.4°F oral",  # unit equivalence in parens
])
def test_split_slash_terms_protects_intact_items(item):
    assert split_slash_terms(item) == [item]


def test_normalize_feature_list_passes_through_without_slash():
    items = ["fever", "hypotension", "back pain"]
    assert normalize_feature_list(items) == items


def test_normalize_feature_list_expands_in_order():
    items = ["fever", "Chills/rigors", "Back/flank pain"]
    assert normalize_feature_list(items) == \
        ["fever", "Chills", "Rigors", "Back pain", "Flank pain"]


def test_normalize_feature_list_handles_none():
    assert normalize_feature_list(None) is None


# --------------------------------------------------------------------------- #
# File-level post-processing: normalize_output_file (deterministic, no Groq)
# --------------------------------------------------------------------------- #
_FNHTR_FEVER = ("Fever (greater than or equal to 38°C/100.4°F oral and a change "
                "of at least 1°C/1.8°F from pre- transfusion value")


def _sample_document():
    return {
        "metadata": {"count": 2},
        "reactions": [
            {
                "reaction": "Acute hemolytic transfusion reaction (AHTR)",
                "definition": "verbatim — Back/flank pain stays in the definition",
                "features": {
                    "time_window": "during, or within 24 hours of cessation of transfusion",
                    "signs_symptoms": [
                        "Back/flank pain", "Chills/rigors", "Oliguria/anuria",
                        "Pain and/or oozing at IV site", "Hypotension"],
                    "lab_markers": [
                        "Plasma discoloration c/w hemolysis",
                        "Positive direct antiglobulin test (DAT) for anti-IgG or anti-C3"],
                },
                "source_page": 13,
            },
            {
                "reaction": "Febrile non-hemolytic transfusion reaction (FNHTR)",
                "definition": "d",
                "features": {
                    "time_window": "during or within 4 hours of cessation of transfusion",
                    "signs_symptoms": [_FNHTR_FEVER, "Chills/rigors"],
                    "lab_markers": None,
                },
                "source_page": 27,
            },
        ],
    }


def _write_doc(tmp_path, doc):
    path = tmp_path / "reaction_features.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_normalize_output_file_splits_confirmed_cases(tmp_path):
    path = _write_doc(tmp_path, _sample_document())
    doc = normalize_output_file(path)
    ahtr = doc["reactions"][0]["features"]
    assert ahtr["signs_symptoms"] == [
        "Back pain", "Flank pain", "Chills", "Rigors", "Oliguria", "Anuria",
        "Pain and/or oozing at IV site", "Hypotension"]
    # Protected lab markers unchanged.
    assert ahtr["lab_markers"] == [
        "Plasma discoloration c/w hemolysis",
        "Positive direct antiglobulin test (DAT) for anti-IgG or anti-C3"]
    # File overwritten in place.
    assert json.loads(path.read_text())["reactions"][0]["features"]["signs_symptoms"][0] \
        == "Back pain"


def test_normalize_output_file_no_cross_item_merge_on_unclosed_paren(tmp_path):
    # FNHTR's "Fever (…" leaves a "(" unclosed; the following Chills/rigors must
    # still split cleanly and must NOT be merged into the Fever item.
    path = _write_doc(tmp_path, _sample_document())
    doc = normalize_output_file(path)
    assert doc["reactions"][1]["features"]["signs_symptoms"] == \
        [_FNHTR_FEVER, "Chills", "Rigors"]


def test_normalize_output_file_leaves_other_fields_untouched(tmp_path):
    doc0 = _sample_document()
    path = _write_doc(tmp_path, doc0)
    doc = normalize_output_file(path)
    for i in (0, 1):
        got, want = doc["reactions"][i], doc0["reactions"][i]
        assert got["reaction"] == want["reaction"]
        assert got["definition"] == want["definition"]
        assert got["source_page"] == want["source_page"]
        assert got["features"]["time_window"] == want["features"]["time_window"]


def test_normalize_output_file_passes_through_lists_without_slash(tmp_path):
    doc0 = {"reactions": [{
        "reaction": "X", "definition": "d", "source_page": 1,
        "features": {"time_window": None,
                     "signs_symptoms": ["fever", "chills", "hypotension"],
                     "lab_markers": None}}]}
    path = _write_doc(tmp_path, doc0)
    doc = normalize_output_file(path)
    feats = doc["reactions"][0]["features"]
    assert feats["signs_symptoms"] == ["fever", "chills", "hypotension"]
    assert feats["lab_markers"] is None


# --------------------------------------------------------------------------- #
# Compound signs_symptoms splitting — Pattern 1, Pattern 2, and guards
# --------------------------------------------------------------------------- #
# Pattern 1 — shared prefix with trailing comma/and list.
def test_pattern1_shared_prefix_list():
    assert _split_shared_prefix_list("Edema of lips, tongue and uvula") == \
        ["Edema of lips", "Edema of tongue", "Edema of uvula"]


def test_pattern1_generalizes_to_new_prefix():
    assert _split_shared_prefix_list("Pain in arms, legs and back") == \
        ["Pain in arms", "Pain in legs", "Pain in back"]


def test_pattern1_protected_threshold_phrase_unchanged():
    item = "Fever (greater than or equal to 38°C and a change of 1°C)"
    assert _split_shared_prefix_list(item) is None
    assert _split_symptom_item(item) == [item]


# Pattern 2 — two findings joined by "and" sharing a prepositional suffix.
def test_pattern2_shared_suffix_pair():
    assert _split_shared_suffix_pair("Erythema and edema of the periorbital area") == \
        ["Erythema of the periorbital area", "Edema of the periorbital area"]


def test_pattern2_generalizes_to_new_pair():
    assert _split_shared_suffix_pair("Swelling and redness of the ankle") == \
        ["Swelling of the ankle", "Redness of the ankle"]


def test_pattern2_no_short_leading_nouns_unchanged():
    item = "Radiographic evidence of bilateral infiltrates"
    assert _split_shared_suffix_pair(item) is None
    assert _split_symptom_item(item) == [item]


def test_pattern2_no_and_unchanged():
    item = "Acute or worsening pulmonary edema"
    assert _split_shared_suffix_pair(item) is None
    assert _split_symptom_item(item) == [item]


# Guards — shared protections both patterns honor.
def test_guard_number_unchanged():
    item = "Drop in systolic BP of 30 mmHg and 25% from baseline"
    assert _split_symptom_item(item) == [item]


def test_guard_and_or_unchanged():
    item = "Pain and/or oozing at IV site"
    assert _split_symptom_item(item) == [item]


def test_guard_long_phrase_on_either_side_unchanged():
    item = "Evidence of acute respiratory distress and worsening pulmonary edema signs"
    assert _split_symptom_item(item) == [item]


def test_guard_no_and_or_comma_unchanged():
    assert _split_symptom_item("Generalized flushing") == ["Generalized flushing"]


# File-level: splits signs_symptoms only, leaves everything else untouched.
def _allergic_document():
    return {
        "metadata": {"count": 2},
        "reactions": [
            {
                "reaction": "Allergic reaction",
                "definition": "verbatim allergic definition — untouched",
                "features": {
                    "time_window": "during or within 4 hours of cessation of transfusion",
                    "signs_symptoms": [
                        "Conjunctival edema",
                        "Edema of lips, tongue and uvula",
                        "Erythema and edema of the periorbital area",
                        "Generalized flushing", "Hypotension",
                        "Localized angioedema", "Maculopapular rash",
                        "Pruritus (itching)", "Respiratory distress",
                        "bronchospasm", "Urticaria (hives)"],
                    # lab_markers deliberately carries a splittable-looking string
                    # to prove it is never processed.
                    "lab_markers": ["Erythema and edema of the periorbital area"],
                },
                "source_page": 24,
            },
            {
                "reaction": "Other reaction",
                "definition": "d",
                "features": {
                    "time_window": None,
                    "signs_symptoms": ["thrombocytopenia"],
                    "lab_markers": None,
                },
                "source_page": 29,
            },
        ],
    }


def test_split_signs_symptoms_produces_expected_allergic_list(tmp_path):
    path = _write_doc(tmp_path, _allergic_document())
    doc = split_signs_symptoms(path)
    assert doc["reactions"][0]["features"]["signs_symptoms"] == [
        "Conjunctival edema",
        "Edema of lips", "Edema of tongue", "Edema of uvula",
        "Erythema of the periorbital area", "Edema of the periorbital area",
        "Generalized flushing", "Hypotension",
        "Localized angioedema", "Maculopapular rash",
        "Pruritus (itching)", "Respiratory distress",
        "bronchospasm", "Urticaria (hives)"]
    # Overwritten in place.
    on_disk = json.loads(path.read_text())
    assert "Edema of tongue" in on_disk["reactions"][0]["features"]["signs_symptoms"]


def test_split_signs_symptoms_leaves_other_fields_and_reactions_untouched(tmp_path):
    doc0 = _allergic_document()
    path = _write_doc(tmp_path, doc0)
    doc = split_signs_symptoms(path)
    allergic = doc["reactions"][0]["features"]
    # lab_markers never processed even though it holds a splittable-looking string.
    assert allergic["lab_markers"] == ["Erythema and edema of the periorbital area"]
    assert allergic["time_window"] == doc0["reactions"][0]["features"]["time_window"]
    assert doc["reactions"][0]["definition"] == doc0["reactions"][0]["definition"]
    # The other reaction is byte-identical.
    assert doc["reactions"][1] == doc0["reactions"][1]


# --------------------------------------------------------------------------- #
# Faithfulness — prompt content + parenthetical rejoin + time-window flag
# --------------------------------------------------------------------------- #
def test_prompt_contains_all_five_faithfulness_rules():
    prompt = (config.PROMPTS_DIR / "features.md").read_text(encoding="utf-8")
    assert "Faithfulness Rules" in prompt
    for n in range(1, 6):
        assert f"Rule {n} —" in prompt, f"missing Rule {n}"
    # Issue 1 & 2 are prompt-only; the exemplars must be present verbatim.
    assert "greater than or equal to 38°C/100.4°F oral" in prompt   # Fever threshold
    assert "age-group variants" in prompt                           # HTR age variants


def test_rejoin_orphaned_parenthetical_fragment():
    # Model split "Pruritus (itching)" into two items -> rejoined into one.
    assert normalize_feature_list(["Pruritus", "(itching)"]) == ["Pruritus (itching)"]


def test_rejoin_orphaned_semicolon_fragment():
    assert normalize_feature_list(["Respiratory distress", "; bronchospasm"]) == \
        ["Respiratory distress; bronchospasm"]


def test_rejoin_closes_open_paren_split_across_items():
    # A long parenthetical split mid-way is stitched back together.
    assert normalize_feature_list(["Fever (>= 38°C", "and rising)"]) == \
        ["Fever (>= 38°C and rising)"]


def test_rejoin_leaves_clean_items_untouched():
    assert normalize_feature_list(["Pruritus (itching)", "Urticaria (hives)"]) == \
        ["Pruritus (itching)", "Urticaria (hives)"]


def test_validation_flags_null_time_window_when_definition_states_one(tmp_cache):
    # Issue 4: model returned null but the definition clearly states a window.
    client = FakeClient([_resp(_features_json(lab_markers=["new alloantibody"]))])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    definition = ("new alloantibody identified between 24 hours and 28 days "
                  "after cessation of transfusion")
    entries = Pipeline(fx).build_entries([ReactionRecord("DSTR", definition, 26)])
    report = Pipeline.validate(entries, single=True)
    assert any("time_window null but definition mentions a time window" in f
               for f in report.flags)


def test_validation_no_time_window_flag_when_definition_has_none(tmp_cache):
    client = FakeClient([_resp(_features_json(signs_symptoms=["fever"]))])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    entries = Pipeline(fx).build_entries([ReactionRecord("TTI", "fever and chills", 11)])
    report = Pipeline.validate(entries, single=True)
    assert not any("time_window null" in f for f in report.flags)


# --------------------------------------------------------------------------- #
# Call path: success / cache / retry / truncation
# --------------------------------------------------------------------------- #
def test_features_for_success_and_caches(tmp_cache):
    client = FakeClient([_resp(_features_json(
        time_window="within 24 hours", signs_symptoms=["back pain"]))])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    feats = fx.features_for("AHTR", "Definitive: within 24 hours ...")
    assert feats == {"time_window": "within 24 hours",
                     "signs_symptoms": ["back pain"], "lab_markers": None}
    # Cache file written, stamped with the current schema version.
    cached = json.loads((tmp_cache / "ahtr.json").read_text())
    assert cached["schema_version"] == config.FEATURES_SCHEMA_VERSION
    assert cached["features"]["time_window"] == "within 24 hours"


def test_features_for_normalizes_slash_terms(tmp_cache):
    # The model returns a combined "/" item; the safety net splits it before
    # caching/return.
    client = FakeClient([_resp(_features_json(
        signs_symptoms=["fever", "Chills/rigors", "Back/flank pain"]))])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    feats = fx.features_for("AHTR", "def")
    assert feats["signs_symptoms"] == \
        ["fever", "Chills", "Rigors", "Back pain", "Flank pain"]
    cached = json.loads((tmp_cache / "ahtr.json").read_text())
    assert cached["features"]["signs_symptoms"][1:3] == ["Chills", "Rigors"]


def test_cache_hit_skips_model_call(tmp_cache):
    client = FakeClient([_resp(_features_json(signs_symptoms=["one"]))])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    fx.features_for("DHTR", "def")            # populates cache, 1 call
    fx.features_for("DHTR", "def")            # served from cache, no new call
    assert client.chat.completions.calls == 1


def test_stale_version_cache_is_ignored(tmp_cache):
    # A pre-existing cache file from the retired flat-list shape must not be reused.
    tmp_cache.mkdir(parents=True)
    (tmp_cache / "dhtr.json").write_text(json.dumps(
        {"schema_version": 1, "reaction": "DHTR", "features": ["old", "list"]}))
    client = FakeClient([_resp(_features_json(time_window="within 24 hours"))])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    feats = fx.features_for("DHTR", "def")
    assert client.chat.completions.calls == 1        # recomputed, not served
    assert feats["time_window"] == "within 24 hours"


def test_no_cache_bypasses_cache(tmp_cache):
    client = FakeClient([_resp(_features_json(time_window="a")),
                         _resp(_features_json(time_window="b"))])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    fx.features_for("DHTR", "def")
    feats = fx.features_for("DHTR", "def", use_cache=False)
    assert client.chat.completions.calls == 2
    assert feats["time_window"] == "b"


def test_reasoning_effort_passed(tmp_cache):
    client = FakeClient([_resp(_features_json(time_window="x"))])
    FeatureExtractor(client=client, cache_dir=tmp_cache).features_for("R", "d")
    assert client.chat.completions.kwargs["reasoning_effort"] == config.REASONING_EFFORT
    assert client.chat.completions.kwargs["response_format"] == {"type": "json_object"}


def test_transient_error_retried(no_sleep, tmp_cache, monkeypatch):
    class _Transient(Exception):
        pass
    monkeypatch.setattr(features, "_transient_exceptions", lambda: (_Transient,))
    client = FakeClient([_Transient("429"),
                         _resp(_features_json(time_window="ok"))])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    assert fx.features_for("R", "d")["time_window"] == "ok"
    assert client.chat.completions.calls == 2


def test_truncation_retries_then_succeeds(no_sleep, tmp_cache):
    client = FakeClient([_resp(_features_json(time_window="partial"), finish="length"),
                         _resp(_features_json(time_window="full"))])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    feats = fx.features_for("R", "d")
    assert feats["time_window"] == "full"
    assert client.chat.completions.calls == 2
    # Retry used the elevated budget.
    assert client.chat.completions.kwargs["max_completion_tokens"] == \
        config.MAX_COMPLETION_TOKENS_RETRY


def test_truncation_twice_is_hard_failure(no_sleep, tmp_cache):
    client = FakeClient([_resp(_features_json(time_window="p"), finish="length"),
                         _resp(_features_json(time_window="p"), finish="length")])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    with pytest.raises(TruncationError):
        fx.features_for("R", "d")


def test_json_validate_failed_retried_then_surfaced(no_sleep, tmp_cache):
    import groq
    import httpx
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
    err = groq.BadRequestError("bad", response=resp,
                               body={"error": {"code": "json_validate_failed"}})
    # Fails MAX_RETRIES times -> FeatureError.
    client = FakeClient([err])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    with pytest.raises(features.FeatureError):
        fx.features_for("R", "d")
    assert client.chat.completions.calls == config.MAX_RETRIES


# --------------------------------------------------------------------------- #
# Pipeline wiring (mocked extractor) — assembly, --single, validation
# --------------------------------------------------------------------------- #
def _records():
    return [
        ReactionRecord("TTI", "def tti", 11),
        ReactionRecord("AHTR", "def ahtr", 13),
    ]


def test_pipeline_assembles_entries(tmp_cache):
    client = FakeClient([_resp(_features_json(
        time_window="within 24 hours", signs_symptoms=["fever"]))])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    entries = Pipeline(fx).build_entries(_records())
    assert [e.reaction for e in entries] == ["TTI", "AHTR"]
    assert entries[0].features["time_window"] == "within 24 hours"
    assert entries[0].features["signs_symptoms"] == ["fever"]


def test_pipeline_single_selects_one_by_page(tmp_cache):
    client = FakeClient([_resp(_features_json(time_window="x"))])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    entries = Pipeline(fx).build_entries(_records(), only="13")
    assert [e.source_page for e in entries] == [13]
    assert client.chat.completions.calls == 1


def test_pipeline_single_selects_one_by_name(tmp_cache):
    client = FakeClient([_resp(_features_json(time_window="x"))])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    entries = Pipeline(fx).build_entries(_records(), only="tti")
    assert [e.reaction for e in entries] == ["TTI"]


def test_dstr_no_signs_symptoms_is_null(tmp_cache):
    # A purely serologic reaction (e.g. DSTR): time window + labs, no symptoms.
    client = FakeClient([_resp(_features_json(
        time_window="between 24 hours and 28 days",
        lab_markers=["newly detected antibody"]))])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    feats = fx.features_for("DSTR", "def")
    assert feats["signs_symptoms"] is None
    assert feats["lab_markers"] == ["newly detected antibody"]


def test_tad_no_lab_markers_is_null(tmp_cache):
    # A reaction stating a time window + symptom but no labs (e.g. TAD).
    client = FakeClient([_resp(_features_json(
        time_window="within 24 hours", signs_symptoms=["respiratory distress"]))])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    feats = fx.features_for("TAD", "def")
    assert feats["lab_markers"] is None


def test_validation_flags_all_null_features(tmp_cache):
    client = FakeClient([_resp(_features_json())])   # all three null
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    entries = Pipeline(fx).build_entries([ReactionRecord("TTI", "d", 11)])
    report = Pipeline.validate(entries, single=True)
    assert any("all feature fields null" in f for f in report.flags)


def test_validation_ok_when_some_field_present(tmp_cache):
    client = FakeClient([_resp(_features_json(time_window="within 24 hours"))])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    entries = Pipeline(fx).build_entries([ReactionRecord("TTI", "d", 11)])
    report = Pipeline.validate(entries, single=True)
    assert not any("feature" in f for f in report.flags)


def test_validation_full_run_flags_wrong_count(tmp_cache):
    client = FakeClient([_resp(_features_json(time_window="x"))])
    fx = FeatureExtractor(client=client, cache_dir=tmp_cache)
    entries = Pipeline(fx).build_entries([ReactionRecord("TTI", "d", 11)])
    report = Pipeline.validate(entries, single=False)
    assert any("expected 12" in f for f in report.flags)
