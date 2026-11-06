"""Dedup-step tests with a FAKE (streaming) Groq client — no network, no spend.

The dedup call now STREAMS and the model returns the RETAINED features as a plain
JSON list (double-negative removal). The fake yields streamed chunks:
    chunk.choices[0].delta.content   -> a piece of the JSON text
    chunk.choices[0].finish_reason   -> "stop" | "length" (on the final chunk)
The reconciliation (keep original order, every retained item must be verbatim in
the input) is enforced in code, not trusted from the model.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from reaction_extractor import config, features
from reaction_extractor import dedup_features as df
from reaction_extractor.dedup_features import (
    FeatureDeduplicator,
    reconcile_retained,
    build_deduplicated_symptoms_labmarkers,
)


# --------------------------------------------------------------------------- #
# Fake streaming client
# --------------------------------------------------------------------------- #
def _stream(content: str, finish: str = "stop"):
    """A fake streaming response: one content chunk, then a final finish chunk."""
    return [
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=content), finish_reason=None)]),
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=None), finish_reason=finish)]),
    ]


def _retained(items, finish: str = "stop"):
    """A stream whose content is the JSON list of RETAINED features."""
    return _stream(json.dumps(items), finish)


class _FakeCompletions:
    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = 0
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
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
    monkeypatch.setattr(df.time, "sleep", lambda *_: None)


@pytest.fixture
def tmp_cache(tmp_path):
    return tmp_path / "cache"


# --------------------------------------------------------------------------- #
# reconcile_retained — pure deterministic logic
# --------------------------------------------------------------------------- #
def test_reconcile_removes_complement_and_preserves_order():
    items = ["A", "B", "C", "D"]
    result = reconcile_retained(items, ["A", "C", "D"])   # B dropped
    assert result["deduplicated"] == ["A", "C", "D"]
    assert result["removed_indices"] == [1]
    assert [x["removed_value"] for x in result["removed_items"]] == ["B"]


def test_reconcile_empty_removal_keeps_all():
    items = ["A", "B", "C"]
    result = reconcile_retained(items, items)
    assert result["deduplicated"] == items
    assert result["removed_items"] == []


def test_reconcile_enforces_original_order_ignoring_model_reordering():
    items = ["A", "B", "C"]
    result = reconcile_retained(items, ["C", "A"])   # model reordered + dropped B
    assert result["deduplicated"] == ["A", "C"]      # rebuilt in ORIGINAL order
    assert [x["removed_value"] for x in result["removed_items"]] == ["B"]


def test_reconcile_unknown_retained_item_raises():
    with pytest.raises(features.FeatureError):
        reconcile_retained(["A", "B"], ["A", "HALLUCINATED"])


def test_reconcile_non_list_raises():
    with pytest.raises(features.FeatureError):
        reconcile_retained(["A"], {"not": "a list"})


def test_reconcile_non_string_items_raise():
    with pytest.raises(features.FeatureError):
        reconcile_retained(["A"], ["A", 123])


# --------------------------------------------------------------------------- #
# deduplicate() — cache + call path (new streaming params)
# --------------------------------------------------------------------------- #
def test_deduplicate_calls_model_and_caches(tmp_cache):
    items = ["A", "B"]
    client = FakeClient([_retained(["A"])])            # keep A, drop B
    dedup = FeatureDeduplicator(client=client, cache_dir=tmp_cache)
    result = dedup.deduplicate(items)
    assert result["deduplicated"] == ["A"]
    kw = client.chat.completions.kwargs
    # Exact sandbox-validated params; NO strict JSON mode; reasoning "default".
    assert kw["stream"] is True
    assert "response_format" not in kw
    assert kw["reasoning_effort"] == "default"
    assert kw["temperature"] == 0.0
    assert kw["top_p"] == 0.95
    assert kw["max_completion_tokens"] == df.DEDUP_MAX_COMPLETION_TOKENS
    assert kw["model"] == df.DEDUP_MODEL
    # Second call with the same items is served from cache (no new model call).
    dedup.deduplicate(items)
    assert client.chat.completions.calls == 1


def test_no_cache_bypasses_cache(tmp_cache):
    items = ["A", "B"]
    client = FakeClient([_retained(["A"]), _retained(["A", "B"])])
    dedup = FeatureDeduplicator(client=client, cache_dir=tmp_cache)
    dedup.deduplicate(items)                            # populates cache
    result = dedup.deduplicate(items, use_cache=False)
    assert client.chat.completions.calls == 2
    assert result["deduplicated"] == items             # second response kept all


def test_changed_input_invalidates_cache(tmp_cache):
    client = FakeClient([_retained(["A"]), _retained(["C"])])
    dedup = FeatureDeduplicator(client=client, cache_dir=tmp_cache)
    dedup.deduplicate(["A", "B"])
    dedup.deduplicate(["C", "D"])                       # different list -> new call
    assert client.chat.completions.calls == 2


# --------------------------------------------------------------------------- #
# Malformed / truncated responses never silently drop items
# --------------------------------------------------------------------------- #
def test_invalid_json_raises(tmp_cache):
    client = FakeClient([_stream("not json at all")])
    dedup = FeatureDeduplicator(client=client, cache_dir=tmp_cache)
    with pytest.raises(features.FeatureError):
        dedup.deduplicate(["A"])


def test_non_list_response_raises(tmp_cache):
    client = FakeClient([_stream(json.dumps({"retained": ["A"]}))])   # object, not list
    dedup = FeatureDeduplicator(client=client, cache_dir=tmp_cache)
    with pytest.raises(features.FeatureError):
        dedup.deduplicate(["A"])


def test_hallucinated_item_raises(tmp_cache):
    client = FakeClient([_retained(["A", "GHOST"])])   # GHOST not in the input
    dedup = FeatureDeduplicator(client=client, cache_dir=tmp_cache)
    with pytest.raises(features.FeatureError):
        dedup.deduplicate(["A", "B"])


def test_truncation_is_hard_failure(no_sleep, tmp_cache):
    client = FakeClient([_retained(["A"], finish="length")])
    dedup = FeatureDeduplicator(client=client, cache_dir=tmp_cache)
    with pytest.raises(features.TruncationError):
        dedup.deduplicate(["A", "B"])


def test_fenced_and_reasoning_wrapped_json_is_parsed(tmp_cache):
    """A response with a <think> block and code fences still parses."""
    wrapped = "<think>deciding…</think>\n```json\n" + json.dumps(["A"]) + "\n```"
    client = FakeClient([_stream(wrapped)])
    dedup = FeatureDeduplicator(client=client, cache_dir=tmp_cache)
    assert dedup.deduplicate(["A", "B"])["deduplicated"] == ["A"]


# --------------------------------------------------------------------------- #
# build_deduplicated_symptoms_labmarkers — file I/O, audit, invariants
# --------------------------------------------------------------------------- #
def _source_doc(items):
    return {"metadata": {"total_features": len(items)},
            "extended_symptoms_labmarkers": items}


def test_build_writes_two_parallel_files_with_invariant(tmp_path, tmp_cache):
    items = ["A", "B", "C"]
    src = tmp_path / "extended_symptoms_labmarkers.json"
    src.write_text(json.dumps(_source_doc(items)), encoding="utf-8")
    out = tmp_path / "deduplicated_symptoms_labmarkers.json"
    removed_out = tmp_path / "dedup_removed_items.json"

    client = FakeClient([_retained(["A", "C"])])       # drop B
    dedup = FeatureDeduplicator(client=client, cache_dir=tmp_cache)
    build_deduplicated_symptoms_labmarkers(
        input_path=src, output_path=out, removed_output_path=removed_out,
        deduplicator=dedup)

    # --- File 1: clean deduplicated list, no audit trail inside it. ---
    written = json.loads(out.read_text(encoding="utf-8"))
    assert list(written.keys()) == ["metadata", "deduplicated_symptoms_labmarkers"]
    assert "removed_items" not in written
    meta = written["metadata"]
    assert (meta["original_count"], meta["removed_count"], meta["final_count"]) == (3, 1, 2)
    assert meta["final_count"] == meta["original_count"] - meta["removed_count"]
    assert meta["final_count"] == len(written["deduplicated_symptoms_labmarkers"])
    assert written["deduplicated_symptoms_labmarkers"] == ["A", "C"]
    assert meta["source_filename"] == "extended_symptoms_labmarkers.json"

    # --- File 2: removed-items audit trail, self-contained. ---
    removed = json.loads(removed_out.read_text(encoding="utf-8"))
    assert list(removed.keys()) == ["metadata", "removed_items"]
    assert removed["metadata"]["removed_count"] == 1
    assert removed["removed_items"] == [
        {"removed_value": "B", "reason": "mechanically-generated double negative"}]


def test_build_empty_removals_keeps_all(tmp_path, tmp_cache):
    items = ["A", "B", "C"]
    src = tmp_path / "extended_symptoms_labmarkers.json"
    src.write_text(json.dumps(_source_doc(items)), encoding="utf-8")
    out = tmp_path / "out.json"
    removed_out = tmp_path / "removed.json"

    client = FakeClient([_retained(items)])            # keep everything
    dedup = FeatureDeduplicator(client=client, cache_dir=tmp_cache)
    build_deduplicated_symptoms_labmarkers(
        input_path=src, output_path=out, removed_output_path=removed_out,
        deduplicator=dedup)

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["deduplicated_symptoms_labmarkers"] == items
    assert written["metadata"]["final_count"] == 3
    removed = json.loads(removed_out.read_text(encoding="utf-8"))
    assert removed["removed_items"] == []
    assert removed["metadata"]["removed_count"] == 0


def test_source_file_not_modified(tmp_path, tmp_cache):
    items = ["A", "B", "C"]
    src = tmp_path / "extended_symptoms_labmarkers.json"
    original = json.dumps(_source_doc(items))
    src.write_text(original, encoding="utf-8")

    client = FakeClient([_retained(["A", "C"])])
    dedup = FeatureDeduplicator(client=client, cache_dir=tmp_cache)
    build_deduplicated_symptoms_labmarkers(
        input_path=src, output_path=tmp_path / "out.json", deduplicator=dedup)

    assert src.read_text(encoding="utf-8") == original


def test_prompt_targets_double_negatives():
    prompt = (config.PROMPTS_DIR / "dedup.md").read_text(encoding="utf-8")
    assert "double negative" in prompt.lower()
    assert "Return ONLY a valid JSON list" in prompt
