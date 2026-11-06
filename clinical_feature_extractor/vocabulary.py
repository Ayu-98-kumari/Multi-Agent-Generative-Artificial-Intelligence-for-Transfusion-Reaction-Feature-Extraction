"""Canonical clinical-feature vocabulary: load + tiered index.

The vocabulary is the deduplicated feature list produced by the
``reaction_extractor`` module. This module only *reads* it (never writes) and
builds the lookups so a model-returned feature can be matched against the list
and then normalized to the canonical spelling.

Matching is two-tier and fully deterministic (no model, no fuzzy scoring):
  1. **Exact** (case-insensitive) match — highest confidence.
  2. **Formatting-normalized** fallback — collapses cosmetic differences that
     vary document to document (whitespace, parentheses/brackets, edge
     punctuation) while preserving every *meaningful* token: words, numbers,
     thresholds, units, ``%``, ``°``, ``<``/``>``, ``/``. So ``less than 90%`` and
     ``38°C`` still distinguish entries — no false merges on values. A normalized
     match is accepted ONLY when it is unambiguous (the normalized form maps to
     exactly one vocabulary entry); ambiguous forms fall back to exact-only.

Both tiers always return the **vocabulary's** spelling, so output stays
consistent with the canonical list.

Matching contract:
  * ``match("Dyspnea")`` -> ``"dyspnea"`` (exact, canonical casing)
  * ``match("  cough ")`` -> canonical ``"cough"`` (surrounding space ignored)
  * ``match("Fever (...value)")`` -> ``"Fever (...value"`` (normalized: the list
    entry is missing its closing paren; matched anyway, list spelling returned)
  * ``match("not on the list")`` -> ``None`` (caller drops + logs it)
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

from . import config

# Characters treated as cosmetic grouping and dropped during normalization.
_GROUPING_TRANS = str.maketrans({c: " " for c in "()[]{}"})
# Punctuation/whitespace trimmed from the ends of a normalized string.
_EDGE_CHARS = " .,;:"
# A hyphen plus any whitespace on either side. Removed entirely during
# normalization so a hyphen is OPTIONAL: ``pre-transfusion``, ``pre- transfusion``
# and ``pretransfusion`` all collapse to the same joined token.
_HYPHEN_WS = re.compile(r"\s*-\s*")


def _normalize(text: str) -> str:
    """Collapse cosmetic formatting while preserving meaningful tokens.

    General text-matching rules only (no word/entry is special-cased), applied to
    a temporary copy purely for comparison — the stored value is never changed:

      1. Unicode NFC (visually identical characters compare equal, e.g. ``°``).
      2. Lowercase.
      3. Drop grouping brackets/parentheses, so a balance difference (one side
         missing a closing ``)``) does not block a match.
      4. Remove hyphens entirely (with any adjacent whitespace), so a hyphen is
         optional: ``pre-transfusion``, ``pre- transfusion`` and ``pretransfusion``
         all compare equal.
      5. Collapse whitespace runs to one space; trim edge whitespace/punctuation.

    Numbers, units, ``%``, ``°``, ``<``/``>``, ``/`` and words are all preserved,
    so distinct thresholds never collapse together.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower().translate(_GROUPING_TRANS)
    text = _HYPHEN_WS.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(_EDGE_CHARS)


class VocabularyError(RuntimeError):
    """The vocabulary source is missing or malformed."""


class Vocabulary:
    """A tiered, casing-normalizing view over the canonical feature list.

    ``items`` preserves the source order and casing. ``match`` maps a feature to
    its canonical spelling via an exact (case-insensitive) tier and a
    formatting-normalized fallback tier, or returns ``None`` for off-list terms.

    Two determinism guards:
      * On a casing collision within the source (two entries equal ignoring case)
        the first-seen spelling wins.
      * A normalized key shared by two or more entries is marked *ambiguous* and
        removed from the normalized tier, so a normalized match can never guess
        between two different features.
    """

    def __init__(self, items: list[str]):
        self.items: list[str] = list(items)
        self._lookup: dict[str, str] = {}          # exact: lower(text) -> canonical
        self._norm_lookup: dict[str, Optional[str]] = {}  # normalized -> canonical/None
        for item in self.items:
            key = item.strip().lower()
            if key and key not in self._lookup:
                self._lookup[key] = item
            nkey = _normalize(item)
            if not nkey:
                continue
            if nkey not in self._norm_lookup:
                self._norm_lookup[nkey] = item
            elif self._norm_lookup[nkey] != item:
                # Two distinct entries normalize alike -> ambiguous; disable it.
                self._norm_lookup[nkey] = None

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __contains__(self, feature: str) -> bool:
        return self.match(feature) is not None

    def match(self, feature: str) -> Optional[str]:
        """Return the canonical spelling for ``feature``, or None if off-list.

        Tier 1: exact case-insensitive. Tier 2: formatting-normalized, accepted
        only when unambiguous. Both return the vocabulary's spelling.
        """
        if not isinstance(feature, str):
            return None
        exact = self._lookup.get(feature.strip().lower())
        if exact is not None:
            return exact
        return self._norm_lookup.get(_normalize(feature))


def load_vocabulary(path: Optional[Path] = None) -> Vocabulary:
    """Load the canonical feature list from the read-only dedup JSON."""
    path = Path(path) if path else config.VOCABULARY_JSON
    if not path.exists():
        raise VocabularyError(f"vocabulary source not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise VocabularyError(f"could not read vocabulary source {path}: {exc}") from exc
    if not isinstance(data, dict) or config.VOCABULARY_KEY not in data:
        raise VocabularyError(
            f"vocabulary source {path} missing key {config.VOCABULARY_KEY!r}")
    items = data[config.VOCABULARY_KEY]
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise VocabularyError(
            f"vocabulary under {config.VOCABULARY_KEY!r} must be a list[str]")
    if not items:
        raise VocabularyError(f"vocabulary source {path} contains no features")
    return Vocabulary(items)
