"""Hand-written golden facts — specific clinical values that must survive parsing.

These catch subtle VLM misreads that schema/coverage/structure checks cannot.

Each fact is SCOPED to a specific table, identified by a ``page`` the table
covers (validation locates the table block whose page_span contains that page,
including the stitched Table 8 spanning pages 19-22). The ``check`` then runs on
that table's text only — NOT the whole document. Scoping is essential here:
several values recur across reactions (DHTR vs DSTR both cite "24 hours and 28
days"; HTR adult vs pediatric BP thresholds; "within 24 hours" appears in AHTR
and TAD), so a document-wide substring would pass spuriously.

Golden-fact failures are FLAGGED for review (severity "flag"), never hard-blocking.

To extend: add a ``GoldenFact`` with the page its target table lives on and a
``check`` over normalized (lowercased, whitespace-collapsed) table text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


def normalize_for_golden(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass(frozen=True)
class GoldenFact:
    name: str
    description: str
    page: int                       # a page the target table covers
    check: Callable[[str], bool]    # runs on the located table's normalized text


def _all_present(*needles: str) -> Callable[[str], bool]:
    lowered = [n.lower() for n in needles]
    return lambda text: all(n in text for n in lowered)


def _any_present(*needles: str) -> Callable[[str], bool]:
    lowered = [n.lower() for n in needles]
    return lambda text: any(n in text for n in lowered)


# --------------------------------------------------------------------------- #
# All facts are table-scoped (uniformly). page = a page the target table covers.
# --------------------------------------------------------------------------- #
GOLDEN_FACTS: list[GoldenFact] = [
    # --- seeded five ---------------------------------------------------- #
    GoldenFact("taco_window_12h",
               "TACO (Table 5) onset within 12 hours of transfusion.",
               14, _all_present("12 hours")),
    GoldenFact("trali_window_6h",
               "TRALI (Table 6) onset within 6 hours of transfusion.",
               15, _all_present("6 hours")),
    GoldenFact("ahtr_onset_24h",
               "AHTR (Table 4) onset within 24 hours of transfusion.",
               13, _all_present("24 hours")),
    GoldenFact("incident_codes_present",
               "Incident codes UT 06 and PC 03 survive in Table 8.",
               19, _all_present("ut 06", "pc 03")),
    GoldenFact("babesia_footnote_14_states",
               "Table 2 footnote references the 14 endemic/contiguous states for Babesia.",
               8, _all_present("babesia", "14 states")),

    # --- eight signed-off candidates ------------------------------------ #
    GoldenFact("trali_pao2_fio2_300",
               "TRALI (Table 6) hypoxemia threshold PaO2/FiO2 <= 300 mm Hg.",
               15, _all_present("pao2/fio2", "300 mm hg")),
    GoldenFact("dhtr_antibody_window_24h_28d",
               "DHTR (Section 6) antibodies developed between 24 hours and 28 days.",
               25, _all_present("24 hours and 28 days")),
    GoldenFact("htr_adult_systolic_bp",
               "HTR (Section 6) adult systolic BP drop >= 30 mmHg and <= 80 mmHg.",
               28, _all_present("30 mmhg", "80 mmhg")),
    GoldenFact("ptp_definite_timing_5_12d",
               "PTP (Section 6) definite case occurs 5-12 days post-transfusion.",
               29, _all_present("5-12 days")),
    GoldenFact("ptp_thrombocytopenia_20pct",
               "PTP (Section 6) thrombocytopenia: platelets to less than 20% of baseline.",
               29, _all_present("less than 20%")),
    GoldenFact("tad_respiratory_distress_24h",
               "TAD (Section 6) acute respiratory distress within 24 hours.",
               30, _all_present("respiratory distress", "within 24 hours")),
    GoldenFact("ahtr_immune_dat_antiigg",
               "AHTR (Table 4) immune-mediated: positive DAT for anti-IgG.",
               13, _all_present("direct antiglobulin test", "anti-igg")),
    GoldenFact("taco_bnp_biomarker",
               "TACO (Table 5) cites elevated brain natriuretic peptide (BNP).",
               14, _all_present("natriuretic peptide")),

    # --- stitch-integrity fact ------------------------------------------ #
    GoldenFact("table8_tail_codes_present",
               "Stitched Table 8 retains a last-fragment code (MS 99 / UT 26, page 22).",
               19, _any_present("ms 99", "ut 26")),
]
