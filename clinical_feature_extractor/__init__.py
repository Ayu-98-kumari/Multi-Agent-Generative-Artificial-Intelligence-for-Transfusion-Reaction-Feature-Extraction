"""Clinical feature extractor.

A standalone module that reads clinical notes from a spreadsheet, extracts
clinical features against a fixed canonical vocabulary using a Groq model, runs a
deterministic vocabulary post-check (case-insensitive match -> canonical casing,
off-list features dropped + logged), writes the validated JSON back into the
sheet, and exports the sheet to CSV.

It never imports from or modifies ``pdf_parser`` or ``reaction_extractor``; it
only *reads* ``reaction_extractor/output/deduplicated_symptoms_labmarkers.json``
as a read-only vocabulary source.
"""
