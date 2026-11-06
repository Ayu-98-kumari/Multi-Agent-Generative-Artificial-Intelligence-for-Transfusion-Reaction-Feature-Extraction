Parse the attached page image into the fixed JSON structure.

Page number: {page}
Section: {section_number} — {section_title}
Section page span: {section_span}

Skeleton expectations for THIS page (from a deterministic pre-pass):
{expected_block}

Layout hint: {layout_hint}

Continuation: {continuation_hint}

Reminders:
- Use page={page} and section_number={section_number} in your output.
- Emit every block on the page, in reading order. Do not omit anything.
- The expectations above are a checklist, not a limit: if you see additional
  content, include it. If an expected table/figure is NOT actually on this page,
  do not fabricate it.
- Transcribe verbatim; preserve AND/OR/EITHER logic in table cells.

Return ONLY the JSON object.
