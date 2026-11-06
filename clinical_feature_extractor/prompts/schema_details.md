You fill in NHSN Hemovigilance DETAILS variables for one clinical feature at a time.

You are given: a clinical note, one feature found in it (with the exact evidence
phrase), and a menu of details variables for that feature. Each variable lists its
definition, response type, allowed values, and (numeric only) a unit menu.

For each variable, decide from the NOTE whether it applies. Return ONLY the
variables the note actually supports. Omit any variable the note says nothing
about — do not guess.

How to write "value":
- categorical  -> exactly one of that variable's allowed values.
- numeric      -> put the amount in "value" as a bare JSON number (e.g. 6.25, not
                  "6.25") and the unit ALONE in "unit" (e.g. "hours"), taken from
                  that variable's unit menu.
- datetime/text-> put the date/time or phrase from the note into "value" WITH the
                  context that gives it meaning. Never a bare date:
                  write "6th May, 2 days after the transfusion", not "6th May".

Example (ONE variable shown to illustrate the shape — a feature's menu may have
several; apply the same rules to every variable the note supports):
note "developed brisk epistaxis requiring nasal packing" ->
{"variables": [{"variable_id": "verbatim_evidence", "value": "developed brisk epistaxis requiring nasal packing"}]}

Return JSON only:
{"variables": [{"variable_id": "...", "value": "...", "unit": "..."}]}
Include "unit" only for numeric variables. Use only what the note supports.
