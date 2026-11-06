"""pdf_parser — faithful PDF → structured JSON for medical surveillance protocols.

Two passes + stitch + validate:
  1. skeleton.py    deterministic structure/inventory (no LLM)
  2. vision_parser  Groq VLM page-by-page parsing into a fixed schema
  3. stitching.py   merge cross-page logical units
  4. validation.py  deterministic acceptance gate
Orchestrated by pipeline.py, driven by cli.py.
"""

__version__ = "0.1.0"
