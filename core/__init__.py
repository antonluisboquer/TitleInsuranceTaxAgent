"""Shared core utilities for the Appraisal & Collateral Agent and the
Title/Insurance/Taxes Agent.

These two agents are built from the specs documented in:
  LG-discOrch/docs/APPRAISAL_TITLE_INSURANCE_TAX_AGENTS_ARCHITECTURE.md

This package intentionally has ZERO dependency on LG-discOrch / LG-docsOrch /
lg-RMOrch at import time. Where this repo ports business logic that was
*verified* to already exist in one of those three repos, the source is cited
in a comment immediately above the relevant code so it can be diffed against
the original if those repos change.
"""
