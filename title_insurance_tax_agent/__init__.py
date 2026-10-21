"""Title/Insurance/Taxes Agent.

Spec (source of truth, see architecture doc Part 2):
    Role: Verify title commitment, insurance docs, and tax information
    Triggers: Title commitment uploaded / Insurance policy uploaded / Tax
              document uploaded
    Responsibilities:
      - Validate title commitment includes chain of title
      - Check insurance coverage matches loan requirements
      - Calculate escrow amounts (DETERMINISTIC — Python code, no LLM)
      - Verify tax information (look at title commitment per business rule #13)
      - Flag missing or incorrect information
    Model: GPT-5 mini (document validation) + Python (deterministic escrow math)
    Tools: search_documents(query, loan_guid) ·
           validate_title_commitment(document) ·
           validate_insurance_coverage(document, loan_requirements) ·
           calculate_escrow(property_taxes, insurance_premium, months)

MVP scope (this pass): all 4 named tools, implemented with the deterministic
business rules already proven in lg-RMOrch (review_preliminary_title,
review_hazard_flood_insurance) plus the RM Job Aid's numeric thresholds, and
the fully-specified calculate_escrow math.

Explicitly OUT of scope for this pass (tracked as backlog in README.md):
  - Chain of title (T2) — SKIPPED per decision; standard title commitments
    don't carry historical ownership data, needs a new document source.
    validate_title_commitment does not claim to check this.
  - search_documents is a deterministic bucket/keyword matcher, NOT a
    ChromaDB/vector semantic search (T1) — see docstring on search_documents.
  - "Verify tax info per title commitment business rule #13" (T4) — no repo
    has ever found "rule #13"; spec needs clarification before building.
  - J-T3 (premium-paid-in-full receipt/invoice), J-T4 (refi coverage-gap
    date-range), J-T5 (wind/hail coverage TYPE), J-T6 (flood "lesser of 3"
    full rule), J-T7 (DSCR rent-loss coverage TYPE), J-T8 (HO6-specific
    fields), J-T10 (CPL/E&O 3-way match), J-T11 (wire instructions schema),
    J-T12 (NY attorney list) — all need new extraction fields/doc types not
    present in any of the 3 source repos.
"""

from .agent import build_title_insurance_tax_agent
from .tools import (
    calculate_escrow,
    search_documents,
    validate_insurance_coverage,
    validate_title_commitment,
)

__all__ = [
    "build_title_insurance_tax_agent",
    "search_documents",
    "validate_title_commitment",
    "validate_insurance_coverage",
    "calculate_escrow",
]
