"""Adapter for using LG-docsOrch's regression-test golden fixtures as
realistic input data for this repo's tools.

LG-docsOrch (a sibling Fintor repo) maintains a golden-fixture regression
suite at ``tests/regression/fixtures/<loan>/frozen_state.json`` -- a frozen
snapshot of real, previously-closed loan files (extracted document fields +
LOS fields) used there to catch regressions in its own extraction/flagging
pipeline.

That data is genuinely useful here too, just for a different purpose: it's
realistic *input* for our tools -- real currency-string formats
("$1,890.83"), real "M/D/YYYY" dates, nested contact/company dicts, fields
that are sometimes missing -- which is a much better stress test than
hand-written synthetic fixtures alone.

This module maps docsOrch's doc_fields/los_fields key names into the shape
this repo's ``tool_schema/*.json`` extraction schemas (and tool function
signatures) expect. It does NOT attempt to replicate docsOrch's own
golden_flags.json expected output: this repo implements different (and
additional) business rules than docsOrch does, so there's no meaningful
"expected flags" comparison to make across repos. See
``tests/test_golden_fixtures.py`` for how this is used.

If the sibling repo isn't present on disk (e.g. CI, or a machine with only
this repo checked out), ``available_fixtures()`` returns an empty list and
dependent tests skip themselves automatically.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# This repo and LG-docsOrch are expected to be sibling directories, e.g.:
#   .../Fintor/LG-TitleInsuranceTaxAgent   (this repo)
#   .../Fintor/LG-docsOrch                 (sibling)
_THIS_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_FIXTURES_DIR = (
    _THIS_REPO.parent / "LG-docsOrch" / "tests" / "regression" / "fixtures"
)
FIXTURES_DIR = Path(os.environ.get("DOCSORCH_FIXTURES_DIR", str(_DEFAULT_FIXTURES_DIR)))


def available_fixtures() -> List[str]:
    """Names of golden-fixture loan directories that have a frozen_state.json.
    Returns [] if the sibling repo/fixtures aren't present on disk."""
    if not FIXTURES_DIR.is_dir():
        return []
    return sorted(
        p.name
        for p in FIXTURES_DIR.iterdir()
        if p.is_dir() and (p / "frozen_state.json").is_file()
    )


def load_frozen_state(name: str) -> Dict[str, Any]:
    path = FIXTURES_DIR / name / "frozen_state.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _unwrap(entry: Any) -> Any:
    """doc_fields/los_fields entries look like
    {"value": ..., "source_document"|"field_id": ..., ...}; already-plain
    values pass through unchanged."""
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def doc_value(doc_fields: Dict[str, Any], *keys: str, source: Optional[str] = None) -> Any:
    """First present, non-None value among `keys` in docsOrch's doc_fields.

    `source`, if given, restricts the match to entries whose
    `source_document` equals it -- docsOrch sometimes reuses a field name
    (e.g. "property_address") across multiple document types, and we want
    the copy that actually came from the document we care about.
    """
    for key in keys:
        entry = doc_fields.get(key)
        if entry is None:
            continue
        if source and isinstance(entry, dict) and entry.get("source_document") != source:
            continue
        value = _unwrap(entry)
        if value is not None:
            return value
    return None


def los_value(los_fields: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        entry = los_fields.get(key)
        if entry is None:
            continue
        value = _unwrap(entry)
        if value is not None:
            return value
    return None


# ---------------------------------------------------------------------------
# Field mappings: this repo's tool_schema field names <- docsOrch doc_fields
# ---------------------------------------------------------------------------

def build_title_fields(doc_fields: Dict[str, Any]) -> Dict[str, Any]:
    title_company = doc_value(doc_fields, "title_company")
    title_company_name = title_company.get("company_name") if isinstance(title_company, dict) else title_company
    return {
        "title_number": doc_value(doc_fields, "commitment_number", "escrow_number"),
        "settlement_agent_name": doc_value(doc_fields, "settlement_agent", source="Title Report") or title_company_name,
        "title_company": title_company_name,
        "effective_date": doc_value(doc_fields, "effective_date", source="Title Report"),
        "vesting_type": doc_value(doc_fields, "final_vesting", "vesting_type"),
        "legal_description": doc_value(doc_fields, "legal_description"),
    }


def build_insurance_fields(doc_fields: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "policy_number": doc_value(doc_fields, "policy_number", source="Evidence of Insurance"),
        "hazard_insurance_premium": doc_value(doc_fields, "hazard_insurance_premium"),
        "hazard_insurance_company": doc_value(doc_fields, "hazard_insurance_company"),
        "coverage_amount": doc_value(doc_fields, "hazard_insurance_coverage"),
        "deductible": doc_value(doc_fields, "deductible", source="Evidence of Insurance"),
        "effective_date": doc_value(doc_fields, "coverage_start_date"),
        "expiration_date": doc_value(doc_fields, "coverage_end_date"),
        "wind_hail_deductible": doc_value(doc_fields, "wind_hail_deductible"),
        "replacement_cost": doc_value(doc_fields, "replacement_cost", source="Evidence of Insurance"),
        "mortgagee_name": doc_value(doc_fields, "mortgagee_name"),
    }


def build_tax_fields(doc_fields: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "annual_taxes": doc_value(doc_fields, "annual_taxes"),
        "tax_parcel_number": doc_value(doc_fields, "tax_parcel_number"),
        "property_address": doc_value(doc_fields, "property_address"),
    }


def build_cpl_fields(doc_fields: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "agent_name": doc_value(doc_fields, "settlement_agent_name", source="Closing Protection Letter"),
        "effective_date": doc_value(doc_fields, "cpl_issue_date"),
        "expiration_date": doc_value(doc_fields, "cpl_expiration_date"),
    }


def build_loan_context(frozen_state: Dict[str, Any]) -> Dict[str, Any]:
    los_fields = frozen_state.get("los_fields", {}) or {}
    loan_profile = frozen_state.get("loan_profile", {}) or {}
    purpose = str(loan_profile.get("purpose") or "").lower()
    return {
        "loan_amount": los_value(los_fields, "loan_amount", "base_loan_amount"),
        "property_state": los_value(los_fields, "property_state") or loan_profile.get("state"),
        "estimated_closing_date": los_value(los_fields, "note_date", "closing_date"),
        "transaction_type": "refinance" if "refinance" in purpose else "purchase",
        "loan_type": loan_profile.get("loan_type"),
    }
