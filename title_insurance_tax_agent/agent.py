"""LangGraph tool-calling agent wrapper for the Title/Insurance/Taxes Agent.

Matches the spec's "Model: GPT-5 mini (document validation) + Python
(deterministic escrow math)" line: calculate_escrow is a plain Python
function tool (no LLM judgment involved in its math), while the other three
tools are exposed to the LLM for document-validation reasoning.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from .tools import (
    DEFAULT_CUSHION_MONTHS,
    calculate_aggregate_escrow,
    calculate_escrow,
    search_documents,
    validate_closing_agent_match,
    validate_insurance_coverage,
    validate_no_coverage_gap,
    validate_ny_closing_attorney,
    validate_premium_paid_in_full,
    validate_property_taxes,
    validate_title_commitment,
    validate_wire_instructions,
)

DEFAULT_MODEL = os.environ.get("TITLE_INSURANCE_TAX_AGENT_MODEL", "openai:gpt-4o-mini")

SYSTEM_PROMPT = """You are the Title/Insurance/Taxes Agent for loan processing.

Role: Verify title commitment, insurance docs, and tax information.

Triggers (handled by whatever invokes you — title commitment uploaded,
insurance policy uploaded, or tax document uploaded): for each loan you are
given a loan GUID and whichever document(s) are relevant.

Responsibilities:
1. If you need to locate a document in the loan's eFolder, call
   search_documents with a query like "title commitment" or "flood insurance".
   This is a bucket/keyword search, not a semantic search — be specific with
   your query terms.
2. For a title commitment, call validate_title_commitment with the extracted
   document fields and loan context (loan_amount, estimated_closing_date,
   property_state). IMPORTANT: this tool does NOT check chain of title —
   standard title commitments don't contain that data. If asked about chain
   of title, say explicitly that it is not currently verifiable, don't guess.
3. For insurance documents (hazard/flood), call validate_insurance_coverage
   with the extracted HOI/flood fields and loan context (loan_amount,
   property_type, is_flood_zone). This checks deductible caps, coverage vs.
   loan amount, and the 12-month policy term.
4. For escrow on a simple tax+insurance loan, call calculate_escrow with
   property_taxes, insurance_premium, and months (default 2 per the RM Job
   Aid). For a full Aggregate Escrow Account with multiple disbursement
   items (different tax types, insurance types, each with their own due
   dates), call calculate_aggregate_escrow instead — it computes the real
   aggregate adjustment (always <= 0 per RM Job Aid p.15) rather than a flat
   formula. Both are deterministic math — always call the tool rather than
   computing it yourself.
5. For a premium receipt/invoice, call validate_premium_paid_in_full to
   confirm the premium is paid in full or properly invoiced for closing.
6. For a refinance, call validate_no_coverage_gap with the old and new
   insurance policy fields to check for a coverage gap and binder-timing
   issues.
7. For a property tax document, call validate_property_taxes with the
   extracted tax fields and loan context (transaction_type, property_state,
   days_to_closing, etc.) — this applies the RM Job Aid's scenario-specific
   rules (new construction millage rate, CA purchase 1.25%, purchases with
   exemptions, refinances, and the 60-day current-tax-bill rule).
8. For wholesale loans, call validate_closing_agent_match with the title,
   CPL, and E&O fields to cross-match the agent name across all three
   documents and check CPL/E&O standards.
9. Before releasing any wire instructions, ALWAYS call
   validate_wire_instructions to confirm the beneficiary matches the title
   commitment's settlement agent — this is a wire-fraud prevention check and
   must never be skipped.
10. For NY properties, call validate_ny_closing_attorney with the closing
    attorney's name and the approved-attorney list to confirm they're
    approved.
11. Summarize all flags you've collected across these tools, clearly
    separating critical issues from items needing follow-up, as your final
    answer.

Known limitations (state them plainly if relevant, don't fabricate a check):
chain of title is not currently implemented — standard title commitments
don't carry historical ownership data, and no deed-history/abstract document
source has been scoped for this agent yet.
"""


def build_title_insurance_tax_agent(
    efolder_index: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    mortgagee_clauses: Optional[Dict[str, str]] = None,
    ny_approved_attorneys: Optional[List[str]] = None,
    model: Optional[str] = None,
):
    """Build a runnable LangGraph react-agent for the Title/Insurance/Taxes Agent.

    Args:
        efolder_index: {bucket_name: [{"doc_id", "filename", "text"}, ...]}
            backing search_documents. Defaults to empty (no documents found).
        mortgagee_clauses: override the default (NQMF-example) mortgagee
            clause table used by validate_title_commitment.
        ny_approved_attorneys: list of approved NY closing attorney names
            (sourced from the external intranet list — not hardcoded here,
            same pattern as mortgagee_clauses). Defaults to empty.
        model: chat model identifier understood by
            `langchain.chat_models.init_chat_model`.
    """
    efolder_index = efolder_index or {}
    ny_approved_attorneys = ny_approved_attorneys or []

    @tool
    def search_documents_tool(query: str, loan_guid: str) -> list:
        """Find documents in the loan's eFolder matching a free-text query
        (e.g. "title commitment", "flood insurance", "tax summary"). This is
        bucket/keyword matching, not semantic search."""
        return search_documents(query, loan_guid, efolder_index)

    @tool
    def validate_title_commitment_tool(
        document_fields: dict,
        loan_context: dict,
        doc_expiration_screen_date: Optional[str] = None,
        expiration_condition_date: Optional[str] = None,
    ) -> dict:
        """Validate a title commitment's extracted fields: lender coverage
        vs. loan amount, Title Effective Date within 90 days of closing, a
        3-way date consistency check (doc vs. Document Expirations screen vs.
        expiration condition), and a mortgagee-clause reminder for the
        property's state. Does NOT check chain of title (not currently
        supported)."""
        return validate_title_commitment(document_fields, loan_context, mortgagee_clauses,
                                          doc_expiration_screen_date, expiration_condition_date)

    @tool
    def validate_insurance_coverage_tool(hoi_fields: dict, loan_context: dict,
                                          flood_fields: Optional[dict] = None,
                                          ho6_fields: Optional[dict] = None) -> dict:
        """Validate hazard & flood insurance: HOI deductible <= 5% of
        coverage, flood deductible <= $10K/$25K (Condo/PUD) with flood-zone
        status read from the flood document's own in_sfha field when
        present, HOI coverage vs. the lesser of loan amount/replacement
        cost, HOI mortgagee clause match, HOI policy term ~12 months,
        wind/hail/hurricane coverage type, DSCR 6-month rent-loss coverage
        type, condo HO6 deductible cap, and the flood 'lesser of 3' coverage
        rule."""
        return validate_insurance_coverage(hoi_fields, loan_context, flood_fields, ho6_fields, mortgagee_clauses)

    @tool
    def calculate_escrow_tool(property_taxes: float, insurance_premium: float,
                               months: int = DEFAULT_CUSHION_MONTHS) -> dict:
        """Deterministically calculate a simple escrow cushion: monthly tax +
        monthly insurance, times the cushion-months (default 2). For a full
        multi-item Aggregate Escrow Account, use calculate_aggregate_escrow
        instead."""
        return calculate_escrow(property_taxes, insurance_premium, months)

    @tool
    def calculate_aggregate_escrow_tool(disbursement_items: list,
                                         cushion_months: int = DEFAULT_CUSHION_MONTHS) -> dict:
        """Deterministically calculate the full Aggregate Escrow Account:
        each disbursement item's own due-date schedule (e.g.
        [{"name": "County Tax", "annual_amount": 6000, "due_months": [12]}]),
        the pooled aggregate cushion, and the aggregate adjustment (which
        should always be <= 0 per RM Job Aid p.15)."""
        return calculate_aggregate_escrow(disbursement_items, cushion_months)

    @tool
    def validate_premium_paid_in_full_tool(receipt_fields: Optional[dict], policy_fields: dict,
                                            loan_context: dict) -> dict:
        """Verify an insurance premium is paid in full (receipt) or properly
        invoiced for payment at closing, including the refinance 60-day rule."""
        return validate_premium_paid_in_full(receipt_fields, policy_fields, loan_context)

    @tool
    def validate_no_coverage_gap_tool(old_policy_fields: dict, new_policy_fields: dict,
                                       loan_context: dict) -> dict:
        """For refinances: verify there's no gap between the old policy's
        expiration and the new policy's effective date, and check
        binder-timing rules."""
        return validate_no_coverage_gap(old_policy_fields, new_policy_fields, loan_context)

    @tool
    def validate_property_taxes_tool(tax_fields: Optional[dict], loan_context: dict) -> dict:
        """Verify the property tax amount per the applicable scenario (new
        construction, CA purchase, purchase with exemptions, or refinance),
        plus the 60-day current-tax-bill rule."""
        return validate_property_taxes(tax_fields, loan_context)

    @tool
    def validate_closing_agent_match_tool(title_fields: dict, cpl_fields: Optional[dict],
                                           eo_fields: Optional[dict],
                                           loan_context: Optional[dict] = None) -> dict:
        """Cross-match the settlement/closing agent name across the Title
        Commitment, CPL, and E&O policy, plus CPL age and E&O coverage
        minimums. Not applicable to Correspondent loans."""
        return validate_closing_agent_match(title_fields, cpl_fields, eo_fields, loan_context)

    @tool
    def validate_wire_instructions_tool(wire_fields: dict, title_fields: dict) -> dict:
        """Wire-fraud prevention: verify the wire instructions' beneficiary
        matches the settlement agent named on the title commitment. ALWAYS
        call this before confirming any wire instructions are safe to use."""
        return validate_wire_instructions(wire_fields, title_fields)

    @tool
    def validate_ny_closing_attorney_tool(attorney_name: Optional[str], property_state: Optional[str] = None) -> dict:
        """For NY properties, verify the closing attorney is on the
        NQMF-approved list."""
        return validate_ny_closing_attorney(attorney_name, ny_approved_attorneys, property_state)

    tools = [
        search_documents_tool,
        validate_title_commitment_tool,
        validate_insurance_coverage_tool,
        calculate_escrow_tool,
        calculate_aggregate_escrow_tool,
        validate_premium_paid_in_full_tool,
        validate_no_coverage_gap_tool,
        validate_property_taxes_tool,
        validate_closing_agent_match_tool,
        validate_wire_instructions_tool,
        validate_ny_closing_attorney_tool,
    ]

    return create_react_agent(model=model or DEFAULT_MODEL, tools=tools, prompt=SYSTEM_PROMPT)
