"""Deterministic, no-API-key demo of the Title/Insurance/Taxes Agent pipeline.

Runs search_documents -> validate_title_commitment -> validate_insurance_coverage
-> calculate_escrow using canned extracted-field dicts (standing in for what
an LLM extraction step would produce from real PDFs), so this can run
anywhere with just Python.

To run the real LLM-orchestrated agent instead, see the bottom of this file.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from title_insurance_tax_agent.tools import (
    calculate_aggregate_escrow,
    calculate_escrow,
    search_documents,
    validate_insurance_coverage,
    validate_property_taxes,
    validate_title_commitment,
    validate_wire_instructions,
)

LOAN_GUID = "22222222-2222-2222-2222-222222222222"

EFOLDER_INDEX = {
    "Title Commitment": [
        {"doc_id": "t1", "filename": "title_commitment.pdf", "text": "preliminary title commitment order #998877"},
    ],
    "Evidence of Insurance": [
        {"doc_id": "i1", "filename": "hoi_dec_page.pdf", "text": "homeowners hazard dwelling policy declaration"},
    ],
}

LOAN_CONTEXT = {
    "loan_amount": 400_000,
    "estimated_closing_date": "2026-03-01",
    "property_state": "CA",
    "property_type": "SFR",
    "is_flood_zone": False,
}

# Canned extracted field dicts, standing in for what an LLM extraction step
# would produce from the real PDFs using tool_schema/title_report.json and
# tool_schema/evidence_of_insurance.json.
TITLE_FIELDS = {
    "title_number": "998877",
    "lender_coverage": 400_000,
    "effective_date": "2026-01-15",  # ~45 days before closing — within 90-day window
    "vesting_type": "Fee Simple",
    "settlement_agent_name": "Acme Title & Escrow",
}

HOI_FIELDS = {
    "policy_number": "HOI-55443322",
    "coverage_amount": 450_000,
    "deductible": 15_000,  # > 5% of 450,000 -> should flag critical
    "effective_date": "2026-01-01",
    "expiration_date": "2027-01-01",  # 12-month term -> OK
}


def main():
    print(f"=== Title/Insurance/Taxes Agent demo — loan {LOAN_GUID} ===\n")

    found = search_documents("title commitment", LOAN_GUID, EFOLDER_INDEX)
    print(f"1. search_documents('title commitment') found {len(found)} doc(s):", found)

    title_result = validate_title_commitment(TITLE_FIELDS, LOAN_CONTEXT)
    print(f"\n2. Title commitment validation (chain_of_title_verified="
          f"{title_result['chain_of_title_verified']}):")
    for f in title_result["flags"]:
        print(f"   - [{f['severity']}] {f['title']}: {f['details']}")

    insurance_result = validate_insurance_coverage(HOI_FIELDS, LOAN_CONTEXT)
    print("\n3. Insurance coverage validation:")
    for f in insurance_result["flags"]:
        print(f"   - [{f['severity']}] {f['title']}: {f['details']}")

    escrow_result = calculate_escrow(property_taxes=7_200, insurance_premium=1_800, months=2)
    print("\n4. Deterministic escrow calculation:", escrow_result)

    aggregate_result = calculate_aggregate_escrow([
        {"name": "County Tax", "annual_amount": 7_200, "due_months": [12]},
        {"name": "HOI", "annual_amount": 1_800, "due_months": [1]},
    ])
    print("\n5. Aggregate escrow (multi-item, real adjustment):")
    print(f"   non_aggregate_cushion={aggregate_result['non_aggregate_cushion']}, "
          f"aggregate_cushion={aggregate_result['aggregate_cushion']}, "
          f"aggregate_adjustment={aggregate_result['aggregate_adjustment']}")

    tax_result = validate_property_taxes(
        {"annual_taxes": 7_200},
        {"transaction_type": "refinance", "proposed_tax_amount": 7_200},
    )
    print(f"\n6. Property tax validation (expected {tax_result['expected_amount']} via "
          f"'{tax_result['method']}'):")
    for f in tax_result["flags"]:
        print(f"   - [{f['severity']}] {f['title']}: {f['details']}")

    # Wire-fraud check: beneficiary name does NOT match the title agent on
    # purpose, to demonstrate the critical flag firing.
    wire_result = validate_wire_instructions(
        {"beneficiary_name": "Suspicious Holdings LLC"}, TITLE_FIELDS
    )
    print("\n7. Wire instructions check (intentionally mismatched beneficiary):")
    for f in wire_result["flags"]:
        print(f"   - [{f['severity']}] {f['title']}: {f['details']}")


if __name__ == "__main__":
    main()

# --- To run the real LLM-orchestrated agent instead ---
#
#   export OPENAI_API_KEY=sk-...
#   from title_insurance_tax_agent import build_title_insurance_tax_agent
#
#   agent = build_title_insurance_tax_agent(efolder_index=EFOLDER_INDEX)
#   result = agent.invoke({"messages": [("user",
#       f"Review the title commitment and HOI policy for loan {LOAN_GUID}. "
#       f"Loan context: {LOAN_CONTEXT}. Title fields: {TITLE_FIELDS}. "
#       f"HOI fields: {HOI_FIELDS}.")]})
#   print(result["messages"][-1].content)
