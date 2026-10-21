import pytest

from title_insurance_tax_agent.tools import (
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


# ---------------------------------------------------------------------------
# search_documents (T1 MVP: bucket/keyword matching, not semantic search)
# ---------------------------------------------------------------------------

EFOLDER_INDEX = {
    "Title Commitment": [{"doc_id": "d1", "filename": "title_commitment.pdf", "text": "title report order #12345"}],
    "Evidence of Insurance": [{"doc_id": "d2", "filename": "hoi_dec_page.pdf", "text": "homeowner hazard policy"}],
    "Unassigned": [{"doc_id": "d3", "filename": "random.pdf", "text": "irrelevant content"}],
}


def test_search_documents_matches_known_bucket():
    results = search_documents("title commitment", "guid-1", EFOLDER_INDEX)
    assert any(r["doc_id"] == "d1" for r in results)


def test_search_documents_matches_by_keyword_in_unlabeled_bucket():
    results = search_documents("hazard", "guid-1", EFOLDER_INDEX)
    assert any(r["doc_id"] == "d2" for r in results)


def test_search_documents_no_match_returns_empty():
    # Deliberately avoids any word that appears in the fixture bucket names,
    # filenames, or text (e.g. "report" would spuriously match "title report").
    results = search_documents("comparable sales grid", "guid-1", EFOLDER_INDEX)
    assert results == []


def test_search_documents_avoid_checks_filename_not_body_text():
    """Regression test: a real title commitment's own chain-of-title
    narrative legitimately mentions prior recorded deeds, but
    title_report.json's attachment_avoid list includes "deed" (meant to
    filter out an actual standalone Deed document sharing the same eFolder
    bucket). The avoid-check must only apply to the filename, or every
    title commitment that discusses its chain of title gets wrongly
    excluded. Found via examples/run_semantic_search_demo.py."""
    efolder_index = {
        "Title Report": [{
            "doc_id": "tc1",
            "filename": "Title Commitment.pdf",
            "text": "This commitment traces the chain of title through a Warranty Deed "
                    "recorded in 2006 and a Deed of Trust recorded in 2019.",
        }],
    }
    results = search_documents("title commitment", "guid-1", efolder_index)
    assert any(r["doc_id"] == "tc1" for r in results), (
        "A legitimate title commitment whose body text mentions 'deed' must still be found."
    )


def test_search_documents_disambiguates_shared_bucket_by_doc_type():
    """'Property: Title Company Documents' is a single Encompass eFolder
    bucket shared by CPL, E&O, AND Wire Instructions documents. Each
    document in the bucket must resolve to its OWN doc type (and have its
    OWN avoid-list applied), not whichever schema happened to claim the
    bucket name first. Found via examples/run_semantic_search_demo.py."""
    efolder_index = {
        "Property: Title Company Documents": [
            {"doc_id": "eo1", "filename": "Errors and Omissions Policy.pdf",
             "text": "E&O coverage $1,000,000. Does not provide closing protection."},
            {"doc_id": "wi1", "filename": "Wire Instructions.pdf",
             "text": "Beneficiary: Brown County Abstract Company. ABA routing number provided."},
        ],
    }

    wire_results = search_documents("wire instructions", "guid-1", efolder_index)
    assert len(wire_results) == 1
    assert wire_results[0]["doc_id"] == "wi1"
    assert wire_results[0]["document_type"] == "wire_instructions"

    eo_results = search_documents("errors and omissions", "guid-1", efolder_index)
    assert len(eo_results) == 1
    assert eo_results[0]["doc_id"] == "eo1"
    assert eo_results[0]["document_type"] == "errors_omissions"


def test_search_documents_finds_cpl_eo_and_wire_docs_now_registered():
    """CPL/E&O/Wire Instructions/Insurance Receipt were added to
    validate_closing_agent_match / validate_wire_instructions /
    validate_premium_paid_in_full but were missing from
    DOCUMENT_TYPE_INDEX entirely, so search_documents couldn't find them
    by bucket name at all."""
    efolder_index = {
        "Closing Protection Letter": [{"doc_id": "cpl1", "filename": "CPL.pdf", "text": "Closing protection letter."}],
    }
    results = search_documents("closing protection letter", "guid-1", efolder_index)
    assert any(r["doc_id"] == "cpl1" and r["document_type"] == "cpl" for r in results)


def test_search_documents_avoid_still_excludes_misfiled_deed_by_filename():
    """The avoid-check should still do its job when the avoid-word is
    actually in the filename (the case it was designed for: an actual
    recorded Deed document misfiled into the Title Report bucket)."""
    efolder_index = {
        "Title Report": [{
            "doc_id": "deed1",
            "filename": "Recorded Deed.pdf",
            "text": "Warranty deed conveying the property.",
        }],
    }
    results = search_documents("title commitment", "guid-1", efolder_index)
    assert results == []


# ---------------------------------------------------------------------------
# validate_title_commitment (T2 minus chain of title)
# ---------------------------------------------------------------------------

def test_validate_title_commitment_flags_coverage_below_loan_amount():
    fields = {"lender_coverage": 350_000, "effective_date": "2026-01-01"}
    loan_context = {"loan_amount": 400_000, "estimated_closing_date": "2026-01-15", "property_state": "CA"}
    result = validate_title_commitment(fields, loan_context)
    coverage_flags = [f for f in result["flags"] if f["field"] == "lender_coverage"]
    assert coverage_flags[0]["severity"] == "warning"
    assert result["chain_of_title_verified"] is False


def test_validate_title_commitment_coverage_adequate():
    fields = {"lender_coverage": 400_000, "effective_date": "2026-01-01"}
    loan_context = {"loan_amount": 400_000, "estimated_closing_date": "2026-01-15", "property_state": "TX"}
    result = validate_title_commitment(fields, loan_context)
    coverage_flags = [f for f in result["flags"] if f["field"] == "lender_coverage"]
    assert coverage_flags[0]["severity"] == "info"


def test_validate_title_commitment_effective_date_too_old_is_critical():
    fields = {"lender_coverage": 400_000, "effective_date": "2025-01-01"}
    loan_context = {"loan_amount": 400_000, "estimated_closing_date": "2026-01-15", "property_state": "NY"}
    result = validate_title_commitment(fields, loan_context)
    date_flags = [f for f in result["flags"] if f["field"] == "effective_date"]
    assert date_flags[0]["severity"] == "critical"


def test_validate_title_commitment_effective_date_within_window():
    fields = {"lender_coverage": 400_000, "effective_date": "2026-01-01"}
    loan_context = {"loan_amount": 400_000, "estimated_closing_date": "2026-01-15", "property_state": "FL"}
    result = validate_title_commitment(fields, loan_context)
    date_flags = [f for f in result["flags"] if f["field"] == "effective_date"]
    assert date_flags[0]["severity"] == "info"


def test_validate_title_commitment_never_claims_chain_of_title():
    result = validate_title_commitment({}, {})
    assert result["chain_of_title_verified"] is False
    assert "chain_of_title_note" in result
    assert "chain_of_title" not in result["checks_performed"]


# ---------------------------------------------------------------------------
# validate_insurance_coverage (T4 + J-T2)
# ---------------------------------------------------------------------------

def test_validate_insurance_coverage_deductible_exceeds_5_percent():
    hoi = {"coverage_amount": 400_000, "deductible": 25_000, "effective_date": "2026-01-01",
           "expiration_date": "2027-01-01"}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000})
    deductible_flags = [f for f in result["flags"] if f["field"] == "deductible"]
    assert deductible_flags[0]["severity"] == "critical"


def test_validate_insurance_coverage_deductible_within_5_percent():
    hoi = {"coverage_amount": 400_000, "deductible": 10_000, "effective_date": "2026-01-01",
           "expiration_date": "2027-01-01"}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000})
    deductible_flags = [f for f in result["flags"] if f["field"] == "deductible"]
    assert deductible_flags[0]["severity"] == "info"


def test_validate_insurance_coverage_below_loan_amount_is_warning():
    hoi = {"coverage_amount": 250_000, "deductible": 5_000, "effective_date": "2026-01-01",
           "expiration_date": "2027-01-01"}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000})
    coverage_flags = [f for f in result["flags"] if f["field"] == "coverage_amount"]
    assert coverage_flags[0]["severity"] == "warning"


def test_validate_insurance_coverage_term_not_12_months_is_warning():
    hoi = {"coverage_amount": 400_000, "deductible": 10_000, "effective_date": "2026-01-01",
           "expiration_date": "2026-07-01"}  # 6 month term
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000})
    term_flags = [f for f in result["flags"] if f["field"] == "policy_term"]
    assert term_flags[0]["severity"] == "warning"


def test_validate_insurance_coverage_term_12_months_is_ok():
    hoi = {"coverage_amount": 400_000, "deductible": 10_000, "effective_date": "2026-01-01",
           "expiration_date": "2027-01-01"}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000})
    term_flags = [f for f in result["flags"] if f["field"] == "policy_term"]
    assert term_flags[0]["severity"] == "info"


def test_validate_insurance_coverage_missing_hoi_is_critical():
    result = validate_insurance_coverage(None, {"loan_amount": 300_000})
    assert result["flags"][0]["severity"] == "critical"


def test_validate_insurance_coverage_flood_deductible_standard_cap():
    hoi = {"coverage_amount": 400_000, "deductible": 10_000, "effective_date": "2026-01-01",
           "expiration_date": "2027-01-01"}
    flood = {"deductible": 15_000}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000, "property_type": "SFR"}, flood)
    flood_flags = [f for f in result["flags"] if f["field"] == "flood_deductible"]
    assert flood_flags[0]["severity"] == "critical"  # 15K > 10K standard cap


def test_validate_insurance_coverage_flood_deductible_condo_pud_cap():
    hoi = {"coverage_amount": 400_000, "deductible": 10_000, "effective_date": "2026-01-01",
           "expiration_date": "2027-01-01"}
    flood = {"deductible": 15_000}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000, "property_type": "Condo"}, flood)
    flood_flags = [f for f in result["flags"] if f["field"] == "flood_deductible"]
    assert flood_flags[0]["severity"] == "info"  # 15K <= 25K condo/pud cap


def test_validate_insurance_coverage_flood_zone_without_flood_policy_is_critical():
    hoi = {"coverage_amount": 400_000, "deductible": 10_000, "effective_date": "2026-01-01",
           "expiration_date": "2027-01-01"}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000, "is_flood_zone": True})
    flood_flags = [f for f in result["flags"] if f["field"] == "flood_insurance"]
    assert flood_flags[0]["severity"] == "critical"


def test_validate_insurance_coverage_in_sfha_from_document_overrides_missing_loan_context_flag():
    """A flood determination's own in_sfha=True should trigger the
    required-but-missing flag even when loan_context never set
    is_flood_zone at all (previously this check only fired if an external
    caller remembered to pass is_flood_zone)."""
    hoi = {"coverage_amount": 400_000, "deductible": 10_000, "effective_date": "2026-01-01",
           "expiration_date": "2027-01-01"}
    flood = {"in_sfha": True}  # determination present, but no actual flood POLICY (no deductible)
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000}, flood)
    flood_flags = [f for f in result["flags"] if f["field"] == "flood_insurance"]
    assert flood_flags[0]["severity"] == "critical"


def test_validate_insurance_coverage_in_sfha_false_overrides_stale_loan_context_flag():
    """A flood determination's own in_sfha=False should suppress the
    required-but-missing flag even if an external (stale) loan_context
    still says is_flood_zone=True -- the document is the authoritative,
    current source."""
    hoi = {"coverage_amount": 400_000, "deductible": 10_000, "effective_date": "2026-01-01",
           "expiration_date": "2027-01-01"}
    flood = {"in_sfha": False}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000, "is_flood_zone": True}, flood)
    flood_flags = [f for f in result["flags"] if f["field"] == "flood_insurance"]
    assert flood_flags[0]["severity"] == "info"
    assert "not required" in flood_flags[0]["title"].lower()


def test_validate_insurance_coverage_in_sfha_as_string_false_is_not_required():
    """Regression test for core.booleans.as_bool(): before it existed,
    `in_sfha = flood_fields.get("in_sfha")` would leave in_sfha as the
    literal string "False", and the later `elif in_sfha is False:` identity
    check would never match a string -- silently dropping the "not
    required" signal and falling through to treat it as flood-zone-unknown."""
    hoi = {"coverage_amount": 400_000, "deductible": 10_000, "effective_date": "2026-01-01",
           "expiration_date": "2027-01-01"}
    flood = {"in_sfha": "False"}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000, "is_flood_zone": True}, flood)
    flood_flags = [f for f in result["flags"] if f["field"] == "flood_insurance"]
    assert flood_flags[0]["severity"] == "info"
    assert "not required" in flood_flags[0]["title"].lower()


def test_validate_insurance_coverage_hoi_coverage_uses_lesser_of_loan_or_replacement_cost():
    # Replacement cost (250K) is LOWER than loan amount (300K) -- coverage of
    # 275K satisfies the loan-amount-only proxy but should now fail against
    # the real "lesser of" minimum (250K)... wait, 275K > 250K, so it passes.
    # Use a replacement cost that's actually binding:
    hoi = {"coverage_amount": 275_000, "deductible": 10_000, "effective_date": "2026-01-01",
           "expiration_date": "2027-01-01", "replacement_cost": 290_000}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000})
    coverage_flags = [f for f in result["flags"] if f["field"] == "coverage_amount"]
    # required minimum = min(300_000, 290_000) = 290_000; coverage 275_000 < 290_000
    assert coverage_flags[0]["severity"] == "warning"
    assert "290,000" in coverage_flags[0]["details"] or "290000" in coverage_flags[0]["details"]


def test_validate_insurance_coverage_hoi_coverage_falls_back_to_loan_amount_without_replacement_cost():
    hoi = {"coverage_amount": 310_000, "deductible": 10_000, "effective_date": "2026-01-01",
           "expiration_date": "2027-01-01"}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000})
    coverage_flags = [f for f in result["flags"] if f["field"] == "coverage_amount"]
    assert coverage_flags[0]["severity"] == "info"
    assert "replacement cost not extracted" in coverage_flags[0]["details"]


def test_validate_insurance_coverage_loss_payee_clause_matches_expected():
    expected = "NQM Funding, LLC. ISAOA/ATIMA, 4800 N Federal Hwy, Bldg. E, Suite 200, Boca Raton, FL 33431 (include loan number)"
    hoi = {"coverage_amount": 400_000, "deductible": 10_000, "effective_date": "2026-01-01",
           "expiration_date": "2027-01-01", "loss_payee_clause": expected}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000, "property_state": "WA"})
    clause_flags = [f for f in result["flags"] if f["field"] == "loss_payee_clause"]
    assert clause_flags[0]["severity"] == "info"


def test_validate_insurance_coverage_loss_payee_clause_mismatch_is_flagged():
    hoi = {"coverage_amount": 400_000, "deductible": 10_000, "effective_date": "2026-01-01",
           "expiration_date": "2027-01-01", "loss_payee_clause": "Some Random Bank, its successors and/or assigns"}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000, "property_state": "WA"})
    clause_flags = [f for f in result["flags"] if f["field"] == "loss_payee_clause"]
    assert clause_flags[0]["severity"] == "warning"


# ---------------------------------------------------------------------------
# calculate_escrow (T3 — the one spec tool explicitly called deterministic)
# ---------------------------------------------------------------------------

def test_calculate_escrow_matches_architecture_doc_sketch():
    result = calculate_escrow(property_taxes=6000, insurance_premium=1200, months=2)
    assert result["monthly_tax"] == 500.0
    assert result["monthly_insurance"] == 100.0
    assert result["cushion_months"] == 2
    assert result["cushion_amount"] == 1200.0  # (500 + 100) * 2
    assert result["total_monthly_escrow"] == 600.0


def test_calculate_escrow_default_months_is_two():
    result = calculate_escrow(property_taxes=1200, insurance_premium=600)
    assert result["cushion_months"] == 2


def test_calculate_escrow_rejects_negative_inputs():
    with pytest.raises(ValueError):
        calculate_escrow(property_taxes=-1, insurance_premium=100)
    with pytest.raises(ValueError):
        calculate_escrow(property_taxes=100, insurance_premium=-1)


def test_calculate_escrow_rejects_non_positive_months():
    with pytest.raises(ValueError):
        calculate_escrow(property_taxes=100, insurance_premium=100, months=0)


def test_calculate_escrow_includes_jt9_caveat_about_aggregate_adjustment():
    result = calculate_escrow(property_taxes=1200, insurance_premium=600)
    assert "aggregate adjustment" in result["caveat"]


# ---------------------------------------------------------------------------
# validate_title_commitment — 3-way effective date consistency (J-T1)
# ---------------------------------------------------------------------------

def test_title_commitment_3way_dates_consistent():
    fields = {"lender_coverage": 400_000, "effective_date": "2026-01-01"}
    loan_context = {"loan_amount": 400_000, "estimated_closing_date": "2026-01-15", "property_state": "CA"}
    result = validate_title_commitment(fields, loan_context,
                                        doc_expiration_screen_date="2026-01-01",
                                        expiration_condition_date="2026-01-01")
    consistency_flags = [f for f in result["flags"] if f["field"] == "effective_date_consistency"]
    assert consistency_flags[0]["severity"] == "info"


def test_title_commitment_3way_dates_mismatch():
    fields = {"lender_coverage": 400_000, "effective_date": "2026-01-01"}
    loan_context = {"loan_amount": 400_000, "estimated_closing_date": "2026-01-15", "property_state": "CA"}
    result = validate_title_commitment(fields, loan_context,
                                        doc_expiration_screen_date="2025-12-01",
                                        expiration_condition_date="2026-01-01")
    consistency_flags = [f for f in result["flags"] if f["field"] == "effective_date_consistency"]
    assert consistency_flags[0]["severity"] == "warning"


def test_title_commitment_3way_dates_not_provided_is_warned_not_skipped():
    fields = {"lender_coverage": 400_000, "effective_date": "2026-01-01"}
    loan_context = {"loan_amount": 400_000, "estimated_closing_date": "2026-01-15", "property_state": "CA"}
    result = validate_title_commitment(fields, loan_context)
    consistency_flags = [f for f in result["flags"] if f["field"] == "effective_date_consistency"]
    assert consistency_flags[0]["severity"] == "warning"


# ---------------------------------------------------------------------------
# validate_insurance_coverage — Wind/Hail (J-T5), DSCR (J-T7), HO6 (J-T8)
# ---------------------------------------------------------------------------

def _base_hoi():
    return {"coverage_amount": 400_000, "deductible": 10_000, "effective_date": "2026-01-01",
            "expiration_date": "2027-01-01"}


def test_wind_hail_excluded_is_critical():
    hoi = {**_base_hoi(), "wind_hail_hurricane_included": "excluded"}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000})
    flags = [f for f in result["flags"] if f["field"] == "wind_hail_hurricane_included"]
    assert flags[0]["severity"] == "critical"


def test_wind_hail_included_is_ok():
    hoi = {**_base_hoi(), "wind_hail_hurricane_included": "included"}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000})
    flags = [f for f in result["flags"] if f["field"] == "wind_hail_hurricane_included"]
    assert flags[0]["severity"] == "info"


def test_dscr_actual_loss_sustained_is_critical():
    hoi = {**_base_hoi(), "dscr_rent_loss_coverage_type": "actual_loss_sustained"}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000, "is_dscr": True})
    flags = [f for f in result["flags"] if f["field"] == "dscr_rent_loss_coverage_type"]
    assert flags[0]["severity"] == "critical"


def test_dscr_scheduled_rent_below_6_months_is_critical():
    hoi = {**_base_hoi(), "dscr_rent_loss_coverage_type": "scheduled_rent", "dscr_rent_loss_months": 3}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000, "is_dscr": True})
    flags = [f for f in result["flags"] if f["field"] == "dscr_rent_loss_coverage_type"]
    assert flags[0]["severity"] == "critical"


def test_dscr_scheduled_rent_6_months_is_ok():
    hoi = {**_base_hoi(), "dscr_rent_loss_coverage_type": "scheduled_rent", "dscr_rent_loss_months": 6}
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000, "is_dscr": True})
    flags = [f for f in result["flags"] if f["field"] == "dscr_rent_loss_coverage_type"]
    assert flags[0]["severity"] == "info"


def test_dscr_check_skipped_when_not_a_dscr_loan():
    hoi = _base_hoi()
    result = validate_insurance_coverage(hoi, {"loan_amount": 300_000, "is_dscr": False})
    flags = [f for f in result["flags"] if f["field"] == "dscr_rent_loss_coverage_type"]
    assert flags == []


def test_ho6_deductible_exceeds_5_percent_of_own_coverage():
    ho6 = {"ho6_dwelling_coverage_amount": 100_000, "deductible": 10_000}
    result = validate_insurance_coverage(_base_hoi(), {"loan_amount": 300_000}, ho6_fields=ho6)
    flags = [f for f in result["flags"] if f["field"] == "ho6_deductible"]
    assert flags[0]["severity"] == "critical"


def test_ho6_deductible_within_cap():
    ho6 = {"ho6_dwelling_coverage_amount": 100_000, "deductible": 4_000}
    result = validate_insurance_coverage(_base_hoi(), {"loan_amount": 300_000}, ho6_fields=ho6)
    flags = [f for f in result["flags"] if f["field"] == "ho6_deductible"]
    assert flags[0]["severity"] == "info"


# ---------------------------------------------------------------------------
# validate_insurance_coverage — flood "lesser of 3" rule (J-T6)
# ---------------------------------------------------------------------------

def test_flood_lesser_of_3_coverage_below_minimum():
    flood = {"flood_insurance_coverage": 150_000, "deductible": 5_000}
    loan_context = {"loan_amount": 300_000, "unpaid_principal_balance": 280_000,
                     "nfip_max": 250_000}
    result = validate_insurance_coverage(_base_hoi(), loan_context, flood)
    flags = [f for f in result["flags"] if f["field"] == "flood_coverage_lesser_of_3"]
    assert flags[0]["severity"] == "critical"  # 150K < min(280K, 250K)=250K


def test_flood_lesser_of_3_coverage_meets_minimum():
    flood = {"flood_insurance_coverage": 260_000, "deductible": 5_000}
    loan_context = {"loan_amount": 300_000, "unpaid_principal_balance": 280_000,
                     "nfip_max": 250_000}
    result = validate_insurance_coverage(_base_hoi(), loan_context, flood)
    flags = [f for f in result["flags"] if f["field"] == "flood_coverage_lesser_of_3"]
    assert flags[0]["severity"] == "info"  # 260K >= min(280K, 250K)=250K


def test_flood_lesser_of_3_uses_default_nfip_max_when_not_provided():
    flood = {"flood_insurance_coverage": 200_000, "deductible": 5_000}
    loan_context = {"loan_amount": 300_000, "unpaid_principal_balance": 500_000}
    result = validate_insurance_coverage(_base_hoi(), loan_context, flood)
    flags = [f for f in result["flags"] if f["field"] == "flood_coverage_lesser_of_3"]
    assert flags[0]["severity"] == "critical"  # 200K < default NFIP max 250K


# ---------------------------------------------------------------------------
# validate_premium_paid_in_full (J-T3)
# ---------------------------------------------------------------------------

def test_premium_paid_in_full_matches_policy():
    receipt = {"amount": 1200, "paid_in_full": True}
    policy = {"hazard_insurance_premium": 1200}
    result = validate_premium_paid_in_full(receipt, policy, {"transaction_type": "purchase"})
    assert result["flags"][0]["severity"] == "info"


def test_premium_receipt_missing_is_warning():
    result = validate_premium_paid_in_full(None, {"hazard_insurance_premium": 1200}, {})
    assert result["flags"][0]["severity"] == "warning"


def test_premium_invoice_only_refinance_within_60_days_is_critical():
    receipt = {"amount": 1200, "paid_in_full": False}
    policy = {"hazard_insurance_premium": 1200}
    loan_context = {"transaction_type": "refinance", "days_to_closing": 30}
    result = validate_premium_paid_in_full(receipt, policy, loan_context)
    assert result["flags"][0]["severity"] == "critical"


def test_premium_invoice_only_purchase_is_acceptable():
    receipt = {"amount": 1200, "paid_in_full": False}
    policy = {"hazard_insurance_premium": 1200}
    loan_context = {"transaction_type": "purchase"}
    result = validate_premium_paid_in_full(receipt, policy, loan_context)
    assert result["flags"][0]["severity"] == "info"


def test_premium_paid_in_full_treats_string_false_as_not_paid():
    """Regression test for core.booleans.as_bool(): paid_in_full extracted
    as the literal string "False" must route through the invoice branch,
    not the paid-in-full branch (bool("False") is True in Python)."""
    receipt = {"amount": 1200, "paid_in_full": "False"}
    policy = {"hazard_insurance_premium": 1200}
    loan_context = {"transaction_type": "refinance", "days_to_closing": 30}
    result = validate_premium_paid_in_full(receipt, policy, loan_context)
    assert result["flags"][0]["severity"] == "critical"
    assert "within 60 days" in result["flags"][0]["title"].lower()


def test_premium_paid_in_full_treats_string_true_as_paid():
    receipt = {"amount": 1200, "paid_in_full": "True"}
    policy = {"hazard_insurance_premium": 1200}
    result = validate_premium_paid_in_full(receipt, policy, {"transaction_type": "purchase"})
    assert result["flags"][0]["severity"] == "info"
    assert "paid in full" in result["flags"][0]["title"].lower()


# ---------------------------------------------------------------------------
# validate_no_coverage_gap (J-T4)
# ---------------------------------------------------------------------------

def test_no_coverage_gap_detects_gap():
    old_policy = {"expiration_date": "2026-01-01"}
    new_policy = {"effective_date": "2026-01-15"}
    result = validate_no_coverage_gap(old_policy, new_policy, {"note_date": "2026-01-15"})
    gap_flags = [f for f in result["flags"] if f["field"] == "coverage_gap"]
    assert gap_flags[0]["severity"] == "critical"


def test_no_coverage_gap_continuous_coverage():
    old_policy = {"expiration_date": "2026-01-15"}
    new_policy = {"effective_date": "2026-01-10"}
    result = validate_no_coverage_gap(old_policy, new_policy, {"note_date": "2026-01-15"})
    gap_flags = [f for f in result["flags"] if f["field"] == "coverage_gap"]
    assert gap_flags[0]["severity"] == "info"


def test_no_coverage_gap_binder_too_far_before_note_date():
    old_policy = {"expiration_date": "2025-12-01"}
    new_policy = {"effective_date": "2025-12-01", "paid_in_full": False}
    result = validate_no_coverage_gap(old_policy, new_policy, {"note_date": "2026-01-15"})
    binder_flags = [f for f in result["flags"] if f["field"] == "binder_timing"]
    assert len(binder_flags) == 1
    assert binder_flags[0]["severity"] == "warning"


# ---------------------------------------------------------------------------
# validate_closing_agent_match (J-T10)
# ---------------------------------------------------------------------------

def test_closing_agent_match_correspondent_not_applicable():
    result = validate_closing_agent_match({}, {}, {}, {"channel": "correspondent"})
    assert result["flags"][0]["severity"] == "info"


def test_closing_agent_match_all_names_match():
    title = {"settlement_agent_name": "ABC Title Co"}
    cpl = {"agent_name": "ABC Title Co", "effective_date": "2026-01-01"}
    eo = {"agent_name": "ABC Title Co", "coverage_amount": 1_500_000}
    result = validate_closing_agent_match(title, cpl, eo, {"estimated_closing_date": "2026-01-10"})
    match_flags = [f for f in result["flags"] if f["field"] == "closing_agent_match"]
    assert match_flags[0]["severity"] == "info"


def test_closing_agent_match_name_mismatch_is_critical():
    title = {"settlement_agent_name": "ABC Title Co"}
    cpl = {"agent_name": "XYZ Title Co"}
    result = validate_closing_agent_match(title, cpl, None)
    match_flags = [f for f in result["flags"] if f["field"] == "closing_agent_match"]
    assert match_flags[0]["severity"] == "critical"


def test_closing_agent_match_eo_coverage_below_minimum():
    title = {"settlement_agent_name": "ABC Title Co"}
    eo = {"agent_name": "ABC Title Co", "coverage_amount": 400_000}
    result = validate_closing_agent_match(title, None, eo)
    eo_flags = [f for f in result["flags"] if f["field"] == "eo_coverage"]
    assert eo_flags[0]["severity"] == "critical"


def test_closing_agent_match_eo_coverage_meets_attorney_minimum():
    title = {"settlement_agent_name": "Jane Attorney"}
    eo = {"agent_name": "Jane Attorney", "coverage_amount": 500_000, "is_attorney": True}
    result = validate_closing_agent_match(title, None, eo)
    eo_flags = [f for f in result["flags"] if f["field"] == "eo_coverage"]
    assert eo_flags[0]["severity"] == "info"


# ---------------------------------------------------------------------------
# validate_wire_instructions (J-T11)
# ---------------------------------------------------------------------------

def test_wire_instructions_match_is_safe():
    result = validate_wire_instructions({"beneficiary_name": "ABC Title Co"},
                                         {"settlement_agent_name": "ABC Title Co"})
    assert result["flags"][0]["severity"] == "info"


def test_wire_instructions_mismatch_is_fraud_risk():
    result = validate_wire_instructions({"beneficiary_name": "Totally Different LLC"},
                                         {"settlement_agent_name": "ABC Title Co"})
    assert result["flags"][0]["severity"] == "critical"
    assert "FRAUD" in result["flags"][0]["title"]


def test_wire_instructions_missing_data_never_silently_passes():
    result = validate_wire_instructions({}, {})
    assert result["flags"][0]["severity"] == "critical"


# ---------------------------------------------------------------------------
# validate_ny_closing_attorney (J-T12)
# ---------------------------------------------------------------------------

def test_ny_attorney_not_applicable_outside_ny():
    result = validate_ny_closing_attorney("Jane Attorney", ["Jane Attorney"], property_state="TX")
    assert result["flags"][0]["severity"] == "info"


def test_ny_attorney_approved():
    result = validate_ny_closing_attorney("Jane Attorney", ["Jane Attorney", "John Lawyer"], property_state="NY")
    assert result["flags"][0]["severity"] == "info"


def test_ny_attorney_not_approved():
    result = validate_ny_closing_attorney("Unknown Esq", ["Jane Attorney"], property_state="NY")
    assert result["flags"][0]["severity"] == "critical"


# ---------------------------------------------------------------------------
# validate_property_taxes (T4 — the biggest gap)
# ---------------------------------------------------------------------------

def test_property_taxes_new_construction_uses_title_confirmed_millage_rate():
    loan_context = {
        "transaction_type": "new_construction",
        "title_confirmed_millage_rate": 0.02,
        "sales_price": 500_000,
        "appraised_value": 510_000,
        "proposed_tax_amount": 10_200,
    }
    result = validate_property_taxes({"annual_taxes": 10_200}, loan_context)
    assert result["expected_amount"] == pytest.approx(0.02 * 510_000)
    assert "millage rate" in result["method"]
    tax_flags = [f for f in result["flags"] if f["field"] == "property_tax_amount"]
    assert tax_flags[0]["severity"] == "info"


def test_property_taxes_new_construction_falls_back_to_1_5_percent():
    loan_context = {"transaction_type": "new_construction", "sales_price": 400_000, "appraised_value": 390_000}
    result = validate_property_taxes({}, loan_context)
    assert result["expected_amount"] == pytest.approx(0.015 * 400_000)


def test_property_taxes_ca_purchase_uses_title_confirmed_amount():
    loan_context = {"transaction_type": "purchase", "property_state": "CA",
                     "title_confirmed_tax_amount": 6_000, "proposed_tax_amount": 6_000}
    result = validate_property_taxes({}, loan_context)
    assert result["expected_amount"] == 6_000
    assert result["method"] == "title-confirmed tax amount"


def test_property_taxes_ca_purchase_uses_1_25_percent_of_assessed_value():
    loan_context = {"transaction_type": "purchase", "property_state": "CA",
                     "revised_assessed_value": 480_000}
    result = validate_property_taxes({}, loan_context)
    assert result["expected_amount"] == pytest.approx(0.0125 * 480_000)


def test_property_taxes_purchase_with_exemption_flags_prior_owner_homestead():
    loan_context = {"transaction_type": "purchase", "has_property_exemption": True,
                     "is_primary_residence": False}
    tax_fields = {"annual_taxes": 4_000, "exemptions": ["homestead"]}
    result = validate_property_taxes(tax_fields, loan_context)
    exemption_flags = [f for f in result["flags"] if f["field"] == "property_tax_exemption"]
    assert exemption_flags[0]["severity"] == "warning"
    assert "homestead" in exemption_flags[0]["details"].lower() or "['homestead']" in exemption_flags[0]["details"]


def test_property_taxes_ordinary_purchase_uses_current_tax_bill():
    """Non-CA, non-exemption, non-new-construction purchase (the most common
    real-world case) -- confirmed by replaying LG-docsOrch golden fixtures,
    which have no handling before this test was added. See
    test_golden_fixtures.py."""
    loan_context = {"transaction_type": "purchase", "property_state": "TX"}
    result = validate_property_taxes({"annual_taxes": 2_287.12}, loan_context)
    assert result["expected_amount"] == pytest.approx(2_287.12)
    assert result["method"] == "current tax bill amount"
    tax_flags = [f for f in result["flags"] if f["field"] == "property_tax_amount"]
    assert tax_flags[0]["severity"] == "info"


def test_property_taxes_ordinary_purchase_without_bill_falls_back_to_smartasset():
    loan_context = {"transaction_type": "purchase", "property_state": "TX", "smartasset_estimate": 2_100}
    result = validate_property_taxes({}, loan_context)
    assert result["expected_amount"] == 2_100
    assert "smartasset" in result["method"]


def test_property_taxes_ordinary_purchase_without_bill_or_estimate_is_ptf():
    loan_context = {"transaction_type": "purchase", "property_state": "TX"}
    result = validate_property_taxes({}, loan_context)
    assert result["expected_amount"] is None
    purchase_flags = [f for f in result["flags"] if f["field"] == "property_tax_purchase"]
    assert purchase_flags[0]["action"] == "ptf"


def test_property_taxes_refinance_requires_current_tax_bill():
    loan_context = {"transaction_type": "refinance"}
    result = validate_property_taxes({}, loan_context)
    refi_flags = [f for f in result["flags"] if f["field"] == "property_tax_refinance"]
    assert refi_flags[0]["severity"] == "critical"


def test_property_taxes_refinance_with_current_bill_is_fine():
    loan_context = {"transaction_type": "refinance"}
    result = validate_property_taxes({"annual_taxes": 5_000}, loan_context)
    assert result["expected_amount"] == 5_000


def test_property_taxes_60_day_rule_without_current_bill_is_critical():
    loan_context = {"transaction_type": "purchase", "days_to_closing": 30}
    result = validate_property_taxes({}, loan_context)
    bill_flags = [f for f in result["flags"] if f["field"] == "current_tax_bill"]
    assert bill_flags[0]["severity"] == "critical"


def test_property_taxes_delinquent_flag():
    result = validate_property_taxes({"annual_taxes": 5000, "is_delinquent_60_plus_days": True},
                                      {"transaction_type": "refinance"})
    delinquent_flags = [f for f in result["flags"] if f["field"] == "property_tax_delinquent"]
    assert delinquent_flags[0]["severity"] == "warning"


def test_property_taxes_delinquent_flag_treats_string_false_as_not_delinquent():
    """Regression test for core.booleans.as_bool(): is_delinquent_60_plus_days
    extracted as the string "False" must not raise the delinquency flag."""
    result = validate_property_taxes({"annual_taxes": 5000, "is_delinquent_60_plus_days": "False"},
                                      {"transaction_type": "refinance"})
    delinquent_flags = [f for f in result["flags"] if f["field"] == "property_tax_delinquent"]
    assert delinquent_flags == []


def test_property_taxes_exemption_excluded_treats_string_booleans_correctly():
    """Regression test for core.booleans.as_bool(): has_property_exemption
    and is_primary_residence extracted as literal strings ("True"/"False")
    must route and gate the same as real Python bools. Before the fix,
    `not "False"` was False, which incorrectly suppressed the
    prior-owner-exemption-must-be-excluded warning for a non-primary-
    residence purchase."""
    loan_context = {
        "transaction_type": "purchase",
        "has_property_exemption": "True",
        "is_primary_residence": "False",
    }
    tax_fields = {"annual_taxes": 4_000, "exemptions": ["senior"]}
    result = validate_property_taxes(tax_fields, loan_context)
    assert result["expected_amount"] == 4_000
    exemption_flags = [f for f in result["flags"] if f["field"] == "property_tax_exemption"]
    assert exemption_flags, "expected the prior-owner-exemption warning to fire"
    assert exemption_flags[0]["severity"] == "warning"


def test_property_taxes_mismatch_vs_proposed_amount_flags_overwrite():
    loan_context = {"transaction_type": "refinance", "proposed_tax_amount": 3_000}
    result = validate_property_taxes({"annual_taxes": 5_000}, loan_context)
    tax_flags = [f for f in result["flags"] if f["field"] == "property_tax_amount"]
    assert tax_flags[0]["action"] == "overwrite"


# ---------------------------------------------------------------------------
# calculate_aggregate_escrow (J-T9 — real aggregate adjustment)
# ---------------------------------------------------------------------------

def test_calculate_aggregate_escrow_two_items_negative_adjustment():
    result = calculate_aggregate_escrow([
        {"name": "County Tax", "annual_amount": 6000, "due_months": [12]},
        {"name": "HOI", "annual_amount": 1200, "due_months": [1]},
    ])
    assert result["aggregate_adjustment"] < 0
    assert result["flags"][0]["severity"] == "info"
    assert "negative" in result["flags"][0]["title"].lower()


def test_calculate_aggregate_escrow_single_item_adjustment_is_zero():
    result = calculate_aggregate_escrow([{"name": "Tax", "annual_amount": 6000, "due_months": [12]}])
    assert result["aggregate_adjustment"] == 0


def test_calculate_aggregate_escrow_quarterly_due_dates():
    result = calculate_aggregate_escrow([
        {"name": "Tax", "annual_amount": 4000, "due_months": [3, 6, 9, 12]},
        {"name": "HOI", "annual_amount": 1200, "due_months": [1]},
    ])
    assert result["aggregate_adjustment"] <= 0
    assert result["monthly_deposit"] == pytest.approx((4000 + 1200) / 12, abs=0.01)


def test_calculate_aggregate_escrow_empty_raises():
    with pytest.raises(ValueError):
        calculate_aggregate_escrow([])


def test_calculate_aggregate_escrow_item_without_due_months_raises():
    with pytest.raises(ValueError):
        calculate_aggregate_escrow([{"name": "Tax", "annual_amount": 1000, "due_months": []}])
