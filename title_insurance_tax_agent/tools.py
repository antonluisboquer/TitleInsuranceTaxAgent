"""Pure, directly-testable implementations of the Title/Insurance/Taxes
Agent's 4 spec tools. See title_insurance_tax_agent/agent.py for the
LangGraph tool-calling wrapper that exposes these to an LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.booleans import as_bool
from core.dates import days_between, months_between, parse_date
from core.models import Flag
from core.money import format_currency, to_number
from core.schema_loader import load_schema

TITLE_SCHEMA = load_schema("title_report")
INSURANCE_SCHEMA = load_schema("evidence_of_insurance")
FLOOD_SCHEMA = load_schema("flood_insurance")
TAX_SCHEMA = load_schema("tax_summary")
CPL_SCHEMA = load_schema("cpl")
ERRORS_OMISSIONS_SCHEMA = load_schema("errors_omissions")
WIRE_INSTRUCTIONS_SCHEMA = load_schema("wire_instructions")
INSURANCE_RECEIPT_SCHEMA = load_schema("insurance_receipt")

# ---------------------------------------------------------------------------
# Tool 1/4: search_documents(query, loan_guid)
# ---------------------------------------------------------------------------
#
# Architecture doc item T1: no repo has ChromaDB/vector/embeddings — document
# retrieval everywhere is exact bucket-name/keyword matching
# (EFOLDER_BUCKET_MAPPING in LG-docsOrch's document_type_registry.py). The
# doc explicitly says: "if the Title/Insurance/Taxes Agent genuinely needs
# free-text querying across a loan's docs [build ChromaDB]; otherwise
# reconsider whether bucket-based lookup is sufficient." This MVP implements
# the "sufficient" bucket/keyword version using the same bucket_names /
# attachment_keywords already embedded in each tool_schema/*.json file, and
# does NOT pretend to do semantic search.

DOCUMENT_TYPE_INDEX: Dict[str, Dict[str, Any]] = {
    "title_report": TITLE_SCHEMA,
    "evidence_of_insurance": INSURANCE_SCHEMA,
    "flood_insurance": FLOOD_SCHEMA,
    "tax_summary": TAX_SCHEMA,
    # Added later for J-T10/J-T11/J-T3 (validate_closing_agent_match,
    # validate_wire_instructions, validate_premium_paid_in_full) but
    # initially left out of this index -- meaning search_documents couldn't
    # find CPL/E&O/Wire Instructions/receipt documents at all, even though
    # this same agent validates against them. Fixed alongside the
    # attachment_avoid bug below (found via run_semantic_search_demo.py).
    "cpl": CPL_SCHEMA,
    "errors_omissions": ERRORS_OMISSIONS_SCHEMA,
    "wire_instructions": WIRE_INSTRUCTIONS_SCHEMA,
    "insurance_receipt": INSURANCE_RECEIPT_SCHEMA,
}


def search_documents(
    query: str,
    loan_guid: str,
    efolder_index: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Find documents in a loan's eFolder matching `query`.

    NOT a semantic/vector search (see module docstring — that's backlog
    item T1). Matching is deterministic, two routes:
      1. The query names a known bucket (e.g. "title commitment" ->
         bucket_names in title_report.json) -> every doc in that bucket.
      2. Otherwise, a plain substring match of the query's words against
         each document's own filename/text.
    This is the same "exact bucket-name/keyword matching" approach already
    used in production across all 3 source repos (e.g.
    EFOLDER_BUCKET_MAPPING in LG-docsOrch's document_type_registry.py).

    Args:
        query: free-text query, e.g. "title commitment" or "flood insurance"
        loan_guid: loan identifier (kept in the signature to match the spec;
            this MVP implementation searches within the provided
            `efolder_index` rather than calling out to a live eFolder API)
        efolder_index: {bucket_name: [{"doc_id", "filename", "text"}, ...]}
            representing the loan's eFolder contents, however your
            deployment sources it.

    Returns:
        List of matching documents, each annotated with the document_type
        and bucket_name it matched under.
    """
    query_terms = [t for t in query.lower().split() if t]
    matches: List[Dict[str, Any]] = []

    # Build bucket-name -> [doc_type, ...] lookup once. Buckets can be
    # claimed by MORE THAN ONE doc type -- e.g. "Property: Title Company
    # Documents" legitimately holds CPL, E&O, AND Wire Instructions
    # documents together in Encompass. A naive "first schema to claim a
    # bucket wins" (the original approach) would apply only ONE of those
    # doc types' attachment_avoid lists to every document in the bucket --
    # e.g. if errors_omissions claimed the bucket first, a literal "Wire
    # Instructions.pdf" would get excluded outright by errors_omissions's
    # own avoid-list (which lists "wire" to avoid miscategorizing a wire
    # doc AS an E&O policy), even when the user is explicitly searching for
    # wire instructions. Each doc is now resolved against ALL doc types
    # that claim its bucket, per-document.
    bucket_to_doc_types: Dict[str, List[str]] = {}
    for doc_type, schema in DOCUMENT_TYPE_INDEX.items():
        for bucket_name in schema.get("bucket_names", []):
            bucket_to_doc_types.setdefault(bucket_name.lower(), []).append(doc_type)

    # Each bucket/doc is evaluated exactly once (not once per doc type), so a
    # single document never shows up duplicated under every doc type.
    for bucket_name, docs in efolder_index.items():
        bucket_lower = bucket_name.lower()
        candidate_doc_types = bucket_to_doc_types.get(bucket_lower, [])  # [] if bucket is unrecognized

        for doc in docs:
            filename_haystack = str(doc.get("filename", "")).lower()

            # attachment_avoid is meant to catch a DIFFERENT document type
            # that got misfiled into this bucket (e.g. a standalone recorded
            # Deed sitting in the same "Title Report" bucket as the actual
            # title commitment, or a Wire Instructions doc sitting in the
            # same shared bucket as an E&O policy) -- so it's checked
            # against the document's filename/title ONLY, never its full
            # body text. A real title commitment legitimately discusses
            # prior recorded deeds in its own chain-of-title narrative;
            # checking avoid-words against full text would wrongly exclude
            # it. (Found via examples/run_semantic_search_demo.py while
            # comparing this MVP against a ChromaDB semantic-search POC.)
            #
            # Resolve which (if any) of the bucket's candidate doc types
            # this specific document actually is: a candidate is eliminated
            # if the filename hits THAT candidate's own avoid-list.
            surviving_doc_types = [
                dt for dt in candidate_doc_types
                if not any(
                    a in filename_haystack
                    for a in (x.lower() for x in DOCUMENT_TYPE_INDEX[dt].get("attachment_avoid", []))
                )
            ]
            if candidate_doc_types and not surviving_doc_types:
                continue  # every candidate's avoid-list excluded this doc

            resolved_doc_type: Optional[str] = None
            if len(surviving_doc_types) == 1:
                resolved_doc_type = surviving_doc_types[0]
            elif len(surviving_doc_types) > 1:
                # Still ambiguous (e.g. multiple doc types share a bucket and
                # neither's avoid-list ruled the doc out) -- break the tie
                # using each candidate's attachment_keywords against the
                # filename; fall back to the first candidate if still tied.
                resolved_doc_type = next(
                    (dt for dt in surviving_doc_types
                     if any(k.lower() in filename_haystack
                            for k in DOCUMENT_TYPE_INDEX[dt].get("attachment_keywords", []))),
                    surviving_doc_types[0],
                )

            # Route 1: the bucket is a known bucket for the resolved doc type
            # AND the query names that bucket (e.g. query "title commitment"
            # + bucket "Title Commitment") -> include the doc.
            bucket_named_in_query = resolved_doc_type is not None and any(
                term in bucket_lower for term in query_terms
            )
            # Route 2: regardless of bucket, the query terms appear directly
            # in the document's own filename/text.
            full_haystack = f"{filename_haystack} {doc.get('text', '')}".lower()
            term_hit = bool(query_terms) and any(term in full_haystack for term in query_terms)
            if bucket_named_in_query or term_hit:
                matches.append({"document_type": resolved_doc_type or "unknown", "bucket": bucket_name, **doc})

    return matches


# ---------------------------------------------------------------------------
# Tool 2/4: validate_title_commitment(document)
# ---------------------------------------------------------------------------

# Example mortgagee-clause table (lg-RMOrch / RM Job Aid p.17, NQMF-specific).
# This is organization-specific — replace with your own clause table when
# wiring this up for a different lender/client.
DEFAULT_MORTGAGEE_CLAUSES = {
    "NY": "Great Home Mortgage of New York, in lieu of true name NP, Inc. ISAOA/ATIMA, "
          "4800 N Federal Hwy, Bldg. E, Suite 200, Boca Raton, FL 33431 (include loan number)",
    "TX": "ISAOA/ATIMA clause NOT required for TX loans.",
    "_default": "NQM Funding, LLC. ISAOA/ATIMA, 4800 N Federal Hwy, Bldg. E, Suite 200, "
                "Boca Raton, FL 33431 (include loan number)",
}

TITLE_EFFECTIVE_DATE_MAX_DAYS = 90  # RM Job Aid p.17 / lg-RMOrch review_preliminary_title


def _mortgagee_clause(prop_state: str, clauses: Dict[str, str]) -> str:
    return clauses.get((prop_state or "").upper(), clauses.get("_default", "n/a"))


def validate_title_commitment(
    document_fields: Dict[str, Any],
    loan_context: Dict[str, Any],
    mortgagee_clauses: Optional[Dict[str, str]] = None,
    doc_expiration_screen_date: Optional[Any] = None,
    expiration_condition_date: Optional[Any] = None,
) -> Dict[str, Any]:
    """Validate a title commitment against NQMF/RM-Job-Aid-style standards.

    Deterministic checks ported from lg-RMOrch's review_preliminary_title
    (substep 6.3) + RM Job Aid p.17:
      - lender coverage >= loan amount
      - Title Effective Date within 90 days of closing
      - mortgagee clause expectation for the property's state
      - Title Effective Date 3-way consistency (J-T1): the title document's
        own effective date, the Document Expirations screen date, and the
        expiration condition's date must all agree. RMOrch's version only
        checked doc-vs-closing (2-way); the job aid (p.17) requires the
        3-way sync. Pass `doc_expiration_screen_date` / `expiration_condition_date`
        to enable this; omitted values are reported as unverifiable, not
        silently skipped.

    `document_fields` should be the output of extracting title_report.json
    (e.g. title_number, lender_coverage/coverage, effective_date, vesting_type).
    `loan_context` expects: loan_amount, estimated_closing_date, property_state,
    borrower_name (optional, for a light vesting/name sanity check).

    NOT implemented (intentionally — see architecture doc item T2): chain of
    title. Standard title commitments only show CURRENT vesting + exceptions,
    not historical ownership; validating chain of title needs a different
    document source (a chain-of-title report / deed-history pull), which has
    not been scoped. This function returns `chain_of_title_verified: False`
    and does not fabricate a check for it.
    """
    mortgagee_clauses = mortgagee_clauses or DEFAULT_MORTGAGEE_CLAUSES
    flags: List[Dict[str, Any]] = []

    # -- Lender coverage >= loan amount --
    coverage = to_number(
        document_fields.get("lender_coverage")
        or document_fields.get("coverage")
        or document_fields.get("liability")
    )
    loan_amount = to_number(loan_context.get("loan_amount"))
    if coverage is not None and loan_amount is not None:
        if coverage < loan_amount:
            flags.append(
                Flag(field="lender_coverage", title="Title coverage below loan amount",
                     severity="warning", action="ptf",
                     details=f"Lender coverage {format_currency(coverage)} is below the loan "
                             f"amount {format_currency(loan_amount)}.",
                     suggestion="Require the title policy coverage to be >= the loan amount.").to_dict()
            )
        else:
            flags.append(
                Flag(field="lender_coverage", title="Title coverage adequate",
                     severity="info", action="ignore",
                     details=f"Lender coverage {format_currency(coverage)} >= "
                             f"loan amount {format_currency(loan_amount)}.").to_dict()
            )
    else:
        flags.append(
            Flag(field="lender_coverage", title="Coverage not extracted",
                 severity="warning", action="ptf",
                 details="Lender coverage amount was not found; confirm manually.").to_dict()
        )

    # -- Title Effective Date within 90 days of closing --
    effective_date = parse_date(document_fields.get("effective_date") or document_fields.get("title_report_date"))
    closing_date = parse_date(loan_context.get("estimated_closing_date"))
    if effective_date and closing_date:
        gap_days = days_between(effective_date, closing_date)
        if gap_days > TITLE_EFFECTIVE_DATE_MAX_DAYS:
            flags.append(
                Flag(field="effective_date", title="Title Effective Date too old",
                     severity="critical", action="critical",
                     details=f"Title Effective Date is {gap_days} days from closing "
                             f"(max {TITLE_EFFECTIVE_DATE_MAX_DAYS}). Job Aid p.17 also requires "
                             "this to match the Document Expirations screen and expiration "
                             "conditions — confirm those separately (not automated here).",
                     suggestion="Request an updated title commitment / date-down endorsement.").to_dict()
            )
        else:
            flags.append(
                Flag(field="effective_date", title="Title Effective Date within window",
                     severity="info", action="ignore",
                     details=f"{gap_days} days from closing (<= {TITLE_EFFECTIVE_DATE_MAX_DAYS}).").to_dict()
            )
    else:
        flags.append(
            Flag(field="effective_date", title="Could not verify 90-day window",
                 severity="warning", action="ptf",
                 details="Effective date and/or closing date missing; cannot verify the "
                         "90-day window.").to_dict()
        )

    # -- Title Effective Date 3-way consistency (J-T1) --
    doc_expiration_screen = parse_date(doc_expiration_screen_date)
    expiration_condition = parse_date(expiration_condition_date)
    if effective_date and (doc_expiration_screen or expiration_condition):
        mismatches = []
        if doc_expiration_screen and doc_expiration_screen != effective_date:
            mismatches.append(f"Document Expirations screen ({doc_expiration_screen})")
        if expiration_condition and expiration_condition != effective_date:
            mismatches.append(f"expiration condition ({expiration_condition})")
        if mismatches:
            flags.append(
                Flag(field="effective_date_consistency", title="Title Effective Date 3-way mismatch",
                     severity="warning", action="ptf",
                     details=f"Title document effective date ({effective_date}) does not match: "
                             f"{', '.join(mismatches)}.",
                     suggestion="Sync the Document Expirations screen and expiration condition "
                                "dates to the title document's actual effective date.").to_dict()
            )
        else:
            flags.append(
                Flag(field="effective_date_consistency", title="Title Effective Date 3-way consistent",
                     severity="info", action="ignore",
                     details="Title document, Document Expirations screen, and expiration "
                             "condition dates all agree.").to_dict()
            )
    elif effective_date:
        flags.append(
            Flag(field="effective_date_consistency", title="3-way date consistency not verified",
                 severity="warning", action="ptf",
                 details="Document Expirations screen date and/or expiration condition date not "
                         "provided; only the doc-vs-closing-date check (above) was performed.").to_dict()
        )

    # -- Mortgagee clause reminder (informational — text matching against a
    #    real commitment PDF is left to the extraction/human review step) --
    prop_state = loan_context.get("property_state", "")
    flags.append(
        Flag(field="mortgagee_clause", title="Mortgagee clause reminder",
             severity="info", action="ptf",
             details=f"Expected clause for state '{prop_state or 'unknown'}': "
                     f"{_mortgagee_clause(prop_state, mortgagee_clauses)}",
             suggestion="Confirm the commitment reflects this exact clause.").to_dict()
    )

    return {
        "checks_performed": ["lender_coverage_vs_loan_amount", "effective_date_within_90_days",
                              "effective_date_3way_consistency", "mortgagee_clause_reminder"],
        "flags": flags,
        "chain_of_title_verified": False,
        "chain_of_title_note": (
            "Not implemented. Standard title commitments do not contain historical ownership "
            "data; chain-of-title validation requires a new document source (see architecture "
            "doc Part 2, item T2). Needs scoping before building."
        ),
    }


# ---------------------------------------------------------------------------
# Tool 3/4: validate_insurance_coverage(document, loan_requirements)
# ---------------------------------------------------------------------------

HOI_DEDUCTIBLE_MAX_PCT = 0.05  # RM Job Aid p.13
FLOOD_DEDUCTIBLE_MAX_STANDARD = 10_000.0  # RM Job Aid p.13
FLOOD_DEDUCTIBLE_MAX_CONDO_PUD = 25_000.0  # RM Job Aid p.13
HOI_TERM_MONTHS = 12  # RM Job Aid p.13 (J-T2)
HOI_TERM_TOLERANCE_MONTHS = 1.0  # allow +/- 1 month for date-math imprecision
DSCR_RENT_LOSS_MIN_MONTHS = 6  # RM Job Aid p.13 (J-T7)
DSCR_UNACCEPTABLE_COVERAGE_TYPE = "actual_loss_sustained"  # RM Job Aid p.13: "not acceptable without the maximum # of months"
HO6_DEDUCTIBLE_MAX_PCT = 0.05  # RM Job Aid p.13 (J-T8) — same 5% cap, but vs. the HO6's OWN dwelling coverage
DEFAULT_NFIP_MAX_RESIDENTIAL = 250_000.0  # NFIP maximum building coverage for 1-4 unit residential


def validate_insurance_coverage(
    hoi_fields: Optional[Dict[str, Any]],
    loan_context: Dict[str, Any],
    flood_fields: Optional[Dict[str, Any]] = None,
    ho6_fields: Optional[Dict[str, Any]] = None,
    mortgagee_clauses: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Validate hazard & flood insurance against RM-Job-Aid-style standards.

    Deterministic checks ported from lg-RMOrch's review_hazard_flood_insurance
    (substep 5.1) + RM Job Aid p.13:
      - HOI deductible <= 5% of coverage
      - Flood deductible <= $10K ($25K for Condo/PUD); flood-zone status is
        read from the flood document's own `in_sfha` field when present,
        falling back to `loan_context["is_flood_zone"]` otherwise
      - HOI coverage >= lesser of (loan amount, replacement cost) when
        replacement cost is extracted; falls back to loan-amount-only when
        it isn't
      - HOI loss payee / mortgagee clause matches the expected clause for
        the property's state (same `mortgagee_clauses` table as
        `validate_title_commitment`)
      - Policy term ~= 12 months (J-T2)
      - Wind/Hail/Hurricane coverage type (J-T5): flags if the policy
        explicitly excludes wind/hail without a separate policy
      - Flood "lesser of 3" rule (J-T6): coverage must be >= the LESSER of
        (a) unpaid principal balance, (b) replacement cost of improvements,
        (c) the NFIP maximum — a real implementation of what was previously
        a simplified "coverage >= loan amount" proxy
      - DSCR 6-month rent-loss coverage TYPE (J-T7): "actual loss sustained"
        is explicitly NOT acceptable regardless of months; only
        "scheduled_rent" with >= 6 months counts
      - Condo HO6 dwelling-coverage deductible <= 5% of the HO6's OWN
        dwelling coverage (J-T8) — distinct from the main HOI 5% rule

    `loan_context` expects: loan_amount, property_type (for condo/PUD flood
    cap), is_dscr (optional, bool), is_flood_zone (optional, bool),
    unpaid_principal_balance (optional, for J-T6), nfip_max (optional,
    defaults to $250,000 for 1-4 unit residential, for J-T6).

    Still NOT implemented (need a document type that doesn't exist in any of
    the 3 source repos' schemas — see `validate_premium_paid_in_full` in this
    module for the J-T3 receipt/invoice check, built separately since it
    needs its own document).
    """
    flags: List[Dict[str, Any]] = []
    hoi_fields = hoi_fields or {}
    flood_fields = flood_fields or {}

    if not hoi_fields:
        flags.append(
            Flag(field="evidence_of_insurance", title="Evidence of Insurance not found",
                 severity="critical", action="critical",
                 details="No Evidence of Insurance document was extracted. HOI is required "
                         "to clear the insurance condition.").to_dict()
        )
    else:
        coverage = to_number(hoi_fields.get("coverage_amount"))
        deductible = to_number(hoi_fields.get("deductible"))
        loan_amount = to_number(loan_context.get("loan_amount"))

        # HOI deductible <= 5% of coverage
        if coverage is not None and deductible is not None:
            cap = HOI_DEDUCTIBLE_MAX_PCT * coverage
            if deductible > cap:
                flags.append(
                    Flag(field="deductible", title="HOI deductible exceeds 5% cap",
                         severity="critical", action="critical",
                         details=f"Deductible {format_currency(deductible)} exceeds 5% of "
                                 f"coverage ({format_currency(cap)}).",
                         suggestion="Return the policy; deductible must be <= 5% of face value.").to_dict()
                )
            else:
                flags.append(
                    Flag(field="deductible", title="HOI deductible within 5% cap",
                         severity="info", action="ignore",
                         details=f"Deductible {format_currency(deductible)} <= 5% of "
                                 f"coverage ({format_currency(cap)}).").to_dict()
                )

        # HOI coverage >= lesser of (loan amount, replacement cost). Mirrors
        # the flood side's already-implemented "lesser of 3" rule (J-T6).
        # `replacement_cost` is modeled on LG-docsOrch's production Evidence
        # of Insurance extraction schema, which has this field where ours
        # previously didn't -- this used to be a simplified "coverage >=
        # loan amount" proxy because replacement cost wasn't extractable.
        replacement_cost = to_number(hoi_fields.get("replacement_cost"))
        if coverage is not None and (loan_amount is not None or replacement_cost is not None):
            candidates = [v for v in (loan_amount, replacement_cost) if v is not None]
            required_minimum = min(candidates)
            method = (
                "lesser of loan amount and replacement cost" if len(candidates) == 2
                else "replacement cost (loan amount not provided)" if replacement_cost is not None
                else "loan amount (replacement cost not extracted)"
            )
            if coverage < required_minimum:
                flags.append(
                    Flag(field="coverage_amount", title="HOI coverage below required minimum",
                         severity="warning", action="ptf",
                         details=f"Coverage {format_currency(coverage)} is below the required "
                                 f"minimum {format_currency(required_minimum)} ({method}).",
                         suggestion="Verify replacement cost; require more coverage if unjustified.").to_dict()
                )
            else:
                flags.append(
                    Flag(field="coverage_amount", title="HOI coverage meets required minimum",
                         severity="info", action="ignore",
                         details=f"Coverage {format_currency(coverage)} >= required minimum "
                                 f"{format_currency(required_minimum)} ({method}).").to_dict()
                )

        # Loss payee / mortgagee clause on the HOI declaration page must
        # match the expected clause for the property's state -- the same
        # table/function used for the title commitment's mortgagee clause
        # reminder. `loss_payee_clause` has been in the schema since this
        # repo's initial pass but was never actually checked against
        # anything until now.
        loss_payee_clause = hoi_fields.get("loss_payee_clause")
        if loss_payee_clause:
            prop_state = loan_context.get("property_state", "")
            expected_clause = _mortgagee_clause(prop_state, mortgagee_clauses or DEFAULT_MORTGAGEE_CLAUSES)
            if str(loss_payee_clause).strip().lower() == str(expected_clause).strip().lower():
                flags.append(
                    Flag(field="loss_payee_clause", title="HOI mortgagee clause matches",
                         severity="info", action="ignore",
                         details="The HOI declaration page's loss payee clause matches the "
                                 "expected clause for this state.").to_dict()
                )
            else:
                flags.append(
                    Flag(field="loss_payee_clause", title="HOI mortgagee clause does not match",
                         severity="warning", action="ptf",
                         details=f"HOI shows: '{loss_payee_clause}'. Expected for state "
                                 f"'{prop_state or 'unknown'}': '{expected_clause}'.",
                         suggestion="Request a corrected declaration page with the correct "
                                    "mortgagee clause.").to_dict()
                )

        # Policy term ~= 12 months (J-T2)
        eff = parse_date(hoi_fields.get("effective_date"))
        exp = parse_date(hoi_fields.get("expiration_date"))
        if eff and exp:
            term_months = months_between(eff, exp)
            if abs(term_months - HOI_TERM_MONTHS) > HOI_TERM_TOLERANCE_MONTHS:
                flags.append(
                    Flag(field="policy_term", title="HOI term is not ~12 months",
                         severity="warning", action="ptf",
                         details=f"Policy term is approximately {term_months:.1f} months "
                                 f"(effective {eff} to {exp}); Job Aid p.13 requires a 12-month term.",
                         suggestion="Request a corrected declaration page with a 12-month term.").to_dict()
                )
            else:
                flags.append(
                    Flag(field="policy_term", title="HOI term is ~12 months",
                         severity="info", action="ignore",
                         details=f"Policy term is approximately {term_months:.1f} months.").to_dict()
                )
        else:
            flags.append(
                Flag(field="policy_term", title="Could not verify policy term",
                     severity="warning", action="ptf",
                     details="Effective and/or expiration date missing; cannot verify the "
                             "12-month term requirement.").to_dict()
            )

        # Wind/Hail/Hurricane coverage type (J-T5)
        wind_hail = (hoi_fields.get("wind_hail_hurricane_included") or "").strip().lower()
        if wind_hail in ("excluded", "excl", "no", "false"):
            flags.append(
                Flag(field="wind_hail_hurricane_included", title="Wind/Hail/Hurricane excluded",
                     severity="critical", action="critical",
                     details="The HOI policy explicitly excludes Wind/Hail/Hurricane coverage.",
                     suggestion="A separate wind/hail/hurricane policy must be obtained.").to_dict()
            )
        elif not wind_hail:
            flags.append(
                Flag(field="wind_hail_hurricane_included", title="Wind/Hail/Hurricane coverage not verified",
                     severity="warning", action="ptf",
                     details="Policy does not state whether Wind/Hail/Hurricane coverage is "
                             "included via a special form or all-perils language; confirm manually.").to_dict()
            )
        else:
            flags.append(
                Flag(field="wind_hail_hurricane_included", title="Wind/Hail/Hurricane coverage included",
                     severity="info", action="ignore",
                     details=f"Policy coverage type: '{wind_hail}'.").to_dict()
            )

        # DSCR 6-month rent-loss coverage TYPE (J-T7)
        if as_bool(loan_context.get("is_dscr")):
            rent_loss_type = (hoi_fields.get("dscr_rent_loss_coverage_type") or "").strip().lower()
            rent_loss_months = to_number(hoi_fields.get("dscr_rent_loss_months"))
            if rent_loss_type == DSCR_UNACCEPTABLE_COVERAGE_TYPE:
                flags.append(
                    Flag(field="dscr_rent_loss_coverage_type", title="DSCR rent-loss coverage type not acceptable",
                         severity="critical", action="critical",
                         details="Policy uses 'actual loss sustained' rent-loss coverage, which "
                                 "RM Job Aid p.13 says is NOT acceptable regardless of the number "
                                 "of months.",
                         suggestion="Require scheduled-rent rent-loss coverage instead.").to_dict()
                )
            elif rent_loss_type == "scheduled_rent" and rent_loss_months is not None:
                if rent_loss_months < DSCR_RENT_LOSS_MIN_MONTHS:
                    flags.append(
                        Flag(field="dscr_rent_loss_coverage_type", title="DSCR rent-loss coverage below 6 months",
                             severity="critical", action="critical",
                             details=f"Scheduled-rent coverage is only {rent_loss_months} months; "
                                     f"DSCR loans require {DSCR_RENT_LOSS_MIN_MONTHS} months PITIA.",
                             suggestion="Request an endorsement extending rent-loss coverage to "
                                        "6 months.").to_dict()
                    )
                else:
                    flags.append(
                        Flag(field="dscr_rent_loss_coverage_type", title="DSCR rent-loss coverage meets requirement",
                             severity="info", action="ignore",
                             details=f"Scheduled-rent coverage of {rent_loss_months} months "
                                     f">= {DSCR_RENT_LOSS_MIN_MONTHS}-month requirement.").to_dict()
                    )
            else:
                flags.append(
                    Flag(field="dscr_rent_loss_coverage_type", title="DSCR rent-loss coverage not verified",
                         severity="warning", action="ptf",
                         details="DSCR loan but rent-loss coverage type/months not extracted; "
                                 "confirm manually.").to_dict()
                )

    # Condo HO6 dwelling-coverage deductible (J-T8) — distinct from the main
    # HOI 5% rule: checked against the HO6 policy's OWN dwelling coverage.
    if ho6_fields:
        ho6_coverage = to_number(ho6_fields.get("ho6_dwelling_coverage_amount") or ho6_fields.get("coverage_amount"))
        ho6_deductible = to_number(ho6_fields.get("deductible"))
        if ho6_coverage is not None and ho6_deductible is not None:
            cap = HO6_DEDUCTIBLE_MAX_PCT * ho6_coverage
            if ho6_deductible > cap:
                flags.append(
                    Flag(field="ho6_deductible", title="HO6 deductible exceeds 5% of its own dwelling coverage",
                         severity="critical", action="critical",
                         details=f"HO6 deductible {format_currency(ho6_deductible)} exceeds 5% of "
                                 f"the HO6's dwelling coverage ({format_currency(cap)}).",
                         suggestion="Return the HO6 policy; deductible must be <= 5% of its own "
                                    "dwelling coverage.").to_dict()
                )
            else:
                flags.append(
                    Flag(field="ho6_deductible", title="HO6 deductible within cap",
                         severity="info", action="ignore",
                         details=f"HO6 deductible {format_currency(ho6_deductible)} <= 5% of "
                                 f"its dwelling coverage ({format_currency(cap)}).").to_dict()
                )
        else:
            flags.append(
                Flag(field="ho6_deductible", title="HO6 coverage/deductible not verified",
                     severity="warning", action="ptf",
                     details="HO6 dwelling coverage and/or deductible not extracted; confirm manually.").to_dict()
            )

    # Flood deductible cap (10K standard, 25K Condo/PUD)
    #
    # Flood-zone status: prefer the flood document's own `in_sfha` flag (from
    # a Flood Certificate/Determination -- modeled on LG-docsOrch's
    # production flood extraction schema, which has this field where ours
    # previously didn't) over an externally-supplied
    # loan_context["is_flood_zone"], since the document is the authoritative,
    # current source when it's available. Falls back to the external flag
    # when no flood document (or an in_sfha-less one) was extracted.
    in_sfha = as_bool(flood_fields.get("in_sfha")) if flood_fields else None
    is_flood_zone = in_sfha if in_sfha is not None else as_bool(loan_context.get("is_flood_zone"))

    flood_deductible = to_number(flood_fields.get("deductible")) if flood_fields else None
    if flood_deductible is not None:
        property_type = (loan_context.get("property_type") or "").lower()
        is_condo_pud = "condo" in property_type or "pud" in property_type
        cap = FLOOD_DEDUCTIBLE_MAX_CONDO_PUD if is_condo_pud else FLOOD_DEDUCTIBLE_MAX_STANDARD
        if flood_deductible > cap:
            flags.append(
                Flag(field="flood_deductible", title="Flood deductible exceeds cap",
                     severity="critical", action="critical",
                     details=f"Flood deductible {format_currency(flood_deductible)} exceeds the "
                             f"{format_currency(cap)} cap.",
                     suggestion="Return the flood policy; deductible over the cap is not acceptable.").to_dict()
            )
        else:
            flags.append(
                Flag(field="flood_deductible", title="Flood deductible within cap",
                     severity="info", action="ignore",
                     details=f"Flood deductible {format_currency(flood_deductible)} <= "
                             f"{format_currency(cap)} cap.").to_dict()
            )
    elif is_flood_zone:
        flags.append(
            Flag(field="flood_insurance", title="No flood documents found in flood zone",
                 severity="critical", action="critical",
                 details="Property is in a flood zone but no flood insurance was extracted.",
                 suggestion="Obtain flood coverage; it cannot be waived in a flood zone.").to_dict()
        )
    elif in_sfha is False:
        flags.append(
            Flag(field="flood_insurance", title="Flood insurance not required",
                 severity="info", action="ignore",
                 details="The flood determination confirms the property is NOT in a Special "
                         "Flood Hazard Area; flood insurance is not required.").to_dict()
        )

    # Flood coverage "lesser of 3" rule (J-T6): coverage must be >= the
    # LESSER of (unpaid principal balance, replacement cost, NFIP max).
    # This replaces the old simplified "coverage >= loan amount" proxy with
    # the real rule, now that the three inputs are available.
    if flood_fields:
        flood_coverage = to_number(flood_fields.get("flood_insurance_coverage"))
        upb = to_number(loan_context.get("unpaid_principal_balance"))
        replacement_cost = to_number(flood_fields.get("replacement_cost_of_improvements"))
        nfip_max = to_number(loan_context.get("nfip_max")) or DEFAULT_NFIP_MAX_RESIDENTIAL

        candidates = [v for v in (upb, replacement_cost, nfip_max) if v is not None]
        if flood_coverage is not None and candidates:
            required_minimum = min(candidates)
            if flood_coverage < required_minimum:
                flags.append(
                    Flag(field="flood_coverage_lesser_of_3", title="Flood coverage below the required minimum",
                         severity="critical", action="critical",
                         details=f"Flood coverage {format_currency(flood_coverage)} is below the "
                                 f"required minimum {format_currency(required_minimum)} (lesser of "
                                 f"UPB={format_currency(upb)}, replacement cost="
                                 f"{format_currency(replacement_cost)}, NFIP max="
                                 f"{format_currency(nfip_max)}).",
                         suggestion="Increase flood coverage to at least the lesser-of-3 minimum.").to_dict()
                )
            else:
                flags.append(
                    Flag(field="flood_coverage_lesser_of_3", title="Flood coverage meets the lesser-of-3 rule",
                         severity="info", action="ignore",
                         details=f"Flood coverage {format_currency(flood_coverage)} >= required "
                                 f"minimum {format_currency(required_minimum)}.").to_dict()
                )
        elif flood_coverage is not None:
            flags.append(
                Flag(field="flood_coverage_lesser_of_3", title="Flood coverage lesser-of-3 not fully verified",
                     severity="warning", action="ptf",
                     details="Flood coverage was extracted but UPB/replacement cost/NFIP max are "
                             "not all available; falling back to the NFIP default max only is not "
                             "sufficient to fully verify this rule.").to_dict()
            )

    return {
        "checks_performed": ["hoi_deductible_cap", "hoi_coverage_lesser_of_loan_or_replacement_cost",
                              "loss_payee_clause_match", "hoi_term_12_months",
                              "wind_hail_hurricane", "dscr_rent_loss_coverage_type", "ho6_deductible_cap",
                              "flood_deductible_cap", "flood_coverage_lesser_of_3"],
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Tool 4/4: calculate_escrow(property_taxes, insurance_premium, months)
# ---------------------------------------------------------------------------

DEFAULT_CUSHION_MONTHS = 2  # RM Job Aid p.15 ("Number of months of cushion would be 2 for each disbursement type")


def calculate_escrow(
    property_taxes: float,
    insurance_premium: float,
    months: int = DEFAULT_CUSHION_MONTHS,
) -> Dict[str, Any]:
    """Deterministic escrow cushion calculation — Python, no LLM, per spec.

    This was architecture doc item T3 (❌ confirmed missing as real math in
    all 3 repos — every existing implementation is an advisory flag/boolean
    consistency check, not a calculation). Formula/months-of-cushion straight
    from the doc's own sketch and RM Job Aid p.15.

    Note on J-T9 ("your aggregate adjustment will always be a negative
    number"): that statement is about Encompass's broader Aggregate Escrow
    Account *adjustment* (new cushion minus previously-collected escrow),
    which needs a "previously collected" input this pure function doesn't
    have. It is NOT the same number as `cushion_amount` below, so no
    assertion is made here that would be incorrect; it's left as a caveat in
    the returned payload for the caller to act on once that input exists.
    """
    if property_taxes < 0 or insurance_premium < 0:
        raise ValueError("property_taxes and insurance_premium must be >= 0.")
    if months <= 0:
        raise ValueError("months must be a positive integer.")

    monthly_tax = property_taxes / 12
    monthly_insurance = insurance_premium / 12
    cushion = (monthly_tax + monthly_insurance) * months

    return {
        "monthly_tax": round(monthly_tax, 2),
        "monthly_insurance": round(monthly_insurance, 2),
        "cushion_months": months,
        "cushion_amount": round(cushion, 2),
        "total_monthly_escrow": round(monthly_tax + monthly_insurance, 2),
        "caveat": (
            "cushion_amount is the escrow cushion only. It is NOT the Encompass "
            "'aggregate adjustment' (which RM Job Aid p.15 / J-T9 says is always negative) "
            "— that figure additionally nets out previously-collected escrow, which is "
            "outside this function's inputs. See calculate_aggregate_escrow() for the "
            "multi-item version that actually computes an aggregate adjustment."
        ),
    }


# ---------------------------------------------------------------------------
# J-T9 (full version): calculate_aggregate_escrow — multi-item aggregate
# escrow analysis with a real "aggregate adjustment" computation, using the
# standard RESPA lowest-balance method.
# ---------------------------------------------------------------------------

class DisbursementItem:
    """One escrow disbursement item (a tax type, or an insurance premium).

    `due_months` is the list of calendar months (1-12) the FULL periodic
    payment is disbursed — e.g. [12] for an annual tax bill, [6, 12] for
    semi-annual, [3, 6, 9, 12] for quarterly (RM Job Aid p.15: "if taxes are
    semi-annual, you will need to input two due dates; if quarterly, four").
    """

    __slots__ = ("name", "annual_amount", "due_months")

    def __init__(self, name: str, annual_amount: float, due_months: List[int]):
        if not due_months:
            raise ValueError(f"DisbursementItem '{name}' needs at least one due month.")
        self.name = name
        self.annual_amount = annual_amount
        self.due_months = sorted(set(due_months))


def _simulate_lowest_balance(items: List[DisbursementItem]) -> float:
    """Simulate 12 months of a pooled escrow account starting at $0, with
    monthly deposits and each item's periodic disbursements, and return the
    lowest (most negative, or least positive) running balance reached.

    This is the standard RESPA "lowest balance" escrow-analysis method: the
    cushion only needs to cover how far into the hole the account actually
    goes given the real timing of deposits vs. disbursements — not a naive
    sum of each item's own worst case.
    """
    monthly_deposit = sum(item.annual_amount / 12 for item in items)
    balance = 0.0
    lowest = 0.0
    for month in range(1, 13):
        balance += monthly_deposit
        for item in items:
            if month in item.due_months:
                balance -= item.annual_amount / len(item.due_months)
        lowest = min(lowest, balance)
    return lowest


def calculate_aggregate_escrow(
    disbursement_items: List[Dict[str, Any]],
    cushion_months: int = DEFAULT_CUSHION_MONTHS,
) -> Dict[str, Any]:
    """Full multi-item Aggregate Escrow Account calculation (RM Job Aid p.15).

    Unlike `calculate_escrow` (which only handles a flat tax+insurance
    cushion, architecture doc item T3), this models each disbursement item's
    own due-date schedule and computes:
      - `non_aggregate_cushion`: sum of each item's own lowest-balance
        cushion, computed as if that item had its OWN separate escrow
        sub-account (can't borrow surplus from any other item)
      - `aggregate_cushion`: the pooled-account cushion, derived from the
        RESPA lowest-balance method on the COMBINED cash flow — one item's
        surplus at a given month can offset another's deficit at the same
        month, which a siloed sub-account could never do
      - `aggregate_adjustment` = aggregate_cushion - non_aggregate_cushion.
        This difference is mathematically guaranteed <= 0: pooling can only
        reduce (never increase) the required cushion, which is exactly RM
        Job Aid p.15 / J-T9's "your aggregate adjustment will always be a
        negative number." The sanity-check flag below exists for the
        boundary case (exactly 0 — no benefit from pooling, e.g. a single
        item) and as a guard against a future refactor accidentally
        breaking the invariant, not because real data can violate it.

    `disbursement_items`: list of {"name": str, "annual_amount": float,
    "due_months": [int, ...]} — e.g.
    [{"name": "County Tax", "annual_amount": 6000, "due_months": [12]},
     {"name": "HOI", "annual_amount": 1200, "due_months": [1]}]
    """
    if not disbursement_items:
        raise ValueError("disbursement_items must not be empty.")

    items = [
        DisbursementItem(i["name"], to_number(i["annual_amount"]) or 0.0, i["due_months"])
        for i in disbursement_items
    ]

    monthly_deposit = sum(item.annual_amount / 12 for item in items)

    # Each item's OWN cushion, as if it had a separate escrow sub-account —
    # simulated independently so it can't benefit from any other item's
    # surplus at the same point in time.
    non_aggregate_cushion = 0.0
    for item in items:
        own_lowest = _simulate_lowest_balance([item])
        own_monthly = item.annual_amount / 12
        non_aggregate_cushion += max(0.0, -own_lowest) + own_monthly * cushion_months

    lowest_point = _simulate_lowest_balance(items)
    aggregate_cushion = max(0.0, -lowest_point) + monthly_deposit * cushion_months
    aggregate_adjustment = aggregate_cushion - non_aggregate_cushion

    flags: List[Dict[str, Any]] = []
    if aggregate_adjustment > 0:
        # Should not happen — see docstring; pooling is mathematically
        # guaranteed to need <= the sum of each item's own siloed cushion.
        flags.append(
            Flag(field="aggregate_adjustment", title="Aggregate adjustment is positive (unexpected)",
                 severity="warning", action="ptf",
                 details=f"Computed aggregate adjustment is {format_currency(aggregate_adjustment)}, "
                         "but RM Job Aid p.15 states this should always be a negative number (or "
                         "zero at worst, with a single item). Double-check the due-date months "
                         "entered for each disbursement item.",
                 suggestion="Verify each item's due_months against the actual tax/insurance "
                            "billing schedule.").to_dict()
        )
    elif aggregate_adjustment == 0:
        flags.append(
            Flag(field="aggregate_adjustment", title="Aggregate adjustment is zero",
                 severity="info", action="ignore",
                 details="No pooling benefit (e.g. a single disbursement item, or items whose "
                         "own lowest points never dip below zero independently).").to_dict()
        )
    else:
        flags.append(
            Flag(field="aggregate_adjustment", title="Aggregate adjustment is negative, as expected",
                 severity="info", action="ignore",
                 details=f"Aggregate adjustment is {format_currency(aggregate_adjustment)} — "
                         "pooling reduced the required cushion vs. separate sub-accounts.").to_dict()
        )

    return {
        "monthly_deposit": round(monthly_deposit, 2),
        "non_aggregate_cushion": round(non_aggregate_cushion, 2),
        "aggregate_cushion": round(aggregate_cushion, 2),
        "aggregate_adjustment": round(aggregate_adjustment, 2),
        "lowest_simulated_balance": round(lowest_point, 2),
        "items": [{"name": i.name, "annual_amount": i.annual_amount, "due_months": i.due_months}
                  for i in items],
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# J-T3: Premium-paid-in-full verification
# ---------------------------------------------------------------------------

def validate_premium_paid_in_full(
    receipt_fields: Optional[Dict[str, Any]],
    policy_fields: Dict[str, Any],
    loan_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify insurance premium is paid in full, or payable at closing with
    an invoice (RM Job Aid p.13, J-T3).

    Rules:
      - All premiums must be paid in full, evidenced by a paid receipt, OR
        paid at closing with an invoice for the premium due.
      - Refinances: premiums due within 60 days of closing must be paid in
        full prior to or at closing.

    `loan_context` expects: transaction_type ("purchase" | "refinance"),
    days_to_closing (optional int, for the refinance 60-day rule).
    """
    flags: List[Dict[str, Any]] = []
    if not receipt_fields:
        flags.append(
            Flag(field="premium_paid_in_full", title="No premium receipt/invoice found",
                 severity="warning", action="ptf",
                 details="RM Job Aid p.13 requires a paid receipt OR an invoice for the premium "
                         "due at closing; neither was found.",
                 suggestion="Request a paid receipt or invoice from the insurance agent.").to_dict()
        )
        return {"flags": flags}

    paid_in_full = bool(as_bool(receipt_fields.get("paid_in_full")))
    amount = to_number(receipt_fields.get("amount"))
    premium = to_number(policy_fields.get("hazard_insurance_premium") or policy_fields.get("flood_insurance_premium"))

    if paid_in_full:
        if amount is not None and premium is not None and abs(amount - premium) > 1.0:
            flags.append(
                Flag(field="premium_paid_in_full", title="Paid receipt amount does not match policy premium",
                     severity="warning", action="ptf",
                     details=f"Receipt shows {format_currency(amount)} paid, but the policy "
                             f"premium is {format_currency(premium)}.").to_dict()
            )
        else:
            flags.append(
                Flag(field="premium_paid_in_full", title="Premium paid in full",
                     severity="info", action="ignore",
                     details=f"Paid receipt confirms {format_currency(amount)} paid in full.").to_dict()
            )
    else:
        # Invoice for an amount due at closing is acceptable...
        is_refinance = (loan_context.get("transaction_type") or "").lower() == "refinance"
        days_to_closing = loan_context.get("days_to_closing")
        if is_refinance and days_to_closing is not None and days_to_closing <= 60:
            flags.append(
                Flag(field="premium_paid_in_full", title="Refinance: premium due within 60 days must be paid in full",
                     severity="critical", action="critical",
                     details=f"This is a refinance closing in {days_to_closing} days; RM Job Aid "
                             f"p.13 requires premiums due within 60 days of closing to be paid in "
                             f"full prior to or at closing, not merely invoiced.",
                     suggestion="Obtain a paid receipt, or ensure the premium is paid at closing "
                                "via the invoice and confirm no coverage gap.").to_dict()
            )
        else:
            flags.append(
                Flag(field="premium_paid_in_full", title="Invoice on file for premium due at closing",
                     severity="info", action="ptf",
                     details=f"Invoice for {format_currency(amount)} is acceptable; ensure it is "
                             f"filed in the ICD placeholder and assigned to the HOI/Flood condition.").to_dict()
            )

    return {"flags": flags}


# ---------------------------------------------------------------------------
# J-T4: No-coverage-gap on refinance
# ---------------------------------------------------------------------------

def validate_no_coverage_gap(
    old_policy_fields: Dict[str, Any],
    new_policy_fields: Dict[str, Any],
    loan_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify there's no gap between the old (refinanced-away) policy's
    expiration and the new policy's effective date, and that a new binder
    isn't dated more than 15 days before the note date unless paid in full
    (RM Job Aid p.13, J-T4). Applies to refinances only.
    """
    flags: List[Dict[str, Any]] = []
    old_expiration = parse_date(old_policy_fields.get("expiration_date"))
    new_effective = parse_date(new_policy_fields.get("effective_date"))
    note_date = parse_date(loan_context.get("note_date"))
    new_policy_paid_in_full = bool(as_bool(new_policy_fields.get("paid_in_full")))

    if old_expiration and new_effective:
        gap_days = (new_effective - old_expiration).days
        if gap_days > 0:
            flags.append(
                Flag(field="coverage_gap", title="Gap in insurance coverage detected",
                     severity="critical", action="critical",
                     details=f"Old policy expired {old_expiration}, new policy isn't effective "
                             f"until {new_effective} — a {gap_days}-day coverage gap.",
                     suggestion="Request a corrected effective date so coverage is continuous.").to_dict()
            )
        else:
            flags.append(
                Flag(field="coverage_gap", title="No coverage gap",
                     severity="info", action="ignore",
                     details=f"New policy effective {new_effective} is on/before old policy "
                             f"expiration {old_expiration} — coverage is continuous.").to_dict()
            )
    else:
        flags.append(
            Flag(field="coverage_gap", title="Could not verify coverage continuity",
                 severity="warning", action="ptf",
                 details="Old policy expiration date and/or new policy effective date missing.").to_dict()
        )

    if note_date and new_effective and not new_policy_paid_in_full:
        days_before_note = (note_date - new_effective).days
        if days_before_note > 15:
            flags.append(
                Flag(field="binder_timing", title="New binder dated too far before note date",
                     severity="warning", action="ptf",
                     details=f"New policy's effective/binder date is {days_before_note} days "
                             f"before the note date ({note_date}); if not paid in full, this "
                             f"cannot exceed 15 days.",
                     suggestion="Obtain a paid-in-full receipt, or a binder dated within 15 days "
                                "of the note date.").to_dict()
            )

    return {"flags": flags}


# ---------------------------------------------------------------------------
# J-T10: CPL / E&O / Title Commitment 3-way agent-name cross-match
# ---------------------------------------------------------------------------

E_AND_O_MIN_COVERAGE = 1_000_000.0
E_AND_O_MIN_COVERAGE_ATTORNEY = 500_000.0
CPL_MAX_AGE_DAYS = 60  # RM Job Aid p.17


def validate_closing_agent_match(
    title_fields: Dict[str, Any],
    cpl_fields: Optional[Dict[str, Any]],
    eo_fields: Optional[Dict[str, Any]],
    loan_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Cross-match the settlement/closing agent name across the Title
    Commitment, CPL, and E&O policy, plus the standalone CPL/E&O standards
    from RM Job Aid p.17 (J-T10 — refines what was advisory-only everywhere
    into a concrete 3-document cross-check). Does not apply to Correspondent
    loans (CPL/E&O are wholesale-only requirements).
    """
    loan_context = loan_context or {}
    flags: List[Dict[str, Any]] = []

    if (loan_context.get("channel") or "").lower() == "correspondent":
        return {"flags": [Flag(field="closing_agent_match", title="Not applicable to Correspondent loans",
                                severity="info", action="ignore",
                                details="CPL and E&O are not required for Correspondent loans.").to_dict()]}

    title_agent = title_fields.get("settlement_agent_name") or title_fields.get("title_company")
    cpl_agent = (cpl_fields or {}).get("agent_name")
    eo_agent = (eo_fields or {}).get("agent_name")

    names = {"title": title_agent, "cpl": cpl_agent, "eo": eo_agent}
    present = {k: v for k, v in names.items() if v}
    if len(present) < 2:
        flags.append(
            Flag(field="closing_agent_match", title="Not enough agent names to cross-match",
                 severity="warning", action="ptf",
                 details=f"Only {len(present)} of 3 documents have an extracted agent name "
                         f"({', '.join(present.keys()) or 'none'}).").to_dict()
        )
    else:
        normalized = {k: str(v).strip().lower() for k, v in present.items()}
        if len(set(normalized.values())) > 1:
            flags.append(
                Flag(field="closing_agent_match", title="Agent name mismatch across documents",
                     severity="critical", action="critical",
                     details=f"Agent names differ: {present}.",
                     suggestion="Confirm the same agent is named on the Title Commitment, CPL, "
                                "and E&O policy.").to_dict()
            )
        else:
            flags.append(
                Flag(field="closing_agent_match", title="Agent name matches across all documents",
                     severity="info", action="ignore",
                     details=f"'{list(present.values())[0]}' matches across {', '.join(present.keys())}.").to_dict()
            )

    if cpl_fields:
        cpl_effective = parse_date(cpl_fields.get("effective_date"))
        closing_date = parse_date(loan_context.get("estimated_closing_date"))
        if cpl_effective and closing_date:
            age = days_between(cpl_effective, closing_date)
            if age > CPL_MAX_AGE_DAYS:
                flags.append(
                    Flag(field="cpl_age", title="CPL effective date too old",
                         severity="critical", action="critical",
                         details=f"CPL is {age} days from closing (max {CPL_MAX_AGE_DAYS}).",
                         suggestion="Request an updated CPL.").to_dict()
                )
            else:
                flags.append(
                    Flag(field="cpl_age", title="CPL within age requirement",
                         severity="info", action="ignore",
                         details=f"CPL is {age} days from closing (<= {CPL_MAX_AGE_DAYS}).").to_dict()
                )

    if eo_fields:
        coverage = to_number(eo_fields.get("coverage_amount"))
        is_attorney = bool(as_bool(eo_fields.get("is_attorney")))
        min_required = E_AND_O_MIN_COVERAGE_ATTORNEY if is_attorney else E_AND_O_MIN_COVERAGE
        if coverage is not None and coverage < min_required:
            flags.append(
                Flag(field="eo_coverage", title="E&O coverage below minimum",
                     severity="critical", action="critical",
                     details=f"E&O coverage {format_currency(coverage)} is below the "
                             f"{format_currency(min_required)} minimum "
                             f"({'attorney' if is_attorney else 'non-attorney'}).",
                     suggestion="Request an updated E&O policy meeting the minimum coverage.").to_dict()
            )
        elif coverage is not None:
            flags.append(
                Flag(field="eo_coverage", title="E&O coverage meets minimum",
                     severity="info", action="ignore",
                     details=f"E&O coverage {format_currency(coverage)} >= "
                             f"{format_currency(min_required)}.").to_dict()
            )

    return {"flags": flags}


# ---------------------------------------------------------------------------
# J-T11: Wire Instructions — beneficiary must match title agent (wire-fraud
# prevention). Never modeled in any of the 3 source repos before this.
# ---------------------------------------------------------------------------

def validate_wire_instructions(
    wire_fields: Dict[str, Any],
    title_fields: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify wire instructions' beneficiary matches the title/settlement
    agent on the title commitment (RM Job Aid p.17, J-T11). This is a
    wire-fraud prevention check — treat any mismatch as critical, never
    silently pass on a missing comparison.
    """
    beneficiary = wire_fields.get("beneficiary_name")
    title_agent = title_fields.get("settlement_agent_name") or title_fields.get("title_company")

    if not beneficiary or not title_agent:
        return {
            "flags": [
                Flag(field="wire_beneficiary_match", title="Cannot verify wire beneficiary",
                     severity="critical", action="critical",
                     details=f"Beneficiary name ({beneficiary!r}) or title agent name "
                             f"({title_agent!r}) is missing — wire fraud checks must not be "
                             f"skipped silently.",
                     suggestion="Do not release wire instructions until the beneficiary can be "
                                "confirmed against the title commitment.").to_dict()
            ]
        }

    if str(beneficiary).strip().lower() != str(title_agent).strip().lower():
        return {
            "flags": [
                Flag(field="wire_beneficiary_match", title="WIRE FRAUD RISK: beneficiary does not match title agent",
                     severity="critical", action="critical",
                     details=f"Wire beneficiary '{beneficiary}' does not match the title/"
                             f"settlement agent '{title_agent}' on the title commitment.",
                     suggestion="STOP. Verbally verify wire instructions directly with the title "
                                "company using a known phone number before releasing funds.").to_dict()
            ]
        }

    return {
        "flags": [
            Flag(field="wire_beneficiary_match", title="Wire beneficiary matches title agent",
                 severity="info", action="ignore",
                 details=f"'{beneficiary}' matches the title commitment's settlement agent.").to_dict()
        ]
    }


# ---------------------------------------------------------------------------
# J-T12: NY Closing Attorney approved-list check
# ---------------------------------------------------------------------------

def validate_ny_closing_attorney(
    attorney_name: Optional[str],
    approved_attorneys: List[str],
    property_state: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify the closing attorney is on the NQMF-approved list (RM Job Aid
    p.18, J-T12). Only applies to NY properties. `approved_attorneys` is
    passed in (not hardcoded) since it's sourced from an external
    intranet-hosted list that changes independently of this codebase —
    same "parameterize, don't hardcode" pattern as `mortgagee_clauses`.
    """
    if (property_state or "").upper() != "NY":
        return {"flags": [Flag(field="ny_attorney", title="Not applicable",
                                severity="info", action="ignore",
                                details="Only NY properties require closing-attorney approval.").to_dict()]}

    if not attorney_name:
        return {"flags": [Flag(field="ny_attorney", title="Closing attorney not identified",
                                severity="warning", action="ptf",
                                details="No closing attorney name provided for this NY property.").to_dict()]}

    approved_normalized = {a.strip().lower() for a in approved_attorneys}
    if attorney_name.strip().lower() in approved_normalized:
        return {"flags": [Flag(field="ny_attorney", title="Closing attorney is approved",
                                severity="info", action="ignore",
                                details=f"'{attorney_name}' is on the NQMF approved list.").to_dict()]}

    return {
        "flags": [
            Flag(field="ny_attorney", title="Closing attorney not on approved list",
                 severity="critical", action="critical",
                 details=f"'{attorney_name}' is not on the NQMF-approved NY attorney list.",
                 suggestion="Submit a request for attorney approval to compliance via the form "
                            "on Operations / NY Attorney Process.").to_dict()
        ]
    }


# ---------------------------------------------------------------------------
# T4 / "Verify tax information": validate_property_taxes
#
# This was the single biggest gap found across all 3 source repos AND this
# repo's own first pass — tax_summary.json was loaded for search_documents()
# but nothing validated a tax amount against it. Implements RM Job Aid p.14
# directly (NOT "per title commitment business rule #13" — no such rule was
# ever found anywhere; see architecture doc Part 2 item 6 for that
# unresolved spec question, which this function does not attempt to answer).
# ---------------------------------------------------------------------------

TAX_BILL_DUE_WINDOW_DAYS = 60  # RM Job Aid p.14
NEW_CONSTRUCTION_FALLBACK_PCT = 0.015  # 1.5% of improved value, if no rate/smartasset available
CA_PURCHASE_ASSESSED_VALUE_PCT = 0.0125  # 1.25% of revised assessed value


def validate_property_taxes(
    tax_fields: Optional[Dict[str, Any]],
    loan_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify property tax amount per RM Job Aid p.14's scenario-specific rules.

    `loan_context` expects:
      - transaction_type: "purchase" | "refinance" | "new_construction"
      - property_state: e.g. "CA"
      - days_to_closing: int (for the 60-day current-tax-bill rule)
      - sales_price, appraised_value (for new construction)
      - title_confirmed_tax_amount, title_confirmed_millage_rate (optional)
      - smartasset_estimate (optional fallback)
      - revised_assessed_value (for CA purchases)
      - has_property_exemption / prior_owner_exemption (for purchases with
        exemptions)
      - proposed_tax_amount: the amount currently entered in Encompass,
        to compare against the computed/expected amount

    Returns the expected tax amount per the applicable scenario, plus flags
    for the 60-day tax bill rule and any mismatch against
    `proposed_tax_amount`.
    """
    loan_context = loan_context or {}
    tax_fields = tax_fields or {}
    flags: List[Dict[str, Any]] = []
    transaction_type = (loan_context.get("transaction_type") or "").lower()

    # -- 60-day current tax bill rule (applies to all scenarios) --
    days_to_closing = loan_context.get("days_to_closing")
    has_current_tax_bill = bool(tax_fields.get("annual_taxes")) and not as_bool(loan_context.get("tax_bill_stale"))
    if days_to_closing is not None and days_to_closing <= TAX_BILL_DUE_WINDOW_DAYS and not has_current_tax_bill:
        flags.append(
            Flag(field="current_tax_bill", title="No current tax bill within 60 days of closing",
                 severity="critical", action="critical",
                 details=f"Closing in {days_to_closing} days but no current tax bill is on file; "
                         "the tax condition cannot be cleared without one.",
                 suggestion="Request a current tax bill from title, or pull current figures from "
                            "https://publicrecords.netronline.com/.").to_dict()
        )

    # -- Determine the expected tax amount by scenario --
    expected_amount: Optional[float] = None
    method = ""

    title_confirmed_amount = to_number(loan_context.get("title_confirmed_tax_amount"))
    title_confirmed_rate = to_number(loan_context.get("title_confirmed_millage_rate"))
    smartasset_estimate = to_number(loan_context.get("smartasset_estimate"))

    if transaction_type == "new_construction":
        sales_price = to_number(loan_context.get("sales_price")) or 0.0
        appraised_value = to_number(loan_context.get("appraised_value")) or 0.0
        base_value = max(sales_price, appraised_value)
        if title_confirmed_rate is not None and base_value:
            expected_amount = title_confirmed_rate * base_value
            method = "title-confirmed millage rate x greater of sales price/appraised value"
        elif smartasset_estimate is not None:
            expected_amount = smartasset_estimate
            method = "smartasset.com/taxes/property-taxes (no title-confirmed rate available)"
        elif base_value:
            expected_amount = NEW_CONSTRUCTION_FALLBACK_PCT * base_value
            method = f"{NEW_CONSTRUCTION_FALLBACK_PCT:.1%} of improved value (fallback — no rate or smartasset available)"

    elif transaction_type == "purchase" and (loan_context.get("property_state") or "").upper() == "CA":
        if title_confirmed_amount is not None:
            expected_amount = title_confirmed_amount
            method = "title-confirmed tax amount"
        else:
            assessed_value = to_number(loan_context.get("revised_assessed_value"))
            if assessed_value is not None:
                expected_amount = CA_PURCHASE_ASSESSED_VALUE_PCT * assessed_value
                method = f"{CA_PURCHASE_ASSESSED_VALUE_PCT:.2%} of revised assessed value"
            elif smartasset_estimate is not None:
                expected_amount = smartasset_estimate
                method = "smartasset.com (lower-tax-rate county fallback)"

    elif transaction_type == "purchase" and as_bool(loan_context.get("has_property_exemption")):
        annual_taxes = to_number(tax_fields.get("annual_taxes"))
        exemptions = tax_fields.get("exemptions") or []
        if not has_current_tax_bill:
            flags.append(
                Flag(field="property_tax_exemption", title="Smartasset not usable for exempt purchases",
                     severity="warning", action="ptf",
                     details="Smartasset is not an option for the purchase of an existing "
                             "dwelling with exemptions; a current tax bill is required.").to_dict()
            )
        elif annual_taxes is not None:
            expected_amount = annual_taxes
            method = "current tax bill amount"
            if exemptions and not as_bool(loan_context.get("is_primary_residence")):
                flags.append(
                    Flag(field="property_tax_exemption", title="Prior-owner exemption must be excluded",
                         severity="warning", action="ptf",
                         details=f"Tax bill lists exemption(s) {exemptions} for the prior owner; "
                                 f"since the new transaction is not a primary residence, these "
                                 f"must NOT be reflected in the buyer's tax estimate. Recompute "
                                 f"without the exemption, and document the calculation via a "
                                 f"processor cert or Notepad entry.",
                         suggestion="Add a processor cert to Property: Property Taxes (TPO) "
                                    "showing how the unexempted figure was derived.").to_dict()
                )

    elif transaction_type == "purchase":
        # Ordinary purchase of an existing dwelling: not new construction, not
        # CA (which uses the assessed-value reassessment rule above), and no
        # property-tax exemption in play. The job aid doesn't call out a
        # special formula for this (by far the most common) case -- it's
        # simply "confirm the current tax bill," same as a refinance.
        annual_taxes = to_number(tax_fields.get("annual_taxes"))
        if not has_current_tax_bill:
            if smartasset_estimate is not None:
                expected_amount = smartasset_estimate
                method = "smartasset.com/taxes/property-taxes (no current tax bill on file)"
            else:
                flags.append(
                    Flag(field="property_tax_purchase", title="Current tax bill not confirmed",
                         severity="warning", action="ptf",
                         details="No current tax bill is on file to verify Section 900 of the "
                                 "2015 Itemization against.",
                         suggestion="Request a current tax bill from title, or pull current "
                                    "figures from smartasset.com/taxes/property-taxes or "
                                    "https://publicrecords.netronline.com/.").to_dict()
                )
        else:
            expected_amount = annual_taxes
            method = "current tax bill amount"

    elif transaction_type == "refinance":
        annual_taxes = to_number(tax_fields.get("annual_taxes"))
        if not has_current_tax_bill:
            flags.append(
                Flag(field="property_tax_refinance", title="Current-year taxes not confirmed",
                     severity="critical", action="critical",
                     details="Refinances require the current year's taxes; smartasset is not an "
                             "option, and the tax condition cannot be cleared without a current "
                             "tax bill.",
                     suggestion="Request a current tax bill from title or pull it from "
                                "https://publicrecords.netronline.com/.").to_dict()
            )
        else:
            expected_amount = annual_taxes
            method = "current-year tax bill"

    if as_bool(tax_fields.get("is_delinquent_60_plus_days")):
        flags.append(
            Flag(field="property_tax_delinquent", title="Taxes 60+ days delinquent",
                 severity="warning", action="ptf",
                 details="Per RM Job Aid p.14, 60+ day delinquent taxes must be listed as a "
                         "payoff on URLA Page 3, NOT in Section 900 of the Itemization.").to_dict()
        )

    if expected_amount is not None:
        proposed_amount = to_number(loan_context.get("proposed_tax_amount"))
        if proposed_amount is not None and abs(proposed_amount - expected_amount) > 1.0:
            flags.append(
                Flag(field="property_tax_amount", title="Proposed tax amount does not match expected",
                     severity="warning", action="overwrite",
                     details=f"Encompass shows {format_currency(proposed_amount)}, but the "
                             f"expected amount per {method} is {format_currency(expected_amount)}.",
                     suggestion="Update Section 900 of the 2015 Itemization to the expected amount.").to_dict()
            )
        else:
            flags.append(
                Flag(field="property_tax_amount", title="Tax amount verified",
                     severity="info", action="ignore",
                     details=f"Expected amount per {method}: {format_currency(expected_amount)}.").to_dict()
            )
    elif not flags:
        flags.append(
            Flag(field="property_tax_amount", title="Could not determine expected tax amount",
                 severity="warning", action="ptf",
                 details="Insufficient inputs for the applicable scenario "
                         f"('{transaction_type or 'unknown'}') to compute an expected tax amount.").to_dict()
        )

    return {"expected_amount": expected_amount, "method": method or None, "flags": flags}
