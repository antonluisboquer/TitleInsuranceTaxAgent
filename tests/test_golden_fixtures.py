"""Golden-fixture-driven tests: exercise this repo's tools against real,
previously-closed loan field data from LG-docsOrch's regression-test
fixtures, instead of only hand-written synthetic data.

Why: hand-written test fixtures tend to be "clean" -- round numbers, ISO
dates, no missing fields. Real extracted data is messier ("$1,890.83" not
1890.83, "3/20/2026" not "2026-03-20", nested contact dicts, fields that are
sometimes just missing) and is a much better test of whether our tools are
actually robust. Replaying these fixtures through validate_property_taxes
is in fact what surfaced a real gap: see the "ordinary purchase" branch
added to validate_property_taxes in title_insurance_tax_agent/tools.py.

This is deliberately NOT a snapshot/expected-output comparison against
docsOrch's own golden_flags.json: this repo checks different (and
additional) business rules than docsOrch does, so there's no meaningful
cross-repo "expected flags" to assert against. Instead these tests assert:
  1. Every tool runs without raising on real-shaped field data.
  2. A handful of outcomes independently verifiable from the raw numbers
     in the fixture (e.g. a known deductible-vs-cap relationship).

Skips automatically if the sibling LG-docsOrch repo / its fixtures aren't
present on disk. Set DOCSORCH_FIXTURES_DIR to point elsewhere if your
checkout layout differs.
"""
import pytest

from core.money import to_number
from title_insurance_tax_agent.tools import (
    validate_closing_agent_match,
    validate_insurance_coverage,
    validate_property_taxes,
    validate_title_commitment,
)

from tests.golden_fixtures import (
    available_fixtures,
    build_cpl_fields,
    build_insurance_fields,
    build_loan_context,
    build_tax_fields,
    build_title_fields,
    load_frozen_state,
)

FIXTURES = available_fixtures()

pytestmark = pytest.mark.skipif(
    not FIXTURES,
    reason=(
        "LG-docsOrch golden fixtures not found on disk (sibling repo not checked "
        "out next to this one, or set DOCSORCH_FIXTURES_DIR to point at "
        "LG-docsOrch/tests/regression/fixtures)."
    ),
)


@pytest.fixture(params=FIXTURES)
def golden_loan(request):
    name = request.param
    state = load_frozen_state(name)
    doc_fields = state.get("doc_fields", {}) or {}
    return {
        "name": name,
        "title": build_title_fields(doc_fields),
        "insurance": build_insurance_fields(doc_fields),
        "tax": build_tax_fields(doc_fields),
        "cpl": build_cpl_fields(doc_fields),
        "loan_context": build_loan_context(state),
    }


def test_fixtures_were_discovered():
    # Sanity check on the adapter itself, not the tools -- if this starts
    # failing, DOCSORCH_FIXTURES_DIR / the sibling checkout has moved.
    assert len(FIXTURES) >= 5


def test_validate_title_commitment_on_every_golden_loan(golden_loan):
    result = validate_title_commitment(golden_loan["title"], golden_loan["loan_context"])
    assert "flags" in result and result["flags"]
    # Never fabricated -- this repo has no chain-of-title data source.
    assert result["chain_of_title_verified"] is False


def test_validate_insurance_coverage_on_every_golden_loan(golden_loan):
    hoi = golden_loan["insurance"]
    if not hoi.get("policy_number"):
        pytest.skip(f"{golden_loan['name']}: no Evidence of Insurance in this fixture")

    result = validate_insurance_coverage(hoi, golden_loan["loan_context"])
    assert "flags" in result and result["flags"]

    coverage = to_number(hoi.get("coverage_amount"))
    deductible = to_number(hoi.get("deductible"))
    if coverage and deductible is not None:
        deductible_flags = [f for f in result["flags"] if f["field"] == "deductible"]
        assert deductible_flags, f"{golden_loan['name']}: expected a deductible flag"
        expected_severity = "critical" if deductible > 0.05 * coverage else "info"
        assert deductible_flags[0]["severity"] == expected_severity, (
            f"{golden_loan['name']}: deductible={deductible} coverage={coverage} "
            f"(5% cap={0.05 * coverage})"
        )


def test_validate_closing_agent_match_on_every_golden_loan(golden_loan):
    # No E&O document in any of these fixtures (docsOrch doesn't extract it
    # either), so this only ever cross-matches Title vs. CPL -- but that's
    # still a real 2-document comparison against real closing-agent names.
    result = validate_closing_agent_match(golden_loan["title"], golden_loan["cpl"], None, golden_loan["loan_context"])
    assert "flags" in result and result["flags"]

    title_agent = golden_loan["title"].get("settlement_agent_name")
    cpl_agent = golden_loan["cpl"].get("agent_name")
    match_flags = [f for f in result["flags"] if f["field"] == "closing_agent_match"]
    assert match_flags, f"{golden_loan['name']}: expected a closing_agent_match flag"

    if title_agent and cpl_agent:
        expected_severity = "info" if title_agent.strip().lower() == cpl_agent.strip().lower() else "critical"
        assert match_flags[0]["severity"] == expected_severity, (
            f"{golden_loan['name']}: title_agent={title_agent!r} cpl_agent={cpl_agent!r}"
        )
    else:
        assert match_flags[0]["severity"] == "warning"


def test_validate_property_taxes_on_every_golden_loan(golden_loan):
    result = validate_property_taxes(golden_loan["tax"], golden_loan["loan_context"])
    assert "flags" in result and result["flags"]

    annual_taxes = golden_loan["tax"].get("annual_taxes")
    if annual_taxes:
        # Every golden fixture has a real, current Tax Summary document on
        # file, so every scenario here should resolve to an expected amount
        # equal to that tax bill (none of these are CA/new-construction/
        # exemption cases -- see build_loan_context).
        assert result["expected_amount"] == pytest.approx(to_number(annual_taxes)), (
            f"{golden_loan['name']}: method={result['method']!r} flags={result['flags']}"
        )


def test_validate_insurance_coverage_replacement_cost_on_every_golden_loan(golden_loan):
    """Confirms the 'lesser of loan amount or replacement cost' HOI rule
    runs cleanly against docsOrch's real replacement_cost values, which
    arrive as digit strings (e.g. "272000")."""
    hoi = golden_loan["insurance"]
    if not hoi.get("policy_number") or hoi.get("replacement_cost") is None:
        pytest.skip(f"{golden_loan['name']}: no HOI + replacement_cost in this fixture")

    result = validate_insurance_coverage(hoi, golden_loan["loan_context"])
    coverage_flags = [f for f in result["flags"] if f["field"] == "coverage_amount"]
    assert coverage_flags, f"{golden_loan['name']}: expected a coverage_amount flag"
    # The method string should reflect that replacement_cost was actually used.
    assert "replacement cost" in coverage_flags[0]["details"].lower()
