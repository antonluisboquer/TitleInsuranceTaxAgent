from datetime import date

from core.booleans import as_bool
from core.dates import days_between, months_between, parse_date
from core.money import format_currency, to_number
from core.schema_loader import field_descriptions, load_schema


# ---------------------------------------------------------------------------
# money.py
# ---------------------------------------------------------------------------

def test_to_number_handles_currency_strings():
    assert to_number("$1,234.56") == 1234.56


def test_to_number_handles_percent_strings():
    assert to_number("5%") == 5.0


def test_to_number_handles_plain_numbers():
    assert to_number(42) == 42.0
    assert to_number(42.5) == 42.5


def test_to_number_returns_none_for_missing_or_garbage():
    assert to_number(None) is None
    assert to_number("") is None
    assert to_number("n/a") is None


def test_format_currency():
    assert format_currency(1234.5) == "$1,234.50"
    assert format_currency(None) == "n/a"


# ---------------------------------------------------------------------------
# booleans.py
#
# The string-"False" case is not hypothetical: LG-docsOrch's real extraction
# output stores several boolean-typed fields (e.g. in_sfha, paid_in_full,
# is_delinquent_60_plus_days) as the literal string "False", not a Python
# bool -- confirmed by replaying its golden fixtures (see
# test_golden_fixtures.py). bool("False") is True in Python, which was
# silently inverting these checks before this helper existed.
# ---------------------------------------------------------------------------

def test_as_bool_passes_through_real_bools():
    assert as_bool(True) is True
    assert as_bool(False) is False


def test_as_bool_parses_the_literal_string_false_as_false():
    assert as_bool("False") is False
    assert as_bool("false") is False
    assert as_bool("FALSE") is False


def test_as_bool_parses_the_literal_string_true_as_true():
    assert as_bool("True") is True
    assert as_bool("true") is True


def test_as_bool_parses_yes_no():
    assert as_bool("yes") is True
    assert as_bool("no") is False


def test_as_bool_returns_none_for_missing():
    assert as_bool(None) is None
    assert as_bool("") is None


def test_as_bool_falls_back_to_truthiness_for_other_strings():
    # A free-text value that isn't a recognized true/false spelling is still
    # treated as truthy, matching the pre-existing (pre-bug-fix) behavior for
    # genuinely non-boolean-shaped data.
    assert as_bool("some narrative text") is True


def test_as_bool_handles_non_string_non_bool_values():
    assert as_bool(1) is True
    assert as_bool(0) is False


# ---------------------------------------------------------------------------
# dates.py
# ---------------------------------------------------------------------------

def test_parse_date_handles_common_formats():
    assert parse_date("2026-01-15") == date(2026, 1, 15)
    assert parse_date("01/15/2026") == date(2026, 1, 15)
    assert parse_date(date(2026, 1, 15)) == date(2026, 1, 15)


def test_parse_date_returns_none_for_garbage():
    assert parse_date(None) is None
    assert parse_date("not a date") is None


def test_days_between():
    assert days_between(date(2026, 1, 1), date(2026, 1, 31)) == 30
    assert days_between(date(2026, 1, 31), date(2026, 1, 1)) == 30  # order-independent


def test_months_between_roughly_twelve():
    term = months_between(date(2026, 1, 1), date(2027, 1, 1))
    assert 11.5 <= term <= 12.5


# ---------------------------------------------------------------------------
# schema_loader.py
# ---------------------------------------------------------------------------

def test_field_descriptions():
    schema = load_schema("title_report")
    descs = field_descriptions(schema)
    assert "title_number" in descs
    assert descs["title_number"]  # non-empty
