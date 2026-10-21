"""Date parsing/diffing helpers.

Mirrors `_parse_date` / `_days_between` from lg-RMOrch/output/tools/_helpers.py,
used by review_preliminary_title (90-day title effective-date check) and
review_hazard_flood_insurance (12-month HOI term check, J-T2).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%Y/%m/%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%m/%d/%y",
)


def parse_date(value: Any) -> Optional[date]:
    """Best-effort parse of a date-like extracted/LOS field into a `date`."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    # Try native ISO parsing first (handles "2026-01-15T00:00:00Z" etc.)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def days_between(a: date, b: date) -> int:
    """Absolute number of days between two dates."""
    return abs((b - a).days)


def months_between(a: date, b: date) -> float:
    """Approximate number of months between two dates (30.44 day months)."""
    return abs((b - a).days) / 30.44
