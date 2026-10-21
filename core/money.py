"""Small numeric/money parsing helpers.

Mirrors the `to_number()` / `_money()` style helpers used throughout
LG-docsOrch and lg-RMOrch (e.g. lg-RMOrch/output/tools/_helpers.py `to_number`,
LG-docsOrch verify_escrow_cushion.py `_money`) so extracted document fields
like "$1,234.56" or "12.5%" parse cleanly into floats.
"""

from __future__ import annotations

from typing import Any, Optional


def to_number(value: Any) -> Optional[float]:
    """Parse a possibly-currency/percentage-formatted value into a float.

    Returns None if the value is missing or cannot be parsed, rather than
    raising, so callers can treat "not extracted" and "not parseable" the
    same way (both become an info/warning flag, not a crash).
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def percent_of(value: Optional[float], pct: float) -> Optional[float]:
    """Return `pct` percent of `value` (pct as a fraction, e.g. 0.05 for 5%)."""
    if value is None:
        return None
    return value * pct


def format_currency(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"
