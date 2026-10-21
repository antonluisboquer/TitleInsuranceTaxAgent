"""Boolean normalization for extracted document/LOS fields.

Found via tests/test_golden_fixtures.py replaying LG-docsOrch's real
extraction output: boolean-typed fields there (e.g. `appraisal_subject_to_
completion`, `appraisal_subject_to_repairs`) are stored as the *string*
``"False"`` / ``"True"``, not a Python bool. Every `if fields.get("some_
flag"):` / `bool(fields.get("some_flag"))` check in this repo was silently
broken for that shape: `bool("False")` is `True` in Python, since it's a
non-empty string. That meant a document explicitly saying "not subject to
repairs" would have been read as "is subject to repairs" -- a false
positive forcing an unnecessary immediate UW resubmit (and, for
`in_sfha`/flood-zone derivation, a false positive on "flood insurance
required").

`as_bool()` is the single place that normalizes this across the codebase.
"""
from __future__ import annotations

from typing import Any, Optional

_TRUE_STRINGS = {"true", "yes", "y", "1"}
_FALSE_STRINGS = {"false", "no", "n", "0"}


def as_bool(value: Any) -> Optional[bool]:
    """Normalize a possibly-string-shaped boolean into a real bool.

    - ``None`` stays ``None`` (field wasn't extracted -- callers decide the
      default, same as `to_number`'s "not extracted" vs. "not parseable"
      split).
    - Real `bool` passes through unchanged.
    - Strings are matched case-insensitively against common true/false
      spellings ("True"/"False", "yes"/"no", "1"/"0"); anything else falls
      through to plain `bool(str)` truthiness (e.g. a non-empty free-text
      value is still treated as truthy, same as before this helper existed).
    - Any other type (int, etc.) falls back to plain `bool(value)`.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
        return bool(text)
    return bool(value)
