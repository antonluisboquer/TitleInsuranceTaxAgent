"""Shared data shapes used by both agents' tools.

`Flag` standardizes on the same four-action vocabulary used by
LG-discOrch's `determine_flag_action()` (tools/document_review.py) and the
`flags.append({...})` pattern used throughout LG-docsOrch / lg-RMOrch, so
output from this repo's tools can be merged into either orchestrator's
existing review-result shape without translation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

Severity = Literal["info", "warning", "critical"]

# Same vocabulary as LG-discOrch's determine_flag_action() return values.
Action = Literal["ignore", "overwrite", "ptf", "critical"]


@dataclass
class Flag:
    field: str
    title: str
    severity: Severity
    action: Action
    details: str
    suggestion: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "field": self.field,
            "title": self.title,
            "severity": self.severity,
            "action": self.action,
            "details": self.details,
            "suggestion": self.suggestion,
        }
        if self.extra:
            d.update(self.extra)
        return d


def info_flag(field_name: str, title: str, details: str, suggestion: str = "") -> Dict[str, Any]:
    return Flag(field=field_name, title=title, severity="info", action="ignore",
                details=details, suggestion=suggestion).to_dict()


def warning_flag(field_name: str, title: str, details: str, suggestion: str = "",
                  action: Action = "ptf") -> Dict[str, Any]:
    return Flag(field=field_name, title=title, severity="warning", action=action,
                details=details, suggestion=suggestion).to_dict()


def critical_flag(field_name: str, title: str, details: str, suggestion: str = "") -> Dict[str, Any]:
    return Flag(field=field_name, title=title, severity="critical", action="critical",
                details=details, suggestion=suggestion).to_dict()
