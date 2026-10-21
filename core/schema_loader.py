"""Loads the JSON-Schema (draft-07) document extraction schemas.

These schemas are ported verbatim from LG-discOrch/tools/tool_schema/*.json
(confirmed identical during the architecture review — see
LG-discOrch/docs/APPRAISAL_TITLE_INSURANCE_TAX_AGENTS_ARCHITECTURE.md).
Keeping the same field `description` text matters: that text is what guides
the extraction LLM to the right spot in the document (e.g. "Look in the
RECONCILIATION section...").
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "tool_schema"


@lru_cache(maxsize=None)
def load_schema(name: str) -> Dict[str, Any]:
    """Load a tool_schema/<name>.json file.

    Args:
        name: schema file name without extension, e.g. "appraisal_report"
    """
    path = SCHEMA_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No schema named '{name}' in {SCHEMA_DIR}. "
            f"Available: {sorted(p.stem for p in SCHEMA_DIR.glob('*.json'))}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def field_descriptions(schema: Dict[str, Any]) -> Dict[str, str]:
    """Flatten {field_name: description} for prompt-building."""
    props = schema.get("properties", {})
    return {name: spec.get("description", "") for name, spec in props.items()}
