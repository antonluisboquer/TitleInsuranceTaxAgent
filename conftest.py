"""Makes `core`, `appraisal_collateral_agent`, and `title_insurance_tax_agent`
importable from the repo root without an installed package."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
