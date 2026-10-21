# Title/Insurance/Taxes Agent

Standalone implementation of the Title/Insurance/Taxes Agent spec documented
in `LG-discOrch/docs/APPRAISAL_TITLE_INSURANCE_TAX_AGENTS_ARCHITECTURE.md`
(a sibling repo's architecture doc — see that file for the full spec text
and the RM Job Aid cross-reference this build is based on).

This repo was split out of the original combined `LG-AppraisalTitleTaxAgents`
repo so this agent and the
[Appraisal & Collateral Agent](../LG-AppraisalCollateralAgent) can be
developed, tested, and deployed independently. The split is clean: this
agent has no import dependency on the other agent, and never did.

This repo is **standalone** — it does not import from or modify
`LG-discOrch`, `LG-docsOrch`, or `lg-RMOrch`. Where it ports business logic
verified to already work in one of those repos, the source is cited directly
above the code (grep for `lg-RMOrch`, `LG-docsOrch`, or `LG-discOrch` in the
`core/` and agent modules).

## What's implemented

Built as a [LangGraph](https://langchain-ai.github.io/langgraph/)
tool-calling react agent, invoked as a step in a sequential pipeline (not
event-driven — see **Deferred** below), with a configurable LLM
("GPT-5 mini" per spec; defaults to `gpt-4o-mini`, swap via env var once that
identifier is available) for document-validation reasoning, plus plain
Python for the escrow math (per the spec's "DETERMINISTIC" requirement).

**Status**: every job-aid item that could be modeled as a pure, testable
function is implemented — 151 unit tests (145 synthetic + golden-fixture
tests, 6 skipped — mostly when the sibling `LG-docsOrch` checkout or
`chromadb` isn't present), all deterministic, no API key required (plus an
optional ChromaDB proof-of-concept, see T1 below, that needs
`pip install chromadb` but still no API key).

| Spec tool | Status | Notes |
|---|---|---|
| `search_documents(query, loan_guid)` | ✅ (MVP) | Deterministic bucket-name + keyword matcher, **not** semantic/vector search (see T1 below) |
| `validate_title_commitment(document)` | ✅ (minus chain of title) | Lender coverage ≥ loan amount, 90-day effective-date window, **3-way effective-date consistency (J-T1)**, mortgagee-clause reminder |
| `validate_insurance_coverage(document, loan_requirements)` | ✅ | HOI deductible ≤5%, flood deductible ≤$10K/$25K, 12-month term (J-T2), **wind/hail/hurricane coverage type (J-T5)**, **DSCR 6-month rent-loss type (J-T7)**, **condo HO6 deductible cap (J-T8)**, **flood "lesser of 3" rule (J-T6)** |
| `calculate_escrow(property_taxes, insurance_premium, months)` | ✅ | Simple flat 2-item cushion formula, per spec's "DETERMINISTIC" requirement |
| `calculate_aggregate_escrow(disbursement_items, months)` | ✅ | J-T9 — full multi-item Aggregate Escrow Account using the RESPA lowest-balance method; computes a *real* aggregate adjustment (mathematically guaranteed ≤ 0, matching the job aid's "always negative" rule) instead of a hardcoded assertion |
| `validate_property_taxes(tax_fields, loan_context)` | ✅ | New construction (title millage rate), CA purchases (1.25% of assessed value), purchases with exemptions (excludes prior-owner homestead), refinances, and the 60-day current-tax-bill rule |
| `validate_premium_paid_in_full` | ✅ | J-T3 — paid receipt vs. invoice-at-closing, including the refinance 60-day rule |
| `validate_no_coverage_gap` | ✅ | J-T4 — old-policy-expiration vs. new-policy-effective date-range + binder-timing check |
| `validate_closing_agent_match` | ✅ | J-T10 — CPL/E&O/Title 3-way agent-name cross-match, plus CPL age and E&O coverage minimums |
| `validate_wire_instructions` | ✅ | J-T11 — wire-fraud prevention: beneficiary must match the title commitment's settlement agent. `wire_instructions.json` schema — never modeled in any of the 3 source repos before this |
| `validate_ny_closing_attorney` | ✅ | J-T12 — approved-list check, parameterized (not hardcoded) like `mortgagee_clauses` |

## What's intentionally deferred (backlog)

| Item | Why deferred |
|---|---|
| **T5** Event-driven triggers | All 3 source repos run as sequential batch orchestrators; this repo matches that pattern |
| **T2** Chain of title | Standard title commitments don't carry historical ownership data — needs a new document source (chain-of-title report / deed-history pull) to be scoped first. `validate_title_commitment` returns `chain_of_title_verified: False` and does not claim to check it |
| **T1 (full)** ChromaDB/vector search | The architecture doc explicitly asks "is bucket-based lookup sufficient?" — this repo implements the bucket/keyword MVP as the primary tool, plus a local ChromaDB proof-of-concept (`core/semantic_search.py`) to answer that question with evidence instead of guessing. See below |
| **T4 "rule #13"** Tax info "per title commitment business rule #13" | No repo (including this one) has ever found a documented "rule #13." `validate_property_taxes` implements the actual job-aid tax rules sourced from the Tax Summary document instead |

### ChromaDB semantic search: what the local POC found, and where it needs to go

`core/semantic_search.py` + `examples/run_semantic_search_demo.py` is a real,
runnable local proof-of-concept — not a stub — answering the architecture
doc's own question: "is bucket-based lookup sufficient, or does this
genuinely need semantic search over document content?"

**Run it**: `pip install chromadb && python3 examples/run_semantic_search_demo.py`
(no OpenAI key needed — Chroma's default embedding model runs fully offline
after a one-time ~80MB download).

**What it found** — two real bugs in the *existing* bucket/keyword MVP:

1. `title_report.json`'s `attachment_avoid: [... "deed", ...]` list (meant to
   filter out a standalone recorded Deed document sharing the same eFolder
   bucket) was being checked against a document's full body text, not just
   its filename, which was silently dropping legitimate title commitments
   from search results (a real commitment's own chain-of-title narrative
   legitimately discusses prior recorded deeds). Same risk existed for
   `flood_insurance.json` ("zone") and `appraisal_report.json` ("license").
   **Fixed**: `attachment_avoid` is now checked against filename only.
2. `DOCUMENT_TYPE_INDEX` only ever had the original 4 schemas —
   `cpl.json`, `errors_omissions.json`, `wire_instructions.json`, and
   `insurance_receipt.json` were added later but never registered, so
   `search_documents` couldn't find the very documents those validation
   functions need. Registering them exposed a second problem:
   `errors_omissions.json` and `wire_instructions.json` claim the same
   eFolder bucket ("Property: Title Company Documents") — a real,
   intentional overlap the original "first schema to claim a bucket wins"
   logic couldn't handle. **Fixed**: each document is now resolved against
   every doc type that claims its bucket individually.

Regression tests: `test_search_documents_avoid_checks_filename_not_body_text`,
`test_search_documents_avoid_still_excludes_misfiled_deed_by_filename`,
`test_search_documents_disambiguates_shared_bucket_by_doc_type`,
`test_search_documents_finds_cpl_eo_and_wire_docs_now_registered`.

**What it also found**: for the motivating "yes/no, does this document
mention chain of title" question, a semantic vector search is actually
overkill — a plain substring check on the full document text would answer
it just as well, and far more cheaply. Semantic search earns its cost on
genuinely fuzzy, paraphrased free-text queries where keyword matching misses
synonyms — not on presence/absence checks for a known phrase.

**Gap list to take this from POC to production**:
1. **Real document text.** Neither this repo nor `LG-docsOrch`'s own
   regression fixtures store full raw document text anywhere — only
   structured per-field extraction results and a pointer to the original
   file in blob storage. You'd need an OCR/text-extraction step per
   document first.
2. **Swap the embedding model.** The POC uses Chroma's local default
   (`all-MiniLM-L6-v2`); the spec calls for OpenAI `text-embedding-3-small`.
3. **Real chunking.** `chunk_text()` is a naive word-count splitter;
   production should chunk by token count and respect page boundaries.
4. **Persistence.** The POC uses `chromadb.EphemeralClient()`; production
   needs `PersistentClient` or a hosted Chroma instance.
5. **Wire it into the agent as an actual tool** (replacing or supplementing
   `search_documents_tool` in `agent.py`) once 1-4 are in place — right now
   it's a standalone module, not exposed to the LLM agent.

### Still worth a second pass (modeled, but with simplifying assumptions)
- `validate_property_taxes`'s new-construction/CA formulas use the job aid's literal percentages (1.5% fallback, 1.25% CA) — real county millage/assessment data would be more accurate when available (the function already prefers `title_confirmed_tax_amount`/`title_confirmed_millage_rate` when present)
- `calculate_aggregate_escrow`'s lowest-balance simulation assumes deposits/disbursements land cleanly within a 12-month window starting from loan consummation; it doesn't model mid-cycle escrow analysis/reserve-shortage adjustments from an existing, already-funded account

### Schema cross-check against LG-docsOrch / LG-discOrch's production extraction schemas
Diffed every `tool_schema/*.json` file's `properties` against LG-docsOrch's real (DynamoDB-sourced) generated schemas and LG-discOrch's static ones. `errors_omissions.json`, `wire_instructions.json`, and `insurance_receipt.json` are genuinely novel to this repo (no sibling-repo equivalent exists at all).
- **`evidence_of_insurance.json` gained `replacement_cost`.** `validate_insurance_coverage`'s HOI-coverage check now implements the real "lesser of loan amount or replacement cost" rule, matching the flood side's already-implemented "lesser of 3" calc (J-T6).
- **`evidence_of_insurance.json`'s existing-but-never-checked `loss_payee_clause` field is now wired up**, compared against the same `mortgagee_clauses` table used by `validate_title_commitment`'s reminder.
- **`flood_insurance.json` gained `in_sfha`.** If a flood document itself states `in_sfha`, that's used (and overrides a stale external flag) — falling back to `loan_context["is_flood_zone"]` only when the document doesn't say.

## Design notes

- **Same flag vocabulary as LG-discOrch.** `core/models.py`'s `Flag` shape
  (`ignore` / `overwrite` / `ptf` / `critical`) matches the vocabulary used
  across the other agent repos so output merges cleanly into an existing
  orchestrator's review-result shape.
- **Organization-specific values are parameterized, not hardcoded.** The
  mortgagee-clause table in `title_insurance_tax_agent/tools.py` defaults to
  the NQMF/RM-Job-Aid example values but is passed as an argument
  (`mortgagee_clauses=`) — swap in your own lender's clause table.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY to run the LLM-orchestrated agent

# Unit tests (151 tests, all deterministic — no API key required)
pytest tests/ -v

# Optional: the ChromaDB semantic-search POC + its tests (pip install chromadb first)
python3 examples/run_semantic_search_demo.py
pytest tests/test_semantic_search.py -v

# Deterministic pipeline demo (no API key required)
python3 examples/run_title_insurance_tax_agent_demo.py
```

### Golden-fixture tests (optional, against real loan data)

`tests/test_golden_fixtures.py` replays real, previously-closed loan field
data from LG-docsOrch's own regression-test fixtures
(`LG-docsOrch/tests/regression/fixtures/*/frozen_state.json`) through this
repo's tools, via the field-name adapter in `tests/golden_fixtures.py`. This
catches things hand-written fixtures don't: real currency-string formats,
real date formats, nested contact dicts, and fields that are sometimes just
missing. (It's how the "ordinary purchase" branch of `validate_property_taxes`
got added — the original scenario list only covered new-construction/CA/
exempt-purchase/refinance, and every golden purchase fixture fell through
with no expected amount.)

It is NOT a comparison against LG-docsOrch's own `golden_flags.json` —
this repo checks different (and additional) rules than docsOrch does, so
there's no meaningful cross-repo "expected output" to assert against.

**Real bug this caught: string-shaped booleans.** Several golden fixtures
have boolean-typed fields extracted as the *literal string* `"False"`, not
a Python `bool`. `bool("False")` is `True` in Python, so every
`if fields.get("some_flag"):` check across this codebase was silently
broken for that shape. `core/booleans.py::as_bool()` is the fix — it's used
at every boolean-field call site in this agent (`in_sfha`, `paid_in_full`,
`has_property_exemption`, `is_primary_residence`,
`is_delinquent_60_plus_days`, `is_dscr`, `is_attorney`), each with its own
regression test proving the string-`"False"` case now resolves correctly.

Automatically skipped if `LG-docsOrch` isn't checked out as a sibling
directory. Point `DOCSORCH_FIXTURES_DIR` elsewhere if your layout differs:

```bash
DOCSORCH_FIXTURES_DIR=/path/to/LG-docsOrch/tests/regression/fixtures pytest tests/test_golden_fixtures.py -v
```

To run the full LLM-orchestrated agent (a tool-calling react agent that
decides which tool to call and summarizes the results itself), set
`OPENAI_API_KEY` and see the commented-out snippet at the bottom of
`examples/run_title_insurance_tax_agent_demo.py`.

## Repo layout

```
core/                           # agent-scoped infrastructure (not shared with the appraisal agent)
  money.py, dates.py            # parsing helpers
  models.py                     # Flag dataclass (ignore/overwrite/ptf/critical)
  schema_loader.py              # loads tool_schema/*.json
  booleans.py                   # as_bool() -- normalizes "True"/"False" strings from real extraction data
  semantic_search.py            # ChromaDB semantic-search POC (T1 research, not yet wired into the agent)

tool_schema/
  title_report.json
  evidence_of_insurance.json    # + wind/hail, DSCR rent-loss type, HO6 fields (J-T5/7/8)
  flood_insurance.json          # + replacement_cost_of_improvements, in_sfha (J-T6)
  tax_summary.json              # + millage_rate, assessed_value, exemptions (T4)
  cpl.json                      # Closing Protection Letter (J-T10)
  errors_omissions.json         # E&O policy (J-T10)
  wire_instructions.json        # never modeled in any of the 3 source repos (J-T11)
  insurance_receipt.json        # premium paid-in-full / invoice (J-T3)

title_insurance_tax_agent/
  tools.py                      # pure functions (unit-testable without an LLM)
  agent.py                      # LangGraph react-agent wrapper

tests/                          # 151 tests, all deterministic
examples/                       # runnable demo scripts
```

## Relationship to the Appraisal & Collateral Agent

This repo and [`LG-AppraisalCollateralAgent`](../LG-AppraisalCollateralAgent)
were split from a single combined repo (`LG-AppraisalTitleTaxAgents`). Each
repo carries its own full copy of the shared `core/` infrastructure
(`booleans`, `dates`, `models`, `money`, `schema_loader`) rather than
depending on a shared package — there was no existing shared-package
registry in this org to depend on, and the two agents' business logic is
otherwise fully independent. If `core/` grows substantially shared surface
area in the future, consider extracting it into its own installable package
rather than hand-syncing bug fixes (like the `as_bool()` fix above) across
both repos.
