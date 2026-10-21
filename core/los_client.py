"""LOS (Loan Origination System / Encompass) field-read abstraction.

Both agents' tools depend on this tiny `LOSClient` protocol rather than
calling Encompass directly, so tests/demos can swap in `MockLOSClient` and
production code can swap in a real client without touching tool logic.

Three implementations:
  - `MockLOSClient`: in-memory fixtures, for tests/demos/local dev.
  - `EncompassLOSClient`: real wiring around `core.encompass_client`
    (vendored from LG-ConstructionEncompass — a dependency-light REST client,
    no private `copilotagent` package needed). Confirmed live against Prod
    2026-10-21. This is the one to use in production.
  - `CallableLOSClient`: generic adapter if you'd rather wire in some other
    already-authenticated batched field-read function instead (e.g. one
    living in LG-discOrch/LG-docsOrch).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class LOSClient(Protocol):
    def get_fields(self, loan_guid: str, field_ids: List[str]) -> Dict[str, Any]:
        """Return {field_id: value} for the requested Encompass field IDs."""
        ...


class MockLOSClient:
    """In-memory LOS client for tests/demos/local dev.

    fixtures: {loan_guid: {field_id: value}}
    """

    def __init__(self, fixtures: Dict[str, Dict[str, Any]]):
        self._fixtures = fixtures

    def get_fields(self, loan_guid: str, field_ids: List[str]) -> Dict[str, Any]:
        loan_fields = self._fixtures.get(loan_guid, {})
        return {fid: loan_fields.get(fid) for fid in field_ids}

    def set_fields(self, loan_guid: str, values: Dict[str, Any]) -> None:
        self._fixtures.setdefault(loan_guid, {}).update(values)


class CallableLOSClient:
    """Adapter around an existing batched field-read function.

    Example (production wiring inside LG-discOrch or a service that already
    has an authenticated EncompassConnect client):

        from tools.encompass_client import get_encompass_client

        def _read(loan_guid, field_ids):
            client = get_encompass_client()
            out = {}
            for i in range(0, len(field_ids), 50):
                batch = field_ids[i:i + 50]
                out.update(client.get_field(loan_guid, batch))
            return out

        los = CallableLOSClient(_read)
    """

    def __init__(self, read_fields_fn: Callable[[str, List[str]], Dict[str, Any]]):
        self._read_fields_fn = read_fields_fn

    def get_fields(self, loan_guid: str, field_ids: List[str]) -> Dict[str, Any]:
        return self._read_fields_fn(loan_guid, field_ids)


class EncompassLOSClient:
    """Real LOS client, backed by `core.encompass_client` (vendored from
    LG-ConstructionEncompass). Confirmed live against Prod Encompass
    2026-10-21: resolved a loan by number via `search_loans` and read back
    loan number / borrower name / loan type / property state fields.

    Env (see .env.example):
        TEST_ENCOMPASS_CLIENT_ID / _CLIENT_SECRET / _INSTANCE_ID / _API_BASE_URL
        PROD_ENCOMPASS_CLIENT_ID / _CLIENT_SECRET / _INSTANCE_ID /
            _USERNAME / _PASSWORD / _SUBJECT_USER_ID / _API_BASE_URL

    Usage:
        los = EncompassLOSClient(env="PROD")
        fields = los.get_fields(loan_guid, ["356", "1041", "1894"])

    Read-only by design: only exposes `get_fields` (-> read_fields) and
    `find_loan_guid` (-> search_loans). Does not expose write_fields.
    """

    def __init__(self, env: str = "TEST", use_cache: bool = True):
        self._env = env
        self._use_cache = use_cache

    def _client(self):
        from core.encompass_client import get_encompass_client

        return get_encompass_client(env=self._env, use_cache=self._use_cache)

    def get_fields(self, loan_guid: str, field_ids: List[str]) -> Dict[str, Any]:
        return self._client().read_fields(loan_guid, field_ids)

    def find_loan_guid(self, loan_number: str) -> List[Dict[str, Any]]:
        """Resolve a human-entered loan number to Encompass loan GUID(s).

        Returns a list of {"loanGuid", "loanNumber", "borrowerName"} dicts
        (usually 0 or 1 entries for an exact loan-number match).
        """
        return self._client().search_loans(loan_number)
