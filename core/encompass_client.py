"""Encompass API client, vendored from LG-ConstructionEncompass/encompass_client.py.

Uses the Encompass REST API directly (no copilotagent dependency), so it has
no sibling-repo/private-package dependency the way the LG-discOrch /
LG-docsOrch clients do. Confirmed live against the real Prod Encompass
instance 2026-10-21 (loan search + field read for a test loan).

This agent is a read-only consumer: only use `search_loans`/`search_loans_pipeline`
and `read_fields` from here (via core.los_client.EncompassLOSClient below).
Do NOT call `write_fields` / `manage_other_income_sources` from this codebase
-- those exist because this file is a straight copy of the source client, not
because this agent is meant to write to Encompass.

Authentication: 2-step OAuth (password grant + token exchange for impersonation),
falling back to client_credentials if the password grant fails.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_MODULE_CLIENT_CACHE: dict = {}
_workflow_env: Optional[str] = None


def reset_encompass_state():
    """Reset cached workflow env and client cache for a fresh run."""
    global _workflow_env
    _workflow_env = None
    _MODULE_CLIENT_CACHE.clear()


def _get_env_var(name: str, env_prefix: Optional[str] = None) -> Optional[str]:
    if env_prefix:
        val = os.getenv(f"{env_prefix}{name}")
        if val:
            return val
    return os.getenv(name)


class EncompassClient:
    """Lightweight Encompass REST client with auto-refresh."""

    def __init__(
        self,
        api_base_url: str,
        client_id: str,
        client_secret: str,
        instance_id: str,
        username: str,
        password: str,
        subject_user_id: str = "",
        scope: str = "lp",
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.instance_id = instance_id
        self.username = username
        self.password = password
        self.subject_user_id = subject_user_id
        self.scope = scope or "lp"
        self.access_token: Optional[str] = None
        # Super-admin (password-grant) token, kept even when we impersonate a
        # subject user. Admins can see settings/templates that impersonated
        # users cannot, so this is used for template listing.
        self.actor_token: Optional[str] = None
        # Cached integration-level token (client_credentials grant). Used for
        # settings/template endpoints whose visibility is filtered per-user.
        self._client_credentials_token: Optional[str] = None

    def authenticate(self) -> str:
        """Authenticate via password grant (+ impersonation), falling back to
        the client_credentials grant if the password grant fails."""
        token_url = f"{self.api_base_url}/oauth2/v1/token"

        try:
            # Step 1: Super Admin token
            r1 = requests.post(token_url, data={
                "grant_type": "password",
                "username": f"{self.username}@encompass:{self.instance_id}",
                "password": self.password,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": self.scope,
            }, timeout=15)
            r1.raise_for_status()
            actor_token = r1.json()["access_token"]
            self.actor_token = actor_token

            if self.subject_user_id:
                # Step 2: Impersonation token
                r2 = requests.post(token_url, data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                    "actor_token_type": "urn:ietf:params:oauth:token-type:access_token",
                    "subject_user_id": self.subject_user_id,
                    "actor_token": actor_token,
                    "scope": self.scope,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                }, timeout=15)
                r2.raise_for_status()
                self.access_token = r2.json()["access_token"]
            else:
                self.access_token = actor_token

            logger.info("[ENCOMPASS] Authenticated via password grant")
            return self.access_token
        except Exception as exc:
            logger.warning(
                f"[ENCOMPASS] Password grant failed ({exc}); "
                "falling back to client_credentials"
            )
            self.access_token = self.get_client_credentials_token(force=True)
            return self.access_token

    def get_client_credentials_token(self, force: bool = False) -> str:
        """Mint (and cache) an integration-level token via client_credentials.

        This authenticates as the integration rather than an impersonated user,
        which is required for settings/template endpoints whose results are
        filtered by per-user visibility.
        """
        if self._client_credentials_token and not force:
            return self._client_credentials_token

        token_url = f"{self.api_base_url}/oauth2/v1/token"
        r = requests.post(token_url, data={
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": self.scope,
            "instance_id": self.instance_id,
        }, timeout=15)
        if r.status_code != 200:
            # Surface the OAuth error body (e.g. {"error":"unauthorized_client"})
            # so callers can report *why* the grant was rejected, not just "400".
            raise RuntimeError(
                f"client_credentials HTTP {r.status_code} at {token_url}: "
                f"{(r.text or '')[:400]}"
            )
        self._client_credentials_token = r.json()["access_token"]
        logger.info("[ENCOMPASS] Obtained client_credentials token")
        return self._client_credentials_token

    def get_actor_token(self) -> Optional[str]:
        """Return the super-admin (password-grant) token, authenticating if needed.

        Returns None if authentication fell back to client_credentials (i.e. the
        password grant failed and no admin token is available).
        """
        if not self.actor_token:
            self._ensure_token()
        return self.actor_token

    def _headers(self, content_type: str = "application/json") -> dict:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": content_type,
            "Accept": "application/json",
        }

    def _ensure_token(self):
        if not self.access_token:
            self.authenticate()

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make a request, retry once on 401 with a fresh token."""
        self._ensure_token()
        kwargs.setdefault("headers", self._headers())
        kwargs.setdefault("timeout", 30)

        r = requests.request(method, url, **kwargs)
        if r.status_code == 401:
            logger.info("[ENCOMPASS] 401 — refreshing token and retrying")
            self.authenticate()
            kwargs["headers"] = self._headers()
            r = requests.request(method, url, **kwargs)
        return r

    # ── Search ──

    def search_loans(self, loan_number: str) -> List[Dict[str, Any]]:
        """Search by loan number via v3 loanPipeline."""
        url = f"{self.api_base_url}/encompass/v3/loanPipeline"
        payload = {
            "fields": ["Loan.Guid", "Loan.LoanNumber", "Loan.BorrowerName"],
            "filter": {
                "canonicalName": "Loan.LoanNumber",
                "value": loan_number,
                "matchType": "exact",
            },
            "includeArchivedLoans": True,
            "loanOwnership": "AllLoans",
        }
        r = self._request("POST", url, json=payload)
        r.raise_for_status()
        data = r.json()
        rows = data if isinstance(data, list) else data.get("rows", [])
        results = []
        for row in rows:
            fields = row.get("fields", {})
            results.append({
                "loanGuid": row.get("loanId"),
                "loanNumber": fields.get("Loan.LoanNumber"),
                "borrowerName": fields.get("Loan.BorrowerName"),
            })
        return results

    # ── Field I/O ──

    def read_fields(self, loan_id: str, field_ids: List[str]) -> Dict[str, Any]:
        """Read fields via POST /fieldReader."""
        url = f"{self.api_base_url}/encompass/v3/loans/{loan_id}/fieldReader"
        r = self._request("POST", url, json=field_ids)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            return data
        # fieldReader returns list of {fieldId, value} or flat list
        result = {}
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    result[entry.get("fieldId", "")] = entry.get("value")
                elif isinstance(entry, str) and len(field_ids) == 1:
                    result[field_ids[0]] = entry
        return result

    def get_application_id(self, loan_id: str) -> str:
        """Get the first application ID from the loan entity."""
        url = f"{self.api_base_url}/encompass/v3/loans/{loan_id}?view=entity"
        r = self._request("GET", url)
        r.raise_for_status()
        apps = r.json().get("applications", [])
        return apps[0].get("id", "") if apps else ""

    # ── Other Income (nested V3 collection — NOT a classic field, so it
    # can't go through read_fields/write_fields above) ──

    def get_other_income_sources(self, loan_id: str, application_id: str) -> List[Dict[str, Any]]:
        """Read the Other Income Sources collection for an application.

        GET /encompass/v3/loans/{loanId}/applications/{applicationId}/otherIncomeSources
        Confirmed live (2026-10-16) — returns a plain JSON array (possibly empty).
        """
        url = f"{self.api_base_url}/encompass/v3/loans/{loan_id}/applications/{application_id}/otherIncomeSources"
        r = self._request("GET", url)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []

    def manage_other_income_sources(
        self, loan_id: str, application_id: str, action: str, items: List[Dict[str, Any]],
    ) -> bool:
        """Add/update/delete entries in the Other Income Sources collection.

        PATCH /encompass/v3/loans/{loanId}/applications/{applicationId}/otherIncomeSources?action={action}
        (official ICE/Encompass "V3 Manage Other Incomes" API — body is a plain
        JSON array, NOT wrapped in an object).

        action="add": items must NOT include "id" (Encompass assigns one) or
            "sourceOfIncomeData" (system-managed — rejected if present).
        action="update"/"delete": items must include the existing "id".

        Confirmed via a live round-trip test against a dummy Encompass loan
        (2026-10-16): add/update/delete all return 204 with this exact shape.
        """
        if action not in ("add", "update", "reorder", "delete"):
            raise ValueError(f"Invalid action: {action!r}")
        url = f"{self.api_base_url}/encompass/v3/loans/{loan_id}/applications/{application_id}/otherIncomeSources"
        r = self._request("PATCH", url, params={"action": action}, json=items)
        if r.status_code not in (200, 204):
            logger.error(f"[ENCOMPASS] manage_other_income_sources({action}) error {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
        return True

    def write_fields(self, loan_id: str, updates: Dict[str, Any]) -> bool:
        """Write fields via PATCH.

        Uses the v3 PATCH endpoint. For fields under applications
        (like borrower info), the application ID is fetched automatically.
        For top-level / custom fields, they go directly on the payload.
        """
        if not updates:
            return True

        app_id = None
        app_fields = {}
        top_fields = {}

        # Separate fields that need application context from top-level fields
        for fid, val in updates.items():
            if fid.startswith(("CX.", "NEWHUD.", "VEND.")):
                top_fields[fid] = val
            elif fid.isdigit():
                # Numeric field IDs need the fieldWriter endpoint
                top_fields[fid] = val
            else:
                top_fields[fid] = val

        # For numeric field IDs, use fieldWriter which translates field IDs
        if top_fields:
            url = f"{self.api_base_url}/encompass/v3/loans/{loan_id}/fieldWriter"
            # The v3 fieldWriter endpoint expects each entry keyed by "id"
            # (the field id), not "fieldId".
            payload = [{"id": fid, "value": str(val)} for fid, val in top_fields.items()]
            r = self._request("POST", url, json=payload)
            if r.status_code not in (200, 204):
                logger.error(f"[ENCOMPASS] fieldWriter error {r.status_code}: {r.text[:300]}")
                r.raise_for_status()

        logger.info(f"[ENCOMPASS] Wrote {len(updates)} fields to loan {loan_id[:12]}")
        return True


# ── Factory ──

def get_encompass_client(env: str = None, state: dict = None, use_cache: bool = True) -> EncompassClient:
    """Get an initialized EncompassClient.

    Environment priority: explicit env > state["env"] > cached > "TEST".
    """
    global _workflow_env

    env_mode = None
    if env:
        env_mode = env.upper()
    elif state and state.get("env"):
        env_mode = state["env"].upper()
    elif _workflow_env:
        env_mode = _workflow_env
    else:
        env_mode = "TEST"

    if env_mode not in ("TEST", "PROD"):
        env_mode = "TEST"

    if _workflow_env is None:
        _workflow_env = env_mode
        logger.info(f"[ENCOMPASS] Workflow env: {env_mode}")

    env_prefix = f"{env_mode}_"
    instance_id = _get_env_var("ENCOMPASS_INSTANCE_ID", env_prefix) or ""
    cache_key = f"{env_mode}_{instance_id}" if instance_id else env_mode

    if use_cache and cache_key in _MODULE_CLIENT_CACHE:
        return _MODULE_CLIENT_CACHE[cache_key]

    client = EncompassClient(
        api_base_url=_get_env_var("ENCOMPASS_API_BASE_URL", env_prefix) or "https://api.elliemae.com",
        client_id=_get_env_var("ENCOMPASS_CLIENT_ID", env_prefix) or "",
        client_secret=_get_env_var("ENCOMPASS_CLIENT_SECRET", env_prefix) or "",
        instance_id=instance_id,
        username=_get_env_var("ENCOMPASS_USERNAME", env_prefix) or "",
        password=_get_env_var("ENCOMPASS_PASSWORD", env_prefix) or "",
        subject_user_id=_get_env_var("ENCOMPASS_SUBJECT_USER_ID", env_prefix) or "",
        scope=_get_env_var("ENCOMPASS_SCOPE", env_prefix) or "lp",
    )

    if use_cache:
        _MODULE_CLIENT_CACHE[cache_key] = client

    return client


class FieldIds:
    """Common Encompass field IDs."""
    BORROWER_FIRST_NAME = "4000"
    BORROWER_LAST_NAME = "4002"
    BORROWER_EMAIL = "1240"
    LOAN_NUMBER = "364"
    LOAN_TYPE = "1172"
    LOAN_PURPOSE = "19"
    LOAN_AMOUNT = "1109"
    PROPERTY_STATE = "14"
    PROPERTY_TYPE = "1041"
    PROPERTY_ADDRESS = "11"
    APPRAISED_VALUE = "356"
    PURCHASE_PRICE = "136"
    LTV = "353"
    CLTV = "976"
    INTEREST_RATE = "3"
    LOAN_TERM = "4"
    APPLICATION_DATE = "745"
    LOAN_STATUS = "1393"  # Loan Status (was FIELDS.2025 — not a valid field id)
    LO_NAME = "LoanTeamMember.Name.Loan Officer"
    PROCESSOR_NAME = "362"
