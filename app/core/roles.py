"""
Role-based access control (RBAC) for FastAPI endpoints.

Role is embedded directly inside the JWT access token (`role` claim).
No database hit is required for role verification — the check is
stateless and handled entirely from the decoded token payload.
"""

import uuid
from typing import Annotated

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError

from app.core.exceptions import ForbiddenException, UnauthorizedException
from app.core.security import decode_token
from app.models.user import UserRoleEnum

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# ── Exported role constants ───────────────────────────────────────────────────
ROLE_DOCTOR = UserRoleEnum.doctor.value
ROLE_ADMIN = UserRoleEnum.admin.value
ROLE_STAFF = UserRoleEnum.staff.value


def extract_token_claims(token: str) -> dict:
    """Decode a JWT and return its payload. Raises UnauthorizedException on failure."""
    try:
        payload = decode_token(token)
        # Accept tokens from other services that may not include a 'type' claim
        return payload
    except (JWTError, ValueError):
        raise UnauthorizedException("Could not validate credentials")


def require_role(*allowed_roles: str):
    """
    Dependency factory — returns a FastAPI dependency that:

    1. Extracts the JWT Bearer token
    2. Decodes and validates it (signature + expiry)
    3. Reads the `role` claim
    4. Raises 403 if the role is not in *allowed_roles*
    5. Returns the parsed token payload on success

    Usage:
        DoctorDep = Depends(require_role("doctor"))
        AdminOrDoctorDep = Depends(require_role("doctor", "admin"))
    """

    async def _guard(
        token: Annotated[str, Depends(oauth2_scheme)],
    ) -> dict:
        payload = extract_token_claims(token)
        # Support both single-string `role` and array `roles` claim formats
        role_single = payload.get("role")
        roles_array = payload.get("roles") or []
        effective_roles = set()
        if role_single:
            effective_roles.add(str(role_single))
        if isinstance(roles_array, (list, tuple)):
            for r in roles_array:
                effective_roles.add(str(r))

        if not any(r in effective_roles for r in allowed_roles):
            raise ForbiddenException(
                f"Access denied — required role(s): {', '.join(allowed_roles)}. Your roles: {', '.join(sorted(effective_roles) or ['none'])}"
            )
        return payload

    return _guard


def get_user_id_from_payload(payload: dict) -> uuid.UUID:
    """Extract and parse the `sub` claim as a UUID."""
    try:
        return uuid.UUID(payload["sub"])
    except (KeyError, ValueError):
        raise UnauthorizedException("Malformed token subject")
