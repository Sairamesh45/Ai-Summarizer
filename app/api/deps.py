import logging
import uuid
from typing import Annotated

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError

log = logging.getLogger(__name__)
from sqlalchemy.ext.asyncio import AsyncSession
from supabase import AsyncClient

from app.core.exceptions import UnauthorizedException
from app.core.roles import ROLE_DOCTOR, require_role
from app.core.security import decode_token
from app.database import get_db
from app.models.user import User, UserRoleEnum
from app.services.auth_service import AuthService
from app.services.document_service import DocumentService
from app.services.patient_service import PatientService
from app.services.storage_service import StorageService
from app.supabase_client import get_supabase_client

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


# ── Database ──────────────────────────────────────────────────────────────────
DBDep = Annotated[AsyncSession, Depends(get_db)]

# ── Supabase ──────────────────────────────────────────────────────────────────
SupabaseDep = Annotated[AsyncClient, Depends(get_supabase_client)]


def _role_from_claims(payload: dict) -> UserRoleEnum:
    """Extract role from JWT claims. Supports both `role` (string) and `roles` (array)."""
    # Try `roles` array first (Clinic-Backend format)
    roles_claim = payload.get("roles")
    if isinstance(roles_claim, (list, tuple)) and roles_claim:
        raw = str(roles_claim[0]).lower()
    elif isinstance(roles_claim, str):
        raw = roles_claim.lower()
    else:
        # Fall back to `role` string (Ai-Summarizer native format)
        raw = str(payload.get("role") or "").lower()

    try:
        return UserRoleEnum(raw)
    except ValueError:
        return UserRoleEnum.staff


# ── Current user ──────────────────────────────────────────────────────────────
async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    db: DBDep,
) -> User:
    if not token:
        raise UnauthorizedException("Missing token")

    # Decode and verify the JWT — raises 401 on bad signature / expiry
    try:
        payload = decode_token(token)
    except (JWTError, ValueError, KeyError) as exc:
        log.warning("Token decode failed (%s): %s", type(exc).__name__, exc)
        raise UnauthorizedException()

    sub = str(payload.get("sub", ""))

    # ── Path A: sub is a UUID → local Ai-Summarizer or Supabase Auth account ──
    try:
        user_id = uuid.UUID(sub)
    except (ValueError, AttributeError):
        user_id = None

    if user_id is not None:
        service = AuthService(db)
        try:
            return await service.get_user_or_404(user_id)
        except Exception:
            # User UUID is valid but not in local users table yet.
            # Auto-provision a local record so FK constraints are satisfied.
            role_enum = _role_from_claims(payload)
            email = str(payload.get("email") or f"auth_{sub}@clinic.local")
            full_name = str(payload.get("name") or payload.get("full_name") or "")
            try:
                user = User(
                    id=user_id,
                    email=email,
                    hashed_password="",  # Supabase Auth manages passwords
                    full_name=full_name or None,
                    is_active=True,
                    role=role_enum,
                )
                db.add(user)
                await db.flush()
                await db.refresh(user)
                log.info("Auto-provisioned local user %s from JWT", user_id)
                return user
            except Exception as provision_err:
                log.warning(
                    "Could not auto-provision user %s: %s", user_id, provision_err
                )
                await db.rollback()
                # Return a transient user so the request can still proceed
                return User(
                    id=user_id,
                    email=email,
                    hashed_password="",
                    full_name=full_name or None,
                    is_active=True,
                    role=role_enum,
                )

    # ── Path B: external token (Clinic-Backend numeric sub) ──────────────────
    # The JWT signature already proves authenticity — no extra HTTP call needed.
    # Build a deterministic UUID from the sub so the same external user always
    # gets the same local UUID (avoids creating duplicate placeholder users).
    role_enum = _role_from_claims(payload)
    deterministic_id = uuid.uuid5(uuid.NAMESPACE_URL, f"clinic-backend:user:{sub}")

    # Try to find or create a persistent local user for this external identity
    service = AuthService(db)
    try:
        existing = await service.get_user_or_404(deterministic_id)
        return existing
    except Exception:
        pass

    try:
        user = User(
            id=deterministic_id,
            email=f"external_{sub}@clinic.local",
            hashed_password="",
            full_name=None,
            is_active=True,
            role=role_enum,
        )
        db.add(user)
        await db.flush()
        await db.refresh(user)
        log.info("Auto-provisioned external user sub=%s → id=%s", sub, deterministic_id)
        return user
    except Exception as exc:
        log.warning("Could not persist external user sub=%s: %s", sub, exc)
        await db.rollback()
        return User(
            id=deterministic_id,
            email=f"external_{sub}@clinic.local",
            hashed_password="",
            full_name=None,
            is_active=True,
            role=role_enum,
        )


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    if not current_user.is_active:
        raise UnauthorizedException("Inactive user")
    return current_user


async def get_current_superuser(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> User:
    from app.core.exceptions import ForbiddenException

    if not current_user.is_superuser:
        raise ForbiddenException("Superuser required")
    return current_user


# ── Service factories ─────────────────────────────────────────────────────────
def get_auth_service(db: DBDep) -> AuthService:
    return AuthService(db)


def get_storage_service(client: SupabaseDep) -> StorageService:
    return StorageService(client)


def get_document_service(db: DBDep) -> DocumentService:
    return DocumentService(db)


def get_patient_service(db: DBDep) -> PatientService:
    return PatientService(db)


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
StorageServiceDep = Annotated[StorageService, Depends(get_storage_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
PatientServiceDep = Annotated[PatientService, Depends(get_patient_service)]
CurrentUser = Annotated[User, Depends(get_current_active_user)]

# ── Role-scoped user dependencies ─────────────────────────────────────────────
# These resolve the JWT role claim WITHOUT an extra DB query, then load User.
# Add more roles by following the same pattern.
DoctorTokenPayload = Annotated[dict, Depends(require_role(ROLE_DOCTOR))]
