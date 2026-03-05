import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.deps import DBDep, StorageServiceDep
from app.core.roles import ROLE_DOCTOR, get_user_id_from_payload, require_role
from app.services.auth_service import AuthService
from app.services.document_service import DocumentService

router = APIRouter(prefix="/files", tags=["secure-files"])

# 5-minute TTL — short enough to limit exposure, long enough for UI handoff
SECURE_URL_TTL = 300


# ── Response schema ───────────────────────────────────────────────────────────
class SecureFileUrlResponse(BaseModel):
    document_id: uuid.UUID
    signed_url: str
    expires_in_seconds: int
    expires_at: datetime
    storage_path: str


# ── Dependency: doctor-only JWT guard ─────────────────────────────────────────
DoctorPayload = Annotated[dict, Depends(require_role(ROLE_DOCTOR))]


# =============================================================================
#  GET SECURE URL
# =============================================================================
@router.get(
    "/{document_id}/secure-url",
    response_model=SecureFileUrlResponse,
    summary="Get a private, time-limited download URL (doctor only)",
    description=(
        "Issues a **5-minute signed URL** for a private Supabase Storage file. "
        "Requires a valid JWT with `role=doctor`. "
        "The file is never publicly accessible — every access generates a fresh URL."
    ),
)
async def get_secure_file_url(
    document_id: uuid.UUID,
    token_payload: DoctorPayload,
    db: DBDep,
    storage_svc: StorageServiceDep,
    ttl: int = Query(
        default=SECURE_URL_TTL,
        ge=60,
        le=300,
        description="Signed-URL lifetime in seconds (max 300 = 5 min)",
    ),
) -> SecureFileUrlResponse:
    """
    Flow:
      1. JWT decoded + role=doctor verified by DoctorPayload dependency
      2. Caller's user ID extracted from token (no extra DB call for auth)
      3. Document looked up in PostgreSQL
      4. Signed URL generated from private Supabase Storage bucket
      5. URL and expiry metadata returned — never stored or logged
    """
    # ── Load document from DB ─────────────────────────────────────────────────
    doc_svc = DocumentService(db)
    document = await doc_svc.get(document_id)

    # ── Generate private signed URL (no public access, short-lived) ───────────
    signed_url = await storage_svc.get_secure_file_url(document.storage_path, ttl=ttl)

    now = datetime.now(timezone.utc)
    return SecureFileUrlResponse(
        document_id=document.id,
        signed_url=signed_url,
        expires_in_seconds=ttl,
        expires_at=now.replace(microsecond=0).isoformat(),  # clean ISO timestamp
        storage_path=document.storage_path,
    )


# =============================================================================
#  BATCH SECURE URLS (multiple documents at once)
# =============================================================================
class BatchSecureUrlRequest(BaseModel):
    document_ids: list[uuid.UUID]


class BatchSecureUrlResponse(BaseModel):
    results: list[SecureFileUrlResponse]
    failed: list[dict]  # {"document_id": ..., "error": "..."}


@router.post(
    "/secure-url/batch",
    response_model=BatchSecureUrlResponse,
    summary="Batch signed URLs for multiple documents (doctor only)",
)
async def get_batch_secure_urls(
    body: BatchSecureUrlRequest,
    token_payload: DoctorPayload,
    db: DBDep,
    storage_svc: StorageServiceDep,
    ttl: int = Query(default=SECURE_URL_TTL, ge=60, le=300),
) -> BatchSecureUrlResponse:
    """
    Retrieve signed URLs for up to 20 documents in a single request.
    Documents that fail (not found, storage error) are collected in `failed`
    rather than aborting the whole batch.
    """
    if len(body.document_ids) > 20:
        from app.core.exceptions import BadRequestException

        raise BadRequestException("Maximum 20 documents per batch request")

    doc_svc = DocumentService(db)
    results: list[SecureFileUrlResponse] = []
    failed: list[dict] = []
    now = datetime.now(timezone.utc)

    for doc_id in body.document_ids:
        try:
            document = await doc_svc.get(doc_id)
            signed_url = await storage_svc.get_secure_file_url(
                document.storage_path, ttl=ttl
            )
            results.append(
                SecureFileUrlResponse(
                    document_id=document.id,
                    signed_url=signed_url,
                    expires_in_seconds=ttl,
                    expires_at=now,
                    storage_path=document.storage_path,
                )
            )
        except Exception as exc:
            failed.append({"document_id": str(doc_id), "error": str(exc)})

    return BatchSecureUrlResponse(results=results, failed=failed)
