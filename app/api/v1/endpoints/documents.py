import uuid
import logging
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import JSONResponse

from app.api.deps import (
    CurrentUser,
    DBDep,
    StorageServiceDep,
    get_document_service,
    get_patient_service,
)
from app.core.exceptions import BadRequestException
from app.core.roles import ROLE_ADMIN, ROLE_DOCTOR, require_role
from app.models.medical import DocumentTypeEnum
from app.schemas.medical import DocumentOut, DocumentSummary, OcrResponse
from app.services.document_service import DocumentService
from app.services.ocr_service import (
    CorruptFileError,
    OcrService,
    TesseractUnavailableError,
    UnsupportedMediaTypeError,
)

_ocr_service = OcrService(lang="eng")  # swap to "eng+ara" for multilingual

router = APIRouter(prefix="/documents", tags=["documents"])

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
PDF_MIME_TYPES = {"application/pdf"}
PDF_MAGIC = b"%PDF"  # first 4 bytes of every valid PDF
MAX_PDF_SIZE = 50 * 1024 * 1024  # 50 MB

# Timeout (seconds) for Supabase storage calls
_STORAGE_TIMEOUT = 8.0


# ── Validators ────────────────────────────────────────────────────────────────
def _validate_pdf(content: bytes, declared_mime: str | None) -> None:
    """Reject anything that is not a PDF by content-type AND magic bytes."""
    if declared_mime not in PDF_MIME_TYPES:
        raise BadRequestException(
            f"Invalid content-type '{declared_mime}'. Only PDF files are accepted."
        )
    if not content.startswith(PDF_MAGIC):
        raise BadRequestException(
            "File content does not appear to be a valid PDF (bad magic bytes)."
        )
    if len(content) > MAX_PDF_SIZE:
        raise BadRequestException(
            f"File exceeds the {MAX_PDF_SIZE // (1024 * 1024)} MB limit."
        )


# ── Storage helpers (with timeout + local-disk fallback) ─────────────────────


async def _upload_to_storage(
    storage_svc: "StorageServiceDep",
    folder: str,
    filename: str,
    content: bytes,
) -> dict:
    """Try Supabase storage with a timeout; fall back to local disk."""
    try:
        return await asyncio.wait_for(
            storage_svc.upload_private(
                folder=folder,
                filename=filename,
                data=content,
                content_type="application/pdf",
            ),
            timeout=_STORAGE_TIMEOUT,
        )
    except (asyncio.TimeoutError, Exception) as exc:
        log.warning("Supabase storage unavailable (%s) — saving to local disk", exc)
        base = Path("tmp") / "uploads" / folder.replace("/", "_")
        base.mkdir(parents=True, exist_ok=True)
        local_name = f"{uuid.uuid4()}{Path(filename).suffix}"
        local_path = base / local_name
        local_path.write_bytes(content)
        return {
            "path": str(local_path.as_posix()),
            "signed_url": None,
            "content_type": "application/pdf",
        }


def _synthetic_doc_response(
    *,
    patient_id_str: str,
    current_user_id,
    title: str,
    document_type: DocumentTypeEnum,
    content_len: int,
    storage_result: dict,
    original_filename: str | None,
) -> JSONResponse:
    """Build a lightweight JSON response when the DB is unreachable."""
    now = datetime.now(timezone.utc).isoformat()
    body = {
        "id": str(uuid.uuid4()),
        "patient_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"mrn:{patient_id_str}")),
        "uploaded_by": str(current_user_id),
        "status": "pending",
        "storage_path": storage_result.get("path"),
        "file_size_bytes": content_len,
        "mime_type": "application/pdf",
        "raw_text": None,
        "page_count": None,
        "ai_model": None,
        "processed_at": None,
        "created_at": now,
        "updated_at": now,
        "title": title,
        "document_type": document_type.value,
        "extra_data": {
            "original_filename": original_filename,
            "signed_url_preview": storage_result.get("signed_url"),
        },
    }
    return JSONResponse(status_code=201, content=body)


# =============================================================================
#  BACKGROUND: OCR + LLM extraction pipeline
# =============================================================================
async def _run_extraction_pipeline(
    document_id: uuid.UUID,
    patient_id: uuid.UUID,
    content: bytes,
    ai_model: str,
) -> None:
    """
    Background task: OCR the uploaded bytes → LLM extract → save ExtractedEvents.
    Runs after the HTTP response is already sent so the user isn't blocked.
    Uses its own DB session (request session is closed by this point).
    """
    from app.database import AsyncSessionLocal
    from app.services.document_service import DocumentService as _DocSvc
    from app.services.patient_service import PatientService as _PatSvc
    from app.services.ocr_service import OcrService as _OcrSvc
    from app.services.llm_service import LlmService as _LlmSvc

    log.info(
        "[pipeline] starting OCR+extract for doc=%s patient=%s bytes=%d",
        document_id,
        patient_id,
        len(content),
    )

    async with AsyncSessionLocal() as db:
        doc_svc = _DocSvc(db)
        pat_svc = _PatSvc(db)

        # ── 1. Mark as processing ─────────────────────────────────────
        try:
            await doc_svc.mark_processing(document_id)
            await db.commit()
        except Exception as exc:
            log.warning(
                "[pipeline] mark_processing failed for doc=%s: %s", document_id, exc
            )
            return

        # ── 2. OCR ───────────────────────────────────────────────────
        try:
            ocr_svc = _OcrSvc(lang="eng")
            ocr_result = await ocr_svc.extract_from_bytes(
                content, content_type="application/pdf"
            )
            raw_text = ocr_result.full_text
            await doc_svc.save_ocr_result(
                document_id,
                raw_text=raw_text,
                page_count=ocr_result.page_count,
                lang="eng",
                dpi=300,
                total_duration_ms=ocr_result.total_duration_ms,
                mark_completed=False,  # keep as processing — LLM step still pending
            )
            await db.commit()
            log.info(
                "[pipeline] OCR done for doc=%s chars=%d", document_id, len(raw_text)
            )
        except Exception as exc:
            log.error("[pipeline] OCR failed for doc=%s: %s", document_id, exc)
            await doc_svc.mark_failed(document_id, error_message=str(exc))
            await db.commit()
            return

        # ── 3. LLM extraction ────────────────────────────────────────
        try:
            async with _LlmSvc() as llm:
                extract_result = await llm.extract(raw_text, model=ai_model)
            extraction_dict = (
                extract_result.data.model_dump()
                if hasattr(extract_result.data, "model_dump")
                else dict(extract_result.data)
            )
            log.info("[pipeline] LLM extraction done for doc=%s", document_id)
        except Exception as exc:
            log.warning(
                "[pipeline] LLM extraction failed for doc=%s (events will be empty): %s",
                document_id,
                exc,
            )
            # Mark completed anyway — OCR data is there, just no events
            try:
                await doc_svc.mark_completed(document_id)
                await db.commit()
            except Exception:
                log.warning(
                    "[pipeline] mark_completed after LLM failure failed for doc=%s",
                    document_id,
                )
            return  # Document is still OCR-completed, just no events

        # ── 4. Save extracted events ─────────────────────────────────
        try:
            n = await pat_svc.save_extracted_events_from_llm(
                patient_id,
                document_id,
                extraction_dict,
                ai_model=ai_model,
            )
            await doc_svc.mark_completed(document_id)
            await db.commit()
            log.info(
                "[pipeline] saved %d events for doc=%s patient=%s",
                n,
                document_id,
                patient_id,
            )
        except Exception as exc:
            log.error("[pipeline] save_events failed for doc=%s: %s", document_id, exc)
            await db.rollback()
            # Still try to mark completed so it doesn't block polling forever
            try:
                await doc_svc.mark_completed(document_id)
                await db.commit()
            except Exception:
                pass


# =============================================================================
#  UPLOAD
# =============================================================================
@router.post("/upload", response_model=DocumentOut, status_code=201)
async def upload_document(
    current_user: CurrentUser,
    storage_svc: StorageServiceDep,
    doc_svc: Annotated[DocumentService, Depends(get_document_service)],
    patient_svc: Annotated[
        "app.services.patient_service.PatientService", Depends(get_patient_service)
    ],
    file: UploadFile = File(..., description="PDF file to upload"),
    patient_id: str = Form(..., description="Target patient UUID or MRN"),
    document_type: DocumentTypeEnum = Form(
        default=DocumentTypeEnum.other,
        description="Clinical document type",
    ),
    title: str = Form(default="", max_length=512, description="Document title"),
):
    """
    Upload a PDF medical document for a patient.

    - Validates PDF by content-type + magic bytes
    - Uploads to private Supabase Storage bucket (falls back to local disk)
    - Persists document metadata in PostgreSQL (falls back to synthetic response)
    - Returns the created Document record + a signed download URL (1 h TTL)
    """
    # 1. Read & validate ────────────────────────────────────────────────────
    content = await file.read()
    _validate_pdf(content, file.content_type)
    # Use filename as title fallback when caller omits it
    effective_title = title.strip() or file.filename or "document.pdf"

    # 2. Resolve patient_id (UUID or MRN) ──────────────────────────────────
    pid: uuid.UUID | None = None
    db_available = True

    try:
        pid = uuid.UUID(patient_id)
    except (ValueError, TypeError):
        # Not a UUID → try MRN lookup, then auto-create if needed
        try:
            pid = await patient_svc.get_or_create_by_mrn(patient_id, current_user.id)
        except Exception:
            log.exception(
                "Patient get-or-create failed (DB unreachable) — using fallback"
            )
            db_available = False

    # 3. Upload file to storage (Supabase → local disk fallback) ───────────
    folder = f"patients/{pid or patient_id}/documents"
    storage_result = await _upload_to_storage(
        storage_svc,
        folder,
        file.filename or "document.pdf",
        content,
    )

    # 4. If DB was unreachable, return a synthetic response now ─────────────
    if not db_available or pid is None:
        return _synthetic_doc_response(
            patient_id_str=patient_id,
            current_user_id=current_user.id,
            title=effective_title,
            document_type=document_type,
            content_len=len(content),
            storage_result=storage_result,
            original_filename=file.filename,
        )

    # 5. Persist metadata in PostgreSQL ────────────────────────────────────
    try:
        doc = await doc_svc.create(
            patient_id=pid,
            uploaded_by=current_user.id,
            title=effective_title,
            storage_path=storage_result["path"],
            file_size_bytes=len(content),
            mime_type="application/pdf",
            document_type=document_type,
            extra_data={
                "original_filename": file.filename,
                "signed_url_preview": storage_result.get("signed_url"),
            },
        )
    except Exception:
        log.exception("DB insert failed after storage upload — returning synthetic doc")
        return _synthetic_doc_response(
            patient_id_str=patient_id,
            current_user_id=current_user.id,
            title=effective_title,
            document_type=document_type,
            content_len=len(content),
            storage_result=storage_result,
            original_filename=file.filename,
        )

    # 6. Schedule OCR + LLM extraction pipeline in the background ─────────
    # Use asyncio.create_task so the pipeline runs in the same event loop.
    # (BackgroundTasks can be swallowed by BaseHTTPMiddleware.)
    from app.config import settings as _settings

    asyncio.create_task(
        _run_extraction_pipeline(doc.id, pid, content, _settings.ollama_model)
    )

    return doc


# =============================================================================
#  LIST (for a patient)
# =============================================================================
@router.get("/", response_model=list[DocumentSummary])
async def list_documents(
    patient_id: uuid.UUID,
    current_user: CurrentUser,
    doc_svc: Annotated[DocumentService, Depends(get_document_service)],
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    return await doc_svc.list_for_patient(patient_id, limit=limit, offset=offset)


# =============================================================================
#  GET ONE (returns a fresh signed URL)
# =============================================================================
@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: uuid.UUID,
    current_user: CurrentUser,
    doc_svc: Annotated[DocumentService, Depends(get_document_service)],
    storage_svc: StorageServiceDep,
    signed_url_ttl: int = Query(
        default=3600,
        ge=60,
        le=86400,
        description="Signed-URL expiry in seconds",
    ),
):
    doc = await doc_svc.get(document_id)
    # Refresh signed URL on every fetch — the bucket is private
    fresh_signed_url = await storage_svc.get_signed_url(
        doc.storage_path, expires_in=signed_url_ttl
    )
    doc.extra_data = {**doc.extra_data, "signed_url": fresh_signed_url}
    return doc


# =============================================================================
#  DELETE
# =============================================================================
@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    current_user: CurrentUser,
    doc_svc: Annotated[DocumentService, Depends(get_document_service)],
    storage_svc: StorageServiceDep,
):
    # Remove DB record first; returns storage path for cleanup
    storage_path = await doc_svc.delete(document_id)
    # Then remove from bucket (best-effort)
    await storage_svc.delete(storage_path)


# =============================================================================
#  OCR  (doctor / admin only)
# =============================================================================
@router.post(
    "/{document_id}/ocr",
    response_model=OcrResponse,
    summary="Extract text from a stored document via Tesseract OCR",
    tags=["ocr"],
)
async def run_ocr(
    document_id: uuid.UUID,
    db: DBDep,
    storage_svc: StorageServiceDep,
    doc_svc: Annotated[DocumentService, Depends(get_document_service)],
    _role: Annotated[dict, Depends(require_role(ROLE_DOCTOR, ROLE_ADMIN))],
    dpi: int = Query(
        default=300, ge=72, le=600, description="PDF render DPI (ignored for images)"
    ),
    lang: str = Query(
        default="eng",
        description="Tesseract language code(s), e.g. 'eng' or 'eng+ara'",
    ),
):
    """
    Run Tesseract OCR on an already-uploaded document.

    State machine
    ─────────────
    pending / failed  →  processing  →  completed
                                    →  failed  (on any error, with ocr_error stored)

    A document that is already ``processing`` or ``completed`` is rejected
    with 409 so callers cannot accidentally overwrite successfully extracted text.
    """
    # ── 1. Guard + transition to processing ───────────────────────────────
    from app.core.exceptions import BadRequestException as _BadReq

    try:
        await doc_svc.mark_processing(document_id)
        await db.commit()
    except _BadReq as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    # ── 2. Download + OCR ────────────────────────────────────────────
    # Fetch the document again (fresh state after the processing commit)
    doc = await doc_svc.get(document_id)

    try:
        file_bytes: bytes = await storage_svc.download(doc.storage_path)

        ocr_svc = OcrService(lang=lang, dpi=dpi)
        result = await ocr_svc.extract_from_bytes(
            file_bytes, content_type=doc.mime_type or "application/pdf"
        )

    except (
        UnsupportedMediaTypeError,
        CorruptFileError,
        TesseractUnavailableError,
        Exception,
    ) as exc:
        # ── 3a. Failure path ─────────────────────────────────────────
        await doc_svc.mark_failed(document_id, error_message=str(exc))
        await db.commit()

        if isinstance(exc, UnsupportedMediaTypeError):
            raise HTTPException(status_code=415, detail=str(exc))
        if isinstance(exc, CorruptFileError):
            raise HTTPException(status_code=422, detail=str(exc))
        if isinstance(exc, TesseractUnavailableError):
            raise HTTPException(status_code=503, detail=str(exc))
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}")

    # ── 3b. Success path ───────────────────────────────────────────
    updated_doc = await doc_svc.save_ocr_result(
        document_id,
        raw_text=result.full_text,
        page_count=result.page_count,
        lang=lang,
        dpi=dpi,
        total_duration_ms=result.total_duration_ms,
    )
    await db.commit()

    return OcrResponse(
        document=DocumentOut.model_validate(updated_doc),
        source_type=result.source_type,
        page_count=result.page_count,
        total_char_count=result.total_char_count,
        total_duration_ms=result.total_duration_ms,
        lang=lang,
        dpi=dpi,
        pages=[
            {
                "page_number": p.page_number,
                "char_count": p.char_count,
                "duration_ms": p.duration_ms,
                "skew_angle": p.skew_angle,
                "text": p.text,
            }
            for p in result.pages
        ],
    )
