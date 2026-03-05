import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import BadRequestException, NotFoundException
from app.models.medical import Document, DocumentStatusEnum, DocumentTypeEnum, Patient
from app.schemas.medical import DocumentCreate, DocumentOut, DocumentUpdate

log = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ── Helpers ───────────────────────────────────────────────────────────────
    async def _patient_exists(self, patient_id: uuid.UUID) -> bool:
        result = await self.db.execute(
            select(Patient.id).where(Patient.id == patient_id)
        )
        return result.scalar_one_or_none() is not None

    async def _get_document(self, document_id: uuid.UUID) -> Document:
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        doc = result.scalar_one_or_none()
        if not doc:
            raise NotFoundException("Document not found")
        return doc

    # ── Create ────────────────────────────────────────────────────────────────
    async def create(
        self,
        *,
        patient_id: uuid.UUID,
        uploaded_by: uuid.UUID,
        title: str,
        storage_path: str,
        file_size_bytes: int,
        mime_type: str,
        document_type: DocumentTypeEnum = DocumentTypeEnum.other,
        extra_data: dict[str, Any] | None = None,
    ) -> Document:
        if not await self._patient_exists(patient_id):
            raise NotFoundException("Patient not found")

        doc = Document(
            patient_id=patient_id,
            uploaded_by=uploaded_by,
            title=title,
            document_type=document_type,
            status=DocumentStatusEnum.pending,
            storage_path=storage_path,
            file_size_bytes=file_size_bytes,
            mime_type=mime_type,
            extra_data=extra_data or {},
        )
        self.db.add(doc)
        await self.db.flush()
        await self.db.refresh(doc)
        return doc

    # ── Read ──────────────────────────────────────────────────────────────────
    async def get(self, document_id: uuid.UUID) -> Document:
        return await self._get_document(document_id)

    async def list_for_patient(
        self, patient_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> list[Document]:
        result = await self.db.execute(
            select(Document)
            .where(Document.patient_id == patient_id)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def has_pending_documents(self, patient_id: uuid.UUID) -> bool:
        """Return True if the patient has any documents still pending or processing."""
        result = await self.db.execute(
            select(func.count())
            .select_from(Document)
            .where(
                Document.patient_id == patient_id,
                Document.status.in_(
                    [
                        DocumentStatusEnum.pending,
                        DocumentStatusEnum.processing,
                    ]
                ),
            )
        )
        return (result.scalar_one() or 0) > 0

    # ── Update ────────────────────────────────────────────────────────────────
    async def update(self, document_id: uuid.UUID, data: DocumentUpdate) -> Document:
        doc = await self._get_document(document_id)
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(doc, field, value)
        await self.db.flush()
        await self.db.refresh(doc)
        return doc

    # ── Delete ────────────────────────────────────────────────────────────────
    async def delete(self, document_id: uuid.UUID) -> str:
        """Deletes the DB record and returns the storage_path for caller to clean up."""
        doc = await self._get_document(document_id)
        storage_path = doc.storage_path
        await self.db.delete(doc)
        await self.db.flush()
        return storage_path

    # ── OCR state-machine ────────────────────────────────────────────────────

    async def mark_processing(self, document_id: uuid.UUID) -> Document:
        """
        Transition document to ``processing``.

        Guards against re-entrant calls: raises BadRequestException if the
        document is already being processed or has already been completed.
        A failed document can be retried (it transitions back to processing).
        """
        doc = await self._get_document(document_id)

        if doc.status == DocumentStatusEnum.processing:
            raise BadRequestException(
                "Document is already being processed. "
                "Wait for it to complete or fail before retrying."
            )
        if doc.status == DocumentStatusEnum.completed:
            raise BadRequestException(
                "Document has already been processed successfully. "
                "Re-processing is not allowed to protect stored data."
            )

        doc.status = DocumentStatusEnum.processing
        # Clear any residual failure info from a previous attempt
        doc.extra_data = {
            k: v
            for k, v in (doc.extra_data or {}).items()
            if k not in ("ocr_error", "ocr_failed_at")
        }
        await self.db.flush()
        await self.db.refresh(doc)
        log.info("Document %s → processing", document_id)
        return doc

    async def save_ocr_result(
        self,
        document_id: uuid.UUID,
        *,
        raw_text: str,
        page_count: int,
        lang: str,
        dpi: int,
        total_duration_ms: float,
        mark_completed: bool = True,
    ) -> Document:
        """
        Persist successful OCR output and optionally transition document to
        ``completed``.

        When ``mark_completed=False`` the document stays in its current status
        (typically ``processing``) so callers that have further pipeline steps
        (e.g. LLM extraction) can keep the document in-progress until
        everything finishes.

        Writes:
          - ``raw_text``        — full extracted text (form-feed separated pages)
          - ``page_count``      — number of pages processed
          - ``processed_at``    — UTC timestamp of completion
          - ``status``          — completed (unless mark_completed is False)
          - ``extra_data``      — updated with ocr_lang, ocr_dpi, ocr_duration_ms
        """
        doc = await self._get_document(document_id)

        doc.raw_text = raw_text
        doc.page_count = page_count
        if mark_completed:
            doc.status = DocumentStatusEnum.completed
        doc.processed_at = datetime.now(timezone.utc)
        doc.extra_data = {
            **(doc.extra_data or {}),
            "ocr_lang": lang,
            "ocr_dpi": dpi,
            "ocr_duration_ms": round(total_duration_ms, 2),
        }

        await self.db.flush()
        await self.db.refresh(doc)
        log.info(
            "Document %s → completed | pages=%d chars=%d ms=%.0f",
            document_id,
            page_count,
            len(raw_text),
            total_duration_ms,
        )
        return doc

    async def mark_completed(self, document_id: uuid.UUID) -> Document:
        """Transition document to ``completed``."""
        doc = await self._get_document(document_id)
        doc.status = DocumentStatusEnum.completed
        if not doc.processed_at:
            doc.processed_at = datetime.now(timezone.utc)
        await self.db.flush()
        await self.db.refresh(doc)
        log.info("Document %s → completed", document_id)
        return doc

    async def mark_failed(
        self,
        document_id: uuid.UUID,
        *,
        error_message: str,
    ) -> Document:
        """
        Transition document to ``failed`` and record the error for debugging.

        The document can be retried by calling :meth:`mark_processing` again,
        which will clear the failure metadata before re-attempting.
        """
        try:
            doc = await self._get_document(document_id)
        except NotFoundException:
            # Document may have been deleted between OCR start and failure — no-op
            log.warning("mark_failed: document %s not found — skipping", document_id)
            return  # type: ignore[return-value]

        doc.status = DocumentStatusEnum.failed
        doc.extra_data = {
            **(doc.extra_data or {}),
            "ocr_error": error_message,
            "ocr_failed_at": datetime.now(timezone.utc).isoformat(),
        }

        await self.db.flush()
        await self.db.refresh(doc)
        log.error("Document %s → failed | %s", document_id, error_message)
        return doc
