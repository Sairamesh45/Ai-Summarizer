"""
Manual test of the full OCR + LLM extraction pipeline.
Run: python scripts/test_pipeline.py
"""

import asyncio
import logging
import pathlib
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("test_pipeline")


DOC_ID = uuid.UUID("50d8f962-6d0e-4d5b-822b-67ce44da02f9")
PATIENT_ID = uuid.UUID("68e2dfdf-eaa1-4e04-8f53-63ac5f6f6395")


async def main():
    from app.database import AsyncSessionLocal
    from sqlalchemy import text

    # ── Reset doc to pending ──────────────────────────────────────────────
    async with AsyncSessionLocal() as db:
        await db.execute(
            text("UPDATE documents SET status='pending' WHERE id=:did"),
            {"did": DOC_ID},
        )
        await db.commit()
        log.info("Reset doc %s to pending", DOC_ID)

    # ── Find the uploaded PDF ─────────────────────────────────────────────
    local_files = list(pathlib.Path("tmp/uploads").rglob("*.pdf"))
    log.info("Local PDF files found: %s", local_files)
    if not local_files:
        log.error("No local PDF files found in tmp/uploads!")
        return

    content = local_files[0].read_bytes()
    log.info("Using file: %s (%d bytes)", local_files[0], len(content))

    # ── Import services ──────────────────────────────────────────────────
    from app.services.document_service import DocumentService
    from app.services.patient_service import PatientService
    from app.services.ocr_service import OcrService
    from app.services.llm_service import LlmService

    async with AsyncSessionLocal() as db:
        doc_svc = DocumentService(db)
        pat_svc = PatientService(db)

        # Step 1: Mark processing
        try:
            await doc_svc.mark_processing(DOC_ID)
            await db.commit()
            log.info("Step 1 OK: mark_processing")
        except Exception:
            log.exception("Step 1 FAILED: mark_processing")
            return

        # Step 2: OCR
        try:
            ocr_svc = OcrService(lang="eng")
            ocr_result = await ocr_svc.extract_from_bytes(
                content, content_type="application/pdf"
            )
            raw_text = ocr_result.full_text
            await doc_svc.save_ocr_result(
                DOC_ID,
                raw_text=raw_text,
                page_count=ocr_result.page_count,
                lang="eng",
                dpi=300,
                total_duration_ms=ocr_result.total_duration_ms,
            )
            await db.commit()
            log.info(
                "Step 2 OK: OCR done — %d chars, %d pages",
                len(raw_text),
                ocr_result.page_count,
            )
            log.info("OCR text (first 500 chars):\n%s", raw_text[:500])
        except Exception:
            log.exception("Step 2 FAILED: OCR")
            return

        # Step 3: LLM extraction
        try:
            async with LlmService() as llm:
                extract_result = await llm.extract(raw_text, model="llama3")
            extraction_dict = (
                extract_result.data.model_dump()
                if hasattr(extract_result.data, "model_dump")
                else dict(extract_result.data)
            )
            log.info("Step 3 OK: LLM extraction — %s", extraction_dict)
        except Exception:
            log.exception("Step 3 FAILED: LLM extraction")
            return

        # Step 4: Save events
        try:
            n = await pat_svc.save_extracted_events_from_llm(
                PATIENT_ID, DOC_ID, extraction_dict, ai_model="llama3"
            )
            await db.commit()
            log.info("Step 4 OK: Saved %d events", n)
        except Exception:
            log.exception("Step 4 FAILED: save_events")

    # ── Verify ────────────────────────────────────────────────────────────
    async with AsyncSessionLocal() as db:
        r = await db.execute(
            text(
                "SELECT event_type, event_data FROM extracted_events WHERE patient_id=:pid"
            ),
            {"pid": PATIENT_ID},
        )
        rows = r.fetchall()
        log.info("=== EVENTS IN DB (%d) ===", len(rows))
        for row in rows:
            log.info("  type=%s data=%s", row[0], row[1])


if __name__ == "__main__":
    asyncio.run(main())
