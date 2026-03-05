"""
LLM Endpoints
=============

POST /api/v1/llm/generate          — send a free-form prompt to the local Ollama model
POST /api/v1/llm/summarize         — summarise medical document text (built-in clinical prompt)
POST /api/v1/llm/extract           — extract structured medical entities as JSON (with retry)
POST /api/v1/llm/physician-summary — generate a concise physician summary from structured data
GET  /api/v1/llm/health            — quick reachability check against Ollama
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import CurrentUser
from app.schemas.llm import (
    LlmExtractRequest,
    LlmExtractResponse,
    LlmGenerateRequest,
    LlmGenerateResponse,
    LlmPhysicianSummaryRequest,
    LlmPhysicianSummaryResponse,
    LlmSummarizeRequest,
    LlmSummarizeResponse,
)
from app.services.llm_service import (
    LlmConnectionError,
    LlmResponseError,
    LlmService,
    LlmTimeoutError,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["llm"])


# ---------------------------------------------------------------------------
# Shared dependency — one LlmService instance per request (lightweight: just
# spins up an httpx client; the TCP connection is kept alive via httpx pool).
# ---------------------------------------------------------------------------


async def get_llm_service() -> LlmService:  # type: ignore[return]
    """FastAPI dependency that yields a ready-to-use LlmService."""
    async with LlmService() as svc:
        yield svc


LlmServiceDep = Depends(get_llm_service)


# ---------------------------------------------------------------------------
# Error translation helper
# ---------------------------------------------------------------------------


def _handle_llm_error(exc: Exception) -> None:
    """Convert LlmService exceptions into appropriate HTTP responses."""
    if isinstance(exc, LlmConnectionError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    if isinstance(exc, LlmTimeoutError):
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    if isinstance(exc, LlmResponseError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    # Unexpected errors bubble up as 500
    log.exception("Unexpected LLM error")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An unexpected error occurred while contacting the LLM service.",
    ) from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    summary="Check Ollama connectivity",
    response_description="Returns {'status': 'ok'} when Ollama is reachable.",
)
async def llm_health(
    _: CurrentUser,
    svc: LlmService = LlmServiceDep,
) -> dict[str, str]:
    """
    Send a minimal no-op prompt to Ollama to verify it is running and the
    configured model is available.
    """
    try:
        await svc.generate("ping", options={"num_predict": 1})
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        _handle_llm_error(exc)


@router.post(
    "/generate",
    response_model=LlmGenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate text from a free-form prompt",
)
async def generate(
    body: LlmGenerateRequest,
    _: CurrentUser,
    svc: LlmService = LlmServiceDep,
) -> LlmGenerateResponse:
    """
    Send a prompt to the locally running Ollama LLaMA model.

    - **prompt** — the text to send to the model (required).
    - **model** — override the default model configured in `.env`.
    - **system** — optional system/instruction prompt.
    - **options** — Ollama model parameters (temperature, top_p, num_ctx, …).
    """
    log.info("POST /llm/generate prompt_len=%d model=%s", len(body.prompt), body.model)
    try:
        return await svc.generate(
            body.prompt,
            model=body.model,
            system=body.system,
            options=body.options or None,
        )
    except Exception as exc:  # noqa: BLE001
        _handle_llm_error(exc)


@router.post(
    "/summarize",
    response_model=LlmSummarizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Summarise medical document text",
)
async def summarize(
    body: LlmSummarizeRequest,
    _: CurrentUser,
    svc: LlmService = LlmServiceDep,
) -> LlmSummarizeResponse:
    """
    Pass raw OCR text through a clinical prompt template and return a
    structured medical summary.

    - **text** — raw document text (max 32 768 chars).
    - **language** — desired output language (default: English).
    - **model** — override the default model.
    - **options** — Ollama model parameters.
    """
    log.info(
        "POST /llm/summarize text_len=%d lang=%s model=%s",
        len(body.text),
        body.language,
        body.model,
    )
    try:
        return await svc.summarize(
            body.text,
            language=body.language,
            model=body.model,
            options=body.options or None,
        )
    except Exception as exc:  # noqa: BLE001
        _handle_llm_error(exc)


@router.post(
    "/extract",
    response_model=LlmExtractResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract structured medical entities from document text",
)
async def extract(
    body: LlmExtractRequest,
    _: CurrentUser,
    svc: LlmService = LlmServiceDep,
) -> LlmExtractResponse:
    """
    Send raw OCR / document text to the Ollama model and get back a structured
    JSON object containing:

    - **document_date** — date of the document
    - **diagnoses** — list of diagnoses
    - **medications** — list of `{name, dosage, frequency}` objects
    - **lab_results** — list of `{test_name, value, unit}` objects
    - **doctor_name** — attending physician
    - **hospital_name** — facility name

    Missing fields are returned as `null`. The model runs with `temperature=0`
    by default for deterministic, hallucination-resistant output.
    """
    log.info("POST /llm/extract text_len=%d model=%s", len(body.text), body.model)
    try:
        return await svc.extract(
            body.text,
            model=body.model,
            options=body.options or None,
        )
    except Exception as exc:  # noqa: BLE001
        _handle_llm_error(exc)


@router.post(
    "/physician-summary",
    response_model=LlmPhysicianSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a concise physician summary from structured medical data",
)
async def physician_summary(
    body: LlmPhysicianSummaryRequest,
    _: CurrentUser,
    svc: LlmService = LlmServiceDep,
) -> LlmPhysicianSummaryResponse:
    """
    Generate a concise ≈150-word summary for a physician from structured medical
    data (typically the output of the ``/llm/extract`` endpoint).

    The model is strictly instructed to use ONLY the provided data — no guessing,
    no hallucination, no invented values.

    - **data** — structured medical dict (``LlmExtractionResult`` shape).
    - **model** — override the default Ollama model.
    - **options** — Ollama model parameters.

    ### Recommended workflow
    1. Call ``POST /llm/extract`` with raw OCR text → receive ``LlmExtractResponse``.
    2. Pass ``response.data`` as the ``data`` field here.
    """
    log.info(
        "POST /llm/physician-summary keys=%s model=%s",
        list(body.data.keys()),
        body.model,
    )
    try:
        return await svc.physician_summary(
            body.data,
            model=body.model,
            options=body.options or None,
        )
    except Exception as exc:  # noqa: BLE001
        _handle_llm_error(exc)
