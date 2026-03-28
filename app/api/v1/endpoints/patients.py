"""
Patient Endpoints
=================

GET /api/v1/patients/{patient_id}/timeline    — paginated clinical event timeline
GET /api/v1/patients/{patient_id}/lab-trends  — time-series for a single lab test
GET /api/v1/patients/{patient_id}/summary     — AI-generated doctor summary (24h cache)
"""

from __future__ import annotations

import logging
import uuid
from datetime import date
from typing import Annotated, Any

log = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import CurrentUser, DBDep
from app.core.exceptions import NotFoundException
from app.core.roles import ROLE_DOCTOR
from app.schemas.medical import (
    DoctorSummaryResponse,
    ExtractedEventOut,
    LabReportResponse,
    LabTrendResponse,
    ManualEventCreate,
    PatientTimeline,
)
from app.services.cache_service import summary_cache
from app.services.document_service import DocumentService
from app.services.llm_service import (
    LlmConnectionError,
    LlmResponseError,
    LlmService,
    LlmTimeoutError,
)
from app.services.patient_service import PatientService

router = APIRouter(prefix="/patients", tags=["patients"])


# ── Dependency ────────────────────────────────────────────────────────────────


def get_patient_service(db: DBDep) -> PatientService:
    return PatientService(db)


PatientServiceDep = Annotated[PatientService, Depends(get_patient_service)]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_patient_uuid(patient_id: str) -> uuid.UUID | None:
    """
    Try to parse patient_id as a UUID.
    Returns None when patient_id is a Clinic-Backend integer like "27".
    Callers return empty data in that case rather than erroring.
    """
    try:
        return uuid.UUID(patient_id)
    except (ValueError, AttributeError):
        return None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get(
    "/{patient_id}/timeline",
    response_model=PatientTimeline,
    summary="Get paginated clinical event timeline for a patient",
)
async def get_patient_timeline(
    patient_id: str,  # str so integer IDs like "27" are accepted
    current_user: CurrentUser,
    svc: PatientServiceDep,
    view: Annotated[
        str | None,
        Query(
            description=(
                "Timeline view preset. "
                "'medicines' — shows only medication events. "
                "'summary' — shows diagnoses, clinical notes, follow-ups, procedures. "
                "Omit for the full timeline."
            ),
        ),
    ] = None,
    event_type: Annotated[
        str | None,
        Query(
            description="Filter by a single event type (overridden by 'view' when set)"
        ),
    ] = None,
    date_from: Annotated[
        date | None,
        Query(description="Include only events on or after this date (YYYY-MM-DD)."),
    ] = None,
    date_to: Annotated[
        date | None,
        Query(description="Include only events on or before this date (YYYY-MM-DD)."),
    ] = None,
    verified_only: Annotated[
        bool,
        Query(description="When true, return only clinician-verified events."),
    ] = False,
    event_data_contains: Annotated[
        str | None,
        Query(
            alias="event_data",
            description=(
                "JSONB containment filter (JSON string). "
                "Uses the GIN index on event_data. "
                'Example: \'{"flag":"HIGH"}\' returns all high-flag lab results.'
            ),
        ),
    ] = None,
    # ── Pagination ────────────────────────────────────────────────────────────
    limit: Annotated[
        int,
        Query(ge=1, le=200, description="Number of events per page (max 200)."),
    ] = 50,
    offset: Annotated[
        int,
        Query(ge=0, description="Number of events to skip."),
    ] = 0,
) -> PatientTimeline:
    """
    Return a chronologically ordered, paginated list of clinical events
    for the specified patient.

    ### View presets
    Use `?view=medicines` to see only medication events.
    Use `?view=summary` to see diagnoses, clinical notes, follow-ups, and procedures.
    Omit `view` for the full unfiltered timeline.

    ### Sort order
    ``event_date ASC NULLS LAST`` → ``created_at ASC``

    Events without a date appear at the end of the timeline so all
    precisely-dated observations are shown first.

    ### Index usage
    | Filter applied          | Index used                        |
    |-------------------------|-----------------------------------|
    | (base)                  | `idx_events_patient_timeline`     |
    | `event_type`            | `idx_events_patient_type`         |
    | `event_data` (JSONB `@>`) | `idx_events_data_gin` (GIN)     |
    | `verified_only`         | `idx_events_unverified` (partial) |

    Any authenticated user may view their patient timeline.
    """
    _empty = PatientTimeline(
        patient_id=uuid.UUID("00000000-0000-0000-0000-000000000000"),
        events=[],
        total=0,
        limit=limit,
        offset=offset,
        has_more=False,
        event_type_counts=[],
    )

    pid = _parse_patient_uuid(patient_id)
    if pid is None:
        # Non-UUID id (e.g. Clinic-Backend integer) — try MRN lookup
        try:
            pid = await svc.resolve_patient_id(patient_id)
        except NotFoundException:
            return _empty
        except Exception:
            log.exception("timeline: resolve_patient_id failed for '%s'", patient_id)
            return _empty

    log.info("timeline: resolved patient_id='%s' → pid=%s", patient_id, pid)

    # Parse optional JSONB filter from query string
    parsed_event_data: dict[str, Any] | None = None
    if event_data_contains:
        import json as _json

        try:
            parsed_event_data = _json.loads(event_data_contains)
            if not isinstance(parsed_event_data, dict):
                parsed_event_data = None
        except ValueError:
            parsed_event_data = None

    # ── Resolve view preset → event_types filter ─────────────────────────────
    resolved_event_types: list[str] | None = None
    if view == "medicines":
        resolved_event_types = ["medication"]
    elif view == "summary":
        resolved_event_types = [
            "diagnosis",
            "clinical_note",
            "follow_up",
            "procedure",
            "other",
        ]

    try:
        return await svc.get_timeline(
            pid,
            event_type=event_type if resolved_event_types is None else None,
            event_types=resolved_event_types,
            date_from=date_from,
            date_to=date_to,
            verified_only=verified_only,
            event_data_contains=parsed_event_data,
            limit=limit,
            offset=offset,
        )
    except Exception:
        log.exception("timeline: get_timeline failed for pid=%s", pid)
        return _empty


@router.get(
    "/{patient_id}/lab-trends",
    response_model=LabTrendResponse,
    summary="Get time-series trend for a specific lab test",
)
async def get_lab_trends(
    patient_id: str,  # str so integer IDs like "27" are accepted
    current_user: CurrentUser,
    svc: PatientServiceDep,
    test_name: Annotated[str, Query(min_length=1, max_length=256)],
    date_from: Annotated[date | None, Query()] = None,
    date_to: Annotated[date | None, Query()] = None,
    verified_only: Annotated[bool, Query()] = False,
) -> LabTrendResponse:
    """
    Return all ``lab_result`` events matching ``test_name`` (case-insensitive)
    for a patient, sorted ``event_date ASC NULLS LAST``.

    Designed for charting longitudinal lab values (e.g. HbA1c over time).

    ### Index usage
    | Predicate                            | Index                          |
    |--------------------------------------|--------------------------------|
    | `event_type = 'lab_result'`           | `idx_events_patient_type`      |
    | `lower(event_data->>'test_name') = ?` | `idx_events_lab_testname_ci`   |

    ``idx_events_lab_testname_ci`` is a partial expression index defined as::

        CREATE INDEX idx_events_lab_testname_ci
        ON extracted_events (lower(event_data->>'test_name'))
        WHERE event_type = 'lab_result'

    PostgreSQL can bitmap-AND both indexes for sub-millisecond lookups.

    ### Response fields
    - **points** — one entry per matching event, sorted chronologically
    - **common_unit** — most frequent unit across all points (for chart axis)
    - **numeric_value** — best-effort float parse of the raw value string
    - **flag** — HIGH | LOW | CRITICAL | NORMAL when present in source data

    Any authenticated user may view lab trends.
    """
    _null_pid = uuid.UUID("00000000-0000-0000-0000-000000000000")
    _empty = LabTrendResponse(
        patient_id=_null_pid,
        test_name=test_name,
        total=0,
        points=[],
        common_unit=None,
    )

    pid = _parse_patient_uuid(patient_id)
    if pid is None:
        # Non-UUID id — try MRN lookup
        try:
            pid = await svc.resolve_patient_id(patient_id)
        except NotFoundException:
            log.info("lab-trends: patient '%s' not found by MRN", patient_id)
            return _empty
        except Exception:
            log.exception("lab-trends: resolve_patient_id failed for '%s'", patient_id)
            return _empty

    log.info("lab-trends: resolved patient_id='%s' → pid=%s", patient_id, pid)

    try:
        return await svc.get_lab_trends(
            pid,
            test_name=test_name,
            date_from=date_from,
            date_to=date_to,
            verified_only=verified_only,
        )
    except Exception:
        log.exception("lab-trends: get_lab_trends failed for pid=%s", pid)
        return _empty


@router.get(
    "/{patient_id}/lab-report",
    response_model=LabReportResponse,
    summary="Get full lab report with normal-range flags for a patient",
)
async def get_lab_report(
    patient_id: str,
    current_user: CurrentUser,
    svc: PatientServiceDep,
    db: DBDep,
) -> LabReportResponse:
    """
    Return ALL lab_result events for a patient with automatic
    HIGH / LOW / NORMAL flagging based on built-in reference ranges.

    Unlike lab-trends (which requires a specific test_name), this endpoint
    returns every lab result at once — ideal for a "Lab Report" card.

    When documents are still being processed (OCR / LLM extraction),
    ``still_processing`` is set to ``True`` so clients can poll.
    """
    _empty = LabReportResponse(
        patient_id=uuid.UUID("00000000-0000-0000-0000-000000000000"),
        total=0,
        items=[],
        abnormal_count=0,
    )

    pid = _parse_patient_uuid(patient_id)
    if pid is None:
        try:
            pid = await svc.resolve_patient_id(patient_id)
        except NotFoundException:
            return _empty
        except Exception:
            log.exception("lab-report: resolve_patient_id failed for '%s'", patient_id)
            return _empty

    try:
        report = await svc.get_lab_report(pid)
    except Exception:
        log.exception("lab-report: get_lab_report failed for pid=%s", pid)
        return _empty

    # Check if any documents are still being processed (OCR + LLM pipeline)
    try:
        doc_svc = DocumentService(db)
        report.still_processing = await doc_svc.has_pending_documents(pid)
    except Exception:
        log.warning("lab-report: has_pending_documents check failed for pid=%s", pid)
        report.still_processing = False

    return report


@router.get(
    "/{patient_id}/summary",
    response_model=DoctorSummaryResponse,
    summary="Get AI-generated doctor-facing patient summary",
    responses={
        202: {
            "description": "Documents are still being processed. Retry after the indicated delay.",
            "content": {
                "application/json": {
                    "example": {
                        "status": "processing",
                        "detail": "Documents are still being processed. Please retry shortly.",
                        "retry_after": 5,
                    }
                }
            },
        },
    },
)
async def get_patient_summary(
    patient_id: str,  # str so integer IDs like "27" are accepted
    current_user: CurrentUser,
    svc: PatientServiceDep,
    db: DBDep,
    refresh: Annotated[
        bool,
        Query(description="Force regeneration even when a cached copy exists."),
    ] = False,
) -> DoctorSummaryResponse:
    """
    Generate a concise 150–200 word narrative summary of the patient's full
    clinical history, powered by a locally-running LLaMA model via Ollama.

    ### Flow
    1. **Cache lookup** — If a fresh summary exists (< 24 h old) it is returned
       immediately with ``cached: true``.
    2. **Event fetch** — All ``ExtractedEvent`` rows for the patient are loaded
       and serialised chronologically (newest-first within groups).
    3. **LLM call** — Events are sent to LLaMA with a strict clinical prompt
       emphasising chronic conditions, medication changes, and abnormal trends.
    4. **Cache store** — The new summary is stored for 24 h.
    5. **Error handling** — Timeout → ``504``; connectivity / model errors → ``503``.

    ### Role requirement
    Only ``doctor`` accounts may call this endpoint.

    Any authenticated user may read; Ollama is only called when needed.
    """
    try:
        pid = await svc.resolve_patient_id(patient_id)
    except NotFoundException as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Database is temporarily unavailable. Please try again shortly.",
        ) from exc

    cache_key = f"patient_summary:{pid}"

    # Stampede-safe: per-key asyncio.Lock prevents multiple concurrent requests
    # from all calling the LLM simultaneously for the same patient.
    async with summary_cache.key_lock(cache_key):
        if not refresh:
            cached_result: DoctorSummaryResponse | None = await summary_cache.get(
                cache_key
            )
            if cached_result is not None:
                cached_result.cached = True
                return cached_result

        # ── Fetch structured events ───────────────────────────────────────
        try:
            events_json, event_count = await svc.get_structured_events_for_summary(pid)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Database is temporarily unavailable. Please try again shortly.",
            ) from exc

        if event_count == 0:
            # Before returning 404, check if documents are still being processed
            from fastapi.responses import JSONResponse

            doc_svc = DocumentService(db)
            try:
                still_processing = await doc_svc.has_pending_documents(pid)
            except Exception:
                still_processing = False

            if still_processing:
                return JSONResponse(
                    status_code=202,
                    content={
                        "status": "processing",
                        "detail": "Documents are still being processed (OCR + AI extraction). Please retry shortly.",
                        "retry_after": 5,
                    },
                )

            raise HTTPException(
                status_code=404,
                detail="No clinical events found for this patient. Upload a medical document first.",
            )

        # ── Call LLaMA ───────────────────────────────────────────────────
        try:
            async with LlmService() as llm:
                result = await llm.doctor_summary(
                    events_json,
                    patient_id=pid,
                    event_count=event_count,
                )
        except LlmTimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail="The LLM model timed out while generating the summary. Try again shortly.",
            ) from exc
        except LlmConnectionError as exc:
            raise HTTPException(
                status_code=503,
                detail="Unable to reach the local LLM service. Ensure Ollama is running.",
            ) from exc
        except LlmResponseError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"The LLM returned an invalid response: {exc}",
            ) from exc

        # ── Cache and return ─────────────────────────────────────────────
        await summary_cache.set(cache_key, result)
        return result


@router.post(
    "/{patient_id}/events",
    response_model=ExtractedEventOut,
    status_code=201,
    summary="Manually create a clinical event for a patient (doctor only)",
)
async def create_manual_event(
    patient_id: str,
    payload: ManualEventCreate,
    current_user: CurrentUser,
    svc: PatientServiceDep,
) -> ExtractedEventOut:
    """Create a clinical event directly. Doctor role required."""
    from app.core.exceptions import ForbiddenException

    role = getattr(current_user, "role", None)
    if str(role) != ROLE_DOCTOR:
        raise ForbiddenException("Doctor role required to create clinical events")

    pid = _parse_patient_uuid(patient_id)
    if pid is None:
        raise HTTPException(
            status_code=422, detail=f"Invalid patient UUID: '{patient_id}'"
        )

    return await svc.create_manual_event(
        patient_id=pid,
        payload=payload,
        reviewed_by=current_user.id,
    )
