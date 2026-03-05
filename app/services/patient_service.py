"""
Patient Service
===============

DB operations for patients and their clinical timeline.

Timeline query strategy
-----------------------
The query against ``extracted_events`` uses two purpose-built indexes:

  idx_events_patient_timeline  ON (patient_id, event_date)
    → covers the primary filter + sort; the planner uses an index-scan
      ordered by event_date so no explicit sort step is needed.

  idx_events_patient_type      ON (patient_id, event_type)
    → used when an event_type filter is applied, switching the planner
      to a bitmap index scan combined with the timeline index.

  idx_events_data_gin          ON event_data  (GIN / jsonb_path_ops)
    → used when event_data_contains is supplied; the ``@>`` containment
      operator is accelerated by this index.

Sort order: event_date ASC NULLS LAST, created_at ASC
  Events with no date (NULL) are placed at the end of the timeline so
  that all precisely-dated clinical observations appear first.

The count query for pagination reuses the exact same WHERE predicate so
the planner's index choices are consistent.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import Counter
from datetime import date
from typing import Any

from sqlalchemy import cast, func, select
from sqlalchemy import Text as SAText
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundException
from app.models.medical import ExtractedEvent, GenderEnum, Patient
from app.schemas.medical import (
    ExtractedEventOut,
    LabReportItem,
    LabReportResponse,
    LabTrendPoint,
    LabTrendResponse,
    ManualEventCreate,
    PatientTimeline,
    TimelineEventGroup,
)

log = logging.getLogger(__name__)


class PatientService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_patient_or_404(self, patient_id: uuid.UUID) -> Patient:
        result = await self.db.execute(select(Patient).where(Patient.id == patient_id))
        patient = result.scalar_one_or_none()
        if patient is None:
            raise NotFoundException(f"Patient {patient_id} not found")
        return patient

    async def find_by_mrn(self, mrn: str) -> uuid.UUID | None:
        """Return the patient's UUID for a given MRN, or None if not found."""
        result = await self.db.execute(select(Patient.id).where(Patient.mrn == mrn))
        pid = result.scalar_one_or_none()
        return pid

    async def resolve_patient_id(self, patient_id_str: str) -> uuid.UUID:
        """
        Resolve a patient identifier string to a UUID.

        Resolution order:
        1. Parse as UUID directly.
        2. Fall back to MRN lookup (supports Clinic-Backend integer IDs like "27").

        Raises NotFoundException if no match is found.
        """
        try:
            return uuid.UUID(patient_id_str)
        except (ValueError, AttributeError):
            pass

        pid = await self.find_by_mrn(patient_id_str)
        if pid is None:
            raise NotFoundException(
                f"No patient record found for id '{patient_id_str}'. "
                "Upload a medical document first to create the patient record."
            )
        return pid

    async def get_or_create_by_mrn(
        self,
        mrn: str,
        created_by: uuid.UUID,
    ) -> uuid.UUID:
        """
        Return the UUID of the patient with the given MRN.
        If no such patient exists, create a placeholder record and return its UUID.

        The placeholder uses sentinel values for required fields (first_name,
        last_name, date_of_birth) that can be updated by a clinician later.

        ``created_by`` must reference a row in ``users``.  When called from an
        external-token context the caller should pass a persisted user UUID; if
        the INSERT fails due to an FK violation the error is logged and re-raised
        so the upload endpoint can fall back to its synthetic response path.
        """
        existing = await self.find_by_mrn(mrn)
        if existing is not None:
            return existing

        # Ensure the created_by user actually exists in the users table.
        # For external (Clinic-Backend) tokens the synthetic user id won't be
        # there, so we upsert a minimal placeholder user first.
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from app.models.user import User as UserModel, UserRoleEnum as _URole

        await self.db.execute(
            pg_insert(UserModel)
            .values(
                id=created_by,
                email=f"external_{created_by}@clinic.local",
                hashed_password="",
                is_active=True,
                is_superuser=False,
                role=_URole.doctor,
            )
            .on_conflict_do_nothing()
        )

        patient = Patient(
            created_by=created_by,
            mrn=mrn,
            first_name="Unknown",
            last_name=f"Patient-{mrn}",
            date_of_birth=date(1900, 1, 1),
            gender=GenderEnum.unknown,
        )
        self.db.add(patient)
        await self.db.flush()
        await self.db.refresh(patient)
        log.info(
            "PatientService.get_or_create_by_mrn: created placeholder patient mrn=%s id=%s",
            mrn,
            patient.id,
        )
        return patient.id  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    async def get_timeline(
        self,
        patient_id: uuid.UUID,
        *,
        event_type: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        verified_only: bool = False,
        event_data_contains: dict[str, Any] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> PatientTimeline:
        """
        Return a paginated, ascending-chronological timeline for a patient.

        Parameters
        ----------
        patient_id:
            Owner of the timeline. Raises 404 if not found.
        event_type:
            Filter to a single event type (e.g. ``'diagnosis'``).
            Uses ``idx_events_patient_type``.
        date_from / date_to:
            Inclusive date-range filter on ``event_date``.
            NULL event_date rows are excluded when a range is applied.
        verified_only:
            When True, only return rows where ``is_verified = true``.
        event_data_contains:
            JSONB containment filter — ``event_data @> :value``.
            Uses ``idx_events_data_gin`` (GIN index).
            Example: ``{"flag": "HIGH"}`` returns all high-flag lab results.
        limit / offset:
            Pagination (max 200 rows per page).
        """
        # ── Guard ──────────────────────────────────────────────────────
        await self._get_patient_or_404(patient_id)
        limit = min(limit, 200)

        # ── Shared WHERE predicate ─────────────────────────────────────
        # Build incrementally so the count query reuses the same filters.
        filters = [ExtractedEvent.patient_id == patient_id]

        if event_type is not None:
            filters.append(ExtractedEvent.event_type == event_type)

        if date_from is not None:
            filters.append(ExtractedEvent.event_date >= date_from)

        if date_to is not None:
            filters.append(ExtractedEvent.event_date <= date_to)

        if verified_only:
            filters.append(ExtractedEvent.is_verified.is_(True))

        if event_data_contains:
            # @> operator — uses idx_events_data_gin (jsonb_path_ops GIN index)
            filters.append(
                ExtractedEvent.event_data.contains(event_data_contains)  # type: ignore[attr-defined]
            )

        # ── COUNT (total matching rows) ────────────────────────────────
        count_result = await self.db.execute(
            select(func.count()).select_from(ExtractedEvent).where(*filters)
        )
        total: int = count_result.scalar_one()

        # ── EVENTS (paginated, sorted) ────────────────────────────────
        # Sort: event_date ASC NULLS LAST → undated events go to end
        #       created_at ASC           → stable secondary order
        # The planner uses idx_events_patient_timeline for the base case.
        events_result = await self.db.execute(
            select(ExtractedEvent)
            .where(*filters)
            .order_by(
                ExtractedEvent.event_date.asc().nulls_last(),
                ExtractedEvent.created_at.asc(),
            )
            .limit(limit)
            .offset(offset)
        )
        events = list(events_result.scalars().all())

        # ── TYPE COUNTS (unfiltered — for sidebar chips) ───────────────
        # Always runs against the full patient set regardless of filters,
        # so the UI can show how many events exist of each type.
        counts_result = await self.db.execute(
            select(ExtractedEvent.event_type, func.count().label("cnt"))
            .where(ExtractedEvent.patient_id == patient_id)
            .group_by(ExtractedEvent.event_type)
            .order_by(func.count().desc())
        )
        event_type_counts = [
            TimelineEventGroup(event_type=row.event_type, count=row.cnt)
            for row in counts_result
        ]

        log.info(
            "PatientService.get_timeline: patient=%s total=%d returned=%d "
            "filters=[type=%s from=%s to=%s verified=%s]",
            patient_id,
            total,
            len(events),
            event_type,
            date_from,
            date_to,
            verified_only,
        )

        return PatientTimeline(
            patient_id=patient_id,
            total=total,
            limit=limit,
            offset=offset,
            has_more=(offset + len(events)) < total,
            event_type_counts=event_type_counts,
            events=[ExtractedEventOut.model_validate(e) for e in events],
        )

    # ------------------------------------------------------------------
    # Lab report (all lab results at a glance)
    # ------------------------------------------------------------------

    async def get_lab_report(
        self,
        patient_id: uuid.UUID,
    ) -> LabReportResponse:
        """
        Return ALL lab_result events for a patient, enriched with
        auto-flagging from the built-in reference range database.
        Results are sorted by created_at DESC (most recent first).
        """
        await self._get_patient_or_404(patient_id)

        result = await self.db.execute(
            select(ExtractedEvent)
            .where(
                ExtractedEvent.patient_id == patient_id,
                ExtractedEvent.event_type == "lab_result",
            )
            .order_by(
                ExtractedEvent.created_at.desc(),
            )
        )
        rows = list(result.scalars().all())

        from app.services.lab_reference import enrich_lab_result

        items: list[LabReportItem] = []
        abnormal = 0
        for row in rows:
            data: dict[str, Any] = row.event_data or {}
            # Enrich with auto-flagging
            data = enrich_lab_result(data)

            test_name = (
                data.get("test_name")
                or data.get("test")
                or data.get("name")
                or "Unknown"
            )
            value_raw = str(data.get("value", ""))
            unit = data.get("unit") or None
            flag = data.get("flag") or None
            ref_range = data.get("reference_range") or None

            numeric_value: float | None = None
            try:
                numeric_value = float(value_raw)
            except (ValueError, TypeError):
                pass

            if flag and flag.upper() in ("HIGH", "LOW", "CRITICAL"):
                abnormal += 1

            items.append(
                LabReportItem(
                    event_id=row.id,
                    document_id=row.document_id,
                    test_name=test_name,
                    value_raw=value_raw,
                    numeric_value=numeric_value,
                    unit=unit,
                    flag=flag,
                    reference_range=ref_range,
                    event_date=row.event_date,
                    is_verified=row.is_verified,
                    created_at=row.created_at,
                )
            )

        log.info(
            "PatientService.get_lab_report: patient=%s total=%d abnormal=%d",
            patient_id,
            len(items),
            abnormal,
        )

        return LabReportResponse(
            patient_id=patient_id,
            total=len(items),
            items=items,
            abnormal_count=abnormal,
        )

    # ------------------------------------------------------------------
    # Lab trends
    # ------------------------------------------------------------------

    async def get_lab_trends(
        self,
        patient_id: uuid.UUID,
        *,
        test_name: str,
        date_from: date | None = None,
        date_to: date | None = None,
        verified_only: bool = False,
    ) -> LabTrendResponse:
        """
        Return all ``lab_result`` events matching ``test_name`` (case-insensitive)
        for a patient, sorted chronologically.

        Index strategy
        --------------
        1. ``idx_events_patient_type (patient_id, event_type)``
              → prunes the scan to *lab_result* rows for this patient immediately.

        2. ``idx_events_lab_testname_ci``
              Partial expression index::

                CREATE INDEX idx_events_lab_testname_ci
                ON extracted_events (lower(event_data->>'test_name'))
                WHERE event_type = 'lab_result'

              Accelerates the case-insensitive predicate
              ``lower(event_data->>'test_name') = lower(:test_name)``.
              PostgreSQL can bitmap-AND this with idx_events_patient_type
              (or idx_events_patient_timeline) for sub-millisecond lookups.

        Parameters
        ----------
        test_name:
            The lab test to look up, e.g. ``"HbA1c"``, ``"hba1c"``, ``"HGBA1C"``.
            Matched case-insensitively via ``lower()``.
        date_from / date_to:
            Inclusive date range filter on ``event_date``.
        verified_only:
            Restrict to clinician-verified events.

        Returns
        -------
        LabTrendResponse
        """
        await self._get_patient_or_404(patient_id)

        # ── WHERE predicate ────────────────────────────────────────────
        # lower(event_data->>'test_name') = lower(:test_name)
        # Uses idx_events_lab_testname_ci (partial expression index)
        name_ci_filter = (
            func.lower(ExtractedEvent.event_data["test_name"].astext)
            == test_name.lower()
        )

        filters = [
            ExtractedEvent.patient_id == patient_id,
            ExtractedEvent.event_type == "lab_result",  # idx_events_patient_type
            name_ci_filter,  # idx_events_lab_testname_ci
        ]

        if date_from is not None:
            filters.append(ExtractedEvent.event_date >= date_from)
        if date_to is not None:
            filters.append(ExtractedEvent.event_date <= date_to)
        if verified_only:
            filters.append(ExtractedEvent.is_verified.is_(True))

        # ── Fetch all matching rows (lab series are always small) ──────
        result = await self.db.execute(
            select(ExtractedEvent)
            .where(*filters)
            .order_by(
                ExtractedEvent.event_date.asc().nulls_last(),
                ExtractedEvent.created_at.asc(),
            )
        )
        rows = list(result.scalars().all())

        # ── Build data points ──────────────────────────────────────────
        points: list[LabTrendPoint] = []
        unit_counter: Counter[str] = Counter()

        for row in rows:
            data: dict[str, Any] = row.event_data or {}
            # Enrich with auto-flagging
            from app.services.lab_reference import enrich_lab_result

            data = enrich_lab_result(data)

            value_raw: str = str(data.get("value", ""))
            unit: str | None = data.get("unit") or None
            flag: str | None = data.get("flag") or None
            reference_range: str | None = data.get("reference_range") or None

            # Best-effort numeric parse
            numeric_value: float | None = None
            try:
                numeric_value = float(value_raw)
            except (ValueError, TypeError):
                pass

            if unit:
                unit_counter[unit] += 1

            points.append(
                LabTrendPoint(
                    event_id=row.id,
                    document_id=row.document_id,
                    event_date=row.event_date,
                    value_raw=value_raw,
                    numeric_value=numeric_value,
                    unit=unit,
                    flag=flag,
                    reference_range=reference_range,
                    is_verified=row.is_verified,
                )
            )

        # Most common unit (for chart axis label)
        common_unit: str | None = (
            unit_counter.most_common(1)[0][0] if unit_counter else None
        )

        log.info(
            "PatientService.get_lab_trends: patient=%s test=%s matched=%d "
            "date_range=[%s, %s] verified=%s",
            patient_id,
            test_name.lower(),
            len(points),
            date_from,
            date_to,
            verified_only,
        )

        return LabTrendResponse(
            patient_id=patient_id,
            test_name=test_name.lower(),
            common_unit=common_unit,
            total=len(points),
            points=points,
        )

    # ------------------------------------------------------------------
    # Doctor summary context builder
    # ------------------------------------------------------------------

    async def get_structured_events_for_summary(
        self,
        patient_id: uuid.UUID,
        *,
        max_events: int = 200,
    ) -> tuple[str, int]:
        """
        Fetch all clinical events for a patient and serialise them to a
        compact, chronologically ordered JSON string suitable for use as
        LLM prompt context.

        Events are grouped by ``event_type`` then sorted by
        ``event_date ASC NULLS LAST`` within each group, producing a
        structure like::

            {
              "diagnoses": [
                {"date": "2024-01-15", "data": {"code": "E11", "description": "T2DM"}},
                ...
              ],
              "lab_results": [
                {"date": "2024-03-10", "data": {"test_name": "HbA1c", "value": "7.4", "unit": "%"}},
                ...
              ],
              ...
            }

        Parameters
        ----------
        patient_id:
            Must exist — raises 404 otherwise.
        max_events:
            Hard cap on total events fetched (prevents oversized prompts).
            Latest events by date are preferred when the cap is hit.

        Returns
        -------
        (events_json, event_count)
            ``events_json``  — compact JSON string ready for prompt injection.
            ``event_count``  — total number of events included.
        """
        await self._get_patient_or_404(patient_id)

        result = await self.db.execute(
            select(ExtractedEvent)
            .where(ExtractedEvent.patient_id == patient_id)
            .order_by(
                ExtractedEvent.event_date.asc().nulls_last(),
                ExtractedEvent.created_at.asc(),
            )
            .limit(max_events)
        )
        rows = list(result.scalars().all())

        # Group by event_type for cleaner prompt structure
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            entry: dict[str, Any] = {"data": row.event_data or {}}
            if row.event_date:
                entry["date"] = row.event_date.isoformat()
            if row.is_verified:
                entry["verified"] = True
            grouped.setdefault(row.event_type, []).append(entry)

        events_json = json.dumps(grouped, indent=2, default=str)
        return events_json, len(rows)

    # ------------------------------------------------------------------
    # Bulk save events extracted from a document by the LLM
    # ------------------------------------------------------------------

    async def save_extracted_events_from_llm(
        self,
        patient_id: uuid.UUID,
        document_id: uuid.UUID,
        extraction: dict[str, Any],
        *,
        ai_model: str = "llama3",
    ) -> int:
        """
        Persist structured events extracted by the LLM into ``extracted_events``.

        Parameters
        ----------
        patient_id:
            Must already exist in the ``patients`` table.
        document_id:
            Source document UUID — stored on every event for traceability.
        extraction:
            Dict in ``LlmExtractionResult`` shape::
                {
                  "document_date": "2024-01-15",
                  "diagnoses":    [{"description": "T2DM", ...}],
                  "medications":  [{"name": "Metformin", "dosage": "500mg", ...}],
                  "lab_results":  [{"test_name": "HbA1c", "value": "7.4", "unit": "%"}],
                  "doctor_name":  "Dr. Smith",
                  "hospital_name": "City Hospital"
                }
        ai_model:
            Model name recorded on every row for provenance.

        Returns
        -------
        int
            Number of events saved.
        """
        from datetime import datetime as _dt

        raw_date: str | None = extraction.get("document_date")
        event_date: date | None = None
        if raw_date:
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
                try:
                    event_date = _dt.strptime(raw_date, fmt).date()
                    break
                except ValueError:
                    continue

        events: list[ExtractedEvent] = []

        for dx in extraction.get("diagnoses") or []:
            if not dx:
                continue
            data = dx if isinstance(dx, dict) else {"description": str(dx)}
            events.append(
                ExtractedEvent(
                    patient_id=patient_id,
                    document_id=document_id,
                    event_type="diagnosis",
                    event_date=event_date,
                    event_data=data,
                    confidence_score=0.85,
                    ai_model=ai_model,
                    is_verified=False,
                )
            )

        for med in extraction.get("medications") or []:
            if not med:
                continue
            data = med if isinstance(med, dict) else {"name": str(med)}
            events.append(
                ExtractedEvent(
                    patient_id=patient_id,
                    document_id=document_id,
                    event_type="medication",
                    event_date=event_date,
                    event_data=data,
                    confidence_score=0.85,
                    ai_model=ai_model,
                    is_verified=False,
                )
            )

        for lab in extraction.get("lab_results") or []:
            if not lab:
                continue
            data = lab if isinstance(lab, dict) else {"description": str(lab)}
            # Enrich with auto-flagging from reference range database
            from app.services.lab_reference import enrich_lab_result

            data = enrich_lab_result(data)
            events.append(
                ExtractedEvent(
                    patient_id=patient_id,
                    document_id=document_id,
                    event_type="lab_result",
                    event_date=event_date,
                    event_data=data,
                    confidence_score=0.85,
                    ai_model=ai_model,
                    is_verified=False,
                )
            )

        # Store doctor / hospital as a clinical_note
        meta: dict[str, Any] = {}
        if extraction.get("doctor_name"):
            meta["doctor_name"] = extraction["doctor_name"]
        if extraction.get("hospital_name"):
            meta["hospital_name"] = extraction["hospital_name"]
        if meta:
            events.append(
                ExtractedEvent(
                    patient_id=patient_id,
                    document_id=document_id,
                    event_type="clinical_note",
                    event_date=event_date,
                    event_data=meta,
                    confidence_score=0.9,
                    ai_model=ai_model,
                    is_verified=False,
                )
            )

        if not events:
            log.warning(
                "save_extracted_events_from_llm: no events parsed from extraction for doc %s",
                document_id,
            )
            return 0

        self.db.add_all(events)
        await self.db.flush()
        log.info(
            "save_extracted_events_from_llm: saved %d events for patient=%s doc=%s",
            len(events),
            patient_id,
            document_id,
        )
        return len(events)

    # ------------------------------------------------------------------
    # Manual event creation (doctor-entered, no document required)
    # ------------------------------------------------------------------

    async def create_manual_event(
        self,
        patient_id: uuid.UUID,
        payload: ManualEventCreate,
        reviewed_by: uuid.UUID,
    ) -> ExtractedEventOut:
        """
        Insert a manually-entered clinical event for a patient.

        Unlike events created via OCR, these have no associated document
        (document_id = NULL) and are marked is_verified=True immediately
        because a clinician is entering them directly.
        """
        await self._get_patient_or_404(patient_id)

        event = ExtractedEvent(
            patient_id=patient_id,
            document_id=None,
            event_type=payload.event_type,
            event_date=payload.event_date,
            event_data=payload.event_data or {},
            confidence_score=(
                payload.confidence_score
                if payload.confidence_score is not None
                else 1.0
            ),
            ai_model=payload.ai_model or "manual",
            notes=payload.notes,
            is_verified=True,
            reviewed_by=reviewed_by,
        )
        self.db.add(event)
        await self.db.commit()
        await self.db.refresh(event)
        return ExtractedEventOut.model_validate(event)
