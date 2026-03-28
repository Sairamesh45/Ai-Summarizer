import uuid
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.models.medical import DocumentStatusEnum, DocumentTypeEnum, GenderEnum


# =============================================================================
#  PATIENT
# =============================================================================
class PatientBase(BaseModel):
    mrn: str = Field(..., max_length=64, description="Medical record number")
    first_name: str = Field(..., max_length=128)
    last_name: str = Field(..., max_length=128)
    date_of_birth: date
    gender: GenderEnum = GenderEnum.unknown
    phone: str | None = Field(default=None, max_length=32)
    email: str | None = Field(default=None, max_length=255)
    extra_data: dict[str, Any] = Field(default_factory=dict)


class PatientCreate(PatientBase):
    pass


class PatientUpdate(BaseModel):
    first_name: str | None = Field(default=None, max_length=128)
    last_name: str | None = Field(default=None, max_length=128)
    date_of_birth: date | None = None
    gender: GenderEnum | None = None
    phone: str | None = None
    email: str | None = None
    extra_data: dict[str, Any] | None = None


class PatientOut(PatientBase):
    id: uuid.UUID
    created_by: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PatientSummary(BaseModel):
    """Lightweight response for list endpoints."""

    id: uuid.UUID
    mrn: str
    first_name: str
    last_name: str
    date_of_birth: date
    gender: GenderEnum
    created_at: datetime

    model_config = {"from_attributes": True}


# =============================================================================
#  DOCUMENT
# =============================================================================
class DocumentBase(BaseModel):
    title: str = Field(..., max_length=512)
    document_type: DocumentTypeEnum = DocumentTypeEnum.other
    extra_data: dict[str, Any] = Field(default_factory=dict)


class DocumentCreate(DocumentBase):
    patient_id: uuid.UUID
    storage_path: str
    file_size_bytes: int | None = None
    mime_type: str | None = None


class DocumentUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=512)
    document_type: DocumentTypeEnum | None = None
    status: DocumentStatusEnum | None = None
    raw_text: str | None = None
    page_count: int | None = None
    ai_model: str | None = None
    processed_at: datetime | None = None
    extra_data: dict[str, Any] | None = None


class DocumentOut(DocumentBase):
    id: uuid.UUID
    patient_id: uuid.UUID
    uploaded_by: uuid.UUID
    status: DocumentStatusEnum
    storage_path: str
    file_size_bytes: int | None
    mime_type: str | None
    raw_text: str | None
    page_count: int | None
    ai_model: str | None
    processed_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentSummary(BaseModel):
    """Lightweight response for list endpoints — omits raw_text."""

    id: uuid.UUID
    patient_id: uuid.UUID
    title: str
    document_type: DocumentTypeEnum
    status: DocumentStatusEnum
    mime_type: str | None
    file_size_bytes: int | None
    page_count: int | None
    processed_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class OcrPageResult(BaseModel):  # mirrors OcrService.PageResult
    page_number: int
    char_count: int
    duration_ms: float
    skew_angle: float
    text: str


class OcrResponse(BaseModel):
    """
    Combined response for the POST /documents/{id}/ocr endpoint.

    Contains the updated document record (with status=completed) plus
    the per-page OCR breakdown returned by Tesseract.
    """

    document: DocumentOut
    source_type: str
    page_count: int
    total_char_count: int
    total_duration_ms: float
    lang: str
    dpi: int
    pages: list[OcrPageResult]


# =============================================================================
#  EXTRACTED EVENT
# =============================================================================
class ExtractedEventBase(BaseModel):
    event_type: str = Field(..., max_length=64)
    event_date: date | None = None
    event_data: dict[str, Any] = Field(default_factory=dict)
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    ai_model: str | None = Field(default=None, max_length=128)
    notes: str | None = None

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        allowed = {
            "medication",
            "diagnosis",
            "procedure",
            "lab_result",
            "vital_sign",
            "allergy",
            "immunization",
            "clinical_note",
            "follow_up",
            "other",
        }
        if v not in allowed:
            raise ValueError(f"event_type must be one of {sorted(allowed)}")
        return v


class ExtractedEventCreate(ExtractedEventBase):
    document_id: uuid.UUID
    patient_id: uuid.UUID


class ManualEventCreate(ExtractedEventBase):
    """Schema for doctor-entered manual clinical events (no document upload needed).
    patient_id is provided as a URL path parameter — not required in body.
    """


class ExtractedEventUpdate(BaseModel):
    event_date: date | None = None
    event_data: dict[str, Any] | None = None
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    is_verified: bool | None = None
    reviewed_by: uuid.UUID | None = None
    notes: str | None = None


class ExtractedEventOut(ExtractedEventBase):
    id: uuid.UUID
    document_id: uuid.UUID | None
    patient_id: uuid.UUID
    is_verified: bool
    reviewed_by: uuid.UUID | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# =============================================================================
#  TIMELINE (read-only aggregation)
# =============================================================================
class TimelineEventGroup(BaseModel):
    """Events bucketed by event_type for sidebar / filter display."""

    event_type: str
    count: int


class PatientTimeline(BaseModel):
    """Paginated, ordered clinical timeline for a single patient."""

    patient_id: uuid.UUID

    # ── Pagination metadata ───────────────────────────────────────────────────
    total: int = Field(
        description="Total number of events matching the applied filters."
    )
    limit: int
    offset: int
    has_more: bool = Field(description="True when more pages are available.")

    # ── Optional aggregation ─────────────────────────────────────────────────
    event_type_counts: list[TimelineEventGroup] = Field(
        default_factory=list,
        description="Count of events per type for the *unfiltered* patient (useful for sidebar chips).",
    )

    # ── Events ───────────────────────────────────────────────────────────────
    events: list[ExtractedEventOut] = Field(
        description="Events sorted by event_date ASC NULLS LAST, then created_at ASC."
    )


# =============================================================================
#  LAB TRENDS (read-only time-series aggregation)
# =============================================================================
class LabTrendPoint(BaseModel):
    """
    A single data point in a lab-value trend series.

    ``value_raw`` is always the exact string stored in ``event_data.value``.
    ``numeric_value`` is a best-effort float parse — ``None`` when the stored
    value is non-numeric (e.g. "positive", "trace").
    """

    event_id: uuid.UUID
    document_id: uuid.UUID | None = None
    event_date: date | None = Field(
        description="Date of the lab result; null when the source document had no date."
    )
    value_raw: str = Field(description="Exact value string from event_data.")
    numeric_value: float | None = Field(
        default=None,
        description="Parsed numeric value for charting; null when non-numeric.",
    )
    unit: str | None = Field(
        default=None, description="Unit of measurement (e.g. '%', 'mg/dL')."
    )
    flag: str | None = Field(
        default=None,
        description="Interpretation flag: HIGH | LOW | CRITICAL | NORMAL | null.",
    )
    reference_range: str | None = Field(
        default=None, description="Normal range string (e.g. '4.0-5.6 %')."
    )
    is_verified: bool

    model_config = {"from_attributes": True}


class LabTrendResponse(BaseModel):
    """Time-series response for a single lab test across all visits."""

    patient_id: uuid.UUID
    test_name: str = Field(
        description="Normalised test name (lower-cased canonical form)."
    )
    common_unit: str | None = Field(
        default=None,
        description="Most frequent unit across all data points.",
    )
    total: int = Field(description="Total number of matching lab_result events.")
    points: list[LabTrendPoint] = Field(
        description="Data points sorted by event_date ASC NULLS LAST, then created_at ASC."
    )


# =============================================================================
#  LAB REPORT (all lab results at a glance)
# =============================================================================
class LabReportItem(BaseModel):
    """A single lab test result in the full lab report."""

    event_id: uuid.UUID
    document_id: uuid.UUID | None = None
    test_name: str
    value_raw: str
    numeric_value: float | None = None
    unit: str | None = None
    flag: str | None = Field(
        default=None,
        description="HIGH | LOW | CRITICAL | NORMAL | null",
    )
    reference_range: str | None = Field(
        default=None,
        description="Normal range (e.g. '70–100 mg/dL')",
    )
    category: str | None = Field(
        default=None,
        description="Test category (e.g. 'Hematology', 'Lipid Profile', 'Blood Sugar')",
    )
    event_date: date | None = None
    is_verified: bool = False
    created_at: datetime | None = None


class LabReportGroup(BaseModel):
    """A group of lab results sharing the same clinical category."""

    category: str
    items: list["LabReportItem"]
    abnormal_count: int = 0


class LabReportResponse(BaseModel):
    """Full lab report — all lab_result events for a patient, grouped by category."""

    patient_id: uuid.UUID
    total: int = Field(description="Total number of lab_result events.")
    items: list[LabReportItem] = Field(
        description="All lab results sorted by created_at DESC (most recent first)."
    )
    grouped_items: list[LabReportGroup] = Field(
        default_factory=list,
        description="Lab results grouped by clinical category for organised display.",
    )
    abnormal_count: int = Field(
        default=0,
        description="Number of items flagged HIGH, LOW, or CRITICAL.",
    )
    still_processing: bool = Field(
        default=False,
        description="True when one or more documents are still being processed (OCR / LLM). "
        "Clients should poll until this becomes False.",
    )


# =============================================================================
#  DOCTOR SUMMARY (AI-generated, event-based)
# =============================================================================
class DoctorSummaryResponse(BaseModel):
    """
    AI-generated physician-facing patient summary derived from the patient's
    chronological structured clinical events. Formatted as bullet points.
    """

    patient_id: uuid.UUID
    summary: str = Field(description="Full bulleted physician-facing summary text.")
    summary_points: list[str] = Field(
        default_factory=list,
        description="Parsed list of individual bullet-point statements for easy rendering.",
    )
    word_count: int
    model: str = Field(description="Ollama model used to generate the summary.")
    event_count: int = Field(description="Number of events included in the context.")
    cached: bool = Field(
        default=False,
        description="True when this response was served from the 24-hour cache.",
    )
    generated_at: datetime = Field(
        description="UTC timestamp when the summary was originally generated."
    )
    total_duration_ms: float | None = None
