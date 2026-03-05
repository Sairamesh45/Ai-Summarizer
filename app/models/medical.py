import enum
import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


# ── Enums ─────────────────────────────────────────────────────────────────────
class GenderEnum(str, enum.Enum):
    male = "male"
    female = "female"
    other = "other"
    unknown = "unknown"


class DocumentTypeEnum(str, enum.Enum):
    lab_report = "lab_report"
    discharge_summary = "discharge_summary"
    clinical_note = "clinical_note"
    imaging = "imaging"
    prescription = "prescription"
    other = "other"


class DocumentStatusEnum(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


# =============================================================================
#  PATIENT
# =============================================================================
class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # FK → users.id (Supabase auth table lives in auth schema, referenced as string)
    created_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    mrn: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    first_name: Mapped[str] = mapped_column(String(128), nullable=False)
    last_name: Mapped[str] = mapped_column(String(128), nullable=False)
    date_of_birth: Mapped[date] = mapped_column(Date, nullable=False)
    gender: Mapped[GenderEnum] = mapped_column(
        Enum(GenderEnum, name="gender_enum"),
        nullable=False,
        default=GenderEnum.unknown,
        server_default=GenderEnum.unknown.value,
    )

    # Links this patient row to a Supabase auth account (patient portal login).
    # NULL for patients who have no portal access yet.
    auth_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        unique=True,
        index=True,
    )

    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    extra_data: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default="{}"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    documents: Mapped[list["Document"]] = relationship(
        "Document",
        back_populates="patient",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    extracted_events: Mapped[list["ExtractedEvent"]] = relationship(
        "ExtractedEvent",
        back_populates="patient",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # ── Composite / expression indexes (declared here, Alembic picks them up) ─
    __table_args__ = (
        Index(
            "idx_patients_metadata_gin",
            "metadata",
            postgresql_using="gin",
            postgresql_ops={"metadata": "jsonb_path_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<Patient id={self.id} mrn={self.mrn}>"


# =============================================================================
#  DOCUMENT
# =============================================================================
class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
    )
    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )

    title: Mapped[str] = mapped_column(String(512), nullable=False)
    document_type: Mapped[DocumentTypeEnum] = mapped_column(
        Enum(DocumentTypeEnum, name="document_type_enum"),
        nullable=False,
        default=DocumentTypeEnum.other,
        server_default=DocumentTypeEnum.other.value,
    )
    status: Mapped[DocumentStatusEnum] = mapped_column(
        Enum(DocumentStatusEnum, name="document_status_enum"),
        nullable=False,
        default=DocumentStatusEnum.pending,
        server_default=DocumentStatusEnum.pending.value,
    )

    # Supabase Storage
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # AI processing output
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ai_model: Mapped[str | None] = mapped_column(String(128), nullable=True)

    extra_data: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default="{}"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    patient: Mapped["Patient"] = relationship("Patient", back_populates="documents")
    extracted_events: Mapped[list["ExtractedEvent"]] = relationship(
        "ExtractedEvent",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        # Chronological list of docs per patient — most common read
        Index("idx_documents_patient_created", "patient_id", "created_at"),
        # Partial index — only pending/processing rows (small, hot set)
        Index(
            "idx_documents_status_active",
            "status",
            postgresql_where="status IN ('pending', 'processing')",
        ),
        Index("idx_documents_type", "document_type"),
        Index(
            "idx_documents_metadata_gin",
            "metadata",
            postgresql_using="gin",
            postgresql_ops={"metadata": "jsonb_path_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} type={self.document_type} status={self.status}>"


# =============================================================================
#  EXTRACTED EVENT
# =============================================================================
class ExtractedEvent(Base):
    __tablename__ = "extracted_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
    )
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    event_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Structured payload — shape depends on event_type
    # medication  → { name, dose, frequency, route }
    # lab_result  → { test_name, value, unit, reference_range, flag }
    # diagnosis   → { code, system, description, status }
    # vital_sign  → { type, value, unit }
    event_data: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )

    confidence_score: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    ai_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    patient: Mapped["Patient"] = relationship(
        "Patient", back_populates="extracted_events"
    )
    document: Mapped["Document"] = relationship(
        "Document", back_populates="extracted_events"
    )

    __table_args__ = (
        # ① Patient timeline — primary read path
        Index("idx_events_patient_timeline", "patient_id", "event_date"),
        # ② All events from one document
        Index("idx_events_document", "document_id"),
        # ③ Filter by type within patient
        Index("idx_events_patient_type", "patient_id", "event_type"),
        # ④ Unverified events queue (partial)
        Index(
            "idx_events_unverified",
            "is_verified",
            "created_at",
            postgresql_where="is_verified = false",
        ),
        # ⑤ JSONB search inside event_data
        Index(
            "idx_events_data_gin",
            "event_data",
            postgresql_using="gin",
            postgresql_ops={"event_data": "jsonb_path_ops"},
        ),
        # ⑥ Case-insensitive test_name lookup — partial on lab_result rows only
        #    Accelerates: lower(event_data->>'test_name') = lower(:name)
        #    Generated SQL: CREATE INDEX … ON extracted_events
        #      (lower(event_data->>'test_name'))
        #      WHERE event_type = 'lab_result'
        Index(
            "idx_events_lab_testname_ci",
            text("lower(event_data->>'test_name')"),
            postgresql_where=text("event_type = 'lab_result'"),
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ExtractedEvent id={self.id} type={self.event_type} "
            f"date={self.event_date} verified={self.is_verified}>"
        )
