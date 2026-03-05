"""manual_events_nullable_document

Allows doctors to create clinical events without an associated document.
Makes extracted_events.document_id nullable so that manually-entered
prescriptions, diagnoses, and other events can exist independently.

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-03-04 00:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from alembic import op

# ── Revision identifiers ──────────────────────────────────────────────────────
revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Make document_id nullable — doctors can create manual events
    op.alter_column(
        "extracted_events",
        "document_id",
        existing_type=PG_UUID(as_uuid=True),
        nullable=True,
    )

    # Drop the existing CASCADE FK constraint and recreate allowing NULL rows
    op.drop_constraint(
        "extracted_events_document_id_fkey",
        "extracted_events",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "extracted_events_document_id_fkey",
        "extracted_events",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # RLS: allow doctors to INSERT manual events (no document_id)
    # The existing policy events_doctor_insert already covers this since
    # it only checks app_is_doctor() AND app_doctor_assigned_to(patient_id).
    # No additional policy change needed.


def downgrade() -> None:
    # Restore CASCADE FK
    op.drop_constraint(
        "extracted_events_document_id_fkey",
        "extracted_events",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "extracted_events_document_id_fkey",
        "extracted_events",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # Remove rows with NULL document_id before restoring NOT NULL
    op.execute("DELETE FROM extracted_events WHERE document_id IS NULL")

    op.alter_column(
        "extracted_events",
        "document_id",
        existing_type=PG_UUID(as_uuid=True),
        nullable=False,
    )
