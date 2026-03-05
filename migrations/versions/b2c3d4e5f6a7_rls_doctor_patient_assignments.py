"""rls_doctor_patient_assignments

Adds:
- patients.auth_user_id  - links a patient row to a Supabase portal login
- doctor_patient_assignments  - doctor-patient junction table
- public.app_is_admin / app_is_doctor / app_is_patient / app_jwt_patient_id /
  app_doctor_assigned_to  - SECURITY DEFINER helper functions
- public.custom_access_token_hook  - injects app_role + patient_id into JWT
- Full per-operation RLS policies replacing the initial placeholder policies

NOTE 1: Every _x() call contains exactly ONE SQL statement.
        asyncpg raises "cannot insert multiple commands into a prepared statement"
        when multiple semicolons appear in one call.

NOTE 2: Helper functions live in the public schema, not auth schema.
        The auth schema is Supabase-internal and rejects CREATE from external
        connections (InsufficientPrivilegeError).

Revision ID: b2c3d4e5f6a7
Revises: a5080d0be2ee
Create Date: 2026-03-03 00:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "a5080d0be2ee"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _x(sql: str) -> None:
    """Execute a single DDL/DML statement, bypassing asyncpg prepared-stmt limit."""
    op.execute(sa.text(sql))


def upgrade() -> None:
    # 1. patients.auth_user_id
    op.add_column("patients", sa.Column("auth_user_id", sa.UUID(), nullable=True))
    op.create_unique_constraint(
        "uq_patients_auth_user_id", "patients", ["auth_user_id"]
    )
    op.create_index(
        "idx_patients_auth_user_id",
        "patients",
        ["auth_user_id"],
        unique=True,
        postgresql_where=sa.text("auth_user_id IS NOT NULL"),
    )

    # 2. doctor_patient_assignments
    op.create_table(
        "doctor_patient_assignments",
        sa.Column("doctor_id", sa.UUID(), nullable=False),
        sa.Column("patient_id", sa.UUID(), nullable=False),
        sa.Column(
            "assigned_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("assigned_by", sa.UUID(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["doctor_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["patient_id"], ["patients.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["assigned_by"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("doctor_id", "patient_id"),
    )
    op.create_index("idx_dpa_doctor_id", "doctor_patient_assignments", ["doctor_id"])
    op.create_index("idx_dpa_patient_id", "doctor_patient_assignments", ["patient_id"])

    # 3. Enable RLS on the new table
    _x("ALTER TABLE doctor_patient_assignments ENABLE ROW LEVEL SECURITY")

    # 4. Helper functions in PUBLIC schema (auth schema is Supabase-internal)
    _x(
        """CREATE OR REPLACE FUNCTION public.app_is_admin()
        RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER AS $$
            SELECT (auth.jwt() -> 'app_metadata' ->> 'app_role') = 'admin'
        $$"""
    )

    _x(
        """CREATE OR REPLACE FUNCTION public.app_is_doctor()
        RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER AS $$
            SELECT (auth.jwt() -> 'app_metadata' ->> 'app_role') = 'doctor'
        $$"""
    )

    _x(
        """CREATE OR REPLACE FUNCTION public.app_is_patient()
        RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER AS $$
            SELECT (auth.jwt() -> 'app_metadata' ->> 'app_role') = 'patient'
        $$"""
    )

    _x(
        """CREATE OR REPLACE FUNCTION public.app_jwt_patient_id()
        RETURNS UUID LANGUAGE sql STABLE SECURITY DEFINER AS $$
            SELECT (auth.jwt() -> 'app_metadata' ->> 'patient_id')::UUID
        $$"""
    )

    _x(
        """CREATE OR REPLACE FUNCTION public.app_doctor_assigned_to(p_patient_id UUID)
        RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER AS $$
            SELECT EXISTS (
                SELECT 1 FROM doctor_patient_assignments
                WHERE doctor_id  = auth.uid()
                  AND patient_id = p_patient_id
            )
        $$"""
    )

    # 5. Custom Access Token Hook
    _x(
        """CREATE OR REPLACE FUNCTION public.custom_access_token_hook(event JSONB)
        RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER AS $$
        DECLARE
            v_user_id    UUID := (event ->> 'user_id')::UUID;
            v_app_role   TEXT;
            v_patient_id UUID;
        BEGIN
            SELECT role::TEXT INTO v_app_role FROM public.users WHERE id = v_user_id;
            v_app_role := COALESCE(v_app_role, 'patient');
            event := jsonb_set(event, '{claims,app_metadata,app_role}', to_jsonb(v_app_role));
            IF v_app_role = 'patient' THEN
                SELECT id INTO v_patient_id FROM public.patients WHERE auth_user_id = v_user_id;
                IF v_patient_id IS NOT NULL THEN
                    event := jsonb_set(event, '{claims,app_metadata,patient_id}', to_jsonb(v_patient_id::TEXT));
                END IF;
            END IF;
            RETURN event;
        END;
        $$"""
    )

    _x(
        "GRANT EXECUTE ON FUNCTION public.custom_access_token_hook TO supabase_auth_admin"
    )
    _x("REVOKE EXECUTE ON FUNCTION public.custom_access_token_hook FROM PUBLIC")

    # 6. Drop old placeholder policies
    _x("DROP POLICY IF EXISTS patients_created_by ON patients")
    _x("DROP POLICY IF EXISTS documents_via_patient ON documents")
    _x("DROP POLICY IF EXISTS events_via_patient ON extracted_events")

    # 7. PATIENTS policies
    _x(
        "CREATE POLICY patients_admin ON patients FOR ALL TO authenticated USING (public.app_is_admin()) WITH CHECK (public.app_is_admin())"
    )
    _x(
        "CREATE POLICY patients_doctor_insert ON patients FOR INSERT TO authenticated WITH CHECK (public.app_is_doctor())"
    )
    _x(
        "CREATE POLICY patients_doctor_read ON patients FOR SELECT TO authenticated USING (public.app_is_doctor() AND public.app_doctor_assigned_to(id))"
    )
    _x(
        "CREATE POLICY patients_doctor_update ON patients FOR UPDATE TO authenticated USING (public.app_is_doctor() AND public.app_doctor_assigned_to(id)) WITH CHECK (public.app_is_doctor() AND public.app_doctor_assigned_to(id))"
    )
    _x(
        "CREATE POLICY patients_self_read ON patients FOR SELECT TO authenticated USING (public.app_is_patient() AND auth_user_id = auth.uid())"
    )

    # 8. DOCUMENTS policies
    _x(
        "CREATE POLICY documents_admin ON documents FOR ALL TO authenticated USING (public.app_is_admin()) WITH CHECK (public.app_is_admin())"
    )
    _x(
        "CREATE POLICY documents_doctor_read ON documents FOR SELECT TO authenticated USING (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id))"
    )
    _x(
        "CREATE POLICY documents_doctor_insert ON documents FOR INSERT TO authenticated WITH CHECK (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id))"
    )
    _x(
        "CREATE POLICY documents_doctor_update ON documents FOR UPDATE TO authenticated USING (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id)) WITH CHECK (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id))"
    )
    _x(
        "CREATE POLICY documents_doctor_delete ON documents FOR DELETE TO authenticated USING (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id))"
    )
    _x(
        "CREATE POLICY documents_self_read ON documents FOR SELECT TO authenticated USING (public.app_is_patient() AND patient_id = public.app_jwt_patient_id())"
    )

    # 9. EXTRACTED_EVENTS policies
    _x(
        "CREATE POLICY events_admin ON extracted_events FOR ALL TO authenticated USING (public.app_is_admin()) WITH CHECK (public.app_is_admin())"
    )
    _x(
        "CREATE POLICY events_doctor_read ON extracted_events FOR SELECT TO authenticated USING (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id))"
    )
    _x(
        "CREATE POLICY events_doctor_insert ON extracted_events FOR INSERT TO authenticated WITH CHECK (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id))"
    )
    _x(
        "CREATE POLICY events_doctor_update ON extracted_events FOR UPDATE TO authenticated USING (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id)) WITH CHECK (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id))"
    )
    _x(
        "CREATE POLICY events_doctor_delete ON extracted_events FOR DELETE TO authenticated USING (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id))"
    )
    _x(
        "CREATE POLICY events_self_read ON extracted_events FOR SELECT TO authenticated USING (public.app_is_patient() AND patient_id = public.app_jwt_patient_id())"
    )

    # 10. DOCTOR_PATIENT_ASSIGNMENTS policies
    _x(
        "CREATE POLICY dpa_admin ON doctor_patient_assignments FOR ALL TO authenticated USING (public.app_is_admin()) WITH CHECK (public.app_is_admin())"
    )
    _x(
        "CREATE POLICY dpa_doctor_read ON doctor_patient_assignments FOR SELECT TO authenticated USING (public.app_is_doctor() AND doctor_id = auth.uid())"
    )


def downgrade() -> None:
    for policy, table in [
        ("patients_admin", "patients"),
        ("patients_doctor_insert", "patients"),
        ("patients_doctor_read", "patients"),
        ("patients_doctor_update", "patients"),
        ("patients_self_read", "patients"),
        ("documents_admin", "documents"),
        ("documents_doctor_read", "documents"),
        ("documents_doctor_insert", "documents"),
        ("documents_doctor_update", "documents"),
        ("documents_doctor_delete", "documents"),
        ("documents_self_read", "documents"),
        ("events_admin", "extracted_events"),
        ("events_doctor_read", "extracted_events"),
        ("events_doctor_insert", "extracted_events"),
        ("events_doctor_update", "extracted_events"),
        ("events_doctor_delete", "extracted_events"),
        ("events_self_read", "extracted_events"),
        ("dpa_admin", "doctor_patient_assignments"),
        ("dpa_doctor_read", "doctor_patient_assignments"),
    ]:
        _x(f"DROP POLICY IF EXISTS {policy} ON {table}")

    _x(
        "CREATE POLICY patients_created_by ON patients FOR ALL TO authenticated USING (created_by = auth.uid())"
    )
    _x(
        "CREATE POLICY documents_via_patient ON documents FOR ALL TO authenticated USING (patient_id IN (SELECT id FROM patients WHERE created_by = auth.uid()))"
    )
    _x(
        "CREATE POLICY events_via_patient ON extracted_events FOR ALL TO authenticated USING (patient_id IN (SELECT id FROM patients WHERE created_by = auth.uid()))"
    )

    _x("DROP FUNCTION IF EXISTS public.custom_access_token_hook(JSONB)")
    _x("DROP FUNCTION IF EXISTS public.app_doctor_assigned_to(UUID)")
    _x("DROP FUNCTION IF EXISTS public.app_jwt_patient_id()")
    _x("DROP FUNCTION IF EXISTS public.app_is_patient()")
    _x("DROP FUNCTION IF EXISTS public.app_is_doctor()")
    _x("DROP FUNCTION IF EXISTS public.app_is_admin()")

    _x("ALTER TABLE doctor_patient_assignments DISABLE ROW LEVEL SECURITY")
    op.drop_index("idx_dpa_patient_id", table_name="doctor_patient_assignments")
    op.drop_index("idx_dpa_doctor_id", table_name="doctor_patient_assignments")
    op.drop_table("doctor_patient_assignments")

    op.drop_index("idx_patients_auth_user_id", table_name="patients")
    op.drop_constraint("uq_patients_auth_user_id", "patients", type_="unique")
    op.drop_column("patients", "auth_user_id")
