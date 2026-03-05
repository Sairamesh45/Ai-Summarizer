-- =============================================================================
--  Medical Document AI System — PostgreSQL Schema
--  Target: Supabase (PostgreSQL 15+)
-- =============================================================================

-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";        -- trigram for text search on names

-- ── Enums ─────────────────────────────────────────────────────────────────────
CREATE TYPE document_type_enum AS ENUM (
    'lab_report',
    'discharge_summary',
    'clinical_note',
    'imaging',
    'prescription',
    'other'
);

CREATE TYPE document_status_enum AS ENUM (
    'pending',
    'processing',
    'completed',
    'failed'
);

CREATE TYPE gender_enum AS ENUM (
    'male',
    'female',
    'other',
    'unknown'
);

-- =============================================================================
--  PATIENTS
-- =============================================================================
CREATE TABLE patients (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_by      UUID            NOT NULL
                                    REFERENCES auth.users(id)   -- Supabase auth
                                    ON DELETE RESTRICT,

    mrn             VARCHAR(64)     NOT NULL,                   -- medical record number
    first_name      VARCHAR(128)    NOT NULL,
    last_name       VARCHAR(128)    NOT NULL,
    date_of_birth   DATE            NOT NULL,
    gender          gender_enum     NOT NULL DEFAULT 'unknown',

    phone           VARCHAR(32),
    email           VARCHAR(255),

    -- extra demographics / custom fields (filled by AI or operator)
    metadata        JSONB           NOT NULL DEFAULT '{}',

    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_patients_mrn UNIQUE (mrn)
);

COMMENT ON TABLE  patients             IS 'Core patient registry';
COMMENT ON COLUMN patients.mrn         IS 'Unique medical record number within the tenant';
COMMENT ON COLUMN patients.metadata    IS 'Arbitrary JSONB bag: insurance, address, custom fields';

-- ── Indexes: patients ─────────────────────────────────────────────────────────
CREATE INDEX idx_patients_mrn          ON patients (mrn);
CREATE INDEX idx_patients_created_by   ON patients (created_by);
CREATE INDEX idx_patients_dob          ON patients (date_of_birth);
CREATE INDEX idx_patients_last_name    ON patients USING gin (last_name gin_trgm_ops);
CREATE INDEX idx_patients_metadata_gin ON patients USING gin (metadata jsonb_path_ops);

-- ── Auto-update updated_at ────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_patients_updated_at
    BEFORE UPDATE ON patients
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =============================================================================
--  DOCUMENTS
-- =============================================================================
CREATE TABLE documents (
    id              UUID                    PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id      UUID                    NOT NULL
                                            REFERENCES patients(id)
                                            ON DELETE CASCADE,
    uploaded_by     UUID                    NOT NULL
                                            REFERENCES auth.users(id)
                                            ON DELETE RESTRICT,

    title           VARCHAR(512)            NOT NULL,
    document_type   document_type_enum      NOT NULL DEFAULT 'other',
    status          document_status_enum    NOT NULL DEFAULT 'pending',

    -- Supabase Storage
    storage_path    TEXT                    NOT NULL,           -- bucket/folder/uuid.ext
    file_size_bytes BIGINT,
    mime_type       VARCHAR(128),

    -- AI processing
    raw_text        TEXT,                                       -- full extracted text
    page_count      INTEGER,
    ai_model        VARCHAR(128),                               -- model version used

    -- arbitrary per-document metadata
    metadata        JSONB                   NOT NULL DEFAULT '{}',

    created_at      TIMESTAMPTZ             NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ             NOT NULL DEFAULT NOW(),
    processed_at    TIMESTAMPTZ                                 -- when AI finished
);

COMMENT ON TABLE  documents               IS 'Uploaded medical documents tied to a patient';
COMMENT ON COLUMN documents.storage_path  IS 'Path inside Supabase Storage bucket';
COMMENT ON COLUMN documents.raw_text      IS 'Full OCR / parsed text for downstream NLP';
COMMENT ON COLUMN documents.metadata      IS 'Extra fields: source system, checksums, etc.';

-- ── Indexes: documents ────────────────────────────────────────────────────────
-- Primary access pattern: all docs for a patient, newest first
CREATE INDEX idx_documents_patient_created
    ON documents (patient_id, created_at DESC);

CREATE INDEX idx_documents_status
    ON documents (status)
    WHERE status IN ('pending', 'processing');  -- partial: only unfinished docs

CREATE INDEX idx_documents_type
    ON documents (document_type);

CREATE INDEX idx_documents_uploaded_by
    ON documents (uploaded_by);

CREATE INDEX idx_documents_metadata_gin
    ON documents USING gin (metadata jsonb_path_ops);

CREATE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =============================================================================
--  EXTRACTED_EVENTS
-- =============================================================================
CREATE TABLE extracted_events (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Direct FK to both patient and document for fast patient-level queries
    document_id     UUID            NOT NULL
                                    REFERENCES documents(id)
                                    ON DELETE CASCADE,
    patient_id      UUID            NOT NULL
                                    REFERENCES patients(id)
                                    ON DELETE CASCADE,

    -- human reviewer, nullable
    reviewed_by     UUID
                                    REFERENCES auth.users(id)
                                    ON DELETE SET NULL,

    event_type      VARCHAR(64)     NOT NULL,
    -- Examples: medication, diagnosis, procedure, lab_result,
    --           vital_sign, allergy, immunization, follow_up

    event_date      DATE,           -- when the clinical event occurred (not upload date)
    event_data      JSONB           NOT NULL DEFAULT '{}',
    -- Structured payload varies per event_type, e.g.:
    --   medication  → { name, dose, frequency, route }
    --   lab_result  → { test_name, value, unit, reference_range, flag }
    --   diagnosis   → { code, system, description, status }

    confidence_score NUMERIC(5,4)   CHECK (confidence_score BETWEEN 0 AND 1),
    ai_model        VARCHAR(128),
    is_verified     BOOLEAN         NOT NULL DEFAULT FALSE,
    notes           TEXT,           -- reviewer free-text

    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE  extracted_events              IS 'AI-extracted clinical events from documents';
COMMENT ON COLUMN extracted_events.event_type   IS 'medication|diagnosis|procedure|lab_result|vital_sign|allergy|immunization|follow_up';
COMMENT ON COLUMN extracted_events.event_date   IS 'Clinical date of the event, used for timeline queries';
COMMENT ON COLUMN extracted_events.event_data   IS 'Structured JSONB payload specific to event_type';
COMMENT ON COLUMN extracted_events.confidence_score IS '0.0–1.0 AI confidence';

-- ── Indexes: extracted_events ─────────────────────────────────────────────────
-- ① Chronological timeline per patient — the most frequent query
CREATE INDEX idx_events_patient_timeline
    ON extracted_events (patient_id, event_date DESC NULLS LAST);

-- ② All events from a single document
CREATE INDEX idx_events_document
    ON extracted_events (document_id);

-- ③ Filter by event type within a patient
CREATE INDEX idx_events_patient_type
    ON extracted_events (patient_id, event_type);

-- ④ Unverified events queue
CREATE INDEX idx_events_unverified
    ON extracted_events (is_verified, created_at DESC)
    WHERE is_verified = FALSE;

-- ⑤ GIN on event_data for arbitrary JSONB queries  (e.g. search by drug name)
CREATE INDEX idx_events_data_gin
    ON extracted_events USING gin (event_data jsonb_path_ops);

-- ⑥ Reviewer workload
CREATE INDEX idx_events_reviewed_by
    ON extracted_events (reviewed_by)
    WHERE reviewed_by IS NOT NULL;

CREATE TRIGGER trg_events_updated_at
    BEFORE UPDATE ON extracted_events
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =============================================================================
--  ROW-LEVEL SECURITY (Supabase)
-- =============================================================================

-- ── Schema additions ──────────────────────────────────────────────────────────

-- Links a patients row to a Supabase auth account (patient portal login).
-- NULL for patients who have no portal access yet.
ALTER TABLE patients
    ADD COLUMN IF NOT EXISTS auth_user_id UUID
        REFERENCES auth.users(id) ON DELETE SET NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_patients_auth_user_id
    ON patients (auth_user_id)
    WHERE auth_user_id IS NOT NULL;

-- Doctor–Patient assignment junction (many-to-many).
-- Controls which doctors can see which patients.
CREATE TABLE IF NOT EXISTS doctor_patient_assignments (
    doctor_id   UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    patient_id  UUID        NOT NULL REFERENCES patients(id)   ON DELETE CASCADE,
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    assigned_by UUID                 REFERENCES auth.users(id) ON DELETE SET NULL,
    notes       TEXT,
    PRIMARY KEY (doctor_id, patient_id)
);

CREATE INDEX IF NOT EXISTS idx_dpa_doctor_id  ON doctor_patient_assignments (doctor_id);
CREATE INDEX IF NOT EXISTS idx_dpa_patient_id ON doctor_patient_assignments (patient_id);

ALTER TABLE doctor_patient_assignments ENABLE ROW LEVEL SECURITY;

-- ── Enable RLS on core tables ─────────────────────────────────────────────────

ALTER TABLE patients         ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents        ENABLE ROW LEVEL SECURITY;
ALTER TABLE extracted_events ENABLE ROW LEVEL SECURITY;

-- ── Helper functions (SECURITY DEFINER) ──────────────────────────────────────
-- Execute as the postgres superuser, so they can query tables without hitting
-- their own RLS and avoid policy recursion.

CREATE OR REPLACE FUNCTION public.app_is_admin()
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER AS $$
    SELECT (auth.jwt() -> 'app_metadata' ->> 'app_role') = 'admin'
$$;

CREATE OR REPLACE FUNCTION public.app_is_doctor()
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER AS $$
    SELECT (auth.jwt() -> 'app_metadata' ->> 'app_role') = 'doctor'
$$;

CREATE OR REPLACE FUNCTION public.app_is_patient()
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER AS $$
    SELECT (auth.jwt() -> 'app_metadata' ->> 'app_role') = 'patient'
$$;

-- Extracts the patient_id claim embedded in patient-portal JWTs.
CREATE OR REPLACE FUNCTION public.app_jwt_patient_id()
RETURNS UUID LANGUAGE sql STABLE SECURITY DEFINER AS $$
    SELECT (auth.jwt() -> 'app_metadata' ->> 'patient_id')::UUID
$$;

-- Returns TRUE when the calling doctor is assigned to the given patient.
-- SECURITY DEFINER prevents the assignment-table RLS from blocking the sub-query.
CREATE OR REPLACE FUNCTION public.app_doctor_assigned_to(p_patient_id UUID)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER AS $$
    SELECT EXISTS (
        SELECT 1 FROM doctor_patient_assignments
        WHERE doctor_id  = auth.uid()
          AND patient_id = p_patient_id
    )
$$;

-- ── Custom Access Token Hook ──────────────────────────────────────────────────
-- Injects app_role (and patient_id for portal accounts) into the JWT at
-- sign-in time.
-- Register in: Supabase Dashboard → Authentication → Hooks → Custom Access Token

CREATE OR REPLACE FUNCTION public.custom_access_token_hook(event JSONB)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
    v_user_id    UUID := (event ->> 'user_id')::UUID;
    v_app_role   TEXT;
    v_patient_id UUID;
BEGIN
    -- Resolve role from application users table (doctor | admin | staff).
    -- Falls back to 'patient' for portal accounts with no users row.
    SELECT role::TEXT INTO v_app_role
    FROM public.users
    WHERE id = v_user_id;

    v_app_role := COALESCE(v_app_role, 'patient');

    -- Embed the resolved role into JWT app_metadata.
    event := jsonb_set(
        event,
        '{claims,app_metadata,app_role}',
        to_jsonb(v_app_role)
    );

    -- For patient-portal accounts, also embed their patients.id so RLS
    -- policies can compare against patient_id columns without a join.
    IF v_app_role = 'patient' THEN
        SELECT id INTO v_patient_id
        FROM public.patients
        WHERE auth_user_id = v_user_id;

        IF v_patient_id IS NOT NULL THEN
            event := jsonb_set(
                event,
                '{claims,app_metadata,patient_id}',
                to_jsonb(v_patient_id::TEXT)
            );
        END IF;
    END IF;

    RETURN event;
END;
$$;

-- Only supabase_auth_admin may call the hook; revoke from everyone else.
GRANT  EXECUTE ON FUNCTION public.custom_access_token_hook TO supabase_auth_admin;
REVOKE EXECUTE ON FUNCTION public.custom_access_token_hook FROM PUBLIC, authenticated, anon;

-- ── PATIENTS policies ─────────────────────────────────────────────────────────

-- Admin: unrestricted
CREATE POLICY patients_admin ON patients
    FOR ALL TO authenticated
    USING      (public.app_is_admin())
    WITH CHECK (public.app_is_admin());

-- Doctor: INSERT freely (creates a new patient, then gets assigned);
-- SELECT / UPDATE only on assigned patients; never DELETE.
CREATE POLICY patients_doctor_insert ON patients
    FOR INSERT TO authenticated
    WITH CHECK (public.app_is_doctor());

CREATE POLICY patients_doctor_read ON patients
    FOR SELECT TO authenticated
    USING (public.app_is_doctor() AND public.app_doctor_assigned_to(id));

CREATE POLICY patients_doctor_update ON patients
    FOR UPDATE TO authenticated
    USING      (public.app_is_doctor() AND public.app_doctor_assigned_to(id))
    WITH CHECK (public.app_is_doctor() AND public.app_doctor_assigned_to(id));

-- Patient portal: read own row only
CREATE POLICY patients_self_read ON patients
    FOR SELECT TO authenticated
    USING (public.app_is_patient() AND auth_user_id = auth.uid());

-- ── DOCUMENTS policies ────────────────────────────────────────────────────────

-- Admin: unrestricted
CREATE POLICY documents_admin ON documents
    FOR ALL TO authenticated
    USING      (public.app_is_admin())
    WITH CHECK (public.app_is_admin());

-- Doctor: full CRUD on documents for assigned patients
CREATE POLICY documents_doctor_read ON documents
    FOR SELECT TO authenticated
    USING (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id));

CREATE POLICY documents_doctor_insert ON documents
    FOR INSERT TO authenticated
    WITH CHECK (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id));

CREATE POLICY documents_doctor_update ON documents
    FOR UPDATE TO authenticated
    USING      (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id))
    WITH CHECK (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id));

CREATE POLICY documents_doctor_delete ON documents
    FOR DELETE TO authenticated
    USING (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id));

-- Patient portal: read-only access to their own documents
CREATE POLICY documents_self_read ON documents
    FOR SELECT TO authenticated
    USING (public.app_is_patient() AND patient_id = public.app_jwt_patient_id());

-- ── EXTRACTED_EVENTS policies ─────────────────────────────────────────────────

-- Admin: unrestricted
CREATE POLICY events_admin ON extracted_events
    FOR ALL TO authenticated
    USING      (public.app_is_admin())
    WITH CHECK (public.app_is_admin());

-- Doctor: full CRUD for assigned patients
CREATE POLICY events_doctor_read ON extracted_events
    FOR SELECT TO authenticated
    USING (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id));

CREATE POLICY events_doctor_insert ON extracted_events
    FOR INSERT TO authenticated
    WITH CHECK (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id));

CREATE POLICY events_doctor_update ON extracted_events
    FOR UPDATE TO authenticated
    USING      (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id))
    WITH CHECK (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id));

CREATE POLICY events_doctor_delete ON extracted_events
    FOR DELETE TO authenticated
    USING (public.app_is_doctor() AND public.app_doctor_assigned_to(patient_id));

-- Patient portal: read-only, own events
CREATE POLICY events_self_read ON extracted_events
    FOR SELECT TO authenticated
    USING (public.app_is_patient() AND patient_id = public.app_jwt_patient_id());

-- ── DOCTOR_PATIENT_ASSIGNMENTS policies ───────────────────────────────────────

-- Admin: full control (assign / unassign doctors)
CREATE POLICY dpa_admin ON doctor_patient_assignments
    FOR ALL TO authenticated
    USING      (public.app_is_admin())
    WITH CHECK (public.app_is_admin());

-- Doctor: read their own assignment rows (enumerates accessible patients)
CREATE POLICY dpa_doctor_read ON doctor_patient_assignments
    FOR SELECT TO authenticated
    USING (public.app_is_doctor() AND doctor_id = auth.uid());

-- Patients have no access to assignment metadata.
