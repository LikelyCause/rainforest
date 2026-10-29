-- Migration 027: a deployment test's progress, as an export shows it; the values per field that didn't convert
-- ({field: {failed, values, flagged}}); what the upload reader noted; the field that identified each record; and the
-- labels its results are scored by, which name the results file's score columns.
ALTER TABLE meta.deployment_tests
    ADD COLUMN IF NOT EXISTS progress       REAL        NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS status_message TEXT,
    ADD COLUMN IF NOT EXISTS warnings       JSONB,
    ADD COLUMN IF NOT EXISTS notes          TEXT[]      NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS id_field       TEXT,
    ADD COLUMN IF NOT EXISTS labels         TEXT[],
    ADD COLUMN IF NOT EXISTS updated_at     TIMESTAMPTZ NOT NULL DEFAULT now();
