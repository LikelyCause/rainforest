-- Migration 007: users and who did what.
-- session_version is part of every session cookie; bumping it (deactivation, role change) ends the user's sessions.

CREATE TABLE IF NOT EXISTS meta.users (
    id              BIGSERIAL   PRIMARY KEY,
    name            TEXT        NOT NULL,
    email           TEXT,
    role            TEXT        NOT NULL DEFAULT 'member' CHECK (role IN ('lead', 'member')),
    active          BOOLEAN     NOT NULL DEFAULT true,
    oidc_subject    TEXT        UNIQUE,
    session_version INTEGER     NOT NULL DEFAULT 1,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ
);

INSERT INTO meta.users (name, role) SELECT 'Admin', 'lead' WHERE NOT EXISTS (SELECT 1 FROM meta.users);

ALTER TABLE meta.decision_history ADD COLUMN IF NOT EXISTS user_id    BIGINT REFERENCES meta.users (id);
ALTER TABLE meta.bulk_batches     ADD COLUMN IF NOT EXISTS user_id    BIGINT REFERENCES meta.users (id);
ALTER TABLE meta.exports          ADD COLUMN IF NOT EXISTS created_by BIGINT REFERENCES meta.users (id);
ALTER TABLE meta.model_runs       ADD COLUMN IF NOT EXISTS created_by BIGINT REFERENCES meta.users (id);
ALTER TABLE meta.analyses         ADD COLUMN IF NOT EXISTS created_by BIGINT REFERENCES meta.users (id);
ALTER TABLE meta.datasets         ADD COLUMN IF NOT EXISTS created_by BIGINT REFERENCES meta.users (id);

-- Everything that happened before there were users was done by the one admin, now the first lead.
-- History written by background jobs (knock-on target changes) is the exception: user_id stays NULL there.
DO $$
DECLARE first_lead BIGINT := (SELECT id FROM meta.users WHERE role = 'lead' ORDER BY id LIMIT 1);
BEGIN
    UPDATE meta.decision_history SET user_id = first_lead WHERE user_id IS NULL;
    UPDATE meta.bulk_batches SET user_id = first_lead WHERE user_id IS NULL;
    UPDATE meta.exports SET created_by = first_lead WHERE created_by IS NULL;
    UPDATE meta.model_runs SET created_by = first_lead WHERE created_by IS NULL;
    UPDATE meta.analyses SET created_by = first_lead WHERE created_by IS NULL;
    UPDATE meta.datasets SET created_by = first_lead WHERE created_by IS NULL;
END $$;
