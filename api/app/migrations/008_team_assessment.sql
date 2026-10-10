-- Migration 008: recipes (versioned decisions per dataset) and blind team assessment.
-- field_decisions / field_settings stay the team's working decision for the dataset's current recipe version.
-- Assessment rows are keyed by recipe version so a new version starts clean.

CREATE TABLE IF NOT EXISTS meta.recipes (
    id                 BIGSERIAL   PRIMARY KEY,
    dataset_id         BIGINT      NOT NULL REFERENCES meta.datasets (id) ON DELETE CASCADE,
    version            INT         NOT NULL,
    phase              TEXT        NOT NULL DEFAULT 'assessing' CHECK (phase IN ('assessing', 'review', 'finalized')),
    parent_id          BIGINT      REFERENCES meta.recipes (id),
    created_by         BIGINT      REFERENCES meta.users (id),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    finalized_by       BIGINT      REFERENCES meta.users (id),
    finalized_at       TIMESTAMPTZ,
    schema_fingerprint TEXT,
    snapshot           JSONB,
    UNIQUE (dataset_id, version)
);
CREATE UNIQUE INDEX IF NOT EXISTS recipes_one_open_idx ON meta.recipes (dataset_id) WHERE phase <> 'finalized';

-- A finalized snapshot never changes. The one allowed write is filling a missing snapshot (rows this migration
-- creates, which bootstrap snapshots in Python so there is one implementation of the format).
CREATE OR REPLACE FUNCTION meta.recipes_frozen() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.phase = 'finalized' AND OLD.snapshot IS NOT NULL THEN
        RAISE EXCEPTION 'recipe v% of dataset % is finalized and cannot change', OLD.version, OLD.dataset_id;
    END IF;
    RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS recipes_frozen ON meta.recipes;
CREATE TRIGGER recipes_frozen BEFORE UPDATE ON meta.recipes FOR EACH ROW EXECUTE FUNCTION meta.recipes_frozen();

CREATE TABLE IF NOT EXISTS meta.dataset_assessors (
    recipe_id    BIGINT      NOT NULL REFERENCES meta.recipes (id) ON DELETE CASCADE,
    user_id      BIGINT      NOT NULL REFERENCES meta.users (id),
    assigned_by  BIGINT      REFERENCES meta.users (id),
    assigned_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    submitted_at TIMESTAMPTZ,
    PRIMARY KEY (recipe_id, user_id)
);

CREATE TABLE IF NOT EXISTS meta.assessments (
    recipe_id     BIGINT      NOT NULL REFERENCES meta.recipes (id) ON DELETE CASCADE,
    field_id      BIGINT      NOT NULL REFERENCES meta.fields (id) ON DELETE CASCADE,
    user_id       BIGINT      NOT NULL REFERENCES meta.users (id),
    action        TEXT        NOT NULL CHECK (action IN ('keep', 'drop', 'target')),
    semantic_type TEXT        NOT NULL,
    notes         TEXT,
    reviewed      BOOLEAN     NOT NULL DEFAULT false,
    source        TEXT        NOT NULL DEFAULT 'suggested' CHECK (source IN ('suggested', 'user')),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (recipe_id, user_id, field_id)
);

CREATE TABLE IF NOT EXISTS meta.assessment_settings (
    recipe_id             BIGINT      NOT NULL REFERENCES meta.recipes (id) ON DELETE CASCADE,
    field_id              BIGINT      NOT NULL REFERENCES meta.fields (id) ON DELETE CASCADE,
    user_id               BIGINT      NOT NULL REFERENCES meta.users (id),
    family                TEXT        NOT NULL,
    imputation            TEXT        NOT NULL DEFAULT 'none',
    impute_value          TEXT,
    add_missing_indicator BOOLEAN     NOT NULL DEFAULT false,
    transform             TEXT        NOT NULL DEFAULT 'none',
    scaling               TEXT        NOT NULL DEFAULT 'none',
    encoding              TEXT        NOT NULL DEFAULT 'none',
    params                JSONB       NOT NULL DEFAULT '{}',
    source                TEXT        NOT NULL DEFAULT 'suggested' CHECK (source IN ('suggested', 'user')),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (recipe_id, user_id, field_id, family)
);

-- An assessor's bulk edits, for undo; kept apart from bulk_batches because the snapshot holds their choices.
CREATE TABLE IF NOT EXISTS meta.assessment_batches (
    id         UUID        PRIMARY KEY,
    recipe_id  BIGINT      NOT NULL REFERENCES meta.recipes (id) ON DELETE CASCADE,
    user_id    BIGINT      NOT NULL REFERENCES meta.users (id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    summary    JSONB       NOT NULL,
    snapshot   JSONB       NOT NULL,
    undone_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS meta.review_comments (
    id         BIGSERIAL   PRIMARY KEY,
    recipe_id  BIGINT      NOT NULL REFERENCES meta.recipes (id) ON DELETE CASCADE,
    field_id   BIGINT      NOT NULL REFERENCES meta.fields (id) ON DELETE CASCADE,
    user_id    BIGINT      NOT NULL REFERENCES meta.users (id),
    body       TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS review_comments_field_idx ON meta.review_comments (recipe_id, field_id, created_at);

ALTER TABLE meta.exports ADD COLUMN IF NOT EXISTS recipe_id BIGINT REFERENCES meta.recipes (id);

-- Explore results computed from one assessor's choices ("Kept only" while assessing) are theirs alone; the
-- console's role, which doesn't own the table, only sees team results.
ALTER TABLE meta.analyses ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS analyses_team_only ON meta.analyses;
CREATE POLICY analyses_team_only ON meta.analyses FOR SELECT
    USING (coalesce(params ->> 'decisions', 'team') = 'team');

-- Every existing dataset becomes recipe v1, finalized from its current decisions by the first lead.
DO $$
DECLARE first_lead BIGINT := (SELECT id FROM meta.users WHERE role = 'lead' ORDER BY id LIMIT 1);
BEGIN
    INSERT INTO meta.recipes (dataset_id, version, phase, created_by, finalized_by, finalized_at)
    SELECT d.id, 1, 'finalized', first_lead, first_lead, now() FROM meta.datasets d
     WHERE NOT EXISTS (SELECT 1 FROM meta.recipes r WHERE r.dataset_id = d.id);
    UPDATE meta.exports e SET recipe_id = r.id FROM meta.recipes r
     WHERE r.dataset_id = e.dataset_id AND r.version = 1 AND e.recipe_id IS NULL;
END $$;
