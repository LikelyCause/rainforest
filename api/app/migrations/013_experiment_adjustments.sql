-- Migration 013: each team experiment's own copy of its baseline, which its owner may adjust. Shaped like the
-- working tables; a field whose row matches the baseline snapshot isn't adjusted.
CREATE TABLE IF NOT EXISTS meta.experiment_decisions (
    experiment_id BIGINT      NOT NULL REFERENCES meta.experiments (id) ON DELETE CASCADE,
    field_id      BIGINT      NOT NULL REFERENCES meta.fields (id) ON DELETE CASCADE,
    action        TEXT        NOT NULL CHECK (action IN ('keep', 'drop', 'target')),
    semantic_type TEXT        NOT NULL,
    notes         TEXT,
    reviewed      BOOLEAN     NOT NULL DEFAULT true,
    source        TEXT        NOT NULL DEFAULT 'suggested' CHECK (source IN ('suggested', 'user')),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (experiment_id, field_id)
);

CREATE TABLE IF NOT EXISTS meta.experiment_settings (
    experiment_id         BIGINT      NOT NULL REFERENCES meta.experiments (id) ON DELETE CASCADE,
    field_id              BIGINT      NOT NULL REFERENCES meta.fields (id) ON DELETE CASCADE,
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
    PRIMARY KEY (experiment_id, field_id, family)
);

CREATE TABLE IF NOT EXISTS meta.experiment_batches (
    id            UUID        PRIMARY KEY,
    experiment_id BIGINT      NOT NULL REFERENCES meta.experiments (id) ON DELETE CASCADE,
    user_id       BIGINT      NOT NULL REFERENCES meta.users (id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    summary       JSONB       NOT NULL,
    snapshot      JSONB       NOT NULL,
    undone_at     TIMESTAMPTZ
);

-- Deleting a dataset's fields cascades here; these keep that from scanning every experiment's rows.
CREATE INDEX IF NOT EXISTS experiment_decisions_field_idx ON meta.experiment_decisions (field_id);
CREATE INDEX IF NOT EXISTS experiment_settings_field_idx ON meta.experiment_settings (field_id);

-- The experiment's targets in chain order (its copy's target fields, ordered).
ALTER TABLE meta.experiments ADD COLUMN IF NOT EXISTS target_order TEXT[];
