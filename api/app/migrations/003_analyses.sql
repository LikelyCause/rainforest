-- Migration 003: cached Explore analyses (2-D map, feature tests, importance), one job row each.
-- data_version ties a result to the decisions it was computed from, so edits make it stale.

CREATE TABLE IF NOT EXISTS meta.analyses (
    id             BIGSERIAL   PRIMARY KEY,
    dataset_id     BIGINT      NOT NULL REFERENCES meta.datasets (id) ON DELETE CASCADE,
    kind           TEXT        NOT NULL CHECK (kind IN ('map', 'tests', 'importance')),
    params         JSONB       NOT NULL DEFAULT '{}',
    params_key     TEXT        NOT NULL,
    data_version   TEXT        NOT NULL,
    status         TEXT        NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending', 'running', 'done', 'failed')),
    progress       REAL        NOT NULL DEFAULT 0,
    status_message TEXT,
    error          TEXT,
    result         JSONB,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS analyses_lookup_idx ON meta.analyses (dataset_id, kind, params_key, data_version);
