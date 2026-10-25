-- Migration 023: datasets built from MongoDB collections, the pulls that fill them, and the `refreshing` lock a
-- refresh holds. The connection is stored by name; its URI lives only in the API's environment.
ALTER TABLE meta.datasets DROP CONSTRAINT IF EXISTS datasets_file_format_check,
    ADD CONSTRAINT datasets_file_format_check CHECK (file_format IN ('csv', 'parquet', 'generated', 'mongodb'));
ALTER TABLE meta.datasets DROP CONSTRAINT IF EXISTS datasets_status_check,
    ADD CONSTRAINT datasets_status_check
        CHECK (status IN ('pending', 'ingesting', 'profiling', 'correlating', 'ready', 'failed', 'refreshing'));

CREATE TABLE IF NOT EXISTS meta.dataset_sources (
    dataset_id   BIGINT  PRIMARY KEY REFERENCES meta.datasets ON DELETE CASCADE,
    kind         TEXT    NOT NULL CHECK (kind = 'mongodb'),
    connection   TEXT    NOT NULL,
    database     TEXT    NOT NULL,
    collections  JSONB   NOT NULL,  -- [{name, label, filter, projection}], filters as extended JSON
    add_source   BOOLEAN NOT NULL,
    dedupe       BOOLEAN NOT NULL,
    match_on     TEXT[],            -- replaces the shared columns deduplication compares, when set
    created_by   BIGINT  REFERENCES meta.users
);

CREATE TABLE IF NOT EXISTS meta.dataset_pulls (
    id            BIGSERIAL   PRIMARY KEY,
    dataset_id    BIGINT      NOT NULL REFERENCES meta.datasets ON DELETE CASCADE,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    status        TEXT        NOT NULL CHECK (status IN ('running', 'done', 'failed')),
    rows          BIGINT,
    per_source    JSONB,  -- [{label, documents, removed_duplicates}]
    flatten_notes JSONB,  -- {arrays, mixed, renamed, compared_on}
    drift         JSONB,  -- {family: reuse.match output} per finalized baseline, or null
    error         TEXT,
    created_by    BIGINT      REFERENCES meta.users
);
CREATE INDEX IF NOT EXISTS dataset_pulls_dataset_idx ON meta.dataset_pulls (dataset_id, id DESC);

-- The pull an export was built from, so its lineage survives a refresh.
ALTER TABLE meta.exports ADD COLUMN IF NOT EXISTS pull_id BIGINT REFERENCES meta.dataset_pulls (id) ON DELETE SET NULL;
