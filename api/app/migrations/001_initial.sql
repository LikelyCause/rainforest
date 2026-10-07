-- Migration 001: initial schema.
--   meta.*  : datasets, field profiles, correlations, decisions, per-family settings, exports, model runs
--   raw.*   : one table per uploaded dataset (raw.ds_<id>), original values

CREATE SCHEMA IF NOT EXISTS meta;
CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE IF NOT EXISTS meta.datasets (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT        NOT NULL,
    source_filename TEXT,
    file_format     TEXT        NOT NULL CHECK (file_format IN ('csv', 'parquet', 'generated')),
    parquet_path    TEXT,
    raw_table       TEXT,
    target_columns  TEXT[]      NOT NULL DEFAULT '{}',
    row_count       BIGINT,
    column_count    INT,
    status          TEXT        NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'ingesting', 'profiling', 'correlating', 'ready', 'failed')),
    progress        REAL        NOT NULL DEFAULT 0,
    status_message  TEXT,
    error           TEXT,
    warnings        TEXT[]      NOT NULL DEFAULT '{}',
    model_family    TEXT        NOT NULL DEFAULT 'tree',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    profiled_at     TIMESTAMPTZ
);

-- One row per column of a dataset: inferred type + general statistics.
-- Frequently queried statistics are typed columns; everything else lives in `stats`.
CREATE TABLE IF NOT EXISTS meta.fields (
    id                   BIGSERIAL PRIMARY KEY,
    dataset_id           BIGINT  NOT NULL REFERENCES meta.datasets (id) ON DELETE CASCADE,
    ordinal              INT     NOT NULL,
    name                 TEXT    NOT NULL,
    db_column            TEXT    NOT NULL,
    physical_dtype       TEXT    NOT NULL,
    inferred_type        TEXT    NOT NULL,
    inference_confidence REAL,
    inference_reasons    TEXT[]  NOT NULL DEFAULT '{}',
    is_target            BOOLEAN NOT NULL DEFAULT false,
    row_count            BIGINT,
    null_count           BIGINT,
    null_pct             DOUBLE PRECISION,
    n_unique             BIGINT,
    unique_pct           DOUBLE PRECISION,
    mean                 DOUBLE PRECISION,
    std                  DOUBLE PRECISION,
    min_value            DOUBLE PRECISION,
    max_value            DOUBLE PRECISION,
    median               DOUBLE PRECISION,
    skewness             DOUBLE PRECISION,
    kurtosis             DOUBLE PRECISION,
    zeros_pct            DOUBLE PRECISION,
    outlier_pct          DOUBLE PRECISION,
    top_value            TEXT,
    top_value_pct        DOUBLE PRECISION,
    avg_length           DOUBLE PRECISION,
    max_target_assoc     DOUBLE PRECISION,
    leakage_flag         BOOLEAN NOT NULL DEFAULT false,
    stats                JSONB   NOT NULL DEFAULT '{}',
    charts               JSONB   NOT NULL DEFAULT '{}',
    suggestion           JSONB   NOT NULL DEFAULT '{}',  -- {family: suggested decision}
    UNIQUE (dataset_id, name),
    UNIQUE (dataset_id, ordinal)
);
CREATE INDEX IF NOT EXISTS fields_dataset_type_idx ON meta.fields (dataset_id, inferred_type);

-- Pairwise associations, stored once per unordered pair (field_a < field_b).
--   spearman          : numeric x numeric, signed, [-1, 1]
--   cramers_v         : categorical x categorical, [0, 1] (bias corrected)
--   correlation_ratio : numeric x categorical (eta), [0, 1]
CREATE TABLE IF NOT EXISTS meta.field_correlations (
    dataset_id BIGINT           NOT NULL REFERENCES meta.datasets (id) ON DELETE CASCADE,
    field_a    BIGINT           NOT NULL REFERENCES meta.fields (id) ON DELETE CASCADE,
    field_b    BIGINT           NOT NULL REFERENCES meta.fields (id) ON DELETE CASCADE,
    method     TEXT             NOT NULL CHECK (method IN ('spearman', 'cramers_v', 'correlation_ratio')),
    value      DOUBLE PRECISION NOT NULL,
    strength   DOUBLE PRECISION NOT NULL,
    n_obs      BIGINT,
    PRIMARY KEY (field_a, field_b),
    CHECK (field_a < field_b)
);
CREATE INDEX IF NOT EXISTS field_correlations_b_idx ON meta.field_correlations (field_b);
CREATE INDEX IF NOT EXISTS field_correlations_strength_idx ON meta.field_correlations (dataset_id, strength DESC);

-- Your choices. Shared across model families: keep/drop/target, semantic type, notes, review state.
CREATE TABLE IF NOT EXISTS meta.field_decisions (
    field_id      BIGINT      PRIMARY KEY REFERENCES meta.fields (id) ON DELETE CASCADE,
    dataset_id    BIGINT      NOT NULL REFERENCES meta.datasets (id) ON DELETE CASCADE,
    action        TEXT        NOT NULL CHECK (action IN ('keep', 'drop', 'target')),
    semantic_type TEXT        NOT NULL,
    notes         TEXT,
    reviewed      BOOLEAN     NOT NULL DEFAULT false,
    source        TEXT        NOT NULL DEFAULT 'suggested' CHECK (source IN ('suggested', 'user')),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS field_decisions_dataset_idx ON meta.field_decisions (dataset_id);

-- Preprocessing per model family (tree / linear / neural): what a tree model needs differs
-- from a linear one, so these are stored once per family and seeded from that family's suggestion.
CREATE TABLE IF NOT EXISTS meta.field_settings (
    field_id              BIGINT      NOT NULL REFERENCES meta.fields (id) ON DELETE CASCADE,
    family                TEXT        NOT NULL,
    dataset_id            BIGINT      NOT NULL REFERENCES meta.datasets (id) ON DELETE CASCADE,
    imputation            TEXT        NOT NULL DEFAULT 'none',
    impute_value          TEXT,
    add_missing_indicator BOOLEAN     NOT NULL DEFAULT false,
    transform             TEXT        NOT NULL DEFAULT 'none',
    scaling               TEXT        NOT NULL DEFAULT 'none',
    encoding              TEXT        NOT NULL DEFAULT 'none',
    params                JSONB       NOT NULL DEFAULT '{}',
    source                TEXT        NOT NULL DEFAULT 'suggested' CHECK (source IN ('suggested', 'user')),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (field_id, family)
);
CREATE INDEX IF NOT EXISTS field_settings_dataset_idx ON meta.field_settings (dataset_id, family);

-- Audit trail of every decision change.
CREATE TABLE IF NOT EXISTS meta.decision_history (
    id         BIGSERIAL   PRIMARY KEY,
    field_id   BIGINT      NOT NULL REFERENCES meta.fields (id) ON DELETE CASCADE,
    dataset_id BIGINT      NOT NULL REFERENCES meta.datasets (id) ON DELETE CASCADE,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    family     TEXT,
    previous   JSONB,
    current    JSONB       NOT NULL
);
CREATE INDEX IF NOT EXISTS decision_history_field_idx ON meta.decision_history (field_id, changed_at DESC);

CREATE TABLE IF NOT EXISTS meta.exports (
    id            BIGSERIAL   PRIMARY KEY,
    dataset_id    BIGINT      NOT NULL REFERENCES meta.datasets (id) ON DELETE CASCADE,
    status        TEXT        NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'running', 'done', 'failed')),
    progress      REAL        NOT NULL DEFAULT 0,
    status_message TEXT,
    error         TEXT,
    model_family  TEXT        NOT NULL DEFAULT 'tree',
    config        JSONB       NOT NULL DEFAULT '{}',
    output_dir    TEXT,
    n_rows        BIGINT,
    n_features    INT,
    summary       JSONB       NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS exports_dataset_idx ON meta.exports (dataset_id, created_at DESC);

-- One row per ClassifierChain training run on an export.
CREATE TABLE IF NOT EXISTS meta.model_runs (
    id             BIGSERIAL   PRIMARY KEY,
    export_id      BIGINT      NOT NULL REFERENCES meta.exports (id) ON DELETE CASCADE,
    dataset_id     BIGINT      NOT NULL REFERENCES meta.datasets (id) ON DELETE CASCADE,
    model          TEXT        NOT NULL,
    family         TEXT        NOT NULL,
    status         TEXT        NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending', 'running', 'done', 'failed')),
    status_message TEXT,
    error          TEXT,
    config         JSONB       NOT NULL DEFAULT '{}',
    metrics        JSONB       NOT NULL DEFAULT '{}',
    report         JSONB       NOT NULL DEFAULT '{}',
    model_dir      TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS model_runs_dataset_idx ON meta.model_runs (dataset_id, created_at DESC);
