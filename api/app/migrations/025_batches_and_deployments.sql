-- Migration 025: batches of runs (one model per class, or a search over settings), and test deployments that score new
-- data with a set of finished runs.

CREATE TABLE IF NOT EXISTS meta.run_batches (
    id            BIGSERIAL   PRIMARY KEY,
    experiment_id BIGINT      NOT NULL REFERENCES meta.experiments (id) ON DELETE CASCADE,
    export_id     BIGINT      NOT NULL REFERENCES meta.exports (id) ON DELETE CASCADE,
    kind          TEXT        NOT NULL CHECK (kind IN ('one_vs_rest', 'search')),
    name          TEXT        NOT NULL,
    spec          JSONB       NOT NULL,  -- model, device, base params, target, classes or space, sampler, ranking metric
    skipped       JSONB       NOT NULL DEFAULT '[]',  -- [{class, reason}]
    status        TEXT        NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'done', 'cancelled')),
    created_by    BIGINT      REFERENCES meta.users (id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS run_batches_experiment_idx ON meta.run_batches (experiment_id, id DESC);
CREATE INDEX IF NOT EXISTS run_batches_export_idx ON meta.run_batches (export_id);

ALTER TABLE meta.model_runs ADD COLUMN IF NOT EXISTS batch_id BIGINT REFERENCES meta.run_batches (id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS model_runs_batch_idx ON meta.model_runs (batch_id);

-- A batch's pending runs are cancelled rather than failed or deleted.
ALTER TABLE meta.model_runs DROP CONSTRAINT IF EXISTS model_runs_status_check,
    ADD CONSTRAINT model_runs_status_check CHECK (status IN ('pending', 'running', 'done', 'failed', 'cancelled'));

CREATE TABLE IF NOT EXISTS meta.deployments (
    id            BIGSERIAL   PRIMARY KEY,
    dataset_id    BIGINT      NOT NULL REFERENCES meta.datasets (id) ON DELETE CASCADE,
    experiment_id BIGINT      NOT NULL REFERENCES meta.experiments (id) ON DELETE CASCADE,
    -- Restrict: a deployment is frozen files, so the export and runs it scores with stay while it exists.
    export_id     BIGINT      NOT NULL REFERENCES meta.exports (id) ON DELETE RESTRICT,
    batch_id      BIGINT      REFERENCES meta.run_batches (id) ON DELETE SET NULL,
    name          TEXT        NOT NULL,
    models        JSONB       NOT NULL,  -- [{run_id, label, kind: binary|multiclass|regression, positive_class?, threshold?, threshold_rule}]
    status        TEXT        NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'retired')),
    created_by    BIGINT      REFERENCES meta.users (id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    retired_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS deployments_dataset_idx ON meta.deployments (dataset_id, id DESC);
CREATE INDEX IF NOT EXISTS deployments_experiment_idx ON meta.deployments (experiment_id);
CREATE INDEX IF NOT EXISTS deployments_export_idx ON meta.deployments (export_id);

CREATE TABLE IF NOT EXISTS meta.deployment_keys (
    id            BIGSERIAL   PRIMARY KEY,
    deployment_id BIGINT      NOT NULL REFERENCES meta.deployments (id) ON DELETE CASCADE,
    name          TEXT        NOT NULL,
    prefix        TEXT        NOT NULL,        -- the first 8 characters, shown in the list
    hash          TEXT        NOT NULL UNIQUE, -- SHA-256 of the key
    created_by    BIGINT      REFERENCES meta.users (id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at  TIMESTAMPTZ,
    revoked_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS deployment_keys_deployment_idx ON meta.deployment_keys (deployment_id);

CREATE TABLE IF NOT EXISTS meta.deployment_tests (
    id            BIGSERIAL   PRIMARY KEY,
    deployment_id BIGINT      NOT NULL REFERENCES meta.deployments (id) ON DELETE CASCADE,
    filename      TEXT        NOT NULL,
    status        TEXT        NOT NULL CHECK (status IN ('pending', 'running', 'done', 'failed')),
    rows          BIGINT,
    has_labels    BOOLEAN,
    metrics       JSONB,
    error         TEXT,
    owner         INTEGER,
    created_by    BIGINT      REFERENCES meta.users (id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS deployment_tests_deployment_idx ON meta.deployment_tests (deployment_id, id DESC);

-- The SQL console's role is granted SELECT on every meta table, so these follow the experiment policies: a private
-- experiment's batches, deployments and tests stay out of it. Keys have no policy at all, and bootstrap also revokes
-- the role's SELECT on them (db.PRIVATE_TABLES), so it never sees a hash. The app's own role owns the tables, so none
-- of this applies to it.
ALTER TABLE meta.run_batches ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS run_batches_public ON meta.run_batches;
CREATE POLICY run_batches_public ON meta.run_batches FOR SELECT
    USING (experiment_id IN (SELECT id FROM meta.experiments));

ALTER TABLE meta.deployments ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS deployments_public ON meta.deployments;
CREATE POLICY deployments_public ON meta.deployments FOR SELECT
    USING (experiment_id IN (SELECT id FROM meta.experiments));

ALTER TABLE meta.deployment_tests ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS deployment_tests_public ON meta.deployment_tests;
CREATE POLICY deployment_tests_public ON meta.deployment_tests FOR SELECT
    USING (deployment_id IN (SELECT id FROM meta.deployments));

ALTER TABLE meta.deployment_keys ENABLE ROW LEVEL SECURITY;
