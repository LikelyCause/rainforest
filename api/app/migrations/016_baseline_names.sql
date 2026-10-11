-- Migration 016: a name per baseline (one per model family of a dataset), kept outside the versions and their frozen
-- snapshots so a finalized baseline can be renamed.

CREATE TABLE IF NOT EXISTS meta.baseline_names (
    dataset_id BIGINT NOT NULL REFERENCES meta.datasets (id) ON DELETE CASCADE,
    family     TEXT   NOT NULL,
    name       TEXT   NOT NULL CHECK (length(name) BETWEEN 1 AND 60 AND name = btrim(name)),
    PRIMARY KEY (dataset_id, family)
);
CREATE UNIQUE INDEX IF NOT EXISTS baseline_names_unique_idx ON meta.baseline_names (dataset_id, lower(name));

CREATE OR REPLACE FUNCTION meta.baseline_name(ds BIGINT, fam TEXT) RETURNS TEXT LANGUAGE sql STABLE AS
$$ SELECT name FROM meta.baseline_names WHERE dataset_id = ds AND family = fam $$;
