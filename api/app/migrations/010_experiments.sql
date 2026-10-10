-- Migration 010: experiments. Exports and model runs belong to one; an experiment is built either from a recipe
-- version (team) or from one assessor's own assessment of a version (personal).
-- A personal experiment is private to its assessor while that version is assessing; see app/experiments.py.

CREATE TABLE IF NOT EXISTS meta.experiments (
    id          BIGSERIAL   PRIMARY KEY,
    dataset_id  BIGINT      NOT NULL REFERENCES meta.datasets (id) ON DELETE CASCADE,
    name        TEXT        NOT NULL,
    owner       BIGINT      REFERENCES meta.users (id),
    recipe_id   BIGINT      NOT NULL REFERENCES meta.recipes (id) ON DELETE CASCADE,
    assessor_id BIGINT      REFERENCES meta.users (id),
    source      TEXT        NOT NULL,
    notes       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    archived_at TIMESTAMPTZ,
    CONSTRAINT experiments_source_check CHECK (source = CASE WHEN assessor_id IS NULL THEN 'recipe:' || recipe_id
                                                             ELSE 'assessment:' || recipe_id || ':' || assessor_id END)
);
CREATE INDEX IF NOT EXISTS experiments_dataset_idx ON meta.experiments (dataset_id, created_at DESC);

ALTER TABLE meta.exports ADD COLUMN IF NOT EXISTS experiment_id BIGINT REFERENCES meta.experiments (id) ON DELETE CASCADE;
-- What a personal export was built from, frozen when it was requested (team exports read their recipe's snapshot).
ALTER TABLE meta.exports ADD COLUMN IF NOT EXISTS snapshot JSONB;
ALTER TABLE meta.model_runs ADD COLUMN IF NOT EXISTS experiment_id BIGINT REFERENCES meta.experiments (id) ON DELETE CASCADE;
CREATE INDEX IF NOT EXISTS exports_experiment_idx ON meta.exports (experiment_id);
CREATE INDEX IF NOT EXISTS model_runs_experiment_idx ON meta.model_runs (experiment_id);

-- Exports and runs without an experiment (made before experiments existed) go into one "Imported" team experiment
-- per dataset, on the newest recipe version they were exported from. Bootstrap calls this on every start.
CREATE OR REPLACE FUNCTION meta.import_orphan_experiments() RETURNS integer LANGUAGE plpgsql AS $$
DECLARE ds RECORD; rid BIGINT; xid BIGINT; n integer := 0;
BEGIN
    FOR ds IN SELECT DISTINCT dataset_id AS id FROM (
                  SELECT dataset_id FROM meta.exports WHERE experiment_id IS NULL
                  UNION SELECT dataset_id FROM meta.model_runs WHERE experiment_id IS NULL) o LOOP
        SELECT x.id INTO xid FROM meta.experiments x
         WHERE x.dataset_id = ds.id AND x.name = 'Imported' AND x.assessor_id IS NULL ORDER BY x.id LIMIT 1;
        IF xid IS NULL THEN
            rid := coalesce(
                (SELECT max(recipe_id) FROM meta.exports WHERE dataset_id = ds.id),
                (SELECT id FROM meta.recipes WHERE dataset_id = ds.id
                  ORDER BY (phase = 'finalized') DESC, version DESC LIMIT 1));
            CONTINUE WHEN rid IS NULL;
            INSERT INTO meta.experiments (dataset_id, name, recipe_id, source, notes)
            VALUES (ds.id, 'Imported', rid, 'recipe:' || rid,
                    'Exports and runs made before experiments existed.')
            RETURNING id INTO xid;
            n := n + 1;
        END IF;
        UPDATE meta.exports SET experiment_id = xid WHERE dataset_id = ds.id AND experiment_id IS NULL;
        UPDATE meta.model_runs r SET experiment_id = coalesce(e.experiment_id, xid)
          FROM meta.exports e WHERE e.id = r.export_id AND r.dataset_id = ds.id AND r.experiment_id IS NULL;
    END LOOP;
    RETURN n;
END $$;
SELECT meta.import_orphan_experiments();

-- The SQL console's role sees only what everyone may see: no personal experiment while it is private, nor its
-- exports, runs or Explore results. The app's own role owns these tables, so the policies don't apply to it.
-- Exports, runs and analyses defer to the experiments policy (a row is visible when its experiment is).
ALTER TABLE meta.experiments ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS experiments_public ON meta.experiments;
CREATE POLICY experiments_public ON meta.experiments FOR SELECT
    USING (assessor_id IS NULL OR EXISTS (SELECT 1 FROM meta.recipes r WHERE r.id = recipe_id
                                                                      AND r.phase <> 'assessing'));

ALTER TABLE meta.exports ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS exports_public ON meta.exports;
CREATE POLICY exports_public ON meta.exports FOR SELECT
    USING (experiment_id IN (SELECT id FROM meta.experiments));

ALTER TABLE meta.model_runs ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS model_runs_public ON meta.model_runs;
CREATE POLICY model_runs_public ON meta.model_runs FOR SELECT
    USING (experiment_id IN (SELECT id FROM meta.experiments));

DROP POLICY IF EXISTS analyses_team_only ON meta.analyses;
CREATE POLICY analyses_team_only ON meta.analyses FOR SELECT
    USING (coalesce(params ->> 'decisions', 'team') = 'team'
           AND CASE WHEN params ->> 'data' LIKE 'export:%'
                    THEN substr(params ->> 'data', 8)::bigint IN (SELECT id FROM meta.exports) ELSE true END);
