-- Migration 014: how a target is predicted (classes, a number, or bands of a number), on every decision table, kept
-- coherent by one trigger so every writer (including the ones that only move the target role) gets the same rule.
ALTER TABLE meta.field_decisions ADD COLUMN IF NOT EXISTS target_task TEXT, ADD COLUMN IF NOT EXISTS target_bins JSONB;
ALTER TABLE meta.assessments ADD COLUMN IF NOT EXISTS target_task TEXT, ADD COLUMN IF NOT EXISTS target_bins JSONB;
ALTER TABLE meta.experiment_decisions ADD COLUMN IF NOT EXISTS target_task TEXT,
    ADD COLUMN IF NOT EXISTS target_bins JSONB;

ALTER TABLE meta.field_decisions DROP CONSTRAINT IF EXISTS field_decisions_target_task_check,
    ADD CONSTRAINT field_decisions_target_task_check CHECK (target_task IN ('classes', 'regression', 'bins'));
ALTER TABLE meta.assessments DROP CONSTRAINT IF EXISTS assessments_target_task_check,
    ADD CONSTRAINT assessments_target_task_check CHECK (target_task IN ('classes', 'regression', 'bins'));
ALTER TABLE meta.experiment_decisions DROP CONSTRAINT IF EXISTS experiment_decisions_target_task_check,
    ADD CONSTRAINT experiment_decisions_target_task_check CHECK (target_task IN ('classes', 'regression', 'bins'));

-- Mirrors semantics.default_task.
CREATE OR REPLACE FUNCTION meta.default_target_task(semantic_type text, n_unique bigint) RETURNS text
LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE
        WHEN semantic_type IN ('binary', 'categorical') THEN 'classes'
        WHEN semantic_type = 'numeric_continuous' THEN 'regression'
        WHEN semantic_type = 'numeric_discrete' THEN CASE WHEN coalesce(n_unique, 0) <= 100 THEN 'classes'
                                                          ELSE 'regression' END
    END
$$;

CREATE OR REPLACE FUNCTION meta.target_task_coherent() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    levels bigint;
BEGIN
    IF NEW.action <> 'target' THEN
        NEW.target_task := NULL;
        NEW.target_bins := NULL;
        RETURN NEW;
    END IF;
    IF NEW.target_task IS NULL
       OR (NEW.target_task IN ('regression', 'bins')
           AND NEW.semantic_type NOT IN ('numeric_continuous', 'numeric_discrete'))
       OR (NEW.target_task = 'classes' AND NEW.semantic_type = 'numeric_continuous') THEN
        SELECT n_unique INTO levels FROM meta.fields WHERE id = NEW.field_id;
        NEW.target_task := meta.default_target_task(NEW.semantic_type, levels);
    END IF;
    IF NEW.target_task IS DISTINCT FROM 'bins' THEN
        NEW.target_bins := NULL;
    ELSIF NEW.target_bins IS NULL OR NEW.target_bins = 'null'::jsonb THEN
        NEW.target_bins := '{"k": 5}'::jsonb;
    END IF;
    RETURN NEW;
END
$$;

DROP TRIGGER IF EXISTS target_task_coherent ON meta.field_decisions;
CREATE TRIGGER target_task_coherent BEFORE INSERT OR UPDATE ON meta.field_decisions
    FOR EACH ROW EXECUTE FUNCTION meta.target_task_coherent();
DROP TRIGGER IF EXISTS target_task_coherent ON meta.assessments;
CREATE TRIGGER target_task_coherent BEFORE INSERT OR UPDATE ON meta.assessments
    FOR EACH ROW EXECUTE FUNCTION meta.target_task_coherent();
DROP TRIGGER IF EXISTS target_task_coherent ON meta.experiment_decisions;
CREATE TRIGGER target_task_coherent BEFORE INSERT OR UPDATE ON meta.experiment_decisions
    FOR EACH ROW EXECUTE FUNCTION meta.target_task_coherent();

-- Every target so far was predicted as classes.
UPDATE meta.field_decisions SET target_task = 'classes' WHERE action = 'target' AND target_task IS DISTINCT FROM 'classes';
UPDATE meta.assessments SET target_task = 'classes' WHERE action = 'target' AND target_task IS DISTINCT FROM 'classes';
UPDATE meta.experiment_decisions SET target_task = 'classes'
 WHERE action = 'target' AND target_task IS DISTINCT FROM 'classes';

CREATE OR REPLACE FUNCTION pg_temp.with_target_tasks(snap jsonb) RETURNS jsonb LANGUAGE sql AS $$
    SELECT jsonb_set(snap, '{fields}', coalesce((
        SELECT jsonb_object_agg(e.key, jsonb_set(e.value, '{decision}', (e.value -> 'decision') || jsonb_build_object(
                   'target_task', CASE WHEN e.value -> 'decision' ->> 'action' = 'target' THEN 'classes' END,
                   'target_bins', NULL)))
          FROM jsonb_each(snap -> 'fields') e), '{}'::jsonb))
$$;

ALTER TABLE meta.recipes DISABLE TRIGGER recipes_frozen;
UPDATE meta.recipes SET snapshot = pg_temp.with_target_tasks(snapshot)
 WHERE snapshot ? 'fields'
   AND EXISTS (SELECT 1 FROM jsonb_each(snapshot -> 'fields') e WHERE NOT (e.value -> 'decision' ? 'target_task'));
ALTER TABLE meta.recipes ENABLE TRIGGER recipes_frozen;
UPDATE meta.exports SET snapshot = pg_temp.with_target_tasks(snapshot)
 WHERE snapshot ? 'fields'
   AND EXISTS (SELECT 1 FROM jsonb_each(snapshot -> 'fields') e WHERE NOT (e.value -> 'decision' ? 'target_task'));
