-- Migration 018: a class field (binary or categorical) can be a target predicted one-vs-rest. Its task is 'bins' with
-- target_bins = {"positive": [...]}, the classes predicted against the rest; numbers keep their bands.

CREATE OR REPLACE FUNCTION meta.target_task_coherent() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    levels bigint;
    numeric_type boolean := NEW.semantic_type IN ('numeric_continuous', 'numeric_discrete');
    grouped boolean := NEW.semantic_type IN ('binary', 'categorical') AND jsonb_typeof(NEW.target_bins) = 'object'
                       AND NEW.target_bins ? 'positive';
BEGIN
    IF NEW.action <> 'target' THEN
        NEW.target_task := NULL;
        NEW.target_bins := NULL;
        RETURN NEW;
    END IF;
    IF NEW.target_task IS NULL
       OR (NEW.target_task = 'regression' AND NOT numeric_type)
       OR (NEW.target_task = 'bins' AND NOT numeric_type AND NOT grouped)
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
