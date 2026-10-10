-- Migration 009: new recipe versions that reopen some fields, and reuse of a recipe on new data.
-- scope 'all': every field is open (a fresh dataset, or a version a lead edits directly).
-- scope 'reopened': only the fields in recipe_reopened are open; the rest are carried over and locked.
-- reuse: how a recipe of (usually) another dataset was matched to this one; parent_id is that recipe.

ALTER TABLE meta.recipes ADD COLUMN IF NOT EXISTS scope TEXT NOT NULL DEFAULT 'all'
    CHECK (scope IN ('all', 'reopened'));
ALTER TABLE meta.recipes ADD COLUMN IF NOT EXISTS reuse JSONB;

CREATE TABLE IF NOT EXISTS meta.recipe_reopened (
    recipe_id BIGINT NOT NULL REFERENCES meta.recipes (id) ON DELETE CASCADE,
    field_id  BIGINT NOT NULL REFERENCES meta.fields (id) ON DELETE CASCADE,
    PRIMARY KEY (recipe_id, field_id)
);

-- A reused recipe may belong to another dataset, which can be deleted; the reuse summary keeps the provenance.
ALTER TABLE meta.recipes DROP CONSTRAINT IF EXISTS recipes_parent_id_fkey;
ALTER TABLE meta.recipes ADD CONSTRAINT recipes_parent_id_fkey
    FOREIGN KEY (parent_id) REFERENCES meta.recipes (id) ON DELETE SET NULL;

-- Finalized rows stay frozen, except for losing a parent that was deleted.
CREATE OR REPLACE FUNCTION meta.recipes_frozen() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.phase = 'finalized' AND OLD.snapshot IS NOT NULL
       AND NOT (NEW.parent_id IS NULL AND (to_jsonb(NEW) - 'parent_id') = (to_jsonb(OLD) - 'parent_id')) THEN
        RAISE EXCEPTION 'recipe v% of dataset % is finalized and cannot change', OLD.version, OLD.dataset_id;
    END IF;
    RETURN NEW;
END $$;
