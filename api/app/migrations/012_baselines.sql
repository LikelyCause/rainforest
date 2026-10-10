-- Migration 012: a recipe version is a baseline for one model family, and versions count per family.
-- The default keeps 008's DO block (which inserts without a family) working when migrations re-run.
ALTER TABLE meta.recipes ADD COLUMN IF NOT EXISTS family TEXT NOT NULL DEFAULT 'tree';
ALTER TABLE meta.recipes DROP CONSTRAINT IF EXISTS recipes_family_check;
ALTER TABLE meta.recipes ADD CONSTRAINT recipes_family_check CHECK (family IN ('tree', 'linear', 'neural'));
-- Finalized rows are frozen by a trigger; their family is set once here, from the dataset's model family.
ALTER TABLE meta.recipes DISABLE TRIGGER recipes_frozen;
UPDATE meta.recipes r SET family = d.model_family FROM meta.datasets d
 WHERE d.id = r.dataset_id AND r.family <> d.model_family;
ALTER TABLE meta.recipes ENABLE TRIGGER recipes_frozen;
ALTER TABLE meta.recipes DROP CONSTRAINT IF EXISTS recipes_dataset_id_version_key;
CREATE UNIQUE INDEX IF NOT EXISTS recipes_family_version_idx ON meta.recipes (dataset_id, family, version);

-- Default team experiments are found by this flag, not by name, so they can be renamed.
ALTER TABLE meta.experiments ADD COLUMN IF NOT EXISTS is_default BOOLEAN NOT NULL DEFAULT false;
UPDATE meta.experiments SET is_default = true
 WHERE owner IS NULL AND assessor_id IS NULL AND name LIKE 'Team · v%' AND NOT is_default;
UPDATE meta.experiments x
   SET name = 'Team · ' || CASE r.family WHEN 'tree' THEN 'Trees' WHEN 'linear' THEN 'Linear' ELSE 'Neural' END
              || ' v' || r.version
  FROM meta.recipes r WHERE r.id = x.recipe_id AND x.is_default AND x.name LIKE 'Team · v%';
