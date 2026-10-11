-- Migration 019: a derived field keeps its definition ({"kind", "spec"}), so it can be shown and removed.
ALTER TABLE meta.fields ADD COLUMN IF NOT EXISTS derived JSONB;
