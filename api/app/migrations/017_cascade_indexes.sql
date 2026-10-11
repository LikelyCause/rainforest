-- Migration 017: indexes that lead with the column a cascade or per-export query looks up, so deleting a dataset
-- (fields cascade into these tables one row at a time) and listing an export's runs don't scan whole tables.

CREATE INDEX IF NOT EXISTS assessments_field_idx ON meta.assessments (field_id);
CREATE INDEX IF NOT EXISTS assessment_settings_field_idx ON meta.assessment_settings (field_id);
CREATE INDEX IF NOT EXISTS recipe_reopened_field_idx ON meta.recipe_reopened (field_id);
CREATE INDEX IF NOT EXISTS model_runs_export_idx ON meta.model_runs (export_id);
