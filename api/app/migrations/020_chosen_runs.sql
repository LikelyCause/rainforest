-- Migration 020: a run can carry a note, and a lead can mark one run per experiment as the team's chosen model.
ALTER TABLE meta.model_runs ADD COLUMN IF NOT EXISTS note TEXT;
ALTER TABLE meta.model_runs ADD COLUMN IF NOT EXISTS chosen_at TIMESTAMPTZ;
ALTER TABLE meta.model_runs ADD COLUMN IF NOT EXISTS chosen_by BIGINT REFERENCES meta.users (id);
CREATE UNIQUE INDEX IF NOT EXISTS model_runs_one_chosen_idx ON meta.model_runs (experiment_id) WHERE chosen_at IS NOT NULL;
