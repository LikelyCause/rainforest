-- Migration 004: which API process runs each job. The process holds an advisory lock on its owner id for its
-- lifetime, so a job whose owner's lock is gone was interrupted, however recently it last reported progress.

ALTER TABLE meta.datasets    ADD COLUMN IF NOT EXISTS owner INTEGER;
ALTER TABLE meta.exports     ADD COLUMN IF NOT EXISTS owner INTEGER;
ALTER TABLE meta.model_runs  ADD COLUMN IF NOT EXISTS owner INTEGER;
ALTER TABLE meta.analyses    ADD COLUMN IF NOT EXISTS owner INTEGER;
