-- Migration 005: the upload a dataset is ingested from, so startup can remove uploads that no queued or ingesting
-- dataset still needs (left behind by a crash or a deleted row).

ALTER TABLE meta.datasets ADD COLUMN IF NOT EXISTS upload_path TEXT;
