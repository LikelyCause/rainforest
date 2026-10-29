-- Migration 026: the SHA-256 of the export's preprocessor.joblib when a deployment was made, checked like its model
-- files' before every load.
ALTER TABLE meta.deployments ADD COLUMN IF NOT EXISTS preprocessor_sha256 TEXT;
