-- Migration 028: where a dataset's data came from when it wasn't an upload, so an export's script can get the same
-- data: {"kind": "library", "key"} for a sample library dataset, {"kind": "generated", "settings": {rows, fields,
-- seed, target_kind, numeric_target}} for a generated sample. Null for uploads, which keep source_filename and
-- file_format. Datasets from before this have none; Export to code works the source out when it can.
ALTER TABLE meta.datasets ADD COLUMN IF NOT EXISTS source_spec JSONB;
