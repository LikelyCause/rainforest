-- Migration 002: bulk edits are recorded as batches so they can be audited and undone.

ALTER TABLE meta.decision_history ADD COLUMN IF NOT EXISTS batch_id UUID;
CREATE INDEX IF NOT EXISTS decision_history_batch_idx ON meta.decision_history (batch_id) WHERE batch_id IS NOT NULL;

-- snapshot: {field_id: {"decision": <field_decisions row>, "settings": [<field_settings row per family>]}}
-- taken before the batch, so undo restores every family exactly.
CREATE TABLE IF NOT EXISTS meta.bulk_batches (
    id         UUID        PRIMARY KEY,
    dataset_id BIGINT      NOT NULL REFERENCES meta.datasets (id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    summary    JSONB       NOT NULL,
    snapshot   JSONB       NOT NULL,
    undone_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS bulk_batches_dataset_idx ON meta.bulk_batches (dataset_id, created_at DESC);
