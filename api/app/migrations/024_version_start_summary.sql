-- Migration 024: what a new version could not carry over from the version before, because a refresh changed the data
-- since: {missing: [field names], targets_not_carried: [{name, problem}]}, or null when it carried everything.
ALTER TABLE meta.recipes ADD COLUMN IF NOT EXISTS start_summary JSONB;
