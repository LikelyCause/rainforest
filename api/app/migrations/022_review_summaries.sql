-- Migration 022: a finalized version keeps how its team review went (fields adopted, disagreements, how many the lead
-- settled and how many were still open at finalize), since its review flags live in the working tables only until the
-- next version. Versions finalized before this have none; the Baselines page reconstructs what it can.
ALTER TABLE meta.recipes ADD COLUMN IF NOT EXISTS review_summary JSONB;
