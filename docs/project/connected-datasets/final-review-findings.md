# Final-review fix brief (connected datasets, branch head deb711c)

Whole-branch review findings that survived refutation (each reproduced by ≥2 of 3 refuters). Fix every numbered item,
test first (record the failing run), in the worktree `/Users/sentry/Development/Rainforest/.claude/worktrees/connected-datasets`.

## Important

1. **No new baseline version can start once a refresh removes or turns down a target the finalized baseline uses**
   (api/app/recipes.py ~72, start_version / populate from parent). A new version must be startable: carry the parent's
   decisions for fields still present, drop vanished targets from the new version's targets (and the target role of a
   field profiling now rejects), and say so in the new version's reuse/start summary. Test: finalize with target
   `engine`, refresh after `$unset engine` → `POST` new version succeeds with targets without `engine`.
2. **Exports from team and personal experiments skip the "no longer in the data" block** (api/app/routers/exports.py):
   Ruling 10's check (kept fields + targets of the snapshot that are gone) runs only for the default Team export. Apply
   it to every export built from a snapshot or an experiment's adjusted copy or an assessment. Tests for a non-default
   team experiment and a personal experiment.
3. **A refresh discards the remembered datetime format of a field the team retyped to datetime**
   (api/app/processing.py update_field): `update_field` rewrites stats in a way that loses the stored datetime format.
   Keep it (or re-detect it for the retyped type on the new data). Test: retype a text date field to datetime, refresh,
   export still parses the dates.
4. **A collection that is missing (dropped, renamed) or comes back empty on refresh silently removes its rows, fields,
   decisions and history** (api/app/sources/mongodb.py ~326, sources/pull.py). Ruling: a refresh fails (pull `failed`,
   dataset untouched) when a configured collection does not exist, and when it returns 0 documents although its
   previous successful pull had more than 0; the error names the collection. A first connect of an empty collection
   still works (the preview already shows 0). Tests for missing and newly empty.
5. **The `source` label column can change meaning under the same field id** (sources/pull.py): when labels are reordered
   or renamed between pulls, `source` values change while the field keeps its id and decisions. Ruling: labels are fixed
   at connect time (stored in dataset_sources) and the refresh uses the stored labels; if a stored label no longer maps
   to its collection the refresh fails. Verify that is what happens today and test it; if labels can drift, fix.
6. **A failed later pull hides the drift summary of the data actually in use** (ui/src/components/PullCard.tsx): show the
   latest *done* pull's summary (and reopen action), with the failed pull shown above it as a notice with its error.
   Test the selection helper in ui/src/pulls.ts.
7. **The refresh failure handler's writes are neither atomic nor retried**, so a DB error in the handler leaves the
   dataset `refreshing` until an API restart (api/app/refresh.py). Do the failure writes (pull failed + dataset ready)
   in one transaction with a short retry, and make the startup sweep also reset a refreshing dataset whose pull is
   already `failed`/`done`. Test by making the first handler write raise once.
8. **After a refresh removes fields, untouched team experiments report phantom adjustments** (api/app/experiments.py
   ~257; also the manifest `adjustments` and MLflow): a field missing from the data must not count as an adjustment.
   Removed fields are listed separately (e.g. `removed_by_refresh`) on the experiment and its Adjustments card says "N
   fields left the data in pull #n". Tests on the API; UI shows the separate line.
9. **Empty-string keys are split across several columns** (polars renames a "" column per chunk): map an empty key to a
   fixed name (`_empty`, with the usual collision suffixing) in flatten_document and record it under renamed; docs state
   it. Test with documents across two chunks.
10. **Fields deleted by a refresh are listed only in a warning truncated at 20 names** (api/app/refresh.py): store the full
    list on the pull row (`flatten_notes.removed_fields` or a new key in the pull's JSON) and show it in the PullCard
    (collapsed list). Keep the warning short with "and N more".

## Minor (fold in; each small)

11. docs/api.md and docs/data-preparation.md say reads are unaffected during a refresh; Explore lookups and
    correlations are refused. Ruling: keep the API behaviour, fix the docs (Explore is unavailable while a refresh runs).
12. ui/src/types.ts `Pull.has_drift` is never returned by /pulls or /refresh: remove it (it lives on `last_pull`).
13. docs/sql.md: list `meta.dataset_sources` and `meta.dataset_pulls` among the tables the console can read.
14. deploy/README.md: the dialog lists databases through `listDatabases` with `authorizedDatabases`; a read-only user sees
    the databases it can read (state what is true for MongoDB 7).
15. docs/experiments-and-training.md: says any removed baseline field blocks the export; only kept fields and targets do.
16. docs/data-preparation.md profiling claims in the flatten section (`_id` profiles as an identifier only with enough
    rows; arrays as JSON text may profile as categorical; the identifier-name pattern) — state what profiling does.
17. jobs.py: an interrupted connect is told to "upload or generate it again"; say to delete and connect again.

Rules: match surrounding code, low comment density, comments explain why; docs current-state only. Commit in logical
commits ending with "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"; don't push. Python:
`cd api && RAINFOREST_TEST_DATABASE_URL=postgresql://postgres:test@127.0.0.1:55432/postgres RAINFOREST_TEST_MONGO_URL=mongodb://127.0.0.1:27017 /private/tmp/claude-501/-Users-sentry-Development/8be71796-eab6-5caa-b0e3-f1c0eb3a475a/scratchpad/venv/bin/pytest ...`, ruff from the same bin. UI: `cd ui && npm run typecheck && npx vitest run && npx prettier --check src && npx eslint src` (4 theme.test.ts failures and 20 eslint warnings pre-exist). Full API suite and UI checks once at the end.
