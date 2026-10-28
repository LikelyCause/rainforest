# SDD ledger — plan: docs/superpowers/plans/2026-10-25-connected-datasets.md

Spec: docs/superpowers/specs/2026-10-25-connected-datasets-design.md (read; binding).
Worktree: .claude/worktrees/connected-datasets, branch connected-datasets, BASE 117bd62 (origin/main).
Baseline: API 895 passed / 1 skipped (test DB), UI typecheck clean, vitest 182 passed + 4 pre-existing theme.test.ts failures.
Containers: rainforest-testpg (postgres:16 on 127.0.0.1:55432, password test), rainforest-testmongo (mongo:7 on 127.0.0.1:27017).
Venv: /private/tmp/claude-501/-Users-sentry-Development/8be71796-eab6-5caa-b0e3-f1c0eb3a475a/scratchpad/venv (pymongo installed by Task 1).

## Pre-flight scan

| Tasks | Produces vs consumes | Found |
|---|---|---|
| T1 ↔ T2 | registry/get/Source/CollectionSpec/PreviewResult/ColumnInfo | names match |
| T1 ↔ T3 | Source.pull signature declared in T1, implemented in T3 | consistent |
| T3 ↔ T4 | run_connect(dataset_id); _create(fmt="mongodb"); dataset_sources.match_on | match |
| T3 ↔ T5 | pull/assemble reused; load_raw_table(table_suffix) added in T5 | match |
| T5 ↔ T6 | run_refresh(dataset_id, pull_id); status 'refreshing' from T3's migration | match |
| T4 ↔ T7, T2 ↔ T7 | dataset.source/last_pull, sources endpoints | match |
| T5/T6 ↔ T8 | drift keyed by family ('tree','linear','neural') | match |
| T2 ↔ T3/T5/T9 | `mongo` fixture and seed generator | see Ruling 1 |
| T2/T9 ↔ CI files | service blocks added incrementally | no conflict |

Self-consistency: T3's 750 rows = 300 + 500 − 50 duplicates ✓; T5 tests mutate Mongo then refresh ✓; no plan-mandated rubric defects.

Ruling 1 (T2): the seed generator lives in `api/app/sample_mongo.py` from the start (tests import it; Task 9's Compose seeder runs it), so Task 9 doesn't move code. — Cost if wrong: none.
Ruling 2 (T1): booleans flatten as booleans (Parquet boolean; profiling reads them as `binary`); the spec's flatten table gains a row. Preview reports kind `text` for them until Task 3 adds `"boolean"` to ColumnKind — Task 3 carries it. — Cost if wrong: a boolean column previewed as text.

## Progress
Task 1: implementer a0f6d59a7482bb51d (sonnet), commit 33fb8c3, DONE_WITH_CONCERNS (brief file was missing — the Bash that wrote it was refused by the worktree guard; implementer worked from the plan + spec; PullResult left to Task 3; redaction via split_hosts and parent domain for +srv).
Task 1: review (opus): 4 Important (_count fallback dead / views fail; signalling-NaN Decimal128 raises; list_databases hides auth failures; present_share unweighted), 12 minor.
Ruling 3 (T1): folded into fix round 1 — `"boolean"` ColumnKind (supersedes Ruling 2's deferral), the collision precedence rule (a literal key keeps the base name; a dotted key that collides gets `_2`, independent of order), arrays with NaN/inf serialised as null (valid JSON), the laziness assertion in the registry test. — Cost if wrong: a larger fix diff.
Task 1: minor (deferred): +srv entries skip structural checks at startup.
Task 1: minor (deferred): empty-string keys and always-`{}` fields need documenting (data-preparation.md, Task 9).
Task 1: minor (deferred): redaction garbles `mongodb://` when a host is literally `mongo`; greedy regex in _host_tokens with `@` in options; module-level redact without hosts leaves the host.
Task 1: minor (deferred): empty Binary inside an array becomes "b''".
Task 1: minor (deferred): a slow view's count_documents timeout fails the whole listing.
Task 1: minor (deferred): lazy client not locked (two clients on a concurrent first use).
Task 1: minor (deferred): `;` inside a URI splits the entry; docs should say to percent-encode.
Task 1: carry-forward → T3: the pull needs a key→column map stable across documents and chunks (collision rule from Ruling 3), and `pull` must replace `Any` with `PullResult`.
Task 1: fix round 1/5 (8 addressed, 0 open; commits 33fb8c3..85b6148)
Task 1: minor (deferred): a filtered preview whose count timed out returns a whole-collection estimate without flagging it as an upper bound.
Task 1: minor (deferred): Unauthorized detection uses exc.code only (fails loudly otherwise — safe).
Task 1: carry-forward → T7: ColumnKind now includes "boolean" and "mixed"; UI labels must cover both.
Task 1: complete (commits 117bd62..85b6148, review clean)
Task 2: implementer a33e8d6b201eb3a8a (sonnet), commit 52067aa, DONE_WITH_CONCERNS (preview POST added to VIEWER_ALLOWED; filter/projection are JSON strings in request bodies — T3/T4 reuse collection_specs(); make test-mongo alone skips the API tests since they need Postgres too).
Task 2: carry-forward → T4: request bodies carry `filter`/`projection` as JSON strings; reuse `routers/sources.collection_specs()`.
Task 2: review (sonnet): approved, 6 minor.
Task 2: minor (deferred): the no-URI test doesn't assert the host is absent.
Task 2: minor (deferred): bare `except Exception` around json_util.loads (narrow to the decoder error types).
Task 2: minor (deferred): fixture seed not in try/finally (a half-failed seed leaves rf_test_<pid>).
Task 2: minor (deferred): `make test-mongo` alone skips the API tests (needs Postgres too); help text should say so.
Task 2: minor (deferred): a Viewer can run filter-based counts / $where through preview — docs keep "the connection is read-only"; preview accepts the same collection twice (T3/T4 reject duplicates).
Task 2: minor (deferred): three lines >120 chars (auth.py comment, docs/api.md, docs/development.md).
Task 2: complete (commits 85b6148..52067aa, review clean)
Task 3: implementer acf2da7502478195f (opus), commits d2f1bc4 + 8b35140, DONE_WITH_CONCERNS (budget uses the preview's cell estimate; mixed = value kinds change across the pull; process_dataset keeps prior warnings; flatten_notes carries compared_on; a restart fails interrupted pulls).
Task 3: carry-forward → T4: reject duplicate collection labels and the label `__source_rank`; store filter/projection as extended JSON strings.
Task 3: carry-forward → T5: reuse `pull.pull_into` but it needs a status parameter (writes `ingesting` today, refresh needs `refreshing`).
Task 3: review (opus): 1 Important (datetime column crashes on an all-null later chunk), 8 minor.
Ruling 4 (T3): minors 3 (success-path DB error leaves pull `running`/dataset `ingesting` — widen the try) and 4 (warn when deduplication compared nothing) folded into fix round 1. — Cost if wrong: slightly larger fix diff.
Task 3: minor (deferred): memory at target size — assemble peaks ~2.3× the frame (≈8 GiB for 1.5M×300; more with long text); docs (T9) must state it; spec "memory-bounded" applies to the pull only.
Task 3: minor (deferred): no mid-pull budget enforcement after the 200-doc estimate.
Task 3: minor (deferred): non-PyMongoError exceptions stored as str(exc) without redact (no leaking path found).
Task 3: minor (deferred): `renamed` notes keyed by ".".join(path) can collide.
Task 3: minor (deferred): no socket timeout configured for stalled connections.
Task 3: minor (deferred): sweep test doesn't check a live owner's pull is left alone.
Task 3: carry-forward → T5: process_dataset keeps prior warnings; refresh must clear pull/processing-owned warnings before re-running; sweep must also reset `refreshing` datasets (jobs.py lists pending/ingesting/profiling/correlating only); `warnings_for(..., dedupe=)` takes a dedupe argument.
Task 3: fix round 1/5 (3 addressed, 0 open — typed all-null chunks, widened try, no-shared-columns warning; commits 8b35140..9bf6912)
Task 3: minor (deferred): a failed pull after `done` keeps rows/per_source/flatten_notes on the row (T8's pulls UI must check status).
Task 3: complete (commits 52067aa..9bf6912, review clean)
Task 4: implementer a575ef0a7791b78cf (sonnet), commit d844dd8, DONE (15 tests; stores filter/projection as parsed extended-JSON dicts since pull.specs_from expects dicts; match_on validated at pull time; targets as a JSON list).
Ruling 5 (T4): filters/projections are stored as parsed extended-JSON documents (JSONB), not strings — `pull.specs_from` already expects that, and the read model re-serialises them. Supersedes Task 3's "store as strings" carry-forward. — Cost if wrong: byte-level round-trip of the user's JSON isn't preserved (semantics are); JSONB reorders keys, which matters for embedded-document equality filters (docs note deferred).
Task 4: review (sonnet): approved, 6 minor.
Ruling 6 (T4): an explicit `add_source: false` with 2+ collections is a 422 rather than silently coerced — the UI locks the box on, and a refused contradiction beats a silent override. — Cost if wrong: none.
Task 4: minor (deferred): _create + dataset_sources insert are two statements with a compensating delete (not one transaction).
Task 4: minor (deferred): JSONB key reordering can change embedded-document equality filters — needs one docs line.
Task 4: minor (deferred): round-trip test asserts less than its name (no ready/row-count assertion).
Task 4: minor (deferred): last_pull timestamps via jsonb_build_object may format differently from /pulls.
Task 4: minor (deferred): dataset name not stripped (whitespace-only accepted).
Task 4: carry-forward → T5: store SQL NULL (never JSON null) in dataset_pulls.drift when there is no drift; `has_drift` is `drift IS NOT NULL`.
Task 4: complete (commits 9bf6912..d844dd8, review clean)
Task 5: implementer aa2091fa9e5c02d54 (opus), commits d670537 + 56c193f + da2cf58, DONE_WITH_CONCERNS (mark_stale covers running analyses too; COMMIT-after-os.replace window accepted; a target profiling now rejects keeps its role).
Task 5: carry-forward → T6: when setting `refreshing`, also set `datasets.owner = jobs.owner_id()` so a sibling worker's startup sweep doesn't reset a live refresh; the export/run lock during refresh is what makes the Parquet swap safe for concurrent readers.
Task 5: review (opus): approved, 9 minor; two need rulings.
Ruling 7 (T5): `update_field` may rewrite *suggested* decisions and non-user settings (spec: working decisions follow the new stats) but must leave reviewed fields' settings alone and must not bump updated_at when nothing changes. Folded into fix round 1 together with minors 1 (a target profiling now rejects: mirror process_dataset — remove it from the target role with the "Not used as target(s)" warning), 3 (export-based analyses must not be marked stale), 4 (vanished target listed twice), and the test gaps (assert the "Fields no longer in the data" warning; a live owner's refreshing dataset is left alone by the sweep). — Cost if wrong: a larger fix diff.
Task 5: minor (deferred): the swap holds ACCESS EXCLUSIVE through reconciliation; moving it later would shorten reader waits.
Task 5: minor (deferred): commit-failure window after os.replace (keeping raw.parquet.old until after commit would close it).
Task 5: minor (deferred): new fields have no experiment_decisions row, so they're invisible inside existing experiments' adjusted copies (follow-up).
Task 5: fix round 1/5 commit 26107f3 (5 items; rejected targets get a history row with user None; shared "Not used as target(s)" helper). Re-review runs as a Workflow adversarial panel (wf_c096b1ed-39e) — ultracode opted in by the owner from here on.
Ruling 8 (process): from Task 6 on, review gates are Workflow panels (lens reviewers + skeptic refuters) and independent tasks (6, 7, 9) run in parallel in isolated worktrees via a Workflow, then merge; Task 8 follows 6+7. — Cost if wrong: merge conflicts between the three branches (resolved by hand) and more tokens.
Task 5: fix round 1/5 panel (wf_c096b1ed-39e, resumed after a spend-limit interruption): 5/5 items addressed, 0 confirmed findings, 4 minor.
Task 5: minor (deferred): knock-on decision changes from dropping a target (leak flags clearing) get no history rows, unlike the interactive path.
Task 5: minor (deferred): rejected-target test asserts action != 'target' rather than action == the new suggestion.
Task 5: minor (deferred): warning assertion builds the expected string from the helper under test.
Task 5: minor (deferred): live-owner sweep test can't distinguish a live owner from any non-null owner.
Task 5: complete (commits d844dd8..26107f3, review clean)
Tasks 6/7/9: implementation workflow wf_8f8c6d67-4ce (resumed after the spend limit) — T6 6f117b9 (opus), T7 89e8fcd (sonnet), T9 e48440c (sonnet), each on 26107f3; merged as f0d56ae/64e9aa6 with no conflicts; merged tree: API 1022 passed, UI typecheck clean + 190 (4 pre-existing), chart 174/174, lint clean. Worktrees and branches removed.
T6 concerns: blocking export issue fires for every missing snapshot field incl. dropped ones; "pull #<n>" = global dataset_pulls id; PATCH experiments/runs metadata left unlocked; preview 500 on a vanished target fixed.
T7 concerns: compared_on not in the preview response, dialog states the rule in words; manual check against a mocked API only; T8 edits the same DatasetsPage/types regions.
T9 concerns: sample_mongo kept its positional __main__ plus main(argv) --if-empty; docs/configuration.md gained COMPOSE_PROFILES/MONGO_PORT rows; data-preparation.md describes refresh/lock/drift from the other tasks — re-read after merge; operator-managed Secret rotation needs an API restart (documented); live `make up-mongo` connected curated+massive → 750 rows.
Ruling 9 (T6): "pull #<n>" is the global pull id (equals pull_id) everywhere, UI included. — Cost if wrong: numbers look non-sequential per dataset.
Tasks 6/7/9: review panels run as one Workflow (wf_96aa466c-82b); Task 8 implementer ab1423f40ca53ed73 (sonnet) dispatched in parallel (edits merged UI files).
Panel run 1 (cut off by the spend limit): T6 no lenses completed; T7 lens3 → 4 minor (preview lacks compared_on so the dialog states the rule in words; connect's "Too large" 422 loses the numbers; per-collection picks/options survive a connection/database change; duplicate .connect-body CSS rule); T9 lens1+lens3 → minors (make up-mongo bypasses Compose's $$ unescaping for .env values; mongo-seed rebuilds the API image under the GPU override; UI Connect check not performed; a second `dev` source appended when .env names one; literal $$ doubled). No serious findings so far. Panel resumed; Task 8 agent resumed (partial uncommitted work present).
Task 8: implementer ab1423f40ca53ed73 (sonnet), commit 3400c69, DONE_WITH_CONCERNS (route is /d/<id>/baselines; reopen is comma-separated; role-only-gated controls rely on the API's 409; an already-open page shows the banner at its next fetch; live check via API + Vite + Chromium). Panel dispatched (wf_cef194f1-e63) — all three lenses failed on the spend limit; re-run in the fix-wave workflow.
Panel 6/7/9 (wf_96aa466c-82b, completed after resume): confirmed T6-1 export-rows 500 after a refresh removed a target (refuters cut off; fixer must reproduce with a test first); T6-2 export blocking fires for snapshot fields the baseline dropped; T7-1 Connect allowed with a Match on column the preview lacks (the pull fails late); T9-1 docs say the exports list shows the pull, the UI doesn't; T9-2 docs say targets stay targets, false since Task 5's fix. ~30 minors.
Ruling 10 (T6-2): the spec's rationale governs over the literal binding: the blocking issue covers only snapshot fields kept as features plus snapshot targets; a dropped field vanishing does not block. — Cost if wrong: a vanished dropped field goes unmentioned (the export is byte-identical anyway).
Ruling 11 (T7-1): match_on is validated at connect against the preview's merged sampled columns (422 naming the column and saying it was not found in a sample of up to 200 documents per collection), and in the dialog before Connect. — Cost if wrong: a sparse key that misses the sample can't be used as a match key.
Ruling 12 (compared_on): the preview endpoint returns top-level `compared_on` computed by the same rule as assemble (columns shared by every chosen collection's sample, minus _id/source/identifier-kind), or the validated match_on when given; the dialog renders it (spec §4 requires it). — Cost if wrong: the sample-based list can differ from the pull's real list on sparse columns (the pull records the real one).
Ruling 13 (T9-1): the spec requires the pull on the exports list, so the UI shows "pull #<n>" on export rows; the doc stays.
Ruling 14 (Makefile): `make up-mongo` sets RAINFOREST_MONGO_SOURCES only when .env doesn't define it; when .env does, it prints the entry to add instead of re-exporting the raw value (fixes both the $$ unescaping and the duplicate `dev`). — Cost if wrong: one manual .env edit.
Ruling 15 (folded minors): T6 run_refresh failure before its try leaves the dataset refreshing; targets PATCH while refreshing returns the documented lock message; a parametrised lock test over the guarded routes. T7 UI: "Too large" shows the numbers; per-collection picks and options reset on a connection/database change; a failed last pull shows as failed on the badge; duplicate .connect-body; rename `interface Pick`; blank-name explanation; inline validation marked aria-invalid/aria-describedby like existing dialogs. T9 docs: duplicated cleaning sentence; collision-rule wording; no drift summary before a finalized baseline.
Deferred (minor): PATCH experiments/runs metadata unlocked while refreshing (by design); export-rows and derived preview check status without the lock; dialog options after Preview; not-listable typed database vs pull database; Connect pending label; mongo profile and test-mongo both publish 27017; mongo-seed builds a second API image.
Fix wave: one Workflow — Task 8 panel re-run; backend fixer, docs fixer in parallel (isolated worktrees from 3400c69); UI fixer after the Task 8 panel (T7 UI + T9-1 + T8 findings); scoped re-review per fix branch.
Fix wave result (wf_29ce8136-d79, 24 agents): backend 3400c69..81c599d (5 commits), docs ..254644a, UI ..5b14654 (5 commits); every item verified addressed. T8 panel confirmed 6 (Team page hides PullCard when the queue has items; banner names a stale starter; a refresh ending on the list leaves stale caches; null drift says "no finalized baseline" wrongly; experiment workspace writable while refreshing; baseline setup form live while refreshing) — all fixed by the UI fixer. Re-review breakage: 1 Important (preview compared_on uses a different identifier rule than the pull), minors (UI shows the default compared_on after typing Match on; docs/api.md export-preview wording; Makefile shell-env case; deploy docs wording). Unjudged T8: Targets picker opens for a refreshing row.
Merged as de870c6 (conflict in docs/data-preparation.md resolved by hand: UI placement of the drift summary + "no summary before a finalized baseline"; export-preview sentence aligned with Ruling 10). Shared test Postgres recreated (a concurrent run had deactivated Admin). Merged tree: API 1059 passed, UI typecheck clean + 236 (4 pre-existing), eslint 0 errors (20 pre-existing warnings), nginx ok, chart 174/174, lint clean.
Ruling 16: the pull's deduplication uses the preview's identifier rule (identifier kind from value types and the ID name pattern, tracked across the whole pull) instead of profiling.infer_type, so the preview's compared_on is exactly what the pull compares (spec §4: "any column the preview flags as an identifier"). — Cost if wrong: a numeric key column without an ID-like name is compared (rows differing only in that key are not duplicates; Match on covers it).
Ruling 17: PATCH /experiments/{id} and PATCH /runs/{id} are refused while the dataset is refreshing, matching the UI and docs/api.md's lock list (supersedes the earlier deferral). — Cost if wrong: renaming an experiment waits for a refresh.
Round 2 + whole-branch panel: one Workflow.
Round 2 (wf_b740f5cd-635): fix 11 commits de870c6..deb711c, verifier 10/10 addressed. Whole-branch panel (resumed twice after spend limits): 7 lenses, refuters 3 per serious finding. Survived: 10 Important (baseline target removed → no new version; experiment exports skip the gone-field block; refresh loses retyped datetime format; missing/empty collection silently wipes rows/decisions; source labels can drift; failed pull hides live drift summary; failure handler not atomic; phantom adjustments; empty-string keys split; deleted fields truncated at 20). Both claimed Criticals rated Important by refuters. Refuted: Viewer preview oracle; docs-lens export claim. Triage agent not completed (owner turned ultracode off).
Ruling 18: a refresh fails when a configured collection is missing, or returns 0 documents after a previous successful pull returned >0. Ruling 19: Explore stays refused during a refresh; docs fixed. Ruling 20: collection labels are fixed at connect.
Final fix: single implementer from final-fix-brief.md (owner asked: no ultracode from here).
Final fix: 14 commits deb711c..0e4423e; migration 024 (recipes.start_summary). Re-review: 16/17 addressed; item 2 partly — gone-field check falls back to the baseline because experiment_decisions cascade on field delete (over-block A, silent missing feature B). 3 new minors.
Ruling 21: record per-experiment and per-assessment fields-in-use at removal time on the pull (flatten_notes.removed_in_use) and check exports against them. Ruling 22: renamed column names are seeded from the previous pull so they stay stable across refreshes. Fix round sent to the same implementer.
Fix round 2: 5b514ec, d51ab54, 69b46ab, c02d7b0 (pushed with docs as 77007e0; owner asked to push unreviewed). Re-review: 4/4 addressed; 1 Important regression — a personal experiment created after the removal exports unblocked (only pairs with an experiment were recorded); minors: per-pair snapshot cost inside the swap transaction, `names` passed positionally, reserved names push new keys to suffixes.
Ruling 23: every finalized version's assessors and every team experiment are recorded at removal time (empty lists included); a key with no record (created after the removal) falls back to its baseline's snapshot and blocks if the baseline keeps or targets the field — same as the default Team export. — Cost if wrong: an experiment created after a removal that the owner meant to export without the field must start from a new version.
Fix round 3 sent to the same implementer.
Fix round 3: 08a3e79, 2317f68, 871a162, 348c6f7; API 1097 passed. Ruling 24: item 4 of round 3 (release names reserved for absent paths) is reverted — it lets a different key take the old name and, through reconciliation by name, the old field's id and decisions; the permanent `_2` suffix is the lesser cost. My round-3 instruction was the error.
Rounds 3–4 re-review: ready to merge, 4/4 addressed, no correctness defects; minors: pairs CTE DISTINCT over JSONB (1.41 s → 0.02 s) inside the swap transaction, early return removed (empty records on every refresh), KeyPaths typed as tuples, docs wording of the reservation. Final polish round sent. Note: T7's commit 89e8fcd is on 117bd62 (main), not 26107f3 — UI-only files, merge clean; its review package is review-117bd62..89e8fcd.diff copied to the expected 26107f3..89e8fcd name.

Final polish: 015d6b8, 864f36a, 72fe23a, c83d2c5. Final head verified by the controller on a fresh database: API 1098 passed / 1 skipped, ruff clean, UI typecheck clean + 254 passed (4 pre-existing theme.test.ts failures), Prettier clean, ESLint 0 errors (20 pre-existing warnings), nginx ok, chart 174/174, lint clean, smoke.sh syntax ok. Branch reviewed and ready to merge.
