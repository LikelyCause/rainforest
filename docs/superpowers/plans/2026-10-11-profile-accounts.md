# Profile Accounts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accounts come from `.env`; you pick yours from a centred grid of circular profiles with pictures; a Viewer role sees what the team shares and changes nothing.

**Architecture:** Builds on pick mode. Startup syncs `RAINFOREST_USERS` into `meta.users` (stable `key`, claim by `#id`). One deny-by-default check in the request middleware refuses viewers' writes (with a short allowlist), and `source_for` gives viewers a read-only source while a version is being assessed. Pictures are re-encoded with Pillow into `DATA_DIR/avatars/` and served from an open, hash-named route. The UI gains a single `can()` helper, the picker page, a header profile menu and a picture dialog.

**Tech Stack:** FastAPI, psycopg 3, Postgres 16, Pillow, React 18, vitest, nginx.

**Spec:** `docs/superpowers/specs/2026-10-11-profile-accounts-design.md`

## Global Constraints

- `.env`: `RAINFOREST_USERS=key[#id]:Name:role;…`; key `[a-z0-9-]+`; roles `lead`, `member`, `viewer` (labels Team Lead, Team Member, Viewer).
- Invalid list (duplicate key, unknown role, malformed entry, no lead) refuses startup with a message; nothing written.
- Accounts are never deleted; removed keys and keyless accounts are deactivated (sessions ended) when `RAINFOREST_USERS` is set.
- Viewer writes: `403` except pick/logout, `POST /api/datasets/{id}/analyses` (recompute forced off), `POST /api/datasets/{id}/fields/query`, `POST /api/sql`, own picture `PUT/DELETE /api/users/{id}/avatar`.
- Viewers see an assessing version's decisions only from team review on: before that, the latest finalized baseline of the dataset, else the suggestions.
- Pictures: JPEG/PNG/WebP, ≤ 5 MB, ≤ 40 MP, centred square 256², WebP q85, `avatars/<id>-<hash16>.webp`, served at `/api/auth/avatars/<id>/<hash16>.webp` with year-long immutable cache, `nosniff`, `default-src 'none'; sandbox`.
- Self or a lead may change a picture; the picker never uploads.
- Docs current-state only; comments sparse. API tests `"$PT"`; UI `cd ui && npm run -s typecheck && npx vitest run`.

## Review Focus

1. **A write route added later** is closed to viewers without anyone remembering: pinned by `test_every_write_route_refuses_a_viewer` walking `app.routes` (Task 2).
2. **A viewer reading an assessing version through a side door** (field history, Explore analysis by id keyed `team`, export preview, field list): pinned by `test_viewer_never_sees_an_assessing_team_decision` (Task 2).
3. **Editing `.env` so the last lead disappears or a key is renamed**: startup refuses no-lead; a renamed key deactivates the old account and creates a new one (history stays with the old): pinned by `test_sync_*` (Task 1).
4. **A disguised or hostile picture** (SVG, HTML named .png, pixel bomb, 6 MB): refused, nothing written: pinned by `test_avatar_refuses_*` (Task 3).
5. **A stale session after a role change to viewer** must lose write access at once: role change bumps `session_version`: pinned by `test_sync_role_change_signs_out` (Task 1).

---

### Task 1: Accounts from .env

**Files:** Create `api/app/migrations/015_profile_accounts.sql`, `api/tests/test_profile_accounts.py`. Modify `api/app/config.py` (`users: str` from `RAINFOREST_USERS`), `api/app/users.py` (`parse_accounts`, `sync_accounts`, `pickable` with `key`, `avatar_url`), `api/app/db.py` (bootstrap calls the sync instead of seeding Admin when set), `api/app/routers/users.py` (`409` for POST/PATCH when accounts come from `.env`), `api/app/schemas.py` (role literals add `viewer`), `docker-compose.yml` (pass `RAINFOREST_USERS`), `.env.example`.

**Interfaces:**
- `users.Account(key: str, claim_id: int | None, name: str, role: str)`
- `users.parse_accounts(text: str) -> list[Account]` (raises `ValueError` naming the problem)
- `users.sync_accounts(conn, accounts: list[Account]) -> dict` (`{"created": [...], "updated": [...], "deactivated": [...]}`)
- `users.avatar_url(row) -> str | None` = `/api/auth/avatars/{id}/{avatar}.webp`
- `users.managed_by_env() -> bool`

- [ ] **Step 1: Failing tests** — parser: valid line → 3 accounts with roles and `#id`; each refusal (`"a:A:lead;a:B:member"` duplicate, `"x:X:boss"` role, `"bad"` malformed, `"Tanner:Tanner:lead"` uppercase key, `"m:M:member"` no lead) raises with the reason in the message. Sync (DB): with fresh keyless rows, `tanner#<id>` claims that row (same id, name becomes Tanner, key stored); another keyless row is deactivated and its `session_version` bumped; a new key inserts; a second sync with the same list changes nothing; removing a key deactivates; re-adding reactivates the same id; changing member→viewer bumps `session_version` (`test_sync_role_change_signs_out`); `POST /api/users` → 409 when managed; `GET /api/auth/users` lists key and `avatar_url`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** Migration 015: `ALTER TABLE meta.users ADD COLUMN IF NOT EXISTS key TEXT UNIQUE, ADD COLUMN IF NOT EXISTS avatar TEXT; ALTER TABLE meta.users DROP CONSTRAINT IF EXISTS users_role_check, ADD CONSTRAINT users_role_check CHECK (role IN ('lead','member','viewer'));`. Sync locks `meta.users` (SHARE ROW EXCLUSIVE), claims (`UPDATE … SET key=%s WHERE id=%s AND key IS NULL`), upserts by key (role change ⇒ `session_version = session_version + 1`), then `UPDATE … SET active=false, session_version=session_version+1 WHERE active AND (key IS NULL OR NOT key = ANY(%s))`, then reactivates listed keys. Bootstrap: parse first (a `ValueError` aborts startup with "RAINFOREST_USERS: …"); when set, sync replaces the Admin seed.
- [ ] **Step 4: Run** → PASS; full suite. **Step 5: Commit** `"Accounts from .env"`.

### Task 2: The Viewer role

**Files:** Modify `api/app/auth.py` (viewer gate in `check_request`; `/api/auth/check` honours `X-Original-Method`), `ui/nginx.conf` (`proxy_set_header X-Original-Method $request_method;` in `/_auth`), `api/app/assessment.py` (viewer branch in `source_for`; `Source.suggested` kind with `decisions()`/`settings()` from `meta.fields.suggestion`; `from_key("suggested:<rid>")`), `api/app/routers/explore.py` (viewer's source key; `recompute` off; `get_analysis` hides `team`-keyed results of an assessing version from viewers), `api/app/routers/fields.py` (`history` empty for a viewer while assessing), `api/app/routers/team.py` (viewers can't be assessors).

**Interfaces:** `auth.VIEWER_ALLOWED: list[tuple[str, re.Pattern]]`; `assessment.viewer_source(conn, dataset_id, recipe) -> Source`.

- [ ] **Step 1: Failing tests** (`test_profile_accounts.py`): `test_every_write_route_refuses_a_viewer` — for every route in `app.routes` with an unsafe method, call it as a viewer with a placeholder path (ids → 1) and an empty JSON body; assert `403` unless (method, path) matches the allowlist; the analyses/fields-query/sql routes are not 403. `/api/auth/check` with `X-Original-Method: DELETE` → 403 for a viewer, 204 for a member. `test_viewer_never_sees_an_assessing_team_decision`: dataset with a finalized Trees baseline (field `x` kept) and a v2 being assessed where the lead's assessment drops `x` and the team working copy drops `x` too: as viewer, `GET /fields/{x}` decision is `keep` (baseline v1), field list shows keep, history is `[]`, an Explore request's stored `decisions` key is `baseline:<v1>`; on a new dataset with no finalized baseline the decision equals the suggestion; after review opens the viewer sees the team decision. Assigning a viewer as assessor → 422.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (allowlist regexes anchored; the gate runs right after the session user is set). **Step 4: Run** → PASS; full suite. **Step 5: Commit** `"Viewer role: read-only everywhere, sees team review onwards"`.

### Task 3: Profile pictures

**Files:** Modify `api/requirements.txt` (`pillow==11.3.0` or the latest 11.x available), `api/app/config.py` (`avatars_dir`), `api/app/main.py` (create dir; gzip skip), `api/app/auth.py` (open GET prefix rule for avatars; multipart allowed on the avatar path), Create `api/app/avatars.py` (`store(user_id, data: bytes) -> str`, `remove(user_id)`, `path_for(user_id, hash) -> Path | None`), `api/app/routers/users.py` (`PUT/DELETE /api/users/{id}/avatar`), `api/app/auth.py` route `GET /avatars/{user_id}/{name}`.

- [ ] **Step 1: Failing tests**: a 400×300 PNG → 200, stored file is 256×256 WebP, URL in `/api/auth/users`, served with the three headers; replacing deletes the old file; refused: SVG bytes, HTML bytes named `.png`, 6 MB file (413), a 9000×9000 PNG (bomb, 400) — and no file written in each case; another member can't change someone else's (403), a lead can; a viewer can change their own; `GET /api/auth/avatars/1/../../x` and a wrong hash → 404.
- [ ] **Step 2-5:** run → FAIL; implement (read ≤ 5 MB + 1, `Image.MAX_IMAGE_PIXELS = 40_000_000`, `exif_transpose`, RGB/RGBA, `ImageOps.fit((256,256), LANCZOS)`, WebP q85, temp + `os.replace`); run → PASS; commit `"Profile pictures"`.

### Task 4: UI access helper

**Files:** Create `ui/src/can.ts`, `ui/src/can.test.ts`. Modify `types.ts` (`Role` adds `viewer`; `PickUser`/`Me`/`TeamUser` add `key`, `avatar_url`), `format.ts` (`roleLabel`), `team.ts` (`accessOf` viewer first), `hooks.ts` (`useAccess` viewer first), `experiments.ts` (`canEdit`, `addBlocked`), and the ungated controls: DatasetsPage (upload, drop, samples, rename, targets, delete), DatasetLayout (delete failed), ExperimentsPage (new experiment, revert), ExperimentWorkspace (export, train, Train again), DatasetTeamPage (comments, assessor lists exclude viewers), StartBaseline (assessors exclude viewers), explore/state.tsx (recompute), and the inline `role === "lead"` checks via `can`.

**Interfaces:** `can(me: {role: Role} | null | undefined, action: Action): boolean` where `Action = "upload" | "editDataset" | "decide" | "comment" | "createExperiment" | "export" | "train" | "recompute" | "startBaseline" | "assign" | "finalize" | "deleteDataset" | "manageUsers" | "join" | "changeOwnPicture"`; `LEAD_ONLY` set.

- [ ] Test: viewer false for all but `changeOwnPicture`; member true for non-lead actions, false for lead-only; lead true for all. Implement, wire, typecheck, commit `"One access helper; viewers get no edit controls"`.

### Task 5: The picker, profile menu and picture dialog

**Files:** Modify `ui/src/pages/LoginPage.tsx` (pick mode → `ProfilePicker`), Create `ui/src/components/Avatar.tsx` (circle: image or initials on `--cat-N` from a stable hash), `ui/src/components/PictureDialog.tsx`, Modify `App.tsx` (header profile menu), `TeamPage.tsx` (read-only when managed by `.env`; lead can change anyone's picture), `api.ts` (`setAvatar`, `removeAvatar`), `styles.css`.

- [ ] Tests (`avatar.test.ts`): `initials("Test Member") === "TM"`, `initials("tanner") === "T"`, `tint(id)` stable and within 1..8. Picker: `.profile-grid` flex-wrap centred, circle `clamp(72px, 14vmin, 128px)`, name, role badge; muted note. Header: avatar + name button → popover (role, Change picture…, Switch profile, Team). Dialog: file input + drop, preview, Save/Remove, errors from the API. Commit `"Profile picker, profile menu and pictures"`.

### Task 6: Docs, dev-01, verification

- [ ] README (sign-in section and permissions table: Viewer column), `docs/api.md` (auth/users/avatar routes, viewer 403s), `docs/architecture.md` (accounts sync, viewer gate, avatars), `.env.example` (`RAINFOREST_USERS` example). Full suites; PR; CI.
- [ ] dev-01 `.env`: `RAINFOREST_AUTH_MODE=pick`, `RAINFOREST_USERS=tanner#2:Tanner:lead;test-member:Test Member:member;test-viewer:Test Viewer:viewer`; drop `COMPOSE_PROFILES=keycloak` and `RAINFOREST_OIDC_*`; `make up`; stop Keycloak.
- [ ] Browser (Playwright): picker shows 3 centred circles; sign in as each; Tanner sees Lee Lead's history; viewer sees no edit controls on datasets/review/experiments and a forced `PATCH` is 403; upload a picture as Test Member → shows in header and picker; screenshot a picker with 12 accounts (temporary env on a test API, not dev-01). Ultracode review workflow before merge.
