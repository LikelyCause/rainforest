# Profile accounts: pick who you are, from accounts in .env

Goal: no passwords. On first visit the app shows every account as a circle (a picture, an icon or initials) centred
on the page; you pick yours. Accounts and their roles (Team Lead, Team Member, Viewer) come from `.env`. Viewers can
look at everything the team shares and change nothing.

Decisions made with the owner (2026-10-11): build on the existing pick mode; the dev deployment has Tanner (Team
Lead), Test Member and Test Viewer; Tanner takes over dev-01's Lee Lead account and its history; viewers see a
baseline's decisions only once it reaches team review; dev-01 is LAN-only, so the picker needs no gate in front.

## 1. Accounts from .env

```
RAINFOREST_AUTH_MODE=pick
RAINFOREST_USERS=tanner#2:Tanner:lead;test-member:Test Member:member;test-viewer:Test Viewer:viewer
```

- Entries separated by `;`, each `key[#id]:Name:role`. `key` is a stable slug (`[a-z0-9-]+`); `role` is `lead`,
  `member` or `viewer` (shown as Team Lead, Team Member, Viewer).
- `#id` claims an existing account that has no key yet, keeping its id and so its history (assessments, comments,
  experiments, runs). After the first sync the key is stored and `#id` is ignored.
- The list is checked as a whole before anything is written: duplicate keys, an unknown role, a malformed entry or no
  lead stop the API from starting with a message naming the problem.
- Synced at every start, in one transaction under the existing start-up lock: a known key updates its name and role (a
  role change signs that account out); a new key creates an account; an account whose key isn't in the list, and any
  account without a key, is deactivated (never deleted: its work stays attached and readable). A key that returns
  reactivates the same account.
- With `RAINFOREST_USERS` set, accounts can't be added or edited in the app (`POST /api/users`, `PATCH` → `409`); the
  Team page lists them read-only and says they come from `.env`. Without it, pick mode works as today.
- Migration 015: `users.key TEXT UNIQUE`, `users.avatar TEXT` (picture hash), and `viewer` added to the role check.

## 2. The Viewer role

- **Enforced in one place:** the API's request check refuses every POST, PUT, PATCH and DELETE from a viewer with
  `403`, except: sign-in and sign-out, Explore analysis requests (served from cache or computed, never with
  *recompute*), the field finder query, and changing their own picture. The SQL console is refused: it reads every
  table, assessors' work in progress included.
- **MLflow:** nginx passes the original method to the auth check, so a viewer can browse MLflow but not change it.
- **What a viewer sees:** what the team shares. Personal assessments and private experiments stay hidden as today.
  While a baseline version is being assessed, a viewer doesn't see its in-progress team decision: the dataset's pages
  show its latest finalized baseline when it has one, otherwise each field's suggestion (as before anyone decided).
  From team review on, they see the team decision like everyone else. Explore follows the same rule.
- **Not offered to viewers:** assessor lists, *Join*, and every control that changes something (upload, samples,
  rename, targets, decisions, bulk edit, comments, baselines, experiments, export, train, recompute, revert, delete).
  One UI helper decides, so a hidden control and a `403` agree.

## 3. The profile picker

- Full page, centred both ways. Profiles wrap in centred rows (`flex-wrap`, a partial last row stays centred); circle
  size scales with the count (large for a few, smaller for many; it scrolls past about 12).
- Each profile: the circle (picture, else initials on a colour that stays the same per account), the name, the role
  badge (Team Lead in the accent colour, Team Member plain, Viewer outlined).
- Keyboard: Tab between profiles, Enter picks. One muted line under the grid: "Anyone who can open this page can pick
  any profile."
- Picking signs in and returns to the page you came from.

## 4. Profile pictures

- Header: your circle and name open a menu with your role, *Change picture…*, *Switch profile* (back to the picker)
  and *Team*.
- *Change picture…* opens a small dialog: drop or choose a file, see it in the circle, *Save* or *Remove picture*.
  Anyone may change their own picture (viewers too: it isn't project data); a Team Lead may change anyone's from the
  Team page. The picker itself never uploads.
- JPEG, PNG or WebP up to 5 MB, recognised by decoding the file (never by its name or declared type). It's turned
  upright, cropped to a centred square, resized to 256 × 256 and saved as WebP, which drops EXIF (location included)
  and anything hidden in the file. Images over 40 megapixels are refused.
- Stored under `DATA_DIR/avatars/` as `<user id>-<hash>.webp`; replacing deletes the old file. Served at
  `/api/auth/avatars/<user id>/<hash>.webp`, open before sign-in (the picker needs it), cached for a year (the hash
  changes with the picture), with `nosniff` and a sandboxing content policy.

## 5. dev-01

- `.env`: `RAINFOREST_AUTH_MODE=pick`, the `RAINFOREST_USERS` line above, and Keycloak's profile and OIDC settings
  removed; Keycloak is stopped (its data stays).
- Tanner takes over Lee Lead (id 2). Admin, Dana Scientist and Sam Scientist are deactivated with their work kept.
  Test Member and Test Viewer are new.

## Build

One PR: migration and `.env` sync, Viewer enforcement (API, nginx, UI), the picker, pictures, docs, then dev-01.

## Testing

API: parsing (each refusal, nothing written on a bad list); sync (claim by id, update, role change signs out,
deactivate, reactivate, no-lead refusal); a class test that walks every route and asserts a viewer gets `403` on each
write except the listed ones; viewer reads while assessing (finalized baseline, else suggestions) and from review;
pictures (each refused type including SVG and a disguised file, oversized, a pixel bomb, path tricks, headers, the old
file removed, only self or a lead). UI: the access helper for every role and action. Browser on dev-01: the picker
with 3 accounts (and a screenshot with many), sign in as each, a viewer sees no edit controls and a forced write is
refused, a picture upload shows in the picker and header.
