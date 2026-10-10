# 🌲 Rainforest

Rainforest prepares wide, mixed-type data for **tree-based models**. Ingest a dataset, pick its targets, review it
**one field at a time** in a fast UI, store keep / drop / imputation / encoding decisions in Postgres, explore the data
with correlations, embeddings and per-field tests, export a **model-ready** dataset, and train a **`ClassifierChain`**
of LightGBM (or another tree ensemble) over your targets.

Trees are the default and the main path. Rainforest knows that tree ensembles split on value *order*, so it never
suggests scaling or log transforms for them, and it leaves missing values for the model to route. Linear and neural
model families are also available. When you switch to one, it gets its own preprocessing settings (scaling,
imputation, one-hot encoding) without touching the tree settings.

```
┌────────────┐  /api/*   ┌──────────────────────────┐        ┌────────────────────────────────┐
│  ui        │ ────────► │  api (FastAPI)           │ ─────► │  db (Postgres 16)              │
│  React +   │  nginx    │  ingest · profile ·      │  SQL   │  meta.*  fields, stats,        │
│  ECharts   │  proxy    │  correlate · explore ·   │        │          decisions, settings,  │
│  :8080     │           │  export · train chains   │        │          exports, model runs,  │
└────────────┘           └─────────┬────────────────┘        │          analyses              │
                                   │ /data volume             │  raw.ds_<id>  your data        │
                                   │ Parquet copies, exports, └────────────────────────────────┘
                                   │ trained models
```

Developer docs: [docs/architecture.md](docs/architecture.md) (how the pieces fit, where to add things) and
[docs/api.md](docs/api.md) (every HTTP endpoint).

## Quick start

Needs Docker with Compose v2.

```bash
cp .env.example .env    # then set POSTGRES_PASSWORD and READONLY_PASSWORD
make up                 # docker compose up -d --build; the first build takes a few minutes
```

Set the passwords before the first start: Postgres takes them only when it creates its data volume.

Then open `http://localhost:8080` (the Docker host's address from another machine; `UI_PORT` sets the port).
Everything `make up` starts is configured in `.env`, which Compose reads on every command; see
[Configuration](#configuration).

Out of the box sign-in is in *pick* mode: the sign-in page lists the team and you choose who you are, starting with
the built-in lead **Admin**. That's for one person, dev and demos; it doesn't authenticate anyone. A team sets up OIDC
sign-in: see [Deploying for a team](#deploying-for-a-team) (`make up-keycloak` runs a ready-made Keycloak to try it).

1. **Datasets** → upload a CSV / Parquet file, or use **Generate sample ▾** (1k–2M rows, 30–400 fields, at most
   60M cells, with `column_a` / `column_b` targets that are binary + 3-class, both binary or both multiclass). On an
   empty install, *Open the 50k × 150 demo* does the same in one click. The upload is converted to Parquet, loaded into
   `raw.ds_<id>`, profiled and correlated, with live progress.
   Then **Choose target ▾** on the dataset's row (once set, click the target chips to change them): pick one or more
   columns, or *No target (unsupervised)*. A target must be binary, categorical or discrete numeric with at most 100
   distinct values; other fields are disabled in the picker with the reason, and the API refuses them. Saving
   refreshes the target associations and leakage flags. Generated samples already have their two targets set. The row
   menu also renames or deletes a dataset (deleting removes its fields, exports, model runs and their files).
   A new dataset starts as recipe v1 in **Assessing**: a lead assigns assessors (row menu → *Assessors…*, or the
   **Team** step), each of them reviews every field blind, and the team settles the disagreements before a lead
   finalizes. See [Team workflow](#team-workflow).
2. **Review** → walk the fields. Each field shows its signals (missing, skew, outliers, levels, target association
   with a leakage flag), its distribution and quantile strip, top tokens or weekdays, and its strongest
   **associations** (click a bar to jump to that field). Decisions autosave. The sidebar filters by status
   (*To do* · *Kept* · *Dropped* · *All*), by type, and by *Needs attention* (unreviewed fields with a leakage flag,
   mostly missing or constant, with low type confidence, or whose save failed).
3. **Bulk edit** → find fields with presets (*Unreviewed*, *Missing > 20%*, *Skewed numerics*, *Target-associated*,
   *High-card categoricals*, *Text*, *Possible leaks*, *Edited by you*) or your own filters, then **Accept
   suggestions**, **Keep**, **Drop** or **Mark reviewed**. Each bulk action shows a preview of what will change before
   you apply it, and can be undone from the toast. Bulk actions never move a field into or out of the target role,
   and *Keep* passes over identifier and constant fields. The table shows the first 500 matches; bulk actions apply
   to every match except the rows you untick, and the bar says so. *Accept suggestions* leaves preprocessing settings you
   customised alone unless it changes the field's type. *Review these N →* (`R`) opens the shown fields as a
   working set in Review. The **SQL** tab is a read-only console with examples.
   A field counts as *edited by you* once you change its keep / drop or type, and its settings as *customised* once
   they differ from what was suggested; confirming or annotating a suggestion leaves it a suggestion. Decisions still
   at the suggestion and not yet reviewed follow target changes (a new leak flag drops them).
4. **Explore** → correlations heatmap, a 2-D map of rows, per-field statistical tests against a target, and feature
   importance. See [Explore](#explore).
5. **Experiments** → in an experiment, fit the preprocessing on the training split, write the model-ready
   Parquet, and train chains with any compatible model. Runs on an export are compared side by side, and the best on
   the validation split is starred. See [Experiments](#experiments).

The top bar shows the dataset switcher, the workflow steps **Review · Bulk edit · Explore · Team · Experiments**
(Review carries the reviewed count, the Team step the dataset's phase: *Assessing*, *Review* or *Finalized*), the
**Model** switch (*Trees* · *Linear* · *Neural*), *Next unreviewed* (`U`) while fields are left, and on the right
**Team** (the user list), who you're signed in as with your role, and *Sign out*.

### Keyboard shortcuts (Review)

| Key | Action |
|---|---|
| `J` / `→` · `K` / `←` | next / previous field |
| `U` | next unreviewed field (also jumps into Review from the dataset's other pages) |
| `X` | toggle keep / drop |
| `Enter` | confirm and go next |
| `A` | use the suggestion and confirm |
| `Z` | undo the last confirm |
| `N` | add a note |
| `/` | search fields |
| `Esc` | leave a control, back to review |
| `?` | show or hide the shortcut list |

## Sign-in

Everyone signs in as a user, and everything they change is recorded against them (decision history, bulk edits,
uploads, exports, model runs, Explore analyses). A user is a **lead** or a **member**:

| | Member | Lead |
|---|---|---|
| Upload, Explore, experiment (export a finalized recipe or your own assessment, train) | ✓ | ✓ |
| Review and bulk edit their own assessment when assigned; comment in team review | ✓ | ✓ |
| Assign assessors, set the team decision in review, finalize | | ✓ |
| Delete datasets | | ✓ |
| Manage users on the **Team** page | | ✓ (pick mode: add, rename, edit email, set role, deactivate / reactivate; OIDC: deactivate / reactivate) |

`RAINFOREST_AUTH_MODE` chooses how people sign in:

* **`oidc`** (team deployments). Standard OpenID Connect against your identity provider: authorization-code flow
  with PKCE, run by the API. The sign-in page shows *Sign in with <provider>*. A user is created the first time they
  sign in, and their name, email and role are refreshed from the ID token every time. The role comes from the
  token's groups claim: `RAINFOREST_LEAD_GROUP` (default `rainforest-lead`) makes a lead, `RAINFOREST_MEMBER_GROUP`
  (default `rainforest-member`) a member. Someone in neither group, or deactivated, gets a *No access* page. *Sign
  out* also ends the provider's session. Register Rainforest at the provider as a confidential client with the
  redirect URI `<Rainforest URL>/api/auth/oidc/callback` (and `<Rainforest URL>/login` as a post-logout redirect URI),
  and configure a groups claim in the ID token.
* **`pick`** (the default; dev, demos, a single person or a trusted single-team network). The sign-in page lists the
  active users and choosing one signs you in as them; the page says plainly that this doesn't authenticate. Leads
  manage the list on the Team page. The first start creates one lead, *Admin*.

Deactivating a user, or changing their role, signs them out everywhere at once. The last active lead can't be
demoted or deactivated, and the Team page doesn't offer to deactivate yourself. Users are never deleted, so their
history keeps its names; matching is by the provider's subject id, never by email.

## Deploying for a team

`.env` is the whole configuration: Compose reads it on every command, so `make up` (and `make down`, `make logs`, …)
always applies it, and it is what you keep and back up besides the data volumes. A team deployment sets, in `.env`
(the commented *team deployment* block in `.env.example` has all of them):

1. **Sign-in:** `RAINFOREST_AUTH_MODE=oidc` and the `RAINFOREST_OIDC_*` settings for your provider (see
   [Sign-in](#sign-in) and [Configuration](#configuration)). To run the reference Keycloak alongside, add
   `COMPOSE_PROFILES=keycloak`, set `KEYCLOAK_URL` and `RAINFOREST_URL` to the URLs browsers use, and point the OIDC
   settings at it (see [Keycloak](#keycloak)).
2. **GPU** (optional): `COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml` builds the API with CUDA XGBoost and
   gives it the host's NVIDIA GPUs (see [GPU](#gpu)).
3. **Access:** `RAINFOREST_ALLOWED_HOSTS` set to the names and addresses people use, and real passwords for Postgres
   and the console role. Anywhere beyond a trusted LAN, also put a TLS proxy in front of the UI port (see
   [Operational notes](#operational-notes-and-limits)).

Then `make up`. Any combination works, Keycloak with the GPU image included, and every later `make up` keeps it.

`make up-keycloak` and `make up-gpu` are shortcuts for trying an option without editing `.env`: they start the stack
as `.env` describes plus the GPU image, or plus the Keycloak profile with the reference Keycloak's sign-in settings
for any that `.env` leaves empty (none when `.env` names its own issuer), for that run only, and print what to add to
`.env` to keep it. The next `make up` starts the stack as `.env` describes. `make up-keycloak` stops with an error
when `.env` sets `RAINFOREST_AUTH_MODE` to something other than `oidc`.

### Keycloak

The `keycloak` profile runs [Keycloak](https://www.keycloak.org/) on `:8081` (`KEYCLOAK_PORT`) as a reference
identity provider. Keycloak keeps its own `keycloak` database in the same Postgres and imports the `rainforest`
realm on first start (`keycloak/realm-rainforest.json`): the confidential client `rainforest`, the groups
`rainforest-lead` and `rainforest-member` mapped into the ID token's `groups` claim, and these users:

| User | Password | Group |
|---|---|---|
| `lead1` | `lead1-dev-password` | rainforest-lead |
| `ds1` | `ds1-dev-password` | rainforest-member |
| `ds2` | `ds2-dev-password` | rainforest-member |
| `guest1` | `guest1-dev-password` | none (shows the *No access* page) |

With a fresh `.env`, `make up-keycloak` runs it at `http://localhost:8081` and signs Rainforest in against it.
Keycloak's own admin console is at `<KEYCLOAK_URL>/admin` (`admin` / `KEYCLOAK_ADMIN_PASSWORD`, default `admin`).
Keycloak takes a minute or so to start, and signing in fails until it is ready.

**This is a development setup, not a production one:** Keycloak runs in `start-dev` mode over plain HTTP, the
client secret (`rainforest-dev-secret`) and passwords above are public, and the realm allows HTTP from anywhere.
For real use, run Keycloak (or any OIDC provider) in production mode behind TLS, and set the `RAINFOREST_OIDC_*`
settings for it.

Tokens name their issuer by the URL browsers use, so Keycloak runs with `KC_HOSTNAME` set to `KEYCLOAK_URL` and the
API checks tokens against the same issuer. The API container can't reach that URL when it names `localhost`, so it
reads discovery from `http://keycloak:8080` (`RAINFOREST_OIDC_DISCOVERY_URL`), and Keycloak's dynamic backchannel
(`KC_HOSTNAME_BACKCHANNEL_DYNAMIC`) answers it with token and key URLs on that internal name while the issuer and
the browser-facing URLs stay external. For use from other machines, set both URLs in `.env` as those machines see
them, e.g. `KEYCLOAK_URL=http://rainforest.lan:8081` and `RAINFOREST_URL=http://rainforest.lan:8080`, with the issuer
and redirect URL to match (`.env.example` shows the set). `make up-keycloak` derives the issuer and redirect URL from
the two URLs when `.env` leaves them empty, and takes them on its command line too (`make up-keycloak
KEYCLOAK_URL=… RAINFOREST_URL=…`).

The redirect URIs are written into the realm on its first import only. After changing `RAINFOREST_URL`, edit the
`rainforest` client in Keycloak's admin console, or reimport: stop Keycloak (`docker compose stop keycloak`), drop its
database (`docker compose exec db dropdb -U <POSTGRES_USER> keycloak`) and start it again (`make up` with the profile
in `.env`, or `make up-keycloak`).

## Team workflow

Every dataset has **recipes**: numbered versions of its field decisions. A version moves through three phases, shown
on the Datasets list and in the top bar.

1. **Assessing.** A lead assigns assessors on the dataset's **Team** page (also *Assessors…* in the Datasets row
   menu). Each assessor starts from the suggestions and uses Review and Bulk edit as usual, but on their own
   assessment: nobody else sees their choices, the lead included, and the API never returns them to anyone else
   (field lists, field details, history, Bulk edit, the SQL console, Explore). Leads see each assessor's progress
   (fields reviewed, submitted or not), never their choices. Everyone else can browse the fields with their
   suggestions and use Explore, read-only. *Submit assessment* needs every field reviewed; an assessor can unsubmit
   until review opens, and a lead can unassign someone (their choices are kept but don't count). Targets are set
   before anyone is assigned, since every assessor starts from them.
2. **Team review** opens by itself when the last assigned assessor submits (or a lead unassigns the last one who
   hadn't). Fields where every assessor chose the same keep / drop / target and type are **adopted** as the team
   decision; each model family's settings take the most common choice, and a tie takes the suggestion. The rest
   form the **review queue** on the Team page: each assessor's choice, settings and notes side by side, with the
   differences marked, and a comment thread. Everyone can comment; a lead sets the team decision (the same decision
   panel as Review). Bulk edit is for leads only in this phase and changes team decisions.
3. **Finalized.** A lead finalizes once the queue is settled, or confirms finalizing with fields left as they stand.
   Finalizing freezes the version: its snapshot records every field's decision, its settings for every model family,
   its inferred type and key stats, and the target order. A finalized version's decisions can't change; the way to
   change them is a new version.

**New versions.** On a dataset's **Versions** page (linked from the Team page) a lead starts the next version from
the latest finalized one and picks the fields to reopen, with search and the same filters as the Review list (type,
*Needs attention*, name); none are picked by default. Reopened fields go through blind assessment and team review
again with the assessors the lead assigns: assessors only assess those, *Submit assessment* needs just those reviewed,
and the review queue holds only disagreements among them. Every other field is **carried over and locked** (🔒 in
the Review list; the decision panel says why): nobody changes it in that version, the lead included. With nothing
reopened the version opens straight in team review and the lead edits the team decision directly (recorded in each
field's history) and finalizes. The Versions page lists every version (who started and finalized it and when, its
phase, how many fields it reopened, what it was built from) and compares any two: the fields whose decision or
settings differ. Only team decisions are compared; a version still being worked is compared as its team decision
stands, never with anyone's assessment.

**Starting from a recipe.** A new dataset (before anyone is assigned to it) can start from any finalized recipe of
any dataset: *Start from a recipe* on its Versions page (the upload toast links there for leads), or `recipe_id`
on the upload API. Fields match by name. A matched field whose inferred type is unchanged and whose stats haven't
drifted (missing share within 10 points, distinct values within half to double) is carried over and locked, with
the recipe's decision, settings and notes. Everything else is reopened for assessment: new fields, changed types,
drift (with the numbers), a recipe target that isn't a valid target here, settings that aren't valid for this data.
Recipe fields missing from the new data are listed. The recipe's targets carry over when they're still valid targets
here. The summary stays on the dataset's Team and Versions pages, and the version records the recipe it came from.
If nothing needs assessing, the version opens in team review and a lead confirms it by finalizing.

**Exports** live in [experiments](#experiments). A team export is built from a finalized version's snapshot, never
from the decisions being worked on, so an old version always exports the same way while a newer one is in progress;
an assessor can also export their own assessment, privately, while they assess. Explore is shared: raw-data analyses
use the team decision, except that while assessing, *Kept only* uses your own choices, and results are cached per
decision source so they never mix people's choices.

## Experiments

Exports and the model runs trained on them belong to an **experiment**: the dataset's **Experiments** step lists
them (*All*, *Mine*, *Team*, *Archived*) with their export and run counts and their best run on the validation split.
*New experiment* asks for a name and what it is built from:

* **A finalized recipe version** (a team experiment). Everyone sees it and anyone can add exports and runs to it.
  Exporting a finalized version from the API without naming an experiment puts the export in that version's
  **Team · vN** experiment, made on first use.
* **My assessment (vN)** (a personal experiment), offered while you are an assessor of the version being worked.
  Its exports use your own choices as they stand when you export; fields a later version carried over (locked) take
  the team's values. The export freezes that snapshot, so it rebuilds the same way after you change your assessment.
  Only you add exports and runs to it.

**Privacy.** A personal experiment is **private** (🔒) while its recipe version is assessing: nobody else sees it,
the lead included, nor its exports, runs, files, Explore results on its exports or anything about it in the Datasets
list, and opening one by URL answers as if it didn't exist. From team review onwards it is visible to the whole team, like
the assessment it was built from. The SQL console never shows a private experiment, its exports, runs or metrics
(row-level security; the console doesn't know who is asking, so this includes the owner).

An experiment's page shows how it was built and whether it is private, its notes, and the export and train panel
scoped to it: build an export from its source, train on it, compare the runs, and *MLflow ↗* for its runs there. Its
owner (for a personal one, its assessor) or a lead renames it, edits its notes and archives or unarchives it; an
archived experiment takes no new exports or runs.

## Decisions: shared vs per model family

| Shared by every family | Per family (`meta.field_settings`) |
|---|---|
| keep / drop / target, semantic type, notes, reviewed | imputation, missing-indicator, transform, scaling, encoding, encoder params |

The top-bar **Model** switch selects which family's settings you're reviewing. It defaults to *Trees*. Each family has
its own suggestions, so switching never overwrites another family's choices. If you change a field's type, the other
families' settings are re-derived for the new type.

| | **Tree ensembles** (default) | Linear models | Neural / distance |
|---|---|---|---|
| Models | LightGBM, XGBoost\*, histogram GBM, random forest | logistic regression | MLP |
| Missing numerics | left as NaN; the model learns where to send them | median, plus a "was missing" flag from 1% missing | same as linear |
| Skew / outliers | nothing: splits only depend on order | signed log1p when \|skew\| > 1, else winsorize above 3% outliers | same as linear |
| Scaling | **not offered**: it can't change a split | standard | quantile-normal when \|skew\| > 1, else standard |
| Categoricals | ≤ 8 levels one-hot, ≤ 1,000 ordinal codes, else frequency | ≤ 200 levels one-hot (top 30 kept), else frequency | same as linear |
| Binary | 0/1; missing takes the most frequent level under 5% missing, else a "was missing" column | same | same |
| Text | TF-IDF → 32 SVD components | same | same |
| Dates | year, month, day, weekday, hour, day-of-year, epoch days | + median impute + scaling | same |

\* XGBoost trains on the CPU in the default image and on an NVIDIA GPU in the GPU image (see [GPU](#gpu)).

In the UI, settings a family doesn't use appear as a locked line with the reason, for example *"Not used for tree
models — they split on value order, so rescaling changes nothing."* The API rejects them too
(`scaling='standard' is not valid … with tree models`).

**Type inference** (`api/app/profiling.py`) proposes `numeric_continuous`, `numeric_discrete`, `categorical`,
`binary`, `text`, `datetime`, `identifier` or `constant`, with a confidence score and reasons. It detects numbers stored
as text, dates stored as text (15 formats, including `YYYYMMDD` as text or integers), integer and UUID keys, and free
text compared with short categorical strings.
* Parse rates are measured over the whole column, and formats are chosen from values spread across the file, so a
  clean head can't hide junk further down.
* Commas are thousands separators only in valid groups (`"1,234.50"`). A column that writes decimals with a comma
  (`"1,5"`, and no dot decimals) is read as decimals. Any other value with a comma doesn't parse.
* `dd/mm` vs `mm/dd` is decided by values with a part above 12. When nothing decides it, month-first is assumed and
  the reasons say so.
* Values that don't parse as the inferred type, and ±inf, are counted in **Missing** because the export reads them as
  missing. They're also reported as `stats.unparseable_count` (with examples) and `stats.inf_count`, shown as a *Read
  as missing* signal. Unparseable values cap the type confidence at 0.65.
* Integer columns retyped to datetime are read as `YYYYMMDD` when they are valid dates, otherwise as epoch
  timestamps whose unit (s, ms, µs, ns) follows from their magnitude. Whole numbers that all fall within one
  unit's 2000–2100 epoch range are inferred as dates in the first place, unless the name looks like a key
  (`…_id`, `key`, `code`).
* An upload target with more than 100 distinct values isn't made a target; the dataset gets a warning.

Fields with a ≥ 0.95 association to a target are flagged as **possible leakage** and suggested for dropping.

**Associations** use a seeded 100k-row sample:
- Spearman ρ for numeric pairs, over the rows where both are present. One vectorised pass ranks each column over
  its own values; pairs whose missing rows differ and whose first estimate is at least 0.3 are re-ranked within
  their shared rows, so a monotone relation isn't understated when values are missing.
- Bias-corrected Cramér's V for categorical pairs, with missing as its own level. Each categorical keeps its 50
  most common levels (`CORR_MAX_LEVELS`) and groups the rest.
- Correlation ratio η for numeric × categorical pairs.
- **Leak screen:** categoricals with more levels than that, and text fields (compared by exact value), are also
  measured against each target with every level kept. That value replaces the capped one, so a 500-level code that
  determines the target is flagged. It runs during processing, and again in the background whenever a target is
  added.

## Explore

Four views over one dataset, sharing a toolbar whose state lives in the URL:

* **Data:** *Raw fields* (with a *Kept only* toggle) or a finished export (its train split, features as encoded).
* **Target:** one of the dataset's targets. Without one, the views that need it say so and link to the target picker.
* **Sample:** 1k / 5k (default) / 10k / 20k rows, seeded, stratified on a categorical target. Rows with no target
  value are left out.

In raw mode the fields used are the kept fields (*Kept only*) or every non-target field, typed by their review
decision. Text, identifier and constant fields, and fields constant in the sample, are skipped and listed.
Categoricals keep their top 20 levels plus *Other* and *(missing)*; dates become epoch seconds. A numeric target with
more than 20 distinct values is treated as numeric, otherwise as classes.

| View | What it computes |
|---|---|
| **Correlations** | Clustered heatmap of the stored associations (no job, raw data only): the target and the 59 fields most associated with it, or the 60 with the strongest associations overall, ordered by average-linkage clustering on 1 − strength. A slider hides weak cells, you can add fields beyond the 60, and clicking a cell shows the pair with links to Review. |
| **Map** | 2-D embedding of the sampled rows, coloured by the target. **t-SNE** (default; PCA to 50 components first when there are more features, perplexity 5–100, default 30) or **UMAP** (neighbours 5–100, default 15), fixed seed. Numerics are median-filled and standardised, categoricals one-hot. Hover shows the 5 fields that spread the rows most (largest weight in the first two principal components). |
| **Feature tests** | One test per field against the target, with Benjamini–Hochberg adjusted p-values (the UI marks adjusted p < 0.05). Numeric field × class target: one-way ANOVA (effect η²) plus Kruskal–Wallis, box plots per class. Categorical field × class target: χ² (effect Cramér's V), level counts per class. Numeric target: Spearman (effect \|ρ\|) plus Pearson, a scatter of up to 500 points; or ANOVA of the target across the field's levels. Classes with fewer than 5 sampled rows are excluded and listed. |
| **Importance** | Mutual information of each field with the target (needs a target), and PCA: explained variance for up to 20 components with the cumulative line, plus loadings of the first 5 components for the top 20 features. |

Map, Feature tests and Importance run as background jobs and are cached in `meta.analyses` per parameter set. A
result is tied to the decisions it was computed from (every field's keep / drop / target and type; for export data,
the export itself). After you change a decision, the cached result is shown as **Out of date** with a *Recompute*
button rather than recomputed automatically. A new request for the same view supersedes one still running. When the
sample can't support an analysis (no target, too few rows or fields, a single class), the view explains why instead
of failing.

## Export → ClassifierChain

Exports are built from an experiment's source: a finalized recipe version's snapshot, or for a personal
experiment its assessor's assessment frozen at export time (see [Experiments](#experiments)). Each export records
the recipe (`recipe_id`, shown as *recipe vN* in the list) and its experiment, and its manifest's `config` the
`source` (`recipe` or `assessment`), recipe id and version, `experiment_id` and, for a personal export, `assessor_id`.

* **Leakage-safe split:** the train / valid / test split is assigned first, stratified on the target combination
  (single-row combinations are pooled; if the combinations still can't be stratified, on one target, else not at
  all). A target with a class that has a single row blocks the export, naming the class.
  Every statistic is then fitted on training rows only (up to 500k of them by default), and the full file is
  transformed in chunks.
* **Memory:** both steps are sized by output width. A transform chunk holds at most `EXPORT_CHUNK_MB` of dense
  output (rows × features × 8 bytes, 256 MB by default), and the fit sample at most `EXPORT_FIT_MB` (512 MB), so a
  wide export fits on fewer rows (the manifest records `fit_sample_rows` and `chunk_rows`). Peak memory runs at a
  few times these budgets: a 300k-row export peaks at about 1.3–2 GB.
* **Output:** float32 features, `__split`, `__row_id` (joins back to `raw.ds_<id>`), label-encoded targets, the fitted
  `preprocessor.joblib`, and `manifest.json` (feature → source field, class mappings, model family, NaN report, and
  `inf_values`: ±inf per field, written as missing). Feature names never take a target's name or `__row_id` /
  `__split`; a clashing feature gets a numbered suffix.
* **Encodings worth knowing:** a binary field becomes one 0/1 column named after its second level (`smoker_yes`).
  Missing values take the most frequent level, or the level you name as the fill value; any other fill label adds a
  `<field>__isna` column instead of a third code. Numeric fill values are applied before log1p or winsorizing, so
  they're in the field's own units. A field retyped to date / time has its date format detected once over the whole
  column when you save, so every chunk of the export reads it the same way.
* **Before you export**, the page summarises what will go in (kept / dropped / targets / customised fields, kept
  fields by type) and warns about unreviewed fields, kept leakage flags and a lone target (nothing to chain). No
  target, no kept features or a single-row target class blocks the export. Once an export exists, it shows how many
  decisions changed since, with *Re-export with same split*. Validation and test share, seed and the fit-row cap are adjustable.
* **Compatibility:** tree models train on any export. Linear and neural models need an export prepared for their
  family, and they're refused if the export contains NaN. The UI disables incompatible models and shows the reason.
* **Chain:** `ClassifierChain(model, order=<targets>, cv=5)`, in the dataset's target order unless you choose another
  (the UI offers both orders when there are two targets).
  * The chained feature is **out-of-fold** during training, so a later target's model learns from realistic
    predictions of the earlier ones rather than the ground truth.
  * All targets binary → P(class 1) is passed down the chain; any multiclass target → the predicted class code.
  * NaN-capable models are wrapped so sklearn's finiteness check doesn't reject the NaN they handle natively.
* **Classes missing from training** (a rare class that only landed in valid or test) keep their own probability
  column (zero), so metrics stay aligned with the class names; the report lists them under `missing_from_train`. A
  target with a single class in the training split stops the run before fitting.
* **Metrics** per target on the valid and test splits: accuracy, balanced accuracy, log-loss, ROC-AUC (one-vs-rest
  macro for multiclass), precision / recall / F1 (macro and weighted, plus positive-class and specificity for binary
  targets), confusion matrices and a per-class table, plus the top importances per chain link.
* **On the page:** each export has *Train ▾* (model and chain order), *Files ▾* (Parquet, preprocessor, manifest) and
  *Use from code* (the CLI command and class mappings). Runs are listed per export, with that export's targets as
  columns, showing test-split AUC and macro-F1. In each of these columns ★ marks the run that scored best on the
  validation split for that target and metric, so choosing it doesn't use the test split. A run expands into its full metrics and confusion matrices.

```bash
make train EXPORT=1                          # LightGBM
make train EXPORT=1 MODEL=random_forest
docker compose exec api python -m app.modeling.train --export-id 1 --model xgboost \
    --order column_b,column_a --param n_estimators=1000 --param learning_rate=0.03
make train EXPORT=1 MODEL=xgboost DEVICE=cuda  # GPU image only
make models                                  # what's registered / installed, and on which devices
```

`--order` takes the export's own target names (the sample's are `column_a` and `column_b`). Runs from the CLI write
`classifier_chain.joblib` and `report.json` under `/data/models/export<id>_<model>_<time>/` but aren't recorded in
`meta.model_runs`, so they don't appear in the UI's runs table or `meta.v_model_metrics`.

**Adding a model** means adding one `ModelSpec` to `api/app/modeling/registry.py`: key, label, family, NaN support,
builder, default parameters and how to read importances, plus optionally a description, the package it requires and
`cuda_params` for GPU training. It then appears in the CLI, the API and the UI.

## GPU

XGBoost can train on an NVIDIA GPU. The default image ships the 5 MB CPU build (`xgboost-cpu`); the GPU image swaps
in the CUDA build (`xgboost`, built with CUDA 13.3), which adds about 800 MB, mostly its NCCL dependency. Training
on the GPU is a per-run choice: when the API sees a usable GPU, *Train ▾* shows a CPU / GPU switch with the GPU's
name for models that support it (XGBoost: `device="cuda"`, `tree_method="hist"`). The runs table has a Device
column, and each run's `report.json` records `device`. Other models always train on the CPU, and Explore's analyses
run on the CPU.

**Host prerequisites**

* NVIDIA driver **580 or newer** for the default CUDA 13 build. Blackwell cards (RTX 50xx) need the open kernel
  modules (`nvidia-open`). With a 570–579 driver, build with `XGBOOST_GPU_PKG=xgboost-cu12==3.4.1` (CUDA 12.9).
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  registered with Docker:

  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```

* Check that containers can see the GPU:

  ```bash
  docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
  ```

The host needs no CUDA toolkit: the XGBoost wheel bundles its CUDA runtime, and the toolkit mounts the driver.

**Start**

Add `COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml` to `.env` (with `XGBOOST_GPU_PKG` for a 570–579
driver), then:

```bash
make up          # every later `make up` keeps the GPU image while .env names the GPU file
make gpu-check   # what the API sees: name, memory, driver, and whether XGBoost can use it
```

`make up-gpu` builds and starts the GPU image for one run without editing `.env`.

`docker-compose.gpu.yml` builds the API with `GPU=1` and reserves the host's GPUs for it. `GET /api/system` reports
`{"gpu": null}` when no GPU is visible, or the GPU with `available: false` and a `reason` when XGBoost can't use it
(CPU build, driver too old). Each API worker detects this once, in the background as it starts, with a one-tree
test fit, because XGBoost otherwise falls back to the CPU silently. A GPU run that falls back anyway fails rather than
being recorded as a GPU run. With `COMPOSE_FILE` unset (or without the GPU file), `make up` builds the CPU image.

## MLflow

Every model run trained from the UI or API is also logged to an [MLflow](https://mlflow.org) tracking server, which
Compose runs as the `mlflow` service (MLflow 3.17). Its UI is at **`http://localhost:8080/mlflow/`**, behind the
Rainforest sign-in: nginx asks the API whether the session cookie is valid before every request, and sends you to
the sign-in page (and back) if it isn't. The server has no published port of its own.

* **Layout:** one MLflow experiment per dataset, named `rainforest/<dataset id>-<dataset name>` (found again by its
  `rainforest.dataset_id` tag, so renaming a dataset keeps its experiment), and one MLflow run per model run, named
  `run <id> · <model label>` (e.g. `run 12 · LightGBM`).
* **Params:** `model`, `order`, `cv`, `seed`, `device`, `chain_method`, `export_family`, `n_train`, `n_features`, and
  every effective estimator parameter as `estimator.<name>` (defaults, your overrides and the device settings).
* **Metrics:** each scalar metric per split and target as `<split>.<target>.<metric>` (e.g. `test.column_a.roc_auc`),
  plus `fit_seconds`. Characters MLflow doesn't allow in a key become `_`.
* **Artifacts:** `report.json` and `model/classifier_chain.joblib`, the same files the API serves for the run. The
  server stores them on the `mlartifacts` volume and proxies uploads and downloads, so clients only talk HTTP to it.
* **Tags:** `rainforest.dataset_id`, `rainforest.dataset_name`, `rainforest.export_id`, `rainforest.recipe_id`
  and `rainforest.recipe_version` (the recipe version the export was built from), `rainforest.experiment_id` and
  `rainforest.experiment_name` (the Rainforest experiment), `rainforest.run_id`, `rainforest.model`,
  `rainforest.model_family` (the family the export was prepared for), `rainforest.user` (who started the run), and
  `rainforest.git_sha` with `mlflow.source.git.commit` when the image was built with `GIT_SHA` (`make up` passes the
  checkout's commit).
* **Private experiments wait.** MLflow has no access control of its own: anyone who can sign in to Rainforest can
  browse everything in it. So a run in a [private experiment](#experiments) is not logged while its recipe version
  is assessing (the runs table shows *after review*); when review opens, those runs are logged, and runs that finish
  after that are logged at once. The API also flushes any still waiting when it starts.
* **Links:** the runs table has an *MLflow ↗* link per logged run, and the run's detail links to it too; an
  experiment's page links to the dataset's MLflow experiment filtered to its runs.
  `meta.model_runs` stores `mlflow_experiment_id` and `mlflow_run_id`, and the API returns `mlflow_url`.
* **When MLflow is down** the run still finishes; logging gives up after a short timeout (15 s per request, two
  retries) and the run records the reason as `mlflow_error` in its report. The runs table shows *not logged* with the
  reason, and nothing is lost: `make mlflow-backfill` logs every finished run that has no MLflow run (one trained
  while MLflow was down or turned off, for example), except runs of an experiment that is still private. It runs
  only when you start it.
* **Turning it off:** set `MLFLOW_TRACKING_URI=` (empty) in `.env`. Runs are then not logged and show no link. The
  `mlflow` service still starts; stop it with `docker compose stop mlflow` if you don't want it.
* **Storage:** the tracking data lives in a database named `mlflow` in the same Postgres. The `mlflow` service
  creates it on start if it is missing, so it works on an existing `pgdata` volume as well as a new one.

CLI runs (`make train`) aren't model runs in `meta.model_runs`, so they aren't logged to MLflow either.

## Querying

Connect any SQL client to `localhost:5432` on the Docker host (`DB_PORT`; user and password from `.env`), or use the
SQL tab in Bulk edit.
The console runs one query (`SELECT`, `WITH … SELECT`, `VALUES` or `TABLE`) as the SELECT-only `rainforest_ro` role,
in a read-only transaction with a statement timeout. Postgres itself parses the text through a server-side cursor, so a
second statement is refused however it's written, and only the rows shown (up to `SQL_MAX_ROWS`) are fetched rather than the whole result.
`:dataset` in the console is replaced with the current dataset id.

```sql
-- all text fields
SELECT name, null_pct, avg_length FROM meta.v_fields WHERE dataset_id = 1 AND inferred_type = 'text';

-- fields with more than 20% missing that the team keeps
SELECT name, null_pct, imputation FROM meta.v_fields WHERE dataset_id = 1 AND null_pct > 0.2 AND action = 'keep';

-- how one field is prepared for each model family
SELECT family, imputation, transform, scaling, encoding, source FROM meta.v_field_settings WHERE field_id = 42;

-- compare trained models on the test split
SELECT model, target, metric, value FROM meta.v_model_metrics
WHERE dataset_id = 1 AND split = 'test' AND metric IN ('roc_auc', 'roc_auc_ovr_macro') ORDER BY target, value DESC;
```

| Table / view | Contents |
|---|---|
| `meta.datasets` | uploads: status, progress, counts, targets, active `model_family` |
| `meta.fields` | inferred type, confidence and reasons, typed statistics, `leakage_flag`, `stats` / `charts` / per-family `suggestion` JSONB |
| `meta.field_decisions` | shared decision per field (keep / drop / target, type, notes, reviewed) |
| `meta.field_settings` | preprocessing per field **per model family** |
| `meta.decision_history` | append-only audit log of every change (with family, `user_id` and, for bulk edits, `batch_id`) |
| `meta.bulk_batches` | each applied bulk edit: summary and a pre-change snapshot used by undo |
| `meta.field_correlations` | pairwise associations (one row per unordered pair) |
| `meta.users` | users: name, email, role (`lead` / `member`), active, OIDC subject, last seen |
| `meta.recipes` | recipe versions per dataset: phase (`assessing` / `review` / `finalized`), `scope` (`all` / `reopened`), who finalized it, the frozen `snapshot`, and `reuse` (the summary when it started from a recipe) |
| `meta.recipe_reopened` | the fields a `reopened` version works on |
| `meta.experiments` | experiments per dataset: name, owner, `source` (`recipe:<id>` or `assessment:<recipe id>:<user id>`), notes, archived |
| `meta.exports`, `meta.model_runs` | export jobs; training runs with metrics and report; each with its `experiment_id` |
| `meta.analyses` | Explore jobs (map, tests, importance): params, data version, status, result JSONB |
| `meta.v_field_profiles`, `meta.v_fields`, `meta.v_field_settings`, `meta.v_correlations`, `meta.v_model_metrics` | query-friendly views (`v_fields` and `v_field_settings` show the team decision) |
| `raw.ds_<id>` | the uploaded data (sanitized column names; see `meta.fields.db_column`) |

Assessments are private while a dataset is assessed, so the console's role can't read `meta.assessments`,
`meta.assessment_settings`, `meta.assessment_batches`, `meta.dataset_assessors` or `meta.review_comments`, and sees
only the team's Explore results in `meta.analyses`. Personal experiments are private while assessing, so row-level
security hides them from the console in `meta.experiments`, with their rows in `meta.exports`, `meta.model_runs`,
`meta.v_model_metrics` and `meta.analyses`.

**Schema changes** go in `api/app/migrations/NNN_name.sql`. The API applies new migrations once on startup, under an
advisory lock, and records them in `meta.schema_migrations`. Views live in `api/app/views.sql` and are recreated on
every start.

## Configuration

Compose settings, read from `.env` (see `.env.example`) on every `docker compose` / `make` command. A literal `$` in a
value is written `$$`.

| Variable | Default | Purpose |
|---|---|---|
| `COMPOSE_FILE` | `docker-compose.yml` | `docker-compose.yml:docker-compose.gpu.yml` for the GPU image (see [GPU](#gpu)) |
| `COMPOSE_PROFILES` | – | `keycloak` to run the reference Keycloak (see [Keycloak](#keycloak)) |
| `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` | rainforest | database name and owner; Postgres applies them when it creates the `pgdata` volume, so set them before the first start |
| `READONLY_PASSWORD` | rainforest_ro | password for the SQL console's `rainforest_ro` role |
| `UI_PORT`, `API_PORT`, `DB_PORT` | 8080, 8000, 5432 | published ports (`API_PORT` and `DB_PORT` bind to 127.0.0.1) |
| `API_WORKERS` / `JOB_WORKERS` | 2 / 2 | uvicorn workers / background job threads per worker |
| `JOB_SLOTS` | 1 | background jobs running at once across all workers; each uses `cpu_count // JOB_SLOTS` threads, so the default gives one job every core |
| `LOG_LEVEL` | info | API log level |
| `RAINFOREST_AUTH_MODE` | `pick` | `oidc` for team deployments; `pick` lists users to choose from and doesn't authenticate (see [Sign-in](#sign-in)) |
| `RAINFOREST_OIDC_ISSUER` | – | the provider's issuer URL, exactly as it appears in its tokens (e.g. `https://sso.example.com/realms/rainforest`). Required in `oidc` mode |
| `RAINFOREST_OIDC_CLIENT_ID`, `RAINFOREST_OIDC_CLIENT_SECRET` | – | the confidential client registered for Rainforest. Required in `oidc` mode |
| `RAINFOREST_OIDC_REDIRECT_URL` | from the request | `<Rainforest URL>/api/auth/oidc/callback`; set it when Rainforest sits behind a proxy |
| `RAINFOREST_OIDC_DISCOVERY_URL` | issuer + `/.well-known/openid-configuration` | where the API reads discovery, when it can't reach the issuer URL itself (the Keycloak profile: `http://keycloak:8080/realms/rainforest/.well-known/openid-configuration`) |
| `RAINFOREST_OIDC_SCOPES` | `openid profile email` | requested scopes |
| `RAINFOREST_OIDC_GROUPS_CLAIM` | `groups` | ID token claim with the user's groups (Keycloak's full group paths like `/rainforest-lead` also work) |
| `RAINFOREST_OIDC_PROVIDER_NAME` | `SSO` | the name on the *Sign in with …* button |
| `RAINFOREST_LEAD_GROUP`, `RAINFOREST_MEMBER_GROUP` | `rainforest-lead`, `rainforest-member` | groups that make a lead or a member |
| `KEYCLOAK_URL`, `RAINFOREST_URL` | `http://localhost:8081`, `http://localhost:8080` | Keycloak profile only: the URLs browsers use for Keycloak (its issuer) and Rainforest (written into the realm's redirect URIs on first import) |
| `KEYCLOAK_PORT`, `KEYCLOAK_ADMIN_PASSWORD` | 8081, admin | Keycloak profile only: published port and admin console password |
| `RAINFOREST_SESSION_SECRET` | generated | key that signs session cookies; if empty, one is generated on first start and kept in the data volume (`/data/session_secret`) |
| `RAINFOREST_ALLOWED_HOSTS` | `*` | comma-separated hostnames / IPs the API answers to (`*.example.com` allowed). Set this to the names and addresses you use, e.g. `rainforest.lan,192.168.1.20`; `localhost`, `127.0.0.1` and `::1` are always accepted |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000/mlflow` | where the API logs model runs; empty turns MLflow logging off (see [MLflow](#mlflow)) |
| `GIT_SHA` | – | commit tagged on MLflow runs, baked into the API image; `make up` sets it from the checkout |
| `API_EXTRA_PIP` | – | extra packages baked into the API image (e.g. `catboost`) |
| `XGBOOST_GPU_PKG` | `xgboost==3.4.1` | the CUDA XGBoost package for the GPU image (`xgboost-cu12==3.4.1` for 570–579 drivers) |

Tuning settings for the API (`api/app/config.py`), also read from `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `PROFILE_SAMPLE_ROWS` | 500000 | rows for quantiles, histograms and text stats (counts are always exact) |
| `CORR_SAMPLE_ROWS` | 100000 | rows for associations |
| `CORR_MAX_LEVELS` | 50 | categorical levels kept for associations |
| `COPY_BATCH_ROWS` | 100000 | rows per batch when loading `raw.ds_<id>` |
| `EXPORT_CHUNK_MB`, `EXPORT_FIT_MB` | 256, 512 | export memory budgets (dense output per transform chunk, and for the fit sample) |
| `SAMPLE_MAX_CELLS` | 60000000 | cap on rows × fields for generated samples (about 2 GB peak) |
| `DB_POOL_MAX` | 10 | connections per API worker |
| `SQL_TIMEOUT_MS`, `SQL_MAX_ROWS` | 15000, 2000 | SQL console guard rails |
| `SQL_POOL_MAX` | 4 | read-only console connections per API worker (also the number of console queries it runs at once) |

## Development

Each block starts from the repository root.

```bash
# API, against the compose Postgres (`docker compose up -d db`; user and password as in .env)
cd api && python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
DATABASE_URL=postgresql://rainforest:<POSTGRES_PASSWORD>@localhost:5432/rainforest DATA_DIR=./data \
  uvicorn app.main:app --reload

# UI (proxies /api to :8000; set API_URL to point elsewhere)
cd ui && npm ci && npm run dev    # http://localhost:5173
npm run typecheck && npm test

# A sample file to test the upload path
make sample-csv
```

**Tests** live in `api/tests`. Most need no database, including the pick-mode sign-in and request-check tests in
`test_auth.py` and the OIDC redirect and ID-token checks in `test_oidc.py` (against an in-process provider,
`tests/oidc_provider.py`). The API tests in `test_explore.py`, `test_sql_console.py`, `test_data_correctness.py`,
`test_api_cleanup.py`, `test_users.py`, `test_team.py`, `test_recipes.py`, `test_experiments.py` and the OIDC
sign-in flows in `test_oidc.py`, and the job tests in `test_jobs.py`, run against an empty Postgres and skip unless `RAINFOREST_TEST_DATABASE_URL` is set; the API tests
sign in with the `login_as(client, role)` fixture in `conftest.py`.
`test_mlflow.py` logs to a temporary file store, so it needs no server; its database-backed tests follow the same
rule, and one more test runs against a real tracking server when `MLFLOW_TEST_URI` is set (CI runs one).
The UI's tests run with `cd ui && npm test` (Vitest, no browser needed).

```bash
make test                         # API suite locally; DB-backed tests skip
make test-db                      # full suite against a throwaway postgres:16 container (needs a local Docker daemon)
RAINFOREST_TEST_DATABASE_URL=postgresql://user:pw@host:5432/db pytest   # from api/, against your own empty database
```

**CI** (`.github/workflows/ci.yml`) runs on pushes to `main` and on pull requests: the API tests with a Postgres
service (so nothing skips) and an MLflow server container, the UI typecheck, tests and build, and
`docker compose build` for the images.

## Operational notes and limits

* **Background jobs** (ingest, target associations, export, training, Explore analyses) run in a thread pool inside
  the API process, and their state lives in Postgres. At most `JOB_SLOTS` run at once across all workers (one by
  default, using every core); the rest wait as *Queued* and start oldest first, so a quick job submitted behind a long
  training run waits for it. Each job row records the process that runs it, and when a worker starts it fails the queued and
  running jobs of processes that are gone, leaving its siblings' jobs alone (an interrupted target-associations job
  leaves its dataset ready, with a warning to choose the targets again). For multi-node deployments, swap
  `app/jobs.py` for a queue (RQ, Celery, Arq).
* **Disk:** uploads stream to the `/data` volume, and Postgres keeps a full typed copy (`raw.*`). Profiling, Explore
  and export read the Parquet copy. Budget roughly 2–3× the file size.
* **Samples:** text-field and chart statistics come from a sample, and the UI labels them as such. If you change a
  field's type, the stored statistics still describe the inferred type (the UI says so). The export uses your type.
* **Security:** the UI port (`UI_PORT`, 8080) listens on every interface so you can reach it over the LAN; `api` and
  `db` bind to localhost, `mlflow` publishes no port, and the `keycloak` profile publishes `KEYCLOAK_PORT` (8081) on
  every interface. Every API route except `/health` and sign-in needs the
  session cookie, file downloads included, and so does the MLflow UI at `/mlflow/`.
  * Use `RAINFOREST_AUTH_MODE=oidc` for anything but a single person or a trusted network: pick mode lets anyone who
    reaches the UI act as any user, including a lead.
  * The session cookie holds only the user id and their session version, signed (HMAC) with the session secret. It
    is `HttpOnly`, `SameSite=Strict`, lasts 7 days, and is `Secure` when the request came over HTTPS. Every request
    checks the user is still active and the version current, so deactivation and role changes apply at once;
    rotating `RAINFOREST_SESSION_SECRET` signs everyone out.
  * OIDC: PKCE (S256), `state` and `nonce` on every sign-in. The state travels in a short-lived signed cookie
    (`SameSite=Lax`, which the provider's redirect back needs, scoped to `/api/auth/oidc`); the PKCE verifier and
    nonce are derived from it with the server's secret, so nothing else is stored. ID tokens are checked against
    the provider's published keys (asymmetric algorithms only), issuer, audience, expiry and nonce. Provider tokens
    are never stored or sent to the browser.
  * Browser requests that change data must be same-origin (`Origin` / `Referer` match the host;
    `http://localhost:5173` and `http://127.0.0.1:5173` are also accepted for `npm run dev`), and a body must be
    `application/json` (`multipart/form-data` for uploads), which blocks cross-site form posts. Set `RAINFOREST_ALLOWED_HOSTS` as well, so a DNS-rebound name is refused outright.
  * On plain HTTP the session cookie crosses the network unencrypted. Anywhere beyond a trusted LAN, put the UI
    behind a TLS proxy that sets `X-Forwarded-Proto`.
