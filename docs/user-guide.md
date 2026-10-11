# User guide

One section per screen. Terms are defined in [Concepts](concepts.md). The team process behind the *Team* and
*Baselines* screens is in [Team workflow](team-workflow.md).

## Top bar

The top bar has, from left to right:

- The dataset switcher.
- The workflow steps *Review · Bulk edit · Explore · Team · Experiments*. *Review* shows how many fields are reviewed.
  *Team* shows the current baseline version's phase: *Assessing*, *Review* or *Finalized*.
- The *Baseline* chip. It names the current baseline, such as *Trees v2*, and links to the dataset's *Baselines*
  page. While you adjust an experiment, it names that experiment instead.
- *Next unreviewed* (`U`), while fields are left to review.
- On the right, your profile: picture, name and role.

Your profile opens a menu with:

- *Change picture…*
- *Settings*
- *Team*
- *MLflow ↗*, in a new tab, when MLflow is on
- *Switch profile* in pick mode, or *Sign out* with OIDC sign-in

### Profile picture

*Change picture…* sets your picture. A lead can set anyone's picture from the *Team* page.

| Limit | Value |
|---|---|
| Formats | JPEG, PNG or WebP |
| File size | up to 5 MB |
| Image size | up to 40 megapixels |

After you choose a picture, frame it in the circle:

- Drag it, or use the arrow keys, to move it. Hold `Shift` to move further.
- Scroll, pinch, use the slider, or press `+` and `−` to zoom, up to 4×.
- *Reset* centres it again.

Rainforest turns the picture upright, keeps the square you framed and saves it as a 256 × 256 WebP. Nothing of the
file is kept but its pixels. Pictures are stored in `DATA_DIR/avatars`. They appear on the sign-in page, the top
bar and the *Team* page. Without a picture, your initials stand in.

## Settings

*Settings*, in your profile menu, has a section per topic:

| Section | What you can do |
|---|---|
| *Profile* | Change your picture and your name. Your email and role are shown read-only. |
| *Appearance* | Choose the *System*, *Light* or *Dark* theme. *System* follows your device. The choice is kept in this browser. |
| *Team* | Leads only: the user list, as on the *Team* page. |
| *Integrations* | See whether runs are logged to the built-in MLflow server. Connecting your own MLflow deployment, or pulling data from S3 or other storage, is coming soon. |
| *Application* | Coming soon: app settings that today live in `.env`. |

You can rename yourself in pick mode without `RAINFOREST_USERS`. When accounts come from `RAINFOREST_USERS`, a lead
changes names in `.env`. With OIDC sign-in, names come from the identity provider.

## Datasets

The *Datasets* page lists every dataset, with one chip per baseline showing its phase.

### Add a dataset

- **Upload** a CSV or Parquet file. Text files may separate columns with commas, semicolons, tabs or pipes.
  Rainforest detects the separator and notes any separator other than a comma on the dataset.
- **Samples ▾ → Real datasets** adds one of five real datasets that ship with the app, with targets already set.
  They go through the same processing as an upload. See [Data attribution](../README.md#data-attribution) for the
  list.
- **Samples ▾ → Generate** makes a synthetic dataset. You choose 1k–2M rows and 30–400 fields, up to 60M cells in
  all (`SAMPLE_MAX_CELLS`). It has two targets, `column_a` and `column_b`: binary and 3-class, both binary, or both
  multiclass. *Add a numeric target (column_c)* adds a third, continuous target predicted as a number.
- On an empty install, *Open the 50k × 150 demo* generates a sample in one click.

A cell holding only `?`, `NA`, `N/A`, `NaN`, `null`, `None` or `#N/A` counts as missing.

Rainforest converts the upload to Parquet, loads it into `raw.ds_<id>`, then profiles and correlates it. Progress
shows live.

### Choose targets

1. Open *Choose target ▾* on the dataset's row. Once targets are set, click the target chips instead.
2. Pick one or more columns, or *No target (unsupervised)*.
3. Save.

The picker shows how each target will be predicted (see
[Predicting numbers and bands](data-preparation.md#predicting-numbers-and-bands)). A target must be binary,
categorical or numeric. Text, identifier, date and constant fields are disabled in the picker with the reason, and the
API refuses them. Saving refreshes the target associations and leak flags. Generated samples already have their
targets set.

### Row menu

The row menu renames or deletes a dataset. Deleting removes its fields, exports, model runs and their files.
*Assessors…* opens the dataset's *Team* step.

### Before review can start

A new dataset's first baseline (*Trees v1*) waits for a lead to set it up on the dataset's *Team* step. Until then,
*Review* is read-only. See [Starting a baseline](team-workflow.md#starting-a-baseline).

## Review

*Review* walks the fields one at a time.

Each field shows:

- its signals: missing share, skew, outliers, levels, and target association with a leak flag
- its distribution and quantile strip
- its top tokens (text) or weekdays (dates)
- its strongest associations. Click a bar to jump to that field.

The decision panel lists *Why these suggestions*: one line per suggested setting, naming it and the evidence. For
example: *Imputation: median, plus a column flagging missing rows; 12.3% of values are missing.* For a target, the
panel also has *Predict as*.

Decisions save as you make them.

The sidebar filters fields:

- by status: *To do* · *Kept* · *Dropped* · *All*
- by type
- by *Needs attention*: unreviewed fields that have a leak flag, are mostly missing or constant, have low type
  confidence, or failed to save

### Details

- A field still at its suggestion and not yet reviewed follows target changes. A new leak flag drops it.
- In a new version, carried-over fields show 🔒 in the list, and the decision panel says why they are locked.
- While you adjust an experiment, a bar names the experiment. The decision panel says what differs from the baseline
  and offers *Revert to baseline*.

## Keyboard shortcuts

These work in *Review*. Press `?` to show the list.

| Key | Action |
|---|---|
| `J` / `→` · `K` / `←` | next / previous field |
| `U` | next unreviewed field (also jumps into *Review* from the dataset's other pages) |
| `X` | toggle keep / drop |
| `Enter` | confirm and go next |
| `A` | use the suggestion and confirm |
| `Z` | undo the last confirm |
| `N` | add a note |
| `/` | search fields |
| `Esc` | leave a control, back to review |
| `?` | show or hide the shortcut list |

## Bulk edit

*Bulk edit* changes many fields at once.

1. Find fields with a preset or your own filters. The presets are *Unreviewed*, *Missing > 20%*, *Skewed numerics*,
   *Target-associated*, *High-card categoricals*, *Text*, *Possible leaks* and *Edited by you*.
2. Untick any rows you want to leave out.
3. Choose *Accept suggestions*, *Keep*, *Drop* or *Mark reviewed*.
4. Check the preview of what will change, then apply.

You can undo a bulk action from its toast. *Review these N →* (`R`) opens the shown fields as a working set in
*Review*.

### Details

- The table shows the first 500 matches. A bulk action applies to every match except the rows you untick, and the bar
  says so.
- Bulk actions never move a field into or out of the target role.
- *Keep* passes over identifier and constant fields.
- *Accept suggestions* leaves settings you customised alone, unless it changes the field's type.
- In team review, *Bulk edit* is for leads only and changes team decisions.

The *SQL* tab is a read-only SQL console with examples. See [SQL](sql.md).

*Derived fields* adds computed columns. See [Derived fields](data-preparation.md#derived-fields).

## Explore

*Explore* shows correlations, a 2-D map of the rows, per-field tests against a target, and feature importance. See
[Explore](explore.md).

## Team

The dataset's *Team* step runs the team process for the baseline version in progress:

- A lead sets up a new dataset's first baseline here, and adds or removes assessors.
- Assessors *Submit assessment* here. A lead sees each assessor's progress.
- In team review, the review queue lists the disagreements, each assessor's choice side by side, with comments.
- A lead finalizes here.
- When a baseline started from another dataset's baseline, its summary stays here.

The *Team* page in your profile menu is different: it lists users. Leads manage users there. See
[Managing people](team-workflow.md#managing-people).

See [Team workflow](team-workflow.md) for each step.

## Baselines

The dataset's *Baselines* page opens from the top bar's *Baseline* chip and from the *Team* step. It lists every
version of every baseline, with:

- who started and finalized it, and when
- its phase
- how many fields it reopened
- what it was built from

It compares any two versions: the fields whose decision or settings differ. Between baselines of different model
types, it compares decisions only. It compares team decisions only. A version still being worked is compared as its
team decision stands, never with anyone's assessment.

Leads start baselines and versions here, and rename baselines. See [Team workflow](team-workflow.md).

## Experiments

The dataset's *Experiments* step lists its experiments. Filter by *All*, *Mine*, *Team* or *Archived*. Each row
shows the export and run counts and the best run on the validation split.

An experiment's page has the export panel, the *Train a model* panel and the runs table. See
[Experiments and training](experiments-and-training.md).
