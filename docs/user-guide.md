# User guide

One section per screen. Terms are defined in [Concepts](concepts.md). The team process behind the *Baseline* tab and
the team review queue is in [Team workflow](team-workflow.md).

## Top bar

Outside a dataset, the top bar has *Datasets* and, on the right, your profile: picture and name.

Inside a dataset, it has two rows.

The first row says where you are:

- The dataset switcher. It lists every dataset, and *All datasets*.
- The dataset's four places: *Fields · Explore · Baseline · Experiments*. *Fields* shows how many fields are
  reviewed, such as *7/9*. In team review it shows how many fields are still open instead. A dot on *Experiments*
  means an export or run is in progress.
- Your profile picture.

The second row says where the baseline version stands:

- The version switcher, such as *Trees v2*. It lists the dataset's baselines with their phases, and opens
  *Versions and compare*. Leads also get *New version or baseline…*.
- The phase rail: *Set up · Assessing · Team review · Finalized*. Next to it, what opens the next phase, such as
  *Team review opens when all 3 submit · 1 in*, or *4 adopted · 2 open*.
- The lens chip. It says whose decisions are on screen:

  | Chip | What you see |
  |---|---|
  | *Private* | Your own assessment. Nobody else sees it. |
  | *Team decision* | The team decision, in team review. Everyone sees it. |
  | *Frozen snapshot* | A finalized version. Nothing can change it. |
  | *Experiment copy* | An experiment's own copy of its baseline. |
  | *Suggestions only* | The suggestions, before a lead sets the baseline up. While others assess, it reads *Suggestions · not assessing*. |
  | *Read-only* | What a Viewer sees while a version is in progress. |

- The next step, always in the same place. It reads, for example, *Set up the baseline*, *Next field · 12 left*,
  *Submit assessment*, *Settle 2 open fields*, *Finalize Trees v2* or *Start an experiment*.

While you adjust an experiment, the second row names the experiment and what it is built from instead, with
*Back to the experiment*. While you browse a finalized version, it says so, with *Back to versions*. On an
experiment's page, it holds the experiment's own tabs. See [Experiments](#experiments).

Opening a dataset takes you to its next step: the *Baseline* tab while it is being set up, the review queue in team
review, and *Fields* otherwise. From the *Datasets* page, a dataset you worked on reopens on the field you had open.

Your profile opens a menu with:

- *Change picture…*
- *Settings*
- *People*
- *MLflow ↗*, in a new tab, when MLflow is on
- *Switch profile* in pick mode, or *Sign out* with OIDC sign-in

Questions that ask you to confirm, such as deleting a dataset or finalizing, open in the app, not in a browser
dialog.

### Profile picture

*Change picture…* sets your picture. A lead can set anyone's picture from the *People* page.

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
bar and the *People* page. Without a picture, your initials stand in.

## Settings

*Settings*, in your profile menu, has a section per topic:

| Section | What you can do |
|---|---|
| *Profile* | Change your picture and your name. Your email and role are shown read-only. |
| *Appearance* | Choose the *System*, *Light* or *Dark* theme. *System* follows your device. The choice is kept in this browser. |
| *People* | Leads only: the user list, as on the *People* page. |
| *Integrations* | See whether runs are logged to the built-in MLflow server. Connecting your own MLflow deployment, or pulling data from S3 or other storage, is coming soon. |
| *Application* | Coming soon: app settings that today live in `.env`. |

You can rename yourself in pick mode without `RAINFOREST_USERS`. When accounts come from `RAINFOREST_USERS`, a lead
changes names in `.env`. With OIDC sign-in, names come from the identity provider.

## Datasets

The *Datasets* page has two parts.

*Waiting on you* holds a card for each thing you can move forward, your own work first:

- *Finish your assessment*, with your progress, while you assess a version.
- *Settle open fields*, for a lead, when a version is in team review.
- *Set up the first baseline*, for a lead, when a dataset has none yet.
- *Mark a chosen model*, for a lead, when a dataset has runs but no chosen model.

*All datasets* lists every dataset, with a search box. Each row shows:

- its size
- its stage: a bar through *Set up · Assessing · Team review · Finalized*, with the current baseline. A dataset
  with several baselines gets one chip per baseline, showing its phase.
- its targets
- the assessment: your own progress while you assess, else the team's
- *Best result*: the chosen model, else the best run on the validation split, with its experiment
- when it last changed

On a phone, the table becomes a list of cards.

### Add a dataset

- **Upload CSV or Parquet**, or drop a file anywhere on the page. Text files may separate columns with commas,
  semicolons, tabs or pipes. Rainforest detects the separator and notes any separator other than a comma on the dataset.
- **Add a sample ▾ → Real datasets** adds one of five real datasets that ship with the app, with targets already set.
  They go through the same processing as an upload. See [Data attribution](../README.md#data-attribution) for the
  list.
- **Add a sample ▾ → Generate** makes a synthetic dataset. You choose 1k–2M rows and 30–400 fields, up to 60M cells in
  all (`SAMPLE_MAX_CELLS`). It has two targets, `column_a` and `column_b`: binary and 3-class, both binary, or both
  multiclass. *Add a numeric target (column_c)* adds a third, continuous target predicted as a number.
- On an empty install, *Open the 50k × 150 demo* generates a sample in one click.

A cell holding only `?`, `NA`, `N/A`, `NaN`, `null`, `None` or `#N/A` counts as missing.

Rainforest converts the upload to Parquet, loads it into `raw.ds_<id>`, then profiles and correlates it. Progress
shows live.

### Choose targets

1. Open *Choose target ▾* on the dataset's row. Once targets are set, click the target chips instead. On a phone,
   use *Targets…* in the row menu.
2. Pick one or more columns, or *No target (unsupervised)*.
3. Save.

The picker shows how each target will be predicted (see
[Predicting numbers and bands](data-preparation.md#predicting-numbers-and-bands)). A target must be binary,
categorical or numeric. Text, identifier, date and constant fields are disabled in the picker with the reason, and the
API refuses them. Saving refreshes the target associations and leak flags. Generated samples already have their
targets set.

### Row menu

The row menu renames or deletes a dataset. Deleting asks you to confirm, then removes its fields, exports, model runs
and their files. *Assessors…*, for leads while a version is assessing, opens the dataset's *Baseline* tab.

### Before review can start

A new dataset's first baseline (*Trees v1*) waits for a lead to set it up on the dataset's *Baseline* tab. Until then,
*Fields* is read-only, and the lens chip reads *Suggestions only*. See
[Starting a baseline](team-workflow.md#starting-a-baseline).

## Fields

*Fields* is where you decide each field. It has its own switch at the top of the field list:

- *Focus* walks the fields one at a time.
- *Table* changes many fields at once.
- *SQL* is a read-only SQL console. See [SQL](sql.md).
- *Derived* adds computed columns. See [Derived fields](data-preparation.md#derived-fields).
- *Queue*, first in the switch during team review, is the team review queue. See [Team review](#team-review).

*SQL* and *Derived* show only for the baseline in progress, to members and leads.

## Fields · Focus

*Focus* walks the fields one at a time. *Previous* (`K`) and *Next* (`J`) move through them.

Each field shows:

- its signals: missing share, skew, outliers, levels, and target association with a leak flag
- its distribution and quantile strip
- its top tokens (text) or weekdays (dates)
- its strongest associations. Click a bar to jump to that field.

The decision panel is on the right. Its header names whose decision it is, such as *Your assessment* or *Team decision*,
with the same colour as the lens chip. It lists *Why these suggestions*: one line per suggested setting, naming it and
the evidence. For example: *Imputation: median, plus a column flagging missing rows; 12.3% of values are missing.* For a
target, the panel also has *Predict as*.

Decisions save as you make them. *Confirm* (`Enter`) marks the field reviewed and moves on. *Revert all to the
suggestion* puts back every setting you changed.

The field list filters fields:

- by status: *To do* or *All*. *Kept* and *Dropped* are under *Type*.
- by type, under *Type*
- by *Needs a look*: unreviewed fields that have a leak flag, are mostly missing or constant, have low type
  confidence, or failed to save

### Details

- A field still at its suggestion and not yet reviewed follows target changes. A new leak flag drops it.
- In a new version, carried-over fields show 🔒 in the list, and the decision panel says why they are locked.
- While you adjust an experiment, the second row of the top bar names the experiment, with the *Experiment copy*
  chip. The decision panel says what differs from the baseline and offers *Revert to baseline*.
- On a phone, the field list opens from *Fields · 3 of 9* above the field, and *Previous*, *Next* and confirm sit in a
  bar at the bottom.

## Keyboard shortcuts

These work in *Fields · Focus*. Press `?` to show the list. The decision panel shows the most used ones under
*Confirm*.

| Key | Action |
|---|---|
| `J` / `→` · `K` / `←` | next / previous field |
| `U` | next unreviewed field (also jumps into *Focus* from the dataset's other pages) |
| `X` | toggle keep / drop |
| `Enter` | confirm and go next |
| `A` | use the suggestion and confirm |
| `Z` | undo the last confirm |
| `N` | add a note |
| `/` | search fields |
| `Esc` | leave a control, back to review |
| `?` | show or hide the shortcut list |

## Fields · Table

*Table* changes many fields at once.

1. Find fields with a preset or your own filters. The presets are *Unreviewed*, *Missing > 20%*, *Skewed numerics*,
   *Target-associated*, *High-card categoricals*, *Text*, *Possible leaks* and *Edited by you*.
2. Untick any rows you want to leave out.
3. Choose *Accept suggestions*, *Keep*, *Drop* or *Mark reviewed*.
4. Check the preview of what will change, then apply.

You can undo a bulk action from its toast. *Review these N →* (`R`) opens the shown fields as a working set in
*Focus*.

### Details

- The table shows the first 500 matches. A bulk action applies to every match except the rows you untick, and the bar
  says so.
- Bulk actions never move a field into or out of the target role.
- *Keep* passes over identifier and constant fields.
- *Accept suggestions* leaves settings you customised alone, unless it changes the field's type.
- In team review, *Table* is for leads only and changes team decisions.

## Explore

*Explore* shows correlations, a 2-D map of the rows, per-field tests against a target, and feature importance. See
[Explore](explore.md).

## Baseline

The dataset's *Baseline* tab runs the team process for the baseline version in progress:

- A lead sets up a new dataset's first baseline here.
- While a version is assessing, it shows your own progress with *Submit*, or *Unsubmit to make changes*. It lists
  the assessors and their progress. A lead adds assessors with *Add an assessor*, and unassigns them from each
  row's menu.
- In team review, it counts the fields adopted, the disagreements and those still open, with *Open the queue* and,
  for a lead, *Finalize…*.
- Once finalized, it says who finalized the version and when, with *Go to Experiments*, *Browse its fields* and, for
  a lead, *Start a new version*.
- When a baseline started from another dataset's baseline, its summary stays here.

Below, *Versions* lists the dataset's baselines and their versions. For a lead, *Other model types* starts a
*Linear* or *Neural* baseline.

*People*, in your profile menu, is different: it lists users. Leads manage users there. See
[Managing people](team-workflow.md#managing-people).

See [Team workflow](team-workflow.md) for each step.

## Team review

While a version is in team review, *Fields* opens on the review queue. It lists the fields under *Open*, *Settled*
and *Agreed*. Each open field shows every assessor's choice side by side, a discussion, the field's profile, and the
team decision panel. A lead settles each field there. See [Team review](team-workflow.md#team-review).

## Versions and compare

*Versions and compare* opens from the version switcher and from *Compare* on the *Baseline* tab. It lists every
version of every baseline, with:

- who started and finalized it, and when
- its phase
- how many fields it reopened
- what it was built from

*Browse* opens a finalized version's fields, read-only.

It compares any two versions: the fields whose decision or settings differ. Between baselines of different model
types, it compares decisions only. It compares team decisions only. A version still being worked is compared as its
team decision stands, never with anyone's assessment.

Leads start baselines and versions here, and rename baselines. See [Team workflow](team-workflow.md).

## Experiments

The dataset's *Experiments* tab lists its experiments. Filter by *All*, *Mine*, *Team* or *Archived*. Each row
shows the export and run counts and the best run on the validation split. Right after a lead finalizes a version, and
until an experiment is built on it, the tab offers *New experiment from vN*.

On an experiment's page, the second row of the top bar names the experiment, with a menu to switch to another one,
and its tabs:

| Tab | What it holds |
|---|---|
| *Runs* | The runs table. A run's detail opens beside it. |
| *Train* | The model, its settings and *Train*. |
| *Exports* | What goes into an export, the split, row rules, and the exports made. |
| *Adjustments* | What the experiment changes from its baseline, with *Adjust fields*. |
| *Notes* | Its name, notes, chosen model, MLflow link, and *Archive*. |

The row also shows what the experiment is built from, such as *From Trees baseline v2 · as finalized*, and
*Train a model*. See [Experiments and training](experiments-and-training.md).
