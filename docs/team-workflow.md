# Team workflow

How a team agrees on a baseline: who can do what, how blind assessment works, how disagreements are settled, and how a
baseline changes over time. Terms are defined in [Concepts](concepts.md).

## Overview

Every dataset goes through the same steps, whether one person uses Rainforest or a team. On your own, you are the lead
and the only assessor.

1. A lead adds people and uploads the dataset. In pick mode, people are added on the *People* page or in
   `RAINFOREST_USERS`. With OIDC, they come from the identity provider's groups. See
   [Deployment](deployment.md#sign-in).
2. The lead sets up the dataset's baseline on its *Baseline* tab: model type, starting point, targets and assessors.
   The lead can assess too.
3. Each assessor works through *Fields* on their own assessment, which nobody else sees. Then they *Submit
   assessment*. The lead follows everyone's progress on the *Baseline* tab.
4. When the last assessor submits, team review opens. Fields everyone agreed on are adopted. The disagreements form a
   queue in *Fields*. The lead settles them and finalizes.
5. Everyone builds experiments from the finalized baseline. Each person adjusts only their own. You compare results
   on *Experiments* and in MLflow. See [Experiments and training](experiments-and-training.md).

## Roles and permissions

Everyone signs in as a user. Everything a user changes is recorded against them: decision history, bulk edits,
uploads, exports, model runs and *Explore* analyses.

| | Viewer | Member | Lead |
|---|---|---|---|
| Look at datasets, fields, *Explore*, experiments, runs and MLflow; query with the field finder | ✓ | ✓ | ✓ |
| Query with the SQL console (every table except assessments in progress and private experiments) | | ✓ | ✓ |
| Change their own picture | ✓ | ✓ | ✓ |
| Upload, recompute *Explore* results, experiment (export a finalized baseline or your own assessment, train) | | ✓ | ✓ |
| Adjust the experiments they own | | ✓ | ✓ |
| Assess in *Fields · Focus* and *Fields · Table* when assigned; comment in team review | | ✓ | ✓ |
| Start baselines and versions, assign assessors, set the team decision in review, finalize | | | ✓ |
| Delete datasets | | | ✓ |
| Manage users on the *People* page; change anyone's picture | | | ✓ |

What "manage users" means depends on the sign-in mode:

| Sign-in mode | What a lead can do on the *People* page |
|---|---|
| Pick mode without `RAINFOREST_USERS` | add, rename, edit email, set role, deactivate, reactivate |
| Pick mode with `RAINFOREST_USERS` | nothing: the page lists the accounts read-only |
| OIDC | deactivate, reactivate |

### Viewers

A Viewer looks and changes nothing. The API refuses their writes with `403`, MLflow included. The UI shows them no
edit controls. They can't be assessors.

While a baseline version is assessing, Viewers don't see its work in progress. The dataset's pages and *Explore* show
its latest finalized baseline, or each field's suggestion when there is none. Field history is empty. From team review
on, Viewers see the team decision like everyone else.

### Managing people

- Deactivating a user, or changing their role, signs them out everywhere at once.
- The last active lead can't be demoted or deactivated.
- The *People* page doesn't let you deactivate yourself.
- Users are never deleted, so their history keeps their names.
- With OIDC, users are matched by the provider's subject id, never by email.

## Baselines and versions

Every dataset has baselines: the team's decisions for one model type (*Trees*, *Linear* or *Neural*), at most one per
type. Each baseline has numbered versions, labelled like *Trees baseline v2*. A version moves through three phases:
*Assessing*, *Team review* and *Finalized*. Before its first version starts, a new dataset's baseline is in *Set up*.
The phase shows on the *Datasets* list and on the top bar's phase rail: *Set up · Assessing · Team review ·
Finalized*.

One baseline version per dataset is in progress (assessing or in review) at a time. Another baseline or version can
start once it is finalized.

## Starting a baseline

A lead starts a baseline with *Start a baseline…* on *Versions and compare*, or with *Start a linear baseline* or
*Start a neural baseline* on the *Baseline* tab. A new dataset's first baseline (*Trees v1*) is set up in place, with
the same form on the dataset's *Baseline* tab. The form ends with *Start Trees baseline v1*, naming what it starts.

The form asks for:

| Field | What to choose |
|---|---|
| *Model type* | Fixed once the baseline starts. A type that already has a baseline gets its next version. |
| *Name* | Optional. Shown for a model type without a baseline yet. See [Baseline names](#baseline-names). |
| *Start from* | *the suggestions* for that model type; *another baseline of this dataset*; or, for a new dataset's first baseline only, *another dataset's baseline*. |
| *Targets* | Fixed for the baseline. Experiments can change them. |
| *Assessors* | Any active users. Tick yourself to assess too. |

*Another baseline of this dataset* can be a finalized baseline of any type. It copies keep, drop and target, types
and notes. It copies each field's settings where they are valid for the new type. The rest take the new type's
suggestion. Every field is still assessed.

For *another dataset's baseline*, see [Reusing another dataset's baseline](#reusing-another-datasets-baseline).

## Assessing

1. Each assessor starts from the baseline's starting point: the suggestions, or the baseline it was built from.
2. They use *Fields · Focus* and *Fields · Table* as usual, but on their own assessment. The lens chip reads
   *Private*, and the top bar's next step reads *Next field · N left*.
3. When every field is reviewed, they choose *Submit assessment* in the top bar, or *Submit* on the *Baseline* tab,
   and confirm.

An assessment is private. Nobody else sees an assessor's choices, the lead included. The API never returns them to
anyone else: not in field lists, field details, history, *Fields · Table*, the SQL console or *Explore*. Leads see each
assessor's progress (fields reviewed, submitted or not), never their choices. Everyone else can browse the fields at
the starting point and use *Explore*, read-only.

### Details

- A lead adds assessors with *Add an assessor* on the dataset's *Baseline* tab, and unassigns them from each row's
  menu.
- An assessor can unsubmit until team review opens, with *Unsubmit to make changes* on the *Baseline* tab.
- A lead can unassign someone. Their choices are kept but don't count.
- The targets stay as the baseline started, since every assessor starts from them.
- A lead who opens *Fields · Focus* or *Fields · Table* on a dataset they aren't assessing sees *Assess it myself*.
  It assigns them like anyone else.

## Team review

Team review opens by itself when the last assigned assessor submits. It also opens when a lead unassigns the last
assessor who hadn't submitted.

A field is **adopted** as the team decision when every assessor chose the same keep, drop or target, type and
*Predict as*. Its settings take the most common choice. A tie takes the suggestion.

The other fields form the **review queue**. During team review, *Fields* opens on it, and the top bar's next step
reads *Settle N open fields*. The queue lists fields under *Open*, *Settled* and *Agreed*. Each queued field shows
every assessor's choice, settings and notes side by side, with the differences marked, and a discussion.

- Everyone can comment.
- A lead sets the team decision, with the same decision panel as *Focus*, and settles the field.
- A lead can open an agreed field from *Agreed* to change it.
- *Fields · Table* is for leads only in this phase, and changes team decisions.

With a single assessor, every field is adopted and the queue is empty.

## Finalizing

A lead finalizes once the queue is settled, with *Finalize vN* in the top bar or *Finalize…* on the *Baseline* tab.
Rainforest asks to confirm, and says how many fields are agreed, settled and still open. A lead can also finalize
with fields left open: they keep their current team decision. Finalizing takes you to *Experiments*.

Finalizing freezes the version. Its snapshot records every field's decision, its settings for the baseline's model
type, its inferred type and key statistics, and the target order. A finalized version's decisions can't change. To
change them, start a new version.

## New versions

1. Choose *Start a new version* on the *Baseline* tab, or *New version…* on a baseline on *Versions and compare*. It
   starts from the baseline's latest finalized version.
2. Pick the fields to reopen. You can search them, and filter by type and by *Needs attention*. None are picked by
   default. Choose *Start vN, reopening N fields*, or *Start vN and edit directly* with none picked.
3. Assign assessors on the *Baseline* tab.

Reopened fields go through blind assessment and team review again. Assessors only assess those fields. *Submit
assessment* needs only those reviewed. The review queue holds only disagreements among them.

Every other field is **carried over and locked** (🔒 in the *Fields* list). Nobody changes it in that version, the
lead included.

With nothing reopened, the version opens straight in team review. The lead edits the team decision directly, which is
recorded in each field's history, and finalizes.

## Baseline names

A baseline can have a name, given when it starts. A lead changes it with *Rename* (or *Name it*) on the baseline's
card on *Versions and compare*. A blank name removes it.

| | Named baseline | Unnamed baseline |
|---|---|---|
| Top bar, dataset list, export badges | Dropout risk v2 | Trees v2 |
| Pickers and messages | Dropout risk v2 (Trees) | Trees baseline v2 |

- Names are up to 60 characters.
- A dataset's baselines have different names, ignoring case.
- All versions of a baseline share its name.
- Renaming doesn't change a finalized version's decisions.
- What was named at the time keeps its name: an export's manifest, and a team experiment's default name
  ("Team · Dropout risk v1").
- A long name is shortened in the top bar and badges, and shown in full on hover.

## Reusing another dataset's baseline

A new dataset's first baseline can start from any finalized baseline of another dataset, before anyone is assigned to
it. It takes that baseline's model type.

You start it in one of three ways:

- *Another dataset's baseline* in the set-up form.
- The upload toast, which links there for leads.
- `recipe_id` on the upload API. See [docs/api.md](api.md).

Fields match by name. A matched field is **carried over and locked**, with the source's decision, settings and
notes, when both of these hold:

- its inferred type is unchanged
- its statistics haven't drifted: its missing share is within 10 points, and its distinct values are within half to
  double

Everything else is **reopened** for assessment:

- new fields
- changed types
- drift, with the numbers
- a source target that isn't a valid target here
- settings that aren't valid for this data

Source fields missing from the new data are listed. The source's targets carry over when they are still valid targets
here. The summary stays on the dataset's *Baseline* tab and *Versions and compare*, and the version records the baseline
it came from. If nothing needs assessing, the version opens in team review, and a lead confirms it by finalizing.
