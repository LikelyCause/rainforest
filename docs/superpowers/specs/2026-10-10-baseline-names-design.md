# Baseline names

Goal: a baseline can have a name ("Dropout risk"), so datasets' baselines aren't all "Trees v1". The name is shown
wherever the baseline is today, with its version and its model type.

Decisions made with the owner (2026-10-10): the name belongs to the baseline (one per model type per dataset) and its
versions share it; it's optional; a Team Lead can rename it later.

## 1. What is named

- A dataset has at most one baseline per model type (Trees, Linear, Neural), each with versions v1, v2, …. The name
  belongs to that baseline: v1 and v2 of "Dropout risk" are "Dropout risk v1" and "Dropout risk v2".
- Optional. Without a name a baseline reads as today ("Trees v1", "Trees baseline v1").
- Rules: trimmed, 1–60 characters, unique among the dataset's baselines (ignoring case). Blank means no name.
- Stored in `meta.baseline_names (dataset_id, family, name)`, outside the versions, so renaming never touches a
  finalized version's frozen snapshot. Deleting the dataset deletes its names.

## 2. How it reads

| Where | Unnamed | Named |
|---|---|---|
| Short (top bar chip, dataset list badge, export badges) | Trees v2 | Dropout risk v2 |
| Long (pickers, toasts, "built from …", reuse notes) | Trees baseline v2 | Dropout risk v2 (Trees) |
| Baselines page card heading | Trees | Dropout risk, with a *Trees* tag |

- The top bar chip and the dataset list badge cap a long name with an ellipsis; the full label is in the tooltip.
- Every API payload that describes a baseline or version (datasets list and detail, team state, baselines and
  versions lists, version comparison, experiments, exports, the finalized baselines offered for reuse, a reused
  baseline's source) carries its `name` (`recipe_name` beside `recipe_family`/`recipe_version`), and every `label`
  the API builds uses it.
- What is recorded when something is made keeps the label of that moment: an export's manifest, and a new
  experiment's default name ("Team · Dropout risk v1"). Renaming later doesn't rewrite them.

## 3. Setting and changing it

- **Start form** (the Team page's first-baseline setup and the Baselines page's *Start a baseline*): an optional
  *Name* field, shown when the chosen model type has no baseline yet (a new dataset's first baseline included). For a model type that already has one, the form
  shows its current name, renamed on the Baselines page. `POST …/baselines` takes `name`.
- **Rename**: on the Baselines page, a lead's *Rename* on each baseline's card edits the name in place (empty clears
  it). `PATCH /api/datasets/{id}/baselines/{family}` with `{name}`: lead only (`403` for members, and viewers by the
  existing gate), `404` for a model type without a baseline, `422` for a name breaking the rules.

## Testing

API: a name set when starting (first setup, a new model type), refused when too long or taken (any case), blank
stored as none; rename by a lead, refused for a member and a viewer, allowed on a finalized baseline with its snapshot
unchanged; one test that names a baseline and checks every endpoint listed in section 2 shows the name in its
`label`/`name`; a new experiment's default name uses it; deleting the dataset removes the names. UI: the label helper
(short and long, named and unnamed). dev-01, at 768, 1024 and 1440px: set up a named baseline and see it in the
dataset list, top bar, Baselines page, experiments and exports; rename it; a long name is cut with an ellipsis and
shown in full on hover.
