"""Blind assessment: whose field decisions a request reads and writes, assessors and their progress, and the rule
that adopts the fields every assessor agreed on when review opens.

While a recipe version is assessing, each assigned assessor works on their own rows in meta.assessments /
meta.assessment_settings, and every read path takes its decisions from `source_for`, so nobody is ever shown
another person's choices. From review onwards everyone reads the team decision (meta.field_decisions /
meta.field_settings).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from fastapi import HTTPException
from psycopg import sql

from . import recipes
from .db import jsonb
from .processing import upsert_settings
from .semantics import FAMILIES, SETTING_KEYS

SHARED_COLS = ("action", "semantic_type", "notes", "reviewed")
TARGET_COLS = ("target_task", "target_bins")


def core_key(d: dict) -> tuple:
    """What assessors must agree on for a field: its role, its type and, for a target, how it is predicted."""
    return (d["action"], d["semantic_type"], d.get("target_task"),
            json.dumps(d.get("target_bins"), sort_keys=True))


@dataclass(frozen=True)
class Source:
    """The decisions a request reads and writes. Four kinds:
    - team: the team decision of the baseline in progress (the working tables);
    - assessment: one assessor's assessment of a version (`user_id`);
    - experiment: a team experiment's own copy of its baseline (`experiment`);
    - baseline: a finalized baseline's snapshot, read-only (`snapshot_of` set);
    - suggested: each field's suggestion for the version's family, read-only (what a Viewer sees of a version being
      assessed on a dataset with no finalized baseline).
    `denied` is the (status, message) a write gets, or None when the viewer may write."""
    recipe: dict
    user_id: int | None = None
    denied: tuple[int, str] | None = None
    experiment: dict | None = None
    snapshot_of: bool = False
    suggested: bool = False

    @property
    def kind(self) -> str:
        if self.user_id is not None:
            return "assessment"
        if self.experiment is not None:
            return "experiment"
        if self.suggested:
            return "suggested"
        return "baseline" if self.snapshot_of else "team"

    @property
    def personal(self) -> bool:
        return self.kind == "assessment"

    @property
    def team(self) -> bool:
        return self.kind == "team"

    @property
    def family(self) -> str:
        return self.recipe["family"]

    @property
    def key(self) -> str:
        """Stable name of the source, as stored with Explore results."""
        return {"assessment": f"assessment:{self.recipe['id']}:{self.user_id}",
                "experiment": f"experiment:{(self.experiment or {}).get('id')}",
                "baseline": f"baseline:{self.recipe['id']}", "suggested": f"suggested:{self.recipe['id']}",
                "team": "team"}[self.kind]

    def require_write(self) -> None:
        if self.denied:
            raise HTTPException(*self.denied)

    def decisions(self) -> sql.Composable:
        """A relation shaped like meta.field_decisions."""
        if self.kind == "team":
            return sql.SQL("meta.field_decisions")
        if self.kind == "assessment":
            return sql.SQL(
                """(SELECT a.field_id, f.dataset_id, a.action, a.semantic_type, a.target_task, a.target_bins, a.notes,
                           a.reviewed, a.source, a.updated_at
                      FROM meta.assessments a JOIN meta.fields f ON f.id = a.field_id
                     WHERE a.recipe_id = {} AND a.user_id = {})"""
            ).format(sql.Literal(self.recipe["id"]), sql.Literal(self.user_id))
        if self.kind == "experiment":
            return sql.SQL(
                """(SELECT e.field_id, f.dataset_id, e.action, e.semantic_type, e.target_task, e.target_bins, e.notes,
                           e.reviewed, e.source, e.updated_at
                      FROM meta.experiment_decisions e JOIN meta.fields f ON f.id = e.field_id
                     WHERE e.experiment_id = {})"""
            ).format(sql.Literal(self.experiment["id"]))
        if self.kind == "suggested":
            return sql.SQL(
                """(SELECT f.id AS field_id, f.dataset_id, coalesce(sg ->> 'action', 'keep') AS action,
                           coalesce(sg ->> 'semantic_type', f.inferred_type) AS semantic_type,
                           CASE WHEN sg ->> 'action' = 'target'
                                THEN meta.default_target_task(coalesce(sg ->> 'semantic_type', f.inferred_type),
                                                              f.n_unique) END AS target_task,
                           NULL::jsonb AS target_bins, NULL::text AS notes, false AS reviewed, 'suggested' AS source,
                           now() AS updated_at
                      FROM meta.fields f CROSS JOIN LATERAL (SELECT f.suggestion -> {fam} AS sg) s
                     WHERE f.dataset_id = {ds})"""
            ).format(fam=sql.Literal(self.family), ds=sql.Literal(self.recipe["dataset_id"]))
        return sql.SQL(
            """(SELECT f.id AS field_id, f.dataset_id, e.value -> 'decision' ->> 'action' AS action,
                       e.value -> 'decision' ->> 'semantic_type' AS semantic_type,
                       e.value -> 'decision' ->> 'target_task' AS target_task,
                       nullif(e.value -> 'decision' -> 'target_bins', 'null'::jsonb) AS target_bins,
                       e.value -> 'decision' ->> 'notes' AS notes, true AS reviewed, 'suggested' AS source,
                       r.finalized_at AS updated_at
                  FROM meta.recipes r CROSS JOIN LATERAL jsonb_each(r.snapshot -> 'fields') e
                  JOIN meta.fields f ON f.dataset_id = r.dataset_id AND f.name = e.key
                 WHERE r.id = {})"""
        ).format(sql.Literal(self.recipe["id"]))

    def settings(self) -> sql.Composable:
        """A relation shaped like meta.field_settings."""
        if self.kind == "team":
            return sql.SQL("meta.field_settings")
        if self.kind == "suggested":
            return sql.SQL(
                """(SELECT f.id AS field_id, {fam}::text AS family, f.dataset_id, v.imputation, v.impute_value,
                           coalesce(v.add_missing_indicator, false) AS add_missing_indicator, v.transform, v.scaling,
                           v.encoding, coalesce(v.params, '{{}}'::jsonb) AS params, 'suggested' AS source,
                           now() AS updated_at
                      FROM meta.fields f
                      CROSS JOIN LATERAL jsonb_to_record(f.suggestion -> {fam}) AS v(
                          imputation text, impute_value text, add_missing_indicator boolean, transform text,
                          scaling text, encoding text, params jsonb)
                     WHERE f.dataset_id = {ds})"""
            ).format(fam=sql.Literal(self.family), ds=sql.Literal(self.recipe["dataset_id"]))
        if self.kind == "assessment":
            return sql.SQL(
                """(SELECT s.field_id, s.family, f.dataset_id, s.imputation, s.impute_value, s.add_missing_indicator,
                           s.transform, s.scaling, s.encoding, s.params, s.source, s.updated_at
                      FROM meta.assessment_settings s JOIN meta.fields f ON f.id = s.field_id
                     WHERE s.recipe_id = {} AND s.user_id = {})"""
            ).format(sql.Literal(self.recipe["id"]), sql.Literal(self.user_id))
        if self.kind == "experiment":
            return sql.SQL(
                """(SELECT s.field_id, s.family, f.dataset_id, s.imputation, s.impute_value, s.add_missing_indicator,
                           s.transform, s.scaling, s.encoding, s.params, s.source, s.updated_at
                      FROM meta.experiment_settings s JOIN meta.fields f ON f.id = s.field_id
                     WHERE s.experiment_id = {})"""
            ).format(sql.Literal(self.experiment["id"]))
        return sql.SQL(
            """(SELECT f.id AS field_id, {fam}::text AS family, f.dataset_id, v.imputation, v.impute_value,
                       coalesce(v.add_missing_indicator, false) AS add_missing_indicator, v.transform, v.scaling,
                       v.encoding, coalesce(v.params, '{{}}'::jsonb) AS params, 'suggested' AS source,
                       r.finalized_at AS updated_at
                  FROM meta.recipes r CROSS JOIN LATERAL jsonb_each(r.snapshot -> 'fields') e
                  JOIN meta.fields f ON f.dataset_id = r.dataset_id AND f.name = e.key
                  CROSS JOIN LATERAL jsonb_to_record(e.value -> 'settings' -> {fam}) AS v(
                      imputation text, impute_value text, add_missing_indicator boolean, transform text,
                      scaling text, encoding text, params jsonb)
                 WHERE r.id = {rid})"""
        ).format(fam=sql.Literal(self.family), rid=sql.Literal(self.recipe["id"]))

    def fields_view(self) -> sql.Composable:
        """A relation shaped like meta.v_fields (the Bulk edit finder's columns)."""
        if self.kind == "team":
            return sql.SQL("meta.v_fields")
        return sql.SQL(
            """(SELECT p.*, dc.action, dc.semantic_type, dc.target_task, dc.target_bins, dc.reviewed, dc.source AS decision_source, dc.notes,
                       dc.updated_at AS decided_at, s.imputation, s.transform, s.scaling, s.encoding,
                       s.add_missing_indicator, s.params, s.source AS settings_source
                  FROM meta.v_field_profiles p
                  LEFT JOIN {dec} dc ON dc.field_id = p.field_id
                  LEFT JOIN {set} s ON s.field_id = p.field_id AND s.family = {fam}) v"""
        ).format(dec=self.decisions(), set=self.settings(), fam=sql.Literal(self.family))


def assessor_row(conn, recipe_id: int, user_id: int) -> dict | None:
    return conn.execute(
        "SELECT * FROM meta.dataset_assessors WHERE recipe_id = %s AND user_id = %s", (recipe_id, user_id)
    ).fetchone()


def source_for(conn, dataset_id: int, user: dict, recipe: dict | None = None, lock: str | None = None,
               view: str | None = None) -> Source:
    """What `user` reads and may write on this dataset now: by default the baseline in progress (their own
    assessment while they assess it), or with `view` an experiment's copy (`experiment:<id>`) or a finalized
    baseline (`baseline:<recipe_id>`, read-only). Writers pass lock="share" (see recipes.current)."""
    if view:
        return _view_source(conn, dataset_id, user, view, lock)
    recipe = recipe or recipes.current(conn, dataset_id, lock)
    name = recipes.label(recipe)
    if user.get("role") == "viewer" and recipe["phase"] == "assessing":
        return viewer_source(conn, recipe)
    if recipe["phase"] == "assessing":
        mine = assessor_row(conn, recipe["id"], user["id"])
        if mine is None:
            return Source(recipe, None, (403, f"{name} is being assessed; only its assigned assessors make choices."))
        if mine["submitted_at"] is not None:
            return Source(recipe, user["id"], (409, "You submitted your assessment. Unsubmit it to make changes."))
        return Source(recipe, user["id"])
    if recipe["phase"] == "review":
        return Source(recipe, None, None if user["role"] == "lead" else (403, "Only a lead sets the team decision."))
    return Source(recipe, None, (409, f"{name} is finalized; its decisions can't change. Start a new version to "
                                      "change them."))


def viewer_source(conn, recipe: dict) -> Source:
    """What a Viewer reads of a version being assessed: nothing of its work in progress. The dataset's latest finalized
    baseline when it has one, else each field's suggestion."""
    denied = (403, "Viewers can look but not change anything.")
    latest = conn.execute(
        f"SELECT {recipes.RECIPE_COLS} FROM meta.recipes WHERE dataset_id = %s AND phase = 'finalized' "
        "ORDER BY finalized_at DESC, id DESC LIMIT 1", (recipe["dataset_id"],)).fetchone()
    if latest is not None:
        return Source(latest, snapshot_of=True, denied=denied)
    return Source(recipe, suggested=True, denied=denied)


def _view_source(conn, dataset_id: int, user: dict, view: str, lock: str | None) -> Source:
    from . import experiments

    kind, _, raw = view.partition(":")
    if kind not in ("experiment", "baseline") or not raw.isdigit():
        raise HTTPException(422, "view must be experiment:<id> or baseline:<recipe_id>")
    if kind == "baseline":
        recipe = recipes.get(conn, int(raw))
        if recipe is None or recipe["dataset_id"] != dataset_id or recipe["phase"] != "finalized":
            raise HTTPException(404, "No finalized baseline of this dataset with that id")
        return Source(recipe, snapshot_of=True,
                      denied=(409, f"{recipes.label(recipe)} is finalized; its decisions can't change."))
    x = experiments.get_visible(conn, int(raw), user)
    if x["dataset_id"] != dataset_id:
        raise HTTPException(404, "Experiment not found")
    if x["assessor_id"] is not None:
        raise HTTPException(409, "A personal experiment follows your assessment; change your assessment instead.")
    if lock:
        # Writers serialise on the experiment: two saves that both move its target order mustn't deadlock.
        x = conn.execute(f"SELECT {experiments.COLS} FROM {experiments.FROM} WHERE x.id = %s FOR NO KEY UPDATE OF x",
                         (x["id"],)).fetchone()
    return Source(recipes.get(conn, x["recipe_id"]), experiment=x, denied=experiments.adjust_denied(x, user))


def from_key(conn, dataset_id: int, key: str) -> Source:
    """The read side of a stored source key (Explore results)."""
    if key == "team":
        return Source(recipes.current(conn, dataset_id))
    kind, _, rest = key.partition(":")
    if kind == "assessment":
        recipe_id, user_id = rest.split(":")
        recipe = recipes.get(conn, int(recipe_id))
        if recipe is None or recipe["dataset_id"] != dataset_id:
            raise HTTPException(404, "Recipe not found")
        return Source(recipe, int(user_id))
    if kind == "baseline":
        recipe = recipes.get(conn, int(rest))
        if recipe is None or recipe["dataset_id"] != dataset_id:
            raise HTTPException(404, "Baseline not found")
        return Source(recipe, snapshot_of=True)
    if kind == "suggested":
        recipe = recipes.get(conn, int(rest))
        if recipe is None or recipe["dataset_id"] != dataset_id:
            raise HTTPException(404, "Recipe not found")
        return Source(recipe, suggested=True)
    if kind == "experiment":
        from . import experiments
        x = experiments.get(conn, int(rest))
        if x is None or x["dataset_id"] != dataset_id:
            raise HTTPException(404, "Experiment not found")
        return Source(recipes.get(conn, x["recipe_id"]), experiment=x)
    raise HTTPException(422, f"Unknown decision source {key!r}")


def key_owner(key: str | None) -> int | None:
    """The user a stored source key is private to (an assessment's assessor), or None."""
    if not key or not key.startswith("assessment:"):
        return None
    return int(key.rsplit(":", 1)[1])


def personal_snapshot(conn, recipe: dict, user_id: int) -> dict:
    """One assessor's assessment of a version, shaped like a recipe snapshot: their choices on the fields the version
    works on, and the version's own carried-over (locked) values for the rest (its snapshot once finalized, since
    the working tables may hold another baseline by then)."""
    team = recipes.snapshot_of(conn, recipe)
    mine = recipes.snapshot(conn, recipe["dataset_id"], Source(recipe, user_id), family=recipe["family"])
    open_ = recipes.open_field_ids(conn, recipe)
    if open_ is None:
        names = set(mine["fields"])
    else:
        names = {r["name"] for r in conn.execute("SELECT name FROM meta.fields WHERE id = ANY(%s)", (open_,))}
    fields = {n: (mine["fields"][n] if n in names and n in mine["fields"] else f) for n, f in team["fields"].items()}
    return {**team, "fields": fields}


# ------------------------------------------------------------------------------------------------- writes
def write_decision(conn, src: Source, field_id: int, d: dict, decided: bool) -> None:
    """Shared columns of one field's decision; `decided` marks it a real change (source becomes 'user'). A target
    task left out of `d` keeps the stored one (the trigger re-checks it against the type)."""
    table, where = _decision_target(src)
    explicit = d.get("target_task") is not None
    conn.execute(
        sql.SQL("""UPDATE {} SET action = %(action)s, semantic_type = %(semantic_type)s, notes = %(notes)s,
                          reviewed = %(reviewed)s, source = CASE WHEN %(decided)s THEN 'user' ELSE source END,
                          target_task = CASE WHEN %(explicit)s THEN %(target_task)s ELSE target_task END,
                          target_bins = CASE WHEN %(explicit)s THEN %(target_bins)s ELSE target_bins END,
                          updated_at = now()
                    WHERE field_id = %(field_id)s AND {}""").format(table, where),
        {**{c: d[c] for c in SHARED_COLS}, "decided": decided, "field_id": field_id, "explicit": explicit,
         "target_task": d.get("target_task"), "target_bins": jsonb(d["target_bins"]) if d.get("target_bins") else None},
    )


def mark_decided(conn, src: Source, field_id: int) -> None:
    table, where = _decision_target(src)
    conn.execute(sql.SQL("UPDATE {} SET source = 'user' WHERE field_id = %s AND {}").format(table, where), (field_id,))


def write_settings(conn, src: Source, dataset_id: int, field_id: int, family: str, s: dict, source: str) -> None:
    if src.kind == "team":
        upsert_settings(conn, dataset_id, field_id, family, s, source)
        return
    if src.kind == "experiment":
        if family != src.family:
            return  # an experiment's copy holds its baseline's model type only
        conn.execute(
            """INSERT INTO meta.experiment_settings
                  (experiment_id, field_id, family, imputation, impute_value, add_missing_indicator, transform,
                   scaling, encoding, params, source, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
               ON CONFLICT (experiment_id, field_id, family) DO UPDATE SET
                  imputation = EXCLUDED.imputation, impute_value = EXCLUDED.impute_value,
                  add_missing_indicator = EXCLUDED.add_missing_indicator, transform = EXCLUDED.transform,
                  scaling = EXCLUDED.scaling, encoding = EXCLUDED.encoding, params = EXCLUDED.params,
                  source = EXCLUDED.source, updated_at = now()""",
            (src.experiment["id"], field_id, family, s["imputation"], s.get("impute_value"),
             s.get("add_missing_indicator", False), s["transform"], s["scaling"], s["encoding"],
             jsonb(s.get("params") or {}), source),
        )
        return
    conn.execute(
        """INSERT INTO meta.assessment_settings
              (recipe_id, user_id, field_id, family, imputation, impute_value, add_missing_indicator, transform,
               scaling, encoding, params, source, updated_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
           ON CONFLICT (recipe_id, user_id, field_id, family) DO UPDATE SET
              imputation = EXCLUDED.imputation, impute_value = EXCLUDED.impute_value,
              add_missing_indicator = EXCLUDED.add_missing_indicator, transform = EXCLUDED.transform,
              scaling = EXCLUDED.scaling, encoding = EXCLUDED.encoding, params = EXCLUDED.params,
              source = EXCLUDED.source, updated_at = now()""",
        (src.recipe["id"], src.user_id, field_id, family, s["imputation"], s.get("impute_value"),
         s.get("add_missing_indicator", False), s["transform"], s["scaling"], s["encoding"],
         jsonb(s.get("params") or {}), source),
    )


def restore(conn, src: Source, field_id: int, snap: dict) -> None:
    """Put an assessor's (or an experiment's) field back to a bulk snapshot (decision and settings)."""
    dec = snap["decision"]
    table, where = _decision_target(src)
    conn.execute(
        sql.SQL("""UPDATE {} SET action = %s, semantic_type = %s, notes = %s, reviewed = %s, source = %s,
                          target_task = %s, target_bins = %s, updated_at = now()
                    WHERE field_id = %s AND {}""").format(table, where),
        (dec["action"], dec["semantic_type"], dec["notes"], dec["reviewed"], dec["source"], dec.get("target_task"),
         jsonb(dec["target_bins"]) if dec.get("target_bins") else None, field_id),
    )
    families = [s["family"] for s in snap["settings"]]
    stable, swhere = _settings_target(src)
    conn.execute(
        sql.SQL("DELETE FROM {} WHERE field_id = %s AND NOT family = ANY(%s) AND {}").format(stable, swhere),
        (field_id, families),
    )
    for s in snap["settings"]:
        write_settings(conn, src, 0, field_id, s["family"], s, s["source"])


def lock_decisions(conn, src: Source, field_ids: list[int]) -> None:
    table, where = _decision_target(src)
    conn.execute(sql.SQL("SELECT 1 FROM {} WHERE field_id = ANY(%s) AND {} ORDER BY field_id FOR UPDATE").format(
        table, where), (field_ids,))


def _decision_target(src: Source) -> tuple[sql.Composable, sql.Composable]:
    if src.kind == "team":
        return sql.SQL("meta.field_decisions"), sql.SQL("true")
    if src.kind == "experiment":
        return sql.SQL("meta.experiment_decisions"), sql.SQL("experiment_id = {}").format(
            sql.Literal(src.experiment["id"]))
    if src.kind == "assessment":
        return sql.SQL("meta.assessments"), sql.SQL("recipe_id = {} AND user_id = {}").format(
            sql.Literal(src.recipe["id"]), sql.Literal(src.user_id))
    raise HTTPException(409, "A finalized baseline can't change")


def _settings_target(src: Source) -> tuple[sql.Composable, sql.Composable]:
    if src.kind == "experiment":
        return sql.SQL("meta.experiment_settings"), sql.SQL("experiment_id = {}").format(
            sql.Literal(src.experiment["id"]))
    if src.kind == "assessment":
        return sql.SQL("meta.assessment_settings"), sql.SQL("recipe_id = {} AND user_id = {}").format(
            sql.Literal(src.recipe["id"]), sql.Literal(src.user_id))
    raise HTTPException(409, "Only an assessment's or an experiment's settings are restored this way")


# ------------------------------------------------------------------------------------------ assessors
def seed(conn, recipe: dict, user_id: int) -> None:
    """Start an assessor from the team decision: on a new dataset that is the suggestions with the dataset's targets,
    on a later version the version before. Fields carried over (locked) come in reviewed, with their notes, as they
    stand. Rows they already have, from an earlier assignment, stay."""
    recipe_id, dataset_id = recipe["id"], recipe["dataset_id"]
    open_ = recipes.open_field_ids(conn, recipe)
    conn.execute(
        """INSERT INTO meta.assessments (recipe_id, user_id, field_id, action, semantic_type, target_task, target_bins,
                                         notes, reviewed, source)
           SELECT %(r)s, %(u)s, d.field_id, d.action, d.semantic_type, d.target_task, d.target_bins,
                  CASE WHEN o.open THEN NULL ELSE d.notes END, NOT o.open, d.source
             FROM meta.field_decisions d
             CROSS JOIN LATERAL (SELECT %(all)s OR d.field_id = ANY(%(open)s) AS open) o
            WHERE d.dataset_id = %(ds)s
           ON CONFLICT DO NOTHING""",
        {"r": recipe_id, "u": user_id, "ds": dataset_id, "all": open_ is None, "open": open_ or []},
    )
    conn.execute(
        """INSERT INTO meta.assessment_settings
              (recipe_id, user_id, field_id, family, imputation, impute_value, add_missing_indicator, transform,
               scaling, encoding, params)
           SELECT %s, %s, s.field_id, s.family, s.imputation, s.impute_value, s.add_missing_indicator, s.transform,
                  s.scaling, s.encoding, s.params
             FROM meta.field_settings s WHERE s.dataset_id = %s
           ON CONFLICT DO NOTHING""",
        (recipe_id, user_id, dataset_id),
    )


def progress(conn, recipe: dict) -> list[dict]:
    """Per assessor: fields reviewed out of the fields to assess (the reopened ones on a later version) and whether
    they submitted. Never their choices."""
    open_ = recipes.open_field_ids(conn, recipe)
    return conn.execute(
        """SELECT a.user_id, u.name, a.assigned_at, a.assigned_by, a.submitted_at,
                  (SELECT count(*) FROM meta.assessments x
                    WHERE x.recipe_id = a.recipe_id AND x.user_id = a.user_id AND x.reviewed
                      AND (%(all)s OR x.field_id = ANY(%(open)s))) AS reviewed,
                  CASE WHEN %(all)s THEN (SELECT count(*) FROM meta.fields f WHERE f.dataset_id = %(ds)s)
                       ELSE %(n_open)s END AS total
             FROM meta.dataset_assessors a JOIN meta.users u ON u.id = a.user_id
            WHERE a.recipe_id = %(r)s ORDER BY u.name, a.user_id""",
        {"ds": recipe["dataset_id"], "r": recipe["id"], "all": open_ is None, "open": open_ or [],
         "n_open": len(open_ or [])},
    ).fetchall()


# ------------------------------------------------------------------------------------------- adoption
def settings_key(s: dict | None) -> tuple:
    """Settings normalised for comparison (fill values as text, flags as bools, params as a sorted tuple)."""
    s = s or {}
    value = s.get("impute_value")
    return (s.get("imputation"), None if value is None else str(value), bool(s.get("add_missing_indicator")),
            s.get("transform"), s.get("scaling"), s.get("encoding"),
            tuple(sorted((k, repr(v)) for k, v in (s.get("params") or {}).items())))


def adopt(assessments: list[dict], suggestion: Callable[[str, str], dict]) -> dict | None:
    """The team decision for a field every assessor agreed on, or None when they disagree.

    `assessments`: one dict per assessor with `action`, `semantic_type` and `settings` ({family: settings}), in
    assessor order. Agreement means the same action, semantic type and target task (`core_key`). Each family's
    settings are the most common choice; a tie takes `suggestion(semantic_type, family)`.
    """
    if not assessments or len({core_key(a) for a in assessments}) != 1:
        return None
    first = assessments[0]
    families = sorted({f for a in assessments for f in a["settings"]}) or list(FAMILIES)
    settings = {}
    for fam in families:
        choices = [a["settings"][fam] for a in assessments if fam in a["settings"]]
        counts = Counter(settings_key(c) for c in choices)
        top = max(counts.values())
        winners = [k for k, n in counts.items() if n == top]
        if len(winners) == 1:
            settings[fam] = next(c for c in choices if settings_key(c) == winners[0])
        else:
            settings[fam] = suggestion(first["semantic_type"], fam)
        settings[fam] = {k: settings[fam].get(k) for k in SETTING_KEYS}
    return {"action": first["action"], "semantic_type": first["semantic_type"], "target_task": first.get("target_task"),
            "target_bins": first.get("target_bins"), "settings": settings}


def disagreements(by_field: dict[int, list[dict]]) -> set[int]:
    """Fields whose assessors differ on action, semantic type or target task."""
    return {fid for fid, rows in by_field.items() if len({core_key(r) for r in rows}) > 1}


def assessments_by_field(conn, recipe_id: int, user_ids: list[int]) -> dict[int, list[dict]]:
    """Every listed assessor's decision and settings per field, in assessor (user id) order."""
    out: dict[int, list[dict]] = {}
    by_key: dict[tuple[int, int], dict] = {}
    for a in conn.execute(
        """SELECT a.*, u.name AS user_name FROM meta.assessments a JOIN meta.users u ON u.id = a.user_id
            WHERE a.recipe_id = %s AND a.user_id = ANY(%s) ORDER BY a.field_id, a.user_id""",
        (recipe_id, user_ids),
    ).fetchall():
        row = {**a, "settings": {}}
        out.setdefault(a["field_id"], []).append(row)
        by_key[(a["field_id"], a["user_id"])] = row
    for s in conn.execute(
        "SELECT * FROM meta.assessment_settings WHERE recipe_id = %s AND user_id = ANY(%s)", (recipe_id, user_ids)
    ).fetchall():
        row = by_key.get((s["field_id"], s["user_id"]))
        if row is not None:
            row["settings"][s["family"]] = {k: s[k] for k in (*SETTING_KEYS, "source")}
    return out
