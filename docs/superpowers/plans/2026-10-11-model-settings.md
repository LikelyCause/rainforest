# Model Settings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LightGBM's parameters, curated and complete, are set from the Train menu with proper controls, validated, applied to the right links of the chain, shown on the run and reusable with *Train again*.

**Architecture:** A data catalog per model (`modeling/params.py`) is the single source: the API serves it with the model lists, validates run `params` against it (plus a 64-row dry fit for LightGBM's own cross-setting rules), and routes each value to the classifier, the regressor or both by `applies`. The UI builds the settings form from the catalog; pure form logic lives in `ui/src/settings.ts`.

**Tech Stack:** FastAPI, LightGBM 4.6.0, React 18 + react-query, vitest.

**Spec:** `docs/superpowers/specs/2026-10-11-model-settings-design.md`

## Global Constraints

- Groups in this order: `run`, `trees`, `learning`, `sampling`, `regularization`, `classes`, `numbers`, `advanced`.
- Kinds: `choice` (options: value, label, optional help), `int`, `float`, `bool`, `int_or_none` (a number or a labelled "no limit" value).
- `applies`: `both` | `classes` | `numbers`. `when`: `{other_param: [allowed values]}`.
- Left out of LightGBM's catalog: `n_jobs`, `verbose`, `random_state`, `importance_type`.
- Seed default 42; cv: 0 (off) or 2–10, default 5.
- A run stores only changed settings in `config.params` (plus `seed`, `cv`).
- Models without a catalog behave as today.
- Docs describe the current state only; comments sparse.
- API tests: `"$PT"` (scratchpad `pt`, dev-01 test DB tunnel on 55432). UI: `cd ui && npm run -s typecheck && npx vitest run`.

## Review Focus

1. **A setting hidden by `when` but still sent** (switch to dart, set drop_rate, switch back to gbdt): the UI must not send it and the API must refuse it with a clear message. Pinned by `test_when_hidden_setting_is_refused` (Task 1) and `visibleValues drops hidden settings` (Task 3).
2. **Huge `n_estimators` in the dry fit** must not make the request slow: the dry fit caps rounds. Pinned by `test_dry_fit_is_quick_with_many_trees` (Task 2).
3. **A class-only setting on a numbers-only export** (or the reverse) must not break the run: routing drops it for links it doesn't apply to. Pinned by `test_routing_by_applies` (Task 1).
4. **Remembered settings from an older catalog** (a removed or renamed parameter, or a value now out of range) must not block Train: restoring keeps only known, valid values. Pinned by `restore drops unknown and invalid values` (Task 3).
5. **`max_depth` "no limit"** must round-trip as `-1`, not `null`, and show as "No limit". Pinned by `test_int_or_none_uses_its_none_value` (Task 1) and the UI test in Task 3.

---

### Task 1: The catalog and its checks

**Files:**
- Create: `api/app/modeling/params.py`
- Test: `api/tests/test_model_params.py`

**Interfaces:**
- Produces:
  - `Setting` dataclass: `name, label, help, group, kind, default, min=None, max=None, step=None, log=False, options=None, applies="both", when=None, none_value=None, none_label=None`; `to_json() -> dict`.
  - `CATALOG: dict[str, list[Setting]]` (key = model key; only `"lightgbm"` now).
  - `settings_for(model: str) -> list[Setting] | None`.
  - `setting_problems(model: str, params: dict) -> list[str]` (unknown names, kind/range/options, `when`).
  - `route(model: str, params: dict) -> tuple[dict, dict]` → (classifier params, regressor params).
  - `objective_problem(params: dict, target_min: float | None) -> str | None`.

- [ ] **Step 1: Failing tests** (`api/tests/test_model_params.py`)

```python
import lightgbm
import pytest

from app.modeling.params import CATALOG, objective_problem, route, setting_problems, settings_for
from app.modeling.registry import MODELS, _lightgbm_native_params

LGB = settings_for("lightgbm")


def test_every_lightgbm_setting_is_a_real_parameter():
    known = set(lightgbm.LGBMClassifier().get_params()) | set(lightgbm.LGBMRegressor().get_params())
    known |= _lightgbm_native_params()
    for s in LGB:
        if s.group != "run":
            assert s.name in known, s.name


def test_defaults_pass_their_own_rules_and_match_the_registry():
    defaults = {s.name: s.default for s in LGB if s.group != "run"}
    assert setting_problems("lightgbm", {}) == []
    for k, v in MODELS["lightgbm"].default_params.items():
        if k in defaults:
            assert defaults[k] == v, k
    for s in LGB:
        if s.group != "run" and s.when is None:
            assert setting_problems("lightgbm", {s.name: s.default}) == [], s.name


def test_left_out_parameters_are_absent():
    names = {s.name for s in LGB}
    assert not names & {"n_jobs", "verbose", "random_state", "importance_type"}


@pytest.mark.parametrize("params,needle", [
    ({"num_leaves": 1}, "num_leaves"), ({"num_leaves": 2.5}, "whole number"), ({"learning_rate": 5}, "learning_rate"),
    ({"boosting_type": "nope"}, "boosting_type"), ({"extra_trees": "yes"}, "extra_trees"), ({"bogus": 1}, "bogus"),
    ({"subsample": 0}, "subsample"),
])
def test_bad_values_are_named(params, needle):
    problems = setting_problems("lightgbm", params)
    assert problems and needle in problems[0]


def test_when_hidden_setting_is_refused():
    assert setting_problems("lightgbm", {"boosting_type": "dart", "drop_rate": 0.2}) == []
    assert "dart" in setting_problems("lightgbm", {"drop_rate": 0.2})[0]


def test_int_or_none_uses_its_none_value():
    depth = next(s for s in LGB if s.name == "max_depth")
    assert depth.kind == "int_or_none" and depth.none_value == -1 and depth.default == -1
    assert setting_problems("lightgbm", {"max_depth": -1}) == [] and setting_problems("lightgbm", {"max_depth": 0})


def test_routing_by_applies():
    clf, reg = route("lightgbm", {"num_leaves": 127, "class_weight": "balanced", "objective": "huber", "alpha": 2.0})
    assert clf == {"num_leaves": 127, "class_weight": "balanced"}
    assert reg == {"num_leaves": 127, "objective": "huber", "alpha": 2.0}


def test_objectives_that_need_non_negative_numbers():
    assert objective_problem({"objective": "poisson"}, -1.0) and objective_problem({"objective": "tweedie"}, -0.1)
    assert objective_problem({"objective": "gamma"}, 0.0)
    assert objective_problem({"objective": "poisson"}, 0.0) is None and objective_problem({}, -5.0) is None


def test_quantile_alpha_must_be_below_one():
    assert setting_problems("lightgbm", {"objective": "quantile", "alpha": 1.5})
    assert setting_problems("lightgbm", {"objective": "huber", "alpha": 1.5}) == []


def test_catalog_serializes():
    import json
    json.dumps([s.to_json() for s in LGB])
    assert settings_for("mlp") is None and set(CATALOG) == {"lightgbm"}
```

- [ ] **Step 2: Run** `"$PT" tests/test_model_params.py` → FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `params.py`**

```python
"""Each model's settings as data: what the Train menu shows, what a run may set, and which links of the chain a value
reaches (`applies`). A model with no entry here keeps free-form `params`, checked by the registry as before."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

GROUPS = ("run", "trees", "learning", "sampling", "regularization", "classes", "numbers", "advanced")


@dataclass(frozen=True)
class Setting:
    name: str
    label: str
    help: str
    group: str
    kind: str  # choice | int | float | bool | int_or_none
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    log: bool = False
    options: tuple[dict, ...] | None = None
    applies: str = "both"
    when: dict[str, tuple] | None = None
    none_value: int | None = None
    none_label: str | None = None

    def to_json(self) -> dict:
        d = asdict(self)
        d["options"] = list(self.options) if self.options else None
        d["when"] = {k: list(v) for k, v in self.when.items()} if self.when else None
        return d


def _opt(value, label, help=None) -> dict:
    return {"value": value, "label": label, **({"help": help} if help else {})}


RUN = [
    Setting("seed", "Seed", "Fixes the random choices, so the same settings train the same model.", "run", "int", 42,
            0, 2**31 - 1),
    Setting("cv", "Chain folds", "Folds that produce the out-of-fold predictions a chain passes down; off passes "
            "the true values instead.", "run", "choice", 5,
            options=(_opt(0, "Off"), *(_opt(k, str(k)) for k in range(2, 11)))),
]

DART = {"boosting_type": ("dart",)}
GOSS = {"data_sample_strategy": ("goss",)}
LIGHTGBM = RUN + [
    Setting("boosting_type", "Boosting", "How trees are added.", "trees", "choice", "gbdt", options=(
        _opt("gbdt", "Gradient boosting", "Each tree corrects the previous ones. The usual choice."),
        _opt("dart", "DART", "Drops random trees while boosting, which can reduce overfitting; slower."),
        _opt("rf", "Random forest", "Independent trees on bagged rows; needs subsample < 1 and a bagging frequency."))),
    Setting("n_estimators", "Trees", "Boosting rounds. More fits more closely, and slower.", "trees", "int", 600, 10,
            10_000, 50),
    Setting("num_leaves", "Leaves per tree", "The main complexity control. More leaves fit finer patterns and "
            "overfit sooner.", "trees", "int", 63, 2, 4096),
    Setting("max_depth", "Max depth", "Caps how deep a tree grows; leaves per tree usually matters more.", "trees",
            "int_or_none", -1, 1, 64, none_value=-1, none_label="No limit"),
    Setting("min_child_samples", "Min rows per leaf", "Larger values make smoother, more conservative trees.",
            "trees", "int", 40, 1, 10_000),
    Setting("learning_rate", "Learning rate", "How much each tree counts. Lower needs more trees and usually "
            "generalizes better.", "learning", "float", 0.04, 0.001, 1.0, log=True),
    Setting("subsample", "Row sample", "Share of rows each tree sees (bagging).", "sampling", "float", 0.8, 0.1,
            1.0, 0.05),
    Setting("subsample_freq", "Bagging every", "Redraw the row sample every k trees; 0 turns bagging off.",
            "sampling", "int", 1, 0, 100),
    Setting("colsample_bytree", "Feature sample", "Share of features each tree sees.", "sampling", "float", 0.7,
            0.1, 1.0, 0.05),
    Setting("reg_alpha", "L1 penalty", "Pushes leaf values to zero; helps with many weak features.",
            "regularization", "float", 0.0, 0.0, 100.0, 0.1),
    Setting("reg_lambda", "L2 penalty", "Shrinks leaf values; steadies noisy targets.", "regularization", "float",
            1.0, 0.0, 100.0, 0.1),
    Setting("min_split_gain", "Min split gain", "A split must improve the loss by at least this much.",
            "regularization", "float", 0.0, 0.0, 100.0, 0.01),
    Setting("class_weight", "Class weights", "Balanced weights rare classes up, trading precision for recall.",
            "classes", "choice", None, options=(_opt(None, "None"), _opt("balanced", "Balanced")), applies="classes"),
    Setting("objective", "Loss", "What a number target is fitted to minimise.", "numbers", "choice", "regression",
            options=(
                _opt("regression", "Squared error", "The usual choice; large errors count most."),
                _opt("regression_l1", "Absolute error", "Robust to outliers; predicts the median."),
                _opt("huber", "Huber", "Squared for small errors, absolute for large ones."),
                _opt("fair", "Fair", "Like Huber, smoother."),
                _opt("quantile", "Quantile", "Predicts a quantile (alpha) instead of the mean."),
                _opt("mape", "Percentage error", "Relative errors; targets near 0 dominate."),
                _opt("poisson", "Poisson", "Counts; needs targets ≥ 0."),
                _opt("gamma", "Gamma", "Positive, skewed amounts; needs targets > 0."),
                _opt("tweedie", "Tweedie", "Amounts with many zeros; needs targets ≥ 0.")), applies="numbers"),
    Setting("alpha", "Alpha", "Huber's switch point, or the quantile to predict (between 0 and 1).", "numbers",
            "float", 0.9, 0.001, 1000.0, 0.05, applies="numbers", when={"objective": ("huber", "quantile")}),
    Setting("max_bin", "Max bins", "Value buckets per feature; more is finer and slower.", "advanced", "int", 255,
            2, 65_535),
    Setting("min_child_weight", "Min leaf weight", "Minimum summed hessian in a leaf.", "advanced", "float", 0.001,
            0.0, 100.0, 0.001),
    Setting("subsample_for_bin", "Rows for binning", "Rows sampled to choose the buckets.", "advanced", "int",
            200_000, 1_000, 10_000_000, 10_000),
    Setting("extra_trees", "Extremely randomized", "Pick split thresholds at random; faster and less overfit.",
            "advanced", "bool", False),
    Setting("path_smooth", "Path smoothing", "Shrinks leaves towards their parents; helps small leaves.",
            "advanced", "float", 0.0, 0.0, 1000.0, 0.1),
    Setting("max_delta_step", "Max leaf output", "Caps each leaf's value; 0 means no cap.", "advanced", "float",
            0.0, 0.0, 100.0, 0.1),
    Setting("feature_fraction_bynode", "Feature sample per split", "Share of features tried at each split.",
            "advanced", "float", 1.0, 0.1, 1.0, 0.05),
    Setting("linear_tree", "Linear leaves", "Fit a linear model in each leaf instead of a constant.", "advanced",
            "bool", False),
    Setting("data_sample_strategy", "Row sampling", "Bagging samples rows at random; GOSS keeps the rows with the "
            "largest errors.", "advanced", "choice", "bagging", options=(_opt("bagging", "Bagging"),
                                                                         _opt("goss", "GOSS"))),
    Setting("top_rate", "GOSS top share", "Share of large-error rows GOSS keeps.", "advanced", "float", 0.2, 0.01,
            1.0, 0.05, when=GOSS),
    Setting("other_rate", "GOSS other share", "Share of the remaining rows GOSS samples.", "advanced", "float", 0.1,
            0.01, 1.0, 0.05, when=GOSS),
    Setting("drop_rate", "DART drop rate", "Share of trees dropped each round.", "advanced", "float", 0.1, 0.0, 1.0,
            0.05, when=DART),
    Setting("max_drop", "DART max drops", "Most trees dropped in one round.", "advanced", "int", 50, 1, 1000,
            when=DART),
    Setting("skip_drop", "DART skip chance", "Chance of skipping the drop in a round.", "advanced", "float", 0.5,
            0.0, 1.0, 0.05, when=DART),
    Setting("uniform_drop", "DART uniform drop", "Drop trees with equal chance instead of by weight.", "advanced",
            "bool", False, when=DART),
    Setting("xgboost_dart_mode", "XGBoost DART mode", "Use XGBoost's DART normalization.", "advanced", "bool",
            False, when=DART),
    Setting("is_unbalance", "Unbalanced (binary)", "Reweight a binary target's classes; don't combine with the "
            "positive-class weight.", "advanced", "bool", False, applies="classes"),
    Setting("scale_pos_weight", "Positive-class weight", "Weight of the positive class of a binary target.",
            "advanced", "float", 1.0, 0.01, 1000.0, 0.1, applies="classes"),
]

CATALOG: dict[str, list[Setting]] = {"lightgbm": LIGHTGBM}
NON_NEGATIVE = {"poisson": "≥ 0", "tweedie": "≥ 0", "gamma": "> 0"}


def settings_for(model: str) -> list[Setting] | None:
    return CATALOG.get(model)


def _value_problem(s: Setting, v: Any) -> str | None:
    if s.kind == "choice":
        return None if v in [o["value"] for o in s.options or ()] else (
            f"{s.name} must be one of {', '.join(repr(o['value']) for o in s.options or ())}")
    if s.kind == "bool":
        return None if isinstance(v, bool) else f"{s.name} must be true or false"
    if s.kind == "int_or_none" and v == s.none_value:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return f"{s.name} must be a number"
    if s.kind in ("int", "int_or_none") and not float(v).is_integer():
        return f"{s.name} must be a whole number"
    if (s.min is not None and v < s.min) or (s.max is not None and v > s.max):
        return f"{s.name} must be from {s.min:g} to {s.max:g}" + (f" (or {s.none_label})" if s.none_label else "")
    return None


def setting_problems(model: str, params: dict) -> list[str]:
    """What's wrong with a run's `params` for `model`: unknown names, values outside their rules, settings sent while
    the setting they depend on hides them, and LightGBM's quantile alpha outside (0, 1)."""
    catalog = {s.name: s for s in settings_for(model) or [] if s.group != "run"}
    out = []
    for name, v in params.items():
        s = catalog.get(name)
        if s is None:
            out.append(f"{name} isn't a setting of this model")
            continue
        if problem := _value_problem(s, v):
            out.append(problem)
            continue
        for other, allowed in (s.when or {}).items():
            current = params.get(other, catalog[other].default)
            if current not in allowed:
                out.append(f"{name} applies only when {other} is {' or '.join(map(str, allowed))}")
    if params.get("objective") == "quantile" and not 0 < params.get("alpha", 0.9) < 1:
        out.append("alpha must be between 0 and 1 for the quantile loss")
    return out


def route(model: str, params: dict) -> tuple[dict, dict]:
    """Split a run's params into the classifier's and the regressor's by each setting's `applies`."""
    applies = {s.name: s.applies for s in settings_for(model) or []}
    clf = {k: v for k, v in params.items() if applies.get(k, "both") in ("both", "classes")}
    reg = {k: v for k, v in params.items() if applies.get(k, "both") in ("both", "numbers")}
    return clf, reg


def objective_problem(params: dict, target_min: float | None) -> str | None:
    """A loss that needs non-negative (or positive) numbers, against a number target's training minimum."""
    rule = NON_NEGATIVE.get(params.get("objective", ""))
    if rule is None or target_min is None:
        return None
    if (rule == "> 0" and target_min <= 0) or (rule == "≥ 0" and target_min < 0):
        return f"The {params['objective']} loss needs targets {rule}; this target's training minimum is {target_min:g}"
    return None
```

- [ ] **Step 4: Run** → PASS. Check the registry's LightGBM `default_params` (`subsample 0.8`, `subsample_freq 1`, …) agree with the catalog defaults; the test pins it.

- [ ] **Step 5: Commit** `"Model settings catalog, with LightGBM's"`.

### Task 2: API — catalog in the model lists, checks, dry fit, routing

**Files:**
- Modify: `api/app/modeling/registry.py` (`catalog()` adds `settings`; `regressor_params` uses `params.route` for catalogued models; new `dry_fit_problem`)
- Modify: `api/app/modeling/train.py` (classifier params from `route`)
- Modify: `api/app/routers/runs.py` (`start_run` checks)
- Test: `api/tests/test_model_params.py`, `api/tests/test_target_tasks.py` (API)

**Interfaces:**
- Consumes: Task 1.
- Produces: `registry.catalog()` rows gain `settings: list[dict] | None`; `registry.dry_fit_problem(spec, params: dict, kinds: list[str]) -> str | None`; `regressor_params(spec, params)` routes by catalog when one exists.

- [ ] **Step 1: Failing tests** (append to `test_model_params.py`)

```python
import time

from app.modeling.registry import catalog, dry_fit_problem, regressor_params


def test_models_list_carries_settings():
    rows = {m["key"]: m for m in catalog()}
    assert rows["lightgbm"]["settings"][0]["name"] == "seed" and rows["mlp"]["settings"] is None


def test_dry_fit_catches_lightgbm_rules():
    spec = MODELS["lightgbm"]
    assert dry_fit_problem(spec, {"boosting_type": "rf", "subsample": 1.0}, ["binary"])
    assert dry_fit_problem(spec, {"num_leaves": 31}, ["binary", "regression"]) is None


def test_dry_fit_is_quick_with_many_trees():
    t = time.monotonic()
    assert dry_fit_problem(MODELS["lightgbm"], {"n_estimators": 10_000}, ["multiclass", "regression"]) is None
    assert time.monotonic() - t < 3


def test_regressor_gets_routed_settings():
    assert regressor_params(MODELS["lightgbm"], {"class_weight": "balanced", "objective": "huber"}) == {"objective": "huber"}
```

API (append to `test_target_tasks.py`, which has `_finalized_with_price` and `export`):

```python
def _run(lead, e, **body):
    return lead.post(f"/api/exports/{e['id']}/runs", json={"model": "lightgbm", **body})


def test_run_settings_are_checked_applied_and_stored(app_, people):
    lead = people["lead"]
    ds, x = _finalized_with_price(lead)
    e = export(lead, ds, x)
    r = _run(lead, e, params={"num_leaves": 1})
    assert r.status_code == 422 and "num_leaves" in r.text
    r = _run(lead, e, params={"boosting_type": "rf", "subsample": 1.0})
    assert r.status_code == 422 and "bagging" in r.text.lower()
    r = _run(lead, e, params={"objective": "poisson"}, seed=7, cv=0)
    assert r.status_code == 202, r.text
    run = wait(lambda: (g := lead.get(f"/api/runs/{r.json()['id']}").json())["status"] in ("done", "failed") and g)
    assert run["status"] == "done", run.get("error")
    assert run["config"]["params"] == {"objective": "poisson"} and run["config"]["seed"] == 7
    assert run["report"]["params"]["objective"] == "poisson" if "objective" in run["report"]["params"] else True


def test_a_loss_that_needs_positive_targets_is_refused(app_, people):
    lead = people["lead"]
    ds = upload(lead, "x,z,v\n" + "".join(f"{i % 17},{i % 5},{(i % 40) - 20.5}\n" for i in range(400)), targets="")
    start(lead, ds, "tree", [lead], targets=("v",))
    rid = submit_and_finalize(lead, ds)["recipe"]["id"]
    e = export(lead, ds, experiment(lead, ds, rid))
    r = _run(lead, e, params={"objective": "gamma"})
    assert r.status_code == 422 and "gamma" in r.text
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

`registry.py`:

```python
from . import params as model_params
```

`catalog()` adds `"settings": [s.to_json() for s in model_params.settings_for(m.key)] if model_params.settings_for(m.key) else None` for registered models and `"settings": None` for planned ones.

`regressor_params`: when `model_params.settings_for(spec.key)` exists, return `model_params.route(spec.key, params)[1]` filtered to keys the regressor accepts (`known` as today) — the catalog's `applies` replaces `_classifier_only` for it.

```python
def dry_fit_problem(spec: ModelSpec, params: dict[str, Any], kinds: list[str]) -> str | None:
    """Fit the run's settings on 64 synthetic rows, so a combination the library refuses fails now, not minutes into
    a run. Rounds are capped so a large tree count stays quick."""
    clf_params, reg_params = model_params.route(spec.key, params) if model_params.settings_for(spec.key) else (params, params)
    quick = {"n_estimators": min(int(params.get("n_estimators", 5)), 5)} if "n_estimators" in spec.build({}, 0).get_params() else {}
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 4))
    try:
        if any(k != "regression" for k in kinds):
            y = (X[:, 0] > 0).astype(int) if all(k == "binary" for k in kinds if k != "regression") else (X[:, 0] > 0).astype(int) + (X[:, 1] > 0)
            spec.build({**spec.default_params, **clf_params, **quick, "n_jobs": 1} if "n_jobs" in spec.default_params else {**spec.default_params, **clf_params, **quick}, 0).fit(X, y)
        if "regression" in kinds and spec.build_regressor is not None:
            merged = {**spec.default_params, **reg_params}
            spec.build_regressor({**spec.regressor_defaults, **regressor_params(spec, merged), **quick}, 0).fit(X, np.abs(X[:, 0]) + 1)
    except Exception as exc:  # noqa: BLE001 - the library's own message is the useful part
        return f"{spec.label} refused these settings: {exc}"
    return None
```

  Simplify the `y` and `n_jobs` lines when writing (one `base = {**spec.default_params, **clf_params, **quick}` with `n_jobs=1` when the spec has `n_jobs`); the behaviour above is the contract.

`train.py`: where the classifier is built (`build_chain` and the TargetChain branch), pass `route(spec.key, params)[0]` (catalogued models) instead of `params`, so `objective`/`alpha` never reach a classifier.

`runs.py start_run`, after the existing checks:

```python
    if model_params.settings_for(spec.key):
        if problems := model_params.setting_problems(spec.key, req.params):
            raise HTTPException(422, {"message": "Invalid model settings", "errors": problems})
        for t, k in zip(order, kinds):
            if k == "regression" and (p := model_params.objective_problem(req.params, (manifest.get("target_stats") or {}).get(t, {}).get("min"))):
                raise HTTPException(422, p)
        if problem := dry_fit_problem(spec, req.params, kinds):
            raise HTTPException(422, problem)
```

  (`order` = `req.order or manifest["targets"]`, `kinds` as already computed there.) For catalogued models keep `param_problems` too (it rejects `random_state`).

- [ ] **Step 4: Run** the two test files, then the full suite → PASS.
- [ ] **Step 5: Commit** `"Check, dry-fit and route model settings; serve them with the models"`.

### Task 3: UI — settings form logic

**Files:**
- Create: `ui/src/settings.ts`, `ui/src/settings.test.ts`
- Modify: `ui/src/types.ts` (`ModelSetting`, `ModelInfo.settings`)

**Interfaces:**
- Produces:
  - `type SettingValue = string | number | boolean | null`; `interface ModelSetting` (mirrors `Setting.to_json`).
  - `settingProblem(s: ModelSetting, v: SettingValue): string | null`.
  - `isVisible(s: ModelSetting, values: Record<string, SettingValue>, settings: ModelSetting[]): boolean`.
  - `changed(values, settings): Record<string, SettingValue>` (non-run settings that differ from default and are visible).
  - `restore(saved: unknown, settings): Record<string, SettingValue>` (keeps known, valid values only).
  - `runFields(values): { seed: number; cv: number }`.
  - `groupOrder`, `GROUP_LABEL`.

- [ ] **Step 1: Failing tests** (`ui/src/settings.test.ts`)

```ts
import { describe, expect, it } from "vitest";
import { changed, isVisible, restore, runFields, settingProblem } from "./settings";
import type { ModelSetting } from "./types";

const S = (o: Partial<ModelSetting>): ModelSetting => ({ name: "x", label: "X", help: "", group: "trees", kind: "int", default: 1,
  min: 1, max: 10, step: null, log: false, options: null, applies: "both", when: null, none_value: null, none_label: null, ...o });
const settings = [
  S({ name: "seed", group: "run", default: 42, min: 0, max: 2 ** 31 - 1 }),
  S({ name: "cv", group: "run", kind: "choice", default: 5, options: [{ value: 0, label: "Off" }, { value: 5, label: "5" }] }),
  S({ name: "num_leaves", default: 63, min: 2, max: 4096 }),
  S({ name: "max_depth", kind: "int_or_none", default: -1, min: 1, max: 64, none_value: -1, none_label: "No limit" }),
  S({ name: "boosting_type", kind: "choice", default: "gbdt", options: [{ value: "gbdt", label: "GBDT" }, { value: "dart", label: "DART" }] }),
  S({ name: "drop_rate", kind: "float", default: 0.1, min: 0, max: 1, group: "advanced", when: { boosting_type: ["dart"] } }),
];

describe("settings", () => {
  it("checks values", () => {
    expect(settingProblem(settings[2], 1)).toMatch(/2/);
    expect(settingProblem(settings[2], 2.5)).toMatch(/whole/);
    expect(settingProblem(settings[3], -1)).toBeNull();
    expect(settingProblem(settings[3], 0)).toMatch(/No limit/);
    expect(settingProblem(settings[4], "nope")).toMatch(/one of/);
  });
  it("hides settings their condition doesn't meet", () => {
    expect(isVisible(settings[5], {}, settings)).toBe(false);
    expect(isVisible(settings[5], { boosting_type: "dart" }, settings)).toBe(true);
  });
  it("visibleValues drops hidden settings and defaults from what is sent", () => {
    expect(changed({ num_leaves: 63, drop_rate: 0.3, seed: 7 }, settings)).toEqual({});
    expect(changed({ num_leaves: 127, boosting_type: "dart", drop_rate: 0.3 }, settings))
      .toEqual({ num_leaves: 127, boosting_type: "dart", drop_rate: 0.3 });
    expect(changed({ max_depth: -1 }, settings)).toEqual({});
  });
  it("restore drops unknown and invalid values", () => {
    expect(restore({ num_leaves: 127, gone: 1, max_depth: 999, seed: 3 }, settings)).toEqual({ num_leaves: 127, seed: 3 });
    expect(restore("junk", settings)).toEqual({});
  });
  it("reads the run fields with their defaults", () => {
    expect(runFields({}, settings)).toEqual({ seed: 42, cv: 5 });
    expect(runFields({ seed: 7, cv: 0 }, settings)).toEqual({ seed: 7, cv: 0 });
  });
});
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** `settings.ts` mirroring `params.py`'s `_value_problem` (choice/bool/number/whole/range/none value), `isVisible` (every `when` entry's current value — the value in `values` else that setting's default — is in its allowed list), `changed` (non-`run`, visible, `values[k] !== default`), `restore` (object check; keep keys that exist and pass `settingProblem`), `runFields` (`seed`, `cv` from values else defaults). **Step 4: Run** → PASS. **Step 5: Commit** `"Model settings form logic"`.

### Task 4: UI — the Train menu, run detail, Train again

**Files:**
- Create: `ui/src/components/ModelSettings.tsx`
- Modify: `ui/src/pages/ExperimentWorkspace.tsx` (Pop gets `open`/`onOpen`/`wide`; TrainForm gets settings, seed/cv and a `prefill`; RunsGroup/RunDetail pass *Train again*), `ui/src/components/RunDetail.tsx` (settings line + button), `ui/src/api.ts` (`startRun` body adds `params`, `seed`), `ui/src/styles.css`

**Interfaces:**
- Consumes: Task 3.
- Produces: `ModelSettings({ settings, values, onChange, kinds })`; `TrainPrefill = { exportId: number; model: string; order: string[]; values: Record<string, SettingValue> }`.

- [ ] **Step 1:** `ModelSettings.tsx`: a summary button "Model settings · Defaults | N changed" toggling an expanded panel; inside, *Reset all*, then groups in `groupOrder` (`classes` only when `kinds` has a non-regression kind, `numbers` only when it has `regression`; `advanced` collapsed with a search box and "N changed"). Each row: label (help in `title` and as `aria-describedby` text shown on focus), control by kind (choice → `<select>` with option labels, values JSON-encoded so `null`/numbers round-trip; bool → switch checkbox; int/float → `<input type="number">` with min/max/step, kept as text locally so partial input like "0." is allowed, committed when it parses; `int_or_none` → a select "No limit / Set" plus a number input when set), "default X" muted beside it, ↺ when changed, the `.edited` marker, and the problem text under it.
- [ ] **Step 2:** `TrainForm`: when the selected model has `settings`, render `ModelSettings` under the model picker; values start from `restore(localStorage["rf.trainSettings.<exportId>.<model>"])` or the prefill; save on change (try/catch). `startRun` sends `{ model, order, device, params: changed(values), ...runFields(values) }`. Train is disabled while any visible value has a problem. The popover gets `wide` (520px, `max-height: 70vh; overflow:auto`) while settings are open.
- [ ] **Step 3:** `RunDetail`: under the header, "Settings: name value · …" from `run.config.params` (labels from the model's catalog when loaded, else names; "defaults" when empty), with seed/cv when not default, and a **Train again with these settings** button calling `onTrainAgain({exportId, model, order, values: {...config.params, seed, cv}})`. `ExperimentWorkspace` holds `prefill` state; the matching export card's Train Pop opens with it (Pop `open` controlled when `prefill?.exportId === e.id`).
- [ ] **Step 4:** `npm run -s typecheck && npx vitest run && npx vite build` → pass. **Step 5: Commit** `"Model settings in the Train menu; Train again from a run"`.

### Task 5: Docs, deploy, dev-01 check

- [ ] README (Export and training: "Model settings" paragraph; the Train menu walkthrough line), `docs/api.md` (`GET /api/models` / export models `settings`; `POST …/runs` checks, dry fit, objective rule, `seed`/`cv`), `docs/architecture.md` (`modeling/params.py` row). Current state only.
- [ ] Full API suite + UI checks; PR; CI green; merge; `DOCKER_CONTEXT=dev-01 make up`.
- [ ] dev-01 (Playwright, as lead1): Concrete strength from the library → baseline → experiment → export; open Train ▾ → Model settings: set Leaves 127, Learning rate 0.02, Loss Huber with Alpha 2, switch Boosting to DART and set a drop rate, switch back (drop rate disappears); Train; run detail lists the settings; MLflow params show `estimator.num_leaves = 127`; *Train again* opens the menu filled; a bad combination (Random forest with Row sample 1) shows the refusal in the menu. Screenshots for the owner.
