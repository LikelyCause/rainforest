"""Train a chain of models, one per target, on an exported, model-ready dataset.

    python -m app.modeling.train --export-id 3                       # LightGBM (default)
    python -m app.modeling.train --export-id 3 --model xgboost --order column_b,column_a
    python -m app.modeling.train --export-id 3 --model xgboost --device cuda   # GPU image only
    python -m app.modeling.train --list-models

Chain mechanics
---------------
ClassifierChain fits one classifier per target, in `order`. Each later model sees the
original features plus the earlier models' outputs. With `cv=5` those outputs are
out-of-fold during training, so the downstream model learns from realistic (noisy)
upstream predictions instead of the ground truth it will never see at inference time.

* both targets binary   -> chain_method="predict_proba" (the P(class=1) is passed down)
* any target multiclass -> chain_method="predict"       (the predicted class code is passed
  down; sklearn's chain stores one column per target, so a k-class probability vector
  can't be chained)

When any target is predicted as a number, the run uses `TargetChain` instead: a classifier or a regressor per
target, chaining P(class 1) for a binary target, the predicted class for a multiclass one and the predicted value
for a number. Runs with only class targets keep ClassifierChain, so their results are unchanged.

Models come from `registry.MODELS`; which ones may train on an export depends on the
model family the export was prepared for (see `registry.compatibility`).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import polars as pl
from sklearn import config_context
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.multioutput import ClassifierChain

from .. import bands
from ..gpu import effective_device
from .registry import (DEFAULT_MODEL, MODELS, ModelSpec, catalog, compatibility, device_params, get_model,
                       classifier_params, importances, order_problem, param_problems, regression_problem,
                       regressor_params)

log = logging.getLogger("train")

# LightGBM >= 4.6 records generated feature names even for ndarray input; sklearn then warns
# on every chained predict. The arrays are always in manifest order, so the warning is noise.
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)


class NaNTolerantClassifierChain(ClassifierChain):
    """ClassifierChain that lets NaN through to models that route missing values natively.

    sklearn's chain validates X as finite even when the wrapped model (LightGBM, XGBoost,
    HistGradientBoosting, random forest) handles NaN itself — the tree family's "leave
    missing" default depends on this. `assume_finite` (thread-local) skips only that check.
    """

    def fit(self, X, Y, **fit_params):
        with config_context(assume_finite=True):
            return super().fit(X, Y, **fit_params)

    def predict(self, X):
        with config_context(assume_finite=True):
            return super().predict(X)

    def predict_proba(self, X):
        with config_context(assume_finite=True):
            return super().predict_proba(X)


class TargetChain:
    """A chain whose targets may be classes or numbers: one classifier or regressor per target, in `order`. Each
    later model sees the features plus the earlier targets' outputs, out-of-fold with `cv` folds (the ground truth
    when cv is None). A number is passed down standardized (training mean and spread), so it sits on the scale of
    the features a linear or neural model sees."""

    chain_method_ = "per_target"

    def __init__(self, classifier, regressor, kinds: list[str], order: list[int], cv: int | None = 5,
                 handles_nan: bool = False, seed: int = 42):
        self.classifier, self.regressor, self.kinds = classifier, regressor, kinds
        self.order, self.cv, self.handles_nan, self.seed = order, cv, handles_nan, seed

    def _folds(self, kind: str):
        """Class links fold as ClassifierChain does (stratified); number links on shuffled folds, so a file sorted by
        its target doesn't leave each fold predicting values it never saw."""
        from sklearn.model_selection import KFold

        return KFold(self.cv, shuffle=True, random_state=self.seed) if kind == "regression" else self.cv

    def _scaled(self, t: int, values: np.ndarray) -> np.ndarray:
        mean, std = self.scales_.get(t, (0.0, 1.0))
        return (values - mean) / std

    @staticmethod
    def _output(est, X: np.ndarray, kind: str) -> np.ndarray:
        return est.predict_proba(X)[:, 1] if kind == "binary" else est.predict(X)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "TargetChain":
        from sklearn.base import clone
        from sklearn.model_selection import cross_val_predict

        feats = np.zeros((X.shape[0], len(self.order)))
        self.estimators_ = []
        self.scales_ = {t: (float(Y[:, t].mean()), float(Y[:, t].std()) or 1.0)
                        for t, kind in enumerate(self.kinds) if kind == "regression"}
        with config_context(assume_finite=self.handles_nan):
            for pos, t in enumerate(self.order):
                kind = self.kinds[t]
                y = Y[:, t] if kind == "regression" else Y[:, t].astype(np.int64)
                X_aug = np.hstack((X, feats[:, :pos]))
                base = self.regressor if kind == "regression" else self.classifier
                self.estimators_.append(clone(base).fit(X_aug, y))
                if pos == len(self.order) - 1:
                    break
                if self.cv is None:
                    out = y
                else:
                    out = cross_val_predict(clone(base), X_aug, y, cv=self._folds(kind),
                                            method="predict_proba" if kind == "binary" else "predict")
                    out = out[:, 1] if kind == "binary" else out
                feats[:, pos] = self._scaled(t, out) if kind == "regression" else out
        self.order_ = list(self.order)
        return self

    def predict_all(self, X: np.ndarray, n_classes: list[int]) -> list[np.ndarray]:
        """Per target, in target order: the class-probability matrix (column j = encoded class j) or the values."""
        feats = np.zeros((X.shape[0], len(self.order_)))
        out: list[np.ndarray | None] = [None] * len(self.order_)
        with config_context(assume_finite=self.handles_nan):
            for pos, est in enumerate(self.estimators_):
                t = self.order_[pos]
                X_aug = np.hstack((X, feats[:, :pos]))
                if self.kinds[t] == "regression":
                    out[t] = est.predict(X_aug).astype(float)
                else:
                    p = est.predict_proba(X_aug)
                    full = np.zeros((p.shape[0], n_classes[t]))
                    full[:, np.asarray(est.classes_, dtype=int)] = p
                    out[t] = full
                out_t = self._output(est, X_aug, self.kinds[t])
                feats[:, pos] = self._scaled(t, out_t) if self.kinds[t] == "regression" else out_t
        return out  # type: ignore[return-value]


def load_export(export_dir: Path) -> tuple[dict, pl.DataFrame]:
    manifest = json.loads((export_dir / "manifest.json").read_text())
    df = pl.read_parquet(export_dir / manifest["files"]["data"])
    return manifest, df


def build_chain(spec: ModelSpec, target_kinds: list[str], order: list[int], seed: int,
                params: dict | None = None, cv: int | None = 5) -> ClassifierChain:
    all_binary = all(k == "binary" for k in target_kinds)
    estimator = spec.build(classifier_params(spec, {**spec.default_params, **(params or {})}), seed)
    chain_cls = NaNTolerantClassifierChain if spec.handles_nan else ClassifierChain
    # First positional arg is `base_estimator` in sklearn 1.6 and `estimator` from 1.7.
    return chain_cls(
        estimator,
        order=order,
        cv=cv,
        chain_method="predict_proba" if all_binary else "predict",
        random_state=seed,
    )


def chain_predict_proba(chain: ClassifierChain, X: np.ndarray, n_classes: list[int] | None = None) -> list[np.ndarray]:
    """Full class-probability matrix for every target (in original target order).

    `ClassifierChain.predict_proba` only returns P(class=1) per target, which is not
    enough for multiclass targets, so walk the fitted chain exactly as sklearn does.
    Each estimator only knows the classes it saw in training; with `n_classes` (per target,
    in target order) column j of the result is always encoded class j, zero for unseen ones.
    """
    n_targets = len(chain.estimators_)
    feature_chain = np.zeros((X.shape[0], n_targets))
    probas: list[np.ndarray | None] = [None] * n_targets
    method = chain.chain_method_
    for pos, est in enumerate(chain.estimators_):
        X_aug = np.hstack((X, feature_chain[:, :pos]))
        p = est.predict_proba(X_aug)
        target = chain.order_[pos]
        if n_classes is not None:
            full = np.zeros((p.shape[0], n_classes[target]))
            full[:, np.asarray(est.classes_, dtype=int)] = p
            probas[target] = full
        else:
            probas[target] = p
        feature_chain[:, pos] = p[:, 1] if method == "predict_proba" else est.predict(X_aug)
    return probas  # type: ignore[return-value]


ROC_MAX_POINTS = 200


def roc_curves(y_true: np.ndarray, proba: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    """ROC curve per class, one-vs-rest (only the positive class for a binary target), thinned to at most
    ROC_MAX_POINTS points. A class with no positives or no negatives in the split has no curve."""
    k = proba.shape[1]
    out = []
    for c in ([1] if k == 2 else range(k)):
        hit = y_true == c
        if hit.all() or not hit.any():
            continue
        fpr, tpr, _ = roc_curve(hit, proba[:, c])
        if len(fpr) > ROC_MAX_POINTS:
            keep = np.unique(np.linspace(0, len(fpr) - 1, ROC_MAX_POINTS).round().astype(int))
            fpr, tpr = fpr[keep], tpr[keep]
        out.append({"label": names[c], "auc": float(roc_auc_score(hit, proba[:, c])),
                    "fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]})
    return out


def evaluate(y_true: np.ndarray, proba: np.ndarray, labels: list[str] | None = None) -> tuple[dict[str, float], dict[str, Any]]:
    """Score one target. Predictions are the argmax class (i.e. a 0.5 threshold for binary targets).

    Returns (scalars, details): scalars are single numbers (stored in `metrics`, queryable via
    meta.v_model_metrics); details hold the confusion matrix, per-class table and ROC curves.
    """
    k = proba.shape[1]
    classes = list(range(k))
    names = labels if labels and len(labels) == k else [str(c) for c in classes]
    pred = proba.argmax(axis=1)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, proba, labels=classes)),
    }
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, pred, labels=classes, average=avg, zero_division=0)
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = float(p), float(r), float(f)
    cm = confusion_matrix(y_true, pred, labels=classes)
    if k == 2:
        # Positive class = encoded label 1 (see manifest target_classes).
        tn, fp, fn, tp = (int(v) for v in cm.ravel())
        out["precision"] = tp / (tp + fp) if tp + fp else 0.0
        out["recall"] = tp / (tp + fn) if tp + fn else 0.0
        out["f1"] = 2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"]) if tp else 0.0
        out["specificity"] = tn / (tn + fp) if tn + fp else 0.0
    # sklearn returns NaN (not an error) for AUC on a single-class split, which JSON can't carry.
    if len(np.unique(y_true)) >= 2:
        try:
            if k == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
            else:
                out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        except ValueError:  # a class absent from this split
            pass
    p, r, f, support = precision_recall_fscore_support(y_true, pred, labels=classes, zero_division=0)
    details = {
        "labels": names,
        "positive_label": names[1] if k == 2 else None,
        "confusion_matrix": cm.astype(int).tolist(),  # rows = actual, columns = predicted
        "per_class": [
            {"label": names[c], "precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c]),
             "support": int(support[c])}
            for c in classes
        ],
        "roc": roc_curves(y_true, proba, names),
    }
    return out, details


SCATTER_MAX_POINTS = 1000
RESIDUAL_BINS = 30


def _sig(values: np.ndarray) -> list[float]:
    return [float(f"{v:.6g}") for v in values]


BAND_COUNT = 5


def _band_edges(values: np.ndarray) -> list[float] | None:
    return bands.quantile_edges(values, BAND_COUNT) or None


def evaluate_regression(y: np.ndarray, pred: np.ndarray, seed: int = 42,
                        edges: list[float] | None = None) -> tuple[dict[str, float], dict[str, Any]]:
    """Score a numeric target: RMSE, MAE, R² (when the actual values vary) and MAPE (when none is 0). Details hold up
    to SCATTER_MAX_POINTS sampled (actual, predicted) pairs and a histogram of the errors (predicted - actual).

    With `edges` (cut points from the training rows), actual and predicted values are also put in those bands and
    scored as classes, so a number has a confusion matrix and precision, recall and F1 too: how often the model
    predicts the right range."""
    if bad := int((~np.isfinite(pred)).sum()):
        # Scoring the rest would give a broken model believable numbers.
        raise ValueError(f"The model predicted something that isn't a number for {bad:,} of {pred.size:,} rows; it "
                         "probably diverged. Try a lower learning rate or fewer trees.")
    err = pred - y
    out = {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))}
    if y.size >= 2 and np.var(y) > 0:
        out["r2"] = float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))
    if not np.any(y == 0):
        out["mape"] = float(np.mean(np.abs(err) / np.abs(y)))
    idx = np.arange(y.size)
    if y.size > SCATTER_MAX_POINTS:
        idx = np.sort(np.random.default_rng(seed).choice(y.size, SCATTER_MAX_POINTS, replace=False))
    counts, hist_edges = np.histogram(err, bins=RESIDUAL_BINS)
    details = {
        "kind": "regression", "n": int(y.size),
        "scatter": {"actual": _sig(y[idx]), "predicted": _sig(pred[idx])},
        "residuals": {"edges": _sig(hist_edges), "counts": counts.astype(int).tolist()},
    }
    if edges:
        k = len(edges) + 1
        actual, predicted = bands.assign(y, edges).astype(int), bands.assign(pred, edges).astype(int)
        p, r, f, support = precision_recall_fscore_support(actual, predicted, labels=list(range(k)), zero_division=0)
        out["band_accuracy"] = float(accuracy_score(actual, predicted))
        out["band_f1_macro"] = float(np.mean(f))
        labels = bands.band_labels(edges)
        details["bands"] = {
            "labels": labels, "edges": edges,
            "confusion_matrix": confusion_matrix(actual, predicted, labels=list(range(k))).astype(int).tolist(),
            "per_class": [{"label": labels[c], "precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c]),
                           "support": int(support[c])} for c in range(k)],
        }
    return out, details


def train(export_dir: Path, model: str = DEFAULT_MODEL, order_names: list[str] | None = None, seed: int = 42,
          params: dict | None = None, cv: int | None = 5, out_dir: Path | None = None,
          progress: Callable[[str], None] | None = None, device: str = "cpu") -> dict[str, Any]:
    spec = get_model(model)
    if problems := param_problems(spec, params or {}):
        raise ValueError("; ".join(problems))
    if cv == 1:
        raise ValueError("cv must be 0 (chain the ground truth) or 2 or more folds")
    params = {**(params or {}), **device_params(spec, device)}
    manifest, df = load_export(export_dir)
    export_family = manifest.get("model_family", "tree")
    problem = compatibility(spec, export_family, bool(manifest.get("contains_nan", False)))
    if problem:
        raise ValueError(problem)
    feats: list[str] = manifest["feature_names"]
    split_col = manifest["split_column"]
    order_names = order_names or manifest["targets"]
    if problem := order_problem(order_names, manifest["targets"]):
        raise ValueError(problem)
    # The run's targets: those in `order`, kept in manifest order so Y's columns line up with the export.
    targets: list[str] = [t for t in manifest["targets"] if t in order_names]
    order = [targets.index(t) for t in order_names]
    kinds = [manifest["target_kinds"][t] for t in targets]
    numeric = [k == "regression" for k in kinds]
    if problem := regression_problem(spec, kinds):
        raise ValueError(problem)
    say = progress or (lambda msg: log.info(msg))

    def xy(split: str) -> tuple[np.ndarray, np.ndarray]:
        part = df.filter(pl.col(split_col) == split)
        return part.select(feats).to_numpy().astype(np.float32), part.select(targets).to_numpy().astype(np.float64)

    X_tr, Y_tr = xy("train")
    n_classes = [0 if num else len(manifest["target_classes"][t]) for t, num in zip(targets, numeric)]
    missing_from_train: dict[str, list[str]] = {}
    for i, t in enumerate(targets):
        if numeric[i]:
            if np.unique(Y_tr[:, i]).size < 2:
                raise ValueError(f"Target {t!r} has a single value in the training split; there is nothing to predict.")
            continue
        seen = set(np.unique(Y_tr[:, i]).astype(int).tolist())
        if len(seen) < 2:
            raise ValueError(f"Target {t!r} has only one class in the training split; a classifier needs at least two.")
        if absent := [c for k, c in enumerate(manifest["target_classes"][t]) if k not in seen]:
            missing_from_train[t] = absent
    say(f"Fitting {spec.label} chain {' → '.join(order_names)} on {X_tr.shape[0]:,} rows × {X_tr.shape[1]:,} features"
        + (" on the GPU" if device == "cuda" else ""))
    if any(numeric):
        merged = {**spec.default_params, **(params or {})}
        regressor = spec.build_regressor({**spec.regressor_defaults, **regressor_params(spec, merged)}, seed)
        if spec.family != "tree":
            # Gradient-trained and penalized models fit a target on its own scale badly (an MLP stops long before a
            # price in the hundreds of thousands); trees split on order and don't care.
            from sklearn.compose import TransformedTargetRegressor
            from sklearn.preprocessing import StandardScaler

            regressor = TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())
        chain = TargetChain(spec.build(classifier_params(spec, merged), seed), regressor, kinds, order, cv,
                            spec.handles_nan, seed)
    else:
        chain = build_chain(spec, kinds, order, seed, params, cv)
    t0 = time.time()
    chain.fit(X_tr, Y_tr if any(numeric) else Y_tr.astype(np.int64))
    fit_seconds = time.time() - t0
    if device == "cuda":
        # XGBoost quietly drops to the CPU when it can't reach the GPU; don't report that as a GPU run.
        used = effective_device(chain.estimators_[0].get_booster())
        if not used.startswith("cuda"):
            raise RuntimeError(f"GPU training was requested but {spec.label} ran on {used}")

    # Ranges for scoring a number as classes too, cut on the training rows like an export's equal-sized bands.
    band_edges = [_band_edges(Y_tr[:, i]) if num else None for i, num in enumerate(numeric)]
    say("Evaluating on validation and test splits")
    metrics: dict[str, Any] = {}
    details: dict[str, Any] = {}
    for split in ("valid", "test"):
        X, Y = xy(split)
        if len(X) == 0:
            continue
        metrics[split], details[split] = {}, {}
        if isinstance(chain, TargetChain):
            outs = chain.predict_all(X, n_classes)
            for i, t in enumerate(targets):
                metrics[split][t], details[split][t] = (
                    evaluate_regression(Y[:, i], outs[i], seed, band_edges[i]) if numeric[i]
                    else evaluate(Y[:, i].astype(int), outs[i], manifest["target_classes"].get(t)))
            continue
        with config_context(assume_finite=spec.handles_nan):
            probas = chain_predict_proba(chain, X, n_classes)
            predicted = chain.predict(X).astype(int)
        for i, t in enumerate(targets):
            metrics[split][t], details[split][t] = evaluate(Y[:, i].astype(int), probas[i],
                                                            manifest["target_classes"].get(t))
        # Sanity check: the manual walk must agree with sklearn's own chain prediction
        if not np.array_equal(np.column_stack([p.argmax(1) for p in probas]), predicted):
            raise RuntimeError("chain_predict_proba diverged from ClassifierChain.predict")

    top_features = {}
    for pos, est in enumerate(chain.estimators_):
        names = feats + [f"chain::{order_names[k]}" for k in range(pos)]
        imp = importances(spec, getattr(est, "regressor_", est), names)
        if imp is None or len(imp) != len(names):
            continue
        top = np.argsort(imp)[::-1][:25]
        top_features[order_names[pos]] = [
            {"feature": names[k], "source": manifest["feature_source_field"].get(names[k], names[k]),
             "importance": float(imp[k])} for k in top
        ]

    out_dir = out_dir or export_dir.parent.parent / "models" / f"export{export_dir.name}_{spec.key}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(chain, out_dir / "classifier_chain.joblib", compress=3)
    report = {
        "export_dir": str(export_dir),
        "model": spec.key,
        "model_label": spec.label,
        "family": spec.family,
        "export_family": export_family,
        "order": order_names,
        "chain_method": chain.chain_method_,
        "chain_kind": "target_chain" if any(numeric) else "classifier_chain",
        "target_tasks": {t: (manifest.get("target_tasks") or {}).get(t, "classes") for t in targets},
        "target_kinds": dict(zip(targets, kinds)),
        "cv": cv,
        "seed": seed,
        "target_classes": manifest["target_classes"],
        "missing_from_train": missing_from_train,
        "n_train": int(len(X_tr)),
        "n_features": len(feats),
        "fit_seconds": round(fit_seconds, 1),
        "device": device,
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in {**spec.default_params, **(params or {})}.items()},
        "metrics": metrics,
        "evaluation": details,
        "top_features": top_features,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    report["model_dir"] = str(out_dir)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--export-id", type=int, help="export id (looked up under $DATA_DIR/exports/<id>)")
    g.add_argument("--export-dir", type=Path)
    g.add_argument("--list-models", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--order", help="targets to train, in chain order, comma-separated; a subset leaves the others out "
                    "(default: every target, in the order listed under \"targets\" in its manifest.json)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=5,
                    help="folds for out-of-fold chain features: 0 chains the ground truth, else 2 or more")
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                    help="override a model parameter, e.g. --param n_estimators=1000 (repeatable)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="cuda needs the GPU image (XGBoost only)")
    args = ap.parse_args()

    if args.list_models:
        for m in catalog():
            print(f"{m['key']:24s} {m['family']:7s} {'nan-ok' if m['handles_nan'] else 'no-nan':7s} "
                  f"{'planned' if m['planned'] else 'available' if m['available'] else 'not installed':14s} "
                  f"{'/'.join(m['devices']):9s} "
                  f"{m['description']}")
        return
    if args.export_dir is None:
        from ..config import get_settings
        export_dir = get_settings().exports_dir / str(args.export_id)
    else:
        export_dir = args.export_dir
    overrides = {}
    for kv in args.param:
        k, _, v = kv.partition("=")
        try:
            overrides[k] = json.loads(v)
        except json.JSONDecodeError:
            overrides[k] = v
    report = train(export_dir, args.model, args.order.split(",") if args.order else None, args.seed, overrides,
                   args.cv or None, device=args.device)
    print(json.dumps({k: report[k] for k in ("model_dir", "model", "device", "order", "chain_method", "n_train",
                                             "metrics")},
                     indent=2))
    for split, by_target in report["evaluation"].items():
        for target, d in by_target.items():
            if d.get("kind") == "regression":
                m = report["metrics"][split][target]
                print(f"\n{split} · {target} — RMSE {m['rmse']:.4g}, MAE {m['mae']:.4g}, R² {m.get('r2', float('nan')):.4f}")
                continue
            print(f"\n{split} · {target} — confusion matrix (rows = actual, columns = predicted)")
            print(format_confusion(d["labels"], d["confusion_matrix"]))


def format_confusion(labels: list[str], cm: list[list[int]]) -> str:
    w = max(8, *(len(x) + 2 for x in labels), *(len(f"{v:,}") + 2 for row in cm for v in row))
    lines = [" " * w + "".join(f"{x:>{w}}" for x in labels)]
    lines += [f"{a:>{w}}" + "".join(f"{v:>{w},}" for v in row) for a, row in zip(labels, cm)]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
