"""Log finished model runs that have no MLflow run yet (trained before MLflow was set up, or while it was down).

    docker compose exec api python -m app.modeling.mlflow_backfill             # every such run
    docker compose exec api python -m app.modeling.mlflow_backfill --dry-run   # list them
    docker compose exec api python -m app.modeling.mlflow_backfill --run-id 12 --run-id 13
"""
from __future__ import annotations

import argparse
import sys

from ..config import get_settings
from ..db import execute, fetch_all, jsonb
from . import mlflow_log


def pending(run_ids: list[int] | None = None) -> list[dict]:
    """Finished runs with no MLflow run, except those of a private experiment (logged once it stops being private,
    see experiments.flush_deferred)."""
    return fetch_all(
        """SELECT id, dataset_id, model, metrics, report, model_dir FROM meta.model_runs r
           WHERE status = 'done' AND mlflow_run_id IS NULL AND (%(ids)s::bigint[] IS NULL OR id = ANY(%(ids)s))
             AND NOT EXISTS (SELECT 1 FROM meta.experiments x JOIN meta.recipes xr ON xr.id = x.recipe_id
                              WHERE x.id = r.experiment_id AND x.assessor_id IS NOT NULL AND xr.phase = 'assessing')
           ORDER BY id""",
        {"ids": run_ids},
    )


def backfill(run_ids: list[int] | None = None, say=print) -> tuple[int, int]:
    """Returns (logged, failed). A failure is stored in the run's report as `mlflow_error`, as for a new run."""
    logged = failed = 0
    for run in pending(run_ids):
        report = {**(run["report"] or {}), "metrics": run["metrics"] or {}, "model_dir": run["model_dir"]}
        out = mlflow_log.log_model_run(run["id"], report)
        if out["mlflow_error"]:
            failed += 1
            execute("UPDATE meta.model_runs SET report = (report - 'mlflow_pending') || %s WHERE id = %s",
                    (jsonb({"mlflow_error": out["mlflow_error"]}), run["id"]))
            say(f"run {run['id']}: {out['mlflow_error']}")
        else:
            logged += 1
            execute(
                """UPDATE meta.model_runs SET mlflow_experiment_id = %s, mlflow_run_id = %s,
                          report = report - 'mlflow_error' - 'mlflow_pending' WHERE id = %s""",
                (out["mlflow_experiment_id"], out["mlflow_run_id"], run["id"]),
            )
            say(f"run {run['id']}: {mlflow_log.mlflow_url(out['mlflow_experiment_id'], out['mlflow_run_id'])}")
    return logged, failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", type=int, action="append", help="only this run (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="list the runs that would be logged")
    args = ap.parse_args()
    if args.dry_run:
        for r in pending(args.run_id):
            print(f"run {r['id']}: dataset {r['dataset_id']}, {r['model']}")
        return
    if not get_settings().mlflow_tracking_uri:
        sys.exit("MLFLOW_TRACKING_URI is not set.")
    logged, failed = backfill(args.run_id)
    print(f"{logged} logged, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
