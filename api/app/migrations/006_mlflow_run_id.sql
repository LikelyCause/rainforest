-- The MLflow run a model run was logged to (null: not logged, MLflow off, or logging failed; see report.mlflow_error).
ALTER TABLE meta.model_runs ADD COLUMN IF NOT EXISTS mlflow_experiment_id TEXT;
ALTER TABLE meta.model_runs ADD COLUMN IF NOT EXISTS mlflow_run_id TEXT;
