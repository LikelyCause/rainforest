-- Views are dropped and recreated on every startup (after migrations), so they can change freely.
DROP VIEW IF EXISTS meta.v_fields;
DROP VIEW IF EXISTS meta.v_field_settings;
DROP VIEW IF EXISTS meta.v_correlations;
DROP VIEW IF EXISTS meta.v_model_metrics;

-- Flat view of every field, its shared decision and the settings for the dataset's active model family.
CREATE VIEW meta.v_fields AS
SELECT
    f.dataset_id, d.name AS dataset_name, d.model_family, f.id AS field_id, f.ordinal, f.name, f.db_column,
    f.physical_dtype, f.inferred_type, f.inference_confidence, f.is_target, f.leakage_flag,
    f.row_count, f.null_count, f.null_pct, f.n_unique, f.unique_pct, f.mean, f.std, f.min_value, f.max_value,
    f.median, f.skewness, f.kurtosis, f.zeros_pct, f.outlier_pct, f.top_value, f.top_value_pct, f.avg_length,
    f.max_target_assoc,
    dc.action, dc.semantic_type, dc.reviewed, dc.source AS decision_source, dc.notes, dc.updated_at AS decided_at,
    s.imputation, s.transform, s.scaling, s.encoding, s.add_missing_indicator, s.params,
    s.source AS settings_source
FROM meta.fields f
JOIN meta.datasets d ON d.id = f.dataset_id
LEFT JOIN meta.field_decisions dc ON dc.field_id = f.id
LEFT JOIN meta.field_settings s ON s.field_id = f.id AND s.family = d.model_family;

-- Settings for every family side by side.
CREATE VIEW meta.v_field_settings AS
SELECT f.dataset_id, f.id AS field_id, f.name, dc.action, dc.semantic_type, s.family,
       s.imputation, s.impute_value, s.add_missing_indicator, s.transform, s.scaling, s.encoding, s.params,
       s.source, s.updated_at
FROM meta.field_settings s
JOIN meta.fields f ON f.id = s.field_id
JOIN meta.field_decisions dc ON dc.field_id = f.id;

-- Correlations in both directions, with names, for "top-N correlated with X" queries.
CREATE OR REPLACE VIEW meta.v_correlations AS
SELECT c.dataset_id, c.field_a AS field_id, fa.name AS field_name,
       c.field_b AS other_field_id, fb.name AS other_field_name, fb.inferred_type AS other_type,
       fb.is_target AS other_is_target, c.method, c.value, c.strength, c.n_obs
FROM meta.field_correlations c
JOIN meta.fields fa ON fa.id = c.field_a
JOIN meta.fields fb ON fb.id = c.field_b
UNION ALL
SELECT c.dataset_id, c.field_b, fb.name, c.field_a, fa.name, fa.inferred_type,
       fa.is_target, c.method, c.value, c.strength, c.n_obs
FROM meta.field_correlations c
JOIN meta.fields fa ON fa.id = c.field_a
JOIN meta.fields fb ON fb.id = c.field_b;

-- One row per run x split x target x metric, for comparing models in SQL.
CREATE VIEW meta.v_model_metrics AS
SELECT r.dataset_id, r.export_id, r.id AS run_id, r.model, r.family, r.created_at,
       sp.key AS split, tg.key AS target, mt.key AS metric, (mt.value)::text::double precision AS value
FROM meta.model_runs r,
     jsonb_each(r.metrics) sp,
     jsonb_each(sp.value) tg,
     jsonb_each(tg.value) mt
WHERE r.status = 'done' AND jsonb_typeof(mt.value) = 'number';
