-- Migration 021: the linear model trains ridge regression for number targets, but runs stored its classifier's name.
-- Name finished runs after what they trained: "Ridge regression" when every target was a number, "Logistic and
-- ridge regression" for a chain of both (registry.label_for). The model key stays logistic_regression.
UPDATE meta.model_runs r
   SET report = jsonb_set(r.report, '{model_label}', to_jsonb(
         CASE WHEN EXISTS (SELECT 1 FROM jsonb_each_text(r.report -> 'target_kinds') k WHERE k.value <> 'regression')
              THEN 'Logistic and ridge regression' ELSE 'Ridge regression' END))
 WHERE r.model = 'logistic_regression'
   AND jsonb_typeof(r.report -> 'target_kinds') = 'object'
   AND EXISTS (SELECT 1 FROM jsonb_each_text(r.report -> 'target_kinds') k WHERE k.value = 'regression');
