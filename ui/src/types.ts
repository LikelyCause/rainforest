export type DatasetStatus = "pending" | "ingesting" | "profiling" | "correlating" | "ready" | "failed";

export interface Dataset {
  id: number;
  name: string;
  source_filename: string | null;
  file_format: "csv" | "parquet" | "generated";
  target_columns: string[];
  model_family: string;
  row_count: number | null;
  column_count: number | null;
  status: DatasetStatus;
  progress: number;
  status_message: string | null;
  error: string | null;
  warnings: string[];
  created_at: string;
  updated_at: string;
  profiled_at: string | null;
  raw_table: string | null;
}

export type Action = "keep" | "drop" | "target";

export interface FieldListItem {
  id: number;
  ordinal: number;
  name: string;
  inferred_type: string;
  inference_confidence: number | null;
  is_target: boolean;
  null_pct: number | null;
  n_unique: number | null;
  max_target_assoc: number | null;
  leakage: boolean | null;
  action: Action;
  semantic_type: string;
  reviewed: boolean;
  source: "suggested" | "user";
}

export interface HistogramChart {
  kind: "histogram" | "time_histogram";
  edges: number[];
  counts: number[];
  below?: number;
  above?: number;
  xlabel?: string;
}
export interface BarsChart {
  kind: "bars";
  labels: string[];
  counts: number[];
  other?: number;
  missing?: number;
}
export type ChartPayload = HistogramChart | BarsChart;

export interface BoxSummary {
  p01: number | null; p25: number | null; p50: number | null; p75: number | null; p99: number | null;
  min: number | null; max: number | null;
}

export interface Suggestion extends Omit<Decision, "field_id" | "dataset_id" | "notes" | "reviewed" | "source" | "updated_at" | "family" | "settings_source"> {
  reasons: string[];
}

export interface FieldDetail {
  id: number;
  dataset_id: number;
  ordinal: number;
  name: string;
  db_column: string;
  physical_dtype: string;
  inferred_type: string;
  inference_confidence: number | null;
  inference_reasons: string[];
  is_target: boolean;
  row_count: number;
  null_count: number;
  null_pct: number;
  n_unique: number;
  unique_pct: number;
  mean: number | null;
  std: number | null;
  min_value: number | null;
  max_value: number | null;
  median: number | null;
  skewness: number | null;
  kurtosis: number | null;
  zeros_pct: number | null;
  outlier_pct: number | null;
  top_value: string | null;
  top_value_pct: number | null;
  avg_length: number | null;
  max_target_assoc: number | null;
  leakage_flag: boolean;
  active_family: string;
  stats: Record<string, any>;
  charts: {
    distribution?: ChartPayload;
    log_distribution?: HistogramChart;
    tokens?: BarsChart;
    weekday?: BarsChart;
    box?: BoxSummary;
  };
  suggestion: Suggestion;
}

export interface Decision {
  field_id: number;
  dataset_id: number;
  action: Action;
  semantic_type: string;
  imputation: string;
  impute_value: string | null;
  add_missing_indicator: boolean;
  transform: string;
  scaling: string;
  encoding: string;
  params: Record<string, any>;
  notes: string | null;
  reviewed: boolean;
  source: "suggested" | "user";
  settings_source: "suggested" | "user";
  family: string;
  updated_at: string;
}

export type DecisionInput = Omit<Decision, "field_id" | "dataset_id" | "source" | "settings_source" | "updated_at">;

export interface TargetAssoc {
  field_id: number;
  name: string;
  method: CorrMethod;
  value: number;
  strength: number;
}

export interface FieldResponse {
  field: FieldDetail;
  family: string;
  decision: Decision;
  target_associations: TargetAssoc[];
}

export type CorrMethod = "spearman" | "cramers_v" | "correlation_ratio";

export interface Correlation {
  field_id: number;
  name: string;
  type: string;
  is_target: boolean;
  method: CorrMethod;
  value: number;
  strength: number;
  n_obs: number;
}

export type Dim = "imputation" | "transform" | "scaling" | "encoding";

export interface Family {
  value: string;
  label: string;
  examples: string;
  handles_nan: boolean;
  scale_invariant: boolean;
  needs_onehot: boolean;
}

export interface OptionCatalog {
  types: { value: string; label: string }[];
  families: Family[];
  default_family: string;
  family_notes: Record<string, Partial<Record<Dim, string>>>;
  labels: Record<Dim, Record<string, string>>;
  /** allowed[family][type][dim] */
  allowed: Record<string, Record<string, Record<Dim, string[]>>>;
  default_params: Record<string, Record<string, any>>;
  non_feature_types: string[];
}

export interface Summary {
  by_type: { type: string; n: number }[];
  by_action: { action: Action; n: number; reviewed: number }[];
  total_fields: number;
  reviewed: number;
  top_target_associations: { field_id: number; field_name: string; target: string; method: CorrMethod; value: number; strength: number }[];
}

export interface FieldFilter { column: string; op: string; value?: unknown }

export interface SqlResult {
  columns: string[];
  rows: unknown[][];
  row_count: number;
  truncated: boolean;
  elapsed_ms: number;
}

export interface ExportPreview {
  family: string;
  customized: number;
  kept: number;
  dropped: number;
  targets: string[];
  kept_by_type: Record<string, number>;
  unreviewed: number;
  warnings: string[];
  blocking: string[];
}

export interface ExportJob {
  id: number;
  dataset_id: number;
  status: "pending" | "running" | "done" | "failed";
  progress: number;
  status_message: string | null;
  error: string | null;
  model_family: string;
  config: Record<string, any>;
  n_rows: number | null;
  n_features: number | null;
  summary: Record<string, any>;
  created_at: string;
  finished_at: string | null;
}

export interface ModelOption {
  key: string;
  label: string;
  family: string;
  handles_nan: boolean;
  available: boolean;
  description: string;
  compatible?: boolean;
  reason?: string | null;
}

export type Metrics = Record<string, Record<string, Record<string, number>>>; // split -> target -> metric

export interface TargetEvaluation {
  labels: string[];
  positive_label: string | null;
  confusion_matrix: number[][]; // rows = actual, columns = predicted
  per_class: { label: string; precision: number; recall: number; f1: number; support: number }[];
}

export interface ModelRun {
  id: number;
  export_id: number;
  dataset_id: number;
  model: string;
  family: string;
  status: "pending" | "running" | "done" | "failed";
  status_message: string | null;
  error: string | null;
  config: { order?: string[] | null; cv?: number; params?: Record<string, unknown> };
  metrics: Metrics;
  report: {
    order?: string[];
    chain_method?: string;
    fit_seconds?: number;
    model_label?: string;
    n_features?: number;
    evaluation?: Record<string, Record<string, TargetEvaluation>>; // split -> target
  };
  created_at: string;
  finished_at: string | null;
}
