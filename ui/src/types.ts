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
  field_count?: number;
  reviewed_count?: number;
  kept_count?: number;
  dropped_count?: number;
  target_count?: number;
  export_count?: number;
  run_count?: number;
  last_export?: { id: number; model_family: string; status: ExportJob["status"]; created_at: string; n_features: number | null } | null;
  last_run?: { id: number; model: string; model_label: string | null; status: ModelRun["status"]; metric: string | null; value: number | null } | null;
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
  target_rule: { types: string[]; max_levels: number };
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
  last_export?: { id: number; created_at: string; status: string } | null;
  changed_since_export?: number | null;
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

export interface BulkRequest {
  field_ids?: number[];
  filters?: FieldFilter[];
  excluded_ids?: number[];
  accept_suggestion?: boolean;
  action?: "keep" | "drop";
  reviewed?: boolean;
  dry_run?: boolean;
}

export interface BulkResult {
  batch_id: string | null;
  dry_run: boolean;
  matched: number;
  changed: number;
  unchanged: number;
  reviewed_changed: number;
  transitions: Record<string, number>;
  changes: { field_id: number; name: string; before: { action: Action; semantic_type: string; reviewed: boolean }; after: { action: Action; semantic_type: string; reviewed: boolean } }[];
}

export type AnalysisKind = "map" | "tests" | "importance";
export type TargetKind = "categorical" | "numeric" | null;

export interface AnalysisParams {
  data: string;
  kept: boolean;
  target: string | null;
  sample: number;
  method?: "tsne" | "umap";
  perplexity?: number;
  n_neighbors?: number;
}

export interface Skipped { name: string; reason: string }

export interface MapResult {
  method: string;
  n: number;
  target: string | null;
  target_kind: TargetKind;
  classes: string[];
  points: [number, number, number | null][];
  hover_fields: string[];
  hover_values: (string | number | null)[][];
  skipped: Skipped[];
}

export type TestDetail =
  | { kind: "box"; groups: { class: string; n: number; min: number; q1: number; median: number; q3: number; max: number }[] }
  | { kind: "levels"; levels: string[]; counts: number[][] }
  | { kind: "scatter"; points: [number, number][] };

export interface TestRow {
  field_id: number | null;
  name: string;
  type: string;
  test: string;
  statistic: number | null;
  p: number | null;
  p_adj: number | null;
  effect: number | null;
  effect_name: string;
  n: number;
  secondary: { test: string; statistic: number | null; p: number | null } | null;
  detail: TestDetail | null;
}

export interface TestsResult {
  target: string;
  target_kind: TargetKind;
  classes: string[];
  excluded_classes: { name: string; n: number }[];
  n: number;
  rows: TestRow[];
  skipped: Skipped[];
}

export interface ImportanceResult {
  target: string | null;
  target_kind: TargetKind;
  n: number;
  mi: { field_id: number | null; name: string; value: number }[];
  pca: {
    n_features: number;
    explained: number[];
    cumulative: number[];
    components: { index: number; loadings: { feature: string; value: number }[] }[];
  } | null;
  skipped: Skipped[];
}

export type NotEnoughReason = "no_target" | "too_few_fields" | "too_few_rows" | "single_class";

export interface Analysis<R = unknown> {
  id: number;
  dataset_id: number;
  kind: AnalysisKind;
  params: AnalysisParams;
  status: "pending" | "running" | "done" | "failed";
  progress: number;
  status_message: string | null;
  error: string | null;
  result: R | { reason?: NotEnoughReason } | null;
  created_at: string;
  finished_at: string | null;
  stale?: boolean;
}

export interface CorrMatrix {
  fields: { id: number; name: string; type: string; action: Action }[];
  order: number[];
  cells: { a: number; b: number; method: CorrMethod; value: number; strength: number; n: number }[];
}
