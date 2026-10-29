export type DatasetStatus = "pending" | "ingesting" | "profiling" | "correlating" | "ready" | "refreshing" | "failed";
export type Phase = "assessing" | "review" | "finalized";

export interface Dataset {
  id: number;
  name: string;
  source_filename: string | null;
  file_format: "csv" | "parquet" | "generated" | "mongodb";
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
  /** Where a connected dataset's rows come from (never a connection string); null for an upload or a sample. */
  source: DatasetSource | null;
  /** The newest pull of a connected dataset; null without one. */
  last_pull: PullSummary | null;
  /** The current baseline version (the one being worked, else the latest started) and the phase it is in. */
  baseline: BaselineRef | null;
  /** Per model type with a baseline (Trees first): its latest version and latest finalized version. */
  baselines: BaselineSummary[];
  recipe_id: number | null;
  recipe_version: number | null;
  phase: Phase | null;
  /** Datasets list only: the viewer's own assessment of the current version, when they are assigned. */
  my_assessment?: { reviewed: number; submitted: boolean } | null;
  /** Datasets list only: how many assess the current version and how many have submitted (counts, never choices). */
  assessors?: { total: number; submitted: number } | null;
  /** Datasets list only: in team review, the disagreements still open; else null. */
  review_open?: number | null;
  /** Datasets list only: per target column, how it is predicted, as the viewer may read it. */
  target_kinds?: Record<string, PredictAs>;
  /** Datasets list only: `target_kinds` is the suggestion nobody has decided on yet. */
  target_kinds_suggested?: boolean;
  /** Fields the current version reopened; null when every field is open. */
  reopened?: number | null;
  field_count?: number;
  reviewed_count?: number;
  kept_count?: number;
  dropped_count?: number;
  target_count?: number;
  export_count?: number;
  run_count?: number;
  last_export?: {
    id: number;
    model_family: string;
    status: ExportJob["status"];
    created_at: string;
    n_features: number | null;
  } | null;
  last_run?: {
    id: number;
    model: string;
    model_label: string | null;
    status: ModelRun["status"];
    metric: string | null;
    value: number | null;
  } | null;
  /** Datasets list only: the chosen model, else the best finished run on validation. */
  best_run?: {
    run_id: number;
    experiment_id: number;
    experiment_name: string;
    model: string;
    model_label: string;
    chosen: boolean;
    metric: string | null;
    value: number | null;
  } | null;
}

export interface DatasetSource {
  kind: "mongodb";
  connection: string;
  database: string;
  collections: { name: string; label: string; filter?: string | null; projection?: string | null }[];
  add_source: boolean;
  dedupe: boolean;
  match_on: string[] | null;
}

export interface PullSummary {
  id: number;
  status: "running" | "done" | "failed";
  started_at: string;
  finished_at: string | null;
  rows: number | null;
  has_drift: boolean;
}

export interface SourceInfo {
  name: string;
  kind: "mongodb";
}

export interface SourceDatabases {
  databases: string[] | null;
  listable: boolean;
}

export interface SourceCollection {
  name: string;
  documents: number;
}

/** How a flattened column is read: "mixed" becomes text, "json" is an array kept as JSON text. */
export type FlattenKind = "number" | "text" | "datetime" | "json" | "identifier" | "mixed" | "boolean";

export interface PreviewColumn {
  name: string;
  kind: FlattenKind;
  present_share: number;
}

export interface PreviewResult {
  columns: PreviewColumn[];
  per_source: { name: string; label: string; documents: number; sampled: number; columns: number }[];
  estimated_rows: number;
  estimated_cells: number;
  budget: { max_cells: number; fits: boolean };
  /** The columns duplicates are compared on unless Match on replaces them. */
  compared_on: string[];
}

export interface CollectionSpec {
  name: string;
  label?: string;
  filter?: string;
  projection?: string;
}

export interface ConnectRequest {
  name: string;
  source: {
    connection: string;
    database: string;
    collections: CollectionSpec[];
    add_source: boolean;
    dedupe: boolean;
    match_on?: string[];
  };
  targets?: string[];
}

/** How the new fields compare with one model type's latest finalized baseline (the same match a reused recipe gets). */
export interface FamilyDrift {
  carried: string[];
  reopened: ReopenedField[];
  missing: string[];
}

/** A pull's drift summary by model type; only types with a finalized baseline appear. */
export type PullDrift = Record<string, FamilyDrift>;

/** A pull as `…/pulls` and `…/refresh` return it; `has_drift` is only on a dataset's `last_pull`. */
export interface Pull extends Omit<PullSummary, "has_drift"> {
  per_source: { label: string; documents: number; removed_duplicates?: number }[] | null;
  flatten_notes: {
    arrays?: string[];
    mixed?: string[];
    renamed?: Record<string, string>;
    compared_on?: string[];
    source_column?: string | null;
    key_paths?: [string[], string][];
    removed_fields?: string[];
  } | null;
  drift: PullDrift | null;
  error: string | null;
  created_by_name: string | null;
}

/** How the datasets list names a target's task: one of its values, a number, or bands of numbers. */
export type PredictAs = "classes" | "number" | "bands";

export type Action = "keep" | "drop" | "target";
/** How a target is predicted: one of its values, a number, or a band of numbers. */
export type TargetTask = "classes" | "regression" | "bins";
/** Bands of a number (`k` equal-sized, or cut `edges`), or a class field's `positive` classes against the rest. */
export type TargetBins = { k: number } | { edges: number[] } | { positive: string[] };

export type SettingValue = string | number | boolean | null;

/** A run's model, chain order and settings, to fill the Train menu again. */
export interface TrainPrefill {
  exportId: number;
  model: string;
  order: string[];
  values: Record<string, SettingValue>;
  cvFolds?: number;
}

/** One model setting, as GET /api/models publishes it. */
export interface ModelSetting {
  name: string;
  label: string;
  help: string;
  group: string;
  kind: "choice" | "int" | "float" | "bool" | "int_or_none";
  default: SettingValue;
  min: number | null;
  max: number | null;
  step: number | null;
  log: boolean;
  options: { value: SettingValue; label: string; help?: string }[] | null;
  applies: "both" | "classes" | "numbers";
  when: Record<string, SettingValue[]> | null;
  none_value: number | null;
  none_label: string | null;
}

/** A transformer from the registry, as GET /api/options publishes it. */
export interface RegistryOption {
  key: string;
  dimension: "transform" | "encoding";
  types: string[];
  families: string[];
  label: string;
  help: string;
  settings: ModelSetting[];
  handles_nan: boolean;
  needs_target: boolean;
  source: string;
}

/** How a numeric field would be cut into bands, on the whole column (GET /api/fields/{id}/bands). */
export interface BandPreview {
  edges: number[];
  labels: string[];
  counts: number[];
  missing: number;
}

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
  /** Carried over into this recipe version from the one before, so it can't change here. */
  locked?: boolean;
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
  p01: number | null;
  p25: number | null;
  p50: number | null;
  p75: number | null;
  p99: number | null;
  min: number | null;
  max: number | null;
}

export interface Suggestion extends Omit<
  Decision,
  "field_id" | "dataset_id" | "notes" | "reviewed" | "source" | "updated_at" | "family" | "settings_source"
> {
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
  /** For a target: how it is predicted (null for other fields; omitted on save keeps the stored task). */
  target_task?: TargetTask | null;
  target_bins?: TargetBins | null;
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
  /** Why the field can't change in this version (carried over), or null. */
  locked?: string | null;
  /** In an experiment: the baseline's own decision and settings for this field. */
  baseline?: {
    action: Action;
    semantic_type: string;
    notes: string | null;
    target_task?: TargetTask | null;
    target_bins?: TargetBins | null;
    settings: Omit<SettingsRow, "source"> | null;
  } | null;
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
  /** param_rules[encoding][param] */
  param_rules?: Record<string, Record<string, { integer: boolean; min: number; max: number }>>;
  /** Each option's parameters, in form order: an encoding's, or a registry option's. */
  settings: Record<string, ModelSetting[]>;
  transforms: RegistryOption[];
  non_feature_types: string[];
  target_rule: { types: string[]; numeric_types: string[]; max_levels: number; tasks: TargetTask[]; default_k: number };
}

export interface FieldFilter {
  column: string;
  op: string;
  value?: unknown;
}

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
  last_export?: { id: number; created_at: string; status: string; recipe_id: number | null } | null;
  changed_since_export?: number | null;
  recipe?: { id: number; version: number; phase: Phase };
  /** "assessment": the preview of a personal experiment, from its assessor's assessment as it stands. */
  source?: "recipe" | "assessment";
  experiment_id?: number | null;
}

/** Where exports and runs live. Built from a finalized recipe version (team) or one assessor's assessment
 * (personal), which only its assessor sees while that version is assessing. */
export interface Experiment {
  id: number;
  dataset_id: number;
  name: string;
  /** Who created it; null for the Team and Imported experiments the app makes. */
  owner: number | null;
  owner_name: string | null;
  recipe_id: number;
  recipe_family: FamilyKey;
  recipe_version: number;
  recipe_name: string | null;
  recipe_phase: Phase;
  /** The Team experiment the app makes for a finalized baseline's exports. */
  is_default: boolean;
  /** Set on a personal experiment: whose assessment it is built from. */
  assessor_id: number | null;
  /** `recipe:<recipe_id>` or `assessment:<recipe_id>:<user_id>`. */
  source: string;
  notes: string | null;
  created_at: string;
  archived_at: string | null;
  /** Personal and its version still assessing: only its assessor can see it. */
  private: boolean;
  export_count?: number;
  run_count?: number;
  mlflow_experiment_id?: string | null;
  /** Its best finished run on the validation split. */
  best?: { metric: string; value: number; run_id: number; target: string } | null;
  /** The run a lead marked as the experiment's chosen model. */
  chosen?: { run_id: number; model: string; model_label: string | null; by: string | null } | null;
  /** The experiment's targets in chain order (team experiments). */
  target_order?: string[] | null;
  /** List rows and detail: fields that differ from the baseline, and whether the targets do. */
  adjusted_count?: number;
  targets_changed?: boolean;
  /** Whether the viewer may adjust its fields (its owner, on an experiment they made). */
  can_adjust?: boolean;
}

export interface Adjustments {
  fields: { name: string; change: "changed" | "added" | "removed"; changes: DiffChange[] }[];
  targets: { before: string[]; after: string[] } | null;
  count: number;
  targets_changed: boolean;
  /** Fields of the baseline a refresh took out of the data, with the pull that did; not adjustments. */
  removed_by_refresh: { name: string; pull_id: number | null }[];
}

export interface ExperimentDetail extends Experiment {
  exports: ExportJob[];
  runs: ModelRun[];
  adjustments: Adjustments;
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
  recipe_id: number | null;
  recipe_family?: FamilyKey | null;
  recipe_version: number | null;
  recipe_name?: string | null;
  experiment_id: number;
  experiment_name: string | null;
  /** The pull of a connected dataset the export read; null for any other dataset. */
  pull_id: number | null;
  n_rows: number | null;
  n_features: number | null;
  summary: Record<string, any>;
  created_at: string;
  finished_at: string | null;
}

export type Device = "cpu" | "cuda";

export interface GpuInfo {
  available: boolean;
  name: string;
  memory_mb: number | null;
  driver: string | null;
  count: number;
  reason: string | null;
}

export interface SystemInfo {
  gpu: GpuInfo | null;
  /** Whether runs are logged to MLflow (links to it show only then), and the address browsers open it at. */
  mlflow: { enabled: boolean; browseUrl: string | null };
}

/** What the export's row rules leave to split (POST /export-rows): the file's rows, the rows the rules keep, the kept
 * rows the split leaves out because a target is empty, what the split divides, and the groups it keeps together. */
export interface ExportRows {
  rows: number;
  kept: number;
  removed: Record<string, number>;
  missing_target: number;
  split_rows: number;
  groups: number | null;
  /** The exact counts the export's own split makes with the shares and seed asked for; null when it would refuse. */
  split: { train: number; valid: number; test: number } | null;
  split_error: string | null;
}

export interface ModelOption {
  key: string;
  label: string;
  family: string;
  handles_nan: boolean;
  available: boolean;
  description: string;
  /** The name and description when every target is a number and the model trains a different method for them
   * (ridge regression for logistic regression); `mixed_label` names a chain of both. Null: one name for both. */
  regressor_label?: string | null;
  regressor_description?: string | null;
  mixed_label?: string | null;
  devices?: Device[];
  /** On the roadmap: listed, never trainable. */
  planned: boolean;
  compatible?: boolean;
  reason?: string | null;
  /** The settings the Train menu offers; null for a model whose settings aren't catalogued yet. */
  settings?: ModelSetting[] | null;
}

export type Metrics = Record<string, Record<string, Record<string, number>>>; // split -> target -> metric

/** K-fold scores on a run's training rows: per target and metric, the mean, spread and each fold's value. */
export interface CrossValidation {
  folds: number;
  stratified_on: string | null;
  /** Why K or the folds differ from what was asked, or why CV was skipped. */
  note: string | null;
  seconds?: number;
  targets: Record<string, Record<string, { mean: number; std: number; folds: (number | null)[] }>>;
}

export interface TargetEvaluation {
  kind?: undefined;
  labels: string[];
  positive_label: string | null;
  confusion_matrix: number[][]; // rows = actual, columns = predicted
  per_class: { label: string; precision: number; recall: number; f1: number; support: number }[];
  /** One-vs-rest ROC curve per class (only the positive class for binary targets); absent on older runs. */
  roc?: RocCurve[];
  /** Binary targets: the positive class's precision, recall and F1, and the share flagged, at each cut-off. */
  thresholds?: { threshold: number; precision: number; recall: number; f1: number; flagged: number }[];
}

export interface RocCurve {
  label: string;
  auc: number;
  fpr: number[];
  tpr: number[];
}

/** A number target's evaluation: sampled (actual, predicted) pairs and a histogram of the errors. */
export interface RegressionEvaluation {
  kind: "regression";
  n: number;
  scatter: { actual: number[]; predicted: number[] };
  residuals: { edges: number[]; counts: number[] };
  /** Actual and predicted values put in ranges cut on the training rows, scored as classes (older runs lack it). */
  bands?: {
    labels: string[];
    edges: number[];
    confusion_matrix: number[][];
    per_class: { label: string; precision: number; recall: number; f1: number; support: number }[];
  };
}

export interface ModelRun {
  id: number;
  export_id: number;
  dataset_id: number;
  model: string;
  family: string;
  /** `cancelled`: a batch run cancelled before it started. */
  status: "pending" | "running" | "done" | "failed" | "cancelled";
  status_message: string | null;
  error: string | null;
  config: {
    order?: string[] | null;
    cv?: number;
    cv_folds?: number;
    seed?: number;
    params?: Record<string, unknown>;
    device?: Device;
    /** A one-vs-rest run: the class it trains against the rest of `target`. */
    positive_class?: string | null;
    target?: string | null;
    balance?: boolean;
    /** What its batch planned for it, shown before it has a report. */
    batch_plan?: {
      settings?: Record<string, SettingValue>;
      positives?: { train: number; valid: number; test: number };
    };
  };
  /** The batch that started it, or null. */
  batch_id?: number | null;
  metrics: Metrics;
  report: {
    order?: string[];
    chain_method?: string;
    fit_seconds?: number;
    device?: Device;
    model_label?: string;
    n_features?: number;
    evaluation?: Record<string, Record<string, TargetEvaluation | RegressionEvaluation>>; // split -> target
    target_kinds?: Record<string, string>;
    mlflow_error?: string;
    cross_validation?: CrossValidation | null;
    /** Per target (chain order), the features the model leaned on most. */
    top_features?: Record<string, { feature: string; source: string; importance: number }[]>;
    /** Run of a private experiment: logged to MLflow once its recipe version reaches review. */
    mlflow_pending?: boolean;
    /** A one-vs-rest run: precision, recall and F1 of its class at every 0.01 of threshold, per split. */
    threshold_curve?: Record<string, ThresholdPoint[]>;
    f1_threshold?: F1Threshold;
    calibration?: Calibration;
  };
  experiment_id: number;
  created_by?: number | null;
  note?: string | null;
  /** Set when a lead marked this run as the experiment's chosen model. */
  chosen_at?: string | null;
  chosen_by_name?: string | null;
  mlflow_run_id: string | null;
  /** The run in the MLflow UI, on the server's MLFLOW_BROWSE_URL; null when it wasn't logged. */
  mlflow_url: string | null;
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
  changes: {
    field_id: number;
    name: string;
    before: { action: Action; semantic_type: string; reviewed: boolean };
    after: { action: Action; semantic_type: string; reviewed: boolean };
  }[];
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

export interface Skipped {
  name: string;
  reason: string;
}

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
  | {
      kind: "box";
      groups: { class: string; n: number; min: number; q1: number; median: number; q3: number; max: number }[];
    }
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

export type Role = "lead" | "member" | "viewer";
export type AuthMode = "pick" | "oidc";

export interface Me {
  id: number;
  name: string;
  email: string | null;
  role: Role;
  mode: AuthMode;
  key?: string | null;
  avatar_url?: string | null;
  /** Whether you can rename yourself: not when names come from RAINFOREST_USERS or the identity provider. */
  can_rename?: boolean;
}

export interface AuthConfig {
  mode: AuthMode;
  provider_name: string | null;
  lead_group: string;
  member_group: string;
  /** Why sign-in can't work yet (OIDC settings missing), or null. */
  problem: string | null;
  /** Accounts come from RAINFOREST_USERS: they can't be added or edited in the app. */
  accounts_from_env?: boolean;
}

export interface PickUser {
  id: number;
  name: string;
  role: Role;
  key: string | null;
  avatar_url: string | null;
}

export interface TeamUser {
  id: number;
  name: string;
  email: string | null;
  role: Role;
  active: boolean;
  /** Signed in through the identity provider (OIDC mode); its name, email and role come from there. */
  sso: boolean;
  key?: string | null;
  avatar_url?: string | null;
  created_at: string;
  last_seen_at: string | null;
}

export type FamilyKey = "tree" | "linear" | "neural";

export interface BaselineRef {
  id: number;
  family: FamilyKey;
  name: string | null;
  version: number;
  phase: Phase;
  label: string;
  /** A new dataset's first baseline nobody has started on: its model type, start and targets can still change. */
  setup: boolean;
}

export interface BaselineSummary {
  family: FamilyKey;
  name: string | null;
  id: number;
  version: number;
  phase: Phase;
  label: string;
  latest_finalized_id: number | null;
}

/** One model type's baseline and its versions (GET /datasets/{id}/baselines). */
export interface BaselineLine {
  family: FamilyKey;
  label: string;
  name: string | null;
  versions: RecipeVersion[];
  latest_finalized_id: number | null;
}

export interface Recipe {
  id: number;
  dataset_id?: number;
  family: FamilyKey;
  /** The baseline's name, shared by its versions; null when unnamed. */
  name?: string | null;
  /** "Dropout risk v2 (Trees)", or "Trees baseline v2" unnamed. */
  label?: string;
  /** Team state only: a new dataset's first baseline nobody has started on. */
  setup?: boolean;
  version: number;
  phase: Phase;
  /** "all": every field is open; "reopened": only the reopened ones, the rest are carried over and locked. */
  scope: "all" | "reopened";
  parent_id: number | null;
  created_by?: number | null;
  created_at: string;
  finalized_by?: number | null;
  finalized_at: string | null;
  created_by_name?: string | null;
  finalized_by_name?: string | null;
  /** How many fields this version reopened; null when every field is open. */
  reopened?: number | null;
  reuse?: ReuseSummary | null;
  start_summary?: StartSummary | null;
}

/** What a new version could not carry over from the version before, because a refresh changed the data since. */
export interface StartSummary {
  missing: string[];
  targets_not_carried: { name: string; problem: string }[];
}

export type ReopenReason = "new" | "type_changed" | "drift" | "target" | "settings";

export interface ReopenedField {
  name: string;
  reason: ReopenReason;
  before?: string;
  after?: string;
  drift?: { null_pct?: [number, number]; n_unique?: [number, number] };
  problem?: string;
}

/** How a finalized recipe was matched onto a new dataset. */
export interface ReuseSummary {
  source: {
    recipe_id: number;
    dataset_id: number;
    dataset_name: string | null;
    family?: FamilyKey;
    version: number;
    name?: string | null;
  };
  carried: string[];
  reopened: ReopenedField[];
  missing: string[];
  targets: string[];
  targets_not_carried: { name: string; problem: string }[];
}

export interface RecipeVersion extends Recipe {
  created_by_name: string | null;
  finalized_by_name: string | null;
  reopened: number | null;
  reuse: ReuseSummary | null;
  parent_dataset_id: number | null;
  parent_family: FamilyKey | null;
  parent_version: number | null;
  parent_name?: string | null;
  parent_dataset_name: string | null;
  /** How its team review went; null while assessing or when it never had assessors. */
  review?: VersionReview | null;
}

export interface FinalizedRecipe {
  id: number;
  dataset_id: number;
  dataset_name: string;
  family: FamilyKey;
  name: string | null;
  label: string;
  version: number;
  field_count: number;
  finalized_by: number | null;
  finalized_by_name: string | null;
  finalized_at: string;
}

export interface DiffChange {
  key: string;
  /** The model family a setting belongs to; null for the shared decision. */
  family: string | null;
  before: unknown;
  after: unknown;
}

export interface DiffField {
  name: string;
  change: "changed" | "added" | "removed";
  changes: DiffChange[];
}

export interface RecipeDiff {
  a: Pick<Recipe, "id" | "version" | "phase">;
  b: Pick<Recipe, "id" | "version" | "phase">;
  fields: DiffField[];
  targets: { before: string[]; after: string[] } | null;
}

export interface AssessorProgress {
  user_id: number;
  name: string;
  assigned_at: string;
  assigned_by: number | null;
  submitted_at: string | null;
  reviewed: number;
  total: number;
  /** When they last reviewed or changed a field; null while their assessment is as it was seeded. */
  last_active_at?: string | null;
}

/** How a version's team review went (GET /datasets/{id}/baselines). */
export interface VersionReview {
  adopted: number;
  disagreements: number;
  /** Null for versions finalized before the summary was kept. */
  settled: number | null;
  /** Null while in review, and for versions finalized before the summary was kept. */
  open_at_finalize: number | null;
}

export interface ReviewSummary {
  fields: number;
  adopted: number;
  disagreements: number;
  open: number;
}

export interface TeamState {
  recipe: Recipe;
  assessors: AssessorProgress[];
  me: { submitted: boolean; reviewed: number; total: number } | null;
  review: ReviewSummary | null;
}

export interface Assessment {
  user_id: number;
  user_name: string;
  action: Action;
  semantic_type: string;
  target_task?: TargetTask | null;
  target_bins?: TargetBins | null;
  notes: string | null;
}

export interface QueueItem {
  field_id: number;
  name: string;
  inferred_type: string;
  team: { action: Action; semantic_type: string; target_task: TargetTask | null; target_bins: TargetBins | null };
  resolved: boolean;
  comments: number;
  assessments: Assessment[];
}

export interface ReviewQueue {
  recipe: Recipe;
  summary: ReviewSummary;
  queue: QueueItem[];
}

export type SettingsRow = Pick<
  Decision,
  "imputation" | "impute_value" | "add_missing_indicator" | "transform" | "scaling" | "encoding" | "params"
> & {
  source: "suggested" | "user";
};

export interface ReviewComment {
  id: number;
  user_id: number;
  user_name: string;
  body: string;
  created_at: string;
}

export interface FieldReview {
  disagreement: boolean;
  assessments: (Assessment & { settings: Record<string, SettingsRow> })[];
  comments: ReviewComment[];
}

/** A real dataset that ships with the app (GET /api/datasets/library). */
export interface LibraryEntry {
  key: string;
  name: string;
  description: string;
  targets: string[];
  rows: number;
  columns: number;
  source_url: string;
  citation: string;
}

export type DerivedInput =
  | { kind: "formula"; spec: { expr: string } }
  | {
      kind: "groups";
      spec: {
        source: string;
        other: string;
        groups: { label: string; values?: string[]; ranges?: [number, number][] }[];
      };
    }
  | { kind: "count"; spec: { fields: string[]; values: string[] } };
export type DerivedField = { id: number; name: string } & DerivedInput;
export interface DerivedPreview {
  type: "number" | "text";
  missing: number;
  rows: number;
  sample: (string | number | null)[];
  distinct: number;
}

/** One class of a target and its rows in each split, as GET /api/exports/{id}/batch-plan gives them. */
export interface ClassCount {
  class: string;
  train: number;
  valid: number;
  test: number;
  /** Why a batch would skip the class (the server's rule), or null when it trains. */
  reason: string | null;
}
/** A class a one-vs-rest batch doesn't train, and why ("not selected" when it was unticked). */
export interface Skip {
  class: string;
  reason: string;
}
/** A search setting's range of numbers: drawn on a log scale, as whole numbers, or neither. */
export interface Range {
  low: number;
  high: number;
  log: boolean;
  integer: boolean;
}
/** A search's space as the API takes it: per setting, a list of values or a range. */
export type Space = Record<string, SettingValue[] | Range>;

/** GET /api/exports/{id}/batch-plan: the limits a batch is held to and, with a class target, its classes' counts. */
export interface BatchPlan {
  min_positives: number;
  max_runs: number;
  target: string | null;
  classes: ClassCount[] | null;
  valid_rows: number | null;
  /** The classes a batch of every class would skip. */
  skipped: Skip[] | null;
  /** A one-vs-rest search's default classes. */
  search_classes: string[] | null;
}

export type BatchKind = "one_vs_rest" | "search";
export interface Sampler {
  kind: "grid" | "random";
  n?: number | null;
  seed?: number;
}

/** POST /api/exports/{id}/batches. */
export interface BatchRequest {
  kind: BatchKind;
  name?: string | null;
  model: string;
  device: Device;
  params: Record<string, SettingValue>;
  seed?: number;
  cv?: number;
  cv_folds?: number;
  target?: string | null;
  classes?: string[] | null;
  balance?: boolean;
  space?: Space;
  sampler?: Sampler;
  ranking?: string | null;
  search_classes?: string[] | null;
  /** A search on the whole target: the targets each run trains, in chain order. */
  order?: string[] | null;
}

export interface Batch {
  id: number;
  experiment_id: number;
  export_id: number;
  dataset_id: number;
  kind: BatchKind;
  name: string;
  spec: {
    model: string;
    device: Device;
    params: Record<string, SettingValue>;
    seed: number;
    cv: number;
    cv_folds: number;
    target: string | null;
    balance: boolean;
    classes?: string[];
    space?: Space;
    sampler?: Required<Sampler> | null;
    ranking?: string;
    ranked_target?: string;
    search_classes?: string[] | null;
    order?: string[] | null;
  };
  skipped: Skip[];
  status: "running" | "done" | "cancelled";
  created_by: number | null;
  created_by_name: string | null;
  created_at: string;
  finished_at: string | null;
  counts: Record<"pending" | "running" | "done" | "failed" | "cancelled", number>;
}

export interface ThresholdPoint {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  flagged?: number;
}
/** A one-vs-rest run's threshold: the validation F1 maximum, its F1 on each split, and a note when there is none. */
export interface F1Threshold {
  threshold: number;
  valid_f1: number | null;
  test_f1: number | null;
  note: string | null;
}
/** How a one-vs-rest run's scores were calibrated on validation. The validation Brier pair is in-sample (the
 * calibrator was fit there); `test` holds the honest one. */
export interface Calibration {
  method: "isotonic" | "sigmoid";
  valid_positives: number;
  valid_negatives: number;
  brier_before: number;
  brier_after: number;
  test?: { brier_before: number; brier_after: number };
}
export interface SplitScores {
  pr_auc: number | null;
  roc_auc: number | null;
  f1: number | null;
}

/** One run of GET /api/batches/{id}. The one-vs-rest fields are there for a run with a positive class, the search
 * fields for a search's runs. */
export interface BatchRow {
  run_id: number;
  status: ModelRun["status"];
  status_message: string | null;
  error: string | null;
  label: string | null;
  positive_class: string | null;
  retried: boolean;
  fit_seconds: number | null;
  finished_at: string | null;
  positives?: { train: number; valid: number; test: number } | null;
  threshold?: number | null;
  threshold_note?: string | null;
  calibration?: Calibration | null;
  valid?: SplitScores;
  test?: SplitScores;
  /** Null until the run is done. */
  weak?: boolean | null;
  settings?: Record<string, SettingValue>;
  value?: number | null;
  cv_std?: number | null;
}

/** A search's combination, best first; unranked (`rank` null) until every one of its runs has the metric. */
export interface LeaderboardEntry {
  rank: number | null;
  settings: Record<string, SettingValue>;
  run_ids: number[];
  value: number | null;
  /** A one-vs-rest search: each class's value. */
  classes: Record<string, number | null>;
  cv_std: number | null;
  fit_seconds: number;
}

export interface BatchDetail extends Batch {
  rows: BatchRow[];
  leaderboard?: LeaderboardEntry[];
  /** OVR_WEAK_PR_AUC and the fewest test positives a class may have before its row is weak. */
  weak_rule: { pr_auc: number; test_positives: number };
}

/** The 202 of starting a batch: `redraws` is how many random draws the checks threw away. */
export interface BatchStarted {
  batch: Batch;
  run_ids: number[];
  skipped: Skip[];
  redraws: number;
}

/** How a one-vs-rest model's threshold is chosen on its validation curve, as the API stores it. */
export type ThresholdRule = { kind: "f1" } | { kind: "precision"; min: number } | { kind: "fixed"; value: number };

/** One model of a deployment. A binary model scores one class against the rest and has a threshold. */
export interface DeploymentModel {
  run_id: number;
  /** The class a binary model scores, or the target. */
  label: string;
  model_label: string | null;
  kind: "binary" | "multiclass" | "regression";
  target: string;
  positive_class?: string;
  threshold: number | null;
  threshold_rule: ThresholdRule | null;
  /** Set when the rule fell back to the F1 choice. */
  threshold_note?: string;
}

export interface Deployment {
  id: number;
  dataset_id: number;
  experiment_id: number;
  experiment_name: string;
  export_id: number;
  batch_id: number | null;
  batch_name: string | null;
  name: string;
  models: DeploymentModel[];
  status: "active" | "retired";
  created_by: number | null;
  created_by_name: string | null;
  created_at: string;
  retired_at: string | null;
  experiment_archived_at: string | null;
  /** How many unrevoked keys it has. */
  keys: number;
  /** Whether the viewer may retire it and manage its keys: its creator or a lead. */
  can_manage: boolean;
}

/** GET /api/deployments/{id}: what a record must carry and two of the dataset's rows of it, or `problem`. */
export interface DeploymentDetail extends Deployment {
  fields: DeploymentField[] | null;
  example: Record<string, unknown>[] | null;
  problem: string | null;
}
export interface DeploymentField {
  name: string;
  dtype: string;
}

/** POST /api/experiments/{id}/deployments. */
export interface DeploymentRequest {
  name: string | null;
  run_ids?: number[];
  batch_id?: number;
  threshold_rule?: ThresholdRule;
  thresholds?: Record<number, ThresholdRule>;
}

export interface DeploymentKey {
  id: number;
  deployment_id: number;
  name: string;
  /** The key's first 8 characters. */
  prefix: string;
  created_by: number | null;
  created_by_name: string | null;
  created_at: string;
  last_used_at: string | null;
  revoked_at: string | null;
}

/** The 201 of creating a key: the only time `key` is shown. */
export interface NewDeploymentKey extends DeploymentKey {
  key: string;
}

/** Values of a field that didn't convert to its type; `flagged` when more than half of them failed. */
export interface FieldWarning {
  failed: number;
  values: number;
  flagged: boolean;
}

export interface ClassMetrics {
  label: string;
  support: number;
  threshold: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  pr_auc: number | null;
}

/** What a labelled test file says of the deployment. */
export type TestMetrics =
  | {
      kind: "classes";
      labelled_rows: number;
      rows_without_label: number;
      top1_accuracy: number | null;
      rows_label_without_model: number;
      /** Null for a single run. */
      nothing_matched_share: number | null;
      classes: ClassMetrics[];
      /** `counts[i][j]`: rows labelled `labels[i]` whose top is `labels[j]`; the last column a top outside the list. */
      confusion: { labels: string[]; counts: number[][] };
    }
  | {
      kind: "number";
      labelled_rows: number;
      rows_without_label: number;
      rows_label_not_a_number: number;
      mae: number | null;
      rmse: number | null;
      r2: number | null;
    };

export interface DeploymentTest {
  id: number;
  deployment_id: number;
  filename: string;
  status: "pending" | "running" | "done" | "failed";
  progress: number;
  status_message: string | null;
  rows: number | null;
  has_labels: boolean | null;
  metrics: TestMetrics | null;
  warnings: Record<string, FieldWarning> | null;
  /** What the upload reader noted about the file. */
  notes: string[];
  /** The identifier field the results carry, or null for the row number. */
  id_field: string | null;
  /** What the scores are keyed by. */
  labels: string[] | null;
  error: string | null;
  created_by: number | null;
  created_by_name: string | null;
  created_at: string;
  updated_at: string;
  finished_at: string | null;
}

/** One record's result in a test: predict's result with its row number, identifier and the file's label. */
export interface TestResult {
  row: number;
  id: string | number | null;
  label: string | number | null;
  top: { label: string; score: number | null } | null;
  matches: string[] | null;
  none_matched: boolean | null;
  scores: Record<string, number | null>;
}

export interface DeploymentTestDetail extends DeploymentTest {
  /** The first 500 records' results, once it is done. */
  results: TestResult[];
}
