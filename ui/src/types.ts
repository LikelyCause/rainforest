export type DatasetStatus = "pending" | "ingesting" | "profiling" | "correlating" | "ready" | "failed";
export type Phase = "assessing" | "review" | "finalized";

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
  /** The current baseline version (the one being worked, else the latest started) and the phase it is in. */
  baseline: BaselineRef | null;
  /** Per model type with a baseline (Trees first): its latest version and latest finalized version. */
  baselines: BaselineSummary[];
  recipe_id: number | null;
  recipe_version: number | null;
  phase: Phase | null;
  /** Datasets list only: the viewer's own assessment of the current version, when they are assigned. */
  my_assessment?: { reviewed: number; submitted: boolean } | null;
  /** Fields the current version reopened; null when every field is open. */
  reopened?: number | null;
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
/** How a target is predicted: one of its values, a number, or a band of numbers. */
export type TargetTask = "classes" | "regression" | "bins";
export type TargetBins = { k: number } | { edges: number[] };

export type SettingValue = string | number | boolean | null;

/** A run's model, chain order and settings, to fill the Train menu again. */
export interface TrainPrefill { exportId: number; model: string; order: string[]; values: Record<string, SettingValue> }

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

/** How a numeric field would be cut into bands, on the whole column (GET /api/fields/{id}/bands). */
export interface BandPreview { edges: number[]; labels: string[]; counts: number[]; missing: number }

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
    action: Action; semantic_type: string; notes: string | null; target_task?: TargetTask | null;
    target_bins?: TargetBins | null; settings: Omit<SettingsRow, "source"> | null;
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
  non_feature_types: string[];
  target_rule: { types: string[]; numeric_types: string[]; max_levels: number; tasks: TargetTask[]; default_k: number };
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
  experiment_id: number;
  experiment_name: string | null;
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
}

export interface ModelOption {
  key: string;
  label: string;
  family: string;
  handles_nan: boolean;
  available: boolean;
  description: string;
  devices?: Device[];
  /** On the roadmap: listed, never trainable. */
  planned: boolean;
  compatible?: boolean;
  reason?: string | null;
  /** The settings the Train menu offers; null for a model whose settings aren't catalogued yet. */
  settings?: ModelSetting[] | null;
}

export type Metrics = Record<string, Record<string, Record<string, number>>>; // split -> target -> metric

export interface TargetEvaluation {
  kind?: undefined;
  labels: string[];
  positive_label: string | null;
  confusion_matrix: number[][]; // rows = actual, columns = predicted
  per_class: { label: string; precision: number; recall: number; f1: number; support: number }[];
  /** One-vs-rest ROC curve per class (only the positive class for binary targets); absent on older runs. */
  roc?: RocCurve[];
}

export interface RocCurve { label: string; auc: number; fpr: number[]; tpr: number[] }

/** A number target's evaluation: sampled (actual, predicted) pairs and a histogram of the errors. */
export interface RegressionEvaluation {
  kind: "regression";
  n: number;
  scatter: { actual: number[]; predicted: number[] };
  residuals: { edges: number[]; counts: number[] };
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
  config: { order?: string[] | null; cv?: number; seed?: number; params?: Record<string, unknown>; device?: Device };
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
    /** Run of a private experiment: logged to MLflow once its recipe version reaches review. */
    mlflow_pending?: boolean;
  };
  experiment_id: number;
  mlflow_run_id: string | null;
  /** The run in the MLflow UI (served at /mlflow/); null when it wasn't logged. */
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

export type Role = "lead" | "member" | "viewer";
export type AuthMode = "pick" | "oidc";

export interface Me {
  id: number;
  name: string;
  email: string | null;
  role: Role;
  mode: AuthMode;
}

export interface AuthConfig {
  mode: AuthMode;
  provider_name: string | null;
  lead_group: string;
  member_group: string;
  /** Why sign-in can't work yet (OIDC settings missing), or null. */
  problem: string | null;
}

export interface PickUser {
  id: number;
  name: string;
  role: Role;
}

export interface TeamUser {
  id: number;
  name: string;
  email: string | null;
  role: Role;
  active: boolean;
  /** Signed in through the identity provider (OIDC mode); its name, email and role come from there. */
  sso: boolean;
  created_at: string;
  last_seen_at: string | null;
}

export type FamilyKey = "tree" | "linear" | "neural";

export interface BaselineRef {
  id: number;
  family: FamilyKey;
  version: number;
  phase: Phase;
  label: string;
  /** A new dataset's first baseline nobody has started on: its model type, start and targets can still change. */
  setup: boolean;
}

export interface BaselineSummary {
  family: FamilyKey;
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
  versions: RecipeVersion[];
  latest_finalized_id: number | null;
}

export interface Recipe {
  id: number;
  dataset_id: number;
  family: FamilyKey;
  /** "Trees baseline v2". */
  label?: string;
  /** Team state only: a new dataset's first baseline nobody has started on. */
  setup?: boolean;
  version: number;
  phase: Phase;
  /** "all": every field is open; "reopened": only the reopened ones, the rest are carried over and locked. */
  scope: "all" | "reopened";
  parent_id: number | null;
  created_by: number | null;
  created_at: string;
  finalized_by: number | null;
  finalized_at: string | null;
  schema_fingerprint: string | null;
  created_by_name?: string | null;
  finalized_by_name?: string | null;
  /** How many fields this version reopened; null when every field is open. */
  reopened?: number | null;
  reuse?: ReuseSummary | null;
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
  source: { recipe_id: number; dataset_id: number; dataset_name: string | null; family?: FamilyKey; version: number };
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
  parent_dataset_name: string | null;
}

export interface FinalizedRecipe {
  id: number;
  dataset_id: number;
  dataset_name: string;
  family: FamilyKey;
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

export type SettingsRow = Pick<Decision, "imputation" | "impute_value" | "add_missing_indicator" | "transform" | "scaling" | "encoding" | "params"> & {
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
