import type { StartBaselineForm } from "./baselines";
import { budgetMessage } from "./sources";
import { type View, viewQuery } from "./view";
import type {
  Analysis,
  AnalysisKind,
  AnalysisParams,
  AuthConfig,
  DerivedField,
  DerivedInput,
  DerivedPreview,
  BandPreview,
  BaselineLine,
  Batch,
  BatchDetail,
  BatchPlan,
  BatchRequest,
  BatchStarted,
  BulkRequest,
  BulkResult,
  CollectionSpec,
  ConnectRequest,
  CorrMatrix,
  Correlation,
  Dataset,
  Decision,
  DecisionInput,
  Deployment,
  DeploymentDetail,
  DeploymentKey,
  DeploymentRequest,
  DeploymentTest,
  DeploymentTestDetail,
  Device,
  Experiment,
  ExperimentDetail,
  ExportJob,
  ExportPreview,
  ExportRows,
  FamilyKey,
  FieldFilter,
  FieldListItem,
  FieldResponse,
  FieldReview,
  FinalizedRecipe,
  LibraryEntry,
  Me,
  ModelOption,
  ModelRun,
  NewDeploymentKey,
  OptionCatalog,
  PickUser,
  PreviewResult,
  Pull,
  RecipeDiff,
  RecipeVersion,
  ReviewComment,
  ReviewQueue,
  Role,
  SourceCollection,
  SourceDatabases,
  SourceInfo,
  SqlResult,
  SystemInfo,
  TeamState,
  TeamUser,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public details: string[] = [],
  ) {
    super(message);
  }
}

/** Where to send the user back to after signing in; only same-app paths. */
export function safeNext(next: string | null): string {
  if (!next || !next.startsWith("/") || next.startsWith("//") || next.includes("\\")) return "/";
  return next.startsWith("/login") || next.startsWith("/forbidden") ? "/" : next;
}

/** OIDC sign-in starts at the API, which redirects on to the provider. */
export const oidcLoginUrl = (next: string) => `/api/auth/oidc/login?next=${encodeURIComponent(next)}`;
/** Paths nginx serves itself rather than the single-page app, so reaching one needs a full page load. */
export function isServerPath(path: string): boolean {
  return path === "/mlflow" || path.startsWith("/mlflow/");
}

let unauthorizedHandler = (next: string) => window.location.assign(`/login?next=${encodeURIComponent(next)}`);

/** App.tsx swaps in a router navigation so a lapsed session doesn't reload the page. */
export function setUnauthorizedHandler(fn: (next: string) => void) {
  unauthorizedHandler = fn;
}

function signedOut() {
  if (window.location.pathname === "/login") return;
  unauthorizedHandler(window.location.pathname + window.location.search);
}

/** What the API answers, with a 409, to a write on a dataset a refresh holds (api/app/routers/_lock.py). */
export const REFRESH_LOCKED = "A refresh of this dataset is running; try again once it finishes.";

export const isRefreshLock = (e: unknown) => e instanceof ApiError && e.status === 409 && e.message === REFRESH_LOCKED;

let refreshLockHandler = () => {};

/** main.tsx refetches the datasets on a lock refusal: a page opened before the refresh began learns of it then. */
export function setRefreshLockHandler(fn: () => void) {
  refreshLockHandler = fn;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: init?.body instanceof FormData ? init.headers : { "content-type": "application/json", ...init?.headers },
  });
  if (res.status === 401) signedOut();
  if (res.status === 204) return undefined as T;
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const error = errorFromBody(res.status, body);
    if (isRefreshLock(error)) refreshLockHandler();
    throw error;
  }
  return body as T;
}

/** The API's error shapes as one ApiError: a plain detail, {message, errors | detail}, a pull over the size limit
 * ({message, estimated, max}), or FastAPI's 422 list. */
export function errorFromBody(status: number, body: any, fallback = `Request failed (${status})`): ApiError {
  const detail = body?.detail;
  if (typeof detail === "string") return new ApiError(detail, status);
  if (detail?.message && typeof detail.estimated === "number" && typeof detail.max === "number")
    return new ApiError(detail.message, status, [budgetMessage(detail.estimated, detail.max)]);
  if (detail?.message)
    return new ApiError(detail.message, status, detail.errors ?? (detail.detail ? [detail.detail] : []));
  if (Array.isArray(detail))
    return new ApiError(
      "Invalid request",
      status,
      detail.map((d: any) => `${d.loc?.join(".")}: ${d.msg}`),
    );
  return new ApiError(fallback, status);
}

const json = (method: string, data: unknown): RequestInit => ({ method, body: JSON.stringify(data) });

export const api = {
  me: () => request<Me>("/api/auth/me"),
  updateMe: (data: { name: string }) => request<Me>("/api/auth/me", json("PATCH", data)),
  authConfig: () => request<AuthConfig>("/api/auth/config"),
  pickUsers: () => request<PickUser[]>("/api/auth/users"),
  pick: (userId: number) => request<Me>("/api/auth/pick", json("POST", { user_id: userId })),
  /** In OIDC mode `logout_url` ends the provider's session too; the browser should go there next. */
  logout: () => request<{ logout_url: string | null }>("/api/auth/logout", { method: "POST" }),
  users: () => request<TeamUser[]>("/api/users"),
  /** `crop`: the square to keep, in the upright picture's pixels; without it the server keeps the centred one. */
  setAvatar: (userId: number, file: File, crop?: { x: number; y: number; size: number }) => {
    const body = new FormData();
    body.append("file", file);
    if (crop) {
      body.append("crop_x", String(crop.x));
      body.append("crop_y", String(crop.y));
      body.append("crop_size", String(crop.size));
    }
    return request<TeamUser>(`/api/users/${userId}/avatar`, { method: "PUT", body });
  },
  removeAvatar: (userId: number) => request<TeamUser>(`/api/users/${userId}/avatar`, { method: "DELETE" }),
  createUser: (data: { name: string; email?: string | null; role: Role }) =>
    request<TeamUser>("/api/users", json("POST", data)),
  patchUser: (id: number, data: { name?: string; email?: string | null; role?: Role; active?: boolean }) =>
    request<TeamUser>(`/api/users/${id}`, json("PATCH", data)),
  options: () => request<OptionCatalog>("/api/options"),
  datasets: () => request<Dataset[]>("/api/datasets"),
  dataset: (id: number) => request<Dataset>(`/api/datasets/${id}`),
  deleteDataset: (id: number) =>
    request<{ fields: number; exports: number; runs: number }>(`/api/datasets/${id}`, { method: "DELETE" }),
  patchDataset: (id: number, data: { name?: string; target_columns?: string[] }) =>
    request<Dataset>(`/api/datasets/${id}`, json("PATCH", data)),
  createSample: (data: { rows: number; fields: number; seed: number; target_kind: string; numeric_target?: boolean }) =>
    request<Dataset>("/api/datasets/sample", json("POST", data)),
  library: () => request<LibraryEntry[]>("/api/datasets/library"),
  addFromLibrary: (key: string) => request<Dataset>(`/api/datasets/library/${key}`, json("POST", {})),
  sampleLimits: () => request<{ max_cells: number }>("/api/datasets/sample/limits"),
  sources: () => request<SourceInfo[]>("/api/sources"),
  sourceDatabases: (name: string) => request<SourceDatabases>(`/api/sources/${encodeURIComponent(name)}/databases`),
  sourceCollections: (name: string, database: string) =>
    request<SourceCollection[]>(
      `/api/sources/${encodeURIComponent(name)}/databases/${encodeURIComponent(database)}/collections`,
    ),
  sourcePreview: (name: string, data: { database: string; collections: CollectionSpec[] }) =>
    request<PreviewResult>(`/api/sources/${encodeURIComponent(name)}/preview`, json("POST", data)),
  connectDataset: (data: ConnectRequest) => request<Dataset>("/api/datasets/connect", json("POST", data)),
  datasetPulls: (id: number) => request<Pull[]>(`/api/datasets/${id}/pulls`),
  refreshDataset: (id: number) => request<Pull>(`/api/datasets/${id}/refresh`, { method: "POST" }),
  fields: (id: number, view: View = null) => request<FieldListItem[]>(`/api/datasets/${id}/fields${viewQuery(view)}`),
  field: (fieldId: number, view: View = null) => request<FieldResponse>(`/api/fields/${fieldId}${viewQuery(view)}`),
  correlations: (fieldId: number, limit = 20) =>
    request<Correlation[]>(`/api/fields/${fieldId}/correlations?limit=${limit}`),
  saveDecision: (fieldId: number, d: DecisionInput, view: View = null) =>
    request<Decision>(`/api/fields/${fieldId}/decision${viewQuery(view)}`, json("PUT", d)),
  bulk: (datasetId: number, data: BulkRequest, view: View = null) =>
    request<BulkResult>(`/api/datasets/${datasetId}/decisions/bulk${viewQuery(view)}`, json("POST", data)),
  bulkUndo: (datasetId: number, batchId: string, view: View = null) =>
    request<{ restored: number }>(`/api/datasets/${datasetId}/decisions/bulk/${batchId}/undo${viewQuery(view)}`, {
      method: "POST",
    }),
  queryFields: (
    datasetId: number,
    data: { filters: FieldFilter[]; order_by: string; descending: boolean; limit?: number },
    view: View = null,
  ) =>
    request<Record<string, unknown>[]>(`/api/datasets/${datasetId}/fields/query${viewQuery(view)}`, json("POST", data)),
  sql: (sql: string) => request<SqlResult>("/api/sql", json("POST", { sql })),
  sqlExamples: () => request<{ title: string; sql: string }[]>("/api/sql/examples"),
  exportPreview: (id: number, family: string, recipeId?: number | null, experimentId?: number | null) =>
    request<ExportPreview>(
      `/api/datasets/${id}/export-preview?family=${family}${recipeId ? `&recipe_id=${recipeId}` : ""}${experimentId ? `&experiment_id=${experimentId}` : ""}`,
    ),
  /** How many rows these row rules leave to split, counted by the export's own code. */
  exportRows: (id: number, rules: Record<string, unknown>, experimentId?: number | null) =>
    request<ExportRows>(
      `/api/datasets/${id}/export-rows`,
      json("POST", { ...rules, experiment_id: experimentId ?? null }),
    ),
  exports: (id: number, experimentId?: number) =>
    request<ExportJob[]>(`/api/datasets/${id}/exports${experimentId ? `?experiment_id=${experimentId}` : ""}`),
  startExport: (id: number, cfg: Record<string, unknown>) =>
    request<ExportJob>(`/api/datasets/${id}/exports`, json("POST", cfg)),
  modelsForExport: (exportId: number) => request<ModelOption[]>(`/api/exports/${exportId}/models`),
  system: () => request<SystemInfo>("/api/system"),
  startRun: (
    exportId: number,
    data: {
      model: string;
      order?: string[];
      cv?: number;
      cv_folds?: number;
      seed?: number;
      params?: Record<string, unknown>;
      device?: Device;
    },
  ) => request<ModelRun>(`/api/exports/${exportId}/runs`, json("POST", data)),
  batchPlan: (exportId: number, target?: string | null) =>
    request<BatchPlan>(`/api/exports/${exportId}/batch-plan${target ? `?target=${encodeURIComponent(target)}` : ""}`),
  startBatch: (exportId: number, body: BatchRequest) =>
    request<BatchStarted>(`/api/exports/${exportId}/batches`, json("POST", body)),
  batches: (experimentId: number) => request<Batch[]>(`/api/experiments/${experimentId}/batches`),
  batch: (id: number) => request<BatchDetail>(`/api/batches/${id}`),
  cancelBatch: (id: number) => request<BatchDetail>(`/api/batches/${id}/cancel`, { method: "POST" }),
  deployments: (datasetId: number) => request<Deployment[]>(`/api/datasets/${datasetId}/deployments`),
  deployment: (id: number) => request<DeploymentDetail>(`/api/deployments/${id}`),
  createDeployment: (experimentId: number, body: DeploymentRequest) =>
    request<DeploymentDetail>(`/api/experiments/${experimentId}/deployments`, json("POST", body)),
  retireDeployment: (id: number) => request<DeploymentDetail>(`/api/deployments/${id}/retire`, { method: "POST" }),
  activateDeployment: (id: number) => request<DeploymentDetail>(`/api/deployments/${id}/activate`, { method: "POST" }),
  deploymentKeys: (id: number) => request<DeploymentKey[]>(`/api/deployments/${id}/keys`),
  createDeploymentKey: (id: number, name: string) =>
    request<NewDeploymentKey>(`/api/deployments/${id}/keys`, json("POST", { name })),
  revokeDeploymentKey: (id: number, keyId: number) =>
    request<DeploymentKey>(`/api/deployments/${id}/keys/${keyId}`, { method: "DELETE" }),
  deploymentTests: (id: number) => request<DeploymentTest[]>(`/api/deployments/${id}/tests`),
  deploymentTest: (testId: number) => request<DeploymentTestDetail>(`/api/deployment-tests/${testId}`),
  derived: (datasetId: number) => request<DerivedField[]>(`/api/datasets/${datasetId}/derived`),
  previewDerived: (datasetId: number, body: DerivedInput) =>
    request<DerivedPreview>(`/api/datasets/${datasetId}/derived/preview`, json("POST", body)),
  addDerived: (datasetId: number, body: DerivedInput & { name: string }) =>
    request<{ id: number; name: string }>(`/api/datasets/${datasetId}/derived`, json("POST", body)),
  removeDerived: (datasetId: number, fieldId: number) =>
    request<{ removed: string }>(`/api/datasets/${datasetId}/derived/${fieldId}`, { method: "DELETE" }),
  run: (runId: number) => request<ModelRun>(`/api/runs/${runId}`),
  updateRun: (runId: number, body: { note?: string; chosen?: boolean }) =>
    request<ModelRun>(`/api/runs/${runId}`, json("PATCH", body)),
  runs: (datasetId: number, experimentId?: number) =>
    request<ModelRun[]>(`/api/datasets/${datasetId}/runs${experimentId ? `?experiment_id=${experimentId}` : ""}`),
  experiments: (datasetId: number) => request<Experiment[]>(`/api/datasets/${datasetId}/experiments`),
  experiment: (id: number) => request<ExperimentDetail>(`/api/experiments/${id}`),
  createExperiment: (datasetId: number, data: { name: string; source: string; notes?: string | null }) =>
    request<Experiment>(`/api/datasets/${datasetId}/experiments`, json("POST", data)),
  revertAdjustments: (id: number, fieldIds: number[] | null) =>
    request<ExperimentDetail>(`/api/experiments/${id}/adjustments/revert`, json("POST", { field_ids: fieldIds })),
  patchExperiment: (id: number, data: { name?: string; notes?: string | null; archived?: boolean }) =>
    request<Experiment>(`/api/experiments/${id}`, json("PATCH", data)),
  createAnalysis: (
    datasetId: number,
    data: { kind: AnalysisKind; params: AnalysisParams; recompute?: boolean; view?: View },
  ) => request<Analysis>(`/api/datasets/${datasetId}/analyses`, json("POST", data)),
  analysis: (id: number) => request<Analysis>(`/api/analyses/${id}`),
  team: (datasetId: number) => request<TeamState>(`/api/datasets/${datasetId}/team`),
  assign: (datasetId: number, userId: number) =>
    request<TeamState>(`/api/datasets/${datasetId}/assessors`, json("POST", { user_id: userId })),
  unassign: (datasetId: number, userId: number) =>
    request<TeamState>(`/api/datasets/${datasetId}/assessors/${userId}`, { method: "DELETE" }),
  submitAssessment: (datasetId: number) =>
    request<TeamState>(`/api/datasets/${datasetId}/assessment/submit`, { method: "POST" }),
  unsubmitAssessment: (datasetId: number) =>
    request<TeamState>(`/api/datasets/${datasetId}/assessment/unsubmit`, { method: "POST" }),
  reviewQueue: (datasetId: number) => request<ReviewQueue>(`/api/datasets/${datasetId}/review`),
  fieldReview: (fieldId: number) => request<FieldReview>(`/api/fields/${fieldId}/review`),
  bands: (fieldId: number, q: { k: number } | { edges: number[] }) =>
    request<BandPreview>(`/api/fields/${fieldId}/bands?${"k" in q ? `k=${q.k}` : `edges=${q.edges.join(",")}`}`),
  addComment: (fieldId: number, body: string) =>
    request<ReviewComment[]>(`/api/fields/${fieldId}/comments`, json("POST", { body })),
  finalize: (datasetId: number, acceptOpen: boolean) =>
    request<TeamState>(`/api/datasets/${datasetId}/finalize`, json("POST", { accept_open: acceptOpen })),
  versions: (datasetId: number) => request<RecipeVersion[]>(`/api/datasets/${datasetId}/recipes`),
  baselines: (datasetId: number) => request<BaselineLine[]>(`/api/datasets/${datasetId}/baselines`),
  startBaseline: (datasetId: number, data: Omit<StartBaselineForm, "name"> & { name?: string }) =>
    request<TeamState>(`/api/datasets/${datasetId}/baselines`, json("POST", data)),
  renameBaseline: (datasetId: number, family: string, name: string) =>
    request<BaselineLine[]>(`/api/datasets/${datasetId}/baselines/${family}`, json("PATCH", { name })),
  finalizedRecipes: () => request<FinalizedRecipe[]>("/api/recipes"),
  /** `a` and `b` are recipe ids. */
  recipeDiff: (datasetId: number, a: number, b: number) =>
    request<RecipeDiff>(`/api/datasets/${datasetId}/recipes/diff?a=${a}&b=${b}`),
  newVersion: (datasetId: number, family: FamilyKey, reopen: number[]) =>
    request<TeamState>(`/api/datasets/${datasetId}/recipes`, json("POST", { family, reopen })),
  corrMatrix: (
    datasetId: number,
    q: { kept: boolean; target: string | null; limit?: number; add?: number[]; view?: View },
  ) => {
    const p = new URLSearchParams({ kept: q.kept ? "1" : "0", limit: String(q.limit ?? 60) });
    if (q.view) p.set("view", q.view);
    if (q.target) p.set("target", q.target);
    if (q.add?.length) p.set("add", q.add.join(","));
    return request<CorrMatrix>(`/api/datasets/${datasetId}/explore/correlations?${p}`);
  },
};

/** POST a form with upload progress (fetch has no upload progress events). */
function uploadForm<T>(path: string, form: FormData, onProgress: (frac: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", path);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body: any = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* non-JSON error page */
      }
      if (xhr.status === 401) signedOut();
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else {
        const error = errorFromBody(xhr.status, body, `Upload failed (${xhr.status})`);
        if (isRefreshLock(error)) refreshLockHandler();
        reject(error);
      }
    };
    xhr.onerror = () => reject(new ApiError("Network error during upload", 0));
    xhr.send(form);
  });
}

export function uploadDataset(
  file: File,
  name: string,
  targets: string,
  onProgress: (frac: number) => void,
): Promise<Dataset> {
  const form = new FormData();
  form.append("file", file);
  if (name) form.append("name", name);
  if (targets) form.append("targets", targets);
  return uploadForm("/api/datasets/upload", form, onProgress);
}

/** A file of records for a deployment to score in the background. */
export function uploadDeploymentTest(
  deploymentId: number,
  file: File,
  onProgress: (frac: number) => void,
): Promise<DeploymentTest> {
  const form = new FormData();
  form.append("file", file);
  return uploadForm(`/api/deployments/${deploymentId}/tests`, form, onProgress);
}

export const deploymentTestDownload = (testId: number, format: "csv" | "parquet") =>
  `/api/deployment-tests/${testId}/download?format=${format}`;
