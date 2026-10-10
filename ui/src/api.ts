import type {
  Analysis, AnalysisKind, AnalysisParams, CorrMatrix, BulkRequest, BulkResult, Correlation, Dataset, DecisionInput, Decision, ExportJob, ExportPreview, FieldFilter,
  FieldListItem, FieldResponse, ModelOption, ModelRun, OptionCatalog, SqlResult, Summary, SystemInfo, Device,
} from "./types";

export class ApiError extends Error {
  constructor(message: string, public status: number, public details: string[] = []) {
    super(message);
  }
}

/** Where to send the user back to after signing in; only same-app paths. */
export function safeNext(next: string | null): string {
  return next && next.startsWith("/") && !next.startsWith("//") && !next.startsWith("/login") ? next : "/";
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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: init?.body instanceof FormData ? init.headers : { "content-type": "application/json", ...init?.headers },
  });
  if (res.status === 401 && path !== "/api/auth/login") signedOut();
  if (res.status === 204) return undefined as T;
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = body?.detail;
    if (typeof detail === "string") throw new ApiError(detail, res.status);
    if (detail?.message) throw new ApiError(detail.message, res.status, detail.errors ?? (detail.detail ? [detail.detail] : []));
    if (Array.isArray(detail)) throw new ApiError("Invalid request", res.status, detail.map((d: any) => `${d.loc?.join(".")}: ${d.msg}`));
    throw new ApiError(`Request failed (${res.status})`, res.status);
  }
  return body as T;
}

const json = (method: string, data: unknown): RequestInit => ({ method, body: JSON.stringify(data) });

export const api = {
  me: () => request<{ user: string }>("/api/auth/me"),
  login: (username: string, password: string) =>
    request<{ user: string }>("/api/auth/login", json("POST", { username, password })),
  logout: () => request<void>("/api/auth/logout", { method: "POST" }),
  options: () => request<OptionCatalog>("/api/options"),
  datasets: () => request<Dataset[]>("/api/datasets"),
  dataset: (id: number) => request<Dataset>(`/api/datasets/${id}`),
  deleteDataset: (id: number) => request<{ fields: number; exports: number; runs: number }>(`/api/datasets/${id}`, { method: "DELETE" }),
  patchDataset: (id: number, data: { model_family?: string; name?: string; target_columns?: string[] }) =>
    request<Dataset>(`/api/datasets/${id}`, json("PATCH", data)),
  createSample: (data: { rows: number; fields: number; seed: number; target_kind: string }) =>
    request<Dataset>("/api/datasets/sample", json("POST", data)),
  sampleLimits: () => request<{ max_cells: number }>("/api/datasets/sample/limits"),
  summary: (id: number) => request<Summary>(`/api/datasets/${id}/summary`),
  fields: (id: number) => request<FieldListItem[]>(`/api/datasets/${id}/fields`),
  field: (fieldId: number, family: string) => request<FieldResponse>(`/api/fields/${fieldId}?family=${family}`),
  correlations: (fieldId: number, limit = 20) =>
    request<Correlation[]>(`/api/fields/${fieldId}/correlations?limit=${limit}`),
  saveDecision: (fieldId: number, d: DecisionInput & { family: string }) =>
    request<Decision>(`/api/fields/${fieldId}/decision`, json("PUT", d)),
  bulk: (datasetId: number, data: BulkRequest) =>
    request<BulkResult>(`/api/datasets/${datasetId}/decisions/bulk`, json("POST", data)),
  bulkUndo: (datasetId: number, batchId: string) =>
    request<{ restored: number }>(`/api/datasets/${datasetId}/decisions/bulk/${batchId}/undo`, { method: "POST" }),
  queryFields: (datasetId: number, data: { filters: FieldFilter[]; order_by: string; descending: boolean; limit?: number }) =>
    request<Record<string, unknown>[]>(`/api/datasets/${datasetId}/fields/query`, json("POST", data)),
  sql: (sql: string) => request<SqlResult>("/api/sql", json("POST", { sql })),
  sqlExamples: () => request<{ title: string; sql: string }[]>("/api/sql/examples"),
  exportPreview: (id: number, family: string) =>
    request<ExportPreview>(`/api/datasets/${id}/export-preview?family=${family}`),
  exports: (id: number) => request<ExportJob[]>(`/api/datasets/${id}/exports`),
  startExport: (id: number, cfg: Record<string, number | string>) =>
    request<ExportJob>(`/api/datasets/${id}/exports`, json("POST", cfg)),
  modelsForExport: (exportId: number) => request<ModelOption[]>(`/api/exports/${exportId}/models`),
  system: () => request<SystemInfo>("/api/system"),
  startRun: (exportId: number, data: { model: string; order?: string[]; cv?: number; params?: Record<string, unknown>; device?: Device }) =>
    request<ModelRun>(`/api/exports/${exportId}/runs`, json("POST", data)),
  runs: (datasetId: number) => request<ModelRun[]>(`/api/datasets/${datasetId}/runs`),
  createAnalysis: (datasetId: number, data: { kind: AnalysisKind; params: AnalysisParams; recompute?: boolean }) =>
    request<Analysis>(`/api/datasets/${datasetId}/analyses`, json("POST", data)),
  analysis: (id: number) => request<Analysis>(`/api/analyses/${id}`),
  corrMatrix: (datasetId: number, q: { kept: boolean; target: string | null; limit?: number; add?: number[] }) => {
    const p = new URLSearchParams({ kept: q.kept ? "1" : "0", limit: String(q.limit ?? 60) });
    if (q.target) p.set("target", q.target);
    if (q.add?.length) p.set("add", q.add.join(","));
    return request<CorrMatrix>(`/api/datasets/${datasetId}/explore/correlations?${p}`);
  },
};

/** Upload with progress (fetch has no upload progress events). */
export function uploadDataset(
  file: File, name: string, targets: string, onProgress: (frac: number) => void,
): Promise<Dataset> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    if (name) form.append("name", name);
    if (targets) form.append("targets", targets);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/datasets/upload");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body: any = null;
      try { body = JSON.parse(xhr.responseText); } catch { /* non-JSON error page */ }
      if (xhr.status === 401) signedOut();
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new ApiError(typeof body?.detail === "string" ? body.detail : `Upload failed (${xhr.status})`, xhr.status));
    };
    xhr.onerror = () => reject(new ApiError("Network error during upload", 0));
    xhr.send(form);
  });
}
