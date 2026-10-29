// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, Route, Routes, useNavigate, type NavigateFunction } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api";
import { confirmDialog } from "../components/kit";
import { showError, toast } from "../toast";
import type { BatchDetail, BatchRow } from "../types";
import { BatchPage } from "./BatchPage";

vi.mock("../api", async (orig) => ({
  ...(await orig<typeof import("../api")>()),
  api: { experiment: vi.fn(), batch: vi.fn(), cancelBatch: vi.fn(), exports: vi.fn(), modelsForExport: vi.fn() },
}));
vi.mock("../components/DatasetLayout", () => ({
  useDatasetContext: () => ({ id: 1, status: "ready" }),
  useCan: () => () => true,
}));
vi.mock("../hooks", async (orig) => ({
  ...(await orig<typeof import("../hooks")>()),
  useMe: () => ({ data: { id: 1, role: "editor" } }),
}));
vi.mock("../components/Chart", () => ({ Chart: () => null }));
vi.mock("../components/kit", async (orig) => ({
  ...(await orig<typeof import("../components/kit")>()),
  confirmDialog: vi.fn(),
}));
vi.mock("../toast", async (orig) => ({
  ...(await orig<typeof import("../toast")>()),
  toast: vi.fn(),
  showError: vi.fn(),
}));

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const row = (run_id: number, positive_class: string, extra: Partial<BatchRow> = {}): BatchRow => ({
  run_id,
  status: "done",
  status_message: null,
  error: null,
  label: `${positive_class} vs rest`,
  positive_class,
  retried: false,
  fit_seconds: 1,
  finished_at: "2026-10-28T00:00:00Z",
  positives: { train: 100, valid: 40, test: 40 },
  threshold: 0.31,
  threshold_note: null,
  calibration: null,
  valid: { pr_auc: 0.8, roc_auc: 0.9, f1: 0.7 },
  test: { pr_auc: 0.78, roc_auc: 0.88, f1: 0.68 },
  weak: false,
  ...extra,
});

const batch = (id: number, extra: Partial<BatchDetail> = {}): BatchDetail => ({
  id,
  experiment_id: 1,
  export_id: 9,
  dataset_id: 1,
  kind: "one_vs_rest",
  name: `Batch ${id}`,
  spec: { model: "xgb", device: "cpu", params: {}, seed: 42, cv: 5, cv_folds: 0, target: "vehicle", balance: false },
  skipped: [],
  status: "done",
  created_by: 1,
  created_by_name: "Ann",
  created_at: "2026-10-28T00:00:00Z",
  finished_at: "2026-10-28T00:01:00Z",
  counts: { pending: 0, running: 0, done: 2, failed: 0, cancelled: 0 },
  rows: [row(id * 10 + 1, "V01"), row(id * 10 + 2, "V02")],
  weak_rule: { pr_auc: 0.5, test_positives: 30 },
  ...extra,
});
const running = (id: number) =>
  batch(id, {
    status: "running",
    finished_at: null,
    counts: { pending: 1, running: 1, done: 1, failed: 0, cancelled: 0 },
    rows: [
      row(id * 10 + 1, "V01"),
      row(id * 10 + 2, "V02", { status: "running" }),
      row(id * 10 + 3, "V03", { status: "pending" }),
    ],
  });

const batches = new Map<number, BatchDetail>();
let root: Root;
let qc: QueryClient;
let host: HTMLDivElement;
let navigate: NavigateFunction;

function Nav() {
  navigate = useNavigate();
  return null;
}

async function until(check: () => boolean) {
  for (let i = 0; i < 100; i++) {
    if (check()) return;
    await act(() => new Promise((r) => setTimeout(r, 10)));
  }
  throw new Error(`timed out; the page reads: ${host.textContent}`);
}
const text = () => host.textContent ?? "";
const checkbox = (label: string) => host.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`)!;
const button = (label: string) => [...host.querySelectorAll("button")].find((b) => b.textContent === label)!;

async function mount(id: number) {
  qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  await act(async () =>
    root.render(
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={[`/d/1/experiments/1/batches/${id}`]}>
          <Nav />
          <Routes>
            <Route path="/d/:datasetId/experiments/:experimentId/batches/:batchId" element={<BatchPage />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    ),
  );
  await until(() => text().includes(`Batch ${id}`));
}

beforeEach(() => {
  vi.clearAllMocks();
  batches.clear();
  vi.mocked(api.experiment).mockResolvedValue({
    id: 1,
    archived_at: null,
    assessor_id: null,
    owner_name: null,
  } as never);
  vi.mocked(api.exports).mockResolvedValue([]);
  vi.mocked(api.modelsForExport).mockResolvedValue([]);
  vi.mocked(api.batch).mockImplementation(async (id) => {
    const b = batches.get(id);
    if (!b) throw new ApiError("Not found", 404);
    return b;
  });
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
  qc.clear();
});

describe("BatchPage", () => {
  it("starts another batch's page with nothing selected", async () => {
    batches.set(2, batch(2));
    batches.set(4, batch(4));
    await mount(2);
    act(() => checkbox("Select V01").click());
    expect(text()).toContain("1 run selected");
    act(() => navigate("/d/1/experiments/1/batches/4"));
    await until(() => text().includes("Batch 4"));
    expect(text()).not.toContain("run selected");
  });

  describe("Cancel", () => {
    const finished = "The batch had already finished; nothing was cancelled.";

    it("sends nothing when the batch ended while the confirm was open", async () => {
      batches.set(3, running(3));
      let answer: (ok: boolean) => void = () => {};
      vi.mocked(confirmDialog).mockImplementation(() => new Promise((r) => (answer = r)));
      await mount(3);
      act(() => button("Cancel the batch").click());
      act(() => void qc.setQueryData(["batch", 3], batch(3)));
      await act(async () => answer(true));
      expect(api.cancelBatch).not.toHaveBeenCalled();
      expect(toast).toHaveBeenCalledWith(expect.objectContaining({ message: finished, tone: "info" }));
    });

    it("refetches on a 409 and says nothing was cancelled once the batch has ended", async () => {
      batches.set(3, running(3));
      vi.mocked(confirmDialog).mockResolvedValue(true);
      vi.mocked(api.cancelBatch).mockImplementation(async () => {
        batches.set(3, batch(3));
        throw new ApiError("The batch has already ended", 409);
      });
      await mount(3);
      await act(async () => button("Cancel the batch").click());
      await until(() => vi.mocked(toast).mock.calls.length + vi.mocked(showError).mock.calls.length > 0);
      expect(toast).toHaveBeenCalledWith(expect.objectContaining({ message: finished, tone: "info" }));
      expect(showError).not.toHaveBeenCalled();
      await until(() => !text().includes("Cancel the batch"));
    });

    it("shows the error on a 409 while the batch is still running", async () => {
      batches.set(3, running(3));
      vi.mocked(confirmDialog).mockResolvedValue(true);
      vi.mocked(api.cancelBatch).mockRejectedValue(new ApiError("A refresh is running on this dataset", 409));
      await mount(3);
      await act(async () => button("Cancel the batch").click());
      await until(() => vi.mocked(toast).mock.calls.length + vi.mocked(showError).mock.calls.length > 0);
      expect(showError).toHaveBeenCalledWith("The batch couldn't be cancelled", expect.any(ApiError));
      expect(toast).not.toHaveBeenCalled();
    });
  });
});
