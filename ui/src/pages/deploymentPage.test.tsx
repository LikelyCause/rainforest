// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";
import type { DeploymentDetail, DeploymentKey, DeploymentTest, DeploymentTestDetail } from "../types";
import { DeploymentPage } from "./DeploymentPage";

vi.mock("../api", async (orig) => ({
  ...(await orig<typeof import("../api")>()),
  api: {
    deployment: vi.fn(),
    experiment: vi.fn(),
    deploymentKeys: vi.fn(),
    deploymentTests: vi.fn(),
    deploymentTest: vi.fn(),
    createDeploymentKey: vi.fn(),
  },
}));
vi.mock("../components/DatasetLayout", () => ({
  useDatasetContext: () => ({ id: 1, status: "ready" }),
  useCan: () => () => true,
}));
let role = "member";
vi.mock("../hooks", async (orig) => ({
  ...(await orig<typeof import("../hooks")>()),
  useMe: () => ({ data: { id: 1, role } }),
}));

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const labels = Array.from({ length: 12 }, (_, i) => `C${String(i + 1).padStart(2, "0")}`);
const deployment = (extra: Partial<DeploymentDetail> = {}): DeploymentDetail => ({
  id: 3,
  dataset_id: 1,
  experiment_id: 1,
  experiment_name: "Vehicles",
  export_id: 2,
  batch_id: 4,
  batch_name: "Per class",
  name: "Per class",
  models: labels.map((l, i) => ({
    run_id: i + 1,
    label: l,
    model_label: null,
    kind: "binary",
    target: "vehicle",
    positive_class: l,
    threshold: 0.3,
    threshold_rule: { kind: "f1" },
  })),
  status: "active",
  created_by: 1,
  created_by_name: "Ann",
  created_at: "2026-10-28T00:00:00Z",
  retired_at: null,
  experiment_archived_at: null,
  keys: 0,
  can_manage: true,
  fields: [{ name: "x1", dtype: "Float64" }],
  example: [{ x1: 1 }, { x1: 2 }],
  problem: null,
  ...extra,
});
const test: DeploymentTest = {
  id: 7,
  deployment_id: 3,
  filename: "new.csv",
  status: "done",
  progress: 1,
  status_message: "Done",
  rows: 2,
  has_labels: false,
  metrics: null,
  warnings: { x1: { failed: 2, values: 3, flagged: true } },
  notes: [],
  id_field: null,
  labels,
  error: null,
  created_by: 1,
  created_by_name: "Ann",
  created_at: "2026-10-28T00:00:00Z",
  updated_at: "2026-10-28T00:00:00Z",
  finished_at: "2026-10-28T00:00:00Z",
};
const result = (row: number) => ({
  row,
  id: null,
  label: null,
  top: { label: "C01", score: 0.9 },
  matches: ["C01"],
  none_matched: false,
  scores: Object.fromEntries(labels.map((l) => [l, 0.123])),
});

let root: Root;
let host: HTMLDivElement;
let qc: QueryClient;
const text = () => host.textContent ?? "";
const button = (label: string) => [...host.querySelectorAll("button")].find((b) => b.textContent?.trim() === label);

async function until(check: () => boolean) {
  for (let i = 0; i < 100; i++) {
    if (check()) return;
    await act(() => new Promise((r) => setTimeout(r, 10)));
  }
  throw new Error(`timed out; the page reads: ${text()}`);
}

async function mount() {
  qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  await act(async () =>
    root.render(
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={["/d/1/deployments/3"]}>
          <Routes>
            <Route path="/d/:datasetId/deployments/:deploymentId" element={<DeploymentPage />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    ),
  );
  await until(() => text().includes("Deployment #3"));
}

let keys: DeploymentKey[] = [];
beforeEach(() => {
  vi.clearAllMocks();
  role = "member";
  keys = [];
  vi.mocked(api.deployment).mockResolvedValue(deployment());
  vi.mocked(api.experiment).mockResolvedValue({ id: 1, archived_at: null, assessor_id: null } as never);
  vi.mocked(api.deploymentKeys).mockImplementation(async () => keys);
  vi.mocked(api.deploymentTests).mockResolvedValue([test]);
  vi.mocked(api.deploymentTest).mockResolvedValue({ ...test, results: [result(1), result(2)] } as DeploymentTestDetail);
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
  qc.clear();
});

describe("DeploymentPage", () => {
  it("shows a new key once, and after Done only its prefix", async () => {
    const row: DeploymentKey = {
      id: 1,
      deployment_id: 3,
      name: "script",
      prefix: "rf_abcde",
      created_by: 1,
      created_by_name: "Ann",
      created_at: "2026-10-28T00:00:00Z",
      last_used_at: null,
      revoked_at: null,
    };
    vi.mocked(api.createDeploymentKey).mockImplementation(async () => {
      keys = [row];
      return { ...row, key: "rf_abcdefghijklmnopqrstuvwxyz012345" };
    });
    await mount();
    const name = host.querySelector<HTMLInputElement>('input[aria-label="New key\'s name"]')!;
    act(() => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(name, "script");
      name.dispatchEvent(new Event("input", { bubbles: true }));
    });
    act(() => button("Create key")!.click());
    await until(() => text().includes("rf_abcdefghijklmnopqrstuvwxyz012345"));
    expect(text()).toContain("you won't see this key again");
    await until(() => text().includes("rf_abcde…"));
    act(() => button("Done")!.click());
    expect(text()).not.toContain("rf_abcdefghijklmnopqrstuvwxyz012345");
    expect(text()).toContain("rf_abcde…");
  });

  it("hides more than 10 score columns behind Show all scores, and flags a field past half", async () => {
    await mount();
    await until(() => host.querySelectorAll(".deploy-results tbody tr").length === 2);
    const heads = () => [...host.querySelectorAll(".deploy-results th")].map((th) => th.textContent);
    expect(heads()).not.toContain("C01");
    act(() => button("Show all scores (12)")!.click());
    expect(heads()).toContain("C12");
    expect(text()).toContain("More than half failed");
  });

  it("gives a viewer no key, retire or upload controls", async () => {
    role = "viewer";
    vi.mocked(api.deployment).mockResolvedValue(deployment({ can_manage: false }));
    await mount();
    await until(() => text().includes("Viewers can look but not change anything."));
    expect(button("Create key")).toBeUndefined();
    expect(button("Retire…")).toBeUndefined();
    expect(text()).toContain("Its creator or a lead creates and revokes keys.");
    expect(button("Upload CSV or Parquet")!.disabled).toBe(true);
  });
});
