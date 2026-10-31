// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ModelRun } from "../types";
import { RunsPage } from "./RunsPage";

const run = (id: number, extra: Partial<ModelRun> = {}): ModelRun =>
  ({
    id,
    export_id: 2,
    dataset_id: 1,
    experiment_id: 4,
    model: "lightgbm",
    family: "tree",
    status: "done",
    status_message: null,
    error: null,
    config: { order: ["strength"] },
    metrics: { valid: { strength: { r2: 0.912, rmse: 4.1 } } },
    report: { model_label: "LightGBM", target_kinds: { strength: "regression" } },
    mlflow_run_id: null,
    mlflow_url: null,
    created_at: "2026-10-30T10:00:00Z",
    finished_at: "2026-10-30T10:01:00Z",
    ...extra,
  }) as ModelRun;

const RUNS = [
  run(1, { chosen_at: "2026-10-30T11:00:00Z", chosen_by_name: "Tanner" }),
  run(2, { status: "running", metrics: {}, report: {} as ModelRun["report"] }),
  run(3, { status: "failed", experiment_id: 5, error: "boom" }),
];

vi.mock("../api", async (orig) => ({
  ...(await orig<typeof import("../api")>()),
  api: {
    runs: vi.fn(async () => RUNS),
    experiments: vi.fn(async () => [
      { id: 4, name: "Baseline trees" },
      { id: 5, name: "Fewer fields" },
    ]),
  },
}));
vi.mock("../components/DatasetLayout", () => ({
  useDatasetContext: () => ({ id: 1, status: "ready" }),
}));

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** Lets the queries resolve and render, a tick at a time, until `done` holds or a second passes. */
async function settle(done: () => boolean) {
  for (let i = 0; i < 50 && !done(); i++) await act(() => new Promise((r) => setTimeout(r, 20)));
}

let root: Root;
let host: HTMLDivElement;

async function mount() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  await act(async () =>
    root.render(
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={["/d/1/runs"]}>
          <Routes>
            <Route path="/d/:datasetId/runs" element={<RunsPage />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    ),
  );
  await settle(() => document.querySelectorAll("tbody tr").length === 3);
}

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

const rows = () => [...document.querySelectorAll("tbody tr")];

describe("runs page", () => {
  it("lists every run newest first, with its experiment, score and a link into that experiment", async () => {
    await mount();
    expect(rows().map((r) => r.querySelector("td")?.textContent)).toEqual([
      "LightGBM#3",
      "lightgbm#2",
      "LightGBM#1Chosen",
    ]);
    const first = rows()[2];
    expect(first.textContent).toContain("Baseline trees");
    expect(first.textContent).toContain("0.912 R²");
    expect(first.querySelector("a")?.getAttribute("href")).toBe("/d/1/experiments/4?tab=runs&export=2&run=1");
    expect(rows()[0].textContent).toContain("Fewer fields");
    expect(rows()[0].textContent).toContain("Failed");
  });

  it("filters to the runs still in progress", async () => {
    await mount();
    const inProgress = [...document.querySelectorAll("button")].find((b) => b.textContent === "In progress")!;
    await act(async () => inProgress.click());
    expect(rows().map((r) => r.querySelector("td")?.textContent)).toEqual(["lightgbm#2"]);
  });
});
