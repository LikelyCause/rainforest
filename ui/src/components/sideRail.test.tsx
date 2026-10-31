// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SideRail } from "./SideRail";

const DS = {
  id: 1,
  name: "Concrete strength",
  status: "ready",
  row_count: 1030,
  column_count: 9,
  phase: "finalized",
  baseline: {
    id: 5,
    family: "tree",
    name: null,
    version: 1,
    phase: "finalized",
    label: "Trees baseline v1",
    setup: false,
  },
};
const FIELDS = [
  { id: 1, reviewed: true },
  { id: 2, reviewed: true },
  { id: 3, reviewed: false },
];
const TEAM = {
  assessors: [],
  recipe: { finalized_at: "2026-10-30T12:00:00Z", finalized_by_name: "Tanner" },
};

vi.mock("../hooks", () => ({
  useMe: () => ({ data: { id: 2, name: "Tanner", role: "lead", mode: "pick", email: null } }),
  useDataset: (id: number, enabled: boolean) => ({ data: enabled && id === 1 ? DS : undefined }),
  useFields: () => ({ data: FIELDS }),
  useTeam: () => ({ data: TEAM }),
  useAccess: () => ({ source: "baseline", canEdit: false, note: "" }),
}));
vi.mock("../api", async (orig) => ({
  ...(await orig<typeof import("../api")>()),
  api: {
    datasets: vi.fn(async () => [DS, { ...DS, id: 2, name: "Leaves" }]),
    exports: vi.fn(async () => []),
    runs: vi.fn(async () => []),
    baselines: vi.fn(async () => [
      {
        family: "tree",
        name: null,
        label: "Trees",
        versions: [{ id: 5, family: "tree", version: 1, name: null, phase: "finalized" }],
      },
    ]),
    system: vi.fn(async () => ({ mlflow: { enabled: false, browseUrl: null } })),
    logout: vi.fn(),
  },
}));

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root;
let host: HTMLDivElement;

async function mount(path: string) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  await act(async () =>
    root.render(
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={[path]}>
          <Routes>
            <Route path="*" element={<SideRail />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    ),
  );
}

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

const links = (label: string) =>
  [...document.querySelectorAll<HTMLElement>(`nav[aria-label="${label}"] a`)].map((a) => a.textContent);

const open = async () => {
  const button = document.querySelector<HTMLButtonElement>(".rail-dataset")!;
  await act(async () => button.click());
  for (let n = 0; n < 50 && !document.querySelector('.focus-menu [aria-label="Baseline"] a[href="/d/1/baseline"]'); n++)
    await act(() => new Promise((r) => setTimeout(r, 20)));
};

const store = new Map<string, string>();
vi.stubGlobal("localStorage", {
  getItem: (k: string) => store.get(k) ?? null,
  setItem: (k: string, v: string) => store.set(k, v),
  removeItem: (k: string) => store.delete(k),
});

describe("side rail", () => {
  afterEach(() => store.clear());

  it("with no dataset in focus, greys out the dataset's places and keeps People out", async () => {
    await mount("/");
    expect(links("Main")).toEqual(["Datasets", "Settings"]);
    expect(links("Dataset")).toEqual([]);
    const places = [...document.querySelectorAll('nav[aria-label="Dataset"] [aria-disabled="true"]')];
    expect(places.map((p) => p.textContent)).toEqual(["Fields", "Explore", "Experiments", "Runs", "Models"]);
    expect(document.querySelector(".rail-dataset")?.textContent).toContain("Choose a dataset");
  });

  it("inside a dataset, lists its places with the review count and marks the one open", async () => {
    await mount("/d/1/review");
    expect(links("Dataset")).toEqual(["Fields2/3", "Explore", "Experiments", "Runs", "Models"]);
    expect(document.querySelector('[aria-current="page"]')?.textContent).toBe("Fields2/3");
    expect(document.querySelector('nav[aria-label="Dataset"] a[href="/d/1/models"]')).not.toBeNull();
  });

  it("names the dataset and the baseline in focus, and keeps them after leaving the dataset", async () => {
    await mount("/d/1/runs");
    expect(document.querySelector(".rail-dataset")?.textContent).toContain("Concrete strength");
    expect(document.querySelector(".rail-dataset")?.textContent).toContain("Trees v1 · Finalized");
    act(() => root.unmount());
    host.remove();
    await mount("/settings");
    expect(document.querySelector(".rail-dataset")?.textContent).toContain("Concrete strength");
    expect(links("Dataset")).toEqual(["Fields2/3", "Explore", "Experiments", "Runs", "Models"]);
  });

  it("its menu switches dataset and lists the baseline: phase, who finalized it, its versions", async () => {
    await mount("/d/1/experiments");
    await open();
    const menu = document.querySelector(".focus-menu")!;
    expect(menu.querySelector('a[href="/d/2/experiments"]')?.textContent).toBe("Leaves");
    const baseline = menu.querySelector('[aria-label="Baseline"]')!;
    expect(baseline.querySelector('[aria-current="step"]')?.textContent).toBe("Finalized");
    expect(baseline.textContent).toContain("by Tanner");
    expect(baseline.querySelector('a[href="/d/1/baseline"]')?.textContent).toContain("Trees v1");
  });

  it("keeps Settings and People out of the profile menu", async () => {
    await mount("/");
    const profile = document.querySelector<HTMLButtonElement>('button[aria-label^="Profile menu"]')!;
    await act(async () => profile.click());
    const items = [...document.querySelectorAll('[role="menuitem"]')].map((i) => i.textContent);
    expect(items).toEqual(["Change picture…", "Switch profile"]);
  });
});
