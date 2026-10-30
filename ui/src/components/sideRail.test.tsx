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
    datasets: vi.fn(async () => []),
    exports: vi.fn(async () => []),
    runs: vi.fn(async () => []),
    baselines: vi.fn(async () => []),
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

describe("side rail", () => {
  it("outside a dataset, offers the app's places and no dataset's", async () => {
    await mount("/");
    expect(links("Main")).toEqual(["Datasets", "People", "Settings"]);
    expect(document.querySelector('nav[aria-label="Dataset"]')).toBeNull();
    expect(document.querySelector('[aria-label="Baseline"]')).toBeNull();
  });

  it("inside a dataset, lists its places with the review count and marks the one open", async () => {
    await mount("/d/1/review");
    expect(links("Dataset")).toEqual(["Fields2/3", "Explore", "Baseline", "Experiments", "Deployments"]);
    expect(document.querySelector('[aria-current="page"]')?.textContent).toBe("Fields2/3");
    expect(links("Main")[0]).toBe("All datasets");
    expect(document.body.textContent).toContain("1,030 rows · 9 fields");
  });

  it("shows where the baseline stands: its version, the current phase, who finalized it", async () => {
    await mount("/d/1/experiments");
    const status = document.querySelector('[aria-label="Baseline"]')!;
    expect(status.textContent).toContain("Trees v1");
    expect(status.querySelector('[aria-current="step"]')?.textContent).toBe("Finalized");
    expect(status.textContent).toContain("by Tanner");
  });

  it("keeps Settings and People out of the profile menu, which the rail already has", async () => {
    await mount("/");
    const profile = document.querySelector<HTMLButtonElement>('button[aria-label^="Profile menu"]')!;
    await act(async () => profile.click());
    const items = [...document.querySelectorAll('[role="menuitem"]')].map((i) => i.textContent);
    expect(items).toEqual(["Change picture…", "Switch profile"]);
  });
});
