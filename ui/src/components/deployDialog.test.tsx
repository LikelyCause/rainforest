// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api";
import { showError, toast } from "../toast";
import type { DeploymentDetail } from "../types";
import { DeployDialog } from "./DeployDialog";

vi.mock("../api", async (orig) => ({
  ...(await orig<typeof import("../api")>()),
  api: { createDeployment: vi.fn() },
}));
vi.mock("../toast", async (orig) => ({
  ...(await orig<typeof import("../toast")>()),
  toast: vi.fn(),
  showError: vi.fn(),
}));

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root;
let host: HTMLDivElement;
let qc: QueryClient;
const text = () => document.body.textContent ?? "";
const button = (label: string) => [...document.querySelectorAll("button")].find((b) => b.textContent === label)!;

async function until(check: () => boolean) {
  for (let i = 0; i < 100; i++) {
    if (check()) return;
    await act(() => new Promise((r) => setTimeout(r, 10)));
  }
  throw new Error(`timed out; the page reads: ${text()}`);
}

function type(input: HTMLInputElement, value: string) {
  const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
  act(() => {
    set.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

async function mount() {
  qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  await act(async () =>
    root.render(
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={["/d/1/experiments/1/batches/4"]}>
          <Routes>
            <Route
              path="/d/1/experiments/1/batches/4"
              element={
                <DeployDialog
                  experimentId={1}
                  datasetId={1}
                  batchId={4}
                  runIds={null}
                  models={[
                    { runId: 1, label: "V01", binary: true },
                    { runId: 2, label: "V02", binary: true },
                  ]}
                  defaultName="Per class"
                  origin="Per class #4"
                  onClose={() => {}}
                />
              }
            />
            <Route path="/d/1/models/:id" element={<p>deployment page</p>} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    ),
  );
}

beforeEach(() => vi.clearAllMocks());
afterEach(() => {
  act(() => root.unmount());
  host.remove();
  qc.clear();
});

const REFUSAL = "There are two runs for 'V01' (runs 8, 10); deploy one";

describe("DeployDialog", () => {
  it("keeps the API's refusal in the dialog, word for word, not in a toast", async () => {
    vi.mocked(api.createDeployment).mockRejectedValue(new ApiError(REFUSAL, 422));
    await mount();
    act(() => button("Deploy").click());
    await until(() => text().includes(REFUSAL));
    expect(document.querySelector('[role="dialog"] [role="alert"]')?.textContent).toContain(REFUSAL);
    expect(showError).not.toHaveBeenCalled();
    expect(toast).not.toHaveBeenCalled();
    await act(() => new Promise((r) => setTimeout(r, 50)));
    expect(text()).toContain(REFUSAL);
    // Changing the form clears a refusal that was about the request as sent.
    type(document.querySelector<HTMLInputElement>('input[placeholder="Per class"]')!, "Another");
    expect(text()).not.toContain(REFUSAL);
  });

  it("sends the batch with the rule for all and one model's own, then opens the deployment", async () => {
    vi.mocked(api.createDeployment).mockResolvedValue({ id: 9, name: "Per class" } as DeploymentDetail);
    await mount();
    act(() =>
      [...document.querySelectorAll<HTMLButtonElement>('[role="radio"]')]
        .find((b) => b.textContent === "Fixed")!
        .click(),
    );
    expect(button("Deploy").disabled).toBe(true);
    type(document.querySelector<HTMLInputElement>('input[aria-label="Threshold"]')!, "0.4");
    const own = document.querySelector<HTMLSelectElement>('select[aria-label="Threshold rule for V02"]')!;
    act(() => {
      own.value = "precision";
      own.dispatchEvent(new Event("change", { bubbles: true }));
    });
    type(document.querySelector<HTMLInputElement>('input[aria-label="Minimum precision for V02"]')!, "0.8");
    act(() => button("Deploy").click());
    await until(() => text().includes("deployment page"));
    expect(api.createDeployment).toHaveBeenCalledWith(1, {
      name: null,
      batch_id: 4,
      threshold_rule: { kind: "fixed", value: 0.4 },
      thresholds: { 2: { kind: "precision", min: 0.8 } },
    });
  });
});
