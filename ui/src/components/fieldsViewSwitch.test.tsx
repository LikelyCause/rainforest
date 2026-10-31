// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Dataset } from "../types";
import { FieldsViewSwitch } from "./FieldsViewSwitch";

const me = { current: { id: 2, name: "Tanner", role: "lead" } };
vi.mock("../hooks", () => ({ useMe: () => ({ data: me.current }) }));

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root;
let host: HTMLDivElement;

async function mount(path: string, ds: Partial<Dataset> = {}) {
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  const d = { id: 1, phase: "assessing", status: "ready", ...ds } as Dataset;
  await act(async () =>
    root.render(
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="*" element={<FieldsViewSwitch ds={d} />} />
        </Routes>
      </MemoryRouter>,
    ),
  );
}

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  me.current = { id: 2, name: "Tanner", role: "lead" };
});

const pressed = () => document.querySelector('[aria-pressed="true"]')?.textContent;
const options = () => [...document.querySelectorAll("button")].map((b) => b.textContent);

describe("fields view switch", () => {
  it("marks the view the route is on", async () => {
    await mount("/d/1/review/4");
    expect(pressed()).toBe("Focus");
    act(() => root.unmount());
    await mount("/d/1/bulk?tab=sql");
    expect(pressed()).toBe("SQL");
    act(() => root.unmount());
    await mount("/d/1/bulk?tab=derived");
    expect(pressed()).toBe("Derived");
  });

  it("puts the queue first in team review and marks it there", async () => {
    await mount("/d/1/team", { phase: "review" });
    expect(options()).toEqual(["Queue", "Focus", "Table", "SQL", "Derived"]);
    expect(pressed()).toBe("Queue");
  });

  it("falls back to Table when the tab asked for isn't the viewer's", async () => {
    me.current = { id: 4, name: "Viewer", role: "viewer" };
    await mount("/d/1/bulk?tab=sql");
    expect(options()).toEqual(["Focus", "Table"]);
    expect(pressed()).toBe("Table");
  });

  it("hides Derived while a refresh holds the dataset", async () => {
    await mount("/d/1/bulk", { status: "refreshing" });
    expect(options()).toEqual(["Focus", "Table", "SQL"]);
  });
});
