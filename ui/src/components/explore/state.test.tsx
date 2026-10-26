import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AnalysisState, type AnalysisHandle } from "./state";

const allowed = vi.hoisted(() => ({ recompute: true }));
vi.mock("../DatasetLayout", () => ({ useCan: () => (action: string) => action === "recompute" && allowed.recompute }));

const failed = {
  row: { status: "failed", error: "Out of memory\ntraceback", result: null, stale: false },
  query: {},
  recompute: () => {},
  recomputing: false,
  recomputeError: null,
} as unknown as AnalysisHandle<unknown>;

const render = () =>
  renderToStaticMarkup(
    <AnalysisState a={failed} datasetId={1}>
      {() => null}
    </AnalysisState>,
  );

describe("AnalysisState, failed", () => {
  beforeEach(() => {
    allowed.recompute = true;
  });

  it("offers Retry to someone who may recompute", () => {
    expect(render()).toContain("Retry");
  });

  it("offers no Retry when recomputing isn't allowed (a viewer, or a refresh holds the dataset)", () => {
    allowed.recompute = false;
    const html = render();
    expect(html).toContain("Analysis failed.");
    expect(html).not.toContain("Retry");
  });
});
