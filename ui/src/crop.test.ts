import { describe, expect, it } from "vitest";
import { MAX_ZOOM, centred, clampFrame, cropSquare, zoomAt } from "./crop";

const wide = { w: 400, h: 200 };
const tall = { w: 300, h: 900 };
const V = 200;

describe("framing a picture", () => {
  it("starts centred, keeping the middle square", () => {
    expect(cropSquare(centred(wide, V), wide, V)).toEqual({ x: 100, y: 0, size: 200 });
    const t = cropSquare(centred(tall, V), tall, V);
    expect([t.x, t.y, t.size].map((v) => Math.round(v * 1e6) / 1e6)).toEqual([0, 300, 300]);
  });

  it("never lets the picture uncover the view", () => {
    expect(clampFrame({ zoom: 1, x: 50, y: 30 }, wide, V)).toEqual({ zoom: 1, x: 0, y: 0 });
    expect(clampFrame({ zoom: 1, x: -999, y: -5 }, wide, V)).toEqual({ zoom: 1, x: -200, y: 0 });
    expect(clampFrame({ zoom: 0.2, x: 0, y: 0 }, wide, V).zoom).toBe(1);
    expect(clampFrame({ zoom: 99, x: 0, y: 0 }, wide, V).zoom).toBe(MAX_ZOOM);
  });

  it("dragged to an edge, keeps that edge", () => {
    expect(cropSquare(clampFrame({ zoom: 1, x: -999, y: 0 }, wide, V), wide, V)).toEqual({ x: 200, y: 0, size: 200 });
  });

  it("zooms around the point under the cursor", () => {
    const f = zoomAt(centred(wide, V), wide, V, 2, 0, 0); // the view's top-left corner stays on picture point (100, 0)
    expect(cropSquare(f, wide, V)).toEqual({ x: 100, y: 0, size: 100 });
    const mid = zoomAt(centred(wide, V), wide, V, 2); // around the middle: the middle 100 px
    expect(cropSquare(mid, wide, V)).toEqual({ x: 150, y: 50, size: 100 });
  });

  it("zooming out again re-covers the view", () => {
    const f = zoomAt({ zoom: 4, x: -600, y: -600 }, wide, V, 1);
    const c = cropSquare(f, wide, V);
    expect(c.size).toBe(200);
    expect(c.x + c.size).toBeLessThanOrEqual(400);
    expect(c.y).toBe(0);
  });
});
