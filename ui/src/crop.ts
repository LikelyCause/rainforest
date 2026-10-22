/** Framing a picture in a square view: the picture always covers the view, at 1× (just covering) up to MAX_ZOOM. */

export const MAX_ZOOM = 4;

/** `x`, `y`: where the picture's top-left corner sits in the view, in view pixels (0 or less). */
export interface Frame {
  zoom: number;
  x: number;
  y: number;
}

export interface Size {
  w: number;
  h: number;
}

/** View pixels per picture pixel at 1×: the picture's shorter side just fills the view. */
export const coverScale = (img: Size, view: number) => Math.max(view / img.w, view / img.h);

const between = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);

/** The same frame, moved and zoomed only as far as keeps the view covered. */
export function clampFrame(f: Frame, img: Size, view: number): Frame {
  const zoom = between(f.zoom, 1, MAX_ZOOM);
  const s = coverScale(img, view) * zoom;
  return { zoom, x: between(f.x, view - img.w * s, 0), y: between(f.y, view - img.h * s, 0) };
}

/** 1×, centred: what the server keeps when nobody frames anything. */
export function centred(img: Size, view: number): Frame {
  const s = coverScale(img, view);
  return { zoom: 1, x: (view - img.w * s) / 2, y: (view - img.h * s) / 2 };
}

/** Zoom to `zoom`, keeping the picture point under view point (`px`, `py`) where it is. */
export function zoomAt(f: Frame, img: Size, view: number, zoom: number, px = view / 2, py = view / 2): Frame {
  const base = coverScale(img, view);
  const z = between(zoom, 1, MAX_ZOOM);
  const before = base * f.zoom;
  const after = base * z;
  return clampFrame(
    { zoom: z, x: px - ((px - f.x) / before) * after, y: py - ((py - f.y) / before) * after },
    img,
    view,
  );
}

/** The square the view shows, in picture pixels: what the server crops. */
export function cropSquare(f: Frame, img: Size, view: number): { x: number; y: number; size: number } {
  const s = coverScale(img, view) * f.zoom;
  return { x: (0 - f.x) / s, y: (0 - f.y) / s, size: view / s }; // 0 - x, not -x: no -0 at the edges
}
