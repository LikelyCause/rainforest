import { useSearchParams } from "react-router-dom";

/** Whose decisions Review, Bulk edit and the finder work on: null for the baseline in progress (the viewer's own
 * assessment while they assess it), `experiment:<id>` for a team experiment's copy, `baseline:<id>` for a finalized
 * baseline, read-only. Carried in the URL as ?view=. */
export type View = string | null;

export function parseView(v: string | null | undefined): { kind: "experiment" | "baseline"; id: number } | null {
  const m = /^(experiment|baseline):(\d+)$/.exec(v ?? "");
  return m ? { kind: m[1] as "experiment" | "baseline", id: Number(m[2]) } : null;
}

export function withView(path: string, view: View): string {
  if (!view) return path;
  return `${path}${path.includes("?") ? "&" : "?"}view=${encodeURIComponent(view)}`;
}

/** The view's part of a react-query key; prefix invalidation of ["fields", id] still reaches every view. */
export const viewKey = (view: View) => view ?? "current";

/** `?view=` in a query string. */
export const viewQuery = (view: View, sep: "?" | "&" = "?") => (view ? `${sep}view=${encodeURIComponent(view)}` : "");

export function useView(): View {
  const [params] = useSearchParams();
  const v = params.get("view");
  return parseView(v) ? v : null;
}
