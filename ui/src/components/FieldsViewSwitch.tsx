import { useLocation, useNavigate } from "react-router-dom";
import { can } from "../can";
import { useMe } from "../hooks";
import { useView, withView } from "../view";
import { Segmented } from "./kit";
import type { Dataset } from "../types";

export type FieldsView = "focus" | "table" | "sql" | "derived" | "queue";

/** Fields' own view switch: Focus · Table · SQL · Derived (and the queue first during team review). Each is a place,
 * so pressing one navigates; `?view=` comes along. SQL and Derived only show for the baseline in progress, to those
 * allowed them. It sits in the page bar and reads which view is open from the route. */
export function FieldsViewSwitch({ ds, className }: { ds: Dataset; className?: string }) {
  const nav = useNavigate();
  const view = useView();
  const me = useMe().data;
  const { pathname, search } = useLocation();
  // The console reads every table, work in progress included, so it isn't a Viewer's.
  const sql = can(me, "querySql") && !view;
  const derive = can(me, "editDataset", ds.status === "refreshing") && !view;
  const tab = new URLSearchParams(search).get("tab");
  const current: FieldsView = pathname.endsWith("/team")
    ? "queue"
    : !pathname.endsWith("/bulk")
      ? "focus"
      : tab === "sql" && sql
        ? "sql"
        : tab === "derived" && derive
          ? "derived"
          : "table";
  const base = `/d/${ds.id}`;
  const options: { value: FieldsView; label: string; title?: string }[] = [
    ...(ds.phase === "review" && !view
      ? [{ value: "queue" as const, label: "Queue", title: "The team review queue" }]
      : []),
    { value: "focus", label: "Focus", title: "One field at a time" },
    { value: "table", label: "Table", title: "Filter fields and change many at once" },
    ...(sql ? [{ value: "sql" as const, label: "SQL", title: "Query the field metadata and raw data" }] : []),
    ...(derive ? [{ value: "derived" as const, label: "Derived", title: "Fields computed from other fields" }] : []),
  ];
  const to: Record<FieldsView, string> = {
    queue: `${base}/team`,
    focus: withView(`${base}/review`, view),
    table: withView(`${base}/bulk`, view),
    sql: `${base}/bulk?tab=sql`,
    derived: `${base}/bulk?tab=derived`,
  };
  const go = (v: FieldsView) => {
    if (v === current) return;
    // Within the table page only `tab` changes, so the finder's presets and filters survive a look at SQL.
    if (pathname.endsWith("/bulk") && (v === "table" || v === "sql" || v === "derived")) {
      const next = new URLSearchParams(search);
      if (v === "table") next.delete("tab");
      else next.set("tab", v);
      const q = next.toString();
      return nav(`${pathname}${q ? `?${q}` : ""}`);
    }
    nav(to[v]);
  };
  return (
    <Segmented pressed label="Fields view" className={className} value={current} options={options} onChange={go} />
  );
}
