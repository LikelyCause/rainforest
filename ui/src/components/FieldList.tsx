import { useEffect, useRef } from "react";
import type { FieldListItem } from "../types";
import { TYPE_SHORT } from "../format";

export type StatusFilter = "all" | "unreviewed" | "keep" | "drop";

interface Props {
  fields: FieldListItem[];
  visible: FieldListItem[];
  activeId: number | null;
  search: string;
  onSearch: (s: string) => void;
  types: Set<string>;
  onToggleType: (t: string) => void;
  status: StatusFilter;
  onStatus: (s: StatusFilter) => void;
  onSelect: (id: number) => void;
  searchRef: React.RefObject<HTMLInputElement>;
}

export function FieldList({
  fields, visible, activeId, search, onSearch, types, onToggleType, status, onStatus, onSelect, searchRef,
}: Props) {
  const activeEl = useRef<HTMLDivElement>(null);
  const hasActive = visible.some((f) => f.id === activeId);
  useEffect(() => {
    activeEl.current?.scrollIntoView({ block: "nearest" });
  }, [activeId, hasActive]);

  const typeCounts = new Map<string, number>();
  fields.forEach((f) => typeCounts.set(f.semantic_type, (typeCounts.get(f.semantic_type) ?? 0) + 1));
  const reviewed = fields.filter((f) => f.reviewed).length;

  return (
    <aside className="sidebar" aria-label="Fields">
      <div className="sidebar-head">
        <input
          ref={searchRef}
          type="search"
          placeholder="Search fields   /"
          value={search}
          onChange={(e) => onSearch(e.target.value)}
          onKeyDown={(e) => e.key === "Escape" && (e.target as HTMLInputElement).blur()}
          aria-label="Search fields"
        />
        <div className="chips" role="group" aria-label="Filter by status">
          {(["all", "unreviewed", "keep", "drop"] as StatusFilter[]).map((s) => (
            <button key={s} className={`chip${status === s ? " on" : ""}`} onClick={() => onStatus(s)}>{s}</button>
          ))}
        </div>
        <div className="chips" role="group" aria-label="Filter by type">
          {[...typeCounts.entries()].sort((a, b) => b[1] - a[1]).map(([t, n]) => (
            <button key={t} className={`chip${types.has(t) ? " on" : ""}`} onClick={() => onToggleType(t)} title={t}>
              {TYPE_SHORT[t] ?? t} <span style={{ opacity: 0.7 }}>{n}</span>
            </button>
          ))}
        </div>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <span className="muted" style={{ fontSize: 12 }}>{visible.length} of {fields.length} shown</span>
          <span className="muted num" style={{ fontSize: 12 }}>{reviewed} reviewed</span>
        </div>
      </div>
      <div className="field-list" role="listbox" aria-label="Field list">
        {visible.map((f) => {
          const active = f.id === activeId;
          return (
            <div
              key={f.id}
              ref={active ? activeEl : undefined}
              role="option"
              aria-selected={active}
              className={`field-item${active ? " active" : ""}${f.action === "drop" ? " dropped" : ""}`}
              onClick={() => onSelect(f.id)}
              title={`${f.name} — ${f.semantic_type}${f.reviewed ? " · reviewed" : ""}`}
            >
              <span className={`dot${f.action === "target" ? " target" : f.reviewed ? " reviewed" : ""}`} aria-hidden="true" />
              <span className="name">{f.name}</span>
              <span className="row" style={{ gap: 4, flexWrap: "nowrap" }}>
                {f.leakage && f.action !== "target" && <span className="badge bad" title="Possible target leakage">leak</span>}
                {f.action === "target" ? <span className="badge accent">target</span> : <span className="type-tag">{TYPE_SHORT[f.semantic_type]}</span>}
              </span>
            </div>
          );
        })}
        {visible.length === 0 && <div className="empty">No fields match.</div>}
      </div>
    </aside>
  );
}
