import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import type { FieldListItem } from "../types";
import { TYPE_SHORT } from "../format";
import { attentionOf } from "../signals";
import type { View } from "../view";
import { useFailedSaves } from "../saveQueue";
import { Empty, StatusGlyph, glyphFor } from "./ui";

export type StatusFilter = "todo" | "kept" | "dropped" | "all";

export interface WorkingSet { key: string; label: string; ids: number[] }

interface Props {
  /** The decision source the list shows (view.ts), for its failed-save marks. */
  view?: View;
  fields: FieldListItem[];
  visible: FieldListItem[];
  settled: Set<number>;
  activeId: number | null;
  search: string;
  onSearch: (s: string) => void;
  types: Set<string>;
  onTypes: (t: string[]) => void;
  status: StatusFilter;
  onStatus: (s: StatusFilter) => void;
  attention: boolean;
  onAttention: (on: boolean) => void;
  workingSet: WorkingSet | null;
  onClearSet: () => void;
  onClearFilters: () => void;
  onSelect: (id: number) => void;
  searchRef: React.RefObject<HTMLInputElement>;
}

const ATT_MARK = {
  leak: { text: "⚠", cls: "bad", title: "Possible target leakage" },
  missing: { text: "∅", cls: "warn", title: "Mostly missing or constant" },
  confidence: { text: "?", cls: "warn", title: "Low type confidence" },
  edited: { text: "✎", cls: "edited", title: "Edited, not yet confirmed" },
} as const;

export const statusCounts = (fields: FieldListItem[]) => ({
  todo: fields.filter((f) => !f.reviewed).length,
  kept: fields.filter((f) => f.reviewed && f.action === "keep").length,
  dropped: fields.filter((f) => f.reviewed && f.action === "drop").length,
  all: fields.length,
});

export function FieldList(p: Props) {
  const { fields, visible, activeId, onSelect } = p;
  const listRef = useRef<HTMLDivElement>(null);
  const activeEl = useRef<HTMLDivElement>(null);
  const [typesOpen, setTypesOpen] = useState(false);
  const failed = useFailedSaves(p.view ?? null);
  useEffect(() => {
    activeEl.current?.scrollIntoView({ block: "nearest" });
  }, [activeId, visible.length]);

  const typeCounts = new Map<string, number>();
  fields.forEach((f) => typeCounts.set(f.semantic_type, (typeCounts.get(f.semantic_type) ?? 0) + 1));
  const counts = statusCounts(fields);
  const attentionCount = fields.filter((f) => !f.reviewed && (attentionOf(f) && attentionOf(f) !== "edited" || failed.has(f.id))).length;

  const onSearchKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Escape") {
      e.preventDefault();
      if (p.search) p.onSearch("");
      else e.currentTarget.blur();
    } else if (e.key === "Enter" && visible[0]) {
      e.preventDefault();
      onSelect(visible[0].id);
      listRef.current?.focus();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      listRef.current?.focus();
    }
  };

  const onListKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = visible.findIndex((f) => f.id === activeId);
    const to = (j: number) => {
      e.preventDefault();
      e.stopPropagation();
      const f = visible[Math.max(0, Math.min(visible.length - 1, j))];
      if (f) onSelect(f.id);
    };
    if (e.key === "ArrowDown") to(i + 1);
    else if (e.key === "ArrowUp") to(i - 1);
    else if (e.key === "Home") to(0);
    else if (e.key === "End") to(visible.length - 1);
  };

  return (
    <aside className="sidebar" aria-label="Fields">
      <div className="sidebar-head">
        <input ref={p.searchRef} type="search" placeholder="Search fields   /" value={p.search} aria-label="Search fields"
          onChange={(e) => p.onSearch(e.target.value)} onKeyDown={onSearchKey} />
        {p.workingSet ? (
          <div className="set-pill" title={`Working set from Bulk edit: ${p.workingSet.label}`}>
            <span aria-hidden="true">◆</span>
            <span className="grow">
              <b>{p.workingSet.label}</b> · {p.workingSet.ids.length} fields · {fields.filter((f) => !f.reviewed).length} left
            </span>
            <button type="button" className="ghost small" aria-label="Leave working set" onClick={p.onClearSet}>✕</button>
          </div>
        ) : (
          <>
            <div className="status-seg" role="radiogroup" aria-label="Status">
              {([["todo", "To do"], ["kept", "Kept"], ["dropped", "Dropped"], ["all", "All"]] as const).map(([k, label]) => (
                <button key={k} type="button" role="radio" aria-checked={p.status === k} className={p.status === k ? "on" : ""}
                  onClick={() => p.onStatus(k)}>
                  <b>{counts[k]}</b>{label}
                </button>
              ))}
            </div>
            <div className="sidebar-tools">
              <button type="button" className={`chip attention${p.attention ? " on" : ""}`} aria-pressed={p.attention}
                title="Leakage, mostly missing or constant, low type confidence, failed saves"
                onClick={() => p.onAttention(!p.attention)}>
                ⚑ Needs attention <b className="num">{attentionCount}</b>
              </button>
              <button type="button" className={`chip${p.types.size ? " on" : ""}`} aria-expanded={typesOpen}
                onClick={() => setTypesOpen(!typesOpen)}>
                {p.types.size ? `Type: ${p.types.size}` : "Type"} ▾
              </button>
              {typesOpen && (
                <div className="popover" role="group" aria-label="Filter by type"
                  onKeyDown={(e) => e.key === "Escape" && setTypesOpen(false)}>
                  <button type="button" className="link small-text" style={{ alignSelf: "flex-start", padding: "4px 6px" }}
                    onClick={() => { p.onTypes([]); setTypesOpen(false); }}>All types</button>
                  {[...typeCounts.entries()].sort((a, b) => b[1] - a[1]).map(([t, n]) => (
                    <label key={t} className="check">
                      <span className="row" style={{ gap: 8 }}>
                        <input type="checkbox" checked={p.types.has(t)}
                          onChange={() => p.onTypes(p.types.has(t) ? [...p.types].filter((x) => x !== t) : [...p.types, t])} />
                        {t.replace("_", " ")}
                      </span>
                      <span className="muted num">{n}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>
          </>
        )}
      </div>
      <div ref={listRef} className="field-list" role="listbox" aria-label="Field list" tabIndex={0}
        aria-activedescendant={activeId !== null && visible.some((f) => f.id === activeId) ? `fl-${activeId}` : undefined}
        onKeyDown={onListKey}>
        {visible.map((f) => {
          const active = f.id === activeId;
          const att = attentionOf(f);
          const mark = failed.has(f.id) ? null : att ? ATT_MARK[att] : null;
          return (
            <div key={f.id} id={`fl-${f.id}`} ref={active ? activeEl : undefined} role="option" aria-selected={active}
              className={`field-item${active ? " active" : ""}${f.action === "drop" ? " dropped" : ""}${p.settled.has(f.id) && !active ? " settled" : ""}`}
              onClick={() => onSelect(f.id)} title={`${f.name} — ${f.semantic_type}${f.reviewed ? " · reviewed" : ""}${f.locked ? " · carried over, locked" : ""}`}>
              <StatusGlyph kind={glyphFor(f, failed.has(f.id))} reviewed={f.reviewed} />
              <span className="name">{f.name}</span>
              <span className={`att${mark ? ` ${mark.cls}` : ""}`} title={f.locked ? "Carried over from the version before; locked" : mark?.title}>{f.locked ? "🔒" : mark?.text}</span>
              <span className="type-tag">{TYPE_SHORT[f.semantic_type]}</span>
            </div>
          );
        })}
        {visible.length === 0 && (
          <Empty title="No fields match" action={<button type="button" className="small" onClick={p.onClearFilters}>Clear filters</button>} />
        )}
      </div>
    </aside>
  );
}
