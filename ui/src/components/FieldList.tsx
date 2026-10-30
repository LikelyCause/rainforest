import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import type { Dataset, FieldListItem } from "../types";
import { TYPE_SHORT } from "../format";
import { attentionOf } from "../signals";
import type { View } from "../view";
import { useFailedSaves } from "../saveQueue";
import { Icon, useDismiss } from "./kit";
import { Empty, ReviewMeter, StatusGlyph, glyphFor } from "./ui";

export type StatusFilter = "todo" | "kept" | "dropped" | "all";

export interface WorkingSet {
  key: string;
  label: string;
  ids: number[];
}

interface Props {
  /** The decision source the list shows (view.ts), for its failed-save marks. */
  view?: View;
  /** The dataset, for the view switch at the top; without it the switch is left out. */
  ds?: Dataset;
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
  /** Closes the list where it is a drawer (narrow screens); shown only there. */
  onClose?: () => void;
}

const ATT_MARK = {
  leak: { text: "possible leak", cls: "bad", title: "Possible target leakage" },
  missing: { text: "needs a look", cls: "warn", title: "Mostly missing or constant" },
  confidence: { text: "needs a look", cls: "warn", title: "Low type confidence" },
  edited: { text: "edited", cls: "edited", title: "Edited, not yet confirmed" },
} as const;

const TYPE_WORD: Record<string, string> = {
  numeric_continuous: "number",
  numeric_discrete: "count",
  categorical: "category",
  binary: "yes/no",
  text: "text",
  datetime: "date",
  identifier: "id",
  constant: "constant",
};

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
  const typesRef = useRef<HTMLDivElement>(null);
  const [typesOpen, setTypesOpen] = useState(false);
  const closeTypes = useCallback(() => setTypesOpen(false), []);
  useDismiss(typesRef, typesOpen, closeTypes);
  const failed = useFailedSaves(p.view ?? null);
  useEffect(() => {
    activeEl.current?.scrollIntoView({ block: "nearest" });
  }, [activeId, visible.length]);

  const typeCounts = new Map<string, number>();
  fields.forEach((f) => typeCounts.set(f.semantic_type, (typeCounts.get(f.semantic_type) ?? 0) + 1));
  const counts = statusCounts(fields);
  const attentionCount = fields.filter(
    (f) => !f.reviewed && ((attentionOf(f) && attentionOf(f) !== "edited") || failed.has(f.id)),
  ).length;
  const targets = fields.filter((f) => f.reviewed && f.action === "target").length;

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

  const statusChip = (k: StatusFilter, label: string) => (
    <button
      key={k}
      type="button"
      role="radio"
      aria-checked={p.status === k}
      className={`filter-chip${p.status === k ? " on" : ""}`}
      onClick={() => p.onStatus(k)}
    >
      {label} <b className="num">{counts[k]}</b>
    </button>
  );

  return (
    <aside className="sidebar field-sidebar" aria-label="Fields">
      <div className="sidebar-head">
        {p.onClose && (
          <div className="drawer-head">
            <b>Fields</b>
            <button type="button" className="ghost small" aria-label="Close the field list" onClick={p.onClose}>
              <Icon name="x" />
            </button>
          </div>
        )}
        <label className="search-box">
          <Icon name="search" />
          <input
            ref={p.searchRef}
            type="search"
            placeholder="Search fields"
            value={p.search}
            aria-label="Search fields"
            aria-keyshortcuts="/"
            onChange={(e) => p.onSearch(e.target.value)}
            onKeyDown={onSearchKey}
          />
          <kbd aria-hidden="true">/</kbd>
        </label>
        {p.workingSet ? (
          <div className="set-pill" title={`Working set from the table: ${p.workingSet.label}`}>
            <Icon name="table" />
            <span className="grow">
              <b>{p.workingSet.label}</b> · {p.workingSet.ids.length} fields ·{" "}
              {fields.filter((f) => !f.reviewed).length} left
            </span>
            <button type="button" className="ghost small" aria-label="Leave working set" onClick={p.onClearSet}>
              <Icon name="x" />
            </button>
          </div>
        ) : (
          <div className="filter-chips" ref={typesRef}>
            <div role="radiogroup" aria-label="Status" className="filter-group">
              {statusChip("todo", "To do")}
              {statusChip("all", "All")}
              {(p.status === "kept" || p.status === "dropped") &&
                statusChip(p.status, p.status === "kept" ? "Kept" : "Dropped")}
            </div>
            <button
              type="button"
              className={`filter-chip attention${p.attention ? " on" : ""}`}
              aria-pressed={p.attention}
              title="Leakage, mostly missing or constant, low type confidence, failed saves"
              onClick={() => p.onAttention(!p.attention)}
            >
              Needs a look <b className="num">{attentionCount}</b>
            </button>
            <button
              type="button"
              className={`filter-chip${p.types.size ? " on" : ""}`}
              aria-expanded={typesOpen}
              aria-haspopup="true"
              title="Kept or dropped fields, and fields by type"
              onClick={() => setTypesOpen(!typesOpen)}
            >
              {p.types.size ? `Type: ${p.types.size}` : "Type"} <Icon name="chevron-down" size={12} />
            </button>
            {typesOpen && (
              <div className="popover filter-popover">
                <div role="radiogroup" aria-label="Status" className="popover-chips">
                  {(["todo", "all", "kept", "dropped"] as const).map((k) =>
                    statusChip(k, { todo: "To do", all: "All", kept: "Kept", dropped: "Dropped" }[k]),
                  )}
                </div>
                <hr />
                <div role="group" aria-label="Filter by type" className="stack type-filter">
                  <button
                    type="button"
                    className="link small-text type-filter-all"
                    onClick={() => {
                      p.onTypes([]);
                      setTypesOpen(false);
                    }}
                  >
                    All types
                  </button>
                  {[...typeCounts.entries()]
                    .sort((a, b) => b[1] - a[1])
                    .map(([t, n]) => (
                      <label key={t} className="check">
                        <span className="row type-filter-name">
                          <input
                            type="checkbox"
                            checked={p.types.has(t)}
                            onChange={() =>
                              p.onTypes(p.types.has(t) ? [...p.types].filter((x) => x !== t) : [...p.types, t])
                            }
                          />
                          {t.replace("_", " ")}
                        </span>
                        <span className="muted num">{n}</span>
                      </label>
                    ))}
                </div>
              </div>
            )}
          </div>
        )}
        <div className="review-progress">
          <ReviewMeter
            kept={counts.kept}
            dropped={counts.dropped}
            targets={targets}
            total={counts.all}
            keptColor="var(--lens, var(--good))"
            label={`${counts.all - counts.todo} of ${counts.all} fields reviewed`}
          />
          <span className="num small-text muted">
            {counts.kept} kept · {counts.dropped} dropped · {targets} target{targets === 1 ? "" : "s"} · {counts.todo}{" "}
            to do
          </span>
        </div>
      </div>
      <div
        ref={listRef}
        className="field-list"
        role="listbox"
        aria-label="Field list"
        tabIndex={0}
        aria-activedescendant={
          activeId !== null && visible.some((f) => f.id === activeId) ? `fl-${activeId}` : undefined
        }
        onKeyDown={onListKey}
      >
        {visible.map((f) => {
          const active = f.id === activeId;
          const att = attentionOf(f);
          const isFailed = failed.has(f.id);
          const mark = isFailed || f.locked ? null : att ? ATT_MARK[att] : null;
          const right = isFailed
            ? { text: "not saved", cls: "bad", title: "This field's last change didn't save" }
            : f.action === "target"
              ? { text: "target", cls: "target", title: undefined }
              : (mark ?? (f.reviewed && f.action === "drop" ? { text: "dropped", cls: "", title: undefined } : null));
          return (
            <div
              key={f.id}
              id={`fl-${f.id}`}
              ref={active ? activeEl : undefined}
              role="option"
              aria-selected={active}
              className={`field-item${active ? " active" : ""}${f.action === "drop" ? " dropped" : ""}${p.settled.has(f.id) && !active ? " settled" : ""}`}
              onClick={() => onSelect(f.id)}
              title={`${f.name} — ${f.semantic_type}${f.reviewed ? " · reviewed" : ""}${f.locked ? " · carried over, locked" : ""}`}
            >
              <StatusGlyph kind={f.locked && !isFailed ? "locked" : glyphFor(f, isFailed)} reviewed={f.reviewed} />
              <span className="name">{f.name}</span>
              <span
                className={`type-tag${right ? ` ${right.cls}` : ""}`}
                title={right?.title ?? f.semantic_type.replace("_", " ")}
              >
                {right?.text ?? TYPE_WORD[f.semantic_type] ?? TYPE_SHORT[f.semantic_type] ?? f.semantic_type}
              </span>
            </div>
          );
        })}
        {visible.length === 0 && (
          <Empty
            title="No fields match"
            action={
              <button type="button" className="small" onClick={p.onClearFilters}>
                Clear filters
              </button>
            }
          />
        )}
      </div>
    </aside>
  );
}
