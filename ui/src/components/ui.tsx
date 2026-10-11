import type { CSSProperties, ReactNode } from "react";
import { describeError } from "../toast";
import { Banner, Icon, type IconName } from "./kit";

/** The one way to show a failed request: message plus the API's per-field details. */
export function ErrorAlert({ error, title, children }: { error: unknown; title?: string; children?: ReactNode }) {
  if (!error) return null;
  const e = error as { message?: string; details?: string[] };
  return (
    <Banner tone="bad" title={title}>
      {e.message ?? describeError(error)}
      {e.details && e.details.length > 0 && <ul>{e.details.map((d) => <li key={d}>{d}</li>)}</ul>}
      {children}
    </Banner>
  );
}

/** An explanation behind a tappable label: hover shows it on desktop, a tap opens it anywhere. */
export function InfoTip({ label, text, className, style, ariaLabel }: {
  label: ReactNode; text: string; className?: string; style?: CSSProperties; ariaLabel?: string;
}) {
  return (
    <details className={`info-tip${className ? ` ${className}` : ""}`} style={style}>
      <summary title={text} aria-label={ariaLabel}>{label}</summary>
      <div className="info-tip-body" role="note">{text}</div>
    </details>
  );
}

export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty-state">
      <b>{title}</b>
      {children && <span>{children}</span>}
      {action}
    </div>
  );
}

export type GlyphKind = "todo" | "kept" | "dropped" | "target" | "failed" | "locked";

const GLYPH: Record<GlyphKind, { icon: IconName; label: string }> = {
  todo: { icon: "circle", label: "to do" }, kept: { icon: "check", label: "kept" }, dropped: { icon: "x", label: "dropped" },
  target: { icon: "target", label: "target" }, failed: { icon: "alert", label: "save failed" }, locked: { icon: "lock", label: "carried over, locked" },
};

/** Field status mark shared by the field list, bulk edit results and finish screens. */
export function StatusGlyph({ kind, reviewed }: { kind: GlyphKind; reviewed?: boolean }) {
  const g = GLYPH[kind];
  return (
    <span className={`glyph ${kind}`} role="img" aria-label={g.label + (kind === "target" && reviewed ? ", reviewed" : "")}>
      <Icon name={g.icon} size={16} />
      {kind === "target" && reviewed && <span className="sub" aria-hidden="true"><Icon name="check" size={10} /></span>}
    </span>
  );
}

export function glyphFor(f: { action: string; reviewed: boolean }, failed = false): GlyphKind {
  if (failed) return "failed";
  if (f.action === "target") return "target";
  if (!f.reviewed) return "todo";
  return f.action === "drop" ? "dropped" : "kept";
}

/** Stacked review meter: kept / dropped / target over the to-do track. With a label it is a progressbar of the
 * reviewed share; without one it is decoration beside counts that say the same. */
export function ReviewMeter({ kept, dropped, targets, total, width, label, keptColor = "var(--good)" }: {
  kept: number; dropped: number; targets: number; total: number; width?: number; label?: string; keptColor?: string;
}) {
  const pct = (n: number) => `${total ? (100 * n) / total : 0}%`;
  const done = kept + dropped + targets;
  return (
    <span className="stack-meter" style={{ width }}
      {...(label ? { role: "progressbar", "aria-label": label, "aria-valuenow": done, "aria-valuemin": 0, "aria-valuemax": total } : { "aria-hidden": true })}>
      <i style={{ width: pct(kept), background: keptColor }} />
      <i style={{ width: pct(dropped), background: "var(--border-strong)" }} />
      <i style={{ width: pct(targets), background: "var(--target)" }} />
    </span>
  );
}
