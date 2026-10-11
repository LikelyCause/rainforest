import type { CSSProperties, ReactNode } from "react";
import { describeError } from "../toast";

/** The one way to show a failed request: message plus the API's per-field details. */
export function ErrorAlert({ error, title, children }: { error: unknown; title?: string; children?: ReactNode }) {
  if (!error) return null;
  const e = error as { message?: string; details?: string[] };
  return (
    <div className="alert bad" role="alert">
      <span aria-hidden="true">✕</span>
      <div className="grow">
        {title ? <><b>{title}</b> {e.message ?? describeError(error)}</> : e.message ?? describeError(error)}
        {e.details && e.details.length > 0 && <ul>{e.details.map((d) => <li key={d}>{d}</li>)}</ul>}
        {children}
      </div>
    </div>
  );
}

/** An explanation behind a tappable label: hover shows it on desktop, a tap opens it anywhere. */
export function InfoTip({ label, text, className, style }: { label: ReactNode; text: string; className?: string; style?: CSSProperties }) {
  return (
    <details className={`info-tip${className ? ` ${className}` : ""}`} style={style}>
      <summary title={text}>{label}</summary>
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

export type GlyphKind = "todo" | "kept" | "dropped" | "target" | "failed";

/** Field status mark shared by the field list, bulk edit results and finish screens. */
export function StatusGlyph({ kind, reviewed }: { kind: GlyphKind; reviewed?: boolean }) {
  const text = { todo: "", kept: "✓", dropped: "✕", target: "◎", failed: "!" }[kind];
  const label = { todo: "to do", kept: "kept", dropped: "dropped", target: "target", failed: "save failed" }[kind];
  return (
    <span className={`glyph ${kind}`} role="img" aria-label={label + (kind === "target" && reviewed ? ", reviewed" : "")}>
      {text}
      {kind === "target" && reviewed && <span className="sub" aria-hidden="true">✓</span>}
    </span>
  );
}

export function glyphFor(f: { action: string; reviewed: boolean }, failed = false): GlyphKind {
  if (failed) return "failed";
  if (f.action === "target") return "target";
  if (!f.reviewed) return "todo";
  return f.action === "drop" ? "dropped" : "kept";
}

/** Stacked review meter: kept / dropped / target over the to-do track. */
export function ReviewMeter({ kept, dropped, targets, total, width }: { kept: number; dropped: number; targets: number; total: number; width?: number }) {
  const pct = (n: number) => `${total ? (100 * n) / total : 0}%`;
  return (
    <span className="stack-meter" style={{ width }} aria-hidden="true">
      <i style={{ width: pct(kept), background: "var(--good)" }} />
      <i style={{ width: pct(dropped), background: "var(--text-muted)" }} />
      <i style={{ width: pct(targets), background: "var(--target)" }} />
    </span>
  );
}
