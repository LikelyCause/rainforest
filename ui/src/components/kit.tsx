import { useEffect, useRef, useSyncExternalStore, type ReactNode, type RefObject } from "react";
import { Link, NavLink } from "react-router-dom";
import type { Phase } from "../types";

/* One component per job: tabs move between places, segmented chooses a value, one banner, one dialog, one chip. */

const PATHS: Record<string, ReactNode> = {
  "chevron-down": <path d="M6 9l6 6 6-6" />,
  "chevron-right": <path d="M9 6l6 6-6 6" />,
  "chevron-left": <path d="M15 6l-6 6 6 6" />,
  "arrow-right": (
    <>
      <path d="M5 12h14" />
      <path d="M13 6l6 6-6 6" />
    </>
  ),
  "arrow-left": (
    <>
      <path d="M19 12H5" />
      <path d="M11 6l-6 6 6 6" />
    </>
  ),
  check: <path d="M5 12.5l4.5 4.5L19 7.5" />,
  x: <path d="M6 6l12 12M18 6L6 18" />,
  circle: <circle cx="12" cy="12" r="7" />,
  target: (
    <>
      <circle cx="12" cy="12" r="8" />
      <circle cx="12" cy="12" r="3" />
    </>
  ),
  lock: (
    <>
      <rect x="5" y="11" width="14" height="10" rx="2" />
      <path d="M8 11V8a4 4 0 0 1 8 0v3" />
    </>
  ),
  user: (
    <>
      <circle cx="12" cy="8" r="4" />
      <path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6" />
    </>
  ),
  users: (
    <>
      <circle cx="9" cy="8" r="3.5" />
      <path d="M2.5 20c1-3.5 3.5-5.5 6.5-5.5s5.5 2 6.5 5.5" />
      <path d="M16 4.5a3.5 3.5 0 0 1 0 7" />
      <path d="M18 14.8c2 .8 3.2 2.6 3.8 5.2" />
    </>
  ),
  flask: (
    <>
      <path d="M9 3h6" />
      <path d="M10 3v6L4.5 19a1.5 1.5 0 0 0 1.3 2h12.4a1.5 1.5 0 0 0 1.3-2L14 9V3" />
    </>
  ),
  play: <path d="M7 5l12 7-12 7z" />,
  search: (
    <>
      <circle cx="11" cy="11" r="7" />
      <path d="M20 20l-3.5-3.5" />
    </>
  ),
  upload: (
    <>
      <path d="M12 16V4" />
      <path d="M7 9l5-5 5 5" />
      <path d="M4 20h16" />
    </>
  ),
  download: (
    <>
      <path d="M12 4v12" />
      <path d="M7 11l5 5 5-5" />
      <path d="M4 20h16" />
    </>
  ),
  more: (
    <>
      <circle cx="5" cy="12" r="1" />
      <circle cx="12" cy="12" r="1" />
      <circle cx="19" cy="12" r="1" />
    </>
  ),
  external: (
    <>
      <path d="M14 4h6v6" />
      <path d="M20 4l-9 9" />
      <path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" />
    </>
  ),
  star: <path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z" />,
  plus: <path d="M12 5v14M5 12h14" />,
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5" />
      <path d="M12 8h.01" />
    </>
  ),
  reset: (
    <>
      <path d="M4 12a8 8 0 1 0 2.3-5.6" />
      <path d="M4 4v4h4" />
    </>
  ),
  alert: (
    <>
      <path d="M12 3l9.5 17h-19z" />
      <path d="M12 10v4" />
      <path d="M12 17h.01" />
    </>
  ),
  comment: <path d="M4 5h16v11H9l-5 4z" />,
  table: (
    <>
      <rect x="3.5" y="5" width="17" height="14" rx="2" />
      <path d="M3.5 10h17M9 10v9" />
    </>
  ),
  list: <path d="M9 6h11M9 12h11M9 18h11M4.5 6h.01M4.5 12h.01M4.5 18h.01" />,
  chart: <path d="M4 20V11M10 20V5M16 20v-6M3 20h18" />,
  layers: (
    <>
      <path d="M12 4l8.5 4.5L12 13 3.5 8.5z" />
      <path d="M3.5 13L12 17.5l8.5-4.5" />
    </>
  ),
  rocket: (
    <>
      <path d="M9.5 15.5l-2.8-2.8C8.5 7.5 12.5 4 19.5 4.5c.5 7-3 11-8.2 12.8z" />
      <path d="M6.5 13.5c-1.5.5-2.5 2.5-2.5 6 3.5 0 5.5-1 6-2.5" />
      <circle cx="14.5" cy="9.5" r="1.5" />
    </>
  ),
  database: (
    <>
      <ellipse cx="12" cy="6" rx="7.5" ry="2.5" />
      <path d="M4.5 6v12c0 1.4 3.4 2.5 7.5 2.5s7.5-1.1 7.5-2.5V6M4.5 12c0 1.4 3.4 2.5 7.5 2.5s7.5-1.1 7.5-2.5" />
    </>
  ),
  settings: (
    <>
      <path d="M4 7h9M17 7h3M4 12h3M11 12h9M4 17h11M19 17h1" />
      <circle cx="15" cy="7" r="2" />
      <circle cx="9" cy="12" r="2" />
      <circle cx="17" cy="17" r="2" />
    </>
  ),
};
export type IconName = keyof typeof PATHS;

/** A stroke icon in the text colour. Decorative unless given a label. */
export function Icon({
  name,
  size = 16,
  label,
  className,
  style,
}: {
  name: IconName;
  size?: number;
  label?: string;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <svg
      className={`ic${className ? ` ${className}` : ""}`}
      viewBox="0 0 24 24"
      width={size}
      height={size}
      style={style}
      {...(label ? { role: "img", "aria-label": label } : { "aria-hidden": true })}
    >
      {PATHS[name]}
    </svg>
  );
}

export interface TabItem {
  to: string;
  label: ReactNode;
  /** Shown muted after the label, such as "30/37". */
  count?: ReactNode;
  /** Overrides the router's own active test (a tab that covers several routes). */
  active?: boolean;
  end?: boolean;
  title?: string;
}

/** Underline tabs that move between places. A tab that passes `active` is a plain link, so its
 * aria-current always matches what it shows; the others follow the router. */
export function Tabs({ items, label, className }: { items: TabItem[]; label: string; className?: string }) {
  return (
    <nav className={`tabs${className ? ` ${className}` : ""}`} aria-label={label}>
      {items.map((t) => {
        const body = (
          <>
            {t.label}
            {t.count != null && <span className="tab-count num">{t.count}</span>}
          </>
        );
        return t.active === undefined ? (
          <NavLink
            key={t.to + String(t.label)}
            to={t.to}
            end={t.end}
            title={t.title}
            className={({ isActive }) => `tab${isActive ? " on" : ""}`}
          >
            {body}
          </NavLink>
        ) : (
          <Link
            key={t.to + String(t.label)}
            to={t.to}
            title={t.title}
            className={`tab${t.active ? " on" : ""}`}
            aria-current={t.active ? "page" : undefined}
          >
            {body}
          </Link>
        );
      })}
    </nav>
  );
}

export interface SegOption<T extends string> {
  value: T;
  label: ReactNode;
  title?: string;
  disabled?: boolean;
}

/** Chooses one value. A radio group by default; `pressed` for a view toggle. */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
  pressed,
  full,
  className,
  size,
}: {
  value: T;
  options: SegOption<T>[];
  onChange: (v: T) => void;
  label: string;
  pressed?: boolean;
  full?: boolean;
  className?: string;
  size?: "small";
}) {
  return (
    <div
      className={`segmented${full ? " full" : ""}${size === "small" ? " soft" : ""}${className ? ` ${className}` : ""}`}
      role={pressed ? "group" : "radiogroup"}
      aria-label={label}
    >
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          title={o.title}
          disabled={o.disabled}
          {...(pressed ? { "aria-pressed": o.value === value } : { role: "radio", "aria-checked": o.value === value })}
          className={o.value === value ? "on" : ""}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export type Tone = "info" | "good" | "warn" | "bad";
const TONE_ICON: Record<Tone, IconName> = { info: "info", good: "check", warn: "alert", bad: "x" };

/** The one banner: a tone, an optional bold lead, the text, and an optional action on the right. */
export function Banner({
  tone = "info",
  title,
  children,
  action,
  icon = true,
  className,
  role,
}: {
  tone?: Tone;
  title?: ReactNode;
  children?: ReactNode;
  action?: ReactNode;
  icon?: boolean;
  className?: string;
  role?: string;
}) {
  return (
    <div
      className={`alert ${tone}${className ? ` ${className}` : ""}`}
      role={role ?? (tone === "bad" ? "alert" : undefined)}
    >
      {icon && <Icon name={TONE_ICON[tone]} className="alert-icon" />}
      <div className="grow">
        {title && <b>{title}</b>}
        {title && children ? " " : null}
        {children}
      </div>
      {action}
    </div>
  );
}

export type ChipTone = "plain" | "good" | "warn" | "bad" | "accent" | "private" | "team" | "frozen" | "experiment";

export function Chip({
  tone = "plain",
  icon,
  children,
  title,
  className,
}: {
  tone?: ChipTone;
  icon?: IconName;
  children: ReactNode;
  title?: string;
  className?: string;
}) {
  return (
    <span className={`badge${tone === "plain" ? "" : ` ${tone}`}${className ? ` ${className}` : ""}`} title={title}>
      {icon && <Icon name={icon} size={13} />}
      {children}
    </span>
  );
}

/** A job's status, written out once: queued, running, done, failed. */
export function StatusBadge({
  status,
  children,
}: {
  status: "pending" | "running" | "done" | "failed" | "cancelled" | string;
  children?: ReactNode;
}) {
  const [tone, text]: [ChipTone, string] =
    status === "done"
      ? ["good", "Done"]
      : status === "failed"
        ? ["bad", "Failed"]
        : status === "running"
          ? ["accent", "Running"]
          : status === "pending"
            ? ["accent", "Queued"]
            : ["plain", status.charAt(0).toUpperCase() + status.slice(1)];
  return <Chip tone={tone}>{children ?? text}</Chip>;
}

/** A labelled progress bar. */
export function ProgressBar({
  value,
  max,
  label,
  color,
  width,
  className,
}: {
  value: number;
  max: number;
  label: string;
  color?: string;
  width?: number | string;
  className?: string;
}) {
  const pct = max > 0 ? Math.min(100, (100 * value) / max) : 0;
  return (
    <span
      className={`meter${className ? ` ${className}` : ""}`}
      role="progressbar"
      aria-valuenow={value}
      aria-valuemin={0}
      aria-valuemax={max}
      aria-label={label}
      style={{ width }}
    >
      <span style={{ width: `${pct}%`, background: color }} />
    </span>
  );
}

export type Stage = "setup" | Phase;
const STAGES: { key: Stage; label: string }[] = [
  { key: "setup", label: "Set up" },
  { key: "assessing", label: "Assessing" },
  { key: "review", label: "Team review" },
  { key: "finalized", label: "Finalized" },
];
const STAGE_COLOR: Record<Stage, string> = {
  setup: "var(--accent)",
  assessing: "var(--lens-private)",
  review: "var(--lens-team)",
  finalized: "var(--good)",
};
export const STAGE_LABEL: Record<Stage, string> = {
  setup: "Set up",
  assessing: "Assessing",
  review: "Team review",
  finalized: "Finalized",
};

export const stageOf = (ds: { phase: Phase | null; baseline: { setup: boolean } | null }): Stage | null =>
  !ds.baseline ? null : ds.baseline.setup ? "setup" : ds.phase;

/** The sequence a baseline version goes through, with the current stage marked. */
export function PhaseRail({ stage, compact }: { stage: Stage; compact?: boolean }) {
  const at = STAGES.findIndex((s) => s.key === stage);
  return (
    <ol className={`phase-rail${compact ? " compact" : ""}`} aria-label="Version phase">
      {STAGES.map((s, i) => {
        const done = i < at || (stage === "finalized" && i === at);
        const current = i === at;
        return (
          <li
            key={s.key}
            aria-current={current ? "step" : undefined}
            title={s.label}
            className={[current && "current", done && "done", !current && !done && "todo"].filter(Boolean).join(" ")}
          >
            {i > 0 && <span className="conn" aria-hidden="true" />}
            <span
              className="dot"
              aria-hidden="true"
              style={current && !done ? { borderColor: STAGE_COLOR[s.key] } : undefined}
            >
              {done ? (
                <Icon name="check" size={12} />
              ) : current ? (
                <i style={{ background: STAGE_COLOR[s.key] }} />
              ) : null}
            </span>
            <span className="phase-label">{s.label}</span>
          </li>
        );
      })}
    </ol>
  );
}

/** The four stage segments a dataset row shows. */
export function StageBar({ stage }: { stage: Stage | null }) {
  const at = stage ? STAGES.findIndex((s) => s.key === stage) : -1;
  return (
    <span className="stage-bar" aria-hidden="true">
      {STAGES.map((s, i) => (
        <i
          key={s.key}
          style={{
            background:
              i < at || (stage === "finalized" && i === at) ? "var(--good)" : i === at ? STAGE_COLOR[s.key] : undefined,
          }}
        />
      ))}
    </span>
  );
}

export type Lens = "private" | "team" | "frozen" | "experiment" | "suggestions";
const LENS: Record<Lens, { icon: IconName; tone: ChipTone }> = {
  private: { icon: "user", tone: "private" },
  team: { icon: "users", tone: "team" },
  frozen: { icon: "lock", tone: "frozen" },
  experiment: { icon: "flask", tone: "experiment" },
  suggestions: { icon: "info", tone: "frozen" },
};
export const LENS_COLOR: Record<Lens, string> = {
  private: "var(--lens-private)",
  team: "var(--lens-team)",
  frozen: "var(--lens-frozen)",
  experiment: "var(--lens-experiment)",
  suggestions: "var(--lens-frozen)",
};

/** Whose decisions are on screen. */
export function LensChip({ lens, children, title }: { lens: Lens; children: ReactNode; title?: string }) {
  // "Private · your assessment": on a phone only the first part shows.
  const [head, ...rest] = typeof children === "string" ? children.split(" · ") : [children];
  return (
    <Chip tone={LENS[lens].tone} icon={LENS[lens].icon} title={title} className="lens-chip">
      {head}
      {rest.length > 0 && <span className="lens-more"> · {rest.join(" · ")}</span>}
    </Chip>
  );
}

/** Closes a popover on a click outside `ref` or on Escape. */
export function useDismiss(ref: RefObject<HTMLElement>, open: boolean, close: () => void) {
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && close();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close();
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [ref, open, close]);
}

/** True while a modal dialog is open: page-wide single-key shortcuts stand down until it closes. */
export const isModalOpen = () => document.querySelector('[aria-modal="true"]') !== null;

// Open dialogs, oldest first: Escape and the focus trap belong to the topmost one only.
const dialogStack: object[] = [];
const FOCUSABLE =
  'button:not(:disabled), [href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])';

/** The one dialog: a title, what will happen, and its buttons. Escape or the scrim closes it; focus stays inside. */
export function Dialog({
  title,
  children,
  onClose,
  actions,
  wide,
  labelledBy,
  role = "dialog",
}: {
  title: ReactNode;
  children?: ReactNode;
  onClose: () => void;
  actions?: ReactNode;
  wide?: boolean;
  labelledBy?: string;
  /** "alertdialog" for a confirmation of something that can't be undone. */
  role?: "dialog" | "alertdialog";
}) {
  const ref = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  const id = labelledBy ?? "dialog-title";
  useEffect(() => {
    const token = {};
    dialogStack.push(token);
    const top = () => dialogStack[dialogStack.length - 1] === token;
    const prev = document.activeElement as HTMLElement | null;
    const el = ref.current;
    (
      el?.querySelector<HTMLElement>("[data-autofocus]") ??
      el?.querySelector<HTMLElement>("button, input, select, textarea") ??
      el
    )?.focus();
    // Document-level, so Escape works after a click on the dialog's text or after the focused control goes away.
    const onKey = (e: KeyboardEvent) => {
      if (!top() || !el) return;
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopImmediatePropagation();
        closeRef.current();
        return;
      }
      if (e.key !== "Tab") return;
      const items = [...el.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(
        (x) => x.offsetParent !== null || x === document.activeElement,
      );
      const active = document.activeElement;
      if (!items.length) {
        e.preventDefault();
        el.focus();
        return;
      }
      const first = items[0],
        last = items[items.length - 1];
      if (!el.contains(active) || active === el) {
        e.preventDefault();
        (e.shiftKey ? last : first).focus();
      } else if (e.shiftKey && active === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    };
    // Focus that lands outside (a click on the scrim's edge, a script) comes back to the dialog.
    const onFocus = (e: FocusEvent) => top() && el && !el.contains(e.target as Node) && el.focus();
    document.addEventListener("keydown", onKey);
    document.addEventListener("focusin", onFocus);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("focusin", onFocus);
      dialogStack.splice(dialogStack.indexOf(token), 1);
      if (prev?.isConnected) prev.focus();
    };
  }, []);
  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div
        ref={ref}
        className={`dialog${wide ? " wide" : ""}`}
        role={role}
        aria-modal="true"
        aria-labelledby={id}
        tabIndex={-1}
      >
        <h2 id={id}>{title}</h2>
        {children}
        {actions && <div className="dialog-actions">{actions}</div>}
      </div>
    </div>
  );
}

export interface ConfirmOptions {
  title: string;
  /** What will happen, in plain words. */
  body?: ReactNode;
  confirm: string;
  cancel?: string;
  danger?: boolean;
}

// A module store, like toasts, so any handler can await a confirmation.
let pending: (ConfirmOptions & { resolve: (ok: boolean) => void }) | null = null;
const confirmListeners = new Set<() => void>();
const emitConfirm = () => confirmListeners.forEach((l) => l());

/** Asks in an in-app dialog; resolves true when confirmed. Replaces window.confirm. */
export function confirmDialog(opts: ConfirmOptions): Promise<boolean> {
  pending?.resolve(false);
  return new Promise((resolve) => {
    pending = { ...opts, resolve };
    emitConfirm();
  });
}

function settle(ok: boolean) {
  const p = pending;
  pending = null;
  emitConfirm();
  p?.resolve(ok);
}

export function ConfirmRegion() {
  const p = useSyncExternalStore(
    (l) => {
      confirmListeners.add(l);
      return () => {
        confirmListeners.delete(l);
      };
    },
    () => pending,
  );
  if (!p) return null;
  return (
    // A destructive confirm starts on Cancel, so a stray Enter can't destroy anything.
    <Dialog
      title={p.title}
      onClose={() => settle(false)}
      labelledBy="confirm-title"
      role={p.danger ? "alertdialog" : "dialog"}
      actions={
        <>
          <button type="button" onClick={() => settle(false)} data-autofocus={p.danger ? true : undefined}>
            {p.cancel ?? "Cancel"}
          </button>
          <button
            type="button"
            className={p.danger ? "danger" : "primary"}
            data-autofocus={p.danger ? undefined : true}
            onClick={() => settle(true)}
          >
            {p.confirm}
          </button>
        </>
      }
    >
      {p.body && <div className="dialog-body">{p.body}</div>}
    </Dialog>
  );
}
