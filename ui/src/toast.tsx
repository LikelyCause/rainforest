import { useEffect, useSyncExternalStore } from "react";

export interface ToastAction {
  label: string;
  kbd?: string;
  run: () => void;
}

export interface Toast {
  id: number;
  message: string;
  detail?: string;
  tone?: "info" | "bad" | "good";
  action?: ToastAction;
  /** ms; errors default to sticky (0). */
  duration?: number;
}

// A module-level store so non-React code (the save queue) can raise toasts. One toast at a time: a new one replaces the last.
let current: Toast | null = null;
let seq = 0;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());

export function toast(t: Omit<Toast, "id">): number {
  current = { ...t, id: ++seq };
  emit();
  return current.id;
}

export function dismissToast(id?: number) {
  if (current && (id === undefined || current.id === id)) {
    current = null;
    emit();
  }
}

export const showError = (message: string, err: unknown, action?: ToastAction) =>
  toast({ message, detail: describeError(err), tone: "bad", action, duration: 0 });

export function describeError(err: unknown): string {
  if (!err) return "";
  const e = err as { message?: string; details?: string[] };
  return [e.message ?? String(err), ...(e.details ?? [])].join(" — ");
}

const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => listeners.delete(l);
};

export function ToastRegion() {
  const t = useSyncExternalStore(subscribe, () => current);
  useEffect(() => {
    if (!t) return;
    const ms = t.duration ?? (t.tone === "bad" ? 0 : 3000);
    if (!ms) return;
    const timer = window.setTimeout(() => dismissToast(t.id), ms);
    return () => window.clearTimeout(timer);
  }, [t]);
  return (
    <div className="toast-region" role="status" aria-live="polite">
      {t && (
        <div className={`toast${t.tone && t.tone !== "info" ? ` ${t.tone}` : ""}`} key={t.id}>
          <div className="msg">
            {t.message}
            {t.detail && <small>{t.detail}</small>}
          </div>
          {t.action && (
            <button
              type="button"
              onClick={() => {
                t.action!.run();
                dismissToast(t.id);
              }}
            >
              {t.action.label}
              {t.action.kbd && <kbd>{t.action.kbd}</kbd>}
            </button>
          )}
          {(t.tone === "bad" || t.duration === 0) && (
            <button type="button" aria-label="Dismiss" onClick={() => dismissToast(t.id)}>
              ✕
            </button>
          )}
        </div>
      )}
    </div>
  );
}
