/**
 * Toast provider and its live region.
 *
 * The region is `aria-live="polite"` and `role="status"`, so a message is announced after whatever
 * the screen reader is currently saying instead of cutting it off. Nothing here takes focus, and a
 * toast never covers the result area — it sits in the bottom-left corner, above the footer.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { t } from "@/i18n";
import { cx } from "@/lib/cx";
import { ToastContext, type Toast, type ToastApi, type ToastTone } from "./toastContext";
import styles from "./Toast.module.css";

/** Default lifetime of a toast in milliseconds. */
export const DEFAULT_TOAST_MS = 6000;

let counter = 0;

function nextId(): string {
  counter += 1;
  return `toast-${String(counter)}`;
}

export interface ToastProviderProps {
  children: ReactNode;
  /** Render the live region (turn off in tests that assert on the queue only). */
  renderRegion?: boolean;
}

/** Provide the toast queue to the tree below and render its live region. */
export function ToastProvider({ children, renderRegion = true }: ToastProviderProps): JSX.Element {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const timers = useRef(new Map<string, ReturnType<typeof setTimeout>>());

  const dismiss = useCallback((id: string) => {
    const timer = timers.current.get(id);
    if (timer !== undefined) {
      clearTimeout(timer);
      timers.current.delete(id);
    }
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const clear = useCallback(() => {
    for (const timer of timers.current.values()) {
      clearTimeout(timer);
    }
    timers.current.clear();
    setToasts([]);
  }, []);

  const show = useCallback(
    (
      message: string,
      options: { tone?: ToastTone; duration?: number; action?: Toast["action"] } = {},
    ): string => {
      const id = nextId();
      const duration = options.duration ?? DEFAULT_TOAST_MS;
      const toast: Toast = {
        id,
        message,
        tone: options.tone ?? "info",
        duration,
        ...(options.action === undefined ? {} : { action: options.action }),
      };
      setToasts((current) => [...current, toast]);
      if (duration > 0) {
        timers.current.set(
          id,
          setTimeout(() => {
            timers.current.delete(id);
            setToasts((current) => current.filter((item) => item.id !== id));
          }, duration),
        );
      }
      return id;
    },
    [],
  );

  useEffect(() => {
    const pending = timers.current;
    return () => {
      for (const timer of pending.values()) {
        clearTimeout(timer);
      }
      pending.clear();
    };
  }, []);

  const api = useMemo<ToastApi>(
    () => ({ show, dismiss, clear, toasts }),
    [show, dismiss, clear, toasts],
  );

  return (
    <ToastContext.Provider value={api}>
      {children}
      {renderRegion ? <ToastRegion toasts={toasts} onDismiss={dismiss} /> : null}
    </ToastContext.Provider>
  );
}

export interface ToastRegionProps {
  toasts: readonly Toast[];
  onDismiss: (id: string) => void;
}

/** The polite live region every toast is announced through. */
export function ToastRegion({ toasts, onDismiss }: ToastRegionProps): JSX.Element {
  return (
    <div className={styles.region} role="status" aria-live="polite" aria-label={t("toast.region")}>
      {toasts.map((toast) => (
        <div key={toast.id} className={cx(styles.toast, styles[toast.tone])}>
          <p className={styles.message}>{toast.message}</p>
          {toast.action === undefined ? null : (
            <button type="button" className={styles.action} onClick={toast.action.onAction}>
              {toast.action.label}
            </button>
          )}
          <button
            type="button"
            className={styles.close}
            onClick={() => {
              onDismiss(toast.id);
            }}
            aria-label={t("common.close")}
          >
            <span aria-hidden="true">×</span>
          </button>
        </div>
      ))}
    </div>
  );
}
