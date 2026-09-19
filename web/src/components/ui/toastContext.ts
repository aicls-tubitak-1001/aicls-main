/**
 * Toast context and hook.
 *
 * Split from the provider component so the module exports either components or plain values, but
 * never both — which keeps React Fast Refresh reliable during a live demo.
 */
import { createContext, useContext } from "react";

/** Semantic tone of a toast; `critical` is reserved for something the lecturer must notice. */
export type ToastTone = "info" | "ok" | "warn" | "critical";

/** One transient message. */
export interface Toast {
  readonly id: string;
  readonly message: string;
  readonly tone: ToastTone;
  /** Milliseconds before it disappears; `0` keeps it until dismissed. */
  readonly duration: number;
  /** Optional single action (an undo, a retry). */
  readonly action?: { readonly label: string; readonly onAction: () => void };
}

/** What {@link useToast} hands a component. */
export interface ToastApi {
  /** Queue a message; returns its id so the caller can dismiss it early. */
  show: (
    message: string,
    options?: { tone?: ToastTone; duration?: number; action?: Toast["action"] },
  ) => string;
  /** Remove one message. */
  dismiss: (id: string) => void;
  /** Remove every message (route change, presentation mode). */
  clear: () => void;
  /** Everything currently on screen. */
  toasts: readonly Toast[];
}

const NOOP_API: ToastApi = {
  show: () => "",
  dismiss: () => undefined,
  clear: () => undefined,
  toasts: [],
};

/** Context filled in by `ToastProvider`; inert when no provider is mounted. */
export const ToastContext = createContext<ToastApi>(NOOP_API);

/**
 * Access the toast queue.
 *
 * Toasts never steal focus and never block the screen: they are announced through an `aria-live`
 * polite region, which is exactly what a lecture needs — the message is read out, the board keeps
 * showing the result.
 */
export function useToast(): ToastApi {
  return useContext(ToastContext);
}
