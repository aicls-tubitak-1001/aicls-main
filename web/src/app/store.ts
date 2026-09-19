/**
 * Tiny observable store for cross-feature UI state.
 *
 * The architecture sketch named `zustand`; a 40-line store built on `useSyncExternalStore` covers
 * what this SPA needs (one session, one workspace, a handful of flags) without another runtime
 * dependency to audit. The API is intentionally the familiar `getState / setState / subscribe`
 * triple, so swapping in a library later is mechanical.
 */
import { useSyncExternalStore } from "react";

import type { ComputeOut, InputKind, OperationOut } from "@/api/types";
import type { SocketStatus } from "@/api/ws";

/** A minimal observable state container. */
export interface Store<T> {
  getState(): T;
  setState(patch: Partial<T> | ((state: T) => Partial<T>)): void;
  subscribe(listener: (state: T) => void): () => void;
}

/**
 * Create a store around an initial state object.
 *
 * `setState` merges a patch (or the result of a producer function) and notifies subscribers only
 * when at least one key actually changed identity.
 */
export function createStore<T extends object>(initial: T): Store<T> {
  let state = initial;
  const listeners = new Set<(state: T) => void>();
  return {
    getState: () => state,
    setState(patch) {
      const partial = typeof patch === "function" ? patch(state) : patch;
      let changed = false;
      for (const key of Object.keys(partial) as (keyof T)[]) {
        if (!Object.is(state[key], partial[key])) {
          changed = true;
          break;
        }
      }
      if (!changed) {
        return;
      }
      state = { ...state, ...partial };
      for (const listener of [...listeners]) {
        listener(state);
      }
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}

/** Subscribe a component to a slice of a store. */
export function useStore<T extends object, S>(store: Store<T>, selector: (state: T) => S): S {
  return useSyncExternalStore(
    (listener) => store.subscribe(listener),
    () => selector(store.getState()),
    () => selector(store.getState()),
  );
}

/** What the command bar is currently waiting for. */
export type CommandPhase = "idle" | "busy" | "awaiting-confirmation";

/** Application-wide UI state. */
export interface AppState {
  /** Active workspace (null before the first workspace is created). */
  workspaceId: string | null;
  /** Raw text typed or transcribed into the command bar. */
  commandText: string;
  /** LaTeX currently on the canvas / in the preview (`context.last_latex`). */
  inputLatex: string;
  /** How the pending command was entered — drives the confirmation policy (api.md §3.6). */
  inputKind: InputKind;
  /** Last successful outcome rendered by `ResultCard`. */
  outcome: ComputeOut | null;
  /** Recent operations shown by `HistoryPanel`. */
  history: OperationOut[];
  /** Command bar phase. */
  phase: CommandPhase;
  /** Ink/audio WebSocket status shown by the connection badge. */
  socketStatus: SocketStatus;
  /** True while the server has asked the client to slow down (api.md §4.1). */
  throttled: boolean;
  /** Turkish error text shown in the banner, if any. */
  errorText: string | null;
  /** Whether the book panel is open. */
  bookOpen: boolean;
}

const INITIAL_STATE: AppState = {
  workspaceId: null,
  commandText: "",
  inputLatex: "",
  inputKind: "text",
  outcome: null,
  history: [],
  phase: "idle",
  socketStatus: "closed",
  throttled: false,
  errorText: null,
  bookOpen: false,
};

/** The singleton application store. */
export const appStore = createStore<AppState>({ ...INITIAL_STATE });

/** Read a slice of the application store inside a component. */
export function useAppState<S>(selector: (state: AppState) => S): S {
  return useStore(appStore, selector);
}

/** Reset the store to its initial state (logout, and tests). */
export function resetAppState(): void {
  appStore.setState({ ...INITIAL_STATE });
}
