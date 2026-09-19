/**
 * Presentation mode: one attribute, `data-density`, on the document element.
 *
 * A lecturer switching to the wall-mounted board needs bigger type and bigger hit targets, not a
 * different application. `styles/tokens.css` therefore keys the whole type ramp and the control
 * heights off `--density-scale`, and this module only flips the attribute that changes it:
 *
 * - `"normal"` — the desk default (16 px body, 44 px controls).
 * - `"presentation"` — 1.35× type, 56 px controls, secondary chrome hidden (`.chrome-secondary`).
 *
 * The choice is remembered in `localStorage`, which is a per-browser convenience and never
 * personal data; a browser that blocks storage simply starts at `"normal"` every time.
 *
 * The attribute is global, so the state behind it is too: {@link applyDensity} notifies every
 * mounted {@link useDensity}. Without that, the switch in the application shell and the switch on
 * the workspace would hold two private copies of one truth, and pressing one would leave the other
 * showing the wrong label — the sort of small lie that costs a lecturer thirty seconds in front of
 * a class.
 */
import { useCallback, useEffect, useState, useSyncExternalStore } from "react";

/** The two densities the design system defines. */
export type Density = "normal" | "presentation";

/** Attribute written on `<html>`; read by `styles/tokens.css`. */
export const DENSITY_ATTRIBUTE = "data-density";

const STORAGE_KEY = "aicls.density";

function isDensity(value: string | null): value is Density {
  return value === "normal" || value === "presentation";
}

/** Read the stored density; falls back to `"normal"` when storage is unavailable. */
export function readDensity(): Density {
  try {
    const stored = globalThis.localStorage?.getItem(STORAGE_KEY) ?? null;
    return isDensity(stored) ? stored : "normal";
  } catch {
    return "normal";
  }
}

/** The density currently written on the document. */
export function currentDensity(): Density {
  const value = globalThis.document?.documentElement.getAttribute(DENSITY_ATTRIBUTE) ?? null;
  return isDensity(value) ? value : "normal";
}

/** Everything currently rendering the density (one entry per mounted {@link useDensity}). */
const listeners = new Set<() => void>();

/**
 * Watch the density for changes.
 *
 * @param listener - Called after every {@link applyDensity}.
 * @returns The unsubscribe function.
 */
export function subscribeToDensity(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/**
 * Apply a density to the document and remember it.
 *
 * The attribute is always written (including `"normal"`), so a test or a screenshot can assert on
 * the current mode without inferring it from its absence. Every subscriber is then notified, so
 * two switches in two parts of the shell can never disagree about which mode is on.
 */
export function applyDensity(density: Density): void {
  globalThis.document?.documentElement.setAttribute(DENSITY_ATTRIBUTE, density);
  try {
    globalThis.localStorage?.setItem(STORAGE_KEY, density);
  } catch {
    // Private mode or blocked storage: the choice simply does not survive a reload.
  }
  for (const listener of [...listeners]) {
    listener();
  }
}

/** The other density (what a toggle switches to). */
export function otherDensity(density: Density): Density {
  return density === "presentation" ? "normal" : "presentation";
}

/**
 * React binding for the presentation-mode switch.
 *
 * @returns The current density, a setter and a toggle. The toggle is what the keyboard shortcut
 *   and the header button both call.
 */
export function useDensity(): {
  density: Density;
  setDensity: (next: Density) => void;
  toggleDensity: () => void;
  isPresentation: boolean;
} {
  // The document attribute is the single source of truth; every hook reads it and re-renders when
  // any of them changes it. `getServerSnapshot` keeps the hook usable if the app is ever rendered
  // outside a browser.
  const density = useSyncExternalStore(subscribeToDensity, currentDensity, () => "normal" as const);

  // The stored preference has to reach the document once, on the first mount of the session.
  const [restored, setRestored] = useState(false);
  useEffect(() => {
    if (restored) {
      return;
    }
    setRestored(true);
    applyDensity(readDensity());
  }, [restored]);

  const setDensity = useCallback((next: Density) => {
    applyDensity(next);
  }, []);

  const toggleDensity = useCallback(() => {
    applyDensity(otherDensity(currentDensity()));
  }, []);

  return { density, setDensity, toggleDensity, isPresentation: density === "presentation" };
}
