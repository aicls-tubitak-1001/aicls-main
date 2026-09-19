/**
 * Colour scheme handling.
 *
 * The default is `"system"`: the CSS in `styles/tokens.css` reacts to `prefers-color-scheme`, so no
 * JavaScript runs for most users. An explicit choice writes `data-theme` on `<html>` (which the
 * token file honours) and is remembered in `localStorage` — a per-browser convenience, never
 * personal data.
 */
import { useEffect, useState } from "react";

/** User-visible options for the appearance switch. */
export type ThemePreference = "system" | "light" | "dark";

/** The scheme actually in effect once "system" is resolved. */
export type ResolvedTheme = "light" | "dark";

const STORAGE_KEY = "aicls.theme";
const DARK_QUERY = "(prefers-color-scheme: dark)";

function isPreference(value: string | null): value is ThemePreference {
  return value === "system" || value === "light" || value === "dark";
}

/** Read the stored preference; falls back to `"system"` when storage is unavailable. */
export function readThemePreference(): ThemePreference {
  try {
    const stored = globalThis.localStorage?.getItem(STORAGE_KEY) ?? null;
    return isPreference(stored) ? stored : "system";
  } catch {
    return "system";
  }
}

/** True when the browser currently prefers a dark scheme. */
export function prefersDark(): boolean {
  return typeof globalThis.matchMedia === "function" && globalThis.matchMedia(DARK_QUERY).matches;
}

/** Resolve a preference against the OS setting. */
export function resolveTheme(preference: ThemePreference): ResolvedTheme {
  if (preference === "system") {
    return prefersDark() ? "dark" : "light";
  }
  return preference;
}

/**
 * Apply a preference to the document and remember it.
 *
 * `"system"` removes the attribute so the media query takes over again.
 */
export function applyTheme(preference: ThemePreference): void {
  const root = globalThis.document?.documentElement;
  if (root !== undefined) {
    if (preference === "system") {
      root.removeAttribute("data-theme");
    } else {
      root.setAttribute("data-theme", preference);
    }
  }
  try {
    globalThis.localStorage?.setItem(STORAGE_KEY, preference);
  } catch {
    // Private mode or blocked storage: the choice simply does not survive a reload.
  }
}

/**
 * React binding for the appearance switch.
 *
 * @returns The current preference, the resolved scheme, and a setter that applies both.
 */
export function useTheme(): {
  preference: ThemePreference;
  resolved: ResolvedTheme;
  setPreference: (next: ThemePreference) => void;
} {
  const [preference, setPreferenceState] = useState<ThemePreference>(readThemePreference);
  const [resolved, setResolved] = useState<ResolvedTheme>(() => resolveTheme(preference));

  useEffect(() => {
    applyTheme(preference);
    setResolved(resolveTheme(preference));
    if (preference !== "system" || typeof globalThis.matchMedia !== "function") {
      return undefined;
    }
    const media = globalThis.matchMedia(DARK_QUERY);
    const onChange = (): void => {
      setResolved(media.matches ? "dark" : "light");
    };
    media.addEventListener("change", onChange);
    return () => {
      media.removeEventListener("change", onChange);
    };
  }, [preference]);

  return {
    preference,
    resolved,
    setPreference: (next: ThemePreference) => {
      setPreferenceState(next);
    },
  };
}
