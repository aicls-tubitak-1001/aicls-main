/**
 * Application store, theme preference and the i18n catalogue.
 *
 * The i18n test is the guard that keeps Turkish text out of the components: every value must be a
 * non-empty string and every key must be plain ASCII (English), matching the backend rule that
 * Turkish user-facing strings live in exactly one place.
 */
import { afterEach, describe, expect, it, vi } from "vitest";

import { appStore, createStore, resetAppState } from "@/app/store";
import { applyTheme, prefersDark, readThemePreference, resolveTheme } from "@/app/theme";
import { messages, t } from "@/i18n";

afterEach(() => {
  resetAppState();
  globalThis.localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
  vi.restoreAllMocks();
});

describe("createStore", () => {
  it("merges patches and notifies subscribers", () => {
    const store = createStore({ count: 0, label: "a" });
    const seen: number[] = [];
    store.subscribe((state) => seen.push(state.count));

    store.setState({ count: 1 });
    store.setState((state) => ({ count: state.count + 1 }));

    expect(store.getState()).toEqual({ count: 2, label: "a" });
    expect(seen).toEqual([1, 2]);
  });

  it("does not notify when nothing changed", () => {
    const store = createStore({ count: 0 });
    const listener = vi.fn();
    store.subscribe(listener);

    store.setState({ count: 0 });

    expect(listener).not.toHaveBeenCalled();
  });

  it("stops notifying after unsubscribe", () => {
    const store = createStore({ count: 0 });
    const listener = vi.fn();
    const off = store.subscribe(listener);

    off();
    store.setState({ count: 5 });

    expect(listener).not.toHaveBeenCalled();
  });
});

describe("appStore", () => {
  it("starts idle with no outcome", () => {
    expect(appStore.getState().phase).toBe("idle");
    expect(appStore.getState().outcome).toBeNull();
  });

  it("resets to the initial state", () => {
    appStore.setState({ commandText: "x karenin türevini al", phase: "busy" });
    resetAppState();

    expect(appStore.getState().commandText).toBe("");
    expect(appStore.getState().phase).toBe("idle");
  });
});

describe("theme", () => {
  it("defaults to the system preference", () => {
    expect(readThemePreference()).toBe("system");
  });

  it("remembers an explicit choice and stamps the document", () => {
    applyTheme("dark");

    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    expect(readThemePreference()).toBe("dark");
  });

  it("removes the attribute again for the system preference", () => {
    applyTheme("dark");
    applyTheme("system");

    expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
    expect(readThemePreference()).toBe("system");
  });

  it("resolves the system preference against prefers-color-scheme", () => {
    vi.spyOn(globalThis, "matchMedia").mockReturnValue({
      matches: true,
      media: "(prefers-color-scheme: dark)",
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
    } as unknown as MediaQueryList);

    expect(prefersDark()).toBe(true);
    expect(resolveTheme("system")).toBe("dark");
    expect(resolveTheme("light")).toBe("light");
  });

  it("falls back to the system preference when storage holds junk", () => {
    globalThis.localStorage.setItem("aicls.theme", "neon");

    expect(readThemePreference()).toBe("system");
  });
});

describe("i18n", () => {
  it("returns the Turkish string for a key", () => {
    expect(t("nav.workspace")).toBe("Çalışma Alanı");
  });

  it("interpolates placeholders", () => {
    expect(t("workspace.strokeCount", { count: 3 })).toBe("İz sayısı: 3");
  });

  it("leaves unknown placeholders untouched", () => {
    expect(t("steps.step", {})).toBe("{{index}}. adım");
  });

  it("has a non-empty Turkish value for every key", () => {
    for (const [key, value] of Object.entries(messages)) {
      expect(value.trim(), `boş değer: ${key}`).not.toBe("");
      expect(key, `anahtar İngilizce ASCII olmalı: ${key}`).toMatch(/^[a-zA-Z0-9.+-]+$/);
    }
  });

  it("carries the contractual unsupported-problem sentence verbatim", () => {
    expect(t("errors.unsupportedProblem")).toBe("Bu problem tipi şu an için desteklenmemektedir.");
  });
});
