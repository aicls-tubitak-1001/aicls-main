/**
 * Presentation mode — the `data-density` attribute.
 *
 * One attribute drives the whole type ramp and every control size, so these tests check the two
 * things that can silently break it: the attribute actually reaching `<html>`, and the choice
 * surviving (or failing safely) across a reload.
 */
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  applyDensity,
  currentDensity,
  DENSITY_ATTRIBUTE,
  otherDensity,
  readDensity,
  useDensity,
} from "@/app/density";
import { isOnline, useOnline } from "@/app/online";

beforeEach(() => {
  globalThis.localStorage.clear();
  document.documentElement.removeAttribute(DENSITY_ATTRIBUTE);
});

afterEach(() => {
  vi.unstubAllGlobals();
  globalThis.localStorage.clear();
  document.documentElement.removeAttribute(DENSITY_ATTRIBUTE);
});

describe("applyDensity", () => {
  it("writes the attribute the token file keys off", () => {
    applyDensity("presentation");

    expect(document.documentElement.getAttribute(DENSITY_ATTRIBUTE)).toBe("presentation");
    expect(currentDensity()).toBe("presentation");
  });

  it("writes 'normal' explicitly rather than removing the attribute", () => {
    applyDensity("presentation");
    applyDensity("normal");

    expect(document.documentElement.getAttribute(DENSITY_ATTRIBUTE)).toBe("normal");
  });

  it("remembers the choice for the next lecture", () => {
    applyDensity("presentation");

    expect(readDensity()).toBe("presentation");
  });

  it("falls back to normal when storage is unavailable or holds nonsense", () => {
    globalThis.localStorage.setItem("aicls.density", "enormous");
    expect(readDensity()).toBe("normal");

    vi.stubGlobal("localStorage", {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
    });
    expect(readDensity()).toBe("normal");
    expect(() => {
      applyDensity("presentation");
    }).not.toThrow();
  });

  it("names the other density for the toggle", () => {
    expect(otherDensity("normal")).toBe("presentation");
    expect(otherDensity("presentation")).toBe("normal");
  });
});

describe("useDensity", () => {
  it("applies the stored density on mount and toggles both ways", () => {
    const { result } = renderHook(() => useDensity());

    expect(result.current.density).toBe("normal");
    expect(result.current.isPresentation).toBe(false);
    expect(document.documentElement.getAttribute(DENSITY_ATTRIBUTE)).toBe("normal");

    act(() => {
      result.current.toggleDensity();
    });

    expect(result.current.density).toBe("presentation");
    expect(result.current.isPresentation).toBe(true);
    expect(document.documentElement.getAttribute(DENSITY_ATTRIBUTE)).toBe("presentation");

    act(() => {
      result.current.toggleDensity();
    });

    expect(document.documentElement.getAttribute(DENSITY_ATTRIBUTE)).toBe("normal");
  });

  it("accepts a density set directly (the header select)", () => {
    const { result } = renderHook(() => useDensity());

    act(() => {
      result.current.setDensity("presentation");
    });

    expect(document.documentElement.getAttribute(DENSITY_ATTRIBUTE)).toBe("presentation");
    expect(readDensity()).toBe("presentation");
  });
});

describe("useOnline", () => {
  it("follows the browser's online and offline events", () => {
    const { result } = renderHook(() => useOnline());

    expect(result.current).toBe(true);

    act(() => {
      Object.defineProperty(globalThis.navigator, "onLine", { value: false, configurable: true });
      globalThis.dispatchEvent(new Event("offline"));
    });
    expect(result.current).toBe(false);

    act(() => {
      Object.defineProperty(globalThis.navigator, "onLine", { value: true, configurable: true });
      globalThis.dispatchEvent(new Event("online"));
    });
    expect(result.current).toBe(true);
  });

  it("assumes a connection when the browser does not report one", () => {
    expect(isOnline()).toBe(true);
  });
});
