/**
 * The keyboard set.
 *
 * The rule worth testing: a bare key never fires while someone is typing a formula, a modified
 * binding always fires, and a browser combination (Ctrl/Cmd) is never stolen.
 */
import { describe, expect, it } from "vitest";

import { isTypingTarget, matchShortcut, SHORTCUTS, shortcutSummary } from "@/app/shortcuts";

function event(
  key: string,
  options: { altKey?: boolean; ctrlKey?: boolean; metaKey?: boolean; target?: EventTarget } = {},
): Parameters<typeof matchShortcut>[0] {
  return {
    key,
    altKey: options.altKey ?? false,
    ctrlKey: options.ctrlKey ?? false,
    metaKey: options.metaKey ?? false,
    target: options.target ?? document.body,
  };
}

describe("matchShortcut", () => {
  it("binds the documented actions", () => {
    expect(matchShortcut(event("/"))?.id).toBe("focusCommand");
    expect(matchShortcut(event("m", { altKey: true }))?.id).toBe("toggleMic");
    expect(matchShortcut(event("t", { altKey: true }))?.id).toBe("clearCanvas");
    expect(matchShortcut(event("s", { altKey: true }))?.id).toBe("togglePresentation");
    expect(matchShortcut(event("b", { altKey: true }))?.id).toBe("toggleBook");
    expect(matchShortcut(event("y", { altKey: true }))?.id).toBe("repeatLast");
    expect(matchShortcut(event("?"))?.id).toBe("showHelp");
  });

  it("ignores a bare key while a formula is being typed", () => {
    const input = document.createElement("input");

    expect(matchShortcut(event("/", { target: input }))).toBeNull();
    expect(matchShortcut(event("m", { altKey: true, target: input }))?.id).toBe("toggleMic");
  });

  it("never steals a browser combination", () => {
    expect(matchShortcut(event("k", { ctrlKey: true }))).toBeNull();
    expect(matchShortcut(event("p", { metaKey: true }))).toBeNull();
  });

  it("is case-insensitive on letters", () => {
    expect(matchShortcut(event("M", { altKey: true }))?.id).toBe("toggleMic");
  });

  it("returns null for an unbound key", () => {
    expect(matchShortcut(event("q", { altKey: true }))).toBeNull();
  });
});

describe("isTypingTarget", () => {
  it("recognises the places a lecturer types", () => {
    const editable = document.createElement("div");
    editable.setAttribute("contenteditable", "true");

    expect(isTypingTarget(document.createElement("input"))).toBe(true);
    expect(isTypingTarget(document.createElement("textarea"))).toBe(true);
    expect(isTypingTarget(document.createElement("select"))).toBe(true);
    expect(isTypingTarget(document.createElement("button"))).toBe(false);
    expect(isTypingTarget(null)).toBe(false);
    // jsdom leaves `isContentEditable` undefined, so the guard reads the attribute as well.
    expect(isTypingTarget(editable)).toBe(true);
  });
});

describe("shortcutSummary", () => {
  it("lists every action once, joining its alternative bindings", () => {
    const summary = shortcutSummary();
    const ids = summary.map((entry) => entry.id);

    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toContain("focusCommand");
    expect(summary.find((entry) => entry.id === "focusCommand")?.display).toBe("/ / Alt + K");
    expect(summary).toHaveLength(new Set(SHORTCUTS.map((item) => item.id)).size);
  });
});
