/**
 * The documented keyboard set.
 *
 * A lecturer who has driven this board for a term should be able to run a whole session without
 * looking for a button: focus the command bar, toggle the microphone, clear the canvas, switch to
 * presentation mode, open the book panel, repeat the last command. The bindings live here — one
 * table, one matcher, one hook — so the help dialog and the handlers can never disagree.
 *
 * Rules that keep them safe on a shared machine:
 * - a bare key (`/`, `?`) is ignored while the focus is in a text field, so typing a fraction
 *   never triggers a shortcut;
 * - a modified binding (`Alt+…`) works everywhere, including inside the command bar;
 * - `Alt` is used rather than `Ctrl`, which browsers reserve (Ctrl+P, Ctrl+K, Ctrl+B …);
 * - every shortcut also exists as a visible control. The keyboard is an accelerator, never the
 *   only way to reach a function.
 */
import { useEffect } from "react";

import type { MessageKey } from "@/i18n";

/** Every action the shortcut layer knows about. */
export type ShortcutId =
  | "focusCommand"
  | "toggleMic"
  | "clearCanvas"
  | "togglePresentation"
  | "toggleBook"
  | "repeatLast"
  | "showHelp";

/** One binding. */
export interface Shortcut {
  readonly id: ShortcutId;
  /** `event.key` values that trigger it (lower-cased for letters). */
  readonly keys: readonly string[];
  /** Whether `Alt` must be held. */
  readonly alt: boolean;
  /** Human-readable form for the help dialog, e.g. `"Alt + M"`. */
  readonly display: string;
  /** i18n key of the Turkish description. */
  readonly labelKey: MessageKey;
  /** May it fire while a text field has focus? */
  readonly allowInInput: boolean;
}

/** The shortcut table, in the order the help dialog lists them. */
export const SHORTCUTS: readonly Shortcut[] = [
  {
    id: "focusCommand",
    keys: ["/"],
    alt: false,
    display: "/",
    labelKey: "shortcuts.focusCommand",
    allowInInput: false,
  },
  {
    id: "focusCommand",
    keys: ["k"],
    alt: true,
    display: "Alt + K",
    labelKey: "shortcuts.focusCommand",
    allowInInput: true,
  },
  {
    id: "toggleMic",
    keys: ["m"],
    alt: true,
    display: "Alt + M",
    labelKey: "shortcuts.toggleMic",
    allowInInput: true,
  },
  {
    id: "clearCanvas",
    keys: ["t"],
    alt: true,
    display: "Alt + T",
    labelKey: "shortcuts.clearCanvas",
    allowInInput: true,
  },
  {
    id: "togglePresentation",
    keys: ["s"],
    alt: true,
    display: "Alt + S",
    labelKey: "shortcuts.togglePresentation",
    allowInInput: true,
  },
  {
    id: "toggleBook",
    keys: ["b"],
    alt: true,
    display: "Alt + B",
    labelKey: "shortcuts.toggleBook",
    allowInInput: true,
  },
  {
    id: "repeatLast",
    keys: ["y"],
    alt: true,
    display: "Alt + Y",
    labelKey: "shortcuts.repeatLast",
    allowInInput: true,
  },
  {
    id: "showHelp",
    keys: ["?"],
    alt: false,
    display: "?",
    labelKey: "shortcuts.showHelp",
    allowInInput: false,
  },
];

/**
 * Whether the event target is a place where the user is typing.
 *
 * The `contenteditable` attribute is read as well as `isContentEditable`, because jsdom (and only
 * jsdom) leaves the property undefined — the unit tests must judge the same elements the browser
 * does.
 */
export function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const editable = target.getAttribute("contenteditable");
  if (
    target.isContentEditable ||
    editable === "" ||
    editable === "true" ||
    editable === "plaintext-only"
  ) {
    return true;
  }
  const tag = target.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select";
}

/**
 * Which action a keyboard event triggers, if any.
 *
 * @param event - The `keydown` event.
 * @param typing - Whether the focus is in a text field (defaults to inspecting `event.target`).
 */
export function matchShortcut(
  event: Pick<KeyboardEvent, "key" | "altKey" | "ctrlKey" | "metaKey" | "target">,
  typing?: boolean,
): Shortcut | null {
  if (event.ctrlKey || event.metaKey) {
    return null;
  }
  const inField = typing ?? isTypingTarget(event.target);
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  for (const shortcut of SHORTCUTS) {
    if (shortcut.alt !== event.altKey) {
      continue;
    }
    if (!shortcut.keys.includes(key)) {
      continue;
    }
    if (inField && !shortcut.allowInInput) {
      continue;
    }
    return shortcut;
  }
  return null;
}

/** The unique actions, for the help dialog (a shortcut with two bindings is listed once). */
export function shortcutSummary(): { id: ShortcutId; labelKey: MessageKey; display: string }[] {
  const byId = new Map<ShortcutId, { id: ShortcutId; labelKey: MessageKey; display: string }>();
  for (const shortcut of SHORTCUTS) {
    const existing = byId.get(shortcut.id);
    if (existing === undefined) {
      byId.set(shortcut.id, {
        id: shortcut.id,
        labelKey: shortcut.labelKey,
        display: shortcut.display,
      });
    } else {
      existing.display = `${existing.display} / ${shortcut.display}`;
    }
  }
  return [...byId.values()];
}

/** Handlers a screen supplies; an action without a handler is simply not bound. */
export type ShortcutHandlers = Partial<Record<ShortcutId, () => void>>;

/**
 * Bind the shortcut table for as long as the component is mounted.
 *
 * @param handlers - What to run for each action.
 * @param options.enabled - Turn the whole set off (e.g. while a dialog is open).
 *
 * @example
 * ```ts
 * useShortcuts({
 *   focusCommand: () => inputRef.current?.focus(),
 *   togglePresentation: toggleDensity,
 * });
 * ```
 */
export function useShortcuts(
  handlers: ShortcutHandlers,
  options: { enabled?: boolean } = {},
): void {
  const { enabled = true } = options;
  useEffect(() => {
    if (!enabled) {
      return undefined;
    }
    const onKeyDown = (event: KeyboardEvent): void => {
      const shortcut = matchShortcut(event);
      if (shortcut === null) {
        return;
      }
      const handler = handlers[shortcut.id];
      if (handler === undefined) {
        return;
      }
      event.preventDefault();
      handler();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
    };
    // `handlers` is intentionally spread into the dependency list by identity: callers pass a
    // fresh object each render, and rebinding one listener is cheaper than memoising at every
    // call site.
  }, [enabled, handlers]);
}
