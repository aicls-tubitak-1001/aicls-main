/**
 * The palette, measured rather than asserted from memory.
 *
 * Contrast is a contractual property of this interface, not a matter of taste: the platform is
 * driven from a wall-mounted board that a room reads from four metres away, and the usability
 * study (İP-9) is run on whatever these tokens resolve to. So the tokens are parsed straight out
 * of `styles/tokens.css` and every pair that carries meaning is recomputed with the WCAG 2.x
 * relative-luminance formula. Editing a hex without checking it fails here, with the measured
 * ratio in the message.
 *
 * Three thresholds are used, each for what WCAG actually asks of that pair:
 *
 * - **7:1 (AAA)** for the result area, which is the one thing the back row must read;
 * - **4.5:1 (AA)** for every other text-on-background pair;
 * - **3:1 (AA non-text, WCAG 1.4.11)** for `--color-border-strong`, which is the only visible
 *   boundary of a secondary button and of a text field.
 *
 * `--color-border` is deliberately not in the list: it draws panel hairlines and table rules, and
 * is never the sole indication of a control.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

/**
 * The token file with its comments removed.
 *
 * Stripping them first matters: the file's own header prose quotes the selectors this test slices
 * on (`@media (prefers-color-scheme: dark)`, `:root[data-theme="dark"]`), so parsing the raw text
 * would cut the light palette off at the documentation instead of at the rule.
 */
const TOKENS = readFileSync(resolve("src/styles/tokens.css"), "utf-8").replaceAll(
  /\/\*[\s\S]*?\*\//g,
  "",
);

/** Channel-wise sRGB → linear, per WCAG 2.x. */
function linear(channel: number): number {
  const c = channel / 255;
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** Relative luminance of a `#rrggbb` colour. */
function luminance(hex: string): number {
  const value = hex.replace("#", "");
  const r = Number.parseInt(value.slice(0, 2), 16);
  const g = Number.parseInt(value.slice(2, 4), 16);
  const b = Number.parseInt(value.slice(4, 6), 16);
  return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b);
}

/** WCAG contrast ratio between two `#rrggbb` colours, rounded to two decimals. */
export function contrastRatio(foreground: string, background: string): number {
  const a = luminance(foreground);
  const b = luminance(background);
  const ratio = (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
  return Math.round(ratio * 100) / 100;
}

/**
 * The colour tokens of one block of `tokens.css`.
 *
 * The light palette is everything declared before the first `@media`; the dark palette is the
 * explicit `:root[data-theme="dark"]` block, which carries the same values as the
 * `prefers-color-scheme` branch (the file states both so the toggle wins in either direction).
 */
function paletteOf(source: string): Record<string, string> {
  const colours: Record<string, string> = {};
  for (const match of source.matchAll(/--(color-[\w-]+):\s*(#[0-9a-fA-F]{6})/g)) {
    colours[match[1] ?? ""] = (match[2] ?? "").toLowerCase();
  }
  return colours;
}

const LIGHT = paletteOf(TOKENS.slice(0, TOKENS.indexOf("@media (prefers-color-scheme: dark)")));
const DARK = paletteOf(
  TOKENS.slice(
    TOKENS.indexOf(':root[data-theme="dark"]'),
    TOKENS.indexOf("@media (prefers-contrast"),
  ),
);

/** One measured pair. */
interface Pair {
  readonly fg: string;
  readonly bg: string;
  /** Minimum acceptable ratio. */
  readonly min: number;
}

/** Text pairs that must clear AA, plus the two that must clear AAA. */
const TEXT_PAIRS: readonly Pair[] = [
  { fg: "color-result-text", bg: "color-result-bg", min: 7 },
  { fg: "color-text", bg: "color-surface", min: 7 },
  { fg: "color-text", bg: "color-bg", min: 4.5 },
  { fg: "color-text", bg: "color-surface-muted", min: 4.5 },
  { fg: "color-text", bg: "color-surface-raised", min: 4.5 },
  { fg: "color-text-muted", bg: "color-surface", min: 4.5 },
  { fg: "color-text-muted", bg: "color-bg", min: 4.5 },
  { fg: "color-text-muted", bg: "color-surface-muted", min: 4.5 },
  { fg: "color-accent", bg: "color-surface", min: 4.5 },
  { fg: "color-accent", bg: "color-bg", min: 4.5 },
  { fg: "color-accent", bg: "color-accent-soft", min: 4.5 },
  { fg: "color-accent-contrast", bg: "color-accent", min: 4.5 },
  { fg: "color-text-inverse", bg: "color-accent", min: 4.5 },
  { fg: "color-highlight", bg: "color-surface", min: 4.5 },
  { fg: "color-ok", bg: "color-ok-bg", min: 4.5 },
  { fg: "color-ok", bg: "color-surface", min: 4.5 },
  { fg: "color-warn", bg: "color-warn-bg", min: 4.5 },
  { fg: "color-warn", bg: "color-surface", min: 4.5 },
  { fg: "color-critical", bg: "color-critical-bg", min: 4.5 },
  { fg: "color-critical", bg: "color-surface", min: 4.5 },
  { fg: "color-info", bg: "color-info-bg", min: 4.5 },
  { fg: "color-info", bg: "color-surface", min: 4.5 },
  { fg: "color-ink", bg: "color-canvas", min: 4.5 },
];

/** Control outlines: WCAG 1.4.11 asks 3:1 of anything that identifies a component. */
const BOUNDARY_PAIRS: readonly Pair[] = [
  { fg: "color-border-strong", bg: "color-surface", min: 3 },
  { fg: "color-border-strong", bg: "color-bg", min: 3 },
  { fg: "color-border-strong", bg: "color-surface-muted", min: 3 },
];

describe("palet kontrastı", () => {
  it("parses both palettes out of the token file", () => {
    expect(Object.keys(LIGHT).length).toBeGreaterThan(20);
    expect(Object.keys(DARK).length).toBeGreaterThan(20);
    expect(LIGHT["color-surface"]).toBe("#ffffff");
    expect(DARK["color-surface"]).toBe("#182123");
  });

  for (const [theme, palette] of [
    ["açık tema", LIGHT],
    ["koyu tema", DARK],
  ] as const) {
    describe(theme, () => {
      for (const pair of [...TEXT_PAIRS, ...BOUNDARY_PAIRS]) {
        it(`${pair.fg} on ${pair.bg} is at least ${String(pair.min)}:1`, () => {
          const fg = palette[pair.fg];
          const bg = palette[pair.bg];
          expect(fg, `--${pair.fg} tanımlı değil`).toBeDefined();
          expect(bg, `--${pair.bg} tanımlı değil`).toBeDefined();
          const ratio = contrastRatio(fg ?? "#000000", bg ?? "#ffffff");
          expect(
            ratio,
            `--${pair.fg} (${fg ?? "?"}) / --${pair.bg} (${bg ?? "?"}) = ${String(ratio)}:1`,
          ).toBeGreaterThanOrEqual(pair.min);
        });
      }
    });
  }
});
