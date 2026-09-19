/**
 * Every control is big enough for a finger or a pen.
 *
 * The brief's number is 44 px, and 56 px once presentation mode is on. Both live in one token
 * pair — `--tap-target-min` and `--control-height-md`/`--control-height-lg` — so a rule that uses
 * either grows with the mode and a rule that does not is simply too small on a wall board. The
 * `Button` primitive gets this right for everything built on it; what slips through are the raw
 * elements: a navigation link sized by its own text, a native `<select>` with two pixels of
 * padding, a close button drawn as a glyph.
 *
 * So this test reads the pairs of `X.tsx` / `X.module.css` and, for every interactive element that
 * a CSS-module class is attached to, requires **at least one** of that element's classes to set a
 * height. "At least one" is the right rule because state classes compose: `cx(styles.mic,
 * recording && styles.micActive)` is one 64 px button, not two undersized ones.
 *
 * It is a source check rather than a rendered one because jsdom performs no layout — there is no
 * height to measure at runtime, only the intent expressed in the stylesheet.
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";

import { describe, expect, it } from "vitest";

/** Vitest runs with `web/` as the working directory. */
const WEB_ROOT = resolve(".");
const SOURCE_ROOT = resolve(WEB_ROOT, "src");

/** Elements a lecturer taps, presses or drags. */
const INTERACTIVE_TAGS = new Set([
  "button",
  "a",
  "select",
  "input",
  "textarea",
  "summary",
  "NavLink",
  "Link",
]);

/** Declarations that give an element a real box in the tap-target sense. */
const SIZED = ["min-height", "height:", "control-height", "tap-target", "aspect-ratio"];

/**
 * Classes that are legitimately unsized because the element is not a target on its own.
 *
 * Each entry is `<css path> .<class>` with the reason it is exempt. The map is empty today: every
 * styled control in the client carries a height token, and it is meant to stay that way — an entry
 * added here should be an argued exception, not a way to quiet the test.
 */
const EXEMPT: ReadonlyMap<string, string> = new Map<string, string>();

/** Every `.tsx` that has a CSS module beside it. */
function componentPairs(directory: string): { tsx: string; css: string }[] {
  const pairs: { tsx: string; css: string }[] = [];
  for (const entry of readdirSync(directory)) {
    const full = join(directory, entry);
    if (statSync(full).isDirectory()) {
      if (entry === "generated" || entry === "test") {
        continue;
      }
      pairs.push(...componentPairs(full));
      continue;
    }
    if (!entry.endsWith(".tsx")) {
      continue;
    }
    const css = full.replace(/\.tsx$/, ".module.css");
    try {
      statSync(css);
      pairs.push({ tsx: full, css });
    } catch {
      // No stylesheet beside this component: nothing to check.
    }
  }
  return pairs;
}

/**
 * The text of every JSX opening tag for an interactive element.
 *
 * The scan is brace-aware because `className` is often a callback — React Router's `NavLink` takes
 * `({ isActive }) => …` — and a naive "up to the next `>`" would stop inside the arrow function.
 */
function openingTags(source: string): { tag: string; text: string }[] {
  const tags: { tag: string; text: string }[] = [];
  const opener = /<([A-Za-z][\w.]*)[\s>]/g;
  for (const match of source.matchAll(opener)) {
    const name = match[1] ?? "";
    if (!INTERACTIVE_TAGS.has(name)) {
      continue;
    }
    const start = match.index;
    let depth = 0;
    for (let i = start; i < source.length; i += 1) {
      const char = source[i];
      if (char === "{") {
        depth += 1;
      } else if (char === "}") {
        depth -= 1;
      } else if (char === ">" && depth === 0 && i > start) {
        tags.push({ tag: name, text: source.slice(start, i) });
        break;
      }
    }
  }
  return tags;
}

/**
 * Merge every declaration block of one class into a single string.
 *
 * Comments are removed first. Without that, prose containing a dot followed by word characters —
 * "WCAG 1.4.11", a file name, a selector being explained — reads as a selector to the regex below
 * and swallows the real rule that follows it, so a properly sized control is reported as
 * undersized. The stylesheet's own documentation must not be able to change its meaning.
 */
function ruleBodies(source: string): Map<string, string> {
  const css = source.replaceAll(/\/\*[\s\S]*?\*\//g, "");
  const bodies = new Map<string, string>();
  for (const match of css.matchAll(/\.([\w-]+)([^{]*)\{([^}]*)\}/g)) {
    const name = match[1] ?? "";
    bodies.set(name, (bodies.get(name) ?? "") + (match[3] ?? ""));
  }
  return bodies;
}

/** One element whose classes never mention a height. */
interface Undersized {
  readonly file: string;
  readonly tag: string;
  readonly classes: string[];
}

function undersizedControls(): Undersized[] {
  const findings: Undersized[] = [];
  for (const { tsx, css } of componentPairs(SOURCE_ROOT)) {
    const bodies = ruleBodies(readFileSync(css, "utf-8"));
    const cssPath = relative(WEB_ROOT, css).replaceAll("\\", "/");
    for (const { tag, text } of openingTags(readFileSync(tsx, "utf-8"))) {
      const classes = [...text.matchAll(/\bstyles\.(\w+)/g)]
        .map((match) => match[1] ?? "")
        .filter((name) => bodies.has(name));
      if (classes.length === 0) {
        // No CSS module class on this element: either it is unstyled, or a shared utility class
        // owns it. Nothing this test can judge from the stylesheet beside it.
        continue;
      }
      if (classes.some((name) => EXEMPT.has(`${cssPath} .${name}`))) {
        continue;
      }
      const sized = classes.some((name) => {
        const body = bodies.get(name) ?? "";
        return SIZED.some((token) => body.includes(token));
      });
      if (!sized) {
        findings.push({
          file: `${relative(WEB_ROOT, tsx).replaceAll("\\", "/")} (${cssPath})`,
          tag,
          classes,
        });
      }
    }
  }
  return findings;
}

describe("dokunma hedefleri", () => {
  it("gives every styled control a minimum height", () => {
    const findings = undersizedControls().map(
      (entry) => `${entry.file}  <${entry.tag} class=${entry.classes.join("+")}>`,
    );

    expect(findings, "en az 44 px yüksekliği olmayan denetimler").toEqual([]);
  });

  it("actually inspects the components it claims to", () => {
    // Without this the assertion above would pass on an empty walk.
    const pairs = componentPairs(SOURCE_ROOT);
    expect(pairs.length).toBeGreaterThan(30);
    // A comment mentioning "1.4.11" must not eat the rule that follows it.
    const bodies = ruleBodies(
      ["/* WCAG 1.4.11 */", ".a { min-height: 3rem; }", ".a:hover { color: red; }"].join("\n"),
    );
    expect(bodies.get("a")).toContain("min-height");
    const tags = openingTags(
      'const x = <NavLink to="/a" className={({ isActive }) => (isActive ? styles.on : styles.off)}>b</NavLink>;',
    );
    expect(tags).toHaveLength(1);
    expect(tags[0]?.text).toContain("styles.off");
  });

  it("documents why each exemption is exempt", () => {
    for (const [target, reason] of EXEMPT) {
      expect(reason.length, `${target} için gerekçe`).toBeGreaterThan(30);
    }
  });
});
