/**
 * One place for Turkish, checked mechanically.
 *
 * The rule the whole client is built on — mirroring `aicls.core.messages_tr` on the backend — is
 * that every Turkish sentence a person can read lives in `i18n/tr.json` under an English key, and
 * nowhere else. It is a rule that decays quietly: a `throw new Error("…")` written in Turkish
 * during a debugging session ends up on screen a month later, untranslatable and unreviewable.
 *
 * So this test reads the client's own source and fails on any string literal containing a letter
 * that only Turkish uses. Two things are deliberately allowed:
 *
 * - the catalogue itself, and the tests (which assert on Turkish output by design);
 * - the entries of {@link ALLOWED}, each a verbatim transcript of something the *backend* sends,
 *   quoted in the technical guide as a code sample. Translating those would make them wrong.
 *
 * A message thrown for a developer (a missing canvas context, an impossible sample rate) belongs
 * in English, like every other identifier and comment in the repository.
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { messages } from "@/i18n";

/** Letters that appear in Turkish and in no other language this repository writes. */
const TURKISH_LETTERS = /[çğıöşüÇĞİÖŞÜ]/;

/** Vitest runs with `web/` as the working directory. */
const WEB_ROOT = resolve(".");
const SOURCE_ROOT = resolve(WEB_ROOT, "src");

/** Files whose Turkish literals are quotations of the API, not interface text. */
const ALLOWED: ReadonlyMap<string, string> = new Map([
  [
    "src/features/guide/apiSurface.ts",
    "PROBLEM_EXAMPLE is a verbatim RFC 9457 body from docs/mimari/api.md §2, shown as a code " +
      "sample in the technical guide; its Turkish comes from aicls.core.messages_tr.",
  ],
]);

/** Every `.ts`/`.tsx` file of the application, excluding generated code and the tests. */
function sourceFiles(directory: string): string[] {
  const found: string[] = [];
  for (const entry of readdirSync(directory)) {
    const full = join(directory, entry);
    if (statSync(full).isDirectory()) {
      if (entry === "generated" || entry === "test") {
        continue;
      }
      found.push(...sourceFiles(full));
      continue;
    }
    if (entry.endsWith(".ts") || entry.endsWith(".tsx")) {
      found.push(full);
    }
  }
  return found;
}

/** Matches a single-line double-quoted, single-quoted or backtick string literal. */
const LITERAL = /"([^"\\\n]*)"|'([^'\\\n]*)'|`([^`\\\n]*)`/g;

/** One offending literal. */
interface Offence {
  readonly file: string;
  readonly line: number;
  readonly text: string;
}

/** Find every Turkish string literal outside the catalogue. */
function turkishLiterals(): Offence[] {
  const offences: Offence[] = [];
  for (const file of sourceFiles(SOURCE_ROOT)) {
    const relativePath = relative(WEB_ROOT, file).replaceAll("\\", "/");
    if (ALLOWED.has(relativePath)) {
      continue;
    }
    const lines = readFileSync(file, "utf-8").split("\n");
    lines.forEach((line, index) => {
      const trimmed = line.trim();
      // Comments and docstrings are prose about the code and may name Turkish things.
      if (trimmed.startsWith("*") || trimmed.startsWith("//") || trimmed.startsWith("/*")) {
        return;
      }
      for (const match of line.matchAll(LITERAL)) {
        const literal = match[1] ?? match[2] ?? match[3] ?? "";
        if (TURKISH_LETTERS.test(literal)) {
          offences.push({ file: relativePath, line: index + 1, text: literal.slice(0, 80) });
          return;
        }
      }
    });
  }
  return offences;
}

describe("i18n disiplini", () => {
  it("keeps every Turkish sentence in the catalogue", () => {
    const offences = turkishLiterals().map(
      (offence) => `${offence.file}:${String(offence.line)}  "${offence.text}"`,
    );

    expect(offences, "i18n dışında Türkçe metin").toEqual([]);
  });

  it("still inspects the files it is meant to inspect", () => {
    // A broken walker or a regex that stops matching would make the assertion above vacuous.
    const files = sourceFiles(SOURCE_ROOT);
    expect(files.length).toBeGreaterThan(100);
    expect(TURKISH_LETTERS.test("Çalışma Alanı")).toBe(true);
    expect(TURKISH_LETTERS.test("Workspace")).toBe(false);
  });

  it("documents why each allowed file is allowed", () => {
    for (const [file, reason] of ALLOWED) {
      expect(reason.length, `${file} için gerekçe`).toBeGreaterThan(40);
      expect(() => readFileSync(resolve(WEB_ROOT, file), "utf-8")).not.toThrow();
    }
  });

  it("has a catalogue where every value is Turkish text and every key is English", () => {
    const keys = Object.keys(messages);
    expect(keys.length).toBeGreaterThan(500);
    const nonAscii = keys.filter((key) => !/^[a-zA-Z0-9.]+$/.test(key));
    expect(nonAscii, "İngilizce olmayan anahtarlar").toEqual([]);
    const empty = keys.filter((key) => (messages[key] ?? "").trim() === "");
    expect(empty, "boş çeviriler").toEqual([]);
  });
});
