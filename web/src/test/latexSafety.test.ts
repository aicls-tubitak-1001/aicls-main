/**
 * The browser sanitiser must agree with `aicls.core.latex_safety`, which is the security boundary
 * the whole rendering path relies on (guvenlik-gizlilik.md §6). These cases mirror the Python
 * suite: forbidden commands leave with their arguments, unknown commands leave their groups behind,
 * unknown environments are unwrapped, braces are balanced and the function is idempotent.
 */
import { describe, expect, it } from "vitest";

import {
  ALLOWED_COMMANDS,
  findDisallowedCommands,
  FORBIDDEN_COMMANDS,
  LatexTooLongError,
  MAX_LATEX_LENGTH,
  sanitizeLatex,
} from "@/lib/latexSafety";

describe("sanitizeLatex", () => {
  it("keeps allow-listed mathematics verbatim", () => {
    const latex = "\\int_{0}^{1} x^{2} \\, dx = \\frac{1}{3}";

    expect(sanitizeLatex(latex)).toBe(latex);
  });

  it("keeps allowed environments", () => {
    const latex = "\\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}";

    expect(sanitizeLatex(latex)).toBe(latex);
  });

  it("unwraps unknown environments", () => {
    expect(sanitizeLatex("\\begin{tikzpicture} x \\end{tikzpicture}")).toBe("x");
  });

  it("removes forbidden commands together with their arguments", () => {
    expect(sanitizeLatex("\\href{https://a.example}{tikla}x")).toBe("x");
    expect(sanitizeLatex("\\includegraphics{gizli.png} y")).toBe("y");
    expect(sanitizeLatex("\\def\\evil#1{#1}\\evil{z}")).toBe("{z}");
  });

  it("removes unknown commands but keeps their groups", () => {
    expect(sanitizeLatex("\\foo{x}")).toBe("{x}");
  });

  it("strips math delimiters", () => {
    expect(sanitizeLatex("$x^{2}$")).toBe("x^{2}");
    expect(sanitizeLatex("\\[x\\]")).toBe("x");
  });

  it("balances stray braces", () => {
    expect(sanitizeLatex("{x")).toBe("{x}");
    expect(sanitizeLatex("x}")).toBe("x");
  });

  it("never lets two commands fuse into a new one", () => {
    // \unknown disappears; without the guard \frac + a would become \fraca.
    expect(sanitizeLatex("\\frac\\unknown ab")).toBe("\\frac ab");
  });

  it("is idempotent", () => {
    const inputs = ["\\href{u}{t}x", "\\begin{tikzpicture}a\\end{tikzpicture}", "{x", "\\foo{y}"];

    for (const input of inputs) {
      const once = sanitizeLatex(input);
      expect(sanitizeLatex(once)).toBe(once);
    }
  });

  it("never returns a forbidden command", () => {
    for (const name of FORBIDDEN_COMMANDS) {
      const output = sanitizeLatex(`\\${name}{a}{b} x`);
      expect(output).not.toContain(`\\${name}`);
    }
  });

  it("enforces the length limit when asked", () => {
    const long = "x".repeat(MAX_LATEX_LENGTH + 1);

    expect(() => sanitizeLatex(long, { maxLen: MAX_LATEX_LENGTH })).toThrow(LatexTooLongError);
    expect(sanitizeLatex(long)).toHaveLength(MAX_LATEX_LENGTH + 1);
  });

  it("drops control characters and normalises newlines", () => {
    expect(sanitizeLatex("x\u0000\r\ny")).toBe("x\ny");
  });

  it("returns an empty string for blank input", () => {
    expect(sanitizeLatex("   ")).toBe("");
  });

  it("has a non-trivial allow-list", () => {
    expect(ALLOWED_COMMANDS.has("frac")).toBe(true);
    expect(ALLOWED_COMMANDS.has("href")).toBe(false);
    expect(ALLOWED_COMMANDS.size).toBeGreaterThan(200);
  });
});

describe("findDisallowedCommands", () => {
  it("reports removed commands once, in order", () => {
    expect(findDisallowedCommands("\\href{a}{b}\\foo\\href{c}{d}")).toEqual(["href", "foo"]);
  });

  it("reports nothing for clean input", () => {
    expect(findDisallowedCommands("\\frac{1}{2}")).toEqual([]);
  });
});
