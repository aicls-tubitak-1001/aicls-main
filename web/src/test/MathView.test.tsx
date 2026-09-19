/**
 * `MathView.test.tsx` — contractual test from `docs/mimari/moduller.md#web--sahibi-sg`:
 * `\frac` renders through KaTeX, an unsupported macro falls back to the MathJax slot, and `\href`
 * is rejected by the allow-list sanitiser (guvenlik-gizlilik.md §6).
 */
import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { MathView } from "@/features/results/MathView";

describe("MathView", () => {
  it("renders \\frac with KaTeX", () => {
    const { container } = render(<MathView latex="\frac{d}{dx} x^{2} = 2x" display />);

    const katexRoot = container.querySelector(".katex");
    expect(katexRoot).not.toBeNull();
    expect(container.querySelector('[data-mathjax-fallback="true"]')).toBeNull();
    // Attribute queries rather than `getByRole`: jsdom's getComputedStyle throws on the MathML
    // subtree KaTeX emits for screen readers, which any role query would walk.
    expect(container.querySelector('[role="math"]')).toHaveAttribute(
      "data-latex",
      "\\frac{d}{dx} x^{2} = 2x",
    );
  });

  it("falls back to the MathJax slot for a macro KaTeX does not implement", () => {
    const onError = vi.fn();
    // \sech is on the allow-list (it is a real math function) but KaTeX has no definition for it.
    const { container } = render(<MathView latex="\sech{x}" onError={onError} />);

    const fallback = container.querySelector('[data-mathjax-fallback="true"]');
    expect(fallback).not.toBeNull();
    expect(fallback?.textContent).toBe("\\(\\sech{x}\\)");
    expect(container.querySelector(".katex")).toBeNull();
    expect(onError).toHaveBeenCalledTimes(1);
  });

  it("uses display delimiters in the fallback slot when display is set", () => {
    const { container } = render(<MathView latex="\sech{x}" display />);

    expect(container.querySelector('[data-mathjax-fallback="true"]')?.textContent).toBe(
      "\\[\\sech{x}\\]",
    );
  });

  it("rejects \\href: neither the command nor a link survives", () => {
    const { container } = render(<MathView latex="\href{https://kotu.example}{tikla} + x" />);

    expect(container.innerHTML).not.toContain("href");
    expect(container.innerHTML).not.toContain("kotu.example");
    expect(container.querySelector("a")).toBeNull();
    expect(container.querySelector('[role="math"]')).toHaveAttribute("data-latex", "+ x");
  });

  it("rejects KaTeX HTML extensions and macro definitions", () => {
    const { container } = render(<MathView latex="\htmlStyle{color:red}{x}\def\evil{y}\evil" />);

    expect(container.innerHTML).not.toContain("htmlStyle");
    expect(container.innerHTML).not.toContain("color:red");
    expect(container.innerHTML).not.toContain("\\def");
  });

  it("renders nothing for an empty expression", () => {
    const { container } = render(<MathView latex="   " />);

    expect(container.querySelector(".katex")).toBeNull();
    expect(container.querySelector('[data-mathjax-fallback="true"]')).toBeNull();
  });
});
