/**
 * Renders a LaTeX expression with KaTeX, behind the allow-list sanitiser.
 *
 * Pipeline (guvenlik-gizlilik.md §6, ADR-0019):
 *
 * 1. `sanitizeLatex` strips everything outside the KaTeX math allow-list — `\href`, `\url`,
 *    `\includegraphics`, macro definitions and KaTeX's HTML extensions disappear together with
 *    their arguments, so nothing hostile ever reaches the renderer.
 * 2. `katex.renderToString` runs with `trust: false` and `maxExpand: 1000`; the HTML it returns is
 *    injected with `dangerouslySetInnerHTML`, which is safe precisely because of steps 1 and 2.
 * 3. A command that survives sanitising but KaTeX does not implement (`\sech`, `\csch`, `\arccot`,
 *    ...) throws; the expression is then handed to the **MathJax fallback slot**: a marked element
 *    carrying the sanitised source between MathJax delimiters. `moduller.md` describes MathJax as
 *    "yalnızca desteklenmeyen makro yedeği" — the slot is rendered even when no MathJax bundle is
 *    loaded, so the user sees the source instead of an empty box, and `data-mathjax-fallback`
 *    marks the element for the typesetter and for the nightly KaTeX-coverage check.
 */
import katex from "katex";
import { useEffect, useMemo } from "react";

import { t } from "@/i18n";
import { LatexTooLongError, MAX_LATEX_LENGTH, sanitizeLatex } from "@/lib/latexSafety";
import styles from "./MathView.module.css";

export interface MathViewProps {
  /** Raw LaTeX; sanitised before rendering. */
  latex: string;
  /** Render as a centred display equation instead of inline. */
  display?: boolean;
  /** Called when KaTeX cannot render the (sanitised) expression. */
  onError?: (error: Error) => void;
}

interface RenderOutcome {
  html: string | null;
  sanitized: string;
  error: Error | null;
}

/** KaTeX options fixed by the security policy; never widened per call site. */
const KATEX_OPTIONS = {
  throwOnError: true,
  trust: false,
  strict: "ignore",
  maxExpand: 1000,
} as const;

function render(latex: string, display: boolean): RenderOutcome {
  let sanitized = "";
  try {
    sanitized = sanitizeLatex(latex, { maxLen: MAX_LATEX_LENGTH });
  } catch (error) {
    return {
      html: null,
      sanitized: "",
      error: error instanceof LatexTooLongError ? error : new Error(t("errors.tooLong")),
    };
  }
  if (sanitized === "") {
    return { html: null, sanitized: "", error: null };
  }
  try {
    return {
      html: katex.renderToString(sanitized, { ...KATEX_OPTIONS, displayMode: display }),
      sanitized,
      error: null,
    };
  } catch (error) {
    return {
      html: null,
      sanitized,
      error: error instanceof Error ? error : new Error(t("math.renderError")),
    };
  }
}

/**
 * Display a mathematical expression.
 *
 * @returns A `<span>` (or `<div>` in display mode) containing KaTeX output, or the MathJax
 *   fallback slot when KaTeX rejects the expression.
 */
export function MathView({ latex, display = false, onError }: MathViewProps): JSX.Element {
  const outcome = useMemo(() => render(latex, display), [latex, display]);

  useEffect(() => {
    if (outcome.error !== null) {
      onError?.(outcome.error);
    }
  }, [outcome, onError]);

  const className = `${styles.math} ${display ? styles.display : styles.inline} math-body`;

  if (outcome.html !== null) {
    return (
      <span
        className={className}
        role="math"
        aria-label={t("math.label")}
        data-latex={outcome.sanitized}
        dangerouslySetInnerHTML={{ __html: outcome.html }}
      />
    );
  }

  if (outcome.sanitized === "") {
    return <span className={className} data-latex="" />;
  }

  const delimited = display ? `\\[${outcome.sanitized}\\]` : `\\(${outcome.sanitized}\\)`;
  return (
    <span
      className={`${className} ${styles.fallback}`}
      data-mathjax-fallback="true"
      data-latex={outcome.sanitized}
      title={t("math.fallbackNotice")}
    >
      {delimited}
    </span>
  );
}
