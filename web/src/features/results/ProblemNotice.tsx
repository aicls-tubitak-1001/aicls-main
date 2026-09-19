/**
 * Failure as a normal state.
 *
 * Every backend problem already carries a Turkish sentence (`detail_tr`), and for the one the
 * lecturer will meet most often — a problem type the CAS does not support yet — that sentence is
 * contract text frozen in İP-4: *"Bu problem tipi şu an için desteklenmemektedir."* It must be
 * shown calmly, with a suggestion to rephrase, next to the previous answer that is still on the
 * board. There is no crash screen here and no red for an answer that is simply "not yet".
 *
 * Two tones, decided by the code rather than by the status:
 *
 * - **quiet** (`UNSUPPORTED_PROBLEM`, `NLU_NO_INTENT`, `PARSE_ERROR`, `UNSUPPORTED_CONSTRUCT`) —
 *   the platform understood the question and is saying no; amber, `role="status"`.
 * - **loud** (everything else: transport, 5xx, rate limits) — something is wrong with the system;
 *   the critical palette, still `role="status"` so it never interrupts a screen reader mid-sentence.
 *
 * Structured extras are shown when the backend sent them: the character `position` of a parse
 * error, the refused `construct`, the Turkish `suggestions_tr` of an unmatched command, and the
 * request id a lecturer can quote in a bug report.
 */
import type { AiclsProblem } from "@/api/problem";
import { Button } from "@/components/ui/Button";
import { t } from "@/i18n";
import { cx } from "@/lib/cx";
import styles from "./ProblemNotice.module.css";

/** Codes that mean "understood, but not supported" rather than "broken". */
const QUIET_CODES: ReadonlySet<string> = new Set([
  "UNSUPPORTED_PROBLEM",
  "UNSUPPORTED_CONSTRUCT",
  "NLU_NO_INTENT",
  "PARSE_ERROR",
  "VALIDATION_ERROR",
]);

export interface ProblemNoticeProps {
  /** The parsed problem; `detail_tr` is always a usable Turkish sentence. */
  problem: AiclsProblem;
  /** Retry the same command. */
  onRetry?: () => void;
  /** Dismiss the notice and go back to the board. */
  onDismiss?: () => void;
}

/** Whether the failure is a polite "no" rather than a fault. */
function isQuietProblem(problem: AiclsProblem): boolean {
  return QUIET_CODES.has(problem.code);
}

/** A calm, quotable explanation of why there is no new answer. */
export function ProblemNotice({ problem, onRetry, onDismiss }: ProblemNoticeProps): JSX.Element {
  const quiet = isQuietProblem(problem);
  const construct = typeof problem.construct === "string" ? problem.construct : null;
  const suggestions = problem.suggestions_tr ?? [];

  return (
    <section
      className={cx(styles.notice, quiet ? styles.quiet : styles.loud)}
      role="status"
      aria-live="polite"
      data-testid="problem-notice"
      data-code={problem.code}
    >
      <h2 className={styles.title}>
        {quiet ? t("results.unsupportedTitle") : problem.title || t("errors.unknown")}
      </h2>
      <p className={styles.detail}>{problem.detail_tr}</p>

      {construct === null ? null : (
        <p className={styles.meta}>{t("results.construct", { construct })}</p>
      )}
      {problem.position === undefined ? null : (
        <p className={styles.meta}>{t("errors.position", { position: problem.position })}</p>
      )}
      {suggestions.length === 0 ? null : (
        <p className={styles.meta}>{t("errors.suggestions", { list: suggestions.join(", ") })}</p>
      )}
      {quiet ? <p className={styles.meta}>{t("results.rephrase")}</p> : null}

      <div className={styles.actions}>
        {onRetry === undefined ? null : (
          <Button variant="secondary" onClick={onRetry}>
            {t("common.retry")}
          </Button>
        )}
        {onDismiss === undefined ? null : (
          <Button variant="ghost" onClick={onDismiss}>
            {t("common.dismiss")}
          </Button>
        )}
        {problem.instance === null ? null : (
          <span className={cx(styles.requestId, "chrome-secondary")}>
            {t("errors.requestId", { id: problem.instance })}
          </span>
        )}
      </div>
    </section>
  );
}
