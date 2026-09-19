/**
 * The derivation, step by step.
 *
 * Steps are pedagogy, not proof: `mathengine` emits a short Turkish title, the intermediate
 * expression and (when it has one) the rule it applied. They are the part of the answer a lecturer
 * actually talks through, so they are an ordered list with real numbers, they stay in the DOM when
 * collapsed (a screen reader and a `Ctrl+F` both find them), and every expression goes through
 * `MathView` — the sanitiser and the KaTeX policy apply here exactly as they do to the answer.
 */
import { t } from "@/i18n";
import type { StepDto } from "@/api/types";
import { MathView } from "./MathView";
import styles from "./StepsView.module.css";

export interface StepsViewProps {
  /** Steps of the result, in order. */
  steps: readonly StepDto[];
  /** Start expanded (the default: in a lecture the steps *are* the explanation). */
  defaultOpen?: boolean;
}

/** An ordered list of steps, or a quiet placeholder when the command produced none. */
export function StepsView({ steps, defaultOpen = true }: StepsViewProps): JSX.Element {
  if (steps.length === 0) {
    return <p className={styles.empty}>{t("steps.empty")}</p>;
  }
  return (
    <details className={styles.details} open={defaultOpen}>
      <summary className={styles.summary}>{t("steps.title")}</summary>
      <ol className={styles.list}>
        {steps.map((step, index) => (
          <li key={`${String(index)}-${step.title_tr}`} className={styles.item}>
            <span className={styles.index} aria-hidden="true">
              {index + 1}
            </span>
            <div className={styles.content}>
              <h4 className={styles.title}>
                <span className="visually-hidden">{t("steps.step", { index: index + 1 })}: </span>
                {step.title_tr}
              </h4>
              {step.latex === "" ? null : <MathView latex={step.latex} display />}
              {step.detail_tr === undefined || step.detail_tr === "" ? null : (
                <p className={styles.detail}>{step.detail_tr}</p>
              )}
              {step.rule === undefined || step.rule === "" ? null : (
                <p className={styles.rule}>{step.rule}</p>
              )}
            </div>
          </li>
        ))}
      </ol>
    </details>
  );
}
