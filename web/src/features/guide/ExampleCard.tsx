/**
 * One worked example, laid out the way the result panel lays out a real answer.
 *
 * The order of the rows is deliberate and matches the product: the command, then what the system
 * understood, then the typed command that reached the engine, then the answer, then the plain
 * text form. A reader who has seen this card knows where to look on the workspace screen.
 *
 * Mathematics goes through `MathView`, the same sanitised KaTeX path the workspace uses — the
 * guide is not allowed a shortcut the product does not have.
 */
import { MathView } from "@/features/results/MathView";
import { t } from "@/i18n";
import { CodeBlock } from "./CodeBlock";
import type { WorkedExample } from "./examples";
import styles from "./ExampleCard.module.css";

export interface ExampleCardProps {
  readonly example: WorkedExample;
}

/** A single input modality, demonstrated end to end. */
export function ExampleCard({ example }: ExampleCardProps): JSX.Element {
  const mode = t(example.modeKey);
  const headingId = `ornek-${example.id}`;
  return (
    <article
      className={styles.card}
      aria-labelledby={headingId}
      data-testid={`guide-example-${example.id}`}
    >
      <h3 id={headingId} className={styles.mode}>
        {mode}
      </h3>
      <p className={styles.how}>{t(example.howKey)}</p>

      <CodeBlock code={t(example.commandKey)} label={t("guide.example.command")} compact />

      <dl className={styles.rows}>
        <div className={styles.row}>
          <dt className={styles.term}>{t("guide.example.understood")}</dt>
          <dd className={styles.definition}>
            <MathView latex={example.inputLatex} />
          </dd>
        </div>
        <div className={styles.row}>
          <dt className={styles.term}>{t("guide.example.casCommand")}</dt>
          <dd className={styles.definition}>
            <code className={styles.code}>{example.casCommand}</code>
          </dd>
        </div>
        <div className={`${styles.row} ${styles.resultRow}`}>
          <dt className={styles.term}>{t("guide.example.result")}</dt>
          <dd className={styles.definition}>
            <MathView latex={example.resultLatex} display />
          </dd>
        </div>
        <div className={styles.row}>
          <dt className={styles.term}>{t("guide.example.plain")}</dt>
          <dd className={styles.definition}>
            <code className={styles.code}>{example.resultPlain}</code>
          </dd>
        </div>
      </dl>

      <p className={styles.note}>{t(example.noteKey)}</p>
    </article>
  );
}
