/**
 * Audience A: the lecturer and the student.
 *
 * Written to be read once before the first lecture and consulted afterwards by anchor. It answers,
 * in order, the questions someone standing at the board actually asks: what is this, how do I give
 * it a command, why is it asking me a question, why does it sometimes refuse, what am I looking at,
 * and which key does what.
 *
 * Every example is a real Temel Analiz problem the reader can retype immediately (`examples.ts`),
 * and every claim about behaviour matches the product: the clarification is a normal answer, the
 * refusal is the frozen contract sentence, and the shortcut table is generated from the bindings
 * rather than copied.
 */
import { Link } from "react-router-dom";

import { MathView } from "@/features/results/MathView";
import { t, type MessageKey } from "@/i18n";
import { Callout } from "./Callout";
import { CodeBlock } from "./CodeBlock";
import { ExampleCard } from "./ExampleCard";
import { CLARIFICATION_EXAMPLE, WORKED_EXAMPLES } from "./examples";
import { GUIDE_ROUTES } from "./routes";
import { Section } from "./Section";
import { ShortcutTable } from "./ShortcutTable";
import { USAGE } from "./sections";
import styles from "./GuideArticle.module.css";

const RESULT_ROWS: readonly { termKey: MessageKey; descKey: MessageKey }[] = [
  { termKey: "guide.usage.result.input.term", descKey: "guide.usage.result.input.desc" },
  { termKey: "guide.usage.result.answer.term", descKey: "guide.usage.result.answer.desc" },
  { termKey: "guide.usage.result.plain.term", descKey: "guide.usage.result.plain.desc" },
  { termKey: "guide.usage.result.steps.term", descKey: "guide.usage.result.steps.desc" },
  { termKey: "guide.usage.result.plot.term", descKey: "guide.usage.result.plot.desc" },
  { termKey: "guide.usage.result.timings.term", descKey: "guide.usage.result.timings.desc" },
  { termKey: "guide.usage.result.model.term", descKey: "guide.usage.result.model.desc" },
  { termKey: "guide.usage.result.cache.term", descKey: "guide.usage.result.cache.desc" },
];

/** The usage guide's article body. */
export function UsageGuide(): JSX.Element {
  return (
    <>
      <p className={styles.lead}>{t("guide.usage.lead")}</p>

      <Section {...USAGE["ai-cls-nedir"]}>
        <p>{t("guide.usage.what.p1")}</p>
        <p>{t("guide.usage.what.p2")}</p>
        <p>
          {t("guide.usage.what.p3")}{" "}
          <Link to={GUIDE_ROUTES.status}>{t("guide.usage.what.statusLink")}</Link>
        </p>
      </Section>

      <Section {...USAGE["komut-vermenin-uc-yolu"]}>
        <p>{t("guide.usage.inputs.intro")}</p>
        <p className={styles.panelLabel}>{t("guide.usage.inputs.tryIt")}</p>
        {WORKED_EXAMPLES.map((example) => (
          <ExampleCard key={example.id} example={example} />
        ))}
      </Section>

      <Section {...USAGE["aciklama-istegi"]}>
        <p>{t("guide.usage.clarify.p1")}</p>
        <p>{t("guide.usage.clarify.p2")}</p>
        <p>{t("guide.usage.clarify.p3")}</p>

        <div className={styles.panel} data-testid="guide-clarification-example">
          <h3 className={styles.panelTitle}>{t("guide.usage.clarify.exampleTitle")}</h3>
          <CodeBlock
            code={t(CLARIFICATION_EXAMPLE.commandKey)}
            label={t("guide.example.command")}
            compact
          />
          <p className={styles.question}>{t(CLARIFICATION_EXAMPLE.questionKey)}</p>
          <p className={styles.panelLabel}>{t("guide.usage.clarify.candidatesLabel")}</p>
          <ul className={styles.candidates}>
            {CLARIFICATION_EXAMPLE.candidates.map((candidate) => (
              <li key={candidate.command} className={styles.candidate}>
                <span className={styles.candidateLabel}>{t(candidate.labelKey)}</span>
                <MathView latex={candidate.latex} />
                <code className={styles.inlineCode}>{candidate.command}</code>
              </li>
            ))}
          </ul>
        </div>
      </Section>

      <Section {...USAGE["kapsam-disi-problem"]}>
        <p>{t("guide.usage.unsupported.p1")}</p>
        {/* Quoted from the catalogue the product itself renders, so the two can never diverge. */}
        <Callout tone="warn" label={t("guide.usage.unsupported.heading")}>
          <p className={styles.contractSentence}>{t("errors.unsupportedProblem")}</p>
        </Callout>
        <p>{t("guide.usage.unsupported.p2")}</p>
        <p>{t("guide.usage.unsupported.p3")}</p>
      </Section>

      <Section {...USAGE["sonuc-paneli"]}>
        <p>{t("guide.usage.result.intro")}</p>
        <dl className={styles.terms}>
          {RESULT_ROWS.map((row) => (
            <div key={row.termKey} className={styles.termRow}>
              <dt className={styles.term}>{t(row.termKey)}</dt>
              <dd className={styles.definition}>{t(row.descKey)}</dd>
            </div>
          ))}
        </dl>
      </Section>

      <Section {...USAGE["kitap-paneli"]}>
        <p>{t("guide.usage.book.p1")}</p>
        <CodeBlock code={t("command.example.book")} label={t("guide.example.command")} />
        <p>{t("guide.usage.book.p2")}</p>
      </Section>

      <Section {...USAGE["gecmis-ve-disa-aktarma"]}>
        <p>{t("guide.usage.history.p1")}</p>
        <p>{t("guide.usage.history.p2")}</p>
      </Section>

      <Section {...USAGE["geri-bildirim"]}>
        <p>{t("guide.usage.feedback.p1")}</p>
        <p>{t("guide.usage.feedback.p2")}</p>
      </Section>

      <Section {...USAGE["sunum-modu"]}>
        <p>{t("guide.usage.presentation.p1")}</p>
        <p>{t("guide.usage.presentation.p2")}</p>
      </Section>

      <Section {...USAGE["klavye-kisayollari"]}>
        <p>{t("guide.usage.shortcuts.intro")}</p>
        <ShortcutTable />
        <p>{t("guide.usage.shortcuts.note")}</p>
      </Section>
    </>
  );
}
